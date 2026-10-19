"""Precision and shard-aware diagnostics for exact GRPO with nested FSDP."""
import math
import torch
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import CustomPolicy


def chunk_mixed_precision_policy(chunk_type):
    """Keep FP32 optimizer masters/root modules; use BF16 transformer forwards.

    Disabling input casts preserves the sampler's FP32 residual/input behavior.
    BF16 parameters are supplied and restored by FSDP instead of allocating an
    autocast copy of every linear weight at every autoregressive replay step.
    """
    precision = MixedPrecision(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
        buffer_dtype=torch.float32, keep_low_precision_grads=False,
        cast_forward_inputs=False, cast_root_forward_inputs=False,
    )
    return CustomPolicy(lambda module: {"mixed_precision": precision}
                        if isinstance(module, chunk_type) else False)


def transformer_chunks(model):
    chunks = getattr(model, "block_chunks", None)
    if chunks is None or len(chunks) == 0:
        chunks = getattr(model, "blocks", None)
    return list(chunks) if chunks is not None else []


def transformer_precision(model):
    """Report effective forward dtype separately from local master storage."""
    chunks = transformer_chunks(model)
    if not chunks:
        chunks = list(getattr(model, "unregistered_blocks", []))
    forward, masters = set(), set()
    for chunk in chunks:
        parameters = list(chunk.parameters())
        storage = {str(p.dtype).replace("torch.", "") for p in parameters}
        masters.update(storage)
        mp = getattr(chunk, "mixed_precision", None)
        dtype = getattr(mp, "param_dtype", None)
        forward.update({str(dtype).replace("torch.", "")} if dtype is not None else storage)
    if len(forward) > 1:
        raise ValueError(f"transformer chunks have inconsistent forward dtypes: {sorted(forward)}")
    return {"block_forward_dtype": next(iter(forward), None),
            "block_master_dtypes": sorted(masters)}


def chunk_gradient_statistics(model, device, require_nonzero=False):
    """Aggregate local original-parameter shards after backward, before clipping.

    FSDP original parameters may have zero-sized local shards. Sum reduction
    reconstructs each chunk's global element counts and squared gradient norm.
    No parameter unshard or extra forward is performed.
    """
    chunks = transformer_chunks(model)
    stats = torch.zeros((len(chunks), 6), device=device, dtype=torch.float64)
    for index, chunk in enumerate(chunks):
        for parameter in chunk.parameters():
            if not parameter.requires_grad:
                continue
            stats[index, 5] += parameter.numel()
            grad = parameter.grad
            if grad is None or grad.numel() == 0:
                continue
            detached = grad.detach()
            norm = torch.linalg.vector_norm(detached, dtype=torch.float64)
            stats[index, 0] += norm.square()
            stats[index, 1] += detached.numel()
            stats[index, 2] += torch.count_nonzero(detached)
            stats[index, 3] += (~torch.isfinite(detached).all()).to(torch.float64)
            stats[index, 4] += 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
    rows = []
    for index, values in enumerate(stats.cpu().tolist()):
        square_norm, elements, nonzero, nonfinite, tensors, trainable = values
        finite = nonfinite == 0 and math.isfinite(square_norm)
        row = dict(chunk=index, trainable_parameters=int(trainable),
                   gradient_parameters=int(elements), nonzero_gradient_elements=int(nonzero),
                   nonfinite_gradient_tensors=int(nonfinite), gradient_tensor_shards=int(tensors),
                   grad_norm=square_norm ** 0.5 if finite else None, finite=finite)
        if trainable and (not finite or (require_nonzero and not nonzero)):
            raise RuntimeError(f"transformer chunk {index} has invalid or zero gradient: {row}")
        rows.append(row)
    return rows


def validate_replay_execution_schedule(traces):
    """Fail collectively before FSDP forwards with incompatible AR lengths.

    Each saved trace step executes all wrapped transformer chunks once, plus
    the same text-prefill call per trace. The ordered step counts therefore
    determine collective ordering; clip IDs alone do not. With SP=1, token
    sequence shapes do not change FSDP parameter collective payloads.
    """
    signature = (len(traces), tuple(len(trace["step_clipids"]) for trace in traces))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        signatures = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(signatures, signature)
        if any(value != signatures[0] for value in signatures[1:]):
            raise RuntimeError(
                "FSDP exact replay requires equal execution schedules across ranks "
                f"before model forward; received (trace_count, ordered_step_counts)={signatures}. "
                "Use batches with the same ordered replay lengths on every rank."
            )
    return signature
