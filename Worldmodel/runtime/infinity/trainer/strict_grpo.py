"""Small, independently testable contracts for exact sampled-token PPO replay."""
import math
import torch


def clipped_policy_loss(new_logprob, old_logprob, advantages, ratio_eps):
    """PPO uses the minimum for both signs of the advantage."""
    if not 0.0 < float(ratio_eps) < 1.0:
        raise ValueError("grpo_ratio_eps must lie in (0, 1)")
    # A clip contains millions of bits: summed logp is about -6e5, where
    # FP32 spacing is 0.0625. Preserve FP64 BEFORE subtracting old from new;
    # casting the already rounded difference cannot recover small policy steps.
    new, old, adv = new_logprob.double(), old_logprob.double(), advantages.float()
    if new.shape != old.shape or new.shape != adv.shape:
        raise ValueError("policy logprobs and advantages must have equal shapes")
    if not all(bool(torch.isfinite(x).all()) for x in (new, old, adv)):
        raise FloatingPointError("non-finite policy logprob or advantage")
    delta = new - old
    if bool((delta.abs() > 60).any()):
        raise FloatingPointError("policy log-ratio exceeds 60; check trace/checkpoint consistency")
    ratio = delta.exp()
    clipped = ratio.clamp(1.0 - ratio_eps, 1.0 + ratio_eps)
    loss = -torch.minimum(ratio * adv, clipped * adv)
    return loss, ratio, delta


def load_sampling_trace(path):
    # Trace files are trusted local training artifacts, containing tensors and
    # schedule metadata. They must never be loaded from an untrusted upload.
    trace = torch.load(path, map_location="cpu", weights_only=False)
    if int(trace.get("schema_version", 0)) < 3:
        raise ValueError(f"{path}: regenerate StageA with trace schema >= 3 and FP64 summed logprobs")
    if trace.get("logprob_accumulation_dtype") != "float64":
        raise ValueError(f"{path}: exact PPO requires float64 logprob accumulation")
    if trace.get("old_logprob_mode") != "sampling":
        raise ValueError(f"{path}: exact PPO requires sampling-time old_logprob")
    for key in ("idx_trace", "step_clipids", "step_is_sampled", "scale_schedule", "context_info", "text_cond_tuple", "tau_list", "block_parameter_dtype", "autocast_dtype"):
        if key not in trace or trace[key] is None:
            raise ValueError(f"{path}: missing {key}")
    if not isinstance(trace["idx_trace"], list) or not trace["idx_trace"]:
        raise ValueError(f"{path}: idx_trace must be a nonempty list")
    if len(trace["idx_trace"]) != len(trace["step_clipids"]) or len(trace["idx_trace"]) != len(trace["step_is_sampled"]):
        raise ValueError(f"{path}: token trace and step_clipids lengths differ")
    target = int(trace["clipid_target"])
    tokens = sum(int(torch.as_tensor(x).numel()) for x, c, sampled in zip(trace["idx_trace"], trace["step_clipids"], trace["step_is_sampled"]) if int(c) == target and bool(sampled))
    if tokens <= 0:
        raise ValueError(f"{path}: no sampled tokens for target clip {target}")
    counts = trace.get("sample_token_count_by_clip")
    if counts is None or target >= len(counts) or int(counts[target]) != tokens:
        raise ValueError(f"{path}: target-clip token count differs from sampling trace")
    if not math.isfinite(float(trace["sample_logprob"])):
        raise FloatingPointError(f"{path}: non-finite sampling logprob")
    by_clip = trace.get("sample_logprob_by_clip")
    if not isinstance(by_clip, (tuple, list)) or target >= len(by_clip):
        raise ValueError(f"{path}: missing per-clip FP64 sampling logprobs")
    target_logprob = by_clip[target]
    if not torch.is_tensor(target_logprob) or target_logprob.dtype != torch.float64 or target_logprob.numel() != 1:
        raise ValueError(f"{path}: target-clip sampling logprob must be one float64 tensor value")
    if float(target_logprob.reshape(-1)[0]) != float(trace["sample_logprob"]):
        raise ValueError(f"{path}: target-clip logprob differs from saved sample_logprob")
    cond = trace["text_cond_tuple"]
    if not isinstance(cond, (tuple, list)) or len(cond) != 4:
        raise ValueError(f"{path}: text_cond_tuple must contain compact text, lengths, offsets, max length")
    if not torch.is_tensor(cond[0]) or not bool(torch.isfinite(cond[0]).all()):
        raise ValueError(f"{path}: invalid saved text conditions")
    if bool(int(trace.get("noise_input", 0))):
        raise ValueError(f"{path}: noise_input=1 requires recorded initial noises; use noise_input=0")
    return trace


def move_text_conditions(condition, device):
    kv, lens, offsets, max_length = condition
    return (kv.to(device=device, dtype=torch.float32).clone(), list(lens),
            torch.as_tensor(offsets, device=device, dtype=torch.int32), int(max_length))


def parameter_probes(model):
    """Compact update diagnostics; full frozen equality is checked in CPU tests."""
    probes = {}
    for name, parameter in model.named_parameters():
        if parameter.numel():
            flat = parameter.detach().reshape(-1)
            indices = sorted({0, flat.numel() // 2, flat.numel() - 1})
            probes[name] = (bool(parameter.requires_grad), flat[indices].float().cpu().clone())
    return probes


def compare_parameter_probes(before, after):
    if before.keys() != after.keys():
        raise RuntimeError("parameter set changed during policy update")
    frozen_max = trainable_max = 0.0
    for name, (trainable, values) in before.items():
        if trainable != after[name][0] or values.shape != after[name][1].shape:
            raise RuntimeError(f"parameter layout changed during update: {name}")
        change = float((values - after[name][1]).abs().max())
        if not trainable:
            frozen_max = max(frozen_max, change)
        else:
            trainable_max = max(trainable_max, change)
    if frozen_max != 0.0:
        raise RuntimeError(f"frozen parameter probes changed by {frozen_max}")
    return frozen_max, trainable_max
