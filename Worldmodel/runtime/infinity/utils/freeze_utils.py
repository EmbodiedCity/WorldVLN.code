from typing import Dict, List, Tuple


def _resolve_block_container(model) -> Tuple[object, str]:
    if hasattr(model, 'block_chunks') and len(model.block_chunks) > 0:
        return model.block_chunks, 'block_chunks'
    if hasattr(model, 'blocks') and len(model.blocks) > 0:
        return model.blocks, 'blocks'
    raise AttributeError('model has neither block_chunks nor blocks for partial freeze')


def summarize_parameter_status(model, sample_limit: int = 12) -> Dict[str, object]:
    summary = {
        'total_params': 0,
        'trainable_params': 0,
        'frozen_params': 0,
        'total_param_tensors': 0,
        'trainable_param_tensors': 0,
        'frozen_param_tensors': 0,
        'trainable_samples': [],
        'frozen_samples': [],
    }

    for name, para in model.named_parameters():
        summary['total_params'] += para.numel()
        summary['total_param_tensors'] += 1
        if para.requires_grad:
            summary['trainable_params'] += para.numel()
            summary['trainable_param_tensors'] += 1
            if len(summary['trainable_samples']) < sample_limit:
                summary['trainable_samples'].append(name)
        else:
            summary['frozen_params'] += para.numel()
            summary['frozen_param_tensors'] += 1
            if len(summary['frozen_samples']) < sample_limit:
                summary['frozen_samples'].append(name)

    return summary


def _print_partial_freeze_summary(summary: Dict[str, object]) -> None:
    total_params = summary['total_params']
    trainable_params = summary['trainable_params']
    frozen_params = summary['frozen_params']
    trainable_ratio = (trainable_params / total_params) if total_params else 0.0
    frozen_ratio = (frozen_params / total_params) if total_params else 0.0

    print(
        '[partial-freeze] '
        f"enabled={summary['enabled']} mode={summary['mode']} "
        f"freeze_chunk_prefix={summary['freeze_chunk_prefix']} "
        f"total_chunks={summary['total_chunks']} "
        f"trainable_chunks={summary['trainable_chunk_indices']}",
        flush=True,
    )
    print(
        '[partial-freeze] '
        f"total={total_params / 1e9:.4f}B trainable={trainable_params / 1e9:.4f}B ({trainable_ratio:.2%}) "
        f"frozen={frozen_params / 1e9:.4f}B ({frozen_ratio:.2%})",
        flush=True,
    )
    print(
        '[partial-freeze] '
        f"trainable_tensors={summary['trainable_param_tensors']} frozen_tensors={summary['frozen_param_tensors']}",
        flush=True,
    )
    print(f"[partial-freeze] trainable_samples={summary['trainable_samples']}", flush=True)
    print(f"[partial-freeze] frozen_samples={summary['frozen_samples']}", flush=True)


def apply_stageb_partial_freeze(model, freeze_chunk_prefix: int, print_summary: bool = True) -> Dict[str, object]:
    block_container, block_container_name = _resolve_block_container(model)
    total_chunks = len(block_container)

    if freeze_chunk_prefix < 0:
        raise ValueError(f'freeze_chunk_prefix must be >= 0, got {freeze_chunk_prefix}')
    if total_chunks <= 0:
        raise ValueError('partial freeze requires at least one transformer block container')
    if freeze_chunk_prefix > total_chunks:
        raise ValueError(
            f'freeze_chunk_prefix={freeze_chunk_prefix} must not exceed total_chunks={total_chunks}; '
            'equality explicitly selects heads-only training'
        )

    if freeze_chunk_prefix == 0:
        summary = summarize_parameter_status(model)
        summary.update({
            'enabled': False,
            'mode': 'full-train',
            'freeze_chunk_prefix': 0,
            'total_chunks': total_chunks,
            'block_container_name': block_container_name,
            'trainable_chunk_indices': list(range(total_chunks)),
        })
        if print_summary:
            _print_partial_freeze_summary(summary)
        return summary

    model.requires_grad_(False)

    for chunk_idx in range(freeze_chunk_prefix, total_chunks):
        block_container[chunk_idx].requires_grad_(True)

    for module_name in ('norm_hidden_sates', 'head', 'semantic_head2'):
        if hasattr(model, module_name):
            getattr(model, module_name).requires_grad_(True)

    summary = summarize_parameter_status(model)
    if not summary['trainable_params']:
        raise ValueError('partial freeze leaves no trainable output heads')
    summary.update({
        'enabled': True,
        'mode': 'heads-only' if freeze_chunk_prefix == total_chunks else 'chunk-prefix',
        'freeze_chunk_prefix': freeze_chunk_prefix,
        'total_chunks': total_chunks,
        'block_container_name': block_container_name,
        'trainable_chunk_indices': list(range(freeze_chunk_prefix, total_chunks)),
    })
    if print_summary:
        _print_partial_freeze_summary(summary)
    return summary
