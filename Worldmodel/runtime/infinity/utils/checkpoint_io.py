"""CPU loading for large, trusted training checkpoints."""
import os

import torch


def load_checkpoint_cpu(path):
    path = os.fspath(path)
    # Optimizer tensors in a full checkpoint can greatly exceed the policy
    # weights. Map their storages instead of eagerly copying all of them to RAM.
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        # Compatibility with torch versions predating mmap/weights_only flags.
        return torch.load(path, map_location="cpu")
    except RuntimeError as error:
        if "mmap can only be used" not in str(error):
            raise
        return torch.load(path, map_location="cpu", weights_only=False)


def extract_policy_state_dict(checkpoint):
    if "trainer" not in checkpoint:
        return checkpoint
    trainer = checkpoint["trainer"]
    for key in ("gpt_fsdp", "gpt_wo_ddp"):
        if isinstance(trainer.get(key), dict):
            return trainer[key]
    raise KeyError("checkpoint trainer contains neither gpt_fsdp nor gpt_wo_ddp policy weights")
