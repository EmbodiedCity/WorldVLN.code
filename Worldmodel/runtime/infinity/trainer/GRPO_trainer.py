#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GRPO trainer.

This class reuses InfinityTrainer and selects its exact sampled-token PPO
replay path via `args.trainer_type=grpo`, using precomputed replay advantages.
"""

from infinity.trainer.sft_trainer import InfinityTrainer


class GRPOTrainer(InfinityTrainer):
    pass

