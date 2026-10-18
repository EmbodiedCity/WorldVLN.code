"""Small CPU contract tests for the production sampling/replay implementation.

The heavyweight model is not imported: its actual policy methods are extracted
and exercised on a tiny differentiable KV block, with a frozen bit-code decoder.
Run with Python + torch, passing the production infinity.py path if necessary.
"""
import ast
import copy
import json
import math
import os
import sys
import types
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


if "CORE_SOURCE" not in globals():
    source = Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].endswith("infinity.py") else Path(__file__).resolve().parents[1] / "Worldmodel/runtime/infinity/models/infinity.py"
    CORE_SOURCE = source.read_text(encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1].endswith("infinity.py"):
        sys.argv = sys.argv[:1]
tree = ast.parse(CORE_SOURCE)
names = {"_policy_trace_layout", "_policy_log_probs", "_policy_checked_tokens"}
methods = {"ar_infer_infinity_elegant", "_ar_infer_infinity_elegant_impl", "ar_infer_infinity_elegant_replay_logprob", "forward"}
model_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Infinity")
selected = [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
selected.extend(copy.deepcopy(n) for n in model_class.body if isinstance(n, ast.FunctionDef) and n.name in methods)
namespace = dict(globals())
namespace["sp_manager"] = types.SimpleNamespace(sp_on=lambda: False)
namespace["get_first_full_spatial_size_scale_index"] = lambda schedule: 1


class QuietProgress:
    def __init__(self, **kwargs):
        pass

    def update(self, value):
        pass

    def close(self):
        pass


namespace["tqdm"] = types.SimpleNamespace(tqdm=QuietProgress)
mock_schedule = types.ModuleType("infinity.schedules.infinity_elegant")
mock_schedule.interpolate = lambda codes, size, mode, **kwargs: F.interpolate(codes, size=size[-3:], mode=mode)
sys.modules["infinity.schedules.infinity_elegant"] = mock_schedule
exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), "production_policy_methods", "exec"), namespace)


class KVBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.7))
        self.attn = types.SimpleNamespace(kv_caching=self.reset_cache)
        self.cached = None

    def reset_cache(self, enable, reset=True):
        self.enabled = enable
        if reset:
            self.cached = None

    def forward(self, x, scale_ind=None, last_repetition_step=True, **kwargs):
        value = x * self.weight
        if self.cached is not None:
            value = value + self.cached.mean() * 0.2
        if last_repetition_step:
            self.cached = value
        return value


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([KVBlock()])
        self.unregistered_blocks = list(self.blocks)
        self.num_block_chunks = 1
        self.num_of_label_value = 2
        self.vae_embed_dim = 1
        self.apply_spatial_patchify = False
        self.other_args = types.SimpleNamespace(noise_input=0, use_two_stage_lfq=False)
        self.rope2d_freqs_grid = {"freqs_text": torch.zeros(1, 1, 1, 1, 1)}
        self.rng = torch.Generator(device="cpu")
        self.head = torch.nn.Linear(1, 2, bias=True)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[-0.3], [0.8]]))
            self.head.bias.copy_(torch.tensor([0.1, -0.1]))

    def prepare_text_conditions(self, label, cfg, B, negative, *args, **kwargs):
        prefix = label[0].reshape(1, 1, 1)
        if any(float(v) != 1.0 for v in cfg):
            prefix = torch.cat((prefix, prefix * -0.5), dim=0)
        return prefix, [1]

    def embeds_codes2input(self, codes, repeat=1):
        if self.apply_spatial_patchify:
            patched = F.pixel_unshuffle(codes.permute(0, 2, 1, 3, 4), 2)
            codes = patched.mean(dim=2, keepdim=True).permute(0, 2, 1, 3, 4)
        return codes.flatten(2).transpose(1, 2).repeat(repeat, 1, 1)

    def get_logits_during_infer(self, x, **kwargs):
        return self.head(x)


for name in methods:
    setattr(TinyPolicy, name, namespace[name])


def bit_codes(bits, label_type):
    return bits.float().permute(0, 4, 1, 2, 3)


class PolicyTraceTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyPolicy()
        self.args = types.SimpleNamespace(
            image_scale_repetition="[2,1]", video_scale_repetition="[1,2]",
            max_repeat_times=2, semantic_scales=1, use_cfg=1, use_apg=0,
            grpo_replay_checkpointing=0,
        )
        self.kw = dict(
            vae=types.SimpleNamespace(quantizer=types.SimpleNamespace(
                lfq_detail=types.SimpleNamespace(indices_to_codes=bit_codes),
                z_interplote_up="nearest", z_interplote_down="nearest",
            )),
            scale_schedule=[(1, 1, 1), (1, 2, 2)] * 3,
            label_B_or_BLT=(torch.ones(1, 1), [1], torch.tensor([0, 1]), 1),
            cfg_list=[1.5] * 6, tau_list=[0.8] * 6, top_k=0, top_p=1.0,
            args=self.args, get_visual_rope_embeds=lambda *args: None,
            context_info={}, gt_leak=2,
            gt_ls_Bl=[torch.zeros(1, 1, 1, 1, 1, dtype=torch.long)] * 2 + [torch.ones(1, 1, 2, 2, 1, dtype=torch.long)],
        )

    def sample(self, **overrides):
        kw = {**self.kw, **overrides}
        return self.model.ar_infer_infinity_elegant(**kw, g_seed=7, return_trace=True, return_summed_code_only=True)

    def replay(self, trace, **overrides):
        kw = dict(self.kw)
        label = kw.pop("label_B_or_BLT")
        kw.update(
            forced_idx_trace=trace["idx_trace"], logprob_clipid_target=1,
            expected_step_clipids=trace["step_clipids"],
            expected_step_is_sampled=trace["step_is_sampled"],
            expected_token_count=trace["token_count_by_clip"][1],
        )
        kw.update(overrides)
        return self.model(label, policy_trace_replay_kwargs=kw)

    def test_actual_loop_matches_sampling_and_backpropagates(self):
        trace = self.sample()
        scored = self.replay(trace)
        torch.testing.assert_close(scored.detach(), trace["sample_logprob_by_clip"][1], atol=1e-6, rtol=0)
        self.assertEqual(trace["token_count_by_clip"], [0, 9, 9])
        self.assertEqual(trace["step_is_sampled"], [False] * 3 + [True] * 6)
        self.assertEqual(float(trace["sample_logprob_by_clip"][0]), 0.0)
        (-scored.mean()).backward()
        for parameter in self.model.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(float(self.model.blocks[0].weight.grad.abs()), 0.0)
        self.assertIsNone(self.model.blocks[0].cached)

    def test_repeat_cap_stays_aligned(self):
        self.args.max_repeat_times = 1
        trace = self.sample()
        self.assertEqual(len(trace["idx_trace"]), 6)
        torch.testing.assert_close(self.replay(trace).detach(), trace["sample_logprob_by_clip"][1], atol=1e-6, rtol=0)

    def test_malformed_trace_is_rejected_and_cache_cleaned(self):
        trace = self.sample()
        with self.assertRaisesRegex(ValueError, "steps"):
            self.replay(trace, forced_idx_trace=trace["idx_trace"][:-1])
        bad = [t.clone() for t in trace["idx_trace"]]
        bad[0].fill_(1)
        with self.assertRaisesRegex(ValueError, "GT prefix"):
            self.replay(trace, forced_idx_trace=bad)
        self.assertIsNone(self.model.blocks[0].cached)
        self.assertFalse(self.model.blocks[0].enabled)
        with self.assertRaisesRegex(ValueError, "bits"):
            self.replay(trace, expected_token_count=123)
        with self.assertRaisesRegex(ValueError, "clip mapping"):
            self.replay(trace, expected_step_clipids=[0])

    def test_truncated_distribution_uses_same_support(self):
        for top_k, top_p in [(1, 1.0), (0, 0.6)]:
            trace = self.sample(top_k=top_k, top_p=top_p)
            score = self.replay(trace, top_k=top_k, top_p=top_p)
            torch.testing.assert_close(score.detach(), trace["sample_logprob_by_clip"][1], atol=1e-6, rtol=0)

    def test_patchified_gt_preserves_vae_pixel_order(self):
        self.model.apply_spatial_patchify = True
        self.model.head = torch.nn.Linear(1, 8)
        small_gt = torch.tensor([0, 1, 1, 0], dtype=torch.long).reshape(1, 1, 2, 2, 1)
        large_gt = torch.arange(16).remainder(2).reshape(1, 1, 4, 4, 1)
        self.kw["gt_ls_Bl"] = [small_gt, small_gt, large_gt]
        trace = self.sample()
        trace_patch = trace["idx_trace"][0].reshape(1, 1, 1, 1, 4)
        reconstructed_gt = F.pixel_shuffle(trace_patch.permute(0, 1, 4, 2, 3), 2).permute(0, 1, 3, 4, 2)
        torch.testing.assert_close(reconstructed_gt, small_gt)
        torch.testing.assert_close(self.replay(trace).detach(), trace["sample_logprob_by_clip"][1], atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
