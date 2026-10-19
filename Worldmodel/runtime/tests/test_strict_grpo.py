"""CPU contract tests; no model downloads, GPU, AirSim, or training dataset needed.

Run: python -m unittest discover -s tests -p test_strict_grpo.py -v
The real trainer method is extracted by AST to avoid importing GPU-only kernels.
"""
import ast
from contextlib import nullcontext
import json
import math
import os
import os.path as osp
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace, ModuleType
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from infinity.trainer.strict_grpo import (
    clipped_policy_loss, load_sampling_trace, move_text_conditions,
)
# A single-process boundary stub keeps these CPU tests independent of cluster
# setup, timezone packages, and the production CUDA collective backend.
dist = ModuleType("infinity.utils.dist")
dist.get_world_size = lambda: 1
dist.get_rank = lambda: 0
dist.is_master = lambda: True
dist.allreduce = lambda value: None
dist.barrier = lambda: None
sys.modules["infinity.utils.dist"] = dist
from infinity.utils.amp_opt import AmpOptimizer


ROOT = Path(__file__).resolve().parents[1]


def actual_method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.1, -0.2]))
        self.frozen = torch.nn.Parameter(torch.tensor([2.0, 3.0]), requires_grad=False)
        self.cond_drop_rate = 0.1

    def forward(self, condition, _unused, policy_trace_replay_kwargs):
        assert self.cond_drop_rate == 0.0
        assert not self.training
        return self.logits.log_softmax(-1)[1].double().reshape(1) * 2


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        return self.module(*args, **kwargs)


class TestExactPolicyContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.policy = Policy()
        self.old = float((self.policy.logits.log_softmax(-1)[1].double() * 2).detach())
        self.trace = dict(
            schema_version=3, logprob_accumulation_dtype="float64", old_logprob_mode="sampling", clipid_target=1,
            idx_trace=[torch.zeros((1, 1), dtype=torch.long), torch.ones((1, 2), dtype=torch.long)],
            step_clipids=[0, 1], step_is_sampled=[False, True],
            sample_token_count_by_clip=[0, 2], sample_logprob=self.old,
            sample_logprob_by_clip=[torch.zeros(1, dtype=torch.float64), torch.tensor([self.old], dtype=torch.float64)],
            scale_schedule=[(1, 1, 1), (1, 1, 1)], context_info={0: {}, 1: {}},
            text_cond_tuple=(torch.ones((1, 2)), [1], torch.tensor([0, 1], dtype=torch.int32), 1),
            tau_list=[1.0, 1.0], infinity_cfg=1.0, gt_leak=1, noise_input=0,
            block_parameter_dtype="float32", autocast_dtype="float32",
        )
        self.path = self.root / "trace.pt"
        torch.save(self.trace, self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_signed_advantage_clipping_and_gradient(self):
        new = torch.tensor([0.5, 2.0]).log().requires_grad_()
        losses, _, _ = clipped_policy_loss(new, torch.zeros(2), torch.full((2,), -1.0), .2)
        self.assertTrue(torch.allclose(losses, torch.tensor([.8, 2.0], dtype=torch.float64)))
        losses.sum().backward()
        self.assertTrue(torch.allclose(new.grad, torch.tensor([0.0, 2.0])))

    def test_large_summed_logprob_preserves_small_delta_and_gradient(self):
        old = torch.tensor([-600000.0], dtype=torch.float64)
        new = torch.tensor([-600000.0 + .001], dtype=torch.float64, requires_grad=True)
        # Reproduce the old numerical failure, then verify the actual objective
        # preserves the same change with the production clip epsilon.
        self.assertEqual(float(new.detach().float() - old.float()), 0.0)
        loss, ratio, delta = clipped_policy_loss(new, old, torch.tensor([.25]), .02)
        self.assertEqual(delta.dtype, torch.float64)
        self.assertEqual(ratio.dtype, torch.float64)
        self.assertAlmostEqual(float(delta.detach()), .001, delta=1e-10)
        self.assertAlmostEqual(float(ratio.detach()), math.exp(.001), delta=1e-10)
        loss.sum().backward()
        self.assertEqual(new.grad.dtype, torch.float64)
        self.assertAlmostEqual(float(new.grad), -.25 * math.exp(.001), delta=1e-10)

    def test_missing_mask_and_approximate_scores_are_rejected(self):
        for key, value in [("step_is_sampled", None), ("old_logprob_mode", "trace_ce"),
                           ("sample_token_count_by_clip", [0, 3]), ("schema_version", 2),
                           ("logprob_accumulation_dtype", None), ("logprob_accumulation_dtype", "float32"),
                           ("sample_logprob_by_clip", [torch.zeros(1), torch.tensor([self.old])])]:
            invalid = dict(self.trace)
            invalid[key] = value
            torch.save(invalid, self.path)
            with self.assertRaises(ValueError):
                load_sampling_trace(str(self.path))

    def test_saved_condition_is_copied(self):
        moved = move_text_conditions(self.trace["text_cond_tuple"], "cpu")
        moved[0].zero_()
        self.assertEqual(float(self.trace["text_cond_tuple"][0].sum()), 2.0)

    def test_nonfinite_and_huge_ratios_are_rejected(self):
        for value in [float("nan"), float("inf"), 61.0]:
            with self.assertRaises(FloatingPointError):
                clipped_policy_loss(torch.tensor([value]), torch.zeros(1), torch.ones(1), .2)

    def test_real_trainer_wrapper_update_and_frozen_parameters(self):
        namespace = dict(torch=torch, nullcontext=nullcontext, os=os, osp=osp, json=json, dist=dist)
        method = actual_method(ROOT / "infinity/trainer/sft_trainer.py", "InfinityTrainer", "_train_sampling_trace", namespace)
        wrapper = Wrapper(self.policy)
        optimizer = torch.optim.AdamW([self.policy.logits], lr=1e-4, weight_decay=0.0)
        amp = AmpOptimizer("gpt", 0, optimizer, wrapper, 1.0, 5.0, 0)
        trainer = SimpleNamespace(gpt=wrapper, gpt_wo_ddp=self.policy, vae_local=None,
                                  gpt_opt=amp, get_visual_rope_embeds=None)
        args = SimpleNamespace(
            device="cpu", grpo_aux_sft_coef=0.0, grpo_kl_beta=0.0,
            grpo_require_nonnegative_adv=1, grpo_adv_clip=5.0, grpo_validate_updates=1,
            grpo_replay_save_on_cpu=0, grpo_replay_save_on_cpu_pin=0,
            grpo_ratio_eps=.2, grpo_hybrid_rl_coef=1.0,
            grpo_initial_logprob_atol=.05, use_fsdp_model_ema=False,
            grpo_diagnostics_path=str(self.root / "updates.jsonl"), local_out_path=str(self.root),
        )
        before = self.policy.logits.detach().clone()
        frozen = self.policy.frozen.detach().clone()
        for step in range(3):
            norm, _ = method(trainer, epoch=0, it=step, g_it=step, stepping=True,
                             clip_decay_ratio=1.0, metric_lg=SimpleNamespace(update=lambda **kw: None),
                             args=args, trace_files=[[str(self.path)]], old_logprobs=[self.old],
                             advantages=[1.0], clip_ids=[1], rewards=[1.0])
            self.assertGreater(float(norm), 0.0)
        self.assertEqual(wrapper.calls, 3)
        self.assertFalse(torch.equal(self.policy.logits, before))
        self.assertTrue(torch.equal(self.policy.frozen, frozen))
        self.assertTrue(self.policy.training)
        self.assertEqual(self.policy.cond_drop_rate, .1)
        rows = [json.loads(line) for line in (self.root / "updates.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(rows[0]["ratio_min"], 1.0)
        self.assertGreater(rows[0]["trainable_probe_max_change"], 0)
        self.assertEqual(rows[0]["frozen_probe_max_change"], 0)

    def test_trace_dataset_does_not_decode_video(self):
        namespace = dict(Tuple=tuple, osp=osp)
        method = actual_method(ROOT / "infinity/dataset/dataset_joint_vi.py", "JointViIterableDataset", "prepare_video_input", namespace)
        dataset = SimpleNamespace(other_args=SimpleNamespace(trainer_type="grpo", grpo_new_logprob_mode="trace_replay"))
        success, item = method(dataset, dict(caption="Original instruction.", grpo_trace_files=[str(self.path)]))
        self.assertTrue(success)
        self.assertTrue(item["strict_trace_replay"])
        self.assertIsNone(item["img_T3HW"])

    def test_optimizer_rejects_invalid_gradient_before_update(self):
        for expression, error in [(lambda p: p.sqrt().sum(), FloatingPointError),
                                  (lambda p: (p * 0).sum(), RuntimeError)]:
            parameter = torch.nn.Parameter(torch.zeros(1))
            model = torch.nn.ParameterList([parameter])
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            amp = AmpOptimizer("gpt", 0, optimizer, model, 1.0, 5.0, 0)
            with self.assertRaises(error):
                amp.backward_clip_step(0, 0, 0, True, expression(parameter), require_nonzero_grad=True)
            self.assertFalse(optimizer.state)
            self.assertEqual(float(parameter.detach()), 0.0)

    def test_gradient_callback_observes_unclipped_grad_before_optimizer_step(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        model = torch.nn.ParameterList([parameter])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        amp = AmpOptimizer("gpt", 0, optimizer, model, 1.0, 1.0, 0)
        seen = []
        amp.backward_clip_step(0, 0, 0, True, (parameter * 5).sum(),
                               gradient_callback=lambda: seen.append((float(parameter), float(parameter.grad))))
        self.assertEqual(seen, [(1.0, 5.0)])
        self.assertAlmostEqual(float(parameter), 0.9, places=6)

    def test_partial_freeze_optimizer_excludes_frozen_weights(self):
        import infinity.utils.lr_control as lr_control
        # Production dist initialization extends print with a `force` keyword.
        lr_control.print = lambda *args, **kwargs: None
        names, parameters, groups = lr_control.filter_params(self.policy, {"logits": 1})
        self.assertEqual(names, ["logits"])
        self.assertEqual(len(parameters), 1)
        self.assertIs(parameters[0], self.policy.logits)
        self.assertTrue(all(p.requires_grad for group in groups for p in group["params"]))

    def test_explicit_heads_only_freeze(self):
        from infinity.utils.freeze_utils import apply_stageb_partial_freeze
        model = torch.nn.Module()
        model.block_chunks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(2)])
        model.word_embed = torch.nn.Linear(2, 2)
        model.norm_hidden_sates = torch.nn.LayerNorm(2)
        model.head = torch.nn.Linear(2, 2)
        model.semantic_head2 = torch.nn.Linear(2, 2)
        summary = apply_stageb_partial_freeze(model, 2, print_summary=False)
        self.assertEqual(summary["mode"], "heads-only")
        self.assertEqual(summary["trainable_chunk_indices"], [])
        self.assertFalse(any(p.requires_grad for p in model.block_chunks.parameters()))
        self.assertFalse(model.word_embed.weight.requires_grad)
        self.assertTrue(all(p.requires_grad for p in model.head.parameters()))
        with self.assertRaises(ValueError):
            apply_stageb_partial_freeze(model, 3, print_summary=False)

    def test_mmap_checkpoint_and_both_policy_keys(self):
        from infinity.utils.checkpoint_io import load_checkpoint_cpu, extract_policy_state_dict
        for key in ("gpt_fsdp", "gpt_wo_ddp"):
            checkpoint = self.root / (key + ".pth")
            torch.save({"trainer": {key: {"weight": torch.ones(2)}}}, checkpoint)
            state = extract_policy_state_dict(load_checkpoint_cpu(checkpoint))
            self.assertTrue(torch.equal(state["weight"], torch.ones(2)))


if __name__ == "__main__":
    unittest.main()
