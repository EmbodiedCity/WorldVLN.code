"""CPU checks for opt-in full-backbone precision and gradient diagnostics."""
import sys
import json
import os
import tempfile
from datetime import timedelta
from pathlib import Path
import unittest
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from infinity.utils.fsdp_grpo import (
    chunk_mixed_precision_policy, transformer_precision, chunk_gradient_statistics,
    validate_replay_execution_schedule,
)


class Chunk(torch.nn.Sequential):
    pass


def _schedule_worker(rank, init_file, output_directory):
    torch.distributed.init_process_group("gloo", rank=rank, world_size=2,
                                         init_method="file://" + init_file,
                                         timeout=timedelta(seconds=20))
    results = []
    try:
        cases = [([81], [81]), ([81], [120]), ([81, 120], [120, 81]), ([81], [81, 81])]
        for by_rank in cases:
            traces = [{"step_clipids": [0] * steps} for steps in by_rank[rank]]
            try:
                validate_replay_execution_schedule(traces)
                results.append("passed")
            except RuntimeError as error:
                if "equal execution schedules" not in str(error):
                    raise
                results.append("rejected")
        Path(output_directory, f"rank{rank}.json").write_text(json.dumps(results))
    finally:
        torch.distributed.destroy_process_group()


class FSDPGRPOTest(unittest.TestCase):
    def test_two_rank_schedule_guard_accepts_equal_and_rejects_all_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            init_file = os.path.join(directory, "init")
            torch.multiprocessing.spawn(_schedule_worker, args=(init_file, directory),
                                        nprocs=2, join=True)
            for rank in range(2):
                self.assertEqual(json.loads(Path(directory, f"rank{rank}.json").read_text()),
                                 ["passed", "rejected", "rejected", "rejected"])

    def make_model(self):
        model = torch.nn.Module()
        model.block_chunks = torch.nn.ModuleList([
            Chunk(torch.nn.Linear(2, 2)), Chunk(torch.nn.Linear(2, 2)),
        ])
        model.head = torch.nn.Linear(2, 1)
        return model

    def test_policy_only_overrides_chunks_not_root_or_heads(self):
        model = self.make_model()
        policy = chunk_mixed_precision_policy(Chunk)
        overrides = policy._run_policy(model, set(), {"mixed_precision": None})
        self.assertEqual(set(overrides), set(model.block_chunks))
        for values in overrides.values():
            mp = values["mixed_precision"]
            self.assertEqual(mp.param_dtype, torch.bfloat16)
            self.assertEqual(mp.reduce_dtype, torch.float32)
            self.assertEqual(mp.buffer_dtype, torch.float32)
            self.assertFalse(mp.cast_forward_inputs)
            self.assertFalse(mp.cast_root_forward_inputs)
            self.assertFalse(mp.keep_low_precision_grads)
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))

    def test_forward_dtype_is_distinct_from_master_dtype(self):
        model = self.make_model()
        self.assertEqual(transformer_precision(model)["block_forward_dtype"], "float32")
        policy = chunk_mixed_precision_policy(Chunk)
        for chunk, values in policy._run_policy(model, set(), {}).items():
            chunk.mixed_precision = values["mixed_precision"]
        self.assertEqual(transformer_precision(model), {
            "block_forward_dtype": "bfloat16", "block_master_dtypes": ["float32"],
        })

    def test_each_chunk_gradient_finite_nonzero_and_empty_shards_safe(self):
        model = self.make_model()
        model.block_chunks[0].register_parameter("empty_shard", torch.nn.Parameter(torch.empty(0)))
        x = torch.tensor([[1., 2.]])
        model.head(model.block_chunks[1](model.block_chunks[0](x))).sum().backward()
        rows = chunk_gradient_statistics(model, "cpu", require_nonzero=True)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertTrue(row["finite"])
            self.assertGreater(row["nonzero_gradient_elements"], 0)
            self.assertGreater(row["grad_norm"], 0)
            self.assertEqual(row["trainable_parameters"], 6)
        model.block_chunks[0][0].weight.grad.fill_(float("inf"))
        with self.assertRaises(RuntimeError):
            chunk_gradient_statistics(model, "cpu", require_nonzero=True)

    def test_zero_chunk_gradient_rejected_before_step(self):
        model = self.make_model()
        for parameter in model.block_chunks.parameters():
            parameter.grad = torch.zeros_like(parameter)
        with self.assertRaises(RuntimeError):
            chunk_gradient_statistics(model, "cpu", require_nonzero=True)


if __name__ == "__main__":
    unittest.main()
