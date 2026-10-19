"""CPU checks for the action/reward contract; no simulator or model imports."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pose = load('pose_math_test', ROOT / 'action_aware_grpo/runtime/pose_math.py')
rollout = load('rollout_test', ROOT / 'Worldmodel/runtime/tools/GRPO/generate_candidate_trajectories_real.py')
reward = load('reward_test', ROOT / 'Worldmodel/runtime/tools/GRPO/reward_uavflow.py')


class OnlineContract(unittest.TestCase):
    def test_body_translation_uses_previous_orientation(self):
        actions = [[0, 0, 0, 0, 90, 0], [100, 0, 0, 0, 0, 0]]
        np.testing.assert_allclose(rollout._integrate_actions(actions)[-1][:3], [0, 100, 0], atol=1e-10)
        combined = pose.compose_body_delta([0]*6, [100, 0, 0, 0, 90, 0])
        np.testing.assert_allclose(combined[:3], [100, 0, 0], atol=1e-10)

    def test_full_rotation_and_initial_frame(self):
        start = [120, -30, 400, 10, 70, -15]
        action = [30, 20, -2, -8, 35, 12]
        end = pose.compose_body_delta(start, action)
        rel = pose.world_poses_to_initial([start, end])[-1]
        np.testing.assert_allclose(pose.pose_to_matrix(rel), pose.pose_to_matrix(action), atol=1e-10)

    def test_angle_wrap_in_all6_reward(self):
        metrics = reward._clip_mse([[0, 0, 0, 0, 179, 0]], [[0, 0, 0, 0, -179, 0]])
        self.assertAlmostEqual(metrics['mse_yaw_deg2'], 4.0)
        self.assertAlmostEqual(metrics['mse_all6_mrad'], np.deg2rad(2)**2 / 6)

    def test_world_gt_uses_same_full_initial_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'world.json'
            start = [20, 30, 400, 10, 70, 12]
            delta = [100, 0, 0, 0, 20, 0]
            p.write_text(json.dumps([start, pose.compose_body_delta(start, delta)]))
            relative = reward._gt_in_initial_frame(str(p), 'world')
            np.testing.assert_allclose(pose.pose_to_matrix(relative[-1]), pose.pose_to_matrix(delta), atol=1e-10)
            with self.assertRaises(ValueError):
                reward._gt_in_initial_frame(str(p), 'initial_body')

    def test_reward_uses_observed_complete_trajectory(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            truth = pose.integrate_body_actions([[2, 0, 0, 0, 2, 0]]*48)
            gt_path = folder/'gt.json'
            gt_path.write_text(json.dumps(truth), encoding='utf-8')
            rows = []
            for i in range(2):
                tid = f'task_k{i}'
                target = folder/tid
                target.mkdir()
                trajectory = {
                    'sampling_backend': 'remote_sim', 'pose_source': 'simulator_observed',
                    'relative_poses': {'poses': truth[1:]},
                    'sample_logprob_segments': [-100.0]*3, 'sample_logprob_total': -300.0,
                    'trace_files': ['a.pt', 'b.pt', 'c.pt']}
                (target/'trajectory.json').write_text(json.dumps(trajectory), encoding='utf-8')
                rows.append({'traj_id': tid, 'grpo_group_id': 'same_task', 'video_path': 'reference.mp4',
                             'tarsier2_caption': 'move and turn', 'gt_pose_json': str(gt_path)})
            source = folder/'rows.jsonl'
            source.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
            output = folder/'reward.jsonl'
            command = [sys.executable, str(ROOT/'Worldmodel/runtime/tools/GRPO/reward_uavflow.py'),
                       '--replay_jsonl', str(source), '--trajectory_json_dir', str(folder),
                       '--output_jsonl', str(output), '--require_all_trajectories', '1',
                       '--require_old_logprob', '1']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            replay = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(replay), 6)
            self.assertTrue(all(r['grpo_succ'] == 1 for r in replay))
            self.assertTrue(all(r['grpo_mse']['mse_xyz_m2'] < 1e-18 for r in replay))
            bad = json.loads((folder/'task_k0/trajectory.json').read_text())
            bad['pose_source'] = 'commanded'
            (folder/'task_k0/trajectory.json').write_text(json.dumps(bad), encoding='utf-8')
            failure = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failure.returncode, 0)
            self.assertIn('observed', failure.stderr)


if __name__ == '__main__':
    unittest.main()
