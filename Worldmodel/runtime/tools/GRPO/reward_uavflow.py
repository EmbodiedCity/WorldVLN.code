#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from typing import Any, Dict, List

import numpy as np


def _yaw_wrap_deg(d: np.ndarray) -> np.ndarray:
    return (d + 180.0) % 360.0 - 180.0


def _angles_wrap_deg(d: np.ndarray) -> np.ndarray:
    """Wrap degrees to (-180, 180]."""
    return (d + 180.0) % 360.0 - 180.0


def _load_gt_poses(path: str) -> List[List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list) and obj and isinstance(obj[0], list):
        return [[float(v) for v in row[:6]] for row in obj]
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        out: List[List[float]] = []
        for row in obj:
            c = row.get("commanded", row.get("observed", {})) if isinstance(row, dict) else {}
            out.append(
                [
                    float(c.get("x", 0.0)),
                    float(c.get("y", 0.0)),
                    float(c.get("z", 0.0)),
                    float(c.get("roll", 0.0)),
                    float(c.get("yaw", 0.0)),
                    float(c.get("pitch", 0.0)),
                ]
            )
        return out
    return []


def _to_relative_cmdeg(poses: List[List[float]]) -> List[List[float]]:
    """
    Normalize a pose list to a relative trajectory (cm/deg), anchored at the first pose.
    This makes success-threshold checks robust when input json stores absolute/world poses.
    """
    if not poses:
        return poses
    arr = np.asarray(poses, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 6:
        return poses
    base = arr[0:1, :6].copy()
    out = arr[:, :6].copy()
    out[:, 0:3] = out[:, 0:3] - base[:, 0:3]
    # angle relative diff with wrap
    out[:, 3:6] = _angles_wrap_deg(out[:, 3:6] - base[:, 3:6])
    return [[float(v) for v in row.tolist()] for row in out]


def _gt_in_initial_frame(path: str, frame: str) -> List[List[float]]:
    poses = _load_gt_poses(path)
    if not poses:
        raise ValueError(f"empty ground-truth trajectory: {path}")
    module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "action_aware_grpo", "runtime", "pose_math.py"))
    spec = importlib.util.spec_from_file_location("worldvln_reward_pose_math", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if frame == "world":
        return module.world_poses_to_initial(poses)
    if frame != "initial_body":
        raise ValueError(f"unsupported gt_pose_frame={frame}; expected initial_body or world")
    if not np.allclose(module.pose_to_matrix(poses[0]), np.eye(4), atol=1e-5):
        raise ValueError("initial_body ground truth must start at the identity pose; set gt_pose_frame=world for absolute poses")
    for pose in poses:
        module.pose_to_matrix(pose)  # finite values and six-dimensional shape
    return poses


def _to_m_rad(arr_deg_cm: np.ndarray) -> np.ndarray:
    """
    Input/Output layout: [x, y, z, roll, yaw, pitch]
    - xyz: cm -> m
    - ryp: deg -> rad
    """
    out = arr_deg_cm.astype(np.float64).copy()
    out[:, 0:3] = out[:, 0:3] / 100.0
    out[:, 3:6] = out[:, 3:6] * (math.pi / 180.0)
    return out


def _yaw_wrap_rad(d: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(d), np.cos(d))


def _clip_mse(pred_poses: List[List[float]], gt_poses: List[List[float]]) -> Dict[str, float]:
    n = min(len(pred_poses), len(gt_poses))
    if n <= 0:
        return {
            "mse_all6_mrad": 0.0,
            "mse_xyz_m2": 0.0,
            "mse_yaw_rad2": 0.0,
            "mse_xyz_cm2": 0.0,
            "mse_yaw_deg2": 0.0,
        }
    p_cmdeg = np.asarray(pred_poses[:n], dtype=np.float64)
    g_cmdeg = np.asarray(gt_poses[:n], dtype=np.float64)
    p = _to_m_rad(p_cmdeg)
    g = _to_m_rad(g_cmdeg)
    yaw_diff_rad = _yaw_wrap_rad(p[:, 4] - g[:, 4])
    all_diff = p - g
    all_diff[:, 3:6] = _yaw_wrap_rad(all_diff[:, 3:6])
    mse_all6_mrad = float(np.mean(all_diff ** 2))
    mse_xyz_m2 = float(np.mean((p[:, :3] - g[:, :3]) ** 2))
    mse_yaw_rad2 = float(np.mean(yaw_diff_rad**2))
    # keep cm/deg diagnostics for easy inspection
    yaw_diff_deg = _yaw_wrap_deg(p_cmdeg[:, 4] - g_cmdeg[:, 4])
    mse_xyz_cm2 = float(np.mean((p_cmdeg[:, :3] - g_cmdeg[:, :3]) ** 2))
    mse_yaw_deg2 = float(np.mean(yaw_diff_deg**2))
    return {
        "mse_all6_mrad": mse_all6_mrad,
        "mse_xyz_m2": mse_xyz_m2,
        "mse_yaw_rad2": mse_yaw_rad2,
        "mse_xyz_cm2": mse_xyz_cm2,
        "mse_yaw_deg2": mse_yaw_deg2,
    }


def _reward_from_mse(mse: Dict[str, float], alpha_xyz: float, alpha_yaw: float, alpha_all6: float) -> float:
    """
    Use unit-aligned m/rad losses and an inverse map to avoid underflow collapse.
    val >= 0 ; reward in (0, 1].
    """
    val = (
        alpha_xyz * mse["mse_xyz_m2"]
        + alpha_yaw * mse["mse_yaw_rad2"]
        + alpha_all6 * mse["mse_all6_mrad"]
    )
    return float(1.0 / (1.0 + max(0.0, val)))


def _act_mse_scalar(mse: Dict[str, float], alpha_xyz: float, alpha_yaw: float, alpha_all6: float) -> float:
    """Scalar loss-like metric (>=0, lower is better) used for group-relative act reward."""
    return float(
        alpha_xyz * float(mse.get("mse_xyz_m2", 0.0))
        + alpha_yaw * float(mse.get("mse_yaw_rad2", 0.0))
        + alpha_all6 * float(mse.get("mse_all6_mrad", 0.0))
    )


def _zscore_exp_reward(
    xs: np.ndarray,
    *,
    eps: float,
    zmax: float,
) -> Dict[str, np.ndarray]:
    """
    GRPO-style group-relative shaping:
    - z = (x - mean) / max(std, eps)
    - z_tilde = clip(max(0, z), 0, zmax)
    - r = exp(-z_tilde) in (0, 1]
    Lower x is better; x below mean => z<0 => r=1.
    """
    xs = xs.astype(np.float64)
    mu = float(np.mean(xs)) if xs.size > 0 else 0.0
    sig = float(np.sqrt(np.mean((xs - mu) ** 2))) if xs.size > 0 else 0.0
    denom = max(float(sig), float(eps))
    z = (xs - mu) / denom
    z_tilde = np.maximum(0.0, z)
    if zmax and zmax > 0:
        z_tilde = np.minimum(z_tilde, float(zmax))
    r = np.exp(-z_tilde)
    return {
        "mu": np.asarray([mu], dtype=np.float64),
        "sigma": np.asarray([sig], dtype=np.float64),
        "z": z,
        "z_tilde": z_tilde,
        "r": r,
    }


def _minstd_exp_reward(
    xs: np.ndarray,
    *,
    eps: float,
    zmax: float,
) -> Dict[str, np.ndarray]:
    """
    Alternative group-relative shaping (non-negative, <=1 after exp mapping):
    - xmin = min(x)
    - denom = max(std(x), eps)
    - xminus = (x - xmin) / denom   (>=0)
    - xminus_tilde = clip(xminus, 0, zmax)
    - r = exp(-xminus_tilde) in (0, 1]

    Notes:
    - Lower x is better; the best sample in-group gets r=1.
    - This removes the need for max(0, z) clipping because xminus is non-negative by construction.
    """
    xs = xs.astype(np.float64)
    if xs.size <= 0:
        xmin = 0.0
        mu = 0.0
        sig = 0.0
        denom = max(float(eps), 1.0)
        xminus = xs
    else:
        xmin = float(np.min(xs))
        mu = float(np.mean(xs))
        sig = float(np.sqrt(np.mean((xs - mu) ** 2)))
        denom = max(float(sig), float(eps))
        xminus = (xs - xmin) / denom
    xminus_tilde = np.maximum(0.0, xminus)
    if zmax and zmax > 0:
        xminus_tilde = np.minimum(xminus_tilde, float(zmax))
    r = np.exp(-xminus_tilde)
    return {
        "xmin": np.asarray([xmin], dtype=np.float64),
        "mu": np.asarray([mu], dtype=np.float64),
        "sigma": np.asarray([sig], dtype=np.float64),
        "xminus": xminus,
        "xminus_tilde": xminus_tilde,
        "r": r,
    }


def _loo_adv(r: np.ndarray) -> np.ndarray:
    """Leave-one-out advantage: A_i = r_i - mean_{j!=i}(r_j)."""
    r = r.astype(np.float64)
    k = int(r.size)
    if k <= 1:
        return np.zeros_like(r, dtype=np.float64)
    ssum = float(np.sum(r))
    out = np.zeros_like(r, dtype=np.float64)
    for i in range(k):
        others_mean = (ssum - float(r[i])) / max(1.0, float(k - 1))
        out[i] = float(r[i]) - others_mean
    return out


def _rank01_average_ties(vals: np.ndarray) -> np.ndarray:
    vals = vals.astype(np.float64)
    n = int(vals.shape[0])
    if n <= 1:
        return np.zeros((n,), dtype=np.float64)
    order = np.argsort(vals, kind="mergesort")
    ranks = np.zeros((n,), dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j)
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks / float(max(1, n - 1))


def _centered_rank_adv(vals: np.ndarray) -> np.ndarray:
    rank01 = _rank01_average_ties(vals)
    return (rank01 - 0.5) * 2.0


def _print_final_weight_summary(rows: List[Dict[str, Any]], out_path: str) -> None:
    total = len(rows)
    if total <= 0:
        print(f"[reward_uavflow][summary] empty output: {out_path}")
        return
    adv = np.asarray([float(r.get("grpo_adv_final", 0.0)) for r in rows], dtype=np.float64)
    success_raw = np.asarray([float(r.get("grpo_reward_task_success_raw", 0.0)) for r in rows], dtype=np.float64)
    task_raw = np.asarray([float(r.get("grpo_reward_task_raw", 0.0)) for r in rows], dtype=np.float64)
    pos_frac = float(np.mean(adv > 0.0))
    zero_frac = float(np.mean(adv == 0.0))
    neg_frac = float(np.mean(adv < 0.0))
    success_negative = int(np.sum((success_raw > 0.0) & (adv < 0.0)))
    high_task_negative = int(np.sum((task_raw >= 0.8) & (adv < 0.0)))
    print(
        "[reward_uavflow][summary] "
        f"rows={total} pos_frac={pos_frac:.6f} zero_frac={zero_frac:.6f} neg_frac={neg_frac:.6f} "
        f"success_negative={success_negative} high_task_negative={high_task_negative} out={out_path}"
    )


def _task_success_from_final(
    pred_poses: List[List[float]],
    gt_poses: List[List[float]],
    pos_thresh_m: float,
    yaw_thresh_deg: float,
) -> float:
    if not pred_poses or not gt_poses:
        return 0.0
    p = np.asarray(pred_poses[-1], dtype=np.float64)  # cm/deg
    g = np.asarray(gt_poses[-1], dtype=np.float64)    # cm/deg
    pos_err_m = float(np.linalg.norm((p[:3] - g[:3]) / 100.0))
    yaw_err_deg = float(abs(((p[4] - g[4] + 180.0) % 360.0) - 180.0))
    return 1.0 if (pos_err_m <= float(pos_thresh_m) and yaw_err_deg <= float(yaw_thresh_deg)) else 0.0


def _dense_task_reward_from_final(
    pred_poses: List[List[float]],
    gt_poses: List[List[float]],
    pos_scale_m: float,
    yaw_scale_deg: float,
    pos_weight: float,
    yaw_weight: float,
) -> Dict[str, float]:
    """
    Dense terminal task reward aligned with final xyz + yaw evaluation:
      cost = w_pos * (pos_err_m / pos_scale_m)^2 + w_yaw * (yaw_err_deg / yaw_scale_deg)^2
      reward = 1 / (1 + cost)
    """
    if not pred_poses or not gt_poses:
        return {
            "pos_err_m": 0.0,
            "yaw_err_deg": 180.0,
            "cost": float(max(0.0, pos_weight) * 1e6 + max(0.0, yaw_weight) * 1e6),
            "reward": 0.0,
        }
    p = np.asarray(pred_poses[-1], dtype=np.float64)  # cm/deg
    g = np.asarray(gt_poses[-1], dtype=np.float64)  # cm/deg
    pos_err_m = float(np.linalg.norm((p[:3] - g[:3]) / 100.0))
    yaw_err_deg = float(abs(((p[4] - g[4] + 180.0) % 360.0) - 180.0))
    pos_scale = max(float(pos_scale_m), 1e-6)
    yaw_scale = max(float(yaw_scale_deg), 1e-6)
    cost = float(
        max(0.0, float(pos_weight)) * (pos_err_m / pos_scale) ** 2
        + max(0.0, float(yaw_weight)) * (yaw_err_deg / yaw_scale) ** 2
    )
    return {
        "pos_err_m": pos_err_m,
        "yaw_err_deg": yaw_err_deg,
        "cost": cost,
        "reward": float(1.0 / (1.0 + max(0.0, cost))),
    }


def _compose_task_reward_raw(
    task_mode: str,
    dense_raw: float,
    success_raw: float,
    enable_success_bonus: int,
    task_dense_weight: float,
    task_success_weight: float,
) -> float:
    mode = str(task_mode or "raw_dense").strip().lower()
    if mode == "raw_succ":
        return float(success_raw)
    if int(enable_success_bonus) != 1:
        return float(dense_raw)
    return float(
        max(0.0, float(task_dense_weight)) * float(dense_raw)
        + max(0.0, float(task_success_weight)) * float(success_raw)
    )


def _split_into_clips(poses: List[List[float]], clip_len: int, num_clips: int) -> List[List[List[float]]]:
    """
    Split a full 49f trajectory's relative poses into per-clip segments.

    Typical layout:
    - 49 frames => 48 relative deltas/poses aligned to frames 1..48
    - 3 clips => 16 poses each: [0:16], [16:32], [32:48]
    """
    out: List[List[List[float]]] = []
    L = int(max(0, len(poses)))
    clip_len_i = int(max(1, clip_len))
    n_i = int(max(1, num_clips))
    for ci in range(n_i):
        st = ci * clip_len_i
        ed = min(L, st + clip_len_i)
        out.append(poses[st:ed] if (st < ed) else [])
    return out


def _reward_act_with_clip_decay(
    pred_poses: List[List[float]],
    gt_poses: List[List[float]],
    clip_len: int,
    num_clips: int,
    clip_alpha: float,
    alpha_xyz: float,
    alpha_yaw: float,
    alpha_all6: float,
) -> Dict[str, Any]:
    """
    Compute action-level reward by summing per-clip rewards with temporal decay:
      r_act = r0 * 1 + r1 * alpha + r2 * alpha^2
    """
    pred_clips = _split_into_clips(pred_poses, clip_len=clip_len, num_clips=num_clips)
    gt_clips = _split_into_clips(gt_poses, clip_len=clip_len, num_clips=num_clips)
    r_list: List[float] = []
    mse_list: List[Dict[str, float]] = []
    for ci in range(int(num_clips)):
        mse = _clip_mse(pred_clips[ci], gt_clips[ci])
        r = _reward_from_mse(mse=mse, alpha_xyz=alpha_xyz, alpha_yaw=alpha_yaw, alpha_all6=alpha_all6)
        r_list.append(float(r))
        mse_list.append(mse)
    clip_alpha_f = float(clip_alpha)
    w_list = [float(clip_alpha_f**ci) for ci in range(int(num_clips))]
    r_total = float(sum(r_list[ci] * w_list[ci] for ci in range(int(num_clips))))
    return {
        "reward_act": r_total,
        "reward_act_clips": r_list,
        "reward_act_weights": w_list,
        "mse_clips": mse_list,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay_jsonl", type=str, required=True, help="input replay jsonl")
    ap.add_argument("--trajectory_json_dir", type=str, required=True, help="directory of trajectory.json files")
    ap.add_argument("--output_jsonl", type=str, required=True)
    ap.add_argument(
        "--output_mode",
        type=str,
        default="clip",
        choices=["clip", "traj"],
        help="clip: expand each rollout into 3 clip samples (recommended); traj: keep 1 sample per rollout.",
    )
    # In m/rad domain:
    # - xyz term is in m^2
    # - yaw term is in rad^2
    ap.add_argument("--alpha_xyz", type=float, default=1.0)
    ap.add_argument("--alpha_yaw", type=float, default=1.0)
    ap.add_argument("--alpha_all6", type=float, default=0.2)
    # GRPO-ish reward shaping (group-relative z-score -> exp mapping):
    ap.add_argument(
        "--act_reward_mode",
        type=str,
        default="zscore_exp",
        choices=["inv1p", "zscore_exp", "minstd_exp"],
        help=(
            "inv1p: use 1/(1+val) per-sample; "
            "zscore_exp: group-relative z-score then exp(-max(0,z)); "
            "minstd_exp: (x-min)/std then exp(-x) (non-negative, <=1) before LOO."
        ),
    )
    ap.add_argument("--zscore_eps", type=float, default=1e-6)
    ap.add_argument("--zscore_zmax", type=float, default=10.0)
    ap.add_argument("--enable_ce_reward", type=int, default=1)
    ap.add_argument("--lambda_act", type=float, default=1.0)
    ap.add_argument("--lambda_task", type=float, default=1.0)
    ap.add_argument("--lambda_ce", type=float, default=0.3)
    # Clip-level temporal decay for action reward:
    ap.add_argument("--clip_len", type=int, default=16)
    ap.add_argument("--num_clips", type=int, default=3)
    ap.add_argument("--clip_alpha", type=float, default=0.9)
    # Success thresholds:
    # - clip_* are used for clip-end diagnostics
    # - task_* are used for success-aligned bonus / traj success
    ap.add_argument("--clip_task_pos_thresh_m", type=float, default=0.6)
    ap.add_argument("--clip_task_yaw_thresh_deg", type=float, default=2.5)
    ap.add_argument("--task_pos_thresh_m", type=float, default=3.0)
    ap.add_argument("--task_yaw_thresh_deg", type=float, default=10.0)
    # Dense terminal task reward knobs:
    #   reward = 1 / (1 + w_pos * (pos_err_m / pos_scale_m)^2 + w_yaw * (yaw_err_deg / yaw_scale_deg)^2 )
    ap.add_argument("--task_pos_scale_m", type=float, default=2.0)
    ap.add_argument("--task_yaw_scale_deg", type=float, default=10.0)
    ap.add_argument("--task_pos_weight", type=float, default=1.0)
    ap.add_argument("--task_yaw_weight", type=float, default=1.0)
    ap.add_argument("--task_enable_success_bonus", type=int, default=1)
    ap.add_argument("--task_dense_weight", type=float, default=0.85)
    ap.add_argument("--task_success_weight", type=float, default=0.15)
    # Task reward mode:
    # - raw_dense: use dense + optional success bonus directly
    # - raw_succ: use success-only task reward directly
    # - dense_loo/loo: apply LOO to the raw task reward within a group
    ap.add_argument("--task_reward_mode", type=str, default="raw_dense", choices=["raw_succ", "loo", "raw_dense", "dense_loo"])
    ap.add_argument("--require_old_logprob", type=int, default=1)
    ap.add_argument("--require_all_trajectories", type=int, default=1)
    args = ap.parse_args()

    in_path = os.path.abspath(args.replay_jsonl)
    out_path = os.path.abspath(args.output_jsonl)
    traj_root = os.path.abspath(args.trajectory_json_dir)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows: List[Dict[str, Any]] = []
    with open(in_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    n, ok = 0, 0
    missing_traj = 0
    missing_oldlp = 0
    mode = str(args.output_mode or "clip").strip().lower()
    if mode == "traj":
        # Trajectory-level (legacy): 1 sample per rollout
        for idx, meta in enumerate(rows):
            n += 1
            traj_id = str(meta.get("traj_id", meta.get("id", idx + 1)))
            traj_path = os.path.join(traj_root, str(traj_id), "trajectory.json")
            try:
                if not os.path.exists(traj_path):
                    missing_traj += 1
                    continue
                with open(traj_path, "r", encoding="utf-8") as f:
                    tj = json.load(f)
                if tj.get("sampling_backend") == "remote_sim" and tj.get("pose_source") != "simulator_observed":
                    raise ValueError("remote_sim reward requires observed execution poses")
                pred = tj.get("relative_poses", {}).get("poses", [])
                try:
                    if "sample_logprob_total" not in tj:
                        missing_oldlp += 1
                    meta["grpo_old_logprob"] = float(tj.get("sample_logprob_total", meta.get("grpo_old_logprob", 0.0)))
                except Exception:
                    missing_oldlp += 1
                    pass
                if isinstance(tj.get("trace_files", None), list):
                    meta["grpo_trace_files"] = tj.get("trace_files", [])
                gt_path = meta.get("gt_pose_json", "")
                if not gt_path or not os.path.exists(gt_path):
                    continue
                gt = _gt_in_initial_frame(gt_path, str(meta.get("gt_pose_frame", "initial_body")))
                gt_use = gt[1:] if isinstance(gt, list) and len(gt) > 1 else []
                act_pack = _reward_act_with_clip_decay(
                    pred_poses=pred,
                    gt_poses=gt_use,
                    clip_len=int(args.clip_len),
                    num_clips=int(args.num_clips),
                    clip_alpha=float(args.clip_alpha),
                    alpha_xyz=float(args.alpha_xyz),
                    alpha_yaw=float(args.alpha_yaw),
                    alpha_all6=float(args.alpha_all6),
                )
                succ = _task_success_from_final(
                    pred_poses=pred,
                    gt_poses=gt_use,
                    pos_thresh_m=float(args.task_pos_thresh_m),
                    yaw_thresh_deg=float(args.task_yaw_thresh_deg),
                )
                dense_task = _dense_task_reward_from_final(
                    pred_poses=pred,
                    gt_poses=gt_use,
                    pos_scale_m=float(args.task_pos_scale_m),
                    yaw_scale_deg=float(args.task_yaw_scale_deg),
                    pos_weight=float(args.task_pos_weight),
                    yaw_weight=float(args.task_yaw_weight),
                )
                task_mode = str(args.task_reward_mode or "raw_dense").strip().lower()
                dense_raw = float(dense_task["reward"])
                success_raw = float(succ)
                task_raw = _compose_task_reward_raw(
                    task_mode=task_mode,
                    dense_raw=dense_raw,
                    success_raw=success_raw,
                    enable_success_bonus=int(args.task_enable_success_bonus),
                    task_dense_weight=float(args.task_dense_weight),
                    task_success_weight=float(args.task_success_weight),
                )
                meta["grpo_reward_act"] = float(act_pack["reward_act"])
                meta["grpo_reward_act_clips"] = act_pack["reward_act_clips"]
                meta["grpo_reward_act_weights"] = act_pack["reward_act_weights"]
                meta["grpo_succ"] = float(succ)
                meta["grpo_task_final_pos_err_m"] = float(dense_task["pos_err_m"])
                meta["grpo_task_final_yaw_err_deg"] = float(dense_task["yaw_err_deg"])
                meta["grpo_task_final_cost"] = float(dense_task["cost"])
                meta["grpo_reward_task_dense_raw"] = float(dense_raw)
                meta["grpo_reward_task_success_raw"] = float(success_raw)
                meta["grpo_reward_task_raw"] = float(task_raw)
                meta["grpo_mse"] = _clip_mse(pred, gt_use)
                meta["grpo_mse_clips"] = act_pack["mse_clips"]
                ok += 1
            except Exception as exc:
                raise RuntimeError(f"invalid reward input for trajectory {meta.get('traj_id')}: {exc}") from exc

        # task reward mode for traj-level: raw_succ or loo within original groups
        groups: Dict[str, List[int]] = {}
        for i, meta in enumerate(rows):
            gid = str(meta.get("grpo_group_id", ""))
            if not gid:
                gid = f"__single_{i}"
            groups.setdefault(gid, []).append(i)
        task_mode = str(args.task_reward_mode or "raw_dense").strip().lower()
        if task_mode in ("loo", "dense_loo"):
            for _, inds in groups.items():
                if len(inds) <= 1:
                    for i in inds:
                        rows[i]["grpo_reward_task"] = 0.0
                    continue
                raws = np.asarray([float(rows[i].get("grpo_reward_task_raw", 0.0)) for i in inds], dtype=np.float64)
                advs = _loo_adv(raws)
                for off, i in enumerate(inds):
                    rows[i]["grpo_reward_task"] = float(advs[off])
        else:
            for meta in rows:
                meta["grpo_reward_task"] = float(meta.get("grpo_reward_task_raw", 0.0))
        for meta in rows:
            ra = float(meta.get("grpo_reward_act", meta.get("grpo_reward", 0.0)))
            rt = float(meta.get("grpo_reward_task", 0.0))
            meta["grpo_reward"] = float(ra + rt)
        for _, inds in groups.items():
            scores = np.asarray([float(rows[i].get("grpo_reward", 0.0)) for i in inds], dtype=np.float64)
            advs = _centered_rank_adv(scores)
            rank01 = _rank01_average_ties(scores)
            for off, i in enumerate(inds):
                rows[i]["grpo_score_final"] = float(scores[off])
                rows[i]["grpo_rank_final"] = float(rank01[off])
                rows[i]["grpo_adv_final"] = float(advs[off])

        out_rows = rows
    else:
        # Clip-level (recommended): expand each rollout into num_clips clip samples.
        out_rows: List[Dict[str, Any]] = []
        clip_len = int(args.clip_len)
        num_clips = int(args.num_clips)
        clip_alpha = float(args.clip_alpha)
        task_id_key = "grpo_group_id"

        for idx, meta in enumerate(rows):
            n += 1
            base_traj_id = str(meta.get("traj_id", meta.get("id", idx + 1)))
            traj_path = os.path.join(traj_root, str(base_traj_id), "trajectory.json")
            try:
                if not os.path.exists(traj_path):
                    missing_traj += 1
                    continue
                with open(traj_path, "r", encoding="utf-8") as f:
                    tj = json.load(f)
                if tj.get("sampling_backend") == "remote_sim" and tj.get("pose_source") != "simulator_observed":
                    raise ValueError("remote_sim reward requires observed execution poses")
                pred_all = tj.get("relative_poses", {}).get("poses", [])
                if len(pred_all) != clip_len * num_clips:
                    raise ValueError("reward requires a complete observed trajectory")
                seg_lp = tj.get("sample_logprob_segments", None)
                seg_tr = tj.get("trace_files", None)
                if not isinstance(seg_lp, list) or len(seg_lp) < num_clips:
                    missing_oldlp += 1
                    seg_lp = [tj.get("sample_logprob_total", meta.get("grpo_old_logprob", 0.0))] * num_clips
                if not isinstance(seg_tr, list) or len(seg_tr) < num_clips:
                    seg_tr = [[] for _ in range(num_clips)]
                gt_path = meta.get("gt_pose_json", "")
                if not gt_path or not os.path.exists(gt_path):
                    continue
                gt_rel = _gt_in_initial_frame(gt_path, str(meta.get("gt_pose_frame", "initial_body")))
                gt_use_all = gt_rel[1:] if isinstance(gt_rel, list) and len(gt_rel) > 1 else []

                gid_base = str(meta.get(task_id_key, "")) or f"task_{idx:06d}"

                # Whole-trajectory success (diagnostics) using traj-level thresholds.
                succ_traj = 0.0
                traj_dense_task = {
                    "pos_err_m": 0.0,
                    "yaw_err_deg": 180.0,
                    "cost": 1e6,
                    "reward": 0.0,
                }
                traj_task_raw = 0.0
                try:
                    if len(pred_all) > 0 and len(gt_use_all) > 0:
                        succ_traj = _task_success_from_final(
                            pred_poses=[pred_all[-1]],
                            gt_poses=[gt_use_all[-1]],
                            pos_thresh_m=float(args.task_pos_thresh_m),
                            yaw_thresh_deg=float(args.task_yaw_thresh_deg),
                        )
                        traj_dense_task = _dense_task_reward_from_final(
                            pred_poses=[pred_all[-1]],
                            gt_poses=[gt_use_all[-1]],
                            pos_scale_m=float(args.task_pos_scale_m),
                            yaw_scale_deg=float(args.task_yaw_scale_deg),
                            pos_weight=float(args.task_pos_weight),
                            yaw_weight=float(args.task_yaw_weight),
                        )
                        traj_task_raw = _compose_task_reward_raw(
                            task_mode=str(args.task_reward_mode or "raw_dense").strip().lower(),
                            dense_raw=float(traj_dense_task["reward"]),
                            success_raw=float(succ_traj),
                            enable_success_bonus=int(args.task_enable_success_bonus),
                            task_dense_weight=float(args.task_dense_weight),
                            task_success_weight=float(args.task_success_weight),
                        )
                except Exception:
                    succ_traj = 0.0
                    traj_dense_task = {
                        "pos_err_m": 0.0,
                        "yaw_err_deg": 180.0,
                        "cost": 1e6,
                        "reward": 0.0,
                    }
                    traj_task_raw = 0.0

                for clip_pos in range(1, num_clips + 1):
                    st = (clip_pos - 1) * clip_len
                    ed = st + clip_len
                    pred_seg = pred_all[st:ed]
                    gt_seg = gt_use_all[st:ed]
                    mse = _clip_mse(pred_seg, gt_seg)
                    act_mse_scalar = _act_mse_scalar(
                        mse=mse,
                        alpha_xyz=float(args.alpha_xyz),
                        alpha_yaw=float(args.alpha_yaw),
                        alpha_all6=float(args.alpha_all6),
                    )
                    r_act_inv = _reward_from_mse(
                        mse=mse,
                        alpha_xyz=float(args.alpha_xyz),
                        alpha_yaw=float(args.alpha_yaw),
                        alpha_all6=float(args.alpha_all6),
                    )
                    w = float(clip_alpha ** (clip_pos - 1))
                    # r_act will be finalized after we have group stats (z-score-exp mode).
                    r_act = float(r_act_inv * w)

                    # Dense reward and success bonus are both aligned with the current clip end.
                    # Whole-trajectory success is kept separately for diagnostics only.
                    succ = 0.0
                    dense_task = {
                        "pos_err_m": 0.0,
                        "yaw_err_deg": 180.0,
                        "cost": 1e6,
                        "reward": 0.0,
                    }
                    if len(pred_all) >= ed and len(gt_use_all) >= ed:
                        succ = _task_success_from_final(
                            pred_poses=[pred_all[ed - 1]],
                            gt_poses=[gt_use_all[ed - 1]],
                            pos_thresh_m=float(args.clip_task_pos_thresh_m),
                            yaw_thresh_deg=float(args.clip_task_yaw_thresh_deg),
                        )
                        dense_task = _dense_task_reward_from_final(
                            pred_poses=[pred_all[ed - 1]],
                            gt_poses=[gt_use_all[ed - 1]],
                            pos_scale_m=float(args.task_pos_scale_m),
                            yaw_scale_deg=float(args.task_yaw_scale_deg),
                            pos_weight=float(args.task_pos_weight),
                            yaw_weight=float(args.task_yaw_weight),
                        )

                    # Use 17-frame windows to satisfy 4n+1 rule: [0..16], [16..32], [32..48]
                    begin_frame = (clip_pos - 1) * clip_len
                    end_frame = begin_frame + clip_len
                    clip_meta = dict(meta)
                    clip_meta["begin_frame_id"] = int(begin_frame)
                    clip_meta["end_frame_id"] = int(end_frame)
                    clip_meta["traj_id"] = f"{base_traj_id}_c{clip_pos}"
                    clip_meta["grpo_group_id"] = f"{gid_base}_clip{clip_pos}"
                    clip_meta["grpo_clip_id"] = int(clip_pos)
                    clip_meta["grpo_traj_group_id"] = str(gid_base)
                    clip_meta["grpo_old_logprob"] = float(seg_lp[clip_pos - 1])
                    clip_meta["grpo_trace_files"] = [seg_tr[clip_pos - 1]] if seg_tr[clip_pos - 1] else []
                    clip_meta["grpo_act_mse_scalar"] = float(act_mse_scalar)
                    clip_meta["grpo_reward_act_inv_raw"] = float(r_act_inv)
                    clip_meta["grpo_reward_act"] = float(r_act)
                    clip_meta["grpo_succ"] = float(succ)
                    clip_meta["grpo_succ_traj"] = float(succ_traj)
                    clip_meta["grpo_traj_final_pos_err_m"] = float(traj_dense_task["pos_err_m"])
                    clip_meta["grpo_traj_final_yaw_err_deg"] = float(traj_dense_task["yaw_err_deg"])
                    clip_meta["grpo_traj_final_cost"] = float(traj_dense_task["cost"])
                    clip_meta["grpo_reward_task_traj_dense_raw"] = float(traj_dense_task["reward"])
                    clip_meta["grpo_reward_task_traj_success_raw"] = float(succ_traj)
                    clip_meta["grpo_reward_task_traj_raw"] = float(traj_task_raw)
                    clip_meta["grpo_task_final_pos_err_m"] = float(dense_task["pos_err_m"])
                    clip_meta["grpo_task_final_yaw_err_deg"] = float(dense_task["yaw_err_deg"])
                    clip_meta["grpo_task_final_cost"] = float(dense_task["cost"])
                    clip_meta["grpo_mse"] = mse
                    clip_meta["grpo_ce_nll"] = float(max(0.0, -float(clip_meta.get("grpo_old_logprob", 0.0))))
                    clip_meta["grpo_reward_ce_raw"] = 0.0
                    clip_meta["grpo_reward_ce_adv"] = 0.0
                    dense_raw = float(dense_task["reward"])
                    success_raw = float(succ)
                    task_raw = _compose_task_reward_raw(
                        task_mode=str(args.task_reward_mode or "raw_dense").strip().lower(),
                        dense_raw=dense_raw,
                        success_raw=success_raw,
                        enable_success_bonus=int(args.task_enable_success_bonus),
                        task_dense_weight=float(args.task_dense_weight),
                        task_success_weight=float(args.task_success_weight),
                    )
                    clip_meta["grpo_reward_task_dense_raw"] = float(dense_raw)
                    clip_meta["grpo_reward_task_success_raw"] = float(success_raw)
                    clip_meta["grpo_reward_task_raw"] = float(task_raw)
                    clip_meta["grpo_reward_task"] = 0.0
                    clip_meta["grpo_reward"] = float(r_act)  # will add task after LOO
                    clip_meta["grpo_reward_decay"] = float(w)
                    out_rows.append(clip_meta)
                    ok += 1
            except Exception as exc:
                raise RuntimeError(f"invalid reward input for trajectory {meta.get('traj_id')}: {exc}") from exc

        # Group LOO baseline for task reward at clip group granularity.
        groups: Dict[str, List[int]] = {}
        for i, meta in enumerate(out_rows):
            gid = str(meta.get("grpo_group_id", "")) or f"__single_{i}"
            groups.setdefault(gid, []).append(i)

        task_mode = str(args.task_reward_mode or "raw_dense").strip().lower()
        for _, inds in groups.items():
            task_raws = np.asarray([float(out_rows[i].get("grpo_reward_task_raw", 0.0)) for i in inds], dtype=np.float64)
            if task_mode in ("loo", "dense_loo"):
                task_vals = _loo_adv(task_raws) if len(inds) > 1 else np.zeros_like(task_raws, dtype=np.float64)
            else:
                task_vals = task_raws
            for off, i in enumerate(inds):
                w = float(out_rows[i].get("grpo_reward_decay", 1.0))
                out_rows[i]["grpo_reward_task_adv"] = float(task_vals[off] if task_mode in ("loo", "dense_loo") else 0.0)
                out_rows[i]["grpo_reward_task"] = float(task_vals[off] * w)
                out_rows[i]["grpo_reward"] = float(out_rows[i].get("grpo_reward_act", 0.0) + out_rows[i].get("grpo_reward_task", 0.0))

        # Act reward shaping and optional CE reward shaping are group-relative (GRPO-ish).
        act_mode = str(args.act_reward_mode or "zscore_exp").strip().lower()
        do_ce = int(args.enable_ce_reward) == 1
        eps = float(args.zscore_eps)
        zmax = float(args.zscore_zmax)
        for _, inds in groups.items():
            idx = np.asarray(inds, dtype=np.int64)
            # ---- act: based on mse scalar (lower better) ----
            act_x = np.asarray([float(out_rows[i].get("grpo_act_mse_scalar", 0.0)) for i in idx], dtype=np.float64)
            if act_mode == "zscore_exp":
                pack = _zscore_exp_reward(act_x, eps=eps, zmax=zmax)
                r_nodecay = pack["r"]
                a_nodecay = _loo_adv(r_nodecay)
                for j, i in enumerate(idx.tolist()):
                    w = float(out_rows[i].get("grpo_reward_decay", 1.0))
                    out_rows[i]["grpo_reward_act_raw"] = float(r_nodecay[j])
                    out_rows[i]["grpo_reward_act_adv"] = float(a_nodecay[j] * w)
                    # Use LOO advantage as act reward (more GRPO-like).
                    out_rows[i]["grpo_reward_act"] = float(a_nodecay[j] * w)
                    out_rows[i]["grpo_reward_act_z"] = float(pack["z"][j])
                    out_rows[i]["grpo_reward_act_mu"] = float(pack["mu"][0])
                    out_rows[i]["grpo_reward_act_sigma"] = float(pack["sigma"][0])
            elif act_mode == "minstd_exp":
                pack = _minstd_exp_reward(act_x, eps=eps, zmax=zmax)
                r_nodecay = pack["r"]
                a_nodecay = _loo_adv(r_nodecay)
                for j, i in enumerate(idx.tolist()):
                    w = float(out_rows[i].get("grpo_reward_decay", 1.0))
                    out_rows[i]["grpo_reward_act_raw"] = float(r_nodecay[j])
                    out_rows[i]["grpo_reward_act_adv"] = float(a_nodecay[j] * w)
                    out_rows[i]["grpo_reward_act"] = float(a_nodecay[j] * w)
                    out_rows[i]["grpo_reward_act_xminus"] = float(pack["xminus"][j])
                    out_rows[i]["grpo_reward_act_xmin"] = float(pack["xmin"][0])
                    out_rows[i]["grpo_reward_act_mu"] = float(pack["mu"][0])
                    out_rows[i]["grpo_reward_act_sigma"] = float(pack["sigma"][0])
            else:
                # inv1p legacy: keep r_act_inv_raw * decay (already filled)
                for i in idx.tolist():
                    out_rows[i]["grpo_reward_act_raw"] = float(out_rows[i].get("grpo_reward_act_inv_raw", 0.0))
                    out_rows[i]["grpo_reward_act_adv"] = 0.0

            # ---- ce: based on NLL from old logprob (lower better) ----
            if do_ce:
                ce_x = np.asarray([float(out_rows[i].get("grpo_ce_nll", 0.0)) for i in idx], dtype=np.float64)
                # Keep CE shaping in sync with act shaping when act_mode is minstd_exp.
                pack_ce = _minstd_exp_reward(ce_x, eps=eps, zmax=zmax) if act_mode == "minstd_exp" else _zscore_exp_reward(ce_x, eps=eps, zmax=zmax)
                r_ce_raw_nodecay = pack_ce["r"]
                a_ce_nodecay = _loo_adv(r_ce_raw_nodecay)
                for j, i in enumerate(idx.tolist()):
                    w = float(out_rows[i].get("grpo_reward_decay", 1.0))
                    out_rows[i]["grpo_reward_ce_raw"] = float(r_ce_raw_nodecay[j] * w)
                    out_rows[i]["grpo_reward_ce_adv"] = float(a_ce_nodecay[j] * w)
                    if act_mode == "minstd_exp":
                        out_rows[i]["grpo_reward_ce_xminus"] = float(pack_ce["xminus"][j])
                        out_rows[i]["grpo_reward_ce_xmin"] = float(pack_ce["xmin"][0])
                    else:
                        out_rows[i]["grpo_reward_ce_z"] = float(pack_ce["z"][j])
                    out_rows[i]["grpo_reward_ce_mu"] = float(pack_ce["mu"][0])
                    out_rows[i]["grpo_reward_ce_sigma"] = float(pack_ce["sigma"][0])

            # ---- final combined reward ----
            for i in idx.tolist():
                r_act = float(out_rows[i].get("grpo_reward_act", 0.0))
                r_task = float(out_rows[i].get("grpo_reward_task", 0.0))
                r_ce = float(out_rows[i].get("grpo_reward_ce_adv", 0.0)) if do_ce else 0.0
                out_rows[i]["grpo_reward"] = float(float(args.lambda_act) * r_act + float(args.lambda_task) * r_task + float(args.lambda_ce) * r_ce)
                # Reference logprob for KL anchor (default to behavior/old logprob for offline stage).
                if "grpo_ref_logprob" not in out_rows[i]:
                    out_rows[i]["grpo_ref_logprob"] = float(out_rows[i].get("grpo_old_logprob", 0.0))
            scores = np.asarray([float(out_rows[i].get("grpo_reward", 0.0)) for i in idx], dtype=np.float64)
            score_pos = np.clip(scores, 0.0, None)
            traj_gate = np.asarray(
                [
                    1.0
                    if float(out_rows[i].get("grpo_succ_traj", 0.0)) > 0.0
                    else np.clip(float(out_rows[i].get("grpo_reward_task_traj_raw", 0.0)), 0.0, 1.0)
                    for i in idx.tolist()
                ],
                dtype=np.float64,
            )
            final_w = np.clip(score_pos * traj_gate, 0.0, 1.0)
            rank01 = _rank01_average_ties(scores)
            for j, i in enumerate(idx.tolist()):
                out_rows[i]["grpo_score_final"] = float(scores[j])
                out_rows[i]["grpo_score_final_raw"] = float(scores[j])
                out_rows[i]["grpo_score_final_pos"] = float(score_pos[j])
                out_rows[i]["grpo_traj_gate"] = float(traj_gate[j])
                out_rows[i]["grpo_rank_final"] = float(rank01[j])
                out_rows[i]["grpo_adv_final"] = float(final_w[j])

    if not out_rows:
        raise RuntimeError("reward produced no valid replay rows")
    _print_final_weight_summary(out_rows, out_path)
    with open(out_path, "w", encoding="utf-8") as fout:
        for meta in out_rows:
            fout.write(json.dumps(meta, ensure_ascii=False) + "\n")

    if int(args.require_all_trajectories) == 1 and int(missing_traj) > 0:
        raise RuntimeError(f"strict mode: missing trajectory.json for {missing_traj} rows")
    if int(args.require_old_logprob) == 1 and int(missing_oldlp) > 0:
        raise RuntimeError(f"strict mode: missing sample_logprob_total for {missing_oldlp} rows")
    print(f"[reward_uavflow] processed={n}, updated={ok}, out={out_path}")


if __name__ == "__main__":
    main()

