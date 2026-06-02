#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import types
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

def _count_candidates(candidates_jsonl: str, num_shards: int, shard_id: int) -> Tuple[int, int]:
    total_global = 0
    total_shard = 0
    global_idx = -1
    with open(os.path.abspath(candidates_jsonl), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            global_idx += 1
            total_global += 1
            if (global_idx % max(1, int(num_shards))) == int(shard_id):
                total_shard += 1
    return total_global, total_shard


def _load_api_module(py_path: str):
    spec = importlib.util.spec_from_file_location("rl_infinity_api_mod", py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    # dataclass in target module relies on sys.modules[cls.__module__]
    # during module execution; register before exec_module.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _integrate_actions(actions: List[List[float]]) -> List[List[float]]:
    pose = [0.0] * 6
    out = []
    for a in actions:
        if len(a) != 6:
            continue
        pose = [pose[i] + float(a[i]) for i in range(6)]
        out.append(list(pose))
    return out


def _all_finite_6d(actions: List[List[float]]) -> bool:
    if not isinstance(actions, list) or len(actions) <= 0:
        return False
    for a in actions:
        if not (isinstance(a, list) and len(a) == 6):
            return False
        for v in a:
            try:
                x = float(v)
            except Exception:
                return False
            if not math.isfinite(x):
                return False
    return True


def _all_finite(xs: List[float]) -> bool:
    for v in xs:
        try:
            x = float(v)
        except Exception:
            return False
        if not math.isfinite(x):
            return False
    return True


def _require_trace_ce_ok(trace_paths: List[str]) -> None:
    """
    Strict GRPO safety guard:
    When StageB uses trace_ce for new_logprob, StageA must record old_logprob in the same definition
    (teacher-forcing CE/full-softmax). If StageA falls back to sampling-time logprob, PPO ratio/KL becomes
    inconsistent and can collapse video quality.
    """
    mode = (os.environ.get("INFINITY_STAGEA_OLD_LOGPROB_MODE", "") or "").strip().lower()
    strict = int((os.environ.get("INFINITY_STAGEA_OLD_LOGPROB_STRICT", "0") or "0").strip() or "0") == 1
    if not (mode == "trace_ce" and strict):
        return
    import torch  # local import to keep startup light

    for p in trace_paths:
        if not p or (not os.path.exists(p)):
            continue
        tr = torch.load(p, map_location="cpu")
        if tr.get("sample_logprob_trace_ce", None) is None:
            raise RuntimeError(f"trace_ce strict: missing sample_logprob_trace_ce in trace: {p}")


def _make_req(**kwargs):
    obj = types.SimpleNamespace()
    for k, v in kwargs.items():
        setattr(obj, k, v)
    return obj


def _http_post_json(url: str, payload: Dict[str, Any], *, timeout_s: int) -> Dict[str, Any]:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        str(url),
        data=raw,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")
        except Exception:
            detail = str(e)
        raise RuntimeError(f"HTTP {e.code} calling {url}: {detail[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"request to {url} failed: {e}") from e
    obj = json.loads(body or "{}")
    if not isinstance(obj, dict):
        raise RuntimeError(f"non-dict response from {url}: {type(obj)}")
    return obj


def _task_id_from_video_path(video_path: str) -> str:
    return os.path.basename(os.path.dirname(os.path.abspath(video_path)))


def _build_obj_info_from_task_json(task_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "obj_id" not in task_json or "use_obj" not in task_json:
        return None
    if "target_pos" in task_json and isinstance(task_json["target_pos"], list) and len(task_json["target_pos"]) == 6:
        obj_pos = [float(x) for x in task_json["target_pos"][:3]]
        obj_rot = [float(x) for x in task_json["target_pos"][3:]]
    else:
        raw_pos = task_json.get("obj_pos", None)
        raw_rot = task_json.get("obj_rot", [0, 0, 0])
        if not (isinstance(raw_pos, list) and len(raw_pos) >= 3):
            return None
        obj_pos = [float(x) for x in raw_pos[:3]]
        obj_rot = [float(x) for x in (raw_rot[:3] if isinstance(raw_rot, list) else [0, 0, 0])]
    return {
        "use_obj": int(task_json["use_obj"]),
        "obj_id": int(task_json["obj_id"]),
        "obj_pos": obj_pos,
        "obj_rot": obj_rot,
    }


def _resolve_uavflow_task_meta(row: Dict[str, Any], video_path: str, task_json_root: str) -> Dict[str, Any]:
    task_id = str(row.get("task_id", "") or "").strip() or _task_id_from_video_path(video_path)
    task_json_path = os.path.join(os.path.abspath(task_json_root), f"{task_id}.json")
    if not os.path.exists(task_json_path):
        raise FileNotFoundError(f"missing UAV-Flow task json for task_id={task_id}: {task_json_path}")
    with open(task_json_path, "r", encoding="utf-8") as f:
        task_json = json.load(f)
    if not isinstance(task_json, dict):
        raise ValueError(f"bad task json (expect dict): {task_json_path}")
    initial_pos = task_json.get("initial_pos", None)
    if not (isinstance(initial_pos, list) and len(initial_pos) >= 6):
        raise ValueError(f"bad initial_pos in task json: {task_json_path}")
    instruction = str(task_json.get("instruction") or task_json.get("instruction_unified") or row.get("tarsier2_caption") or row.get("instruction") or "").strip()
    if not instruction:
        raise ValueError(f"empty instruction in task json: {task_json_path}")
    return {
        "task_id": task_id,
        "instruction": instruction,
        "initial_pose": [float(x) for x in initial_pos[:6]],
        "obj_info": _build_obj_info_from_task_json(task_json),
        "task_json_path": task_json_path,
    }


def _sim_reset(
    *,
    base_url: str,
    session_id: str,
    traj_id: str,
    row: Dict[str, Any],
    prompt: str,
    seed: int,
    timeout_s: int,
    task_json_root: str,
) -> Dict[str, Any]:
    video_path = str(row.get("video_path", "") or "")
    task_meta = _resolve_uavflow_task_meta(row, video_path, task_json_root)
    payload = {
        "session_id": str(session_id),
        "traj_id": str(traj_id),
        "prompt": str(prompt or task_meta["instruction"]),
        "seed": int(seed),
        "video_path": video_path,
        "gt_pose_json": str(row.get("gt_pose_json", "") or ""),
        "task_id": str(task_meta["task_id"]),
        "initial_pose": task_meta["initial_pose"],
        "obj_info": task_meta["obj_info"],
    }
    return _http_post_json(base_url.rstrip("/") + "/reset", payload, timeout_s=int(timeout_s))


def _sim_step_actions(
    *,
    base_url: str,
    session_id: str,
    actions: List[List[float]],
    segment_index: int,
    timeout_s: int,
) -> Dict[str, Any]:
    payload = {
        "session_id": str(session_id),
        "actions": actions,
        "segment_index": int(segment_index),
    }
    return _http_post_json(base_url.rstrip("/") + "/step_actions", payload, timeout_s=int(timeout_s))


def _run_remote_sim_rollout(
    *,
    mod,
    row: Dict[str, Any],
    traj_id: str,
    session_id: str,
    prompt: str,
    seed: int,
    simulator_base_url: str,
    simulator_timeout_s: int,
    action_head_batch_size: int,
    action_head_stride: int,
    action_head_pre_resize_hw: int,
    task_json_root: str,
) -> Tuple[List[Any], Dict[str, Any]]:
    reset_resp = _sim_reset(
        base_url=simulator_base_url,
        session_id=session_id,
        traj_id=traj_id,
        row=row,
        prompt=prompt,
        seed=seed,
        timeout_s=int(simulator_timeout_s),
        task_json_root=task_json_root,
    )
    init_images = reset_resp.get("images_base64", None)
    if not isinstance(init_images, list) or len(init_images) != 1:
        raise RuntimeError("simulator reset must return exactly one initial frame")

    sim_world_poses: List[Any] = []
    if isinstance(reset_resp.get("world_poses", None), list):
        sim_world_poses.extend(reset_resp["world_poses"])

    responses: List[Any] = []
    seg_exec_meta: List[Dict[str, Any]] = []
    images_for_model = list(init_images)
    timings: Dict[str, float] = {}
    for seg_i in range(3):
        t0 = time.perf_counter()
        req = _make_req(
            session_id=session_id,
            instruction=prompt if seg_i == 0 else None,
            prompt=None,
            negative_prompt="",
            images_base64=images_for_model,
            reset_session=(seg_i == 0),
            action_head_mode="actionhead_ref_vit",
            action_head_batch_size=int(action_head_batch_size),
            action_head_stride=int(action_head_stride),
            action_head_pre_resize_hw=int(action_head_pre_resize_hw),
            allow_future_segments=True,
            prefix_mode=False,
            allow_future_last_segment=True,
            seed=int(seed),
            debug=False,
        )
        resp = mod._predict_delta_actions_impl(req)
        timings[f"seg{seg_i:02d}_sec"] = time.perf_counter() - t0
        responses.append(resp)

        seg_actions = getattr(resp, "actions", None)
        if not _all_finite_6d(seg_actions):
            raise RuntimeError(f"invalid simulator-backend actions for segment {seg_i}")

        step_resp = _sim_step_actions(
            base_url=simulator_base_url,
            session_id=session_id,
            actions=[[float(v) for v in action[:6]] for action in seg_actions],
            segment_index=seg_i,
            timeout_s=int(simulator_timeout_s),
        )
        seg_exec_meta.append(
            {
                "segment_index": int(seg_i),
                "frame_indices": step_resp.get("frame_indices", []),
                "done": bool(step_resp.get("done", False)),
            }
        )
        if isinstance(step_resp.get("world_poses", None), list):
            sim_world_poses.extend(step_resp["world_poses"])
        if seg_i < 2:
            images_next = step_resp.get("images_base64", None)
            if not isinstance(images_next, list) or len(images_next) != 16:
                raise RuntimeError(f"simulator step {seg_i} must return 16 frames for the next model call")
            images_for_model = list(images_next)

    task_meta = _resolve_uavflow_task_meta(row, str(row.get("video_path", "") or ""), task_json_root)
    return responses, {
        "sampling_backend": "remote_sim",
        "simulator_base_url": str(simulator_base_url),
        "simulator_session_id": str(session_id),
        "simulator_world_poses": sim_world_poses,
        "simulator_segment_exec": seg_exec_meta,
        "task_id": str(task_meta["task_id"]),
        "task_json_name": os.path.basename(str(task_meta["task_json_path"])),
        **timings,
    }


def _disable_api_cache_dump(mod) -> None:
    # Disable heavy debug/cache writes in api server for offline rollout speed/storage.
    def _noop(*_args, **_kwargs):
        return None

    if hasattr(mod, "_save_latent_tensor"):
        mod._save_latent_tensor = _noop
    if hasattr(mod, "_save_latent_video_clip"):
        mod._save_latent_video_clip = _noop
    if hasattr(mod, "_save_pred_video"):
        mod._save_pred_video = _noop
    if hasattr(mod, "infinity_save_video"):
        mod.infinity_save_video = None


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_infinity_repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    default_package_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    default_repo_root = os.path.abspath(os.path.join(default_package_root, ".."))
    default_actionhead_repo_root = os.path.join(default_repo_root, "Worldmodel", "action_decoder", "actionhead_runtime")
    default_task_json_root = os.path.join(default_package_root, "data", "uavflow_tasks", "select_from_train_jsons")

    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates_jsonl", type=str, required=True)
    ap.add_argument("--trajectory_root", type=str, required=True)
    ap.add_argument("--api_py", type=str, required=True, help="Path to action_aware_grpo/grpo_server.py")
    ap.add_argument("--infinity_server_config", type=str, required=True, help="Path to action_aware_grpo/config.json")
    ap.add_argument("--infinity_ckpt", type=str, required=True)
    ap.add_argument(
        "--infinity_repo_root",
        type=str,
        default=default_infinity_repo_root,
        help="Path to Infinity repo root used by api server imports.",
    )
    ap.add_argument("--actionhead_ckpt", type=str, required=True)
    ap.add_argument("--actionhead_run_config", type=str, required=True)
    ap.add_argument(
        "--actionhead_repo_root",
        type=str,
        default=default_actionhead_repo_root,
        help="Path to actionhead repo root used by api server.",
    )
    ap.add_argument("--num_frames", type=int, default=49)
    ap.add_argument("--action_head_batch_size", type=int, default=8)
    ap.add_argument("--action_head_stride", type=int, default=1)
    ap.add_argument("--action_head_pre_resize_hw", type=int, default=0)
    ap.add_argument("--dump_debug_cache", type=int, default=0, help="1=keep api cache mp4/pt, 0=disable dump")
    ap.add_argument("--failed_jsonl", type=str, default="", help="Optional path to save failed traj rows")
    ap.add_argument("--timing_jsonl", type=str, default="", help="Optional path to save per-traj timing records")
    ap.add_argument("--num_shards", type=int, default=1, help="Total shard count for parallel rollout.")
    ap.add_argument("--shard_id", type=int, default=0, help="Current shard id in [0, num_shards).")
    ap.add_argument(
        "--max_retry",
        type=int,
        default=int(os.environ.get("INFINITY_ROLLOUT_MAX_RETRY", "3")),
        help="Max retry count when rollout produces invalid numbers (NaN/Inf/missing).",
    )
    ap.add_argument(
        "--retry_seed_step",
        type=int,
        default=int(os.environ.get("INFINITY_ROLLOUT_RETRY_SEED_STEP", "9973")),
        help="Seed increment per retry (resample).",
    )
    ap.add_argument(
        "--progress_every_n",
        type=int,
        default=int(os.environ.get("STAGEA_PROGRESS_EVERY_N", "10")),
        help="Print shard progress every N processed candidates.",
    )
    ap.add_argument(
        "--rollout_backend",
        type=str,
        default=str(os.environ.get("UAVFLOW_STAGEA_ROLLOUT_BACKEND", "remote_sim")),
        choices=["remote_sim"],
        help="StageA open-source path only keeps remote_sim: locally infer and step a remote UAV-Flow simulator service.",
    )
    ap.add_argument(
        "--simulator_base_url",
        type=str,
        default=str(os.environ.get("UAVFLOW_SIMULATOR_BASE_URL", "http://127.0.0.1:8765")),
        help="Base URL for the remote simulator service when --rollout_backend=remote_sim.",
    )
    ap.add_argument(
        "--simulator_timeout_s",
        type=int,
        default=int(os.environ.get("UAVFLOW_SIMULATOR_TIMEOUT_S", "120")),
        help="HTTP timeout when communicating with the remote simulator service.",
    )
    ap.add_argument(
        "--uavflow_task_json_root",
        type=str,
        default=str(os.environ.get("UAVFLOW_TASK_JSON_ROOT", default_task_json_root)),
        help="Directory containing rollout task json files named <task_id>.json.",
    )
    args = ap.parse_args()

    os.environ["INFINITY_SERVER_CONFIG"] = os.path.abspath(args.infinity_server_config)
    os.environ["INFINITY_CKPT"] = os.path.abspath(args.infinity_ckpt)
    os.environ["INFINITY_REPO_ROOT"] = os.path.abspath(args.infinity_repo_root)
    os.environ["ACTIONHEAD_CKPT"] = os.path.abspath(args.actionhead_ckpt)
    os.environ["ACTIONHEAD_RUN_CONFIG"] = os.path.abspath(args.actionhead_run_config)
    os.environ["ACTIONHEAD_REPO_ROOT"] = os.path.abspath(args.actionhead_repo_root)
    os.environ["ACTION_HEAD_MODE"] = "actionhead_ref_vit"
    os.environ["INFINITY_DISABLE_P2P_LOAD"] = "1"
    if int(args.dump_debug_cache) != 1:
        os.environ["INFINITY_LATENT_CACHE_ROOT"] = os.path.abspath(args.trajectory_root)

    mod = _load_api_module(os.path.abspath(args.api_py))
    if int(args.dump_debug_cache) != 1:
        _disable_api_cache_dump(mod)
    os.makedirs(os.path.abspath(args.trajectory_root), exist_ok=True)
    failed_path = os.path.abspath(args.failed_jsonl) if str(args.failed_jsonl).strip() else ""
    failed_fp = None
    if failed_path:
        os.makedirs(os.path.dirname(failed_path), exist_ok=True)
        failed_fp = open(failed_path, "w", encoding="utf-8")
    timing_path = os.path.abspath(args.timing_jsonl) if str(args.timing_jsonl).strip() else ""
    timing_fp = None
    if timing_path:
        os.makedirs(os.path.dirname(timing_path), exist_ok=True)
        timing_fp = open(timing_path, "w", encoding="utf-8")

    # Warm model once
    cfg = mod._get_server_config()
    mod._init_models(cfg=cfg)

    n, ok = 0, 0
    failed_cnt = 0
    num_shards = max(1, int(args.num_shards))
    shard_id = int(args.shard_id)
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"bad shard_id={shard_id} for num_shards={num_shards}")
    total_global, total_shard = _count_candidates(args.candidates_jsonl, num_shards=num_shards, shard_id=shard_id)
    progress_every_n = max(1, int(args.progress_every_n))
    stage_start = time.perf_counter()
    print(
        f"[progress] shard={shard_id}/{num_shards} assigned={total_shard} global_total={total_global} "
        f"progress_every_n={progress_every_n}"
    )
    global_idx = -1
    with open(os.path.abspath(args.candidates_jsonl), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            global_idx += 1
            if (global_idx % num_shards) != shard_id:
                continue
            n += 1
            row: Dict[str, Any] = json.loads(line)
            traj_id = str(row.get("traj_id", f"traj_{n:06d}"))
            sid = f"GRPO_{traj_id}"
            video_path = str(row.get("video_path", ""))
            prompt = str(row.get("tarsier2_caption", row.get("instruction", "")))
            seed = int(row.get("seed", 0))
            if not video_path or not prompt:
                continue
            try:
                t0_all = time.perf_counter()
                rollout_backend = str(args.rollout_backend).strip().lower()

                # Retry rollout if actions/logprob/trace contain invalid values.
                # This is useful when model inference produces NaN/Inf or trace write fails intermittently.
                max_retry = max(1, int(args.max_retry))
                seed_step = int(args.retry_seed_step)
                r0 = r1 = r2 = None
                dt0 = dt1 = dt2 = 0.0
                backend_meta: Dict[str, Any] = {"sampling_backend": rollout_backend}
                for attempt in range(max_retry):
                    seed_try = int(seed) + int(attempt) * int(seed_step)
                    sid_try = f"{sid}_try{attempt}"
                    try:
                        responses_try, backend_meta_try = _run_remote_sim_rollout(
                            mod=mod,
                            row=row,
                            traj_id=traj_id,
                            session_id=sid_try,
                            prompt=prompt,
                            seed=seed_try,
                            simulator_base_url=str(args.simulator_base_url),
                            simulator_timeout_s=int(args.simulator_timeout_s),
                            action_head_batch_size=int(args.action_head_batch_size),
                            action_head_stride=int(args.action_head_stride),
                            action_head_pre_resize_hw=int(args.action_head_pre_resize_hw),
                            task_json_root=str(args.uavflow_task_json_root),
                        )
                        if len(responses_try) != 3:
                            raise RuntimeError(f"remote simulator rollout returned {len(responses_try)} segments, expected 3")
                        r0, r1, r2 = responses_try
                        backend_meta = dict(backend_meta_try)
                        dt0 = float(backend_meta.get("seg00_sec", 0.0))
                        dt1 = float(backend_meta.get("seg01_sec", 0.0))
                        dt2 = float(backend_meta.get("seg02_sec", 0.0))

                        actions_try: List[List[float]] = []
                        for rr in (r0, r1, r2):
                            aa = getattr(rr, "actions", None)
                            if isinstance(aa, list):
                                actions_try.extend(aa)

                        seg_oldlp_try: List[float] = []
                        seg_trace_try: List[str] = []
                        for rr in (r0, r1, r2):
                            try:
                                seg_oldlp_try.append(float(getattr(rr, "segment_old_logprob", 0.0) or 0.0))
                            except Exception:
                                seg_oldlp_try.append(0.0)
                            seg_trace_try.append(str(getattr(rr, "segment_trace_path", "") or ""))

                        # Basic validity checks: finite actions/logprobs and existing trace file(s).
                        if (not _all_finite_6d(actions_try)) or (not _all_finite(seg_oldlp_try)):
                            raise RuntimeError("invalid rollout numbers (NaN/Inf/non-6D)")
                        if not any((p and os.path.exists(p)) for p in seg_trace_try):
                            raise RuntimeError("missing trace files")
                        # Strict trace_ce guard: trace files must contain sample_logprob_trace_ce.
                        _require_trace_ce_ok(seg_trace_try)

                        # Success
                        seed = seed_try
                        sid = sid_try
                        break
                    except Exception as e:
                        # Retry next attempt
                        r0 = r1 = r2 = None
                        if attempt == max_retry - 1:
                            raise RuntimeError(f"rollout failed after {max_retry} attempts: {e}") from e

                actions = []
                for rr in (r0, r1, r2):
                    aa = getattr(rr, "actions", None)
                    if isinstance(aa, list):
                        actions.extend(aa)
                poses = _integrate_actions(actions)
                seg_old_logprobs: List[float] = []
                seg_trace_paths: List[str] = []
                for rr in (r0, r1, r2):
                    try:
                        seg_old_logprobs.append(float(getattr(rr, "segment_old_logprob", 0.0) or 0.0))
                    except Exception:
                        seg_old_logprobs.append(0.0)
                    seg_trace_paths.append(str(getattr(rr, "segment_trace_path", "") or ""))

                out_dir = os.path.join(os.path.abspath(args.trajectory_root), traj_id)
                os.makedirs(out_dir, exist_ok=True)
                # Keep trace_files aligned to seg index (0..2). DO NOT append-only,
                # otherwise missing segs would shift indices and break clip alignment.
                trace_files: List[str] = ["", "", ""]
                for si, src in enumerate(seg_trace_paths):
                    if not src or (not os.path.exists(src)):
                        continue
                    dst = os.path.join(out_dir, f"seg{si:02d}_trace.pt")
                    try:
                        shutil.copy2(src, dst)
                        if 0 <= int(si) < len(trace_files):
                            trace_files[int(si)] = dst
                    except Exception:
                        continue
                payload = {
                    "traj_id": traj_id,
                    "seed": int(seed),
                    "candidate_id": int(row.get("candidate_id", 0)),
                    "video_path": video_path,
                    "prompt": prompt,
                    "num_actions": len(actions),
                    "actions": actions,
                    "relative_poses": {
                        "start_pose": [0.0] * 6,
                        "poses": poses,
                        "final_pose": (poses[-1] if poses else [0.0] * 6),
                    },
                    "sample_logprob_segments": seg_old_logprobs,
                    "sample_logprob_total": float(sum(seg_old_logprobs)),
                    "trace_files": trace_files,
                    "note": "Real rollout via Infinity + actionhead_ref_vit from api server.",
                    **backend_meta,
                }
                with open(os.path.join(out_dir, "trajectory.json"), "w", encoding="utf-8") as wf:
                    json.dump(payload, wf, ensure_ascii=False, indent=2)
                dt_all = time.perf_counter() - t0_all
                lp_total = float(sum(seg_old_logprobs))
                print(
                    f"[timing] traj_id={traj_id} total={dt_all:.2f}s "
                    f"seg00={dt0:.2f}s seg01={dt1:.2f}s seg02={dt2:.2f}s actions={len(actions)} oldlp={lp_total:.3f}"
                )
                if timing_fp is not None:
                    timing_fp.write(
                        json.dumps(
                            {
                                "traj_id": traj_id,
                                "candidate_id": int(row.get("candidate_id", 0)),
                                "video_path": video_path,
                                "total_sec": dt_all,
                                "seg00_sec": dt0,
                                "seg01_sec": dt1,
                                "seg02_sec": dt2,
                                "num_actions": len(actions),
                                "old_logprob_total": lp_total,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    timing_fp.flush()
                ok += 1
            except Exception as e:
                print(f"[generate_candidate_trajectories_real] skip traj_id={traj_id}: {e}")
                failed_cnt += 1
                if failed_fp is not None:
                    failed = {
                        "traj_id": traj_id,
                        "candidate_id": int(row.get("candidate_id", 0)),
                        "video_path": video_path,
                        "error": str(e),
                    }
                    failed_fp.write(json.dumps(failed, ensure_ascii=False) + "\n")
                    failed_fp.flush()
            elapsed = time.perf_counter() - stage_start
            if (n % progress_every_n == 0) or (n == total_shard):
                avg_wall = elapsed / max(1, n)
                eta_sec = max(0.0, avg_wall * max(0, total_shard - n))
                print(
                    f"[progress] shard={shard_id}/{num_shards} processed={n}/{total_shard} "
                    f"wrote={ok} failed={failed_cnt} avg_wall={avg_wall:.1f}s eta_min={eta_sec/60.0:.1f} "
                    f"global_total={total_global}"
                )
            continue
    if failed_fp is not None:
        failed_fp.close()
    if timing_fp is not None:
        timing_fp.close()
    print(
        f"[generate_candidate_trajectories_real] shard={shard_id}/{num_shards} "
        f"processed={n}, wrote={ok}, root={os.path.abspath(args.trajectory_root)}"
    )


if __name__ == "__main__":
    main()

