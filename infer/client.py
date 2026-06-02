#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Client script for the WorldVLN online inference server.

Protocol:
- One trajectory corresponds to one `session_id` (typically the route folder name).
- Upload frames incrementally: first 1 warmup frame, then `step` frames per call until `num_frames` is reached.

Server output:
- Delta actions are in cm/deg, ordered as [dx, dy, dz, droll, dyaw, dpitch].
- action_head_mode=tsformer_latent: emits 4 macro actions per segment.
- action_head_mode=actionhead_ref_vit: emits `step` per-frame actions per segment (step=16 -> 16 actions).

Client outputs (two JSON files per segment, filename includes session_id):
1) actions json:
   - actions_server_order: Nx6 in server order (N depends on action_head_mode)
   - actions_client_order: Nx6 in an alternate order used by some downstream tooling
   - action_frames: per-action frame identifiers (dataset: path; unrealcv: saved filename)
   - cumsum_*: cumulative sums of actions (within the file)
2) poses json (absolute coordinates):
   - segment 0: points include start + N endpoints (1+N points)
   - later segments: points include N endpoints only
   - pose order is [x, y, z, roll, yaw, pitch] in cm/deg
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _time_id() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def _sorted_frame_paths(images_dir: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    names = [n for n in os.listdir(images_dir) if n.lower().endswith(exts)]
    names.sort()
    return [os.path.join(images_dir, n) for n in names]


def _take_with_pad(paths: List[str], n: int, pad_short_real: bool) -> List[str]:
    """Pad by repeating last frame path (official scripts behavior)."""
    if len(paths) >= int(n):
        return paths[: int(n)]
    if not paths:
        raise ValueError("no real frames found")
    if not bool(pad_short_real):
        raise ValueError(f"need >={n} frames, got {len(paths)} (use --pad_short_real 1 to pad)")
    return paths + [paths[-1]] * (int(n) - int(len(paths)))


def _image_to_data_url_jpeg(path: str, quality: int = 90) -> str:
    return _image_to_data_url(path, codec="jpeg", quality=int(quality))


def _pil_to_data_url_jpeg(img: Image.Image, quality: int = 90) -> str:
    return _pil_to_data_url(img, codec="jpeg", quality=int(quality))


def _image_to_data_url(path: str, *, codec: str, quality: int = 90) -> str:
    img = Image.open(path).convert("RGB")
    return _pil_to_data_url(img, codec=str(codec), quality=int(quality))


def _pil_to_data_url(img: Image.Image, *, codec: str, quality: int = 90) -> str:
    img = img.convert("RGB")
    bio = BytesIO()
    c = str(codec).lower().strip()
    if c in ("jpg", "jpeg"):
        img.save(bio, format="JPEG", quality=int(quality))
        mime = "image/jpeg"
    elif c == "png":
        img.save(bio, format="PNG")
        mime = "image/png"
    else:
        raise ValueError(f"unsupported image codec: {codec}")
    b64 = base64.b64encode(bio.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_pil_jpeg(img: Image.Image, path: str, *, quality: int = 95) -> None:
    parent = os.path.dirname(path)
    if parent:
        _ensure_dir(parent)
    img.convert("RGB").save(path, format="JPEG", quality=int(quality))


def _safe_np_image_to_pil_rgb(img_any: Any) -> Image.Image:
    """
    UnrealCV get_image(...) often returns np.ndarray(H,W,3) (many implementations use BGR).
    Convert to a PIL RGB image as robustly as possible.
    """
    if isinstance(img_any, Image.Image):
        return img_any.convert("RGB")
    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise RuntimeError("numpy is required to convert UnrealCV images") from e
    if not isinstance(img_any, np.ndarray):
        raise TypeError(f"Unsupported image type: {type(img_any)}")
    arr = img_any
    if arr.ndim == 3 and int(arr.shape[2]) == 3:
        try:
            import cv2  # type: ignore

            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        except Exception:
            arr = arr[:, :, ::-1]
        return Image.fromarray(arr.astype("uint8"), mode="RGB")
    if arr.ndim == 2:
        return Image.fromarray(arr.astype("uint8"), mode="L").convert("RGB")
    raise ValueError(f"Unsupported ndarray image shape: {getattr(arr, 'shape', None)}")


def _reorder_server_to_client(a6: List[float]) -> List[float]:
    """
    server order: [dx,dy,dz,droll,dyaw,dpitch]
    client order: [dx,dy,dz,droll,dpitch,dyaw]
    """
    if len(a6) != 6:
        raise ValueError(f"action must be 6D, got {len(a6)}")
    dx, dy, dz, droll, dyaw, dpitch = [float(x) for x in a6]
    return [dx, dy, dz, droll, dpitch, dyaw]


def _cumsum_actions(actions: List[List[float]]) -> List[List[float]]:
    out: List[List[float]] = []
    cur = [0.0] * 6
    for a in actions:
        cur = [cur[i] + float(a[i]) for i in range(6)]
        out.append(cur)
    return out


def _apply_action_to_pose(pose_xyz_rpy: List[float], action_dxdy_dz_droll_dyaw_dpitch: List[float]) -> List[float]:
    """
    pose: [x,y,z,roll,yaw,pitch] in cm/deg
    action: [dx,dy,dz,droll,dyaw,dpitch] in cm/deg
    simple world-frame integration: pose_next = pose + delta
    """
    if len(pose_xyz_rpy) != 6:
        raise ValueError(f"pose must be 6D, got {len(pose_xyz_rpy)}")
    if len(action_dxdy_dz_droll_dyaw_dpitch) != 6:
        raise ValueError(f"action must be 6D, got {len(action_dxdy_dz_droll_dyaw_dpitch)}")
    return [float(pose_xyz_rpy[i]) + float(action_dxdy_dz_droll_dyaw_dpitch[i]) for i in range(6)]


def _apply_action_to_pose_with_frame(
    pose_xyz_rpy: List[float],
    action_dxdy_dz_droll_dyaw_dpitch: List[float],
    *,
    action_frame: str,
    body_apply_order: str = "yaw_first",
    integrate_roll_pitch: bool = True,
) -> List[float]:
    """
    pose: [x,y,z,roll,yaw,pitch] in cm/deg
    action: [dx,dy,dz,droll,dyaw,dpitch] in cm/deg

    - action_frame="world": direct add (world-frame deltas)
    - action_frame="body": treat (dx,dy) as body-frame (forward/right) and rotate by yaw; body_apply_order selects
      whether to turn first then translate, translate first then turn, or use midpoint integration.

    Note: dz is added along Z(up) directly; this is typically acceptable when pitch/roll are small or ignored.
    """
    if len(pose_xyz_rpy) != 6 or len(action_dxdy_dz_droll_dyaw_dpitch) != 6:
        raise ValueError("pose/action must be 6D")
    x, y, z, roll, yaw, pitch = [float(v) for v in pose_xyz_rpy]
    dx, dy, dz, droll, dyaw, dpitch = [float(v) for v in action_dxdy_dz_droll_dyaw_dpitch]

    fr = str(action_frame).strip().lower()
    if fr == "world":
        x += dx
        y += dy
        z += dz
    elif fr == "body":
        # Use yaw to rotate body (forward/right) into world (x/y).
        import math

        order = str(body_apply_order).strip().lower()
        if order in ("yaw_first", "rotate_first", "turn_first"):
            yaw = float(yaw) + float(dyaw)
            theta = math.radians(yaw)
        elif order in ("trans_first", "translate_first", "move_first"):
            theta = math.radians(yaw)
            yaw = float(yaw) + float(dyaw)
        elif order in ("midpoint", "mid", "half"):
            theta = math.radians(float(yaw) + 0.5 * float(dyaw))
            yaw = float(yaw) + float(dyaw)
        else:
            raise ValueError(f"bad body_apply_order={body_apply_order}, expected yaw_first|trans_first|midpoint")
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        x += dx * cos_t - dy * sin_t
        y += dx * sin_t + dy * cos_t
        z += dz
    else:
        raise ValueError(f"bad action_frame={action_frame}, expected world|body")

    # Angles
    if fr == "world":
        yaw += dyaw
    if bool(integrate_roll_pitch):
        roll += droll
        pitch += dpitch
    return [x, y, z, roll, yaw, pitch]


def _http_post_json(url: str, payload: Dict, *, timeout_s: int = 120) -> Dict:
    try:
        import requests  # type: ignore
    except Exception as e:
        raise RuntimeError("requests is required for client HTTP calls") from e

    r = requests.post(url, json=payload, timeout=int(timeout_s))
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    return r.json()


@dataclass
class Route:
    route_dir: str
    route_id: str
    images_dir: str
    meta_path: str
    raw_logs_path: Optional[str]


def _discover_routes(dataset_root: str) -> List[Route]:
    routes: List[Route] = []
    for name in sorted(os.listdir(dataset_root)):
        rd = os.path.join(dataset_root, name)
        if not os.path.isdir(rd):
            continue
        images_dir = os.path.join(rd, "images")
        meta_path = os.path.join(rd, "meta.json")
        if not os.path.isdir(images_dir) or not os.path.exists(meta_path):
            continue
        raw_logs = os.path.join(rd, "raw_logs.json")
        routes.append(
            Route(
                route_dir=rd,
                route_id=os.path.basename(rd.rstrip("/")),
                images_dir=images_dir,
                meta_path=meta_path,
                raw_logs_path=raw_logs if os.path.exists(raw_logs) else None,
            )
        )
    return routes


def _load_prompt(meta_path: str) -> str:
    meta = _read_json(meta_path)
    prompt = (meta.get("instruction") or meta.get("instruction_unified") or meta.get("prompt") or "").strip()
    return str(prompt)


def _load_start_pose_cm_deg(raw_logs_path: Optional[str]) -> List[float]:
    if not raw_logs_path or not os.path.exists(raw_logs_path):
        return [0.0] * 6
    arr = _read_json(raw_logs_path)
    if not isinstance(arr, list) or len(arr) == 0:
        return [0.0] * 6
    p0 = arr[0]
    if not (isinstance(p0, (list, tuple)) and len(p0) == 6):
        return [0.0] * 6
    return [float(x) for x in p0]  # [x,y,z,roll,yaw,pitch] in cm/deg


def _load_num_frames_step_from_config(config_json: str) -> Tuple[int, int]:
    cfg = _read_json(config_json)
    if not isinstance(cfg, dict):
        raise ValueError(f"bad config json: {config_json}")
    inf = cfg.get("infinity", cfg)
    num_frames = int(inf.get("num_frames", 81))
    step = int(inf.get("step", 16))
    if num_frames <= 0 or step <= 0:
        raise ValueError(f"bad num_frames/step in config: num_frames={num_frames} step={step}")
    return num_frames, step


def _load_instruction_and_initial_pose_from_task_json(task_json_path: str) -> Tuple[str, List[float]]:
    """
    Read a rollout task json (e.g. select_from_train_jsons/*.json) and extract:
    - instruction (or instruction_unified)
    - initial_pos: [x,y,z,roll,yaw,pitch] in cm/deg
    """
    d = _read_json(task_json_path)
    if not isinstance(d, dict):
        raise ValueError(f"bad task json (expect dict): {task_json_path}")
    instr = (d.get("instruction") or d.get("instruction_unified") or "").strip()
    if not instr:
        raise ValueError(f"empty instruction in task json: {task_json_path}")
    initial_pos = d.get("initial_pos", None)
    if not (isinstance(initial_pos, list) and len(initial_pos) >= 6):
        raise ValueError(f"bad initial_pos in task json: {task_json_path}")
    init6 = [float(x) for x in initial_pos[:6]]
    return instr, init6


def _build_obj_info_from_task_json(task_json_path: str) -> Optional[Dict[str, Any]]:
    """
    Align with batch_run_act_all.py:
    - only place an object when both obj_id and use_obj exist
    - prefer target_pos[:3]/target_pos[3:] as obj_pos/obj_rot when available
    - otherwise fall back to obj_pos/obj_rot
    """
    d = _read_json(task_json_path)
    if not isinstance(d, dict):
        return None
    if "obj_id" not in d or "use_obj" not in d:
        return None
    if "target_pos" in d and isinstance(d["target_pos"], list) and len(d["target_pos"]) == 6:
        obj_pos = [float(x) for x in d["target_pos"][:3]]
        obj_rot = [float(x) for x in d["target_pos"][3:]]
    else:
        raw_pos = d.get("obj_pos", None)
        raw_rot = d.get("obj_rot", [0, 0, 0])
        if not (isinstance(raw_pos, list) and len(raw_pos) >= 3):
            return None
        obj_pos = [float(x) for x in raw_pos[:3]]
        obj_rot = [float(x) for x in (raw_rot[:3] if isinstance(raw_rot, list) else [0, 0, 0])]
    return {
        "use_obj": int(d["use_obj"]),
        "obj_id": int(d["obj_id"]),
        "obj_pos": obj_pos,
        "obj_rot": obj_rot,
    }


def _init_marker_objects_if_needed(env: Any) -> None:
    """
    Create/initialize marker objects in the scene (once), aligned with batch_run_act_all.py init behavior.
    """
    # Avoid repeated init across tasks in same process.
    if bool(getattr(env.unwrapped, "_xjc_marker_inited", False)):
        return
    try:
        time.sleep(1.0)
        env.unwrapped.unrealcv.new_obj("bp_character_C", "BP_Character_21", [0, 0, 0])
        env.unwrapped.unrealcv.set_appearance("BP_Character_21", 0)
        env.unwrapped.unrealcv.set_obj_rotation("BP_Character_21", [0, 0, 0])
        time.sleep(1.0)
        env.unwrapped.unrealcv.new_obj("BP_BaseCar_C", "BP_Character_22", [1000, 0, 0])
        env.unwrapped.unrealcv.set_appearance("BP_Character_22", 2)
        env.unwrapped.unrealcv.set_obj_rotation("BP_Character_22", [0, 0, 0])
        env.unwrapped.unrealcv.set_phy("BP_Character_22", 0)
        time.sleep(1.0)
        env.unwrapped._xjc_marker_inited = True
    except Exception:
        # Objects may already exist or class names may differ; do not block the main control flow.
        env.unwrapped._xjc_marker_inited = True


def _create_obj_if_needed_unrealcv(env: Any, obj_info: Optional[Dict[str, Any]]) -> None:
    """
    Place the task object; logic aligned with batch_run_act_all.py create_obj_if_needed.
    """
    if obj_info is None:
        return
    use_obj = obj_info.get("use_obj", None)
    obj_id = obj_info.get("obj_id", None)
    obj_pos = obj_info.get("obj_pos", None)
    obj_rot = obj_info.get("obj_rot", None)
    if obj_pos is None:
        return
    try:
        if int(use_obj) == 1:
            env.unwrapped.unrealcv.set_appearance("BP_Character_21", int(obj_id))
            env.unwrapped.unrealcv.set_obj_location("BP_Character_21", obj_pos)
            env.unwrapped.unrealcv.set_obj_rotation("BP_Character_21", obj_rot if obj_rot is not None else [0, 0, 0])
            env.unwrapped.unrealcv.set_obj_location("BP_Character_22", [0, 0, -1000])
            env.unwrapped.unrealcv.set_obj_location("BP_Character_21", obj_pos)
        elif int(use_obj) == 2:
            env.unwrapped.unrealcv.set_appearance("BP_Character_22", 2)
            env.unwrapped.unrealcv.set_obj_location("BP_Character_22", [obj_pos[0], obj_pos[1], 0])
            env.unwrapped.unrealcv.set_obj_rotation("BP_Character_22", obj_rot if obj_rot is not None else [0, 0, 0])
            env.unwrapped.unrealcv.set_phy("BP_Character_22", 0)
            env.unwrapped.unrealcv.set_obj_location("BP_Character_21", [0, 0, -1000])
            env.unwrapped.unrealcv.set_obj_location("BP_Character_22", [obj_pos[0], obj_pos[1], 0])
        if int(use_obj) in (1, 2):
            time.sleep(1.0)
    except Exception:
        # Do not interrupt the main control flow; avoid failing due to scene-asset differences.
        pass


def _setup_unrealcv_camera_follow(env: Any, *, cam_id: int = 0) -> None:
    """
    Bind the camera to the UAV position to approximate a first-person view.
    Mirrors batch_run_act_all.py set_cam logic.
    """
    x, y, z = env.unwrapped.unrealcv.get_obj_location(env.unwrapped.player_list[0])
    roll, yaw, pitch = env.unwrapped.unrealcv.get_obj_rotation(env.unwrapped.player_list[0])  # [roll, yaw, pitch]
    cam_loc = [x, y, z]
    cam_rot = [roll, pitch, yaw]  # UnrealCV set_cam rotation order
    env.unwrapped.unrealcv.set_cam(int(cam_id), cam_loc, cam_rot)


def _apply_pose_unrealcv(
    env: Any,
    *,
    pose_xyz_rpy: List[float],
    yaw_offset_deg: float = -180.0,
) -> None:
    """
    Apply [x,y,z,roll,yaw,pitch] to the simulator (cm/deg).
    For the drone, gym_unrealcv set_obj_rotation may not take effect reliably, so we use set_rotation(yaw).
    """
    if len(pose_xyz_rpy) < 6:
        raise ValueError(f"pose must be 6D, got {len(pose_xyz_rpy)}")
    x, y, z, _roll, yaw, _pitch = [float(v) for v in pose_xyz_rpy[:6]]
    env.unwrapped.unrealcv.set_obj_location(env.unwrapped.player_list[0], [x, y, z])
    env.unwrapped.unrealcv.set_rotation(env.unwrapped.player_list[0], float(yaw) + float(yaw_offset_deg))
    _setup_unrealcv_camera_follow(env, cam_id=0)


def _capture_unrealcv_lit_pil(env: Any, *, cam_id: int = 0) -> Image.Image:
    _setup_unrealcv_camera_follow(env, cam_id=int(cam_id))
    img = env.unwrapped.unrealcv.get_image(int(cam_id), "lit")
    return _safe_np_image_to_pil_rgb(img)


def _angle_diff_deg(a: float, b: float) -> float:
    d = (float(a) - float(b) + 180.0) % 360.0 - 180.0
    return abs(d)


def _wait_pose_settle(
    env: Any,
    *,
    target_pose_xyz_rpy: List[float],
    yaw_offset_deg: float,
    max_tries: int = 20,
    sleep_s: float = 0.05,
    pos_tol_cm: float = 1.0,
    yaw_tol_deg: float = 1.0,
) -> None:
    """Wait for set_obj_location/set_rotation to take effect, reducing the chance of capturing a stale first frame."""
    tx, ty, tz = [float(v) for v in target_pose_xyz_rpy[:3]]
    tyaw_set = float(target_pose_xyz_rpy[4]) + float(yaw_offset_deg)
    for _ in range(int(max_tries)):
        try:
            x, y, z = env.unwrapped.unrealcv.get_obj_location(env.unwrapped.player_list[0])
            _roll, yaw_now, _pitch = env.unwrapped.unrealcv.get_obj_rotation(env.unwrapped.player_list[0])
            pos_ok = abs(float(x) - tx) <= pos_tol_cm and abs(float(y) - ty) <= pos_tol_cm and abs(float(z) - tz) <= pos_tol_cm
            yaw_ok = _angle_diff_deg(float(yaw_now), tyaw_set) <= yaw_tol_deg
            if pos_ok and yaw_ok:
                return
        except Exception:
            pass
        time.sleep(float(sleep_s))


def _split_action_to_substeps(action6: List[float], substeps: int) -> List[List[float]]:
    if len(action6) != 6:
        raise ValueError(f"action must be 6D, got {len(action6)}")
    k = int(substeps)
    if k <= 0:
        raise ValueError(f"substeps must be >0, got {substeps}")
    a = [float(x) for x in action6]
    return [[a[i] / float(k) for i in range(6)] for _ in range(k)]


def run_one_task_unrealcv(
    *,
    task_json_path: str,
    env: Any,
    env_id: str,
    server_base_url: str,
    out_root: str,
    run_id: str,
    num_frames: int,
    step: int,
    max_actions: int = 0,
    timeout_s: int,
    action_head_mode: str,
    action_head_batch_size: int,
    action_head_stride: int,
    action_head_pre_resize_hw: int,
    image_codec: str,
    jpeg_quality: int,
    yaw_offset_deg: float = -180.0,
    allow_future_last_segment: bool = False,
    action_frame: str = "world",
    body_apply_order: str = "yaw_first",
    save_images: bool = True,
) -> None:
    """
    Online mode (gym_unrealcv):
    - Read instruction + initial_pos from task_json
    - Capture 256x256 lit RGB (resolution set by ConfigUEWrapper)
    - Upload frames incrementally: 1, step, step, ... (prefix_mode=false; history is stored server-side by session_id)
    - tsformer_latent: after receiving 4 macro actions, split each into 4 sub-actions to execute step(=16) steps;
      capture and save one frame after each executed step
    - actionhead_ref_vit: directly receive step(=16) per-frame actions; execute one-by-one and save frames

    Note: on n==1 the server may only warm up and emit no actions. This client advances the server timeline by
    capturing/uploading `step` frames in place to make seg0 available.
    """
    instruction, init_pose = _load_instruction_and_initial_pose_from_task_json(task_json_path)
    obj_info = _build_obj_info_from_task_json(task_json_path)

    base_name = os.path.splitext(os.path.basename(task_json_path))[0]
    session_id = f"{base_name}__{run_id}"
    out_dir = os.path.join(os.path.abspath(out_root), f"client_run_{run_id}", base_name)
    _ensure_dir(out_dir)
    images_dir = os.path.join(out_dir, "images")
    _ensure_dir(images_dir)

    # Reset env state per task
    try:
        env.reset()
    except Exception:
        pass
    try:
        env.unwrapped.unrealcv.set_viewport(env.unwrapped.player_list[0])
        env.unwrapped.unrealcv.set_phy(env.unwrapped.player_list[0], 0)
    except Exception:
        pass

    _init_marker_objects_if_needed(env)
    _create_obj_if_needed_unrealcv(env, obj_info)
    _apply_pose_unrealcv(env, pose_xyz_rpy=init_pose, yaw_offset_deg=float(yaw_offset_deg))
    _wait_pose_settle(env, target_pose_xyz_rpy=init_pose, yaw_offset_deg=float(yaw_offset_deg))
    # Give the camera extra time to refresh to reduce stale-view risk for the first frame.
    time.sleep(1.0)

    summary = {
        "mode": "unrealcv",
        "task_json": os.path.abspath(task_json_path),
        "env_id": str(env_id),
        "session_id": session_id,
        "server_base_url": server_base_url,
        "endpoint": "/v1/predict_delta_actions",
        "instruction": instruction,
        "initial_pose_cm_deg_order_xyz_roll_yaw_pitch": init_pose,
        "num_frames": int(num_frames),
        "step": int(step),
        "max_actions": int(max_actions),
        "prefix_mode": False,
        "allow_future_last_segment": bool(allow_future_last_segment),
        "camera": {"cam_id": 0, "viewmode": "lit"},
        "image_source": {"type": "gym_unrealcv", "client_capture_resolution": list(getattr(env.unwrapped, "resolution", [None, None]))},
        "yaw_offset_deg_applied_in_unrealcv": float(yaw_offset_deg),
        "action_frame_for_integration": str(action_frame),
        "body_apply_order": str(body_apply_order),
        "action_head_mode": str(action_head_mode),
        "action_head_batch_size": int(action_head_batch_size),
        "action_head_stride": int(action_head_stride),
        "action_head_pre_resize_hw": int(action_head_pre_resize_hw),
        "image_codec": str(image_codec),
        "jpeg_quality": int(jpeg_quality),
        "units": {"translation": "cm", "angles": "deg"},
        "time": int(time.time()),
    }
    with open(os.path.join(out_dir, f"{base_name}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    def _post_frames(frames: List[Image.Image], *, include_instruction: bool) -> Dict:
        payload: Dict[str, Any] = {
            "session_id": session_id,
            "images_base64": [_pil_to_data_url(im, codec=str(image_codec), quality=int(jpeg_quality)) for im in frames],
            "prefix_mode": False,
            "allow_future_last_segment": bool(allow_future_last_segment),
            "allow_future_segments": True,
            "action_head_mode": str(action_head_mode),
            "action_head_batch_size": int(action_head_batch_size),
            "action_head_stride": int(action_head_stride),
            "action_head_pre_resize_hw": int(action_head_pre_resize_hw),
        }
        if include_instruction:
            payload["instruction"] = instruction
            payload["reset_session"] = True
        return _http_post_json(server_base_url.rstrip("/") + "/v1/predict_delta_actions", payload, timeout_s=int(timeout_s))

    # Frame 1: initial
    frame_idx = 1
    actions_executed = 0
    cur_pose = init_pose[:]  # [x,y,z,roll,yaw,pitch]
    im0 = _capture_unrealcv_lit_pil(env, cam_id=0)
    if bool(save_images):
        _save_pil_jpeg(im0, os.path.join(images_dir, f"frame_{frame_idx:04d}.jpg"), quality=95)
    resp = _post_frames([im0], include_instruction=True)

    def _write_segment_logs(
        seg: int,
        call_i: int,
        frames_in_call: int,
        resp_obj: Dict,
        actions_server: List[List[float]],
        pose_before: List[float],
        action_image_names: Optional[List[str]] = None,
    ) -> None:
        actions_client = [_reorder_server_to_client(a) for a in actions_server]
        cumsum_server = _cumsum_actions(actions_server)
        cumsum_client = _cumsum_actions(actions_client)
        actions_json = {
            "session_id": session_id,
            "task_json": os.path.abspath(task_json_path),
            "segment_index": int(seg),
            "call_index": int(call_i),
            "frames_in_call": int(frames_in_call),
            "num_received_frames": resp_obj.get("num_received_frames", None),
            "done": resp_obj.get("done", None),
            "prefix_latents": resp_obj.get("prefix_latents", None),
            "units": {"translation": "cm", "angles": "deg"},
            "action_head_mode": str(action_head_mode),
            "num_actions": int(len(actions_server)),
            "action_order_server": ["dx", "dy", "dz", "droll", "dyaw", "dpitch"],
            "action_order_client": ["dx", "dy", "dz", "droll", "dpitch", "dyaw"],
            "actions_server_order": actions_server,
            "actions_client_order": actions_client,
            "action_frames": action_image_names or [],
            "cumsum_server_order": cumsum_server,
            "cumsum_client_order": cumsum_client,
        }
        with open(os.path.join(out_dir, f"{base_name}_seg{seg:02d}_actions.json"), "w", encoding="utf-8") as f:
            json.dump(actions_json, f, ensure_ascii=False, indent=2)

        # Pose points after each action (macro or per-frame)
        p = pose_before[:]
        pts: List[List[float]] = []
        if int(seg) == 0:
            pts.append(p[:])
        for a in actions_server:
            p = _apply_action_to_pose_with_frame(
                p,
                a,
                action_frame=str(action_frame),
                body_apply_order=str(body_apply_order),
                integrate_roll_pitch=True,
            )
            pts.append(p[:])
        poses_json = {
            "session_id": session_id,
            "task_json": os.path.abspath(task_json_path),
            "segment_index": int(seg),
            "call_index": int(call_i),
            "units": {"translation": "cm", "angles": "deg"},
            "pose_order": ["x", "y", "z", "roll", "yaw", "pitch"],
            "points": pts,
        }
        with open(os.path.join(out_dir, f"{base_name}_seg{seg:02d}_poses.json"), "w", encoding="utf-8") as f:
            json.dump(poses_json, f, ensure_ascii=False, indent=2)

    # Determine expected action count for execution
    mode_l = str(action_head_mode).strip().lower()
    is_per_frame_mode = mode_l in ("actionhead_ref_vit", "actionhead_ref", "actionhead_vit", "ref_vit", "actionhead")
    expected_actions = int(step) if bool(is_per_frame_mode) else 4

    call_i = 0
    last_upload_n = 1  # first upload: 1 frame
    max_actions_i = int(max_actions)
    # Stop conditions:
    # - max_actions>0: stop after executing that many actions (strict protocol: 48 actions => 1+48 frames)
    # - else: fall back to num_frames bound
    while True:
        if max_actions_i > 0 and int(actions_executed) >= int(max_actions_i):
            break
        if max_actions_i <= 0 and frame_idx >= int(num_frames):
            break
        call_i += 1
        actions = resp.get("actions", [])
        seg = int(resp.get("segment_index", -1))
        done = bool(resp.get("done", False))

        pending_actions: Optional[List[List[float]]] = None
        if isinstance(actions, list) and len(actions) == int(expected_actions) and seg >= 0:
            pending_actions = [[float(x) for x in a] for a in actions]
            if max_actions_i > 0:
                remain = int(max_actions_i) - int(actions_executed)
                if remain <= 0:
                    pending_actions = []
                elif 0 < remain < len(pending_actions):
                    pending_actions = pending_actions[:remain]
            _write_segment_logs(seg, call_i=call_i, frames_in_call=int(last_upload_n), resp_obj=resp, actions_server=pending_actions, pose_before=cur_pose[:])

        if done:
            break

        # Collect next 'step' frames (incremental upload)
        new_frames: List[Image.Image] = []
        if pending_actions is None:
            # Hold position to fill frames until server is ready to emit a segment.
            for _ in range(int(step)):
                if max_actions_i > 0:
                    # In strict action-count mode, do NOT advance frames without executing actions.
                    break
                if frame_idx >= int(num_frames):
                    break
                im = _capture_unrealcv_lit_pil(env, cam_id=0)
                frame_idx += 1
                if bool(save_images):
                    _save_pil_jpeg(im, os.path.join(images_dir, f"frame_{frame_idx:04d}.jpg"), quality=95)
                new_frames.append(im)
        else:
            total_needed = int(step)
            produced = 0
            action_image_names: List[str] = []
            if bool(is_per_frame_mode):
                # Execute per-frame actions directly: (typically 16 actions -> 16 frames)
                for a in pending_actions:
                    if produced >= total_needed:
                        break
                    if max_actions_i > 0 and int(actions_executed) >= int(max_actions_i):
                        break
                    if max_actions_i <= 0 and frame_idx >= int(num_frames):
                        break
                    next_pose = _apply_action_to_pose_with_frame(
                        cur_pose,
                        a,
                        action_frame=str(action_frame),
                        body_apply_order=str(body_apply_order),
                        integrate_roll_pitch=False,
                    )
                    cur_pose[0] = float(next_pose[0])
                    cur_pose[1] = float(next_pose[1])
                    cur_pose[2] = float(next_pose[2])
                    cur_pose[4] = float(next_pose[4])
                    _apply_pose_unrealcv(env, pose_xyz_rpy=cur_pose, yaw_offset_deg=float(yaw_offset_deg))
                    im = _capture_unrealcv_lit_pil(env, cam_id=0)
                    frame_idx += 1
                    produced += 1
                    actions_executed += 1
                    name = f"frame_{frame_idx:04d}.jpg"
                    if bool(save_images):
                        _save_pil_jpeg(im, os.path.join(images_dir, name), quality=95)
                    action_image_names.append(name)
                    new_frames.append(im)
            else:
                # Execute 4 macro actions, each split into 4 substeps => 16 frames
                substeps_per_action = 4
                for a in pending_actions:
                    for sub in _split_action_to_substeps(a, substeps=substeps_per_action):
                        if produced >= total_needed:
                            break
                        if max_actions_i > 0 and int(actions_executed) >= int(max_actions_i):
                            break
                        if max_actions_i <= 0 and frame_idx >= int(num_frames):
                            break
                        next_pose = _apply_action_to_pose_with_frame(
                            cur_pose,
                            sub,
                            action_frame=str(action_frame),
                            body_apply_order=str(body_apply_order),
                            integrate_roll_pitch=False,
                        )
                        cur_pose[0] = float(next_pose[0])
                        cur_pose[1] = float(next_pose[1])
                        cur_pose[2] = float(next_pose[2])
                        cur_pose[4] = float(next_pose[4])
                        _apply_pose_unrealcv(env, pose_xyz_rpy=cur_pose, yaw_offset_deg=float(yaw_offset_deg))
                        im = _capture_unrealcv_lit_pil(env, cam_id=0)
                        frame_idx += 1
                        produced += 1
                        actions_executed += 1
                        name = f"frame_{frame_idx:04d}.jpg"
                        if bool(save_images):
                            _save_pil_jpeg(im, os.path.join(images_dir, name), quality=95)
                        action_image_names.append(name)
                        new_frames.append(im)
                    if produced >= total_needed:
                        break
                    if max_actions_i > 0 and int(actions_executed) >= int(max_actions_i):
                        break
                    if max_actions_i <= 0 and frame_idx >= int(num_frames):
                        break
            # If produced is insufficient: only pad by in-place sampling in num_frames mode; do not pad in max_actions mode
            # (strictly one frame per executed action).
            if max_actions_i <= 0:
                while produced < total_needed and frame_idx < int(num_frames):
                    im = _capture_unrealcv_lit_pil(env, cam_id=0)
                    frame_idx += 1
                    produced += 1
                    if bool(save_images):
                        _save_pil_jpeg(im, os.path.join(images_dir, f"frame_{frame_idx:04d}.jpg"), quality=95)
                    new_frames.append(im)

            # Patch the latest segment action log with per-action frame names (best-effort):
            # We rewrite the file only if it exists and lengths match.
            try:
                if seg >= 0 and action_image_names and len(action_image_names) == len(pending_actions):
                    p_actions = os.path.join(out_dir, f"{base_name}_seg{seg:02d}_actions.json")
                    if os.path.exists(p_actions):
                        obj = _read_json(p_actions)
                        obj["action_frames"] = action_image_names
                        with open(p_actions, "w", encoding="utf-8") as f:
                            json.dump(obj, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        if not new_frames:
            break
        last_upload_n = int(len(new_frames))
        resp = _post_frames(new_frames, include_instruction=False)

    # Best-effort: update summary with final counters
    try:
        p_sum = os.path.join(out_dir, f"{base_name}_summary.json")
        if os.path.exists(p_sum):
            obj = _read_json(p_sum)
            obj["final"] = {
                "frames_captured": int(frame_idx),
                "actions_executed": int(actions_executed),
            }
            with open(p_sum, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _chunks_stream(paths: List[str], *, num_frames: int, step: int) -> List[List[str]]:
    p = paths[: int(num_frames)]
    chunks: List[List[str]] = []
    chunks.append(p[:1])
    idx = 1
    while idx < int(num_frames):
        chunks.append(p[idx : min(int(num_frames), idx + int(step))])
        idx += int(step)
    return chunks


def _obs_points(pred_num_frames: int, step: int) -> List[int]:
    """Same points logic as server: [1, 1+step, 1+2*step, ..., num_frames]."""
    end = int(pred_num_frames)
    if end <= 0:
        return []
    pts = [1]
    k = 1
    while True:
        v = 1 + k * int(step)
        if v >= end:
            break
        pts.append(v)
        k += 1
    if pts[-1] != end:
        pts.append(end)
    return pts


def _chunks_prefix(paths: List[str], *, num_frames: int, step: int) -> List[List[str]]:
    """
    Prefix mode: send full prefix each call to match v2v semantics:
      call0: [1]
      call1: [1..17]
      call2: [1..33]
      call3: [1..49]
    """
    p = paths[: int(num_frames)]
    pts = _obs_points(int(num_frames), int(step))
    if not pts or pts[0] != 1 or pts[-1] != int(num_frames):
        raise ValueError(f"bad points computed: {pts} for num_frames={num_frames} step={step}")
    return [p[: int(k)] for k in pts]


def _chunks_prefix_from_points(paths: List[str], *, points: List[int]) -> List[List[str]]:
    """Build prefix chunks given explicit absolute points (e.g. [1,17,33])."""
    if not points or int(points[0]) != 1:
        raise ValueError(f"bad points: {points}")
    max_k = int(max(points))
    if len(paths) < max_k:
        raise ValueError(f"need >={max_k} frames for points={points}, got {len(paths)}")
    p = paths[:max_k]
    return [p[: int(k)] for k in points]


def run_one_route(
    *,
    route: Route,
    server_base_url: str,
    out_root: str,
    save_dir_name: Optional[str],
    session_id: str,
    num_frames: int,
    step: int,
    prefix_mode: bool = False,
    allow_future_last_segment: bool = False,
    action_frame: str = "world",
    body_apply_order: str = "yaw_first",
    timeout_s: int = 120,
    image_codec: str = "jpeg",
    jpeg_quality: int = 90,
    pad_short_real: bool = False,
    action_head_mode: str = "actionhead_ref_vit",
    action_head_batch_size: int = 8,
    action_head_stride: int = 1,
    action_head_pre_resize_hw: int = 256,
) -> None:
    prompt = _load_prompt(route.meta_path)
    if not prompt:
        raise RuntimeError(f"empty prompt: {route.meta_path}")
    start_pose = _load_start_pose_cm_deg(route.raw_logs_path)

    real_paths = _sorted_frame_paths(route.images_dir)
    real_count = int(len(real_paths))

    obs_points = _obs_points(int(num_frames), int(step))  # e.g. [1,17,33,49]
    if len(obs_points) < 2:
        raise RuntimeError(f"bad points computed: {obs_points} for num_frames={num_frames} step={step}")

    # If we allow emitting the last segment without requiring points[-1] real frames,
    # only the prefix up to points[-2] (e.g. 33) must exist in the dataset.
    send_points = obs_points
    if bool(prefix_mode) and bool(allow_future_last_segment) and int(obs_points[-1]) > int(obs_points[-2]):
        send_points = obs_points[:-1]  # drop final 49

    max_need = int(max(send_points))
    frame_paths = _take_with_pad(real_paths, max_need, bool(pad_short_real))

    paths_for_map: List[str] = []
    if prefix_mode:
        chunks = _chunks_prefix_from_points(frame_paths, points=[int(k) for k in send_points])
        paths_for_map = frame_paths
        # trigger last segment (seg02) without adding more real frames:
        # resend the max prefix once more (server sees no new frames but will emit last seg).
        if bool(allow_future_last_segment) and int(obs_points[-1]) > int(obs_points[-2]):
            chunks.append(chunks[-1])
    else:
        # stream mode uses incremental chunks, requires full num_frames frames
        full_paths = _take_with_pad(real_paths, int(num_frames), bool(pad_short_real))
        chunks = _chunks_stream(full_paths, num_frames=int(num_frames), step=int(step))
        paths_for_map = full_paths

    out_dir = os.path.join(out_root, str(save_dir_name or session_id))
    os.makedirs(out_dir, exist_ok=True)

    # Save input summary once
    summary = {
        "session_id": session_id,
        "route_id": route.route_id,
        "route_dir": route.route_dir,
        "server_base_url": server_base_url,
        "endpoint": "/v1/predict_delta_actions",
        "prompt": prompt,
        "start_pose_cm_deg_order_xyz_roll_yaw_pitch": start_pose,
        "num_frames": int(num_frames),
        "step": int(step),
        "real_frame_count": real_count,
        "pad_short_real": bool(pad_short_real),
        "allow_future_last_segment": bool(allow_future_last_segment),
        "action_frame_for_integration": str(action_frame),
        "body_apply_order": str(body_apply_order),
        "frames_sent": {"total": int(num_frames), "chunks": [len(c) for c in chunks], "prefix_mode": bool(prefix_mode)},
        "action_head_mode": str(action_head_mode),
        "action_head_batch_size": int(action_head_batch_size),
        "action_head_stride": int(action_head_stride),
        "action_head_pre_resize_hw": int(action_head_pre_resize_hw),
        "image_codec": str(image_codec),
        "jpeg_quality": int(jpeg_quality),
        "action_order_server": ["dx", "dy", "dz", "droll", "dyaw", "dpitch"],
        "pose_order": ["x", "y", "z", "roll", "yaw", "pitch"],
        "units": {"translation": "cm", "angles": "deg"},
        "time": int(time.time()),
    }
    with open(os.path.join(out_dir, f"{session_id}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    cur_pose = start_pose[:]  # absolute pose in cm/deg

    for call_i, chunk_paths in enumerate(chunks):
        images_b64 = [_image_to_data_url(p, codec=str(image_codec), quality=int(jpeg_quality)) for p in chunk_paths]
        payload = {
            "session_id": session_id,
            "images_base64": images_b64,
            "prefix_mode": bool(prefix_mode),
            "allow_future_last_segment": bool(allow_future_last_segment),
            "action_head_mode": str(action_head_mode),
            "action_head_batch_size": int(action_head_batch_size),
            "action_head_stride": int(action_head_stride),
            "action_head_pre_resize_hw": int(action_head_pre_resize_hw),
        }
        if call_i == 0:
            payload["instruction"] = prompt

        resp = _http_post_json(server_base_url.rstrip("/") + "/v1/predict_delta_actions", payload, timeout_s=int(timeout_s))

        actions = resp.get("actions", [])
        seg = int(resp.get("segment_index", -1))
        # Warmup call (first frame) may return no actions with segment_index=-1.
        if seg < 0:
            continue
        # Validate action count by mode
        mode_l = str(action_head_mode).strip().lower()
        if mode_l in ("actionhead_ref_vit", "actionhead_ref", "actionhead_vit", "ref_vit", "actionhead"):
            if seg < 0 or seg >= len(obs_points) - 1:
                raise RuntimeError(f"bad segment_index from server: seg={seg} obs_points={obs_points}")
            expected_n = int(obs_points[seg + 1]) - int(obs_points[seg])  # usually == step
        else:
            expected_n = 4

        if not isinstance(actions, list) or len(actions) != int(expected_n):
            raise RuntimeError(
                f"server returned invalid actions at call={call_i}, segment={seg}: {type(actions)} len={getattr(actions,'__len__',lambda:None)()}"
            )

        actions_server = [[float(x) for x in a] for a in actions]
        actions_client = [_reorder_server_to_client(a) for a in actions_server]
        cumsum_server = _cumsum_actions(actions_server)
        cumsum_client = _cumsum_actions(actions_client)

        # Per-action frame mapping (dataset mode):
        # Only meaningful for actionhead_ref_vit (per-frame actions).
        action_frames: List[Dict[str, Any]] = []
        if mode_l in ("actionhead_ref_vit", "actionhead_ref", "actionhead_vit", "ref_vit", "actionhead") and seg >= 0 and seg < len(obs_points) - 1:
            obs_len = int(obs_points[seg])
            for i in range(int(len(actions_server))):
                abs_from = int(obs_len) + int(i)
                abs_to = int(obs_len) + int(i) + 1
                img_path_upload: Optional[str] = None
                img_path_real: Optional[str] = None

                # Upload-mapped frame (includes padding/repeats when --pad_short_real=1).
                if 1 <= abs_to <= len(paths_for_map):
                    img_path_upload = paths_for_map[int(abs_to) - 1]

                # Real dataset frame (may exist even if not uploaded in allow_future_last_segment mode).
                if 1 <= abs_to <= int(real_count):
                    img_path_real = real_paths[int(abs_to) - 1]

                is_real = 1 <= abs_to <= int(real_count)
                is_padded = (not is_real) and (img_path_upload is not None)

                # Backward-compatible single path: prefer real, else upload (padded) if available.
                img_path = img_path_real or img_path_upload
                action_frames.append(
                    {
                        "index": int(i),
                        "abs_from": int(abs_from),
                        "abs_to": int(abs_to),
                        "image_path": img_path,
                        "image_path_real": img_path_real,
                        "image_path_upload": img_path_upload,
                        "image_available": bool(img_path) and os.path.exists(str(img_path)),
                        "image_available_real": bool(img_path_real) and os.path.exists(str(img_path_real)),
                        "image_available_upload": bool(img_path_upload) and os.path.exists(str(img_path_upload)),
                        "is_real_frame": bool(is_real),
                        "is_padded_frame": bool(is_padded),
                    }
                )

        # Write actions json (one file per segment)
        actions_json = {
            "session_id": session_id,
            "route_id": route.route_id,
            "segment_index": seg,
            "call_index": call_i,
            "frames_in_call": len(chunk_paths),
            "num_received_frames": resp.get("num_received_frames", None),
            "done": resp.get("done", None),
            "prefix_latents": resp.get("prefix_latents", None),
            "units": {"translation": "cm", "angles": "deg"},
            "action_head_mode": str(action_head_mode),
            "num_actions": int(len(actions_server)),
            "action_order_server": ["dx", "dy", "dz", "droll", "dyaw", "dpitch"],
            "action_order_client": ["dx", "dy", "dz", "droll", "dpitch", "dyaw"],
            "actions_server_order": actions_server,
            "actions_client_order": actions_client,
            "action_frames": action_frames,
            "cumsum_server_order": cumsum_server,
            "cumsum_client_order": cumsum_client,
        }
        with open(os.path.join(out_dir, f"{session_id}_seg{seg:02d}_actions.json"), "w", encoding="utf-8") as f:
            json.dump(actions_json, f, ensure_ascii=False, indent=2)

        # Absolute pose points (integrate with configured frame/order)
        pose_points: List[List[float]] = []
        if seg == 0:
            pose_points.append(cur_pose[:])

        for a in actions_server:
            cur_pose = _apply_action_to_pose_with_frame(
                cur_pose,
                a,
                action_frame=str(action_frame),
                body_apply_order=str(body_apply_order),
                integrate_roll_pitch=True,
            )
            pose_points.append(cur_pose[:])

        poses_json = {
            "session_id": session_id,
            "route_id": route.route_id,
            "segment_index": seg,
            "call_index": call_i,
            "units": {"translation": "cm", "angles": "deg"},
            "pose_order": ["x", "y", "z", "roll", "yaw", "pitch"],
            "points": pose_points,  # seg0: 1+N points (start + N), else: N points
        }
        with open(os.path.join(out_dir, f"{session_id}_seg{seg:02d}_poses.json"), "w", encoding="utf-8") as f:
            json.dump(poses_json, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="dataset", choices=["dataset", "unrealcv"], help="dataset: send frames from dataset_root; unrealcv: capture frames from gym_unrealcv using task json(s).")
    ap.add_argument("--dataset_root", type=str, default="", help="(mode=dataset) Dataset root containing route folders with images/ + meta.json (+ raw_logs.json optional).")
    ap.add_argument("--task_json", type=str, default="", help="(mode=unrealcv) Single task json path, e.g. ./select_from_train_jsons/2025-03-30_11-49-14.json")
    ap.add_argument("--json_folder", type=str, default="", help="(mode=unrealcv) Folder containing multiple task json files.")
    ap.add_argument("--json_order", type=str, default="asc", choices=["asc", "desc"], help="(mode=unrealcv) Order for iterating --json_folder by filename.")
    ap.add_argument("--json_start", type=str, default="", help="(mode=unrealcv) Optional lower bound filename for --json_folder (e.g. 2025-03-30_12-02-10 or 2025-03-30_12-02-10.json).")
    ap.add_argument("--json_start_exclusive", type=int, default=0, choices=[0, 1], help="(mode=unrealcv) If 1, start strictly after --json_start (resume mode).")
    ap.add_argument("--json_end", type=str, default="", help="(mode=unrealcv) Optional upper bound filename for --json_folder (inclusive). Same format as --json_start.")
    ap.add_argument("--env_id", type=str, default="UnrealTrack-DowntownWest-ContinuousColor-v0", help="(mode=unrealcv) gym env id.")
    ap.add_argument("--time_dilation", type=int, default=10, help="(mode=unrealcv) Time dilation wrapper parameter.")
    ap.add_argument("--seed", type=int, default=0, help="(mode=unrealcv) Random seed.")
    ap.add_argument("--resolution", type=str, default="256,256", help="(mode=unrealcv) Capture resolution as 'W,H'. Default 256,256.")
    ap.add_argument("--ue_port", type=int, default=0, help="(mode=unrealcv) If >0, set UnrealCV socket base port in unrealcv.ini before launching UE. Useful to run multiple UE instances in parallel (e.g. 9393/9394).")
    ap.add_argument("--yaw_offset_deg", type=float, default=-180.0, help="(mode=unrealcv) Yaw offset applied when calling UnrealCV set_rotation. batch_run_act_all.py uses -180.")
    ap.add_argument("--action_frame", type=str, default="body", choices=["world", "body"], help="How to interpret dx/dy when integrating poses/logs: world=direct add; body=dx,dy in body forward/right rotated by yaw.")
    ap.add_argument("--body_apply_order", type=str, default="yaw_first", choices=["yaw_first", "trans_first", "midpoint"], help="Only for --action_frame=body. yaw_first: yaw+=dyaw then translate; trans_first: translate with old yaw then yaw+=dyaw; midpoint: translate with yaw+0.5*dyaw then yaw+=dyaw.")
    ap.add_argument("--max_actions", type=int, default=48, help="(mode=unrealcv) If >0, stop after executing this many actions (each action -> 1 captured frame). Default 48 (i.e. 1+48=49 frames total). Set 0 to use --num_frames bound instead.")
    ap.add_argument("--server_url", type=str, default="http://127.0.0.1:8002", help="Server base URL (no trailing endpoint).")
    ap.add_argument("--out_dir", type=str, default=r"E:\xjc\UAV-Flow-main1\UAV-Flow-main\UAV-Flow-Eval\cache", help="Where to write per-session json outputs.")
    ap.add_argument("--route_id", type=str, default="", help="If set, only run this route id (folder name).")
    ap.add_argument("--max_routes", type=int, default=0, help="If >0, limit number of routes processed.")
    ap.add_argument("--select_n", type=int, default=0, help="If >0, pick first N routes after filtering (deterministic).")
    ap.add_argument("--min_real_frames", type=int, default=0, help="If >0, only keep routes with real image count >= this threshold.")
    ap.add_argument("--pad_short_real", type=int, default=0, choices=[0, 1], help="If 1, pad short routes by repeating last frame to reach required frames.")
    ap.add_argument("--run_id", type=str, default="", help="Optional run id. If empty, uses timestamp.")
    ap.add_argument("--run_subdir", type=int, default=1, choices=[0, 1], help="If 1, write outputs under out_dir/client_run_<run_id>/ to avoid overwrite.")
    ap.add_argument("--session_id_mode", type=str, default="route_run", choices=["route", "route_run"], help="Server session_id naming to avoid overwriting server latent dirs.")
    ap.add_argument("--allow_future_last_segment", type=int, default=1, choices=[0, 1], help="If 1, allow seg02 emission with only 33 real prefix frames (34-49 predicted). Requires server support.")
    ap.add_argument("--dry_run", type=int, default=0, choices=[0, 1], help="If 1, only print selected routes and exit.")
    ap.add_argument("--image_codec", type=str, default="jpeg", choices=["jpeg", "jpg", "png"], help="Image codec for uploading frames to server.")
    ap.add_argument("--jpeg_quality", type=int, default=90, help="Only used when --image_codec=jpeg/jpg.")
    ap.add_argument("--prefix_mode", type=int, default=0, choices=[0, 1], help="If 1, send full prefix each call: 1,1-17,1-33,... (requires server support).")
    ap.add_argument("--timeout_s", type=int, default=600, help="HTTP request timeout (seconds). seg0 inference can take several minutes.")
    ap.add_argument("--action_head_mode", type=str, default="actionhead_ref_vit", choices=["tsformer_latent", "actionhead_ref_vit"], help="Which action head mode to request from server.")
    ap.add_argument("--action_head_batch_size", type=int, default=8)
    ap.add_argument("--action_head_stride", type=int, default=1)
    ap.add_argument("--action_head_pre_resize_hw", type=int, default=256, help="Intermediate pre-resize before actionhead preprocessing (default 256).")
    ap.add_argument("--config_json", type=str, default="", help="Optional: server-style config.json; if set, reads infinity.num_frames/step from it.")
    ap.add_argument("--num_frames", type=int, default=81, help="Fallback if --config_json not set.")
    ap.add_argument("--step", type=int, default=16, help="Fallback if --config_json not set.")
    args = ap.parse_args()

    if args.config_json.strip():
        num_frames, step = _load_num_frames_step_from_config(args.config_json.strip())
    else:
        num_frames, step = int(args.num_frames), int(args.step)

    run_id = (args.run_id or "").strip() or _time_id()

    out_root = os.path.abspath(args.out_dir)
    os.makedirs(out_root, exist_ok=True)

    if bool(int(args.dry_run)):
        print(f"[dry_run] run_id={run_id} mode={args.mode} out_root={out_root}")
        return

    if str(args.mode).strip().lower() == "dataset":
        if not str(args.dataset_root).strip():
            raise SystemExit("--dataset_root is required when --mode=dataset")
        dataset_root = os.path.abspath(args.dataset_root)
        routes = _discover_routes(dataset_root)
        if args.route_id.strip():
            routes = [r for r in routes if r.route_id == args.route_id.strip()]
        if args.min_real_frames and int(args.min_real_frames) > 0:
            keep: List[Route] = []
            for r in routes:
                try:
                    n = len(_sorted_frame_paths(r.images_dir))
                except Exception:
                    n = 0
                if int(n) >= int(args.min_real_frames):
                    keep.append(r)
            routes = keep

        # Deterministic selection
        if args.select_n and int(args.select_n) > 0:
            routes = routes[: int(args.select_n)]
        if args.max_routes and args.max_routes > 0:
            routes = routes[: int(args.max_routes)]
        if not routes:
            raise SystemExit("No valid routes found.")

        out_root_eff = out_root
        if bool(int(args.run_subdir)):
            out_root_eff = os.path.join(out_root_eff, f"client_run_{run_id}")
        os.makedirs(out_root_eff, exist_ok=True)

        for r in routes:
            if str(args.session_id_mode) == "route_run":
                session_id = f"{r.route_id}__{run_id}"
            else:
                session_id = r.route_id
            save_dir_name = r.route_id  # stable folder name under this run
            try:
                real_n = len(_sorted_frame_paths(r.images_dir))
            except Exception:
                real_n = -1
            print(f"[run] mode=dataset route_id={r.route_id} session_id={session_id} real_frames={real_n} images_dir={r.images_dir}")
            try:
                run_one_route(
                    route=r,
                    server_base_url=args.server_url,
                    out_root=out_root_eff,
                    save_dir_name=save_dir_name,
                    session_id=session_id,
                    num_frames=int(num_frames),
                    step=int(step),
                    prefix_mode=bool(int(args.prefix_mode)),
                    allow_future_last_segment=bool(int(args.allow_future_last_segment)),
                    action_frame=str(args.action_frame),
                    body_apply_order=str(args.body_apply_order),
                    timeout_s=int(args.timeout_s),
                    image_codec=str(args.image_codec),
                    jpeg_quality=int(args.jpeg_quality),
                    pad_short_real=bool(int(args.pad_short_real)),
                    action_head_mode=str(args.action_head_mode),
                    action_head_batch_size=int(args.action_head_batch_size),
                    action_head_stride=int(args.action_head_stride),
                    action_head_pre_resize_hw=int(args.action_head_pre_resize_hw),
                )
            except Exception as e:
                print(f"[fail] route_id={r.route_id} err={e}")
        return

    # mode=unrealcv
    task_paths: List[str] = []
    if str(args.task_json).strip():
        task_paths = [os.path.abspath(str(args.task_json).strip())]
    elif str(args.json_folder).strip():
        jf = os.path.abspath(str(args.json_folder).strip())
        if not os.path.isdir(jf):
            raise SystemExit(f"--json_folder is not a dir: {jf}")
        names = [n for n in os.listdir(jf) if n.lower().endswith(".json")]
        names.sort(reverse=(str(args.json_order).strip().lower() == "desc"))
        # Optional filename bounds
        def _norm_json_name(s: str) -> str:
            s = str(s or "").strip()
            if not s:
                return ""
            return s if s.lower().endswith(".json") else (s + ".json")

        start_name = _norm_json_name(str(args.json_start))
        end_name = _norm_json_name(str(args.json_end))
        if start_name:
            if bool(int(args.json_start_exclusive)):
                names = [n for n in names if str(n) > str(start_name)]
            else:
                names = [n for n in names if str(n) >= str(start_name)]
        if end_name:
            names = [n for n in names if str(n) <= str(end_name)]
        task_paths = [os.path.join(jf, n) for n in names]
    else:
        raise SystemExit("--task_json or --json_folder is required when --mode=unrealcv")
    if not task_paths:
        raise SystemExit("No task json files found.")

    # parse resolution string "W,H"
    try:
        parts = [p.strip() for p in str(args.resolution).split(",")]
        res_w = int(parts[0])
        res_h = int(parts[1])
    except Exception as e:
        raise SystemExit(f"bad --resolution '{args.resolution}', expect 'W,H'") from e

    # Lazy import gym_unrealcv stack
    try:
        import gym  # type: ignore
        import gym_unrealcv  # noqa: F401  # type: ignore
        from gym_unrealcv.envs.wrappers import configUE, time_dilation  # type: ignore
    except Exception as e:
        raise SystemExit(f"mode=unrealcv requires gym & gym_unrealcv imports, but failed: {e}")

    env = gym.make(str(args.env_id))
    if int(args.time_dilation) > 0:
        env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
    try:
        env.unwrapped.agents_category = ["drone"]
    except Exception:
        pass
    env = configUE.ConfigUEWrapper(env, resolution=(int(res_w), int(res_h)))
    try:
        env.seed(int(args.seed))
    except Exception:
        pass
    try:
        # Optional: pin UnrealCV port base for parallel UE instances
        if int(args.ue_port) > 0:
            try:
                env.unwrapped.ue_binary.write_port(int(args.ue_port))
            except Exception:
                # Best-effort only; launcher will still auto-increment if needed
                pass
        env.reset()
        env.unwrapped.unrealcv.set_viewport(env.unwrapped.player_list[0])
        env.unwrapped.unrealcv.set_phy(env.unwrapped.player_list[0], 0)
    except Exception:
        pass

    for p in task_paths:
        print(f"[run] mode=unrealcv task={p} env_id={args.env_id} resolution={res_w}x{res_h}")
        try:
            run_one_task_unrealcv(
                task_json_path=p,
                env=env,
                env_id=str(args.env_id),
                server_base_url=args.server_url,
                out_root=out_root,
                run_id=run_id,
                num_frames=int(num_frames),
                step=int(step),
                max_actions=int(args.max_actions),
                timeout_s=int(args.timeout_s),
                action_head_mode=str(args.action_head_mode),
                action_head_batch_size=int(args.action_head_batch_size),
                action_head_stride=int(args.action_head_stride),
                action_head_pre_resize_hw=int(args.action_head_pre_resize_hw),
                image_codec=str(args.image_codec),
                jpeg_quality=int(args.jpeg_quality),
                yaw_offset_deg=float(args.yaw_offset_deg),
                allow_future_last_segment=bool(int(args.allow_future_last_segment)),
                action_frame=str(args.action_frame),
                body_apply_order=str(args.body_apply_order),
                save_images=True,
            )
        except Exception as e:
            print(f"[fail] task={p} err={e}")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()

