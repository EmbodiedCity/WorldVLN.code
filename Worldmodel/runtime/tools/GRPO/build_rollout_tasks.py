#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build GRPO rollout task jsonl from a source json.

Output lines are compatible with the existing video dataset reader and include:
  - video_path, begin_frame_id, end_frame_id, fps
  - tarsier2_caption
  - grpo_reward (default 0.0)
  - grpo_old_logprob (default 0.0)
  - optional gt_pose_json for reward computation
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List


def _read_any_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        if isinstance(obj.get("items"), list):
            return [x for x in obj["items"] if isinstance(x, dict)]
        if isinstance(obj.get("data"), list):
            return [x for x in obj["data"] if isinstance(x, dict)]
        return [obj]
    raise ValueError(f"Unsupported json format: {type(obj)}")


def _pick(d: Dict[str, Any], keys: Iterable[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _norm_item(x: Dict[str, Any], default_fps: int, fallback_group_id: str) -> Dict[str, Any]:
    video_path = _pick(x, ("video_path", "source_video", "video", "path"), "")
    caption = _pick(x, ("tarsier2_caption", "instruction_unified", "instruction", "caption", "prompt"), "")
    begin = int(_pick(x, ("begin_frame_id", "start_frame", "start"), 0))
    end = int(_pick(x, ("end_frame_id", "end_frame", "end"), max(begin, begin + 48)))
    fps = int(_pick(x, ("fps", "video_fps"), default_fps))
    gt_pose_json = _pick(x, ("preprocessed_logs_json", "gt_pose_json", "pose_json", "coordinates_path"), "")
    group_id = _pick(x, ("grpo_group_id", "group_id"), "")
    clip_id = int(_pick(x, ("grpo_clip_id", "clip_id"), 1))
    if not group_id:
        group_id = str(fallback_group_id)
    out = {
        "video_path": str(video_path),
        "begin_frame_id": int(begin),
        "end_frame_id": int(end),
        "fps": int(fps),
        "tarsier2_caption": str(caption),
        "grpo_reward": float(_pick(x, ("grpo_reward", "reward"), 0.0)),
        "grpo_reward_act": float(_pick(x, ("grpo_reward_act", "reward_act", "grpo_reward", "reward"), 0.0)),
        "grpo_reward_task": float(_pick(x, ("grpo_reward_task", "reward_task"), 0.0)),
        "grpo_old_logprob": float(_pick(x, ("grpo_old_logprob", "old_logprob"), 0.0)),
        "grpo_group_id": str(group_id),
        "grpo_clip_id": int(clip_id),
    }
    if gt_pose_json:
        out["gt_pose_json"] = str(gt_pose_json)
    # Dataset layout and coordinate frames are user configuration, not paths
    # baked into the released source tree.
    out["gt_pose_frame"] = str(x.get("gt_pose_frame", "initial_body"))
    if x.get("task_id"):
        out["task_id"] = str(x["task_id"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", type=str, required=True)
    ap.add_argument("--output_jsonl", type=str, required=True)
    ap.add_argument("--default_fps", type=int, default=16)
    args = ap.parse_args()

    items = _read_any_json(os.path.abspath(args.input_json))
    lines: List[Dict[str, Any]] = []
    for i, x in enumerate(items):
        try:
            y = _norm_item(x, args.default_fps, fallback_group_id=f"task_{i:06d}")
            if not y["video_path"] or not y["tarsier2_caption"]:
                continue
            lines.append(y)
        except Exception:
            continue

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
    with open(os.path.abspath(args.output_jsonl), "w", encoding="utf-8") as f:
        for item in lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[build_rollout_tasks] wrote {len(lines)} lines -> {os.path.abspath(args.output_jsonl)}")


if __name__ == "__main__":
    main()

