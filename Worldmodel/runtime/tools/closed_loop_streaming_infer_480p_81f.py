#!/usr/bin/env python3
# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
Helper script for InfinityStar closed-loop / streaming inference.

In the open-source layout, this file is mainly used for:

- A shared argument builder reused by `infer/server.py` and `action_aware_grpo/grpo_server.py`.
- Offline debugging: replay a sequence of real frames through the streaming pipeline.

Prepare a directory of real frames (named in chronological order) and the corresponding prompt.
The script will sort frames automatically and print inference results.

Example:
  python3 tools/closed_loop_streaming_infer_480p_81f.py \
    --ckpt ./checkpoints/model.pth \
    --route_dir ./data/reference_route \
    --prompt_key instruction \
    --out_dir ./output_streaming_closed_loop
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
import sys
import time
from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch
from PIL import Image

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.infinity_streaming_session import InfinityStreamingSession
from tools.run_infinity import load_tokenizer, load_transformer, load_visual_tokenizer, save_video, transform


def _is_safetensors_shard_dir(path: str) -> bool:
    if not path or not osp.isdir(path):
        return False
    return bool(glob.glob(osp.join(path, "*.safetensors"))) or bool(glob.glob(osp.join(path, "*.safetensors.index.json")))


def _resolve_checkpoint_layout(ckpt: str) -> dict[str, str]:
    ckpt = osp.abspath(ckpt)

    gpt_dir = osp.join(ckpt, "gpt")
    vae_dir = osp.join(ckpt, "vae")
    if osp.isdir(ckpt) and _is_safetensors_shard_dir(gpt_dir) and _is_safetensors_shard_dir(vae_dir):
        return {
            "checkpoint_layout": "hf_repo",
            "checkpoint_type": "torch_shard",
            "model_path": gpt_dir,
            "vae_model_path": vae_dir,
        }

    if _is_safetensors_shard_dir(ckpt):
        return {
            "checkpoint_layout": "torch_shard",
            "checkpoint_type": "torch_shard",
            "model_path": ckpt,
            "vae_model_path": "",
        }

    return {
        "checkpoint_layout": "torch",
        "checkpoint_type": "torch",
        "model_path": ckpt,
        "vae_model_path": "",
    }


def _sorted_images(image_dir: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(osp.join(image_dir, e)))
    files = sorted(files)
    return files


def _read_prompt(prompt: Optional[str], prompt_json: Optional[str], prompt_key: str) -> str:
    if prompt and prompt.strip():
        return prompt.strip()
    if not prompt_json:
        raise ValueError("Must provide --prompt or --prompt_json")
    with open(prompt_json, "r", encoding="utf-8") as f:
        pj = json.load(f)
    p = pj.get(prompt_key) or pj.get("instruction_unified") or pj.get("instruction") or pj.get("prompt")
    if not isinstance(p, str) or not p.strip():
        raise ValueError(f"Prompt not found in {prompt_json} (key={prompt_key})")
    return p.strip()


def _make_args(
    *,
    ckpt: str,
    pn: str,
    fps: int,
    num_frames: int,
    seed: int,
    dynamic_scale_schedule: str,
    mask_type: str,
    cfg: float,
    tau_image: float,
    tau_video: float,
) -> SimpleNamespace:
    ckpt_dir = osp.abspath(os.environ.get("CHECKPOINTS_DIR", osp.join(REPO_ROOT, "checkpoint")))
    ckpt_layout = _resolve_checkpoint_layout(ckpt)
    # Prefer the repo's Args (has a complete set of defaults) to avoid missing-field crashes.
    # Fallback to SimpleNamespace if Tap/Args is unavailable in current python env.
    try:
        from infinity.utils.arg_util import Args as _Args  # type: ignore
        a = _Args()
    except Exception:
        a = SimpleNamespace()
    a.pn = pn
    a.fps = int(fps)
    # Keep both names (some utilities use video_fps).
    a.video_fps = int(fps)
    a.video_frames = int(num_frames)  # model configured with the max length (81 in our use-case)
    a.temporal_compress_rate = 4
    a.videovae = 10
    a.vae_type = 64
    a.vae_path = osp.abspath(os.environ.get("VAE_PATH", osp.join(ckpt_dir, "infinitystar_videovae.pth")))
    a.text_encoder_ckpt = osp.abspath(os.environ.get("T5_PATH", osp.join(ckpt_dir, "text_encoder", "flan-t5-xl-official")))
    a.text_channels = 2048
    # Keep aliases used across repo scripts.
    a.Ct5 = a.text_channels
    a.tlen = 512
    a.simple_text_proj = 1
    a.model_type = "infinity_qwen8b"
    a.model_path = ckpt_layout["model_path"]
    a.checkpoint_type = ckpt_layout["checkpoint_type"]
    a.checkpoint_layout = ckpt_layout["checkpoint_layout"]
    a.vae_model_path = ckpt_layout["vae_model_path"]

    # match finetune schedule family
    a.dynamic_scale_schedule = dynamic_scale_schedule
    a.mask_type = mask_type

    # inference knobs (match tools/infer_video_480p.py & our finetune inference script)
    a.use_flex_attn = True
    block_dtype = os.environ.get("INFINITY_BLOCK_DTYPE", "float32").strip().lower()
    if block_dtype not in {"float32", "bfloat16"}:
        raise ValueError("INFINITY_BLOCK_DTYPE must be float32 or bfloat16")
    # Keep FP32 master weights by default for subsequent policy optimization.
    # BF16 storage is supported for a frozen-backbone/heads-only policy update.
    a.bf16 = int(block_dtype == "bfloat16")
    a.use_apg = 1
    a.use_cfg = 0
    a.cfg = float(cfg)
    a.tau_image = float(tau_image)
    a.tau_video = float(tau_video)
    a.apg_norm_threshold = 0.05
    a.append_duration2caption = 1
    a.use_two_stage_lfq = 1
    a.detail_scale_min_tokens = 350
    a.semantic_scales = 11
    # These are required by the VideoVAE constructor (global_args.*) and also used by Infinity heads.
    # Match repo defaults / finetune scripts.
    a.semantic_scale_dim = 16
    a.detail_scale_dim = 64
    a.use_learnable_dim_proj = 0
    a.use_feat_proj = 2
    # Additional args accessed by Infinity __init__ / attention mask / RoPE helpers.
    a.context_frames = getattr(a, "context_frames", 10000)
    a.steps_per_frame = getattr(a, "steps_per_frame", 3)
    a.inject_sync = getattr(a, "inject_sync", 0)
    a.rope_type = getattr(a, "rope_type", "4d")
    a.image_batch_size = getattr(a, "image_batch_size", 0)
    a.video_batch_size = getattr(a, "video_batch_size", 1)
    a.train_with_var_seq_len = getattr(a, "train_with_var_seq_len", 0)
    a.train_max_token_len = getattr(a, "train_max_token_len", -1)
    a.noise_input = getattr(a, "noise_input", 0)
    a.max_repeat_times = 10000
    a.apply_spatial_patchify = 0
    a.num_of_label_value = 2
    a.rope2d_each_sa_layer = 1
    a.rope2d_normalized_by_hw = 2
    a.pad_to_multiplier = 128
    a.seed = int(seed)

    # repetition configs (14 scales for 0.40M)
    a.image_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]"
    a.video_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]"
    return a


def _load_obs_video_bcthw(frame_paths: List[str], tgt_h: int, tgt_w: int) -> torch.Tensor:
    """Return [1,3,T,H,W] float in [-1,1] (CPU tensor)."""
    frames = []
    for p in frame_paths:
        pil = Image.open(p).convert("RGB")
        frames.append(transform(pil, tgt_h, tgt_w))  # [3,H,W] in [-1,1]
    video_T3HW = torch.stack(frames, dim=0)  # [T,3,H,W]
    return video_T3HW.permute(1, 0, 2, 3).unsqueeze(0)  # [1,3,T,H,W]

def _take_with_pad(paths: List[str], n: int, pad_short_real: bool) -> List[str]:
    if len(paths) >= n:
        return paths[:n]
    if not paths:
        raise ValueError("no real frames found")
    if not pad_short_real:
        raise ValueError(f"Not enough real frames: need={n} but only={len(paths)} (use --pad_short_real to pad)")
    # Repeat last frame path to reach n.
    return paths + [paths[-1]] * (n - len(paths))


def _obs_points(total_gt_frames: int, pred_num_frames: int, step: int) -> List[int]:
    end = min(int(total_gt_frames), int(pred_num_frames))
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="global_step_*.pth (checkpoint_type=torch)")
    ap.add_argument("--route_dir", type=str, default="", help="Route directory (reads route_dir/images and route_dir/meta.json by default)")
    ap.add_argument("--frames_dir", type=str, default="", help="Directory of real observation frames (sorted by filename). Can be omitted if --route_dir is set.")
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--prompt_json", type=str, default="")
    ap.add_argument("--prompt_key", type=str, default="instruction_unified")
    ap.add_argument("--negative_prompt", type=str, default="")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--step", type=int, default=16)
    ap.add_argument("--save_full_pred", action="store_true", help="Also save the full N-frame predicted video for each inference call")
    ap.add_argument("--pad_short_real", action="store_true", default=True, help="If real frames are fewer than num_frames, pad by repeating the last frame (enabled by default)")
    ap.add_argument("--no_pad_short_real", action="store_false", dest="pad_short_real", help="Disable padding; fail/stop early when real frames are insufficient")
    ap.add_argument("--watch", action="store_true", help="Watch mode: wait for new frames under frames_dir and run inference once enough frames are available")
    ap.add_argument("--poll_interval", type=float, default=0.5, help="Polling interval (seconds) for watch mode")
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--pn", type=str, default="0.40M")
    ap.add_argument("--h_div_w_template", type=float, default=0.562)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg", type=float, default=34.0)
    ap.add_argument("--tau_image", type=float, default=1.0)
    ap.add_argument("--tau_video", type=float, default=0.4)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=0.0)

    ap.add_argument("--dynamic_scale_schedule", type=str, default="infinity_elegant_clip20frames_v2_allpt")
    ap.add_argument("--mask_type", type=str, default="infinity_elegant_clip20frames_v2_allpt")
    args_cli = ap.parse_args()

    # local rank support (torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    ckpt = osp.abspath(args_cli.ckpt)
    route_dir = osp.abspath(args_cli.route_dir) if args_cli.route_dir else ""
    frames_dir = osp.abspath(args_cli.frames_dir) if args_cli.frames_dir else ""
    if route_dir:
        if not frames_dir:
            frames_dir = osp.join(route_dir, "images")
        if not args_cli.prompt_json:
            args_cli.prompt_json = osp.join(route_dir, "meta.json")
    out_dir = osp.abspath(args_cli.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    prompt = _read_prompt(args_cli.prompt, args_cli.prompt_json or None, args_cli.prompt_key)
    prompt = prompt.strip()
    negative_prompt = (args_cli.negative_prompt or "").strip()

    if not frames_dir:
        raise ValueError("Must provide --frames_dir or --route_dir")

    frame_paths = _sorted_images(frames_dir)
    if not frame_paths:
        raise FileNotFoundError(f"No image frames found under {frames_dir} (write at least 1 frame first)")
    if (not args_cli.watch) and (len(frame_paths) < int(args_cli.num_frames)) and bool(args_cli.pad_short_real):
        print(
            f"[warn] real_frames={len(frame_paths)} < num_frames={args_cli.num_frames}; "
            f"padding by repeating the last frame to {args_cli.num_frames} for writing GT cache"
        )

    # Build args + load models
    a = _make_args(
        ckpt=ckpt,
        pn=args_cli.pn,
        fps=args_cli.fps,
        num_frames=args_cli.num_frames,
        seed=args_cli.seed,
        dynamic_scale_schedule=args_cli.dynamic_scale_schedule,
        mask_type=args_cli.mask_type,
        cfg=args_cli.cfg,
        tau_image=args_cli.tau_image,
        tau_video=args_cli.tau_video,
    )

    text_tokenizer, text_encoder = load_tokenizer(t5_path=a.text_encoder_ckpt)
    vae = load_visual_tokenizer(a).float().to("cuda")
    infinity = load_transformer(vae, a)
    infinity.eval().requires_grad_(False)

    session = InfinityStreamingSession(
        args=a,
        infinity_model=infinity,
        vae=vae,
        text_tokenizer=text_tokenizer,
        text_encoder=text_encoder,
        h_div_w_template=float(args_cli.h_div_w_template),
    )

    # output schedule determines target resolution & tau list
    sched_out = session.build_schedule_for_num_frames(num_frames=int(args_cli.num_frames))
    tgt_h, tgt_w = sched_out.tgt_h, sched_out.tgt_w
    tau = [float(a.tau_image)] * int(sched_out.tower_split_index) + [float(a.tau_video)] * (len(sched_out.scale_schedule) - int(sched_out.tower_split_index))

    # prompt with duration tag (match training prompt format)
    dur_s = (int(args_cli.num_frames) - 1) // int(args_cli.fps)
    prompt_infer = f"<<<t={dur_s}s>>>{prompt}" if int(getattr(a, "append_duration2caption", 0)) else prompt

    # Init session (text cache as GT). IMPORTANT: cfg != 1 -> bs=2, so GT caches will be written with bs=2.
    session.reset(prompt_infer, negative_prompt=negative_prompt, cfg_scale=float(a.cfg))

    # Loop points (offline): 1, 17, 33, ... , min(81, len(frames))
    total_for_points = int(args_cli.num_frames) if (not args_cli.watch and bool(args_cli.pad_short_real)) else len(frame_paths)
    points = _obs_points(total_gt_frames=total_for_points, pred_num_frames=int(args_cli.num_frames), step=int(args_cli.step))
    if not args_cli.watch:
        print(f"[info] offline total_gt_frames={len(frame_paths)} pred_num_frames={args_cli.num_frames} points={points}")
    print(f"[info] tgt_h={tgt_h} tgt_w={tgt_w} cfg={a.cfg} tau_image={a.tau_image} tau_video={a.tau_video} top_k={args_cli.top_k} top_p={args_cli.top_p}")

    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = osp.join(out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    with open(osp.join(run_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "ckpt": ckpt,
                "route_dir": route_dir,
                "frames_dir": frames_dir,
                "real_frame_count": len(frame_paths),
                "num_frames": int(args_cli.num_frames),
                "step": int(args_cli.step),
                "pad_short_real": bool(args_cli.pad_short_real),
                "fps": int(args_cli.fps),
                "pn": str(args_cli.pn),
                "h_div_w_template": float(args_cli.h_div_w_template),
                "cfg": float(a.cfg),
                "tau_image": float(a.tau_image),
                "tau_video": float(a.tau_video),
                "top_k": int(args_cli.top_k),
                "top_p": float(args_cli.top_p),
                "dynamic_scale_schedule": str(args_cli.dynamic_scale_schedule),
                "mask_type": str(args_cli.mask_type),
                "prompt": prompt,
                "prompt_infer": prompt_infer,
                "negative_prompt": negative_prompt,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    def _write_gt_obs(cur_frame_paths: List[str], obs_len: int) -> None:
        """Overwrite GT obs cache with cumulative real frames [1..obs_len]."""
        padded = _take_with_pad(cur_frame_paths, int(obs_len), bool(args_cli.pad_short_real))
        obs_cpu = _load_obs_video_bcthw(padded, tgt_h, tgt_w)  # [1,3,T,H,W] CPU
        obs = obs_cpu.to("cuda", non_blocking=True)
        session.compute_kv_cache_gt(obs)

    def _infer_full(step_i: int) -> torch.Tensor:
        """Infer full N frames, return uint8 BGR tensor [T,H,W,3] (first sample)."""
        seed = int(args_cli.seed) + int(step_i)
        t0 = time.time()
        _, img = session.infer_chunk(
            num_frames=int(args_cli.num_frames),
            cfg_list=float(a.cfg),
            tau_list=tau,
            top_k=int(args_cli.top_k),
            top_p=float(args_cli.top_p),
            seed=seed,
            negative_prompt=negative_prompt,
            low_vram_mode=True,
        )
        dt = time.time() - t0
        vid = img[0] if isinstance(img, torch.Tensor) and img.dim() == 5 else img
        print(f"[infer] step={step_i} seed={seed} time={dt:.2f}s cache_stats={session.cache_stats()}")
        return vid

    def _save_segment(vid: torch.Tensor, *, step_i: int, obs_len: int, next_obs_len: int) -> None:
        """
        Save the predicted segment that will be overwritten by next GT write:
        frames (obs_len+1 .. next_obs_len) in 1-indexed convention.
        """
        # vid: [T,H,W,3] with T == num_frames
        start0 = int(obs_len)  # 0-based: obs_len=1 -> start from frame index 1 (2nd frame)
        end0 = int(next_obs_len)
        seg = vid[start0:end0]
        seg_start_1idx = int(obs_len) + 1
        seg_end_1idx = int(next_obs_len)
        save_path = osp.join(run_dir, f"seg_{step_i:02d}_pred_{seg_start_1idx:03d}_{seg_end_1idx:03d}.mp4")
        save_video(seg, fps=int(args_cli.fps), save_filepath=save_path, force_all_keyframes=True)
        print(f"[save] seg step={step_i} pred={seg_start_1idx}-{seg_end_1idx} saved={save_path}")

    def _save_full(vid: torch.Tensor, *, step_i: int, obs_len: int) -> None:
        save_path = osp.join(run_dir, f"full_{step_i:02d}_obs{obs_len:03d}_pred{int(args_cli.num_frames):03d}.mp4")
        save_video(vid, fps=int(args_cli.fps), save_filepath=save_path, force_all_keyframes=True)
        print(f"[save] full step={step_i} saved={save_path}")

    # Always start by writing GT obs for the first frame (obs_len=1).
    session.correction_clear_pred()
    _write_gt_obs(frame_paths, obs_len=1)

    if not args_cli.watch:
        # Offline: iterate intervals [1->17], [17->33], ... and save each predicted segment BEFORE writing next GT.
        if len(points) < 2:
            raise ValueError(f"points={points} too short to do segmented saving (need at least 2 points)")
        for i in range(len(points) - 1):
            obs_len = int(points[i])
            next_obs_len = int(points[i + 1])
            vid = _infer_full(step_i=i)
            if args_cli.save_full_pred:
                _save_full(vid, step_i=i, obs_len=obs_len)
            _save_segment(vid, step_i=i, obs_len=obs_len, next_obs_len=next_obs_len)

            # now overwrite caches with GT up to next_obs_len
            session.correction_clear_pred()
            _write_gt_obs(frame_paths, obs_len=next_obs_len)
        return

    # watch mode: wait for new frames as 1 -> 1+step -> ... -> num_frames
    points_watch = _obs_points(total_gt_frames=10**9, pred_num_frames=int(args_cli.num_frames), step=int(args_cli.step))
    # Ensure at least the first frame exists, then write GT obs_len=1 already done above.
    for i in range(len(points_watch) - 1):
        obs_len = int(points_watch[i])
        next_obs_len = int(points_watch[i + 1])
        vid = _infer_full(step_i=i)
        if args_cli.save_full_pred:
            _save_full(vid, step_i=i, obs_len=obs_len)
        _save_segment(vid, step_i=i, obs_len=obs_len, next_obs_len=next_obs_len)

        # wait until enough frames exist for next_obs_len, then overwrite GT cache
        while True:
            cur = _sorted_images(frames_dir)
            if len(cur) >= next_obs_len:
                frame_paths = cur
                break
            time.sleep(float(args_cli.poll_interval))
        session.correction_clear_pred()
        _write_gt_obs(frame_paths, obs_len=next_obs_len)


if __name__ == "__main__":
    main()

