#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Online inference API server used by action-aware GRPO workflows (weights-resident, streaming architecture).

Goals:
- Load the world model and action head once at startup (kept resident in memory/GPU).
- The client streams RGB frames per trajectory (`session_id`):
  - First call: typically 1 warmup frame (and an optional instruction/prompt).
  - Next calls: typically `step` frames per call (accumulated until `num_frames`).
- The server runs the world model to produce summed_codes (latents), then predicts delta actions.
  This entry point retains the legacy TSformer(P2P, window_size=2) path for compatibility.
- Output delta actions are in cm/deg, ordered as: [dx, dy, dz, droll, dyaw, dpitch].

Example:
  export INFINITY_CKPT=/path/to/global_step_xxx.pth
  uvicorn grpo_server:app --host 0.0.0.0 --port 8002

Self-test (requires real checkpoints and a route_dir):
  python3 grpo_server.py --self_test \
    --infinity_ckpt "$INFINITY_CKPT" \
    --route_dir /path/to/route_dir
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

# Optional (only needed for actionhead reference-video mode)
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

# Optional server dependencies (allow running offline eval without fastapi/pydantic installed)
FASTAPI_AVAILABLE = True
try:
    from fastapi import FastAPI, HTTPException  # type: ignore
    from pydantic import BaseModel, Field  # type: ignore
except Exception:
    FASTAPI_AVAILABLE = False

    class HTTPException(RuntimeError):  # minimal stub
        def __init__(self, status_code: int = 500, detail: str = ""):
            super().__init__(f"HTTP {status_code}: {detail}")
            self.status_code = status_code
            self.detail = detail

    class BaseModel:  # minimal stub
        pass

    def Field(default=None, **kwargs):  # noqa: N802
        return default


# -------------------------
# 0) Paths / sys.path
# -------------------------
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

TSFORMER_ROOT = REPO / "Worldmodel" / "action_decoder" / "actionhead_runtime"

if not TSFORMER_ROOT.exists():
    raise FileNotFoundError(f"TSformer repo not found: {TSFORMER_ROOT}")

# TSformer modules
sys.path.insert(0, str(TSFORMER_ROOT))

# -------------------------
# 1) InfinityStar dynamic import (supports INFINITY_REPO_ROOT)
# -------------------------
# NOTE: Worldmodel repo is selected at runtime so this server can embed different
# copies.
DEFAULT_INFINITY_REPO_ROOT = REPO / "Worldmodel" / "runtime"

# Filled by _import_infinity_modules()
InfinityStreamingSession = None  # type: ignore
SelfCorrection = None  # type: ignore
get_dynamic_resolution_meta = None  # type: ignore
_make_infinity_args = None  # type: ignore
load_tokenizer = None  # type: ignore
load_transformer = None  # type: ignore
load_visual_tokenizer = None  # type: ignore
infinity_transform = None  # type: ignore
infinity_save_video = None  # type: ignore
infinity_gen_one_example = None  # type: ignore


def _get_infinity_repo_root() -> Path:
    p = os.environ.get("INFINITY_REPO_ROOT", "").strip()
    if p:
        return Path(p).expanduser().resolve()
    return DEFAULT_INFINITY_REPO_ROOT


def _import_infinity_modules(repo_root: Path) -> None:
    """
    Dynamically import InfinityStar python modules from `repo_root`.
    Must be called before using Infinity-related symbols.
    """
    global InfinityStreamingSession, SelfCorrection, get_dynamic_resolution_meta
    global _make_infinity_args, load_tokenizer, load_transformer, load_visual_tokenizer, infinity_transform, infinity_save_video, infinity_gen_one_example

    if InfinityStreamingSession is not None:
        return
    if not repo_root.exists():
        raise FileNotFoundError(f"InfinityStar repo not found: {repo_root}")
    # Put selected repo at highest priority.
    sys.path.insert(0, str(repo_root))

    from tools.closed_loop_streaming_infer_480p_81f import _make_args as __make_args  # type: ignore
    from tools.infinity_streaming_session import InfinityStreamingSession as __ISS  # type: ignore
    from tools.run_infinity import (  # type: ignore
        load_tokenizer as __load_tokenizer,
        load_transformer as __load_transformer,
        load_visual_tokenizer as __load_visual_tokenizer,
        gen_one_example as __gen_one_example,
        save_video as __save_video,
        transform as __transform,
    )
    from infinity.models.self_correction import SelfCorrection as __SelfCorrection  # type: ignore
    from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta as __get_dyn  # type: ignore

    _make_infinity_args = __make_args
    InfinityStreamingSession = __ISS
    load_tokenizer = __load_tokenizer
    load_transformer = __load_transformer
    load_visual_tokenizer = __load_visual_tokenizer
    infinity_gen_one_example = __gen_one_example
    infinity_save_video = __save_video
    infinity_transform = __transform
    SelfCorrection = __SelfCorrection
    get_dynamic_resolution_meta = __get_dyn


# -------------------------
# 2) Action head only mode
# -------------------------


# -------------------------
# 3) Config (env defaults)
# -------------------------
DEFAULT_TS_CKPT = str(TSFORMER_ROOT / "adapter_p2p" / "new_stage2_resume70_to100_bs256" / "p2p_epoch_100.pth")
DEFAULT_TS_STATS = str(TSFORMER_ROOT / "adapter_p2p" / "uav-flow_p2p" / "p2p_target_stats.json")

DEFAULT_NUM_FRAMES = int(os.environ.get("INFINITY_NUM_FRAMES", "81"))
DEFAULT_STEP = int(os.environ.get("INFINITY_STEP", "16"))
DEFAULT_FPS = int(os.environ.get("INFINITY_FPS", "16"))
DEFAULT_PN = os.environ.get("INFINITY_PN", "0.40M")
DEFAULT_H_DIV_W = float(os.environ.get("INFINITY_H_DIV_W_TEMPLATE", "0.562"))

DEFAULT_DYNAMIC_SCHEDULE = os.environ.get("INFINITY_DYNAMIC_SCALE_SCHEDULE", "infinity_elegant_clip20frames_v2_allpt")
DEFAULT_MASK_TYPE = os.environ.get("INFINITY_MASK_TYPE", "infinity_elegant_clip20frames_v2_allpt")
DEFAULT_CFG = float(os.environ.get("INFINITY_CFG", "34.0"))
DEFAULT_TAU_IMAGE = float(os.environ.get("INFINITY_TAU_IMAGE", "1.0"))
DEFAULT_TAU_VIDEO = float(os.environ.get("INFINITY_TAU_VIDEO", "0.4"))
DEFAULT_TOP_K = int(os.environ.get("INFINITY_TOP_K", "900"))
DEFAULT_TOP_P = float(os.environ.get("INFINITY_TOP_P", "0.97"))
DEFAULT_GT_LEAK_FIRST = int(os.environ.get("INFINITY_GT_LEAK_FIRST", "14"))

# Default config file location (can be overridden by INFINITY_SERVER_CONFIG).
DEFAULT_SERVER_CONFIG_JSON = str((ROOT / "config.json").resolve())


def _obs_points(pred_num_frames: int, step: int) -> List[int]:
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


@dataclass
class InfinityConfig:
    ckpt: str = ""
    num_frames: int = DEFAULT_NUM_FRAMES
    step: int = DEFAULT_STEP
    fps: int = DEFAULT_FPS
    pn: str = DEFAULT_PN
    h_div_w_template: float = DEFAULT_H_DIV_W
    dynamic_scale_schedule: str = DEFAULT_DYNAMIC_SCHEDULE
    mask_type: str = DEFAULT_MASK_TYPE
    cfg: float = DEFAULT_CFG
    tau_image: float = DEFAULT_TAU_IMAGE
    tau_video: float = DEFAULT_TAU_VIDEO
    top_k: int = DEFAULT_TOP_K
    top_p: float = DEFAULT_TOP_P
    gt_leak_first: int = DEFAULT_GT_LEAK_FIRST

    # closed-loop / rolling-tail knobs (match batch_closed_loop_streaming_infer_routes.py)
    rolling_tail_infer: bool = False
    rolling_infer_mode: str = "stable_full"  # stable_full | tail_window
    tail_window_frames: int = 33
    tail_window_start_step: int = 1
    v2v_history_injection: str = "gt_obs"  # gt_obs | official_leak | hybrid_leak_gtobs
    late_v2v_history_injection: Optional[str] = None
    late_step_start: int = 2
    late_top_k: int = 300
    late_top_p: float = 0.90
    lock_seed_across_steps: bool = False

    def points(self) -> List[int]:
        return _obs_points(pred_num_frames=int(self.num_frames), step=int(self.step))

    def pt_total(self) -> int:
        # pt = (num_frames - 1)//temporal_compress_rate + 1, temporal_compress_rate=4 in this repo
        return (int(self.num_frames) - 1) // 4 + 1


@dataclass
class TSformerConfig:
    ckpt: str = DEFAULT_TS_CKPT
    stats: str = DEFAULT_TS_STATS


@dataclass
class ServerConfig:
    infinity: InfinityConfig = field(default_factory=InfinityConfig)
    tsformer: TSformerConfig = field(default_factory=TSformerConfig)
    infinity_repo_root: Path = field(default_factory=_get_infinity_repo_root)


_SRV_CFG: Optional[ServerConfig] = None


def _load_server_config_from_json(path: str) -> ServerConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    inf = raw.get("infinity", raw) if isinstance(raw, dict) else {}
    ts = raw.get("tsformer", {}) if isinstance(raw, dict) else {}

    inf_cfg = InfinityConfig(
        ckpt=str(inf.get("ckpt") or inf.get("checkpoint") or "").strip(),
        num_frames=int(inf.get("num_frames", DEFAULT_NUM_FRAMES)),
        step=int(inf.get("step", DEFAULT_STEP)),
        fps=int(inf.get("fps", DEFAULT_FPS)),
        pn=str(inf.get("pn", DEFAULT_PN)),
        h_div_w_template=float(inf.get("h_div_w_template", DEFAULT_H_DIV_W)),
        dynamic_scale_schedule=str(inf.get("dynamic_scale_schedule", DEFAULT_DYNAMIC_SCHEDULE)),
        mask_type=str(inf.get("mask_type", DEFAULT_MASK_TYPE)),
        cfg=float(inf.get("cfg", DEFAULT_CFG)),
        tau_image=float(inf.get("tau_image", DEFAULT_TAU_IMAGE)),
        tau_video=float(inf.get("tau_video", DEFAULT_TAU_VIDEO)),
        top_k=int(inf.get("top_k", DEFAULT_TOP_K)),
        top_p=float(inf.get("top_p", DEFAULT_TOP_P)),
        gt_leak_first=int(inf.get("gt_leak_first", DEFAULT_GT_LEAK_FIRST)),
        rolling_tail_infer=bool(inf.get("rolling_tail_infer", False)),
        rolling_infer_mode=str(inf.get("rolling_infer_mode", "stable_full")),
        tail_window_frames=int(inf.get("tail_window_frames", 33)),
        tail_window_start_step=int(inf.get("tail_window_start_step", 1)),
        v2v_history_injection=str(inf.get("v2v_history_injection", "gt_obs")),
        late_v2v_history_injection=(str(inf.get("late_v2v_history_injection")).strip() if inf.get("late_v2v_history_injection") is not None else None),
        late_step_start=int(inf.get("late_step_start", 2)),
        late_top_k=int(inf.get("late_top_k", 300)),
        late_top_p=float(inf.get("late_top_p", 0.90)),
        lock_seed_across_steps=bool(inf.get("lock_seed_across_steps", False)),
    )

    ts_cfg = TSformerConfig(
        ckpt=str(ts.get("ckpt", DEFAULT_TS_CKPT)).strip(),
        stats=str(ts.get("stats", DEFAULT_TS_STATS)).strip(),
    )

    return ServerConfig(infinity=inf_cfg, tsformer=ts_cfg, infinity_repo_root=_get_infinity_repo_root())


def _get_server_config() -> ServerConfig:
    global _SRV_CFG
    if _SRV_CFG is not None:
        return _SRV_CFG

    cfg_path = os.environ.get("INFINITY_SERVER_CONFIG", "").strip()
    if not cfg_path:
        cfg_path = DEFAULT_SERVER_CONFIG_JSON if os.path.exists(DEFAULT_SERVER_CONFIG_JSON) else ""

    if cfg_path:
        cfg = _load_server_config_from_json(cfg_path)
    else:
        cfg = ServerConfig()

    # Allow env vars to override checkpoint paths.
    # We intentionally let env take precedence so offline rollout can swap policies per-iteration
    # without editing config.json (e.g., loop StageA->StageB->StageA).
    env_inf_ckpt = os.environ.get("INFINITY_CKPT", "").strip()
    if env_inf_ckpt:
        cfg.infinity.ckpt = env_inf_ckpt
    env_ts_ckpt = os.environ.get("TS_P2P_CKPT", "").strip()
    if env_ts_ckpt:
        cfg.tsformer.ckpt = env_ts_ckpt
    env_ts_stats = os.environ.get("TS_P2P_STATS", "").strip()
    if env_ts_stats:
        cfg.tsformer.stats = env_ts_stats

    # Allow env vars to override runtime knobs even when config.json provides them.
    # This is important for GRPO experiments where StageA must align with StageB's
    # logprob scoring mode (e.g. teacher-forcing `trace_ce` expects cfg=1, tau=1).
    try:
        v = os.environ.get("INFINITY_CFG", "").strip()
        if v:
            cfg.infinity.cfg = float(v)
    except Exception:
        pass
    try:
        v = os.environ.get("INFINITY_TAU_IMAGE", "").strip()
        if v:
            cfg.infinity.tau_image = float(v)
    except Exception:
        pass
    try:
        v = os.environ.get("INFINITY_TAU_VIDEO", "").strip()
        if v:
            cfg.infinity.tau_video = float(v)
    except Exception:
        pass
    try:
        v = os.environ.get("INFINITY_TOP_K", "").strip()
        if v:
            cfg.infinity.top_k = int(float(v))
    except Exception:
        pass
    try:
        v = os.environ.get("INFINITY_TOP_P", "").strip()
        if v:
            cfg.infinity.top_p = float(v)
    except Exception:
        pass

    _SRV_CFG = cfg
    return cfg


# -------------------------
# 4) Utilities
# -------------------------
_DATA_URL_SPLIT_RE = re.compile(r"^data:image/[^;]+;base64,", flags=re.IGNORECASE)


def _load_image_from_base64(s: str) -> Image.Image:
    if not isinstance(s, str) or not s.strip():
        raise ValueError("empty image string")
    b64 = _DATA_URL_SPLIT_RE.sub("", s.strip())
    raw = base64.b64decode(b64)
    return Image.open(BytesIO(raw)).convert("RGB")


def _sorted_image_paths(images_dir: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    names = [n for n in os.listdir(images_dir) if n.lower().endswith(exts)]
    names.sort()
    return [os.path.join(images_dir, n) for n in names]


def _to_cm_deg(deltas_m_rad: torch.Tensor) -> torch.Tensor:
    """
    deltas: [..., 6] = [dx,dy,dz,droll,dyaw,dpitch] in (m, rad)
    -> (cm, deg)
    """
    out = deltas_m_rad.clone()
    out[..., 0:3] = out[..., 0:3] * 100.0
    out[..., 3:6] = out[..., 3:6] * (180.0 / math.pi)
    return out


def _prompt_with_duration(prompt: str, *, num_frames: int, fps: int, append_tag: bool = True) -> str:
    if not append_tag:
        return prompt
    dur_s = (int(num_frames) - 1) // max(1, int(fps))
    return f"<<<t={dur_s}s>>>{prompt}"


# -------------------------
# 5) Model holders (loaded once)
# -------------------------
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPE = torch.bfloat16 if (_DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16 if _DEVICE == "cuda" else torch.float32

_infinity_args = None
_infinity_session_template: Optional[InfinityStreamingSession] = None
_infinity_self_correction: Optional[SelfCorrection] = None

_ts_model: Optional[torch.nn.Module] = None
_ts_mean: Optional[torch.Tensor] = None
_ts_std: Optional[torch.Tensor] = None

# ActionHead (reference-video TimesFormer) optional mode:
# - input: 4-frame windows (stride=1), aggregated to per-frame deltas
# - output: per-frame 6D deltas, then converted to our API units (cm/deg)
_ah_vit_cls = None  # type: ignore
_ah_model: Optional[torch.nn.Module] = None
_ah_stats: Optional[Dict[str, "np.ndarray"]] = None  # type: ignore[name-defined]
_ah_preprocess = None  # type: ignore
_AH_KITTI_MEAN = [0.34721234, 0.36705238, 0.36066107]
_AH_KITTI_STD = [0.30737526, 0.31515116, 0.32020183]
_AH_TARGET_H = 192
_AH_TARGET_W = 640

DEFAULT_ACTIONHEAD_REPO_ROOT = REPO / "Worldmodel" / "action_decoder" / "actionhead_runtime"


def _get_actionhead_repo_root() -> Path:
    p = os.environ.get("ACTIONHEAD_REPO_ROOT", "").strip()
    if p:
        return Path(p).expanduser().resolve()
    return DEFAULT_ACTIONHEAD_REPO_ROOT


def _import_actionhead_modules(repo_root: Path) -> None:
    """
    Import TimesFormer VisionTransformer for the actionhead reference-video mode.
    NOTE: we intentionally do NOT import any `datasets.*` modules here to avoid
    name collisions with the latent TSformer repo (both have a `datasets` package).
    """
    global _ah_vit_cls, _ah_preprocess
    if _ah_vit_cls is not None and _ah_preprocess is not None:
        return
    if np is None:
        raise RuntimeError("numpy is required for actionhead mode")
    if not repo_root.exists():
        raise FileNotFoundError(f"ActionHead repo not found: {repo_root}")
    if str(repo_root) not in sys.path:
        # Append (do not insert at 0) to minimize import shadowing.
        sys.path.append(str(repo_root))
    try:
        from torchvision import transforms as T  # type: ignore
    except Exception as e:
        raise RuntimeError(f"torchvision is required for actionhead mode: {e}")
    from timesformer.models.vit import VisionTransformer  # type: ignore

    _ah_vit_cls = VisionTransformer
    # Match predict_reference_videos_batch copy.py preprocessing:
    # ToPILImage -> Resize((H,W)) -> ToTensor -> Normalize (NO crop)
    _ah_preprocess = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((int(_AH_TARGET_H), int(_AH_TARGET_W))),
            T.ToTensor(),
            T.Normalize(mean=_AH_KITTI_MEAN, std=_AH_KITTI_STD),
        ]
    )


def _default_action_head_mode() -> str:
    """
    Read default action-head mode from env.
    This is used during startup model initialization to decide whether TSformer(P2P)
    must be loaded eagerly.
    """
    return os.environ.get("ACTION_HEAD_MODE", "").strip().lower()


def _use_actionhead_ref_mode_by_default() -> bool:
    mode = _default_action_head_mode()
    return mode in ("actionhead_ref_vit", "actionhead_ref", "actionhead_vit", "ref_vit", "actionhead")


def _load_actionhead_stats(run_config_path: str) -> Dict[str, "np.ndarray"]:  # type: ignore[name-defined]
    assert np is not None
    with open(run_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("use_raw_for_labels") is not True:
        raise ValueError("actionhead_ref_vit requires previous-body SE(3) labels (use_raw_for_labels=true)")
    if int(cfg.get("window_size", 4)) != 4:
        raise ValueError("actionhead_ref_vit requires a four-frame action-head checkpoint")
    stats = cfg.get("label_stats") or {}
    need = ("mean_angles", "std_angles", "mean_t", "std_t")
    for k in need:
        if k not in stats:
            raise ValueError(f"run_config.json missing label_stats.{k}")
    out: Dict[str, "np.ndarray"] = {}
    out["mean_angles"] = np.asarray(stats["mean_angles"], dtype=np.float32)
    out["std_angles"] = np.asarray(stats["std_angles"], dtype=np.float32)
    out["mean_t"] = np.asarray(stats["mean_t"], dtype=np.float32)
    out["std_t"] = np.asarray(stats["std_t"], dtype=np.float32)
    divisor = float(cfg.get("translation_divisor", 100.0))
    if not math.isfinite(divisor) or divisor <= 0:
        raise ValueError("action-head translation_divisor must be finite and positive")
    out["translation_divisor"] = divisor
    return out


def _init_actionhead_model(*, ckpt_path: str, run_config_path: str) -> None:
    global _ah_model, _ah_stats
    if _ah_model is not None and _ah_stats is not None:
        return
    repo_root = _get_actionhead_repo_root()
    _import_actionhead_modules(repo_root)
    assert _ah_vit_cls is not None

    device = torch.device("cuda" if _DEVICE == "cuda" else "cpu")
    model = _ah_vit_cls(  # type: ignore[misc]
        img_size=(int(_AH_TARGET_H), int(_AH_TARGET_W)),
        num_classes=18,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=lambda *a, **kw: torch.nn.LayerNorm(*a, eps=1e-6, **kw),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        num_frames=4,
        attention_type="divided_space_time",
    )
    ckpt = torch.load(os.path.abspath(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Action-head checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.to(device).eval()

    _ah_model = model
    _ah_stats = _load_actionhead_stats(os.path.abspath(run_config_path))

# Concurrency: one GPU pipeline lock (single-process test stage)
try:
    import asyncio

    _LOCK: "asyncio.Lock" = asyncio.Lock()
except Exception:
    _LOCK = None  # type: ignore


def _load_tsformer_p2p(
    *,
    ckpt_path: str,
    stats_path: str,
    device: str,
) -> Tuple[torch.nn.Module, Optional[torch.Tensor], Optional[torch.Tensor]]:
    try:
        from pretrain_latent_p2p import build_p2p_model  # type: ignore
    except Exception as e:
        raise RuntimeError(f"legacy TSformer(P2P) import failed (install its deps like fvcore): {e}")
    args = argparse.Namespace(window_size=2, hidden_dim=96, num_layers=2, device=device, checkpoint=ckpt_path, stats_path=stats_path)
    model = build_p2p_model(args)
    model.to(device).eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    new_sd: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("module."):
            new_sd[k[7:]] = v
        else:
            new_sd[k] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing or unexpected:
        # Keep strict=False: this repo has multiple variants; mismatches are common and usually benign for adapter layers.
        print(f"[TSformer] load_state_dict strict=False, missing={len(missing)} unexpected={len(unexpected)}")

    mean_t = std_t = None
    if stats_path and os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)
        std = torch.tensor(stats["std"], dtype=torch.float32, device=device)
        mean_t, std_t = mean, std
        print(f"[TSformer] loaded stats: {stats_path}")
    else:
        print("[TSformer] stats not found; will output normalized deltas")
    return model, mean_t, std_t


def _init_models(
    *,
    cfg: ServerConfig,
) -> None:
    global _infinity_args, _infinity_session_template, _infinity_self_correction, _ts_model, _ts_mean, _ts_std

    skip_p2p = _use_actionhead_ref_mode_by_default()
    if _infinity_session_template is not None and (_ts_model is not None or skip_p2p):
        return

    print("[Service] initializing models...")
    print(f"[Service] device={_DEVICE} dtype={_DTYPE}")

    # Select InfinityStar repo and import its modules.
    _import_infinity_modules(Path(cfg.infinity_repo_root))

    def _resolve_path(p: str) -> str:
        if not p:
            return p
        if os.path.isabs(p):
            return p
        return str((ROOT / p).resolve())

    cfg.infinity.ckpt = _resolve_path(cfg.infinity.ckpt)
    cfg.tsformer.ckpt = _resolve_path(cfg.tsformer.ckpt)
    cfg.tsformer.stats = _resolve_path(cfg.tsformer.stats)

    if not cfg.infinity.ckpt:
        raise ValueError("InfinityStar checkpoint path is empty (set in config.json or INFINITY_CKPT env)")

    # InfinityStar: build args + load models once
    a = _make_infinity_args(  # type: ignore[misc]
        ckpt=os.path.abspath(cfg.infinity.ckpt),
        pn=str(cfg.infinity.pn),
        fps=int(cfg.infinity.fps),
        num_frames=int(cfg.infinity.num_frames),
        seed=0,
        dynamic_scale_schedule=str(cfg.infinity.dynamic_scale_schedule),
        mask_type=str(cfg.infinity.mask_type),
        cfg=float(cfg.infinity.cfg),
        tau_image=float(cfg.infinity.tau_image),
        tau_video=float(cfg.infinity.tau_video),
    )

    # Align with StageB training defaults (critical for flex-attn packing correctness).
    # Infinity forward pads (visual+text) sequence to `pad_to_multiplier` when train_with_var_seq_len=1.
    # Our trace_ce old_logprob path relies on the same padding regime to match flex-attn mask construction.
    try:
        a.train_with_var_seq_len = 1
        a.pad_to_multiplier = int(getattr(a, "pad_to_multiplier", 128) or 128) if hasattr(a, "pad_to_multiplier") else 128
    except Exception:
        pass
    try:
        a.train_max_token_len = int(getattr(a, "train_max_token_len", 20480) or 20480)
        a.allow_less_one_elem_in_seq = int(getattr(a, "allow_less_one_elem_in_seq", 1) or 1)
    except Exception:
        pass
    try:
        a.use_flex_attn = True
    except Exception:
        pass

    # CRITICAL: `infinity_elegant` schedules rely on `args.frames_inner_clip` to compute
    # scale_pack_info.frame_ss/frame_ee. If it doesn't match the schedule family
    # (e.g. clip4frames vs clip20frames), `freqs_frames[:, frame_ss:frame_ee]` can be empty,
    # causing get_visual_rope_embeds() to fail with size-0 tensors.
    try:
        sched_name = str(cfg.infinity.dynamic_scale_schedule)
        if "clip4frames" in sched_name:
            a.frames_inner_clip = 4
        elif "clip20frames" in sched_name:
            a.frames_inner_clip = 20
    except Exception:
        pass

    text_tokenizer, text_encoder = load_tokenizer(t5_path=a.text_encoder_ckpt)  # type: ignore[misc]
    vae = load_visual_tokenizer(a).float().to(_DEVICE)  # type: ignore[misc]
    infinity = load_transformer(vae, a).to(_DEVICE)  # type: ignore[misc]
    infinity.eval().requires_grad_(False)
    self_correction = SelfCorrection(vae, a)  # type: ignore[misc]

    session = InfinityStreamingSession(  # type: ignore[misc]
        args=a,
        infinity_model=infinity,
        vae=vae,
        text_tokenizer=text_tokenizer,
        text_encoder=text_encoder,
        h_div_w_template=float(cfg.infinity.h_div_w_template),
    )

    _infinity_args = a
    _infinity_session_template = session
    _infinity_self_correction = self_correction

    # TSformer(P2P): load once unless default mode is actionhead_ref_vit.
    # In actionhead_ref_vit mode we predict actions from decoded video windows,
    # so p2p weights are not required and can be incompatible.
    if skip_p2p:
        _ts_model, _ts_mean, _ts_std = None, None, None
        print("[Service] ACTION_HEAD_MODE=actionhead_ref_vit*, skip loading TSformer(P2P).")
    else:
        ts_model, mean_t, std_t = _load_tsformer_p2p(
            ckpt_path=os.path.abspath(cfg.tsformer.ckpt),
            stats_path=os.path.abspath(cfg.tsformer.stats) if cfg.tsformer.stats else "",
            device=_DEVICE,
        )
        _ts_model, _ts_mean, _ts_std = ts_model, mean_t, std_t

    print("[Service] model initialization done.")


# -------------------------
# 6) Per-trajectory state
# -------------------------
@dataclass
class TrajectoryState:
    session_id: str
    prompt_raw: str
    negative_prompt: str = ""
    created_at: float = field(default_factory=lambda: time.time())

    # received frames already transformed to [-1,1] at (tgt_h,tgt_w), each is [3,H,W] CPU tensor
    frames_cpu: List[torch.Tensor] = field(default_factory=list)

    # per-trajectory Infinity session wrapper (holds text tuple); caches live in model, so we keep exported copies here.
    stream: Optional[InfinityStreamingSession] = None
    kv_cache: Optional[Any] = None

    # first-frame i2v alignment helpers (optional)
    gt_ls_Bl_first: Optional[Any] = None

    # closed-loop helpers
    dyn_res: Optional[Any] = None
    h_sel: Optional[str] = None
    firstframe_prepared: bool = False

    # TSformer latent memory (carry one latent across segments)
    last_latent_1: Optional[torch.Tensor] = None  # [1,16,1,H,W] on CPU (float16/float32)
    latent_dir: Optional[str] = None  # on-disk cache directory: "<session_id>_infinity_latnet"

    # target spatial size for transform (determined once)
    tgt_h: Optional[int] = None
    tgt_w: Optional[int] = None
    h_div_w_template: float = float(DEFAULT_H_DIV_W)

    # emission bookkeeping
    last_emitted_segment: int = -1
    # request mode hint (prefix_mode: full prefix [1..K] per call)
    last_req_prefix_mode: bool = False

    def num_frames(self) -> int:
        return len(self.frames_cpu)


_TRAJ: Dict[str, TrajectoryState] = {}
_SESSION_ALIAS: Dict[str, str] = {}


def _make_run_session_id(external_session_id: str) -> str:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    # Add ns suffix to avoid collisions within same second.
    suffix = str(time.time_ns() % 1_000_000_000).rjust(9, "0")
    return f"{external_session_id}__{ts}_{suffix}"


def _get_or_create_traj(session_id: str, prompt: str, negative_prompt: str) -> TrajectoryState:
    cfg = _get_server_config()
    if session_id in _TRAJ:
        st = _TRAJ[session_id]
        # allow client to omit prompt on subsequent calls
        if prompt and prompt.strip():
            st.prompt_raw = prompt.strip()
        if negative_prompt is not None:
            st.negative_prompt = (negative_prompt or "").strip()
        return st

    st = TrajectoryState(
        session_id=session_id,
        prompt_raw=prompt.strip(),
        negative_prompt=(negative_prompt or "").strip(),
        h_div_w_template=float(cfg.infinity.h_div_w_template),
    )
    # latent cache folder (optional but enabled by default)
    root = os.environ.get("INFINITY_LATENT_CACHE_ROOT", "").strip()
    if not root:
        root = str((ROOT / "cache").resolve())
    try:
        os.makedirs(root, exist_ok=True)
        st.latent_dir = os.path.join(root, f"{session_id}_infinity_latnet")
        os.makedirs(st.latent_dir, exist_ok=True)
        # best-effort resume: load last_latent.pt if exists
        last_path = os.path.join(st.latent_dir, "last_latent.pt")
        if os.path.exists(last_path):
            try:
                t = torch.load(last_path, map_location="cpu")
                if isinstance(t, torch.Tensor) and t.ndim == 5 and t.shape[0] == 1 and t.shape[1] == 16 and t.shape[2] == 1:
                    st.last_latent_1 = t.contiguous()
            except Exception:
                pass
    except Exception:
        st.latent_dir = None
        st.last_latent_1 = None
    _TRAJ[session_id] = st
    return st


def _ensure_traj_infinity_session(st: TrajectoryState) -> None:
    assert _infinity_session_template is not None
    assert _infinity_args is not None
    cfg = _get_server_config()

    if st.stream is not None:
        return

    # Create a lightweight per-trajectory wrapper. It shares model/vae/text components with the template.
    tpl = _infinity_session_template
    st.stream = InfinityStreamingSession(  # type: ignore[misc]
        args=_infinity_args,
        infinity_model=tpl.infinity,
        vae=tpl.vae,
        text_tokenizer=tpl.text_tokenizer,
        text_encoder=tpl.text_encoder,
        h_div_w_template=float(st.h_div_w_template),
    )

    prompt_infer = _prompt_with_duration(
        st.prompt_raw,
        num_frames=int(cfg.infinity.num_frames),
        fps=int(cfg.infinity.fps),
        append_tag=bool(getattr(_infinity_args, "append_duration2caption", 0)),
    )
    st.stream.reset(prompt_infer, negative_prompt=st.negative_prompt, cfg_scale=float(cfg.infinity.cfg))
    st.kv_cache = st.stream.infinity.export_kv_cache()


def _import_kv_cache_for_traj(st: TrajectoryState) -> None:
    assert st.stream is not None
    if st.kv_cache is None:
        return
    # Clear any previous session caches by resetting cache storage, then import.
    for blk in st.stream.infinity.unregistered_blocks:
        blk.attn.kv_caching(True, reset=True)
    st.stream.infinity.import_kv_cache(st.kv_cache, overwrite=True)


def _prepare_firstframe_condition_if_needed(st: TrajectoryState) -> None:
    """
    Match batch_closed_loop_streaming_infer_routes.py:
    - Step0 uses first-frame gt_leak injection, and we intentionally do NOT write obs1 into gt_obs cache.
    - We precompute gt_ls_Bl_first + dyn_res/h_sel once per trajectory.
    """
    cfg = _get_server_config()
    assert st.stream is not None
    assert _infinity_args is not None
    assert _infinity_self_correction is not None
    assert get_dynamic_resolution_meta is not None

    if st.firstframe_prepared:
        return
    if st.num_frames() <= 0:
        raise ValueError("no frames received")

    dyn_res, _ = get_dynamic_resolution_meta(_infinity_args.dynamic_scale_schedule, _infinity_args.video_frames)  # type: ignore[misc]
    st.dyn_res = dyn_res

    # Pick nearest h/w template key for dynamic-resolution tables.
    try:
        import numpy as np  # local import; numpy is already used by InfinityStar tools

        h_keys = list(dyn_res.keys())
        h_vals = np.array([float(k) for k in h_keys], dtype=np.float64)
        st.h_sel = h_keys[int(np.argmin(np.abs(h_vals - float(st.h_div_w_template))))]
    except Exception:
        # Fallback: just take the first key.
        st.h_sel = list(dyn_res.keys())[0]

    # Encode first frame into gt_ls_Bl_first for strict i2v alignment.
    obs1 = st.frames_cpu[0].unsqueeze(0).to(_DEVICE, non_blocking=True)  # [1,3,H,W] in [-1,1]
    with torch.no_grad():
        _, _, gt_ls_Bl_first, _, _, _ = st.stream.video_encode(
            vae=st.stream.vae,
            inp_B3HW=obs1,
            vae_features=None,
            self_correction=_infinity_self_correction,
            args=_infinity_args,
            infer_mode=True,
            dynamic_resolution_h_w=dyn_res,
        )
    st.gt_ls_Bl_first = gt_ls_Bl_first
    st.firstframe_prepared = True


def _update_gt_obs_cache_to(st: TrajectoryState, n_frames: int) -> None:
    """Overwrite gt_obs cache with prefix [1..n_frames] (B=1)."""
    assert st.stream is not None
    if n_frames <= 0:
        return
    obs = torch.stack(st.frames_cpu[:n_frames], dim=0)  # [T,3,H,W]
    obs_bcthw = obs.permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # [1,3,T,H,W]
    st.stream.compute_kv_cache_gt(obs_bcthw.to(_DEVICE, non_blocking=True))


def _infer_summed_codes_for_step(
    st: TrajectoryState,
    *,
    step_i: int,
    obs_len: int,
    infer_num_frames: int,
    seed: int,
    top_k: int,
    top_p: float,
    injection: str,
    need_pred_video: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], float, Optional[str]]:
    """
    Run InfinityStar inference for one closed-loop step and return summed_codes [1,16,pt,H,W].
    This mirrors the control flow in batch_closed_loop_streaming_infer_routes.py, but skips VAE decode.
    """
    cfg = _get_server_config()
    assert st.stream is not None
    assert _infinity_args is not None

    # Ensure session uses correct aspect template
    st.stream.h_div_w_template = float(st.h_div_w_template)
    st.stream.correction_clear_pred()

    gt_leak = -1
    gt_ls_Bl = None

    if int(step_i) == 0:
        _prepare_firstframe_condition_if_needed(st)
        gt_leak = int(cfg.infinity.gt_leak_first)
        gt_ls_Bl = st.gt_ls_Bl_first
    else:
        inj = str(injection)
        if inj in ("official_leak", "hybrid_leak_gtobs"):
            if not st.dyn_res or not st.h_sel:
                _prepare_firstframe_condition_if_needed(st)
            assert st.dyn_res is not None and st.h_sel is not None
            assert _infinity_self_correction is not None
            # Encode continuous prefix [1..obs_len] and inject with auto leak depth.
            prefix = torch.stack(st.frames_cpu[:obs_len], dim=0)  # [T,3,H,W]
            prefix_obs = prefix.permute(1, 0, 2, 3).unsqueeze(0).contiguous().to(_DEVICE, non_blocking=True)  # [1,3,T,H,W]
            with torch.no_grad():
                _, _, gt_ls_Bl_prefix, _, _, _ = st.stream.video_encode(
                    vae=st.stream.vae,
                    inp_B3HW=prefix_obs,
                    vae_features=None,
                    self_correction=_infinity_self_correction,
                    args=_infinity_args,
                    infer_mode=True,
                    dynamic_resolution_h_w=st.dyn_res,
                )
            if inj == "hybrid_leak_gtobs":
                # Hybrid: also write gt_obs cache (helps stabilize late segments).
                st.stream.compute_kv_cache_gt(prefix_obs)

            pt_obs = (int(obs_len) - 1) // int(getattr(_infinity_args, "temporal_compress_rate", 4)) + 1
            pt2sched = st.dyn_res[st.h_sel][_infinity_args.pn]["pt2scale_schedule"]
            leak_auto = len(pt2sched[int(pt_obs)])
            gt_leak = int(leak_auto)
            gt_ls_Bl = gt_ls_Bl_prefix
        else:
            # gt_obs mode: write prefix into cache and infer without leak.
            _update_gt_obs_cache_to(st, obs_len)

    sched = st.stream.build_schedule_for_num_frames(int(infer_num_frames))
    tau_list = [float(cfg.infinity.tau_image)] * int(sched.tower_split_index) + [float(cfg.infinity.tau_video)] * (
        len(sched.scale_schedule) - int(sched.tower_split_index)
    )

    # Use the legacy standalone inference wrapper style via gen_one_example(),
    # which internally normalizes cfg/tau lists and handles prompt encoding per-call.
    try:
        trace_sample_logprob = 0.0
        trace_path: Optional[str] = None
        if infinity_gen_one_example is None:
            raise RuntimeError("InfinityStar gen_one_example not imported")
        assert _infinity_args is not None

        prompt_infer = _prompt_with_duration(
            st.prompt_raw,
            num_frames=int(cfg.infinity.num_frames),
            fps=int(cfg.infinity.fps),
            append_tag=bool(getattr(_infinity_args, "append_duration2caption", 0)),
        )

        with torch.no_grad():
            if _DEVICE == "cuda":
                with torch.cuda.amp.autocast(enabled=True, dtype=next(iter(st.stream.infinity.parameters())).dtype):
                    out_gen = infinity_gen_one_example(  # type: ignore[misc]
                        st.stream.infinity,
                        st.stream.vae,
                        st.stream.text_tokenizer,
                        st.stream.text_encoder,
                        prompt_infer,
                        negative_prompt=str(st.negative_prompt or ""),
                        g_seed=int(seed),
                        gt_leak=int(gt_leak),
                        gt_ls_Bl=gt_ls_Bl,
                        cfg_list=float(cfg.infinity.cfg),
                        tau_list=tau_list,
                        scale_schedule=sched.scale_schedule,
                        top_k=int(top_k),
                        top_p=float(top_p),
                        cfg_insertion_layer=[0],
                        vae_type=int(getattr(_infinity_args, "vae_type", 64)),
                        sampling_per_bits=1,
                        enable_positive_prompt=0,
                        low_vram_mode=True,
                        args=_infinity_args,
                        get_visual_rope_embeds=st.stream.get_visual_rope_embeds,
                        context_info=sched.context_info,
                        noise_list=None,
                        return_summed_code_only=True,
                        return_trace=True,
                        return_logprob_by_clip=True,
                    )
            else:
                out_gen = infinity_gen_one_example(  # type: ignore[misc]
                    st.stream.infinity,
                    st.stream.vae,
                    st.stream.text_tokenizer,
                    st.stream.text_encoder,
                    prompt_infer,
                    negative_prompt=str(st.negative_prompt or ""),
                    g_seed=int(seed),
                    gt_leak=int(gt_leak),
                    gt_ls_Bl=gt_ls_Bl,
                    cfg_list=float(cfg.infinity.cfg),
                    tau_list=tau_list,
                    scale_schedule=sched.scale_schedule,
                    top_k=int(top_k),
                    top_p=float(top_p),
                    cfg_insertion_layer=[0],
                    vae_type=int(getattr(_infinity_args, "vae_type", 64)),
                    sampling_per_bits=1,
                    enable_positive_prompt=0,
                    low_vram_mode=True,
                    args=_infinity_args,
                    get_visual_rope_embeds=st.stream.get_visual_rope_embeds,
                    context_info=sched.context_info,
                    noise_list=None,
                    return_summed_code_only=True,
                    return_trace=True,
                    return_logprob_by_clip=True,
                )

        if not isinstance(out_gen, dict):
            raise RuntimeError("GRPO sampling requires a complete trace dict")
        summed_codes = out_gen.get("summed_codes")
        clipid_target = int(step_i) + 1
        by_clip = out_gen.get("sample_logprob_by_clip")
        token_counts = out_gen.get("token_count_by_clip")
        if not isinstance(by_clip, (list, tuple)) or len(by_clip) <= clipid_target:
            raise RuntimeError(f"missing policy logprob for target clip {clipid_target}")
        if not isinstance(token_counts, (list, tuple)) or len(token_counts) <= clipid_target:
            raise RuntimeError(f"missing policy token count for target clip {clipid_target}")
        if int(token_counts[clipid_target]) <= 0:
            raise RuntimeError("target clip contains no policy-sampled tokens")
        value = by_clip[clipid_target]
        if out_gen.get("logprob_accumulation_dtype") != "float64" or not torch.is_tensor(value) or value.dtype != torch.float64:
            raise RuntimeError("exact clip likelihoods require FP64 accumulation and a matching sampler trace")
        trace_sample_logprob = float(value.detach().cpu().item())
        if not math.isfinite(trace_sample_logprob):
            raise RuntimeError("non-finite sampling logprob")
        mode = os.environ.get("INFINITY_STAGEA_OLD_LOGPROB_MODE", "sampling").strip().lower()
        if mode != "sampling":
            raise ValueError("strict GRPO requires INFINITY_STAGEA_OLD_LOGPROB_MODE=sampling; approximate trace_ce is unsupported")
        text_condition = out_gen.get("text_cond_tuple")
        if not isinstance(text_condition, (list, tuple)) or len(text_condition) != 4:
            raise RuntimeError("sampler must return its exact encoded text_cond_tuple")
        if not st.latent_dir:
            raise RuntimeError("GRPO requires a writable trace directory")
        os.makedirs(st.latent_dir, exist_ok=True)
        trace_path = os.path.join(st.latent_dir, f"seg{int(step_i):02d}_trace.pt")
        def _cpu_tree(value):
            if torch.is_tensor(value):
                return value.detach().to("cpu")
            if isinstance(value, tuple):
                return tuple(_cpu_tree(v) for v in value)
            if isinstance(value, list):
                return [_cpu_tree(v) for v in value]
            if isinstance(value, dict):
                return {k: _cpu_tree(v) for k, v in value.items()}
            return value
        behavior_path = os.path.realpath(cfg.infinity.ckpt)
        behavior_stat = os.stat(behavior_path)
        trace = {
            "schema_version": 3,
            "logprob_accumulation_dtype": "float64",
            "old_logprob_mode": "sampling",
            "behavior_checkpoint": behavior_path,
            "behavior_checkpoint_size": int(behavior_stat.st_size),
            "behavior_checkpoint_mtime_ns": int(behavior_stat.st_mtime_ns),
            "block_parameter_dtype": str(next(st.stream.infinity.unregistered_blocks[0].parameters()).dtype),
            "autocast_dtype": "torch.bfloat16",
            "amp_dtype": "torch.bfloat16",
            "prompt_infer": prompt_infer,
            "negative_prompt": str(st.negative_prompt or ""),
            "text_cond_tuple": text_condition,
            "negative_text_cond_tuple": out_gen.get("negative_text_cond_tuple"),
            "segment_index": int(step_i),
            "obs_len": int(obs_len),
            "infer_num_frames": int(infer_num_frames),
            "injection": str(injection),
            "infinity_cfg": float(cfg.infinity.cfg),
            "tau_list": [float(x) for x in tau_list],
            "top_k": int(top_k), "top_p": float(top_p),
            "pn": str(cfg.infinity.pn),
            "h_div_w_template": float(st.h_div_w_template),
            "dynamic_scale_schedule": str(cfg.infinity.dynamic_scale_schedule),
            "mask_type": str(cfg.infinity.mask_type),
            "scale_schedule": sched.scale_schedule,
            "context_info": sched.context_info,
            "gt_leak": int(gt_leak), "gt_ls_Bl": gt_ls_Bl,
            "clipid_target": clipid_target,
            "step_clipids": out_gen.get("step_clipids"),
            "step_is_sampled": out_gen.get("step_is_sampled"),
            "sample_logprob": trace_sample_logprob,
            "sample_logprob_sampling": trace_sample_logprob,
            "sample_logprob_by_clip": by_clip,
            "sample_token_count_by_clip": token_counts,
            "idx_trace": out_gen.get("idx_trace"),
            "chosen_token_logprobs": out_gen.get("chosen_token_logprobs"),
            "image_scale_repetition": str(_infinity_args.image_scale_repetition),
            "video_scale_repetition": str(_infinity_args.video_scale_repetition),
        }
        for key in ("max_repeat_times", "noise_input", "semantic_scales", "use_cfg", "use_apg", "apg_norm_threshold", "frames_inner_clip", "rope2d_normalized_by_hw", "rope2d_each_sa_layer", "rope_type", "semantic_scale_dim", "detail_scale_dim", "detail_scale_min_tokens", "use_two_stage_lfq"):
            if hasattr(_infinity_args, key):
                trace[key] = getattr(_infinity_args, key)
        torch.save(_cpu_tree(trace), trace_path)
        if not isinstance(summed_codes, torch.Tensor):
            raise RuntimeError(f"Unexpected Infinity output type: {type(out_gen)}")

        pred_vid: Optional[torch.Tensor] = None
        want_decode = bool(st.latent_dir) or bool(need_pred_video)
        if want_decode:
            try:
                with torch.no_grad():
                    pred_vid = st.stream.infinity.summed_codes2images(st.stream.vae, summed_codes)  # [1,T,H,W,3], uint8(BGR)
                if st.latent_dir:
                    total_num_frames = int(cfg.infinity.num_frames)
                    _save_pred_video(st, f"seg{int(step_i):02d}_pred_full_{int(total_num_frames):03d}f.mp4", pred_vid)
            except Exception as e:
                pred_vid = None
                print(f"[pred->video] decode/save skipped: {e}")
    except Exception:
        # Print rich debug info to server logs (FastAPI wraps exceptions into HTTP 500 detail).
        print("[InfinityStar] infer_chunk failed. Dumping debug info...")
        print(traceback.format_exc())
        try:
            blk0 = st.stream.infinity.unregistered_blocks[0]
            ck = getattr(blk0.attn, "cached_k", {})
            cv = getattr(blk0.attn, "cached_v", {})
            keys = list(ck.keys())
            print(f"[InfinityStar] cached_k keys (first block): {keys}")
            for k in keys[:10]:
                vk = ck.get(k, None)
                vv = cv.get(k, None)
                sk = tuple(vk.shape) if isinstance(vk, torch.Tensor) else type(vk).__name__
                sv = tuple(vv.shape) if isinstance(vv, torch.Tensor) else type(vv).__name__
                print(f"  - key={k!r} k={sk} v={sv}")
        except Exception:
            print("[InfinityStar] (debug dump failed)")
        raise

    st.stream.correction_clear_pred()
    return summed_codes, pred_vid, float(trace_sample_logprob), trace_path


def _save_latent_tensor(st: TrajectoryState, name: str, t: torch.Tensor) -> None:
    if not st.latent_dir:
        return
    try:
        p = os.path.join(st.latent_dir, name)
        # store float16 CPU to reduce disk
        torch.save(t.detach().to("cpu", dtype=torch.float16).contiguous(), p)
    except Exception:
        return


def _save_latent_video_clip(
    st: TrajectoryState,
    name: str,
    latents_B16THW: torch.Tensor,
    *,
    drop_first_frame: bool,
) -> None:
    """
    Decode latent clip and save mp4 under the same latent directory.
    - seg0: 5 latents -> 17 frames (drop_first_frame=False)
    - seg>0: decode latent5 then drop first boundary frame -> 16 new frames
    """
    if not st.latent_dir or st.stream is None or infinity_save_video is None:
        return
    try:
        model_dtype = next(iter(st.stream.infinity.parameters())).dtype if _DEVICE == "cuda" else torch.float32
        z = latents_B16THW.to(_DEVICE, dtype=model_dtype, non_blocking=(_DEVICE == "cuda"))
        with torch.no_grad():
            frames = st.stream.infinity.summed_codes2images(st.stream.vae, z)  # [1,T,H,W,3], uint8
        clip = frames[0] if isinstance(frames, torch.Tensor) else frames[0]
        if drop_first_frame and int(clip.shape[0]) > 1:
            clip = clip[1:]
        clip_np = clip.detach().cpu().numpy() if isinstance(clip, torch.Tensor) else clip
        if int(clip_np.shape[0]) <= 0:
            return
        # Infinity.summed_codes2images returns uint8 BGR (it flips channel dim).
        # tools.run_infinity.save_video expects BGR and internally flips to RGB for writing.
        out_path = os.path.join(st.latent_dir, name)
        cfg = _get_server_config()
        infinity_save_video(clip_np, fps=int(cfg.infinity.fps), save_filepath=out_path, force_all_keyframes=True)
    except Exception as e:
        print(f"[latent->video] skip {name}: {e}")


def _save_pred_video(
    st: TrajectoryState,
    name: str,
    pred_video_BTHWC: Any,
) -> None:
    """Save predicted video (decoded by Infinity) under latent_dir. File name must be unique to avoid overwrite."""
    if not st.latent_dir or infinity_save_video is None:
        return
    try:
        vid = pred_video_BTHWC
        if isinstance(vid, torch.Tensor):
            vid = vid.detach().cpu().numpy()
        # Expect [B,T,H,W,3] or [T,H,W,3]
        if getattr(vid, "ndim", 0) == 5:
            vid = vid[0]
        if getattr(vid, "ndim", 0) != 4 or int(vid.shape[-1]) != 3:
            return
        # Infinity.summed_codes2images returns BGR; tools.run_infinity.save_video expects BGR.
        out_path = os.path.join(st.latent_dir, name)
        cfg = _get_server_config()
        infinity_save_video(vid, fps=int(cfg.infinity.fps), save_filepath=out_path, force_all_keyframes=True)
    except Exception as e:
        print(f"[pred->video] skip {name}: {e}")


def _slice_abs_latents_from_summed_codes(
    summed_codes: torch.Tensor,
    *,
    abs_lat_start: int,
    abs_lat_end: int,
    infer_num_frames: int,
    total_num_frames: int,
) -> torch.Tensor:
    """
    summed_codes: [1,16,pt_local,H,W] for either full-horizon (infer_num_frames==total_num_frames)
                or tail-window (infer_num_frames < total_num_frames, ending-aligned).
    abs_lat_start/end: 1-indexed absolute latent indices in the full video timeline.
    Return: [1,16,T,H,W] where T == abs_lat_end-abs_lat_start+1 (if within window).
    """
    if abs_lat_end < abs_lat_start:
        raise ValueError(f"bad abs_lat range: {abs_lat_start}..{abs_lat_end}")
    t_local = int(summed_codes.shape[2])

    local_start = int(abs_lat_start)
    local_end = int(abs_lat_end)
    if int(infer_num_frames) != int(total_num_frames):
        window_start_abs = int(total_num_frames) - int(infer_num_frames) + 1  # 1-indexed absolute frame id
        abs_lat_start_window = (int(window_start_abs) - 1) // 4 + 1
        local_start = int(abs_lat_start) - int(abs_lat_start_window) + 1
        local_end = int(abs_lat_end) - int(abs_lat_start_window) + 1

    s0 = max(1, int(local_start))
    e0 = min(int(local_end), int(t_local))
    if e0 < s0:
        raise ValueError(f"latent slice out of range: abs [{abs_lat_start}..{abs_lat_end}] -> local [{local_start}..{local_end}] vs t_local={t_local}")
    out = summed_codes[:, :, (s0 - 1) : e0].contiguous()
    # If the requested slice is partially outside the window, treat as error for now.
    if int(out.shape[2]) != int(abs_lat_end - abs_lat_start + 1):
        raise ValueError(
            f"latent slice length mismatch: need={abs_lat_end-abs_lat_start+1} got={out.shape[2]} "
            f"(abs [{abs_lat_start}..{abs_lat_end}] -> local [{local_start}..{local_end}], infer_num_frames={infer_num_frames})"
        )
    return out


def _slice_abs_frames_from_pred_video_bgr(
    pred_video_BTHWC: object,
    *,
    abs_frame_start: int,
    abs_frame_end: int,
    infer_num_frames: int,
    total_num_frames: int,
) -> "np.ndarray":  # type: ignore[name-defined]
    """
    pred_video_BTHWC: [1,T,H,W,3] or [T,H,W,3] uint8(BGR), where T==infer_num_frames (full or tail-window).
    abs_frame_start/end: 1-indexed absolute pixel frame indices in the full video timeline.
    Returns: [Tslice,H,W,3] uint8(BGR)
    """
    if np is None:
        raise RuntimeError("numpy is required for actionhead mode")
    if abs_frame_end < abs_frame_start:
        raise ValueError(f"bad abs_frame range: {abs_frame_start}..{abs_frame_end}")

    vid = pred_video_BTHWC
    if isinstance(vid, torch.Tensor):
        vid = vid.detach().cpu().numpy()
    if getattr(vid, "ndim", 0) == 5:
        vid = vid[0]
    if getattr(vid, "ndim", 0) != 4 or int(vid.shape[-1]) != 3:
        raise ValueError(f"bad pred_video shape: {getattr(vid,'shape',None)}")
    t_local = int(vid.shape[0])

    if int(infer_num_frames) == int(total_num_frames):
        local_start = int(abs_frame_start) - 1
        local_end = int(abs_frame_end) - 1
    else:
        window_start_abs = int(total_num_frames) - int(infer_num_frames) + 1  # 1-indexed abs frame id
        local_start = int(abs_frame_start) - int(window_start_abs)
        local_end = int(abs_frame_end) - int(window_start_abs)

    s0 = max(0, int(local_start))
    e0 = min(int(local_end), int(t_local) - 1)
    if e0 < s0:
        raise ValueError(
            f"frame slice out of range: abs [{abs_frame_start}..{abs_frame_end}] -> local [{local_start}..{local_end}] vs t_local={t_local} "
            f"(infer_num_frames={infer_num_frames}, total_num_frames={total_num_frames})"
        )
    out = vid[s0 : (e0 + 1)]
    if int(out.shape[0]) != int(abs_frame_end - abs_frame_start + 1):
        raise ValueError(
            f"frame slice length mismatch: need={abs_frame_end-abs_frame_start+1} got={out.shape[0]} "
            f"(abs [{abs_frame_start}..{abs_frame_end}] -> local [{local_start}..{local_end}])"
        )
    return out


def _frame_tensor_chw_neg1to1_to_bgr_uint8(fr_3hw: torch.Tensor) -> "np.ndarray":  # type: ignore[name-defined]
    """
    fr_3hw: torch.Tensor [3,H,W] in [-1,1] (RGB)
    return: uint8 [H,W,3] in BGR
    """
    if np is None:
        raise RuntimeError("numpy is required")
    x = fr_3hw.detach().to("cpu", dtype=torch.float32).clamp(-1.0, 1.0)
    x = (x + 1.0) * 0.5 * 255.0
    x = x.round().clamp(0.0, 255.0).to(torch.uint8)
    rgb = x.permute(1, 2, 0).contiguous().numpy()  # HWC RGB
    return rgb[..., ::-1].copy()  # BGR


@dataclass
class SegmentInferResult:
    latent5_input: torch.Tensor
    summed_codes: torch.Tensor
    pred_vid_bgr: Optional[torch.Tensor]
    infer_num_frames: int
    obs_len: int
    next_obs_len: int
    total_num_frames: int
    sample_logprob: float
    trace_path: Optional[str]


def _infer_latents_for_actions_and_advance_cache(
    st: TrajectoryState,
    *,
    segment_index: int,
    seed: int,
    advance_gt_obs_to_next: bool = True,
    need_pred_video: bool = False,
) -> SegmentInferResult:
    """
    For a given segment i:
    - run InfinityStar inference following the closed-loop/rolling-tail config
    - seg0: slice 5 latent steps needed to predict 4 actions
    - seg>0: slice only 4 NEW latent steps, then combine with stored last_latent_1 to form 5 latents
    - overwrite gt_obs cache to the newly revealed prefix (points[i+1])
    Returns: latent5_input [1,16,5,H,W]
    """
    cfg = _get_server_config()
    points = cfg.infinity.points()
    if segment_index < 0 or segment_index >= len(points) - 1:
        raise ValueError(f"bad segment_index={segment_index}, points={points}")

    obs_len = int(points[segment_index])
    next_obs_len = int(points[segment_index + 1])

    # Step-specific knobs
    lock_seed = bool(cfg.infinity.lock_seed_across_steps)
    local_seed = int(seed) + (0 if lock_seed else int(segment_index))
    use_late = int(segment_index) >= int(cfg.infinity.late_step_start)
    step_top_k = int(cfg.infinity.late_top_k) if use_late else int(cfg.infinity.top_k)
    step_top_p = float(cfg.infinity.late_top_p) if use_late else float(cfg.infinity.top_p)
    inj = str(cfg.infinity.late_v2v_history_injection or cfg.infinity.v2v_history_injection) if use_late else str(cfg.infinity.v2v_history_injection)

    # Early-stop rollout: each segment only infers up to its own next_obs_len.
    # seg00: infer to 17f (predict 2..17), seg01: infer to 33f (predict 18..33), seg02: infer to 49f (predict 34..49).
    infer_num_frames = int(next_obs_len)

    summed_codes, pred_vid, step_logprob, step_trace_path = _infer_summed_codes_for_step(
        st,
        step_i=int(segment_index),
        obs_len=obs_len,
        infer_num_frames=infer_num_frames,
        seed=int(local_seed),
        top_k=int(step_top_k),
        top_p=float(step_top_p),
        injection=inj,
        need_pred_video=bool(need_pred_video),
    )  # [1,16,pt_local,H,W]

    # For early-stop inference, treat the predicted horizon as the "total" timeline for slicing.
    total_num_frames = int(infer_num_frames)
    abs_end_lat = (int(next_obs_len) - 1) // 4 + 1  # absolute end latent after this segment

    latent5_input: torch.Tensor
    force_full_window = bool(getattr(st, "last_req_prefix_mode", False))
    if int(segment_index) == 0 or st.last_latent_1 is None or force_full_window:
        # seg0 (or resume failure): provide full 5-latent window [end-4..end]
        abs_start_lat = max(1, int(abs_end_lat) - 4)
        latent5_input = _slice_abs_latents_from_summed_codes(
            summed_codes,
            abs_lat_start=int(abs_start_lat),
            abs_lat_end=int(abs_end_lat),
            infer_num_frames=int(infer_num_frames),
            total_num_frames=int(total_num_frames),
        )
        # If video is too short to have 5 latents, pad by repeating last.
        if int(latent5_input.shape[2]) < 5:
            rep = latent5_input[:, :, -1:].repeat(1, 1, 5 - int(latent5_input.shape[2]), 1, 1)
            latent5_input = torch.cat([latent5_input, rep], dim=2)
    else:
        # seg>0: only take 4 NEW latents (prev_end+1 .. cur_end), then concat with last_latent_1.
        prev_obs_len = int(points[segment_index])  # equals current obs_len
        prev_end_lat = (int(prev_obs_len) - 1) // 4 + 1
        abs_start_lat_new = int(prev_end_lat) + 1
        abs_end_lat_new = int(abs_end_lat)
        new4 = _slice_abs_latents_from_summed_codes(
            summed_codes,
            abs_lat_start=int(abs_start_lat_new),
            abs_lat_end=int(abs_end_lat_new),
            infer_num_frames=int(infer_num_frames),
            total_num_frames=int(total_num_frames),
        )  # [1,16,4,H,W] expected
        # Keep latents on CPU for downstream TSformer + disk saving, and to avoid
        # device-mismatch when concatenating with st.last_latent_1 (stored on CPU).
        if isinstance(new4, torch.Tensor):
            new4 = new4.detach().to("cpu").contiguous()
        if int(new4.shape[2]) < 4:
            rep = new4[:, :, -1:].repeat(1, 1, 4 - int(new4.shape[2]), 1, 1)
            new4 = torch.cat([new4, rep], dim=2)
        last1 = st.last_latent_1
        if last1 is None:
            # should not happen, but keep safe
            last1 = new4[:, :, :1].clone()
        # ensure shapes align on spatial dims (H,W)
        if last1.shape[-2:] != new4.shape[-2:]:
            raise ValueError(f"latent spatial mismatch: last1={tuple(last1.shape)} new4={tuple(new4.shape)}")
        latent5_input = torch.cat([last1.to(new4.dtype), new4], dim=2).contiguous()

    # Normalize latent5 tensor placement for downstream (TSformer expects CPU->to(cuda) inside).
    latent5_input = latent5_input.detach().to("cpu").contiguous()

    # Advance: overwrite caches with the newly revealed GT prefix [1..next_obs_len].
    # For the last segment, callers may disable this to avoid writing non-real frames
    # into gt_obs cache (e.g. when 34-49 are purely predicted).
    if bool(advance_gt_obs_to_next):
        st.stream.correction_clear_pred()  # type: ignore[union-attr]
        _update_gt_obs_cache_to(st, int(next_obs_len))
        st.stream.correction_clear_pred()  # type: ignore[union-attr]
        st.kv_cache = st.stream.infinity.export_kv_cache()  # type: ignore[union-attr]

    # Update memory + save to disk
    st.last_latent_1 = latent5_input[:, :, -1:].detach().to("cpu").contiguous()
    _save_latent_tensor(st, f"seg{int(segment_index):02d}_latent5_input.pt", latent5_input)
    if int(segment_index) == 0:
        _save_latent_tensor(st, "seg00_latent5.pt", latent5_input)
        _save_latent_video_clip(st, "seg00_latent5_17f.mp4", latent5_input, drop_first_frame=False)
        # Also split and save boundary latent (frame 1) + new4 latents (frames 2..17) explicitly.
        _save_latent_tensor(st, "seg00_first1.pt", latent5_input[:, :, 0:1].contiguous())
        _save_latent_tensor(st, "seg00_new4.pt", latent5_input[:, :, 1:].contiguous())
        _save_latent_video_clip(st, "seg00_new4_16f.mp4", latent5_input, drop_first_frame=True)
    else:
        # also save new4 only for inspection
        _save_latent_tensor(st, f"seg{int(segment_index):02d}_new4.pt", latent5_input[:, :, 1:].contiguous())
        _save_latent_video_clip(
            st,
            f"seg{int(segment_index):02d}_new4_16f.mp4",
            latent5_input,
            drop_first_frame=True,
        )
    _save_latent_tensor(st, "last_latent.pt", st.last_latent_1)
    return SegmentInferResult(
        latent5_input=latent5_input,
        summed_codes=summed_codes,
        pred_vid_bgr=pred_vid,
        infer_num_frames=int(infer_num_frames),
        obs_len=int(obs_len),
        next_obs_len=int(next_obs_len),
        total_num_frames=int(total_num_frames),
        sample_logprob=float(step_logprob),
        trace_path=step_trace_path,
    )


def _tsformer_predict_actions_from_summed_codes(
    summed_codes_BCTHW: torch.Tensor,
    *,
    prefix_latents: int,
) -> torch.Tensor:
    """
    Returns last 4 actions (4,6) in cm/deg.
    """
    assert _ts_model is not None
    assert summed_codes_BCTHW.ndim == 5 and summed_codes_BCTHW.shape[0] == 1, f"expect [1,C,T,H,W], got {tuple(summed_codes_BCTHW.shape)}"

    # InfinityStar's WAN VAE often uses patchified codes: (B, 4*C0, T, H/2, W/2).
    # TSformer adapter is trained on the unpatchified representation (C0=16).
    # If we see C=64, undo patchify -> C=16 and spatial x2.
    if int(summed_codes_BCTHW.shape[1]) == 64:
        x = summed_codes_BCTHW.permute(0, 2, 1, 3, 4).contiguous()  # [B,T,C,H,W]
        x = torch.nn.functional.pixel_shuffle(x, 2)  # [B,T,C/4,H*2,W*2]
        summed_codes_BCTHW = x.permute(0, 2, 1, 3, 4).contiguous()  # [B,C/4,T,H*2,W*2]

    assert int(summed_codes_BCTHW.shape[1]) == 16, f"TSformer expects 16ch latents, got C={int(summed_codes_BCTHW.shape[1])}"

    t_lat = int(summed_codes_BCTHW.shape[2])
    k = int(prefix_latents)
    if k > t_lat:
        k = t_lat
    if k < 2:
        raise ValueError(f"prefix_latents too small: {k}")

    # [1,16,T,H,W] -> [T,16,H,W]
    lat_TCHW = summed_codes_BCTHW[0].permute(1, 0, 2, 3).contiguous()  # [T,16,H,W]
    lat_TCHW = lat_TCHW[:k]

    # windows: (k-1, 2, 16, H, W)
    windows = torch.stack([lat_TCHW[:-1], lat_TCHW[1:]], dim=1)
    windows = windows.to(_DEVICE, dtype=torch.float32)

    with torch.no_grad():
        out = _ts_model(windows)  # (N, 6)
        if _ts_mean is not None and _ts_std is not None:
            out = out * _ts_std + _ts_mean

    # last 4 actions (or pad if fewer)
    if out.shape[0] >= 4:
        last4 = out[-4:]
    else:
        # pad by repeating last
        pads = [out[-1:]] * (4 - int(out.shape[0]))
        last4 = torch.cat([out] + pads, dim=0)

    return _to_cm_deg(last4).detach().cpu()


def _ah_denorm_window_preds(pred_norm: "np.ndarray", stats: Dict[str, "np.ndarray"]) -> "np.ndarray":  # type: ignore[name-defined]
    assert np is not None
    b = int(pred_norm.shape[0])
    pred = pred_norm.reshape(b, 3, 6).astype(np.float32)
    mean_a, std_a = stats["mean_angles"], stats["std_angles"]
    mean_t, std_t = stats["mean_t"], stats["std_t"]
    pred[:, :, 0:3] = pred[:, :, 0:3] * std_a[None, None, :] + mean_a[None, None, :]
    pred[:, :, 3:6] = pred[:, :, 3:6] * std_t[None, None, :] + mean_t[None, None, :]
    return pred


def _ah_aggregate_overlapping_windows(
    *,
    num_frames: int,
    window_starts: List[int],
    window_deltas: "np.ndarray",  # (N,3,6)
    window_size: int = 4,
) -> "np.ndarray":  # (T,6)
    assert np is not None
    acc = np.zeros((int(num_frames), 6), dtype=np.float32)
    cnt = np.zeros((int(num_frames),), dtype=np.int32)
    for i, s in enumerate(window_starts):
        for j in range(1, int(window_size)):
            t = int(s) + int(j)
            if 0 <= t < int(num_frames):
                acc[t] += window_deltas[i, j - 1]
                cnt[t] += 1
    out = np.zeros((int(num_frames), 6), dtype=np.float32)
    mask = cnt > 0
    out[mask] = acc[mask] / cnt[mask, None]
    return out


def _actionhead_ref_predict_actions_cm_deg(
    *,
    frames_rgb_uint8: List["np.ndarray"],  # length: 1(prev)+16(clip)=17, RGB uint8
    batch_size: int = 8,
    stride: int = 1,
    pre_resize_hw: int = 0,
) -> List[List[float]]:
    """
    Reference-video actionhead mode (TimesFormer ViT):
    - Takes a sequence of RGB frames (uint8) length T>=4
    - Runs sliding windows of size=4 with stride
    - Each window predicts 3 deltas (for next 3 frames), aggregated to per-frame deltas
    - Returns actions for frames[1:] (length T-1) in API order [dx,dy,dz,droll,dyaw,dpitch] (cm/deg)
    """
    if np is None:
        raise RuntimeError("numpy is required for actionhead mode")
    if _ah_model is None or _ah_stats is None or _ah_preprocess is None:
        raise RuntimeError("actionhead model not initialized")
    if len(frames_rgb_uint8) < 4:
        return []

    # Optional intermediate resize (debug bridge):
    # Some pipelines want to force 480p -> 256x256 before swapping to a native-480p actionhead.
    # NOTE: The reference actionhead checkpoint you provided is trained with img_size=(192,640),
    # so this is only a *pre-resize* step; the model still receives 192x640 after _ah_preprocess.
    if int(pre_resize_hw) <= 0:
        env_pre = os.environ.get("ACTIONHEAD_PRE_RESIZE_HW", "").strip()
        if env_pre:
            try:
                pre_resize_hw = int(env_pre)
            except Exception:
                pre_resize_hw = 0
    # Default: no intermediate pre-resize (directly preprocess 848x480 -> actionhead input).

    # preprocess to tensors (C,H,W), normalized
    frames_t: List[torch.Tensor] = []
    for f in frames_rgb_uint8:
        if int(pre_resize_hw) > 0:
            try:
                pil = Image.fromarray(f)
                pil = pil.resize((int(pre_resize_hw), int(pre_resize_hw)), resample=Image.BILINEAR)
                f = np.asarray(pil, dtype=np.uint8)  # type: ignore[assignment]
            except Exception:
                pass
        frames_t.append(_ah_preprocess(f))  # type: ignore[misc]

    window_size = 4
    t = int(len(frames_t))
    starts = list(range(0, t - window_size + 1, max(1, int(stride))))
    clips: List[torch.Tensor] = []
    for s in starts:
        # stack to (C,T,H,W)
        x = torch.stack([frames_t[s + i] for i in range(window_size)], dim=0).transpose(0, 1).contiguous()
        clips.append(x)

    preds = []
    device = torch.device("cuda" if _DEVICE == "cuda" else "cpu")
    with torch.no_grad():
        for i in range(0, len(clips), max(1, int(batch_size))):
            batch = torch.stack(clips[i : i + int(batch_size)], dim=0).to(device)  # (B,C,T,H,W)
            out = _ah_model(batch.float())
            preds.append(out.detach().cpu().numpy())
    pred_norm = np.concatenate(preds, axis=0) if preds else np.zeros((0, 18), dtype=np.float32)
    window_deltas = _ah_denorm_window_preds(pred_norm, _ah_stats) if pred_norm.shape[0] > 0 else np.zeros((0, 3, 6), dtype=np.float32)
    deltas = _ah_aggregate_overlapping_windows(num_frames=t, window_starts=starts, window_deltas=window_deltas, window_size=window_size)

    # Convert per-frame deltas to API actions (cm/deg), for frames[1:] only.
    # Here we assume delta format is [dz, dy, dx, tx, ty, tz] with angles in rad and translation in meters.
    out_actions: List[List[float]] = []
    for i in range(1, t):
        dz, dy, dx = [float(x) for x in deltas[i, 0:3]]
        tx, ty, tz = [float(x) for x in deltas[i, 3:6]]
        out_actions.append(
            [
                tx * float(_ah_stats["translation_divisor"]),
                ty * float(_ah_stats["translation_divisor"]),
                tz * float(_ah_stats["translation_divisor"]),
                dx * (180.0 / math.pi),  # roll (x)
                dz * (180.0 / math.pi),  # yaw (z)
                dy * (180.0 / math.pi),  # pitch (y)
            ]
        )
    return out_actions


# -------------------------
# 7) FastAPI schema (optional)
# -------------------------
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="InfinityStar+TSformer Action API",
        description="InfinityStar summed_codes (latents) -> TSformer(P2P) delta actions (cm/deg). num_frames/step are configurable via config.json.",
        version="0.1.0",
    )

    class PredictDeltaActionsRequest(BaseModel):
        session_id: str = Field(..., description="Trajectory/session identifier")
        instruction: Optional[str] = Field(None, description="Prompt/instruction; used on first call or when updating prompt")
        prompt: Optional[str] = Field(None, description="Alias of instruction (compat)")
        negative_prompt: Optional[str] = Field("", description="Optional negative prompt")
        images_base64: List[str] = Field(..., description="RGB images as base64 strings; first call typically 1 frame, later typically 16 frames")
        reset_session: bool = Field(
            False,
            description="If true, forces starting a fresh run even if the same session_id was used before (drops in-memory state and avoids overwriting by using a new internal run session id).",
        )
        action_head_mode: str = Field(
            "tsformer_latent",
            description=(
                "Action head mode. "
                "'tsformer_latent' (default): 5 latents -> 4 actions per segment. "
                "'actionhead_ref_vit': decode Infinity predicted video to RGB frames and run a 4-frame sliding-window ViT "
                "(stride=1, overlapping windows aggregated) to output 16 actions per 16-frame clip."
            ),
        )
        action_head_batch_size: int = Field(8, description="Batch size for actionhead_ref_vit sliding-window inference.")
        action_head_stride: int = Field(1, description="Stride for actionhead_ref_vit sliding-window inference (default 1).")
        action_head_pre_resize_hw: int = Field(
            0,
            description=(
                "Optional intermediate pre-resize before actionhead_ref_vit preprocessing. "
                "If >0, each decoded RGB frame will be resized to (N,N) first (e.g. 256). Use 0 to disable. "
                "Note: the reference actionhead model then resizes to (192,640) internally with torchvision Resize (NO crop), matching predict_reference_videos_batch*.py."
            ),
        )
        allow_future_segments: bool = Field(
            False,
            description=(
                "If true, server may emit actions for segment i once the real prefix reaches points[i] "
                "(instead of requiring points[i+1]). This enables a strict closed-loop protocol: "
                "send 1 frame+prompt -> get 4 actions -> execute to collect 16 frames -> send 16 frames -> get next 4 actions, etc."
            ),
        )
        prefix_mode: bool = Field(
            False,
            description="If true, images_base64 contains the full prefix [1..K] each call. Server will only append the new tail frames to avoid duplicates.",
        )
        allow_future_last_segment: bool = Field(
            False,
            description="If true, allows emitting the last segment (e.g. seg02 for points [1,17,33,49]) once the real prefix reaches points[seg] (33), without requiring points[seg+1] (49) real frames. This matches the semantics '34-49 are predicted'.",
        )
        seed: Optional[int] = Field(
            None,
            description="Optional base seed for sampling. If omitted, server uses 0. With lock_seed_across_steps=true, this seed will be used for all segments of the session (official batch script uses seed=base_seed + global_idx*1000).",
        )
        debug: bool = False

    class PredictDeltaActionsResponse(BaseModel):
        actions: List[List[float]] = Field(
            ...,
            description="Delta actions list; each is [dx_cm,dy_cm,dz_cm,droll_deg,dyaw_deg,dpitch_deg]. Length depends on action_head_mode (tsformer_latent: 4 per segment; actionhead_ref_vit: 16 per 16-frame clip).",
        )
        segment_index: int = Field(
            ...,
            description="Which segment this output corresponds to (0..S-1 where S=len(points)-1 from config). -1 means no new segment emitted.",
        )
        num_received_frames: int
        prefix_latents: int
        done: bool
        used_prompt: Optional[str] = None
        segment_old_logprob: Optional[float] = None
        segment_trace_path: Optional[str] = None

    @app.get("/health")
    async def health():
        cfg = _get_server_config()
        tgt_h = tgt_w = None
        try:
            if _infinity_session_template is not None:
                sched = _infinity_session_template.build_schedule_for_num_frames(int(cfg.infinity.num_frames))
                tgt_h, tgt_w = int(sched.tgt_h), int(sched.tgt_w)
        except Exception:
            tgt_h = tgt_w = None
        return {
            "status": "ok",
            "device": _DEVICE,
            "dtype": str(_DTYPE),
            "ts_ckpt_loaded": _ts_model is not None,
            "infinity_loaded": _infinity_session_template is not None,
            "active_sessions": len(_TRAJ),
            "num_frames": int(cfg.infinity.num_frames),
            "step": int(cfg.infinity.step),
            "points": cfg.infinity.points(),
            "h_div_w_template": float(cfg.infinity.h_div_w_template),
            "tgt_h": tgt_h,
            "tgt_w": tgt_w,
            "rolling_tail_infer": bool(cfg.infinity.rolling_tail_infer),
            "rolling_infer_mode": str(cfg.infinity.rolling_infer_mode),
            "v2v_history_injection": str(cfg.infinity.v2v_history_injection),
        }

    @app.on_event("startup")
    async def _startup_load_models():
        """
        Optional eager-load: if env vars are already set, load weights at startup.
        This keeps the 'weights resident' behavior even before the first request arrives.
        """
        cfg = _get_server_config()
        if not cfg.infinity.ckpt:
            # allow starting server without ckpt; requests will fail fast until config/env is set
            print("[Service] startup: InfinityStar ckpt not set, skip eager model load.")
            return
        _init_models(cfg=cfg)

    @app.post("/v1/predict_delta_actions", response_model=PredictDeltaActionsResponse)
    async def predict_delta_actions(req: "PredictDeltaActionsRequest"):
        # Global lock (single-process safety)
        if _LOCK is not None:
            async with _LOCK:
                return _predict_delta_actions_impl(req)
        return _predict_delta_actions_impl(req)

else:
    app = None  # type: ignore


def _predict_delta_actions_impl(req) -> "PredictDeltaActionsResponse":
    cfg = _get_server_config()
    if not cfg.infinity.ckpt:
        raise HTTPException(status_code=500, detail="InfinityStar ckpt is required (set in config.json or INFINITY_CKPT env var)")
    _init_models(cfg=cfg)

    external_session_id = (req.session_id or "").strip()
    if not external_session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    raw_prompt = (req.instruction or "").strip() or (req.prompt or "").strip()
    allow_future_segments = bool(getattr(req, "allow_future_segments", False))
    # Auto "new run" rule to avoid conflicts with stale in-memory state:
    # If the frontend starts a route by sending exactly 1 frame + prompt/instruction,
    # we treat it as a fresh run even if the same external session_id was reused.
    auto_reset_on_one_frame = os.environ.get("INFINITY_RESET_SESSION_ON_ONE_FRAME", "1").strip() in ("1", "true", "True")
    one_frame_with_prompt = bool(raw_prompt) and int(len(getattr(req, "images_base64", []) or [])) == 1
    want_reset = bool(getattr(req, "reset_session", False)) or (auto_reset_on_one_frame and one_frame_with_prompt)
    if want_reset and not raw_prompt:
        raise HTTPException(status_code=400, detail="reset_session requires instruction/prompt")

    if want_reset:
        old_key = _SESSION_ALIAS.get(external_session_id, external_session_id)
        try:
            if old_key in _TRAJ:
                del _TRAJ[old_key]
        except Exception:
            pass
        try:
            # Also drop legacy state stored directly under external_session_id
            if external_session_id in _TRAJ and external_session_id != old_key:
                del _TRAJ[external_session_id]
        except Exception:
            pass
        _SESSION_ALIAS[external_session_id] = _make_run_session_id(external_session_id)

    session_id = _SESSION_ALIAS.get(external_session_id, external_session_id)
    if session_id not in _TRAJ and not raw_prompt:
        raise HTTPException(status_code=400, detail="First call of a session must provide instruction/prompt")

    st = _get_or_create_traj(session_id, raw_prompt, req.negative_prompt or "")

    # Decode and append frames
    if not req.images_base64:
        raise HTTPException(status_code=400, detail="images_base64 is required")

    new_imgs: List[Image.Image] = []
    try:
        for s in req.images_base64:
            new_imgs.append(_load_image_from_base64(s))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"decode images_base64 failed: {e}")

    # IMPORTANT:
    # Many UAVFlow-style datasets store frames as 256x256, but require a fixed
    # training-time template (e.g. h_div_w_template=0.562 -> 848x480) for inference.
    # So by default we DO NOT override `st.h_div_w_template` from the raw frame aspect.
    #
    # If you really want auto-detection from the first frame aspect, enable it via env:
    #   INFINITY_AUTO_H_DIV_W_TEMPLATE=1
    if st.num_frames() == 0 and os.environ.get("INFINITY_AUTO_H_DIV_W_TEMPLATE", "0").strip() in ("1", "true", "True"):
        w, h = new_imgs[0].size
        if w > 0 and h > 0:
            st.h_div_w_template = float(h) / float(w)

    _ensure_traj_infinity_session(st)

    # Determine (tgt_h,tgt_w) once (schedule derived from configured num_frames)
    if st.tgt_h is None or st.tgt_w is None:
        assert st.stream is not None
        sched = st.stream.build_schedule_for_num_frames(int(cfg.infinity.num_frames))
        st.tgt_h, st.tgt_w = int(sched.tgt_h), int(sched.tgt_w)
        # Optional hard check for target resolution (useful when forcing 640x640 templates).
        req_hw = os.environ.get("INFINITY_REQUIRE_TGT_HW", "").strip()
        if req_hw:
            try:
                parts = [p.strip() for p in str(req_hw).split(",")]
                req_h = int(parts[0])
                req_w = int(parts[1])
                if req_h > 0 and req_w > 0 and (int(st.tgt_h), int(st.tgt_w)) != (int(req_h), int(req_w)):
                    raise HTTPException(
                        status_code=500,
                        detail=f"Target resolution mismatch: got {(int(st.tgt_h), int(st.tgt_w))} but INFINITY_REQUIRE_TGT_HW={(req_h, req_w)}. Check h_div_w_template/dynamic_scale_schedule.",
                    )
            except HTTPException:
                raise
            except Exception:
                # ignore malformed env var
                pass

    # If client sends full prefix each time, keep only the new tail frames.
    st.last_req_prefix_mode = bool(getattr(req, "prefix_mode", False))
    if bool(st.last_req_prefix_mode):
        already = int(st.num_frames())
        if already > int(len(new_imgs)):
            raise HTTPException(
                status_code=400,
                detail=f"prefix_mode expects non-decreasing prefix length, but server already has {already} frames and request has only {len(new_imgs)}",
            )
        new_imgs = new_imgs[already:]

    # Transform new frames to [-1,1] at target size, store on CPU
    for pil in new_imgs:
        if st.num_frames() >= int(cfg.infinity.num_frames):
            break
        if infinity_transform is None:
            raise HTTPException(status_code=500, detail="InfinityStar modules not imported (check INFINITY_REPO_ROOT)")
        fr = infinity_transform(pil, int(st.tgt_h), int(st.tgt_w))  # type: ignore[misc]  # [3,H,W] in [-1,1]
        st.frames_cpu.append(fr.cpu())

    n = st.num_frames()
    done = n >= int(cfg.infinity.num_frames)

    points = cfg.infinity.points()
    if len(points) < 2:
        raise HTTPException(status_code=500, detail=f"bad config points={points} (num_frames={cfg.infinity.num_frames}, step={cfg.infinity.step})")

    # Warmup: first frame prepares first-frame conditioning.
    # By default we return no actions on the warmup call.
    # If allow_future_segments is enabled, we continue and may emit seg0 actions immediately from the first frame.
    if n == 1 and st.last_emitted_segment < 0:
        try:
            _prepare_firstframe_condition_if_needed(st)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"InfinityStar warmup failed: {e}")
        if not allow_future_segments:
            return PredictDeltaActionsResponse(
                actions=[],
                segment_index=-1,
                num_received_frames=n,
                prefix_latents=0,
                done=done,
                used_prompt=st.prompt_raw if req.debug else None,
            )

    next_seg = int(st.last_emitted_segment) + 1
    if next_seg >= (len(points) - 1):
        return PredictDeltaActionsResponse(
            actions=[],
            segment_index=-1,
            num_received_frames=n,
            prefix_latents=0,
            done=done,
            used_prompt=st.prompt_raw if req.debug else None,
        )

    # Segment readiness:
    # - default: require real prefix to reach points[seg+1] (e.g. 49) to emit seg2
    # - special-case: last segment can be emitted once prefix reaches points[seg] (e.g. 33),
    #   because frames (34..49) are purely predicted
    seg = int(next_seg)
    is_last_seg = int(seg) == (len(points) - 2)
    ready_default = n >= int(points[seg + 1])
    ready_last_future = bool(getattr(req, "allow_future_last_segment", False)) and is_last_seg and n >= int(points[seg])
    ready_future = bool(allow_future_segments) and n >= int(points[seg])
    if not (ready_default or ready_last_future or ready_future):
        return PredictDeltaActionsResponse(
            actions=[],
            segment_index=-1,
            num_received_frames=n,
            prefix_latents=0,
            done=done,
            used_prompt=st.prompt_raw if req.debug else None,
        )

    prefix_latents_abs = (int(points[seg + 1]) - 1) // 4 + 1

    # Select action head mode
    mode = str(getattr(req, "action_head_mode", "tsformer_latent") or "tsformer_latent").strip().lower()
    if mode in ("", "default", "tsformer_latent"):
        env_mode = os.environ.get("ACTION_HEAD_MODE", "").strip().lower()
        if env_mode:
            mode = env_mode
    use_actionhead_ref_vit = mode in ("actionhead_ref_vit", "actionhead_ref", "actionhead_vit", "ref_vit", "actionhead")

    # InfinityStar closed-loop inference: produce latents (and optionally decode predicted video),
    # and advance gt_obs cache to the newly revealed prefix (points[seg+1]) when allowed.
    try:
        # Only advance GT cache when we truly have real frames up to points[seg+1].
        # If we emit with future (predicted) tail, do not write non-real frames into GT cache.
        advance_gt = bool(ready_default)
        base_seed = 0
        try:
            if getattr(req, "seed", None) is not None:
                base_seed = int(getattr(req, "seed"))
        except Exception:
            base_seed = 0
        infer_res = _infer_latents_for_actions_and_advance_cache(
            st,
            segment_index=seg,
            seed=int(base_seed),
            advance_gt_obs_to_next=advance_gt,
            need_pred_video=bool(use_actionhead_ref_vit),
        )
    except Exception as e:
        print("[Service] _infer_latents_for_actions_and_advance_cache failed.")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"InfinityStar inference failed: {e}")

    actions: List[List[float]] = []
    if use_actionhead_ref_vit:
        # ActionHead (reference-video) mode: decode predicted video -> 4-frame sliding windows -> per-frame actions.
        ckpt_path = os.environ.get("ACTIONHEAD_CKPT", "").strip() or os.environ.get("ACTIONHEAD_REF_CKPT", "").strip()
        run_cfg = os.environ.get("ACTIONHEAD_RUN_CONFIG", "").strip() or os.environ.get("ACTIONHEAD_REF_RUN_CONFIG", "").strip()
        if not ckpt_path or not run_cfg:
            raise HTTPException(
                status_code=500,
                detail="actionhead_ref_vit requires env ACTIONHEAD_CKPT and ACTIONHEAD_RUN_CONFIG (or ACTIONHEAD_REF_CKPT/ACTIONHEAD_REF_RUN_CONFIG)",
            )
        try:
            _init_actionhead_model(ckpt_path=ckpt_path, run_config_path=run_cfg)
            pred_vid = infer_res.pred_vid_bgr
            if pred_vid is None:
                # Best-effort fallback (should not happen when need_pred_video=True)
                assert st.stream is not None
                with torch.no_grad():
                    pred_vid = st.stream.infinity.summed_codes2images(st.stream.vae, infer_res.summed_codes)
            # IMPORTANT: to match `predict_reference_videos_batch*.py` (window_size=4) behavior across clip boundaries,
            # we must provide up to (window_size-1)=3 frames of history before the clip start. Otherwise the first
            # few deltas inside a clip will be under-averaged and deviate from the offline script.
            #
            # For seg i (points=[1,17,33,49]):
            # - We output actions for transitions [obs_len->obs_len+1 .. next_obs_len-1->next_obs_len] => 16 actions.
            # - We build input frames for actionhead as abs frames [ctx_start .. next_obs_len] where
            #   ctx_start = max(1, (obs_len+1) - 3) = max(1, obs_len-2).
            obs_len = int(infer_res.obs_len)
            next_obs_len = int(infer_res.next_obs_len)
            clip_abs_start = int(obs_len) + 1
            clip_abs_end = int(next_obs_len)
            ctx_start_abs = max(1, int(clip_abs_start) - 3)

            frames_rgb: List["np.ndarray"] = []  # type: ignore[name-defined]
            for abs_i in range(int(ctx_start_abs), int(clip_abs_end) + 1):
                # Prefer real frames if we have them (prefix observations are real and should match offline script).
                if 1 <= int(abs_i) <= int(st.num_frames()):
                    bgr = _frame_tensor_chw_neg1to1_to_bgr_uint8(st.frames_cpu[int(abs_i) - 1])
                else:
                    bgr = _slice_abs_frames_from_pred_video_bgr(
                        pred_vid,
                        abs_frame_start=int(abs_i),
                        abs_frame_end=int(abs_i),
                        infer_num_frames=int(infer_res.infer_num_frames),
                        total_num_frames=int(infer_res.total_num_frames),
                    )[0]
                frames_rgb.append(bgr[..., ::-1].copy())

            actions_all = _actionhead_ref_predict_actions_cm_deg(
                frames_rgb_uint8=frames_rgb,
                batch_size=int(getattr(req, "action_head_batch_size", 8) or 8),
                stride=int(getattr(req, "action_head_stride", 1) or 1),
                pre_resize_hw=int(getattr(req, "action_head_pre_resize_hw", 0) or 0),
            )

            # Slice out exactly the 16 actions for this clip.
            # transitions in frames_rgb are consecutive; action index corresponds to "to-frame" position-1.
            start_idx = int(obs_len) - int(ctx_start_abs)
            end_idx = int(start_idx) + (int(clip_abs_end) - int(obs_len))
            actions = actions_all[int(start_idx) : int(end_idx)]
            if len(actions) != int(clip_abs_end) - int(obs_len):
                raise ValueError(f"actionhead actions length mismatch: got={len(actions)} need={int(clip_abs_end)-int(obs_len)}")
        except HTTPException:
            raise
        except Exception as e:
            print("[Service] actionhead_ref_vit inference failed.")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"actionhead_ref_vit inference failed: {e}")
    else:
        # TSformer(P2P): 5 latents -> 4 actions (cm/deg)
        try:
            latent5 = infer_res.latent5_input
            actions_t = _tsformer_predict_actions_from_summed_codes(latent5, prefix_latents=int(latent5.shape[2]))  # (4,6)
            actions = actions_t.tolist()
        except Exception as e:
            print("[Service] TSformer inference failed.")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"TSformer inference failed: {e}")

    st.last_emitted_segment = seg
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI/pydantic not installed; server mode is unavailable.")
    return PredictDeltaActionsResponse(  # type: ignore[name-defined]
        actions=actions,
        segment_index=seg,
        num_received_frames=n,
        prefix_latents=int(prefix_latents_abs),
        done=bool(done or ((ready_last_future or (ready_future and is_last_seg)) and is_last_seg)),
        used_prompt=st.prompt_raw if getattr(req, "debug", False) else None,
        segment_old_logprob=float(getattr(infer_res, "sample_logprob", 0.0)),
        segment_trace_path=getattr(infer_res, "trace_path", None),
    )


# -------------------------
# 8) Internal self-test (no HTTP)
# -------------------------
def _self_test(
    *,
    infinity_ckpt: str,
    route_dir: str,
    ts_ckpt: str,
    ts_stats: str,
    prompt_key: str = "instruction",
) -> None:
    route_dir = os.path.abspath(route_dir)
    meta_path = os.path.join(route_dir, "meta.json")
    images_dir = os.path.join(route_dir, "images")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images dir not found: {images_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    prompt = str(meta.get(prompt_key) or meta.get("instruction_unified") or meta.get("instruction") or meta.get("prompt") or "").strip()
    if not prompt:
        raise ValueError(f"no prompt found in meta.json (key={prompt_key})")

    paths = _sorted_image_paths(images_dir)
    if not paths:
        raise FileNotFoundError("no images found")

    cfg0 = _get_server_config()
    num_frames = int(cfg0.infinity.num_frames)
    step = int(cfg0.infinity.step)
    # Pad/trim to configured num_frames for deterministic test
    if len(paths) < num_frames:
        paths = paths + [paths[-1]] * (num_frames - len(paths))
    else:
        paths = paths[:num_frames]

    cfg = ServerConfig(
        infinity=InfinityConfig(**{**cfg0.infinity.__dict__, "ckpt": os.path.abspath(infinity_ckpt)}),
        tsformer=TSformerConfig(ckpt=os.path.abspath(ts_ckpt), stats=os.path.abspath(ts_stats)),
        infinity_repo_root=cfg0.infinity_repo_root,
    )
    _init_models(cfg=cfg)

    sid = f"selftest_{int(time.time())}"
    st = _get_or_create_traj(sid, prompt, "")

    # Simulate streaming: 1 frame then chunks of `step` until num_frames.
    chunks: List[List[str]] = []
    chunks.append(paths[:1])
    idx = 1
    while idx < num_frames:
        chunks.append(paths[idx : min(num_frames, idx + step)])
        idx += step
    for i, ch in enumerate(chunks):
        imgs = [Image.open(p).convert("RGB") for p in ch]
        # reuse internal impl by temporarily base64-encoding (keeps code path consistent)
        b64s = []
        for pil in imgs:
            buf = BytesIO()
            pil.save(buf, format="PNG")
            b64s.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        req = PredictDeltaActionsRequest(session_id=sid, instruction=prompt if i == 0 else None, images_base64=b64s, debug=True)
        resp = _predict_delta_actions_impl(req)
        print(f"[self_test] step={i} received_frames={resp.num_received_frames} segment={resp.segment_index} prefix_latents={resp.prefix_latents} done={resp.done}")
        if resp.actions:
            print(f"[self_test] actions(last4) = {resp.actions}")


def _init_tsformer_only(*, ts_ckpt: str, ts_stats: str) -> None:
    """Initialize only TSformer weights/stats (skip InfinityStar)."""
    global _ts_model, _ts_mean, _ts_std
    if _ts_model is not None:
        return
    ts_model, mean_t, std_t = _load_tsformer_p2p(
        ckpt_path=os.path.abspath(ts_ckpt),
        stats_path=os.path.abspath(ts_stats) if ts_stats else "",
        device=_DEVICE,
    )
    _ts_model, _ts_mean, _ts_std = ts_model, mean_t, std_t


def _integrate_relative_pose_points(actions_cm_deg: List[List[float]]) -> Dict[str, object]:
    """
    actions: list of 6D deltas in (cm, deg). We integrate by simple addition to get relative poses.
    Returns:
      - start_pose: [0,0,0,0,0,0]
      - poses: length == len(actions) (pose after each action)
      - final_pose
    """
    pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    poses = []
    for a in actions_cm_deg:
        if len(a) != 6:
            raise ValueError(f"action dim must be 6, got {len(a)}")
        pose = [pose[i] + float(a[i]) for i in range(6)]
        poses.append(pose)
    return {"start_pose": [0.0] * 6, "poses": poses, "final_pose": poses[-1] if poses else [0.0] * 6}


def _offline_eval_from_precomputed_summed_codes(
    *,
    route_dir: str,
    ts_ckpt: str,
    ts_stats: str,
    out_dir: str,
    prompt_key: str = "instruction",
    take_first_pixel_frames: Optional[int] = None,
) -> str:
    """
    Offline evaluation without InfinityStar weights:
    - Load meta.json + video_summed_codes.npy from route_dir
    - Slice summed_codes to match first `take_first_pixel_frames` frames (pt = (N-1)//4 + 1)
    - Produce 20 actions (5 segments * 4 actions) and integrated poses
    - Write two json files under out_dir
    Returns output folder path.
    """
    route_dir = os.path.abspath(route_dir)
    meta_path = os.path.join(route_dir, "meta.json")
    summed_path = os.path.join(route_dir, "reshape_actionhead_data", "video_summed_codes.npy")
    images_dir = os.path.join(route_dir, "images")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    if not os.path.exists(summed_path):
        raise FileNotFoundError(f"video_summed_codes.npy not found: {summed_path}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images dir not found: {images_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    prompt = str(meta.get(prompt_key) or meta.get("instruction_unified") or meta.get("instruction") or meta.get("prompt") or "").strip()

    cfg0 = _get_server_config()
    if take_first_pixel_frames is None:
        take_first_pixel_frames = int(cfg0.infinity.num_frames)

    # Ensure we do have >=N images for the "streaming" semantics (even if we don't run Infinity here)
    img_paths = _sorted_image_paths(images_dir)
    if len(img_paths) < int(take_first_pixel_frames):
        raise ValueError(f"need at least {take_first_pixel_frames} images for this offline eval, got {len(img_paths)}")

    import numpy as np

    z = np.load(summed_path)  # expected (1,16,T_lat,H,W)
    if z.ndim != 5 or z.shape[0] != 1 or z.shape[1] != 16:
        raise ValueError(f"unexpected summed_codes shape: {z.shape} (expect (1,16,T,H,W))")

    # Slice to pt for first N pixel frames (temporal_compress_rate=4)
    pt = (int(take_first_pixel_frames) - 1) // 4 + 1
    if z.shape[2] < pt:
        raise ValueError(f"summed_codes time too short: T_lat={z.shape[2]} but need pt={pt}")
    z = z[:, :, :pt]

    summed_codes = torch.from_numpy(z).to(_DEVICE, dtype=torch.float32)

    _init_tsformer_only(ts_ckpt=ts_ckpt, ts_stats=ts_stats)

    # Simulate segments according to points(num_frames, step).
    all_actions: List[List[float]] = []
    points = _obs_points(pred_num_frames=int(take_first_pixel_frames), step=int(cfg0.infinity.step))
    for seg in range(len(points) - 1):
        abs_end_lat = (int(points[seg + 1]) - 1) // 4 + 1
        abs_start_lat = max(1, int(abs_end_lat) - 4)
        z5 = summed_codes[:, :, (abs_start_lat - 1) : abs_end_lat].contiguous()
        a4 = _tsformer_predict_actions_from_summed_codes(z5, prefix_latents=int(z5.shape[2])).tolist()
        all_actions.extend(a4)
    poses_info = _integrate_relative_pose_points(all_actions)

    route_id = os.path.basename(route_dir.rstrip("/"))
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_run = os.path.join(out_dir, f"offline_eval_{route_id}_{int(time.time())}")
    os.makedirs(out_run, exist_ok=True)

    actions_json = {
        "route_id": route_id,
        "route_dir": route_dir,
        "prompt": prompt,
        "take_first_pixel_frames": int(take_first_pixel_frames),
        "pt_used": int(pt),
        "points": points,
        "ts_ckpt": os.path.abspath(ts_ckpt),
        "ts_stats": os.path.abspath(ts_stats) if ts_stats else "",
        "units": {"translation": "cm", "angles": "deg"},
        "actions": all_actions,
        "num_actions": int(len(all_actions)),
    }
    poses_json = {
        "route_id": route_id,
        "units": {"translation": "cm", "angles": "deg"},
        # poses length == num_actions (pose after each action); start_pose kept separately
        **poses_info,
        "note": "poses length equals num_actions (pose after each action). start_pose is provided separately.",
    }

    with open(os.path.join(out_run, "actions.json"), "w", encoding="utf-8") as f:
        json.dump(actions_json, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_run, "relative_poses.json"), "w", encoding="utf-8") as f:
        json.dump(poses_json, f, ensure_ascii=False, indent=2)

    return out_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--offline_eval_precomputed", action="store_true", help="Offline eval using route_dir/reshape_actionhead_data/video_summed_codes.npy (no InfinityStar weights).")
    ap.add_argument("--infinity_ckpt", type=str, default=os.environ.get("INFINITY_CKPT", ""))
    ap.add_argument("--route_dir", type=str, default="")
    ap.add_argument("--out_dir", type=str, default=str(ROOT / "cache"), help="Output directory for offline eval json files.")
    ap.add_argument("--ts_ckpt", type=str, default=DEFAULT_TS_CKPT)
    ap.add_argument("--ts_stats", type=str, default=DEFAULT_TS_STATS)
    ap.add_argument("--prompt_key", type=str, default="instruction")
    args = ap.parse_args()

    if args.self_test:
        if not args.infinity_ckpt:
            raise SystemExit("--infinity_ckpt or env INFINITY_CKPT is required for --self_test")
        if not args.route_dir:
            raise SystemExit("--route_dir is required for --self_test")
        _self_test(
            infinity_ckpt=args.infinity_ckpt,
            route_dir=args.route_dir,
            ts_ckpt=args.ts_ckpt,
            ts_stats=args.ts_stats,
            prompt_key=args.prompt_key,
        )
    elif args.offline_eval_precomputed:
        if not args.route_dir:
            raise SystemExit("--route_dir is required for --offline_eval_precomputed")
        out_run = _offline_eval_from_precomputed_summed_codes(
            route_dir=args.route_dir,
            ts_ckpt=args.ts_ckpt,
            ts_stats=args.ts_stats,
            out_dir=args.out_dir,
            prompt_key=args.prompt_key,
        )
        print(f"[offline_eval_precomputed] wrote json files to: {out_run}")


if __name__ == "__main__":
    main()

