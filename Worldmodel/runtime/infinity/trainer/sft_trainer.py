# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
from pprint import pformat
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import nullcontext
import math
import os
import os.path as osp
import json
import time
import importlib
import sys

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import torch.distributed as tdist

import infinity.utils.dist as dist
from infinity.models import Infinity
from infinity.models.ema import update_ema
from infinity.models.self_correction import SelfCorrection
from infinity.utils import arg_util, misc, wandb_utils
from infinity.utils.amp_opt import AmpOptimizer
from infinity.schedules import get_encode_decode_func
from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
fulloptstate_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

import queue
import threading

def save_token():
    while True:
        try:
            raw_features, feature_cache_files4images = save_token_queue.get()
            for i in range(len(feature_cache_files4images)):
                if not osp.exists(feature_cache_files4images[i]):
                    os.makedirs(osp.dirname(feature_cache_files4images[i]), exist_ok=True)
                    torch.save(raw_features[i], feature_cache_files4images[i])
                    print(f'Save to {feature_cache_files4images[i]}')
                else:
                    print(f'{feature_cache_files4images[i]} exists, skip')
        except Exception as e:
            print(f"Error saving token: {e}")
        finally:
            save_token_queue.task_done()

save_token_queue = queue.Queue()
saver = threading.Thread(target=save_token, daemon=True)
saver.start()


def _obs_points(pred_num_frames: int, step: int):
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

class InfinityTrainer(object):
    def __init__(
        self, 
        device, 
        raw_scale_schedule: Tuple[int, ...],
        vae_local, 
        gpt_wo_ddp: Infinity, gpt: DDP,
        gpt_opt: AmpOptimizer, 
        label_smooth: float,
        zero=0, 
        vae_type=True, 
        reweight_loss_by_scale=0,
        gpt_wo_ddp_ema=None, 
        gpt_ema=None, 
        use_fsdp_model_ema=False, 
        other_args=None,
    ):
        super(InfinityTrainer, self).__init__()
        
        self.zero = zero
        self.vae_type = vae_type
        
        self.gpt: Union[DDP, FSDP, nn.Module]
        self.gpt, self.vae_local = gpt, vae_local
        self.dynamic_scale_schedule = other_args.dynamic_scale_schedule
        self.steps_per_frame = other_args.steps_per_frame
        self.dynamic_resolution_h_w, self.h_div_w_templates = get_dynamic_resolution_meta(other_args.dynamic_scale_schedule, other_args.video_frames)
        self.gpt_opt: AmpOptimizer = gpt_opt
        self.gpt_wo_ddp: Union[Infinity, torch._dynamo.eval_frame.OptimizedModule] = gpt_wo_ddp  # after torch.compile
        self.gpt_wo_ddp_ema = gpt_wo_ddp_ema
        self.gpt_ema = gpt_ema
        self.self_correction = SelfCorrection(self.vae_local, other_args)
        self.use_fsdp_model_ema = use_fsdp_model_ema
        self.batch_size, self.seq_len = 0, 0
        self.reweight_loss_by_scale = reweight_loss_by_scale
        print(f'self.reweight_loss_by_scale: {self.reweight_loss_by_scale}')
        video_encode, _, get_visual_rope_embeds, get_scale_pack_info = get_encode_decode_func(other_args.dynamic_scale_schedule)
        self.video_encode = video_encode
        # Needed for GRPO trace-replay logprob computation.
        self.get_visual_rope_embeds = get_visual_rope_embeds
        self.get_scale_pack_info = get_scale_pack_info
        
        gpt_uncompiled = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp
        del gpt_uncompiled.rng
        gpt_uncompiled.rng = torch.Generator(device=device)
        del gpt_uncompiled
        
        self.label_smooth = label_smooth

        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='none')
        self.loss_weight = {0:{}, 1:{}}

        # Optional teacher-forcing debug: decode latent clips and run TSformer trajectory export.
        self._tf_dump_enable = bool(int(getattr(other_args, "tf_dump_tsformer_enable", 0)))
        self._tf_dump_interval = int(getattr(other_args, "tf_dump_interval", 0))
        self._tf_dump_step = int(getattr(other_args, "tf_dump_step_frames", 16))
        self._tf_dump_max_samples = max(1, int(getattr(other_args, "tf_dump_max_samples_per_step", 1)))
        self._tf_dump_save_video = bool(int(getattr(other_args, "tf_dump_save_video", 1)))
        self._tf_dump_out_dir = str(getattr(other_args, "tf_dump_out_dir", "") or "").strip()
        self._tf_ts_repo_root = str(getattr(other_args, "tf_dump_tsformer_repo_root", "") or "").strip()
        self._tf_ts_ckpt = str(getattr(other_args, "tf_dump_tsformer_ckpt", "") or "").strip()
        self._tf_ts_stats = str(getattr(other_args, "tf_dump_tsformer_stats", "") or "").strip()
        self._tf_ref_pose_json = str(getattr(other_args, "tf_dump_ref_pose_json", "") or "").strip()
        self._tf_expand_factor = max(1, int(getattr(other_args, "tf_dump_expand_factor", 4)))
        self._tf_ts_model = None
        self._tf_ts_mean = None
        self._tf_ts_std = None
        self._tf_ts_ready = False
        self._tf_ts_init_err = ""
        if self._tf_dump_enable and self._tf_dump_interval <= 0:
            # Safe default to avoid per-iter heavy I/O.
            self._tf_dump_interval = 200
            
        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True
        self.generator = np.random.default_rng(0)
        self._stable_metric_beta = min(0.999, max(0.0, float(getattr(other_args, "grpo_metric_ema_beta", 0.9) or 0.9)))
        self._stable_metric_ema: Dict[str, float] = {}
        self._stable_metric_last: Dict[str, float] = {}
        self._optstep_metric_sums: Dict[str, float] = {}
        self._optstep_metric_counts: Dict[str, int] = {}
        self._optstep_metric_last: Dict[str, float] = {}
        self._optstep_metric_ema: Dict[str, float] = {}
        self._optstep_metric_ema_last: Dict[str, float] = {}

    def _update_scalar_ema(self, store: Dict[str, float], key: str, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        value = float(value)
        if not math.isfinite(value):
            return None
        prev = store.get(key, None)
        if prev is None:
            store[key] = value
        else:
            beta = self._stable_metric_beta
            store[key] = beta * prev + (1.0 - beta) * value
        return store[key]

    def _update_stable_metric_trackers(self, metrics: Dict[str, Optional[float]], stepping: bool) -> None:
        touched: List[str] = []
        for key, value in metrics.items():
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value):
                continue
            ema_val = self._update_scalar_ema(self._stable_metric_ema, key, value)
            if ema_val is not None:
                self._stable_metric_last[key] = ema_val
            self._optstep_metric_sums[key] = self._optstep_metric_sums.get(key, 0.0) + value
            self._optstep_metric_counts[key] = self._optstep_metric_counts.get(key, 0) + 1
            touched.append(key)

        if not stepping:
            return

        for key in touched:
            cnt = self._optstep_metric_counts.get(key, 0)
            if cnt <= 0:
                continue
            optstep_mean = self._optstep_metric_sums[key] / float(cnt)
            self._optstep_metric_last[key] = optstep_mean
            optstep_ema_val = self._update_scalar_ema(self._optstep_metric_ema, key, optstep_mean)
            if optstep_ema_val is not None:
                self._optstep_metric_ema_last[key] = optstep_ema_val
            self._optstep_metric_sums[key] = 0.0
            self._optstep_metric_counts[key] = 0

    def _collect_stable_metrics(self) -> Dict[str, float]:
        stable_metrics: Dict[str, float] = {}
        for key, value in self._stable_metric_last.items():
            stable_metrics[f"{key}_ema"] = value
        for key, value in self._optstep_metric_last.items():
            stable_metrics[f"{key}_optstep"] = value
        for key, value in self._optstep_metric_ema_last.items():
            stable_metrics[f"{key}_optstep_ema"] = value
        return stable_metrics

    def _tf_to_cm_deg(self, deltas_m_rad: torch.Tensor) -> torch.Tensor:
        out = deltas_m_rad.clone()
        out[..., 0:3] = out[..., 0:3] * 100.0
        out[..., 3:6] = out[..., 3:6] * (180.0 / math.pi)
        return out

    def _tf_integrate(self, actions_cm_deg):
        pose = [0.0] * 6
        poses = []
        for a in actions_cm_deg:
            pose = [float(pose[i]) + float(a[i]) for i in range(6)]
            poses.append(pose)
        return {"start_pose": [0.0] * 6, "poses": poses, "final_pose": (poses[-1] if poses else [0.0] * 6)}

    def _tf_expand_actions4_to16(self, actions4):
        # TSformer outputs 4 deltas per clip; expand to 16 frame-level deltas by repeating each delta/factor.
        fac = int(self._tf_expand_factor)
        out = []
        for a in actions4:
            per_frame = [float(v) / float(fac) for v in a]
            for _ in range(fac):
                out.append(list(per_frame))
        return out

    def _tf_clip_positions_from_start(self, start_pose6, actions4):
        frame_deltas = self._tf_expand_actions4_to16(actions4)
        cur = [float(v) for v in start_pose6]
        poses16 = []
        for d in frame_deltas:
            cur = [cur[i] + float(d[i]) for i in range(6)]
            poses16.append(list(cur))
        return poses16

    def _tf_clip_mse_vs_ref(self, clip_actions4, points, ref_poses):
        # points is 1-based frame anchors, e.g. [1,17,33,49]
        per_clip = []
        for seg, a4 in enumerate(clip_actions4):
            if seg >= len(points) - 1:
                break
            start_fid = int(points[seg])  # 1-based
            if start_fid - 1 >= len(ref_poses):
                break
            gt_start = ref_poses[start_fid - 1]
            pred16 = self._tf_clip_positions_from_start(gt_start, a4)
            gt16 = ref_poses[start_fid : start_fid + 16]
            if len(gt16) < len(pred16):
                pred16 = pred16[: len(gt16)]
            if len(pred16) == 0:
                continue
            p = np.asarray(pred16, dtype=np.float64)
            g = np.asarray(gt16, dtype=np.float64)
            mse_all = float(np.mean((p - g) ** 2))
            mse_xyz = float(np.mean((p[:, :3] - g[:, :3]) ** 2))
            mse_yaw = float(np.mean((p[:, 4] - g[:, 4]) ** 2))
            per_clip.append(
                {
                    "segment_index": int(seg),
                    "start_frame_1based": int(start_fid),
                    "pred_frames": int(len(pred16)),
                    "mse_all6": mse_all,
                    "mse_xyz": mse_xyz,
                    "mse_yaw": mse_yaw,
                }
            )
        if not per_clip:
            return {"per_clip": [], "global_mse_all6": None, "global_mse_xyz": None, "global_mse_yaw": None}
        return {
            "per_clip": per_clip,
            "global_mse_all6": float(np.mean([x["mse_all6"] for x in per_clip])),
            "global_mse_xyz": float(np.mean([x["mse_xyz"] for x in per_clip])),
            "global_mse_yaw": float(np.mean([x["mse_yaw"] for x in per_clip])),
        }

    def _tf_maybe_init_tsformer(self, device: str):
        if self._tf_ts_ready or self._tf_ts_init_err:
            return
        if not self._tf_ts_ckpt:
            self._tf_ts_init_err = "tf_dump_tsformer_ckpt is empty"
            return
        repo_root = self._tf_ts_repo_root.strip()
        if not repo_root:
            self._tf_ts_init_err = "tf_dump_tsformer_repo_root is empty"
            return
        try:
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            m = importlib.import_module("pretrain_latent_p2p")
            build_p2p_model = getattr(m, "build_p2p_model")
            args = type("A", (), {})()
            args.window_size = 2
            args.hidden_dim = 96
            args.num_layers = 2
            args.device = device
            args.checkpoint = self._tf_ts_ckpt
            args.stats_path = self._tf_ts_stats
            model = build_p2p_model(args)
            model.to(device).eval()

            ckpt = torch.load(self._tf_ts_ckpt, map_location="cpu")
            sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            new_sd = {}
            for k, v in sd.items():
                if k.startswith("module."):
                    new_sd[k[7:]] = v
                else:
                    new_sd[k] = v
            model.load_state_dict(new_sd, strict=False)

            mean_t = std_t = None
            if self._tf_ts_stats and osp.exists(self._tf_ts_stats):
                with open(self._tf_ts_stats, "r", encoding="utf-8") as f:
                    st = json.load(f)
                mean_t = torch.tensor(st["mean"], dtype=torch.float32, device=device)
                std_t = torch.tensor(st["std"], dtype=torch.float32, device=device)
            self._tf_ts_model = model
            self._tf_ts_mean = mean_t
            self._tf_ts_std = std_t
            self._tf_ts_ready = True
            print(f"[TF-TS] initialized TSformer: ckpt={self._tf_ts_ckpt}")
        except Exception as e:
            self._tf_ts_init_err = str(e)
            print(f"[TF-TS] init failed: {e}")

    @torch.no_grad()
    def _tf_predict_actions_4(self, lat5_BCTHW: torch.Tensor, device: str) -> torch.Tensor:
        assert self._tf_ts_model is not None
        x = lat5_BCTHW
        if int(x.shape[1]) == 64:
            t = x.permute(0, 2, 1, 3, 4).contiguous()
            t = torch.nn.functional.pixel_shuffle(t, 2)
            x = t.permute(0, 2, 1, 3, 4).contiguous()
        if int(x.shape[1]) != 16:
            raise ValueError(f"TSformer expects C=16/64, got {tuple(x.shape)}")
        lat_TCHW = x[0].permute(1, 0, 2, 3).contiguous()
        windows = torch.stack([lat_TCHW[:-1], lat_TCHW[1:]], dim=1).to(device=device, dtype=torch.float32)
        out = self._tf_ts_model(windows)
        if self._tf_ts_mean is not None and self._tf_ts_std is not None:
            out = out * self._tf_ts_std + self._tf_ts_mean
        if int(out.shape[0]) >= 4:
            out = out[-4:]
        else:
            out = torch.cat([out] + [out[-1:]] * (4 - int(out.shape[0])), dim=0)
        return self._tf_to_cm_deg(out).detach().cpu()

    @torch.no_grad()
    def _tf_decode_and_save_clip(self, z5_BCTHW: torch.Tensor, out_mp4: str, fps: int, drop_first: bool):
        if not self._tf_dump_save_video:
            return
        try:
            imageio = importlib.import_module("imageio")
        except Exception:
            return
        try:
            gpt_raw = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, "_orig_mod") else self.gpt_wo_ddp
            model_dtype = next(iter(gpt_raw.parameters())).dtype
            vae_device = next(iter(self.vae_local.parameters())).device
            z = z5_BCTHW.to(device=vae_device, dtype=model_dtype)
            bgr = gpt_raw.summed_codes2images(self.vae_local, z)
            if isinstance(bgr, torch.Tensor):
                bgr = bgr.detach().cpu().numpy()
            clip = bgr[0]
            if drop_first and int(clip.shape[0]) > 1:
                clip = clip[1:]
            if int(clip.shape[0]) <= 0:
                return
            rgb = clip[:, :, :, ::-1]
            os.makedirs(osp.dirname(out_mp4), exist_ok=True)
            imageio.mimsave(out_mp4, rgb, fps=int(fps))
        except Exception as e:
            print(f"[TF-TS] decode/save clip failed: {e}")

    @torch.no_grad()
    def _tf_dump_step_trajectory(self, raw_features_list, g_it: int, args):
        if not self._tf_dump_enable:
            return
        if self._tf_dump_interval <= 0 or ((int(g_it) + 1) % int(self._tf_dump_interval) != 0):
            return
        self._tf_maybe_init_tsformer(args.device)
        if not self._tf_ts_ready:
            if self._tf_ts_init_err:
                print(f"[TF-TS] skipped: {self._tf_ts_init_err}")
            return

        out_root = self._tf_dump_out_dir or osp.join(args.local_out_path, "tf_tsformer_debug")
        points = _obs_points(int(args.video_frames), int(self._tf_dump_step))
        if len(points) < 2:
            return
        ts_tag = time.strftime("%Y%m%d_%H%M%S")
        run_dir = osp.join(out_root, f"it_{int(g_it)+1:08d}_{ts_tag}")
        os.makedirs(run_dir, exist_ok=True)

        for sample_idx, raw in enumerate(raw_features_list[: self._tf_dump_max_samples]):
            z = raw.detach()
            if z.ndim == 4:
                z = z.unsqueeze(0)
            z = z.to("cpu").contiguous()
            if z.shape[0] != 1:
                z = z[:1]
            t_lat = int(z.shape[2])
            actions_all = []
            clip_actions4 = []
            clip_positions16 = []
            sample_dir = osp.join(run_dir, f"sample_{sample_idx:02d}")
            os.makedirs(sample_dir, exist_ok=True)

            for seg in range(len(points) - 1):
                abs_end_lat = (int(points[seg + 1]) - 1) // 4 + 1
                abs_start_lat = max(1, int(abs_end_lat) - 4)
                if abs_start_lat > t_lat:
                    break
                end_lat = min(abs_end_lat, t_lat)
                z5 = z[:, :, (abs_start_lat - 1) : end_lat].contiguous()
                if int(z5.shape[2]) < 2:
                    break
                if int(z5.shape[2]) < 5:
                    rep = z5[:, :, -1:].repeat(1, 1, 5 - int(z5.shape[2]), 1, 1)
                    z5 = torch.cat([z5, rep], dim=2)

                a4 = self._tf_predict_actions_4(z5, args.device).tolist()
                actions_all.extend(a4)
                clip_actions4.append(a4)
                if len(clip_positions16) == 0:
                    # Fallback start pose when no GT anchor is available.
                    start_pose = [0.0] * 6
                else:
                    start_pose = clip_positions16[-1][-1]
                clip_positions16.append(self._tf_clip_positions_from_start(start_pose, a4))
                out_mp4 = osp.join(sample_dir, f"seg{seg:02d}_latent5.mp4")
                self._tf_decode_and_save_clip(z5, out_mp4=out_mp4, fps=int(args.video_fps), drop_first=(seg > 0))

            ref_pose_eval = None
            if self._tf_ref_pose_json and osp.exists(self._tf_ref_pose_json):
                try:
                    with open(self._tf_ref_pose_json, "r", encoding="utf-8") as f:
                        ref_poses = json.load(f)
                    if isinstance(ref_poses, list) and len(ref_poses) > 1 and isinstance(ref_poses[0], list):
                        # Re-anchor each clip at GT clip-start before computing MSE.
                        clip_positions16 = []
                        for seg, a4 in enumerate(clip_actions4):
                            if seg >= len(points) - 1:
                                break
                            start_fid = int(points[seg])
                            if start_fid - 1 >= len(ref_poses):
                                break
                            clip_positions16.append(self._tf_clip_positions_from_start(ref_poses[start_fid - 1], a4))
                        ref_pose_eval = self._tf_clip_mse_vs_ref(clip_actions4, points, ref_poses)
                except Exception as e:
                    ref_pose_eval = {"error": str(e)}

            payload = {
                "global_it": int(g_it) + 1,
                "video_frames": int(args.video_frames),
                "step_frames": int(self._tf_dump_step),
                "points": points,
                "expand_factor": int(self._tf_expand_factor),
                "num_actions": len(actions_all),
                "units": {"translation": "cm", "angles": "deg"},
                "actions": actions_all,
                "clip_actions4": clip_actions4,
                "clip_positions16": clip_positions16,
                "relative_poses": self._tf_integrate(actions_all),
                "ref_pose_json": self._tf_ref_pose_json if self._tf_ref_pose_json else None,
                "clip_mse_vs_ref": ref_pose_eval,
                "note": "Teacher-forcing debug export from VAE raw_features (not autoregressive sampled video).",
            }
            with open(osp.join(sample_dir, "trajectory.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[TF-TS] dumped trajectories to: {run_dir}")
    
    def _train_sampling_trace(self, *, epoch, it, g_it, stepping, clip_decay_ratio,
                              metric_lg, args, trace_files, old_logprobs,
                              advantages, clip_ids, rewards):
        """Replay the actual behavior tokens through the normal model wrapper.

        This path never encodes a decoded rollout video or runs an auxiliary SFT
        forward. FSDP/DDP own the complete forward/backward parameter lifecycle.
        """
        from types import SimpleNamespace
        from infinity.trainer.strict_grpo import (
            clipped_policy_loss, load_sampling_trace, move_text_conditions,
            parameter_probes, compare_parameter_probes,
        )
        from infinity.utils.fsdp_grpo import (
            transformer_precision, transformer_chunks, chunk_gradient_statistics,
            validate_replay_execution_schedule,
        )
        if float(args.grpo_aux_sft_coef) != 0 or float(args.grpo_kl_beta) != 0:
            raise ValueError("exact trace_replay currently requires grpo_aux_sft_coef=0 and grpo_kl_beta=0; no approximate CE/KL fallback is permitted")
        if not isinstance(trace_files, list) or not trace_files:
            raise ValueError("exact GRPO requires a nonempty batch of trace files")
        n = len(trace_files)
        if not isinstance(old_logprobs, list) or not isinstance(advantages, list) or len(old_logprobs) != n or len(advantages) != n:
            raise ValueError("exact GRPO needs aligned, explicit old logprobs and advantages")
        # Load the complete local batch before any model forward. Different
        # autoregressive lengths across ranks would interleave FSDP collectives.
        traces = []
        for paths in trace_files:
            if not isinstance(paths, list) or len(paths) != 1:
                raise ValueError("one clip-sample must reference exactly one complete sampling trace")
            traces.append(load_sampling_trace(paths[0]))
        if int(getattr(args, "zero", 0)):
            validate_replay_execution_schedule(traces)
        model = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, "_orig_mod") else self.gpt_wo_ddp
        device = torch.device(args.device)
        weights = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        if not bool(torch.isfinite(weights).all()):
            raise FloatingPointError("non-finite GRPO advantage")
        if int(args.grpo_require_nonnegative_adv) and bool((weights < 0).any()):
            raise ValueError("negative advantage under the configured nonnegative reward scheme")
        if float(args.grpo_adv_clip) > 0:
            weights = weights.clamp(-float(args.grpo_adv_clip), float(args.grpo_adv_clip))
        old = torch.as_tensor(old_logprobs, dtype=torch.float64, device=device)
        validate = bool(int(getattr(args, "grpo_validate_updates", 1)))
        probes_before = parameter_probes(model) if validate and stepping else None
        original_training = self.gpt.training
        original_drop = model.cond_drop_rate
        # eval() disables stochastic layers; it does not disable autograd.
        self.gpt.eval()
        model.cond_drop_rate = 0.0
        new_values, counts, offload_stats = [], [], []
        precision_report = transformer_precision(model)
        try:
            for si, trace in enumerate(traces):
                actual_dtype = precision_report['block_forward_dtype']
                if actual_dtype and str(trace['block_parameter_dtype']).replace('torch.', '') != actual_dtype:
                    raise ValueError(f'behavior/training transformer forward dtype differs: {trace["block_parameter_dtype"]} != {actual_dtype}; masters={precision_report["block_master_dtypes"]}')
                precision = int(getattr(args, 'fp16', 0))
                amp_dtype = 'bfloat16' if precision == 2 else ('float16' if precision > 0 else 'float32')
                if str(trace['autocast_dtype']).replace('torch.', '') != amp_dtype:
                    raise ValueError(f'behavior/training autocast dtype differs: {trace["autocast_dtype"]} != {amp_dtype}')
                target = int(trace["clipid_target"])
                if clip_ids is not None and int(clip_ids[si]) != target:
                    raise ValueError("metadata clip id differs from the trace target")
                if abs(float(old_logprobs[si]) - float(trace["sample_logprob"])) > 1e-4:
                    raise ValueError("metadata old_logprob differs from the saved sampling logprob")
                replay_args = vars(args).copy()
                for key in ("image_scale_repetition", "video_scale_repetition", "max_repeat_times", "noise_input", "semantic_scales", "use_cfg", "use_apg", "apg_norm_threshold"):
                    if key in trace:
                        replay_args[key] = trace[key]
                replay_args.setdefault("use_cfg", 1)
                replay_args.setdefault("use_apg", 0)
                replay_args.setdefault("max_repeat_times", 999999)
                replay_args.setdefault("apg_norm_threshold", 0.0)
                counts.append(int(trace["sample_token_count_by_clip"][target]))
                replay_kw = dict(
                    vae=self.vae_local, scale_schedule=trace["scale_schedule"], B=1,
                    negative_label_B_or_BLT=(move_text_conditions(trace["negative_text_cond_tuple"], device) if trace.get("negative_text_cond_tuple") is not None else None),
                    g_seed=None, cfg_list=[float(trace["infinity_cfg"])] * len(trace["scale_schedule"]),
                    tau_list=[float(x) for x in trace["tau_list"]],
                    top_k=int(trace.get("top_k", 0)), top_p=float(trace.get("top_p", 1.0)),
                    trunk_scale=1000, gt_leak=int(trace.get("gt_leak", -1)),
                    gt_ls_Bl=trace.get("gt_ls_Bl"), low_vram_mode=False,
                    args=SimpleNamespace(**replay_args), get_visual_rope_embeds=self.get_visual_rope_embeds,
                    context_info=trace["context_info"], forced_idx_trace=trace["idx_trace"],
                    logprob_clipid_target=target, kv_cache_reset=True,
                    skip_text_forward=False, cache_text_as_gt=False,
                    expected_step_clipids=trace["step_clipids"],
                    expected_step_is_sampled=trace["step_is_sampled"], expected_token_count=counts[-1],
                )
                save_ctx = nullcontext()
                if int(args.grpo_replay_save_on_cpu) and device.type == "cuda":
                    from infinity.utils.selective_offload import selective_save_on_cpu
                    save_ctx = selective_save_on_cpu(self.gpt, pin_memory=bool(args.grpo_replay_save_on_cpu_pin))
                with save_ctx as offload, self.gpt_opt.amp_ctx:
                    value = self.gpt(
                        move_text_conditions(trace["text_cond_tuple"], device), None,
                        policy_trace_replay_kwargs=replay_kw,
                    )
                if offload is not None:
                    offload_stats.append(offload.stats)
                if isinstance(value, dict):
                    value = value["sample_logprob"]
                if value.dtype != torch.float64:
                    raise ValueError('strict replay must return FP64 summed logprob; update the sampling/replay core')
                new_values.append(value.reshape(-1)[0].double())
        finally:
            model.cond_drop_rate = original_drop
            self.gpt.train(original_training)
        new = torch.stack(new_values)
        losses, ratio, delta = clipped_policy_loss(new, old, weights, float(args.grpo_ratio_eps))
        if not getattr(self, "_strict_policy_has_updated", False):
            tolerance = float(getattr(args, "grpo_initial_logprob_atol", 0.05))
            if float(delta.detach().abs().max()) > tolerance:
                raise RuntimeError(f"initial replay logprob mismatch: max |new-old|={float(delta.detach().abs().max()):.6g}, tolerance={tolerance}; check behavior checkpoint and trace settings")
        final_loss = losses.mean() * float(args.grpo_hybrid_rl_coef)
        active = ((weights > 0) & (ratio.detach() <= 1.0 + float(args.grpo_ratio_eps))) | ((weights < 0) & (ratio.detach() >= 1.0 - float(args.grpo_ratio_eps)))
        self._strict_accumulated_signal = bool(getattr(self, "_strict_accumulated_signal", False) or active.any().item())
        chunk_gradients = []
        def record_chunk_gradients():
            chunk_gradients.extend(chunk_gradient_statistics(
                model, device, require_nonzero=bool(validate and self._strict_accumulated_signal),
            ))
        grad_norm, scale_log2 = self.gpt_opt.backward_clip_step(
            ep=epoch, it=it, g_it=g_it, stepping=stepping, loss=final_loss,
            clip_decay_ratio=clip_decay_ratio,
            require_nonzero_grad=bool(validate and self._strict_accumulated_signal),
            gradient_callback=(record_chunk_gradients if int(getattr(args, "grpo_chunk_gradient_diagnostics", 0)) else None),
        )
        frozen_grad_count = sum(int(p.grad is not None) for p in model.parameters() if not p.requires_grad)
        if frozen_grad_count:
            raise RuntimeError("frozen parameters unexpectedly have gradients")
        frozen_change = trainable_change = 0.0
        if stepping:
            self._strict_policy_has_updated = True
            self._strict_accumulated_signal = False
            if probes_before is not None:
                frozen_change, trainable_change = compare_parameter_probes(probes_before, parameter_probes(model))
            if args.use_fsdp_model_ema and args.model_ema_decay < 1:
                update_ema(self.gpt_ema, self.gpt)
            self.gpt_opt.optimizer.zero_grad(set_to_none=True)
        chunks = transformer_chunks(model)
        trainable_chunks = [i for i, chunk in enumerate(chunks) if any(p.requires_grad for p in chunk.parameters())]
        parameter_counts = torch.tensor([
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(p.numel() for p in model.parameters() if not p.requires_grad),
        ], device=device, dtype=torch.int64)
        if int(getattr(args, "zero", 0)) and torch.distributed.is_initialized():
            torch.distributed.all_reduce(parameter_counts, op=torch.distributed.ReduceOp.SUM)
        trainable_count, frozen_count = parameter_counts.cpu().tolist()
        summary = dict(
            global_it=int(g_it) + 1, optimizer_step=bool(stepping),
            objective="sampled_token_ppo", loss=float(final_loss.detach()),
            optimization_scope=("heads-only" if chunks and not trainable_chunks else
                                "full-backbone" if chunks and len(trainable_chunks) == len(chunks) else "parameter-subset"),
            trainable_chunk_indices=trainable_chunks,
            trainable_parameters=int(trainable_count), frozen_parameters=int(frozen_count),
            **precision_report, chunk_gradients=chunk_gradients, selective_offload=offload_stats,
            ratio_min=float(ratio.detach().min()), ratio_max=float(ratio.detach().max()),
            logprob_delta_max=float(delta.detach().abs().max()),
            old_logprob=[float(x) for x in old.detach().cpu()],
            new_logprob=[float(x) for x in new.detach().cpu()],
            advantages=[float(x) for x in weights.detach().cpu()], token_counts=counts,
            active_policy_samples=int(active.sum()),
            grad_norm=float(grad_norm) if grad_norm is not None else None,
            frozen_gradient_tensors=frozen_grad_count, frozen_probe_max_change=frozen_change,
            trainable_probe_max_change=trainable_change, rank=int(dist.get_rank()),
        )
        self._last_strict_diagnostics = summary
        metric_lg.update(L=summary["loss"], Acc=0.0, L_i=0.0, L_v=0.0, Acc_i=0.0, Acc_v=0.0,
                         tnm=grad_norm, seq_usage=100.0, adv=float(weights.mean()),
                         ratio_min=summary["ratio_min"], ratio_max=summary["ratio_max"])
        if dist.is_master():
            diagnostics_path = str(getattr(args, "grpo_diagnostics_path", "") or osp.join(args.local_out_path, "strict_grpo_updates.jsonl"))
            os.makedirs(osp.dirname(osp.abspath(diagnostics_path)), exist_ok=True)
            with open(diagnostics_path, "a", encoding="utf-8") as output:
                output.write(json.dumps(summary, ensure_ascii=False, allow_nan=False) + "\n")
            print("[strict-grpo] " + json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
        return grad_norm, scale_log2

    def train_step(
        self, epoch: int, it: int, g_it: int, stepping: bool, clip_decay_ratio: float, metric_lg: misc.MetricLogger,
        raw_features_bcthw: FTen, feature_cache_files4images: list, media: str,
        inp_B3HW: FTen, text_cond_tuple: Union[ITen, FTen], args: arg_util.Args,
        grpo_rewards: Optional[list] = None,
        grpo_old_logprobs: Optional[list] = None,
        grpo_adv_finals: Optional[list] = None,
        grpo_reward_acts: Optional[list] = None,
        grpo_reward_tasks: Optional[list] = None,
        grpo_reward_task_raws: Optional[list] = None,
        grpo_reward_task_dense_raws: Optional[list] = None,
        grpo_reward_task_success_raws: Optional[list] = None,
        grpo_succs: Optional[list] = None,
        grpo_succ_trajs: Optional[list] = None,
        grpo_task_final_costs: Optional[list] = None,
        grpo_task_final_pos_errs: Optional[list] = None,
        grpo_task_final_yaw_errs: Optional[list] = None,
        grpo_reward_ces: Optional[list] = None,
        grpo_ref_logprobs: Optional[list] = None,
        grpo_group_ids: Optional[list] = None,
        grpo_clip_ids: Optional[list] = None,
        grpo_trace_files: Optional[list] = None,
        traj_ids: Optional[list] = None,
        hybrid_roles: Optional[list] = None,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        trainer_type = str(getattr(args, "trainer_type", "sft")).lower()
        if hybrid_roles:
            roles = {str(role or "").lower() for role in hybrid_roles}
            if len(roles) != 1:
                raise ValueError("one training batch must have a single hybrid role")
            trainer_type = next(iter(roles))
        if trainer_type == "grpo":
            if str(args.grpo_new_logprob_mode).lower() != "trace_replay":
                raise ValueError("GRPO supports exact trace_replay only; trace_ce/proxy_ce change the behavior distribution")
            return self._train_sampling_trace(
                epoch=epoch, it=it, g_it=g_it, stepping=stepping, clip_decay_ratio=clip_decay_ratio,
                metric_lg=metric_lg, args=args, trace_files=grpo_trace_files,
                old_logprobs=grpo_old_logprobs, advantages=grpo_adv_finals,
                clip_ids=grpo_clip_ids, rewards=grpo_rewards,
            )
        device = args.device
        B = len(inp_B3HW) + len(raw_features_bcthw)

        if media == 'images':
            is_image_batch = 1
        else:
            is_image_batch = 0
        # [forward]
        with self.gpt_opt.amp_ctx:
            with torch.amp.autocast('cuda', enabled=False):
                raw_features_list = []
                if len(inp_B3HW):
                    with torch.no_grad():
                        for inp_ind, inp in enumerate(inp_B3HW):
                            raw_features_, _, _ = self.vae_local.encode_for_raw_features(inp.unsqueeze(0), scale_schedule=None, slice=args.use_slice)
                            raw_features_list.append(raw_features_)
                            if args.use_vae_token_cache and args.save_vae_token_cache and (not osp.exists(feature_cache_files4images[inp_ind])):
                                os.makedirs(osp.dirname(feature_cache_files4images[inp_ind]), exist_ok=True)
                                save_token_queue.put((raw_features_.cpu().data, [feature_cache_files4images[inp_ind]]))
                if len(raw_features_bcthw):
                    raw_features_bcthw = [item.unsqueeze(0) for item in raw_features_bcthw]
                    raw_features_list = raw_features_list + raw_features_bcthw

            full_pts_this_batch = [item.shape[-3] for item in raw_features_list]
            kv_compact, lens, cu_seqlens_k, max_seqlen_k = text_cond_tuple
            x_BLC, x_BLC_mask, gt_BLC, pred_all_bit_indices, visual_rope_cache, sequece_packing_scales, super_scale_lengths, super_querysid_super_refsid, other_info_by_scale = self.video_encode(
                vae=self.vae_local,
                inp_B3HW=None,
                vae_features=raw_features_list,
                self_correction=self.self_correction,
                args=args,
                device=device,
                rope2d_freqs_grid=self.gpt.rope2d_freqs_grid,
                dynamic_resolution_h_w=self.dynamic_resolution_h_w,
                text_lens=lens,
                tokens_remain=args.train_max_token_len,
            )

            # In strict GRPO (trace_replay), the learning signal comes from replayed logp_new,
            # not from the per-token CE loss graph. Keeping the CE autograd graph alongside
            # the replay graph can exceed 80G even on A100.
            trainer_type = str(getattr(args, "trainer_type", "sft") or "sft").strip().lower()
            # Optional per-batch hybrid override (works best with video_batch_size=1):
            # allow switching between GRPO and sft based on dataset-provided role.
            if isinstance(hybrid_roles, list) and len(hybrid_roles) > 0:
                uniq = {str(x or "").strip().lower() for x in hybrid_roles}
                uniq.discard("")
                if len(uniq) == 1:
                    role0 = next(iter(uniq))
                    if role0 == "sft":
                        trainer_type = "sft"
                    elif role0 == "grpo":
                        trainer_type = "grpo"
            use_grpo = trainer_type == "grpo"
            new_mode = str(getattr(args, "grpo_new_logprob_mode", "trace_replay") or "trace_replay").strip().lower()
            aux = float(getattr(args, "grpo_aux_sft_coef", 0.0) or 0.0)
            pg_only_flag = int(getattr(args, "grpo_pg_only", 1) or 0) == 1
            # In strict GRPO, the only gradient-bearing term is the PPO objective (computed from trace replay/CE).
            # We keep the big packed forward in no_grad to save memory even if aux>0, and implement any auxiliary
            # stabilizer loss inside the strict logprob path (so it doesn't force a full packed backward).
            pg_only = bool(use_grpo and new_mode in ("trace_replay", "trace_ce") and pg_only_flag)

            if pg_only:
                with torch.no_grad():
                    loss, acc_bit, valid_sequence_ratio = self.gpt(
                        text_cond_tuple,
                        x_BLC,
                        gt_BL=gt_BLC,
                        is_image_batch=is_image_batch,
                        visual_rope_cache=visual_rope_cache,
                        sequece_packing_scales=sequece_packing_scales,
                        super_scale_lengths=super_scale_lengths,
                        super_querysid_super_refsid=super_querysid_super_refsid,
                        other_info_by_scale=other_info_by_scale,
                    )  # loss & acc_bit: [seq_len] (metrics only)
            else:
                loss, acc_bit, valid_sequence_ratio = self.gpt(
                    text_cond_tuple,
                    x_BLC,
                    gt_BL=gt_BLC,
                    is_image_batch=is_image_batch,
                    visual_rope_cache=visual_rope_cache,
                    sequece_packing_scales=sequece_packing_scales,
                    super_scale_lengths=super_scale_lengths,
                    super_querysid_super_refsid=super_querysid_super_refsid,
                    other_info_by_scale=other_info_by_scale,
                )  # loss & acc_bit: [seq_len]

            # [loss reweight]
            # import pdb; pdb.set_trace()
            acc_pt2scale_acc = {}
            acc_pt2scale_acc_counter = {}
            for full_pt, scale_schedule in self.dynamic_resolution_h_w[self.h_div_w_templates[0]][args.pn]['pt2scale_schedule'].items():
                acc_pt2scale_acc[full_pt] = [[] for _ in range(len(scale_schedule))]
                acc_pt2scale_acc_counter[full_pt] = [0 for _ in range(len(scale_schedule))]
            
            flatten_L_list, flatten_acc_bit_list, flatten_weight_list = [], [], []
            flatten_pg_obj_list = []
            flatten_sample_ind_list: List[int] = []
            ptr = 0
            global_scale_ind = 0
            # NOTE: use_grpo already computed above (keep consistent).
            reward_t = None
            reward_act_t = None
            reward_task_t = None
            reward_task_raw_t = None
            reward_task_dense_raw_t = None
            reward_task_success_raw_t = None
            succ_t = None
            succ_traj_t = None
            task_cost_t = None
            task_pos_err_t = None
            task_yaw_err_t = None
            reward_ce_t = None
            weight_t = None
            oldlp_t = None
            pg_by_sample: Optional[torch.Tensor] = None
            aux_sft_by_sample: Optional[torch.Tensor] = None
            # Exact GRPO returns through _train_sampling_trace before this SFT path.
            for sample_ind, item in enumerate(sequece_packing_scales):
                full_pt = full_pts_this_batch[sample_ind]
                for si, (pt, ph, pw) in enumerate(item):
                    mul_pt_ph_pw = pt * ph * pw
                    start, end = ptr, ptr+mul_pt_ph_pw
                    ptr = end
                    if x_BLC_mask is None:
                        loss_this_scale = loss[start:end].mean()
                        acc_this_scale = acc_bit[start:end].mean()
                    else:
                        pred_elem_num = x_BLC_mask[start:end].sum()
                        assert pred_elem_num > 0
                        loss_this_scale = loss[start:end].sum() / pred_elem_num
                        acc_this_scale = acc_bit[start:end].sum() / pred_elem_num
                    real_si = other_info_by_scale[global_scale_ind]['real_si']
                    volume_times = np.array(other_info_by_scale[global_scale_ind]['largest_scale']).prod() / mul_pt_ph_pw
                    acc_pt2scale_acc[full_pt][real_si].append(acc_this_scale)
                    acc_pt2scale_acc_counter[full_pt][real_si] += 1
                    if self.reweight_loss_by_scale == 0:
                        weight = 1 * mul_pt_ph_pw
                    else:
                        reweight_value = min(args.max_reweight_value, np.power(volume_times, 1/(1+self.reweight_loss_by_scale)))
                        weight = reweight_value * mul_pt_ph_pw
                    flatten_weight_list.append(weight)
                    flatten_L_list.append(loss_this_scale)
                    if use_grpo and weight_t is not None:
                        adv = weight_t[sample_ind]
                        if pg_by_sample is not None:
                            # Trace-replay PPO objective (per-sample), broadcast to all scales for weighting.
                            pg_obj = pg_by_sample[sample_ind]
                        elif oldlp_t is not None:
                            # Legacy proxy: per-scale negative CE as logp_new estimate (kept for compatibility).
                            logp_new = -loss_this_scale
                            delta = (logp_new - oldlp_t[sample_ind]).to(torch.float32).clamp(min=-60.0, max=60.0)
                            ratio = torch.exp(delta)
                            eps = float(getattr(args, "grpo_ratio_eps", 0.2))
                            ratio_clip = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
                            adv32 = adv.to(torch.float32) if isinstance(adv, torch.Tensor) else torch.tensor(float(adv), device=loss.device, dtype=torch.float32)
                            unclipped = ratio * adv32
                            clipped = ratio_clip * adv32
                            obj = torch.minimum(unclipped, clipped)
                            pg_obj = (-obj).to(loss.dtype)
                            if float(getattr(args, "grpo_kl_beta", 0.0)) > 0:
                                kl_proxy = (logp_new - oldlp_t[sample_ind]) ** 2
                                pg_obj = pg_obj + float(getattr(args, "grpo_kl_beta", 0.0)) * kl_proxy
                        else:
                            # Fallback: reward-weighted policy-gradient proxy without ratio clipping.
                            pg_obj = adv * loss_this_scale
                        if pg_obj is not None:
                            flatten_pg_obj_list.append(pg_obj)
                    flatten_acc_bit_list.append(acc_this_scale)
                    flatten_sample_ind_list.append(sample_ind)
                    global_scale_ind += 1
            flatten_weight_list = torch.tensor(flatten_weight_list, dtype=loss.dtype, device=loss.device)
            flatten_weight_list = flatten_weight_list / flatten_weight_list.sum()
            sft_loss = (torch.stack(flatten_L_list) * flatten_weight_list).sum()
            if use_grpo and len(flatten_pg_obj_list) == len(flatten_L_list) and len(flatten_pg_obj_list) > 0:
                # Non-strict fallback path (e.g. no old_logprob): keep legacy behavior.
                rl_loss = (torch.stack(flatten_pg_obj_list) * flatten_weight_list).sum()
                # Optional hybrid scaling: shrink GRPO update magnitude when mixing with a stronger SFT anchor.
                rl_coef = float(getattr(args, "grpo_hybrid_rl_coef", 1.0) or 1.0)
                if rl_coef != 1.0:
                    rl_loss = rl_loss * rl_coef
                aux = float(getattr(args, "grpo_aux_sft_coef", 0.0) or 0.0)
                # If strict PPO objective is present (pg_by_sample != None), auxiliary is already included there
                # (for trace_ce) and we must NOT force a full packed backward via sft_loss.
                if pg_by_sample is not None:
                    final_loss = rl_loss
                else:
                    final_loss = rl_loss + aux * sft_loss
            else:
                final_loss = sft_loss
            final_acc_bit = (torch.stack(flatten_acc_bit_list) * flatten_weight_list).sum()
        
        # [backward]
        grad_norm_t, scale_log2_t = self.gpt_opt.backward_clip_step(ep=epoch, it=it, g_it=g_it, stepping=stepping, loss=final_loss, clip_decay_ratio=clip_decay_ratio)
        
        # update ema 
        if args.use_fsdp_model_ema and (args.model_ema_decay < 1):
            update_ema(self.gpt_ema, self.gpt)

        # [zero_grad]
        if stepping:
            self.gpt_opt.optimizer.zero_grad(set_to_none=True)

        # Optional debug export: decode per-segment latent clips and run TSformer trajectory.
        self._tf_dump_step_trajectory(raw_features_list=raw_features_list, g_it=g_it, args=args)
        
        # [metric logging]
        if metric_lg.log_every_iter or it == 0 or it in metric_lg.log_iters:
            def sum_dict(acc_pt2scale_acc):
                for full_pt in acc_pt2scale_acc:
                    for si in range(len(acc_pt2scale_acc[full_pt])):
                        acc_pt2scale_acc[full_pt][si] = torch.tensor(acc_pt2scale_acc[full_pt][si]).sum()
                return acc_pt2scale_acc

            def dict2list(acc_pt2scale_acc):
                flatten_acc_pt2scale_acc = []
                for key, val in acc_pt2scale_acc.items():
                    flatten_acc_pt2scale_acc.extend(val)
                return flatten_acc_pt2scale_acc
            
            def list2dict(acc_pt2scale_acc, flatten_acc_pt2scale_acc):
                ptr = 0
                for key in acc_pt2scale_acc:
                    for ind in range(len(acc_pt2scale_acc[key])):
                        acc_pt2scale_acc[key][ind] = flatten_acc_pt2scale_acc[ptr]
                        ptr += 1
                return acc_pt2scale_acc
            
            acc_pt2scale_acc = sum_dict(acc_pt2scale_acc)
            flatten_acc_pt2scale_acc = dict2list(acc_pt2scale_acc)
            flatten_acc_pt2scale_acc_counter = dict2list(acc_pt2scale_acc_counter)

            train_loss = final_loss.item()
            train_acc = final_acc_bit.item()
            grad_norm_scalar = 0.0
            if grad_norm_t is not None:
                grad_norm_scalar = float(torch.nan_to_num(grad_norm_t.detach().to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0).item())
            metrics = torch.tensor(flatten_acc_pt2scale_acc + flatten_acc_pt2scale_acc_counter + [grad_norm_scalar, train_loss, train_acc, is_image_batch, valid_sequence_ratio], device=loss.device)
            tdist.all_reduce(metrics, op=tdist.ReduceOp.SUM)
            flatten_acc_pt2scale_acc, flatten_acc_pt2scale_acc_counter = metrics[:len(flatten_acc_pt2scale_acc)], metrics[len(flatten_acc_pt2scale_acc):2*len(flatten_acc_pt2scale_acc)]
            flatten_acc_pt2scale_acc = flatten_acc_pt2scale_acc / (flatten_acc_pt2scale_acc_counter + 1e-16)
            acc_pt2scale_acc = list2dict(acc_pt2scale_acc, flatten_acc_pt2scale_acc)
            acc_pt2scale_acc_counter = list2dict(acc_pt2scale_acc_counter, flatten_acc_pt2scale_acc_counter)
            grad_norm_t, train_loss, train_acc, is_image_batch, valid_sequence_ratio = metrics[2*len(flatten_acc_pt2scale_acc):] / (dist.get_world_size() + 1e-16)
            if args.num_of_label_value == 1:
                key, base = 'Loss', 1
            else:
                key, base = 'Acc', 100
            rew_mean = None
            reward_act_mean = None
            reward_task_mean = None
            reward_task_raw_mean = None
            reward_task_dense_raw_mean = None
            reward_task_success_raw_mean = None
            succ_hit_clip_ratio = None
            succ_hit_traj_ratio = None
            task_cost_mean = None
            task_pos_err_mean = None
            task_yaw_err_mean = None
            reward_ce_mean = None
            adv_mean = None
            pos_ratio = None
            neff_count = None
            success_bonus_hit_ratio = None
            log_weight_stats = bool(int(getattr(args, "grpo_log_weight_stats", 1)))
            weight_zero_ratio = None
            weight_neg_ratio = None
            weight_mean = None
            weight_min = None
            weight_max = None
            success_negative_ratio = None
            if use_grpo and weight_t is not None:
                def _sum_and_count(tensor):
                    if tensor is None:
                        return (
                            torch.tensor(0.0, dtype=torch.float32, device=loss.device),
                            torch.tensor(0.0, dtype=torch.float32, device=loss.device),
                        )
                    tensor = torch.nan_to_num(tensor.detach().to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
                    return tensor.sum(), torch.tensor(float(tensor.numel()), dtype=torch.float32, device=loss.device)

                rew_sum, rew_cnt = _sum_and_count(reward_t)
                reward_act_sum, reward_act_cnt = _sum_and_count(reward_act_t)
                reward_task_sum, reward_task_cnt = _sum_and_count(reward_task_t)
                reward_task_raw_sum, reward_task_raw_cnt = _sum_and_count(reward_task_raw_t)
                reward_task_dense_raw_sum, reward_task_dense_raw_cnt = _sum_and_count(reward_task_dense_raw_t)
                reward_task_success_raw_sum, reward_task_success_raw_cnt = _sum_and_count(reward_task_success_raw_t)
                task_cost_sum, task_cost_cnt = _sum_and_count(task_cost_t)
                task_pos_err_sum, task_pos_err_cnt = _sum_and_count(task_pos_err_t)
                task_yaw_err_sum, task_yaw_err_cnt = _sum_and_count(task_yaw_err_t)
                reward_ce_sum, reward_ce_cnt = _sum_and_count(reward_ce_t)
                adv_sum, adv_cnt = _sum_and_count(weight_t)
                finite_mask = torch.isfinite(weight_t)
                pos_sum = torch.sum((finite_mask & (weight_t > 0)).to(torch.float32))
                zero_sum = torch.sum((finite_mask & (weight_t == 0)).to(torch.float32))
                neg_sum = torch.sum((finite_mask & (weight_t < 0)).to(torch.float32))
                finite_cnt = torch.sum(finite_mask.to(torch.float32))
                finite_weight_t = torch.masked_select(weight_t.detach().to(torch.float32), finite_mask)
                weight_min_t = torch.tensor(float("inf"), dtype=torch.float32, device=loss.device)
                weight_max_t = torch.tensor(float("-inf"), dtype=torch.float32, device=loss.device)
                if finite_weight_t.numel() > 0:
                    weight_min_t = torch.min(finite_weight_t)
                    weight_max_t = torch.max(finite_weight_t)
                success_hit_sum = torch.tensor(0.0, dtype=torch.float32, device=loss.device)
                success_hit_cnt = torch.tensor(0.0, dtype=torch.float32, device=loss.device)
                success_neg_sum = torch.tensor(0.0, dtype=torch.float32, device=loss.device)
                succ_clip_hit_sum = torch.tensor(0.0, dtype=torch.float32, device=loss.device)
                succ_clip_hit_cnt = torch.tensor(0.0, dtype=torch.float32, device=loss.device)
                succ_traj_hit_sum = torch.tensor(0.0, dtype=torch.float32, device=loss.device)
                succ_traj_hit_cnt = torch.tensor(0.0, dtype=torch.float32, device=loss.device)
                if reward_task_success_raw_t is not None:
                    success_mask = reward_task_success_raw_t > 0
                    success_hit_sum = torch.sum(success_mask.to(torch.float32))
                    success_hit_cnt = torch.tensor(float(reward_task_success_raw_t.numel()), dtype=torch.float32, device=loss.device)
                    success_neg_sum = torch.sum((success_mask & (weight_t < 0)).to(torch.float32))
                if succ_t is not None:
                    succ_clip_hit_sum = torch.sum((succ_t > 0).to(torch.float32))
                    succ_clip_hit_cnt = torch.tensor(float(succ_t.numel()), dtype=torch.float32, device=loss.device)
                if succ_traj_t is not None:
                    succ_traj_hit_sum = torch.sum((succ_traj_t > 0).to(torch.float32))
                    succ_traj_hit_cnt = torch.tensor(float(succ_traj_t.numel()), dtype=torch.float32, device=loss.device)
                reward_stats = torch.stack(
                    [
                        rew_sum, rew_cnt,
                        reward_act_sum, reward_act_cnt,
                        reward_task_sum, reward_task_cnt,
                        reward_task_raw_sum, reward_task_raw_cnt,
                        reward_task_dense_raw_sum, reward_task_dense_raw_cnt,
                        reward_task_success_raw_sum, reward_task_success_raw_cnt,
                        task_cost_sum, task_cost_cnt,
                        task_pos_err_sum, task_pos_err_cnt,
                        task_yaw_err_sum, task_yaw_err_cnt,
                        reward_ce_sum, reward_ce_cnt,
                        adv_sum, adv_cnt,
                        pos_sum, zero_sum, neg_sum, finite_cnt,
                        success_hit_sum, success_hit_cnt,
                        success_neg_sum,
                        succ_clip_hit_sum, succ_clip_hit_cnt,
                        succ_traj_hit_sum, succ_traj_hit_cnt,
                    ]
                )
                tdist.all_reduce(reward_stats, op=tdist.ReduceOp.SUM)
                tdist.all_reduce(weight_min_t, op=tdist.ReduceOp.MIN)
                tdist.all_reduce(weight_max_t, op=tdist.ReduceOp.MAX)

                def _safe_mean(sum_idx, cnt_idx):
                    denom = reward_stats[cnt_idx].item()
                    if denom <= 0:
                        return None
                    return float((reward_stats[sum_idx] / reward_stats[cnt_idx].clamp_min(1.0)).item())

                rew_mean = _safe_mean(0, 1)
                reward_act_mean = _safe_mean(2, 3)
                reward_task_mean = _safe_mean(4, 5)
                reward_task_raw_mean = _safe_mean(6, 7)
                reward_task_dense_raw_mean = _safe_mean(8, 9)
                reward_task_success_raw_mean = _safe_mean(10, 11)
                task_cost_mean = _safe_mean(12, 13)
                task_pos_err_mean = _safe_mean(14, 15)
                task_yaw_err_mean = _safe_mean(16, 17)
                reward_ce_mean = _safe_mean(18, 19)
                adv_mean = _safe_mean(20, 21)
                if reward_stats[25].item() > 0:
                    pos_ratio = float((reward_stats[22] / reward_stats[25].clamp_min(1.0) * 100.0).item())
                    neff_count = float(reward_stats[22].item())
                    if log_weight_stats:
                        weight_zero_ratio = float((reward_stats[23] / reward_stats[25].clamp_min(1.0) * 100.0).item())
                        weight_neg_ratio = float((reward_stats[24] / reward_stats[25].clamp_min(1.0) * 100.0).item())
                        weight_mean = adv_mean
                        weight_min = float(weight_min_t.item()) if torch.isfinite(weight_min_t) else 0.0
                        weight_max = float(weight_max_t.item()) if torch.isfinite(weight_max_t) else 0.0
                if reward_stats[27].item() > 0:
                    success_bonus_hit_ratio = float((reward_stats[26] / reward_stats[27].clamp_min(1.0) * 100.0).item())
                    if log_weight_stats and reward_stats[26].item() > 0:
                        success_negative_ratio = float((reward_stats[28] / reward_stats[26].clamp_min(1.0) * 100.0).item())
                if reward_stats[30].item() > 0:
                    succ_hit_clip_ratio = float((reward_stats[29] / reward_stats[30].clamp_min(1.0) * 100.0).item())
                if reward_stats[32].item() > 0:
                    succ_hit_traj_ratio = float((reward_stats[31] / reward_stats[32].clamp_min(1.0) * 100.0).item())

            stable_metric_inputs = {
                'rew': rew_mean,
                'r_act': reward_act_mean,
                'r_task': reward_task_mean,
                'task_cost': task_cost_mean,
                'pos_err_m': task_pos_err_mean,
                'yaw_err_deg': task_yaw_err_mean,
                'adv': adv_mean,
            }
            self._update_stable_metric_trackers(stable_metric_inputs, stepping=stepping)
            stable_metrics = self._collect_stable_metrics()

            metric_lg.update(
                L=train_loss,
                Acc=train_acc*base,
                L_i=0.,
                Acc_i=0.,
                L_v=0.,
                Acc_v=0.,
                tnm=grad_norm_t,
                seq_usage=valid_sequence_ratio*100.,
                rew=rew_mean,
                r_act=reward_act_mean,
                r_task=reward_task_mean,
                r_task_raw=reward_task_raw_mean,
                r_task_dense_raw=reward_task_dense_raw_mean,
                r_task_success_raw=reward_task_success_raw_mean,
                succ_hit=success_bonus_hit_ratio,
                succ_hit_clip=succ_hit_clip_ratio,
                succ_hit_traj_diag=succ_hit_traj_ratio,
                task_cost=task_cost_mean,
                pos_err_m=task_pos_err_mean,
                yaw_err_deg=task_yaw_err_mean,
                r_ce=reward_ce_mean,
                adv=adv_mean,
                pos=pos_ratio,
                w_zero=weight_zero_ratio if log_weight_stats else None,
                w_neg=weight_neg_ratio if log_weight_stats else None,
                w_mean=weight_mean if log_weight_stats else None,
                w_min=weight_min if log_weight_stats else None,
                w_max=weight_max if log_weight_stats else None,
                succ_neg=success_negative_ratio if log_weight_stats else None,
                neff=neff_count,
            )    # todo: Accm, Acct
            if stable_metrics:
                metric_lg.update(**stable_metrics)
            wandb_log_dict = {
                'Overall/train_loss': train_loss,
                'Overall/train_acc': train_acc*base,
                'Overall/grad_norm_t': grad_norm_t,
                'Overall/video_batch_ratio': (1-is_image_batch)*100., 
                'Overall/valid_sequence_ratio': valid_sequence_ratio*100.,
            }
            if reward_task_raw_mean is not None:
                wandb_log_dict['GRPO/reward_task_raw_mean'] = reward_task_raw_mean
            if reward_task_dense_raw_mean is not None:
                wandb_log_dict['GRPO/reward_task_dense_raw_mean'] = reward_task_dense_raw_mean
            if reward_task_success_raw_mean is not None:
                wandb_log_dict['GRPO/reward_task_success_raw_mean'] = reward_task_success_raw_mean
            if success_bonus_hit_ratio is not None:
                wandb_log_dict['GRPO/success_bonus_hit_ratio'] = success_bonus_hit_ratio
            if log_weight_stats and weight_zero_ratio is not None:
                wandb_log_dict['GRPO/weight_zero_ratio'] = weight_zero_ratio
            if log_weight_stats and weight_neg_ratio is not None:
                wandb_log_dict['GRPO/weight_neg_ratio'] = weight_neg_ratio
            if log_weight_stats and weight_mean is not None:
                wandb_log_dict['GRPO/weight_mean'] = weight_mean
            if log_weight_stats and weight_min is not None:
                wandb_log_dict['GRPO/weight_min'] = weight_min
            if log_weight_stats and weight_max is not None:
                wandb_log_dict['GRPO/weight_max'] = weight_max
            if log_weight_stats and success_negative_ratio is not None:
                wandb_log_dict['GRPO/success_negative_ratio'] = success_negative_ratio
            if succ_hit_clip_ratio is not None:
                wandb_log_dict['GRPO/success_clip_hit_ratio'] = succ_hit_clip_ratio
            if succ_hit_traj_ratio is not None:
                wandb_log_dict['GRPO/success_traj_hit_ratio'] = succ_hit_traj_ratio
            if task_cost_mean is not None:
                wandb_log_dict['GRPO/task_cost_mean'] = task_cost_mean
            if task_pos_err_mean is not None:
                wandb_log_dict['GRPO/task_final_pos_err_mean_m'] = task_pos_err_mean
            if task_yaw_err_mean is not None:
                wandb_log_dict['GRPO/task_final_yaw_err_mean_deg'] = task_yaw_err_mean
            if use_grpo and ('weight_t' in locals()) and (weight_t is not None):
                with torch.no_grad():
                    neff = torch.sum(torch.isfinite(weight_t) & (weight_t > 0)).float()
                    wandb_log_dict['GRPO/N_eff'] = float(neff.item())
            for stable_key, stable_value in stable_metrics.items():
                wandb_log_dict[f'GRPOStable/{stable_key}'] = stable_value
            for full_pt in acc_pt2scale_acc:
                for si in range(len(acc_pt2scale_acc[full_pt])):
                    if acc_pt2scale_acc_counter[full_pt][si] > 0:
                        duration = (full_pt-1) / args.temporal_compress_rate
                        wandb_log_dict[f'Details/{key}/t{duration:04.1f}s/s{si+1:03d}'] = acc_pt2scale_acc[full_pt][si].item() * base
                        wandb_log_dict[f'Details/Num/t{duration:04.1f}s/s{si+1:03d}'] = acc_pt2scale_acc_counter[full_pt][si]
            wandb_utils.log(wandb_log_dict, step=g_it)
        return grad_norm_t, scale_log2_t
        
    def __repr__(self):
        return (
            f'\n'
            f'[VGPTTr.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[VGPTTr.structure]: {super(InfinityTrainer, self).__repr__().replace(InfinityTrainer.__name__, "")}'
        )
    
    def ema_load(self):
        self.cached_state_not_ema = {k: v.cpu() for k, v in self.gpt_wo_ddp.state_dict().items()}
        for pi, p_ema in self.pi_para_copy_for_parallel_ema:
            self.gpt_opt.paras[pi].data.copy_(p_ema)
        for pi, para in enumerate(self.gpt_opt.paras):
            dist.broadcast(para, src_rank=pi % dist.get_world_size())
    
    def ema_recover(self):
        self.gpt_wo_ddp.load_state_dict(self.cached_state_not_ema)
        del self.cached_state_not_ema
        self.cached_state_not_ema = None
    
    def get_config(self):
        return {
            'label_smooth': self.label_smooth,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }
    
    def state_dict(self):
        m = self.vae_local
        if hasattr(m, '_orig_mod'):
            m = m._orig_mod
        state = {'config': self.get_config(), 'vae_local': m.state_dict()}
        
        if self.zero:   # TODO: fixme
            state['gpt_fsdp'] = None
            with FSDP.state_dict_type(self.gpt, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                state['gpt_fsdp'] = self.gpt.state_dict()
                if self.use_fsdp_model_ema:
                    state['gpt_ema_fsdp'] = self.gpt_ema.state_dict()
                state['gpt_fsdp_opt'] = FSDP.optim_state_dict(model=self.gpt, optim=self.gpt_opt.optimizer, optim_state_dict=self.gpt_opt.optimizer.state_dict())
            if self.gpt_opt.scaler is not None:
                state['gpt_opt_scaler'] = self.gpt_opt.scaler.state_dict()
        
        else:
            
            for k in ('gpt_wo_ddp', 'gpt_opt'):
                m = getattr(self, k)
                if m is not None:
                    if hasattr(m, '_orig_mod'):
                        m = m._orig_mod
                    state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        if self.zero:
            with FSDP.state_dict_type(self.gpt, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                gpt_state = state['gpt_fsdp']
                # Honor `strict` for FSDP resume as well.
                # This is critical for compatibility when old checkpoints contain
                # keys from slightly different heads (e.g. semantic_head2.*).
                ret = self.gpt.load_state_dict(gpt_state, strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VGPTTr.load_state_dict][zero] gpt missing:  {missing}')
                    print(f'[VGPTTr.load_state_dict][zero] gpt unexpected:  {unexpected}')
                if self.use_fsdp_model_ema:
                    ema_state = state.get('gpt_ema_fsdp', None)
                    if ema_state is not None:
                        ret_ema = self.gpt_ema.load_state_dict(ema_state, strict=strict)
                        if ret_ema is not None:
                            missing, unexpected = ret_ema
                            print(f'[VGPTTr.load_state_dict][zero] gpt_ema missing:  {missing}')
                            print(f'[VGPTTr.load_state_dict][zero] gpt_ema unexpected:  {unexpected}')
                one_group_opt_state = state.get('gpt_fsdp_opt', None)
                """
                AdamW state['gpt_fsdp_opt']:
                {
                    'state': { <para_name>: {'exp_avg': <unsharded_tensor>, 'exp_avg_sq': <unsharded_tensor>, 'step': <int>} },
                    'param_groups': [
                        {
                            'wd_sc': 1.0, 'lr_sc': 1.0, 'lr': xxx, 'betas': (0.9, 0.97), 'eps': 1e-08, 'weight_decay': 0.02,
                            'amsgrad': False, 'foreach': None, 'maximize': False, 'capturable': False, 'differentiable': False, 'fused': True,
                            'params': [<para_name> x m]
                        } x n
                    ]
                }
                one_group_opt_state['param_groups'] = self.gpt_opt.optimizer.state_dict()['param_groups']
                """
                if one_group_opt_state is not None:
                    try:
                        optim_state_dict = FSDP.optim_state_dict_to_load(
                            model=self.gpt, optim=self.gpt_opt.optimizer, optim_state_dict=one_group_opt_state
                        )
                        self.gpt_opt.optimizer.load_state_dict(optim_state_dict)
                    except Exception as e:
                        if strict:
                            raise
                        print(f'[VGPTTr.load_state_dict][zero] skip optimizer state due to mismatch: {e}')

            if self.gpt_opt.scaler is not None:
                try: self.gpt_opt.scaler.load_state_dict(state['gpt_opt_scaler'])
                except Exception as e: print(f'[fp16 load_state_dict err] {e}')
        else:
            for k in ('gpt_wo_ddp', 'gpt_opt'):
                if skip_vae and 'vae' in k: continue
                m = getattr(self, k)
                if m is not None:
                    if hasattr(m, '_orig_mod'):
                        m = m._orig_mod
                    ret = m.load_state_dict(state[k], strict=strict)
                    if ret is not None:
                        missing, unexpected = ret
                        print(f'[VGPTTr.load_state_dict] {k} missing:  {missing}')
                        print(f'[VGPTTr.load_state_dict] {k} unexpected:  {unexpected}')
        
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VGPT.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err)
