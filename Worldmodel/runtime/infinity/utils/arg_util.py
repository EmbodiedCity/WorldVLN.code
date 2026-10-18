# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

import json
import os
import random
import sys
import time
from collections import OrderedDict
from typing import Union

import numpy as np
import torch
from tap import Tap

import infinity.utils.dist as dist
from infinity.utils.sequence_parallel import SequenceParallelManager as sp_manager


class Args(Tap):
    # ==================================================================================================================
    # ============================================= Paths and Directories ============================================
    # ==================================================================================================================
    local_out_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'local_output')  # Directory to save checkpoints
    data_path: str = ''  # Path to the image dataset
    video_data_path: str = ''  # Path to the video dataset
    bed: str = ''  # Directory to copy checkpoints apart from local_out_path
    vae_path: str = ''  # Path to the VAE checkpoint
    log_txt_path: str = ''  # Path to the log file
    t5_path: str = ''  # Path to the T5 model; if not specified, it will be automatically found
    token_cache_dir: str = ''  # Directory for token cache

    # ==================================================================================================================
    # =============================================== General Training =================================================
    # ==================================================================================================================
    exp_name: str = ''  # Experiment name
    project_name: str = 'infinitystar'  # Name of the wandb project
    tf32: bool = True  # Whether to use TensorFloat32
    auto_resume: bool = True  # Whether to automatically resume from the last checkpoint
    rush_resume: str = ''  # Path to a pretrained infinity checkpoint for rushing resume
    rush_omnistore_resume: str = ''  # Path to an omnistore pretrained checkpoint for rushing resume
    torchshard_resume: str = ''      # Path to an torch shard checkpoint resume
    log_every_iter: bool = False  # Whether to log every iteration
    checkpoint_type: str = 'torch'  # Type of checkpoint: 'torch' or 'onmistore'
    device: str = 'cpu'  # Device to use for training ('cpu' or 'cuda')
    is_master_node: bool = None  # Whether the current node is the master node
    epoch: int = 300  # Number of training epochs (total epochs)
    extra_epochs_after_resume: int = 1  # Ensure at least N epochs run after auto_resume
    cur_epoch: int = 0  # [runtime] current epoch index (for dataset shuffling/logging); do NOT use as total
    log_freq: int = 1  # Logging frequency in stdout
    save_model_iters_freq: int = 1000  # Frequency of saving the model in iterations
    short_cap_prob: float = 0.2  # Probability of training with short captions
    label_smooth: float = 0.0  # Label smoothing factor
    cfg: float = 0.1  # Classifier-free guidance dropout probability
    rand_uncond: bool = False  # Whether to use random, unlearnable unconditional embedding
    twoclip_alternatingtraining: int = 0  # Whether to use two-clip alternating training
    trainer_type: str = 'sft'  # sft | grpo
    wp_it: int = 100  # Warm-up iterations

    # ==================================================================================================================
    # ============================================= Hybrid GRPO + SFT (optional) =======================================
    # ==================================================================================================================
    # Dataset-level shuffle of formed batches (IterableDataset). Disable to preserve deterministic ordering patterns,
    # e.g. 12 offline_grpo clips + 1 sft anchor per trajectory.
    shuffle_batches: int = 1
    # If set (e.g. "sft"), override stepping schedule: optimizer step is taken only when the current batch role matches.
    # Requires dataset to provide `hybrid_roles` in each yielded batch.
    hybrid_step_on_role: str = ""
    # Scale factor applied to GRPO RL loss (useful when mixing with a stronger SFT anchor loss).
    grpo_hybrid_rl_coef: float = 1.0
    freeze_chunk_prefix: int = 0
    partial_freeze_print_summary: int = 1

    # ==================================================================================================================
    # ===================================================== Model ======================================================
    # ==================================================================================================================
    model: str = ''  # Model type: 'b' for VAE training, or any other for GPT training
    sdpa_mem: bool = True  # Whether to use memory-efficient SDPA
    rms_norm: bool = False  # Whether to use RMS normalization
    tau: float = 1  # Tau of self-attention in GPT
    tini: float = -1  # Initialization parameters
    topp: float = 0.0                     # top-p
    topk: float = 0.0                     # top-k
    fused_norm: bool = False  # Whether to use fused normalization
    flash: bool = False  # Whether to use customized flash-attention kernel
    use_flex_attn: bool = False  # Whether to use flex_attn to speed up training
    norm_eps: float = 1e-6  # Epsilon for normalization layers
    Ct5: int = 2048  # Feature dimension of the text encoder
    simple_text_proj: int = 1  # Whether to use a simple text projection
    mask_type: str = 'infinity_elegant_clip20frames_v2'  # Self-attention mask type ('var' or 'video_tower')
    mask_video_first_frame: int = 0  # Whether to mask the first frame of the video when calculating loss

    use_fsdp_model_ema: int = 0  # Whether to use FSDP model EMA
    model_ema_decay: float = 0.9999  # Model EMA decay rate

    rope_type: str = '4d'  # RoPE type ('2d', '3d', or '4d')
    rope2d_each_sa_layer: int = 1  # Apply RoPE2D to each self-attention layer
    rope2d_normalized_by_hw: int = 2  # Apply normalized RoPE2D
    add_lvl_embeding_on_first_block: int = 0  # Apply level PE embedding only to the first block

    # ==================================================================================================================
    # ================================================== Scale Schedule =============================================
    # ==================================================================================================================
    semantic_scales: int = 8  # Number of semantic scales
    semantic_scale_dim: int = 16  # Dimension of semantic scales
    detail_scale_dim: int = 64  # Dimension of detail scales
    use_learnable_dim_proj: int = 0  # Whether to use a learnable dimension projection
    detail_scale_min_tokens: int = 80  # Minimum number of tokens for detail scale
    pn: str = ''  # Pixel numbers, choose from '0.06M', '0.25M', '1M'
    scale_schedule: tuple = None  # [Automatically set] Scale schedule based on pn
    patch_size: int = None  # [Automatically set] Patch size based on scale_schedule
    dynamic_scale_schedule: str = ''  # Dynamic scale schedule for video
    min_scale_ind: int = 3  # Minimum scale index for infinity frame pack
    max_reweight_value: int = 40  # Clipping value for reweighting
    image_scale_repetition: str = '[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]'  # Repetition for image scales
    video_scale_repetition: str = '[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]'  # Repetition for video scales
    inner_scale_boost: int = 0  # Whether to boost inner scales
    drop_720p_last_scale: int = 1  # Whether to drop the last scale for 720p
    reweight_loss_by_scale: int = 0  # Reweight loss by scale

    # ==================================================================================================================
    # ================================================== Optimization ==================================================
    # ==================================================================================================================
    tlr: float = 2e-5  # Learning rate
    grad_clip: float = 5  # Gradient clipping threshold
    cdec: bool = False  # Whether to decay the grad clip thresholds
    opt: str = 'adamw'  # Optimizer type ('adamw' or 'lion')
    ada: str = '0.9_0.97'  # Adam's beta parameters (e.g., '0.9_0.999')
    adam_eps: float = 0.0  # Adam's epsilon
    fused_adam: bool = True  # Whether to use fused Adam optimizer
    disable_weight_decay: int = 1  # Whether to disable weight decay on sparse params
    fp16: int = 2  # Floating point precision: 1 for fp16, 2 for bf16
    
    # ==================================================================================================================
    # ====================================================== Data ======================================================
    # ==================================================================================================================
    video_fps: int = 16  # Frames per second for video
    video_frames: int = 81  # Number of frames per video
    video_sample_mode: str = 'duration'  # 'duration': current behavior; 'fixed_full': uniformly sample `video_frames` over full segment; 'segment_full': sample full segment up to cap with 4n+1 rule; 'segment_full_floor_sec': sample full segment but floor length to 1+video_fps*n (align to 1s pt keys)
    video_batch_size: int = 1  # Batch size for video data
    workers: int = 16  # Number of dataloader workers
    image_batch_size: int = 0  # [Automatically set] Batch size per GPU for image data
    ac: int = 1  # Gradient accumulation steps
    max_train_iters: int = 0  # Stop current invocation after N training iterations; 0 disables this limit
    r_accu: float = 1.0  # [Automatically set] Reciprocal of gradient accumulation
    tlen: int = 512  # Truncate text embedding to this length
    num_of_label_value: int = 2  # Number of label values (2 for bitwise, 0 for index-wise)
    dynamic_resolution_across_gpus: int = 1  # Allow dynamic resolution across GPUs
    enable_dynamic_length_prompt: int = 0  # Enable dynamic length prompt during training
    use_streaming_dataset: int = 0  # Whether to use a streaming dataset
    iterable_data_buffersize: int = 90000  # Buffer size for streaming dataset
    image_batches_multiply: float = 1.0  # Multiplier for the number of image batches per epoch
    down_size_limit: int = 10000  # Download size limit for videos in MB
    addition_pn_list: str = '[]'  # Additional pixel number list
    video_caption_type: str = 'tarsier2_caption'  # Type of video caption to use
    only_images4extract_feats: int = 0  # Whether to only extract features for images
    train_max_token_len: int = -1  # Maximum token length for training
    train_with_var_seq_len: int = 0  # Whether to train with variable sequence length
    video_var_len_prob: str = '[30, 30, 30, 5, 3, 2]'  # Probability distribution for variable video length
    duration_resolution: int = 1  # Resolution for duration
    seq_pack_bucket: int = 1000  # Bucket size for sequence packing
    drop_long_video: int = 0  # Whether to drop long videos
    min_video_frames: int = -1  # Minimum number of video frames
    restrict_data_size: int = -1  # Restrict the size of the dataset
    allow_less_one_elem_in_seq: int = 0  # Allow sequences with less than one element
    train_192pshort: int = 0  # Whether to train with 192p short videos
    steps_per_frame: int = 3  # Steps per frame for the video tower
    add_motion_score2caption: int = 0  # Whether to prepend motion score to the caption
    context_frames: int = 10000  # Context frames for the video tower
    cached_video_frames: int = 81  # Number of cached video frames
    frames_inner_clip: int = 20  # Number of frames in a clip for infinity frame pack
    context_interval: int = 2  # Context interval
    context_from_largest_no: int = 1  # Context from the largest number
    append_duration2caption: int = 0  # Whether to append duration to the caption
    cache_check_mode: int = 0  # Cache check mode
    online_t5: bool = True  # Whether to use online T5 or load local features
    
    # ==================================================================================================================
    # ============================================= Distributed Training ===============================================
    # ==================================================================================================================
    enable_hybrid_shard: bool = False  # Whether to use hybrid FSDP
    inner_shard_degree: int = 8  # Inner degree for FSDP
    zero: int = 0  # DeepSpeed ZeRO stage
    buck: str = 'chunk'  # Module-wise bucketing for FSDP
    fsdp_orig: bool = True  # Whether to use original FSDP
    enable_checkpointing: str = None  # Checkpointing strategy: 'full-block', 'self-attn'
    pad_to_multiplier: int = 128  # Pad sequence length to a multiplier of this value
    sp_size: int = 0  # Sequence parallelism size
    fsdp_save_flatten_model: int = 1  # Whether to save the flattened model in FSDP
    inject_sync: int = 0  # Whether to inject synchronization
    model_init_device: str = 'cuda'  # Device for model initialization
    fsdp_init_device: str = 'cuda'  # Device for FSDP initialization
    
    # ==================================================================================================================
    # ======================================================= VAE ======================================================
    # ==================================================================================================================
    vae_type: int = 64  # VAE type (e.g., 16/32/64 for bsq vae quant bits)
    fake_vae_input: bool = False  # Whether to use fake VAE input for debugging
    use_slice: int = 1  # Whether to use slicing for VAE encoding
    use_vae_token_cache: int = 1  # Whether to use token cache for VAE
    save_vae_token_cache: int = 0  # Whether to save the VAE token cache
    allow_online_vae_feature_extraction: int = 1  # Allow online VAE feature extraction
    use_text_token_cache: int = 0  # Whether to use text token cache
    videovae: int = 10  # Whether to use a video VAE
    use_feat_proj: int = 2  # Whether to use feature projection
    use_two_stage_lfq: int = 0  # Whether to use two-stage LFQ
    casual_multi_scale: int = 0  # Whether to use casual multi-scale
    temporal_compress_rate: int = 4  # Temporal compression rate
    apply_spatial_patchify: int = 0  # Whether to apply spatial patchify

    # ======================= Offline GRPO =======================
    grpo_initial_logprob_atol: float = 0.05
    grpo_validate_updates: int = 1
    grpo_diagnostics_path: str = ''
    grpo_block_dtype: str = 'float32'  # optimizer/master storage; forward dtype also accounts for nested FSDP MP
    grpo_fsdp_compute_dtype: str = 'none'  # opt-in bfloat16 chunks with FP32 FSDP masters/root
    grpo_fsdp_cpu_offload: int = 0  # FSDP CPU parameter/gradient/optimizer offload; ac=1 or controlled ac=2/no_sync
    grpo_chunk_gradient_diagnostics: int = 0
    grpo_adv_clip: float = 10.0
    grpo_ratio_eps: float = 0.2
    grpo_kl_beta: float = 0.0
    grpo_lambda_act: float = 1.0
    grpo_lambda_task: float = 1.0
    grpo_lambda_ce: float = 0.0
    grpo_alpha_decay: float = 0.9
    grpo_require_old_logprob: int = 1
    grpo_weight_mode: str = "raw_reward"  # raw_reward | gate_mean | rank_gate (legacy)
    grpo_aux_sft_coef: float = 0.0  # optional: add coef * SFT loss to RL objective (stability)
    # How to compute logp_new for PPO/GRPO ratio.
    # - trace_replay: strict but very memory heavy; replays the autoregressive generation loop with forced idx_trace.
    # - trace_ce: teacher-forcing evaluation on the forced idx_trace using a single forward pass (no KV-cache replay).
    #   This is the standard PPO/RLHF approach for logprob evaluation (1 forward to score chosen tokens).
    # - proxy_ce: legacy per-scale CE proxy (kept for compatibility).
    grpo_new_logprob_mode: str = "trace_replay"  # trace_replay | trace_ce | proxy_ce (legacy)
    # For strict trace-replay, backward can OOM due to saved activations.
    # When enabled, PyTorch will offload saved tensors to CPU (slower but much lower GPU peak).
    grpo_replay_save_on_cpu: int = 1
    grpo_replay_save_on_cpu_pin: int = 1
    # Activation checkpointing inside replay.
    # NOTE: the current replay path uses stateful KV caching, so torch.utils.checkpoint
    # can raise CheckpointError (tensor-count mismatch). Keep default OFF unless you
    # have verified it for your run.
    grpo_replay_checkpointing: int = 0
    # When strict trace-replay is enabled (and aux_sft_coef==0), we can skip building
    # the large SFT autograd graph for per-token CE losses (keep them as no-grad metrics only).
    # This does NOT change strict PPO/GRPO math (ratio/clip/KL), but reduces GPU peak memory.
    grpo_pg_only: int = 1
    grpo_require_nonnegative_adv: int = 1
    grpo_log_weight_stats: int = 1

    # Teacher-forcing debug export: decode per-segment latents, run TSformer, and save trajectories.
    tf_dump_tsformer_enable: int = 0
    tf_dump_interval: int = 0
    tf_dump_step_frames: int = 16
    tf_dump_max_samples_per_step: int = 1
    tf_dump_out_dir: str = ''
    tf_dump_save_video: int = 1
    tf_dump_tsformer_repo_root: str = ''
    tf_dump_tsformer_ckpt: str = ''
    tf_dump_tsformer_stats: str = ''
    tf_dump_ref_pose_json: str = ''
    tf_dump_expand_factor: int = 4
    

    # ==================================================================================================================
    # ============================================ Bitwise Self-Correction =============================================
    # ==================================================================================================================
    noise_apply_layers: int = 1000  # Apply noise to layers
    noise_apply_strength: str = '-1'  # Noise strength
    noise_apply_requant: int = 1  # Requant after applying noise
    noise_apply_random_one: int = 0  # Requant only one scale randomly
    debug_bsc: int = 0  # Save figures and set breakpoints for debugging BSC
    noise_input: int = 0  # Whether to add noise to the input
    reduce_accumulate_error_method: str = 'bsc'  # Method to reduce accumulation error
    


    ############################  Attention! The following arguments and configurations are set automatically, you can skip reading the following part ###############################
    ############################  Attention! The following arguments and configurations are set automatically, you can skip reading the following part ###############################
    ############################  Attention! The following arguments and configurations are set automatically, you can skip reading the following part ###############################


    # would be automatically set in runtime
    branch: str = '' # subprocess.check_output(f'git symbolic-ref --short HEAD 2>/dev/null || git rev-parse HEAD', shell=True).decode('utf-8').strip() or '[unknown]' # [automatically set; don't specify this]
    commit_id: str = '' # subprocess.check_output(f'git rev-parse HEAD', shell=True).decode('utf-8').strip() or '[unknown]'  # [automatically set; don't specify this]
    commit_msg: str = ''# (subprocess.check_output(f'git log -1', shell=True).decode('utf-8').strip().splitlines() or ['[unknown]'])[-1].strip()    # [automatically set; don't specify this]
    cmd: str = ' '.join(a.replace('--exp_name=', '').replace('--exp_name ', '') for a in sys.argv[7:])  # [automatically set; don't specify this]
    tag: str = 'UK'                     # [automatically set; don't specify this]
    cur_it: str = ''                    # [automatically set; don't specify this]
    MFU: float = None                   # [automatically set; don't specify this]
    HFU: float = None                   # [automatically set; don't specify this]
    # ==================================================================================================================
    # ======================== ignore these parts below since they are only for debug use ==============================
    # ==================================================================================================================
    
    dbg: bool = 'KEVIN_LOCAL' in os.environ       # only used when debug about unused param in DDP
    prof: int = 0           # profile
    prof_freq: int = 50     # profile
    profall: int = 0
    # ==================================================================================================================
    # ======================== ignore these parts above since they are only for debug use ==============================
    # ==================================================================================================================
    
    @property
    def gpt_training(self):
        return len(self.model) > 0

    def set_initial_seed(self, benchmark: bool):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = benchmark
        assert self.seed
        seed = self.seed
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def dump_log(self):
        if not dist.is_local_master():
            return
        nd = {'is_master': dist.is_visualizer()}
        for k, v in {
            'name': self.exp_name, 
            'tag': self.tag, 
            'cmd': self.cmd, 
            'commit': self.commit_id, 
            'branch': self.branch,
            'cur_it': self.cur_it,
            'last_upd': time.strftime("%Y-%m-%d %H:%M", time.localtime()),
            'opt': self.opt,
            'is_master_node': self.is_master_node,
        }.items():
            if hasattr(v, 'item'):v = v.item()
            if v is None or (isinstance(v, str) and len(v) == 0): continue
            nd[k] = v
        
        with open(self.log_txt_path, 'w') as fp:
            json.dump(nd, fp, indent=2)
    
    def state_dict(self, key_ordered=True) -> Union[OrderedDict, dict]:
        d = (OrderedDict if key_ordered else dict)()
        for k in self.class_variables.keys():
            if k not in {'device', 'dbg_ks_fp'}:     # these are not serializable
                d[k] = getattr(self, k)
        return d
    
    def load_state_dict(self, d: Union[OrderedDict, dict, str]):
        if isinstance(d, str):  # for compatibility with old version
            d: dict = eval('\n'.join([l for l in d.splitlines() if '<bound' not in l and 'device(' not in l]))
        for k in d.keys():
            if k in {'is_large_model', 'gpt_training'}:
                continue
            try:
                setattr(self, k, d[k])
            except Exception as e:
                print(f'k={k}, v={d[k]}')
                raise e
    
    @staticmethod
    def set_tf32(tf32: bool):
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high' if tf32 else 'highest')
                print(f'[tf32] [precis] torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}')
            print(f'[tf32] [ conv ] torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}')
            print(f'[tf32] [matmul] torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}')
    
    def __str__(self):
        s = []
        for k in self.class_variables.keys():
            if k not in {'device', 'dbg_ks_fp'}:     # these are not serializable
                s.append(f'  {k:20s}: {getattr(self, k)}')
        s = '\n'.join(s)
        return f'{{\n{s}\n}}\n'


def init_dist_and_get_args():
    for i in range(len(sys.argv)):
        if sys.argv[i].startswith('--local-rank=') or sys.argv[i].startswith('--local_rank='):
            del sys.argv[i]
            break
    args = Args(explicit_bool=True).parse_args(known_only=False)
    
    if len(args.extra_args) > 0 and args.is_master_node == 0:
        print(f'======================================================================================')
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================\n{args.extra_args}')
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================')
        print(f'======================================================================================\n\n')
    
    args.set_tf32(args.tf32)
    
    try: os.makedirs(args.bed, exist_ok=True)
    except: pass
    try: os.makedirs(args.local_out_path, exist_ok=True)
    except: pass
    
    dist.init_distributed_mode(local_out_path=args.local_out_path, fork=False, timeout_minutes=30)
    args.device = dist.get_device()

    # sync seed
    args.seed = int(time.time())
    seed = torch.tensor([args.seed], device=args.device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(seed, op=torch.distributed.ReduceOp.MIN)
    args.seed = seed.item()

    if args.sp_size > 1:
        print(f"INFO: sp_size={args.sp_size}")
        sp_manager.init_sp(args.sp_size)
        
    
    args.r_accu = 1 / args.ac   # gradient accumulation
    args.ada = args.ada or ('0.9_0.96' if args.gpt_training else '0.5_0.9')
    args.opt = args.opt.lower().strip()
    
    # gpt args
    if args.gpt_training:
        assert args.vae_path, 'VAE ckpt must be specified when training GPT'
        from infinity.models import alias_dict
        if args.model in alias_dict:
            args.model = alias_dict[args.model]
    
    args.log_txt_path = os.path.join(args.local_out_path, 'log.txt')
    
    args.enable_checkpointing = None if args.enable_checkpointing in [False, 0, "0"] else args.enable_checkpointing
    args.enable_checkpointing = "full-block" if args.enable_checkpointing in [True, 1, "1"] else args.enable_checkpointing
    assert args.enable_checkpointing in [None, "full-block", "full-attn", "self-attn"], \
        f"only support no-checkpointing or full-block/full-attn checkpointing, but got {args.enable_checkpointing}."
    
    if len(args.exp_name) == 0:
        args.exp_name = os.path.basename(args.bed) or 'test_exp'
    
    if '-' in args.exp_name:
        args.tag, args.exp_name = args.exp_name.split('-', maxsplit=1)
    else:
        args.tag = 'UK'
    
    if dist.is_master():
        os.system(f'rm -rf {os.path.join(args.bed, "ready-node*")} {os.path.join(args.local_out_path, "ready-node*")}')
    
    if args.sdpa_mem:
        from torch.backends.cuda import enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
        enable_flash_sdp(True)
        enable_mem_efficient_sdp(True)
        enable_math_sdp(False)
    print(args)
    if isinstance(args.noise_apply_strength, str):
        args.noise_apply_strength = list(map(float, args.noise_apply_strength.split(',')))
    elif isinstance(args.noise_apply_strength, float):
        args.noise_apply_strength = [args.noise_apply_strength]
    return args
