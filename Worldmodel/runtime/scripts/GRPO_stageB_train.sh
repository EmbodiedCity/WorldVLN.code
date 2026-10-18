#!/usr/bin/env bash
set -euo pipefail
set -x

# Reduce CUDA allocator fragmentation (safe default; can override via env).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Disable W&B prompts by default in this partial-freeze repo copy.
export WANDB_MODE="${WANDB_MODE:-disabled}"

# Minimal GRPO training launcher (single or multi node).
# Fill paths according to your rollout outputs.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PACKAGE_ROOT}/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PACKAGE_ROOT}/outputs}"
. "${ROOT_DIR}/scripts/GRPO_cluster_env.sh"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
FAST_OUT_DIR="${FAST_OUT_DIR:-${OUTPUT_ROOT}/GRPO_data_fast}"
export TORCHELASTIC_EXIT_BARRIER_TIMEOUT="${TORCHELASTIC_EXIT_BARRIER_TIMEOUT:-3600}"
grpo_resolve_cluster_env
# Trainer mode:
# - GRPO: strict GRPO/PPO-style objective on rollout replay
# - sft: standard teacher-forcing training (ignores GRPO fields)
TRAINER_TYPE="${TRAINER_TYPE:-grpo}"
SHUFFLE_BATCHES="${SHUFFLE_BATCHES:-1}"
HYBRID_STEP_ON_ROLE="${HYBRID_STEP_ON_ROLE:-}"
GRPO_HYBRID_RL_COEF="${GRPO_HYBRID_RL_COEF:-1.0}"
# Split read/write roots:
# - REPLAY_FAST_OUT_DIR: where StageA produced replay_meta_*
# - TRAIN_OUT_ROOT: where StageB writes ckpts/run/token_cache
REPLAY_FAST_OUT_DIR="${REPLAY_FAST_OUT_DIR:-${FAST_OUT_DIR}}"
TRAIN_OUT_ROOT="${TRAIN_OUT_ROOT:-${FAST_OUT_DIR}}"
# Split output roots to avoid network FS flush errors:
# - TRAIN_CKPT_ROOT: checkpoint target (can be network path)
# - TRAIN_LOG_ROOT: runtime logs (should prefer local fast disk)
# - TRAIN_TOKEN_CACHE_ROOT: token cache (should prefer local fast disk)
TRAIN_CKPT_ROOT="${TRAIN_CKPT_ROOT:-${TRAIN_OUT_ROOT}}"
TRAIN_LOG_ROOT="${TRAIN_LOG_ROOT:-${FAST_OUT_DIR}}"
TRAIN_TOKEN_CACHE_ROOT="${TRAIN_TOKEN_CACHE_ROOT:-${FAST_OUT_DIR}}"
RUN_ID="${RUN_ID:-}"
# Replay meta selection:
# - If user sets env REPLAY_META_DIR, use it as-is.
# - Else if RUN_ID is set, use replay_meta_${RUN_ID}.
# - Else fallback to the latest replay_meta_* under REPLAY_FAST_OUT_DIR.
if [ -z "${REPLAY_META_DIR+x}" ]; then
  if [ -n "${RUN_ID}" ]; then
    REPLAY_META_DIR="${REPLAY_FAST_OUT_DIR}/replay_meta_${RUN_ID}"
  else
    REPLAY_META_DIR="$(ls -dt "${REPLAY_FAST_OUT_DIR}"/replay_meta_* 2>/dev/null | head -n 1 || true)"
  fi
fi
ROLL_JSONL="${REPLAY_META_DIR}/part_00.jsonl"
# Token length cap used for sequence packing and video_encode tokens_remain.
# 20480 matches the proven finetune config (safer memory). Override via env if needed.
TRAIN_MAX_TOKEN_LEN_ENV_SET=0
if [ "${TRAIN_MAX_TOKEN_LEN+x}" = "x" ]; then
  TRAIN_MAX_TOKEN_LEN_ENV_SET=1
fi
TRAIN_MAX_TOKEN_LEN="${TRAIN_MAX_TOKEN_LEN:-20480}"
ALLOW_LESS_ONE_ELEM_IN_SEQ="${ALLOW_LESS_ONE_ELEM_IN_SEQ:-1}"
ZERO_STAGE="${ZERO_STAGE:-3}"
# Detect whether user explicitly set EPOCHS (so we can avoid resume-epoch guards).
EPOCHS_ENV_SET=0
if [ "${EPOCHS+x}" = "x" ]; then
  EPOCHS_ENV_SET=1
fi
EPOCHS="${EPOCHS:-60}"
EXTRA_EPOCHS="${EXTRA_EPOCHS:-1}"
IMAGE_SCALE_REPETITION="${IMAGE_SCALE_REPETITION:-[3,3,3,3,3,3,3,3,3,3,3,3,3,3]}"
VIDEO_SCALE_REPETITION="${VIDEO_SCALE_REPETITION:-[3,3,3,3,3,3,3,3,3,3,3,3,2,1]}"
# Align with proven finetune settings by default (override via env if needed).
SHORT_CAP_PROB="${SHORT_CAP_PROB:-0.0}"
ENABLE_DYNAMIC_LENGTH_PROMPT="${ENABLE_DYNAMIC_LENGTH_PROMPT:-0}"
REWEIGHT_LOSS_BY_SCALE="${REWEIGHT_LOSS_BY_SCALE:-4}"
CONTEXT_FROM_LARGEST_NO="${CONTEXT_FROM_LARGEST_NO:-0}"
APPEND_DURATION2CAPTION="${APPEND_DURATION2CAPTION:-1}"
WP_IT="${WP_IT:-0}"
USE_TWO_STAGE_LFQ="${USE_TWO_STAGE_LFQ:-1}"
SEMANTIC_SCALES="${SEMANTIC_SCALES:-11}"
DETAIL_SCALE_MIN_TOKENS="${DETAIL_SCALE_MIN_TOKENS:-350}"
SAVE_MODEL_ITERS_FREQ="${SAVE_MODEL_ITERS_FREQ:-1000}"
KEEP_LATEST_CKPTS="${KEEP_LATEST_CKPTS:-0}"
TLR="${TLR:-5e-6}"
GRAD_CLIP="${GRAD_CLIP:-5.0}"
GRPO_WEIGHT_MODE="${GRPO_WEIGHT_MODE:-raw_reward}"          # raw_reward | gate_mean | rank_gate
GRPO_AUX_SFT_COEF="${GRPO_AUX_SFT_COEF:-0.0}"              # add coef * SFT loss to GRPO objective (optional)
GRPO_LAMBDA_ACT="${GRPO_LAMBDA_ACT:-1.0}"                  # weight for act-adv reward level
GRPO_LAMBDA_TASK="${GRPO_LAMBDA_TASK:-1.0}"                # weight for task-adv reward level
GRPO_LAMBDA_CE="${GRPO_LAMBDA_CE:-1.0}"                    # weight for CE-adv reward level (zscore-exp)
GRPO_KL_BETA="${GRPO_KL_BETA:-0.0}"                        # exact replay: nonzero KL requires an explicit per-token reference implementation
GRPO_NEW_LOGPROB_MODE="${GRPO_NEW_LOGPROB_MODE:-trace_replay}"  # trace_replay | trace_ce | proxy_ce
GRPO_BLOCK_DTYPE="${GRPO_BLOCK_DTYPE:-float32}"
GRPO_FSDP_COMPUTE_DTYPE="${GRPO_FSDP_COMPUTE_DTYPE:-none}"
GRPO_FSDP_CPU_OFFLOAD="${GRPO_FSDP_CPU_OFFLOAD:-0}"
GRPO_CHUNK_GRADIENT_DIAGNOSTICS="${GRPO_CHUNK_GRADIENT_DIAGNOSTICS:-0}"
if [ "${TRAINER_TYPE,,}" = "grpo" ] && [ "${GRPO_NEW_LOGPROB_MODE}" != "trace_replay" ]; then
  echo "[FATAL] GRPO requires exact trace_replay; approximate trace_ce/proxy_ce scoring is unsupported."
  exit 1
fi
GRPO_REQUIRE_NONNEGATIVE_ADV="${GRPO_REQUIRE_NONNEGATIVE_ADV:-1}"
GRPO_LOG_WEIGHT_STATS="${GRPO_LOG_WEIGHT_STATS:-1}"
# PPO/GRPO safety knobs (configurable via env).
GRPO_ADV_CLIP="${GRPO_ADV_CLIP:-10.0}"                     # clip per-sample advantage magnitude
GRPO_RATIO_EPS="${GRPO_RATIO_EPS:-0.2}"                    # PPO ratio clip epsilon
# Optional selective activation offload; parameter views stay under FSDP lifecycle.
GRPO_REPLAY_SAVE_ON_CPU="${GRPO_REPLAY_SAVE_ON_CPU:-0}"
# Pinned CPU memory can trigger host OOM / SIGKILL under multi-proc + replay; default off for stability.
GRPO_REPLAY_SAVE_ON_CPU_PIN="${GRPO_REPLAY_SAVE_ON_CPU_PIN:-0}"
# Replay activation checkpointing is NOT safe with current stateful KV caching.
# FORCE OFF here to avoid inheriting a stale env var that can crash or waste memory.
if [ -n "${GRPO_REPLAY_CHECKPOINTING:-}" ] && [ "${GRPO_REPLAY_CHECKPOINTING}" -ne 0 ]; then
  echo "[WARN] Ignoring GRPO_REPLAY_CHECKPOINTING=${GRPO_REPLAY_CHECKPOINTING}; forcing 0 (unsafe with trace_replay)."
fi
GRPO_REPLAY_CHECKPOINTING="0"
GRPO_PG_ONLY="${GRPO_PG_ONLY:-1}"
FREEZE_CHUNK_PREFIX="${FREEZE_CHUNK_PREFIX:-0}"
PARTIAL_FREEZE_PRINT_SUMMARY="${PARTIAL_FREEZE_PRINT_SUMMARY:-1}"
TRAIN_PN="${TRAIN_PN:-0.40M}"
TRAIN_EXP_NAME="${TRAIN_EXP_NAME:-GRPO_uavflow_mvp}"
TRAIN_DYNAMIC_SCALE_SCHEDULE="${TRAIN_DYNAMIC_SCALE_SCHEDULE:-infinity_elegant_clip4frames_v2_allpt}"
TRAIN_MASK_TYPE="${TRAIN_MASK_TYPE:-infinity_elegant_clip4frames_v2_allpt}"
TRAIN_VIDEO_FPS="${TRAIN_VIDEO_FPS:-16}"
TRAIN_VIDEO_FRAMES="${TRAIN_VIDEO_FRAMES:-49}"
GRPO_REQUIRE_OLD_LOGPROB="${GRPO_REQUIRE_OLD_LOGPROB:-1}"
export KEEP_LATEST_CKPTS

# Dataloader workers: too many workers across 8 ranks can cause host OOM / SIGKILL.
WORKERS="${WORKERS:-4}"
# Micro-batch per GPU (number of clip-samples per iteration on each rank).
VIDEO_BATCH_SIZE="${VIDEO_BATCH_SIZE:-1}"
# Gradient accumulation steps (keeps peak VRAM ~constant vs increasing VIDEO_BATCH_SIZE).
AC="${AC:-1}"
MAX_TRAIN_ITERS="${MAX_TRAIN_ITERS:-0}"
# Sequence packing bucket:
# Strict GRPO trace-replay is extremely memory intensive (per-sample replay w/ grad).
# Packing multiple samples into one step can multiply replay graphs and trigger CUDA OOM.
# Setting SEQ_PACK_BUCKET=1 effectively disables packing (one sample per batch) for stability.
SEQ_PACK_BUCKET="${SEQ_PACK_BUCKET:-1}"
# Sequence parallelism (sp_size).
# - trace_replay mode is NOT sequence-parallel safe in this codebase (rope mismatch / incorrect sharding).
# - trace_ce (single-forward teacher-forcing) IS sequence-parallel safe and can benefit from SP_SIZE>1.
if [ "${GRPO_NEW_LOGPROB_MODE}" = "trace_replay" ]; then
  if [ -n "${SP_SIZE:-}" ] && [ "${SP_SIZE}" -ne 1 ]; then
    echo "[WARN] Ignoring SP_SIZE=${SP_SIZE}; forcing SP_SIZE=1 for trace_replay."
  fi
  SP_SIZE="1"
else
  SP_SIZE="${SP_SIZE:-1}"
fi

if [ ! -f "${ROLL_JSONL}" ]; then
  echo "Missing ${ROLL_JSONL}."
  echo "REPLAY_META_DIR=${REPLAY_META_DIR}"
  echo "RUN_ID=${RUN_ID}"
  echo "REPLAY_FAST_OUT_DIR=${REPLAY_FAST_OUT_DIR}"
  echo "Available replay_meta_* under REPLAY_FAST_OUT_DIR:"
  ls -dt "${REPLAY_FAST_OUT_DIR}"/replay_meta_* 2>/dev/null | head -n 20 || true
  echo "Fix: set RUN_ID to an existing one, or set REPLAY_META_DIR to an existing replay_meta_* directory."
  exit 1
fi

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${REPO_ROOT}/action_aware_grpo/models/infinity}"
T5_PATH="${T5_PATH:-${CHECKPOINTS_DIR}/text_encoder/flan-t5-xl-official}"
VAE_PATH="${VAE_PATH:-${CHECKPOINTS_DIR}/infinitystar_videovae.pth}"
TORCHSHARD_RESUME_PATH="${TORCHSHARD_RESUME_PATH:-${CHECKPOINTS_DIR}/infinitystar_8b_480p_weights}"
RUSH_RESUME="${RUSH_RESUME:-}"
RESUME_CKPT="${RESUME_CKPT:-}"

if [ ! -d "${T5_PATH}" ]; then
  echo "Missing T5_PATH directory: ${T5_PATH}"
  echo "Set env T5_PATH to a valid local flan-t5-xl-official folder."
  exit 1
fi
if [ ! -f "${VAE_PATH}" ]; then
  echo "Missing VAE_PATH file: ${VAE_PATH}"
  echo "Set env VAE_PATH to a valid infinitystar_videovae.pth."
  exit 1
fi
if [ -z "${RUSH_RESUME}" ] && [ ! -d "${TORCHSHARD_RESUME_PATH}" ]; then
  echo "Missing TORCHSHARD_RESUME_PATH directory: ${TORCHSHARD_RESUME_PATH}"
  echo "Set env TORCHSHARD_RESUME_PATH to a valid Infinity torchshard folder."
  exit 1
fi
if [ -n "${RUSH_RESUME}" ] && [ ! -f "${RUSH_RESUME}" ]; then
  echo "Missing RUSH_RESUME file: ${RUSH_RESUME}"
  echo "Set env RUSH_RESUME to a valid global_step_xxx.pth."
  exit 1
fi
if [ -n "${RESUME_CKPT}" ] && [ ! -f "${RESUME_CKPT}" ]; then
  echo "Missing RESUME_CKPT file: ${RESUME_CKPT}"
  echo "Set env RESUME_CKPT to a valid global_step_xxx.pth."
  exit 1
fi

run_name="train_run_${RUN_ID:-latest}"
bed_path="${TRAIN_CKPT_ROOT}/${run_name}/ckpts"
token_cache_dir="${TRAIN_TOKEN_CACHE_ROOT}/${run_name}/token_cache"
local_out_path="${TRAIN_LOG_ROOT}/${run_name}/run"
mkdir -p "${local_out_path}" "${bed_path}" "${token_cache_dir}"

# Preferred resume path (matches proven finetune scripts):
# place global_step_xxx.pth under bed_path and let auto_resume(global_step_*) restore trainer/optimizer state.
if [ -n "${RESUME_CKPT}" ]; then
  resume_name="$(basename "${RESUME_CKPT}")"
  resume_dst="${bed_path}/${resume_name}"
  # IMPORTANT: if RESUME_CKPT is already inside bed_path (or exactly equals resume_dst),
  # NEVER rm/ln it. Otherwise we can delete a real checkpoint and create a self-referential
  # symlink loop (ELOOP: too many levels of symbolic links).
  abs_src="$("${PYTHON_BIN}" - "${RESUME_CKPT}" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)"
  abs_dst="$("${PYTHON_BIN}" - "${resume_dst}" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)"
  if [ "${abs_src}" = "${abs_dst}" ]; then
    echo "[WARN] RESUME_CKPT already in bed_path; skip linking: ${RESUME_CKPT}"
  else
    if [ -e "${resume_dst}" ] && [ ! -L "${resume_dst}" ]; then
      echo "[WARN] resume_dst exists and is not a symlink; keep existing file: ${resume_dst}"
    else
      rm -f "${resume_dst}"
      ln -s "${RESUME_CKPT}" "${resume_dst}"
      echo "Linked resume checkpoint: ${resume_dst} -> ${RESUME_CKPT}"
    fi
  fi

  # Avoid immediate exit when resume epoch is already larger than configured EPOCHS.
  # NOTE: If user explicitly set EPOCHS (e.g. EPOCHS=1 to run one extra epoch),
  # we should NOT override it here. train.py already computes start_ep from g_it/iters_train
  # and will adjust args.epoch to (start_ep + extra_epochs_after_resume) automatically.
  resume_epoch="$("${PYTHON_BIN}" - "${RESUME_CKPT}" <<'PY'
import sys, torch
p = sys.argv[1]
try:
    d = torch.load(p, map_location="cpu")
    print(int(d.get("epoch", -1)))
except Exception:
    print(-1)
PY
)"
  if [ "${EPOCHS_ENV_SET}" -eq 0 ]; then
    if [ "${resume_epoch}" -ge 0 ] && [ "${EPOCHS}" -le "${resume_epoch}" ]; then
      EPOCHS=$((resume_epoch + EXTRA_EPOCHS))
      echo "Adjusted EPOCHS to ${EPOCHS} (resume_epoch=${resume_epoch}, EXTRA_EPOCHS=${EXTRA_EPOCHS})"
    fi
  fi
fi

# Safety net: even when RESUME_CKPT is empty, auto_resume may still pick an
# existing global_step_*.pth under bed_path. Ensure EPOCHS is always > resume epoch.
resume_epoch_bed="$("${PYTHON_BIN}" - "${bed_path}" <<'PY'
import glob, os, torch, re, sys
bed = sys.argv[1]
best = ""
best_step = -1
for p in glob.glob(os.path.join(bed, "global_step_*.pth")):
    m = re.search(r"global_step_(\d+)", os.path.basename(p))
    if not m:
        continue
    s = int(m.group(1))
    if s > best_step:
        best_step = s
        best = p
if not best:
    print(-1)
else:
    try:
        d = torch.load(best, map_location="cpu")
        print(int(d.get("epoch", -1)))
    except Exception:
        print(-1)
PY
)"
if [ "${EPOCHS_ENV_SET}" -eq 0 ]; then
  if [ "${resume_epoch_bed}" -ge 0 ] && [ "${EPOCHS}" -le "${resume_epoch_bed}" ]; then
    EPOCHS=$((resume_epoch_bed + EXTRA_EPOCHS))
    echo "Adjusted EPOCHS to ${EPOCHS} from bed ckpt (resume_epoch=${resume_epoch_bed}, EXTRA_EPOCHS=${EXTRA_EPOCHS})"
  fi
fi

# CRITICAL: train.py computes start_ep from start_global_it (g_it) and *current* iters_train.
# When iters_train is small (e.g. 80), start_ep can be very large (e.g. 19000//80=237),
# and training will exit immediately if args.epoch <= start_ep, even if ckpt['epoch'] is smaller/different.
# We estimate iters_train from replay_meta line counts and adjust EPOCHS accordingly.
iters_train_est="$("${PYTHON_BIN}" - "${REPLAY_META_DIR}" "${NPROC_PER_NODE}" <<'PY'
import sys, glob, os
replay_dir = sys.argv[1]
paths = sorted(glob.glob(os.path.join(replay_dir, "part_*.jsonl")))
total = 0
for p in paths:
    try:
        with open(p, "rb") as f:
            for _ in f:
                total += 1
    except FileNotFoundError:
        pass
if total <= 0:
    print(-1)
else:
    # IMPORTANT: train.py uses len(dataset) as iters_train. For our jsonl-backed iterable dataset,
    # len(dataset) corresponds to TOTAL lines across all shards (not per-rank).
    print(int(total))
PY
)"
resume_g_it_bed="$("${PYTHON_BIN}" - "${bed_path}" <<'PY'
import glob, os, re, sys
import torch
bed = sys.argv[1]
best = ""
best_step = -1
for p in glob.glob(os.path.join(bed, "global_step_*.pth")):
    m = re.search(r"global_step_(\d+)", os.path.basename(p))
    if not m:
        continue
    s = int(m.group(1))
    if s > best_step:
        best_step = s
        best = p
if not best:
    print(-1)
else:
    try:
        d = torch.load(best, map_location="cpu")
        print(int(d.get("g_it", best_step)))
    except Exception:
        print(best_step)
PY
)"
if [ "${iters_train_est}" -gt 0 ] && [ "${resume_g_it_bed}" -ge 0 ]; then
  start_ep_est=$((resume_g_it_bed / iters_train_est))
  start_it_est=$((resume_g_it_bed % iters_train_est))
  echo "Estimated iters_train(total)=${iters_train_est}, start_ep=${start_ep_est}, start_it=${start_it_est} from g_it=${resume_g_it_bed}"
  if [ "${EPOCHS}" -le "${start_ep_est}" ]; then
    EPOCHS=$((start_ep_est + EXTRA_EPOCHS))
    echo "Adjusted EPOCHS to ${EPOCHS} from g_it/iters_train (start_ep=${start_ep_est}, EXTRA_EPOCHS=${EXTRA_EPOCHS})"
  fi
fi

TORCHRUN_ARGS=(--nproc_per_node="${NPROC_PER_NODE}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}")
if [ "${NNODES}" -gt 1 ]; then
  TORCHRUN_ARGS+=(
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
  )
fi

echo "[stageB] run_id=${RUN_ID:-latest} nnodes=${NNODES} node_rank=${NODE_RANK} nproc_per_node=${NPROC_PER_NODE} total_gpus=${GRPO_TOTAL_GPUS} sp_size=${SP_SIZE} ac=${AC} replay_meta_dir=${REPLAY_META_DIR}"

"${TORCHRUN_BIN}" "${TORCHRUN_ARGS[@]}" \
"${ROOT_DIR}/train.py" \
  --trainer_type "${TRAINER_TYPE}" \
  --shuffle_batches "${SHUFFLE_BATCHES}" \
  --hybrid_step_on_role "${HYBRID_STEP_ON_ROLE}" \
  --grpo_hybrid_rl_coef "${GRPO_HYBRID_RL_COEF}" \
  --local_out_path "${local_out_path}" \
  --bed "${bed_path}" \
  --data_path '' \
  --video_data_path "${REPLAY_META_DIR}" \
  --t5_path "${T5_PATH}" \
  --vae_type 64 \
  --videovae 10 \
  --vae_path "${VAE_PATH}" \
  --token_cache_dir "${token_cache_dir}" \
  --tlr "${TLR}" \
  --pn "${TRAIN_PN}" \
  --model infinity_qwen8b \
  --zero "${ZERO_STAGE}" \
  --project_name infinity \
  --exp_name "${TRAIN_EXP_NAME}" \
  --checkpoint_type torch \
  --enable_checkpointing full-block \
  --save_model_iters_freq "${SAVE_MODEL_ITERS_FREQ}" \
  --video_fps "${TRAIN_VIDEO_FPS}" \
  --video_frames "${TRAIN_VIDEO_FRAMES}" \
  --video_batch_size "${VIDEO_BATCH_SIZE}" \
  --ac "${AC}" \
  --max_train_iters "${MAX_TRAIN_ITERS}" \
  --short_cap_prob "${SHORT_CAP_PROB}" \
  --use_streaming_dataset 1 \
  --iterable_data_buffersize 1000 \
  --workers "${WORKERS}" \
  --seq_pack_bucket "${SEQ_PACK_BUCKET}" \
  --enable_dynamic_length_prompt "${ENABLE_DYNAMIC_LENGTH_PROMPT}" \
  --reweight_loss_by_scale "${REWEIGHT_LOSS_BY_SCALE}" \
  --dynamic_scale_schedule "${TRAIN_DYNAMIC_SCALE_SCHEDULE}" \
  --mask_type "${TRAIN_MASK_TYPE}" \
  --frames_inner_clip 4 \
  --context_from_largest_no "${CONTEXT_FROM_LARGEST_NO}" \
  --image_scale_repetition "${IMAGE_SCALE_REPETITION}" \
  --video_scale_repetition "${VIDEO_SCALE_REPETITION}" \
  --append_duration2caption "${APPEND_DURATION2CAPTION}" \
  --wp_it "${WP_IT}" \
  --use_two_stage_lfq "${USE_TWO_STAGE_LFQ}" \
  --semantic_scales "${SEMANTIC_SCALES}" \
  --detail_scale_min_tokens "${DETAIL_SCALE_MIN_TOKENS}" \
  --use_flex_attn True \
  --train_with_var_seq_len 1 \
  --train_max_token_len "${TRAIN_MAX_TOKEN_LEN}" \
  --allow_less_one_elem_in_seq "${ALLOW_LESS_ONE_ELEM_IN_SEQ}" \
  --sp_size "${SP_SIZE}" \
  --extra_epochs_after_resume "${EXTRA_EPOCHS}" \
  --epoch "${EPOCHS}" \
  --grpo_adv_clip "${GRPO_ADV_CLIP}" \
  --grpo_ratio_eps "${GRPO_RATIO_EPS}" \
  --grpo_kl_beta "${GRPO_KL_BETA}" \
  --grpo_lambda_act "${GRPO_LAMBDA_ACT}" \
  --grpo_lambda_task "${GRPO_LAMBDA_TASK}" \
  --grpo_lambda_ce "${GRPO_LAMBDA_CE}" \
  --grpo_alpha_decay 0.9 \
  --grpo_require_old_logprob "${GRPO_REQUIRE_OLD_LOGPROB}" \
  --grpo_new_logprob_mode "${GRPO_NEW_LOGPROB_MODE}" \
  --grpo_block_dtype "${GRPO_BLOCK_DTYPE}" \
  --grpo_fsdp_compute_dtype "${GRPO_FSDP_COMPUTE_DTYPE}" \
  --grpo_fsdp_cpu_offload "${GRPO_FSDP_CPU_OFFLOAD}" \
  --grpo_chunk_gradient_diagnostics "${GRPO_CHUNK_GRADIENT_DIAGNOSTICS}" \
  --grpo_require_nonnegative_adv "${GRPO_REQUIRE_NONNEGATIVE_ADV}" \
  --grpo_log_weight_stats "${GRPO_LOG_WEIGHT_STATS}" \
  --grpo_replay_save_on_cpu "${GRPO_REPLAY_SAVE_ON_CPU}" \
  --grpo_replay_save_on_cpu_pin "${GRPO_REPLAY_SAVE_ON_CPU_PIN}" \
  --grpo_replay_checkpointing "${GRPO_REPLAY_CHECKPOINTING}" \
  --grpo_pg_only "${GRPO_PG_ONLY}" \
  --grpo_weight_mode "${GRPO_WEIGHT_MODE}" \
  --grpo_aux_sft_coef "${GRPO_AUX_SFT_COEF}" \
  --grad_clip "${GRAD_CLIP}" \
  --freeze_chunk_prefix "${FREEZE_CHUNK_PREFIX}" \
  --partial_freeze_print_summary "${PARTIAL_FREEZE_PRINT_SUMMARY}" \
  --rush_resume "${RUSH_RESUME}" \
  --torchshard_resume "${TORCHSHARD_RESUME_PATH}"

