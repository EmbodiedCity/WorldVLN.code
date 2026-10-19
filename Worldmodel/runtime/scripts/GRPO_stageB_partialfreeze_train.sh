#!/usr/bin/env bash
set -euo pipefail
set -x

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PACKAGE_ROOT}/outputs}"
. "${ROOT_DIR}/scripts/GRPO_cluster_env.sh"

for arg in "$@"; do
  case "${arg}" in
    *=*) export "${arg}" ;;
  esac
done

AC_ENV_SET=0
if [ "${AC+x}" = "x" ]; then
  AC_ENV_SET=1
fi
SP_SIZE_ENV_SET=0
if [ "${SP_SIZE+x}" = "x" ]; then
  SP_SIZE_ENV_SET=1
fi

PARTIAL_FREEZE_MODE="${PARTIAL_FREEZE_MODE:-main}"
case "${PARTIAL_FREEZE_MODE}" in
  smoke)
    default_freeze_chunk_prefix=5
    default_workers=2
    default_kl_beta=0.0
    default_ratio_eps=0.02
    default_adv_clip=1.0
    default_aux_sft_coef=0.0
    default_max_train_iters=1200
    default_save_model_iters_freq=150
    default_tlr=1e-6
    default_grad_clip=0.5
    default_train_max_token_len=20480
    default_video_batch_size=1
    default_ac=4
    ;;
  main)
    default_freeze_chunk_prefix=4
    default_workers=4
    default_kl_beta=0.0
    default_ratio_eps=0.15
    default_adv_clip=5.0
    default_aux_sft_coef=0.0
    default_max_train_iters=0
    default_save_model_iters_freq=300
    default_tlr=5e-6
    default_grad_clip=5.0
    default_train_max_token_len=20480
    default_video_batch_size=1
    default_ac=1
    ;;
  *)
    echo "[FATAL] Unsupported PARTIAL_FREEZE_MODE=${PARTIAL_FREEZE_MODE}. Use smoke or main."
    exit 1
    ;;
esac

export FREEZE_CHUNK_PREFIX="${FREEZE_CHUNK_PREFIX:-${default_freeze_chunk_prefix}}"
export PARTIAL_FREEZE_PRINT_SUMMARY="${PARTIAL_FREEZE_PRINT_SUMMARY:-1}"
export INFINITY_GRPO_TRAIN_LAST_N_BLOCKS="${INFINITY_GRPO_TRAIN_LAST_N_BLOCKS:-0}"
export GRPO_NEW_LOGPROB_MODE="${GRPO_NEW_LOGPROB_MODE:-trace_replay}"
export GRPO_KL_BETA="${GRPO_KL_BETA:-${default_kl_beta}}"
export GRPO_RATIO_EPS="${GRPO_RATIO_EPS:-${default_ratio_eps}}"
export GRPO_ADV_CLIP="${GRPO_ADV_CLIP:-${default_adv_clip}}"
export GRPO_AUX_SFT_COEF="${GRPO_AUX_SFT_COEF:-${default_aux_sft_coef}}"
export GRPO_REQUIRE_NONNEGATIVE_ADV="${GRPO_REQUIRE_NONNEGATIVE_ADV:-1}"
export TLR="${TLR:-${default_tlr}}"
export GRAD_CLIP="${GRAD_CLIP:-${default_grad_clip}}"
export WORKERS="${WORKERS:-${default_workers}}"
export VIDEO_BATCH_SIZE="${VIDEO_BATCH_SIZE:-${default_video_batch_size}}"
export TRAIN_MAX_TOKEN_LEN="${TRAIN_MAX_TOKEN_LEN:-${default_train_max_token_len}}"
export ZERO_STAGE="${ZERO_STAGE:-3}"
export MAX_TRAIN_ITERS="${MAX_TRAIN_ITERS:-${default_max_train_iters}}"
export SAVE_MODEL_ITERS_FREQ="${SAVE_MODEL_ITERS_FREQ:-${default_save_model_iters_freq}}"
export RUN_ID="${RUN_ID:-partialfreeze_${PARTIAL_FREEZE_MODE}_$(date +%Y%m%d_%H%M%S)}"
export FAST_OUT_DIR="${FAST_OUT_DIR:-${OUTPUT_ROOT}/GRPO_data_fast}"
export TRAIN_CKPT_ROOT="${TRAIN_CKPT_ROOT:-${OUTPUT_ROOT}/checkpoints_partialfreeze}"
export TRAIN_LOG_ROOT="${TRAIN_LOG_ROOT:-${OUTPUT_ROOT}/train_logs}"
export TRAIN_TOKEN_CACHE_ROOT="${TRAIN_TOKEN_CACHE_ROOT:-${OUTPUT_ROOT}/token_cache}"

grpo_resolve_cluster_env
if [ "${SP_SIZE_ENV_SET}" -eq 0 ]; then
  export SP_SIZE="$(grpo_recommend_stageb_sp_size "${GRPO_TOTAL_GPUS}")"
else
  export SP_SIZE="${SP_SIZE}"
fi
if [ "${AC_ENV_SET}" -eq 0 ]; then
  export AC="$(grpo_recommend_stageb_ac "${default_ac}" "${GRPO_TOTAL_GPUS}" "${SP_SIZE}")"
else
  export AC="${AC}"
fi
if [ "${GRPO_NEW_LOGPROB_MODE}" != "trace_replay" ] && [ "$((GRPO_TOTAL_GPUS % SP_SIZE))" -ne 0 ]; then
  echo "[FATAL] total_gpus=${GRPO_TOTAL_GPUS} must be divisible by SP_SIZE=${SP_SIZE}"
  exit 1
fi
echo "[stageB_partialfreeze] mode=${PARTIAL_FREEZE_MODE} total_gpus=${GRPO_TOTAL_GPUS} nnodes=${NNODES} nproc_per_node=${NPROC_PER_NODE} sp_size=${SP_SIZE} ac=${AC} video_batch_size=${VIDEO_BATCH_SIZE}"

bash "${ROOT_DIR}/scripts/GRPO_stageB_train.sh"
