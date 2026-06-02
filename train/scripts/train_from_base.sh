#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

pick_first_existing_path() {
  local candidate
  for candidate in "$@"; do
    if [[ -e "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  printf '%s\n' "$1"
}

require_existing_path() {
  local label="$1"
  local path="$2"
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] ${label} not found: ${path}" >&2
    exit 1
  fi
}

unset NCCL_NET_PLUGIN
unset NCCL_FASTRAK_ENABLE
unset NCCL_FASTRAK_USE_SNAP
unset NCCL_FASTRAK_NUM_FLOWS
unset NCCL_FASTRAK_*
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET_DISABLE="${NCCL_NET_DISABLE:-1}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-NONE}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

TORCHRUN_RDZV_READ_TIMEOUT="${TORCHRUN_RDZV_READ_TIMEOUT:-600}"
ARNOLD_ID="${ARNOLD_ID:-0}"
ARNOLD_WORKER_NUM="${ARNOLD_WORKER_NUM:-1}"
ARNOLD_WORKER_GPU="${ARNOLD_WORKER_GPU:-4}"
ARNOLD_WORKER_0_HOST="${ARNOLD_WORKER_0_HOST:-localhost}"
ARNOLD_WORKER_0_PORT="${ARNOLD_WORKER_0_PORT:-9591}"
PORT="${ARNOLD_WORKER_0_PORT%%,*}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Worldmodel python package lives under Worldmodel/infinity/.
# Add the directory containing the `infinity/` package to PYTHONPATH.
<<<<<<< Updated upstream
=======
# Use the runtime copy of the InfinityStar package (contains `infinity/dataset`).
>>>>>>> Stashed changes
export PYTHONPATH="${REPO_ROOT}/../Worldmodel/runtime${PYTHONPATH:+:${PYTHONPATH}}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
unset MKL_NUM_THREADS || true
unset OPENBLAS_NUM_THREADS || true

cd "${REPO_ROOT}"

if command -v wandb >/dev/null 2>&1; then
  wandb offline >/dev/null 2>&1 || true
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="python3"
fi

EXP_NAME="${EXP_NAME:-finetune_uavflow_from_base}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
SAVE_FREQ_ITERS="${SAVE_FREQ_ITERS:-1000}"
TLR="${TLR:-1e-5}"
VIDEO_FPS="${VIDEO_FPS:-16}"
VIDEO_FRAMES="${VIDEO_FRAMES:-49}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${REPO_ROOT}/checkpoints}"
T5_PATH="${T5_PATH:-${CHECKPOINTS_DIR}/text_encoder/flan-t5-xl-official}"
VAE_PATH="${VAE_PATH:-${CHECKPOINTS_DIR}/infinitystar_videovae.pth}"
TORCHSHARD_RESUME_PATH="${TORCHSHARD_RESUME_PATH:-${CHECKPOINTS_DIR}/infinitystar_8b_480p_weights}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
VIDEO_DATA_PATH="${VIDEO_DATA_PATH:-$(pick_first_existing_path \
  "${DATA_ROOT}/uavflow_49f_from_40_60_split8_jsonl" \
  "${DATA_ROOT}/uavflow_40_60_split8_jsonl" \
  "${DATA_ROOT}/split8_jsonl")}"

require_existing_path "video data path" "${VIDEO_DATA_PATH}"
require_existing_path "T5 path" "${T5_PATH}"
require_existing_path "VAE checkpoint" "${VAE_PATH}"
require_existing_path "base torch shard weights" "${TORCHSHARD_RESUME_PATH}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
LOCAL_OUT_PATH="${LOCAL_OUT_PATH:-${OUTPUT_ROOT}/run_logs/${EXP_NAME}}"
BED_PATH="${BED_PATH:-${OUTPUT_ROOT}/checkpoints/${EXP_NAME}}"
TOKEN_CACHE_DIR="${TOKEN_CACHE_DIR:-${OUTPUT_ROOT}/cache/${EXP_NAME}}"
mkdir -p "${LOCAL_OUT_PATH}" "${BED_PATH}" "${TOKEN_CACHE_DIR}"

noise_apply_strength=()
for ((i = 0; i < 200; i++)); do
  noise_apply_strength+=("0.3")
done
NOISE_APPLY_STRENGTH_STR="$(IFS=,; echo "${noise_apply_strength[*]}")"

"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${ARNOLD_WORKER_GPU}" \
  --nnodes="${ARNOLD_WORKER_NUM}" \
  --master_addr="${ARNOLD_WORKER_0_HOST}" \
  --node_rank="${ARNOLD_ID}" \
  --master_port="${PORT}" \
  --rdzv_conf="read_timeout=${TORCHRUN_RDZV_READ_TIMEOUT}" \
  train.py \
  --local_out_path "${LOCAL_OUT_PATH}" \
  --bed="${BED_PATH}" \
  --data_path='' \
  --video_data_path="${VIDEO_DATA_PATH}" \
  --t5_path="${T5_PATH}" \
  --vae_type=64 \
  --videovae=10 \
  --vae_path="${VAE_PATH}" \
  --token_cache_dir="${TOKEN_CACHE_DIR}" \
  --tlr="${TLR}" \
  --pn 0.40M \
  --model=infinity_qwen8b \
  --project_name=infinity \
  --exp_name="${EXP_NAME}" \
  --checkpoint_type='torch' \
  --enable_checkpointing=full-block \
  --video_fps="${VIDEO_FPS}" \
  --video_frames="${VIDEO_FRAMES}" \
  --short_cap_prob 0.3 \
  --use_streaming_dataset 1 \
  --iterable_data_buffersize 1000 \
  --enable_dynamic_length_prompt 1 \
  --reweight_loss_by_scale 4 \
  --zero=3 \
  --save_model_iters_freq "${SAVE_FREQ_ITERS}" \
  --epoch "${TRAIN_EPOCHS}" \
  --noise_apply_strength="${NOISE_APPLY_STRENGTH_STR}" \
  --dynamic_scale_schedule=infinity_elegant_clip4frames_v2_allpt \
  --mask_type=infinity_elegant_clip4frames_v2_allpt \
  --frames_inner_clip=4 \
  --context_from_largest_no=0 \
  --use_flex_attn=True \
  --use_vae_token_cache=1 \
  --cache_check_mode=0 \
  --allow_online_vae_feature_extraction=1 \
  --train_with_var_seq_len=1 \
  --train_max_token_len=20480 \
  --video_var_len_prob='[20, 15, 10, 4, 2, 1, 10, 3, 2, 10, 3, 2]' \
  --drop_long_video=0 \
  --image_scale_repetition='[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]' \
  --video_scale_repetition='[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]' \
  --append_duration2caption=1 \
  --wp_it=0 \
  --use_two_stage_lfq=1 \
  --semantic_scale_dim=16 \
  --detail_scale_min_tokens=350 \
  --semantic_scales=11 \
  --allow_less_one_elem_in_seq=1 \
  --use_feat_proj=2 \
  --drop_720p_last_scale=1 \
  --twoclip_alternatingtraining=0 \
  --enable_hybrid_shard=0 \
  --restrict_data_size=-1 \
  --sp_size=1 \
  --torchshard_resume="${TORCHSHARD_RESUME_PATH}"
