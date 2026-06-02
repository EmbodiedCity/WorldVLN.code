#!/usr/bin/env bash
set -euo pipefail
set -x

# Usage:
#   bash scripts/GRPO_stageA_collect.sh
#
# This stage builds rollout task jsonl from source json and prepares replay input.

for arg in "$@"; do
  case "${arg}" in
    *=*) export "${arg}" ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PACKAGE_ROOT}/.." && pwd)"
. "${ROOT_DIR}/scripts/GRPO_cluster_env.sh"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PACKAGE_ROOT}/outputs}"
# Default dataset; can be overridden by env SRC_JSON=...
DEFAULT_SRC_JSON="${PACKAGE_ROOT}/data/reference4x_then_extra_uavflow_sim.json"
SRC_JSON="${SRC_JSON:-}"
if [ -z "${SRC_JSON}" ] && [ -f "${DEFAULT_SRC_JSON}" ]; then
  SRC_JSON="${DEFAULT_SRC_JSON}"
fi
# Output roots default to package-local paths.
RL_CACHE_ROOT="${RL_CACHE_ROOT:-${OUTPUT_ROOT}/rlcache}"
FAST_OUT_DIR="${FAST_OUT_DIR:-${OUTPUT_ROOT}/GRPO_data_fast}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
STAGEA_LOCAL_LOG_ROOT="${STAGEA_LOCAL_LOG_ROOT:-${OUTPUT_ROOT}/stagea_logs}"
STAGEA_LOCAL_LOG_FILE_ENV_SET=0
if [ "${STAGEA_LOCAL_LOG_FILE+x}" = "x" ]; then
  STAGEA_LOCAL_LOG_FILE_ENV_SET=1
fi
STAGEA_LOCAL_LOG_ENABLE="${STAGEA_LOCAL_LOG_ENABLE:-1}"
STAGEA_EXPECT_TOP_N="${STAGEA_EXPECT_TOP_N:-}"
OUT_DIR="${RL_CACHE_ROOT}/${RUN_ID}"
TASK_JSONL="${OUT_DIR}/rollout_tasks.jsonl"
CAND_JSONL="${OUT_DIR}/rollout_candidates.jsonl"
REWARD_JSONL="${OUT_DIR}/rollout_tasks_rewarded.jsonl"
REPLAY_JSONL="${OUT_DIR}/rollout_replay.jsonl"
TRAJ_DIR="${OUT_DIR}/trajectories"
FAILED_JSONL="${OUT_DIR}/rollout_failed.jsonl"
TIMING_JSONL="${OUT_DIR}/rollout_timing.jsonl"

REPLAY_META_DIR="${FAST_OUT_DIR}/replay_meta_${RUN_ID}"
MANIFEST_JSON="${FAST_OUT_DIR}/manifest_${RUN_ID}.json"
SUMMARY_JSON="${FAST_OUT_DIR}/summary_${RUN_ID}.json"
TOP_N="${TOP_N:-0}"
K_CAND="${K_CAND:-8}"
N_SHARDS_ENV_SET=0
if [ "${N_SHARDS+x}" = "x" ]; then
  N_SHARDS_ENV_SET=1
fi
N_SHARDS="${N_SHARDS:-}"
USE_REAL_ROLLOUT="${USE_REAL_ROLLOUT:-1}"
STAGEA_NPROC_ENV_SET=0
if [ "${STAGEA_NPROC+x}" = "x" ]; then
  STAGEA_NPROC_ENV_SET=1
fi
STAGEA_NPROC="${STAGEA_NPROC:-}"
STAGEA_GPU_IDS="${STAGEA_GPU_IDS:-}"
STAGEA_PROGRESS_EVERY_N="${STAGEA_PROGRESS_EVERY_N:-10}"
STAGEA_TASK_SEED_STRIDE="${STAGEA_TASK_SEED_STRIDE:-1000003}"
STAGEA_CANDIDATE_SEED_STRIDE="${STAGEA_CANDIDATE_SEED_STRIDE:-65537}"
LAMBDA_ACT="${LAMBDA_ACT:-0.3}"
LAMBDA_TASK="${LAMBDA_TASK:-1.3}"
LAMBDA_CE="${LAMBDA_CE:-0.03}"
ACT_REWARD_MODE="${ACT_REWARD_MODE:-zscore_exp}"   # zscore_exp | minstd_exp | inv1p
ALPHA_XYZ="${ALPHA_XYZ:-1.0}"
ALPHA_YAW="${ALPHA_YAW:-1.0}"
ALPHA_ALL6="${ALPHA_ALL6:-0.2}"
# Task success thresholds:
# - Clip-level thresholds are used to compute per-clip succ for clip-GRPO task reward.
# - Traj-level thresholds are used only for whole-trajectory succ diagnostics / legacy traj-mode.
CLIP_TASK_POS_THRESH_M="${CLIP_TASK_POS_THRESH_M:-0.6}"
CLIP_TASK_YAW_THRESH_DEG="${CLIP_TASK_YAW_THRESH_DEG:-2.5}"
TRAJ_TASK_POS_THRESH_M="${TRAJ_TASK_POS_THRESH_M:-3.0}"
TRAJ_TASK_YAW_THRESH_DEG="${TRAJ_TASK_YAW_THRESH_DEG:-10.0}"
TASK_POS_SCALE_M="${TASK_POS_SCALE_M:-2.0}"
TASK_YAW_SCALE_DEG="${TASK_YAW_SCALE_DEG:-10.0}"
TASK_POS_WEIGHT="${TASK_POS_WEIGHT:-1.0}"
TASK_YAW_WEIGHT="${TASK_YAW_WEIGHT:-1.0}"
TASK_ENABLE_SUCCESS_BONUS="${TASK_ENABLE_SUCCESS_BONUS:-1}"
TASK_DENSE_WEIGHT="${TASK_DENSE_WEIGHT:-0.85}"
TASK_SUCCESS_WEIGHT="${TASK_SUCCESS_WEIGHT:-0.15}"
TASK_REWARD_MODE="${TASK_REWARD_MODE:-raw_dense}"
# StageA strictness for postprocessing:
# - REQUIRE_ALL_TRAJECTORIES=1 will fail-fast if any rollout is missing trajectory.json
# - Default 0: skip missing rollouts (they are already logged in rollout_failed.jsonl)
REQUIRE_ALL_TRAJECTORIES="${REQUIRE_ALL_TRAJECTORIES:-0}"
REQUIRE_OLD_LOGPROB="${REQUIRE_OLD_LOGPROB:-1}"
# Real rollout checkpoints/configs
INFINITY_API_PY="${INFINITY_API_PY:-${REPO_ROOT}/action_aware_grpo/grpo_server.py}"
INFINITY_SERVER_CONFIG="${INFINITY_SERVER_CONFIG:-${REPO_ROOT}/action_aware_grpo/config.json}"
INFINITY_CKPT="${INFINITY_CKPT:-}"
INFINITY_REPO_ROOT="${INFINITY_REPO_ROOT:-${ROOT_DIR}}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${REPO_ROOT}/action_aware_grpo/models/infinity}"
STAGEA_T5_PATH="${T5_PATH:-${CHECKPOINTS_DIR}/text_encoder/flan-t5-xl-official}"
STAGEA_VAE_PATH="${VAE_PATH:-${CHECKPOINTS_DIR}/infinitystar_videovae.pth}"
ACTIONHEAD_CKPT="${ACTIONHEAD_CKPT:-}"
ACTIONHEAD_RUN_CONFIG="${ACTIONHEAD_RUN_CONFIG:-}"
ACTIONHEAD_REPO_ROOT="${ACTIONHEAD_REPO_ROOT:-${REPO_ROOT}/Worldmodel/action_decoder/actionhead_runtime}"
ROLLOUT_TASK_BUILDER_PY="${ROLLOUT_TASK_BUILDER_PY:-${ROOT_DIR}/tools/GRPO/build_rollout_tasks.py}"
ROLLOUT_TRAJECTORY_PY="${ROLLOUT_TRAJECTORY_PY:-${ROOT_DIR}/tools/GRPO/generate_candidate_trajectories_real.py}"
ROLLOUT_REWARD_PY="${ROLLOUT_REWARD_PY:-${ROOT_DIR}/tools/GRPO/reward_uavflow.py}"
ROLLOUT_REWARD_LABEL="${ROLLOUT_REWARD_LABEL:-uavflow}"
INFINITY_REQUIRE_TGT_HW="${INFINITY_REQUIRE_TGT_HW:-}"
ROLLOUT_REQUIRE_TRACE_FILES="${ROLLOUT_REQUIRE_TRACE_FILES:-}"
UAVFLOW_STAGEA_ROLLOUT_BACKEND="${UAVFLOW_STAGEA_ROLLOUT_BACKEND:-remote_sim}"
UAVFLOW_SIMULATOR_BASE_URL="${UAVFLOW_SIMULATOR_BASE_URL:-http://127.0.0.1:8765}"
UAVFLOW_SIMULATOR_TIMEOUT_S="${UAVFLOW_SIMULATOR_TIMEOUT_S:-120}"
UAVFLOW_TASK_JSON_ROOT="${UAVFLOW_TASK_JSON_ROOT:-}"
DEFAULT_UAVFLOW_TASK_JSON_ROOT="${PACKAGE_ROOT}/data/uavflow_tasks/select_from_train_jsons"
if [ -z "${UAVFLOW_TASK_JSON_ROOT}" ] && [ -d "${DEFAULT_UAVFLOW_TASK_JSON_ROOT}" ]; then
  UAVFLOW_TASK_JSON_ROOT="${DEFAULT_UAVFLOW_TASK_JSON_ROOT}"
fi

if [ -z "${SRC_JSON}" ]; then
  echo "[stageA][fatal] SRC_JSON is required. Point it to a UAV-Flow rollout manifest JSON."
  exit 1
fi
if [ ! -f "${SRC_JSON}" ]; then
  echo "[stageA][fatal] missing SRC_JSON: ${SRC_JSON}"
  exit 1
fi
if [ ! -f "${INFINITY_API_PY}" ]; then
  echo "[stageA][fatal] missing INFINITY_API_PY: ${INFINITY_API_PY}"
  exit 1
fi
if [ ! -f "${INFINITY_SERVER_CONFIG}" ]; then
  echo "[stageA][fatal] missing INFINITY_SERVER_CONFIG: ${INFINITY_SERVER_CONFIG}"
  exit 1
fi
if [ -z "${INFINITY_CKPT}" ]; then
  echo "[stageA][fatal] INFINITY_CKPT is required. Export it to a local InfinityStar checkpoint."
  exit 1
fi
if [ ! -f "${INFINITY_CKPT}" ]; then
  echo "[stageA][fatal] missing INFINITY_CKPT: ${INFINITY_CKPT}"
  exit 1
fi
if [ ! -d "${INFINITY_REPO_ROOT}" ]; then
  echo "[stageA][fatal] missing INFINITY_REPO_ROOT: ${INFINITY_REPO_ROOT}"
  exit 1
fi
if [ ! -d "${CHECKPOINTS_DIR}" ]; then
  echo "[stageA][fatal] missing CHECKPOINTS_DIR: ${CHECKPOINTS_DIR}"
  echo "[stageA][fatal] StageA local inference needs the shared InfinityStar assets under CHECKPOINTS_DIR."
  exit 1
fi
if [ ! -d "${STAGEA_T5_PATH}" ]; then
  echo "[stageA][fatal] missing T5_PATH for StageA local inference: ${STAGEA_T5_PATH}"
  exit 1
fi
if [ ! -f "${STAGEA_VAE_PATH}" ]; then
  echo "[stageA][fatal] missing VAE_PATH for StageA local inference: ${STAGEA_VAE_PATH}"
  exit 1
fi
if [ -z "${ACTIONHEAD_CKPT}" ]; then
  echo "[stageA][fatal] ACTIONHEAD_CKPT is required. Export it to a local action-head checkpoint."
  exit 1
fi
if [ ! -f "${ACTIONHEAD_CKPT}" ]; then
  echo "[stageA][fatal] missing ACTIONHEAD_CKPT: ${ACTIONHEAD_CKPT}"
  exit 1
fi
if [ -z "${ACTIONHEAD_RUN_CONFIG}" ]; then
  echo "[stageA][fatal] ACTIONHEAD_RUN_CONFIG is required. Export it to the matching run_config.json."
  exit 1
fi
if [ ! -f "${ACTIONHEAD_RUN_CONFIG}" ]; then
  echo "[stageA][fatal] missing ACTIONHEAD_RUN_CONFIG: ${ACTIONHEAD_RUN_CONFIG}"
  exit 1
fi
if [ ! -d "${ACTIONHEAD_REPO_ROOT}" ]; then
  echo "[stageA][fatal] missing ACTIONHEAD_REPO_ROOT: ${ACTIONHEAD_REPO_ROOT}"
  exit 1
fi
if [ "${UAVFLOW_STAGEA_ROLLOUT_BACKEND}" != "remote_sim" ]; then
  echo "[stageA][fatal] unsupported UAVFLOW_STAGEA_ROLLOUT_BACKEND=${UAVFLOW_STAGEA_ROLLOUT_BACKEND}"
  echo "[stageA][fatal] open-source action_aware_grpo only keeps remote_sim for rollout."
  exit 1
fi
if [ -z "${UAVFLOW_TASK_JSON_ROOT}" ]; then
  echo "[stageA][fatal] UAVFLOW_TASK_JSON_ROOT is required when UAVFLOW_STAGEA_ROLLOUT_BACKEND=remote_sim."
  exit 1
fi
if [ ! -d "${UAVFLOW_TASK_JSON_ROOT}" ]; then
  echo "[stageA][fatal] missing UAVFLOW_TASK_JSON_ROOT: ${UAVFLOW_TASK_JSON_ROOT}"
  exit 1
fi

# StageB(trace_ce) compatibility:
# Use teacher-forcing single-forward logprob to populate old_logprob in StageA outputs.
INFINITY_STAGEA_OLD_LOGPROB_MODE="${INFINITY_STAGEA_OLD_LOGPROB_MODE:-trace_ce}"   # trace_ce | sampling
INFINITY_GRPO_TRACE_CE_TMAX="${INFINITY_GRPO_TRACE_CE_TMAX:-20480}"
export INFINITY_STAGEA_OLD_LOGPROB_MODE INFINITY_GRPO_TRACE_CE_TMAX
#
# IMPORTANT:
# If trace_ce old_logprob fails and we silently fall back to sampling logprob, StageB ratio/KL becomes inconsistent
# and can easily collapse the video quality (checkerboard/garbled frames). Therefore we default to strict mode:
# make StageA treat trace_ce failure as a rollout failure (so the retry loop kicks in).
INFINITY_STAGEA_OLD_LOGPROB_STRICT="${INFINITY_STAGEA_OLD_LOGPROB_STRICT:-1}"
export INFINITY_STAGEA_OLD_LOGPROB_STRICT

# When using trace_ce logprob, strongly recommend cfg=1 and tau=1 to keep the behavior policy
# consistent with the teacher-forcing scoring distribution.
if [ "${INFINITY_STAGEA_OLD_LOGPROB_MODE}" = "trace_ce" ]; then
  export INFINITY_CFG="${INFINITY_CFG:-1.0}"
  export INFINITY_TAU_IMAGE="${INFINITY_TAU_IMAGE:-1.0}"
  export INFINITY_TAU_VIDEO="${INFINITY_TAU_VIDEO:-1.0}"
  # For strict on-policy consistency, sample from the full distribution (no top-k/top-p truncation).
  export INFINITY_TOP_K="${INFINITY_TOP_K:-0}"
  export INFINITY_TOP_P="${INFINITY_TOP_P:-1.0}"
fi

grpo_resolve_cluster_env
if [ "${STAGEA_NPROC_ENV_SET}" -eq 0 ]; then
  STAGEA_NPROC="${NPROC_PER_NODE}"
fi
if [ -z "${STAGEA_GPU_IDS}" ]; then
  STAGEA_GPU_IDS="${GRPO_LOCAL_GPU_IDS}"
fi
if [ "${N_SHARDS_ENV_SET}" -eq 0 ]; then
  N_SHARDS="$((STAGEA_NPROC * NNODES))"
fi
if [ "${STAGEA_NPROC}" -le 0 ]; then
  echo "[stageA][fatal] STAGEA_NPROC must be positive, got ${STAGEA_NPROC}"
  exit 1
fi
if [ "$((STAGEA_NPROC * NNODES))" -ne "${N_SHARDS}" ]; then
  echo "[stageA][fatal] require N_SHARDS == STAGEA_NPROC * NNODES to keep one shard per local worker."
  echo "[stageA][fatal] N_SHARDS=${N_SHARDS} STAGEA_NPROC=${STAGEA_NPROC} NNODES=${NNODES}"
  exit 1
fi

STAGEA_SYNC_ROOT="${STAGEA_SYNC_ROOT:-${OUT_DIR}/_stagea_sync}"
STAGEA_SYNC_TIMEOUT="${STAGEA_SYNC_TIMEOUT:-7200}"
STAGEA_READY_FILE="${STAGEA_SYNC_ROOT}/inputs.ready"
STAGEA_COMPLETE_FILE="${STAGEA_SYNC_ROOT}/stagea.complete"
STAGEA_FAIL_MARKER="${STAGEA_SYNC_ROOT}/${GRPO_NODE_TAG}.fail"
STAGEA_FAIL_GLOB="${STAGEA_SYNC_ROOT}/node_*.fail"
STAGEA_NODE_DONE_FILE="${STAGEA_SYNC_ROOT}/rollout_node_${NODE_RANK}.done"
if [ "${STAGEA_LOCAL_LOG_FILE_ENV_SET}" -eq 0 ]; then
  STAGEA_LOCAL_LOG_FILE="${STAGEA_LOCAL_LOG_ROOT}/${GRPO_NODE_TAG}/stageA_${RUN_ID}.log"
fi

export SRC_JSON INPUT_JSON TOP_N TASK_JSONL CAND_JSONL REWARD_JSONL REPLAY_JSONL REPLAY_META_DIR N_SHARDS FAST_OUT_DIR OUT_DIR RUN_ID RL_CACHE_ROOT MANIFEST_JSON SUMMARY_JSON USE_REAL_ROLLOUT FAILED_JSONL TIMING_JSONL STAGEA_NPROC
export LAMBDA_ACT LAMBDA_TASK LAMBDA_CE
export TASK_ENABLE_SUCCESS_BONUS TASK_DENSE_WEIGHT TASK_SUCCESS_WEIGHT
export K_CAND STAGEA_TASK_SEED_STRIDE STAGEA_CANDIDATE_SEED_STRIDE
export STAGEA_LOCAL_LOG_ROOT STAGEA_LOCAL_LOG_FILE STAGEA_LOCAL_LOG_ENABLE STAGEA_PROGRESS_EVERY_N STAGEA_EXPECT_TOP_N
export ROLLOUT_TASK_BUILDER_PY ROLLOUT_TRAJECTORY_PY ROLLOUT_REWARD_PY ROLLOUT_REWARD_LABEL INFINITY_REQUIRE_TGT_HW ROLLOUT_REQUIRE_TRACE_FILES
export UAVFLOW_STAGEA_ROLLOUT_BACKEND UAVFLOW_SIMULATOR_BASE_URL UAVFLOW_SIMULATOR_TIMEOUT_S UAVFLOW_TASK_JSON_ROOT
export CHECKPOINTS_DIR T5_PATH="${STAGEA_T5_PATH}" VAE_PATH="${STAGEA_VAE_PATH}"
# Rollout retry knobs (for invalid reward/logprob/actions cases).
INFINITY_ROLLOUT_MAX_RETRY="${INFINITY_ROLLOUT_MAX_RETRY:-3}"
INFINITY_ROLLOUT_RETRY_SEED_STEP="${INFINITY_ROLLOUT_RETRY_SEED_STEP:-9973}"
export INFINITY_ROLLOUT_MAX_RETRY INFINITY_ROLLOUT_RETRY_SEED_STEP

mkdir -p "${OUT_DIR}" "${FAST_OUT_DIR}" "${TRAJ_DIR}" "${STAGEA_SYNC_ROOT}"
STAGEA_NODE_SUCCESS=0
stagea_on_exit() {
  local exit_code="$1"
  if [ "${exit_code}" -ne 0 ] && [ "${STAGEA_NODE_SUCCESS}" -ne 1 ]; then
    mkdir -p "${STAGEA_SYNC_ROOT}"
    printf '%s\n' "${exit_code}" > "${STAGEA_FAIL_MARKER}"
  fi
}
trap 'stagea_on_exit "$?"' EXIT
if [ "${STAGEA_LOCAL_LOG_ENABLE}" = "1" ] && [ -z "${STAGEA_LOCAL_LOG_REDIRECTED:-}" ]; then
  mkdir -p "${STAGEA_LOCAL_LOG_ROOT}"
  mkdir -p "$(dirname "${STAGEA_LOCAL_LOG_FILE}")"
  export STAGEA_LOCAL_LOG_REDIRECTED=1
  exec > >(tee -a "${STAGEA_LOCAL_LOG_FILE}") 2>&1
fi
echo "[stageA] local_log_file=${STAGEA_LOCAL_LOG_FILE}"
echo "[stageA] run_id=${RUN_ID} top_n=${TOP_N} k_cand=${K_CAND} stagea_nproc=${STAGEA_NPROC} total_shards=${N_SHARDS} nnodes=${NNODES} node_rank=${NODE_RANK} local_gpus=${STAGEA_GPU_IDS} trace_ce_tmax=${INFINITY_GRPO_TRACE_CE_TMAX}"
if [ -n "${STAGEA_EXPECT_TOP_N}" ] && [ "${TOP_N}" != "${STAGEA_EXPECT_TOP_N}" ]; then
  echo "[stageA][fatal] expected TOP_N=${STAGEA_EXPECT_TOP_N}, got TOP_N=${TOP_N}"
  exit 1
fi

INPUT_JSON="${SRC_JSON}"
if [ "${TOP_N}" -gt 0 ]; then
  INPUT_JSON="${OUT_DIR}/source_top${TOP_N}.json"
fi

if [ "${NODE_RANK}" -eq 0 ]; then
  if [ "${TOP_N}" -gt 0 ]; then
    "${PYTHON_BIN}" - <<'PY'
import json, os
src=os.environ['SRC_JSON']
dst=os.environ['INPUT_JSON']
topn=int(os.environ['TOP_N'])
with open(src,'r',encoding='utf-8') as f:
    obj=json.load(f)
if isinstance(obj,list):
    out=obj[:topn]
else:
    out=obj
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst,'w',encoding='utf-8') as f:
    json.dump(out,f,ensure_ascii=False)
print(f"[stageA] wrote subset json: {dst} (top={topn})")
PY
  fi

  "${PYTHON_BIN}" "${ROLLOUT_TASK_BUILDER_PY}" \
    --input_json "${INPUT_JSON}" \
    --output_jsonl "${TASK_JSONL}" \
    --default_fps 16

  "${PYTHON_BIN}" "${ROOT_DIR}/tools/GRPO/generate_candidate_rollouts.py" \
    --task_jsonl "${TASK_JSONL}" \
    --output_jsonl "${CAND_JSONL}" \
    --k "${K_CAND}" \
    --seed_base 20260320 \
    --task_seed_stride "${STAGEA_TASK_SEED_STRIDE}" \
    --candidate_seed_stride "${STAGEA_CANDIDATE_SEED_STRIDE}"

  "${PYTHON_BIN}" - <<'PY'
import os

def count_nonempty(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n

task_jsonl = os.environ["TASK_JSONL"]
cand_jsonl = os.environ["CAND_JSONL"]
task_cnt = count_nonempty(task_jsonl)
cand_cnt = count_nonempty(cand_jsonl)
print(
    f"[stageA] task_count={task_cnt} candidate_count={cand_cnt} "
    f"(top_n={os.environ.get('TOP_N', '0')}, k_cand={os.environ.get('K_CAND', '8')})"
)
PY
fi

if [ "${NNODES}" -gt 1 ]; then
  if [ "${NODE_RANK}" -eq 0 ]; then
    printf 'ready\n' > "${STAGEA_READY_FILE}"
  else
    grpo_wait_for_path "${STAGEA_READY_FILE}" "${STAGEA_SYNC_TIMEOUT}" "stageA rollout inputs" "${STAGEA_FAIL_GLOB}"
  fi
fi

if [ "${USE_REAL_ROLLOUT}" = "1" ]; then
  PIDS=""
  if [ -n "${STAGEA_GPU_IDS}" ]; then
    IFS=',' read -r -a GPU_ARR <<< "${STAGEA_GPU_IDS}"
    if [ "${#GPU_ARR[@]}" -lt "${STAGEA_NPROC}" ]; then
      echo "STAGEA_GPU_IDS count (${#GPU_ARR[@]}) must be >= STAGEA_NPROC (${STAGEA_NPROC})"
      exit 1
    fi
  fi
  for ((i=0; i<STAGEA_NPROC; i++)); do
    GPU_ID="${i}"
    if [ -n "${STAGEA_GPU_IDS}" ]; then
      GPU_ID="${GPU_ARR[$i]}"
    fi
    GLOBAL_SHARD_ID="$((NODE_RANK * STAGEA_NPROC + i))"
    PART_SUFFIX="$(printf '%02d' "${GLOBAL_SHARD_ID}")"
    EXTRA_TRAJ_ARGS=()
    if [ -n "${INFINITY_REQUIRE_TGT_HW}" ] && [[ "${ROLLOUT_TRAJECTORY_PY}" == *"indoor"* ]]; then
      EXTRA_TRAJ_ARGS+=(--require_tgt_hw "${INFINITY_REQUIRE_TGT_HW}")
    fi
    if [ -n "${ROLLOUT_REQUIRE_TRACE_FILES}" ] && [[ "${ROLLOUT_TRAJECTORY_PY}" == *"indoor"* ]]; then
      EXTRA_TRAJ_ARGS+=(--require_trace_files "${ROLLOUT_REQUIRE_TRACE_FILES}")
    fi
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${ROLLOUT_TRAJECTORY_PY}" \
      --candidates_jsonl "${CAND_JSONL}" \
      --trajectory_root "${TRAJ_DIR}" \
      --api_py "${INFINITY_API_PY}" \
      --infinity_server_config "${INFINITY_SERVER_CONFIG}" \
      --infinity_ckpt "${INFINITY_CKPT}" \
      --infinity_repo_root "${INFINITY_REPO_ROOT}" \
      --actionhead_ckpt "${ACTIONHEAD_CKPT}" \
      --actionhead_run_config "${ACTIONHEAD_RUN_CONFIG}" \
      --actionhead_repo_root "${ACTIONHEAD_REPO_ROOT}" \
      --failed_jsonl "${FAILED_JSONL}.part${PART_SUFFIX}" \
      --timing_jsonl "${TIMING_JSONL}.part${PART_SUFFIX}" \
      --num_shards "${N_SHARDS}" \
      --shard_id "${GLOBAL_SHARD_ID}" \
      --progress_every_n "${STAGEA_PROGRESS_EVERY_N}" \
      --rollout_backend "${UAVFLOW_STAGEA_ROLLOUT_BACKEND}" \
      --simulator_base_url "${UAVFLOW_SIMULATOR_BASE_URL}" \
      --simulator_timeout_s "${UAVFLOW_SIMULATOR_TIMEOUT_S}" \
      --uavflow_task_json_root "${UAVFLOW_TASK_JSON_ROOT}" \
      --max_retry "${INFINITY_ROLLOUT_MAX_RETRY}" \
      --retry_seed_step "${INFINITY_ROLLOUT_RETRY_SEED_STEP}" \
      --dump_debug_cache 0 \
      "${EXTRA_TRAJ_ARGS[@]}" &
    PIDS="${PIDS} $!"
  done
  for p in ${PIDS}; do
    wait "$p"
  done
else
  if [ "${NODE_RANK}" -eq 0 ]; then
    echo "[stageA] USE_REAL_ROLLOUT=0, using synthetic trajectory generator (debug only)."
    "${PYTHON_BIN}" "${ROOT_DIR}/tools/GRPO/generate_candidate_trajectories.py" \
      --candidates_jsonl "${CAND_JSONL}" \
      --trajectory_root "${TRAJ_DIR}" \
      --pos_noise_std 0.02 \
      --yaw_noise_std_deg 2.0
  fi
fi

printf 'done\n' > "${STAGEA_NODE_DONE_FILE}"
if [ "${NNODES}" -gt 1 ] && [ "${NODE_RANK}" -eq 0 ]; then
  grpo_wait_for_markers "${STAGEA_SYNC_ROOT}" "rollout_node_" "${NNODES}" "${STAGEA_SYNC_TIMEOUT}" "${STAGEA_FAIL_GLOB}"
fi

if [ "${NODE_RANK}" -eq 0 ] && [ "${USE_REAL_ROLLOUT}" = "1" ]; then
  "${PYTHON_BIN}" - <<'PY'
import os
root_failed = os.environ["FAILED_JSONL"]
root_timing = os.environ["TIMING_JSONL"]
n = int(os.environ.get("N_SHARDS", "8"))
for target, prefix in [(root_failed, root_failed), (root_timing, root_timing)]:
    with open(target, "w", encoding="utf-8") as wf:
        for i in range(n):
            part = f"{prefix}.part{i:02d}"
            if not os.path.exists(part):
                continue
            with open(part, "r", encoding="utf-8") as rf:
                for line in rf:
                    if line.strip():
                        wf.write(line if line.endswith("\n") else (line + "\n"))
print(f"[stageA] merged shard files -> {root_failed}, {root_timing}")
PY

  # Health check: if zero successes, fail-fast with a concise reason.
  "${PYTHON_BIN}" - <<'PY'
import os, json

def _count_lines(p: str) -> int:
    if not p or (not os.path.exists(p)):
        return 0
    n = 0
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n

cand = os.environ.get("CAND_JSONL", "")
fail = os.environ.get("FAILED_JSONL", "")
tim = os.environ.get("TIMING_JSONL", "")
nc = _count_lines(cand)
nf = _count_lines(fail)
ns = _count_lines(tim)  # success count == timing lines
print(f"[stageA] candidates={nc} success={ns} failed={nf}")
if ns <= 0:
    # Show one failure for quick debugging.
    if fail and os.path.exists(fail):
        with open(fail, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        msg = str(obj.get("error", line.strip()))
                    except Exception:
                        msg = line.strip()
                    print("[stageA][fatal] example_failed:", msg[:400])
                    break
    raise SystemExit(2)
PY
fi

if [ "${NODE_RANK}" -eq 0 ]; then
  "${PYTHON_BIN}" "${ROLLOUT_REWARD_PY}" \
    --replay_jsonl "${CAND_JSONL}" \
    --trajectory_json_dir "${TRAJ_DIR}" \
    --output_jsonl "${REWARD_JSONL}" \
    --output_mode clip \
    --alpha_xyz "${ALPHA_XYZ}" \
    --alpha_yaw "${ALPHA_YAW}" \
    --alpha_all6 "${ALPHA_ALL6}" \
    --act_reward_mode "${ACT_REWARD_MODE}" \
    --zscore_eps 1e-6 \
    --zscore_zmax 10.0 \
    --enable_ce_reward 1 \
    --lambda_act "${LAMBDA_ACT}" \
    --lambda_task "${LAMBDA_TASK}" \
    --lambda_ce "${LAMBDA_CE}" \
    --clip_len 16 \
    --num_clips 3 \
    --clip_alpha 0.9 \
    --clip_task_pos_thresh_m "${CLIP_TASK_POS_THRESH_M}" \
    --clip_task_yaw_thresh_deg "${CLIP_TASK_YAW_THRESH_DEG}" \
    --task_pos_thresh_m "${TRAJ_TASK_POS_THRESH_M}" \
    --task_yaw_thresh_deg "${TRAJ_TASK_YAW_THRESH_DEG}" \
    --task_pos_scale_m "${TASK_POS_SCALE_M}" \
    --task_yaw_scale_deg "${TASK_YAW_SCALE_DEG}" \
    --task_pos_weight "${TASK_POS_WEIGHT}" \
    --task_yaw_weight "${TASK_YAW_WEIGHT}" \
    --task_enable_success_bonus "${TASK_ENABLE_SUCCESS_BONUS}" \
    --task_dense_weight "${TASK_DENSE_WEIGHT}" \
    --task_success_weight "${TASK_SUCCESS_WEIGHT}" \
    --task_reward_mode "${TASK_REWARD_MODE}" \
    --require_old_logprob "${REQUIRE_OLD_LOGPROB}" \
    --require_all_trajectories "${REQUIRE_ALL_TRAJECTORIES}"

  "${PYTHON_BIN}" "${ROOT_DIR}/tools/GRPO/build_replay_dataset.py" \
    --input_jsonl "${REWARD_JSONL}" \
    --output_jsonl "${REPLAY_JSONL}" \
    --lambda_act "${LAMBDA_ACT}" \
    --lambda_task "${LAMBDA_TASK}" \
    --lambda_ce "${LAMBDA_CE}" \
    --alpha_decay 0.9 \
    --mode precomputed_adv

  mkdir -p "${REPLAY_META_DIR}"
  "${PYTHON_BIN}" - <<'PY'
import os, time
src=os.environ['REPLAY_JSONL']
reward_src=os.environ['REWARD_JSONL']
out_dir=os.environ['REPLAY_META_DIR']
n=max(1,int(os.environ.get('N_SHARDS','8')))

def count_nonempty_lines(path: str) -> int:
    total = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                total += 1
    return total

expected = count_nonempty_lines(reward_src)
last_seen = -1
for attempt in range(30):
    got = count_nonempty_lines(src)
    last_seen = got
    if got == expected and got > 0:
        break
    print(f"[stageA] waiting replay_jsonl visibility: got={got}, expected={expected}, attempt={attempt+1}/30")
    time.sleep(2)
else:
    raise RuntimeError(f"[stageA] replay_jsonl not fully visible after retries: got={last_seen}, expected={expected}, src={src}")

fps=[open(os.path.join(out_dir,f"part_{i:02d}.jsonl"),'w',encoding='utf-8') for i in range(n)]
cnt=0
with open(src,'r',encoding='utf-8') as f:
    for j,line in enumerate(f):
        if not line.strip():
            continue
        fps[j % n].write(line)
        cnt += 1
for fp in fps:
    fp.close()
if cnt != expected:
    raise RuntimeError(f"[stageA] shard source count mismatch: cnt={cnt}, expected={expected}")
part_total = 0
for i in range(n):
    p = os.path.join(out_dir, f"part_{i:02d}.jsonl")
    part_total += count_nonempty_lines(p)
if part_total != expected:
    raise RuntimeError(f"[stageA] shard output count mismatch: part_total={part_total}, expected={expected}")
print(f"[stageA] sharded replay: {cnt} lines into {n} parts under {out_dir}")
PY

  "${PYTHON_BIN}" "${ROOT_DIR}/tools/GRPO/summarize_replay_meta.py" \
    --replay_meta_dir "${REPLAY_META_DIR}" \
    --output_json "${SUMMARY_JSON}" \
    --fail_on_negative_adv 1 \
    --fail_on_success_negative 1

  # Fail-fast if StageA produced zero valid rollouts (avoid starting StageB with empty dataset).
  if [ ! -s "${REPLAY_JSONL}" ]; then
    echo "[FATAL] StageA produced empty replay_jsonl: ${REPLAY_JSONL}"
    echo "See failures: ${FAILED_JSONL}"
    exit 1
  fi

  "${PYTHON_BIN}" - <<'PY'
import os, json
manifest = {
  "run_id": os.environ["RUN_ID"],
  "use_real_rollout": os.environ.get("USE_REAL_ROLLOUT", "1"),
  "cache_root": os.environ["RL_CACHE_ROOT"],
  "cache_run_dir": os.environ["OUT_DIR"],
  "task_jsonl": os.path.join(os.environ["OUT_DIR"], "rollout_tasks.jsonl"),
  "candidate_jsonl": os.path.join(os.environ["OUT_DIR"], "rollout_candidates.jsonl"),
  "reward_jsonl": os.path.join(os.environ["OUT_DIR"], "rollout_tasks_rewarded.jsonl"),
  "replay_jsonl": os.path.join(os.environ["OUT_DIR"], "rollout_replay.jsonl"),
  "traj_dir": os.path.join(os.environ["OUT_DIR"], "trajectories"),
  "failed_jsonl": os.path.join(os.environ["OUT_DIR"], "rollout_failed.jsonl"),
  "timing_jsonl": os.path.join(os.environ["OUT_DIR"], "rollout_timing.jsonl"),
  "fast_replay_meta_dir": os.environ["REPLAY_META_DIR"],
  "summary_json": os.environ["SUMMARY_JSON"],
  "k_cand": int(os.environ.get("K_CAND", "8")),
  "stagea_task_seed_stride": int(os.environ.get("STAGEA_TASK_SEED_STRIDE", "1000003")),
  "stagea_candidate_seed_stride": int(os.environ.get("STAGEA_CANDIDATE_SEED_STRIDE", "65537")),
  "alpha_xyz": float(os.environ.get("ALPHA_XYZ", "1.0")),
  "alpha_yaw": float(os.environ.get("ALPHA_YAW", "1.0")),
  "alpha_all6": float(os.environ.get("ALPHA_ALL6", "0.2")),
  "lambda_act": float(os.environ.get("LAMBDA_ACT", "0.3")),
  "lambda_task": float(os.environ.get("LAMBDA_TASK", "1.3")),
  "task_pos_scale_m": float(os.environ.get("TASK_POS_SCALE_M", "2.0")),
  "task_yaw_scale_deg": float(os.environ.get("TASK_YAW_SCALE_DEG", "10.0")),
  "task_pos_weight": float(os.environ.get("TASK_POS_WEIGHT", "1.0")),
  "task_yaw_weight": float(os.environ.get("TASK_YAW_WEIGHT", "1.0")),
  "task_enable_success_bonus": int(os.environ.get("TASK_ENABLE_SUCCESS_BONUS", "1")),
  "task_dense_weight": float(os.environ.get("TASK_DENSE_WEIGHT", "0.85")),
  "task_success_weight": float(os.environ.get("TASK_SUCCESS_WEIGHT", "0.15")),
  "task_reward_mode": os.environ.get("TASK_REWARD_MODE", "raw_dense"),
  "lambda_ce": float(os.environ.get("LAMBDA_CE", "0.03")),
    "actionhead_repo_root": os.environ.get("ACTIONHEAD_REPO_ROOT", ""),
  "infinity_repo_root": os.environ.get("INFINITY_REPO_ROOT", ""),
    "rollout_task_builder_py": os.environ.get("ROLLOUT_TASK_BUILDER_PY", ""),
    "rollout_trajectory_py": os.environ.get("ROLLOUT_TRAJECTORY_PY", ""),
    "rollout_reward_py": os.environ.get("ROLLOUT_REWARD_PY", ""),
    "rollout_reward_label": os.environ.get("ROLLOUT_REWARD_LABEL", ""),
}
with open(os.environ["MANIFEST_JSON"], "w", encoding="utf-8") as f:
  json.dump(manifest, f, ensure_ascii=False, indent=2)
print(f"[stageA] manifest: {os.environ['MANIFEST_JSON']}")
PY

  echo "[stageA] rollout task jsonl ready: ${TASK_JSONL}"
  echo "[stageA] candidate jsonl: ${CAND_JSONL}"
  echo "[stageA] replay jsonl: ${REPLAY_JSONL}"
  echo "[stageA] replay meta dir for train: ${REPLAY_META_DIR} (part_00..part_$(printf '%02d' $((N_SHARDS-1))))"
  echo "[stageA] replay summary json: ${SUMMARY_JSON}"
  echo "[stageA] heavy artifacts stored in cache dir: ${OUT_DIR}"
  printf 'complete\n' > "${STAGEA_COMPLETE_FILE}"
else
  grpo_wait_for_path "${STAGEA_COMPLETE_FILE}" "${STAGEA_SYNC_TIMEOUT}" "stageA completion marker" "${STAGEA_FAIL_GLOB}"
fi

STAGEA_NODE_SUCCESS=1

