#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Match the other launchers' KEY=value overrides without interpreting shell code.
for arg in "$@"; do
  case "${arg}" in
    *=*) export "${arg}" ;;
    *) echo "[FATAL] Expected KEY=value, got: ${arg}" >&2; exit 2 ;;
  esac
done

# This entry point must never silently inherit a partial-freeze smoke setup.
if [ "${FREEZE_CHUNK_PREFIX:-0}" != "0" ]; then
  echo "[FATAL] Full-backbone training requires FREEZE_CHUNK_PREFIX=0; clear the conflicting setting or use run_stageb_partialfreeze.sh." >&2
  exit 2
fi
if [ "${GRPO_BLOCK_DTYPE:-float32}" != "float32" ]; then
  echo "[FATAL] Full-backbone training requires GRPO_BLOCK_DTYPE=float32 for optimizer master weights; Stage A traces must match the configured forward dtype." >&2
  exit 2
fi
export FREEZE_CHUNK_PREFIX=0
export GRPO_BLOCK_DTYPE=float32

echo "[stageB_full] all backbone chunks trainable; transformer master weights=float32"
exec bash "${REPO_ROOT}/Worldmodel/runtime/scripts/GRPO_stageB_train.sh"
