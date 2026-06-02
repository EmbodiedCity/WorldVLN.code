# Remote Simulator Workflow

The remote_sim path keeps policy inference on the training host and delegates only environment reset, action execution, and rendering to the simulator host.

## Service Endpoints

The simulator-side service is implemented by runtime/client.py in service mode. It exposes:

- GET /health
- POST /reset
- POST /step_actions

## Start the Service

On the simulator machine:

```bash
python runtime/client.py \
  --mode service \
  --host 0.0.0.0 \
  --port 8765 \
  --task_json_root /path/to/uavflow_tasks/select_from_train_jsons
```

## Reverse Port Forwarding

If the simulator machine cannot be reached directly from the training host, create a reverse tunnel from the simulator machine to the training host:

```bash
ssh -R 18765:127.0.0.1:8765 user@training-host
```

Then validate from the training host:

```bash
curl --noproxy '*' http://127.0.0.1:18765/health
```

## Rollout Settings

The open-source action-aware GRPO package keeps only remote_sim for rollout. Use the wrapper command below:

```bash
cd ./action_aware_grpo

unset ALL_PROXY all_proxy
export NO_PROXY=127.0.0.1,localhost

export PYTHON_BIN=${PYTHON_BIN:-python}

export CUDA_VISIBLE_DEVICES=0
export GRPO_LOCAL_GPU_IDS=0
export NPROC_PER_NODE=1
export NNODES=1
export NODE_RANK=0

export SRC_JSON=/path/to/reference_video_full_49f_trajectory_prompts.json
export CHECKPOINTS_DIR=/path/to/checkpointsinf
export INFINITY_CKPT=/path/to/infinity/global_step_xxx.pth
export ACTIONHEAD_CKPT=/path/to/actionhead/checkpoint_last.pth
export ACTIONHEAD_RUN_CONFIG=/path/to/actionhead/run_config.json
UAVFLOW_STAGEA_ROLLOUT_BACKEND=remote_sim
UAVFLOW_SIMULATOR_BASE_URL=http://127.0.0.1:18765
export UAVFLOW_SIMULATOR_TIMEOUT_S=120
UAVFLOW_TASK_JSON_ROOT=/path/to/uavflow_tasks/select_from_train_jsons
STAGEA_NPROC=1

export RUN_ID=rl_rollout_smoke_$(date +%Y%m%d_%H%M%S)

bash scripts/run_stagea_collect.sh \
  RUN_ID=${RUN_ID} \
  TOP_N=1 \
  K_CAND=1 \
  STAGEA_NPROC=1 \
  STAGEA_PROGRESS_EVERY_N=1
```

STAGEA_NPROC should remain 1 unless the simulator service is extended to support multiple concurrent sessions.
