# WorldVLN: Autoregressive World Action Model for Aerial Vision-Language Navigation

[![arXiv](https://img.shields.io/badge/arXiv-2605.15964-b31b1b.svg)](https://arxiv.org/abs/2605.15964)
[![Website](https://img.shields.io/badge/Project-Website-blue.svg)](https://embodiedcity.github.io/WorldVLN/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model%20weights-yellow.svg)](https://huggingface.co/EmbodiedCity/WorldVLN)

This repository provides the reference implementation for WorldVLN. It includes an autoregressive inference for closed-loop action prediction, as well as two-stage training pipelines: (1) supervised backbone and action decoder training, and (2) action-aware GRPO–based optimization.


## Installation

We recommend using a single Python 3.10 environment for the released workflows. In our validated launch scripts, the Python interpreter is passed explicitly through `PYTHON_BIN`, so after activating your environment it is recommended to export:

```bash
export PYTHON_BIN=$(which python)
```

### Recommended Environment

1. Create a Python 3.10 environment.

```bash
conda create -n worldvln python=3.10
conda activate worldvln
```

2. Install a PyTorch build that matches your CUDA environment. For the released training and action-aware GRPO workflows, a PyTorch 2.5.1 environment is the recommended baseline.

3. Install the shared dependencies used by the released workflows.

```bash
pip install -r requirements.txt
```

## Setup

### Model Weights

Official WorldVLN weights are available on Hugging Face:

- [WorldVLN weights](https://huggingface.co/EmbodiedCity/WorldVLN)

Download the weights to your preferred checkpoint directory and configure the relevant training or inference scripts to point to them.

Simulation / benchmark resources:

- IndoorUAV simulation environment download guide: [Indoor_UAV](https://modelscope.cn/datasets/valyentine/Indoor_UAV)
- UAV-Flow benchmark and evaluation environment: [buaa-colalab/UAV-Flow](https://github.com/buaa-colalab/UAV-Flow)

## Inference

The repository currently provides two main inference entry points.

![WorldVLN model](./assets/model.png)

### Online Inference Service

#### Quick start

From the repository root:

```bash
export PYTHON_BIN=$(which python)
export INFINITY_CKPT=/path/to/infinity/global_step_xxx.pth
export STAGE2_LATENT2ACTION_CKPT=/path/to/stage2_latent2action_combined.pt

bash infer/run_server.sh
```

Common environment variables:

- `INFINITY_CKPT`: main InfinityStar / WorldVLN checkpoint used by the service
- `STAGE2_LATENT2ACTION_CKPT`: Stage-2 latent-to-action checkpoint for action prediction
- `INFINITY_SERVER_CONFIG`: optional override for `infer/config.json`
- `INFINITY_REPO_ROOT`: optional override for the default `Worldmodel/runtime/`
- `INFINITY_LATENT_CACHE_ROOT`: runtime cache directory used by the service
- `HOST`, `PORT`: bind address for Uvicorn

#### Details

- Entry points: [infer/run_server.sh](./infer/run_server.sh), [infer/server.py](./infer/server.py)
- Configuration: [infer/config.json](./infer/config.json)
- Windows-side client: [infer/client.py](./infer/client.py)

#### Autoregressive I/O contract (how clients talk to the server)

WorldVLN runs in an **autoregressive** closed-loop protocol over a trajectory `session_id`:

- **Input (per call)**: `images_base64` (RGB frames) + an optional `instruction` on the first call.
  - First call typically sends **1 warmup frame**.
  - Next calls send **`step` frames** (default 16) to advance the session timeline.
- **State (server-side)**: the server stores history by `session_id` and maintains a streaming world-model session.
- **Output (per call)**: `actions` as delta actions in **cm/deg** with order `[dx, dy, dz, droll, dyaw, dpitch]`.
  - In the default `tsformer_latent` mode, the server emits **one segment worth of actions** whenever enough frames have been received.

Example (strict closed-loop, enable `allow_future_segments=1`):

- Send **(1 real frame + instruction/prompt)** → get the next **`step` actions** (typically 16).
- Execute them, collect the next **`step` real frames**, send them → get the next **`step` actions**.
- Repeat with the same `session_id` until the trajectory ends.

Clients in this repo follow the same pattern:

- **`infer/client.py`** (simulation / dataset example): sends `1, step, step, ...` frames under a stable `session_id`, and writes per-segment `*_actions.json` / `*_poses.json`.
- **`action_aware_grpo/windows_client.py`** (Windows-side rollout integration / debugging): uses the same session protocol and output format, but is packaged as a Win-friendly client for action-aware GRPO workflows.

## Training

This repository is organized into two stages:

- **Stage 1 (supervised)**: backbone finetuning + action decoder training.
- **Stage 2 (action-aware GRPO)**: rollout collection + GRPO training.

![WorldVLN framework](./assets/framework.png)

### Stage 1: Supervised Training

#### Backbone Training

#### Quick start

```bash
bash train/scripts/train_from_base.sh
```

#### Details

The backbone finetuning workflow is located under [train/](./train).

- Entry point: [train/scripts/train_from_base.sh](./train/scripts/train_from_base.sh)
- Main trainer: [train/train.py](./train/train.py)
- Detailed guide: [train/TRAINING.md](./train/TRAINING.md)

#### Action Decoder Training

#### Quick start (Stage A + Stage B)

```bash
# Stage A: adapter distillation
bash train/action_decoder/scripts/train_stageA_ddp.sh

# Stage B: latent-to-action training
bash train/action_decoder/scripts/train_stageB_ddp.sh
```

#### Details

The action decoder workflow is located under [Worldmodel/action_decoder/src/](./Worldmodel/action_decoder/src) and is organized into two steps (Stage A + Stage B).

The action decoder training entrypoints live under [train/action_decoder/](./train/action_decoder) and are organized into two steps:

- Stage A adapter distillation: [train/action_decoder/scripts/train_stageA_ddp.sh](./train/action_decoder/scripts/train_stageA_ddp.sh)
- Stage B latent-to-action training: [train/action_decoder/scripts/train_stageB_ddp.sh](./train/action_decoder/scripts/train_stageB_ddp.sh)
- Main scripts: [train/action_decoder/tools/train_stageA_ddp.py](./train/action_decoder/tools/train_stageA_ddp.py), [train/action_decoder/tools/train_stageB_ddp.py](./train/action_decoder/tools/train_stageB_ddp.py)

This workflow trains the mapping from visual latent features to 6-DoF motion outputs.

Data contract (training manifest):

```json
{
  "items_train": [
    {
      "latent_path": "path/to/latents.pt",
      "traj_json_path": "path/to/preprocessed_logs.json",
      "images_dir": "path/to/images"
    }
  ]
}
```

Stage A required environment variables:

- `MANIFEST_JSON`
- `TSFORMER_CKPT`
- `INF_VAE_PATH`

Run Stage A:

```bash
bash train/action_decoder/scripts/train_stageA_ddp.sh
```

Stage B required environment variables:

- `MANIFEST_JSON`
- `TSFORMER_PRETRAINED`
- `ADAPTER_CKPT`
- `INFINITYSTAR_VAE_PATH`

Run Stage B:

```bash
bash train/action_decoder/scripts/train_stageB_ddp.sh
```

### Stage 2: Action-aware GRPO

#### Quick start (rollout + train)

Start the local inference service used by rollout:

```bash
INFINITY_CKPT=/path/to/infinity/global_step_xxx.pth \
CHECKPOINTS_DIR=/path/to/checkpointsinf \
ACTIONHEAD_CKPT=/path/to/actionhead/checkpoint_last.pth \
ACTIONHEAD_RUN_CONFIG=/path/to/actionhead/run_config.json \
bash action_aware_grpo/run_infer_server.sh
```

Run rollout collection:

```bash
unset ALL_PROXY all_proxy
export NO_PROXY=127.0.0.1,localhost

SRC_JSON=/path/to/reference_video_full_49f_trajectory_prompts.json \
INFINITY_CKPT=/path/to/infinity/global_step_xxx.pth \
CHECKPOINTS_DIR=/path/to/checkpointsinf \
ACTIONHEAD_CKPT=/path/to/actionhead/checkpoint_last.pth \
ACTIONHEAD_RUN_CONFIG=/path/to/actionhead/run_config.json \
CUDA_VISIBLE_DEVICES=0 \
GRPO_LOCAL_GPU_IDS=0 \
NPROC_PER_NODE=1 \
NNODES=1 \
NODE_RANK=0 \
UAVFLOW_STAGEA_ROLLOUT_BACKEND=remote_sim \
UAVFLOW_SIMULATOR_BASE_URL=http://127.0.0.1:18765 \
UAVFLOW_SIMULATOR_TIMEOUT_S=120 \
UAVFLOW_TASK_JSON_ROOT=/path/to/uavflow_tasks/select_from_train_jsons \
bash action_aware_grpo/scripts/run_stagea_collect.sh RUN_ID=remote_sim_smoke TOP_N=1 K_CAND=1 STAGEA_NPROC=1 STAGEA_PROGRESS_EVERY_N=1
```

Run train (partial-freeze optimization):

```bash
CHECKPOINTS_DIR=/path/to/checkpointsinf \
RUSH_RESUME=/path/to/infinity/global_step_xxx.pth \
REPLAY_META_DIR=/path/to/replay_meta_rollout_smoke \
bash action_aware_grpo/scripts/run_stageb_partialfreeze.sh PARTIAL_FREEZE_MODE=smoke RUN_ID=stageb_smoke
```

#### Details

The action-aware GRPO workflow is located under [action_aware_grpo/](./action_aware_grpo) and is organized into two steps: **rollout** and **train**.

- Server entry point: [action_aware_grpo/grpo_server.py](./action_aware_grpo/grpo_server.py)
- Windows-side client (for action-aware GRPO rollout integration / debugging): [action_aware_grpo/windows_client.py](./action_aware_grpo/windows_client.py)
- Rollout collection: [action_aware_grpo/scripts/run_stagea_collect.sh](./action_aware_grpo/scripts/run_stagea_collect.sh)
- Train (partial-freeze optimization): [action_aware_grpo/scripts/run_stageb_partialfreeze.sh](./action_aware_grpo/scripts/run_stageb_partialfreeze.sh)
- Remote simulator service wrapper: [action_aware_grpo/scripts/run_remote_sim_service.sh](./action_aware_grpo/scripts/run_remote_sim_service.sh)
- Local inference launcher used by rollout: [action_aware_grpo/run_infer_server.sh](./action_aware_grpo/run_infer_server.sh)

At a high level:

- Rollout consumes rollout sources and model assets, then generates rollout caches and replay metadata.
- Train consumes replay metadata and runs optimization to produce updated checkpoints and logs.

For simulator-backed rollout details, see [action_aware_grpo/docs/remote_sim.md](./action_aware_grpo/docs/remote_sim.md).

## Acknowledgement

We sincerely thank the following projects for their exceptional effort: [InfinityStar](https://github.com/FoundationVision/InfinityStar), [TSformer-VO](https://github.com/aofrancani/TSformer-VO).

## Citation

If you find this work useful, welcome to cite the WorldVLN paper:

```bibtex
@misc{zhao2026worldvln,
      title={WorldVLN: Autoregressive World Action Model for Aerial Vision-Language Navigation}, 
      author={Baining Zhao and Jiacheng Xu and Weicheng Feng and Xin Zhang and Zhaolu Wang and Haoyang Wang and Shilong Ji and Ziyou Wang and Jianjie Fang and Zhiheng Zheng and Weichen Zhang and Yu Shang and Wei Wu and Chen Gao and Xinlei Chen and Yong Li},
      year={2026},
      eprint={2605.15964},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.15964}, 
}
```

## License

This project is released under the **CC BY 4.0** license. See `LICENSE`.
