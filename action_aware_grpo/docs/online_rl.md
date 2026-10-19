# Exact online rollout and RL optimization

The released GRPO path uses the same autoregressive loop for sampling and differentiable forced-token replay. Trace schema 3 records the actual encoded text, every scale/repeat token, deterministic GT-prefix mask, target-clip policy-bit count, sampling distribution, model precision and checkpoint identity. Unknown training arguments, approximate `trace_ce`/`proxy_ce`, missing traces and mismatched precision fail explicitly.

All dataset, simulator and checkpoint locations are supplied by configuration. Do not edit source files to insert a workstation path.

## Windows simulator service

Install the UAV-Flow evaluation environment and configure its Gym/UnrealCV UE executable settings. Run the service from that evaluation workspace. Either use its updated `client.py` together with the sibling `sim_service.py` and `pose_math.py`, or invoke this repository's three runtime files using:

```powershell
.\action_aware_grpo\scripts\run_simulator_service.ps1 -EvalRoot "C:\path\to\UAV-Flow-Eval" -TaskJsonRoot "C:\path\to\tasks" -PythonExe "C:\path\to\environment\python.exe"
```

The Python environment must provide the Gym/UnrealCV dependencies. The launcher automatically adds `EvalRoot` to `PYTHONPATH`, so the evaluation workspace's `gym_unrealcv` package and UE configuration are used. Direct CLI calls through `windows_client.py` or `runtime/client.py` require `gym_unrealcv` to be installed or the evaluation workspace to be on `PYTHONPATH`.

The service exposes `/health`, `/reset`, `/step_actions` and `/abort`. The last endpoint releases only the matching session; incorrect ownership returns HTTP 409. One service owns one simulator and supports one active rollout at a time. Use one Stage A worker. Independent parallel workers require independent simulator instances and services.

From Windows, create a reverse tunnel when Linux cannot reach Windows directly:

```powershell
ssh -N -T -o ExitOnForwardFailure=yes -R 18765:127.0.0.1:8765 your-training-host
```

Verify `/health` through `http://127.0.0.1:18765` on Linux. Health performs real simulator reads; rollout responses contain observed world poses rather than commanded targets. The action protocol is previous-body SE(3), cm/deg, `[dx,dy,dz,droll,dyaw,dpitch]`, with `T_next=T_previous@T_delta`.

The Collection actor can constrain roll/pitch. The default service verifies the pose after rendering and reports HTTP 503 when the commanded pose cannot be realized. Configure tolerances through the simulator's service settings; do not treat an unexecuted pose as a successful action.

## Site configuration

The following values are examples to replace with your own assets:

```bash
export PYTHON_BIN=/path/to/environment/bin/python
export CHECKPOINTS_DIR=/path/to/shared/infinity/assets
export T5_PATH=$CHECKPOINTS_DIR/text_encoder/flan-t5-xl-official
export VAE_PATH=$CHECKPOINTS_DIR/infinitystar_videovae.pth
export INFINITY_CKPT=/path/to/policy.pth
export ACTIONHEAD_CKPT=/path/to/actionhead/checkpoint_last.pth
export ACTIONHEAD_RUN_CONFIG=/path/to/actionhead/run_config.json
export SRC_JSON=/path/to/rollout_manifest.json
export UAVFLOW_TASK_JSON_ROOT=/path/to/tasks
export UAVFLOW_SIMULATOR_BASE_URL=http://127.0.0.1:18765
export CUDA_VISIBLE_DEVICES=0 GRPO_LOCAL_GPU_IDS=0
export NPROC_PER_NODE=1 NNODES=1 NODE_RANK=0 STAGEA_NPROC=1
```

Both Stage A and Stage B use `PYTHON_BIN`. Stage B starts its launcher and workers
through that interpreter's `torch.distributed.run` module; no separate
`TORCHRUN_BIN` configuration is needed.

On machines with a small system/home filesystem, also configure temporary and JIT cache storage on a writable data volume. PyTorch's Jiterator cache uses its own variable, separately from `CUDA_CACHE_PATH`:

```bash
export OUTPUT_ROOT=/path/to/fast/output
export TMPDIR=$OUTPUT_ROOT/tmp
export PYTORCH_KERNEL_CACHE_PATH=$OUTPUT_ROOT/pytorch_kernel_cache
mkdir -p "$TMPDIR" "$PYTORCH_KERNEL_CACHE_PATH"
```

The action-head run config must describe previous-body relative labels (`use_raw_for_labels=true`) and a four-frame window. Its translation divisor is read from configuration. The manifest accepts `gt_pose_frame=initial_body` (default, identity initial pose) or `gt_pose_frame=world` (absolute poses). Both predictions and world-frame GT are converted by the same SE(3) initial-frame transform.

## Collect and optimize

### Full-backbone training

Stage A defaults to `INFINITY_BLOCK_DTYPE=float32`. The full-backbone Stage B entry point selects `FREEZE_CHUNK_PREFIX=0` and `GRPO_BLOCK_DTYPE=float32`, so every backbone chunk and the output heads are trainable with FP32 master weights. It rejects conflicting inherited freeze/dtype settings instead of silently performing partial training. The existing `run_stageb_partialfreeze.sh` retains its partial-freeze behavior.

For the default `GRPO_FSDP_COMPUTE_DTYPE=none`, configure the site values above, then collect a new FP32 batch and train it:

```bash
export INFINITY_BLOCK_DTYPE=float32
bash action_aware_grpo/scripts/run_stagea_collect.sh RUN_ID=full_rollout STAGEA_NPROC=1

export RUSH_RESUME=$INFINITY_CKPT
export REPLAY_META_DIR=/path/to/replay_meta_full_rollout
bash action_aware_grpo/scripts/run_stageb_full.sh RUN_ID=full_training
```

For optional `GRPO_FSDP_COMPUTE_DTYPE=bfloat16`, keep `GRPO_BLOCK_DTYPE=float32` and `FREEZE_CHUNK_PREFIX=0`, enable FSDP (`ZERO_STAGE>0`) and BF16 AMP (`fp16=2`, the default), and collect fresh Stage A traces with `INFINITY_BLOCK_DTYPE=bfloat16`. Only nested transformer chunks use BF16 forward parameters; the outer module/output heads and optimizer master parameters remain FP32, with FP32 gradient reduction. `GRPO_CHUNK_GRADIENT_DIAGNOSTICS=1` enables shard-aware finite/nonzero gradient checks per chunk.

`GRPO_FSDP_CPU_OFFLOAD=1` optionally places FSDP master parameters, synchronized gradients and optimizer state on CPU; its default is `0`. It requires exact `trace_replay`, FSDP and fixed stepping with `AC=1` or `AC=2`; role-based stepping is rejected. With `AC=1`, each microbatch updates the policy. With `AC=2`, the first microbatch's complete forward/backward runs inside FSDP `no_sync()`, and the second synchronizes and performs one update; each loss is scaled by `1/2`. `MAX_TRAIN_ITERS` counts microbatches and must be divisible by `AC`. Provide enough replay batches to complete the requested update. Exact trace batches also move the unused T5 encoder to CPU and use the encoded text stored in the trace.

For a bounded CPU-offload configuration, collect new BF16 Stage A traces from the same checkpoint, then supply replay metadata containing two real clip-1 rows, one per candidate:

```bash
export INFINITY_CKPT=/path/to/policy.pth
export INFINITY_BLOCK_DTYPE=bfloat16 INFINITY_STAGEA_OLD_LOGPROB_MODE=sampling
bash action_aware_grpo/scripts/run_stagea_collect.sh RUN_ID=cpu_offload_rollout TOP_N=1 K_CAND=2 STAGEA_NPROC=1

export RUSH_RESUME=$INFINITY_CKPT
export REPLAY_META_DIR=/path/to/replay_meta_two_clip1_samples
export NPROC_PER_NODE=1 NNODES=1 NODE_RANK=0 SP_SIZE=1 ZERO_STAGE=3 AC=2
export FREEZE_CHUNK_PREFIX=0 GRPO_BLOCK_DTYPE=float32 GRPO_FSDP_COMPUTE_DTYPE=bfloat16
export GRPO_FSDP_CPU_OFFLOAD=1 GRPO_REPLAY_SAVE_ON_CPU=1 GRPO_REPLAY_SAVE_ON_CPU_PIN=0
export VIDEO_BATCH_SIZE=1 MAX_TRAIN_ITERS=2 EPOCHS=1 SAVE_MODEL_ITERS_FREQ=2
bash action_aware_grpo/scripts/run_stageb_full.sh RUN_ID=cpu_offload_update
```

Here `global_step_2.pth` labels two microbatches and one optimizer update. CPU parameter offload and selective saved-activation offload are separate options; neither guarantees that a particular model or trace fits the available CPU/GPU memory. The Qwen8B test for this configuration completed two online 49-frame trajectories and six clip rewards, then trained their two complete first clips in one accumulated optimizer update. Both initial replay logprobs matched the sampling records exactly, all six chunks had finite nonzero gradients, and checkpoint comparison confirmed changes in all 36 backbone layers. The test used one A800 80 GiB GPU and reached about 483 GiB process RSS. With one rank, PyTorch uses FSDP NO_SHARD while retaining CPU offload. This validation covers one optimizer update.

The replay data must come from that same checkpoint and precision configuration. Previously collected BF16 heads-only smoke traces cannot be reused for FP32 full-backbone training; collect them again. Model/data paths, simulator settings, GPU topology and available memory remain site configuration. This entry point sets the optimization scope; it does not establish that full-backbone training fits on a particular GPU. The verified full-backbone configuration is the single-rank CPU-offload example above; other precision and GPU topologies require their own resource validation.

### Optional heads-only smoke update

For a frozen-backbone, FP32-output-head smoke update, explicitly match BF16 storage on both sides:

```bash
export INFINITY_BLOCK_DTYPE=bfloat16
export INFINITY_STAGEA_OLD_LOGPROB_MODE=sampling
bash action_aware_grpo/scripts/run_stagea_collect.sh RUN_ID=one_task TOP_N=1 K_CAND=2 STAGEA_NPROC=1

export RUSH_RESUME=$INFINITY_CKPT
export REPLAY_META_DIR=/path/to/replay_meta_one_task
bash action_aware_grpo/scripts/run_stageb_partialfreeze.sh \
  PARTIAL_FREEZE_MODE=smoke RUN_ID=three_updates \
  FREEZE_CHUNK_PREFIX=6 GRPO_BLOCK_DTYPE=bfloat16 \
  MAX_TRAIN_ITERS=3 EPOCHS=1 AC=1 VIDEO_BATCH_SIZE=1 \
  TLR=1e-8 GRPO_RATIO_EPS=0.2 GRPO_KL_BETA=0 GRPO_AUX_SFT_COEF=0
```

`FREEZE_CHUNK_PREFIX=6` is heads-only for the six-chunk Qwen8B architecture. Other architectures must use their own chunk count. For backbone training, FP32 master weights are required for trainable chunks. Collect fresh traces with the forward precision selected above: default FP32 blocks, or BF16 blocks when explicitly enabling nested FSDP BF16 compute.

Each replay row describes one generated clip. Its policy logprob sums the sampled bit likelihoods across every scale/repeat of that target clip; deterministic GT-prefix bits are excluded. PPO uses the joint clip likelihood ratio, `exp(new_logprob - old_logprob)`. Per-bit log-softmax is FP32, while clip sums and the likelihood-ratio calculation remain FP64: at roughly 1.5 million bits per clip, FP32 sums would round away small updates. Old schema 2 traces must be collected again.

Nonzero KL and auxiliary SFT coefficients are rejected because this path does not provide a separately scored per-token reference policy or an SFT anchor. It never substitutes a proxy CE loss for RL. The existing action-aware reward/gate scheme supplies the optimization weights; signed advantages are also supported when explicitly enabled.

Configure reward mixing in Stage A with `LAMBDA_ACT`, `LAMBDA_TASK` and
`LAMBDA_CE`, then rebuild rewards and replay metadata. Stage B consumes the saved
`grpo_adv_final` values directly. Its unused `grpo_lambda_*`, `grpo_alpha_decay`,
`grpo_weight_mode`, `grpo_require_old_logprob`, `grpo_pg_only` and
`grpo_log_weight_stats` CLI options have been removed. Exact replay always
requires old logprobs, uses its own policy graph and writes replay diagnostics.
Stage A's CE reward remains a scalar reward derived from sampling NLL; it is
distinct from an auxiliary SFT loss and from approximate `trace_ce` scoring.

Inspect `strict_grpo_updates.jsonl` for the initial likelihood check, finite gradient norm, active policy samples, trainable-parameter changes, frozen-parameter probes and the optimization scope. A checkpoint is saved after the final requested update. Use the updated checkpoint to collect the next round; do not reuse its predecessor's traces as if they were sampled from the new policy.

## CPU regression checks

```bash
python tests/test_online_contract.py
python tests/test_policy_trace_core.py
python Worldmodel/runtime/tests/test_strict_grpo.py
```

These validate coordinate/reward contracts, the actual shared sampling/replay loop on a tiny differentiable model, and the optimizer/trace contracts. They do not replace actual UE and large-model validation.
