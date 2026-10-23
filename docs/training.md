# Training and installation

## Environment

Training targets Linux/CUDA with 4 GPUs, 32 Ray CPUs, and 4 executor workers. Prepare an environment that supports Qwen3-VL.

`physevolve_env.yml` is a dependency reference. Its Transformers 4.53.0 pin predates Qwen3-VL support, so the runtime needs a compatible Transformers/vLLM stack. Run artifacts record the installed versions. veRL uses the pinned source checkout below.

```bash
# Activate your Qwen3-VL environment first.
bash scripts/setup_verl.sh
python -m pip install --no-deps -e ../verl
export MUJOCO_GL=egl
```

MuJoCo needs a working OpenGL backend. Use EGL and the appropriate NVIDIA driver on a headless training server. On macOS, CPU checks can use `MUJOCO_GL=glfw`; full distributed training is not targeted there.

veRL is fixed to `bcb638649a50e58494a8ddd92085ad1174f674b8`, with four patches for Qwen-VL mRoPE/NestedTensor handling. See [architecture and patches](architecture.md#verl-integration). `setup_verl.sh` refuses dirty checkouts; its explicit `--force` option resets only the selected external veRL checkout.

## Model and paths

The default base is `Qwen/Qwen3-VL-8B-Instruct`. Use a local Hugging Face model directory with `--model /path/to/Qwen3-VL-8B-Instruct` or `MODEL_PATH`. `--out-root` or `OUT_ROOT` selects output storage. Run the commands from the repository root. The launchers default to GPUs `0,1,2,3`; override both `--gpu` and `--n-gpus-per-node` when changing GPU count.

PhysEvolve and Naive both start from the base model. Naive is a separately trained single-trajectory baseline, not a required warmup stage.

## Recipes

| Entry | Training behavior | Checkpoint interval |
| --- | --- | ---: |
| `scripts/train/physevolve.sh` | Shared Proposer/Solver, branch execution, exact VQA reward | 25 |
| `scripts/train/naive.sh` | Single trajectory, Forward/Inverse/Parameter tasks, graded reward | 10 |
| `scripts/train/ablations/no_branch.sh` | Disable branches and their dependent task families | 100 |
| `scripts/train/ablations/freeze_proposer.sh` | Exclude Proposer-role rows from updates; still generate with the shared policy | 100 |

All default to 100 steps. The two ablation wrappers inherit the full PhysEvolve recipe. They enforce their ablation and checkpoint settings after forwarded arguments; do not stack them. Naive retains its own recorded recipe, including 64 Ray CPUs.

The PhysEvolve paper settings are:

| Setting | Value |
| --- | --- |
| Intent groups × proposals | 4 × 4 per step |
| Solver groups | At most 5 items per scene; 32 groups per step; 4 answers per group |
| Generation | Temperature 0.7; top-p 0.95; top-k 20 |
| Token budgets | 6,144 prompt; 4,096 response |
| GRPO | 1 epoch; learning rate 1e-6; clip ratio 0.2; entropy coefficient 0 |
| KL | Actor loss and reward penalty: 0.03 |
| Golden programs | 10 throwing + 8 collision + 6 motion, available through step 100 |
| Curriculum | 8-step family-sampling cold start; adaptive sampling thereafter |
| Proposer instance-learnability reward | Coefficient 0 while demonstrations are shown (all 100 reported steps) |
| Seeds | Executor starts at 9001; curriculum/reference RNG base 9001 |
| Rendering | 32 frames at 480 × 480; one additional color permutation |
| Executor | 0.002 s timestep; 30 s simulation timeout; 15 s rendering timeout; up to 6 distinct branches |

The shell recipe and generated `verl_config.json` specify reward, reference-buffer, and anti-copy settings. Reported results use one run per configuration; construction seeds control scene generation and curriculum sampling.

Family sampling adapts to rolling Solver accuracy after step 8. This is separate from the Proposer's instance-learnability reward, which is disabled throughout the reported demonstration schedule. Proposer updates use the configured validity, intrinsic, and copy-control terms; Solver updates use exact answer correctness.

## Dry-run, preflight, and GPU smoke

```bash
# No weights, Ray, or GPU allocation.
bash scripts/train/physevolve.sh --dry-run --run-id config-check
bash scripts/train/naive.sh --dry-run --run-id naive-config-check
bash scripts/train/ablations/no_branch.sh --dry-run --run-id branch-config-check
bash scripts/train/ablations/freeze_proposer.sh --dry-run --run-id proposer-config-check

# Check installed veRL imports and interfaces without loading the model.
bash scripts/train/physevolve.sh --factory-preflight

# Actual two-step GPU training and checkpoint save.
bash scripts/train/physevolve.sh \
  --max-steps 2 --stage2-cold-start-fewshot-steps 2 --run-id smoke

# Full paper run.
bash scripts/train/physevolve.sh --run-id physevolve
```

Run directories are immutable: choose a fresh `--run-id` for each invocation. To use W&B, append `--wandb`; authentication is then provided by your own environment. Training is console-logged by default.

Check `summary.json` for completion and two PPO updates, and `global_step_2/actor/` for the saved checkpoint.

## Run artifacts and checkpoint export

Each run records `config.yaml`, `verl_config.json`, and `provenance.json` (code revision, dirty-tree flag, model source, installed dependency versions, and construction seeds). Real training also records `resolved_verl_config.json` after merging the pinned veRL configuration. Metrics, per-step artifacts, summaries, and FSDP checkpoints live in the same output directory.

Export the terminal checkpoint for Hugging Face evaluation:

```bash
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir outputs/physevolve/global_step_100/actor \
  --target_dir outputs/physevolve/merged_hf_global_step_100
```

For a complete Linux workflow, the pipeline helper trains, exports each scheduled checkpoint, and runs PhysBench validation and test evaluation:

```bash
bash scripts/run_stage2_train_merge_eval.sh \
  --run-id physevolve --dataset-path /path/to/PhysBench

# Existing training run: export and evaluate, without retraining.
bash scripts/run_stage2_train_merge_eval.sh \
  --run-dir outputs/physevolve --dataset-path /path/to/PhysBench \
  --steps '25 50 75 100'
```

The pipeline's `--run-dir` continues export/evaluation; it does not resume interrupted optimizer training. CausalPhys and ContPhy commands are in the [evaluation guide](evaluation.md). Exported merged weights support evaluation; resuming optimization additionally requires the distributed training state.
