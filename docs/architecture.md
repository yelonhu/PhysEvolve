# Architecture and development

## Method boundary

The same VLM parameters serve as Proposer and Solver. A Proposer emits a PhysDSL program plus a literal `SCENE_MANIFEST` inside `<propose>` tags. The executor validates the program, performs bounded repairs, freezes an immutable scene, and runs baseline and required body-elision branches with matched seeds. It derives physical answer semantics before constructing the multiple-choice surface. The Solver receives only baseline visual evidence and the question/options, and returns an option inside `<answer>` tags.

The executor reads simulator state, verifies interventions, and binds each generated answer. Failed or ambiguous worlds/items are filtered; exact option correctness supplies the Solver reward. Shared-policy GRPO and a rolling family-accuracy curriculum close the training loop.

## Components

| Component | Responsibility |
| --- | --- |
| `physevolve/engine/` | Typed PhysDSL, scene compilation, MuJoCo simulation and rendering |
| `physevolve/executor/` | Sandbox, repair, frozen branches, task admission and traces |
| `physevolve/skills/` | Scene templates, reasoning families and option layouts |
| `physevolve/prompts/` · `tasks/` | Model interfaces and observations |
| `physevolve/verification/` | Exact VQA rewards, proposal rewards and Naive diagnostics |
| `physevolve/training/` | Online construction, curriculum, veRL integration and checkpoints |
| `physevolve/inference/` · `integrations/` | Inference, answer parsing and video payloads |
| `physevolve/smoke/` | Small configuration/IO utilities imported by training |

The Naive baseline (`stage=1`) retains its dense Forward/Inverse/Parameter task and reward definitions. PhysEvolve (`stage=2`) uses `gen_vqa` and `pred_vqa`. Stage 2 does not consume a Stage-1 checkpoint.

Training uses the demonstration programs in `data/golden_scenes/` and the option layouts in `data/task_bank/option_layout_bank.jsonl`. The layout bank retains option text and frequency weights derived from the PhysBench catalog, including test-split observations. Simulated outcomes determine the answers to generated training questions.

The compiler/executor fixtures under `data/fixtures/` retain their original source labels; those labels are identifiers rather than paths in this checkout.

## veRL integration

veRL owns distributed rollout, log probabilities, advantages, optimization, FSDP, Ray, and checkpointing. PhysEvolve owns online physics task construction and rewards.

```text
scripts/train/physevolve.sh
  -> scripts/train/_entry.py
  -> initialized veRL trainer
  -> proposer rollout
  -> PhysDSL validation, simulation, branches, rendering and task construction
  -> solver video rollout and exact reward
  -> GRPO actor update, curriculum update and checkpoint
```

The four patches in `third_party/verl/patches/` are applied in order to the revision in `base_commit.txt`:

1. Jagged NestedTensor unbind fallback.
2. Qwen-VL mRoPE storage recovery and reliable select/chunk behavior.
3. Packing mRoPE position IDs before FSDP/AutoModel calls.
4. CPU protocol regression tests for these layouts.

Sequence lengths come from `input_ids.offsets().diff()`. The model receives position IDs packed as `(4, 1, total_sequence_length)`. These changes preserve multimodal position alignment across padding removal and batching.

## Tests

For a lightweight CPU development environment, separate from the training manifest:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
MUJOCO_GL=egl python -m pytest -q
```

On macOS use `MUJOCO_GL=glfw`. Linux CPU-only CI uses Mesa/OSMesa with `MUJOCO_GL=osmesa`. Tests cover scene schemas, simulator and branch behavior, golden programs, reward/curriculum logic, launcher recipes, answer parsing, and benchmark scoring. Model-dependent integration tests may skip when their optional runtime is absent.

Dry-runs check configuration generation, factory preflight checks runtime interfaces, and GPU smoke runs exercise rollout, updates, and checkpoint creation.
