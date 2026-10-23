# Evaluation

## Data preparation

Obtain benchmark media from [PhysBench](https://physbench.github.io/), [CausalPhys](https://huggingface.co/datasets/haorentang/causalphys), and the [ContPhy project](https://physical-reasoning-project.github.io/). Keep them outside the source tree or under ignored `datasets/`. Pass the local root explicitly.

| Benchmark | Dataset root must contain | Entry argument |
| --- | --- | --- |
| PhysBench | `test.json`, extracted `image/` and `video/` | `--dataset-path` |
| CausalPhys | Official annotation and media hierarchy, as described below | `--dataset-dir` |
| ContPhy | `cloth/`, `fluid/`, `rope/`, `soft/`, each with `question/` and `video/` | `--dataset-dir` |

PhysBench answer files and its official scoring implementation are included under `evaluation/physbench/third_party/`, with their upstream licenses. CausalPhys discovers `{category}/{subcategory}/annotations/*.json` and resolves their media under the sibling `data/` directory; it validates media before inference. ContPhy uses these exact test files:

```text
cloth/question/cloth_final2_test_v1.json       450
fluid/question/fluid_final_test_v1.json       450
rope/question/rope_final2_test_v1.json        600
soft/question/soft_final_test_v1.json         450
```

ContPhy videos live under `{domain}/video/{video_id}/`; the loader prefers `output_Full.mp4` and excludes segmentation/depth/mask videos. Nested duplicate dataset copies are ignored.

## Fixed evaluation contracts

| Benchmark | Scored items | Visual input | Decoding | Scoring |
| --- | ---: | --- | --- | --- |
| PhysBench | 9,802 | 8 uniform frames; original array packing | Greedy, 2,048 tokens | Vendored official answer matching |
| CausalPhys | 3,057 | 5 uniform frames; paper video packing | CoT, greedy, 512 tokens | Final A–D exact match |
| ContPhy | 1,950 | 10 presampled frames; fps 2.0 metadata | Greedy, 64 tokens | Normalized option/open-response exact match |

For paper-matched CausalPhys, set `PHYSEVOLVE_LEGACY_VIDEO_PACKING=1` for that command only. This retains the original processor-side sampling behavior. Use `PHYSEVOLVE_LEGACY_VIDEO_PACKING=0` for ContPhy's presampled protocol. These input paths must not be mixed when comparing results.

CausalPhys normalizes a lowercase gold `b` and excludes five annotations whose gold is not A–D. The parser prefers the final answer block, not letters within the rationale. Unparsed responses count as incorrect.

ContPhy uses an adapted evaluation protocol: one question per generation, with extracted open responses lowercased and stripped of whitespace. It differs from the official GPT-4V multi-question batching and per-option, all-correct-or-fail scoring. Compare model results under the same protocol; its accuracy is not directly comparable with the official readout.

## Commands

Set `MODEL` to the base model identifier or an exported checkpoint directory:

```bash
MODEL=outputs/physevolve/merged_hf_global_step_100

bash scripts/eval/run_physbench_official_eval_multi_gpu.sh \
  --dataset-path /path/to/PhysBench --split test --gpu-list 0,1,2,3 \
  --model-path "$MODEL" --n-video-frames 8 --max-new-tokens 2048 \
  --out-dir outputs/evaluation/physbench

PHYSEVOLVE_LEGACY_VIDEO_PACKING=1 \
bash scripts/eval/run_causalphys_vlm_eval_multi_gpu.sh \
  --dataset-dir /path/to/causalphys --gpu-list 0,1,2,3 \
  --model-path "$MODEL" --prompt-style cot \
  --n-video-frames 5 --max-new-tokens 512 \
  --out-dir outputs/evaluation/causalphys

PHYSEVOLVE_LEGACY_VIDEO_PACKING=0 \
bash scripts/eval/run_contphy_vlm_eval_multi_gpu.sh \
  --dataset-dir /path/to/contphy --gpu-list 0,1,2,3 \
  --model-path "$MODEL" --n-video-frames 10 --max-new-tokens 64 \
  --out-dir outputs/evaluation/contphy
```

Repeat with each baseline/control checkpoint, using separate output directories. For contextual models, pass local weights and `--model-family internvl`, `llama32_vision`, or `pixtral`; retain each benchmark's prompts, item set, and decoding settings. These adapters do not fetch non-Qwen model weights automatically.

CausalPhys and ContPhy support `--dry-run --validate-media --n-samples 8` through their single-process Python entry points. This checks data and decoding without loading a model. PhysBench can first be checked on its 200-item `val` split with actual inference. Full paper results use `test`, not `val`.

Sharded evaluation writes predictions, raw answer audits, configuration metadata, a merged `summary.json`, and a breakdown report. Resume only with identical model, sampling, prompt, and generation settings. Use a fresh output directory when changing protocols.

Run outputs stay under the ignored `outputs/` directory. The repository contains reference programs, fixtures, option layouts, and PhysBench scoring answers; generated training corpora, benchmark media, checkpoints, and prediction dumps are stored separately.

Two reported CausalPhys control runs have aggregate reports but no retained per-item predictions under the paper video-packing protocol. Rechecking those results at item level requires rerunning the corresponding checkpoints with that protocol.
