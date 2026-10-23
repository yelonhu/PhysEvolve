# Golden reference programs

The demonstration pool contains 24 executable reference programs: 10 throwing, 8 collision, and 6 motion variants. The [manifest](manifest.json) lists their paths and scene kinds. Training uses these template pools throughout the 100-step recipe.

```bash
python scripts/smoke/list_golden_scenes.py --render-cmd
python scripts/smoke/render_golden_scene.py \
  --scene data/golden_scenes/stage2/01_throwing_pit_board__v01.py \
  --out-dir outputs/demo --seed 7 --print-vqa --print-branches
```

The renderer writes a baseline video, per-family questions/options/answers, and branch diagnostics. `--no-video` suppresses video files; `--no-save-artifacts` prints diagnostics only.

Active scene templates load the listed programs and their `Plan:` descriptions into the demonstration pool at import time. Editing these files changes the Proposer examples. Run scene and prompt checks after making changes, and use the recorded version when reproducing a published run.
