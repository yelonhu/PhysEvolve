#!/usr/bin/env python3
"""Launch PhysEvolve's zero-shot ContPhy reference evaluation.

Default model:
  Qwen/Qwen3-VL-8B-Instruct

"""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.contphy.evaluator import main


if __name__ == "__main__":
    raise SystemExit(main())
