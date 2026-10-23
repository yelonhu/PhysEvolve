"""Put ``scripts/smoke`` and the repo root at the front of ``sys.path``.

Smoke scripts are run as files (``python scripts/smoke/foo.py``), not as
``python -m scripts.smoke.foo``. The ``scripts`` tree has no ``__init__.py``,
so ``from scripts.smoke...`` is unreliable when another ``scripts`` package
or a non-canonical ``sys.path`` order shadows the repo.
"""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_smoke_sys_path() -> tuple[Path, Path]:
    """Ensure repo root and this directory precede other ``sys.path`` entries."""

    smoke_dir = Path(__file__).resolve().parent
    repo_root = smoke_dir.parents[1]
    for path in (repo_root, smoke_dir):
        entry = str(path)
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    return repo_root, smoke_dir
