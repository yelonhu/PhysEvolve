#!/usr/bin/env python3
"""Compatibility shim. Canonical Stage-2 entry: ``scripts/train/_entry.py``.

Owner commands that still call this path keep working. New wrappers live
under ``scripts/train/``. The real source is executed in this module's
namespace so CLI tests that monkeypatch helpers on the shim still work.
"""

from __future__ import annotations

from pathlib import Path

_ENTRY = Path(__file__).resolve().parents[1] / "train" / "_entry.py"
exec(compile(_ENTRY.read_text(encoding="utf-8"), str(_ENTRY), "exec"), globals())
