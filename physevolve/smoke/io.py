"""Filesystem helpers for immutable smoke-run artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def create_run_dir(out_root: Path, run_id: str) -> Path:
    """Create and return an immutable smoke run directory.

    Raises:
        FileExistsError: if the run directory already exists.
    """

    run_dir = Path(out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_text(path: Path, text: str) -> None:
    """Atomically write UTF-8 text."""

    _atomic_write_text(Path(path), text)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write a small YAML-compatible config file.

    The implementation emits JSON, which is valid YAML 1.2 and avoids
    introducing a runtime dependency in smoke orchestration.
    """

    write_text(
        Path(path),
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
    )


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(item), sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file. Missing files return an empty list."""

    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_json(path: Path, payload: Any) -> None:
    """Atomically write a JSON value."""

    write_text(
        Path(path),
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: tuple[str, ...]) -> None:
    """Write a simple CSV file for human-readable aggregate tables."""

    lines = [",".join(columns)]
    for row in rows:
        values = [_csv_cell(row.get(col, "")) for col in columns]
        lines.append(",".join(values))
    write_text(Path(path), "\n".join(lines) + "\n")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        finally:
            raise


def _csv_cell(value: Any) -> str:
    text = str(value)
    if any(ch in text for ch in (",", "\"", "\n")):
        return "\"" + text.replace("\"", "\"\"") + "\""
    return text


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "append_jsonl",
    "create_run_dir",
    "read_jsonl",
    "write_csv",
    "write_json",
    "write_text",
    "write_yaml",
]
