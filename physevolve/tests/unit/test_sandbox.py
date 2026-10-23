"""Unit tests for :mod:`physevolve.executor.sandbox`.

These tests check that prohibited operations fail and whitelisted operations work.
"""

from __future__ import annotations

import pytest

from physevolve.executor.sandbox import (
    RestrictedExecError,
    execute_proposer_code,
)


# ─── Happy paths ──────────────────────────────────────────────────────


def test_happy_path_build_scene_only() -> None:
    """Minimal Stage-1 source: defines ``build_scene`` and nothing else."""

    src = (
        "def build_scene(api):\n"
        "    api.add_sphere('ball', pos=(0.0, 0.0, 1.0), radius=0.1, mass=1.0)\n"
    )
    build_scene, get_vqa = execute_proposer_code(src, timeout_s=2.0)
    assert callable(build_scene)
    assert get_vqa is None


def test_happy_path_stage2_defines_both() -> None:
    """Stage-2 source returns both callables."""

    src = (
        "def build_scene(api):\n"
        "    pass\n"
        "\n"
        "def get_vqa(sim_state):\n"
        "    return ('q?', ['a', 'b', 'c', 'd'], 0)\n"
    )
    build_scene, get_vqa = execute_proposer_code(src, timeout_s=2.0)
    assert callable(build_scene)
    assert callable(get_vqa)


def test_allowed_builtins_are_callable() -> None:
    """Whitelisted builtins (``abs``, ``range``, ``sorted`` ...) work inside the sandbox."""

    src = (
        "def build_scene(api):\n"
        "    xs = sorted(range(5), reverse=True)\n"
        "    api._result = sum(xs) + abs(-7) + max(xs) + len(xs)\n"
    )
    build_scene, _ = execute_proposer_code(src, timeout_s=2.0)

    class _Dummy:
        pass

    dummy = _Dummy()
    build_scene(dummy)
    assert dummy._result == 0 + 1 + 2 + 3 + 4 + 7 + 4 + 5


# ─── Syntax errors propagate as-is ────────────────────────────────────


def test_syntax_error_propagates() -> None:
    """``ast.parse`` failure raises :class:`SyntaxError` (pipeline maps it)."""

    with pytest.raises(SyntaxError):
        execute_proposer_code("def build_scene(api:\n", timeout_s=2.0)


# ─── AST rejections ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "src",
    [
        "import os\n",
        "from os import path\n",
        "def build_scene(api):\n    import math\n",
    ],
    ids=["import_os", "from_import", "import_in_function"],
)
def test_rejects_imports(src: str) -> None:
    with pytest.raises(RestrictedExecError, match="import"):
        execute_proposer_code(src, timeout_s=2.0)


@pytest.mark.parametrize(
    "banned",
    ["open", "eval", "exec", "compile", "__import__", "globals", "locals", "getattr"],
)
def test_rejects_banned_names(banned: str) -> None:
    src = f"def build_scene(api):\n    {banned}\n"
    with pytest.raises(RestrictedExecError, match=banned):
        execute_proposer_code(src, timeout_s=2.0)


def test_rejects_dunder_attribute() -> None:
    """Block attribute access ending/starting with double underscores."""

    src = (
        "def build_scene(api):\n"
        "    x = (1).__class__\n"
    )
    with pytest.raises(RestrictedExecError, match="dunder"):
        execute_proposer_code(src, timeout_s=2.0)


def test_rejects_missing_build_scene() -> None:
    """Source that parses cleanly but omits ``build_scene`` is rejected."""

    src = "x = 1\n"
    with pytest.raises(RestrictedExecError, match="build_scene"):
        execute_proposer_code(src, timeout_s=2.0)


def test_rejects_non_callable_build_scene() -> None:
    src = "build_scene = 42\n"
    with pytest.raises(RestrictedExecError, match="build_scene"):
        execute_proposer_code(src, timeout_s=2.0)


def test_rejects_non_callable_get_vqa() -> None:
    src = (
        "def build_scene(api):\n    pass\n"
        "get_vqa = 42\n"
    )
    with pytest.raises(RestrictedExecError, match="get_vqa"):
        execute_proposer_code(src, timeout_s=2.0)


# ─── Restricted builtins at runtime ──────────────────────────────────


def test_builtin_not_whitelisted_is_nameerror_at_runtime() -> None:
    """Even if the AST scan missed something, the restricted namespace shields us.

    We craft source that passes the AST scan (no ``eval`` / ``open`` name
    literally present) but references ``__builtins__`` through an
    attribute chain. That chain is blocked by the dunder check too —
    which is precisely the defence-in-depth we want.
    """

    src = "def build_scene(api):\n    __builtins__\n"
    with pytest.raises(RestrictedExecError, match="dunder"):
        execute_proposer_code(src, timeout_s=2.0)


# ─── Timeout ─────────────────────────────────────────────────────────


def test_sandbox_timeout_fires() -> None:
    """Infinite loop at import time triggers :class:`TimeoutError`."""

    src = "while True:\n    pass\n"
    with pytest.raises(TimeoutError):
        execute_proposer_code(src, timeout_s=0.2)


def test_sandbox_timeout_zero_disables_guard() -> None:
    """``timeout_s <= 0`` skips the SIGALRM guard entirely."""

    src = "def build_scene(api):\n    pass\n"
    build_scene, _ = execute_proposer_code(src, timeout_s=0.0)
    assert callable(build_scene)


def test_sandbox_timeout_restores_previous_handler() -> None:
    """After any timeout, the outer handler must be back in place."""

    import signal

    sentinel_called: list[int] = []

    def _sentinel(signum, frame):  # type: ignore[no-untyped-def]
        sentinel_called.append(signum)

    if not hasattr(signal, "SIGALRM"):
        pytest.skip("SIGALRM unsupported on this platform")

    prev = signal.signal(signal.SIGALRM, _sentinel)
    try:
        execute_proposer_code("def build_scene(api):\n    pass\n", timeout_s=0.5)
        # Sentinel must be restored.
        current = signal.getsignal(signal.SIGALRM)
        assert current is _sentinel
    finally:
        signal.signal(signal.SIGALRM, prev)
