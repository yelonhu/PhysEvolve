"""Proposer code sandbox.

Implements :func:`execute_proposer_code` with these restrictions:

* AST pre-scan rejects ``import``, dunder attribute access, ``open``,
  ``eval``, ``exec``, ``compile``, ``__import__``, ``getattr``/``setattr``
  etc. before any code runs.
* ``exec`` runs under a whitelist-only ``__builtins__`` dict — so even
  if a banned name slipped past the AST check it would be a
  :class:`NameError` at call time.
* ``signal.SIGALRM`` + ``signal.setitimer`` wraps the ``exec`` call with
  a wall-clock budget. The previous SIGALRM handler is saved and
  restored on exit so the guard composes with outer handlers (e.g.
  pytest's own timeout plugins).

Platforms without ``SIGALRM`` (notably Windows) silently skip the
timeout — a one-time :class:`UserWarning` is emitted so this is not a
silent footgun in production.
"""

from __future__ import annotations

import ast
import signal
import warnings
from contextlib import contextmanager
from typing import Callable, Iterator, Mapping, Optional

# ─── Public exception ─────────────────────────────────────────────────


class RestrictedExecError(RuntimeError):
    """Raised when a Proposer source string violates the sandbox policy.

    The pipeline layer maps this to
    :attr:`~physevolve.engine.failure.FailureReason.FAILED_SEMANTIC`.
    Subclass of :class:`RuntimeError` rather than :class:`SyntaxError`
    because syntactic validity is already guaranteed by the time we
    reach the AST scan — a banned node is a *semantic* violation of the
    DSL contract, not a parse error.
    """


# ─── Restricted builtins whitelist ───────────────────

_ALLOWED_BUILTIN_NAMES: frozenset[str] = frozenset(
    {
        "abs",
        "min",
        "max",
        "sum",
        "len",
        "range",
        "enumerate",
        "zip",
        "list",
        "tuple",
        "dict",
        "set",
        "float",
        "int",
        "str",
        "bool",
        "True",
        "False",
        "None",
        "print",
        "round",
        "pow",
        "divmod",
        "sorted",
        "reversed",
        "any",
        "all",
    }
)


# Names that must never appear in Proposer source, even under the
# restricted builtins. This catches e.g. someone calling ``eval`` via
# a module-level reference and keeps error messages specific.
_BANNED_NAMES: frozenset[str] = frozenset(
    {
        "open",
        "eval",
        "exec",
        "compile",
        "__import__",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "breakpoint",
        "input",
    }
)


def _build_restricted_builtins(
    allowed: Optional[set[str]] = None,
) -> dict[str, object]:
    """Construct the ``__builtins__`` dict for the restricted namespace."""

    import builtins

    names = allowed if allowed is not None else _ALLOWED_BUILTIN_NAMES
    out: dict[str, object] = {}
    for name in names:
        if hasattr(builtins, name):
            out[name] = getattr(builtins, name)
        elif name in ("True", "False", "None"):
            # Literals are not attributes of ``builtins``; exec will
            # resolve them from the parser regardless, but we keep the
            # names documented as allowed.
            out[name] = eval(name)  # noqa: S307 — literal whitelist only
    return out


# ─── AST inspection ───────────────────────────────────────────────────


def _validate_ast(tree: ast.AST) -> None:
    """Walk the AST and reject forbidden constructs.

    Raises :class:`RestrictedExecError` on the first violation found so
    error messages are specific to a single node.
    """

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise RestrictedExecError(
                f"import statements are not allowed "
                f"(line {getattr(node, 'lineno', '?')})"
            )
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("__") or attr.endswith("__"):
                raise RestrictedExecError(
                    f"dunder attribute access {attr!r} is banned "
                    f"(line {getattr(node, 'lineno', '?')})"
                )
        if isinstance(node, ast.Name):
            if node.id in _BANNED_NAMES:
                raise RestrictedExecError(
                    f"name {node.id!r} is banned in Proposer code "
                    f"(line {getattr(node, 'lineno', '?')})"
                )
            # Also reject dunder *names* (e.g. ``__builtins__``) — these
            # are a back-door into the restricted namespace if somehow
            # exposed. Harmless to block since Proposer code has no
            # legitimate reason to touch them.
            if node.id.startswith("__") or node.id.endswith("__"):
                raise RestrictedExecError(
                    f"dunder name {node.id!r} is banned "
                    f"(line {getattr(node, 'lineno', '?')})"
                )


# ─── Timeout guard ────────────────────────────────────────────────────


def _raise_timeout(signum, frame):  # type: ignore[no-untyped-def]
    """Signal handler that converts SIGALRM into a :class:`TimeoutError`."""

    raise TimeoutError("proposer code exceeded wall-clock budget")


_PLATFORM_WARNED: bool = False


@contextmanager
def _timeout_guard(seconds: float) -> Iterator[None]:
    """Install SIGALRM for ``seconds`` and restore the previous handler on exit.

    On platforms without ``SIGALRM`` this degrades to a no-op; a
    :class:`UserWarning` is emitted once per process so tests / ops see
    the limitation without spamming logs.
    """

    global _PLATFORM_WARNED

    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        if not _PLATFORM_WARNED:
            warnings.warn(
                "SIGALRM not supported on this platform; sandbox timeout disabled",
                stacklevel=2,
            )
            _PLATFORM_WARNED = True
        yield
        return

    if seconds <= 0.0:
        # Zero / negative budget means "no guard" rather than "fire
        # immediately" — simplifies test setup.
        yield
        return

    prev_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        # Disable the timer FIRST so a stray SIGALRM cannot fire after
        # we have already restored the previous handler.
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prev_handler)


# ─── Public entry point ───────────────────────────────────────────────


def execute_proposer_code(
    source: str,
    timeout_s: float = 5.0,
    allowed_builtins: Optional[set[str]] = None,
) -> tuple[Callable, Optional[Callable]]:
    """Parse, validate, and exec a Proposer source string.

    Args:
        source: Python source defining ``build_scene(api)`` and
            optionally ``get_vqa(sim_state)``.
        timeout_s: Wall-clock budget for the ``exec`` call. ``0`` or
            negative disables the timer. Ignored on non-SIGALRM
            platforms (with a one-time warning).
        allowed_builtins: Optional override of the builtin whitelist.
            If ``None`` uses the module default set.

    Returns:
        ``(build_scene, get_vqa_or_None)``.

    Raises:
        SyntaxError: ``ast.parse`` failed. The pipeline maps this to
            :attr:`FailureReason.FAILED_SYNTAX`.
        RestrictedExecError: AST scan found a banned node, or the
            executed code did not define ``build_scene``. The pipeline
            maps this to :attr:`FailureReason.FAILED_SEMANTIC`.
        TimeoutError: The ``exec`` call overran ``timeout_s``. The
            pipeline maps this to :attr:`FailureReason.FAILED_TIMEOUT`.
    """

    build_scene, get_vqa, _vqa_spec, _manifest = execute_proposer_code_with_spec(
        source, timeout_s=timeout_s, allowed_builtins=allowed_builtins
    )
    return build_scene, get_vqa


def execute_proposer_code_with_spec(
    source: str,
    timeout_s: float = 5.0,
    allowed_builtins: Optional[set[str]] = None,
) -> tuple[
    Callable,
    Optional[Callable],
    Optional[dict[str, object]],
    Optional[dict[str, object]],
]:
    """Like :func:`execute_proposer_code` but also returns the VQA specs.

    Stage-2 proposer source defines EITHER:

    * a single-question ``VQA_SPEC = {...}`` literal (typed-skill path), OR
    * a multi-question ``SCENE_MANIFEST = {...}`` literal (scene-template
      path: one reusable scene -> many auto-derived VQAs).

    Both are returned raw; the executor shape-validates them. A non-dict
    value is forwarded wrapped in an ``__invalid_*_type__`` sentinel so the
    pipeline can map the mismatch to ``FAILED_VQA``.

    Returns:
        ``(build_scene, get_vqa_or_None, vqa_spec_or_None, scene_manifest_or_None)``.
    """

    tree = ast.parse(source)
    _validate_ast(tree)
    code = compile(tree, filename="<proposer>", mode="exec")

    builtins_dict = _build_restricted_builtins(allowed_builtins)
    namespace: dict[str, object] = {"__builtins__": builtins_dict}

    with _timeout_guard(timeout_s):
        exec(code, namespace)  # noqa: S102 — restricted namespace by design

    build_scene = namespace.get("build_scene")
    if not callable(build_scene):
        raise RestrictedExecError(
            "proposer source must define a callable `build_scene(api)`"
        )

    get_vqa = namespace.get("get_vqa")
    if get_vqa is not None and not callable(get_vqa):
        raise RestrictedExecError(
            "`get_vqa` is defined but is not callable"
        )

    vqa_spec = _extract_dict_literal(namespace, "VQA_SPEC")
    scene_manifest = _extract_dict_literal(namespace, "SCENE_MANIFEST")

    return build_scene, get_vqa, vqa_spec, scene_manifest


def _extract_dict_literal(
    namespace: Mapping[str, object], key: str
) -> Optional[dict[str, object]]:
    """Return a defensive copy of ``namespace[key]`` or a type sentinel.

    ``None`` when the symbol is absent; a shallow ``dict`` copy when it is a
    dict; otherwise ``{"__invalid_<key.lower()>_type__": <typename>}`` so the
    pipeline can reject non-dict literals as ``FAILED_VQA``.
    """

    raw = namespace.get(key)
    if raw is None:
        return None
    if isinstance(raw, dict):
        return dict(raw)
    return {f"__invalid_{key.lower()}_type__": type(raw).__name__}


__all__ = [
    "RestrictedExecError",
    "execute_proposer_code",
    "execute_proposer_code_with_spec",
]
