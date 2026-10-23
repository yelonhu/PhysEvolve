"""Copy diagnostics for Stage-2 proposer code.

This module is intentionally observational: it reports exact AST repeats and
source similarity against golden fewshots / recent proposer code, but it does
not change rewards. Keep it lightweight so it can run on the training hot path.
"""

from __future__ import annotations

import ast
import hashlib
from collections import Counter, deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from statistics import mean
from typing import Optional, Sequence

from physevolve.inference.parsing import extract_propose_block


@dataclass(frozen=True, slots=True)
class Stage2CopyDiagnostic:
    """Diagnostic result for one parsed Stage-2 proposer program."""

    scene_kind: str
    ast_hash: str
    ast_dump: str
    ast_valid: bool
    exact_fewshot: bool
    similarity_to_fewshot: Optional[float]
    recent_repeat: bool
    max_similarity_to_recent: Optional[float]


class Stage2CopyDiagnostics:
    """Track recent parseable proposer ASTs and compare new programs."""

    def __init__(self, *, max_recent: int = 64) -> None:
        if max_recent <= 0:
            raise ValueError("max_recent must be positive")
        self._recent: deque[tuple[str, str, str]] = deque(maxlen=int(max_recent))

    def analyze(self, code: str, *, scene_kind: str) -> Stage2CopyDiagnostic:
        """Return copy diagnostics for ``code`` without mutating history."""

        ast_dump, ast_hash, ast_valid = canonical_ast_fingerprint(code)
        golden_asts = _golden_asts_for_scene_kind(scene_kind)
        similarity_to_fewshot: Optional[float] = None
        exact_fewshot = False
        if golden_asts:
            exact_fewshot = ast_valid and any(
                ast_hash == golden_hash for _golden_dump, golden_hash in golden_asts
            )
            similarity_to_fewshot = max(
                _similarity(ast_dump, golden_dump)
                for golden_dump, _golden_hash in golden_asts
            )

        recent_hashes = {item[1] for item in self._recent}
        recent_repeat = ast_valid and ast_hash in recent_hashes
        recent_similarities = [_similarity(ast_dump, item[2]) for item in self._recent]
        max_similarity_to_recent = (
            max(recent_similarities) if recent_similarities else None
        )

        return Stage2CopyDiagnostic(
            scene_kind=str(scene_kind),
            ast_hash=ast_hash,
            ast_dump=ast_dump,
            ast_valid=ast_valid,
            exact_fewshot=exact_fewshot,
            similarity_to_fewshot=similarity_to_fewshot,
            recent_repeat=recent_repeat,
            max_similarity_to_recent=max_similarity_to_recent,
        )

    def remember(self, diagnostic: Stage2CopyDiagnostic) -> None:
        """Add one parseable proposer program to the recent-code window."""

        if diagnostic.ast_valid:
            self._recent.append(
                (diagnostic.scene_kind, diagnostic.ast_hash, diagnostic.ast_dump)
            )


def diagnostic_similarity(
    left: Stage2CopyDiagnostic,
    right: Stage2CopyDiagnostic,
) -> float:
    """Return AST-dump similarity between two diagnostic records."""

    return _similarity(left.ast_dump, right.ast_dump)


def canonical_ast_fingerprint(code: str) -> tuple[str, str, bool]:
    """Return ``(canonical_ast_dump, sha256, ast_valid)`` for Python source."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        dump = "\n".join(line.rstrip() for line in code.strip().splitlines())
        return dump, _hash_text(dump), False
    dump = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return dump, _hash_text(dump), True


def summarize_copy_diagnostics(
    diagnostics: Sequence[Stage2CopyDiagnostic],
) -> dict[str, object]:
    """Aggregate per-step copy diagnostics into flat metrics."""

    if not diagnostics:
        return {
            "stage2_copy_diagnostic_count": 0,
            "stage2_copy_exact_fewshot_count": 0,
            "stage2_copy_recent_repeat_count": 0,
            "stage2_copy_similarity_to_fewshot_mean": 0.0,
            "stage2_copy_similarity_to_fewshot_max": 0.0,
            "stage2_copy_similarity_to_recent_mean": 0.0,
            "stage2_copy_similarity_to_recent_max": 0.0,
            "stage2_copy_exact_fewshot_by_scene_kind": {},
        }

    fewshot_values = [
        float(d.similarity_to_fewshot)
        for d in diagnostics
        if d.similarity_to_fewshot is not None
    ]
    recent_values = [
        float(d.max_similarity_to_recent)
        for d in diagnostics
        if d.max_similarity_to_recent is not None
    ]
    by_scene = Counter(
        d.scene_kind for d in diagnostics if d.exact_fewshot and d.scene_kind
    )
    return {
        "stage2_copy_diagnostic_count": len(diagnostics),
        "stage2_copy_exact_fewshot_count": sum(
            1 for d in diagnostics if d.exact_fewshot
        ),
        "stage2_copy_recent_repeat_count": sum(
            1 for d in diagnostics if d.recent_repeat
        ),
        "stage2_copy_similarity_to_fewshot_mean": _mean_or_zero(fewshot_values),
        "stage2_copy_similarity_to_fewshot_max": max(fewshot_values, default=0.0),
        "stage2_copy_similarity_to_recent_mean": _mean_or_zero(recent_values),
        "stage2_copy_similarity_to_recent_max": max(recent_values, default=0.0),
        "stage2_copy_exact_fewshot_by_scene_kind": dict(sorted(by_scene.items())),
    }


@lru_cache(maxsize=16)
def _golden_asts_for_scene_kind(scene_kind: str) -> tuple[tuple[str, str], ...]:
    if not scene_kind:
        return ()
    from physevolve.skills.scene_templates import (
        fewshot_examples_for,
        get_scene_template_registry,
    )

    template = get_scene_template_registry().lookup(scene_kind)
    out: list[tuple[str, str]] = []
    for example in fewshot_examples_for(template):
        code = extract_propose_block(example)
        if code is None:
            continue
        dump, digest, valid = canonical_ast_fingerprint(code)
        if valid:
            out.append((dump, digest))
    return tuple(out)


def _similarity(left: str, right: str) -> float:
    return float(SequenceMatcher(None, left, right).ratio())


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mean_or_zero(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


__all__ = [
    "Stage2CopyDiagnostic",
    "Stage2CopyDiagnostics",
    "canonical_ast_fingerprint",
    "diagnostic_similarity",
    "summarize_copy_diagnostics",
]
