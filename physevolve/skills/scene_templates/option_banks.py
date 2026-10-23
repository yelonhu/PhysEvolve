"""Weighted option layouts for executor-authored VQA items.

``data/task_bank/option_layout_bank.jsonl`` contains layouts and frequency
weights derived from the PhysBench catalog, including its test split. The bank
controls wording and slot order. Each generated item's answer is bound from
the scene template's simulator-derived ``answer_semantic``.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

from physevolve.skills.scene_templates import GeneratedVQA

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LAYOUT_BANK_JSONL = _REPO_ROOT / "data" / "task_bank" / "option_layout_bank.jsonl"

# ─── semantic alias tables (normalized lowercase, no trailing period) ─────

def _norm(text: str) -> str:
    return " ".join(text.lower().strip().rstrip(".").split())


def _alias_map(pairs: Sequence[tuple[str, str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for semantic, surface in pairs:
        out[_norm(surface)] = semantic
    return out


_REGION_ALIASES = _alias_map(
    (
        ("left", "the left pit"),
        ("left", "left pit"),
        ("right", "the right pit"),
        ("right", "right pit"),
        ("middle", "the center pit"),
        ("middle", "the central pit"),
        ("middle", "the middle pit"),
        ("middle", "middle pit"),
        ("middle", "center pit"),
        ("both", "both pits"),
        ("both", "both of the pits"),
        ("neither", "neither pit"),
        ("neither", "neither of the pits"),
        ("neither", "no pit"),
        ("neither", "will not drop into any pit"),
        ("neither", "it will not drop into any pit"),
        ("neither", "it will not drop into either pit"),
        ("neither", "bounce out"),
        ("neither", "another place"),
        ("none", "none of the above"),
        ("none", "none of these actions"),
        ("none", "it remains uncertain"),
        ("none", "it remains suspended"),
        ("uncertain", "uncertain"),
        ("uncertain", "can not answer"),
        ("uncertain", "cannot answer"),
        ("uncertain", "maybe"),
        ("uncertain", "cannot be determined"),
    )
)

_YESNO_ALIASES = _alias_map(
    (
        ("yes", "yes"),
        ("no", "no"),
        ("uncertain", "uncertain"),
        ("uncertain", "can not answer"),
        ("uncertain", "cannot answer"),
        ("uncertain", "maybe"),
        ("uncertain", "cannot be determined"),
        ("uncertain", "it depends"),
    )
)

_POSE_ALIASES = _alias_map(
    (
        ("standing_upright", "standing upright"),
        ("leaning", "leaning"),
        ("lying_horizontally", "lying horizontally"),
        ("floating", "floating"),
        ("floating", "floating mid-air"),
        ("floating", "floating in mid-air"),
        ("floating", "floating in the air"),
        ("floating", "hanging in mid-air"),
        ("floating", "suspended in midair"),
        ("floating", "floating midair"),
        # Distractor-only surfaces (never GT); bind only if answer_semantic matches.
        ("inverted", "inverted"),
        ("inverted", "upside down"),
        ("tilting", "tilting slightly"),
        ("tilting", "slightly tilted"),
        ("tilting", "tilted to the side"),
        ("broken", "broken into pieces"),
        ("broken", "broken"),
        ("broken", "collapsed"),
        ("hanging", "hanging vertically"),
        ("submerged", "partially submerged"),
    )
)

_INVERSE_ALIASES = _alias_map(
    (
        ("remove_a", "remove the {wall_a} and other balls"),
        ("remove_b", "remove the {wall_b} and other balls"),
        ("remove_both", "remove both floating walls and other balls"),
        ("remove_both", "remove both the {wall_a} and {wall_b} and other balls"),
        ("remove_both", "both of the above"),
        ("remove_both", "both a and b"),
        ("none", "none of the above"),
        ("none", "do nothing"),
    )
)

_RELATION_YESNO_ALIASES = _YESNO_ALIASES

_MOTION_BUCKET_ALIASES = _alias_map(
    (
        ("stop", "it comes to a stop"),
        ("stop", "comes to a stop"),
        ("slow_down", "it slows down but keeps moving"),
        ("slow_down", "slows down but keeps moving"),
        ("speed_up", "it speeds up"),
        ("speed_up", "speeds up"),
        ("same_speed", "it continues at roughly the same speed"),
        ("same_speed", "continues at roughly the same speed"),
    )
)

_SPEED_PROFILE_ALIASES = _alias_map(
    (
        ("accelerate", "it accelerates"),
        ("accelerate", "accelerates"),
        ("decelerate", "it decelerates"),
        ("decelerate", "decelerates"),
        ("accelerate_then_decelerate", "it first accelerates and then decelerates"),
        ("accelerate_then_decelerate", "first accelerates and then decelerates"),
        ("constant", "it remains constant"),
        ("constant", "remains constant"),
    )
)

_PAIRWISE_SPEED_ALIASES = _alias_map(
    (
        ("same", "both objects have the same speed"),
        ("crossover", "eventually surpasses"),
        ("crossover", "surpasses it"),
    )
)


@dataclass(frozen=True, slots=True)
class _WeightedLayout:
    options: tuple[str, ...]
    weight: int


# Embedded high-frequency fallbacks when the layout bank is missing or sparse.
_FALLBACK_LAYOUTS: dict[str, tuple[_WeightedLayout, ...]] = {
    "throwing_final_destination": (
        _WeightedLayout(
            ("The left pit", "The right pit", "None of the above", "Both pits"),
            54,
        ),
        _WeightedLayout(
            ("The left pit", "The right pit", "Both pits", "None of the above"),
            17,
        ),
        _WeightedLayout(
            (
                "The left pit",
                "The right pit",
                "The center pit",
                "None of the above",
            ),
            12,
        ),
        _WeightedLayout(
            (
                "The left pit",
                "The right pit",
                "Neither pit",
                "Both pits",
            ),
            8,
        ),
        _WeightedLayout(
            (
                "The left pit",
                "The right pit",
                "It will not drop into any pit",
                "None of the above",
            ),
            4,
        ),
    ),
    "counterfactual_destination": (
        _WeightedLayout(
            ("The left pit", "The right pit", "None of the above", "Both pits"),
            54,
        ),
        _WeightedLayout(
            ("The left pit", "The right pit", "Both pits", "None of the above"),
            17,
        ),
        _WeightedLayout(
            (
                "The left pit",
                "The right pit",
                "The center pit",
                "None of the above",
            ),
            12,
        ),
        _WeightedLayout(
            ("The left pit", "The right pit", "Neither pit", "Both pits"),
            8,
        ),
    ),
    "throwing_current_run_judgment": (
        _WeightedLayout(("Yes", "No", "Can not answer", "Uncertain"), 4),
        _WeightedLayout(("Yes", "No", "Maybe", "Can not answer"), 2),
        _WeightedLayout(("Yes", "No", "Cannot answer", "Maybe"), 2),
    ),
    "inverse_intervention": (
        _WeightedLayout(
            (
                "Remove the {wall_a} and other balls",
                "Remove the {wall_b} and other balls",
                "Both of the above",
                "None of the above",
            ),
            20,
        ),
        _WeightedLayout(
            (
                "Remove the {wall_a} and other balls",
                "Remove the {wall_b} and other balls",
                "None of the above",
                "Remove both floating walls and other balls",
            ),
            15,
        ),
    ),
    "collision_final_state": (
        _WeightedLayout(
            (
                "Standing upright",
                "Leaning",
                "Lying horizontally",
                "Floating",
            ),
            8,
        ),
        _WeightedLayout(
            (
                "Standing upright",
                "Leaning",
                "Lying horizontally",
                "Inverted",
            ),
            6,
        ),
        _WeightedLayout(
            (
                "Standing upright",
                "Leaning",
                "Lying horizontally",
                "Tilting slightly",
            ),
            3,
        ),
        _WeightedLayout(
            (
                "Standing upright",
                "Leaning",
                "Lying horizontally",
                "Broken into pieces",
            ),
            2,
        ),
        _WeightedLayout(
            (
                "Standing upright",
                "Leaning",
                "Lying horizontally",
                "Hanging vertically",
            ),
            2,
        ),
        _WeightedLayout(
            (
                "Standing upright",
                "Leaning",
                "Lying horizontally",
                "Upside down",
            ),
            2,
        ),
    ),
    "collision_final_relation": (
        _WeightedLayout(("Yes", "No", "Sometimes", "Uncertain"), 11),
    ),
    "motion_state_change": (
        _WeightedLayout(
            (
                "It comes to a stop",
                "It slows down but keeps moving",
                "It speeds up",
                "It continues at roughly the same speed",
            ),
            4,
        ),
        _WeightedLayout(
            (
                "It accelerates.",
                "It decelerates.",
                "It first accelerates and then decelerates.",
                "It remains constant.",
            ),
            4,
        ),
    ),
}

_BANKS: dict[str, tuple[_WeightedLayout, ...]] | None = None


def _is_binary_judgment_layout(options: Sequence[str]) -> bool:
    """Accept only explicit yes/no layouts for judgment questions."""

    normalized = {_norm(option) for option in options}
    return {"yes", "no"} <= normalized and not any(
        "pit" in option for option in normalized
    )


def _load_banks() -> dict[str, tuple[_WeightedLayout, ...]]:
    global _BANKS
    if _BANKS is not None:
        return _BANKS

    merged: dict[str, list[_WeightedLayout]] = {
        family: list(layouts) for family, layouts in _FALLBACK_LAYOUTS.items()
    }

    if _LAYOUT_BANK_JSONL.is_file():
        with _LAYOUT_BANK_JSONL.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                family = str(row.get("family") or "")
                template = row.get("options")
                if not family or not isinstance(template, list) or len(template) != 4:
                    continue
                weight = int(row.get("weight") or 1)
                opts = tuple(str(o).strip() for o in template)
                if any(not o for o in opts):
                    continue
                if (
                    family == "throwing_current_run_judgment"
                    and not _is_binary_judgment_layout(opts)
                ):
                    # The family bank also contains destination layouts.
                    # Keep those out of the binary bank even though the runtime
                    # yes/no guard below currently preserves template options.
                    continue
                merged.setdefault(family, []).append(_WeightedLayout(opts, weight))

    _BANKS = {
        family: tuple(layouts)
        for family, layouts in merged.items()
        if layouts
    }
    return _BANKS


def _seed_int(*parts: object) -> int:
    sig = "|".join(repr(p) for p in parts)
    return int(hashlib.md5(sig.encode("utf-8")).hexdigest(), 16) % (2 ** 32)


def _layout_supports_semantic(
    layout: Sequence[str],
    answer_semantic: str,
    *,
    alias_table: Mapping[str, str],
    context: Mapping[str, str],
) -> bool:
    return (
        bind_correct_index(
            layout,
            answer_semantic,
            alias_table=alias_table,
            context=context,
        )
        is not None
    )


def _layout_is_coherent(
    layout: Sequence[str],
    *,
    alias_table: Mapping[str, str],
    context: Mapping[str, str],
) -> bool:
    """Require every slot to resolve under the same alias table.

    Prevents mixing yes/no stems with destination-style bank layouts that
    happen to fuzzy-match a short semantic like ``\"no\"``.
    """

    for slot in layout:
        if _slot_semantic(slot, alias_table, context=context) is None:
            return False
    return True


def _eligible_layouts(
    family: str,
    answer_semantic: str,
    *,
    phrasing: str,
    alias_table: Mapping[str, str],
    context: Mapping[str, str],
) -> tuple[_WeightedLayout, ...]:
    del phrasing  # reserved for future per-row phrasing filters
    banks = _load_banks().get(family) or _FALLBACK_LAYOUTS.get(family, ())
    eligible = [
        bank
        for bank in banks
        if _layout_is_coherent(
            bank.options, alias_table=alias_table, context=context
        )
        and _layout_supports_semantic(
            bank.options,
            answer_semantic,
            alias_table=alias_table,
            context=context,
        )
    ]
    # Empty means caller must fall back to template options — never the full
    # unfiltered family pool, which may lack a compatible answer semantic.
    return tuple(eligible)


def sample_option_layout(
    family: str,
    *,
    phrasing: str,
    seed_key: tuple[str, str, Mapping[str, object]],
    salt: str = "",
    answer_semantic: Optional[str] = None,
    context: Optional[Mapping[str, str]] = None,
) -> list[str]:
    """Return a bank-weighted 4-option layout for ``family``.

    Raises ``KeyError`` when no layout is eligible for ``answer_semantic``.
    ``phrasing`` is reserved for future per-row filters.
    """

    alias_table = _alias_table_for_family(family, phrasing)
    ctx = dict(context or {})
    if answer_semantic:
        banks = _eligible_layouts(
            family,
            answer_semantic,
            phrasing=phrasing,
            alias_table=alias_table,
            context=ctx,
        )
    else:
        banks = _load_banks().get(family) or _FALLBACK_LAYOUTS.get(family, ())
    if not banks:
        raise KeyError(f"no option bank for family {family!r}")

    key, _phrasing, params = seed_key
    seed = _seed_int(salt, key, phrasing, tuple(sorted(params.items())))
    weights = [max(1, b.weight) for b in banks]
    choice = random.Random(seed).choices(range(len(banks)), weights=weights, k=1)[0]
    return list(banks[choice].options)


def _fill_placeholders(text: str, context: Mapping[str, str]) -> str:
    out = text
    for key, value in context.items():
        out = out.replace("{" + key + "}", value)
    return out


_SHORT_ALIAS_MAX_LEN = 3
_WORD_RE = re.compile(r"[a-z0-9]+")


def _slot_semantic(
    text: str,
    alias_table: Mapping[str, str],
    *,
    context: Mapping[str, str],
) -> Optional[str]:
    filled = _fill_placeholders(text, context)
    norm = _norm(filled)
    resolved_aliases = [
        (_norm(_fill_placeholders(alias, context)), semantic)
        for alias, semantic in alias_table.items()
    ]
    for alias_norm, semantic in resolved_aliases:
        if norm == alias_norm:
            return semantic
    # Prefer longer aliases first so \"none of the above\" beats \"no\".
    for alias_norm, semantic in sorted(
        resolved_aliases, key=lambda item: len(item[0]), reverse=True
    ):
        if len(alias_norm) <= _SHORT_ALIAS_MAX_LEN:
            # Short tokens (yes/no): exact equality only — never substring
            # (\"no\" ⊂ \"not\" / \"cannot\" caused GT letter flips).
            if norm == alias_norm:
                return semantic
            continue
        if alias_norm in norm or norm in alias_norm:
            return semantic
        # Word-boundary containment for multi-word aliases.
        alias_words = _WORD_RE.findall(alias_norm)
        slot_words = _WORD_RE.findall(norm)
        if alias_words and all(w in slot_words for w in alias_words):
            return semantic
    return None


def bind_correct_index(
    layout: Sequence[str],
    answer_semantic: str,
    *,
    alias_table: Mapping[str, str],
    context: Mapping[str, str],
) -> Optional[int]:
    """Map ``answer_semantic`` to the index of the matching slot in ``layout``."""

    target = _norm(answer_semantic)
    for idx, slot in enumerate(layout):
        sem = _slot_semantic(slot, alias_table, context=context)
        if sem is not None and _norm(sem) == target:
            return idx
    return None


def _alias_table_for_family(family: str, phrasing: str) -> Mapping[str, str]:
    if family in {
        "throwing_final_destination",
        "counterfactual_destination",
    }:
        return _REGION_ALIASES
    if family == "throwing_current_run_judgment":
        return _YESNO_ALIASES
    if family == "inverse_intervention":
        return _INVERSE_ALIASES
    if family == "collision_final_state":
        return _POSE_ALIASES
    if family == "collision_final_relation" and phrasing == "binary_collide":
        return _RELATION_YESNO_ALIASES
    if family == "motion_state_change":
        if phrasing == "speed_profile":
            return _SPEED_PROFILE_ALIASES
        if phrasing == "pairwise_speed":
            return _PAIRWISE_SPEED_ALIASES
        return _MOTION_BUCKET_ALIASES
    return _REGION_ALIASES


def _layout_context(gen: GeneratedVQA) -> dict[str, str]:
    spec = gen.spec
    ctx: dict[str, str] = {}
    wall_a = spec.get("wall_a_phrase")
    wall_b = spec.get("wall_b_phrase")
    if isinstance(wall_a, str):
        ctx["wall_a"] = wall_a
    if isinstance(wall_b, str):
        ctx["wall_b"] = wall_b
    region_kind = spec.get("region_kind")
    if isinstance(region_kind, str):
        ctx["region_kind"] = region_kind
    return ctx


_YESNO_SEMANTICS = frozenset({"yes", "no"})
_YESNO_PHRASINGS = frozenset({"yes_no", "wrong_pit", "binary_collide"})


def _keep_template_options(gen: GeneratedVQA, answer_semantic: str) -> bool:
    """Skip bank remap for binary yes/no stems (avoid destination pollution)."""

    if _norm(answer_semantic) in _YESNO_SEMANTICS:
        return True
    if str(gen.phrasing) in _YESNO_PHRASINGS:
        return True
    return False


def apply_option_layout(
    gen: GeneratedVQA,
    *,
    scene_kind: str,
    salt: str,
    shuffle_fallback: bool = False,
) -> Optional[tuple[list[str], int]]:
    """Apply a bank layout to ``gen``; return ``(options, correct_idx)`` or skip.

    When ``answer_semantic`` is absent from ``gen.spec``, falls back to the
    template's canonical ``gen.options`` / ``gen.correct_idx`` without shuffle
    (unless ``shuffle_fallback`` is True for debug).

    Yes/no judgment stems always keep template options: the layout bank for
    ``throwing_current_run_judgment`` can mix destination layouts that would
    corrupt stem/option coherence and GT letters.
    """

    spec = gen.spec
    answer_semantic = spec.get("answer_semantic")
    if not isinstance(answer_semantic, str) or not answer_semantic.strip():
        if shuffle_fallback:
            from physevolve.skills.scene_templates.render_utils import (
                shuffle_options_deterministic,
            )

            return shuffle_options_deterministic(
                list(gen.options),
                int(gen.correct_idx),
                seed_key=(f"{scene_kind}:{gen.family}", gen.phrasing, gen.spec),
                salt=salt,
            )
        return list(gen.options), int(gen.correct_idx)

    if _keep_template_options(gen, answer_semantic.strip()):
        return list(gen.options), int(gen.correct_idx)

    seed_key = (f"{scene_kind}:{gen.family}", gen.phrasing, spec)
    context = _layout_context(gen)
    alias_table = _alias_table_for_family(gen.family, gen.phrasing)
    try:
        layout = sample_option_layout(
            gen.family,
            phrasing=gen.phrasing,
            seed_key=seed_key,
            salt=salt,
            answer_semantic=answer_semantic.strip(),
            context=context,
        )
    except KeyError:
        return list(gen.options), int(gen.correct_idx)

    options = [_fill_placeholders(slot, context) for slot in layout]
    correct_idx = bind_correct_index(
        layout,
        answer_semantic,
        alias_table=alias_table,
        context=context,
    )
    if correct_idx is None:
        # Body-specific layouts (e.g. pairwise_speed) keep template options.
        return list(gen.options), int(gen.correct_idx)
    if len(set(_norm(o) for o in options)) != len(options):
        return list(gen.options), int(gen.correct_idx)
    return options, correct_idx


__all__ = [
    "apply_option_layout",
    "bind_correct_index",
    "sample_option_layout",
]
