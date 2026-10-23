"""Score a ``predictions.json`` against the PhysBench answer file.

This implementation is written from scratch (no code lifted from the
upstream repository) but produces the same accuracy decomposition that
PhysBench's leaderboard reports:

* Overall accuracy.
* Per-``task_type`` accuracy (property / relationships / scene / dynamics).
* Per-``sub_type`` accuracy (19 subcategories).
* Per-``ability_type`` accuracy (8 abilities).

The match function is intentionally permissive: an answer is correct if
its first non-whitespace character matches the gold letter
(case-insensitive). This tolerates common VLM artefacts such as
``"Answer: A"`` or ``"(A)"`` without us parsing them ourselves.

Answer file schema (``val_answer.json`` / ``test_answer.json``)::

    [{"idx": <int>,
      "answer": "A",
      "task_type": "dynamics",
      "sub_type":  "collision",
      "ability_type": "prediction",
      "mode":     "general"},
     ...]

Predictions file schema (what :mod:`.runner` writes)::

    [{"idx": <int>, "answer": <str | null>}, ...]
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


# ─── Display ordering (mirrors the order used on the PhysBench site) ─


TASK_TYPE_ORDER: tuple[str, ...] = (
    "property", "relationships", "scene", "dynamics",
)

SUB_TYPE_ORDER: tuple[str, ...] = (
    "number", "mass", "color", "attribute", "size",
    "location", "depth", "distance", "motion",
    "temperature", "viewpoint", "air", "light",
    "collision", "throwing", "manipulation", "fluid",
    "chemistry", "others",
)

ABILITY_TYPE_ORDER: tuple[str, ...] = (
    "identify", "comparison", "static", "dynamic",
    "perception", "prediction", "judgment", "reasoning",
)


# ─── Types ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CategoryStat:
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return (self.correct / self.total) if self.total > 0 else 0.0


@dataclass(frozen=True, slots=True)
class ScoreReport:
    """Aggregate result of :func:`score`.

    ``covered`` is the number of predictions that were joined against a
    known gold answer (items missing from the answer file are ignored,
    matching the upstream behaviour).
    """

    overall: CategoryStat
    by_task_type: dict[str, CategoryStat]
    by_sub_type: dict[str, CategoryStat]
    by_ability_type: dict[str, CategoryStat]
    n_predicted: int
    n_covered: int
    n_unmatched: int

    def to_dict(self) -> dict[str, Any]:
        def _pack(stats: dict[str, CategoryStat]) -> dict[str, dict[str, float]]:
            return {
                k: {
                    "accuracy": round(v.accuracy * 100, 2),
                    "correct": v.correct,
                    "total": v.total,
                }
                for k, v in stats.items()
            }

        return {
            "overall": {
                "accuracy": round(self.overall.accuracy * 100, 2),
                "correct": self.overall.correct,
                "total": self.overall.total,
            },
            "by_task_type": _pack(self.by_task_type),
            "by_sub_type": _pack(self.by_sub_type),
            "by_ability_type": _pack(self.by_ability_type),
            "counts": {
                "n_predicted": self.n_predicted,
                "n_covered": self.n_covered,
                "n_unmatched": self.n_unmatched,
            },
        }


# ─── Core match ──────────────────────────────────────────────────────


def _first_letter(raw: Optional[str]) -> Optional[str]:
    """Best-effort extraction of the first A/B/C/D letter in ``raw``.

    Returns uppercase letter or ``None`` if no candidate is found.
    Unicode whitespace is stripped; other leading characters are not
    (so ``"A. answer"`` matches ``A`` but ``"the A"`` matches ``None`` —
    we do not want false positives from prose mentions).
    """

    if raw is None:
        return None
    stripped = str(raw).strip()
    if not stripped:
        return None
    # Accept forms like "A", "A.", "(A)", "Answer: A"
    # We check char-by-char until we find the first stand-alone letter.
    for i, ch in enumerate(stripped):
        if ch.upper() in {"A", "B", "C", "D"}:
            nxt = stripped[i + 1] if i + 1 < len(stripped) else ""
            prev = stripped[i - 1] if i > 0 else ""
            # Standalone letter: surrounded by non-letters (or at string edge)
            if not prev.isalpha() and not nxt.isalpha():
                return ch.upper()
    return None


def _match(pred: Optional[str], gold: str) -> bool:
    """True iff predicted answer resolves to the same letter as gold."""

    letter = _first_letter(pred)
    if letter is None:
        return False
    return letter == gold.strip().upper()


# ─── Loading ─────────────────────────────────────────────────────────


def _load_json(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"expected list at {path}, got {type(data).__name__}")
    return data


# ─── Public entry ────────────────────────────────────────────────────


def score(
    predictions_file: Path,
    answer_file: Path,
) -> ScoreReport:
    """Compute the leaderboard-style accuracy report.

    Predictions not present in the answer file are counted in
    ``n_unmatched`` but excluded from the accuracy denominators.
    """

    preds = _load_json(predictions_file)
    golds = _load_json(answer_file)

    gold_by_idx: dict[int, dict[str, Any]] = {
        int(row["idx"]): row for row in golds if "idx" in row
    }

    n_correct = 0
    n_total = 0
    n_unmatched = 0

    task_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [correct, total]
    sub_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    ability_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for row in preds:
        if not isinstance(row, dict):
            continue
        idx = row.get("idx")
        if idx is None:
            continue
        gold_row = gold_by_idx.get(int(idx))
        if gold_row is None:
            n_unmatched += 1
            continue

        pred_answer = row.get("answer")
        gold_answer = str(gold_row.get("answer", ""))
        if not gold_answer:
            continue

        hit = _match(pred_answer, gold_answer)
        n_total += 1
        if hit:
            n_correct += 1

        tt = gold_row.get("task_type") or "unknown"
        st = gold_row.get("sub_type") or "unknown"
        at = gold_row.get("ability_type") or "unknown"

        task_counts[tt][1] += 1
        sub_counts[st][1] += 1
        ability_counts[at][1] += 1
        if hit:
            task_counts[tt][0] += 1
            sub_counts[st][0] += 1
            ability_counts[at][0] += 1

    def _pack(counts: dict[str, list[int]]) -> dict[str, CategoryStat]:
        return {k: CategoryStat(correct=v[0], total=v[1]) for k, v in counts.items()}

    return ScoreReport(
        overall=CategoryStat(correct=n_correct, total=n_total),
        by_task_type=_pack(task_counts),
        by_sub_type=_pack(sub_counts),
        by_ability_type=_pack(ability_counts),
        n_predicted=len(preds),
        n_covered=n_total,
        n_unmatched=n_unmatched,
    )


# ─── Reporting ───────────────────────────────────────────────────────


def _fmt_row(name: str, stat: CategoryStat) -> str:
    acc = stat.accuracy * 100
    return f"  {name:<16s}: {acc:6.2f}%  ({stat.correct:4d}/{stat.total:4d})"


def format_report(report: ScoreReport, model_name: str = "model") -> str:
    """Return a human-readable multi-line report string."""

    lines: list[str] = []
    lines.append(f"=== PhysBench Evaluation — {model_name} ===")
    lines.append(
        f"Overall: {report.overall.accuracy * 100:.2f}%  "
        f"({report.overall.correct}/{report.overall.total})"
    )
    lines.append(
        f"Counts: predicted={report.n_predicted}  "
        f"covered={report.n_covered}  unmatched_by_gold={report.n_unmatched}"
    )

    lines.append("")
    lines.append("Task Type:")
    for tt in _ordered(report.by_task_type, TASK_TYPE_ORDER):
        lines.append(_fmt_row(tt, report.by_task_type[tt]))

    lines.append("")
    lines.append("Sub Type:")
    for st in _ordered(report.by_sub_type, SUB_TYPE_ORDER):
        lines.append(_fmt_row(st, report.by_sub_type[st]))

    lines.append("")
    lines.append("Ability Type:")
    for at in _ordered(report.by_ability_type, ABILITY_TYPE_ORDER):
        lines.append(_fmt_row(at, report.by_ability_type[at]))

    return "\n".join(lines)


def format_markdown(report: ScoreReport, model_name: str = "model") -> str:
    """Return a Markdown version of the same report.

    The three tables mirror the columns used on the PhysBench site so
    numbers can be copied directly into the paper / leaderboard.
    """

    def _table(title: str,
               stats: dict[str, CategoryStat],
               order: Iterable[str]) -> list[str]:
        cols = list(_ordered(stats, order))
        headers = "| model | " + " | ".join(cols) + " | avg |"
        sep = "| --- | " + " | ".join(["---"] * len(cols)) + " | --- |"
        accs = [stats[c].accuracy * 100 for c in cols]
        avg = (
            sum(stats[c].correct for c in cols)
            / max(1, sum(stats[c].total for c in cols))
            * 100
        )
        values = (
            f"| {model_name} | "
            + " | ".join(f"{a:.2f}" for a in accs)
            + f" | {avg:.2f} |"
        )
        return [f"**{title}**", "", headers, sep, values, ""]

    out: list[str] = []
    out.append(f"# PhysBench — {model_name}")
    out.append("")
    out.append(
        f"Overall accuracy: **{report.overall.accuracy * 100:.2f}%** "
        f"({report.overall.correct}/{report.overall.total})  "
        f"(predicted={report.n_predicted}, "
        f"covered={report.n_covered}, "
        f"unmatched_by_gold={report.n_unmatched})"
    )
    out.append("")
    out.extend(_table("Task Type Accuracy", report.by_task_type, TASK_TYPE_ORDER))
    out.extend(_table("Sub Type Accuracy", report.by_sub_type, SUB_TYPE_ORDER))
    out.extend(_table("Ability Type Accuracy", report.by_ability_type,
                      ABILITY_TYPE_ORDER))
    return "\n".join(out)


def _ordered(stats: dict[str, CategoryStat], order: Iterable[str]) -> list[str]:
    """Return ``order``-first keys then any extras in insertion order."""

    seen: set[str] = set()
    out: list[str] = []
    for k in order:
        if k in stats:
            out.append(k)
            seen.add(k)
    for k in stats:
        if k not in seen:
            out.append(k)
    return out


def print_report(report: ScoreReport, model_name: str = "model") -> None:
    """Print :func:`format_report` to stdout. Thin convenience wrapper."""

    print(format_report(report, model_name=model_name))


def _md_cell(value: object, max_len: int = 500) -> str:
    """Escape content for a Markdown table cell (pipes, newlines, length)."""

    if value is None:
        s = "—"
    else:
        s = str(value).strip() or "—"
    s = s.replace("\r", " ").replace("\n", " ")
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s.replace("|", r"\|")


def format_answer_log_markdown(
    predictions_file: Path,
    answer_file: Path | None = None,
    *,
    model_name: str = "model",
    run_meta: dict[str, Any] | None = None,
) -> str:
    """Build a per-item Markdown table: gold vs model answer and metadata.

    If ``answer_file`` is ``None`` or missing on disk, the table only lists
    ``idx`` and the model output (e.g. test split with no public gold).
    """

    pred_path = Path(predictions_file)
    preds = _load_json(pred_path)
    rows_in = [
        r for r in preds if isinstance(r, dict) and r.get("idx") is not None
    ]
    rows_in.sort(key=lambda r: int(r["idx"]))

    gold_by_idx: dict[int, dict[str, Any]] = {}
    ans_path: Path | None = None
    if answer_file is not None:
        ans_path = Path(answer_file)
        if ans_path.is_file():
            for g in _load_json(ans_path):
                if isinstance(g, dict) and "idx" in g:
                    gold_by_idx[int(g["idx"])] = g

    have_gold = bool(gold_by_idx)
    lines: list[str] = [
        f"# PhysBench Answer Record — {model_name}",
        "",
    ]
    if run_meta:
        lines.append("## Run Information")
        lines.append("")
        for k, v in run_meta.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    lines.append("## Item Details")
    lines.append("")

    if answer_file is not None and ans_path is not None and not ans_path.is_file():
        lines.append(
            f"> Gold file not found `{ans_path}`，showing predictions only."
        )
        lines.append("")

    if not rows_in:
        lines.append("*(No prediction rows; check ``predictions.json``。)*")
        return "\n".join(lines)

    if have_gold:
        lines.append(
            "| idx | Gold Answer | Model Output | Result | task_type | sub_type | "
            "ability_type | mode |"
        )
        lines.append(
            "| ---: | :---: | :--- | :---: | --- | --- | --- | --- |"
        )
        for r in rows_in:
            idx = int(r["idx"])
            pred_raw = r.get("answer")
            g = gold_by_idx.get(idx)
            if g is not None:
                gold = str(g.get("answer", "") or "")
                gdisp = _md_cell(gold, max_len=32)
            else:
                gold = ""
                gdisp = "—"
            pred_str = None if pred_raw is None else str(pred_raw)
            if g is not None and gold:
                mark = "✓" if _match(pred_str, gold) else "✗"
            else:
                mark = "—"
            tt = (g or {}).get("task_type") or "—"
            st = (g or {}).get("sub_type") or "—"
            ab = (g or {}).get("ability_type") or "—"
            md = (g or {}).get("mode") or "—"
            lines.append(
                f"| {idx} | {gdisp} | {_md_cell(pred_raw)} | {mark} | "
                f"{_md_cell(tt, max_len=64)} | {_md_cell(st, max_len=64)} | "
                f"{_md_cell(ab, max_len=64)} | {_md_cell(md, max_len=32)} |"
            )
    else:
        lines.append("| idx | Model Output |")
        lines.append("| ---: | :--- |")
        for r in rows_in:
            idx = int(r["idx"])
            pred_raw = r.get("answer")
            lines.append(f"| {idx} | {_md_cell(pred_raw)} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        f"*Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC*"
    )
    return "\n".join(lines)


__all__ = [
    "ABILITY_TYPE_ORDER",
    "CategoryStat",
    "ScoreReport",
    "SUB_TYPE_ORDER",
    "TASK_TYPE_ORDER",
    "format_answer_log_markdown",
    "format_markdown",
    "format_report",
    "print_report",
    "score",
]
