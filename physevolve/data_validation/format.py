"""Format checks for Proposer responses and PhysDSL code.

The validator requires a valid ``<propose>`` block. Reasoning tags are optional;
their presence is recorded in diagnostic fields.
"""

from __future__ import annotations

import ast
import re
from typing import Literal, Optional

from physevolve.data_validation.schemas import FormatReport
from physevolve.inference.parsing import extract_propose_block


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_LEGACY_THINK_RE = re.compile(
    r"<thinking>(.*?)</thinking>", re.IGNORECASE | re.DOTALL
)


def validate_raw_format(
    raw_output: str,
    *,
    stage: Literal[1, 2] = 1,
    allow_legacy_thinking_tag: bool = True,
) -> FormatReport:
    """Validate a model output that should contain ``<propose>`` code."""

    errors: list[str] = []
    if not isinstance(raw_output, str) or not raw_output.strip():
        return FormatReport(
            ok=False,
            has_think=False,
            has_propose=False,
            has_build_scene=False,
            has_get_vqa=False,
            errors=("raw output is empty or non-string",),
            extracted_code=None,
        )

    has_canonical_think = _THINK_RE.search(raw_output) is not None
    has_legacy_think = _LEGACY_THINK_RE.search(raw_output) is not None
    has_think = has_canonical_think or (
        allow_legacy_thinking_tag and has_legacy_think
    )
    # <think> is no longer required. The has_think field is preserved
    # for informational diagnostics; only a legacy <thinking> tag still
    # produces an explicit error when the caller asked to forbid it.
    if has_legacy_think and not allow_legacy_thinking_tag:
        errors.append("legacy <thinking> tag is not allowed")

    extracted_code = extract_propose_block(raw_output)
    has_propose = extracted_code is not None
    if extracted_code is None:
        errors.append("missing <propose> block")
        extracted_code = None

    code_report = _validate_code_body(extracted_code, stage=stage)
    errors.extend(code_report.errors)
    return FormatReport(
        ok=len(errors) == 0,
        has_think=has_think,
        has_propose=has_propose,
        has_build_scene=code_report.has_build_scene,
        has_get_vqa=code_report.has_get_vqa,
        errors=tuple(errors),
        extracted_code=extracted_code,
    )


def validate_code_format(
    code: str,
    *,
    stage: Literal[1, 2] = 1,
) -> FormatReport:
    """Validate already-extracted PhysDSL code without wrapper tags."""

    code_report = _validate_code_body(code, stage=stage)
    return FormatReport(
        ok=code_report.ok,
        has_think=False,
        has_propose=False,
        has_build_scene=code_report.has_build_scene,
        has_get_vqa=code_report.has_get_vqa,
        errors=code_report.errors,
        extracted_code=code if isinstance(code, str) else None,
    )


def _validate_code_body(
    code: Optional[str],
    *,
    stage: Literal[1, 2],
) -> FormatReport:
    """Run AST-level checks that do not duplicate sandbox validation."""

    errors: list[str] = []
    has_build_scene = False
    has_get_vqa = False

    if not isinstance(code, str) or not code.strip():
        return FormatReport(
            ok=False,
            has_think=False,
            has_propose=False,
            has_build_scene=False,
            has_get_vqa=False,
            errors=("missing PhysDSL code",),
            extracted_code=None,
        )

    if "```" in code:
        errors.append("extracted code contains markdown fences")

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return FormatReport(
            ok=False,
            has_think=False,
            has_propose=False,
            has_build_scene=False,
            has_get_vqa=False,
            errors=tuple(errors + [f"ast parse failed: {exc.msg}"]),
            extracted_code=code,
        )

    build_scene_count = 0
    get_vqa_count = 0
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name == "build_scene":
            build_scene_count += 1
        elif node.name == "get_vqa":
            get_vqa_count += 1

    has_build_scene = build_scene_count > 0
    has_get_vqa = get_vqa_count > 0

    if build_scene_count == 0:
        errors.append("missing build_scene definition")
    elif build_scene_count > 1:
        errors.append("multiple build_scene definitions")

    if stage == 2 and get_vqa_count == 0:
        errors.append("stage=2 requires get_vqa definition")
    if get_vqa_count > 1:
        errors.append("multiple get_vqa definitions")

    return FormatReport(
        ok=len(errors) == 0,
        has_think=False,
        has_propose=False,
        has_build_scene=has_build_scene,
        has_get_vqa=has_get_vqa,
        errors=tuple(errors),
        extracted_code=code,
    )


__all__ = ["validate_code_format", "validate_raw_format"]
