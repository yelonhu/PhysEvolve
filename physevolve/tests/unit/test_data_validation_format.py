"""Tests for data-validation format checks."""

from __future__ import annotations

from physevolve.data_validation import validate_code_format, validate_raw_format


_VALID_CODE = """\
def build_scene(api):
    api.add_sphere('ball', pos=(0.0, 0.0, 1.0), radius=0.1, mass=1.0)
    api.set_duration(0.5)
"""


def test_format_missing_propose() -> None:
    report = validate_raw_format("<think>intent</think>no code")

    assert report.ok is False
    assert report.has_think is True
    assert report.has_propose is False
    assert "missing <propose> block" in report.errors


def test_format_missing_think_is_not_an_error() -> None:
    # Instruct base model does not emit <think>; missing tag must not
    # block format validation as long as <propose> and build_scene are present.
    raw = f"Plan: build a simple scene.<propose>{_VALID_CODE}</propose>"

    report = validate_raw_format(raw)

    assert report.ok is True
    assert report.has_think is False
    assert report.has_propose is True
    assert report.has_build_scene is True
    assert all("missing <think>" not in err for err in report.errors)


def test_format_extracts_build_scene() -> None:
    raw = f"<think>intent</think><propose>{_VALID_CODE}</propose>"

    report = validate_raw_format(raw)

    assert report.ok is True
    assert report.has_think is True
    assert report.has_propose is True
    assert report.has_build_scene is True
    assert report.extracted_code == _VALID_CODE.strip()


def test_format_accepts_vllm_swallowed_propose_stop() -> None:
    raw = f"<think>intent</think><propose>{_VALID_CODE}"

    report = validate_raw_format(raw)

    assert report.ok is True
    assert report.has_propose is True
    assert report.extracted_code == _VALID_CODE.strip()


def test_format_accepts_assistant_text_with_restored_prefill_think() -> None:
    generated_text = f"intent</think><propose>{_VALID_CODE}</propose>"
    assistant_text = "<think>" + generated_text

    report = validate_raw_format(assistant_text)

    assert report.ok is True
    assert report.has_think is True
    assert report.has_propose is True


def test_format_rejects_markdown_only_output() -> None:
    raw = f"<think>intent</think><propose>```python\n{_VALID_CODE}```</propose>"

    report = validate_raw_format(raw)

    assert report.ok is False
    assert "extracted code contains markdown fences" in report.errors


def test_format_legacy_thinking_can_be_disabled() -> None:
    raw = f"<thinking>legacy</thinking><propose>{_VALID_CODE}</propose>"

    report = validate_raw_format(raw, allow_legacy_thinking_tag=False)

    assert report.ok is False
    assert report.has_think is False
    assert "legacy <thinking> tag is not allowed" in report.errors


def test_format_code_path_does_not_require_wrappers() -> None:
    report = validate_code_format(_VALID_CODE)

    assert report.ok is True
    assert report.has_think is False
    assert report.has_propose is False
    assert report.has_build_scene is True
    assert report.extracted_code == _VALID_CODE
