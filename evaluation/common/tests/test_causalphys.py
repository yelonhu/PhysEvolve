"""CPU-only CausalPhys loader, prompt, and scorer tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.common import causalphys
from evaluation.common.causalphys import (
    _default_out_dir,
    _parse_args,
    _run_signature,
    build_causalphys_prompt,
    load_causalphys_corpus,
    parse_causalphys_response,
    score_causalphys,
    select_causalphys_items,
)


def _write_annotation(
    root: Path,
    *,
    name: str,
    gold: str,
    secret: str,
) -> None:
    """Write one minimal annotation and existing image placeholder."""

    annotation_dir = root / "Perception" / "optics" / "annotations"
    data_dir = root / "Perception" / "optics" / "data"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / f"{name}.png").write_bytes(b"fixture")
    (annotation_dir / f"{name}.json").write_text(
        json.dumps(
            {
                "id": name,
                "question": "Which outcome occurs? A: first B: second",
                "ground_truth_answer": gold,
                "path": f"{name}.png",
                "graph": {"nodes": [{"text": secret}], "edges": []},
            }
        ),
        encoding="utf-8",
    )


def test_loader_normalizes_lowercase_and_excludes_invalid_gold(
    tmp_path: Path,
) -> None:
    """Lowercase B is valid while literal nan is reported and excluded."""

    _write_annotation(tmp_path, name="a", gold="A", secret="secret-a")
    _write_annotation(tmp_path, name="b", gold="b", secret="secret-b")
    _write_annotation(tmp_path, name="bad", gold="nan", secret="secret-bad")

    corpus = load_causalphys_corpus(tmp_path)

    assert corpus.n_annotations == 3
    assert [item.gold_answer for item in corpus.items] == ["A", "B"]
    assert len(corpus.invalid_gold) == 1
    assert corpus.invalid_gold[0].sample_key.endswith("bad.json")


def test_short_and_cot_prompts_contain_no_graph_or_gold(
    tmp_path: Path,
) -> None:
    """Both protocols must expose only the visual question surface."""

    _write_annotation(
        tmp_path,
        name="sample",
        gold="B",
        secret="DO_NOT_LEAK_GRAPH",
    )
    item = load_causalphys_corpus(tmp_path).items[0]
    short_prompt = build_causalphys_prompt(item, prompt_style="short")
    cot_prompt = build_causalphys_prompt(item, prompt_style="cot")

    assert "<rationale>" not in short_prompt
    assert "exactly one option letter" in short_prompt
    assert "<rationale>" in cot_prompt
    assert "<answer>A</answer>" in cot_prompt
    assert "at most 8 sentences" in cot_prompt
    for prompt in (short_prompt, cot_prompt):
        assert "Which outcome occurs?" in prompt
        assert "DO_NOT_LEAK_GRAPH" not in prompt
        assert "ground_truth_answer" not in prompt


def test_protocol_defaults_and_signature_isolation(tmp_path: Path) -> None:
    """Style and token budget must isolate output and resume provenance."""

    cot_args = _parse_args(["--dataset-dir", str(tmp_path)])
    short_args = _parse_args(
        [
            "--dataset-dir",
            str(tmp_path),
            "--prompt-style",
            "short",
            "--max-new-tokens",
            "64",
        ]
    )

    assert cot_args.prompt_style == "cot"
    assert cot_args.max_new_tokens == 512
    cot_signature = _run_signature(cot_args, model_tag="model")
    short_signature = _run_signature(short_args, model_tag="model")
    assert cot_signature["prompt_style"] == "cot"
    assert cot_signature["max_new_tokens"] == 512
    assert "model_family" not in cot_signature
    internvl_args = _parse_args(
        [
            "--dataset-dir",
            str(tmp_path),
            "--model-path",
            "models/InternVL2_5-8B",
            "--model-family",
            "internvl",
        ]
    )
    internvl_signature = _run_signature(
        internvl_args, model_tag="InternVL2_5-8B"
    )
    assert internvl_args.model_family == "internvl"
    assert internvl_signature["model_family"] == "internvl"
    assert cot_signature != internvl_signature
    assert cot_signature != short_signature
    assert _default_out_dir(
        "model",
        n_samples=None,
        prompt_style="cot",
        max_new_tokens=512,
    ) != _default_out_dir(
        "model",
        n_samples=None,
        prompt_style="short",
        max_new_tokens=512,
    )
    assert _default_out_dir(
        "model",
        n_samples=None,
        prompt_style="cot",
        max_new_tokens=512,
    ) != _default_out_dir(
        "model",
        n_samples=None,
        prompt_style="cot",
        max_new_tokens=64,
    )


@pytest.mark.parametrize(
    ("raw_text", "answer", "rationale"),
    [
        (
            "<rationale>Option A is tempting, but motion supports D."
            "</rationale><answer>D</answer>",
            "D",
            "Option A is tempting, but motion supports D.",
        ),
        (
            "<rationale>The second object moves first.</rationale>"
            "<result>Final answer: C</result>",
            "C",
            "The second object moves first.",
        ),
        (
            "The collision precedes the color change.\nFinal answer: B",
            "B",
            "The collision precedes the color change.",
        ),
        ("<answer>D", "D", None),
        ("A", "A", None),
    ],
)
def test_parse_causalphys_response(
    raw_text: str,
    answer: str,
    rationale: str | None,
) -> None:
    """Final answer parsing must preserve a separate audit rationale."""

    parsed = parse_causalphys_response(raw_text)

    assert parsed.answer == answer
    assert parsed.rationale == rationale


def test_parser_ignores_rationale_letter_without_final_answer() -> None:
    """A letter mentioned inside rationale must not become a prediction."""

    parsed = parse_causalphys_response(
        "<rationale>Option A is visually plausible.</rationale>"
    )

    assert parsed.answer is None
    assert parsed.rationale == "Option A is visually plausible."


def test_inference_audits_rationale_and_rejects_protocol_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit rows preserve rationale and resume rejects changed protocol."""

    _write_annotation(tmp_path, name="sample", gold="B", secret="hidden")
    corpus = load_causalphys_corpus(tmp_path)
    selected = list(corpus.items)
    out_dir = tmp_path / "out"
    cot_args = _parse_args(
        [
            "--dataset-dir",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
        ]
    )

    class DummyBackend:
        """Minimal context manager replacing the GPU backend."""

        def __enter__(self) -> "DummyBackend":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(causalphys, "open_eval_backend", lambda *a, **k: DummyBackend())
    monkeypatch.setattr(
        causalphys,
        "build_causalphys_content",
        lambda *args, **kwargs: [{"type": "text", "text": "fixture"}],
    )
    monkeypatch.setattr(
        causalphys,
        "generate_reference_answer",
        lambda *args, **kwargs: type(
            "Result",
            (),
            {
                "text": (
                    "<rationale>Motion supports the second outcome.</rationale>"
                    "<answer>B</answer>"
                ),
                "metadata": {"fixture": True},
            },
        )(),
    )

    assert (
        causalphys._run_inference(
            cot_args,
            corpus=corpus,
            selected=selected,
            shard=selected,
            model_tag="fixture",
            out_dir=out_dir,
        )
        == 0
    )
    audit_row = json.loads(
        (out_dir / "raw_audit.jsonl").read_text(encoding="utf-8").strip()
    )
    assert audit_row["answer"] == "B"
    assert audit_row["rationale"] == "Motion supports the second outcome."
    assert audit_row["prompt_style"] == "cot"

    short_args = _parse_args(
        [
            "--dataset-dir",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
            "--prompt-style",
            "short",
        ]
    )
    with pytest.raises(ValueError, match="resume metadata differs"):
        causalphys._run_inference(
            short_args,
            corpus=corpus,
            selected=selected,
            shard=selected,
            model_tag="fixture",
            out_dir=out_dir,
        )


def test_single_run_journal_seeds_shards_and_reruns_nulls(
    tmp_path: Path,
) -> None:
    """A compatible one-process journal should seed only non-null answers."""

    for name, gold in (("a", "A"), ("b", "B"), ("c", "C"), ("d", "D")):
        _write_annotation(tmp_path, name=name, gold=gold, secret="hidden")
    corpus = load_causalphys_corpus(tmp_path)
    selected = list(corpus.items)
    out_dir = tmp_path / "resume"
    args = _parse_args(
        [
            "--dataset-dir",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
            "--num-shards",
            "2",
        ]
    )
    signature = _run_signature(args, model_tag="fixture")
    source_store = causalphys.PredictionStore(
        out_dir / "predictions.json",
        resume=False,
    )
    source_store.append(selected[0].idx, "A")
    source_store.append(selected[1].idx, None)
    source_store.append(selected[2].idx, "C")
    causalphys.write_json_atomic(
        out_dir / "causalphys_eval_config.json",
        {
            "run_signature": signature,
            "num_shards": 1,
            "shard_id": 0,
            "expected_idx": [item.idx for item in selected],
        },
    )

    seeded_ids: set[int] = set()
    for shard_id in range(2):
        args.shard_id = shard_id
        shard = causalphys.shard_items(
            selected,
            num_shards=2,
            shard_id=shard_id,
        )
        seed = causalphys._single_run_seed_answers(
            args=args,
            out_dir=out_dir,
            signature=signature,
            selected=selected,
            shard=shard,
            has_shard_history=False,
        )
        seeded_ids.update(seed)
        predictions_file = causalphys.shard_filename(
            out_dir / "predictions.json",
            shard_id,
            2,
        )
        store = causalphys.PredictionStore(predictions_file, resume=True)
        for idx, answer in seed.items():
            store.append(idx, answer)
        for item in shard:
            if not store.is_completed(item.idx):
                store.append(item.idx, item.gold_answer)
        store.finalize()
        causalphys.write_json_atomic(
            causalphys.shard_filename(
                out_dir / "causalphys_eval_config.json",
                shard_id,
                2,
            ),
            {
                "run_signature": signature,
                "num_shards": 2,
                "shard_id": shard_id,
                "expected_idx": [item.idx for item in selected],
            },
        )

    assert seeded_ids == {selected[0].idx, selected[2].idx}
    config = causalphys.load_shard_config(
        out_dir,
        prefix="causalphys_eval_config",
    )
    causalphys.merge_prediction_shards(
        out_dir,
        expected_idx=config["expected_idx"],
        expected_num_shards=2,
    )
    assert causalphys.load_prediction_map(out_dir / "predictions.json") == {
        item.idx: item.gold_answer for item in selected
    }


def test_single_run_seed_rejects_signature_mismatch(tmp_path: Path) -> None:
    """A root journal from another protocol must never seed new shards."""

    _write_annotation(tmp_path, name="a", gold="A", secret="hidden")
    selected = list(load_causalphys_corpus(tmp_path).items)
    out_dir = tmp_path / "resume"
    args = _parse_args(
        [
            "--dataset-dir",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
            "--num-shards",
            "4",
        ]
    )
    signature = _run_signature(args, model_tag="fixture")
    source_store = causalphys.PredictionStore(
        out_dir / "predictions.json",
        resume=False,
    )
    source_store.append(selected[0].idx, "A")
    incompatible_signature = {**signature, "prompt_style": "short"}
    causalphys.write_json_atomic(
        out_dir / "causalphys_eval_config.json",
        {
            "run_signature": incompatible_signature,
            "num_shards": 1,
            "shard_id": 0,
            "expected_idx": [item.idx for item in selected],
        },
    )

    with pytest.raises(ValueError, match="incompatible"):
        causalphys._single_run_seed_answers(
            args=args,
            out_dir=out_dir,
            signature=signature,
            selected=selected,
            shard=selected,
            has_shard_history=False,
        )


def test_causalphys_scorer_reports_breakdown_and_parse_rate(
    tmp_path: Path,
) -> None:
    """Local ACC should count parse failures as wrong on valid gold only."""

    _write_annotation(tmp_path, name="a", gold="A", secret="x")
    _write_annotation(tmp_path, name="b", gold="B", secret="y")
    _write_annotation(tmp_path, name="bad", gold="nan", secret="z")
    corpus = load_causalphys_corpus(tmp_path)
    selected = select_causalphys_items(
        corpus,
        n_samples=None,
        seed=7,
    )

    summary = score_causalphys(
        selected,
        {selected[0].idx: "A", selected[1].idx: None},
        corpus=corpus,
        model_tag="fixture",
        prompt_style="cot",
        max_new_tokens=512,
        out_dir=tmp_path / "results",
    )

    assert summary["overall"]["accuracy"] == 0.5
    assert summary["overall"]["parse_rate"] == 0.5
    assert summary["prompt_style"] == "cot"
    assert summary["max_new_tokens"] == 512
    assert summary["n_invalid_gold"] == 1
    assert summary["by_category"]["Perception"]["total"] == 2
