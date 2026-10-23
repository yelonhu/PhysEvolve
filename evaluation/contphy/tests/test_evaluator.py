"""CPU-only tests for the ContPhy zero-shot evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.contphy import evaluator
from evaluation.contphy.evaluator import (
    DOMAIN_ORDER,
    EXPECTED_COUNTS,
    EXPECTED_N,
    ContPhyItem,
    ContPhyMissingVideoError,
    ContPhyPackingError,
    ContPhyRGBSelectionError,
    build_contphy_multimodal_content,
    build_contphy_prompt,
    frames_and_metadata_from_content,
    load_contphy_corpus,
    parse_contphy_answer,
    post_prompt_for_answer_type,
    score_contphy,
    select_contphy_items,
    select_rgb_video,
)


_LIVE_DATASET = Path("datasets/contphy")
_JSON_NAMES = {
    "cloth": "cloth_final2_test_v1.json",
    "fluid": "fluid_final_test_v1.json",
    "rope": "rope_final2_test_v1.json",
    "soft": "soft_final_test_v1.json",
}


def _write_tiny_mp4(
    path: Path,
    *,
    n_frames: int = 4,
    fps: float = 5.0,
) -> None:
    """Write a tiny mp4 fixture, skipping if OpenCV cannot encode."""

    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (8, 8),
    )
    if not writer.isOpened():
        pytest.skip("cv2 could not write a fake mp4 — skipping")
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[:] = 80
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        pytest.skip("cv2 wrote an empty mp4 — skipping")


def _question_row(
    *,
    question: str,
    answer: str,
    answer_type: str,
    qtype: str,
    question_id: str,
    video_id: str,
) -> dict[str, object]:
    """Return one ContPhy-shaped annotation object."""

    return {
        "question": question,
        "answer": answer,
        "answer_type": answer_type,
        "qtype": qtype,
        "question_family": qtype,
        "question_id": question_id,
        "split": "test",
        "video_id": video_id,
        "positive": [answer],
        "negative": [],
        "program": "unused",
    }


def _write_domain_json(
    root: Path,
    domain: str,
    rows: list[dict[str, object]],
    *,
    nested_trap: bool = False,
) -> Path:
    """Write a zip-native test JSON, optionally with a nested duplicate file."""

    path = root / domain / "question" / _JSON_NAMES[domain]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")
    if nested_trap:
        trap = root / domain / domain / "question" / _JSON_NAMES[domain]
        trap.parent.mkdir(parents=True, exist_ok=True)
        trap.write_text(
            json.dumps(
                [
                    _question_row(
                        question="nested trap should be ignored",
                        answer="yes",
                        answer_type="open_ended",
                        qtype="trap",
                        question_id="trap000",
                        video_id="trap",
                    )
                ]
            ),
            encoding="utf-8",
        )
    return path


def _write_trial(
    root: Path,
    domain: str,
    video_id: str,
    *,
    include_rgb: bool = True,
    include_seg: bool = True,
) -> Path:
    """Write a trial directory with RGB and/or seg mp4 files."""

    trial = root / domain / "video" / video_id
    trial.mkdir(parents=True, exist_ok=True)
    if include_rgb:
        _write_tiny_mp4(trial / "output_Full.mp4")
    if include_seg:
        _write_tiny_mp4(trial / "output_Full_seg.mp4")
    return trial


def _write_fixture(root: Path, *, create_videos: bool = True) -> Path:
    """Write a four-domain fixture covering all answer types and a nested trap."""

    _write_domain_json(
        root,
        "cloth",
        [
            _question_row(
                question="Is the left cloth harder to bend?",
                answer="no",
                answer_type="open_ended",
                qtype="bending_compliance",
                question_id="c001",
                video_id="10",
            ),
            _question_row(
                question="Does the gray pillar fall over?\nA. Yes.\nB. No.",
                answer="B",
                answer_type="single_choice",
                qtype="predictive_fall_over",
                question_id="c002",
                video_id="10",
            ),
        ],
    )
    _write_domain_json(
        root,
        "fluid",
        [
            _question_row(
                question=(
                    "Which stick will the fluid pass?\n"
                    "A. Purple stick\nB. Gray stick\nC. Brown stick"
                ),
                answer="B C",
                answer_type="multiple_choice",
                qtype="predictive_stick",
                question_id="f001",
                video_id="20",
            ),
        ],
        nested_trap=True,
    )
    _write_domain_json(
        root,
        "rope",
        [
            _question_row(
                question=(
                    "If we want the red pulley to rotate anti-clockwise, "
                    "what can we do?\nA. Decrease cyan\nB. Decrease brown cube\n"
                    "C. Decrease red cube\nD. Increase red cube\n"
                    "E. Decrease brown sphere"
                ),
                answer="E",
                answer_type="multiple_choice",
                qtype="rope_goaldriven",
                question_id="r001",
                video_id="30",
            ),
        ],
    )
    _write_domain_json(
        root,
        "soft",
        [
            _question_row(
                question="Is the plasticity of the brown ball much greater?",
                answer="yes",
                answer_type="open_ended",
                qtype="plasticity",
                question_id="s001",
                video_id="40",
            ),
        ],
    )
    if create_videos:
        _write_trial(root, "cloth", "10")
        _write_trial(root, "fluid", "20")
        _write_trial(root, "rope", "30")
        _write_trial(root, "soft", "40")
    return root


def _dummy_item(**overrides: object) -> ContPhyItem:
    """Return a gold-bearing item that is never used as a prompt leak check."""

    payload: dict[str, object] = {
        "idx": 0,
        "sample_key": "cloth:c001",
        "domain": "cloth",
        "question": "Is the left cloth harder to bend?",
        "gold_answer": "no",
        "answer_type": "open_ended",
        "qtype": "bending_compliance",
        "question_family": "bending_compliance",
        "question_id": "c001",
        "video_id": "10",
        "trial_dir": Path("/tmp/unused"),
        "video_path": Path("/tmp/unused/output_Full.mp4"),
        "video_name": "output_Full.mp4",
    }
    payload.update(overrides)
    return ContPhyItem(**payload)  # type: ignore[arg-type]


def test_rgb_picker_drops_seg_and_depth(tmp_path: Path) -> None:
    """ls then drop seg/depth/mask; prefer output_Full.mp4."""

    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "output_Full_seg.mp4").write_bytes(b"seg")
    (trial / "depth.mp4").write_bytes(b"depth")
    (trial / "mask_overlay.mp4").write_bytes(b"mask")
    (trial / "output_Full.mp4").write_bytes(b"rgb")
    (trial / "zzz_other.mp4").write_bytes(b"other")

    chosen = select_rgb_video(trial)

    assert chosen.name == "output_Full.mp4"
    assert "seg" not in chosen.name.lower()
    assert "depth" not in chosen.name.lower()


def test_rgb_picker_zero_rgb_raises(tmp_path: Path) -> None:
    """A trial with only seg/depth must fail before any model load."""

    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "output_Full_seg.mp4").write_bytes(b"seg")

    with pytest.raises(ContPhyRGBSelectionError, match="no RGB mp4"):
        select_rgb_video(trial)


def test_loader_rejects_missing_video(tmp_path: Path) -> None:
    """A JSON row whose trial mp4 is absent must fail before inference."""

    _write_fixture(tmp_path, create_videos=False)

    with pytest.raises(ContPhyMissingVideoError, match="videos are missing"):
        load_contphy_corpus(tmp_path, strict_official_counts=False)


def test_official_count_guard_rejects_fixture(tmp_path: Path) -> None:
    """Production loading should reject an incomplete fixture JSON."""

    _write_fixture(tmp_path)

    with pytest.raises(ValueError, match="must contain 450 test rows"):
        load_contphy_corpus(tmp_path)


def test_loader_ignores_nested_domain_copies(tmp_path: Path) -> None:
    """fluid/fluid (and similar nested dumps) must not be indexed."""

    _write_fixture(tmp_path)
    corpus = load_contphy_corpus(tmp_path, strict_official_counts=False)

    assert len(corpus.items) == 5
    assert corpus.n_by_domain == {
        "cloth": 2,
        "fluid": 1,
        "rope": 1,
        "soft": 1,
    }
    assert all(item.sample_key != "fluid:trap000" for item in corpus.items)
    assert all(item.qtype != "trap" for item in corpus.items)
    assert all(item.video_path.is_file() for item in corpus.items)
    assert all("seg" not in item.video_name.lower() for item in corpus.items)
    assert [item.video_name for item in corpus.items] == ["output_Full.mp4"] * 5


def test_prompts_are_table8_suffixes_and_gold_free(tmp_path: Path) -> None:
    """Each answer_type gets its Table 8 suffix; gold is not concatenated."""

    _write_fixture(tmp_path)
    corpus = load_contphy_corpus(tmp_path, strict_official_counts=False)
    by_type = {item.answer_type: item for item in corpus.items}

    open_item = by_type["open_ended"]
    single_item = by_type["single_choice"]
    mc_item = by_type["multiple_choice"]

    open_prompt = build_contphy_prompt(open_item)
    single_prompt = build_contphy_prompt(single_item)
    mc_prompt = build_contphy_prompt(mc_item)

    assert open_prompt == open_item.question + post_prompt_for_answer_type(
        "open_ended"
    )
    assert single_prompt == single_item.question + post_prompt_for_answer_type(
        "single_choice"
    )
    assert mc_prompt == mc_item.question + post_prompt_for_answer_type(
        "multiple_choice"
    )
    assert open_prompt.endswith(post_prompt_for_answer_type("open_ended"))
    assert "all correct choices" in mc_prompt
    assert 'respond with "A B C"' in mc_prompt
    assert 'respond with "A"' in single_prompt
    assert "can not answer" in open_prompt
    assert not open_prompt.rstrip().endswith(open_item.gold_answer)
    assert not single_prompt.rstrip().endswith(single_item.gold_answer)
    assert not mc_prompt.rstrip().endswith(mc_item.gold_answer)
    assert "<answer>" not in open_prompt


def test_parser_covers_choice_and_open_ended_and_unparsed_is_wrong(
    tmp_path: Path,
) -> None:
    """Local scoring uses A-E multi-letter parse and short-answer exact match."""

    _write_fixture(tmp_path)
    corpus = load_contphy_corpus(tmp_path, strict_official_counts=False)

    assert parse_contphy_answer("A B C", "multiple_choice") == "A B C"
    assert parse_contphy_answer("C B A", "multiple_choice") == "A B C"
    assert parse_contphy_answer("<answer>E</answer>", "multiple_choice") == "E"
    assert parse_contphy_answer("B", "single_choice") == "B"
    assert parse_contphy_answer("Yes", "open_ended") == "yes"
    assert parse_contphy_answer("NO", "open_ended") == "no"
    assert parse_contphy_answer("can not answer", "open_ended") == "can not answer"
    assert parse_contphy_answer("I cannot tell", "multiple_choice") is None
    assert parse_contphy_answer("   ", "open_ended") is None

    items = list(corpus.items)
    # cloth open no, cloth single B, fluid MC B C, rope MC E, soft open yes
    summary = score_contphy(
        items,
        {
            0: "no",
            1: None,
            2: "B C",
            3: "A",
            4: "yes",
        },
        corpus=corpus,
        model_tag="fixture",
        n_video_frames=10,
        max_new_tokens=64,
        out_dir=tmp_path / "out",
    )
    assert summary["official_readout_comparable"] is False
    assert summary["dynamic_per_option_scoring"] is False
    assert summary["parser"] == "parse_contphy_answer"
    assert summary["overall"]["correct"] == 3
    assert summary["overall"]["parsed"] == 4
    assert summary["overall"]["total"] == 5
    assert summary["overall"]["accuracy"] == pytest.approx(3 / 5)
    assert summary["by_domain"]["cloth"]["correct"] == 1
    assert summary["by_answer_type"]["open_ended"]["correct"] == 2


def test_evaluator_does_not_import_physbench_mc_parser() -> None:
    """ContPhy must not import the A-D PhysBench letter parser or ContPhy repos."""

    source = Path(evaluator.__file__).read_text(encoding="utf-8")
    assert not hasattr(evaluator, "parse_physbench_mc_answer")
    assert "from evaluation.physbench.prompter" not in source
    assert "ContPhy_Public" not in source
    assert "ContPhy-Gen" not in source


def test_packing_is_presampled_tuple_with_fps_2() -> None:
    """Qwen3-VL content must wrap 10 frames with fps=2.0 metadata."""

    frames = np.full((10, 32, 32, 3), 40, dtype=np.uint8)
    content = build_contphy_multimodal_content(frames, _dummy_item())
    unpacked, metadata = frames_and_metadata_from_content(content)

    assert content[0]["type"] == "video"
    assert isinstance(content[0]["video"], tuple)
    assert unpacked.shape == (10, 32, 32, 3)
    assert metadata["fps"] == 2.0
    assert "do_sample_frames" not in metadata
    assert content[1]["type"] == "text"
    assert content[1]["text"].endswith(post_prompt_for_answer_type("open_ended"))


def test_run_signature_records_presampled_protocol() -> None:
    """Resume metadata must include domains, fps=2.0, and per-type suffixes."""

    args = evaluator._parse_args([])
    signature = evaluator._run_signature(args, model_tag="fixture")

    assert args.n_video_frames == 10
    assert args.max_new_tokens == 64
    assert signature["protocol"] == "contphy_test_presampled_v1"
    assert signature["do_sample_frames"] is False
    assert signature["video_fps"] == 2.0
    assert signature["parser"] == "parse_contphy_answer"
    assert signature["domain_order"] == list(DOMAIN_ORDER)
    assert set(signature["post_prompts"]) == {
        "multiple_choice",
        "single_choice",
        "open_ended",
    }
    default_dir = evaluator._default_out_dir(
        "Qwen3-VL-8B-Instruct",
        n_samples=None,
        n_video_frames=10,
    )
    assert default_dir.name.endswith("_contphy_test_f10_presampled")
    assert "model_family" not in signature
    internvl_args = evaluator._parse_args(
        [
            "--model-path",
            "models/InternVL2_5-8B",
            "--model-family",
            "internvl",
        ]
    )
    internvl_signature = evaluator._run_signature(
        internvl_args, model_tag="InternVL2_5-8B"
    )
    assert internvl_args.model_family == "internvl"
    assert internvl_signature["model_family"] == "internvl"
    assert internvl_signature["protocol"] == "contphy_test_presampled_v1"
    assert internvl_signature["do_sample_frames"] is False
    assert internvl_signature["video_fps"] == 2.0


def test_token_gate_stops_resampled_shard(tmp_path: Path, monkeypatch) -> None:
    """A ~211-token prompt must abort instead of writing answers."""

    _write_fixture(tmp_path)
    corpus = load_contphy_corpus(tmp_path, strict_official_counts=False)
    selected = list(corpus.items)
    args = evaluator._parse_args(
        [
            "--dataset-dir",
            str(tmp_path),
            "--n-video-frames",
            "10",
            "--no-resume",
        ]
    )
    monkeypatch.setattr(
        evaluator,
        "generate_reference_answer",
        lambda *args, **kwargs: type(
            "Result",
            (),
            {
                "text": "A)",
                "metadata": {"prompt_token_count": 211},
            },
        )(),
    )
    monkeypatch.setattr(
        evaluator,
        "open_eval_backend",
        lambda *args, **kwargs: _FakeBackend(),
    )
    with pytest.raises(ContPhyPackingError, match="prompt_token_count=211"):
        evaluator._run_inference(
            args,
            corpus=corpus,
            selected=selected,
            shard=selected[:1],
            model_tag="fixture",
            out_dir=tmp_path / "out_presampled",
        )


def test_token_gate_skipped_for_non_qwen_family(tmp_path: Path, monkeypatch) -> None:
    """InternVL / Llama / Pixtral must not inherit the Qwen fps=24 token abort."""

    _write_fixture(tmp_path)
    corpus = load_contphy_corpus(tmp_path, strict_official_counts=False)
    selected = list(corpus.items)[:1]
    args = evaluator._parse_args(
        [
            "--dataset-dir",
            str(tmp_path),
            "--n-video-frames",
            "10",
            "--no-resume",
            "--model-family",
            "internvl",
            "--model-path",
            "models/InternVL2_5-8B",
        ]
    )
    monkeypatch.setattr(
        evaluator,
        "generate_reference_answer",
        lambda *args, **kwargs: type(
            "Result",
            (),
            {
                "text": "A",
                "metadata": {"prompt_token_count": 211},
            },
        )(),
    )
    monkeypatch.setattr(
        evaluator,
        "open_eval_backend",
        lambda *args, **kwargs: _FakeBackend(),
    )
    assert (
        evaluator._run_inference(
            args,
            corpus=corpus,
            selected=selected,
            shard=selected,
            model_tag="fixture",
            out_dir=tmp_path / "internvl_out_presampled",
        )
        == 0
    )


def test_select_n_samples_is_deterministic(tmp_path: Path) -> None:
    """The same seed should yield the same --n-samples subset."""

    _write_fixture(tmp_path)
    corpus = load_contphy_corpus(tmp_path, strict_official_counts=False)
    first = select_contphy_items(corpus, n_samples=2, seed=7)
    second = select_contphy_items(corpus, n_samples=2, seed=7)
    assert [item.idx for item in first] == [item.idx for item in second]
    assert len(first) == 2


def test_validate_media_checks_frames_fps_and_rgb_name(tmp_path: Path) -> None:
    """--validate-media must decode N frames at fps=2.0 from a non-seg mp4."""

    _write_fixture(tmp_path)
    corpus = load_contphy_corpus(tmp_path, strict_official_counts=False)
    payload = evaluator._validate_media_item(
        corpus.items[0],
        n_video_frames=10,
    )

    assert payload["video_shape"][0] == 10
    assert payload["video_fps"] == 2.0
    assert payload["do_sample_frames_in_metadata"] is False
    assert payload["video_name"] == "output_Full.mp4"
    assert payload["gold_in_trailing_completion"] is False


def test_select_domain_filter(tmp_path: Path) -> None:
    """--domain must keep only the requested zip-native domain."""

    _write_fixture(tmp_path)
    corpus = load_contphy_corpus(tmp_path, strict_official_counts=False)
    cloth = select_contphy_items(corpus, n_samples=None, seed=0, domains=["cloth"])
    assert {item.domain for item in cloth} == {"cloth"}
    assert len(cloth) == 2


@pytest.mark.skipif(
    not (_LIVE_DATASET / "cloth" / "question" / "cloth_final2_test_v1.json").is_file(),
    reason="local ContPhy dump is missing",
)
def test_live_corpus_has_official_count_and_no_missing_videos() -> None:
    """The on-disk dump must index 1950 test rows with every RGB video present."""

    corpus = load_contphy_corpus(_LIVE_DATASET)
    assert len(corpus.items) == EXPECTED_N
    assert corpus.n_json_rows == EXPECTED_N
    assert corpus.n_by_domain == EXPECTED_COUNTS
    assert all(item.video_path.is_file() for item in corpus.items)
    assert all("seg" not in item.video_name.lower() for item in corpus.items)
    assert all("depth" not in item.video_name.lower() for item in corpus.items)
    nested_keys = {item.sample_key for item in corpus.items}
    assert "fluid:trap000" not in nested_keys
    assert len(nested_keys) == EXPECTED_N


class _FakeBackend:
    """Context manager stand-in so packing tests never load weights."""

    def __enter__(self) -> "_FakeBackend":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None
