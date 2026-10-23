"""Tests for skill-conditioned curriculum prompts and sampling.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from physevolve.prompts.proposer_stage1 import PHYSICS_SKILL_TARGETS
from physevolve.training.online_step_builder import (
    PhysEvolveOnlineStepBuilder,
    PhysEvolveOnlineStepBuilderConfig,
)


def _make_builder(tmp_path: Path, **overrides) -> PhysEvolveOnlineStepBuilder:
    config = PhysEvolveOnlineStepBuilderConfig(
        scenes_per_batch=overrides.pop("scenes_per_batch", 2),
        group_size=overrides.pop("group_size", 1),
        task_modes=overrides.pop("task_modes", ("forward",)),
        response_length=16,
        skill_seed=overrides.pop("skill_seed", 0),
        **overrides,
    )
    # The sampler does not touch the backend or executor; we can pass None
    # because we never call ``__call__``.
    return PhysEvolveOnlineStepBuilder(
        config,
        run_dir=tmp_path,
        backend_factory=lambda _config: None,  # type: ignore[arg-type]
        executor_fn=lambda *_a, **_kw: None,
    )


def test_pick_skill_is_deterministic_given_seed(tmp_path: Path) -> None:
    builder = _make_builder(tmp_path, skill_seed=42)
    first = [builder._pick_skill(step=1, item_idx=i) for i in range(16)]
    second = [builder._pick_skill(step=1, item_idx=i) for i in range(16)]
    assert first == second


def test_pick_skill_returns_known_vocabulary(tmp_path: Path) -> None:
    builder = _make_builder(tmp_path)
    for step in range(1, 8):
        for item_idx in range(8):
            skill = builder._pick_skill(step=step, item_idx=item_idx)
            assert skill in PHYSICS_SKILL_TARGETS, skill


def test_pick_skill_covers_most_of_vocabulary_in_64_samples(tmp_path: Path) -> None:
    """Sampling 8 items x 8 steps should cover at least 6/8 skills.

    With a uniform-random sampler over an 8-element vocabulary, the
    expected coverage of 64 draws is ~7.99/8; allowing 6/8 leaves
    significant slack for skill_seed variation without flaking.
    """

    builder = _make_builder(tmp_path)
    seen: set[str] = set()
    for step in range(1, 9):
        for item_idx in range(8):
            seen.add(builder._pick_skill(step=step, item_idx=item_idx))
    assert len(seen) >= 6, sorted(seen)


def test_pick_skill_distribution_is_not_collapsed(tmp_path: Path) -> None:
    """No single skill may dominate the sampler.

    Sampling 8 items x 16 steps = 128 draws over an 8-element vocab;
    expected per-skill count is 16. We require the most common skill
    to be at most 6 times more frequent than the least common one,
    which is a generous bound on a Multinomial(128, 1/8) sampler.
    """

    builder = _make_builder(tmp_path)
    counts: Counter[str] = Counter()
    for step in range(1, 17):
        for item_idx in range(8):
            counts[builder._pick_skill(step=step, item_idx=item_idx)] += 1
    most = max(counts.values())
    least = min(counts.values()) if len(counts) == len(PHYSICS_SKILL_TARGETS) else 1
    assert least >= 1
    assert most <= 6 * least, counts


def test_skill_seed_changes_assignment(tmp_path: Path) -> None:
    """Different skill_seed values must produce different assignment streams."""

    builder_a = _make_builder(tmp_path, skill_seed=0)
    builder_b = _make_builder(tmp_path, skill_seed=1)
    a = [builder_a._pick_skill(step=1, item_idx=i) for i in range(16)]
    b = [builder_b._pick_skill(step=1, item_idx=i) for i in range(16)]
    assert a != b, "skill_seed had no effect on the sampler"
