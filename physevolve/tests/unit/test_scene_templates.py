"""Tests for the scene-template (multi-GT) path.

Covers the "one reusable scene -> many verifier-grounded VQAs" pipeline:

* the ``throwing_pit_board`` template enumerates final-destination / yes-no /
  counterfactual / inverse questions from one simulation (+ branch sims);
* every derived question's ground truth comes from the (baseline or branch)
  trajectory, never from the Proposer;
* deterministic option shuffling spreads the correct letter;
* ``rgba_to_color_name`` + colour-permutation keep colour labels aligned with
  the body identity (GT unchanged when colours are swapped);
* the full executor path (``run_executor(stage=2)`` with ``SCENE_MANIFEST``)
  returns a non-trivial list of payloads.

All physics here is real MuJoCo (CPU); the fewshot scene is the golden
``throwing_pit_board`` example, which is exercised end-to-end.
"""

from __future__ import annotations

from collections import Counter

import pytest

from physevolve.executor import run_executor
from physevolve.inference.parsing import extract_propose_block
from physevolve.skills.scene_templates import (
    fewshot_examples_for,
    get_scene_template_registry,
)
from physevolve.skills.scene_templates.colors import (
    color_permuted_scene,
    rgba_to_color_name,
)


def _fewshot_code() -> str:
    template = get_scene_template_registry().lookup("throwing_pit_board")
    code = extract_propose_block(template.fewshot_example)
    assert code is not None
    return code


# ─── registry ────────────────────────────────────────────────────────


def test_registry_has_throwing_pit_board() -> None:
    reg = get_scene_template_registry()
    assert "throwing_pit_board" in reg.available_scene_kinds()
    tpl = reg.lookup("throwing_pit_board")
    assert "throwing_final_destination" in tpl.families
    assert "counterfactual_destination" in tpl.families
    assert "inverse_intervention" in tpl.families


def test_unknown_scene_kind_raises() -> None:
    with pytest.raises(ValueError):
        get_scene_template_registry().lookup("does_not_exist")


# ─── end-to-end executor (multi-GT) ───────────────────────────────────


def test_throwing_pit_board_fewshot_generates_multiple_questions() -> None:
    result = run_executor(_fewshot_code(), stage=2, seed=7)
    assert result.ok, result.error_message
    assert result.vqa_payloads is not None
    # One scene -> several questions.
    assert len(result.vqa_payloads) >= 3
    # Every payload is a valid (question, 4 options, correct_idx in [0,3]).
    for item in result.vqa_payloads:
        question, options, correct_idx = item["payload"]
        assert isinstance(question, str) and question.strip()
        assert len(options) == 4 and all(isinstance(o, str) and o for o in options)
        assert 0 <= int(correct_idx) < 4
        assert "family" in item["spec"]
    # Questions read like natural language (no physdsl / underscores leaking).
    for item in result.vqa_payloads:
        assert "_" not in item["payload"][0]


def test_throwing_pit_board_covers_branch_families() -> None:
    result = run_executor(_fewshot_code(), stage=2, seed=7)
    assert result.ok
    families = {item["spec"]["family"] for item in (result.vqa_payloads or [])}
    # The validated fewshot must exercise both branch families at the default
    # six-distinct-simulation cap.
    assert "throwing_final_destination" in families
    assert "counterfactual_destination" in families
    assert "inverse_intervention" in families


def test_throwing_branch_order_completes_inverse_and_reuses_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physevolve.executor import branch_sim
    from physevolve.skills.scene_templates import throwing_pit_board as tpb

    manifest = {
        "balls": ["ball_a", "ball_b", "ball_c"],
        "walls": ["wall_a", "wall_b", "wall_c", "wall_d"],
        "families": ["counterfactual_destination", "inverse_intervention"],
    }
    specs = tpb._required_branches(manifest)
    assert [spec.key for spec in specs[:6]] == [
        "inv|ball_a|a",
        "inv|ball_a|b",
        "inv|ball_a|both",
        "inv|ball_b|a",
        "inv|ball_b|b",
        "inv|ball_b|both",
    ]

    calls: list[frozenset[str]] = []

    def fake_simulate_with_removed(
        _scene_def: object,
        remove_names: tuple[str, ...],
        *,
        seed: int,
        sim_timeout_s: float,
    ) -> object:
        del seed, sim_timeout_s
        calls.append(frozenset(remove_names))
        return object()

    monkeypatch.setattr(
        branch_sim,
        "simulate_with_removed",
        fake_simulate_with_removed,
    )
    views = branch_sim.run_branch_sims(
        object(),
        specs,
        seed=7,
        sim_timeout_s=1.0,
        max_branches=6,
    )

    assert len(calls) == 6
    assert all(f"inv|{ball}|both" in views for ball in ("ball_a", "ball_b"))
    for ball in ("ball_a", "ball_b"):
        assert views[f"cf|{ball}|wall_a"] is views[f"inv|{ball}|a"]
        assert views[f"cf|{ball}|wall_b"] is views[f"inv|{ball}|b"]
        assert f"cf|{ball}|wall_c" not in views
        assert f"cf|{ball}|wall_d" not in views


def test_throwing_cf_only_branch_order_is_unchanged() -> None:
    from physevolve.skills.scene_templates import throwing_pit_board as tpb

    manifest = {
        "balls": ["ball_a", "ball_b", "ball_c"],
        "walls": ["wall_a", "wall_b", "wall_c"],
        "families": ["counterfactual_destination"],
    }
    specs = tpb._required_branches(manifest)

    assert [spec.key for spec in specs] == [
        f"cf|{ball}|{wall}"
        for ball in ("ball_a", "ball_b")
        for wall in ("wall_a", "wall_b", "wall_c")
    ]


def test_generated_questions_are_deterministic() -> None:
    a = run_executor(_fewshot_code(), stage=2, seed=7)
    b = run_executor(_fewshot_code(), stage=2, seed=7)
    assert [it["payload"] for it in a.vqa_payloads] == [
        it["payload"] for it in b.vqa_payloads
    ]


def test_option_shuffle_spreads_letters_across_questions() -> None:
    # Across many derived questions (over several seeds), the correct letter
    # should not be pinned to a single slot.
    letters: Counter[int] = Counter()
    for seed in range(8):
        result = run_executor(_fewshot_code(), stage=2, seed=seed)
        if not result.ok or not result.vqa_payloads:
            continue
        for item in result.vqa_payloads:
            letters[int(item["payload"][2])] += 1
    assert sum(letters.values()) > 0
    # At least 3 distinct correct-answer positions appear.
    assert len(letters) >= 3, letters


# ─── colour naming + permutation (GT-by-identity) ─────────────────────


def test_rgba_to_color_name_basic() -> None:
    assert rgba_to_color_name((0.9, 0.15, 0.15, 1.0)) == "red"
    assert rgba_to_color_name((0.2, 0.3, 0.9, 1.0)) == "blue"
    assert rgba_to_color_name((0.2, 0.75, 0.25, 1.0)) == "green"


def test_color_permutation_preserves_physics_and_remaps_labels() -> None:
    from physevolve.engine.physdsl import PhysDSLApi
    from physevolve.engine.simulator import simulate

    import numpy as np

    from physevolve.executor.sandbox import execute_proposer_code_with_spec

    build_scene, _g, _spec, _man = execute_proposer_code_with_spec(_fewshot_code())
    api = PhysDSLApi()
    build_scene(api)
    scene = api.freeze()

    # Balls are neutrally named (ball_a/b/c); colour comes from the rgba only.
    permuted, cmap = color_permuted_scene(
        scene, ["ball_a", "ball_b"], ["white", "orange"]
    )
    assert cmap == {"ball_a": "white", "ball_b": "orange"}

    s_orig = simulate(scene, seed=7)
    s_perm = simulate(permuted, seed=7)
    # Colours do not affect physics: trajectories identical -> GT (by body
    # identity) unchanged; only the colour label flips.
    for body in s_orig.trajectories:
        assert np.allclose(
            s_orig.trajectories[body].positions,
            s_perm.trajectories[body].positions,
        )
    assert rgba_to_color_name(
        next(b.rgba for b in permuted.bodies if b.name == "ball_a")
    ) == "white"


# ─── manifest validation ──────────────────────────────────────────────


def test_manifest_rejects_fixed_ball() -> None:
    code = """
def build_scene(api):
    # Two visible platforms; the gap between them is the pit.
    api.add_box("left_platform", pos=(-0.6, 0.0, 0.10), size=(0.25, 0.4, 0.10),
                mass=20.0, rgba=(0.4, 0.4, 0.4, 1.0), friction=1.0,
                restitution=0.0, fixed=True)
    api.add_box("right_platform", pos=(0.6, 0.0, 0.10), size=(0.25, 0.4, 0.10),
                mass=20.0, rgba=(0.4, 0.4, 0.4, 1.0), friction=1.0,
                restitution=0.0, fixed=True)
    # A free moving body so the scene is NOT trivial (reaches manifest check).
    api.add_sphere("mover", pos=(-0.8, 0.3, 0.30), radius=0.06, mass=0.5,
                   rgba=(0.2, 0.75, 0.25, 1.0), friction=0.6, restitution=0.1,
                   fixed=False)
    api.set_velocity("mover", linear=(1.5, 0.0, 0.0), angular=(0.0, 0.0, 0.0))
    # The queried ball is FIXED -> manifest must reject it.
    api.add_sphere("red_ball", pos=(-0.5, -0.3, 0.20), radius=0.06,
                   mass=0.5, rgba=(0.9, 0.15, 0.15, 1.0), friction=0.6,
                   restitution=0.0, fixed=True)
    api.set_duration(1.5)

SCENE_MANIFEST = {
    "scene_kind": "throwing_pit_board",
    "balls": ["red_ball"],
    "platforms": ["left_platform", "right_platform"],
    "families": ["throwing_final_destination"],
}
"""
    result = run_executor(code, stage=2, seed=1)
    # red_ball is fixed -> manifest invalid -> FAILED_VQA.
    assert not result.ok
    assert result.status.value == "vqa"


# ─── gap-region derivation (no rendering) ──────────────────────────────
#
# These exercise the simplified throwing contract: the manifest names platform
# bodies, and the executor derives left/middle/right gap regions from repaired
# platform geometry. Fake SimStateView stubs keep them CPU-only.


def _build_gap_scene(*, platform_overrides=None, ball_fixed=False):
    """Freeze a minimal valid gap-pit scene; return (scene, manifest).

    Three platforms create three gaps: left (between p_left & p_mid), middle
    (between p_mid & p_right) and right (off the right edge of p_right).
    ``platform_overrides`` lets a test mutate a platform to provoke failures.
    """

    from physevolve.engine.physdsl import PhysDSLApi

    platforms = {
        "p_left": dict(pos=(-0.75, 0.0, 0.10), size=(0.25, 0.4, 0.10)),
        "p_mid": dict(pos=(0.05, 0.0, 0.10), size=(0.25, 0.4, 0.10)),
        "p_right": dict(pos=(0.85, 0.0, 0.10), size=(0.25, 0.4, 0.10)),
    }
    for name, override in (platform_overrides or {}).items():
        platforms[name].update(override)

    api = PhysDSLApi()
    for name, kw in platforms.items():
        api.add_box(name, pos=kw["pos"], size=kw["size"], mass=10.0,
                    rgba=(0.4, 0.4, 0.4, 1.0), friction=1.0,
                    restitution=0.0, fixed=True)
    api.add_box("red_wall", pos=(-0.2, 0.25, 0.45), size=(0.04, 0.18, 0.12),
                mass=10.0, rgba=(0.9, 0.15, 0.15, 1.0), friction=0.5,
                restitution=0.0, fixed=True)
    api.add_box("blue_wall", pos=(0.3, -0.2, 0.45), size=(0.04, 0.18, 0.12),
                mass=10.0, rgba=(0.2, 0.3, 0.9, 1.0), friction=0.5,
                restitution=0.0, fixed=True)
    api.add_sphere("ball_a", pos=(-1.0, 0.0, 0.60), radius=0.06, mass=0.5,
                   rgba=(0.95, 0.55, 0.15, 1.0), friction=0.6, restitution=0.1,
                   fixed=ball_fixed)
    api.set_velocity("ball_a", linear=(1.6, 0.0, 0.0),
                     angular=(0.0, 0.0, 0.0))
    api.set_duration(2.0)
    scene = api.freeze()
    # p_left x=[-1.0,-0.5], p_mid x=[-0.2,0.3], p_right x=[0.6,1.1].
    manifest = {
        "scene_kind": "throwing_pit_board",
        "balls": ["ball_a"],
        "platforms": ["p_left", "p_mid", "p_right"],
        "walls": ["red_wall", "blue_wall"],
        "region_kind": "pit",
        "families": ["throwing_final_destination", "inverse_intervention"],
    }
    return scene, manifest


class _FakeBody:
    def __init__(self, pos, vel, status="stable", initial_pos=None):
        self.final_pos = pos
        self.final_vel = vel
        self.final_status = status
        self.initial_pos = initial_pos if initial_pos is not None else pos


class _FakeView:
    def __init__(self, bodies):
        self._bodies = bodies

    def __contains__(self, name):
        return name in self._bodies

    def __getitem__(self, name):
        return self._bodies[name]


def test_regions_are_derived_from_platform_gaps() -> None:
    from physevolve.skills.scene_templates import throwing_pit_board as tpb

    scene, manifest = _build_gap_scene()
    regions = tpb._derive_regions_from_platforms(tpb._platforms(manifest), scene)
    assert regions is not None
    assert regions["left"]["x"] == [-0.5, -0.2]
    assert regions["middle"]["x"] == [0.3, 0.6]
    # z_ceiling = lowest platform top = 0.10 + 0.10.
    assert abs(float(regions["left"]["z_ceiling"]) - 0.20) < 1e-9


def test_throwing_compile_manifest_derives_pits_from_platforms() -> None:
    from physevolve.executor.repair import RepairTrace
    from physevolve.skills.scene_templates import throwing_pit_board as tpb

    scene, manifest = _build_gap_scene()
    raw = dict(manifest)
    trace = RepairTrace()
    compiled = tpb._compile_manifest(raw, scene, trace)
    assert compiled["pits"]["left"] == {"x": [-0.5, -0.2]}
    assert compiled["pits"]["middle"] == {"x": [0.3, 0.6]}
    assert compiled["pits"]["right"] == {"x": [1.1, 1.5]}
    assert any(action.kind == "derive_throwing_pits" for action in trace.actions)


def test_throwing_compile_manifest_overrides_authored_pits() -> None:
    from physevolve.executor.repair import RepairTrace
    from physevolve.skills.scene_templates import throwing_pit_board as tpb

    scene, manifest = _build_gap_scene()
    raw = dict(manifest)
    raw["pits"] = {
        "left": {"x": [-0.9, -0.6]},  # invalid authored platform footprint
        "middle": {"x": [0.3, 0.6]},
        "right": {"x": [1.42, 1.49]},
    }
    trace = RepairTrace()
    compiled = tpb._compile_manifest(raw, scene, trace)
    assert compiled["pits"]["left"] == {"x": [-0.5, -0.2]}
    assert compiled["pits"]["right"] == {"x": [1.1, 1.5]}
    assert any(action.kind == "derive_throwing_pits" for action in trace.actions)


def test_throwing_compile_manifest_aligns_role_name_spelling() -> None:
    from physevolve.executor.repair import RepairTrace
    from physevolve.skills.scene_templates import throwing_pit_board as tpb

    scene, manifest = _build_gap_scene()
    raw = dict(manifest)
    raw["platforms"] = ["P Left", "p_mid", "p_right"]
    trace = RepairTrace()
    compiled = tpb._compile_manifest(raw, scene, trace)
    assert compiled["platforms"][0] == "p_left"
    assert any(action.kind == "align_manifest_roles" for action in trace.actions)


def test_settle_in_gap_rejects_moving_ejected_on_platform_and_out() -> None:
    from physevolve.skills.scene_templates import throwing_pit_board as tpb

    scene, manifest = _build_gap_scene()
    regions = tpb._derive_regions_from_platforms(tpb._platforms(manifest), scene)
    assert regions is not None
    # At rest in the left gap on the ground -> resolved "left".
    rest = _FakeView({"ball_a": _FakeBody((-0.35, 0.0, 0.06), (0.01, 0, 0))})
    assert tpb._settled_region_in(rest, "ball_a", regions) == "left"
    # Still moving -> None.
    moving = _FakeView({"ball_a": _FakeBody((-0.35, 0.0, 0.06), (1.5, 0, 0))})
    assert tpb._settled_region_in(moving, "ball_a", regions) is None
    # Ejected -> None.
    ej = _FakeView(
        {"ball_a": _FakeBody((-0.35, 0.0, 0.06), (0.01, 0, 0), "ejected")}
    )
    assert tpb._settled_region_in(ej, "ball_a", regions) is None
    # Resting ON a platform (x not in any gap interval) -> None.
    on_plat = _FakeView({"ball_a": _FakeBody((0.05, 0.0, 0.27), (0.01, 0, 0))})
    assert tpb._settled_region_in(on_plat, "ball_a", regions) is None
    # Stuck high on a wall inside a gap x-range (z above ceiling+band) -> None.
    high = _FakeView({"ball_a": _FakeBody((-0.35, 0.0, 0.9), (0.01, 0, 0))})
    assert tpb._settled_region_in(high, "ball_a", regions) is None


def test_inverse_question_requires_all_simulated_branches_and_uses_unique_lever() -> None:
    """Inverse GT must come from complete branch views, never a guessed outcome."""

    from physevolve.skills.scene_templates import throwing_pit_board as tpb

    scene, manifest = _build_gap_scene()
    regions = tpb._derive_regions_from_platforms(tpb._platforms(manifest), scene)
    assert regions is not None

    def view_at(x: float) -> _FakeView:
        return _FakeView(
            {
                "ball_a": _FakeBody(
                    (x, 0.0, 0.06),
                    (0.01, 0.0, 0.0),
                    initial_pos=(-1.0, 0.0, 0.60),
                )
            }
        )

    baseline = view_at(-0.35)  # left pit
    branch_views = {
        "inv|ball_a|a": view_at(0.45),  # middle pit: unique achiever
        "inv|ball_a|b": view_at(1.25),  # right pit
        "inv|ball_a|both": view_at(1.25),  # right pit
    }
    kwargs = {
        "ball": "ball_a",
        "wall_a": "red_wall",
        "wall_b": "blue_wall",
        "region_kind": "pit",
        "regions": regions,
        "baseline_view": baseline,
        "body_colors": {"ball_a": "orange", "red_wall": "red", "blue_wall": "blue"},
        "spatial": {},
    }

    incomplete = dict(branch_views)
    incomplete.pop("inv|ball_a|both")
    assert tpb._build_inverse_question(branch_views=incomplete, **kwargs) is None

    vqa = tpb._build_inverse_question(branch_views=branch_views, **kwargs)
    assert vqa is not None
    assert vqa.spec["target"] == "middle"
    assert vqa.spec["answer_semantic"] == "remove_a"
    assert vqa.options[vqa.correct_idx] == "Remove the red floating wall and other balls"


def test_contains_color_word() -> None:
    from physevolve.skills.scene_templates.colors import contains_color_word

    assert contains_color_word("green_ball")
    assert contains_color_word("blue_wall")
    assert contains_color_word("orange")
    assert not contains_color_word("ball_a")
    assert not contains_color_word("p_left")
    assert not contains_color_word("striker")


def test_validate_rejects_colour_named_ball() -> None:
    """The throwing manifest must reject a colour-baked ball body name."""

    from physevolve.engine.physdsl import PhysDSLApi
    from physevolve.skills.scene_templates import throwing_pit_board as tpb

    api = PhysDSLApi()
    for name, x in (("p_left", -0.75), ("p_mid", 0.05), ("p_right", 0.85)):
        api.add_box(name, pos=(x, 0.0, 0.10), size=(0.25, 0.4, 0.10), mass=10.0,
                    rgba=(0.4, 0.4, 0.4, 1.0), friction=1.0, restitution=0.0,
                    fixed=True)
    # A COLOUR-NAMED ball -> must be rejected (use ball_a + rgba instead).
    api.add_sphere("green_ball", pos=(-1.0, 0.0, 0.60), radius=0.06, mass=0.5,
                   rgba=(0.2, 0.75, 0.25, 1.0), friction=0.6, restitution=0.1,
                   fixed=False)
    api.set_velocity("green_ball", linear=(1.6, 0.0, 0.0), angular=(0.0, 0.0, 0.0))
    api.set_duration(2.0)
    scene = api.freeze()
    manifest = {
        "scene_kind": "throwing_pit_board",
        "balls": ["green_ball"],
        "platforms": ["p_left", "p_mid", "p_right"],
        "region_kind": "pit",
        "families": ["throwing_final_destination"],
    }
    reason = tpb._validate_manifest(manifest, scene)
    assert reason is not None
    assert "colour-named" in reason


def test_spatial_ref_map_and_ref_phrase() -> None:
    """Unambiguous extreme balls get a spatial phrase; others keep colour."""

    from physevolve.skills.scene_templates import throwing_pit_board as tpb

    view = _FakeView(
        {
            "ball_a": _FakeBody((-1.0, 0, 0.06), (0, 0, 0), initial_pos=(-1.0, 0, 0.6)),
            "ball_b": _FakeBody((0.0, 0, 0.06), (0, 0, 0), initial_pos=(0.0, 0, 0.6)),
            "ball_c": _FakeBody((1.0, 0, 0.06), (0, 0, 0), initial_pos=(1.0, 0, 0.6)),
        }
    )
    balls = ["ball_a", "ball_b", "ball_c"]
    spatial = tpb._spatial_ref_map(balls, view)
    assert spatial["ball_a"] == "leftmost launched ball"
    assert spatial["ball_c"] == "rightmost launched ball"
    assert "ball_b" not in spatial  # middle ball is not a clear extreme
    colors = {"ball_a": "green", "ball_b": "orange", "ball_c": "blue"}
    # Extremes -> spatial phrase; middle -> colour phrase.
    assert tpb._ref_phrase("ball_a", colors, spatial) == "leftmost launched ball"
    assert tpb._ref_phrase("ball_b", colors, spatial) == "orange ball"


def test_spatial_ref_map_skips_ambiguous_extremes() -> None:
    """Balls clustered in x get no spatial phrase (would be ambiguous)."""

    from physevolve.skills.scene_templates import throwing_pit_board as tpb

    view = _FakeView(
        {
            "ball_a": _FakeBody((-0.05, 0, 0.06), (0, 0, 0), initial_pos=(-0.05, 0, 0.6)),
            "ball_b": _FakeBody((0.0, 0, 0.06), (0, 0, 0), initial_pos=(0.0, 0, 0.6)),
        }
    )
    spatial = tpb._spatial_ref_map(["ball_a", "ball_b"], view)
    assert spatial == {}


def test_throwing_generates_catalog_phrasing_variants() -> None:
    from physevolve.skills.scene_templates import throwing_pit_board as tpb

    scene, manifest = _build_gap_scene()
    manifest = dict(manifest)
    manifest["balls"] = ["ball_a"]
    manifest["families"] = [
        "throwing_final_destination",
        "throwing_current_run_judgment",
    ]
    regions = tpb._derive_regions_from_platforms(tpb._platforms(manifest), scene)
    assert regions is not None
    view = _FakeView(
        {
            "ball_a": _FakeBody(
                (-0.35, 0.0, 0.06),
                (0.01, 0, 0),
                initial_pos=(-1.0, 0.0, 0.60),
            )
        }
    )
    gen = list(tpb._generate_questions(manifest, view, {}, {}, scene))
    by_phrasing = {g.phrasing: g for g in gen}

    assert {"forward", "fate", "yes_no", "wrong_pit"} <= set(by_phrasing)
    assert by_phrasing["fate"].family == "throwing_final_destination"
    assert "eventually happen" in by_phrasing["fate"].question
    assert by_phrasing["wrong_pit"].family == "throwing_current_run_judgment"
    assert "different pit than" in by_phrasing["wrong_pit"].question


def test_validate_ignores_authored_pits_when_platforms_are_valid() -> None:
    from physevolve.skills.scene_templates import throwing_pit_board as tpb

    scene, manifest = _build_gap_scene()
    manifest = dict(manifest)
    manifest["pits"] = {
        "left": {"x": [-0.9, -0.6]},  # old bad footprint, now ignored
        "middle": {"x": [0.0, 0.1]},
        "right": {"x": [1.42, 1.49]},
    }
    reason = tpb._validate_manifest(manifest, scene)
    assert reason is None


def test_validate_rejects_platforms_without_positive_gaps() -> None:
    from physevolve.skills.scene_templates import throwing_pit_board as tpb

    scene, manifest = _build_gap_scene(
        platform_overrides={"p_mid": {"pos": (-0.25, 0.0, 0.10)}}
    )
    reason = tpb._validate_manifest(manifest, scene)
    assert reason is not None and "positive-width visible x-gaps" in reason


def test_validate_rejects_too_few_platforms() -> None:
    from physevolve.skills.scene_templates import throwing_pit_board as tpb

    scene, manifest = _build_gap_scene()
    manifest = dict(manifest)
    manifest["platforms"] = ["p_left"]  # only one
    reason = tpb._validate_manifest(manifest, scene)
    assert reason is not None and ">= 3 platforms" in reason


def test_validate_accepts_golden_gap_scene() -> None:
    from physevolve.skills.scene_templates import throwing_pit_board as tpb

    scene, manifest = _build_gap_scene()
    assert tpb._validate_manifest(manifest, scene) is None


# ─── camera_depth (pure-geometry, no rendering) ────────────────────────


def _build_camera_scene():
    """Freeze a static 4-cube + camera scene; return (scene, manifest).

    red_cube is nearest the camera, yellow_cube is farthest.
    """

    from physevolve.engine.physdsl import PhysDSLApi

    api = PhysDSLApi()
    api.add_plane("ground", pos=(0.0, 0.0, 0.0), size=(5.0, 5.0))
    api.add_box("red_cube", pos=(0.0, -1.0, 0.15), size=(0.12, 0.12, 0.12),
                rgba=(0.9, 0.15, 0.15, 1.0), fixed=True)
    api.add_box("green_cube", pos=(0.6, 0.2, 0.15), size=(0.12, 0.12, 0.12),
                rgba=(0.2, 0.75, 0.25, 1.0), fixed=True)
    api.add_box("blue_cube", pos=(-0.7, 0.5, 0.15), size=(0.12, 0.12, 0.12),
                rgba=(0.2, 0.3, 0.9, 1.0), fixed=True)
    api.add_box("yellow_cube", pos=(0.1, 1.2, 0.15), size=(0.12, 0.12, 0.12),
                rgba=(0.95, 0.9, 0.2, 1.0), fixed=True)
    api.add_camera("cam0", pos=(0.0, -2.6, 1.1), target=(0.0, 0.0, 0.15))
    api.set_duration(1.0)
    scene = api.freeze()
    manifest = {
        "scene_kind": "camera_depth_closest_object",
        "phrasing": "static",
        "candidate_bodies": ["red_cube", "green_cube", "blue_cube", "yellow_cube"],
        "camera": "cam0",
    }
    return scene, manifest


def test_camera_depth_registered() -> None:
    reg = get_scene_template_registry()
    assert "camera_depth_closest_object" in reg.available_scene_kinds()
    tpl = reg.lookup("camera_depth_closest_object")
    assert tpl.families == ("camera_depth_closest_object",)


def test_camera_depth_closest_gt() -> None:
    from physevolve.skills.scene_templates import camera_depth as cd

    scene, manifest = _build_camera_scene()
    assert cd._validate_manifest(manifest, scene) is None
    gen = list(cd._generate_questions(manifest, None, {}, {}, scene))
    assert len(gen) == 1
    vqa = gen[0]
    assert vqa.family == "camera_depth_closest_object"
    assert vqa.options[vqa.correct_idx] == "The red cube"


def test_camera_depth_farthest_gt() -> None:
    from physevolve.skills.scene_templates import camera_depth as cd

    scene, manifest = _build_camera_scene()
    manifest = dict(manifest)
    manifest["phrasing"] = "farthest"
    gen = list(cd._generate_questions(manifest, None, {}, {}, scene))
    assert gen[0].options[gen[0].correct_idx] == "The yellow cube"


def test_camera_depth_rejects_ambiguous() -> None:
    from physevolve.skills.scene_templates import camera_depth as cd

    scene, manifest = _build_camera_scene()
    # Override candidate set with two equidistant cubes via a tweaked scene:
    # put green_cube symmetric to red_cube about the camera's y so distances tie.
    from physevolve.engine.physdsl import PhysDSLApi

    api = PhysDSLApi()
    api.add_plane("ground", pos=(0.0, 0.0, 0.0), size=(5.0, 5.0))
    # red & green both 1.6 m from cam eye (0,-2.6,1.1): mirror across x.
    api.add_box("red_cube", pos=(-0.5, -1.2, 0.15), size=(0.12, 0.12, 0.12),
                rgba=(0.9, 0.15, 0.15, 1.0), fixed=True)
    api.add_box("green_cube", pos=(0.5, -1.2, 0.15), size=(0.12, 0.12, 0.12),
                rgba=(0.2, 0.75, 0.25, 1.0), fixed=True)
    api.add_box("blue_cube", pos=(-0.7, 0.5, 0.15), size=(0.12, 0.12, 0.12),
                rgba=(0.2, 0.3, 0.9, 1.0), fixed=True)
    api.add_box("yellow_cube", pos=(0.1, 1.2, 0.15), size=(0.12, 0.12, 0.12),
                rgba=(0.95, 0.9, 0.2, 1.0), fixed=True)
    api.add_camera("cam0", pos=(0.0, -2.6, 1.1), target=(0.0, 0.0, 0.15))
    api.set_duration(1.0)
    tied = api.freeze()
    reason = cd._validate_manifest(manifest, tied)
    assert reason is not None and "ambiguous" in reason


def test_camera_depth_rejects_missing_body() -> None:
    from physevolve.skills.scene_templates import camera_depth as cd

    scene, manifest = _build_camera_scene()
    manifest = dict(manifest)
    manifest["candidate_bodies"] = ["red_cube", "green_cube", "blue_cube", "ghost"]
    reason = cd._validate_manifest(manifest, scene)
    assert reason is not None and "ghost" in reason


def test_camera_depth_bypass_attaches_only_to_dynamics_sources() -> None:
    from physevolve.executor.pipeline import _camera_depth_bypass_payload
    from physevolve.skills.scene_templates.colors import body_color_map

    scene, depth_manifest = _build_camera_scene()
    candidates = list(depth_manifest["candidate_bodies"])
    source_manifest = {
        "scene_kind": "collision_arena",
        "movers": candidates,
    }
    item = _camera_depth_bypass_payload(
        source_scene_kind="collision_arena",
        source_manifest=source_manifest,
        scene_def=scene,
        baseline_view=None,
        body_colors=body_color_map(scene),
        registry=get_scene_template_registry(),
        scene_id="depth-bypass-test",
    )

    assert item is not None
    assert item["spec"]["family"] == "camera_depth_closest_object"
    assert item["spec"]["scene_kind"] == "collision_arena"
    assert item["spec"]["attached_scene_kind"] == "camera_depth_closest_object"
    assert item["spec"]["bypass"] is True
    assert len(item["payload"][1]) == 4

    assert (
        _camera_depth_bypass_payload(
            source_scene_kind="throwing_pit_board",
            source_manifest=source_manifest,
            scene_def=scene,
            baseline_view=None,
            body_colors=body_color_map(scene),
            registry=get_scene_template_registry(),
            scene_id="no-throw-depth",
        )
        is None
    )


# ─── grouped scene_kinds registered (collision_arena / motion_lab) ────
#
# Detailed per-family GT/validation for the grouped templates lives in
# test_collision_arena.py and test_motion_lab.py (CPU, fake SimStateView). Here
# we only assert the registry exposes the three active scene_kinds plus the two
# frozen ones, and that their fewshots parse.

_ACTIVE_GROUPED_SCENE_KINDS = (
    "throwing_pit_board",
    "collision_arena",
    "motion_lab",
)
_FROZEN_SCENE_KINDS = (
    "motion_phenomenon_classification",
    "camera_depth_closest_object",
)


def test_registry_has_active_and_frozen_scene_kinds() -> None:
    reg = get_scene_template_registry()
    kinds = set(reg.available_scene_kinds())
    for sk in _ACTIVE_GROUPED_SCENE_KINDS + _FROZEN_SCENE_KINDS:
        assert sk in kinds, f"missing scene_kind {sk!r}"


def test_grouped_fewshots_parse_and_carry_scene_manifest() -> None:
    reg = get_scene_template_registry()
    for sk in _ACTIVE_GROUPED_SCENE_KINDS + _FROZEN_SCENE_KINDS:
        tpl = reg.lookup(sk)
        code = extract_propose_block(tpl.fewshot_example)
        assert code is not None, f"{sk} fewshot has no <propose> block"
        assert "SCENE_MANIFEST" in code
        assert f'"scene_kind": "{sk}"' in code


def test_active_grouped_fewshot_variants_parse_build_and_validate() -> None:
    from physevolve.engine.physdsl import PhysDSLApi
    from physevolve.executor.sandbox import execute_proposer_code_with_spec

    expected_counts = {
        "throwing_pit_board": 10,
        "collision_arena": 8,
        "motion_lab": 6,
    }
    reg = get_scene_template_registry()
    for sk, expected_count in expected_counts.items():
        tpl = reg.lookup(sk)
        examples = fewshot_examples_for(tpl)
        assert len(examples) == expected_count
        for idx, example in enumerate(examples):
            code = extract_propose_block(example)
            assert code is not None, f"{sk} variant {idx} has no <propose> block"
            build_scene, _get_vqa, _spec, manifest = execute_proposer_code_with_spec(
                code
            )
            api = PhysDSLApi()
            build_scene(api)
            scene = api.freeze()
            compiled = (
                tpl.compile_manifest(manifest, scene, None)
                if tpl.compile_manifest is not None
                else manifest
            )
            reason = tpl.validate_manifest(compiled, scene)
            assert reason is None, f"{sk} variant {idx} rejected: {reason}"
            if sk == "throwing_pit_board":
                assert '"pits": {' not in example
                assert "pits" not in manifest
