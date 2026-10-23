"""Validate, simulate, and render Proposer programs.

The pipeline runs the sandbox, builds and freezes the scene, then simulates
its baseline and required branches. Stage-2 templates derive questions and
answers from read-only simulator views. Bounded repair actions are recorded.

Expected failures become ``ExecutorResult`` statuses; unexpected programming
errors propagate to the caller.
"""

from __future__ import annotations

import threading
import time
from itertools import combinations
from typing import Any, Callable, Literal, Optional

import mujoco

from physevolve.engine.failure import (
    DSLValidationError,
    FailureReason,
    InitialOverlapError,
    SimulationDivergedError,
    SimulationTimeoutError,
)
from physevolve.engine.physdsl import PhysDSLApi
from physevolve.engine.schema import SceneDef, SimState, compute_scene_id
from physevolve.executor.result import (
    ExecutorResult,
    make_failure,
    make_ok,
    make_trivial,
)
from physevolve.executor.repair import (
    RepairPolicy,
    RepairTrace,
    repair_scene_def,
    wrap_physdsl_api,
)
from physevolve.executor.sandbox import (
    RestrictedExecError,
    _timeout_guard,
    execute_proposer_code_with_spec,
)
from physevolve.executor.state_view import sim_state_public_view
from physevolve.skills.scene_templates import (
    GeneratedVQA,
    SceneTemplate,
    SceneTemplateRegistry,
    get_scene_template_registry,
)
from physevolve.skills.scene_templates.option_banks import apply_option_layout


# ─── Private renderer/sim wrappers ─────────────────────────────────────


# Scene_kinds whose questions are inherently STATIC (a spatial relationship,
# not a dynamics outcome). For these the no-motion triviality gate is skipped.
_STATIC_SCENE_KINDS: frozenset[str] = frozenset({"camera_depth_closest_object"})
_CAMERA_DEPTH_BYPASS_SOURCE_KINDS: frozenset[str] = frozenset(
    {"collision_arena", "motion_lab"}
)
_CAMERA_DEPTH_SCENE_KIND = "camera_depth_closest_object"
_MAX_CAMERA_DEPTH_BYPASS_CANDIDATES = 12


class _RenderError(RuntimeError):
    """Internal marker that a failure came from ``render_multiview``.

    Allows the outer try/except to distinguish renderer failures from
    physics failures even though both can surface as ``RuntimeError``.
    """


# Guards the module-attribute swap in :func:`_simulate_guarded`. The
# monkeypatch mutates ``engine.simulator.render_multiview``; without a
# lock, two threads entering ``run_executor`` concurrently could
# interleave their save/restore and one thread's ``_render_trap`` would
# survive into the other thread. Parallel fan-out uses processes
# (not threads), but single-process multithreaded harnesses, test
# runners with ``pytest-xdist -n auto``, and notebooks all share this
# module and must not corrupt each other.
_SIMULATE_LOCK: threading.Lock = threading.Lock()


def _repair_trace_payload(trace: RepairTrace) -> Optional[dict]:
    """Return JSON-safe repair metadata only when something was applied."""

    return trace.as_dict() if trace.applied else None


def _simulate_guarded(scene_def: SceneDef, seed: int, sim_timeout_s: float) -> SimState:
    """Run the full sim+render, re-raising renderer failures as _RenderError.

    The engine-level :func:`simulate` already enforces ``sim_timeout_s``
    via its per-step wall-clock check and re-raises MuJoCo exceptions
    unchanged — we only need to reclassify renderer-side errors.
    """

    from physevolve.engine import simulator as _sim  # local import to keep
    # top-level import graph lean (engine.simulator pulls in mujoco and
    # the renderer's EGL cache).

    with _SIMULATE_LOCK:
        real_render = _sim.render_multiview

        def _render_trap(*args, **kwargs):  # type: ignore[no-untyped-def]
            try:
                return real_render(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 — render failures are diverse.
                raise _RenderError(str(e)) from e

        # Monkeypatch only for the duration of this call; unconditionally
        # restored via try/finally even on re-raise.
        _sim.render_multiview = _render_trap  # type: ignore[assignment]
        try:
            return _sim.simulate(
                scene_def, seed=seed, sim_timeout_s=sim_timeout_s
            )
        finally:
            _sim.render_multiview = real_render  # type: ignore[assignment]


# ─── VQA validation ───────────────────────────────────────────────────


def _validate_vqa_payload(payload: Any) -> tuple[str, list[str], int]:
    """Shape-validate a ``(question, options, correct_idx)`` 3-tuple.

    Used as a defence-in-depth guard after the typed skill card has
    rendered the payload. Raises :class:`ValueError` on any shape /
    type / range violation; the pipeline maps this to
    :attr:`FailureReason.FAILED_VQA`.
    """

    if not isinstance(payload, tuple) or len(payload) != 3:
        raise ValueError(
            "VQA payload must be a (question, options, correct_idx) 3-tuple"
        )
    question, options, correct_idx = payload
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(options, list) or len(options) != 4:
        raise ValueError("options must be a list of exactly 4 strings")
    for i, opt in enumerate(options):
        if not isinstance(opt, str):
            raise ValueError(f"option {i} is not a string")
    if not isinstance(correct_idx, int) or isinstance(correct_idx, bool):
        raise ValueError("correct_idx must be an int (not bool)")
    if not (0 <= correct_idx < 4):
        raise ValueError("correct_idx must satisfy 0 <= idx < 4")
    return question, list(options), correct_idx


# ─── Main entry point ─────────────────────────────────────────────────


def run_executor(
    proposer_code: str,
    *,
    stage: Literal[1, 2] = 1,
    seed: int = 0,
    sandbox_timeout_s: float = 5.0,
    sim_timeout_s: float = 30.0,
    scene_template_registry: Optional[SceneTemplateRegistry] = None,
    max_branches: int = 6,
    color_variants_per_scene: int = 0,
    camera_depth_bypass_enabled: bool = False,
    skip_branch_sims: bool = False,
    trace: Optional[dict] = None,
) -> ExecutorResult:
    """Execute Proposer source through the full executor pipeline.

    Args:
        proposer_code: Python source defining ``build_scene(api)`` and
            (if ``stage == 2``) a top-level ``VQA_SPEC = {...}`` dict
            literal naming a typed Stage-2 skill.
        stage: ``1`` for Stage-1 curriculum (tri-mode tasks, no VQA),
            ``2`` for Stage-2 typed VQA self-play.
        seed: Simulation seed, mixed into ``scene_id``.
        sandbox_timeout_s: Wall-clock budget for the sandboxed ``exec``
            and, separately, for any Stage-2 helper callable invoked.
        sim_timeout_s: Wall-clock budget for the physics pipeline.
        scene_template_registry: Optional override of the canonical scene
            template registry. ``None`` uses
            :func:`get_scene_template_registry`. Tests inject custom registries.
        camera_depth_bypass_enabled: Attach at most one geometry-grounded
            camera-depth question to eligible collision/motion scenes. Disabled
            by default so earlier recipes remain unchanged.
        skip_branch_sims: When True, skip ``run_branch_sims`` and pass empty
            branch views to question generation. Disabled by default so the
            Stage-2 verifier still grounds counterfactual / inverse families.
        trace: Optional **diagnostic** sink. When a dict is passed, the
            Stage-2 branch fills it in-place with the manifest the
            executor parsed, baseline/branch settle outcomes, and the
            pre-shuffle option derivation for every generated VQA. This
            is read-only debugging metadata for the rollout probe; it has
            no effect on the returned :class:`ExecutorResult`. ``None``
            (default) keeps the hot path allocation-free.

    Returns:
        A fully-populated :class:`ExecutorResult`. The ``status`` field
        captures the docs/architecture.md failure taxonomy. Unexpected
        exceptions (assertion errors, etc.) propagate — those are bugs.
    """

    t0 = time.perf_counter()
    scene_id = ""
    repair_trace = RepairTrace()
    repair_policy = RepairPolicy(enabled=(stage == 2))

    # ─── 1. Sandbox ──────────────────────────────────────────────────
    try:
        (
            build_scene,
            _legacy_get_vqa,
            vqa_spec,
            scene_manifest,
        ) = execute_proposer_code_with_spec(
            proposer_code, timeout_s=sandbox_timeout_s
        )
    except SyntaxError as e:
        return make_failure(
            FailureReason.FAILED_SYNTAX, scene_id, e, time.perf_counter() - t0
        )
    except RestrictedExecError as e:
        return make_failure(
            FailureReason.FAILED_SEMANTIC, scene_id, e, time.perf_counter() - t0
        )
    except TimeoutError as e:
        return make_failure(
            FailureReason.FAILED_TIMEOUT, scene_id, e, time.perf_counter() - t0
        )
    except NameError as e:
        return make_failure(
            FailureReason.FAILED_RUNTIME,
            scene_id,
            e,
            time.perf_counter() - t0,
        )

    # Resolve bounded repair from the authored manifest before build_scene.
    # The default policy is unchanged for throwing/motion; only
    # collision_arena receives the conservative overlap override.
    repair_scene_kind = (
        scene_manifest.get("scene_kind")
        if isinstance(scene_manifest, dict)
        else None
    )
    repair_policy = repair_policy.for_scene_kind(
        repair_scene_kind if isinstance(repair_scene_kind, str) else None
    )

    # ─── 2. build_scene → freeze ─────────────────────────────────────
    # Stage-2 uses continuous mass/friction/restitution (values such as
    # mass=2.0 survive instead of snapping to the discrete Stage-1 ladder).
    api = PhysDSLApi(continuous_levels=(stage == 2))
    build_api = wrap_physdsl_api(api, trace=repair_trace, policy=repair_policy)
    try:
        build_scene(build_api)
        scene_def = build_api.freeze()
        scene_def = repair_scene_def(
            scene_def, trace=repair_trace, policy=repair_policy
        )
    except DSLValidationError as e:
        return make_failure(
            FailureReason.FAILED_SEMANTIC,
            scene_id,
            e,
            time.perf_counter() - t0,
            repair_trace=_repair_trace_payload(repair_trace),
        )
    except Exception as e:  # noqa: BLE001 — any non-DSL exception is RUNTIME.
        return make_failure(
            FailureReason.FAILED_RUNTIME,
            scene_id,
            e,
            time.perf_counter() - t0,
            repair_trace=_repair_trace_payload(repair_trace),
        )

    scene_id = compute_scene_id(scene_def, seed)

    # ─── 3. Simulate + render ────────────────────────────────────────
    try:
        sim_state = _simulate_guarded(scene_def, seed, sim_timeout_s)
    except InitialOverlapError as e:
        return make_failure(
            FailureReason.FAILED_OVERLAP,
            scene_id,
            e,
            time.perf_counter() - t0,
            repair_trace=_repair_trace_payload(repair_trace),
        )
    except SimulationDivergedError as e:
        return make_failure(
            FailureReason.FAILED_NAN,
            scene_id,
            e,
            time.perf_counter() - t0,
            repair_trace=_repair_trace_payload(repair_trace),
        )
    except SimulationTimeoutError as e:
        return make_failure(
            FailureReason.FAILED_TIMEOUT,
            scene_id,
            e,
            time.perf_counter() - t0,
            repair_trace=_repair_trace_payload(repair_trace),
        )
    except _RenderError as e:
        return make_failure(
            FailureReason.FAILED_RENDER,
            scene_id,
            e,
            time.perf_counter() - t0,
            repair_trace=_repair_trace_payload(repair_trace),
        )
    except RuntimeError as e:
        # Simulator-side model/name mismatches can surface as RuntimeError
        # after MJCF compilation; treat them as scene compile failures so a
        # single bad Proposer scene does not terminate training.
        return make_failure(
            FailureReason.FAILED_COMPILE,
            scene_id,
            e,
            time.perf_counter() - t0,
            repair_trace=_repair_trace_payload(repair_trace),
        )
    except (mujoco.FatalError, mujoco.UnexpectedError, ValueError) as e:
        # MJCF compile errors surface as mujoco.FatalError or — for
        # some invalid-string paths — as ValueError from
        # ``MjModel.from_xml_string``.
        return make_failure(
            FailureReason.FAILED_COMPILE,
            scene_id,
            e,
            time.perf_counter() - t0,
            repair_trace=_repair_trace_payload(repair_trace),
        )

    # ─── 4. Triviality gate ──────────────────────────────────────────
    from physevolve.engine.simulator import is_trivial as _is_trivial

    # Static scene_kinds (e.g. camera depth) are intentionally motionless -- a
    # spatial relationship is the whole question -- so the no-motion triviality
    # gate must not fire for them.
    _manifest_kind = (
        scene_manifest.get("scene_kind")
        if isinstance(scene_manifest, dict)
        else None
    )
    if _manifest_kind not in _STATIC_SCENE_KINDS and _is_trivial(sim_state):
        return make_trivial(
            scene_id,
            sim_state,
            time.perf_counter() - t0,
            repair_trace=_repair_trace_payload(repair_trace),
        )

    # ─── 5. Stage-2 VQA branch ───────────────────────────────────────
    vqa_payload: Optional[tuple[str, list[str], int]] = None
    resolved_vqa_spec: Optional[dict] = None
    vqa_payloads: Optional[list[dict]] = None
    if stage == 2:
        # Unified scene-manifest path: the Proposer always authors a
        # SCENE_MANIFEST; the executor simulates once (+ branch sims) and
        # auto-derives one or more verifier-grounded VQAs. The single-question
        # Stage 2 requires SCENE_MANIFEST instead of VQA_SPEC.
        if scene_manifest is None:
            hint = (
                " (a deprecated `VQA_SPEC` was found instead; author a "
                "`SCENE_MANIFEST` naming the scene_kind)"
                if vqa_spec is not None
                else ""
            )
            return make_failure(
                FailureReason.FAILED_VQA,
                scene_id,
                ValueError(
                    "stage=2 requires a top-level `SCENE_MANIFEST = {...}` in "
                    f"Proposer code{hint}"
                ),
                time.perf_counter() - t0,
                repair_trace=_repair_trace_payload(repair_trace),
            )
        try:
            vqa_payloads, resolved_vqa_spec = _run_scene_manifest(
                scene_manifest,
                scene_def,
                sim_state,
                scene_id=scene_id,
                sandbox_timeout_s=sandbox_timeout_s,
                sim_timeout_s=sim_timeout_s,
                seed=seed,
                registry=scene_template_registry
                or get_scene_template_registry(),
                max_branches=max_branches,
                color_variants_per_scene=color_variants_per_scene,
                camera_depth_bypass_enabled=camera_depth_bypass_enabled,
                skip_branch_sims=skip_branch_sims,
                trace=trace,
                repair_trace=repair_trace,
            )
            # Mirror the first question onto vqa_payload for back-compat with
            # callers that only read one question.
            vqa_payload = tuple(vqa_payloads[0]["payload"])  # type: ignore[assignment]
        except TimeoutError as e:
            return make_failure(
                FailureReason.FAILED_TIMEOUT,
                scene_id,
                e,
                time.perf_counter() - t0,
                repair_trace=_repair_trace_payload(repair_trace),
            )
        except Exception as e:  # noqa: BLE001 — manifest contract violation.
            return make_failure(
                FailureReason.FAILED_VQA,
                scene_id,
                e,
                time.perf_counter() - t0,
                repair_trace=_repair_trace_payload(repair_trace),
            )

    return make_ok(
        scene_id,
        sim_state,
        vqa_payload,
        time.perf_counter() - t0,
        vqa_spec=resolved_vqa_spec,
        vqa_payloads=vqa_payloads,
        repair_trace=_repair_trace_payload(repair_trace),
    )


def _collect_branch_views(
    template: SceneTemplate,
    scene_manifest: dict,
    scene_def: SceneDef,
    *,
    seed: int,
    sim_timeout_s: float,
    max_branches: int,
    skip_branch_sims: bool = False,
) -> dict:
    """Run required branch sims, or return {} when the ablation skips them."""

    if skip_branch_sims:
        return {}
    branches = template.required_branches(scene_manifest)
    if not branches:
        return {}
    from physevolve.executor.branch_sim import run_branch_sims

    return run_branch_sims(
        scene_def,
        branches,
        seed=seed,
        sim_timeout_s=sim_timeout_s,
        max_branches=max_branches,
    )


def _run_scene_manifest(
    scene_manifest: dict,
    scene_def: SceneDef,
    sim_state: SimState,
    *,
    scene_id: str,
    sandbox_timeout_s: float,
    sim_timeout_s: float,
    seed: int,
    registry: SceneTemplateRegistry,
    max_branches: int,
    color_variants_per_scene: int = 0,
    camera_depth_bypass_enabled: bool = False,
    skip_branch_sims: bool = False,
    trace: Optional[dict] = None,
    repair_trace: Optional[RepairTrace] = None,
) -> tuple[list[dict], dict]:
    """Run the scene-template multi-GT path; return (payloads, resolved_spec).

    Raises on any contract violation (mapped to FAILED_VQA by the caller).
    When ``trace`` is a dict it is filled in-place with diagnostic
    ground-truth (manifest, baseline/branch settle outcomes, pre-shuffle
    option derivation) for the rollout probe; it never affects the result.
    """

    if "__invalid_scene_manifest_type__" in scene_manifest:
        raise ValueError(
            "SCENE_MANIFEST must be a dict; got "
            f"{scene_manifest['__invalid_scene_manifest_type__']}"
        )
    scene_kind = scene_manifest.get("scene_kind")
    if not isinstance(scene_kind, str) or not scene_kind:
        raise ValueError("SCENE_MANIFEST['scene_kind'] must be a non-empty string")
    template = registry.lookup(scene_kind)

    with _timeout_guard(sandbox_timeout_s):
        if template.compile_manifest is not None:
            scene_manifest = dict(
                template.compile_manifest(
                    scene_manifest,
                    scene_def,
                    repair_trace,
                )
            )
        reason = template.validate_manifest(scene_manifest, scene_def)
        if reason is not None:
            raise ValueError(f"manifest invalid: {reason}")

    baseline_view = sim_state_public_view(sim_state)

    branch_views = _collect_branch_views(
        template,
        scene_manifest,
        scene_def,
        seed=seed,
        sim_timeout_s=sim_timeout_s,
        max_branches=max_branches,
        skip_branch_sims=skip_branch_sims,
    )

    from physevolve.skills.scene_templates.colors import body_color_map

    body_colors = body_color_map(scene_def)
    with _timeout_guard(sandbox_timeout_s):
        generated = list(
            template.generate_questions(
                scene_manifest, baseline_view, branch_views, body_colors, scene_def
            )
        )
        generated = _filter_generated_vqas(generated)

    camera_depth_bypass_payload = None
    if camera_depth_bypass_enabled:
        camera_depth_bypass_payload = _camera_depth_bypass_payload(
            source_scene_kind=scene_kind,
            source_manifest=scene_manifest,
            scene_def=scene_def,
            baseline_view=baseline_view,
            body_colors=body_colors,
            registry=registry,
            scene_id=scene_id,
        )

    if trace is not None:
        _fill_manifest_trace(
            trace,
            scene_kind=scene_kind,
            manifest=scene_manifest,
            baseline_view=baseline_view,
            branches=([] if skip_branch_sims else template.required_branches(scene_manifest)),
            branch_views=branch_views,
            body_colors=body_colors,
            generated=generated,
        )
        trace["camera_depth_bypass"] = (
            None
            if camera_depth_bypass_payload is None
            else {
                "attached": True,
                "spec": dict(camera_depth_bypass_payload.get("spec", {})),
            }
        )

    payloads: list[dict] = [
        item
        for gen in generated
        if (item := _payload_from_generated(gen, scene_kind=scene_kind, salt=scene_id))
        is not None
    ]
    if camera_depth_bypass_payload is not None:
        payloads.append(camera_depth_bypass_payload)

    # Colour-permutation variants (anti text-space collapse): re-render the
    # scene with the template's recolourable bodies recoloured and append the
    # (identity-stable) questions so the Solver cannot shortcut on a fixed
    # colour->answer mapping. Physics is unchanged by colour, so GT is the same
    # by body identity; only the colour word in the question (derived from the
    # new rgba) changes. Each variant carries its OWN sim_state (its distinct
    # video). The recolour role is template-declared (throwing -> balls); scene
    # kinds whose option text references bodies by their baked name return [] so
    # a recolour cannot desync the wording.
    if color_variants_per_scene > 0 and template.color_variant_bodies is not None:
        recolor_bodies = [
            str(b)
            for b in template.color_variant_bodies(scene_manifest)
            if isinstance(b, str)
        ]
        if recolor_bodies:
            payloads.extend(
                _color_variant_payloads(
                    template=template,
                    scene_manifest=scene_manifest,
                    scene_def=scene_def,
                    recolor_bodies=recolor_bodies,
                    branch_views=branch_views,
                    scene_kind=scene_kind,
                    scene_id=scene_id,
                    seed=seed,
                    sim_timeout_s=sim_timeout_s,
                    sandbox_timeout_s=sandbox_timeout_s,
                    n_variants=int(color_variants_per_scene),
                )
            )

    if not payloads:
        raise ValueError(
            f"scene_kind {scene_kind!r} generated no verifiable questions"
        )
    resolved_spec = {
        "scene_kind": scene_kind,
        "n_questions": len(payloads),
        "families": sorted({p["spec"]["family"] for p in payloads}),
    }
    return payloads, resolved_spec


def _camera_depth_bypass_payload(
    *,
    source_scene_kind: str,
    source_manifest: dict,
    scene_def: SceneDef,
    baseline_view: Any,
    body_colors: dict[str, str],
    registry: SceneTemplateRegistry,
    scene_id: str,
) -> Optional[dict]:
    """Build one optional camera-depth payload for a dynamics scene.

    Candidate bodies referenced by the source manifest are tried first, then
    other non-plane bodies. Invalid/ambiguous four-body combinations are skipped;
    the source dynamics scene remains valid even when no depth question fits.
    """

    if source_scene_kind not in _CAMERA_DEPTH_BYPASS_SOURCE_KINDS:
        return None
    if not scene_def.cameras:
        return None

    eligible_names = {
        body.name
        for body in scene_def.bodies
        if body.geom_type.value != "plane"
    }
    if len(eligible_names) < 4:
        return None

    preferred: list[str] = []

    def _collect_manifest_body_names(value: Any) -> None:
        if isinstance(value, str):
            if value in eligible_names and value not in preferred:
                preferred.append(value)
            return
        if isinstance(value, dict):
            for child in value.values():
                _collect_manifest_body_names(child)
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                _collect_manifest_body_names(child)

    _collect_manifest_body_names(source_manifest)
    ordered_names = preferred + sorted(eligible_names - set(preferred))
    ordered_names = ordered_names[:_MAX_CAMERA_DEPTH_BYPASS_CANDIDATES]
    if len(ordered_names) < 4:
        return None

    try:
        depth_template = registry.lookup(_CAMERA_DEPTH_SCENE_KIND)
    except ValueError:
        return None
    camera_name = scene_def.cameras[0].name
    for candidate_names in combinations(ordered_names, 4):
        depth_manifest = {
            "scene_kind": _CAMERA_DEPTH_SCENE_KIND,
            "families": [_CAMERA_DEPTH_SCENE_KIND],
            "phrasing": "static",
            "candidate_bodies": list(candidate_names),
            "camera": camera_name,
        }
        try:
            reason = depth_template.validate_manifest(depth_manifest, scene_def)
            if reason is not None:
                continue
            generated = _filter_generated_vqas(
                list(
                    depth_template.generate_questions(
                        depth_manifest,
                        baseline_view,
                        {},
                        body_colors,
                        scene_def,
                    )
                )
            )
        except (TypeError, ValueError):
            continue
        if len(generated) != 1:
            continue
        item = _payload_from_generated(
            generated[0],
            scene_kind=_CAMERA_DEPTH_SCENE_KIND,
            salt=f"{scene_id}:camera-depth-bypass",
        )
        if item is None:
            continue
        item["spec"].update(
            {
                "scene_kind": source_scene_kind,
                "attached_scene_kind": _CAMERA_DEPTH_SCENE_KIND,
                "bypass": True,
            }
        )
        return item
    return None


def _filter_generated_vqas(generated: list[GeneratedVQA]) -> list[GeneratedVQA]:
    """Drop degenerate generated questions before they reach the Solver."""

    out: list[GeneratedVQA] = []
    seen_questions: set[tuple[str, tuple[str, ...]]] = set()
    for gen in generated:
        if len(set(gen.options)) != len(gen.options):
            continue
        if not (0 <= int(gen.correct_idx) < len(gen.options)):
            continue
        key = (gen.question.strip(), tuple(gen.options))
        if key in seen_questions:
            continue
        seen_questions.add(key)
        out.append(gen)
    return out


# Cap colour-variant questions per variant so the chunked Solver rollout stays
# bounded even at high executor-OK rates (each question carries a full video).
_MAX_VARIANT_QUESTIONS: int = 4


def _payload_from_generated(
    gen: GeneratedVQA,
    *,
    scene_kind: str,
    salt: str,
    sim_state: Optional[SimState] = None,
) -> Optional[dict]:
    """Apply catalog layout + validate one ``GeneratedVQA`` into a payload item.

    Returns ``None`` when the sampled option layout cannot bind the verifier
    semantic answer (question is dropped rather than emitting false GT).

    ``sim_state`` is attached only for colour-permuted variant questions whose
    video differs from the scene baseline; baseline questions leave it out so
    downstream code reuses the result's single ``sim_state``.
    """

    if not isinstance(gen, GeneratedVQA):
        raise ValueError("generate_questions must return GeneratedVQA items")
    layout_result = apply_option_layout(
        gen,
        scene_kind=scene_kind,
        salt=salt,
        shuffle_fallback=False,
    )
    if layout_result is None:
        return None
    options, correct_idx = layout_result
    payload = _validate_vqa_payload((gen.question, options, correct_idx))
    item: dict = {
        "payload": list(payload),
        "spec": {
            "scene_kind": scene_kind,
            "family": gen.family,
            "phrasing": gen.phrasing,
            **dict(gen.spec),
        },
    }
    if sim_state is not None:
        item["sim_state"] = sim_state
    return item


def _color_variant_payloads(
    *,
    template: SceneTemplate,
    scene_manifest: dict,
    scene_def: SceneDef,
    recolor_bodies: list[str],
    branch_views: dict,
    scene_kind: str,
    scene_id: str,
    seed: int,
    sim_timeout_s: float,
    sandbox_timeout_s: float,
    n_variants: int,
) -> list[dict]:
    """Render ``n_variants`` colour-permuted variants and return their payloads.

    Each variant recolours the template's ``recolor_bodies`` to a deterministic
    distinct-colour assignment (seeded by ``scene_id`` + variant index),
    re-simulates+renders (colours never affect physics, so trajectories / GT are
    identity-stable), and re-derives questions whose colour words follow the new
    rgba. Other bodies (e.g. walls) are left unrecoloured so branch-based
    wording and the cached ``branch_views`` stay consistent. A failed variant
    sim is skipped.
    """

    import random

    from physevolve.skills.scene_templates.colors import (
        PALETTE_COLOR_NAMES,
        body_color_map,
        color_permuted_scene,
    )

    palette = sorted(PALETTE_COLOR_NAMES)
    out: list[dict] = []
    for v in range(n_variants):
        rng = random.Random(f"{scene_id}:colorvar:{v}")
        if len(recolor_bodies) <= len(palette):
            colors = rng.sample(palette, len(recolor_bodies))
        else:  # more bodies than palette colours: allow repeats.
            colors = [rng.choice(palette) for _ in recolor_bodies]
        variant_scene, _assignment = color_permuted_scene(
            scene_def, recolor_bodies, colors
        )
        try:
            variant_sim = _simulate_guarded(variant_scene, seed, sim_timeout_s)
        except Exception:  # noqa: BLE001 — a failed variant just yields no Qs.
            continue
        variant_view = sim_state_public_view(variant_sim)
        variant_colors = body_color_map(variant_scene)
        try:
            with _timeout_guard(sandbox_timeout_s):
                gens = list(
                    template.generate_questions(
                        scene_manifest,
                        variant_view,
                        branch_views,
                        variant_colors,
                        variant_scene,
                    )
                )
        except Exception:  # noqa: BLE001 — skip a variant that fails to render Qs.
            continue
        for gen in gens[:_MAX_VARIANT_QUESTIONS]:
            item = _payload_from_generated(
                gen,
                scene_kind=scene_kind,
                salt=f"{scene_id}:cv{v}",
                sim_state=variant_sim,
            )
            if item is not None:
                out.append(item)
    return out


def _ball_outcome(view: "SimStateView", ball: str) -> Optional[dict]:
    """Compact per-ball outcome (final pos + speed) for the probe trace."""

    if ball not in view:
        return None
    body = view[ball]
    fv = body.final_vel
    final_speed = float((fv[0] ** 2 + fv[1] ** 2 + fv[2] ** 2) ** 0.5)
    return {
        "final_pos": [round(float(x), 4) for x in body.final_pos],
        "final_speed": round(final_speed, 4),
        "final_status": str(body.final_status),
    }


def _fill_manifest_trace(
    trace: dict,
    *,
    scene_kind: str,
    manifest: dict,
    baseline_view: "SimStateView",
    branches: list,
    branch_views: dict,
    body_colors: dict,
    generated: list,
) -> None:
    """Populate ``trace`` with scene-template diagnostics (probe only)."""

    balls = [b for b in (manifest.get("balls") or []) if isinstance(b, str)]
    trace["path"] = "scene_manifest"
    trace["scene_kind"] = scene_kind
    trace["body_colors"] = dict(body_colors)
    trace["manifest"] = manifest
    trace["baseline_outcomes"] = {
        ball: _ball_outcome(baseline_view, ball) for ball in balls
    }
    branch_records: list[dict] = []
    for spec in branches:
        view = branch_views.get(spec.key)
        branch_records.append(
            {
                "key": spec.key,
                "removed": list(spec.remove),
                "ran": view is not None,
                "outcomes": (
                    {ball: _ball_outcome(view, ball) for ball in balls}
                    if view is not None
                    else None
                ),
            }
        )
    trace["branches"] = branch_records
    trace["generated"] = [
        {
            "family": gen.family,
            "phrasing": gen.phrasing,
            "question": gen.question,
            "options": list(gen.options),
            "correct_idx": int(gen.correct_idx),
            "correct_option": list(gen.options)[int(gen.correct_idx)],
            "spec": dict(gen.spec),
        }
        for gen in generated
        if isinstance(gen, GeneratedVQA)
    ]


# Re-export for public-ish consumers that don't want to reach into
# ``state_view`` directly.
__all__ = [
    "run_executor",
    "sim_state_public_view",
]


# Convenience: exposing the raw build_scene callable type for users of
# the pipeline who might want to call it themselves (e.g. for the
# inverse-replay scorer). Not documented as stable.
BuildSceneFn = Callable[[PhysDSLApi], None]
