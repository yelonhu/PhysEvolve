"""Scene templates: one reusable scene -> many auto-derived VQAs.

Dynamics items are predominantly *one scene -> many questions*: the
same multi-ball + floating-wall + pit video is re-asked by changing the queried
colour, the target pit, or a hypothetical intervention. The single typed
``VQA_SPEC`` path (one question per scene) cannot express that.

A :class:`SceneTemplate` lets the Proposer author one high-quality, reusable
scene via a ``SCENE_MANIFEST`` literal (named coloured bodies + named regions +
removable obstacles + which question families to ask). The executor simulates
the scene ONCE (plus a capped set of branch simulations for counterfactual /
inverse families), then the template's :meth:`SceneTemplate.generate_questions`
auto-enumerates a list of verifier-grounded :class:`GeneratedVQA` from the
single (and branch) simulation results.

This sits ALONGSIDE the typed skill cards (:mod:`physevolve.skills`), which
still serve the single-question families (collision / motion / mass / friction).
Ground truth always comes from the simulated body trajectories; colours and
region names are only labels.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from physevolve.engine.schema import SceneDef
    from physevolve.executor.repair import RepairTrace
    from physevolve.executor.state_view import SimStateView


@dataclass(frozen=True, slots=True)
class GeneratedVQA:
    """One auto-derived VQA produced from a scene template.

    Attributes:
        question: Natural-language question text authored by the executor.
        options: Exactly 4 natural-language option strings.
        correct_idx: GT index into ``options`` (0..3), from the simulation.
        family: The question-family key this VQA belongs to (for metrics).
        phrasing: The phrasing variant used.
        spec: Audit dict ``{family, phrasing, target, ...}`` for logging /
            novelty signatures. Never contains the answer beyond ``correct_idx``
            being implied by the simulation.
    """

    question: str
    options: list[str]
    correct_idx: int
    family: str
    phrasing: str
    spec: dict

    def as_payload(self) -> tuple[str, list[str], int]:
        """Return the ``(question, options, correct_idx)`` payload tuple."""

        return (self.question, list(self.options), int(self.correct_idx))


@dataclass(frozen=True, slots=True)
class BranchSpec:
    """A requested branch simulation: remove these bodies, then re-simulate.

    Attributes:
        key: Stable identifier the template uses to look the result up in the
            ``branch_views`` mapping passed to ``generate_questions``.
        remove: Body names to delete from the scene before re-simulating.
    """

    key: str
    remove: tuple[str, ...]


# A template's question generator. Args:
#   manifest      - the validated SCENE_MANIFEST dict
#   baseline_view - SimStateView of the unmodified run
#   branch_views  - {branch_key: SimStateView} (empty if disabled/failed)
#   body_colors   - {body_name: visible colour name} from the scene's rgba, so
#                   questions refer to bodies by what the Solver actually sees.
#   scene_def     - the frozen SceneDef, so a template can DERIVE regions /
#                   target footprints from the actual body geometry (binding
#                   the question wording to a visible structure).
GenerateFn = Callable[
    [
        Mapping[str, object],
        "SimStateView",
        Mapping[str, "SimStateView"],
        Mapping[str, str],
        "SceneDef",
    ],
    Sequence[GeneratedVQA],
]

# Optional deterministic manifest compiler. It runs after ``build_scene`` is
# frozen and before template validation, so it can align role names with actual
# body names and derive manifest fields from authored geometry.
CompileManifestFn = Callable[
    [Mapping[str, object], "SceneDef", Optional["RepairTrace"]],
    Mapping[str, object],
]


@dataclass(frozen=True, slots=True)
class SceneTemplate:
    """A reusable scene kind that emits many VQAs from one simulation.

    Attributes:
        scene_kind: Unique key (matches ``SCENE_MANIFEST['scene_kind']``).
        families: Question families this template can emit.
        validate_manifest: ``(manifest, scene_def) -> Optional[str]`` returning
            a failure reason (mapped to ``FAILED_VQA``) or ``None``.
        required_branches: ``(manifest) -> list[BranchSpec]`` listing the branch
            sims the executor should run (deduped + capped by the executor).
        generate_questions: see :data:`GenerateFn`.
        compile_manifest: Optional semantic compiler invoked before validation.
            It may derive manifest-only values from authored geometry or repair
            obvious role-name mismatches, and should record any action in the
            provided repair trace.
        manifest_schema_doc: Human-readable schema rendered into the prompt.
        scene_constraints: Per-template scene authoring rules for the prompt.
        authoring_playbook: Legacy scene-specific construction rules kept for
            compatibility with older prompt renderers.
        authoring_skeleton: Legacy topology checklist kept for compatibility.
        golden_design_card: Legacy distilled design notes kept for compatibility.
        main_training_wiring_card: Main-training fewshot distillation. This is
            the single prompt-facing substitute for the executable golden: it
            teaches role wiring, API order, manifest shape, invariants, and
            variation axes without full golden coordinates or body names.
        fewshot_example: Backward-compatible primary runnable golden
            ``build_scene`` + ``SCENE_MANIFEST``.
        fewshot_examples: Optional ordered golden variants for cold-start prompt
            rotation. When empty, callers fall back to ``fewshot_example``.
        color_variant_bodies: ``(manifest) -> list[str]`` naming the queryable
            bodies whose colour the executor may permute to mint extra "multiple questions per scene"
            variants (the colour word in the question must be derived from the
            body's rgba, not its name, for this to stay correct). Templates whose
            option text references bodies by their baked name return ``[]`` so a
            recolour does not desync the wording. ``None`` disables variants.
    """

    scene_kind: str
    families: tuple[str, ...]
    validate_manifest: Callable[[Mapping[str, object], "SceneDef"], Optional[str]]
    required_branches: Callable[[Mapping[str, object]], list[BranchSpec]]
    generate_questions: GenerateFn
    compile_manifest: Optional[CompileManifestFn] = None
    manifest_schema_doc: str = ""
    scene_constraints: str = ""
    authoring_playbook: str = ""
    authoring_skeleton: str = ""
    golden_design_card: str = ""
    main_training_wiring_card: str = ""
    fewshot_example: str = ""
    fewshot_examples: tuple[str, ...] = ()
    color_variant_bodies: Optional[
        Callable[[Mapping[str, object]], list[str]]
    ] = None


@dataclass(frozen=True, slots=True)
class SceneTemplateRegistry:
    """Closed catalog of :class:`SceneTemplate` instances."""

    templates: Mapping[str, SceneTemplate]

    def lookup(self, scene_kind: str) -> SceneTemplate:
        try:
            return self.templates[scene_kind]
        except KeyError:
            available = sorted(self.templates)
            raise ValueError(
                f"unknown scene_kind {scene_kind!r}; available: {available}"
            ) from None

    def available_scene_kinds(self) -> tuple[str, ...]:
        return tuple(sorted(self.templates))


def build_scene_template_registry(
    templates: Sequence[SceneTemplate],
) -> SceneTemplateRegistry:
    table: dict[str, SceneTemplate] = {}
    for tpl in templates:
        if tpl.scene_kind in table:
            raise ValueError(f"duplicate scene_kind {tpl.scene_kind!r}")
        table[tpl.scene_kind] = tpl
    return SceneTemplateRegistry(templates=table)


def fewshot_examples_for(template: SceneTemplate) -> tuple[str, ...]:
    """Return all prompt-ready executable golden examples for ``template``."""

    examples = tuple(example.strip() for example in template.fewshot_examples if example.strip())
    if examples:
        return examples
    primary = template.fewshot_example.strip()
    return (primary,) if primary else ()


def select_fewshot_example(template: SceneTemplate, index: int = 0) -> str:
    """Return one executable golden example by deterministic modulo index."""

    examples = fewshot_examples_for(template)
    if not examples:
        raise ValueError(f"scene_kind {template.scene_kind!r} has no fewshot examples")
    return examples[int(index) % len(examples)]


def load_prompt_fewshot_examples(
    filenames: Sequence[str],
    *,
    drop_manifest_keys: Sequence[str] = (),
) -> tuple[str, ...]:
    """Load prompt-ready golden examples from ``data/golden_scenes/stage2``.

    Golden scene files are runnable Python modules with a module docstring plus
    ``build_scene`` / ``SCENE_MANIFEST``. Cold-start prompts need the latter
    wrapped in ``<propose>`` and, for fields removed from the current Proposer
    contract, a small prompt-facing manifest normalization.
    """

    root = Path(__file__).resolve().parents[3]
    base = root / "data" / "golden_scenes" / "stage2"
    examples: list[str] = []
    for filename in filenames:
        source = (base / filename).read_text(encoding="utf-8")
        plan, code = _split_golden_source(source)
        for key in drop_manifest_keys:
            code = _drop_manifest_key_block(code, key)
        examples.append(f"{plan}\n<propose>\n{code.rstrip()}\n</propose>")
    return tuple(examples)


def _split_golden_source(source: str) -> tuple[str, str]:
    """Return ``(plan_line, code_without_module_docstring)``."""

    module = ast.parse(source)
    doc = ast.get_docstring(module) or ""
    lines = source.splitlines()
    start_line = 0
    if module.body and isinstance(module.body[0], ast.Expr):
        value = module.body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            start_line = int(getattr(module.body[0], "end_lineno", 0))
    plan = _plan_from_docstring(doc)
    code = "\n".join(lines[start_line:]).strip()
    return plan, code


def _plan_from_docstring(doc: str) -> str:
    """Extract a concise Plan line from a golden scene docstring."""

    raw_lines = [line.strip() for line in doc.splitlines()]
    for idx, line in enumerate(raw_lines):
        if line.startswith("Plan:"):
            parts = [line]
            for extra in raw_lines[idx + 1 :]:
                if not extra:
                    break
                if ":" in extra and not extra.startswith(("and ", "that ", "the ")):
                    break
                parts.append(extra)
            return " ".join(parts)
    for line in raw_lines:
        if line:
            return f"Plan: {line}"
    return "Plan: validated executable golden scene."


def _drop_manifest_key_block(source: str, key: str) -> str:
    """Remove one top-level manifest key block from prompt-facing source text."""

    out: list[str] = []
    skipping = False
    brace_depth = 0
    needle = f'"{key}"'
    for line in source.splitlines():
        stripped = line.strip()
        if not skipping and stripped.startswith(f"{needle}:"):
            skipping = True
            brace_depth = line.count("{") - line.count("}")
            if brace_depth <= 0:
                skipping = False
            continue
        if skipping:
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                skipping = False
            continue
        out.append(line)
    return "\n".join(out)


_REGISTRY_CACHE: Optional[SceneTemplateRegistry] = None


def get_scene_template_registry() -> SceneTemplateRegistry:
    """Return the canonical scene-template registry (built lazily)."""

    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        from physevolve.skills.scene_templates import (
            camera_depth,
            collision_arena,
            motion_lab,
            motion_phenomenon,
            throwing_pit_board,
        )

        _REGISTRY_CACHE = build_scene_template_registry(
            [
                # Active grouped scene kinds (each a throwing-level multi-question
                # template; a manifest declares a SUBSET of the family menu).
                throwing_pit_board.make_template(),
                collision_arena.make_template(),
                motion_lab.make_template(),
                # Frozen scene kinds remain registered but are not
                # bound to an active question family (see question_families.py).
                motion_phenomenon.make_template(),
                camera_depth.make_template(),
            ]
        )
    return _REGISTRY_CACHE


__all__ = [
    "BranchSpec",
    "CompileManifestFn",
    "GeneratedVQA",
    "GenerateFn",
    "SceneTemplate",
    "SceneTemplateRegistry",
    "build_scene_template_registry",
    "fewshot_examples_for",
    "get_scene_template_registry",
    "load_prompt_fewshot_examples",
    "select_fewshot_example",
]
