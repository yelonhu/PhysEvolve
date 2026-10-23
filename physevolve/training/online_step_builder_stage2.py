"""Stage-2 VQA online self-play step builder (docs/architecture.md).

This module mirrors :class:`PhysEvolveOnlineStepBuilder` (Stage 1) but emits
the Stage-2 typed VQA loop:

1. Proposer prompt is rendered per-item via
   :func:`physevolve.prompts.proposer_stage2.build_proposer_scene_template_prompt`
   for the scene_kind of the question family picked by adaptive
   sampling
   (:func:`physevolve.training.adaptive_curriculum_stage2.pick_stage2_question_family`).
   Every active family is served by a scene
   template (1-family for collision/motion/property; multi-question for
   throwing), so the Proposer always authors ``build_scene`` + ``SCENE_MANIFEST``.
2. Executor runs with ``stage=2`` so the scene template's verifier
   (reusing the typed skill cards' ``feasibility_check`` +
   ``ground_truth_extractor``) produces the authoritative ``vqa_payloads``.
3. Solver rollout uses ``TaskMode.VQA`` (single request per OK scene,
   not three dense requests).
4. Solver reward is the VQA exact-match scalar
   (:func:`physevolve.verification.vqa_scorer.score_vqa`).
5. Proposer reward reuses the Stage-1 learnability + intrinsic shaping
   term; the novelty signature is extended with
   ``(skill_target, phrasing, params_signature)``.
6. Problem-type tuple is
   :data:`physevolve.training.verl_config.PHYSEVOLVE_STAGE2_PROBLEM_TYPES`:
   ``("gen_vqa", "pred_vqa")``.

Stage-2 is trained **from scratch** on the base Instruct VL model; it does
NOT consume a Stage-1 checkpoint. See ``docs/architecture.md`` and the Stage-2
training entry (``scripts/train/_entry.py``).
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict, deque
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from physevolve.engine.failure import PROPOSER_PENALTY, FailureReason
from physevolve.executor import run_executor
from physevolve.inference.hf_backend import DEFAULT_MODEL, GenerationResult
from physevolve.inference.parsing import extract_propose_block
from physevolve.prompts.proposer_stage2 import (
    build_proposer_scene_template_prompt,
)
from physevolve.skills.question_families import (
    BRANCH_ELISION_FAMILIES,
    active_families,
    lookup_family,
    validate_registry,
)
from physevolve.skills.scene_templates.render_utils import (
    canonical_params_signature,
)
from physevolve.smoke.io import append_jsonl, write_text
from physevolve.smoke.run_config import SmokeGenerationConfig
from physevolve.tasks.router import TaskMode
from physevolve.training.adaptive_curriculum import (
    DEFAULT_ACC_WINDOW,
    DEFAULT_COLD_START_STEPS,
    ReferenceBuffer,
    RollingAccTracker,
    format_reference_examples,
)
from physevolve.training.adaptive_curriculum_stage2 import (
    DEFAULT_FAMILY_PRIORITY_BOOST,
    DEFAULT_FAMILY_WEIGHT_MULTIPLIERS,
    pick_stage2_question_families_for_step,
    stage2_family_pick_weight_snapshot,
)
from physevolve.training.online_step import OnlineSelfPlayStep
from physevolve.training.online_step_builder import (
    BackendFactory,
    ExecutorFn,
    GenerationBackend,
    GenerationSpec,
    _concat_rollouts,
    _default_backend_factory,
    _generated_token_count,
    _learnability_min_rollouts as _stage1_learnability_min_rollouts,
    _maybe_collect_rollout,
    _proposer_failure_record as _stage1_proposer_failure_record,
    _rate,
    _valid_generation_length,
    call_generate_batch,
    call_generate_batch_chunked,
)
from physevolve.training.reward_batch import RewardRecord, build_reward_batch
from physevolve.training.selfplay_batch import (
    build_solver_requests,
    executor_result_record,
    save_video_mp4,
    score_solver_response,
    video_request_record,
)
from physevolve.training.stage2_copy_diagnostics import (
    Stage2CopyDiagnostic,
    Stage2CopyDiagnostics,
    diagnostic_similarity,
    summarize_copy_diagnostics,
)
from physevolve.training.verl_config import (
    PHYSEVOLVE_STAGE2_PROBLEM_TYPES,
)
from physevolve.verification.proposer_reward import (
    DEFAULT_INTRINSIC_COMPONENT_WEIGHTS,
    DEFAULT_INTRINSIC_WEIGHT,
    CopyControl,
    IntrinsicInputs,
    SceneSignature,
    compute_proposer_reward,
    compute_proposer_reward_with_intrinsic,
    compute_scene_signature,
    family_difficulty_weight,
)


# ─── Configuration ────────────────────────────────────────────────────


_REFERENCE_GEOMETRY_REPAIR_ACTION_CAP = 3


@dataclass(frozen=True, slots=True)
class PhysEvolveStage2StepBuilderConfig:
    """Configuration for the Stage-2 typed VQA online self-play step builder.

    Stage-2 picks a question family per item via the adaptive sampler; each family resolves to a scene template (the verifier).
    ``question_families`` lets callers restrict the vocabulary (e.g. for
    ablations); ``None`` means "use every active P0 family".
    """

    model: str = DEFAULT_MODEL
    scenes_per_batch: int = 2
    seed_start: int = 9001
    group_size: int = 1
    sandbox_timeout_s: float = 5.0
    sim_timeout_s: float = 30.0
    response_length: int = 4096
    executor_workers: int = 1
    # Phase 4 dispatches the VQA solver rollout in fixed-size chunks so the
    # per-row video preprocessing peak stays bounded (multi-question scenes
    # can otherwise produce dozens of video-bearing solver rows per step and
    # OOM a single generate_sequences call). 0 disables chunking (single
    # batched dispatch).
    solver_rollout_chunk_size: int = 8
    # Colour permutation: the executor renders
    # this many additional colour-permuted variants per OK scene-template scene,
    # each producing its own video + recomputed (identity-stable) questions, so
    # the Solver cannot shortcut on a fixed colour->answer mapping. 0 disables
    # the extra variants (the baseline scene is still seed-recoloured).
    color_variants_per_scene: int = 1
    # Stage-2 multi-GT scenes can derive many verifier-grounded questions from
    # one OK proposer rollout. These caps bound the number of UNIQUE VQA question
    # groups sent to the Solver; each selected group is still rolled out
    # ``group_size`` times for GRPO. 0 disables the corresponding cap.
    max_solver_questions_per_scene: int = 4
    max_solver_groups_per_step: int = 32
    camera_depth_bypass_enabled: bool = False
    # Stage-2 ablation: drop counterfactual / inverse families from sampling,
    # skip executor branch sims, and drop those payloads before Solver select.
    disable_branch_elision: bool = False
    generation: SmokeGenerationConfig = SmokeGenerationConfig(
        max_new_tokens=4096,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        stop_sequences=("</propose>", "</answer>"),
    )
    # Keep the default Proposer reward focused on verified learnability.
    intrinsic_enabled: bool = False
    intrinsic_weight: float = DEFAULT_INTRINSIC_WEIGHT
    intrinsic_component_weights: tuple[float, float, float, float, float] = (
        DEFAULT_INTRINSIC_COMPONENT_WEIGHTS
    )
    # Scale on the per-status Proposer failure penalty. Raised above 1.0 so a
    # failed rollout produces a stronger negative reward than the learnability
    # band: this guarantees intra-group reward variance (hence a GRPO gradient)
    # even when the synthetic questions are all answered correctly and the
    # learnability tent would otherwise be flat-zero across the group.
    proposer_failure_weight: float = 1.5
    novelty_buffer_size: int = 64
    # Question-family layer: Stage-2 samples over PhysEvolve question families
    # (each resolves to a verifying scene template). ``question_families=None``
    # uses the active P0 families; the priority boost emphasises the shapes the
    # curriculum pushes hardest.
    question_families: Optional[tuple[str, ...]] = None
    question_family_priority_boost: float = DEFAULT_FAMILY_PRIORITY_BOOST
    # Keep recipe's global 1.5 branch-family default reproducible while allowing a
    # launcher-scoped, paired CF/inverse backoff for balance recipes such as paper.
    branch_family_weight_multiplier: float = float(
        DEFAULT_FAMILY_WEIGHT_MULTIPLIERS["counterfactual_destination"]
    )
    # Per-step scene-kind minimums inside ``scenes_per_batch``. Quota slots use
    # the same family tent restricted to that scene kind; remaining slots keep
    # the unrestricted sampler.
    min_scene_kind_picks_per_step: tuple[tuple[str, int], ...] = ()
    cold_start_uniform_steps: int = DEFAULT_COLD_START_STEPS
    skill_acc_window: int = DEFAULT_ACC_WINDOW
    rng_seed: int = 9001
    # Save up to this many OK-scene videos as MP4 every ``video_save_freq``
    # steps. 0 (default) disables video saving entirely.
    video_save_freq: int = 0
    video_save_max_per_step: int = 4
    # Diagnostic only (rollout probe): when True, every OK Stage-2
    # executor call writes a ``groundtruth/<sample_id>.json`` capturing the
    # raw simulator ground-truth + manifest + branch/option derivation.
    # Off by default so the training hot path is unaffected.
    emit_groundtruth_trace: bool = False
    # ── Anti-copy settings ────────────────────────────────────
    # Executable examples are shown through cold_start_fewshot_steps.
    # Reference summaries may accompany them or continue after that window.
    # Copy penalties and their activation schedule are configured separately.
    cold_start_fewshot_steps: int = 0
    reference_buffer_enabled: bool = False
    reference_k: int = 2
    reference_buffer_size: int = 32
    # When True, inject ReferenceBuffer summaries even while executable fewshot
    # is shown (recipe hybrid). Legacy False keeps fewshot/buffer mutually exclusive.
    reference_buffer_during_fewshot: bool = False
    # AST anti-copy reward. ``copy_penalty_enabled`` is the master switch.
    copy_penalty_enabled: bool = False
    # Reward forced on an exact golden-fewshot copy / recent-window AST repeat
    # / intra-group AST duplicate (overrides learnability + diversity).
    copy_exact_reward: float = -0.5
    # OK reward cap applied when AST similarity >= ``copy_similarity_threshold``.
    copy_cap: float = 0.1
    copy_similarity_threshold: float = 0.95
    # Intra-GRPO-group AST dedup: keep one canonical rollout, flag the rest as
    # duplicates (forced duplicate reward and NOT sent to the Solver).
    batch_dedup_enabled: bool = False
    # learnability saturation: penalise ``mean_acc`` above the threshold so
    # "synthetic questions all answered correctly" stops being an attractor.
    learnability_saturation_threshold: float = 0.85
    learnability_saturation_weight: float = 0.0
    # AST edit-distance diversity bonus: diversity_weight * (1 - similarity),
    # kept at learnability magnitude (default 0 keeps it off).
    diversity_edit_distance_weight: float = 0.0
    # anti-copy / saturation gate independent of fewshot schedule.
    # ``None`` falls back to ``cold_start_fewshot_steps``.
    # Set to 0 so controls activate from step 1 while fewshot stays on for the
    # full run.
    anti_copy_start_step: Optional[int] = None
    # Scale on the learnability tent (recipe default 0.5 via launcher).
    learnability_weight: float = 1.0
    # While executable fewshot is still shown (``step <= cold_start_fewshot_steps``),
    # use this learnability scale instead. ``None`` = same as ``learnability_weight``
    # (legacy). recipe sets ``0.0`` so verbatim copy + high solver acc cannot earn a
    # positive learnability reward that fights ``copy_exact_reward``.
    learnability_weight_during_fewshot: Optional[float] = None
    # Family rolling-acc difficulty multiplier: reward *= clip(1+β*(0.5-acc)).
    family_difficulty_beta: float = 0.0
    # Multiply ``pred_vqa`` reward rows before GRPO (raise Solver gradient share).
    solver_reward_scale: float = 1.0

    def __post_init__(self) -> None:
        quotas = tuple(
            (str(scene_kind), int(count))
            for scene_kind, count in self.min_scene_kind_picks_per_step
        )
        object.__setattr__(self, "min_scene_kind_picks_per_step", quotas)
        if self.scenes_per_batch <= 0:
            raise ValueError("scenes_per_batch must be positive")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.response_length <= 0:
            raise ValueError("response_length must be positive")
        if self.executor_workers <= 0:
            raise ValueError("executor_workers must be positive")
        if self.solver_rollout_chunk_size < 0:
            raise ValueError("solver_rollout_chunk_size must be non-negative")
        if self.color_variants_per_scene < 0:
            raise ValueError("color_variants_per_scene must be non-negative")
        if self.max_solver_questions_per_scene < 0:
            raise ValueError("max_solver_questions_per_scene must be non-negative")
        if self.max_solver_groups_per_step < 0:
            raise ValueError("max_solver_groups_per_step must be non-negative")
        if self.intrinsic_weight < 0:
            raise ValueError("intrinsic_weight must be non-negative")
        if self.proposer_failure_weight < 0:
            raise ValueError("proposer_failure_weight must be non-negative")
        if len(self.intrinsic_component_weights) != 5:
            raise ValueError(
                "intrinsic_component_weights must have 5 entries "
                "(motion, interaction, complexity, diversity, parameter)"
            )
        if any(float(w) < 0 for w in self.intrinsic_component_weights):
            raise ValueError("intrinsic_component_weights must be non-negative")
        if self.novelty_buffer_size <= 0:
            raise ValueError("novelty_buffer_size must be positive")
        if self.cold_start_uniform_steps < 0:
            raise ValueError("cold_start_uniform_steps must be non-negative")
        if self.skill_acc_window <= 0:
            raise ValueError("skill_acc_window must be positive")
        if self.question_family_priority_boost <= 0:
            raise ValueError("question_family_priority_boost must be positive")
        if self.branch_family_weight_multiplier <= 0:
            raise ValueError("branch_family_weight_multiplier must be positive")
        if any(not scene_kind for scene_kind, _count in quotas):
            raise ValueError("scene-kind quota names must be non-empty")
        if any(count <= 0 for _scene_kind, count in quotas):
            raise ValueError("scene-kind quota counts must be positive")
        if len({scene_kind for scene_kind, _count in quotas}) != len(quotas):
            raise ValueError("scene-kind quotas must not contain duplicate names")
        quota_total = sum(count for _scene_kind, count in quotas)
        if quota_total > self.scenes_per_batch:
            raise ValueError(
                "min_scene_kind_picks_per_step requires "
                f"{quota_total} picks but scenes_per_batch={self.scenes_per_batch}"
            )
        if self.cold_start_fewshot_steps < 0:
            raise ValueError("cold_start_fewshot_steps must be non-negative")
        if self.reference_k < 0:
            raise ValueError("reference_k must be non-negative")
        if self.reference_buffer_size <= 0:
            raise ValueError("reference_buffer_size must be positive")
        if not 0.0 <= self.copy_similarity_threshold <= 1.0:
            raise ValueError("copy_similarity_threshold must be in [0, 1]")
        if not 0.0 <= self.learnability_saturation_threshold <= 1.0:
            raise ValueError("learnability_saturation_threshold must be in [0, 1]")
        if self.learnability_saturation_weight < 0:
            raise ValueError("learnability_saturation_weight must be non-negative")
        if self.diversity_edit_distance_weight < 0:
            raise ValueError("diversity_edit_distance_weight must be non-negative")
        if self.anti_copy_start_step is not None and self.anti_copy_start_step < 0:
            raise ValueError("anti_copy_start_step must be non-negative")
        if self.learnability_weight < 0:
            raise ValueError("learnability_weight must be non-negative")
        if (
            self.learnability_weight_during_fewshot is not None
            and self.learnability_weight_during_fewshot < 0
        ):
            raise ValueError("learnability_weight_during_fewshot must be non-negative")
        if self.family_difficulty_beta < 0:
            raise ValueError("family_difficulty_beta must be non-negative")
        if self.solver_reward_scale <= 0:
            raise ValueError("solver_reward_scale must be positive")
        if self.question_families is not None:
            if len(self.question_families) == 0:
                raise ValueError("question_families override cannot be empty")
            available_fam = set(active_families(include_p1=False))
            unknown_fam = [f for f in self.question_families if f not in available_fam]
            if unknown_fam:
                raise ValueError(
                    f"question_families contains unknown / non-active "
                    f"family(ies) {unknown_fam}; active P0 families="
                    f"{sorted(available_fam)}"
                )
        available_scene_kinds = {
            lookup_family(family).scene_kind for family in self.resolved_families()
        }
        unknown_scene_kinds = [
            scene_kind
            for scene_kind, _count in quotas
            if scene_kind not in available_scene_kinds
        ]
        if unknown_scene_kinds:
            raise ValueError(
                "min_scene_kind_picks_per_step contains scene kind(s) with no "
                f"active family: {unknown_scene_kinds}"
            )

    def resolved_families(self) -> tuple[str, ...]:
        """Return the active question-family sampling vocabulary."""

        if not self.disable_branch_elision:
            if self.question_families is not None:
                return tuple(self.question_families)
            return active_families(include_p1=False)
        if self.question_families is not None:
            vocab = tuple(self.question_families)
        else:
            vocab = active_families(include_p1=False)
        filtered = tuple(
            name for name in vocab if name not in BRANCH_ELISION_FAMILIES
        )
        if not filtered:
            raise ValueError(
                "disable_branch_elision removed every remaining question family"
            )
        return filtered


# ─── Internal per-row contexts ───────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Stage2ProposerCtx:
    """Per-row Stage-2 proposer context captured before the batched rollout."""

    item_idx: int
    rollout_idx: int
    sample_id: str
    rollout_sample_id: str
    seed: int
    proposer_uid: str
    question_family: str
    scene_kind: Optional[str] = None
    fewshot_index: int = 0


@dataclass(frozen=True, slots=True)
class _Stage2PendingProposer:
    """OK-executor proposer awaiting Solver rewards for the OK-branch reward."""

    result: Any
    generation: GenerationResult
    sample_id: str


@dataclass(frozen=True, slots=True)
class _Stage2SolverCtx:
    """Per-row Stage-2 solver context captured before the batched rollout."""

    candidate_idx: int
    request: Any
    sim_state: Any
    source_proposer_record_idx: int
    source_sample_id: str
    source_item_idx: int
    solver_rollout_idx: int
    solver_uid: str
    # The question-family of THIS derived question (a scene can derive
    # questions across several families); drives per-family reward attribution.
    question_family: str = ""


# ─── Builder ──────────────────────────────────────────────────────────


class PhysEvolveStage2StepBuilder:
    """Build one Stage-2 VQA online self-play step per call.

    Public surface intentionally matches :class:`PhysEvolveOnlineStepBuilder`
    (Stage 1): ``__call__(step) -> OnlineSelfPlayStep`` with the same
    artifact shapes (``step_dir``, ``rollout_batch``,
    ``token_level_scores``). Downstream veRL plumbing therefore consumes
    Stage-1 and Stage-2 step builders interchangeably.
    """

    def __init__(
        self,
        config: PhysEvolveStage2StepBuilderConfig,
        *,
        run_dir: Path,
        backend_factory: BackendFactory | None = None,
        executor_fn: ExecutorFn | None = None,
    ) -> None:
        self.config = config
        self.run_dir = Path(run_dir)
        # Stage-2 step builder takes the same backend factory contract as
        # Stage-1: a context manager yielding a GenerationBackend. The
        # default uses the canonical HF backend; tests inject fakes.
        self.backend_factory = backend_factory or _default_backend_factory
        self.executor_fn = executor_fn or run_executor
        self._novelty_buffer: deque[SceneSignature] = deque(
            maxlen=int(config.novelty_buffer_size)
        )
        # Question-family layer: the sampling axis is the question
        # family; each family resolves to a verifying scene template.
        # ``validate_registry`` fails fast if a family binds to a missing /
        # inconsistent scene template.
        validate_registry()
        self._families: tuple[str, ...] = config.resolved_families()
        self._family_tracker: RollingAccTracker = RollingAccTracker(
            window_size=int(config.skill_acc_window)
        )
        self._family_weight_multipliers = dict(DEFAULT_FAMILY_WEIGHT_MULTIPLIERS)
        for branch_family in (
            "counterfactual_destination",
            "inverse_intervention",
        ):
            self._family_weight_multipliers[branch_family] = float(
                config.branch_family_weight_multiplier
            )
        self._family_rng: random.Random = random.Random(int(config.rng_seed) + 1)
        self._copy_diagnostics = Stage2CopyDiagnostics(
            max_recent=int(config.novelty_buffer_size)
        )
        # ReferenceBuffer keyed by scene_kind: after cold start the prompt
        # drops the executable fewshot and injects these no-coordinate
        # one-line summaries instead ("vary, do NOT copy").
        self._reference_buffer = ReferenceBuffer(
            maxlen=int(config.reference_buffer_size)
        )
        self._reference_rng: random.Random = random.Random(int(config.rng_seed) + 2)

    def _build_copy_control(
        self,
        diag: Optional[Stage2CopyDiagnostic],
        *,
        is_batch_duplicate: bool,
        active: bool,
    ) -> Optional[CopyControl]:
        """Translate copy diagnostics + config into a :class:`CopyControl`.

        Returns ``None`` (a reward no-op) during the cold-start window or
        when no anti-copy / saturation / diversity term is enabled, so the
        legacy reward path is reproduced exactly.
        """

        cfg = self.config
        any_term = (
            cfg.copy_penalty_enabled
            or cfg.batch_dedup_enabled
            or cfg.learnability_saturation_weight > 0.0
            or cfg.diversity_edit_distance_weight > 0.0
        )
        if not active or not any_term or diag is None:
            return None

        dup = bool(cfg.batch_dedup_enabled and is_batch_duplicate)
        sim_fewshot = float(diag.similarity_to_fewshot or 0.0)
        sim_recent = float(diag.max_similarity_to_recent or 0.0)
        near = max(sim_fewshot, sim_recent)
        is_exact = bool(
            cfg.copy_penalty_enabled and (diag.exact_fewshot or diag.recent_repeat)
        )
        # Reward distance from the golden/fewshot template (not recent peers) so
        # editing the shown example creates a positive gradient vs verbatim copy.
        # Exact / batch-duplicate rollouts get 0 here; ``apply_copy_control`` still
        # overrides them to ``copy_penalty``.
        if is_exact or dup:
            diversity_bonus = 0.0
        else:
            diversity_bonus = float(cfg.diversity_edit_distance_weight) * (
                1.0 - sim_fewshot
            )
        return CopyControl(
            is_exact_copy=is_exact,
            is_batch_duplicate=dup,
            copy_penalty=(
                _batch_duplicate_reward(cfg) if dup else float(cfg.copy_exact_reward)
            ),
            near_copy_similarity=near,
            near_copy_threshold=float(cfg.copy_similarity_threshold),
            copy_cap=(float(cfg.copy_cap) if cfg.copy_penalty_enabled else None),
            diversity_bonus=diversity_bonus,
            saturation_threshold=float(cfg.learnability_saturation_threshold),
            saturation_weight=float(cfg.learnability_saturation_weight),
            enabled=True,
        )

    def _write_groundtruth_trace(
        self,
        step_dir: Path,
        ctx: "_Stage2ProposerCtx",
        result: Any,
        gt_trace: dict,
    ) -> None:
        """Write a per-sample diagnostic ground-truth JSON (probe only).

        Merges the raw simulator ground-truth (from ``result.sim_state``)
        with the executor's manifest/branch/option-derivation trace. Best
        effort: any failure here must never break the rollout.
        """

        import json

        from physevolve.executor.trace import summarize_sim_state

        try:
            record: dict[str, Any] = {
                "sample_id": ctx.rollout_sample_id,
                "question_family": ctx.question_family,
                "scene_kind": ctx.scene_kind,
                "seed": ctx.seed,
            }
            sim_state = getattr(result, "sim_state", None)
            if sim_state is not None:
                record["sim_groundtruth"] = summarize_sim_state(sim_state)
            record["executor_trace"] = gt_trace
            write_text(
                step_dir / "groundtruth" / f"{ctx.rollout_sample_id}.json",
                json.dumps(record, indent=2, ensure_ascii=False) + "\n",
            )
        except Exception:  # noqa: BLE001 — diagnostics must not break rollout.
            pass

    def __call__(self, step: int) -> OnlineSelfPlayStep:
        step_dir = self.run_dir / "steps" / f"step_{int(step):06d}"

        # ── Bookkeeping accumulators ─────────────────────────────────
        reward_records: list[RewardRecord] = []
        failure_counts: Counter[str] = Counter()
        physevolve_problem_types_seen: set[str] = {"gen_vqa"}
        unsupported_problem_types_seen: set[str] = set()
        solver_reward_by_task: dict[str, list[float]] = defaultdict(list)
        solver_parse_ok = 0
        solver_schema_ok = 0
        solver_requests = 0
        solver_video_requests = 0
        executor_calls = 0
        executor_ok = 0
        ok_scene_ids: set[str] = set()
        vqa_payload_present_count = 0
        vqa_correct_idx_counts: Counter[int] = Counter()
        # Scene-template (multi-GT) accounting: total derived questions and the
        # number of OK scenes that produced >= 1 question.
        vqa_question_total_raw = 0
        vqa_question_total_before_step_cap = 0
        vqa_scene_with_questions_raw = 0
        vqa_scene_with_questions_selected = 0
        # Per-scene_kind / per-phrasing aggregations (diagnostics).
        vqa_scene_kind_counts: Counter[str] = Counter()
        vqa_phrasing_counts: Counter[str] = Counter()
        vqa_reward_by_scene_kind: dict[str, list[float]] = defaultdict(list)
        executor_calls_by_scene_kind: Counter[str] = Counter()
        executor_ok_by_scene_kind: Counter[str] = Counter()
        vqa_payload_by_scene_kind: Counter[str] = Counter()
        repair_applied_count = 0
        repair_applied_ok_count = 0
        repair_tier_counts: Counter[str] = Counter()
        repair_action_counts: Counter[str] = Counter()
        reference_recorded_tier_counts: Counter[str] = Counter()
        # Per-question-family aggregations (question-family layer).
        vqa_question_family_counts: Counter[str] = Counter()
        vqa_reward_by_family: dict[str, list[float]] = defaultdict(list)
        solver_selection_by_family: Counter[str] = Counter()
        raw_payload_by_family: Counter[str] = Counter()
        preferred_family_missing_by_family: Counter[str] = Counter()
        relation_semantic_counts: Counter[str] = Counter()
        _save_videos = (
            self.config.video_save_freq > 0
            and int(step) % self.config.video_save_freq == 0
        )
        _video_save_count = 0
        proposer_rollout_s = 0.0
        executor_s = 0.0
        solver_rollout_s = 0.0
        generation_token_count = 0
        rollout_batches: list[object] = []
        pending_proposers: dict[int, _Stage2PendingProposer] = {}
        proposer_solver_rewards: dict[int, list[float]] = defaultdict(list)
        # proposer_record_idx -> {question_family: [solver rewards]} so Phase 6
        # can update the per-family rolling tracker at QUESTION granularity (a
        # single scene can derive questions across several families).
        proposer_family_rewards: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # Map proposer_record_idx -> scene_kind for per-scene_kind diagnostics.
        record_scene_kind: dict[int, str] = {}
        # Map proposer_record_idx -> question_family for the family tracker.
        record_family: dict[int, str] = {}
        # Map proposer_record_idx -> proposer GRPO uid for group reward std.
        record_proposer_uid: dict[int, str] = {}
        # Skill snapshots are deduplicated to avoid redundant disk
        # writes when scenes_per_batch > #skills (or when adaptive_pick
        # repeatedly samples the same target on a given step).
        prompt_snapshot_written: set[str] = set()
        copy_diagnostics: list[Stage2CopyDiagnostic] = []
        # Anti-copy bookkeeping. ``controls_active`` gates copy penalties /
        # batch dedup / saturation / diversity. Decoupled from fewshot via
        # ``anti_copy_start_step`` (recipe); when unset, falls back to
        # ``cold_start_fewshot_steps`` (legacy).
        anti_copy_gate = (
            int(self.config.anti_copy_start_step)
            if self.config.anti_copy_start_step is not None
            else int(self.config.cold_start_fewshot_steps)
        )
        controls_active = int(step) > anti_copy_gate
        fewshot_active = int(step) <= int(self.config.cold_start_fewshot_steps)
        effective_learnability_weight = _effective_learnability_weight(
            self.config, fewshot_active=fewshot_active
        )
        record_copy_diag: dict[int, Stage2CopyDiagnostic] = {}
        record_batch_duplicate: dict[int, bool] = {}
        seen_diag_by_uid: dict[str, list[Stage2CopyDiagnostic]] = defaultdict(list)
        batch_duplicate_count = 0
        duplicate_failure_count = 0
        copy_penalty_applied_count = 0
        saturation_penalty_count = 0
        preferred_family_missing_count = 0
        geometry_repaired_ok_count = 0

        question_family_picks = pick_stage2_question_families_for_step(
            self._family_rng,
            self._family_tracker,
            step=int(step),
            n_picks=int(self.config.scenes_per_batch),
            min_scene_kind_picks=self.config.min_scene_kind_picks_per_step,
            families=self._families,
            cold_start_uniform_steps=int(self.config.cold_start_uniform_steps),
            priority_boost=float(self.config.question_family_priority_boost),
            family_weight_multipliers=self._family_weight_multipliers,
        )
        scene_kind_pick_counts: Counter[str] = Counter(
            str(scene_kind)
            for family in question_family_picks
            if (scene_kind := lookup_family(family).scene_kind) is not None
        )

        with self.backend_factory(self.config) as backend:
            # ── Phase 1a: per-row proposer context + prompts ────────
            proposer_ctxs: list[_Stage2ProposerCtx] = []
            proposer_specs: list[GenerationSpec] = []
            for item_idx, question_family in enumerate(question_family_picks):
                sample_id = f"step{int(step):06d}_sample{item_idx + 1:04d}"
                base_seed = (
                    self.config.seed_start
                    + (int(step) - 1) * self.config.scenes_per_batch
                    + item_idx
                )
                proposer_uid = f"{sample_id}:proposer"
                # Adaptive sampling over the QUESTION-FAMILY axis:
                # pick a question family once per item (priority
                # families boosted), then resolve it to the verifying skill
                # card. All rollouts inside the GRPO group share the family +
                # skill (variance-reduction friendly); different items can see
                # different picks.
                scene_kind = lookup_family(question_family).scene_kind
                vqa_question_family_counts[question_family] += self.config.group_size
                if scene_kind is not None:
                    vqa_scene_kind_counts[scene_kind] += self.config.group_size
                # Unified manifest path: every active family is served by a
                # grouped scene template (throwing / collision_arena / motion_lab).
                # The Proposer always authors a SCENE_MANIFEST; the executor has a
                # single scene-template path.
                if scene_kind is None:
                    raise ValueError(
                        f"active family {question_family!r} has no scene_kind; "
                        "every family must bind to a scene template after the "
                        "manifest unification"
                    )
                # Anti-copy prompt schedule: show the executable golden fewshot
                # only during cold start; afterwards drop it and inject
                # ReferenceBuffer one-line summaries so the policy keeps variance
                # and cannot collapse onto a verbatim copy.
                include_fewshot = int(step) <= int(
                    self.config.cold_start_fewshot_steps
                )
                fewshot_index = int(step) + int(item_idx)
                reference_examples: tuple[str, ...] = ()
                inject_reference = (
                    self.config.reference_buffer_enabled
                    and self.config.reference_k > 0
                    and (
                        not include_fewshot
                        or bool(self.config.reference_buffer_during_fewshot)
                    )
                )
                if inject_reference:
                    sampled = self._reference_buffer.sample(
                        self._reference_rng,
                        scene_kind,
                        k=int(self.config.reference_k),
                        allow_cross_skill_fallback=False,
                    )
                    # Skip empty / short blocks until the buffer has warmed up.
                    if len(sampled) >= int(self.config.reference_k):
                        reference_examples = format_reference_examples(sampled)
                proposer_prompt = build_proposer_scene_template_prompt(
                    scene_kind,
                    question_family=question_family,
                    include_fewshot=include_fewshot,
                    fewshot_index=fewshot_index,
                    reference_examples=reference_examples,
                )
                snapshot_key = (
                    f"{question_family}__fs{fewshot_index}"
                    if include_fewshot
                    else question_family
                )
                if snapshot_key not in prompt_snapshot_written:
                    write_text(
                        step_dir
                        / "prompt_snapshots"
                        / f"proposer__stage2__{snapshot_key}.txt",
                        proposer_prompt,
                    )
                    prompt_snapshot_written.add(snapshot_key)
                for rollout_idx in range(self.config.group_size):
                    rollout_sample_id = f"{sample_id}_r{rollout_idx:02d}"
                    seed = base_seed + rollout_idx
                    proposer_ctxs.append(
                        _Stage2ProposerCtx(
                            item_idx=item_idx,
                            rollout_idx=rollout_idx,
                            sample_id=sample_id,
                            rollout_sample_id=rollout_sample_id,
                            seed=seed,
                            proposer_uid=proposer_uid,
                            question_family=question_family,
                            scene_kind=scene_kind,
                            fewshot_index=fewshot_index,
                        )
                    )
                    proposer_specs.append(
                        GenerationSpec(
                            prompt=proposer_prompt,
                            video=None,
                            max_new_tokens=self.config.generation.max_new_tokens,
                            temperature=self.config.generation.temperature,
                            top_p=self.config.generation.top_p,
                            top_k=self.config.generation.top_k,
                            stop_sequences=("</propose>",),
                            rollout_metadata={
                                "uid": proposer_uid,
                                "group_id": proposer_uid,
                                "index": item_idx,
                                "role": "proposer",
                                "problem_type": "gen_vqa",
                                "scene_id": None,
                                "stage": 2,
                                "question_family": question_family,
                                "scene_kind": scene_kind,
                                "fewshot_index": fewshot_index,
                            },
                        )
                    )

            # ── Phase 1b: one batched proposer rollout ──────────────
            rollout_start = perf_counter()
            proposer_results = call_generate_batch(backend, proposer_specs)
            proposer_rollout_s += perf_counter() - rollout_start
            _maybe_collect_rollout(backend, rollout_batches)

            # ── Phase 2: per-row parse + Stage-2 executor + bookkeeping ──
            current_solver_candidates: list[
                tuple[_Stage2ProposerCtx, Any, int, Any, str]
            ] = []
            for ctx, proposer_generation in zip(
                proposer_ctxs, proposer_results, strict=True
            ):
                generation_token_count += _generated_token_count(proposer_generation)
                proposer_text = proposer_generation.text
                proposer_record_idx = len(reward_records)
                record_scene_kind[proposer_record_idx] = ctx.scene_kind or ""
                record_family[proposer_record_idx] = ctx.question_family
                record_proposer_uid[proposer_record_idx] = ctx.proposer_uid
                reward_records.append(
                    RewardRecord(
                        problem_type="gen_vqa",
                        reward=0.0,
                        valid_response_length=_valid_generation_length(
                            proposer_generation,
                            self.config.response_length,
                        ),
                        response_length=self.config.response_length,
                        metadata={
                            "stage": 2,
                            "scene_kind": ctx.scene_kind,
                        },
                    )
                )
                append_jsonl(
                    step_dir / "raw_proposer_outputs.jsonl",
                    {
                        "sample_id": ctx.rollout_sample_id,
                        "group_sample_id": ctx.sample_id,
                        "rollout_idx": ctx.rollout_idx,
                        "seed": ctx.seed,
                        "problem_type": "gen_vqa",
                        "scene_kind": ctx.scene_kind,
                        "question_family": ctx.question_family,
                        "fewshot_index": ctx.fewshot_index,
                        "raw_text": proposer_text,
                        "generation_metadata": proposer_generation.metadata,
                    },
                )

                code = extract_propose_block(proposer_text)
                if code is None:
                    status = FailureReason.FAILED_SYNTAX
                    failure_counts[status.value] += 1
                    reward_records[proposer_record_idx] = _stage2_proposer_failure_record(
                        sample_id=ctx.rollout_sample_id,
                        status=status,
                        generation=proposer_generation,
                        response_length=self.config.response_length,
                        scene_kind=ctx.scene_kind,
                    )
                    continue

                write_text(
                    step_dir / "code" / f"{ctx.rollout_sample_id}.py",
                    code.rstrip() + "\n",
                )
                copy_diagnostic = self._copy_diagnostics.analyze(
                    code, scene_kind=ctx.scene_kind or ""
                )
                copy_diagnostics.append(copy_diagnostic)
                is_batch_duplicate = False
                if (
                    self.config.batch_dedup_enabled
                    and controls_active
                    and copy_diagnostic.ast_valid
                ):
                    uid_diags = seen_diag_by_uid[ctx.proposer_uid]
                    is_batch_duplicate = any(
                        copy_diagnostic.ast_hash == prev.ast_hash
                        or diagnostic_similarity(copy_diagnostic, prev)
                        >= float(self.config.copy_similarity_threshold)
                        for prev in uid_diags
                    )
                    if is_batch_duplicate:
                        batch_duplicate_count += 1
                    else:
                        uid_diags.append(copy_diagnostic)
                record_copy_diag[proposer_record_idx] = copy_diagnostic
                record_batch_duplicate[proposer_record_idx] = is_batch_duplicate
                # Copy memory is deliberately broader than prompt reference
                # memory: even invalid/repaired attempts are useful anti-repeat
                # evidence, but only OK+VQA scenes are allowed into prompts.
                self._copy_diagnostics.remember(copy_diagnostic)
                executor_calls += 1
                if ctx.scene_kind:
                    executor_calls_by_scene_kind[ctx.scene_kind] += 1
                executor_start = perf_counter()
                # Stage=2 triggers get_vqa validation in the executor.
                gt_trace: dict = {}
                exec_kwargs: dict[str, Any] = dict(
                    stage=2,
                    seed=ctx.seed,
                    sandbox_timeout_s=self.config.sandbox_timeout_s,
                    sim_timeout_s=self.config.sim_timeout_s,
                    color_variants_per_scene=self.config.color_variants_per_scene,
                    camera_depth_bypass_enabled=(
                        self.config.camera_depth_bypass_enabled
                    ),
                )
                if self.config.disable_branch_elision:
                    exec_kwargs["skip_branch_sims"] = True
                if self.config.emit_groundtruth_trace:
                    exec_kwargs["trace"] = gt_trace
                result = self.executor_fn(code, **exec_kwargs)
                executor_s += perf_counter() - executor_start
                result_row = executor_result_record(result)
                append_jsonl(
                    step_dir / "executor_results.jsonl",
                    {"sample_id": ctx.rollout_sample_id, **result_row},
                )
                failure_counts[str(result_row["status"])] += 1
                executor_ok += int(bool(result_row["ok"]))
                if bool(result_row["ok"]) and ctx.scene_kind:
                    executor_ok_by_scene_kind[ctx.scene_kind] += 1
                if bool(result_row.get("repair_applied", False)):
                    repair_applied_count += 1
                    if bool(result_row["ok"]):
                        repair_applied_ok_count += 1
                    repair_tier_counts[str(result_row.get("repair_tier", "none"))] += 1
                    for action in result_row.get("repair_actions", []):
                        if isinstance(action, dict):
                            repair_action_counts[str(action.get("kind", "unknown"))] += 1

                ok = bool(getattr(result, "ok", False))
                vqa_payload = getattr(result, "vqa_payload", None)
                vqa_payloads = getattr(result, "vqa_payloads", None)
                resolved_spec = getattr(result, "vqa_spec", None)
                if self.config.disable_branch_elision:
                    vqa_payloads = _drop_branch_elision_payloads(vqa_payloads)
                    if vqa_payloads:
                        vqa_payload = _payload_tuple_from_item(vqa_payloads[0])
                    elif vqa_payloads is not None:
                        vqa_payload = None
                if ok and _is_geometry_repaired(result):
                    geometry_repaired_ok_count += 1
                # Count executor-emitted payloads before the preferred-family
                # gate and both Solver caps. Together with sampled-family and
                # Solver-selection metrics this exposes the complete funnel.
                if ok and vqa_payloads:
                    for item in vqa_payloads:
                        family = _payload_family(item)
                        if family:
                            raw_payload_by_family[family] += 1
                        spec = item.get("spec", {})
                        if spec.get("family") == "collision_final_relation":
                            semantic = spec.get("expected_relation")
                            if isinstance(semantic, str) and semantic:
                                relation_semantic_counts[semantic] += 1
                elif ok and vqa_payload is not None and isinstance(resolved_spec, dict):
                    family = resolved_spec.get("family")
                    if isinstance(family, str) and family:
                        raw_payload_by_family[family] += 1

                preferred_family_missing = bool(
                    ok
                    and vqa_payload is not None
                    and not _result_supports_question_family(
                        result, ctx.question_family
                    )
                )
                if preferred_family_missing:
                    preferred_family_missing_count += 1
                    preferred_family_missing_by_family[ctx.question_family] += 1
                    failure_counts["preferred_family_missing"] += 1
                    if is_batch_duplicate:
                        duplicate_failure_count += 1
                    failure_record = _stage2_proposer_failure_record(
                        sample_id=ctx.rollout_sample_id,
                        status=FailureReason.FAILED_VQA,
                        generation=proposer_generation,
                        response_length=self.config.response_length,
                        scene_kind=ctx.scene_kind,
                    )
                    reward_records[proposer_record_idx] = replace(
                        failure_record,
                        scene_id=getattr(result, "scene_id", ""),
                        metadata={
                            **(failure_record.metadata or {}),
                            "question_family": ctx.question_family,
                            "preferred_family_missing": True,
                            "batch_duplicate": bool(is_batch_duplicate),
                            "vqa_payload_present": bool(vqa_payload is not None),
                            **_repair_metadata(result),
                        },
                    )
                    append_jsonl(
                        step_dir / "reward_records.jsonl",
                        {
                            "sample_id": ctx.rollout_sample_id,
                            "scene_id": getattr(result, "scene_id", ""),
                            "problem_type": "gen_vqa",
                            "reward": float(reward_records[proposer_record_idx].reward),
                            "status": FailureReason.FAILED_VQA.value,
                            "preferred_family_missing": True,
                            "required_family": ctx.question_family,
                            "scene_kind": ctx.scene_kind,
                            "batch_duplicate": bool(is_batch_duplicate),
                            **_repair_metadata(result),
                        },
                    )
                    continue

                resolved_phrasing: Optional[str] = None
                solver_result = result
                if self.config.emit_groundtruth_trace and ok:
                    self._write_groundtruth_trace(
                        step_dir, ctx, result, gt_trace
                    )
                if ok and result_row["scene_id"]:
                    ok_scene_ids.add(str(result_row["scene_id"]))
                    if (
                        _save_videos
                        and _video_save_count < self.config.video_save_max_per_step
                        and getattr(result, "sim_state", None) is not None
                        and result.sim_state.video_frames is not None
                    ):
                        video_path = (
                            step_dir
                            / "videos"
                            / f"{ctx.rollout_sample_id}_{ctx.scene_kind}.mp4"
                        )
                        save_video_mp4(result.sim_state.video_frames, video_path)
                        _video_save_count += 1
                if ok and vqa_payload is not None:
                    vqa_payload_present_count += 1
                    if ctx.scene_kind:
                        vqa_payload_by_scene_kind[ctx.scene_kind] += 1
                    if vqa_payloads:
                        # Scene-template path: count every derived question's
                        # letter + phrasing/family before any Solver sampling.
                        vqa_scene_with_questions_raw += 1
                        vqa_question_total_raw += len(vqa_payloads)
                        for item in vqa_payloads:
                            vqa_correct_idx_counts[int(item["payload"][2])] += 1
                            spec = item.get("spec", {})
                            ph = spec.get("phrasing")
                            if isinstance(ph, str) and ph:
                                vqa_phrasing_counts[ph] += 1
                        selected_vqa_payloads = _select_stage2_vqa_payloads(
                            list(vqa_payloads),
                            preferred_family=ctx.question_family,
                            max_questions=int(
                                self.config.max_solver_questions_per_scene
                            ),
                            seed_key=(
                                f"stage2:vqa-select:{self.config.rng_seed}:"
                                f"{int(step)}:{ctx.rollout_sample_id}"
                            ),
                        )
                        if selected_vqa_payloads:
                            vqa_scene_with_questions_selected += 1
                            vqa_question_total_before_step_cap += len(
                                selected_vqa_payloads
                            )
                            solver_result = replace(
                                result,
                                vqa_payload=_payload_tuple_from_item(
                                    selected_vqa_payloads[0]
                                ),
                                vqa_payloads=selected_vqa_payloads,
                            )
                    else:
                        vqa_question_total_raw += 1
                        vqa_question_total_before_step_cap += 1
                        vqa_scene_with_questions_raw += 1
                        vqa_scene_with_questions_selected += 1
                        vqa_correct_idx_counts[int(vqa_payload[2])] += 1
                        if isinstance(resolved_spec, dict):
                            ph = resolved_spec.get("phrasing")
                            if isinstance(ph, str) and ph:
                                resolved_phrasing = ph
                                vqa_phrasing_counts[ph] += 1

                if (
                    ok
                    and getattr(result, "sim_state", None) is not None
                    and not is_batch_duplicate
                ):
                    # One Solver request per derived question. ``vqa_payloads``
                    # is ordered identically to the requests, so we align each
                    # request with its question-family for per-family reward
                    # attribution (a scene can derive several families).
                    requests = build_solver_requests(
                        solver_result,
                        task_modes=(TaskMode.VQA,),
                        sample_id=ctx.rollout_sample_id,
                    )
                    question_specs = list(getattr(solver_result, "vqa_payloads", None) or [])
                    for q_idx, request in enumerate(requests):
                        physevolve_problem_types_seen.add(request.problem_type)
                        q_family = ""
                        if q_idx < len(question_specs):
                            q_family = str(
                                question_specs[q_idx].get("spec", {}).get(
                                    "family", ""
                                )
                            )
                        if not q_family:
                            q_family = ctx.question_family
                        current_solver_candidates.append(
                            (ctx, request, proposer_record_idx, result.sim_state, q_family)
                        )
                    pending_proposers[proposer_record_idx] = _Stage2PendingProposer(
                        result=result,
                        generation=proposer_generation,
                        sample_id=ctx.rollout_sample_id,
                    )
                else:
                    # Failed scene, OK-with-no-payload, OR an OK batch
                    # duplicate that we deliberately did not send to the Solver.
                    # The copy control forces an overlap-aware duplicate
                    # reward for OK duplicates. Failed duplicates are also
                    # shaped below because failure rewards ignore copy_control.
                    dup_control = self._build_copy_control(
                        record_copy_diag.get(proposer_record_idx),
                        is_batch_duplicate=is_batch_duplicate,
                        active=controls_active,
                    )
                    proposer_reward = compute_proposer_reward(
                        result,
                        solver_rewards=[],
                        failure_weight=self.config.proposer_failure_weight,
                        copy_control=dup_control,
                    )
                    if is_batch_duplicate:
                        duplicate_failure_count += int(not ok)
                        proposer_reward = min(
                            float(proposer_reward),
                            _batch_duplicate_reward(self.config),
                        )
                        copy_penalty_applied_count += 1
                    reward_records[proposer_record_idx] = RewardRecord(
                        problem_type="gen_vqa",
                        reward=float(proposer_reward),
                        valid_response_length=_valid_generation_length(
                            proposer_generation, self.config.response_length
                        ),
                        response_length=self.config.response_length,
                        scene_id=getattr(result, "scene_id", ""),
                        metadata={
                            "n_solver_rewards": 0,
                            "stage": 2,
                            "vqa_payload_present": bool(vqa_payload is not None),
                            "batch_duplicate": bool(is_batch_duplicate),
                            "duplicate_failure": bool(is_batch_duplicate and not ok),
                            **_repair_metadata(result),
                        },
                    )
                    if is_batch_duplicate:
                        # Audit the deduplicated rollout so logs show why it got
                        # the copy reward and skipped the Solver.
                        append_jsonl(
                            step_dir / "reward_records.jsonl",
                            {
                                "sample_id": ctx.rollout_sample_id,
                                "scene_id": getattr(result, "scene_id", ""),
                                "problem_type": "gen_vqa",
                                "reward": float(proposer_reward),
                                "n_solver_rewards": 0,
                                "scene_kind": ctx.scene_kind,
                                "question_family": ctx.question_family,
                                "batch_duplicate": True,
                                "duplicate_failure": bool(not ok),
                            },
                        )

            # ── Phase 3: cap Solver question groups + prepare specs ──
            available_solver_candidates = list(current_solver_candidates)
            selected_solver_candidates = _select_stage2_solver_candidates_for_step(
                available_solver_candidates,
                max_groups=int(self.config.max_solver_groups_per_step),
            )
            solver_variant_request_count = 0
            for _src_ctx, request, _src_record_idx, sim_state, src_family in (
                selected_solver_candidates
            ):
                solver_selection_by_family[src_family] += 1
                if getattr(sim_state, "scene_id", None) != request.scene_id:
                    solver_variant_request_count += 1

            solver_ctxs: list[_Stage2SolverCtx] = []
            solver_specs: list[GenerationSpec] = []
            for candidate_idx, (
                src_ctx,
                request,
                src_record_idx,
                sim_state,
                src_family,
            ) in enumerate(selected_solver_candidates):
                physevolve_problem_types_seen.add(request.problem_type)
                solver_requests += self.config.group_size
                write_text(
                    step_dir / "prompt_snapshots" / f"{request.request_id}.txt",
                    request.prompt,
                )
                request_row = video_request_record(request)
                solver_video_requests += self.config.group_size * int(
                    bool(request_row["video_payload_present"])
                )
                append_jsonl(step_dir / "solver_requests.jsonl", request_row)

                solver_uid = f"{request.request_id}:solver:{candidate_idx}"
                for solver_rollout_idx in range(self.config.group_size):
                    solver_ctxs.append(
                        _Stage2SolverCtx(
                            candidate_idx=candidate_idx,
                            request=request,
                            sim_state=sim_state,
                            source_proposer_record_idx=src_record_idx,
                            source_sample_id=src_ctx.rollout_sample_id,
                            source_item_idx=src_ctx.item_idx,
                            solver_rollout_idx=solver_rollout_idx,
                            solver_uid=solver_uid,
                            question_family=src_family,
                        )
                    )
                    solver_specs.append(
                        GenerationSpec(
                            prompt=request.prompt,
                            video=request.video,
                            max_new_tokens=self.config.generation.max_new_tokens,
                            temperature=self.config.generation.temperature,
                            top_p=self.config.generation.top_p,
                            top_k=self.config.generation.top_k,
                            stop_sequences=("</answer>",),
                            rollout_metadata={
                                "uid": solver_uid,
                                "group_id": solver_uid,
                                "index": src_ctx.item_idx,
                                "role": "solver",
                                "problem_type": request.problem_type,
                                "scene_id": request.scene_id,
                                "stage": 2,
                            },
                        )
                    )

            # ── Phase 4: chunked batched VQA solver rollout ──────────
            # Multi-question scenes (throwing) at high executor-OK rates can
            # push the solver row count (each row carrying a full video) high
            # enough to OOM a single generate_sequences dispatch. Chunking
            # bounds the concurrently-materialised video tensors; the helper
            # drains the rollout after every chunk so all rows still reach the
            # PPO update in order (zero training-quality impact).
            rollout_start = perf_counter()
            solver_results = call_generate_batch_chunked(
                backend,
                solver_specs,
                rollout_batches,
                self.config.solver_rollout_chunk_size,
            )
            solver_rollout_s += perf_counter() - rollout_start

            # ── Phase 5: per-row VQA scoring + reward records ────────
            for solver_ctx, solver_generation in zip(
                solver_ctxs, solver_results, strict=True
            ):
                request = solver_ctx.request
                generation_token_count += _generated_token_count(solver_generation)
                solver_text = solver_generation.text
                append_jsonl(
                    step_dir / "solver_outputs.jsonl",
                    {
                        "sample_id": solver_ctx.source_sample_id,
                        "request_id": request.request_id,
                        "solver_rollout_idx": solver_ctx.solver_rollout_idx,
                        "scene_id": request.scene_id,
                        "problem_type": request.problem_type,
                        "task_mode": request.mode.value,
                        "raw_text": solver_text,
                        "generation_metadata": solver_generation.metadata,
                    },
                )

                score = score_solver_response(
                    request,
                    solver_text,
                    sim_state=solver_ctx.sim_state,
                )
                solver_parse_ok += int(score.answer_parse_ok)
                solver_schema_ok += int(score.schema_ok)
                proposer_solver_rewards[solver_ctx.source_proposer_record_idx].append(
                    score.reward
                )
                if solver_ctx.question_family:
                    proposer_family_rewards[
                        solver_ctx.source_proposer_record_idx
                    ][solver_ctx.question_family].append(score.reward)
                solver_reward_by_task[request.problem_type].append(score.reward)
                reward_records.append(
                    RewardRecord(
                        problem_type=request.problem_type,
                        reward=score.reward,
                        valid_response_length=_valid_generation_length(
                            solver_generation, self.config.response_length
                        ),
                        response_length=self.config.response_length,
                        scene_id=request.scene_id,
                        metadata={
                            "request_id": request.request_id,
                            "schema_ok": score.schema_ok,
                            "answer_parse_ok": score.answer_parse_ok,
                            "stage": 2,
                        },
                    )
                )
                append_jsonl(
                    step_dir / "reward_records.jsonl",
                    {
                        "sample_id": solver_ctx.source_sample_id,
                        "request_id": request.request_id,
                        "solver_rollout_idx": solver_ctx.solver_rollout_idx,
                        "scene_id": request.scene_id,
                        "problem_type": request.problem_type,
                        "reward": score.reward,
                        "schema_ok": score.schema_ok,
                        "answer_parse_ok": score.answer_parse_ok,
                        "parsed_answer": score.parsed_answer,
                    },
                )

            # ── Phase 6: OK-branch Proposer reward (learnability + intrinsic) ──
            intrinsic_components_log: dict[str, list[float]] = defaultdict(list)
            history_snapshot: tuple[SceneSignature, ...] = tuple(self._novelty_buffer)
            scored_signatures: list[SceneSignature] = []
            for proposer_record_idx in sorted(pending_proposers):
                pending = pending_proposers[proposer_record_idx]
                result = pending.result
                proposer_generation = pending.generation
                rewards = proposer_solver_rewards[proposer_record_idx]
                scene_kind = record_scene_kind.get(proposer_record_idx, "")
                resolved_spec = getattr(result, "vqa_spec", None) or {}
                resolved_phrasing = (
                    resolved_spec.get("phrasing")
                    if isinstance(resolved_spec, dict)
                    else None
                )
                resolved_params = (
                    resolved_spec.get("params")
                    if isinstance(resolved_spec, dict)
                    else None
                )
                params_sig = (
                    canonical_params_signature(resolved_params)
                    if isinstance(resolved_params, dict)
                    else None
                )

                intrinsic_inputs: Optional[IntrinsicInputs] = None
                signature: Optional[SceneSignature] = None
                if (
                    self.config.intrinsic_enabled
                    and getattr(result, "sim_state", None) is not None
                ):
                    signature = compute_scene_signature(
                        result.sim_state,
                        skill_target=scene_kind or None,
                        phrasing=resolved_phrasing,
                        params_signature=params_sig,
                    )
                    history_for_scene = history_snapshot + tuple(scored_signatures)
                    intrinsic_inputs = IntrinsicInputs(
                        sim_state=result.sim_state,
                        signature=signature,
                        history=history_for_scene,
                        weight=float(self.config.intrinsic_weight),
                        component_weights=tuple(self.config.intrinsic_component_weights),
                    )

                # Stage-2 only has one Solver task per scene, so the
                # learnability minimum-rollouts threshold is just
                # ``group_size`` (or however many actually came back).
                min_rollouts = _stage2_learnability_min_rollouts(
                    self.config, len(rewards)
                )
                copy_control = self._build_copy_control(
                    record_copy_diag.get(proposer_record_idx),
                    is_batch_duplicate=record_batch_duplicate.get(
                        proposer_record_idx, False
                    ),
                    active=controls_active,
                )
                # learnability from the sampled family's solver mean
                # (not the scene-wide pool) so multi-family scenes do not dilute
                # throwing / collision difficulty signals.
                sampled_family = record_family.get(proposer_record_idx, "")
                fam_bucket = proposer_family_rewards.get(proposer_record_idx, {})
                learnability_rewards = rewards
                if sampled_family and fam_bucket.get(sampled_family):
                    learnability_rewards = list(fam_bucket[sampled_family])
                _mean_acc = (
                    sum(float(r) for r in learnability_rewards)
                    / float(len(learnability_rewards))
                    if learnability_rewards
                    else 0.0
                )
                if intrinsic_inputs is not None:
                    proposer_reward, breakdown = compute_proposer_reward_with_intrinsic(
                        result,
                        solver_rewards=learnability_rewards,
                        intrinsic=intrinsic_inputs,
                        failure_weight=self.config.proposer_failure_weight,
                        learnability_weight=float(effective_learnability_weight),
                        min_rollouts_for_learnability=min_rollouts,
                        copy_control=copy_control,
                    )
                else:
                    proposer_reward = compute_proposer_reward(
                        result,
                        solver_rewards=learnability_rewards,
                        failure_weight=self.config.proposer_failure_weight,
                        learnability_weight=float(effective_learnability_weight),
                        min_rollouts_for_learnability=min_rollouts,
                        copy_control=copy_control,
                    )
                    breakdown = None

                proposer_reward_pre_family = float(proposer_reward)
                family_weight = 1.0
                if (
                    float(self.config.family_difficulty_beta) > 0.0
                    and sampled_family
                    and result.status == FailureReason.OK
                ):
                    family_weight = family_difficulty_weight(
                        self._family_tracker.mean(sampled_family),
                        float(self.config.family_difficulty_beta),
                    )
                    proposer_reward = float(proposer_reward) * family_weight

                if copy_control is not None and (
                    copy_control.is_exact_copy or copy_control.is_batch_duplicate
                ):
                    copy_penalty_applied_count += 1
                if (
                    copy_control is not None
                    and copy_control.saturation_weight > 0.0
                    and len(learnability_rewards) >= min_rollouts
                    and _mean_acc > copy_control.saturation_threshold
                ):
                    saturation_penalty_count += 1

                metadata: dict[str, Any] = {
                    "n_solver_rewards": len(rewards),
                    "n_learnability_rewards": len(learnability_rewards),
                    "stage": 2,
                    "vqa_payload_present": True,
                    "scene_kind": scene_kind,
                    "question_family": record_family.get(proposer_record_idx, ""),
                    "proposer_uid": record_proposer_uid.get(proposer_record_idx, ""),
                    "proposer_reward_pre_family_weight": proposer_reward_pre_family,
                    "proposer_reward_family_weight": float(family_weight),
                    "stage2_effective_learnability_weight": float(
                        effective_learnability_weight
                    ),
                    "stage2_diversity_bonus": float(
                        copy_control.diversity_bonus if copy_control is not None else 0.0
                    ),
                    **_repair_metadata(result),
                }
                if copy_control is not None:
                    metadata["copy_exact"] = bool(copy_control.is_exact_copy)
                    metadata["copy_near_similarity"] = float(
                        copy_control.near_copy_similarity
                    )
                if resolved_phrasing is not None:
                    metadata["phrasing"] = resolved_phrasing
                if breakdown is not None:
                    metadata["intrinsic_components"] = dict(breakdown.components)
                    metadata["intrinsic_weighted_sum"] = breakdown.weighted_sum
                    metadata["intrinsic_total"] = breakdown.total
                    for name, value in breakdown.components.items():
                        intrinsic_components_log[name].append(float(value))

                reward_records[proposer_record_idx] = RewardRecord(
                    problem_type="gen_vqa",
                    reward=float(proposer_reward),
                    valid_response_length=_valid_generation_length(
                        proposer_generation, self.config.response_length
                    ),
                    response_length=self.config.response_length,
                    scene_id=getattr(result, "scene_id", ""),
                    metadata=metadata,
                )
                proposer_row: dict[str, Any] = {
                    "sample_id": pending.sample_id,
                    "scene_id": getattr(result, "scene_id", ""),
                    "problem_type": "gen_vqa",
                    "reward": float(proposer_reward),
                    "n_solver_rewards": len(rewards),
                    "scene_kind": scene_kind,
                    "question_family": record_family.get(proposer_record_idx, ""),
                    "phrasing": resolved_phrasing,
                    **_repair_metadata(result),
                }
                if breakdown is not None:
                    for name, value in breakdown.components.items():
                        proposer_row[f"intrinsic/{name}"] = float(value)
                    proposer_row["intrinsic/weighted_sum"] = float(breakdown.weighted_sum)
                    proposer_row["intrinsic/total"] = float(breakdown.total)
                append_jsonl(step_dir / "reward_records.jsonl", proposer_row)
                if signature is not None:
                    scored_signatures.append(signature)

                # Per-family rolling tracker update. The family
                # tracker is updated at QUESTION granularity: each derived
                # question's solver accuracy updates ITS OWN family, so a
                # multi-family scene drives every family it served rather than
                # only the sampled one. The per-scene_kind mean is kept for
                # diagnostics only (it does not drive sampling).
                question_family = record_family.get(proposer_record_idx, "")
                if rewards and scene_kind:
                    mean_acc = sum(float(r) for r in rewards) / float(len(rewards))
                    vqa_reward_by_scene_kind[scene_kind].append(mean_acc)
                fam_rewards = proposer_family_rewards.get(proposer_record_idx, {})
                if fam_rewards:
                    for fam, fam_rl in fam_rewards.items():
                        if not fam_rl:
                            continue
                        fam_mean = sum(float(r) for r in fam_rl) / float(len(fam_rl))
                        self._family_tracker.update(fam, fam_mean)
                        vqa_reward_by_family[fam].append(fam_mean)
                elif rewards and question_family:
                    # Fallback (no per-question family captured): attribute the
                    # scene mean to the sampled family.
                    fam_mean = sum(float(r) for r in rewards) / float(len(rewards))
                    self._family_tracker.update(question_family, fam_mean)
                    vqa_reward_by_family[question_family].append(fam_mean)

                # Record this OK scene into the per-scene_kind ReferenceBuffer so
                # later (post-cold-start) prompts can cite coordinate-free
                # summaries instead of the executable fewshot.
                if (
                    self.config.reference_buffer_enabled
                    and scene_kind
                    and getattr(result, "sim_state", None) is not None
                    and _ok_scene_is_reference_memory(result)
                ):
                    payload_families = _result_payload_families(result)
                    self._reference_buffer.record(
                        scene_kind,
                        result.sim_state,
                        str(getattr(result, "scene_id", "") or "scene"),
                        scene_kind=scene_kind,
                        families=payload_families,
                        n_questions=len(
                            getattr(result, "vqa_payloads", None) or []
                        ),
                        repair_tier=_repair_tier(result),
                    )
                    reference_recorded_tier_counts[_repair_tier(result)] += 1

            for signature in scored_signatures:
                self._novelty_buffer.append(signature)

        proposer_group_reward_std = _proposer_group_reward_std(
            reward_records, record_proposer_uid
        )

        # ── Phase 7: assemble step output ────────────────────────────
        scale = float(self.config.solver_reward_scale)
        if scale != 1.0:
            reward_records = [
                replace(row, reward=float(row.reward) * scale)
                if row.problem_type == "pred_vqa"
                else row
                for row in reward_records
            ]
        reward_batch = build_reward_batch(
            reward_records, response_length=self.config.response_length
        )
        mean_reward_by_task = {
            task: float(np.mean(values))
            for task, values in sorted(solver_reward_by_task.items())
            if values
        }
        unsupported_problem_types_seen |= set(physevolve_problem_types_seen) - set(PHYSEVOLVE_STAGE2_PROBLEM_TYPES)
        intrinsic_means = {
            f"intrinsic/{name}_mean": float(np.mean(values))
            for name, values in sorted(intrinsic_components_log.items())
            if values
        }
        # vqa_payload_present_rate uses executor_ok as the denominator: an
        # OK scene without a VQA payload counts against the Proposer's
        # ability to emit a valid get_vqa.
        vqa_payload_present_rate = _rate(vqa_payload_present_count, executor_ok)
        # Denominator is the total number of derived questions (multi-GT), so
        # the letter distribution reflects every question, not just one/scene.
        _idx_denom = vqa_question_total_raw or vqa_payload_present_count
        vqa_correct_idx_distribution = {
            str(letter): float(_rate(vqa_correct_idx_counts.get(idx, 0), _idx_denom))
            for idx, letter in enumerate(("A", "B", "C", "D"))
        }
        solver_scene_selected_count = len(
            {c[0].rollout_sample_id for c in selected_solver_candidates}
        )
        # Per-scene_kind mean reward (diagnostics; the live sampling axis is the
        # question family below).
        mean_reward_by_scene_kind = {
            scene_kind: float(np.mean(values))
            for scene_kind, values in sorted(vqa_reward_by_scene_kind.items())
            if values
        }
        # Question-family layer: per-family mean reward + sampler weights +
        # rolling acc so the dashboard surfaces which question
        # shapes are learnable and whether the priority boost is taking hold.
        mean_reward_by_family = {
            family: float(np.mean(values))
            for family, values in sorted(vqa_reward_by_family.items())
            if values
        }
        family_pick_weights = stage2_family_pick_weight_snapshot(
            self._family_tracker,
            families=self._families,
            priority_boost=float(self.config.question_family_priority_boost),
            family_weight_multipliers=self._family_weight_multipliers,
        )
        family_acc_snapshot = self._family_tracker.snapshot()
        reference_buffer_sizes = self._reference_buffer.snapshot_sizes()
        active_scene_kinds = sorted(
            {
                str(lookup_family(family).scene_kind)
                for family in self._families
                if lookup_family(family).scene_kind is not None
            }
        )
        executor_ok_rate_by_scene_kind = {
            scene_kind: _rate(
                executor_ok_by_scene_kind.get(scene_kind, 0),
                executor_calls_by_scene_kind.get(scene_kind, 0),
            )
            for scene_kind in active_scene_kinds
            if executor_calls_by_scene_kind.get(scene_kind, 0) > 0
        }
        vqa_payload_rate_by_scene_kind = {
            scene_kind: _rate(
                vqa_payload_by_scene_kind.get(scene_kind, 0),
                executor_ok_by_scene_kind.get(scene_kind, 0),
            )
            for scene_kind in active_scene_kinds
            if executor_ok_by_scene_kind.get(scene_kind, 0) > 0
        }
        metrics = {
            "online_proposer_generation_count": self.config.scenes_per_batch * self.config.group_size,
            "executor_call_count": executor_calls,
            "n_executor_ok": executor_ok,
            "n_executor_failed": max(0, executor_calls - executor_ok),
            "unique_scene_ratio": _rate(len(ok_scene_ids), executor_ok),
            "failure_reason_distribution": dict(sorted(failure_counts.items())),
            "n_solver_requests": solver_requests,
            "n_solver_groups_active": len(selected_solver_candidates),
            "solver_video_request_count": solver_video_requests,
            "answer_parse_rate": _rate(solver_parse_ok, solver_requests),
            "schema_ok_rate": _rate(solver_schema_ok, solver_requests),
            "mean_reward_by_task": mean_reward_by_task,
            "all_rewards_finite": reward_batch.metrics["reward_tensor_finite"],
            "executor_workers_configured": int(self.config.executor_workers),
            "executor_parallelism_active": False,
            "time/proposer_rollout_s": float(proposer_rollout_s),
            "time/executor_s": float(executor_s),
            "time/solver_rollout_s": float(solver_rollout_s),
            "n_proposer_rollouts_per_batch": len(proposer_specs),
            "n_solver_rollouts_per_batch": len(solver_specs),
            "online_generation_token_count": int(generation_token_count),
            "physevolve_problem_types_seen": sorted(physevolve_problem_types_seen),
            "unsupported_problem_types_seen": sorted(unsupported_problem_types_seen),
            "expected_problem_types": list(PHYSEVOLVE_STAGE2_PROBLEM_TYPES),
            "stage": 2,
            "vqa_payload_present_rate": vqa_payload_present_rate,
            "vqa_payload_present_count": vqa_payload_present_count,
            "vqa_question_total": vqa_question_total_raw,
            "vqa_question_total_raw": vqa_question_total_raw,
            "vqa_question_total_before_step_cap": vqa_question_total_before_step_cap,
            "vqa_question_total_selected": len(selected_solver_candidates),
            "vqa_questions_per_scene": _rate(
                vqa_question_total_raw, vqa_scene_with_questions_raw
            ),
            "vqa_questions_per_scene_before_step_cap": _rate(
                vqa_question_total_before_step_cap, vqa_scene_with_questions_selected
            ),
            "vqa_questions_per_scene_selected": _rate(
                len(selected_solver_candidates), solver_scene_selected_count
            ),
            "vqa_correct_idx_distribution": vqa_correct_idx_distribution,
            "vqa_scene_kind_counts": dict(sorted(vqa_scene_kind_counts.items())),
            "vqa_phrasing_counts": dict(sorted(vqa_phrasing_counts.items())),
            "vqa_reward_by_scene_kind": mean_reward_by_scene_kind,
            "vqa_question_family_counts": dict(sorted(vqa_question_family_counts.items())),
            "vqa_reward_by_family": mean_reward_by_family,
            "vqa_family_pick_weights": family_pick_weights,
            "vqa_family_rolling_acc": family_acc_snapshot,
            "stage2_raw_payload_by_family": dict(
                sorted(raw_payload_by_family.items())
            ),
            "stage2_preferred_family_missing_by_family": dict(
                sorted(preferred_family_missing_by_family.items())
            ),
            "stage2_relation_semantic_counts": dict(
                sorted(relation_semantic_counts.items())
            ),
            "stage2_families_active": list(self._families),
            "stage2_scene_kinds_active": active_scene_kinds,
            "stage2_min_scene_kind_picks_per_step": dict(
                self.config.min_scene_kind_picks_per_step
            ),
            "stage2_scene_kind_pick_counts": dict(
                sorted(scene_kind_pick_counts.items())
            ),
            "stage2_executor_calls_by_scene_kind": dict(
                sorted(executor_calls_by_scene_kind.items())
            ),
            "stage2_executor_ok_by_scene_kind": dict(
                sorted(executor_ok_by_scene_kind.items())
            ),
            "stage2_executor_ok_rate_by_scene_kind": executor_ok_rate_by_scene_kind,
            "stage2_vqa_payload_by_scene_kind": dict(
                sorted(vqa_payload_by_scene_kind.items())
            ),
            "stage2_vqa_payload_rate_by_scene_kind": vqa_payload_rate_by_scene_kind,
            "stage2_repair_applied_count": int(repair_applied_count),
            "stage2_repair_applied_ok_count": int(repair_applied_ok_count),
            "stage2_repair_applied_rate": _rate(repair_applied_count, executor_calls),
            "stage2_repair_ok_rate": _rate(repair_applied_ok_count, executor_ok),
            "stage2_repair_tier_counts": dict(sorted(repair_tier_counts.items())),
            "stage2_repair_action_counts": dict(sorted(repair_action_counts.items())),
            "stage2_geometry_repaired_ok_count": int(geometry_repaired_ok_count),
            "stage2_preferred_family_missing_count": int(
                preferred_family_missing_count
            ),
            "n_solver_groups_available": len(available_solver_candidates),
            "n_solver_groups_selected": len(selected_solver_candidates),
            "solver_groups_dropped": max(
                0, len(available_solver_candidates) - len(selected_solver_candidates)
            ),
            "solver_selection_by_family": dict(sorted(solver_selection_by_family.items())),
            "solver_selection_variant_count": int(solver_variant_request_count),
            "stage2_max_solver_questions_per_scene": int(
                self.config.max_solver_questions_per_scene
            ),
            "stage2_max_solver_groups_per_step": int(
                self.config.max_solver_groups_per_step
            ),
            "stage2_camera_depth_bypass_enabled": bool(
                self.config.camera_depth_bypass_enabled
            ),
            "novelty_buffer_size": len(self._novelty_buffer),
            "novelty_buffer_capacity": int(self.config.novelty_buffer_size),
            "intrinsic_enabled": bool(self.config.intrinsic_enabled),
            "intrinsic_weight": float(self.config.intrinsic_weight),
            "intrinsic_component_weights": [
                float(w) for w in self.config.intrinsic_component_weights
            ],
            "stage2_anti_copy_active": bool(controls_active),
            "stage2_anti_copy_start_step": int(anti_copy_gate),
            "stage2_learnability_weight": float(self.config.learnability_weight),
            "stage2_learnability_weight_during_fewshot": (
                None
                if self.config.learnability_weight_during_fewshot is None
                else float(self.config.learnability_weight_during_fewshot)
            ),
            "stage2_effective_learnability_weight": float(
                effective_learnability_weight
            ),
            "stage2_fewshot_active": bool(fewshot_active),
            "stage2_proposer_group_reward_std_mean": float(
                proposer_group_reward_std
            ),
            "stage2_family_difficulty_beta": float(self.config.family_difficulty_beta),
            "stage2_branch_family_weight_multiplier": float(
                self.config.branch_family_weight_multiplier
            ),
            "stage2_solver_reward_scale": float(self.config.solver_reward_scale),
            "stage2_cold_start_fewshot_steps": int(
                self.config.cold_start_fewshot_steps
            ),
            "stage2_batch_duplicate_count": int(batch_duplicate_count),
            "stage2_duplicate_failure_count": int(duplicate_failure_count),
            "stage2_copy_penalty_applied_count": int(copy_penalty_applied_count),
            "stage2_saturation_penalty_count": int(saturation_penalty_count),
            "stage2_reference_buffer_sizes": reference_buffer_sizes,
            "stage2_reference_buffer_empty_scene_kinds": [
                scene_kind
                for scene_kind in active_scene_kinds
                if reference_buffer_sizes.get(scene_kind, 0) == 0
            ],
            "stage2_reference_buffer_during_fewshot": bool(
                self.config.reference_buffer_during_fewshot
            ),
            "stage2_reference_recorded_repair_tier_counts": dict(
                sorted(reference_recorded_tier_counts.items())
            ),
            "stage2_diversity_edit_distance_weight": float(
                self.config.diversity_edit_distance_weight
            ),
            **summarize_copy_diagnostics(copy_diagnostics),
            **intrinsic_means,
        }
        append_jsonl(step_dir / "step_summary.jsonl", metrics)
        artifacts: dict[str, object] = {"step_dir": step_dir.as_posix()}
        rollout_batch = _concat_rollouts(rollout_batches)
        if rollout_batch is not None:
            artifacts["rollout_batch"] = rollout_batch
            artifacts["token_level_scores"] = reward_batch.tensor
        return OnlineSelfPlayStep(
            step=int(step),
            reward_batch=reward_batch,
            metrics=metrics,
            artifacts=artifacts,
        )


# ─── Internal helpers ────────────────────────────────────────────────


def _stage2_proposer_failure_record(
    *,
    sample_id: str,
    status: FailureReason,
    generation: GenerationResult,
    response_length: int,
    scene_kind: Optional[str] = None,
) -> RewardRecord:
    """Stage-2 variant: same payload as Stage-1 but ``problem_type='gen_vqa'``.

    Stage-1's ``_proposer_failure_record`` hardcodes
    ``problem_type='gen_physics'``; rather than duplicating the body, we
    reuse it and rewrite the field so the Stage-1 helper stays the
    single source of truth for penalty values and metadata shape.
    """

    record = _stage1_proposer_failure_record(
        sample_id=sample_id,
        status=status,
        generation=generation,
        response_length=response_length,
    )
    metadata = dict(record.metadata)
    metadata["stage"] = 2
    if scene_kind:
        metadata["scene_kind"] = scene_kind
    return RewardRecord(
        problem_type="gen_vqa",
        reward=record.reward,
        valid_response_length=record.valid_response_length,
        response_length=record.response_length,
        scene_id=record.scene_id,
        metadata=metadata,
    )


def _repair_metadata(result: Any) -> dict[str, Any]:
    """Flatten executor repair metadata for RewardRecord rows."""

    trace = getattr(result, "repair_trace", None)
    if not isinstance(trace, dict):
        return {"repair_applied": False}
    return {
        "repair_applied": bool(trace.get("applied", False)),
        "repair_tier": str(trace.get("tier", "none")),
        "repair_reward_discount": float(trace.get("reward_discount", 1.0)),
        "repair_action_count": len(trace.get("actions", []) or []),
    }


def _repair_tier(result: Any) -> str:
    """Return executor repair tier, defaulting to ``"none"``."""

    trace = getattr(result, "repair_trace", None)
    if not isinstance(trace, dict) or not bool(trace.get("applied", False)):
        return "none"
    return str(trace.get("tier", "none") or "none")


def _repair_action_count(result: Any) -> int:
    """Return the number of executor repair actions attached to a result."""

    trace = getattr(result, "repair_trace", None)
    if not isinstance(trace, dict):
        return 0
    return len(trace.get("actions", []) or [])


def _is_geometry_repaired(result: Any) -> bool:
    """Whether executor repaired authored geometry, not just metadata."""

    return _repair_tier(result) == "geometry"


def _ok_scene_is_reference_memory(result: Any) -> bool:
    """Return whether an OK+VQA scene is suitable prompt reference memory."""

    if not bool(getattr(result, "ok", False)):
        return False
    tier = _repair_tier(result)
    if tier in {"none", "minor", "manifest"}:
        return True
    if tier == "geometry":
        return _repair_action_count(result) <= _REFERENCE_GEOMETRY_REPAIR_ACTION_CAP
    return False


def _effective_learnability_weight(
    config: PhysEvolveStage2StepBuilderConfig, *, fewshot_active: bool
) -> float:
    """Return the learnability scale for this step.

    During the fewshot-display window, ``learnability_weight_during_fewshot``
    overrides ``learnability_weight`` when set (recipe uses 0.0). Outside that
    window, or when the override is ``None``, the main weight is used.
    """

    if fewshot_active and config.learnability_weight_during_fewshot is not None:
        return float(config.learnability_weight_during_fewshot)
    return float(config.learnability_weight)


def _proposer_group_reward_std(
    reward_records: Sequence[Any],
    record_proposer_uid: Mapping[int, str],
) -> float:
    """Mean intra-GRPO-group reward std over ``gen_vqa`` proposer rows.

    Zero std means every rollout in a group got the same reward (no GRPO
    advantage). Logged to monitor whether copy/edit shaping creates
    within-group variance.
    """

    by_uid: dict[str, list[float]] = defaultdict(list)
    for idx, row in enumerate(reward_records):
        if getattr(row, "problem_type", None) != "gen_vqa":
            continue
        uid = record_proposer_uid.get(idx) or ""
        if not uid:
            meta = getattr(row, "metadata", None) or {}
            uid = str(meta.get("proposer_uid", "") or "")
        if not uid:
            continue
        by_uid[uid].append(float(getattr(row, "reward", 0.0)))
    if not by_uid:
        return 0.0
    stds: list[float] = []
    for rewards in by_uid.values():
        if len(rewards) < 2:
            stds.append(0.0)
            continue
        mean = sum(rewards) / float(len(rewards))
        var = sum((r - mean) ** 2 for r in rewards) / float(len(rewards))
        stds.append(var**0.5)
    return float(sum(stds) / float(len(stds)))


def _batch_duplicate_reward(config: PhysEvolveStage2StepBuilderConfig) -> float:
    """Return a duplicate reward that is worse than common overlap failure."""

    overlap_scaled = (
        float(PROPOSER_PENALTY[FailureReason.FAILED_OVERLAP])
        * float(config.proposer_failure_weight)
    )
    return min(float(config.copy_exact_reward), overlap_scaled - 1e-6)


def _result_payload_families(result: Any) -> tuple[str, ...]:
    """Return unique VQA families actually emitted by the executor."""

    families: list[str] = []
    payloads = getattr(result, "vqa_payloads", None)
    if isinstance(payloads, list):
        for item in payloads:
            if isinstance(item, dict):
                fam = _payload_family(item)
                if fam and fam not in families:
                    families.append(fam)
        if families:
            return tuple(families)
    spec = getattr(result, "vqa_spec", None)
    if isinstance(spec, dict):
        fam = spec.get("family")
        if isinstance(fam, str) and fam and fam not in families:
            families.append(fam)
        raw_families = spec.get("families")
        if isinstance(raw_families, (list, tuple)):
            for value in raw_families:
                fam = str(value)
                if fam and fam not in families:
                    families.append(fam)
    return tuple(families)


def _result_supports_question_family(result: Any, question_family: str) -> bool:
    """Whether the executor emitted the sampled Stage-2 question family."""

    if not question_family:
        return True
    families = _result_payload_families(result)
    if families:
        return question_family in families
    # Back-compat for old single-payload tests/fakes with no family metadata.
    return getattr(result, "vqa_payload", None) is not None


def _stage2_learnability_min_rollouts(
    config: PhysEvolveStage2StepBuilderConfig, observed: int
) -> int:
    """Return the minimum Solver samples needed for an OK-scene reward.

    Stage-2 may select several VQA questions from one scene, but each selected
    question still contributes a full GRPO group. Use the configured
    ``group_size`` as the minimum meaningful sample count, capped by what was
    actually observed after question-level sampling.
    """

    planned = max(1, int(config.group_size))
    if observed > 0:
        return min(planned, observed)
    return planned


def _drop_branch_elision_payloads(
    payloads: list[dict] | None,
) -> list[dict] | None:
    """Drop counterfactual / inverse payloads for the w/o-branch ablation."""

    if not payloads:
        return payloads
    return [
        item
        for item in payloads
        if _payload_family(item) not in BRANCH_ELISION_FAMILIES
    ]


def _payload_tuple_from_item(item: dict) -> tuple[str, list[str], int]:
    """Return the authoritative VQA payload tuple from a vqa_payloads item."""

    question, options, correct_idx = item["payload"]
    return (str(question), list(options), int(correct_idx))


def _payload_family(item: dict) -> str:
    """Return a payload item's question family, or ``""`` when absent."""

    spec = item.get("spec", {})
    if not isinstance(spec, dict):
        return ""
    family = spec.get("family", "")
    return str(family) if family is not None else ""


def _payload_phrasing(item: dict) -> str:
    """Return a payload item's phrasing, or ``""`` when absent."""

    spec = item.get("spec", {})
    if not isinstance(spec, dict):
        return ""
    phrasing = spec.get("phrasing", "")
    return str(phrasing) if phrasing is not None else ""


def _payload_is_variant(item: dict) -> bool:
    """Whether a payload item carries its own colour-variant video."""

    return item.get("sim_state") is not None


def _select_stage2_vqa_payloads(
    payloads: list[dict],
    *,
    preferred_family: str,
    max_questions: int,
    seed_key: str,
) -> list[dict]:
    """Select a representative subset of a scene's VQA payloads.

    The executor still generates and logs the full verifier-grounded list. This
    helper only decides which unique question groups are worth sending to the
    expensive video Solver rollout for this PPO step.
    """

    if not payloads:
        return []
    if max_questions <= 0 or len(payloads) <= max_questions:
        return list(payloads)

    limit = max(1, min(int(max_questions), len(payloads)))
    rng = random.Random(seed_key)
    selected_indices: list[int] = []
    selected_set: set[int] = set()

    def _add(idx: int) -> None:
        if idx not in selected_set and len(selected_indices) < limit:
            selected_indices.append(idx)
            selected_set.add(idx)

    # Keep the sampled family represented when the scene template produced it.
    for want_variant in (False, True):
        for idx, item in enumerate(payloads):
            if _payload_family(item) == preferred_family and (
                _payload_is_variant(item) is want_variant
            ):
                _add(idx)
                break
        if selected_indices:
            break

    # Fallback: preserve at least one baseline question if possible.
    if not selected_indices:
        for idx, item in enumerate(payloads):
            if not _payload_is_variant(item):
                _add(idx)
                break
    if not selected_indices:
        _add(0)

    buckets: dict[tuple[str, str, bool], list[int]] = defaultdict(list)
    for idx, item in enumerate(payloads):
        if idx in selected_set:
            continue
        buckets[
            (_payload_family(item), _payload_phrasing(item), _payload_is_variant(item))
        ].append(idx)
    for values in buckets.values():
        rng.shuffle(values)

    # Round-robin across family / phrasing buckets, preferring baseline videos
    # before variants. A variant is restored below when the cap leaves room.
    while len(selected_indices) < limit:
        active_keys = [key for key, values in buckets.items() if values]
        if not active_keys:
            break
        for key in sorted(active_keys, key=lambda x: (x[2], x[0], x[1])):
            values = buckets[key]
            if values:
                _add(values.pop(0))
            if len(selected_indices) >= limit:
                break

    # Preserve a small amount of colour anti-shortcut signal when the scene had
    # variants and the per-scene cap is not just one question.
    if limit > 1 and not any(_payload_is_variant(payloads[i]) for i in selected_indices):
        variant_indices = [
            idx
            for idx, item in enumerate(payloads)
            if idx not in selected_set and _payload_is_variant(item)
        ]
        if variant_indices:
            rng.shuffle(variant_indices)
            replaced = selected_indices[-1]
            selected_set.remove(replaced)
            selected_indices[-1] = variant_indices[0]
            selected_set.add(variant_indices[0])

    return [payloads[idx] for idx in selected_indices]


def _select_stage2_solver_candidates_for_step(
    candidates: list[tuple[_Stage2ProposerCtx, Any, int, Any, str]],
    *,
    max_groups: int,
) -> list[tuple[_Stage2ProposerCtx, Any, int, Any, str]]:
    """Apply a step-level cap to selected Stage-2 Solver question groups."""

    if not candidates:
        return []
    if max_groups <= 0 or len(candidates) <= max_groups:
        return list(candidates)

    limit = max(1, int(max_groups))
    selected_indices: list[int] = []
    selected_set: set[int] = set()
    seen_proposers: set[int] = set()

    def _add(idx: int) -> None:
        if idx not in selected_set and len(selected_indices) < limit:
            selected_indices.append(idx)
            selected_set.add(idx)

    # First pass: keep one question per OK proposer when the budget allows.
    for idx, candidate in enumerate(candidates):
        proposer_record_idx = int(candidate[2])
        if proposer_record_idx in seen_proposers:
            continue
        seen_proposers.add(proposer_record_idx)
        _add(idx)
        if len(selected_indices) >= limit:
            return [candidates[i] for i in selected_indices]

    # Fill remaining budget by question family round-robin so throwing cannot
    # monopolise every remaining slot in a mixed step.
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, candidate in enumerate(candidates):
        if idx in selected_set:
            continue
        buckets[str(candidate[4])].append(idx)
    while len(selected_indices) < limit:
        active_keys = [key for key, values in buckets.items() if values]
        if not active_keys:
            break
        for key in sorted(active_keys):
            values = buckets[key]
            if values:
                _add(values.pop(0))
            if len(selected_indices) >= limit:
                break

    return [candidates[i] for i in selected_indices]


__all__ = [
    "PhysEvolveStage2StepBuilder",
    "PhysEvolveStage2StepBuilderConfig",
]


# Suppress unused-import lint for the Stage-1 learnability helper —
# it's intentionally imported above so the parallel reference is visible
# and refactors that move it stay in sync.
_ = _stage1_learnability_min_rollouts
_: Callable[..., Any] = AbstractContextManager  # type: ignore[assignment]
_ = GenerationBackend
