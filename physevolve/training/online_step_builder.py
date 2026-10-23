"""Reusable online self-play step builder for training."""

from __future__ import annotations

import random
from collections import Counter, defaultdict, deque
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Optional, Protocol, Sequence

import numpy as np

from physevolve.engine.failure import PROPOSER_PENALTY, FailureReason
from physevolve.executor import run_executor
from physevolve.inference.hf_backend import DEFAULT_MODEL, GenerationResult, HFBackend
from physevolve.inference.parsing import extract_propose_block
from physevolve.prompts.proposer_stage1 import (
    PHYSICS_SKILL_TARGETS,
    build_proposer_stage1_prompt,
)
from physevolve.smoke.io import append_jsonl, write_text
from physevolve.smoke.run_config import SmokeGenerationConfig
from physevolve.training.adaptive_curriculum import (
    DEFAULT_ACC_WINDOW,
    DEFAULT_COLD_START_STEPS,
    DEFAULT_DIFFICULTY_EASY_BELOW,
    DEFAULT_DIFFICULTY_HARD_AT_OR_ABOVE,
    DEFAULT_REFERENCE_BUFFER_SIZE,
    DEFAULT_REFERENCE_K,
    ReferenceBuffer,
    RollingAccTracker,
    adaptive_pick,
    derive_difficulty,
    format_reference_examples,
    pick_weight_snapshot,
)
from physevolve.training.reward_batch import RewardRecord, build_reward_batch
from physevolve.training.online_step import OnlineSelfPlayStep
from physevolve.training.selfplay_batch import (
    build_solver_requests,
    executor_result_record,
    save_video_mp4,
    score_solver_response,
    task_modes_from_strings,
    video_request_record,
)
from physevolve.training.verl_config import PHYSEVOLVE_STAGE1_PROBLEM_TYPES
from physevolve.verification.proposer_reward import (
    DEFAULT_INTRINSIC_COMPONENT_WEIGHTS,
    DEFAULT_INTRINSIC_WEIGHT,
    IntrinsicInputs,
    SceneSignature,
    compute_proposer_reward,
    compute_proposer_reward_with_intrinsic,
    compute_scene_signature,
)


@dataclass(frozen=True, slots=True)
class GenerationSpec:
    """One prompt-shaped generation request fed into a backend batch.

    The PhysEvolve step builder collects all proposer (and later all solver)
    requests for a single PPO step into a flat sequence of these and passes
    them to ``backend.generate_batch(...)`` in one shot so the underlying
    vLLM async server can actually run the rollouts concurrently.
    """

    prompt: str
    video: Any = None
    max_new_tokens: int = 4096
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: Optional[int] = 20
    stop_sequences: tuple[str, ...] = ()
    rollout_metadata: Optional[dict[str, Any]] = None


class GenerationBackend(Protocol):
    """Minimal text/video generation surface for online self-play.

    Backends MUST implement ``generate_with_metadata`` (single-row) and SHOULD
    implement ``generate_batch`` (multi-row) when they can pipeline multiple
    rollouts concurrently. ``call_generate_batch`` below falls back to a
    per-row loop when ``generate_batch`` is unavailable.
    """

    def generate_with_metadata(
        self,
        prompt: str,
        video=None,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        stop_sequences: list[str],
        rollout_metadata: dict[str, object] | None = None,
    ) -> GenerationResult:
        """Generate one completion with metadata."""


def call_generate_batch(
    backend: object,
    specs: Sequence[GenerationSpec],
) -> list[GenerationResult]:
    """Run a batched generation, falling back to per-row when unsupported.

    Real veRL trainer backends override ``generate_batch`` to send all
    ``specs`` through a single ``async_rollout_manager.generate_sequences``
    call. Lightweight fakes (smoke tests, HF) keep working through the
    per-row fallback without any code changes.
    """

    if not specs:
        return []
    method = getattr(backend, "generate_batch", None)
    if method is not None:
        results = list(method(specs))
        if len(results) != len(specs):
            raise RuntimeError(
                f"generate_batch returned {len(results)} results for {len(specs)} specs"
            )
        return results
    return [_call_generate_with_metadata(backend, spec) for spec in specs]


def call_generate_batch_chunked(
    backend: object,
    specs: Sequence[GenerationSpec],
    rollout_batches: list[object],
    chunk_size: int,
) -> list[GenerationResult]:
    """Run a batched generation in fixed-size chunks, collecting each chunk.

    Dispatching every spec in one ``generate_batch`` call lets veRL fan all
    rows out concurrently, but each row's multimodal payload (e.g. a full
    Stage-2 solver video) is preprocessed up front, so a single large batch
    can blow up GPU memory. Splitting the dispatch into chunks bounds how many
    rows -- and therefore how many concurrently materialised video tensors --
    hit ``generate_sequences`` at once.

    Correctness requirement: the trainer backend stores the rollout DataProto
    of the *latest* ``generate_batch`` call in ``_last_rollout`` and the caller
    drains it via :func:`_maybe_collect_rollout`. A naive chunk loop that
    collected only once at the end would let each chunk overwrite the previous
    chunk's ``_last_rollout``, silently dropping all but the final chunk from
    the PPO update. We therefore collect the rollout immediately after *every*
    chunk so all rows reach training in dispatch order.

    ``chunk_size <= 0`` (or a size >= ``len(specs)``) degenerates to a single
    batched call followed by a single collect, matching the un-chunked path.
    """

    if not specs:
        return []
    size = int(chunk_size) if int(chunk_size) > 0 else len(specs)
    results: list[GenerationResult] = []
    for start in range(0, len(specs), size):
        chunk = specs[start : start + size]
        results.extend(call_generate_batch(backend, chunk))
        _maybe_collect_rollout(backend, rollout_batches)
    return results


def _call_generate_with_metadata(
    backend: object,
    spec: GenerationSpec,
) -> GenerationResult:
    """Adapt a single GenerationSpec to the per-row backend protocol."""

    return backend.generate_with_metadata(  # type: ignore[attr-defined]
        spec.prompt,
        video=spec.video,
        max_new_tokens=int(spec.max_new_tokens),
        temperature=float(spec.temperature),
        top_p=float(spec.top_p),
        top_k=None if spec.top_k is None else int(spec.top_k),
        stop_sequences=list(spec.stop_sequences),
        rollout_metadata=dict(spec.rollout_metadata or {}),
    )


BackendFactory = Callable[["PhysEvolveOnlineStepBuilderConfig"], AbstractContextManager[GenerationBackend]]
ExecutorFn = Callable[..., object]


@dataclass(frozen=True, slots=True)
class _SolverCandidate:
    request: Any
    sim_state: Any
    source_proposer_record_idx: int | None = None
    source_sample_id: str = ""
    source_item_idx: int = 0


@dataclass(frozen=True, slots=True)
class _PendingProposer:
    """OK-executor proposer awaiting solver rewards for the OK-branch reward."""

    result: Any
    generation: GenerationResult
    skill: str
    difficulty: str
    sample_id: str


@dataclass(frozen=True, slots=True)
class _ProposerCtx:
    """Per-row proposer context captured before the batched rollout."""

    item_idx: int
    rollout_idx: int
    sample_id: str
    rollout_sample_id: str
    seed: int
    skill: str
    difficulty: str
    proposer_uid: str


@dataclass(frozen=True, slots=True)
class _SolverCtx:
    """Per-row solver context captured before the batched rollout."""

    candidate_idx: int
    candidate: _SolverCandidate
    solver_rollout_idx: int
    solver_uid: str


@dataclass(frozen=True, slots=True)
class PhysEvolveOnlineStepBuilderConfig:
    """Configuration for one or more online PhysEvolve self-play steps."""

    model: str = DEFAULT_MODEL
    scenes_per_batch: int = 2
    task_modes: tuple[str, ...] = ("forward", "inverse", "parameter")
    seed_start: int = 7001
    stage: int = 1
    group_size: int = 1
    sandbox_timeout_s: float = 5.0
    sim_timeout_s: float = 30.0
    replay_timeout_s: float = 15.0
    response_length: int = 4096
    executor_workers: int = 1
    generation: SmokeGenerationConfig = SmokeGenerationConfig(
        max_new_tokens=4096,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        stop_sequences=("</propose>", "</answer>"),
    )
    # Skill-conditioned Proposer curriculum.
    skill_targets: tuple[str, ...] = PHYSICS_SKILL_TARGETS
    skill_seed: int = 0
    # Sample eight skills; derive each skill's difficulty from rolling accuracy.
    acc_tracker_window: int = DEFAULT_ACC_WINDOW
    reference_buffer_size: int = DEFAULT_REFERENCE_BUFFER_SIZE
    reference_examples_per_prompt: int = DEFAULT_REFERENCE_K
    cold_start_uniform_steps: int = DEFAULT_COLD_START_STEPS
    # Difficulty escalation thresholds driving :func:`derive_difficulty`.
    # Lowering ``difficulty_hard_at_or_above`` escalates sooner; raising
    # ``difficulty_easy_below`` keeps the Solver in the easier band
    # longer. Defaults are deliberately wide to give the medium band
    # ample headroom (the "learnable" zone).
    difficulty_easy_below: float = DEFAULT_DIFFICULTY_EASY_BELOW
    difficulty_hard_at_or_above: float = DEFAULT_DIFFICULTY_HARD_AT_OR_ABOVE
    # Intrinsic shaping supplements the learnability reward.
    intrinsic_enabled: bool = True
    intrinsic_weight: float = DEFAULT_INTRINSIC_WEIGHT
    intrinsic_component_weights: tuple[float, float, float, float, float] = (
        DEFAULT_INTRINSIC_COMPONENT_WEIGHTS
    )
    novelty_buffer_size: int = 64
    # Save up to this many OK-scene videos as MP4 every ``video_save_freq``
    # steps. 0 (default) disables video saving entirely.
    video_save_freq: int = 0
    video_save_max_per_step: int = 4

    def __post_init__(self) -> None:
        if self.scenes_per_batch <= 0:
            raise ValueError("scenes_per_batch must be positive")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.response_length <= 0:
            raise ValueError("response_length must be positive")
        if self.executor_workers <= 0:
            raise ValueError("executor_workers must be positive")
        if len(self.skill_targets) == 0:
            raise ValueError("skill_targets must be non-empty")
        if self.acc_tracker_window <= 0:
            raise ValueError("acc_tracker_window must be positive")
        if self.reference_buffer_size <= 0:
            raise ValueError("reference_buffer_size must be positive")
        if self.reference_examples_per_prompt < 0:
            raise ValueError(
                "reference_examples_per_prompt must be non-negative"
            )
        if self.cold_start_uniform_steps < 0:
            raise ValueError("cold_start_uniform_steps must be non-negative")
        if not (0.0 <= self.difficulty_easy_below < self.difficulty_hard_at_or_above <= 1.0):
            raise ValueError(
                "require 0.0 <= difficulty_easy_below < "
                "difficulty_hard_at_or_above <= 1.0; got "
                f"easy_below={self.difficulty_easy_below}, "
                f"hard_at_or_above={self.difficulty_hard_at_or_above}"
            )
        if self.intrinsic_weight < 0:
            raise ValueError("intrinsic_weight must be non-negative")
        if len(self.intrinsic_component_weights) != 5:
            raise ValueError(
                "intrinsic_component_weights must have 5 entries "
                "(motion, interaction, complexity, diversity, parameter)"
            )
        if any(float(w) < 0 for w in self.intrinsic_component_weights):
            raise ValueError("intrinsic_component_weights must be non-negative")
        if self.novelty_buffer_size <= 0:
            raise ValueError("novelty_buffer_size must be positive")
        task_modes_from_strings(self.task_modes)


class PhysEvolveOnlineStepBuilder:
    """Build real online self-play batches for the training adapter."""

    def __init__(
        self,
        config: PhysEvolveOnlineStepBuilderConfig,
        *,
        run_dir: Path,
        backend_factory: BackendFactory | None = None,
        executor_fn: ExecutorFn | None = None,
    ) -> None:
        self.config = config
        self.run_dir = Path(run_dir)
        self.backend_factory = backend_factory or _default_backend_factory
        self.executor_fn = executor_fn or run_executor
        self.task_modes = task_modes_from_strings(config.task_modes)
        self._solver_candidate_pool: list[_SolverCandidate] = []
        self._solver_pool_cursor = 0
        self._novelty_buffer: deque[SceneSignature] = deque(
            maxlen=int(config.novelty_buffer_size)
        )
        # cross-step adaptive-difficulty state. Both objects are
        # owned by the builder (single Ray-actor instance per run) so we
        # do not need any locking; updates happen serially in Phase 6.
        self._acc_tracker = RollingAccTracker(
            window_size=int(config.acc_tracker_window)
        )
        self._reference_buffer = ReferenceBuffer(
            maxlen=int(config.reference_buffer_size)
        )

    def _pick_skill_and_difficulty(
        self, step: int, item_idx: int
    ) -> tuple[str, str]:
        """Pick a ``skill`` (adaptive) and derive its ``difficulty`` band.

        difficulty is no longer a separate sampling axis. We
        pick **one skill** weighted by per-skill learnability (8-way
        adaptive pick), then deterministically map the skill's rolling
        Solver accuracy to a difficulty band via
        :func:`derive_difficulty`. The same ``(skill, difficulty)``
        return shape is preserved so the rest of the builder is
        unchanged.

        All ``group_size`` rollouts inside a single GRPO item share the
        same ``(skill, difficulty)`` so the relative-advantage signal
        stays meaningful; the picked pair varies across items inside a
        batch and across steps.
        """

        seed = f"{int(step)}:{int(item_idx)}:{int(self.config.skill_seed)}"
        rng = random.Random(seed)
        skill = adaptive_pick(
            rng,
            self._acc_tracker,
            skills=tuple(self.config.skill_targets),
            step=int(step),
            cold_start_uniform_steps=int(self.config.cold_start_uniform_steps),
        )
        # ``None`` if the bucket is empty so :func:`derive_difficulty`
        # can apply its cold-start rule (returns ``"easy"``); otherwise
        # pass the rolling mean so the threshold logic runs.
        bucket_count = self._acc_tracker.count(skill)
        mean_acc: Optional[float]
        if bucket_count == 0:
            mean_acc = None
        else:
            mean_acc = self._acc_tracker.mean(skill)
        difficulty = derive_difficulty(
            mean_acc,
            easy_below=float(self.config.difficulty_easy_below),
            hard_at_or_above=float(self.config.difficulty_hard_at_or_above),
        )
        return skill, difficulty

    def _pick_skill(self, step: int, item_idx: int) -> str:
        """Deprecated thin wrapper around :meth:`_pick_skill_and_difficulty`.

        Kept so external smoke / archive code that monkey-patches
        ``_pick_skill`` keeps importing the symbol. New callers should
        use :meth:`_pick_skill_and_difficulty` directly because they need
        the difficulty for the prompt builder and tracker update.
        """

        return self._pick_skill_and_difficulty(step, item_idx)[0]

    def __call__(self, step: int):
        """Build one online self-play step and return an `OnlineSelfPlayStep`."""

        step_dir = self.run_dir / "steps" / f"step_{int(step):06d}"

        reward_records: list[RewardRecord] = []
        failure_counts: Counter[str] = Counter()
        physevolve_problem_types_seen: set[str] = {"gen_physics"}
        unsupported_problem_types_seen: set[str] = set()
        solver_reward_by_task: dict[str, list[float]] = defaultdict(list)
        # Naive-baseline dual-track collectors (Naive diagnostics):
        # exact-match success rate + summary-statistic collapse, per task.
        exact_match_by_task: dict[str, list[bool]] = defaultdict(list)
        collapse_by_task: dict[str, list[str | None]] = defaultdict(list)
        solver_parse_ok = 0
        solver_schema_ok = 0
        solver_requests = 0
        solver_video_requests = 0
        executor_calls = 0
        executor_ok = 0
        ok_scene_ids: set[str] = set()
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
        pending_proposers: dict[int, _PendingProposer] = {}
        proposer_solver_rewards: dict[int, list[float]] = defaultdict(list)
        current_solver_candidates: list[_SolverCandidate] = []
        skill_counts: Counter[str] = Counter()
        snapshotted_skills: set[str] = set()
        # difficulty is no longer a sampling axis, so we
        # only record the *latest* difficulty assigned per skill in
        # the current step (useful for confirming escalation kicked
        # in at the expected rolling-acc thresholds).
        latest_difficulty_per_skill: dict[str, str] = {}
        cold_start_active = int(step) <= int(self.config.cold_start_uniform_steps)

        with self.backend_factory(self.config) as backend:
            # ── Phase 1a: prepare per-row proposer context + prompts ─────
            proposer_ctxs: list[_ProposerCtx] = []
            proposer_specs: list[GenerationSpec] = []
            for item_idx in range(self.config.scenes_per_batch):
                sample_id = f"step{int(step):06d}_sample{item_idx + 1:04d}"
                base_seed = self.config.seed_start + (int(step) - 1) * self.config.scenes_per_batch + item_idx
                proposer_uid = f"{sample_id}:proposer"
                skill, difficulty = self._pick_skill_and_difficulty(step, item_idx)
                skill_counts[skill] += 1
                latest_difficulty_per_skill[skill] = difficulty
                # Sample reference examples per-item with a deterministic
                # RNG so two GRPO rollouts inside the same item see the
                # same references (variance-reduction friendly).
                reference_rng = random.Random(
                    f"{int(step)}:{item_idx}:ref:{int(self.config.skill_seed)}"
                )
                reference_entries = self._reference_buffer.sample(
                    reference_rng,
                    skill,
                    k=int(self.config.reference_examples_per_prompt),
                )
                reference_strings = format_reference_examples(reference_entries)
                proposer_prompt = build_proposer_stage1_prompt(
                    skill,
                    difficulty_level=difficulty,
                    reference_examples=reference_strings,
                )
                if skill not in snapshotted_skills:
                    write_text(
                        step_dir / "prompt_snapshots" / f"proposer__{skill}.txt",
                        proposer_prompt,
                    )
                    snapshotted_skills.add(skill)
                for rollout_idx in range(self.config.group_size):
                    rollout_sample_id = f"{sample_id}_r{rollout_idx:02d}"
                    seed = base_seed + rollout_idx
                    proposer_ctxs.append(
                        _ProposerCtx(
                            item_idx=item_idx,
                            rollout_idx=rollout_idx,
                            sample_id=sample_id,
                            rollout_sample_id=rollout_sample_id,
                            seed=seed,
                            skill=skill,
                            difficulty=difficulty,
                            proposer_uid=proposer_uid,
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
                                "problem_type": "gen_physics",
                                "scene_id": None,
                                "skill_target": skill,
                                "difficulty_level": difficulty,
                            },
                        )
                    )

            # ── Phase 1b: one batched proposer rollout ───────────────────
            rollout_start = perf_counter()
            proposer_results = call_generate_batch(backend, proposer_specs)
            proposer_rollout_s += perf_counter() - rollout_start
            _maybe_collect_rollout(backend, rollout_batches)

            # ── Phase 2: per-row parse + executor + bookkeeping ──────────
            for ctx, proposer_generation in zip(proposer_ctxs, proposer_results, strict=True):
                generation_token_count += _generated_token_count(proposer_generation)
                proposer_text = proposer_generation.text
                proposer_record_idx = len(reward_records)
                reward_records.append(
                    RewardRecord(
                        problem_type="gen_physics",
                        reward=0.0,
                        valid_response_length=_valid_generation_length(
                            proposer_generation,
                            self.config.response_length,
                        ),
                        response_length=self.config.response_length,
                        metadata={
                            "skill_target": ctx.skill,
                            "difficulty_level": ctx.difficulty,
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
                        "problem_type": "gen_physics",
                        "skill_target": ctx.skill,
                        "difficulty_level": ctx.difficulty,
                        "raw_text": proposer_text,
                        "generation_metadata": proposer_generation.metadata,
                    },
                )

                code = extract_propose_block(proposer_text)
                if code is None:
                    status = FailureReason.FAILED_SYNTAX
                    failure_counts[status.value] += 1
                    reward_records[proposer_record_idx] = _proposer_failure_record(
                        sample_id=ctx.rollout_sample_id,
                        status=status,
                        generation=proposer_generation,
                        response_length=self.config.response_length,
                        skill=ctx.skill,
                        difficulty=ctx.difficulty,
                    )
                    continue

                write_text(step_dir / "code" / f"{ctx.rollout_sample_id}.py", code.rstrip() + "\n")
                executor_calls += 1
                executor_start = perf_counter()
                result = self.executor_fn(
                    code,
                    stage=self.config.stage,
                    seed=ctx.seed,
                    sandbox_timeout_s=self.config.sandbox_timeout_s,
                    sim_timeout_s=self.config.sim_timeout_s,
                )
                executor_s += perf_counter() - executor_start
                result_row = executor_result_record(result)
                append_jsonl(step_dir / "executor_results.jsonl", {"sample_id": ctx.rollout_sample_id, **result_row})
                failure_counts[str(result_row["status"])] += 1
                executor_ok += int(bool(result_row["ok"]))
                if bool(result_row["ok"]) and result_row["scene_id"]:
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
                            / f"{ctx.rollout_sample_id}_{ctx.skill}.mp4"
                        )
                        save_video_mp4(result.sim_state.video_frames, video_path)
                        _video_save_count += 1

                if getattr(result, "ok", False) and getattr(result, "sim_state", None) is not None:
                    requests = build_solver_requests(
                        result,
                        task_modes=self.task_modes,
                        sample_id=ctx.rollout_sample_id,
                    )
                    for request in requests:
                        physevolve_problem_types_seen.add(request.problem_type)
                        candidate = _SolverCandidate(
                            request=request,
                            sim_state=result.sim_state,
                            source_proposer_record_idx=proposer_record_idx,
                            source_sample_id=ctx.rollout_sample_id,
                            source_item_idx=ctx.item_idx,
                        )
                        current_solver_candidates.append(candidate)
                    pending_proposers[proposer_record_idx] = _PendingProposer(
                        result=result,
                        generation=proposer_generation,
                        skill=ctx.skill,
                        difficulty=ctx.difficulty,
                        sample_id=ctx.rollout_sample_id,
                    )
                else:
                    proposer_reward = compute_proposer_reward(result, solver_rewards=[])
                    reward_records[proposer_record_idx] = (
                        RewardRecord(
                            problem_type="gen_physics",
                            reward=float(proposer_reward),
                            valid_response_length=_valid_generation_length(
                                proposer_generation,
                                self.config.response_length,
                            ),
                            response_length=self.config.response_length,
                            scene_id=getattr(result, "scene_id", ""),
                            metadata={
                                "n_solver_rewards": 0,
                                "skill_target": ctx.skill,
                                "difficulty_level": ctx.difficulty,
                            },
                        )
                    )

            # ── Phase 3: solver candidate selection + pool maintenance ───
            planned_solver_groups = self.config.scenes_per_batch * len(self.task_modes)
            selected_solver_candidates = self._select_solver_candidates(
                planned_solver_groups,
                current_solver_candidates,
            )
            self._add_solver_candidates_to_pool(current_solver_candidates)

            # ── Phase 4a: prepare per-row solver context + prompts ───────
            solver_ctxs: list[_SolverCtx] = []
            solver_specs: list[GenerationSpec] = []
            for candidate_idx, candidate in enumerate(selected_solver_candidates):
                request = candidate.request
                physevolve_problem_types_seen.add(request.problem_type)
                solver_requests += self.config.group_size
                write_text(
                    step_dir / "prompt_snapshots" / f"{request.request_id}.txt",
                    request.prompt,
                )
                request_row = video_request_record(request)
                solver_video_requests += self.config.group_size * int(bool(request_row["video_payload_present"]))
                append_jsonl(step_dir / "solver_requests.jsonl", request_row)

                solver_uid = f"{request.request_id}:solver:{candidate_idx}"
                for solver_rollout_idx in range(self.config.group_size):
                    solver_ctxs.append(
                        _SolverCtx(
                            candidate_idx=candidate_idx,
                            candidate=candidate,
                            solver_rollout_idx=solver_rollout_idx,
                            solver_uid=solver_uid,
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
                                "index": candidate.source_item_idx,
                                "role": "solver",
                                "problem_type": request.problem_type,
                                "scene_id": request.scene_id,
                            },
                        )
                    )

            # ── Phase 4b: one batched solver rollout ─────────────────────
            rollout_start = perf_counter()
            solver_results = call_generate_batch(backend, solver_specs)
            solver_rollout_s += perf_counter() - rollout_start
            if solver_specs:
                _maybe_collect_rollout(backend, rollout_batches)

            # ── Phase 5: per-row scoring + reward records ────────────────
            for ctx, solver_generation in zip(solver_ctxs, solver_results, strict=True):
                candidate = ctx.candidate
                request = candidate.request
                generation_token_count += _generated_token_count(solver_generation)
                solver_text = solver_generation.text
                append_jsonl(
                    step_dir / "solver_outputs.jsonl",
                    {
                        "sample_id": candidate.source_sample_id,
                        "request_id": request.request_id,
                        "solver_rollout_idx": ctx.solver_rollout_idx,
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
                    sim_state=candidate.sim_state,
                    replay_timeout_s=self.config.replay_timeout_s,
                )
                solver_parse_ok += int(score.answer_parse_ok)
                solver_schema_ok += int(score.schema_ok)
                if candidate.source_proposer_record_idx is not None:
                    proposer_solver_rewards[candidate.source_proposer_record_idx].append(score.reward)
                solver_reward_by_task[request.problem_type].append(score.reward)
                # Naive dual track: exact-match + collapse, alongside soft reward.
                exact_match_by_task[request.problem_type].append(bool(score.exact_match))
                collapse_by_task[request.problem_type].append(score.collapse_type)
                reward_records.append(
                    RewardRecord(
                        problem_type=request.problem_type,
                        reward=score.reward,
                        valid_response_length=_valid_generation_length(
                            solver_generation,
                            self.config.response_length,
                        ),
                        response_length=self.config.response_length,
                        scene_id=request.scene_id,
                        metadata={
                            "request_id": request.request_id,
                            "schema_ok": score.schema_ok,
                            "answer_parse_ok": score.answer_parse_ok,
                        },
                    )
                )
                append_jsonl(
                    step_dir / "reward_records.jsonl",
                    {
                        "sample_id": candidate.source_sample_id,
                        "request_id": request.request_id,
                        "solver_rollout_idx": ctx.solver_rollout_idx,
                        "scene_id": request.scene_id,
                        "problem_type": request.problem_type,
                        "reward": score.reward,
                        "schema_ok": score.schema_ok,
                        "answer_parse_ok": score.answer_parse_ok,
                        "parsed_answer": score.parsed_answer,
                    },
                )

            intrinsic_components_log: dict[str, list[float]] = defaultdict(list)
            reward_by_skill: dict[str, list[float]] = defaultdict(list)
            history_snapshot: tuple[SceneSignature, ...] = tuple(self._novelty_buffer)
            scored_signatures: list[SceneSignature] = []
            # Accuracy diagnostic: count OK scenes whose mean Solver
            # accuracy falls inside the "actually learnable" band. We
            # track this *before* the per-scene reward shaping so it
            # reflects raw Solver competence, not intrinsic mass.
            learnable_band_count = 0
            scored_ok_count = 0
            for proposer_record_idx in sorted(pending_proposers):
                pending = pending_proposers[proposer_record_idx]
                result = pending.result
                proposer_generation = pending.generation
                skill = pending.skill
                difficulty = pending.difficulty
                rewards = proposer_solver_rewards[proposer_record_idx]

                # update per-skill rolling acc tracker and
                # the reference buffer for *validated* OK scenes only
                # (enough solver rollouts to make mean_acc meaningful).
                # Reward shaping is reused from Stage-1; only the
                # cross-step state is new.
                min_rollouts = _learnability_min_rollouts(self.config, len(rewards))
                mean_acc: Optional[float] = None
                if len(rewards) >= min_rollouts and rewards:
                    mean_acc = sum(float(r) for r in rewards) / float(len(rewards))
                    self._acc_tracker.update(skill, mean_acc)
                    scored_ok_count += 1
                    if 0.15 <= mean_acc <= 0.85:
                        learnable_band_count += 1
                    if getattr(result, "sim_state", None) is not None:
                        self._reference_buffer.record(
                            skill,
                            result.sim_state,
                            scene_id=str(getattr(result, "scene_id", "") or ""),
                        )

                intrinsic_inputs: Optional[IntrinsicInputs] = None
                signature: Optional[SceneSignature] = None
                if (
                    self.config.intrinsic_enabled
                    and getattr(result, "sim_state", None) is not None
                ):
                    signature = compute_scene_signature(result.sim_state)
                    history_for_scene = history_snapshot + tuple(scored_signatures)
                    intrinsic_inputs = IntrinsicInputs(
                        sim_state=result.sim_state,
                        signature=signature,
                        history=history_for_scene,
                        weight=float(self.config.intrinsic_weight),
                        component_weights=tuple(self.config.intrinsic_component_weights),
                    )

                if intrinsic_inputs is not None:
                    proposer_reward, breakdown = compute_proposer_reward_with_intrinsic(
                        result,
                        solver_rewards=rewards,
                        intrinsic=intrinsic_inputs,
                        min_rollouts_for_learnability=min_rollouts,
                    )
                else:
                    proposer_reward = compute_proposer_reward(
                        result,
                        solver_rewards=rewards,
                        min_rollouts_for_learnability=min_rollouts,
                    )
                    breakdown = None

                metadata: dict[str, Any] = {
                    "n_solver_rewards": len(rewards),
                    "skill_target": skill,
                    "difficulty_level": difficulty,
                }
                if mean_acc is not None:
                    metadata["mean_solver_acc"] = float(mean_acc)
                if breakdown is not None:
                    metadata["intrinsic_components"] = dict(breakdown.components)
                    metadata["intrinsic_weighted_sum"] = breakdown.weighted_sum
                    metadata["intrinsic_total"] = breakdown.total
                    for name, value in breakdown.components.items():
                        intrinsic_components_log[name].append(float(value))
                reward_by_skill[skill].append(float(proposer_reward))

                reward_records[proposer_record_idx] = (
                    RewardRecord(
                        problem_type="gen_physics",
                        reward=float(proposer_reward),
                        valid_response_length=_valid_generation_length(
                            proposer_generation,
                            self.config.response_length,
                        ),
                        response_length=self.config.response_length,
                        scene_id=getattr(result, "scene_id", ""),
                        metadata=metadata,
                    )
                )
                proposer_row: dict[str, Any] = {
                    "sample_id": pending.sample_id,
                    "scene_id": getattr(result, "scene_id", ""),
                    "problem_type": "gen_physics",
                    "skill_target": skill,
                    "difficulty_level": difficulty,
                    "reward": float(proposer_reward),
                    "n_solver_rewards": len(rewards),
                }
                if mean_acc is not None:
                    proposer_row["mean_solver_acc"] = float(mean_acc)
                if breakdown is not None:
                    for name, value in breakdown.components.items():
                        proposer_row[f"intrinsic/{name}"] = float(value)
                    proposer_row["intrinsic/weighted_sum"] = float(breakdown.weighted_sum)
                    proposer_row["intrinsic/total"] = float(breakdown.total)
                append_jsonl(step_dir / "reward_records.jsonl", proposer_row)
                if signature is not None:
                    scored_signatures.append(signature)

            for signature in scored_signatures:
                self._novelty_buffer.append(signature)

        reward_batch = build_reward_batch(reward_records, response_length=self.config.response_length)
        mean_reward_by_task = {
            task: float(np.mean(values)) for task, values in sorted(solver_reward_by_task.items()) if values
        }
        # Track dense reward, exact-match accuracy, and collapse categories
        # separately for each Naive task mode.
        exact_match_accuracy_by_task = {
            task: float(np.mean([bool(v) for v in values]))
            for task, values in sorted(exact_match_by_task.items())
            if values
        }
        collapse_distribution_by_task: dict[str, dict[str, int]] = {}
        for task, values in sorted(collapse_by_task.items()):
            if not values:
                continue
            counts: Counter[str] = Counter()
            for label in values:
                if label is not None:
                    counts[str(label)] += 1
            collapse_distribution_by_task[task] = dict(sorted(counts.items()))
        unsupported_problem_types_seen |= set(physevolve_problem_types_seen) - set(PHYSEVOLVE_STAGE1_PROBLEM_TYPES)
        intrinsic_means = {
            f"intrinsic/{name}_mean": float(np.mean(values))
            for name, values in sorted(intrinsic_components_log.items())
            if values
        }
        proposer_reward_by_skill = {
            skill: float(np.mean(values))
            for skill, values in sorted(reward_by_skill.items())
            if values
        }
        # Adaptive-curriculum diagnostics. The sampler axis is
        # now skill-only; difficulty is a derived knob and surfaces as a
        # per-skill assignment snapshot.
        acc_by_skill = self._acc_tracker.snapshot()
        pick_weight_by_skill = pick_weight_snapshot(
            self._acc_tracker,
            skills=tuple(self.config.skill_targets),
        )
        reference_buffer_sizes = self._reference_buffer.snapshot_sizes()
        effective_learnable_fraction = (
            float(learnable_band_count) / float(scored_ok_count)
            if scored_ok_count > 0
            else 0.0
        )
        difficulty_assignments_by_skill = dict(
            sorted(latest_difficulty_per_skill.items())
        )
        metrics = {
            "online_proposer_generation_count": self.config.scenes_per_batch * self.config.group_size,
            "executor_call_count": executor_calls,
            "n_executor_ok": executor_ok,
            "n_executor_failed": max(0, executor_calls - executor_ok),
            "unique_scene_ratio": _rate(len(ok_scene_ids), executor_ok),
            "failure_reason_distribution": dict(sorted(failure_counts.items())),
            "n_solver_requests": solver_requests,
            "planned_solver_groups": planned_solver_groups,
            "selected_solver_groups": len(selected_solver_candidates),
            "solver_backfill_shortfall": max(0, planned_solver_groups - len(selected_solver_candidates)),
            "solver_candidate_pool_size": len(self._solver_candidate_pool),
            "solver_video_request_count": solver_video_requests,
            "answer_parse_rate": _rate(solver_parse_ok, solver_requests),
            "schema_ok_rate": _rate(solver_schema_ok, solver_requests),
            "mean_reward_by_task": mean_reward_by_task,
            "exact_match_accuracy_by_task": exact_match_accuracy_by_task,
            "collapse_distribution_by_task": collapse_distribution_by_task,
            # The single-world Naive path creates no interventions or CF items.
            "n_intervention_operations": 0,
            "n_counterfactual_items": 0,
            "naive_interface": "single_world_no_intervention",
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
            "expected_problem_types": list(PHYSEVOLVE_STAGE1_PROBLEM_TYPES),
            "skill_target_counts": dict(sorted(skill_counts.items())),
            "proposer_reward_by_skill": proposer_reward_by_skill,
            "acc_by_skill": acc_by_skill,
            "pick_weight_by_skill": pick_weight_by_skill,
            "difficulty_assignments_by_skill": difficulty_assignments_by_skill,
            "reference_buffer_sizes": reference_buffer_sizes,
            "effective_learnable_fraction": float(effective_learnable_fraction),
            "scored_ok_proposer_count": int(scored_ok_count),
            "cold_start_active": bool(cold_start_active),
            "novelty_buffer_size": len(self._novelty_buffer),
            "novelty_buffer_capacity": int(self.config.novelty_buffer_size),
            "intrinsic_enabled": bool(self.config.intrinsic_enabled),
            "intrinsic_weight": float(self.config.intrinsic_weight),
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

    def _select_solver_candidates(
        self,
        planned_groups: int,
        current_candidates: list[_SolverCandidate],
    ) -> list[_SolverCandidate]:
        """Select current candidates first, then prior pool entries.

        Current candidates are added to the persistent pool only after this
        selection, so a just-created task cannot be duplicated in the same PPO
        step when the reference pool is still short.
        """

        if planned_groups <= 0:
            return []
        selected = list(current_candidates[:planned_groups])
        remaining = planned_groups - len(selected)
        if remaining <= 0 or not self._solver_candidate_pool:
            return selected

        pool_size = len(self._solver_candidate_pool)
        for _ in range(remaining):
            selected.append(self._solver_candidate_pool[self._solver_pool_cursor % pool_size])
            self._solver_pool_cursor += 1
        return selected

    def _add_solver_candidates_to_pool(self, candidates: list[_SolverCandidate]) -> None:
        for candidate in candidates:
            self._solver_candidate_pool.append(
                _SolverCandidate(
                    request=candidate.request,
                    sim_state=candidate.sim_state,
                    source_proposer_record_idx=None,
                    source_sample_id=candidate.source_sample_id,
                    source_item_idx=candidate.source_item_idx,
                )
            )


def _default_backend_factory(
    config: PhysEvolveOnlineStepBuilderConfig,
) -> AbstractContextManager[GenerationBackend]:
    return HFBackend(config.model)


def _proposer_failure_record(
    *,
    sample_id: str,
    status: FailureReason,
    generation: GenerationResult,
    response_length: int,
    skill: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> RewardRecord:
    metadata: dict[str, Any] = {"sample_id": sample_id, "status": status.value}
    if skill is not None:
        metadata["skill_target"] = skill
    if difficulty is not None:
        metadata["difficulty_level"] = difficulty
    return RewardRecord(
        problem_type="gen_physics",
        reward=float(PROPOSER_PENALTY[status]),
        valid_response_length=_valid_generation_length(generation, response_length),
        response_length=response_length,
        scene_id="",
        metadata=metadata,
    )


def _valid_generation_length(generation: GenerationResult, fallback: int) -> int:
    value = generation.metadata.get("generated_token_count") if generation.metadata else None
    try:
        length = int(value)
    except (TypeError, ValueError):
        length = int(fallback)
    return max(1, min(length, int(fallback)))


def _generated_token_count(generation: GenerationResult) -> int:
    """Return generated token count when the backend reports it."""

    value = generation.metadata.get("generated_token_count") if generation.metadata else None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _maybe_collect_rollout(backend: object, rollouts: list[object]) -> None:
    consume = getattr(backend, "consume_last_rollout", None)
    if consume is None:
        return
    rollout = consume()
    if rollout is not None:
        rollouts.append(rollout)


def _concat_rollouts(rollouts: list[object]) -> object | None:
    if not rollouts:
        return None
    try:
        from verl import DataProto
    except Exception:
        return None
    for rollout in rollouts:
        meta_info = getattr(rollout, "meta_info", None)
        if isinstance(meta_info, dict):
            # Per-call timings are not stable metadata; keeping them here
            # makes DataProto.concat reject otherwise aligned rollout rows.
            meta_info.pop("timing", None)
    return DataProto.concat(rollouts)


def _rate(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def _learnability_min_rollouts(config: PhysEvolveOnlineStepBuilderConfig, observed: int) -> int:
    """Derive a non-silent learnability threshold from the active solver plan."""

    planned = max(1, len(config.task_modes) * int(config.group_size))
    if observed > 0:
        return min(planned, observed)
    return planned


__all__ = [
    "BackendFactory",
    "GenerationBackend",
    "GenerationSpec",
    "PhysEvolveOnlineStepBuilder",
    "PhysEvolveOnlineStepBuilderConfig",
    "call_generate_batch",
    "call_generate_batch_chunked",
]
