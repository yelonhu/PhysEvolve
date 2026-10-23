"""Online self-play batch construction primitives for training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from physevolve.engine.failure import FailureReason
from physevolve.executor.result import ExecutorResult
from physevolve.integrations.video_bridge import VideoBundle, video_bundle_from_observation
from physevolve.tasks.router import SolverObservation, TaskMode, assemble_observation, parse_solver_output
from physevolve.verification.dense_scorer import score_forward, score_inverse, score_parameter
from physevolve.verification.naive_diagnostics import naive_diagnostics
from physevolve.verification.vqa_scorer import score_vqa


_MODE_TO_PROBLEM_TYPE: dict[TaskMode, str] = {
    TaskMode.FORWARD: "pred_forward",
    TaskMode.INVERSE: "pred_inverse",
    TaskMode.PARAMETER: "pred_parameter",
    TaskMode.VQA: "pred_vqa",
}


def mode_to_problem_type(mode: TaskMode) -> str:
    """Map a PhysEvolve :class:`TaskMode` to its Stage-1 ``problem_type`` string."""

    return _MODE_TO_PROBLEM_TYPE[mode]


@dataclass(frozen=True, slots=True)
class SolverRequest:
    """One Solver prompt/video request derived from an OK Proposer scene."""

    request_id: str
    scene_id: str
    mode: TaskMode
    problem_type: str
    prompt: str
    observation: SolverObservation
    video: VideoBundle


@dataclass(frozen=True, slots=True)
class SolverScore:
    """Parsed Solver output and dense reward for one request.

    ``exact_match`` / ``collapse_type`` carry the naive-baseline diagnostic
    tracks (Naive diagnostics) alongside the soft ``reward``:

    * ``exact_match`` — strict all-bodies-correct boolean (the success rate
      reported separately from dense reward). ``None`` for the VQA path, which has
      no soft track.
    * ``collapse_type`` — when ``exact_match`` is ``False``, one of the
      :data:`physevolve.verification.naive_diagnostics.COLLAPSE_*` labels
      if the wrong prediction falls on a summary statistic, else ``None``.

    Both default to ``None`` so the Stage-2 VQA path is unaffected.
    """

    request_id: str
    scene_id: str
    problem_type: str
    raw_text: str
    parsed_answer: dict[str, Any] | None
    reward: float
    answer_parse_ok: bool
    schema_ok: bool
    exact_match: bool | None = None
    collapse_type: str | None = None


def build_solver_requests(
    executor_result: ExecutorResult,
    *,
    task_modes: Sequence[TaskMode],
    sample_id: str,
    camera_idx: int = 0,
) -> tuple[SolverRequest, ...]:
    """Build per-mode Solver requests from an OK executor result.

    Stage-2 ``TaskMode.VQA`` is included in ``task_modes`` only when the
    executor result actually carries a ``vqa_payload``; if a VQA mode is
    requested on a result without a payload, that mode is silently
    skipped (the pipeline already returned ``FAILED_VQA`` upstream and
    the Proposer was penalised). Mixing VQA with the dense modes is
    allowed but Stage-2 builders typically only pass ``(VQA,)``.
    """

    if not executor_result.ok or executor_result.sim_state is None:
        return ()
    requests: list[SolverRequest] = []
    for mode in task_modes:
        if mode is TaskMode.VQA:
            # Scene-template multi-GT path: one request per derived question.
            payloads = _vqa_payload_list(executor_result)
            if not payloads:
                # Stage-1 result accidentally requested VQA: skip cleanly.
                continue
            multi = len(payloads) > 1 or executor_result.vqa_payloads is not None
            for q_idx, (payload, spec, item_sim_state) in enumerate(payloads):
                # Colour-permuted variants carry their OWN sim_state (a
                # re-rendered video with recoloured balls); baseline questions
                # leave it None and reuse the scene's single sim_state.
                question_sim_state = (
                    item_sim_state
                    if item_sim_state is not None
                    else executor_result.sim_state
                )
                obs = assemble_observation(
                    question_sim_state,
                    mode,
                    vqa_payload=payload,
                )
                bundle = video_bundle_from_observation(obs)
                request_id = (
                    f"{sample_id}_{mode.value}_q{q_idx:02d}" if multi
                    else f"{sample_id}_{mode.value}"
                )
                _ = camera_idx  # observation builders select the canonical camera.
                requests.append(
                    SolverRequest(
                        request_id=request_id,
                        scene_id=obs.scene_id,
                        mode=mode,
                        problem_type=mode_to_problem_type(mode),
                        prompt=obs.text_prompt,
                        observation=obs,
                        video=bundle,
                    )
                )
            continue
        obs = assemble_observation(executor_result.sim_state, mode)
        bundle = video_bundle_from_observation(obs)
        request_id = f"{sample_id}_{mode.value}"
        _ = camera_idx  # observation builders already select the canonical camera.
        requests.append(
            SolverRequest(
                request_id=request_id,
                scene_id=obs.scene_id,
                mode=mode,
                problem_type=mode_to_problem_type(mode),
                prompt=obs.text_prompt,
                observation=obs,
                video=bundle,
            )
        )
    return tuple(requests)


def _vqa_payload_list(
    executor_result: ExecutorResult,
) -> list[tuple[tuple[str, list[str], int], dict, Any]]:
    """Return ``[(payload, spec, sim_state_or_None), ...]`` for the VQA mode.

    Prefers the scene-template ``vqa_payloads`` (multi-GT); falls back to the
    single ``vqa_payload`` typed-skill path. Empty list when neither present.
    The third tuple element is the question's OWN :class:`SimState` when the
    question came from a colour-permuted variant (its video differs from the
    baseline); ``None`` means "reuse the scene's baseline sim_state".
    """

    if executor_result.vqa_payloads:
        out = []
        for item in executor_result.vqa_payloads:
            payload = tuple(item["payload"])  # (question, options, correct_idx)
            out.append(
                (payload, dict(item.get("spec", {})), item.get("sim_state"))
            )
        return out
    if executor_result.vqa_payload is not None:
        return [
            (executor_result.vqa_payload, dict(executor_result.vqa_spec or {}), None)
        ]
    return []


def score_solver_response(
    request: SolverRequest,
    raw_text: str,
    *,
    sim_state: Any,
    replay_timeout_s: float = 15.0,
) -> SolverScore:
    """Parse and score one Solver response."""

    parsed = parse_solver_output(raw_text, request.mode)
    if parsed is None:
        return SolverScore(
            request_id=request.request_id,
            scene_id=request.scene_id,
            problem_type=request.problem_type,
            raw_text=raw_text,
            parsed_answer=None,
            reward=0.0,
            answer_parse_ok=False,
            schema_ok=False,
            exact_match=False,
            collapse_type=None,
        )

    if request.mode is TaskMode.FORWARD:
        reward = score_forward(parsed, request.observation.ground_truth)
    elif request.mode is TaskMode.INVERSE:
        reward = score_inverse(
            parsed,
            request.observation.ground_truth,
            sim_state,
            replay_timeout_s=replay_timeout_s,
        )
    elif request.mode is TaskMode.PARAMETER:
        reward = score_parameter(parsed, request.observation.ground_truth)
    elif request.mode is TaskMode.VQA:
        reward = score_vqa(parsed, request.observation.ground_truth)
    else:  # pragma: no cover - enum is exhaustive.
        raise ValueError(f"unsupported task mode: {request.mode!r}")

    # Naive-baseline dual track (Naive diagnostics): alongside the soft
    # r_m = k(d), record strict exact-match and whether a wrong prediction
    # collapses to a summary statistic. The dispatcher returns (False, None)
    # for VQA, so the Stage-2 discrete-bottleneck path is unaffected.
    exact_match, collapse_type = naive_diagnostics(
        request.mode,
        parsed,
        request.observation.ground_truth,
        sim_state,
    )

    return SolverScore(
        request_id=request.request_id,
        scene_id=request.scene_id,
        problem_type=request.problem_type,
        raw_text=raw_text,
        parsed_answer=parsed,
        reward=float(reward),
        answer_parse_ok=True,
        schema_ok=True,
        exact_match=exact_match,
        collapse_type=collapse_type,
    )


def executor_result_record(result: ExecutorResult) -> dict[str, Any]:
    """Serialize the executor fields training rewards and summaries need.

    Stage-2 OK results include ``vqa_question``, ``vqa_options``,
    ``vqa_correct_idx``, and ``vqa_spec`` so that audit tools can inspect
    the full question without parsing Solver prompt snapshots.
    """

    status = result.status.value if isinstance(result.status, FailureReason) else str(result.status)
    row: dict[str, Any] = {
        "scene_id": result.scene_id,
        "status": status,
        "ok": bool(result.ok),
        "error_message": result.error_message,
        "proposer_penalty": float(result.proposer_penalty),
        "walltime_s": float(result.walltime_s),
    }
    if result.vqa_payload is not None:
        question, options, correct_idx = result.vqa_payload
        row["vqa_question"] = question
        row["vqa_options"] = list(options)
        row["vqa_correct_idx"] = int(correct_idx)
    if result.vqa_spec is not None:
        row["vqa_spec"] = dict(result.vqa_spec)
    repair_trace = getattr(result, "repair_trace", None)
    if isinstance(repair_trace, dict):
        row["repair_trace"] = dict(repair_trace)
        row["repair_applied"] = bool(repair_trace.get("applied", False))
        row["repair_tier"] = str(repair_trace.get("tier", "none"))
        row["repair_reward_discount"] = float(
            repair_trace.get("reward_discount", 1.0)
        )
        row["repair_actions"] = list(repair_trace.get("actions", []))
    # Scene-template (multi-GT) path: serialize EVERY derived question so audit
    # tools / the rollout probe can show the whole "one scene -> many VQAs" set.
    if getattr(result, "vqa_payloads", None):
        row["vqa_payloads"] = [
            {
                "question": item["payload"][0],
                "options": list(item["payload"][1]),
                "correct_idx": int(item["payload"][2]),
                "spec": dict(item.get("spec", {})),
            }
            for item in result.vqa_payloads
        ]
    return row


def video_request_record(request: SolverRequest) -> dict[str, Any]:
    """Return a JSON-safe summary of the multimodal Solver request.

    ``prompt_text`` is included so that ``solver_requests.jsonl`` can be
    used as a self-contained audit log without reading snapshot files.
    For Stage-2 this is the rendered MC question; for Stage-1 it is the
    forward / inverse / parameter task description.
    """

    frames = request.video.frames
    return {
        "request_id": request.request_id,
        "scene_id": request.scene_id,
        "task_mode": request.mode.value,
        "problem_type": request.problem_type,
        "prompt_text": request.prompt,
        "video_payload_present": True,
        "video_shape": list(frames.shape),
        "video_dtype": str(frames.dtype),
        "video_metadata": dict(request.video.metadata),
        "multi_modal_keys": ["video"],
    }


def save_video_mp4(frames: np.ndarray, path: Path, camera_idx: int = 0) -> None:
    """Write a numpy uint8 frame array to an MP4 file using imageio.

    Args:
        frames: Either ``(T, H, W, 3)`` or ``(n_cameras, T, H, W, 3)`` uint8.
            When 5-D, ``camera_idx`` selects which camera track to save.
        path: Destination file path; parent directory is created if absent.
        camera_idx: Camera index to use when ``frames`` is 5-D
            (``SimState.video_frames`` layout).

    The function is a no-op if ``frames`` is empty. ``imageio`` is imported
    lazily so callers that never save videos pay no import cost.

    Raises:
        ImportError: If ``imageio[ffmpeg]`` is not installed.
        ValueError: If ``frames`` has an unexpected shape or dtype.
    """

    if frames.size == 0:
        return
    if frames.ndim == 5:
        # SimState.video_frames is (n_cameras, T, H, W, 3).
        frames = frames[camera_idx]
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(
            f"frames must be shape (T, H, W, 3) or (n_cameras, T, H, W, 3); got {frames.shape}"
        )
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)

    try:
        import imageio  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "imageio is required for video saving; install it with: pip install imageio[ffmpeg]"
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    import os
    fps = int(os.environ.get("PHYSEVOLVE_VIDEO_FPS", "10"))
    imageio.mimwrite(str(path), [frames[i] for i in range(frames.shape[0])], fps=fps, codec="libx264")


def task_modes_from_strings(values: Sequence[str]) -> tuple[TaskMode, ...]:
    """Validate task-mode CLI strings."""

    modes: list[TaskMode] = []
    for value in values:
        modes.append(TaskMode(str(value)))
    return tuple(modes)


__all__ = [
    "SolverRequest",
    "SolverScore",
    "build_solver_requests",
    "executor_result_record",
    "mode_to_problem_type",
    "save_video_mp4",
    "score_solver_response",
    "task_modes_from_strings",
    "video_request_record",
]
