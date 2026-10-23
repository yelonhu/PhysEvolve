"""Read-only trajectory views used by scene-template verifiers.

``SimStateView`` exposes body positions, velocities, status, speed and height
statistics, and collision queries. It omits raw video frames and scene definitions.
Initial positions and velocities refer to the simulator's pre-step snapshot.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from physevolve.engine.schema import BodyTrajectory, CollisionEvent, SimState


class _BodyView:
    """Per-body read-only projection over a :class:`BodyTrajectory`.

    Instances are short-lived and created on demand by
    :class:`SimStateView.__getitem__`. Fields exposed here are the
    supported read-only trajectory fields.

    ``initial_pos`` / ``initial_vel`` come from the pre-step snapshot
    captured by the simulator right after ``_apply_initial_conditions``
    — i.e. the true t = 0 state as authored by the Proposer. They are
    NOT read from trajectory row 0 (which is the post-step state).
    """

    __slots__ = ("_traj", "_initial_pos", "_initial_vel")

    def __init__(
        self,
        traj: BodyTrajectory,
        initial_pos: np.ndarray,
        initial_vel: np.ndarray,
    ) -> None:
        self._traj = traj
        self._initial_pos = initial_pos
        self._initial_vel = initial_vel

    @property
    def initial_pos(self) -> tuple[float, float, float]:
        p = self._initial_pos
        return (float(p[0]), float(p[1]), float(p[2]))

    @property
    def final_pos(self) -> tuple[float, float, float]:
        p = self._traj.positions[-1]
        return (float(p[0]), float(p[1]), float(p[2]))

    @property
    def initial_vel(self) -> tuple[float, float, float]:
        v = self._initial_vel
        return (float(v[0]), float(v[1]), float(v[2]))

    @property
    def final_vel(self) -> tuple[float, float, float]:
        v = self._traj.lin_velocities[-1]
        return (float(v[0]), float(v[1]), float(v[2]))

    @property
    def final_status(self) -> str:
        # Enum value is the raw string — matches the Stage-2 prompt
        # contract (no Python enum required on the Proposer side).
        return self._traj.final_status.value

    @property
    def max_speed(self) -> float:
        speeds = np.linalg.norm(self._traj.lin_velocities, axis=1)
        return float(np.max(speeds))

    @property
    def max_height(self) -> float:
        """Apex z reached anywhere along the trajectory (metres).

        Unlike ``max(initial_pos[2], final_pos[2])``, this captures the
        mid-flight peak of a bouncing / arcing body.
        """

        return float(np.max(self._traj.positions[:, 2]))

    @property
    def rebound_height(self) -> float:
        """Highest z reached AFTER the trajectory's lowest point (metres).

        For a body dropped onto a surface, the global apex is just its
        release height and is identical for two balls dropped together,
        so it cannot reveal a restitution difference. The post-impact
        rebound apex can: a high-restitution ball bounces back higher.

        Computed as ``max(z)`` over the steps at/after the index of the
        minimum z. If the lowest point is the final step (still settling /
        no rebound), this is just the final z.
        """

        z = self._traj.positions[:, 2]
        low_idx = int(np.argmin(z))
        return float(np.max(z[low_idx:]))


class SimStateView:
    """Stage-2 Proposer-facing projection of a :class:`SimState`.

    Crucially, ``video_frames`` and ``scene_def`` are NOT exposed: the
    VQA judge must reason over trajectory summaries, not raw pixels or
    its own ground-truth scene definition.
    """

    __slots__ = (
        "_trajectories",
        "_collisions",
        "_body_names",
        "_initial_positions",
        "_initial_lin_velocities",
    )

    def __init__(self, sim_state: SimState) -> None:
        self._trajectories = sim_state.trajectories
        self._collisions = sim_state.collisions
        self._body_names = tuple(sim_state.trajectories.keys())
        self._initial_positions = sim_state.initial_positions
        self._initial_lin_velocities = sim_state.initial_lin_velocities

    # ─── Body access ──────────────────────────────────────────────────

    @property
    def body_names(self) -> list[str]:
        """Return a fresh mutable list so user code can't mutate our tuple."""

        return list(self._body_names)

    def __getitem__(self, name: str) -> _BodyView:
        traj = self._trajectories.get(name)
        if traj is None:
            raise KeyError(f"unknown body {name!r}")
        return _BodyView(
            traj,
            initial_pos=self._initial_positions[name],
            initial_vel=self._initial_lin_velocities[name],
        )

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._trajectories

    def __iter__(self):
        return iter(self._body_names)

    def __len__(self) -> int:
        return len(self._body_names)

    # ─── Collision queries ────────────────────────────────────────────

    def collided(self, body_a: str, body_b: str) -> bool:
        """``True`` iff any recorded collision involves this unordered pair."""

        return self._find_collision(body_a, body_b) is not None

    def time_of_collision(
        self, body_a: str, body_b: str
    ) -> Optional[float]:
        """Step index of the earliest collision for ``(a, b)`` or ``None``.

        Returned as a ``float`` (seconds only if multiplied by timestep
        by the caller) to match the Stage-2 prompt. The aggregation
        window is 10 simulation steps, so the returned time
        is the ``t_step`` of the window-start, not the exact contact
        instant.
        """

        event = self._find_collision(body_a, body_b)
        return None if event is None else float(event.t_step)

    # ─── Internal ─────────────────────────────────────────────────────

    def _find_collision(
        self, body_a: str, body_b: str
    ) -> Optional[CollisionEvent]:
        """Earliest event whose unordered pair matches ``(a, b)``."""

        earliest: Optional[CollisionEvent] = None
        for ev in self._collisions:
            if {ev.body_a, ev.body_b} == {body_a, body_b}:
                if earliest is None or ev.t_step < earliest.t_step:
                    earliest = ev
        return earliest


def sim_state_public_view(sim_state: SimState) -> SimStateView:
    """Construct a :class:`SimStateView` wrapping ``sim_state``."""

    return SimStateView(sim_state)


__all__ = [
    "SimStateView",
    "sim_state_public_view",
]
