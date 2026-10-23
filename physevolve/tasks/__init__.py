"""Task router + per-mode observation builders.

Public surface:

* :class:`TaskMode`, :class:`SolverObservation`
* :func:`select_task_mode`, :func:`assemble_observation`
* :func:`parse_solver_output`
"""

from physevolve.tasks.router import (
    SolverObservation,
    TaskMode,
    assemble_observation,
    parse_solver_output,
    select_task_mode,
)

__all__ = [
    "TaskMode",
    "SolverObservation",
    "select_task_mode",
    "assemble_observation",
    "parse_solver_output",
]
