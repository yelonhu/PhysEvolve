"""Operational smoke-test helpers used by the Stage-2 training entry.

The trainer imports :class:`~physevolve.smoke.run_config.SmokeGenerationConfig`
and IO helpers.
"""

from physevolve.smoke.run_config import (
    DEFAULT_PROMPT_VERSION,
    SmokeGenerationConfig,
    build_run_id,
    model_short_name,
)

__all__ = [
    "DEFAULT_PROMPT_VERSION",
    "SmokeGenerationConfig",
    "build_run_id",
    "model_short_name",
]
