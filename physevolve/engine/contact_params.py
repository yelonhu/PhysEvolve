"""MuJoCo contact parameters derived from DSL restitution.

MuJoCo does not expose a direct coefficient-of-restitution geom
attribute. For the current MuJoCo soft-contact model, restitution-like
behavior is controlled through ``solref`` damping while ``solimp`` shapes
the contact impedance curve. This module isolates that upstream-specific
mapping so future MuJoCo version drift is fixed in one place.
"""

from __future__ import annotations

import math

# MuJoCo's default contact impedance. Keep this independent from the DSL
# restitution scalar; the scalar belongs in the solref damping term below.
DEFAULT_CONTACT_SOLIMP: tuple[float, float, float] = (0.9, 0.95, 0.001)

# Positive-format solref time constant. The default simulator timestep is
# 0.002 s, so 0.025 s stays comfortably above MuJoCo's 2*timestep stability
# guideline while keeping contacts firm enough for short golden scenes.
DEFAULT_CONTACT_TIMECONST: float = 0.025

# Keep the positive-format dampratio strictly positive so MuJoCo parses the
# pair as (timeconst, dampratio), while remaining effectively elastic.
_MIN_DAMPRATIO: float = 0.05

# The analytic oscillator inverse under-damps MuJoCo's default soft contacts
# enough to inject energy in equal-mass impacts. This single empirical scale
# is calibrated by CPU regression tests below the engine layer.
_DAMPING_CALIBRATION: float = 2.4


def restitution_to_dampratio(restitution: float) -> float:
    """Approximate a coefficient of restitution as a damping ratio.

    The mapping is the inverse of the standard under-damped oscillator
    relation ``e = exp(-pi*zeta / sqrt(1 - zeta**2))``. It is a contact
    calibration heuristic, not a guarantee of exact COR for every mass and
    geometry combination in MuJoCo's soft solver.
    """

    if not 0.0 <= restitution <= 1.0:
        raise ValueError(f"restitution must be in [0, 1], got {restitution}")
    if restitution == 0.0:
        return 1.0
    if restitution == 1.0:
        return _MIN_DAMPRATIO

    log_e = math.log(restitution)
    dampratio = (
        _DAMPING_CALIBRATION
        * -log_e
        / math.sqrt(math.pi * math.pi + log_e * log_e)
    )
    return max(_MIN_DAMPRATIO, min(1.0, dampratio))


def restitution_to_mjcf_contact(
    restitution: float,
) -> dict[str, tuple[float, ...]]:
    """Return MJCF contact attributes for a positive DSL restitution value."""

    if restitution <= 0.0:
        return {}
    return {
        "solref": (DEFAULT_CONTACT_TIMECONST, restitution_to_dampratio(restitution)),
        "solimp": DEFAULT_CONTACT_SOLIMP,
    }


__all__ = [
    "DEFAULT_CONTACT_SOLIMP",
    "DEFAULT_CONTACT_TIMECONST",
    "restitution_to_dampratio",
    "restitution_to_mjcf_contact",
]
