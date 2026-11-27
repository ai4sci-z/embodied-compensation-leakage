"""Diagnostic quantities: the Lambda family, common mode, design graph, regimes.

`two_way_fit` is the only module here that produces estimates of theta_tau and
gamma_delta. `design_graph` computes design quantities only and `common_mode`
reads gamma off the injected ground truth; neither is an estimator. Proposal
11.2 is explicit about this split, because without `two_way_fit` propositions
1(e), 1(e') and 3 have no measured counterpart to compare against.
"""

from portal_trace.diagnostics.units import (
    Amplitude,
    DimensionError,
    EffectiveSampleCount,
    Energy,
    amp,
    energy,
)

__all__ = [
    "Amplitude",
    "DimensionError",
    "EffectiveSampleCount",
    "Energy",
    "amp",
    "energy",
]
