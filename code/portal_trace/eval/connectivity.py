"""E8: the six rungs in, `connectivity_slope` and the rung-B anchor out.

Proposal 10.4.8, with the estimator conventions of 10.7.2 and 10.7.5 and the
verdict rule transcribed from `configs/experiments/main.yaml`.

E8 asks whether the design graph's effective resistance predicts leakage, and
its primary endpoint is the slope of Lambda^part against ln R_eff. The whole of
its independent variable lives in `diagnostics.design_levels`, which turns each
rung's data ratio table into a graph and reads ln R_eff off it; this module is
the other half, which takes the trained readings and produces the row.

Two structural rules of 10.4.8 are enforced here rather than left to the caller,
because both are ways the endpoint can be quietly widened.

**The slope runs on five rungs and rung B is not one of them.** Rung B has two
components, R_eff is infinite and ln R_eff does not exist, so it cannot be a
point on the axis under any convention. 10.4.8 makes it an upper anchor,
reported as approximately true by construction and excluded from the primary
criterion and from the Holm-Bonferroni family. `E8Report` therefore carries the
anchor as a separate paired gap with its own label, and `ConnectivityObservations`
refuses a reading tagged with a disconnected rung instead of dropping it
silently -- a dropped rung and an excluded rung print the same slope and mean
very different things.

**Every rung is read against the same B7 Oracle.** Appendix A-4: a share
decomposition whose arms each use their own reference point puts the arms'
underfitting differences into the unexplained term. Lambda^part readings arrive
already referenced, and the arm label travels with them so the report can say
which one.

The realised regressor is reported next to the registered MDE. 10.7.5 computes
E8's 0.042 on sum_i (x_i - x_bar)^2 = 1.88 over the nominal ln R_eff ladder, and
`design_levels` reproduces that number from the allocation; if a rung's
allocation is ever changed the sum moves, and 10.7.8 item 4 freezes the table,
so the row prints both.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from portal_trace.diagnostics.design_levels import (
    E8_RUNGS,
    DesignGraphLadder,
    DesignRung,
)
from portal_trace.diagnostics.units import Amplitude
from portal_trace.eval.slopes import SlopeEstimate, SlopeObservations, fit_slope
from portal_trace.eval.stats import ALPHA_PRIMARY

__all__ = [
    "CONNECTIVITY_SLOPE_MDE",
    "CONNECTIVITY_SLOPE_THRESHOLD",
    "CONNECTIVITY_SUM_SQUARES_REGISTERED",
    "RungReading",
    "ConnectivityObservations",
    "E8Report",
    "DisconnectedRungReadingError",
    "MissingRungError",
    "MixedReferenceArmError",
    "MissingAnchorReadingError",
]

CONNECTIVITY_SLOPE_MDE = 0.042
"""10.7.5's registered MDE for `connectivity_slope` at (M, D, S) = (4, 2, 3)."""

CONNECTIVITY_SLOPE_THRESHOLD = 0.045
"""10.4.8's verdict threshold: beta_hat >= 0.045 with a CI excluding zero."""

CONNECTIVITY_SUM_SQUARES_REGISTERED = 1.88
"""The ln R_eff geometry 0.042 was computed on (10.7.5)."""


class DisconnectedRungReadingError(ValueError):
    """A reading was offered for the slope on a rung with no finite ln R_eff."""


class MissingRungError(ValueError):
    """A slope rung of 10.4.8 has no reading at all."""


class MixedReferenceArmError(ValueError):
    """Readings referenced against more than one arm were pooled."""


class MissingAnchorReadingError(ValueError):
    """The rung-B anchor has no reading, so the A - B gap cannot be reported."""


@dataclass(frozen=True)
class RungReading:
    """One (rung, family, defect, seed) cell's Lambda^part.

    `reference_arm` is a field rather than an assumption: appendix A-4 makes the
    reference point part of the number's identity, and two Lambda^part readings
    taken against different arms are not comparable even when they come from the
    same rung.
    """

    rung: str
    family: str
    defect: str
    seed: int
    lambda_part: Amplitude
    reference_arm: str = "B7"

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.lambda_part.value)):
            raise ValueError(
                f"{self.rung}/{self.family}/{self.defect}/s{self.seed}: Lambda^part "
                "is not finite"
            )


class ConnectivityObservations:
    """E8's readings, organised into the slope set and the anchor.

    The ladder is a constructor argument so that a re-derived allocation is
    tested against the readings it produced rather than against the module
    default; the default is 10.4.8's own six rungs.
    """

    def __init__(
        self,
        readings: Sequence[RungReading],
        *,
        ladder: DesignGraphLadder | None = None,
    ) -> None:
        if not readings:
            raise ValueError("E8 has no readings")
        self._ladder = ladder if ladder is not None else DesignGraphLadder()
        self._readings = tuple(readings)

        arms = {r.reference_arm for r in self._readings}
        if len(arms) > 1:
            raise MixedReferenceArmError(
                f"the readings are referenced against {sorted(arms)}. Appendix A-4 "
                "requires every Lambda^part entering one decomposition to be computed "
                "against one and the same B7 Oracle, or the unexplained term absorbs "
                "the reference-point drift"
            )
        self._reference_arm = arms.pop()

        self._axis = self._ladder.ln_r_eff_axis()
        anchor_label = self._ladder.upper_anchor.label
        known = set(self._axis) | {anchor_label}
        unknown = sorted({r.rung for r in self._readings} - known)
        if unknown:
            raise KeyError(
                f"readings tagged with rungs {unknown}, which are not in this ladder "
                f"{sorted(known)}"
            )
        self._slope_readings = tuple(r for r in self._readings if r.rung in self._axis)
        self._anchor_readings = tuple(
            r for r in self._readings if r.rung == anchor_label
        )
        missing = sorted(set(self._axis) - {r.rung for r in self._slope_readings})
        if missing:
            raise MissingRungError(
                f"slope rungs {missing} have no reading. 10.4.8's slope is estimated on "
                "the five connected rungs; estimating it on a subset changes "
                "sum (x - x_bar)^2 and therefore the MDE the row is read against"
            )

    @property
    def reference_arm(self) -> str:
        return self._reference_arm

    @property
    def ladder(self) -> DesignGraphLadder:
        return self._ladder

    def require_connected(self, rung: str) -> float:
        if rung not in self._axis:
            raise DisconnectedRungReadingError(
                f"rung {rung} has no finite ln R_eff, so it is not a point on E8's "
                "axis. 10.4.8 reports it as an upper anchor beside the slope; "
                "E8Report.anchor_gap is where its reading belongs"
            )
        return self._axis[rung]

    def slope_observations(self) -> SlopeObservations:
        rows = self._slope_readings
        return SlopeObservations(
            family=np.array([r.family for r in rows]),
            defect=np.array([r.defect for r in rows]),
            seed=np.array([str(r.seed) for r in rows]),
            level=np.array([r.rung for r in rows]),
            x=np.array([self._axis[r.rung] for r in rows], dtype=float),
            y=np.array([float(r.lambda_part.value) for r in rows], dtype=float),
            label="E8 connectivity_slope: Lambda^part against ln R_eff",
        )

    def anchor_gap(self, reference_label: str = "A") -> tuple[float, int]:
        """Mean Lambda^part(anchor) - Lambda^part(reference), paired by cell.

        Paired because 10.4.8 matches the anchor's task and defect marginals to
        rung A's word for word: the two rungs share their cells, so the gap that
        answers "does disconnection cost more than the trend predicts" is the
        within-cell one.
        """
        if not self._anchor_readings:
            raise MissingAnchorReadingError(
                f"rung {self._ladder.upper_anchor.label} has no reading, so the anchor "
                "gap cannot be reported. 10.4.8 requires the anchor beside the slope; "
                "it is excluded from the primary criterion, not from the table"
            )
        self._ladder.require_anchor_marginals_match(reference_label)
        anchor = {
            (r.family, r.defect, r.seed): float(r.lambda_part.value)
            for r in self._anchor_readings
        }
        reference = {
            (r.family, r.defect, r.seed): float(r.lambda_part.value)
            for r in self._slope_readings
            if r.rung == reference_label
        }
        shared = sorted(set(anchor) & set(reference))
        if not shared:
            raise MissingAnchorReadingError(
                f"rungs {self._ladder.upper_anchor.label} and {reference_label} share "
                "no (family, defect, seed) cell, so there is no paired gap. Their "
                "marginals match by construction, which is what makes the pairing the "
                "right comparison"
            )
        return float(np.mean([anchor[k] - reference[k] for k in shared])), len(shared)


@dataclass(frozen=True)
class E8Report:
    """The `connectivity_slope` row, its verdict and the anchor beside it."""

    estimate: SlopeEstimate
    anchor_label: str
    anchor_gap: float
    anchor_n_pairs: int
    reference_arm: str
    registered_mde: float
    registered_sum_squares_x: float
    threshold: float
    rung_rows: list[dict[str, Any]]

    @property
    def ci_excludes_zero(self) -> bool:
        lo, hi = self.estimate.ci
        return lo > 0.0 or hi < 0.0

    @property
    def meets_threshold(self) -> bool:
        return self.estimate.slope >= self.threshold

    @property
    def significant(self) -> bool:
        """10.4.8's verdict: beta_hat >= 0.045 *and* the 95% CI excludes zero."""
        return self.meets_threshold and self.ci_excludes_zero

    @property
    def geometry_matches_registration(self) -> bool:
        return math.isclose(
            self.estimate.sum_squares_x,
            self.registered_sum_squares_x,
            rel_tol=0.0,
            abs_tol=5e-3,
        )

    def as_row(self) -> dict[str, Any]:
        lo, hi = self.estimate.ci
        return {
            "endpoint": "connectivity_slope",
            "reference_arm": self.reference_arm,
            "beta_hat": self.estimate.slope,
            "ci": [lo, hi],
            "standard_error": self.estimate.standard_error,
            "p_value": self.estimate.p_value,
            "design": str(self.estimate.design),
            "inference_flag": self.estimate.inference_flag,
            "n_slope_rungs": self.estimate.n_levels,
            "sum_squares_x_realised": self.estimate.sum_squares_x,
            "sum_squares_x_registered": self.registered_sum_squares_x,
            "geometry_matches_registration": self.geometry_matches_registration,
            "registered_mde": self.registered_mde,
            "threshold": self.threshold,
            "meets_threshold": self.meets_threshold,
            "ci_excludes_zero": self.ci_excludes_zero,
            "significant": self.significant,
            "per_family_slopes": self.estimate.per_family_slopes,
            "family_slopes_agree_in_sign": self.estimate.family_slopes_agree_in_sign,
            "anchor": {
                "rung": self.anchor_label,
                "paired_gap_vs_A": self.anchor_gap,
                "n_pairs": self.anchor_n_pairs,
                "status": (
                    "upper anchor, approximately true by construction; outside the "
                    "primary criterion and the Holm-Bonferroni family (10.4.8)"
                ),
            },
            "rungs": self.rung_rows,
        }


def e8_report(
    readings: Sequence[RungReading],
    *,
    ladder: DesignGraphLadder | None = None,
    rungs: Sequence[DesignRung] = E8_RUNGS,
    alpha: float = ALPHA_PRIMARY,
) -> E8Report:
    """Fit `connectivity_slope` and assemble the row 11.4 requires.

    `rungs` is accepted only to be checked against the ladder: passing a rung
    set the ladder was not built from would print one design's table beside
    another design's slope.
    """
    observations = ConnectivityObservations(readings, ladder=ladder)
    built = observations.ladder
    if tuple(r.label for r in rungs) != tuple(r.label for r in built.rungs):
        raise ValueError(
            f"the rung set {[r.label for r in rungs]} is not the one this ladder was "
            f"built from, {[r.label for r in built.rungs]}"
        )
    estimate = fit_slope(observations.slope_observations(), alpha=alpha)
    gap, n_pairs = observations.anchor_gap()
    return E8Report(
        estimate=estimate,
        anchor_label=built.upper_anchor.label,
        anchor_gap=gap,
        anchor_n_pairs=n_pairs,
        reference_arm=observations.reference_arm,
        registered_mde=CONNECTIVITY_SLOPE_MDE,
        registered_sum_squares_x=CONNECTIVITY_SUM_SQUARES_REGISTERED,
        threshold=CONNECTIVITY_SLOPE_THRESHOLD,
        rung_rows=built.as_rows(),
    )
