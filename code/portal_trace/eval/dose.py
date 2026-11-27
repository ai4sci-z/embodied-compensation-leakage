"""E3's dose curves under X13: three calibers reported together, or nothing.

Confounder X13 (10.7.2's table, 10.4.3, 14.5 item 13) is dose-dependent cell
exclusion. Two mechanisms remove cells from an E3 dose axis and both get more
likely as the dose rises: the ILC teacher stops converging, and the demonstrator
switches contact mode so that (A3) no longer holds and the trajectory is not a
compensation of the same plan. The high levels therefore keep only the cells
that could still be taught, and T2's predicted descent of Lambda is partly
manufactured by that selection. This is informative missingness, not noise, and
no amount of extra seeds touches it.

10.4.3 fixes the response, and it is a reporting obligation rather than a test:

1. the **balanced panel** B -- the (embodiment, defect, task) keys judged
   converged at *every* level -- which carries the main narrative, reported with
   |B| and the per-defect panel retention rate;
2. the **all-cells** curve alongside, as a coverage statement;
3. a **worst-case fill bound**, which imputes every excluded cell in the
   direction least favourable to T2 and reports the imputed slope and the width
   of the bound. **If the bound does not preserve the sign, E3's dose conclusion
   is demoted to exploratory.**

Two things about the bound are worth stating because they decide what it means.

The imputation rule is 10.4.3's own, transcribed: a hole above the level
centroid is filled with that defect's Lambda at the *lowest* level and a hole
below it with the *highest*. Written that way the rule is adverse to a monotone
prediction in either direction, so it serves the descending |delta| axis and the
ascending k axis without a sign flag. What it does *not* do is stay adverse when
the observed anchors are not themselves ordered the way the prediction says, and
`WorstCaseFill.anchors_follow_prediction` reports whether they are -- when they
are not, the interval is still an interval, but calling it the worst case would
be a claim about data the rule never looked at.

The level exactly at the centroid, which exists whenever the axis has an odd
number of levels, has zero leverage on the fitted slope: its deviation from the
mean level is zero, so either anchor gives the same bound. It is filled with the
midpoint of the two anchors, which is the only choice that does not pull the
*curve* towards one end of an axis it cannot pull the slope along.

The slope is fitted per (family, defect) on that cell's own measured regressor
and then averaged. 10.7.5 is explicit that E3's sum of squares is computed per
defect, which is why E3 has no row in `SLOPE_MDE_TABLE` at all: the regressor is
`defect.observable.{defect}.{level}.{family}` from the lock file, so two defects
on the same nominal level sit at different places on the axis, and one pooled
line would be fitted against a geometry that belongs to neither. T2's registered
criterion is the rank test (Spearman rho <= -0.7 on the top three levels), so
that is computed for each of the three calibers too.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats as sps

from portal_trace.diagnostics.units import Amplitude
from portal_trace.eval.stats import Design, preregistered_inference
from portal_trace.teacher.ilc import Outcome

__all__ = [
    "T2_SPEARMAN_THRESHOLD",
    "T2_RANK_TEST_TOP_LEVELS",
    "DoseAxis",
    "PredictedDirection",
    "CurveCaliber",
    "PanelKey",
    "DoseReading",
    "DoseCurve",
    "WorstCaseFill",
    "DoseConclusionStatus",
    "X13DosePanel",
    "CurveNotEstimableError",
    "IncompleteDosePanelError",
    "WorstCaseAnchorMissingError",
    "EmptyBalancedPanelError",
    "ConstantExclusionRateError",
]

T2_SPEARMAN_THRESHOLD = -0.7
"""T2's rank criterion on the |delta| axis (10.4.3): Spearman rho <= -0.7."""

T2_RANK_TEST_TOP_LEVELS = 3
"""...evaluated on the top three amplitude levels, not on all five (10.4.3)."""


class DoseAxis(str, Enum):
    """The three orthogonal dose axes of 10.4.3, with their level counts.

    The counts are part of the design and are checked against the readings: the
    balanced panel is defined as "converged at every level", so a panel built
    over four of the |delta| axis's five levels is a different set with the same
    name.
    """

    DEFECT_AMPLITUDE = "delta"
    MIXING_WEIGHT = "alpha"
    ILC_ITERATIONS = "k"

    @property
    def n_levels(self) -> int:
        return 4 if self is DoseAxis.MIXING_WEIGHT else 5

    @property
    def design(self) -> Design:
        """(M, D, S) from 10.4.3's dose table."""
        if self is DoseAxis.DEFECT_AMPLITUDE:
            return Design(2, 2, 2, "E3 |delta| axis")
        return Design(2, 1, 2, f"E3 {self.value} axis")


class PredictedDirection(str, Enum):
    """Which way the preregistered prediction says Lambda moves along this axis.

    Declared by the caller rather than derived from the axis. T2 predicts a
    descent in |delta| and 10.4.3 predicts a rise in k, but the alpha axis's
    registered prediction (T1) is about `Lambda^part - E_rho[w]` and not about
    Lambda's own monotonicity, so an axis cannot supply this on its own without
    inventing a prediction for that case.
    """

    DECREASING = "decreasing"
    INCREASING = "increasing"

    @property
    def sign(self) -> float:
        return -1.0 if self is PredictedDirection.DECREASING else 1.0


class CurveCaliber(str, Enum):
    """The three curves 10.4.3 requires to be reported together."""

    BALANCED_PANEL = "balanced panel (converged at every level)"
    ALL_CELLS = "all cells"
    WORST_CASE_FILL = "worst-case fill bound"


class DoseConclusionStatus(str, Enum):
    """10.4.3's demotion rule, as an outcome rather than a note."""

    AS_REGISTERED = "as registered: the worst-case bound preserves the sign"
    DEMOTED_TO_EXPLORATORY = (
        "demoted to exploratory: the worst-case fill bound does not preserve the "
        "sign of the balanced-panel slope (10.4.3)"
    )


class IncompleteDosePanelError(ValueError):
    """A (key, level) of the dose grid has no reading at all.

    Distinct from an excluded cell, which has a reading whose outcome says why
    there is no Lambda. The worst-case bound imputes exclusions; it cannot
    impute a cell nobody ran, because there is no record of whether the ILC run
    happened.
    """


class WorstCaseAnchorMissingError(ValueError):
    """A defect has no admitted reading at one end of the axis.

    The fill rule of 10.4.3 anchors on that defect's Lambda at the lowest and
    the highest level. With one end empty there is nothing to fill towards, and
    substituting the nearest available level would silently narrow the bound.
    """


class EmptyBalancedPanelError(ValueError):
    """No key converged at every level, so the main narrative has no curve."""


class ConstantExclusionRateError(ValueError):
    """The exclusion rate is the same at every level, so it ranks against nothing.

    Not the same as "the exclusions do not covary with the dose": a rank
    correlation against a constant has no value, and the most common way to get
    here is the good case of no exclusions at all.
    """


@dataclass(frozen=True)
class PanelKey:
    """The (embodiment, defect, task) triple 10.4.3 defines the panel over."""

    family: str
    defect: str
    task: str

    def __str__(self) -> str:
        return f"{self.family}|{self.defect}|{self.task}"


@dataclass(frozen=True)
class DoseReading:
    """One (key, level) cell of a dose axis, admitted or not.

    `lambda_part` is present exactly when the cell is admitted. A limit-cycle or
    non-converged ILC run produces no demonstrations for the cell, so it has no
    policy reading to carry; and a cell whose mode-sequence hash says the
    demonstrator changed plan is excluded by (A3) even though ILC converged
    (X6, whose handling 10.7.2's table merges into X13's).

    `x` is the measured regressor for this (family, defect, level) --
    `defect.observable.{defect}.{level}.{family}` from the lock file for the
    |delta| axis. It is present on excluded cells too, because the injected
    amplitude is a property of the injection and not of whether the teacher
    managed to track it.
    """

    key: PanelKey
    level: int
    x: float
    outcome: Outcome
    mode_switch_excluded: bool
    lambda_part: Amplitude | None = None

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError(f"{self.key}: level index must be non-negative")
        if not math.isfinite(self.x):
            raise ValueError(f"{self.key} level {self.level}: regressor x is not finite")
        if self.outcome is Outcome.CONVERGED and self.mode_switch_excluded:
            if self.lambda_part is not None:
                raise ValueError(
                    f"{self.key} level {self.level}: the cell is excluded by the (A3) "
                    "mode-sequence check but carries a Lambda. An excluded cell's "
                    "demonstrations do not enter the training pool, so there is no "
                    "reading to attach"
                )
        elif self.outcome is Outcome.CONVERGED and self.lambda_part is None:
            raise ValueError(
                f"{self.key} level {self.level}: ILC converged and the cell was not "
                "excluded by (A3), so it has demonstrations and must carry its "
                "Lambda^part reading"
            )
        elif self.outcome is not Outcome.CONVERGED and self.lambda_part is not None:
            raise ValueError(
                f"{self.key} level {self.level}: outcome is {self.outcome.value} but a "
                "Lambda is attached. 9.4.2 keeps the three ILC outcomes apart because a "
                "limit cycle's demonstrations are a mode-selection artefact, not a "
                "weaker compensation; they do not enter the pool"
            )

    @property
    def admitted(self) -> bool:
        return self.outcome is Outcome.CONVERGED and not self.mode_switch_excluded

    @property
    def value(self) -> float:
        if self.lambda_part is None:
            raise ValueError(f"{self.key} level {self.level} has no Lambda reading")
        return float(self.lambda_part.value)


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    centred = x - x.mean()
    sxx = float(np.dot(centred, centred))
    if sxx <= 0.0:
        raise ValueError("the dose regressor has no spread inside this cell")
    return float(np.dot(centred, y - y.mean()) / sxx)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        raise ValueError(
            f"Spearman's rho on {x.size} points is not the rank test of 10.4.3, which "
            f"runs on the top {T2_RANK_TEST_TOP_LEVELS} levels"
        )
    return float(sps.spearmanr(x, y).statistic)


class CurveNotEstimableError(ValueError):
    """No cell of a curve has enough surviving levels to be fitted at all.

    The all-cells curve can lose a cell entirely to X13's exclusions, and a cell
    with one surviving level has no slope. Losing one cell is reported next to
    the curve; losing all of them means the caliber has no curve, and saying so
    is the point of the three-caliber rule.
    """


@dataclass(frozen=True)
class DoseCurve:
    """One caliber's curve: the per-level means, the per-cell slopes, the rank test.

    `slope` is the mean of the per-(family, defect) within-cell slopes, which is
    10.7.5's estimator for this endpoint. `sum_squares_x` is reported per cell
    for the same reason: on this axis the regressor is measured per defect and
    per family, so there is no single geometry the curve sits on.
    """

    caliber: CurveCaliber
    axis: DoseAxis
    level_means: dict[int, float]
    level_counts: dict[int, int]
    per_cell_slopes: dict[str, float]
    per_cell_sum_squares_x: dict[str, float]
    per_cell_spearman: dict[str, float]
    cells_without_slope: tuple[str, ...]
    cells_without_rank_test: tuple[str, ...]
    n_keys: int
    n_readings: int
    n_imputed: int

    def __post_init__(self) -> None:
        if not self.per_cell_slopes:
            raise CurveNotEstimableError(
                f"{self.caliber.value} on the {self.axis.value} axis: no "
                "(family, defect) cell keeps two levels, so this caliber has no slope. "
                f"Cells dropped: {list(self.cells_without_slope)}"
            )

    @property
    def slope(self) -> float:
        return float(np.mean(list(self.per_cell_slopes.values())))

    @property
    def spearman_top_levels(self) -> float:
        """Mean per-cell Spearman rho over the top levels; T2's own criterion."""
        if not self.per_cell_spearman:
            raise CurveNotEstimableError(
                f"{self.caliber.value} on the {self.axis.value} axis: no cell keeps "
                f"{T2_RANK_TEST_TOP_LEVELS} points on the top levels, so T2's rank test "
                f"has nothing to run on. Cells dropped: {list(self.cells_without_rank_test)}"
            )
        return float(np.mean(list(self.per_cell_spearman.values())))

    @property
    def slope_sign(self) -> float:
        return math.copysign(1.0, self.slope) if self.slope != 0.0 else 0.0

    def satisfies_t2_rank_test(self) -> bool:
        return self.spearman_top_levels <= T2_SPEARMAN_THRESHOLD

    def as_row(self) -> dict[str, Any]:
        return {
            "caliber": self.caliber.value,
            "axis": self.axis.value,
            "n_keys": self.n_keys,
            "n_readings": self.n_readings,
            "n_imputed": self.n_imputed,
            "level_means": dict(sorted(self.level_means.items())),
            "level_counts": dict(sorted(self.level_counts.items())),
            "slope": self.slope,
            "per_cell_slopes": dict(sorted(self.per_cell_slopes.items())),
            "per_cell_sum_squares_x": dict(sorted(self.per_cell_sum_squares_x.items())),
            "per_cell_spearman": dict(sorted(self.per_cell_spearman.items())),
            "cells_without_slope": list(self.cells_without_slope),
            "cells_without_rank_test": list(self.cells_without_rank_test),
        }


@dataclass(frozen=True)
class WorstCaseFill:
    """The imputation behind the third curve, with what it was anchored on."""

    low_anchor: dict[str, float]
    high_anchor: dict[str, float]
    centroid_level: float
    imputed: dict[tuple[str, int], float]
    predicted_direction: PredictedDirection

    @property
    def anchors_follow_prediction(self) -> bool:
        """Whether the observed anchors are ordered the way the prediction says.

        The fill rule of 10.4.3 is adverse to a monotone prediction only while
        the low-level anchor sits on the side the prediction puts it: under a
        predicted descent the lowest level must read higher than the highest.
        When it does not, the imputation is still the registered one and the
        interval is still reported, but it is no longer the worst case for these
        data and the report says so rather than keeping the name.
        """
        sign = self.predicted_direction.sign
        return all(
            sign * (self.high_anchor[d] - self.low_anchor[d]) >= 0.0
            for d in self.low_anchor
        )

    def as_row(self) -> dict[str, Any]:
        return {
            "rule": (
                "holes above the level centroid take the defect's lowest-level mean, "
                "holes below take its highest-level mean (10.4.3)"
            ),
            "centroid_level": self.centroid_level,
            "low_anchor": dict(sorted(self.low_anchor.items())),
            "high_anchor": dict(sorted(self.high_anchor.items())),
            "n_imputed": len(self.imputed),
            "predicted_direction": self.predicted_direction.value,
            "anchors_follow_prediction": self.anchors_follow_prediction,
        }


class X13DosePanel:
    """One dose axis's readings, and the three curves 10.4.3 reports together.

    The three curves are produced by one object rather than by three calls
    because reporting one of them alone is the failure X13's entry legislates
    against: the balanced panel on its own hides how much was dropped, the
    all-cells curve on its own is the selected one, and the bound on its own has
    no sign to be compared with.
    """

    def __init__(
        self,
        readings: Sequence[DoseReading],
        *,
        axis: DoseAxis,
        predicted_direction: PredictedDirection,
    ) -> None:
        if not readings:
            raise ValueError(f"{axis.value} axis: no readings")
        self._axis = axis
        self._direction = predicted_direction
        self._readings = tuple(readings)
        self._levels = tuple(sorted({r.level for r in self._readings}))
        if len(self._levels) != axis.n_levels:
            raise ValueError(
                f"{axis.value} axis: {len(self._levels)} levels {list(self._levels)}, "
                f"but 10.4.3 fixes {axis.n_levels}. The balanced panel is defined as "
                "'converged at every level', so a panel over a different level set is "
                "a different object wearing the same name"
            )
        self._keys = tuple(
            sorted({r.key for r in self._readings}, key=str)
        )
        self._by_cell: dict[tuple[PanelKey, int], DoseReading] = {}
        for reading in self._readings:
            slot = (reading.key, reading.level)
            if slot in self._by_cell:
                raise ValueError(f"{reading.key} level {reading.level} appears twice")
            self._by_cell[slot] = reading
        missing = [
            f"{key} level {level}"
            for key in self._keys
            for level in self._levels
            if (key, level) not in self._by_cell
        ]
        if missing:
            raise IncompleteDosePanelError(
                f"{axis.value} axis: no reading at all for {missing}. An excluded cell "
                "carries its ILC outcome and its measured regressor; a cell with no "
                "record cannot be told apart from one that was never run, and the "
                "worst-case bound of 10.4.3 fills exclusions, not gaps"
            )

    # -- what X13 asks to be reported about the exclusions ------------------

    @property
    def axis(self) -> DoseAxis:
        return self._axis

    @property
    def levels(self) -> tuple[int, ...]:
        return self._levels

    @property
    def keys(self) -> tuple[PanelKey, ...]:
        return self._keys

    def balanced_panel(self) -> tuple[PanelKey, ...]:
        """B: the keys admitted at every level (10.4.3 item 1)."""
        return tuple(
            key
            for key in self._keys
            if all(self._by_cell[(key, lv)].admitted for lv in self._levels)
        )

    def panel_retention_by_defect(self) -> dict[str, float]:
        """|B ∩ defect| / |keys of that defect|, which 10.4.3 requires by name."""
        panel = set(self.balanced_panel())
        out: dict[str, float] = {}
        for defect in sorted({k.defect for k in self._keys}):
            of_defect = [k for k in self._keys if k.defect == defect]
            out[defect] = len([k for k in of_defect if k in panel]) / len(of_defect)
        return out

    def outcome_counts_by_level(self) -> dict[int, dict[str, int]]:
        """The three ILC outcomes kept apart, plus the (A3) exclusions (9.4.2).

        10.7.2's X13 row says the three outcomes are listed separately and not
        merged into one exclusion rate, and the mode-switch exclusion is a
        fourth column rather than a fourth outcome: it applies to runs that did
        converge.
        """
        out: dict[int, dict[str, int]] = {}
        for level in self._levels:
            at_level = [self._by_cell[(k, level)] for k in self._keys]
            out[level] = {
                Outcome.CONVERGED.value: sum(
                    1 for r in at_level if r.outcome is Outcome.CONVERGED
                ),
                Outcome.LIMIT_CYCLE.value: sum(
                    1 for r in at_level if r.outcome is Outcome.LIMIT_CYCLE
                ),
                Outcome.NOT_CONVERGED.value: sum(
                    1 for r in at_level if r.outcome is Outcome.NOT_CONVERGED
                ),
                "mode_switch_excluded": sum(1 for r in at_level if r.mode_switch_excluded),
                "admitted": sum(1 for r in at_level if r.admitted),
            }
        return out

    def exclusion_rate_by_level(self) -> dict[int, float]:
        counts = self.outcome_counts_by_level()
        return {
            level: 1.0 - counts[level]["admitted"] / len(self._keys)
            for level in self._levels
        }

    def exclusion_covaries_with_dose(self) -> float:
        """Spearman rho of the exclusion rate against the level index.

        The number that makes X13 a confounder rather than a worry: a rate that
        rises with the dose is missingness correlated with the main regressor.
        """
        rates = self.exclusion_rate_by_level()
        values = np.array([rates[lv] for lv in self._levels], dtype=float)
        if np.unique(values).size == 1:
            raise ConstantExclusionRateError(
                f"{self._axis.value} axis: the exclusion rate is {values[0]} at every "
                "level, so its rank correlation with the dose does not exist. At zero "
                "it is the case X13 is worried about not arising"
            )
        return _spearman(np.array(self._levels, dtype=float), values)

    # -- the three curves ---------------------------------------------------

    def _cell_label(self, key: PanelKey) -> str:
        return f"{key.family}|{key.defect}"

    def _curve(
        self,
        caliber: CurveCaliber,
        values: Mapping[tuple[PanelKey, int], float],
        *,
        n_imputed: int,
    ) -> DoseCurve:
        keys_used = sorted({k for k, _ in values}, key=str)
        level_means: dict[int, float] = {}
        level_counts: dict[int, int] = {}
        for level in self._levels:
            at_level = [v for (k, lv), v in values.items() if lv == level]
            level_counts[level] = len(at_level)
            if at_level:
                level_means[level] = float(np.mean(at_level))

        slopes: dict[str, float] = {}
        sum_squares: dict[str, float] = {}
        rho: dict[str, float] = {}
        no_slope: list[str] = []
        no_rank_test: list[str] = []
        cells = sorted({self._cell_label(k) for k in keys_used})
        top = self._levels[-T2_RANK_TEST_TOP_LEVELS:]
        for cell in cells:
            rows = [
                (self._by_cell[(k, lv)].x, v)
                for (k, lv), v in values.items()
                if self._cell_label(k) == cell
            ]
            xs = np.array([r[0] for r in rows], dtype=float)
            ys = np.array([r[1] for r in rows], dtype=float)
            positions = np.unique(xs)
            if positions.size < 2:
                no_slope.append(cell)
            else:
                slopes[cell] = _ols_slope(xs, ys)
                centred = positions - positions.mean()
                sum_squares[cell] = float(np.dot(centred, centred))
            top_rows = [
                (self._by_cell[(k, lv)].x, v)
                for (k, lv), v in values.items()
                if self._cell_label(k) == cell and lv in top
            ]
            if len(top_rows) < T2_RANK_TEST_TOP_LEVELS:
                no_rank_test.append(cell)
            else:
                rho[cell] = _spearman(
                    np.array([r[0] for r in top_rows], dtype=float),
                    np.array([r[1] for r in top_rows], dtype=float),
                )
        return DoseCurve(
            caliber=caliber,
            axis=self._axis,
            level_means=level_means,
            level_counts=level_counts,
            per_cell_slopes=slopes,
            per_cell_sum_squares_x=sum_squares,
            per_cell_spearman=rho,
            cells_without_slope=tuple(no_slope),
            cells_without_rank_test=tuple(no_rank_test),
            n_keys=len(keys_used),
            n_readings=len(values),
            n_imputed=n_imputed,
        )

    def balanced_curve(self) -> DoseCurve:
        panel = self.balanced_panel()
        if not panel:
            raise EmptyBalancedPanelError(
                f"{self._axis.value} axis: no (embodiment, defect, task) key converged "
                f"at all {self._axis.n_levels} levels, so the balanced panel of 10.4.3 "
                "is empty and the main narrative has no curve. The per-level exclusion "
                "counts are still reportable; the dose conclusion is not"
            )
        values = {
            (key, lv): self._by_cell[(key, lv)].value
            for key in panel
            for lv in self._levels
        }
        return self._curve(CurveCaliber.BALANCED_PANEL, values, n_imputed=0)

    def all_cells_curve(self) -> DoseCurve:
        values = {
            (r.key, r.level): r.value for r in self._readings if r.admitted
        }
        return self._curve(CurveCaliber.ALL_CELLS, values, n_imputed=0)

    def worst_case_fill(self) -> WorstCaseFill:
        """10.4.3's imputation, built from the admitted readings at the two ends."""
        low, high = self._levels[0], self._levels[-1]
        low_anchor: dict[str, float] = {}
        high_anchor: dict[str, float] = {}
        for defect in sorted({k.defect for k in self._keys}):
            for level, anchor in ((low, low_anchor), (high, high_anchor)):
                admitted = [
                    self._by_cell[(k, level)].value
                    for k in self._keys
                    if k.defect == defect and self._by_cell[(k, level)].admitted
                ]
                if not admitted:
                    raise WorstCaseAnchorMissingError(
                        f"{self._axis.value} axis: defect {defect} has no admitted cell "
                        f"at level {level}, which is one of the two anchors 10.4.3's "
                        "fill rule reads. Filling from the nearest surviving level "
                        "instead would narrow the bound by exactly the amount the "
                        "exclusion cost, which is the quantity the bound exists to show"
                    )
                anchor[defect] = float(np.mean(admitted))

        centroid = float(np.mean(self._levels))
        imputed: dict[tuple[str, int], float] = {}
        for key in self._keys:
            for level in self._levels:
                if self._by_cell[(key, level)].admitted:
                    continue
                if level > centroid:
                    filled = low_anchor[key.defect]
                elif level < centroid:
                    filled = high_anchor[key.defect]
                else:
                    filled = 0.5 * (low_anchor[key.defect] + high_anchor[key.defect])
                imputed[(str(key), level)] = filled
        return WorstCaseFill(
            low_anchor=low_anchor,
            high_anchor=high_anchor,
            centroid_level=centroid,
            imputed=imputed,
            predicted_direction=self._direction,
        )

    def worst_case_curve(self) -> DoseCurve:
        fill = self.worst_case_fill()
        values: dict[tuple[PanelKey, int], float] = {}
        for key in self._keys:
            for level in self._levels:
                reading = self._by_cell[(key, level)]
                values[(key, level)] = (
                    reading.value
                    if reading.admitted
                    else fill.imputed[(str(key), level)]
                )
        return self._curve(
            CurveCaliber.WORST_CASE_FILL, values, n_imputed=len(fill.imputed)
        )

    # -- the demotion rule ---------------------------------------------------

    def bound_width(self) -> float:
        """|worst-case slope - balanced-panel slope|.

        The main narrative runs on the balanced panel, so the bound is the
        distance the imputation can move that number; 10.4.3 asks for the
        imputed slope and this width together.
        """
        return abs(self.worst_case_curve().slope - self.balanced_curve().slope)

    def sign_preserved(self) -> bool:
        balanced = self.balanced_curve().slope_sign
        worst = self.worst_case_curve().slope_sign
        return balanced != 0.0 and worst == balanced

    def sign_preserved_by_cell(self) -> dict[str, bool]:
        balanced = self.balanced_curve().per_cell_slopes
        worst = self.worst_case_curve().per_cell_slopes
        return {
            cell: (
                slope != 0.0
                and cell in worst
                and math.copysign(1.0, worst[cell]) == math.copysign(1.0, slope)
            )
            for cell, slope in balanced.items()
        }

    def conclusion_status(self) -> DoseConclusionStatus:
        if self.sign_preserved():
            return DoseConclusionStatus.AS_REGISTERED
        return DoseConclusionStatus.DEMOTED_TO_EXPLORATORY

    def report(self) -> dict[str, Any]:
        """Everything 10.4.3 requires reported together, in one object.

        The three curves are fields of one dictionary for the same reason the
        class exists: X13's handling is the set, and a caller that wanted only
        the balanced panel would have to drop the other two by hand.
        """
        balanced = self.balanced_curve()
        fill = self.worst_case_fill()
        worst = self.worst_case_curve()
        _, inference_flag = preregistered_inference(self._axis.design)
        try:
            exclusion_rho: float | None = self.exclusion_covaries_with_dose()
            exclusion_rho_absent = None
        except ConstantExclusionRateError as exc:
            exclusion_rho = None
            exclusion_rho_absent = str(exc)
        return {
            "axis": self._axis.value,
            "design": str(self._axis.design),
            "inference_flag": inference_flag,
            "n_keys": len(self._keys),
            "balanced_panel_size": len(self.balanced_panel()),
            "panel_retention_by_defect": self.panel_retention_by_defect(),
            "outcome_counts_by_level": self.outcome_counts_by_level(),
            "exclusion_rate_by_level": self.exclusion_rate_by_level(),
            "exclusion_spearman_vs_level": exclusion_rho,
            "exclusion_spearman_absent_because": exclusion_rho_absent,
            "curves": [
                balanced.as_row(),
                self.all_cells_curve().as_row(),
                worst.as_row(),
            ],
            "worst_case_fill": fill.as_row(),
            "balanced_spearman_top_levels": balanced.spearman_top_levels,
            "balanced_satisfies_t2_rank_test": balanced.satisfies_t2_rank_test(),
            "worst_case_spearman_top_levels": worst.spearman_top_levels,
            "bound_width": self.bound_width(),
            "sign_preserved": self.sign_preserved(),
            "sign_preserved_by_cell": self.sign_preserved_by_cell(),
            "status": self.conclusion_status().value,
        }
