"""E5 endpoint B: route (S) anchoring on real data (proposal 10.4.6, 12 item 9).

Endpoint B fits delta_hat per institution (delay ell, transmission gain, static
bias -- 3 to 6 parameters), builds a subtractable compensation model from the
physical basis psi_j(delta_hat), and tests on a **held-out institution**: after
subtracting the model, how much does the between-institution variance component
fall? The preregistered prediction is at least 15%.

Why this endpoint is route (S) and not route (P)
------------------------------------------------
Proposition 1(d) is a five-way choice, and route (S), the execution-signal
channel, works by *widening the observation* from (x, a) to (x, a, Delta x^ach)
rather than by restricting M or C. It needs no (ER)-type exclusion restriction,
which makes endpoint B the only identification route on real-robot data that
does not lean on an untestable one (12 item 9). That is a claim about what this
code may be used to argue, so it is stated here and not only in the paper.

It also cuts against the pessimistic reading of the common mode: common-mode
unidentifiability holds only *without an anchor*, and on real data the execution
signature is an anchor.

What the held-out split does and does not protect against
---------------------------------------------------------
9.6.6 freezes the split: three institutions are held out and **the split key is
the institution id, not the trajectory id**, because otherwise one institution's
trajectories land on both sides and the held-out test is void.

delta_hat is a per-institution physical parameter -- that institution's delay,
transmission gain and static bias -- so it is estimated from that institution's
own command-to-motion map on both sides of the split. What is held out is the
*procedure*: the basis psi_j, the filter, the segmentation and the referencing
are fixed on the training institutions and then run unchanged on three
institutions that took no part in fixing them. The training-side drop is
therefore reported too, and reported as what it is: an in-sample number, printed
next to the held-out one so the gap between them is visible.

The transfer-only reading -- give the held-out institutions the *mean* of the
training institutions' delta_hat -- is not offered, and the reason is worth
recording: a correction that is identical across institutions leaves a
between-institution variance component almost unchanged, because that component
is already measured around the grand mean. Such a test would return roughly zero
by construction, whatever the truth.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from portal_trace.real.pair import DroidLabId, PairedPoseSeries
from portal_trace.real.segments import SegmentSet
from portal_trace.real.transfer_ratio import (
    PoseComponentGroup,
    SegmentReferencing,
    TCompReading,
    VarianceComponentFit,
    between_lab_variance_component,
    collect_observations,
)

__all__ = [
    "PhysicalBasis",
    "DeltaHat",
    "PSI_BASIS_DROID",
    "fit_delta_hat",
    "subtract_compensation_model",
    "LabSplit",
    "institution_folds",
    "HeldOutFoldResult",
    "EndpointBResult",
    "held_out_lab_test",
    "PREREGISTERED_VARIANCE_DROP",
    "N_HELD_OUT_LABS",
]

PREREGISTERED_VARIANCE_DROP = 0.15
"""On the held-out institutions, the between-lab component falls by >= 15% (10.4.6)."""

N_HELD_OUT_LABS = 3
"""9.6.6 holds out three institutions, keyed by institution id.

Three is also the floor: the endpoint reads a between-*institution* variance
component on the held-out side, which needs more than one institution to exist
at all, and 10.7.2's warning about REML at three groups applies to it. The
before/after ratio is taken with the same estimator on both sides, so the bias
is common to numerator and denominator, but the fold-level number is noisy and
`EndpointBResult` reports every fold rather than only their median.
"""


@dataclass(frozen=True)
class PhysicalBasis:
    """The psi_j of route (S): c = sum_j delta_j psi_j(x).

    Each entry maps a paired series to a (T, D) field. Route (S) requires the
    compensation field to be *parameterised on a physical basis* -- that is what
    makes delta identifiable from the command-to-motion map G_delta rather than
    from a decomposition of the action field (12 item 9). A free-form regression
    onto arbitrary features would not be route (S).
    """

    names: tuple[str, ...]
    fields: tuple[Callable[[PairedPoseSeries], np.ndarray], ...]

    def __post_init__(self) -> None:
        if len(self.names) != len(self.fields):
            raise ValueError("every basis field needs a name")
        if not 3 <= len(self.names) <= 6:
            raise ValueError(
                f"10.4.6 fixes delta_hat at 3 to 6 parameters per institution "
                f"(delay, transmission gain, static bias); got {len(self.names)}"
            )

    def design(self, series: PairedPoseSeries, *, columns: Sequence[int]) -> np.ndarray:
        """(T * len(columns), J) design matrix, one column per basis field."""
        blocks = []
        for fn in self.fields:
            f = np.asarray(fn(series), dtype=float)
            if f.shape[0] != series.n_frames:
                raise ValueError(
                    f"basis field returned {f.shape[0]} frames, series has "
                    f"{series.n_frames}"
                )
            blocks.append(f[:, columns].reshape(-1, 1))
        return np.concatenate(blocks, axis=1)


def _delay_field(series: PairedPoseSeries) -> np.ndarray:
    """Pure delay: the achieved rate, whose coefficient is the lag in seconds.

    For ach_t = cmd_{t-ell}, a first-order expansion gives
    cmd_t - ach_t ~ ell * d(ach)/dt, so regressing the residual on the achieved
    rate recovers ell directly in seconds.
    """
    return series.ach_rate


def _gain_field(series: PairedPoseSeries) -> np.ndarray:
    """Transmission gain error: proportional to the achieved displacement itself."""
    return series.ach


def _bias_field(series: PairedPoseSeries) -> np.ndarray:
    """Static bias: a constant offset per pose component."""
    return np.ones_like(series.ach)


PSI_BASIS_DROID = PhysicalBasis(
    names=("delay_s", "gain_error", "static_bias"),
    fields=(_delay_field, _gain_field, _bias_field),
)
"""The three-parameter basis 10.4.6 names: delay, transmission gain, static bias."""


@dataclass(frozen=True)
class DeltaHat:
    """One institution's fitted delta, on a named physical basis."""

    lab: DroidLabId
    basis_names: tuple[str, ...]
    coefficients: np.ndarray
    n_frames: int
    residual_rms: float

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.basis_names, self.coefficients.tolist(), strict=True))


def fit_delta_hat(
    series_by_lab: Mapping[DroidLabId, Sequence[PairedPoseSeries]],
    *,
    basis: PhysicalBasis,
    component_group: PoseComponentGroup,
) -> dict[DroidLabId, DeltaHat]:
    """Fit delta_hat per institution from the command-to-motion residual.

    The target is cmd - ach, the execution residual. 10.4.6 excludes the
    *variance ratio* of that residual as an endpoint -- it measures the plant's
    failure to execute, i.e. delta before compensation -- but the residual is
    exactly the right regressand for estimating delta itself, which is what
    endpoint B needs and what makes the two uses consistent rather than
    contradictory.
    """
    cols = list(component_group.value)
    out: dict[DroidLabId, DeltaHat] = {}
    for lab, series_list in series_by_lab.items():
        designs, targets = [], []
        for series in series_list:
            designs.append(basis.design(series, columns=cols))
            targets.append(series.residual[:, cols].reshape(-1))
        if not designs:
            raise ValueError(f"institution {lab} has no series to fit delta_hat from")
        x = np.concatenate(designs, axis=0)
        y = np.concatenate(targets, axis=0)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        out[DroidLabId(lab)] = DeltaHat(
            lab=DroidLabId(lab),
            basis_names=basis.names,
            coefficients=beta,
            n_frames=int(x.shape[0] // len(cols)),
            residual_rms=float(np.sqrt(np.mean(resid**2))),
        )
    return out


def subtract_compensation_model(
    series: PairedPoseSeries,
    delta_hat: DeltaHat,
    *,
    basis: PhysicalBasis,
    component_group: PoseComponentGroup,
) -> np.ndarray:
    """cmd - sum_j delta_hat_j psi_j(x): the command with the model taken off.

    Returns an array shaped like `series.cmd`; components outside the group are
    passed through untouched, so the caller cannot accidentally compare a
    partially corrected translation against an uncorrected rotation.
    """
    cols = list(component_group.value)
    design = basis.design(series, columns=cols)
    modelled = (design @ delta_hat.coefficients).reshape(series.n_frames, len(cols))
    out = series.cmd.astype(float, copy=True)
    out[:, cols] -= modelled
    return out


@dataclass(frozen=True)
class LabSplit:
    """One 9.6.6 split of the institutions: three held out, the rest training."""

    training: tuple[DroidLabId, ...]
    held_out: tuple[DroidLabId, ...]

    def __post_init__(self) -> None:
        overlap = set(self.training) & set(self.held_out)
        if overlap:
            raise ValueError(
                f"institutions {sorted(overlap)} are on both sides of the split; 9.6.6 "
                "keys the split on the institution id precisely so that this cannot "
                "happen"
            )
        if len(self.held_out) < 2:
            raise ValueError(
                f"the held-out side carries {len(self.held_out)} institution(s); a "
                "between-institution variance component needs at least two, and 9.6.6 "
                f"preregisters {N_HELD_OUT_LABS}"
            )


def institution_folds(
    labs: Sequence[DroidLabId], *, n_held_out: int = N_HELD_OUT_LABS
) -> tuple[LabSplit, ...]:
    """Partition the institutions into held-out folds of `n_held_out` (9.6.6).

    Deterministic in the sorted institution ids, and every institution is held
    out exactly once. When the count does not divide, the last fold absorbs the
    remainder rather than dropping it -- with DROID's 13 institutions and folds
    of 3 that gives 3, 3, 3, 4.
    """
    ordered = [DroidLabId(str(x)) for x in sorted({str(x) for x in labs})]
    if len(ordered) < n_held_out + 2:
        raise ValueError(
            f"{len(ordered)} institutions cannot be split into a training side and a "
            f"held-out side of {n_held_out}; 10.4.6 runs this over DROID's 13"
        )
    n_folds = len(ordered) // n_held_out
    folds: list[LabSplit] = []
    for f in range(n_folds):
        lo = f * n_held_out
        hi = len(ordered) if f == n_folds - 1 else lo + n_held_out
        held = tuple(ordered[lo:hi])
        folds.append(
            LabSplit(
                training=tuple(x for x in ordered if x not in set(held)),
                held_out=held,
            )
        )
    return tuple(folds)


@dataclass(frozen=True)
class HeldOutFoldResult:
    """One fold: the between-institution component before and after subtraction.

    `training_*` is the same quantity on the institutions the procedure was
    fixed on. It is in sample and labelled as such; it is carried so that the
    held-out number can be read against it instead of on its own.

    The fits themselves are kept rather than only their variances, because a
    fold whose *before* component sits at the REML boundary has no denominator:
    a ratio taken against it is arbitrarily large and arbitrarily signed. Such a
    fold reports `admissible = False` and a NaN drop, which is 11.3's zero
    variance component trigger reaching this endpoint.
    """

    held_out: tuple[DroidLabId, ...]
    before_fit: VarianceComponentFit
    after_fit: VarianceComponentFit
    training_before_fit: VarianceComponentFit
    training_after_fit: VarianceComponentFit
    delta_hat: dict[DroidLabId, dict[str, float]]
    n_held_out_segments: int

    @staticmethod
    def _drop(before: VarianceComponentFit, after: VarianceComponentFit) -> float:
        if before.degenerate:
            return float("nan")
        return 1.0 - after.sigma2_lab / before.sigma2_lab

    @property
    def sigma2_lab_before(self) -> float:
        return self.before_fit.sigma2_lab

    @property
    def sigma2_lab_after(self) -> float:
        return self.after_fit.sigma2_lab

    @property
    def admissible(self) -> bool:
        return not self.before_fit.degenerate

    @property
    def relative_drop(self) -> float:
        return self._drop(self.before_fit, self.after_fit)

    @property
    def training_relative_drop(self) -> float:
        return self._drop(self.training_before_fit, self.training_after_fit)

    @property
    def meets_prediction(self) -> bool:
        return self.admissible and self.relative_drop >= PREREGISTERED_VARIANCE_DROP


@dataclass(frozen=True)
class EndpointBResult:
    """Endpoint B over the 9.6.6 institution folds."""

    folds: tuple[HeldOutFoldResult, ...]
    basis_names: tuple[str, ...]
    component_group: PoseComponentGroup
    referencing: SegmentReferencing
    reading: TCompReading

    @property
    def admissible_folds(self) -> tuple[HeldOutFoldResult, ...]:
        return tuple(r for r in self.folds if r.admissible)

    @property
    def median_relative_drop(self) -> float:
        """Median over the admissible folds; NaN when none of them is."""
        drops = [r.relative_drop for r in self.admissible_folds]
        return float(np.median(drops)) if drops else float("nan")

    @property
    def median_training_relative_drop(self) -> float:
        drops = [
            r.training_relative_drop
            for r in self.folds
            if not r.training_before_fit.degenerate
        ]
        return float(np.median(drops)) if drops else float("nan")

    @property
    def n_meeting_prediction(self) -> int:
        return sum(1 for r in self.folds if r.meets_prediction)

    def describe(self) -> str:
        n_ok = len(self.admissible_folds)
        lines = [
            f"endpoint B (route S anchoring), basis {list(self.basis_names)}, "
            f"{self.component_group.name}, {self.reading.name}",
            f"  median held-out drop in the between-institution component: "
            f"{self.median_relative_drop:+.1%} "
            f"(preregistered >= {PREREGISTERED_VARIANCE_DROP:.0%}), over the "
            f"{n_ok}/{len(self.folds)} admissible folds",
            f"  median training-side drop (in sample, for contrast): "
            f"{self.median_training_relative_drop:+.1%}",
            f"  folds meeting the prediction: {self.n_meeting_prediction}/"
            f"{len(self.folds)}",
        ]
        for r in self.folds:
            if r.admissible:
                lines.append(
                    f"    held out {list(r.held_out)}: {r.sigma2_lab_before:.4g} -> "
                    f"{r.sigma2_lab_after:.4g} ({r.relative_drop:+.1%}); training side "
                    f"{r.training_relative_drop:+.1%}"
                )
            else:
                lines.append(
                    f"    held out {list(r.held_out)}: INADMISSIBLE -- the "
                    f"between-institution component before subtraction is "
                    f"{r.sigma2_lab_before:.4g}, an intraclass share of "
                    f"{r.before_fit.intraclass_share:.2g}, which is the REML boundary "
                    "and not a denominator (10.4.6 first-layer exit; 11.3 zero "
                    "variance component trigger)"
                )
        lines.append(
            "  route (S): identification comes from widening the observation to "
            "(x, a, Delta x^ach); no (ER)-type exclusion restriction is used "
            "(proposition 1(d), 12 item 9)"
        )
        lines.append(
            "  observational association only (10.4.6): (A2) excludes human "
            "teleoperation and DROID is human teleoperation"
        )
        return "\n".join(lines)


def _fit_and_subtract(
    segment_sets: Sequence[SegmentSet],
    *,
    basis: PhysicalBasis,
    component_group: PoseComponentGroup,
) -> tuple[dict[DroidLabId, DeltaHat], dict[str, np.ndarray]]:
    """Fit delta_hat per institution on this side of the split and subtract it."""
    by_lab: dict[DroidLabId, list[PairedPoseSeries]] = {}
    for sset in segment_sets:
        by_lab.setdefault(DroidLabId(str(sset.series.lab)), []).append(sset.series)
    fitted = fit_delta_hat(by_lab, basis=basis, component_group=component_group)
    override = {
        sset.series.episode_id: subtract_compensation_model(
            sset.series,
            fitted[DroidLabId(str(sset.series.lab))],
            basis=basis,
            component_group=component_group,
        )
        for sset in segment_sets
    }
    return fitted, override


def held_out_lab_test(
    segment_sets: Sequence[SegmentSet],
    *,
    basis: PhysicalBasis,
    component_group: PoseComponentGroup,
    referencing: SegmentReferencing,
    n_phase_bins: int,
    reading: TCompReading = TCompReading.WHOLE_SEGMENT,
    n_held_out: int = N_HELD_OUT_LABS,
) -> EndpointBResult:
    """Endpoint B: subtract psi_j(delta_hat) and read the drop on the held-out side.

    For each 9.6.6 fold the two sides are analysed separately and never mixed:
    delta_hat is fitted per institution within a side, the model is subtracted
    from that side's commands, and the between-institution variance component of
    the commands is recomputed on that side alone. The held-out side's number is
    the endpoint; the training side's is reported next to it as the in-sample
    contrast.

    `n_held_out` is 9.6.6's split size rather than a tuning knob; it is a
    parameter only so that a smaller pilot set can be analysed with the fold
    structure intact, and the fold sizes are recorded in the result.
    """
    labs = sorted({DroidLabId(str(s.series.lab)) for s in segment_sets})
    pre_window_only = reading is TCompReading.PRE_LATENCY_WINDOW

    def component(
        side: Sequence[SegmentSet], cmd_override: dict[str, np.ndarray] | None
    ) -> VarianceComponentFit:
        obs = collect_observations(
            side,
            pre_window_only=pre_window_only,
            component_group=component_group,
            referencing=referencing,
            n_phase_bins=n_phase_bins,
            cmd_override=cmd_override,
        )
        return between_lab_variance_component(
            values=obs.cmd,
            lab=obs.lab,
            task_category=obs.task_category,
            phase_bin=obs.phase_bin,
        )

    results: list[HeldOutFoldResult] = []
    for split in institution_folds(labs, n_held_out=n_held_out):
        held = set(split.held_out)
        held_side = [s for s in segment_sets if DroidLabId(str(s.series.lab)) in held]
        train_side = [s for s in segment_sets if DroidLabId(str(s.series.lab)) not in held]

        fitted_held, override_held = _fit_and_subtract(
            held_side, basis=basis, component_group=component_group
        )
        _fitted_train, override_train = _fit_and_subtract(
            train_side, basis=basis, component_group=component_group
        )
        results.append(
            HeldOutFoldResult(
                held_out=split.held_out,
                before_fit=component(held_side, None),
                after_fit=component(held_side, override_held),
                training_before_fit=component(train_side, None),
                training_after_fit=component(train_side, override_train),
                delta_hat={lab: d.as_dict() for lab, d in fitted_held.items()},
                n_held_out_segments=sum(s.n_segments for s in held_side),
            )
        )

    return EndpointBResult(
        folds=tuple(results),
        basis_names=basis.names,
        component_group=component_group,
        referencing=referencing,
        reading=reading,
    )
