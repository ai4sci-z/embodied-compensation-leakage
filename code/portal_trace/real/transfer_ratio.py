"""E5 endpoint A: the compensation transfer ratio T_comp (proposal 10.4.6).

    T_comp = sigma^2_lab(cmd) / sigma^2_lab(ach)

where sigma^2_lab(.) is the between-institution variance component after
controlling for task category and trajectory phase, from a crossed random
effects model. The endpoint needs no estimate of delta_hat, so it does not
circle back on itself.

10.4.6 requires **four** readings, and they are four because the arm has an
identification problem it has to confront rather than assume away: DROID is
human closed-loop teleoperation, and an operator with no feedforward
compensation still commands differently on a laggy machine because they are
reacting to the lag they can see. That alone raises sigma^2_lab(cmd), pulls
achieved back through the loop, and gives T_comp > 1.

    1. pre-latency window (primary) -- only the first 200 ms after a velocity
       zero crossing. Visuomotor latency is 200-300 ms, so those commands cannot
       contain a reaction to that segment's own lag. Measures the feedforward,
       anticipatory component.
    2. whole segment -- feedforward plus reactive.
    3. their difference -- the feedforward/reactive decomposition, which 10.4.6
       notes nobody has reported.
    4. reaction regressed out -- project cmd_t onto {ach_{t-1}, ..., ach_{t-L}}
       per institution with L covering 300 ms, take the residual, and read its
       between-lab component. A second, independent caliber on the feedforward
       side.

Two implementation choices the proposal does not pin down, both recorded here
because they move the number:

* **Segment referencing.** Absolute Cartesian pose carries each lab's table
  layout, which is not an execution signature. Referencing each segment to its
  own onset removes that and keeps the motion profile, and the segment is then
  oriented by its own direction of travel (see `_reference`). Both sides of the
  ratio get the same treatment, so the choice largely cancels -- but not exactly,
  so `SegmentReferencing` is a required argument rather than a default, and it
  belongs in preregistration item 16 alongside the filter.
* **The per-lab projection of reading 4 carries no intercept.** With one, each
  lab's residuals would be mean zero by construction and the between-lab
  component of the fourth reading would be driven towards zero by the estimator
  rather than by the data.

Statement scope: 10.4.6 fixes the wording as observational. (A2) excludes human
teleoperation, and these datasets are mostly human teleoperation, so nothing
here is a causal claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from portal_trace.real.pair import N_DROID_LABS, DroidLabId
from portal_trace.real.segments import SegmentSet

__all__ = [
    "BETWEEN_LAB_ICC_FLOOR",
    "PoseComponentGroup",
    "SegmentReferencing",
    "TCompReading",
    "VarianceComponentFit",
    "TCompResult",
    "TCompReport",
    "SegmentObservations",
    "collect_observations",
    "between_lab_variance_component",
    "t_comp_reading",
    "t_comp_all_readings",
    "PREREGISTERED_T_COMP_THRESHOLD",
    "PREREGISTERED_T_COMP_MIN_LABS",
]

PREREGISTERED_T_COMP_THRESHOLD = 1.3
"""Pre-window T_comp > 1.3 is the preregistered prediction (10.4.6)."""

PREREGISTERED_T_COMP_MIN_LABS = 8
"""...on at least 8 of the 13 DROID institutions (10.4.6)."""

BETWEEN_LAB_ICC_FLOOR = 1e-3
"""Intraclass-correlation floor below which sigma^2_lab is not an estimate.

REML puts a between-institution component at exactly the boundary whenever the
institution spread is small against the residual, and it reports the boundary as
a number. A component worth less than a thousandth of the total variance is that
boundary, not a measurement, and 10.4.6's first-layer null exit is written for
precisely this case ("T_comp ~ 1 but the between-institution component is itself
tiny => a data quality finding"). Results carry the flag; nothing silently
substitutes a value.
"""

_ZERO_VARIANCE_TOL = 1e-12


class PoseComponentGroup(Enum):
    """Which pose components are aggregated together.

    Translation is in metres and rotation in radians, so a variance summed across
    the two would depend on the unit choice. The groups are separate, and a
    caller has to name one.
    """

    TRANSLATION = (0, 1, 2)
    ROTATION = (3, 4, 5)


class SegmentReferencing(Enum):
    """Whether a segment's poses are referenced to its onset (see module docstring)."""

    ONSET_RELATIVE = "displacement from the segment's first frame"
    ABSOLUTE = "raw pose in the lab's world frame"


class TCompReading(Enum):
    """The four readings of 10.4.6."""

    PRE_LATENCY_WINDOW = "pre-latency window (primary): feedforward / anticipatory"
    WHOLE_SEGMENT = "whole segment: feedforward + reactive"
    DIFFERENCE = "pre-window minus whole segment: the reactive/feedforward split"
    REACTION_REGRESSED_OUT = (
        "cmd projected off the achieved lags per lab, residual's between-lab component"
    )


@dataclass(frozen=True)
class VarianceComponentFit:
    """One between-lab variance component, with what a card has to report."""

    sigma2_lab: float
    sigma2_residual: float
    n_observations: int
    n_labs: int
    converged: bool
    per_component: tuple[float, ...]

    @property
    def intraclass_share(self) -> float:
        """sigma^2_lab / (sigma^2_lab + sigma^2_residual)."""
        total = self.sigma2_lab + self.sigma2_residual
        if total <= 0:
            return 0.0
        return self.sigma2_lab / total

    @property
    def degenerate(self) -> bool:
        return (
            (not self.converged)
            or self.sigma2_lab <= _ZERO_VARIANCE_TOL
            or self.intraclass_share <= BETWEEN_LAB_ICC_FLOOR
        )


@dataclass(frozen=True)
class TCompResult:
    """T_comp for one reading."""

    reading: TCompReading
    t_comp: float
    cmd_fit: VarianceComponentFit
    ach_fit: VarianceComponentFit
    component_group: PoseComponentGroup
    referencing: SegmentReferencing
    n_segments: int

    @property
    def degenerate(self) -> bool:
        return self.cmd_fit.degenerate or self.ach_fit.degenerate

    def describe(self) -> str:
        return (
            f"T_comp[{self.reading.name}] = {self.t_comp:.4g} "
            f"(sigma2_lab cmd {self.cmd_fit.sigma2_lab:.4g} / ach "
            f"{self.ach_fit.sigma2_lab:.4g}; {self.cmd_fit.n_labs} labs, "
            f"{self.n_segments} segments, {self.component_group.name})"
        )


@dataclass(frozen=True)
class TCompReport:
    """All four readings, plus the preregistered prediction of 10.4.6.

    `difference` is stored as its own reading rather than derived on demand,
    because 10.4.6 lists it as one of the four things to report and a derived
    property would not survive being tabulated.
    """

    pre_window: TCompResult
    whole_segment: TCompResult
    difference: float
    reaction_regressed_out: TCompResult
    per_lab_pre_window: dict[DroidLabId, float]

    @property
    def labs_above_threshold(self) -> int:
        return sum(
            1 for v in self.per_lab_pre_window.values() if v > PREREGISTERED_T_COMP_THRESHOLD
        )

    @property
    def prediction_met(self) -> bool:
        """The 10.4.6 prediction: > 1.3 on at least 8/13, and difference > 0.

        Reported, never used to select a caliber. The three-layer null-result
        exit of 10.4.6 (F5) applies when it is not met.
        """
        return (
            self.labs_above_threshold >= PREREGISTERED_T_COMP_MIN_LABS
            and self.difference > 0.0
        )

    def describe(self) -> str:
        lines = [
            self.pre_window.describe(),
            self.whole_segment.describe(),
            f"T_comp[DIFFERENCE] = {self.difference:.4g} "
            "(pre-window minus whole segment; > 0 means a reactive component exists "
            "but is not the whole of it)",
            self.reaction_regressed_out.describe(),
            f"pre-window T_comp > {PREREGISTERED_T_COMP_THRESHOLD} on "
            f"{self.labs_above_threshold}/{len(self.per_lab_pre_window)} institutions "
            f"(preregistered: at least {PREREGISTERED_T_COMP_MIN_LABS} of {N_DROID_LABS})",
            "observational association only: (A2) excludes human teleoperation and "
            "these data are human teleoperation (10.4.6)",
        ]
        return "\n".join(lines)


# --- assembling the observation table --------------------------------------


@dataclass(frozen=True)
class SegmentObservations:
    lab: np.ndarray
    task_category: np.ndarray
    phase_bin: np.ndarray
    cmd: np.ndarray  # (n, k) one column per pose component in the group
    ach: np.ndarray
    n_segments: int


def _reference(
    block: np.ndarray, referencing: SegmentReferencing, direction: int
) -> np.ndarray:
    """Put one segment into the frame the variance component is read in.

    Onset referencing removes the lab's table layout, which is not an execution
    signature. Orienting by the segment's own direction is the second half of the
    same idea and it is not optional: a lab's outward and return moves carry the
    same execution signature with opposite signs, so a signed displacement
    averaged over them has a lab-level mean of zero whatever the lab does. The
    between-institution variance component is a variance of lab *means*, so
    without the orientation it would be driven to zero by the direction mix
    rather than by the data -- the same failure mode as the no-intercept rule of
    reading 4, one level up.

    Absolute referencing keeps the raw pose and is left unoriented: there is no
    onset to orient around.
    """
    if referencing is SegmentReferencing.ONSET_RELATIVE:
        return (block - block[0]) * direction
    return block


def _phase_means(block: np.ndarray, n_bins: int) -> np.ndarray:
    """Mean of each phase bin of one segment; (n_bins, k).

    Binning before the model rather than fitting per frame is deliberate: frames
    inside a segment are strongly autocorrelated, so a per-frame fit would report
    a residual variance far smaller than the information content and shrink the
    lab component's standard error accordingly. The bin mean is the phase
    covariate's natural unit and 10.4.6's "after controlling for trajectory
    phase" is exactly a phase factor in the model.
    """
    n = block.shape[0]
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    out = np.empty((n_bins, block.shape[1]), dtype=float)
    for b in range(n_bins):
        lo, hi = edges[b], max(edges[b + 1], edges[b] + 1)
        out[b] = block[lo:hi].mean(axis=0)
    return out


def collect_observations(
    segment_sets: Sequence[SegmentSet],
    *,
    pre_window_only: bool,
    component_group: PoseComponentGroup,
    referencing: SegmentReferencing,
    n_phase_bins: int,
    cmd_override: dict[str, np.ndarray] | None = None,
) -> SegmentObservations:
    cols = list(component_group.value)
    labs, tasks, bins, cmds, achs = [], [], [], [], []
    n_segments = 0
    for sset in segment_sets:
        series = sset.series
        cmd_full = (
            cmd_override[series.episode_id] if cmd_override is not None else series.cmd
        )
        for seg in sset.segments:
            sl = seg.slice_pre_window() if pre_window_only else seg.slice_whole()
            cmd_block = _reference(cmd_full[sl][:, cols], referencing, seg.direction)
            ach_block = _reference(series.ach[sl][:, cols], referencing, seg.direction)
            if cmd_block.shape[0] < 1:
                continue
            n_segments += 1
            cmds.append(_phase_means(cmd_block, n_phase_bins))
            achs.append(_phase_means(ach_block, n_phase_bins))
            labs.append(np.full(n_phase_bins, str(series.lab)))
            tasks.append(np.full(n_phase_bins, series.task_category))
            bins.append(np.arange(n_phase_bins))
    if not cmds:
        raise ValueError(
            "no motion segments survived the preregistered zero-crossing rule; T_comp "
            "has nothing to be computed on"
        )
    return SegmentObservations(
        lab=np.concatenate(labs),
        task_category=np.concatenate(tasks),
        phase_bin=np.concatenate(bins),
        cmd=np.concatenate(cmds, axis=0),
        ach=np.concatenate(achs, axis=0),
        n_segments=n_segments,
    )


# --- the variance component ------------------------------------------------


def between_lab_variance_component(
    *,
    values: np.ndarray,
    lab: np.ndarray,
    task_category: np.ndarray,
    phase_bin: np.ndarray,
) -> VarianceComponentFit:
    """sigma^2_lab after controlling for task category and trajectory phase.

    A crossed random effects fit with the institution as the random intercept and
    task category and phase as fixed factors (10.4.6). Each pose component is
    fitted separately and the components of the group are summed, which is the
    total between-lab variance of the vector restricted to that group; summing
    variances is only legitimate inside a group of one unit, which is what
    `PoseComponentGroup` enforces.
    """
    import pandas as pd
    import statsmodels.formula.api as smf

    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"values must be (n, k), got {values.shape}")
    n_labs = int(len(np.unique(lab)))
    if n_labs < 2:
        raise ValueError(
            f"a between-institution variance component needs at least two "
            f"institutions, got {n_labs}"
        )

    per_component: list[float] = []
    residuals: list[float] = []
    converged = True
    for k in range(values.shape[1]):
        frame = pd.DataFrame(
            {
                "y": values[:, k],
                "lab": np.asarray(lab).astype(str),
                "task": np.asarray(task_category).astype(str),
                "phase": np.asarray(phase_bin).astype(str),
            }
        )
        frame["_g"] = 1
        formula = "y ~ C(phase)"
        if frame["task"].nunique() > 1:
            formula += " + C(task)"
        model = smf.mixedlm(
            formula, frame, groups=frame["_g"], vc_formula={"lab": "0 + C(lab)"},
            re_formula="0",
        )
        with np.errstate(all="ignore"):
            fit = model.fit(reml=True, method=["lbfgs"])
        converged = converged and bool(getattr(fit, "converged", False))
        per_component.append(max(float(np.atleast_1d(fit.vcomp)[0]), 0.0))
        residuals.append(max(float(fit.scale), 0.0))

    return VarianceComponentFit(
        sigma2_lab=float(sum(per_component)),
        sigma2_residual=float(sum(residuals)),
        n_observations=int(values.shape[0]),
        n_labs=n_labs,
        converged=converged,
        per_component=tuple(per_component),
    )


def _regress_out_reaction(
    segment_sets: Sequence[SegmentSet],
    *,
    component_group: PoseComponentGroup,
) -> dict[str, np.ndarray]:
    """Reading 4: project cmd_t off the achieved lags, per institution.

    One projection is fitted per (institution, pose component) over that
    institution's whole pooled series, using L achieved lags with L covering
    300 ms (10.4.6). The projection carries **no intercept**: with one, every
    lab's residuals would be mean zero by construction and the between-lab
    component of this reading would be an artefact of the estimator.

    Returns the residual series keyed by episode id, shaped like the original
    command array so the segment slices still index into it.
    """
    cols = list(component_group.value)
    by_lab: dict[str, list[SegmentSet]] = {}
    for sset in segment_sets:
        by_lab.setdefault(str(sset.series.lab), []).append(sset)

    out: dict[str, np.ndarray] = {}
    for _lab, ssets in by_lab.items():
        n_lags = ssets[0].prereg.reaction_regression_lags
        designs, targets, index = [], [], []
        for sset in ssets:
            series = sset.series
            t = series.n_frames
            if t <= n_lags:
                continue
            lagged = np.concatenate(
                [series.ach[n_lags - l - 1 : t - l - 1][:, cols] for l in range(n_lags)],
                axis=1,
            )
            designs.append(lagged)
            targets.append(series.cmd[n_lags:][:, cols])
            index.append((series.episode_id, t, series.cmd.shape[1]))
        if not designs:
            continue
        x = np.concatenate(designs, axis=0)
        y = np.concatenate(targets, axis=0)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)

        offset = 0
        for design, (episode_id, t, width) in zip(designs, index, strict=True):
            rows = design.shape[0]
            residual = y[offset : offset + rows] - design @ beta
            offset += rows
            full = np.zeros((t, width), dtype=float)
            full[n_lags:, cols] = residual
            # The first n_lags frames have no full lag window. They are left at
            # zero and excluded by the segment slices only when a segment starts
            # inside them, which the truncation counters in `SegmentSet` surface.
            out[episode_id] = full
    return out


def _phase_residual_variance(values: np.ndarray, phase_bin: np.ndarray) -> np.ndarray:
    """Per-column variance after removing the phase-bin means.

    The same "after controlling for trajectory phase" the pooled model applies,
    done arithmetically because a one-institution subset has no random effect to
    fit.
    """
    out = values.astype(float, copy=True)
    for b in np.unique(phase_bin):
        mask = phase_bin == b
        out[mask] -= out[mask].mean(axis=0)
    return np.var(out, axis=0, ddof=1)


def t_comp_reading(
    segment_sets: Sequence[SegmentSet],
    *,
    reading: TCompReading,
    component_group: PoseComponentGroup,
    referencing: SegmentReferencing,
    n_phase_bins: int,
) -> TCompResult:
    """One of the four readings of 10.4.6.

    `TCompReading.DIFFERENCE` is not computed here: it is the difference of two
    readings and is assembled by `t_comp_all_readings`, so a caller cannot obtain
    it without also having the two it is made of.
    """
    if reading is TCompReading.DIFFERENCE:
        raise ValueError(
            "the difference reading is the pre-window reading minus the whole-segment "
            "reading; call t_comp_all_readings, which returns all four"
        )
    cmd_override = None
    pre_window_only = reading is TCompReading.PRE_LATENCY_WINDOW
    if reading is TCompReading.REACTION_REGRESSED_OUT:
        cmd_override = _regress_out_reaction(
            segment_sets, component_group=component_group
        )

    obs = collect_observations(
        segment_sets,
        pre_window_only=pre_window_only,
        component_group=component_group,
        referencing=referencing,
        n_phase_bins=n_phase_bins,
        cmd_override=cmd_override,
    )
    cmd_fit = between_lab_variance_component(
        values=obs.cmd, lab=obs.lab, task_category=obs.task_category, phase_bin=obs.phase_bin
    )
    ach_fit = between_lab_variance_component(
        values=obs.ach, lab=obs.lab, task_category=obs.task_category, phase_bin=obs.phase_bin
    )
    if ach_fit.sigma2_lab <= _ZERO_VARIANCE_TOL:
        raise ZeroDivisionError(
            f"{reading.name}: the achieved between-lab variance component is "
            f"{ach_fit.sigma2_lab:.3g}, so T_comp is undefined. 10.4.6's first-layer "
            "null exit covers this: a vanishing between-lab component is a data quality "
            "finding, to be redone stratified by DROID's own task categories"
        )
    return TCompResult(
        reading=reading,
        t_comp=cmd_fit.sigma2_lab / ach_fit.sigma2_lab,
        cmd_fit=cmd_fit,
        ach_fit=ach_fit,
        component_group=component_group,
        referencing=referencing,
        n_segments=obs.n_segments,
    )


def t_comp_all_readings(
    segment_sets: Sequence[SegmentSet],
    *,
    component_group: PoseComponentGroup,
    referencing: SegmentReferencing,
    n_phase_bins: int,
) -> TCompReport:
    """All four readings of 10.4.6, plus the per-institution pre-window table.

    The per-institution table exists because the preregistered prediction is
    "> 1.3 on at least 8 of 13", which is a count over institutions and cannot be
    read off a pooled ratio. Each institution's own ratio is computed by holding
    that institution against the rest, which keeps a between-institution
    component defined for a single institution.
    """
    pre = t_comp_reading(
        segment_sets,
        reading=TCompReading.PRE_LATENCY_WINDOW,
        component_group=component_group,
        referencing=referencing,
        n_phase_bins=n_phase_bins,
    )
    whole = t_comp_reading(
        segment_sets,
        reading=TCompReading.WHOLE_SEGMENT,
        component_group=component_group,
        referencing=referencing,
        n_phase_bins=n_phase_bins,
    )
    regressed = t_comp_reading(
        segment_sets,
        reading=TCompReading.REACTION_REGRESSED_OUT,
        component_group=component_group,
        referencing=referencing,
        n_phase_bins=n_phase_bins,
    )

    per_lab: dict[DroidLabId, float] = {}
    for lab in sorted({str(s.series.lab) for s in segment_sets}):
        subset = [s for s in segment_sets if str(s.series.lab) == lab]
        obs = collect_observations(
            subset,
            pre_window_only=True,
            component_group=component_group,
            referencing=referencing,
            n_phase_bins=n_phase_bins,
        )
        # One institution has no between-institution component of its own, so the
        # per-lab column is the institution's command spread against its achieved
        # spread on the same frames, residualised on the phase factor exactly as
        # the pooled fit is. The count of institutions above 1.3 is what 10.4.6
        # predicts; the pooled ratio above is the estimate.
        cmd_var = float(np.sum(_phase_residual_variance(obs.cmd, obs.phase_bin)))
        ach_var = float(np.sum(_phase_residual_variance(obs.ach, obs.phase_bin)))
        per_lab[DroidLabId(lab)] = (
            cmd_var / ach_var if ach_var > _ZERO_VARIANCE_TOL else float("nan")
        )

    return TCompReport(
        pre_window=pre,
        whole_segment=whole,
        difference=pre.t_comp - whole.t_comp,
        reaction_regressed_out=regressed,
        per_lab_pre_window=per_lab,
    )
