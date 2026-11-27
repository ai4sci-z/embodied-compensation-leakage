"""The five-regime partition, its joint calibration, occupancy and z probe.

Implements proposal 9.7.1-9.7.5 and the reporting rules of 11.3 and 11.4.

The partition
-------------
r1 reversal, r2 quasi-static load, r3 free-space fast, r4 contact, r0 the rest,
decided from the short-history window only -- K_hist frames of joint velocity,
the tau_g(q) channel and the wrist force/torque channel -- because 9.7.2
requires the label to be available to the policy and to the estimator at the
same time. Exclusivity follows the priority r4 > r1 > r2 > r3 > r0, so that R
is a partition and not a cover: proposition 1(e')(i) needs the per-regime
conditional means to be uncorrelated, which a cover would break.

Joint calibration
-----------------
The five thresholds are one coupled system, not five independent quantiles.
f_th appears in r4 and in r3, and the exclusivity priority makes eps_v change
how many frames are left for r2 and r3. 9.7.1 therefore specifies coordinate
descent in priority order, with the occupancy denominator held at |F| (all
frames) while the numerator is searched only over the frames the higher-
priority regimes have not taken. Target occupancy 0.15, admissible band
[0.05, 0.40], at most five rounds; a regime that cannot be pushed into the band
merges into r0 and the descent is re-run.

Thresholds are calibrated per (task, family), frozen into the lock file and
never adjusted per defect -- otherwise the regime definition would depend on
delta and gamma_{delta,r} would not be comparable across defects. Nothing in
this module reads a threshold from a literal: `RegimeThresholds.from_lock` is
the only production entry point, and the 9.7.1 table values appear once, as the
initialisation of the descent, behind a provenance tag that `label_frames`
refuses to use for anything else.

Specification inconsistency found while implementing
----------------------------------------------------
The 9.7.1 pseudo-code annotates the bisection directions as "f_th, eps_qs,
eps_fs decreasing, eps_v, tau_min increasing". Read against the decision rules
in the same table, three of the five are the wrong way round: occupancy is
*increasing* in eps_qs (r2 is ||qdot|| < eps_qs) and *decreasing* in eps_v and
tau_min (both appear as lower bounds a frame must exceed). Rather than encode
either version, `_solve_threshold` establishes the direction from the data at
the bracket ends and then bisects, which is correct under both readings.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier

from portal_trace.diagnostics.design_graph import (
    N0_PRIMARY_FRAMES,
    N0_SENSITIVITY_FRAMES,
    CellObservation,
    RegimeRestriction,
)
from portal_trace.lockfile import REQUIRED_MEASUREMENTS, LockFile
from portal_trace.types import REGIME_ORDINAL, Regime

__all__ = [
    "K_HIST_PRIMARY",
    "K_HIST_LADDER",
    "STEPS_SINCE_CROSSING",
    "TARGET_OCCUPANCY",
    "OCCUPANCY_BAND",
    "MAX_CALIBRATION_ROUNDS",
    "R1_PROXY_MIN_BALANCED_ACCURACY",
    "Z_PROBE_BASELINE_MARGIN",
    "THRESHOLD_NAMES",
    "Provenance",
    "RegimeThresholds",
    "RegimeSignals",
    "RegimeFeatures",
    "RegimeCalibration",
    "ZProbeResult",
    "TVReading",
    "label_frames",
    "calibrate_regime_thresholds",
    "occupancy_mass",
    "within_regime_tv",
    "cross_embodiment_tv",
    "per_regime_connectivity",
    "regime_proxy_quality",
    "run_z_probe",
]

# 9.7.1: the primary short-history window and the ablation ladder of 9.7.3.
K_HIST_PRIMARY = 5
K_HIST_LADDER = (1, 2, 5, 10)
# 9.7.1: "t is within n steps of the nearest zero crossing", n = 3, justified
# there by b_j = 0.5 deg of backlash taking ~0.9 control steps at 0.2 rad/s.
STEPS_SINCE_CROSSING = 3

# 9.7.1: the occupancy target and the admissible band. Below 0.05 the cell has
# too few effective samples (9.7.5's n_min); above 0.40 the within-regime
# oscillation osc_r(c_delta) is no longer small and (A5)(i) fails.
TARGET_OCCUPANCY = 0.15
OCCUPANCY_BAND = (0.05, 0.40)
MAX_CALIBRATION_ROUNDS = 5
_CONVERGENCE_TOLERANCE = 1e-3

# 9.7.3: the r1 rule is a proxy for the hidden meshing state; below this
# balanced accuracy the per-regime conclusions at that K_hist are exploratory.
R1_PROXY_MIN_BALANCED_ACCURACY = 0.70
# 9.7.3: if the z probe beats the majority-class baseline by less than this,
# z is substantively unobservable at that (defect, level, K_hist).
Z_PROBE_BASELINE_MARGIN = 0.10

# The r2 grid of 9.7.1 ("the product grid of (eps_qs, tau_min)"). The
# resolution is an implementation choice, not a calibrated threshold: it only
# bounds how close to m* the chosen grid point can land.
_R2_GRID_POINTS = 24
_BISECTION_ITERATIONS = 60

THRESHOLD_NAMES = ("f_th", "eps_v", "eps_qs", "tau_min", "eps_fs")

# Priority order of the exclusivity rule, minus r0 which is the complement.
_CALIBRATION_ORDER = (
    Regime.R4_CONTACT,
    Regime.R1_REVERSAL,
    Regime.R2_QUASISTATIC,
    Regime.R3_FREESPACE,
)

# Which threshold each regime's occupancy is solved for. f_th also appears in
# r3's out-of-contact screen, which is the coupling 9.7.1 cites as the reason
# the five cannot be calibrated independently.
_THRESHOLDS_OF: dict[Regime, frozenset[str]] = {
    Regime.R4_CONTACT: frozenset({"f_th"}),
    Regime.R1_REVERSAL: frozenset({"eps_v"}),
    Regime.R2_QUASISTATIC: frozenset({"eps_qs", "tau_min"}),
    Regime.R3_FREESPACE: frozenset({"eps_fs"}),
}


class Provenance(str, Enum):
    """Where a threshold set came from.

    `CALIBRATION_PRIOR` is the 9.7.1 table read as the starting point of the
    coordinate descent. It is not a usable threshold set: 9.7.1 marks four of
    the five values as pending calibration, and 11.3 forbids a measurement from
    having a default.
    """

    LOCK = "lock"
    CALIBRATION_PRIOR = "calibration_prior"
    CALIBRATION_ITERATE = "calibration_iterate"


@dataclass(frozen=True)
class RegimeThresholds:
    """The five thresholds of the 9.7.1 decision table.

    `unmeasured` names the thresholds the descent never calibrated, which
    happens when their regime merged into r0: with no occupancy target left
    there is nothing to solve for, and keeping the 9.7.1 prior in the slot
    would turn an initialisation into a value the rest of the pipeline reads.
    Those slots hold NaN and `value_of` raises on them.
    """

    f_th: float
    eps_v: float
    eps_qs: float
    tau_min: float
    eps_fs: float
    provenance: Provenance
    unmeasured: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        unknown = sorted(self.unmeasured - set(THRESHOLD_NAMES))
        if unknown:
            raise KeyError(f"unknown threshold names in `unmeasured`: {unknown}")
        for name in THRESHOLD_NAMES:
            if name in self.unmeasured:
                continue
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"threshold {name} must be finite and non-negative, got {value}")

    @property
    def measured_names(self) -> tuple[str, ...]:
        return tuple(n for n in THRESHOLD_NAMES if n not in self.unmeasured)

    def value_of(self, name: str) -> float:
        if name not in THRESHOLD_NAMES:
            raise KeyError(f"unknown threshold {name!r}; the five are {THRESHOLD_NAMES}")
        if name in self.unmeasured:
            raise ValueError(
                f"threshold {name} was never calibrated because its regime merged into r0 "
                "(9.7.1); there is no value to read and the 9.7.1 prior is not a substitute"
            )
        return float(getattr(self, name))

    @classmethod
    def from_lock(
        cls,
        lock: LockFile,
        *,
        task: str,
        family: str,
        names: Sequence[str] = THRESHOLD_NAMES,
    ) -> RegimeThresholds:
        """The only production source of thresholds (9.7.1, 11.3).

        `names` is the set the calibration actually froze; when a regime merged
        into r0 its threshold is absent from the lock and must be absent here
        too, rather than being filled in.
        """
        unknown = sorted(set(names) - set(THRESHOLD_NAMES))
        if unknown:
            raise KeyError(f"unknown threshold names: {unknown}")
        decl = REQUIRED_MEASUREMENTS["regime.threshold.{name}.{task}.{family}"]
        values = {
            name: float(lock.get(decl, name=name, task=task, family=family))
            for name in names
        }
        for name in THRESHOLD_NAMES:
            values.setdefault(name, math.nan)
        return cls(
            provenance=Provenance.LOCK,
            unmeasured=frozenset(set(THRESHOLD_NAMES) - set(names)),
            **values,
        )

    def write_to_lock(
        self,
        lock: LockFile,
        *,
        task: str,
        family: str,
        source_commit: str,
        notes: str = "",
        extra: Mapping[str, Any] | None = None,
    ) -> tuple[str, ...]:
        """Freeze the calibrated thresholds into `registry/lock.json` (9.7.1).

        Refuses to write anything that has not been through the descent: the
        9.7.1 prior is an initialisation, and writing it would make a pending
        calibration look measured. Uncalibrated slots are skipped, and the
        names that were written are returned so the reader can ask for exactly
        those.
        """
        if self.provenance is Provenance.CALIBRATION_PRIOR:
            raise ValueError(
                "refusing to freeze the 9.7.1 prior as a measurement; run "
                "calibrate_regime_thresholds first"
            )
        decl = REQUIRED_MEASUREMENTS["regime.threshold.{name}.{task}.{family}"]
        written = self.measured_names
        for name in written:
            lock.put(
                decl,
                self.value_of(name),
                source_commit=source_commit,
                notes=notes,
                extra=dict(extra or {}),
                name=name,
                task=task,
                family=family,
            )
        return written

    def replace_threshold(self, name: str, value: float) -> RegimeThresholds:
        if name not in THRESHOLD_NAMES:
            raise KeyError(f"unknown threshold {name!r}; the five are {THRESHOLD_NAMES}")
        return replace(
            self,
            **{name: float(value)},
            provenance=Provenance.CALIBRATION_ITERATE,
            unmeasured=self.unmeasured - {name},
        )

    def drop(self, names: Sequence[str]) -> RegimeThresholds:
        """Mark thresholds as never calibrated and blank their slots."""
        dropped = set(names)
        unknown = sorted(dropped - set(THRESHOLD_NAMES))
        if unknown:
            raise KeyError(f"unknown threshold names: {unknown}")
        return replace(
            self,
            **{name: math.nan for name in dropped},
            unmeasured=self.unmeasured | frozenset(dropped),
        )

    def as_dict(self) -> dict[str, float | None]:
        return {
            name: (None if name in self.unmeasured else float(getattr(self, name)))
            for name in THRESHOLD_NAMES
        }


# 9.7.1 table: the prior values the coordinate descent starts from. Four of the
# five are marked as pending calibration there, which is why this is an
# initialisation and not a default (contract 3.1).
CALIBRATION_PRIOR = RegimeThresholds(
    f_th=2.0,
    eps_v=0.02,
    eps_qs=0.05,
    tau_min=5.0,
    eps_fs=0.15,
    provenance=Provenance.CALIBRATION_PRIOR,
)


@dataclass(frozen=True)
class RegimeSignals:
    """The raw channels the 9.7.1 rules read, all of them policy-visible.

    `qdot_history` is (N, K_hist, J) with the last window position being the
    current frame. `wrist_ft` is the six-dimensional force/torque channel added
    to proprio in 9.7.2; the r3/r4 rules use its force triple, which is what
    carries the newton-valued threshold f_th.
    """

    qdot_history: np.ndarray
    tau_g: np.ndarray
    wrist_ft: np.ndarray
    ee_velocity: np.ndarray

    def __post_init__(self) -> None:
        q = np.asarray(self.qdot_history, dtype=np.float64)
        if q.ndim != 3:
            raise ValueError(f"qdot_history must be (N, K_hist, J), got shape {q.shape}")
        n = q.shape[0]
        for name, arr, width in (
            ("tau_g", self.tau_g, None),
            ("wrist_ft", self.wrist_ft, 6),
            ("ee_velocity", self.ee_velocity, 3),
        ):
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim != 2 or a.shape[0] != n:
                raise ValueError(f"{name} must be ({n}, d), got shape {a.shape}")
            if width is not None and a.shape[1] != width:
                raise ValueError(
                    f"{name} must have {width} columns (9.7.2), got {a.shape[1]}"
                )

    @property
    def n_frames(self) -> int:
        return int(np.asarray(self.qdot_history).shape[0])

    @property
    def k_hist(self) -> int:
        return int(np.asarray(self.qdot_history).shape[1])


@dataclass(frozen=True)
class RegimeFeatures:
    """The five scalars the decision rules threshold, precomputed once.

    Reducing each rule to "scalar against threshold" is what makes the
    coordinate descent tractable: the calibration evaluates occupancy tens of
    thousands of times, and recomputing sign crossings each time would dominate
    the cost. `reversal_strength` is the largest velocity magnitude at which a
    qualifying sign flip happened inside the recency window, so r1 holds exactly
    when it exceeds eps_v.
    """

    wrist_force_norm: np.ndarray
    reversal_strength: np.ndarray
    qdot_norm: np.ndarray
    tau_g_inf: np.ndarray
    ee_speed: np.ndarray
    k_hist: int
    steps_since_crossing: int

    @property
    def n_frames(self) -> int:
        return int(self.wrist_force_norm.shape[0])

    @classmethod
    def from_signals(
        cls,
        signals: RegimeSignals,
        *,
        steps_since_crossing: int = STEPS_SINCE_CROSSING,
    ) -> RegimeFeatures:
        q = np.asarray(signals.qdot_history, dtype=np.float64)
        n, k_hist, _ = q.shape
        if k_hist < 2:
            raise ValueError(
                f"the reversal rule compares consecutive frames and needs K_hist >= 2, "
                f"got {k_hist}"
            )
        if steps_since_crossing < 1:
            raise ValueError(f"n must be at least 1 step, got {steps_since_crossing}")
        earlier = q[:, :-1, :]
        later = q[:, 1:, :]
        flipped = np.sign(earlier) * np.sign(later) < 0
        # Distance in control steps from the current frame to the crossing that
        # sits between window positions i and i+1.
        lag = (k_hist - 1) - np.arange(k_hist - 1)
        recent = lag <= steps_since_crossing
        qualifying = flipped & recent[None, :, None]
        both_sides = np.minimum(np.abs(earlier), np.abs(later))
        strength = np.where(qualifying, both_sides, 0.0)
        return cls(
            wrist_force_norm=np.linalg.norm(
                np.asarray(signals.wrist_ft, dtype=np.float64)[:, :3], axis=1
            ),
            reversal_strength=strength.reshape(n, -1).max(axis=1),
            qdot_norm=np.linalg.norm(q[:, -1, :], axis=1),
            tau_g_inf=np.max(np.abs(np.asarray(signals.tau_g, dtype=np.float64)), axis=1),
            ee_speed=np.linalg.norm(np.asarray(signals.ee_velocity, dtype=np.float64), axis=1),
            k_hist=k_hist,
            steps_since_crossing=steps_since_crossing,
        )

    def rule_mask(self, regime: Regime, thresholds: RegimeThresholds) -> np.ndarray:
        """The bare decision rule of 9.7.1, before exclusivity is applied."""
        if regime is Regime.R4_CONTACT:
            return self.wrist_force_norm > thresholds.value_of("f_th")
        if regime is Regime.R1_REVERSAL:
            return self.reversal_strength > thresholds.value_of("eps_v")
        if regime is Regime.R2_QUASISTATIC:
            return (self.qdot_norm < thresholds.value_of("eps_qs")) & (
                self.tau_g_inf > thresholds.value_of("tau_min")
            )
        if regime is Regime.R3_FREESPACE:
            fast = self.ee_speed > thresholds.value_of("eps_fs")
            if "f_th" in thresholds.unmeasured:
                # r4 merged into r0, so f_th has no occupancy target and was not
                # calibrated. 9.7.1 gives no other target for it, so the
                # out-of-contact screen is dropped rather than run on a prior.
                return fast
            return fast & (self.wrist_force_norm <= thresholds.value_of("f_th"))
        if regime is Regime.R0_OTHER:
            taken = np.zeros(self.n_frames, dtype=bool)
            for r in _CALIBRATION_ORDER:
                if _THRESHOLDS_OF[r] & thresholds.unmeasured:
                    continue
                taken |= self.rule_mask(r, thresholds)
            return ~taken
        raise ValueError(f"unhandled regime {regime}")


def label_frames(
    features: RegimeFeatures,
    thresholds: RegimeThresholds,
    *,
    active_regimes: Sequence[Regime] = _CALIBRATION_ORDER,
    for_calibration: bool = False,
) -> np.ndarray:
    """Assign one regime ordinal per frame under the 9.7.1 exclusivity order.

    `active_regimes` is the priority order after any merge into r0 (9.7.1's
    under-occupancy rule); regimes absent from it are folded into r0 and the
    effective regime count drops accordingly.
    """
    if thresholds.provenance is not Provenance.LOCK and not for_calibration:
        raise ValueError(
            f"thresholds carry provenance {thresholds.provenance.value}; production "
            "labelling reads the frozen values with RegimeThresholds.from_lock "
            "(9.7.1, 11.3). Pass for_calibration=True only from the calibrator itself"
        )
    order = [r for r in _CALIBRATION_ORDER if r in set(active_regimes)]
    if not order:
        raise ValueError("no regimes are active; the partition would be r0 alone")
    blocked = {r: sorted(_THRESHOLDS_OF[r] & thresholds.unmeasured) for r in order}
    missing = {r.value: names for r, names in blocked.items() if names}
    if missing:
        raise ValueError(
            f"these regimes are active but their thresholds were never calibrated: "
            f"{missing}. Either drop them from `active_regimes` or run the calibration"
        )
    labels = np.full(features.n_frames, REGIME_ORDINAL[Regime.R0_OTHER], dtype=np.uint8)
    assigned = np.zeros(features.n_frames, dtype=bool)
    for regime in order:
        mask = features.rule_mask(regime, thresholds) & ~assigned
        labels[mask] = REGIME_ORDINAL[regime]
        assigned |= mask
    return labels


def occupancy_mass(labels: np.ndarray) -> dict[Regime, float]:
    """m_r, with |F| as the denominator so the five masses sum to one."""
    labels = np.asarray(labels)
    if labels.size == 0:
        raise ValueError("no frames; occupancy mass is undefined")
    counts = np.bincount(labels.astype(np.int64), minlength=len(Regime))
    total = float(labels.size)
    return {r: float(counts[REGIME_ORDINAL[r]]) / total for r in Regime}


@dataclass(frozen=True)
class TVReading:
    """An empirical total variation distance, on a partition the caller states.

    TV is a supremum over measurable sets, so the value on any one partition is
    a *lower* bound on the true distance. That is the useful direction for A-19,
    where the quantity exists to show that a term the additivity argument treats
    as negligible is not: a lower bound that is already large settles it, and a
    small one settles nothing. `null_median` says how large the reading would be
    between two samples of these sizes drawn from the same distribution, which
    on a fine partition and few frames is not small.
    """

    value: float
    n_bins: int
    n_a: int
    n_b: int
    null_median: float
    null_upper: float

    @property
    def excess(self) -> float:
        """How far the reading clears the same-distribution null."""
        return self.value - self.null_median

    @property
    def separated(self) -> bool:
        """Whether the two samples are distinguishable on this partition at all."""
        return self.value > self.null_upper

    def describe(self) -> str:
        verdict = "separated" if self.separated else "within the null band"
        return (
            f"TV = {self.value:.4f} on {self.n_bins} bins from n = ({self.n_a}, "
            f"{self.n_b}); same-distribution null median {self.null_median:.4f}, "
            f"95th percentile {self.null_upper:.4f}; {verdict}. A lower bound on "
            "the true TV, since it reads one partition rather than the supremum"
        )


def _tv_on_counts(counts_a: np.ndarray, counts_b: np.ndarray) -> float:
    """Half the L1 distance of two normalised histograms: the [0, 1] convention.

    9.7's TV is read on [0, 1] throughout (proposal 1319), so the factor of two
    that turns the L1 distance into a probability is applied here and nowhere
    else.
    """
    pa = counts_a / counts_a.sum()
    pb = counts_b / counts_b.sum()
    return float(0.5 * np.abs(pa - pb).sum())


def _empirical_tv(
    bins_a: np.ndarray,
    bins_b: np.ndarray,
    *,
    n_bins: int,
    rng: np.random.Generator,
    n_null: int,
) -> TVReading:
    counts_a = np.bincount(bins_a, minlength=n_bins).astype(np.float64)
    counts_b = np.bincount(bins_b, minlength=n_bins).astype(np.float64)
    value = _tv_on_counts(counts_a, counts_b)

    # The same-distribution null, by relabelling the pooled sample. Without it a
    # fine partition on few frames reads as a large distance between a sample
    # and itself, and the A-19 term would look established when it is not.
    #
    # Relabelling is drawn as counts rather than as a permutation of the frames.
    # A random split of the pooled sample into a side of n_a frames puts exactly
    # a multivariate hypergeometric draw from the pooled bin counts on that side,
    # and the statistic reads nothing but those counts -- so this is the same
    # null, exactly, at O(n_null * n_bins) instead of O(n_null * n_frames). The
    # sides here run to millions of frames per regime, which is where permuting
    # the frames stops being affordable.
    n_a = bins_a.size
    pooled_counts = (counts_a + counts_b).astype(np.int64)
    drawn = rng.multivariate_hypergeometric(pooled_counts, n_a, size=n_null).astype(np.float64)
    null = 0.5 * np.abs(drawn / n_a - (pooled_counts - drawn) / bins_b.size).sum(axis=1)
    return TVReading(
        value=value,
        n_bins=n_bins,
        n_a=int(n_a),
        n_b=int(bins_b.size),
        null_median=float(np.median(null)),
        null_upper=float(np.quantile(null, 0.95)),
    )


def _check_partition(bins: np.ndarray, n_bins: int, label: str) -> np.ndarray:
    arr = np.asarray(bins)
    if arr.ndim != 1:
        raise ValueError(f"{label}: the partition is one code per frame, got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(
            f"{label}: the partition must be integer bin codes. A continuous state "
            "has no empirical TV -- the caller chooses the partition, because it "
            "sets how tight every bound in proposition 1(e) is"
        )
    if arr.size and (arr.min() < 0 or arr.max() >= n_bins):
        raise ValueError(
            f"{label}: bin codes span [{arr.min()}, {arr.max()}] but n_bins is {n_bins}"
        )
    return arr.astype(np.int64)


def within_regime_tv(
    labels_a: np.ndarray,
    bins_a: np.ndarray,
    labels_b: np.ndarray,
    bins_b: np.ndarray,
    *,
    regime: Regime,
    n_bins: int,
    rng: np.random.Generator,
    n_null: int = 200,
    n_min: int = N0_PRIMARY_FRAMES,
) -> TVReading:
    """TV between two occupancies *conditioned on* one regime (A-19, 9.7).

    The second term of the 1(e) decomposition. Equal visit frequency cancels the
    occupancy-mass term and leaves this one untouched: two tasks can put the same
    mass in a regime and sit in different parts of it, and the mean compensation
    then still differs. Reported alongside `occupancy_mass`, never instead of it.

    `bins_*` is a per-frame partition code of the state, supplied by the caller.
    """
    ordinal = REGIME_ORDINAL[regime]
    la, lb = np.asarray(labels_a), np.asarray(labels_b)
    ba = _check_partition(bins_a, n_bins, "bins_a")
    bb = _check_partition(bins_b, n_bins, "bins_b")
    for labels, bins, name in ((la, ba, "a"), (lb, bb, "b")):
        if labels.shape != bins.shape:
            raise ValueError(
                f"side {name}: {labels.size} labels against {bins.size} partition "
                "codes; both index the same frames"
            )
    sub_a, sub_b = ba[la == ordinal], bb[lb == ordinal]
    for sub, name in ((sub_a, "a"), (sub_b, "b")):
        if sub.size < n_min:
            raise ValueError(
                f"side {name} has {sub.size} frames in {regime.value}, below the "
                f"{n_min}-frame admission floor of 9.7.5. A conditional TV on fewer "
                "frames than that is dominated by its own sampling noise; drop the "
                "regime from this comparison rather than reading the number"
            )
    return _empirical_tv(sub_a, sub_b, n_bins=n_bins, rng=rng, n_null=n_null)


def cross_embodiment_tv(
    bins_defective: np.ndarray,
    bins_clean: np.ndarray,
    *,
    n_bins: int,
    rng: np.random.Generator,
    n_null: int = 200,
) -> TVReading:
    """TV(rho_{tau,delta}, rho_{tau,0}): same task, the two sides of a twin pair.

    The third of A-19's three quantities, and the one that bounds the task-intent
    interaction A_{tau,delta}. Unconditional on purpose: proposal 1285 is explicit
    that the restricted measure bounds only the within-regime term, and that the
    global TV is what bounds both. Reporting the first two conditions without
    this one gives a bound with one term unmeasured.
    """
    bd = _check_partition(bins_defective, n_bins, "bins_defective")
    bc = _check_partition(bins_clean, n_bins, "bins_clean")
    if bd.size == 0 or bc.size == 0:
        raise ValueError("a twin side with no frames has no occupancy to compare")
    return _empirical_tv(bd, bc, n_bins=n_bins, rng=rng, n_null=n_null)


def _solve_threshold(
    occupancy_of: Callable[[float], float],
    lo: float,
    hi: float,
    target: float,
) -> tuple[float, float, float]:
    """Bisect a monotone occupancy-versus-threshold curve onto `target`.

    Returns (threshold, achieved occupancy, maximum attainable occupancy). The
    direction of monotonicity is read off the bracket ends rather than assumed,
    because the directions annotated in the 9.7.1 pseudo-code disagree with the
    decision rules in the same table for three of the five thresholds.
    """
    occ_lo = occupancy_of(lo)
    occ_hi = occupancy_of(hi)
    best = max(occ_lo, occ_hi)
    if occ_lo == occ_hi:
        return (lo if occ_lo >= occ_hi else hi), occ_lo, best
    decreasing = occ_lo > occ_hi
    if decreasing:
        if target >= occ_lo:
            return lo, occ_lo, best
        if target <= occ_hi:
            return hi, occ_hi, best
    else:
        if target <= occ_lo:
            return lo, occ_lo, best
        if target >= occ_hi:
            return hi, occ_hi, best
    left, right = lo, hi
    for _ in range(_BISECTION_ITERATIONS):
        mid = 0.5 * (left + right)
        occ = occupancy_of(mid)
        if math.isclose(occ, target, rel_tol=0.0, abs_tol=1e-12):
            return mid, occ, best
        if (occ > target) == decreasing:
            left = mid
        else:
            right = mid
    mid = 0.5 * (left + right)
    return mid, occupancy_of(mid), best


@dataclass(frozen=True)
class RegimeCalibration:
    """Result of the 9.7.1 joint calibration for one (task, family)."""

    task: str
    family: str
    thresholds: RegimeThresholds
    occupancy: Mapping[Regime, float]
    under_occupied: tuple[Regime, ...]
    merged_into_r0: tuple[Regime, ...]
    rounds: int
    converged: bool
    n_frames: int

    @property
    def n_regimes_effective(self) -> int:
        """5 minus whatever merged into r0 (9.7.1, reported per 11.4)."""
        return len(Regime) - len(self.merged_into_r0)

    @property
    def active_regimes(self) -> tuple[Regime, ...]:
        merged = set(self.merged_into_r0)
        return tuple(r for r in _CALIBRATION_ORDER if r not in merged)

    @property
    def frozen_threshold_names(self) -> tuple[str, ...]:
        """The thresholds this calibration actually determined."""
        return self.thresholds.measured_names

    def freeze(
        self,
        lock: LockFile,
        *,
        source_commit: str,
        notes: str = "",
    ) -> tuple[str, ...]:
        """Write the calibrated thresholds and their context into the lock file.

        The occupancy the descent landed on and the regimes that merged go into
        the entry's `extra`, because 11.4's model card reports the effective
        regime count next to every per-regime quantity.
        """
        return self.thresholds.write_to_lock(
            lock,
            task=self.task,
            family=self.family,
            source_commit=source_commit,
            notes=notes,
            extra={
                "occupancy": {r.value: self.occupancy[r] for r in Regime},
                "merged_into_r0": [r.value for r in self.merged_into_r0],
                "n_regimes_effective": self.n_regimes_effective,
                "rounds": self.rounds,
                "converged": self.converged,
                "n_frames": self.n_frames,
                "target_occupancy": TARGET_OCCUPANCY,
                "band": list(OCCUPANCY_BAND),
            },
        )

    @property
    def in_band(self) -> dict[Regime, bool]:
        low, high = OCCUPANCY_BAND
        return {
            r: low <= self.occupancy[r] <= high
            for r in _CALIBRATION_ORDER
            if r not in set(self.merged_into_r0)
        }

    @property
    def all_in_band(self) -> bool:
        return all(self.in_band.values())

    def as_rows(self) -> list[dict[str, Any]]:
        rows = [
            {
                "task": self.task,
                "family": self.family,
                "regime": r.value,
                "m": self.occupancy[r],
                "in_band": self.in_band.get(r),
                "merged_into_r0": r in set(self.merged_into_r0),
            }
            for r in Regime
        ]
        rows.append(
            {
                "task": self.task,
                "family": self.family,
                "regime": "-",
                "thresholds": self.thresholds.as_dict(),
                "rounds": self.rounds,
                "converged": self.converged,
                "n_regimes_effective": self.n_regimes_effective,
                "n_frames": self.n_frames,
            }
        )
        return rows


def _descend(
    features: RegimeFeatures,
    active: Sequence[Regime],
    *,
    target: float,
    max_rounds: int,
) -> tuple[RegimeThresholds, dict[Regime, float], set[Regime], int, bool]:
    """One coordinate-descent run over the active regimes, in priority order."""
    n_frames = features.n_frames
    if n_frames == 0:
        raise ValueError("no frames to calibrate on")
    order = [r for r in _CALIBRATION_ORDER if r in set(active)]
    inactive_thresholds = frozenset().union(
        *(_THRESHOLDS_OF[r] for r in _CALIBRATION_ORDER if r not in order)
    ) if len(order) < len(_CALIBRATION_ORDER) else frozenset()
    thresholds = replace(
        CALIBRATION_PRIOR, provenance=Provenance.CALIBRATION_ITERATE
    ).drop(inactive_thresholds)
    under: set[Regime] = set()
    band_low = OCCUPANCY_BAND[0]

    brackets = {
        "f_th": (0.0, float(features.wrist_force_norm.max()) * (1 + 1e-9) + 1e-12),
        "eps_v": (0.0, float(features.reversal_strength.max()) * (1 + 1e-9) + 1e-12),
        "eps_qs": (0.0, float(features.qdot_norm.max()) * (1 + 1e-9) + 1e-12),
        "tau_min": (0.0, float(features.tau_g_inf.max()) * (1 + 1e-9) + 1e-12),
        "eps_fs": (0.0, float(features.ee_speed.max()) * (1 + 1e-9) + 1e-12),
    }

    converged = False
    rounds = 0
    for round_index in range(1, max_rounds + 1):
        rounds = round_index
        previous = thresholds.as_dict()
        remaining = np.ones(n_frames, dtype=bool)
        under = set()
        for regime in order:
            if regime is Regime.R2_QUASISTATIC:
                thresholds, attainable = _fit_r2(
                    features, thresholds, remaining, target, brackets
                )
            else:
                name = {
                    Regime.R4_CONTACT: "f_th",
                    Regime.R1_REVERSAL: "eps_v",
                    Regime.R3_FREESPACE: "eps_fs",
                }[regime]

                occupancy_of = _single_threshold_occupancy(
                    features, thresholds, regime, name, remaining
                )
                lo, hi = brackets[name]
                value, _, attainable = _solve_threshold(occupancy_of, lo, hi, target)
                thresholds = thresholds.replace_threshold(name, value)
            if attainable < band_low:
                under.add(regime)
                # 9.7.1: take the extremal value that maximises occupancy, and
                # let the merge rule deal with the regime afterwards.
                thresholds = _extremal_for(features, thresholds, regime, remaining, brackets)
            remaining &= ~features.rule_mask(regime, thresholds)
        current = thresholds.as_dict()
        deltas = [
            abs(current[k] - previous[k]) / max(abs(previous[k]), 1e-12)
            for k in thresholds.measured_names
            if previous[k] is not None and current[k] is not None
        ]
        if deltas and max(deltas) < _CONVERGENCE_TOLERANCE:
            converged = True
            break

    labels = label_frames(features, thresholds, active_regimes=order, for_calibration=True)
    return thresholds, occupancy_mass(labels), under, rounds, converged


def _single_threshold_occupancy(
    features: RegimeFeatures,
    thresholds: RegimeThresholds,
    regime: Regime,
    name: str,
    remaining: np.ndarray,
) -> Callable[[float], float]:
    """Occupancy of `regime` over the un-taken pool, as a function of one threshold."""
    n_frames = features.n_frames

    def occupancy_of(value: float) -> float:
        trial = thresholds.replace_threshold(name, value)
        return float(np.count_nonzero(features.rule_mask(regime, trial) & remaining)) / n_frames

    return occupancy_of


def _fit_r2(
    features: RegimeFeatures,
    thresholds: RegimeThresholds,
    remaining: np.ndarray,
    target: float,
    brackets: Mapping[str, tuple[float, float]],
) -> tuple[RegimeThresholds, float]:
    """r2 has two thresholds; 9.7.1 fixes them on a product grid.

    eps_qs sets the occupancy and tau_min screens inside it, so the pair is
    chosen as the grid point whose occupancy is closest to m*. Returns the
    updated thresholds and the maximum occupancy attainable on the grid, which
    is what the under-occupancy rule tests.
    """
    n_frames = features.n_frames
    qs_grid = np.quantile(
        features.qdot_norm, np.linspace(0.0, 1.0, _R2_GRID_POINTS)
    )
    tau_grid = np.quantile(
        features.tau_g_inf, np.linspace(0.0, 1.0, _R2_GRID_POINTS)
    )
    qs_grid = np.unique(np.concatenate([qs_grid, [brackets["eps_qs"][1]]]))
    tau_grid = np.unique(np.concatenate([tau_grid, [0.0]]))
    best: tuple[float, float, float] | None = None
    attainable = 0.0
    for eps_qs in qs_grid:
        slow = (features.qdot_norm < eps_qs) & remaining
        if not slow.any():
            continue
        loaded_values = features.tau_g_inf[slow]
        for tau_min in tau_grid:
            occ = float(np.count_nonzero(loaded_values > tau_min)) / n_frames
            attainable = max(attainable, occ)
            distance = abs(occ - target)
            if best is None or distance < best[0]:
                best = (distance, float(eps_qs), float(tau_min))
    if best is None:
        return thresholds, 0.0
    _, eps_qs, tau_min = best
    updated = thresholds.replace_threshold("eps_qs", eps_qs).replace_threshold(
        "tau_min", tau_min
    )
    return updated, attainable


def _extremal_for(
    features: RegimeFeatures,
    thresholds: RegimeThresholds,
    regime: Regime,
    remaining: np.ndarray,
    brackets: Mapping[str, tuple[float, float]],
) -> RegimeThresholds:
    """The threshold value that maximises an under-occupied regime's share."""
    n_frames = features.n_frames
    if regime is Regime.R2_QUASISTATIC:
        return thresholds.replace_threshold(
            "eps_qs", brackets["eps_qs"][1]
        ).replace_threshold("tau_min", 0.0)
    name = {
        Regime.R4_CONTACT: "f_th",
        Regime.R1_REVERSAL: "eps_v",
        Regime.R3_FREESPACE: "eps_fs",
    }[regime]
    candidates = brackets[name]
    scored = []
    for value in candidates:
        trial = thresholds.replace_threshold(name, value)
        occ = float(np.count_nonzero(features.rule_mask(regime, trial) & remaining)) / n_frames
        scored.append((occ, value))
    return thresholds.replace_threshold(name, max(scored)[1])


def calibrate_regime_thresholds(
    features: RegimeFeatures,
    *,
    task: str,
    family: str,
    target: float = TARGET_OCCUPANCY,
    max_rounds: int = MAX_CALIBRATION_ROUNDS,
) -> RegimeCalibration:
    """The 9.7.1 joint calibration, including the merge rule.

    Runs coordinate descent in priority order; if a regime cannot be pushed to
    the 0.05 floor it merges into r0 and the descent is re-run, because the
    remaining frame pool changed.
    """
    if not 0.0 < target < 1.0:
        raise ValueError(f"the occupancy target must lie in (0, 1), got {target}")
    if not OCCUPANCY_BAND[0] <= target <= OCCUPANCY_BAND[1]:
        raise ValueError(
            f"the target {target} is outside the admissible band {OCCUPANCY_BAND} (9.7.1)"
        )
    thresholds, occupancy, under, rounds, converged = _descend(
        features, _CALIBRATION_ORDER, target=target, max_rounds=max_rounds
    )
    merged: tuple[Regime, ...] = ()
    if under:
        merged = tuple(r for r in _CALIBRATION_ORDER if r in under)
        active = [r for r in _CALIBRATION_ORDER if r not in under]
        if not active:
            raise ValueError(
                f"every regime is under-occupied on ({task}, {family}); the partition "
                "collapses to r0 and no per-regime conclusion exists"
            )
        thresholds, occupancy, under_again, rounds_2, converged = _descend(
            features, active, target=target, max_rounds=max_rounds
        )
        rounds += rounds_2
        under = under_again
    return RegimeCalibration(
        task=task,
        family=family,
        thresholds=replace(thresholds, provenance=Provenance.CALIBRATION_ITERATE),
        occupancy=occupancy,
        under_occupied=tuple(r for r in _CALIBRATION_ORDER if r in under),
        merged_into_r0=merged,
        rounds=rounds,
        converged=converged,
        n_frames=features.n_frames,
    )


def per_regime_connectivity(
    observations: Mapping[Regime, Sequence[CellObservation]],
    *,
    n0_levels: Sequence[int] = N0_SENSITIVITY_FRAMES,
    primary_n0: int = N0_PRIMARY_FRAMES,
    heteroscedastic: bool = False,
    key_pairs: Sequence[tuple[str, str]] = (),
) -> dict[str, Any]:
    """9.7.5's per-regime verdict, with the 11.3 sensitivity sweep.

    Reports the primary N_0 = 200 caliber and the {50, 200, 800} sweep, flags
    the regimes whose verdict changes across the sweep, and applies the
    preregistered rule that such regimes are treated at their most conservative
    level. The second, stricter admission rule of 9.7.5 (m >= 0.02 and
    n >= 2000 raw frames) is reported alongside, because the two calibers
    the proposal states for the same set do not agree.

    `key_pairs` are the task pairs of 9.7.5's list P. The section requires the
    component count, lambda_2 and R_eff on every key pair to be reported
    together -- the first two say whether the contrast is identifiable at all
    and the third says how precisely -- so the pairs travel into every row
    rather than being recovered from the graph afterwards.
    """
    if primary_n0 not in n0_levels:
        raise ValueError(
            f"the primary caliber N_0 = {primary_n0} must be one of the swept levels "
            f"{tuple(n0_levels)} so the sensitivity table contains it (11.3)"
        )
    per_level: dict[int, dict[str, RegimeRestriction]] = {}
    for n0 in n0_levels:
        per_level[n0] = {
            regime.value: RegimeRestriction.build(
                regime.value,
                cells,
                admission="n0",
                n0=n0,
                heteroscedastic=heteroscedastic,
            )
            for regime, cells in observations.items()
        }
    occupancy_caliber = {
        regime.value: RegimeRestriction.build(
            regime.value, cells, admission="occupancy", heteroscedastic=heteroscedastic
        )
        for regime, cells in observations.items()
    }
    unstable = []
    for regime in observations:
        statuses = {per_level[n0][regime.value].status for n0 in n0_levels}
        if len(statuses) > 1:
            unstable.append(regime.value)
    reportable = {}
    for regime in observations:
        # Most conservative level wins where the sweep disagrees (11.3).
        reportable[regime.value] = all(
            per_level[n0][regime.value].reportable for n0 in n0_levels
        )
    return {
        "primary_n0": primary_n0,
        "key_pairs": tuple(key_pairs),
        "primary": {k: v.as_row(key_pairs) for k, v in per_level[primary_n0].items()},
        "sensitivity": {
            n0: {k: v.as_row(key_pairs) for k, v in level.items()}
            for n0, level in per_level.items()
        },
        "occupancy_caliber": {
            k: v.as_row(key_pairs) for k, v in occupancy_caliber.items()
        },
        "unstable_across_n0": tuple(unstable),
        "reportable": reportable,
    }


@dataclass(frozen=True)
class ZProbeResult:
    """The (A5)(ii) probe of 9.7.3, per (defect, level, K_hist)."""

    defect: str
    level: int
    k_hist: int
    target: str
    balanced_accuracy: float
    majority_baseline: float
    n_classes: int
    n_test: int

    @property
    def margin(self) -> float:
        return self.balanced_accuracy - self.majority_baseline

    @property
    def z_observable(self) -> bool:
        return self.margin >= Z_PROBE_BASELINE_MARGIN

    @property
    def reading_qualifier(self) -> str:
        if self.z_observable:
            return "z recoverable from the K_hist window"
        return (
            "z substantively unobservable at this (defect, level, K_hist): the leakage "
            "conclusion reads only as a reading under inputs in which z is undetermined, "
            "and must be printed next to E_phi (9.7.3)"
        )

    def as_row(self) -> dict[str, Any]:
        return {
            "defect": self.defect,
            "level": self.level,
            "k_hist": self.k_hist,
            "target": self.target,
            "balanced_accuracy": self.balanced_accuracy,
            "majority_baseline": self.majority_baseline,
            "margin": self.margin,
            "z_observable": self.z_observable,
            "n_classes": self.n_classes,
            "n_test": self.n_test,
            "qualifier": self.reading_qualifier,
        }


def _split_by_episode(
    episode_index: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three-way split by demonstration entry (9.4.10's caliber).

    Splitting by frame would put neighbouring frames of one trajectory on both
    sides and inflate every accuracy in this module.
    """
    episodes = np.unique(episode_index)
    if episodes.size < 3:
        raise ValueError(
            f"a three-way split by demonstration needs at least three episodes, got "
            f"{episodes.size}"
        )
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(episodes)
    n_train = max(1, int(round(0.6 * shuffled.size)))
    n_val = max(1, int(round(0.2 * shuffled.size)))
    if n_train + n_val >= shuffled.size:
        n_train = shuffled.size - 2
        n_val = 1
    train = np.isin(episode_index, shuffled[:n_train])
    val = np.isin(episode_index, shuffled[n_train : n_train + n_val])
    test = np.isin(episode_index, shuffled[n_train + n_val :])
    return train, val, test


def run_z_probe(
    policy_inputs: np.ndarray,
    z_labels: np.ndarray,
    episode_index: np.ndarray,
    *,
    defect: str,
    level: int,
    k_hist: int,
    target: str,
    seed: int,
    max_iter: int = 400,
) -> ZProbeResult:
    """The 9.7.3 probe: can K_hist frames of policy input recover z?

    Two hidden layers of width 128, input restricted to the policy's own
    K_hist-frame window, split three ways by demonstration entry, scored with
    balanced accuracy against the majority-class baseline. The specification is
    fixed by 9.7.3 and is not a tuning surface: the number is only meaningful
    as "what a small probe with the policy's information can do".
    """
    x = np.asarray(policy_inputs, dtype=np.float64)
    y = np.asarray(z_labels)
    episodes = np.asarray(episode_index)
    if x.ndim != 2:
        raise ValueError(f"policy_inputs must be (N, d), got shape {x.shape}")
    if y.shape[0] != x.shape[0] or episodes.shape[0] != x.shape[0]:
        raise ValueError("policy_inputs, z_labels and episode_index must agree on length")
    n_classes = int(np.unique(y).size)
    if n_classes < 2:
        raise ValueError(
            f"z takes a single value on these frames ({np.unique(y)}); a classifier "
            "cannot measure observability of a constant"
        )
    train, val, test = _split_by_episode(episodes, seed)
    fit_mask = train | val
    if np.unique(y[fit_mask]).size < n_classes or np.unique(y[test]).size < 2:
        raise ValueError(
            "the demonstration-level split left a class entirely on one side; balanced "
            "accuracy would be undefined. Report the split rather than reshuffling it"
        )
    model = MLPClassifier(
        hidden_layer_sizes=(128, 128),
        max_iter=max_iter,
        random_state=seed,
    )
    mean = x[fit_mask].mean(axis=0)
    scale = x[fit_mask].std(axis=0)
    scale[scale == 0.0] = 1.0
    model.fit((x[fit_mask] - mean) / scale, y[fit_mask])
    predicted = model.predict((x[test] - mean) / scale)
    values, counts = np.unique(y[fit_mask], return_counts=True)
    majority = np.full(int(test.sum()), values[int(np.argmax(counts))])
    return ZProbeResult(
        defect=defect,
        level=level,
        k_hist=k_hist,
        target=target,
        balanced_accuracy=float(balanced_accuracy_score(y[test], predicted)),
        majority_baseline=float(balanced_accuracy_score(y[test], majority)),
        n_classes=n_classes,
        n_test=int(test.sum()),
    )


def regime_proxy_quality(
    predicted_r1: np.ndarray, true_engagement: np.ndarray, *, k_hist: int
) -> dict[str, Any]:
    """How well the r1 rule proxies the hidden meshing state (9.7.3).

    Balanced accuracy, F1 and the confusion matrix, per the reporting list in
    9.7.3, plus the preregistered rule that below 0.70 balanced accuracy the
    per-regime conclusions at that K_hist are exploratory only.
    """
    predicted = np.asarray(predicted_r1).astype(bool)
    truth = np.asarray(true_engagement).astype(bool)
    if predicted.shape != truth.shape:
        raise ValueError(f"shapes disagree: {predicted.shape} vs {truth.shape}")
    if np.unique(truth).size < 2:
        raise ValueError(
            "the true meshing state is constant on these frames; a proxy cannot be scored"
        )
    accuracy = float(balanced_accuracy_score(truth, predicted))
    return {
        "k_hist": k_hist,
        "balanced_accuracy": accuracy,
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, predicted).tolist(),
        "threshold": R1_PROXY_MIN_BALANCED_ACCURACY,
        "confirmatory": accuracy >= R1_PROXY_MIN_BALANCED_ACCURACY,
    }
