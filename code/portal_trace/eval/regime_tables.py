"""P1's three mandatory outputs (proposal 10.3.2, format fixed by 9.7.4/9.7.5).

10.3.2 closes with the list: "P1's mandatory outputs are therefore three -- the
per-regime occupancy-mass table, the within-regime distribution distance, and
the cross-embodiment same-task TV." The three are produced here.

Output 1, the occupancy table
-----------------------------
9.7.4 fixes both the size and the columns. The size is 8 x 36 x 5 = 1440 rows:
eight tasks (six core plus two held out, 9.5.2), thirty-six defect
configurations (six morphology families x six core defects, 9.6.5) and five
regimes (9.7.1). That product is stated there as "the single source of truth for
the whole table's size", so `NOMINAL_OCCUPANCY_ROWS` derives it from the same
three factors rather than writing 1440 down. The columns are quoted verbatim:

    task, defect, family, regime, m, n_frames, n_eff, tau_corr, n_demos,
    m_ci_lo, m_ci_hi, osc_r, included

with `merged_from` added because the same paragraph requires a merged regime to
appear as an `r0` row carrying that note, and requires the realised row count to
be reported next to the nominal 1440.

The CI is a trajectory-level block bootstrap at B = 2000, not a per-frame one:
neighbouring frames inside one demonstration are strongly autocorrelated, and a
frame-level resample would report an interval a factor of sqrt(tau_corr) too
narrow on the very quantity tau_corr exists to correct.

Output 2, the per-regime connectivity verdict
---------------------------------------------
`diagnostics.regimes.per_regime_connectivity` already computes it; what this
module adds is the reporting shape 9.7.5 demands. Three things have to appear
together -- the component count, lambda_2(G|_r) and R_eff on every key task pair
-- and the two admission calibers have to appear side by side, because the
proposal states both for the same edge set and they are not equivalent: N_0 raw
frames (swept over 50 / 200 / 800, primary 200) against m >= 0.02 and n >= 2000.
The stricter pair carries the main narrative and the N_0 sweep is the
sensitivity upper bound. The deliverable W2 names is the "do not report" list,
so it is computed rather than left to a reader of the rows.

Output 3, the cross-embodiment TV
---------------------------------
TV(rho_{tau,delta}, rho_{tau,0}) per (task, configuration). 10.3.2 calls it the
only measurement of the first term of proposition 1(e)'s bias bound, which is
the task-scale term and usually the dominant one.

osc_r
-----
9.7.4 asks for osc_r(c_delta) in every row so the reader can judge (A5)(i), and
the symbol table at 14.4 defines it as the Chebyshev radius of the compensation
field's range restricted to regime r. The exact Chebyshev radius is a smallest-
enclosing-ball problem in six dimensions over millions of points. What is
computed here is the radius about the mean, which is the enclosing radius of a
feasible centre and therefore satisfies rad_r <= osc_r_reported <= 2 rad_r. The
direction matters: osc_r enters 1(e')'s bias bound as an upper bound, so
over-reporting keeps the bound true and under-reporting would silently turn a
valid bound into a false one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from portal_trace.diagnostics.design_graph import (
    M_MIN_OCCUPANCY,
    N0_PRIMARY_FRAMES,
    N0_SENSITIVITY_FRAMES,
    N_MIN_RAW_FRAMES,
    CellObservation,
)
from portal_trace.diagnostics.regimes import cross_embodiment_tv, per_regime_connectivity
from portal_trace.eval.calibration_tables import CalibrationTable
from portal_trace.lockfile import REQUIRED_MEASUREMENTS, LockFile
from portal_trace.registry.tasks import CORE_TASKS, HELDOUT_TASKS
from portal_trace.registry.twins import N_TWIN_CELLS
from portal_trace.types import REGIME_ORDINAL, Regime

__all__ = [
    "NOMINAL_TASKS",
    "NOMINAL_CONFIGURATIONS",
    "NOMINAL_OCCUPANCY_ROWS",
    "BLOCK_BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_INTERVAL",
    "MissingRegimeInput",
    "OccupancyCell",
    "CrossEmbodimentTVCell",
    "configuration_id",
    "oscillation_radius",
    "build_occupancy_table",
    "build_connectivity_report",
    "build_cross_embodiment_tv_table",
]

# 9.7.4's three factors. The eight tasks are 9.5.2's six core plus two held out,
# which leaves out the two unmodified nut environments 9.5.4 keeps for P0; the
# thirty-six configurations are 9.6.5's six families x six core defects, which is
# the same 36 that `registry.twins` counts as cells (contract 3.6).
NOMINAL_TASKS = len(CORE_TASKS) + len(HELDOUT_TASKS)
NOMINAL_CONFIGURATIONS = N_TWIN_CELLS
NOMINAL_OCCUPANCY_ROWS = NOMINAL_TASKS * NOMINAL_CONFIGURATIONS * len(Regime)

if NOMINAL_OCCUPANCY_ROWS != 1440:
    raise AssertionError(
        f"9.7.4 fixes the occupancy table at 8 x 36 x 5 = 1440 rows and calls that "
        f"product the single source of truth for the table's size; the registries give "
        f"{NOMINAL_TASKS} x {NOMINAL_CONFIGURATIONS} x {len(Regime)} = "
        f"{NOMINAL_OCCUPANCY_ROWS}"
    )
if len(HELDOUT_TASKS) != 2:
    raise AssertionError(
        f"9.5.2 puts two tasks outside every training arm and 9.7.4 counts them among "
        f"the eight; the registry declares {len(HELDOUT_TASKS)}"
    )

# 9.7.4: the CI comes from a trajectory-level block bootstrap at B = 2000.
BLOCK_BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_INTERVAL = (0.025, 0.975)


class MissingRegimeInput(LookupError):
    """A P1 output was requested and the frames behind it are not there."""

    def __init__(self, output: str, what: str, produced_by: str) -> None:
        self.output = output
        self.what = what
        self.produced_by = produced_by
        super().__init__(
            f"P1 output {output!r} cannot be built: {what}. It is produced by "
            f"{produced_by}"
        )


def configuration_id(family: str, defect: str) -> str:
    """The name of one of 9.6.5's thirty-six defect configurations.

    9.6.5 makes a configuration the (family, defect) pair, and 11.4's lock key
    for tau_corr is scoped by ``{task}.{defect}.{regime}``. Those two only agree
    if the `defect` slot holds the configuration: tau_corr is measured per cell
    per regime (9.7.5), and a cell is a (task, configuration) pair, so a key
    carrying the bare defect code would be one quantity standing for six.
    """
    return f"{family}:{defect}"


def oscillation_radius(values: np.ndarray) -> float:
    """osc_r(c_delta): the enclosing radius of the field's range about its mean.

    An upper bound on the Chebyshev radius, within a factor of two of it. See
    the module docstring for why the bound is taken in this direction.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"c_delta values must be (n_frames, d), got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("no frames in this regime, so the field has no range here")
    centre = arr.mean(axis=0)
    return float(np.max(np.linalg.norm(arr - centre, axis=1)))


@dataclass(frozen=True)
class OccupancyCell:
    """One (task, family, defect) cell of the 9.7.4 table, per demonstration.

    Frames arrive grouped by demonstration and not concatenated, because the
    block bootstrap resamples demonstrations: flattening first would throw away
    the only structure the interval depends on.

    `merged_into_r0` names the regimes 9.7.1's under-occupancy rule folded into
    r0 for this cell's (task, family). Those regimes get no row of their own and
    the r0 row records them, which is what makes the realised row count fall
    below 1440 in a way a reader can account for.
    """

    task: str
    family: str
    defect: str
    episode_labels: tuple[np.ndarray, ...]
    episode_c_true: tuple[np.ndarray, ...]
    merged_into_r0: tuple[Regime, ...] = ()

    def __post_init__(self) -> None:
        if not self.episode_labels:
            raise ValueError(
                f"{self.task}/{self.family}/{self.defect}: a cell with no demonstrations "
                "has no occupancy and no bootstrap"
            )
        if len(self.episode_labels) != len(self.episode_c_true):
            raise ValueError(
                f"{self.task}/{self.family}/{self.defect}: {len(self.episode_labels)} "
                f"label arrays against {len(self.episode_c_true)} c_true arrays"
            )
        for i, (labels, c_true) in enumerate(
            zip(self.episode_labels, self.episode_c_true, strict=True)
        ):
            if labels.ndim != 1:
                raise ValueError(f"episode {i}: labels must be one per frame")
            if c_true.ndim != 2 or c_true.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"episode {i}: c_true is {c_true.shape} against {labels.shape[0]} "
                    "labelled frames; both index the same frames"
                )
        if Regime.R0_OTHER in self.merged_into_r0:
            raise ValueError("r0 is the complement and cannot be merged into itself")

    @property
    def configuration(self) -> str:
        return configuration_id(self.family, self.defect)

    @property
    def reported_regimes(self) -> tuple[Regime, ...]:
        merged = set(self.merged_into_r0)
        return tuple(r for r in Regime if r not in merged)

    @property
    def n_demos(self) -> int:
        return len(self.episode_labels)

    @property
    def counts(self) -> np.ndarray:
        """(n_demos, n_regimes) frame counts, the bootstrap's sufficient statistic."""
        return np.stack(
            [
                np.bincount(labels.astype(np.int64), minlength=len(Regime))
                for labels in self.episode_labels
            ]
        )


def _bootstrap_interval(counts: np.ndarray, ordinal: int, rng: np.random.Generator) -> tuple[
    float, float
]:
    """Percentile interval for m_r under a trajectory-level resample (9.7.4).

    The ratio is recomputed on each resample rather than averaged over
    demonstrations: m is a frame share, so a long demonstration weighs more, and
    a mean over per-demonstration shares would be a different estimand.
    """
    n_demos = counts.shape[0]
    draws = rng.integers(0, n_demos, size=(BLOCK_BOOTSTRAP_RESAMPLES, n_demos))
    resampled = counts[draws]
    numerator = resampled[:, :, ordinal].sum(axis=1)
    denominator = resampled.sum(axis=(1, 2))
    if np.any(denominator == 0):
        raise ValueError(
            "a bootstrap resample drew only empty demonstrations; the cell's frame "
            "counts are degenerate"
        )
    shares = numerator / denominator
    lo, hi = BOOTSTRAP_INTERVAL
    return float(np.quantile(shares, lo)), float(np.quantile(shares, hi))


def build_occupancy_table(
    cells: Sequence[OccupancyCell], *, lock: LockFile, seed: int
) -> CalibrationTable:
    """P1 output 1: the per-regime occupancy-mass table of 9.7.4.

    tau_corr is read from the lock file per (task, configuration, regime) and
    has no default: it sets n_eff, and 9.7.5 makes n_eff the edge weight of
    every design-graph quantity downstream.

    `seed` fixes the bootstrap draw, because 9.1.6 makes every published number
    reproducible from its address and an interval drawn from an unseeded
    generator is not.
    """
    if not cells:
        raise MissingRegimeInput(
            "regime_occupancy_mass",
            "no labelled demonstrations were supplied",
            "the 9.7.1 labelling run over the collected demonstrations",
        )
    decl = REQUIRED_MEASUREMENTS["stats.tau_corr.{task}.{defect}.{regime}"]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for cell in cells:
        counts = cell.counts
        total = int(counts.sum())
        if total == 0:
            raise ValueError(
                f"{cell.task}/{cell.configuration}: the cell has demonstrations but no "
                "frames, so no occupancy mass is defined on it"
            )
        c_true = np.concatenate(cell.episode_c_true, axis=0)
        labels = np.concatenate(cell.episode_labels, axis=0)
        for regime in cell.reported_regimes:
            ordinal = REGIME_ORDINAL[regime]
            n_frames = int(counts[:, ordinal].sum())
            tau_corr = float(
                lock.get(
                    decl,
                    task=cell.task,
                    defect=cell.configuration,
                    regime=regime.value,
                )
            )
            if tau_corr <= 0.0:
                raise ValueError(
                    f"tau_corr for {cell.task}/{cell.configuration}/{regime.value} is "
                    f"{tau_corr}; n_eff divides by it"
                )
            m = n_frames / total
            ci_lo, ci_hi = _bootstrap_interval(counts, ordinal, rng)
            in_regime = c_true[labels == ordinal]
            rows.append(
                {
                    "task": cell.task,
                    "defect": cell.defect,
                    "family": cell.family,
                    "regime": regime.value,
                    "m": m,
                    "n_frames": n_frames,
                    "n_eff": n_frames / tau_corr,
                    "tau_corr": tau_corr,
                    "n_demos": int(np.count_nonzero(counts[:, ordinal])),
                    "m_ci_lo": ci_lo,
                    "m_ci_hi": ci_hi,
                    "osc_r": (
                        oscillation_radius(in_regime) if in_regime.shape[0] else None
                    ),
                    "included": m >= M_MIN_OCCUPANCY and n_frames >= N_MIN_RAW_FRAMES,
                    "merged_from": (
                        [r.value for r in cell.merged_into_r0]
                        if regime is Regime.R0_OTHER
                        else []
                    ),
                }
            )
    notes = [
        f"nominal size {NOMINAL_TASKS} tasks x {NOMINAL_CONFIGURATIONS} configurations x "
        f"{len(Regime)} regimes = {NOMINAL_OCCUPANCY_ROWS} rows (9.7.4)",
        f"realised {len(rows)} rows; a regime merged into r0 by 9.7.1's under-occupancy "
        "rule has no row of its own and is named in the r0 row's merged_from",
        f"`included` is 9.7.5's reporting admission: m >= {M_MIN_OCCUPANCY} and "
        f"n_frames >= {N_MIN_RAW_FRAMES} raw frames, which is the stricter of the two "
        "calibers and is not the N_0 edge-admission rule",
        f"m_ci is a trajectory-level block bootstrap, B = {BLOCK_BOOTSTRAP_RESAMPLES}, "
        f"seed {seed}",
        "osc_r is the enclosing radius of c_delta's range about its mean in the regime: "
        "an upper bound on the Chebyshev radius, which is the direction 1(e')'s bias "
        "bound needs",
    ]
    return CalibrationTable(
        name="regime_occupancy_mass",
        title="per-regime occupancy mass, per (task, defect configuration, regime)",
        proposal_ref="10.3.2 output 1, 9.7.4",
        columns=(
            "task", "defect", "family", "regime", "m", "n_frames", "n_eff", "tau_corr",
            "n_demos", "m_ci_lo", "m_ci_hi", "osc_r", "included", "merged_from",
        ),
        units={
            "m": "share", "n_frames": "frames", "n_eff": "effective frames",
            "tau_corr": "frames",
        },
        rows=tuple(rows),
        notes=tuple(notes),
    )


def build_connectivity_report(
    observations: Mapping[Regime, Sequence[CellObservation]],
    *,
    key_pairs: Sequence[tuple[str, str]],
    n0_levels: Sequence[int] = N0_SENSITIVITY_FRAMES,
    primary_n0: int = N0_PRIMARY_FRAMES,
    heteroscedastic: bool = False,
) -> dict[str, Any]:
    """P1 output 2: the per-regime connectivity verdict at both calibers.

    `key_pairs` is 9.7.5's list P and is required rather than defaulted: the
    section reports R_eff "on every key pair", and which task contrasts are key
    is a preregistered choice, not something this function can supply.

    The returned `do_not_report` list is the deliverable W2 names. It is taken
    at the stricter caliber -- m >= 0.02 and n >= 2000 raw frames -- because the
    proposal makes that one the main narrative and the N_0 sweep the sensitivity
    upper bound; the sweep's own verdicts are in `n0_caliber`, and the regimes
    whose verdict moves across 50 / 200 / 800 are listed separately and treated
    at their most conservative level.
    """
    if not observations:
        raise MissingRegimeInput(
            "per_regime_connectivity",
            "no per-regime cell observations were supplied",
            "the 9.7.4 occupancy table, one CellObservation per (task, configuration)",
        )
    if not key_pairs:
        raise MissingRegimeInput(
            "per_regime_connectivity",
            "no key task pairs were supplied, and 9.7.5 requires R_eff on all of them",
            "the preregistered key-pair list P of 9.7.5",
        )
    report = per_regime_connectivity(
        observations,
        n0_levels=n0_levels,
        primary_n0=primary_n0,
        heteroscedastic=heteroscedastic,
        key_pairs=key_pairs,
    )
    strict = report["occupancy_caliber"]
    do_not_report = tuple(
        sorted(name for name, row in strict.items() if not row["reportable"])
    )
    return {
        "strict_caliber": strict,
        "n0_caliber": {
            "primary_n0": report["primary_n0"],
            "primary": report["primary"],
            "sensitivity": report["sensitivity"],
            "unstable_across_n0": report["unstable_across_n0"],
            "reportable": report["reportable"],
        },
        "key_pairs": tuple(key_pairs),
        "do_not_report": do_not_report,
        "calibers_disagree": tuple(
            sorted(
                name
                for name, row in strict.items()
                if row["reportable"] != report["reportable"][name]
            )
        ),
    }


@dataclass(frozen=True)
class CrossEmbodimentTVCell:
    """One twin pair's state occupancy on the same task, as partition codes.

    The partition is the caller's: TV is a supremum over measurable sets, so a
    reading on one partition is a lower bound and the partition decides how
    tight it is (`diagnostics.regimes.TVReading`).
    """

    task: str
    family: str
    defect: str
    bins_defective: np.ndarray
    bins_clean: np.ndarray
    n_bins: int


def build_cross_embodiment_tv_table(
    cells: Sequence[CrossEmbodimentTVCell],
    *,
    rng: np.random.Generator,
    n_null: int = 200,
) -> CalibrationTable:
    """P1 output 3: TV(rho_{tau,delta}, rho_{tau,0}) per (task, configuration).

    Unconditional on the regime, on purpose. The within-regime distance is a
    different row of proposition 1(e)'s decomposition and bounds a different
    term; reporting only the conditional one leaves the task-scale term, which
    10.3.2 calls the usually dominant one, unmeasured.
    """
    if not cells:
        raise MissingRegimeInput(
            "cross_embodiment_tv",
            "no twin-pair occupancies were supplied",
            "the same rollouts the 9.7.4 occupancy table is built from, binned onto a "
            "stated state partition",
        )
    rows = []
    for cell in cells:
        reading = cross_embodiment_tv(
            cell.bins_defective,
            cell.bins_clean,
            n_bins=cell.n_bins,
            rng=rng,
            n_null=n_null,
        )
        rows.append(
            {
                "task": cell.task,
                "defect": cell.defect,
                "family": cell.family,
                "tv": reading.value,
                "n_bins": reading.n_bins,
                "n_defective": reading.n_a,
                "n_clean": reading.n_b,
                "null_median": reading.null_median,
                "null_upper": reading.null_upper,
                "excess": reading.excess,
                "separated": reading.separated,
            }
        )
    return CalibrationTable(
        name="cross_embodiment_tv",
        title="cross-embodiment same-task total variation, per (task, configuration)",
        proposal_ref="10.3.2 output 3, A-19",
        columns=(
            "task", "defect", "family", "tv", "n_bins", "n_defective", "n_clean",
            "null_median", "null_upper", "excess", "separated",
        ),
        units={"tv": "probability", "null_median": "probability", "null_upper": "probability"},
        rows=tuple(rows),
        notes=(
            "a reading on one partition is a lower bound on the true TV; the "
            "same-distribution null says how much of it the sample sizes explain",
            "this is the only measurement of the first term of proposition 1(e)'s bias "
            "bound, the task-scale one (10.3.2)",
        ),
    )
