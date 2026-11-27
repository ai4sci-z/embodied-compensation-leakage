"""E11: the mismatch evaluation arm, a 6x6 cross matrix (proposal 10.4.11).

The question is one sentence: does harm come from the *existence* of
compensation, or from the *mismatch* between source and target? The two answers
are separated by evaluating every already-trained policy on deployment
embodiments carrying a defect other than its own, and regressing the harm
reading on two candidate explanatory variables at once --

    ||Delta c||,  Delta c = c_{delta_src} - c_{delta_tgt},   and   ||c_{delta_src}||.

T10's preregistered prediction is that the first partial coefficient is
significantly positive while the second, *after controlling for the first*, is
not (its 95% CI contains zero). If the second stays significant the mismatch
framework is incomplete and, as 10.4.11's zero-result clause says in as many
words, the main text has to demote "the unidentifiable component happens not to
enter harm" to a theoretical statement with no empirical support.

Four structural points this module enforces rather than assumes.

**The diagonal is the natural zero level, and it is reported separately.** On a
diagonal cell delta' = delta, so Delta c = 0 identically and the cell *is* the
main setting's target embodiment of 10.4.0 -- its readings are reused from the
main evaluation at zero incremental cost. Appendix A-9 requires the diagonal and
off-diagonal readings to be listed apart and never merged, because the
off-diagonal cells are where delta_tgt is outside the source defect population
and are therefore the test site for proposition 4's empirical premise. Asking
this module for a pooled number raises.

**The target is never a clean twin.** Appendix A-1 and (A7): no real robot may
be declared defect-free, so harm is defined on the mismatch field, which exists
for any pair of real embodiments, and never on transfer to a delta = 0 target.
A cell whose target is the clean twin is refused.

**The `eval_init` manifest is reused across defective targets, deliberately.**
The manifest is drawn per (core task, morphology family) and has no defect
dimension; `seed_env` does not contain `defect_id`, so a defective target and
its clean twin have the same initial-state distribution. 10.4.11 asks for this
to be written down explicitly, because the natural assumption is that each
target defect needs a fresh manifest. Here it is a checked precondition: every
cell in one matrix has to carry the same manifest id and the same length, which
is also what makes the bootstrap paired.

**Harm is a paired difference against the diagonal that shares the target.**
Not against the diagonal that shares the source: the pairing has to be on the
deployment embodiment, because that is what fixes the state distribution and the
shared initial-state indices, and a difference taken across two deployment
targets would carry the difference between two rollout distributions as well as
the mismatch.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from portal_trace.defects.params import CORE_DEFECTS
from portal_trace.eval.endpoints import SafetyQuantity
from portal_trace.eval.report import Verdict
from portal_trace.eval.stats import (
    ALPHA_PRIMARY,
    N_FIXED_INITIAL_STATES,
    BootstrapResult,
    paired_block_bootstrap_multi,
)
from portal_trace.lockfile import REQUIRED_MEASUREMENTS, LockFile
from portal_trace.registry.families import MAIN_FAMILIES

__all__ = [
    "N_SOURCE_DEFECTS",
    "N_TARGET_DEFECTS",
    "N_OFF_DIAGONAL_TARGETS",
    "N_POLICIES",
    "N_CORE_TASKS",
    "N_INITIAL_STATES",
    "ROLLOUTS_PER_POLICY_TARGET",
    "SECONDS_PER_ROLLOUT",
    "HORIZON_FRAMES",
    "N_EVAL_CAMERAS",
    "TAU_FRAME",
    "TAU_STEP",
    "E11_CPU_HOURS",
    "E11_GPU_HOURS",
    "CPU_HOURS_PER_POLICY",
    "CToDPromotion",
    "C_TO_D_PROMOTION",
    "E11CostModel",
    "E11_COST",
    "BudgetCrossCheck",
    "BUDGET_CROSS_CHECK",
    "TransferDirection",
    "MismatchCell",
    "MismatchMatrix",
    "FailureProfile",
    "PartialRegression",
    "partial_regression",
    "E11Report",
    "e11_report",
    "CleanTwinTargetError",
    "DiagonalMismatchError",
    "IncompleteCrossMatrixError",
    "ManifestMismatchError",
    "MergedDiagonalError",
    "CollinearExplanatoryVariablesError",
    "MissingDiagonalCellError",
]


# ---------------------------------------------------------------------------
# The grid and its cost, from 10.4.11's own arithmetic
# ---------------------------------------------------------------------------

N_SOURCE_DEFECTS = len(CORE_DEFECTS)
N_TARGET_DEFECTS = len(CORE_DEFECTS)
N_OFF_DIAGONAL_TARGETS = N_TARGET_DEFECTS - 1
"""5. The diagonal target is the main evaluation's own, so it costs nothing here."""

N_CONFIGS = len(MAIN_FAMILIES) * len(CORE_DEFECTS)
N_SEEDS = 3
N_POLICIES = N_CONFIGS * N_SEEDS
"""108 = 36 configs x 3 seeds, the S36 arm's policies (10.4.11's cost line)."""

N_CORE_TASKS = 6
N_INITIAL_STATES = N_FIXED_INITIAL_STATES
"""10.7.1's fixed 50 initial states per (task, family), reused on defective targets."""

ROLLOUTS_PER_POLICY_TARGET = N_CORE_TASKS * N_INITIAL_STATES
SECONDS_PER_ROLLOUT = 4.0
"""10.4.11's per-rollout unit cost, which 9.8.4 identifies as the rendering term."""

PHYSICS_SECONDS_PER_ROLLOUT = 0.5
"""9.8.4's physics term. 0.5 + 4.0 is 9.8.1's blended 4.5 s evaluation rollout."""

SECONDS_PER_HOUR = 3600.0

HORIZON_FRAMES = 400
"""Frames in one evaluation rollout: 9.8.1's image-materialisation row's 400."""

N_EVAL_CAMERAS = 2
"""The two 84x84 cameras of 9.6.1, which is the caliber tau_frame is measured in."""

TAU_FRAME = REQUIRED_MEASUREMENTS["sim.tau_frame.{family}.{task}"]
TAU_STEP = REQUIRED_MEASUREMENTS["sim.tau_step.{family}.{task}"]


class CleanTwinTargetError(ValueError):
    """A cell deploys onto a defect-free twin, which appendix A-1 forbids."""


class DiagonalMismatchError(ValueError):
    """A cell's diagonal status and its mismatch norm disagree."""


class IncompleteCrossMatrixError(ValueError):
    """The 6x6 cross matrix is missing cells."""


class ManifestMismatchError(ValueError):
    """Cells in one matrix were read on different initial-state manifests."""


class MergedDiagonalError(ValueError):
    """A diagonal and an off-diagonal reading were asked for as one number."""


class CollinearExplanatoryVariablesError(ValueError):
    """||Delta c|| and ||c_src|| cannot be separated on this matrix."""


class MissingDiagonalCellError(ValueError):
    """An off-diagonal cell has no diagonal cell to be paired against."""


@dataclass(frozen=True)
class E11CostModel:
    """10.4.11's cost formula, kept as an arithmetic rather than as `180`.

    policies x off-diagonal targets x tasks x states x seconds. The diagonal is
    absent from the product on purpose: those readings are the main evaluation's,
    reused rather than repeated.
    """

    n_policies: int = N_POLICIES
    n_off_diagonal_targets: int = N_OFF_DIAGONAL_TARGETS
    n_tasks: int = N_CORE_TASKS
    n_states: int = N_INITIAL_STATES
    seconds_per_rollout: float = SECONDS_PER_ROLLOUT

    @property
    def rollouts_per_policy_target(self) -> int:
        return self.n_tasks * self.n_states

    @property
    def n_rollouts(self) -> int:
        return self.n_policies * self.n_off_diagonal_targets * self.rollouts_per_policy_target

    @property
    def cpu_hours(self) -> float:
        return self.n_rollouts * self.seconds_per_rollout / SECONDS_PER_HOUR

    @property
    def cpu_hours_per_policy(self) -> float:
        """10.8.4's weekly column: 300 rollouts x 4.0 s = 0.333 CPU-h per policy."""
        return self.rollouts_per_policy_target * self.seconds_per_rollout / SECONDS_PER_HOUR

    @property
    def gpu_hours(self) -> float:
        """Zero. E11 trains nothing; it cross-evaluates policies that exist."""
        return 0.0

    def measured_seconds_per_rollout(
        self, lock: LockFile, *, family: str, task: str
    ) -> float:
        """The rendering cost of one rollout, from the measured tau_frame.

        10.4.11's 4.0 s is 9.8.1's rendering unit -- that row's image
        materialisation line reads `400 x 2 x tau_frame = 4.0 s`, and 9.8.4
        records that the evaluation rollout's 4.5 s is `0.5 s physics + 4.0 s
        rendering, rendering 89%`. So the 4.0 in the E11 cost line is not an
        independent constant: it is `HORIZON_FRAMES x N_EVAL_CAMERAS x
        tau_frame`, and tau_frame is a W1-1 quantity the proposal marks as to be
        measured. 9.8.4 says the same thing in general terms -- every CPU number
        in 9.8.1 except training and the E4 fit is a linear function of tau_step
        and tau_frame.

        This is the honest version of the E11 budget: the planning figure above
        is the proposal's transcribed arithmetic, and this is what the line
        actually costs once the constant behind it has been measured. It raises
        through `LockFile.get` when tau_frame is absent, naming the script.
        """
        tau_frame = float(lock.get(TAU_FRAME, family=family, task=task))
        return HORIZON_FRAMES * N_EVAL_CAMERAS * tau_frame

    def measured_cpu_hours(self, lock: LockFile, *, family: str, task: str) -> float:
        """The E11 line re-costed on the measured tau_frame rather than on 4.0 s."""
        seconds = self.measured_seconds_per_rollout(lock, family=family, task=task)
        return self.n_rollouts * seconds / SECONDS_PER_HOUR

    def overspend_factor(self, lock: LockFile, *, family: str, task: str) -> float:
        """9.8.4's varsigma for this line: measured over assumed.

        Above 4 it triggers 9.8.4's branch 0, the MJX port, which comes before
        any arm is cut. A zero-training arm like E11 is the cheapest thing in the
        study to re-cost, so it is a useful early read on that factor.
        """
        measured = self.measured_seconds_per_rollout(lock, family=family, task=task)
        return measured / self.seconds_per_rollout


E11_COST = E11CostModel()
E11_CPU_HOURS = E11_COST.cpu_hours
E11_GPU_HOURS = E11_COST.gpu_hours
CPU_HOURS_PER_POLICY = E11_COST.cpu_hours_per_policy


@dataclass(frozen=True)
class BudgetCrossCheck:
    """9.8.1's E11 line against 10.4.11's, checked rather than assumed.

    The two sections cost this arm independently and they agree exactly. 10.4.11
    writes `108 policies x 5 off-diagonal targets x 6 tasks x 50 initial states
    x 4.0 s = 180 CPU-h`; 9.8.1's budget table writes the same line as `108 x 5 x
    300 rollouts x 4.0 s = 180`, with 300 being that section's way of writing
    `6 x 50` and with the diagonal noted as reused from the main evaluation. Two
    factorisations of one product, so the check is that they multiply out to the
    same number, which is what `agree` reports.

    The unit cost is a decomposition, not a second opinion. 9.8.4 gives the
    evaluation rollout as `0.5 s physics + 4.0 s rendering, rendering 89%`, and
    9.1.3's throughput row gives the blended figure the same way, as
    `0.5 + 400 x 2 x tau_frame ~ 4.5 s`. So the 4.0 in the E11 line is the
    rendering term, it is not an independent constant -- it is
    `400 frames x 2 cameras x tau_frame` -- and `tau_frame` is marked to be
    measured in 9.1.3. `E11CostModel.measured_seconds_per_rollout` is where it
    stops being a transcription.
    """

    section_10_4_11_rollouts: int = N_POLICIES * N_OFF_DIAGONAL_TARGETS * ROLLOUTS_PER_POLICY_TARGET
    section_10_4_11_seconds: float = SECONDS_PER_ROLLOUT
    # 9.8.1's table row, in that row's own factorisation: 300 rollouts per
    # (policy, target) rather than 6 tasks x 50 states written out.
    section_9_8_1_rollouts: int = N_POLICIES * N_OFF_DIAGONAL_TARGETS * 300
    section_9_8_1_seconds: float = SECONDS_PER_ROLLOUT
    # The printed total both sections give, which the two products must reproduce.
    printed_cpu_hours: float = 180.0

    @property
    def section_10_4_11_cpu_hours(self) -> float:
        return self.section_10_4_11_rollouts * self.section_10_4_11_seconds / SECONDS_PER_HOUR

    @property
    def section_9_8_1_cpu_hours(self) -> float:
        return self.section_9_8_1_rollouts * self.section_9_8_1_seconds / SECONDS_PER_HOUR

    @property
    def cpu_hour_difference(self) -> float:
        return self.section_10_4_11_cpu_hours - self.section_9_8_1_cpu_hours

    @property
    def blended_seconds_per_rollout(self) -> float:
        """9.1.3's per-rollout figure: the rendering term plus 9.8.4's physics."""
        return self.section_10_4_11_seconds + PHYSICS_SECONDS_PER_ROLLOUT

    @property
    def rendering_share(self) -> float:
        """9.8.4's "rendering 89%", recomputed from the two terms."""
        return self.section_10_4_11_seconds / self.blended_seconds_per_rollout

    @property
    def agree(self) -> bool:
        return self.cpu_hour_difference == 0.0

    def as_row(self) -> dict[str, Any]:
        return {
            "10.4.11": {
                "rollouts": self.section_10_4_11_rollouts,
                "seconds_per_rollout": self.section_10_4_11_seconds,
                "cpu_hours": self.section_10_4_11_cpu_hours,
                "factorisation": (
                    f"{N_POLICIES} policies x {N_OFF_DIAGONAL_TARGETS} off-diagonal "
                    f"targets x {N_CORE_TASKS} tasks x {N_INITIAL_STATES} states"
                ),
            },
            "9.8.1": {
                "rollouts": self.section_9_8_1_rollouts,
                "seconds_per_rollout": self.section_9_8_1_seconds,
                "cpu_hours": self.section_9_8_1_cpu_hours,
                "factorisation": (
                    f"{N_POLICIES} x {N_OFF_DIAGONAL_TARGETS} x 300 rollouts, the "
                    "diagonal reused from the main evaluation"
                ),
            },
            "agree": self.agree,
            "cpu_hour_difference": self.cpu_hour_difference,
            "printed_cpu_hours": self.printed_cpu_hours,
            "blended_seconds_per_rollout": self.blended_seconds_per_rollout,
            "rendering_share": self.rendering_share,
            "unit_cost_note": (
                "9.8.4 decomposes the evaluation rollout as 0.5 s physics + 4.0 s "
                "rendering; the E11 line's 4.0 s is the rendering term, which is "
                "400 frames x 2 cameras x tau_frame"
            ),
        }


BUDGET_CROSS_CHECK = BudgetCrossCheck()

# Both sections' factorisations have to multiply out to the printed 180 CPU-h,
# and the two unit-cost terms to 9.8.4's 89% rendering share. Checked here rather
# than transcribed, so that a change to any factor is caught at import.
if not BUDGET_CROSS_CHECK.agree:
    raise AssertionError(
        "10.4.11 and 9.8.1 factorise E11's cost differently and the two products "
        f"disagree: {BUDGET_CROSS_CHECK.section_10_4_11_cpu_hours} against "
        f"{BUDGET_CROSS_CHECK.section_9_8_1_cpu_hours} CPU-h"
    )
if abs(BUDGET_CROSS_CHECK.section_10_4_11_cpu_hours - BUDGET_CROSS_CHECK.printed_cpu_hours) > 1e-9:
    raise AssertionError(
        "E11's cost formula no longer reproduces the 180 CPU-h both 10.4.11 and "
        f"9.8.1 print: {BUDGET_CROSS_CHECK.section_10_4_11_cpu_hours}"
    )
if round(BUDGET_CROSS_CHECK.rendering_share, 2) != 0.89:
    raise AssertionError(
        "9.8.4 puts the rendering share at 89%; the two unit-cost terms give "
        f"{BUDGET_CROSS_CHECK.rendering_share:.0%}"
    )


@dataclass(frozen=True)
class CToDPromotion:
    """10.4.11's C->D arm: an existing evaluation relabelled, not a new arm.

    The harm-asymmetry claim is about what *pooled training* does, so it cannot
    rest on open-loop replay alone -- P1's cross replay is open loop (appendix
    A-17). 10.4.11 gets a closed-loop trained C->D arm for free by promoting the
    already-run evaluation of the single-embodiment and Oracle policies on
    defective embodiments. Nothing is trained, and the promotion is recorded so
    that the arm's provenance is visible in the results table.
    """

    promoted_from: tuple[str, ...] = ("B1", "B7")
    new_training_runs: int = 0
    new_gpu_hours: float = 0.0
    reason: str = (
        "appendix A-17: harm asymmetry is a claim about pooled training, so it "
        "needs a closed-loop trained C->D arm; P1's cross replay is open loop"
    )


C_TO_D_PROMOTION = CToDPromotion()


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------


class TransferDirection(Enum):
    """Which way the compensation amplitude steps across the cell.

    10.4.11 asks for the failure-type profile "in both directions" and 11.1's
    harm-asymmetry row names them D->C and C->D. In the cross matrix neither side
    is clean, so the direction is read off the compensation amplitudes: a policy
    whose source carries more compensation than its deployment target is stepping
    down, which is the D->C-like direction, and the reverse is C->D-like. The
    diagonal is neither.
    """

    DIAGONAL = "diagonal: Delta c = 0"
    MORE_TO_LESS = "source compensation larger than target (D->C-like)"
    LESS_TO_MORE = "source compensation smaller than target (C->D-like)"


@dataclass(frozen=True)
class MismatchCell:
    """One (family, source defect, target defect, seed) of the cross matrix.

    The four per-rollout arrays are aligned on the shared `eval_init` manifest:
    entry i of each is the same initial state under the same task. That alignment
    is what makes the bootstrap paired and what makes a difference between two
    cells with the same target a paired difference.
    """

    family: str
    source_defect: str
    target_defect: str
    seed: int
    manifest_id: str
    success: np.ndarray
    constraint_violation: np.ndarray
    peak_cfrc_ext: np.ndarray
    terminal_overshoot: np.ndarray
    delta_c_norm: float
    c_src_norm: float
    c_tgt_norm: float
    target_is_clean_twin: bool = False

    def __post_init__(self) -> None:
        if self.target_is_clean_twin:
            raise CleanTwinTargetError(
                f"{self.family}/{self.source_defect}->{self.target_defect}: the "
                "deployment target is a defect-free twin. Appendix A-1 and (A7): no "
                "real robot may be declared defect-free, so no endpoint in this study "
                "is defined on a delta = 0 deployment target. The clean twin is the "
                "B7 Oracle's reference point and never a transfer target"
            )
        if self.source_defect not in CORE_DEFECTS:
            raise ValueError(
                f"source defect {self.source_defect!r} is not one of the six core "
                f"defects {list(CORE_DEFECTS)}"
            )
        if self.target_defect not in CORE_DEFECTS:
            raise ValueError(
                f"target defect {self.target_defect!r} is not one of the six core "
                f"defects {list(CORE_DEFECTS)}"
            )
        if self.family not in MAIN_FAMILIES:
            raise ValueError(
                f"family {self.family!r} is not one of the six main families "
                f"{list(MAIN_FAMILIES)}"
            )
        n = len(self.success)
        if n == 0:
            raise ValueError(f"{self.key}: no rollouts")
        for name in ("constraint_violation", "peak_cfrc_ext", "terminal_overshoot"):
            arr = np.asarray(getattr(self, name))
            if arr.shape != (n,):
                raise ValueError(
                    f"{self.key}: {name} has {arr.shape} entries against {n} rollouts. "
                    "Every per-rollout array is aligned on the same eval_init manifest"
                )
        for name in ("success", "constraint_violation"):
            arr = np.asarray(getattr(self, name))
            if not np.all((arr == 0) | (arr == 1)):
                raise ValueError(f"{self.key}: {name} takes 0/1 indicators per rollout")
        for name in ("delta_c_norm", "c_src_norm", "c_tgt_norm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{self.key}: {name} = {value} is not a finite norm")
        if self.is_diagonal and self.delta_c_norm != 0.0:
            raise DiagonalMismatchError(
                f"{self.key}: the source and target defects are the same, so "
                f"Delta c = c_src - c_tgt is identically zero, but ||Delta c|| = "
                f"{self.delta_c_norm}. The diagonal is 10.4.11's natural zero level "
                "and a non-zero norm there means the two fields were evaluated on "
                "different embodiments or at different levels"
            )
        if not self.is_diagonal and self.delta_c_norm == 0.0:
            raise DiagonalMismatchError(
                f"{self.key}: an off-diagonal cell with ||Delta c|| = 0. Two different "
                "core defects whose compensation fields coincide exactly would make "
                "the cell indistinguishable from the diagonal, and the regression's "
                "explanatory variable would carry a structural zero that is not one"
            )

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (self.family, self.source_defect, self.target_defect, self.seed)

    @property
    def label(self) -> str:
        return (
            f"{self.family}/{self.source_defect}->{self.target_defect}"
            f"/seed{self.seed}"
        )

    @property
    def is_diagonal(self) -> bool:
        return self.source_defect == self.target_defect

    @property
    def n_rollouts(self) -> int:
        return int(len(self.success))

    @property
    def direction(self) -> TransferDirection:
        if self.is_diagonal:
            return TransferDirection.DIAGONAL
        if self.c_src_norm > self.c_tgt_norm:
            return TransferDirection.MORE_TO_LESS
        if self.c_src_norm < self.c_tgt_norm:
            return TransferDirection.LESS_TO_MORE
        raise DiagonalMismatchError(
            f"{self.label}: the source and target compensation amplitudes are exactly "
            "equal on an off-diagonal cell, so the transfer has no direction. "
            "10.4.11 reports the failure-type profile in both directions and this "
            "cell belongs to neither"
        )

    def rate(self, quantity: SafetyQuantity) -> float:
        return float(np.mean(self.series(quantity)))

    def series(self, quantity: SafetyQuantity) -> np.ndarray:
        if quantity is SafetyQuantity.CONSTRAINT_VIOLATION_RATE:
            return np.asarray(self.constraint_violation, dtype=float)
        if quantity is SafetyQuantity.PEAK_CFRC_EXT:
            return np.asarray(self.peak_cfrc_ext, dtype=float)
        if quantity is SafetyQuantity.TERMINAL_OVERSHOOT:
            return np.asarray(self.terminal_overshoot, dtype=float)
        raise ValueError(
            f"E11 records {[q.name for q in _E11_QUANTITIES]}; {quantity.name} is not "
            "one of the readings 10.4.11 asks this arm for"
        )

    @property
    def success_rate(self) -> float:
        return float(np.mean(self.success))

    @property
    def task_failure_rate(self) -> float:
        return 1.0 - self.success_rate


_E11_QUANTITIES: tuple[SafetyQuantity, ...] = (
    SafetyQuantity.CONSTRAINT_VIOLATION_RATE,
    SafetyQuantity.PEAK_CFRC_EXT,
    SafetyQuantity.TERMINAL_OVERSHOOT,
)


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------


class MismatchMatrix:
    """The complete 6x6 cross matrix, checked for the preconditions of 10.4.11."""

    def __init__(self, cells: Sequence[MismatchCell]) -> None:
        if not cells:
            raise IncompleteCrossMatrixError("the cross matrix has no cells")
        keys = [c.key for c in cells]
        if len(set(keys)) != len(keys):
            duplicated = sorted({k for k in keys if keys.count(k) > 1})
            raise IncompleteCrossMatrixError(f"duplicate cells {duplicated}")
        self._cells = {c.key: c for c in cells}

        manifests = {c.manifest_id for c in cells}
        if len(manifests) != 1:
            raise ManifestMismatchError(
                f"the cells carry {len(manifests)} different eval_init manifests "
                f"{sorted(manifests)}. 10.4.11 reuses one manifest across the "
                "defective targets -- it is drawn per (core task, morphology family) "
                "and carries no defect dimension, and seed_env contains no defect_id, "
                "so a defective target and its clean twin see the same initial-state "
                "distribution. Two manifests break the pairing the bootstrap needs"
            )
        lengths = {c.n_rollouts for c in cells}
        if len(lengths) != 1:
            raise ManifestMismatchError(
                f"the cells carry {sorted(lengths)} rollouts; one manifest gives one "
                "length, and the bootstrap resamples a shared index across the cells"
            )
        self._manifest_id = next(iter(manifests))
        self._n_rollouts = next(iter(lengths))

        families = sorted({c.family for c in cells})
        seeds = sorted({c.seed for c in cells})
        missing: list[tuple[str, str, str, int]] = []
        for fam in families:
            for src in CORE_DEFECTS:
                for tgt in CORE_DEFECTS:
                    for seed in seeds:
                        if (fam, src, tgt, seed) not in self._cells:
                            missing.append((fam, src, tgt, seed))
        if missing:
            expected = len(families) * N_SOURCE_DEFECTS * N_TARGET_DEFECTS * len(seeds)
            raise IncompleteCrossMatrixError(
                f"{len(missing)} of the {expected} "
                f"cells are absent, the first being {missing[0]}. E11 is a complete "
                f"{N_SOURCE_DEFECTS}x{N_TARGET_DEFECTS} cross matrix at every seed: a "
                "partially filled matrix makes the two explanatory variables "
                "correlated through the missing pattern rather than through the "
                "physics. Produce the rest with "
                "`portal_trace.runners.mismatch.cross_evaluate`"
            )
        self._families = tuple(families)
        self._seeds = tuple(seeds)

    @property
    def manifest_id(self) -> str:
        return self._manifest_id

    @property
    def n_rollouts(self) -> int:
        return self._n_rollouts

    @property
    def families(self) -> tuple[str, ...]:
        return self._families

    @property
    def seeds(self) -> tuple[int, ...]:
        return self._seeds

    @property
    def cells(self) -> tuple[MismatchCell, ...]:
        return tuple(self._cells[k] for k in sorted(self._cells))

    @property
    def diagonal(self) -> tuple[MismatchCell, ...]:
        return tuple(c for c in self.cells if c.is_diagonal)

    @property
    def off_diagonal(self) -> tuple[MismatchCell, ...]:
        return tuple(c for c in self.cells if not c.is_diagonal)

    def pooled(self) -> tuple[MismatchCell, ...]:
        raise MergedDiagonalError(
            "the diagonal and off-diagonal readings of the cross matrix are reported "
            "apart and never merged (appendix A-9). The off-diagonal cells are where "
            "delta_tgt falls outside the source defect population, which is the test "
            "site for proposition 4's empirical premise; pooling them with the "
            "diagonal averages the premise together with the case it is a premise "
            "about. Use `diagonal` and `off_diagonal`"
        )

    def diagonal_for(self, cell: MismatchCell) -> MismatchCell:
        """The cell this one is paired against: same family, same target, same seed.

        The pairing is on the *deployment* embodiment. A difference taken against
        the diagonal that shares the source instead would compare two rollouts on
        two different bodies, so it would carry the difference between two state
        distributions as well as the mismatch.
        """
        key = (cell.family, cell.target_defect, cell.target_defect, cell.seed)
        try:
            return self._cells[key]
        except KeyError as exc:
            raise MissingDiagonalCellError(
                f"{cell.label} has no diagonal cell {key} to be paired against. Harm "
                "is the paired success-rate difference against the diagonal that "
                "shares the deployment target, which is 10.4.11's natural zero level"
            ) from exc

    def harm(self, cell: MismatchCell) -> float:
        """SR(diagonal) - SR(cell), the paired success-rate difference in pp units."""
        return self.diagonal_for(cell).success_rate - cell.success_rate

    def explanatory(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(||Delta c||, ||c_src||, harm) over the off-diagonal cells."""
        rows = self.off_diagonal
        return (
            np.array([c.delta_c_norm for c in rows], dtype=float),
            np.array([c.c_src_norm for c in rows], dtype=float),
            np.array([self.harm(c) for c in rows], dtype=float),
        )


# ---------------------------------------------------------------------------
# The failure-type profile (10.4.11's endpoint)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureProfile:
    """(constraint-violation rate, task-failure rate) with the two safety readings.

    The two-dimensional reading is the point of 11.1's harm-asymmetry row: the
    claim is that the two transfer directions fail in different *ways*, not that
    one fails more. A scalar summary of the pair would erase exactly that.
    """

    direction: TransferDirection
    n_cells: int
    constraint_violation_rate: float
    task_failure_rate: float
    peak_cfrc_ext: float
    terminal_overshoot: float

    @property
    def profile(self) -> tuple[float, float]:
        return (self.constraint_violation_rate, self.task_failure_rate)

    def as_row(self) -> dict[str, Any]:
        return {
            "direction": self.direction.value,
            "n_cells": self.n_cells,
            "constraint_violation_rate": self.constraint_violation_rate,
            "task_failure_rate": self.task_failure_rate,
            "peak_cfrc_ext": self.peak_cfrc_ext,
            "terminal_overshoot": self.terminal_overshoot,
        }


def failure_profiles(matrix: MismatchMatrix) -> dict[TransferDirection, FailureProfile]:
    """One profile per direction, with the diagonal kept as its own row."""
    grouped: dict[TransferDirection, list[MismatchCell]] = {
        direction: [] for direction in TransferDirection
    }
    for cell in matrix.cells:
        grouped[cell.direction].append(cell)
    out: dict[TransferDirection, FailureProfile] = {}
    for direction, rows in grouped.items():
        if not rows:
            continue
        out[direction] = FailureProfile(
            direction=direction,
            n_cells=len(rows),
            constraint_violation_rate=float(
                np.mean([c.rate(SafetyQuantity.CONSTRAINT_VIOLATION_RATE) for c in rows])
            ),
            task_failure_rate=float(np.mean([c.task_failure_rate for c in rows])),
            peak_cfrc_ext=float(
                np.mean([c.rate(SafetyQuantity.PEAK_CFRC_EXT) for c in rows])
            ),
            terminal_overshoot=float(
                np.mean([c.rate(SafetyQuantity.TERMINAL_OVERSHOOT) for c in rows])
            ),
        )
    return out


# ---------------------------------------------------------------------------
# The partial regression (T10)
# ---------------------------------------------------------------------------

MAX_EXPLANATORY_CORRELATION = 0.99
"""Above this the two partial coefficients are not separable on this matrix.

Not a tuning knob with a preregistered effect attached: it is the point at which
"the coefficient of ||c_src|| is not significant after controlling for
||Delta c||" stops being a statement about the physics and becomes a statement
about the design matrix. 10.4.11's whole prediction is a comparison between the
two partials, so a matrix that cannot separate them is refused rather than
reported with wide intervals.
"""


@dataclass(frozen=True)
class PartialRegression:
    """T10: harm ~ b0 + b1 ||Delta c|| + b2 ||c_src||, bootstrapped."""

    delta_c: BootstrapResult
    c_src: BootstrapResult
    intercept: BootstrapResult
    n_cells: int
    explanatory_correlation: float
    n_rollouts: int

    @property
    def delta_c_significantly_positive(self) -> bool:
        return self.delta_c.ci[0] > 0.0

    @property
    def c_src_not_significant(self) -> bool:
        lo, hi = self.c_src.ci
        return lo <= 0.0 <= hi

    @property
    def t10_holds(self) -> bool:
        return self.delta_c_significantly_positive and self.c_src_not_significant

    @property
    def verdict(self) -> str:
        if self.t10_holds:
            return "T10 supported"
        if not self.delta_c_significantly_positive:
            return "not significant"
        return "F: the mismatch framework is incomplete"

    @property
    def verdict_kind(self) -> Verdict:
        """T10 in 11.3's three-valued vocabulary.

        The prediction is a conjunction -- ||Delta c|| positive *and* ||c_src||
        cleared -- so failing either half is `NOT_SIGNIFICANT`. There is no
        `INCONCLUSIVE` branch here because the two conditions that would produce
        one, a degenerate design matrix and an incomplete matrix, are refusals
        rather than results: `partial_regression` raises before it returns.
        """
        return Verdict.SIGNIFICANT if self.t10_holds else Verdict.NOT_SIGNIFICANT

    @property
    def zero_result_consequence(self) -> str | None:
        """10.4.11's zero-result clause, verbatim in its consequence."""
        if self.c_src_not_significant:
            return None
        return (
            "||c_src|| stays significant after controlling for ||Delta c||, so the "
            "mismatch framework is incomplete and part of the harm is set by the "
            "absolute amplitude of the source compensation. Proposition 4's empirical "
            "premise is then missing, and the main text must demote 'the "
            "unidentifiable component does not enter harm' to a theoretical statement "
            "explicitly marked as having no empirical support (10.4.11)"
        )

    def as_row(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "verdict_kind": self.verdict_kind.value,
            "delta_c_coefficient": self.delta_c.point,
            "delta_c_ci": list(self.delta_c.ci),
            "c_src_coefficient": self.c_src.point,
            "c_src_ci": list(self.c_src.ci),
            "intercept": self.intercept.point,
            "intercept_ci": list(self.intercept.ci),
            "delta_c_significantly_positive": self.delta_c_significantly_positive,
            "c_src_not_significant": self.c_src_not_significant,
            "t10_holds": self.t10_holds,
            "n_off_diagonal_cells": self.n_cells,
            "n_rollouts_per_cell": self.n_rollouts,
            "explanatory_correlation": self.explanatory_correlation,
            "zero_result_consequence": self.zero_result_consequence,
        }


def _ols(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coefficients


def partial_regression(
    matrix: MismatchMatrix,
    *,
    rng: np.random.Generator,
    n_resamples: int,
    level: float = 1.0 - ALPHA_PRIMARY,
) -> PartialRegression:
    """Fit T10's two-covariate regression, with the preregistered paired bootstrap.

    The uncertainty comes from resampling the shared initial-state index -- one
    index vector per resample, applied to every cell -- and refitting. That is
    the inference the experiment declares (`bootstrap: paired, resampling shared
    initial-state indices`), and it is the right one here for the same reason it
    is right everywhere else in the study: the initial-state manifest is common
    to the cells, so that layer is perfectly correlated across them and
    resampling each cell on its own would inflate the variance of every contrast
    between them.

    The regression is fitted on the off-diagonal cells alone. The diagonal is
    reported beside it and never merged in: those cells have ||Delta c|| = 0 and
    harm identically zero *by construction*, so putting them in would pin the
    fitted line through the origin with structural zeros and inflate the
    ||Delta c|| coefficient for a reason that has nothing to do with the physics.
    """
    off = matrix.off_diagonal
    if len(off) < 4:
        raise IncompleteCrossMatrixError(
            f"{len(off)} off-diagonal cells: a two-covariate regression needs more "
            "residual degrees of freedom than that"
        )
    delta_c = np.array([c.delta_c_norm for c in off], dtype=float)
    c_src = np.array([c.c_src_norm for c in off], dtype=float)
    if float(np.std(delta_c)) <= 0.0 or float(np.std(c_src)) <= 0.0:
        raise CollinearExplanatoryVariablesError(
            "one of the two explanatory variables is constant over the off-diagonal "
            "cells, so its partial coefficient is not identified"
        )
    correlation = float(np.corrcoef(delta_c, c_src)[0, 1])
    if abs(correlation) > MAX_EXPLANATORY_CORRELATION:
        raise CollinearExplanatoryVariablesError(
            f"||Delta c|| and ||c_src|| correlate at {correlation:+.4f} over the "
            f"{len(off)} off-diagonal cells, above {MAX_EXPLANATORY_CORRELATION}. "
            "T10 is a comparison between two partial coefficients, so on this matrix "
            "it would be a statement about the design rather than about the harm. The "
            "full 6x6 grid exists to break this correlation; a matrix this collinear "
            "means cells are missing or the compensation norms were computed at one "
            "defect level only"
        )
    design = np.column_stack([np.ones_like(delta_c), delta_c, c_src])

    arms: dict[str, np.ndarray] = {}
    for cell in off:
        arms[f"off::{cell.label}"] = np.asarray(cell.success, dtype=float)
        diagonal = matrix.diagonal_for(cell)
        arms[f"dia::{diagonal.label}"] = np.asarray(diagonal.success, dtype=float)

    keys = [(f"dia::{matrix.diagonal_for(c).label}", f"off::{c.label}") for c in off]

    def coefficients(resampled: Mapping[str, np.ndarray]) -> np.ndarray:
        harm = np.array(
            [
                float(resampled[dia].mean() - resampled[o].mean())
                for dia, o in keys
            ],
            dtype=float,
        )
        return _ols(design, harm)

    statistics = {
        "intercept": lambda a: float(coefficients(a)[0]),
        "delta_c": lambda a: float(coefficients(a)[1]),
        "c_src": lambda a: float(coefficients(a)[2]),
    }
    boot = paired_block_bootstrap_multi(
        arms, statistics, n_resamples=n_resamples, rng=rng, level=level
    )
    return PartialRegression(
        delta_c=boot["delta_c"],
        c_src=boot["c_src"],
        intercept=boot["intercept"],
        n_cells=len(off),
        explanatory_correlation=correlation,
        n_rollouts=matrix.n_rollouts,
    )


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class E11Report:
    """The two E11 endpoint rows plus the diagonal / off-diagonal split."""

    regression: PartialRegression
    profiles: dict[TransferDirection, FailureProfile]
    matrix: MismatchMatrix
    cost: E11CostModel = E11_COST
    promotion: CToDPromotion = C_TO_D_PROMOTION

    @property
    def diagonal_harm_is_structurally_zero(self) -> bool:
        """Every diagonal cell is its own pairing partner, so its harm is exactly 0."""
        return all(self.matrix.harm(c) == 0.0 for c in self.matrix.diagonal)

    @property
    def diagonal_c_src_range(self) -> tuple[float, float]:
        """The spread of ||c_src|| over the diagonal, where harm is zero.

        Proposition 4's landing reading: on and near the diagonal the harm tends
        to zero *even where ||c_src|| is large*. The range is what makes that
        sentence checkable -- a diagonal whose source amplitudes were all small
        would show zero harm for a reason proposition 4 does not claim.
        """
        values = [c.c_src_norm for c in self.matrix.diagonal]
        return (min(values), max(values))

    def as_row(self) -> dict[str, Any]:
        return {
            "experiment": "E11",
            "mismatch_partial_regression": self.regression.as_row(),
            "failure_type_profile": {
                direction.name: profile.as_row()
                for direction, profile in self.profiles.items()
            },
            "diagonal": {
                "n_cells": len(self.matrix.diagonal),
                "role": (
                    "10.4.11's natural zero level and 10.4.0's main-setting target; "
                    "the readings are reused from the main evaluation at zero "
                    "incremental cost"
                ),
                "harm_is_structurally_zero": self.diagonal_harm_is_structurally_zero,
                "c_src_norm_range": list(self.diagonal_c_src_range),
                "reported_separately": True,
            },
            "off_diagonal": {
                "n_cells": len(self.matrix.off_diagonal),
                "role": (
                    "delta_tgt is outside the source defect population, so Delta gamma "
                    "carries an unidentifiable component whose only route out is the "
                    "physical parameterisation of the compensation field, routes (P) "
                    "and (S). This is proposition 4's empirical premise being tested"
                ),
                "reported_separately": True,
            },
            "eval_init_manifest": {
                "id": self.matrix.manifest_id,
                "n_rollouts": self.matrix.n_rollouts,
                "reused_on_defective_targets": True,
                "why": (
                    "the manifest is drawn per (core task, morphology family) and has "
                    "no defect dimension; seed_env carries no defect_id, so a "
                    "defective target and its clean twin have the same initial-state "
                    "distribution (10.4.11)"
                ),
            },
            "c_to_d_arm": {
                "promoted_from": list(self.promotion.promoted_from),
                "new_training_runs": self.promotion.new_training_runs,
                "new_gpu_hours": self.promotion.new_gpu_hours,
                "reason": self.promotion.reason,
            },
            "cost": {
                "gpu_hours": self.cost.gpu_hours,
                "cpu_hours": self.cost.cpu_hours,
                "n_rollouts": self.cost.n_rollouts,
                "cpu_hours_per_policy": self.cost.cpu_hours_per_policy,
                "budget_cross_check": BUDGET_CROSS_CHECK.as_row(),
            },
        }


def e11_report(
    matrix: MismatchMatrix,
    *,
    rng: np.random.Generator,
    n_resamples: int,
    level: float = 1.0 - ALPHA_PRIMARY,
) -> E11Report:
    """Both E11 endpoints, off one cross matrix."""
    return E11Report(
        regression=partial_regression(
            matrix, rng=rng, n_resamples=n_resamples, level=level
        ),
        profiles=failure_profiles(matrix),
        matrix=matrix,
    )


def _self_check() -> None:
    """10.4.11's and 10.8.4's printed numbers, recomputed from their factors."""
    if N_SOURCE_DEFECTS != 6 or N_TARGET_DEFECTS != 6:
        raise AssertionError(
            f"10.4.11's cross matrix is 6x6; the core defect list gives "
            f"{N_SOURCE_DEFECTS}x{N_TARGET_DEFECTS}"
        )
    if N_POLICIES != 108:
        raise AssertionError(
            f"10.4.11's cost line reads 108 policies = 36 configs x 3 seeds; "
            f"{len(MAIN_FAMILIES)} families x {len(CORE_DEFECTS)} defects x {N_SEEDS} "
            f"seeds gives {N_POLICIES}"
        )
    if E11_COST.n_rollouts != 162_000:
        raise AssertionError(
            f"108 x 5 x 6 x 50 is 162,000 rollouts; the cost model gives "
            f"{E11_COST.n_rollouts}"
        )
    if E11_COST.cpu_hours != 180.0:
        raise AssertionError(
            f"10.4.11's cost line evaluates to 180 CPU-h; the model gives "
            f"{E11_COST.cpu_hours}"
        )
    if E11_COST.gpu_hours != 0.0:
        raise AssertionError("E11 trains nothing, so its GPU cost is zero")
    if round(E11_COST.cpu_hours_per_policy, 3) != 0.333:
        raise AssertionError(
            f"10.8.4's weekly column reads 300 rollouts x 4.0 s = 0.333 CPU-h per "
            f"policy; the model gives {E11_COST.cpu_hours_per_policy}"
        )
    if BUDGET_CROSS_CHECK.section_9_8_1_rollouts != 162_000:
        raise AssertionError(
            "9.8.1's E11 line reads 108 x 5 x 300 rollouts = 162,000; the factors "
            f"here give {BUDGET_CROSS_CHECK.section_9_8_1_rollouts}"
        )
    if not BUDGET_CROSS_CHECK.agree:
        raise AssertionError(
            "10.4.11 and 9.8.1 cost E11 with different factorisations of the same "
            "product and both print 180 CPU-h; the two no longer multiply out to "
            f"the same number ({BUDGET_CROSS_CHECK.cpu_hour_difference} CPU-h apart)"
        )


_self_check()
