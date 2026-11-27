"""X2's two budget calibers for the clean control arm, reported side by side.

Proposal 6.4 layer 3, with the pool arithmetic of 10.4.0's data ratio table and
the step arithmetic of 9.8.1.

Confounder X2 is the data-volume difference: the pooled arm sees seven times the
demonstrations the single-embodiment arm does, so it shrinks less towards the
dataset mean, and the difference between the two carries a component along
+(mu_tau - a_bar) that the surrogate-defect null does not match -- the surrogates
match amplitude, regime support, spatial frequency and acting joint, and not
alignment with the shrinkage field. Layer 3's answer is a clean control
`pi_hat_clean`: same architecture, same parameter count, same budget, same step
count, trained entirely on defect-free bodies, giving a Lambda baseline at a
matched underfitting level. And it says the quiet part: **both budget calibers
are reported**, fixed target-token budget and fixed total-token budget.

Reporting one of them is the failure mode this closes, and it fails in both
directions. A clean control matched on the target's own tokens sees a seventh of
the total, so a critic reads its Lambda as "of course, it was trained on less".
A clean control matched on the total sees seven times the target data, so the
same critic reads it as "of course, it saw more of the body it is evaluated on".
Neither caliber can be made to answer both, which is why the answer is the pair;
`X2BudgetReport` refuses to render with one of them missing rather than printing
the one that happens to be available.

The clean control is not a reference point
------------------------------------------
Appendix A-4 rules an independently trained clean policy out as the reference
point of any Lambda: it has no upper anchor, and it is contaminated by exactly
the data-volume-induced shrinkage difference this module is measuring. B7 Oracle
stays the reference arm of every reading here. The clean control is a control,
and `X2BudgetReport.reference_arm` is B7 on both rows.

What the proposal fixes and what it leaves open
-----------------------------------------------
Fixed: N_tgt = 200 per (config, core task), K = 6 sources at 200 each, a pool of
1 400, alpha = 0.857, horizon 400, six core tasks, at least 60 epochs, and the
sequence-window accounting -- T_seq = 10, 16 sequences per batch -- that turns
the pooled arm's 3.36e6 frames into the 1.26e6 gradient steps 9.8.1 quotes.
`_self_check` reproduces both that number and 9.8.1's 1.8e5 for the
single-embodiment arm, so the arithmetic is checked rather than transcribed.

Open: what a clean control at each caliber is actually made of. "Fixed target
token budget" is taken as the target body's own token count held at the pooled
arm's 200 demonstrations per task, with no source data, since any source data
would raise the total and the caliber is the one that does not control the
total. "Fixed total token budget" is taken as the pooled arm's whole 1 400 per
task given to the target's clean twin, which is the composition that makes the
pair bracket the dilution explanation: one arm holds the target data fixed and
varies the total, the other holds the total fixed and varies the target share.
A clean control at 200 target plus 6 x 200 clean sources would hold *both*
fixed, which sounds better until one notices it is B7's composition, and B7 is
already the reference point every one of these numbers is measured against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from portal_trace.policies.schedule import MIN_EPOCHS
from portal_trace.registry.tasks import HORIZON

__all__ = [
    "N_TARGET_DEMOS",
    "N_SOURCE_DEMOS",
    "K_SOURCES",
    "POOL_DEMOS_PER_TASK",
    "ALPHA_NOMINAL",
    "N_CORE_TASKS",
    "SEQUENCE_WINDOW",
    "BATCH_SEQUENCES",
    "BudgetCaliber",
    "PoolComposition",
    "TokenBudget",
    "CleanControlReading",
    "X2BudgetReport",
    "MAIN_POOL",
    "clean_control_pool",
    "BudgetCaliberMissingError",
    "SingleCaliberReportError",
]

# 10.4.0's main-arm data ratio table, transcribed.
N_TARGET_DEMOS = 200
K_SOURCES = 6
N_SOURCE_DEMOS = 200
POOL_DEMOS_PER_TASK = N_TARGET_DEMOS + K_SOURCES * N_SOURCE_DEMOS
ALPHA_NOMINAL = 0.857
N_CORE_TASKS = 6

# 9.8.1's sample unit: a length-10 sequence window, 16 sequences per batch, which
# is robomimic's BC-RNN caliber and the one the 1.26e6 figure is computed in.
# 10.4.0's optimiser row quotes batch 64 for the chunked decoder; the two are
# different units and only this one closes 9.8.1's arithmetic, so the step count
# is derived here and the optimiser batch stays where it is.
SEQUENCE_WINDOW = 10
BATCH_SEQUENCES = 16


class BudgetCaliberMissingError(ValueError):
    """One of X2's two calibers has no reading."""


class SingleCaliberReportError(ValueError):
    """A report was built from one caliber."""


class BudgetCaliber(str, Enum):
    """The two calibers of 6.4 layer 3."""

    FIXED_TARGET_TOKENS = "fixed target token budget"
    FIXED_TOTAL_TOKENS = "fixed total token budget"


@dataclass(frozen=True)
class PoolComposition:
    """How many demonstrations a training pool holds, and from where.

    Per (config, core task), which is the unit 10.4.0's table is written in.
    `clean` records whether the bodies are defect-free, because it is the only
    thing separating the clean control from the arm it is matched to and a pool
    that lost the flag would print as the wrong arm.
    """

    label: str
    target_demos: int
    n_sources: int
    demos_per_source: int
    clean: bool
    n_tasks: int = N_CORE_TASKS
    horizon: int = HORIZON

    def __post_init__(self) -> None:
        if self.target_demos < 0 or self.demos_per_source < 0 or self.n_sources < 0:
            raise ValueError(f"{self.label}: demonstration counts are non-negative")
        if self.total_demos <= 0:
            raise ValueError(f"{self.label}: an empty pool trains nothing")
        if self.n_tasks <= 0:
            raise ValueError(f"{self.label}: n_tasks must be positive")
        if self.horizon != HORIZON:
            raise ValueError(
                f"{self.label}: horizon {self.horizon} against 9.6.1's fixed "
                f"{HORIZON}. The token count is demonstrations x horizon precisely "
                "because the horizon does not vary; X14's alpha_frame = alpha_demo "
                "identity rests on the same fact"
            )

    @property
    def source_demos(self) -> int:
        return self.n_sources * self.demos_per_source

    @property
    def total_demos(self) -> int:
        return self.target_demos + self.source_demos

    @property
    def alpha_demo(self) -> float:
        return self.source_demos / self.total_demos

    @property
    def target_tokens(self) -> int:
        """Frames from the target body, over all core tasks."""
        return self.target_demos * self.horizon * self.n_tasks

    @property
    def total_tokens(self) -> int:
        return self.total_demos * self.horizon * self.n_tasks

    def budget(self, *, epochs: int = MIN_EPOCHS) -> TokenBudget:
        return TokenBudget(pool=self, epochs=epochs)


@dataclass(frozen=True)
class TokenBudget:
    """A pool's token and gradient-step accounting at a fixed epoch count."""

    pool: PoolComposition
    epochs: int = MIN_EPOCHS

    def __post_init__(self) -> None:
        if self.epochs < MIN_EPOCHS:
            raise ValueError(
                f"{self.pool.label}: {self.epochs} epochs is below 10.4.0's floor of "
                f"{MIN_EPOCHS}. The epoch count is part of the endpoint definition -- "
                "undertraining raises E[eps^2], depresses Phi and reads out a false low "
                "leakage -- and 9.8.4 keeps it out of the degradation ladder"
            )

    @property
    def steps_per_epoch(self) -> float:
        return self.pool.total_tokens / (SEQUENCE_WINDOW * BATCH_SEQUENCES)

    @property
    def gradient_steps(self) -> float:
        return self.epochs * self.steps_per_epoch

    def as_row(self) -> dict[str, Any]:
        return {
            "pool": self.pool.label,
            "clean": self.pool.clean,
            "target_demos": self.pool.target_demos,
            "n_sources": self.pool.n_sources,
            "demos_per_source": self.pool.demos_per_source,
            "total_demos": self.pool.total_demos,
            "alpha_demo": self.pool.alpha_demo,
            "target_tokens": self.pool.target_tokens,
            "total_tokens": self.pool.total_tokens,
            "epochs": self.epochs,
            "gradient_steps": self.gradient_steps,
        }


MAIN_POOL = PoolComposition(
    label="B2 pooled (10.4.0 main arm)",
    target_demos=N_TARGET_DEMOS,
    n_sources=K_SOURCES,
    demos_per_source=N_SOURCE_DEMOS,
    clean=False,
)


def clean_control_pool(
    caliber: BudgetCaliber, reference: PoolComposition = MAIN_POOL
) -> PoolComposition:
    """The clean control matched to `reference` under one of X2's two calibers."""
    if reference.clean:
        raise ValueError(
            f"{reference.label} is already a clean pool; X2's control is matched to the "
            "pooled arm, whose sources carry defects"
        )
    if caliber is BudgetCaliber.FIXED_TARGET_TOKENS:
        return PoolComposition(
            label=f"clean control, {caliber.value}",
            target_demos=reference.target_demos,
            n_sources=0,
            demos_per_source=0,
            clean=True,
            n_tasks=reference.n_tasks,
            horizon=reference.horizon,
        )
    return PoolComposition(
        label=f"clean control, {caliber.value}",
        target_demos=reference.total_demos,
        n_sources=0,
        demos_per_source=0,
        clean=True,
        n_tasks=reference.n_tasks,
        horizon=reference.horizon,
    )


@dataclass(frozen=True)
class CleanControlReading:
    """The clean control's Lambda^part at one caliber, against B7.

    `lambda_part` is an ordinary float rather than an `Amplitude` because it is
    consumed here only as a number to be printed beside its twin; the typed
    quantity lives where it is estimated.
    """

    caliber: BudgetCaliber
    lambda_part: float
    epochs: int = MIN_EPOCHS
    reference_arm: str = "B7"

    def __post_init__(self) -> None:
        if not math.isfinite(self.lambda_part):
            raise ValueError(f"{self.caliber.value}: Lambda^part is not finite")
        if self.reference_arm != "B7":
            raise ValueError(
                f"{self.caliber.value}: reference arm {self.reference_arm!r}. Appendix "
                "A-4 rules an independently trained clean policy out as a reference "
                "point -- it has no upper anchor and it carries the very shrinkage "
                "difference X2 is about -- so the clean control is a control and B7 "
                "stays the reference"
            )

    def pool(self, reference: PoolComposition = MAIN_POOL) -> PoolComposition:
        return clean_control_pool(self.caliber, reference)

    def budget(self, reference: PoolComposition = MAIN_POOL) -> TokenBudget:
        return self.pool(reference).budget(epochs=self.epochs)


class X2BudgetReport:
    """Both calibers, or nothing.

    The constructor takes a mapping and refuses a missing caliber there rather
    than at render time: a report object that exists with one row will be
    printed with one row.
    """

    def __init__(
        self,
        readings: dict[BudgetCaliber, CleanControlReading],
        *,
        pooled_lambda_part: float,
        reference: PoolComposition = MAIN_POOL,
    ) -> None:
        missing = [c for c in BudgetCaliber if c not in readings]
        if missing:
            raise BudgetCaliberMissingError(
                f"no reading for {[c.value for c in missing]}. 6.4 layer 3 reports both "
                "calibers side by side: matched on the target's tokens the clean "
                "control sees a seventh of the total, matched on the total it sees "
                "seven times the target data, and either one alone leaves the "
                "'leakage is really target-data dilution' reading open in the "
                "direction the missing one closes. Train the missing control on "
                f"{clean_control_pool(missing[0], reference).label}"
            )
        for caliber, reading in readings.items():
            if reading.caliber is not caliber:
                raise ValueError(
                    f"reading filed under {caliber.value} is labelled "
                    f"{reading.caliber.value}"
                )
        if len(readings) < 2:
            raise SingleCaliberReportError(
                "X2's control is the pair of calibers, not either of them"
            )
        if not math.isfinite(pooled_lambda_part):
            raise ValueError("the pooled arm's Lambda^part is not finite")
        self._readings = dict(readings)
        self._pooled = pooled_lambda_part
        self._reference = reference

    @property
    def reference_arm(self) -> str:
        return "B7"

    def excess_over_clean(self, caliber: BudgetCaliber) -> float:
        """Pooled Lambda^part minus the clean control's, at one caliber."""
        return self._pooled - self._readings[caliber].lambda_part

    def dilution_explanation_survives(self) -> bool:
        """Whether either caliber puts the clean control level with the pooled arm.

        A clean control that reads as high a Lambda^part as the pooled arm under
        *either* caliber is the X2 alternative coming true: the reading would be
        a shrinkage artefact of the data volume rather than compensation
        leakage. Both calibers have to stay below it for the layer to have done
        its job, which is again why the pair is the object.
        """
        return any(
            self.excess_over_clean(c) <= 0.0 for c in BudgetCaliber
        )

    def as_report(self) -> dict[str, Any]:
        rows = []
        for caliber in BudgetCaliber:
            reading = self._readings[caliber]
            row = reading.budget(self._reference).as_row()
            row["caliber"] = caliber.value
            row["lambda_part"] = reading.lambda_part
            row["excess_over_clean"] = self.excess_over_clean(caliber)
            rows.append(row)
        return {
            "confounder": "X2 (data volume)",
            "layer": "6.4 layer 3, capacity- and volume-matched clean control",
            "reference_arm": self.reference_arm,
            "reference_pool": self._reference.budget().as_row(),
            "pooled_lambda_part": self._pooled,
            "calibers": rows,
            "both_calibers_reported": True,
            "dilution_explanation_survives": self.dilution_explanation_survives(),
            "note": (
                "the clean control is a control and never a reference point "
                "(appendix A-4)"
            ),
        }


def _self_check() -> None:
    """9.8.1's two worked step counts, recomputed from the pool arithmetic."""
    if MAIN_POOL.total_demos != 1_400:
        raise AssertionError(
            f"the main pool is {MAIN_POOL.total_demos} demonstrations per "
            "(config, task); 10.4.0's table fixes 1400"
        )
    if abs(MAIN_POOL.alpha_demo - ALPHA_NOMINAL) > 5e-4:
        raise AssertionError(
            f"alpha_demo = {MAIN_POOL.alpha_demo:.4f} against 10.4.0's {ALPHA_NOMINAL}"
        )
    if MAIN_POOL.total_tokens != 3_360_000:
        raise AssertionError(
            f"the pooled arm's training set is {MAIN_POOL.total_tokens} frames; 9.8.1 "
            "gives 3.36e6"
        )
    pooled_steps = MAIN_POOL.budget().gradient_steps
    if abs(pooled_steps - 1.26e6) / 1.26e6 > 0.01:
        raise AssertionError(
            f"the pooled arm needs {pooled_steps:.3g} gradient steps; 9.8.1 gives "
            "1.26e6"
        )
    single = clean_control_pool(BudgetCaliber.FIXED_TARGET_TOKENS)
    if single.total_tokens != 480_000:
        raise AssertionError(
            f"the fixed-target-token pool is {single.total_tokens} frames; 9.8.1's "
            "single-embodiment arithmetic gives 4.8e5"
        )
    if abs(single.budget().gradient_steps - 1.8e5) / 1.8e5 > 0.01:
        raise AssertionError(
            f"the fixed-target-token pool needs {single.budget().gradient_steps:.3g} "
            "gradient steps; 9.8.1 gives 1.8e5"
        )
    total = clean_control_pool(BudgetCaliber.FIXED_TOTAL_TOKENS)
    if total.total_tokens != MAIN_POOL.total_tokens:
        raise AssertionError(
            "the fixed-total-token caliber does not match the pooled arm's total"
        )
    if total.target_tokens == MAIN_POOL.target_tokens:
        raise AssertionError(
            "the two calibers coincide on the target token count, so they are one "
            "caliber written twice"
        )


_self_check()
