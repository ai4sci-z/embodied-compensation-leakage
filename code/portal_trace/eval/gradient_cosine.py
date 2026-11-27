"""E2': the similarity-gradient bridge, and where it breaks (proposal 10.4.2).

Morphology clustering (the EG of arXiv 2602.18025) groups sources by a signal
that is **static and purely geometric**: an FGW distance computed once before
training, from graph structure and node features only. Its load-bearing
empirical claim is that morphology similarity predicts gradient cosine
similarity. E2' asks whether that bridge still carries along the defect axis,
and the answer it produces is a two-sample statement, not a p-value on somebody
else's regression line.

What this module owns
---------------------
* the **measurement protocol** of 10.4.2 as a type: three checkpoints, BC loss,
  training-split frames, 4 000 frames per source split into the same number of
  minibatches on both sides, and the upper anchor at the same batch size
  (`MinibatchPlan`, `Checkpoint`);
* the **estimator**: a per-parameter-tensor cosine weighted by parameter count
  (`weighted_tensor_cosine`);
* the **null**: the no-conflict upper anchor, two disjoint minibatches from one
  and the same source (`SourcePairLevel.UPPER_ANCHOR`);
* the **verdict rule T12** and the **F3 exit trigger** (`TwoSampleReading`,
  `E2PrimeReport`);
* **our own regression line** on the non-trivial pool, with our own r and
  confidence band, and the twin pair as that line's origin residual
  (`OwnRegressionLine`);
* the **variance decomposition** of J(B3) - J(B2) into a cross-family-pair and
  a same-family-cross-defect source, and the **P7 positive control**.

Three refusals worth stating, because each closes a way the reading could be
quietly widened.

**A single checkpoint is refused.** 10.4.2: "gradient conflict is a training
process quantity, a single reading is not enough". `GradientCosineObservations`
therefore requires all three of 25%, 50% and final, and the headline verdict is
the conjunction over them; the per-checkpoint verdicts are reported beside it,
so a bridge that breaks only at the end is visible rather than averaged away.

**Unequal batch sizes are refused.** The cosine between two minibatch gradients
is a strong function of the effective batch size -- it rises towards one as the
batches grow, because both estimates converge on the full-batch gradient. Two
levels read at different batch sizes would differ for that reason alone, so
`MinibatchPlan` is an argument every reading carries and every reading in one
report has to share, the upper anchor included.

**The baseline's r = 0.63 is not a comparison baseline.** It is fitted on legged
platforms, offline RL, another node feature set and another gradient-cosine
caliber, so its x axis is not on our scale. `BaselineGradientCosineClaim` is
citation-only: asking it for a p value that could be read as an independent
sample, or for a number to test our own r against, raises.

Wording discipline (10.4.2, verbatim): say "morphology distance has no
predictive power for gradient cosine along the defect axis" and "its
similarity-gradient bridge breaks at this point". Do not say "we show
experimentally that morphology clustering does not work". `E2PrimeReport.wording`
is the sentence the report prints, and it is the first of those two.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Any

import numpy as np
from scipy import stats as sps

from portal_trace.eval.report import Verdict
from portal_trace.eval.stats import (
    ALPHA_PRIMARY,
    BootstrapResult,
    paired_block_bootstrap_multi,
)

__all__ = [
    "DELTA_COS_INDISTINGUISHABLE",
    "TWIN_PAIR_CI_UPPER_CEILING",
    "UPPER_ANCHOR_EXPECTED_FLOOR",
    "FRAMES_PER_SOURCE",
    "N_BOOTSTRAP_MINIBATCHES",
    "CHECKPOINT_FRACTIONS",
    "NONTRIVIAL_POOL_CLEAN_FAMILIES",
    "P7_MIN_GAIN_PP",
    "E2PRIME_RUNS",
    "E2PRIME_GPU_HOURS",
    "GPU_HOURS_PER_RUN",
    "SAME_DOF_REACH_M",
    "Checkpoint",
    "SourcePairLevel",
    "MinibatchPlan",
    "BaselineGradientCosineClaim",
    "BASELINE_CLAIM",
    "weighted_tensor_cosine",
    "CosineSample",
    "GradientCosineObservations",
    "TwoSampleReading",
    "OwnRegressionLine",
    "VarianceDecomposition",
    "P7PositiveControl",
    "E2PrimeRunBudget",
    "E2PRIME_RUN_BUDGET",
    "E2PrimeReport",
    "e2prime_report",
    "two_sample_reading",
    "fit_own_regression_line",
    "n_source_pairs",
    "CONSTRUCTIVE_FACTS",
    "DISTANCE_LEVELS",
    "WORDING",
    "BASELINE_EG_IMPROVEMENT",
    "BASELINE_SUBOPTIMAL_SHARE",
    "E2PRIME_SEEDS_PER_ARM",
    "DegenerateGradientTensorError",
    "GradientStructureMismatch",
    "MissingCheckpointError",
    "MissingUpperAnchorError",
    "UnequalMinibatchCountError",
    "MixedMinibatchPlanError",
    "BaselineIsNotAComparisonBaseline",
    "PositiveControlFailed",
    "DegenerateRegressorError",
]


# ---------------------------------------------------------------------------
# The preregistered constants of 10.4.2
# ---------------------------------------------------------------------------

DELTA_COS_INDISTINGUISHABLE = 0.15
"""T12: the twin-pair and cross-DoF distributions agree when |delta cos| < 0.15."""

TWIN_PAIR_CI_UPPER_CEILING = 0.85
"""T12's second half: the twin pair's 95% CI upper bound is below 0.85."""

UPPER_ANCHOR_EXPECTED_FLOOR = 0.9
"""10.4.2 expects the no-conflict anchor above 0.9. Reported, never assumed.

It is an expectation about the apparatus, not a criterion: an anchor that comes
in below it means the cosine estimator is noisy at this batch size, which makes
every other reading in the table harder to interpret and is therefore flagged
rather than corrected.
"""

FRAMES_PER_SOURCE = 4_000
"""10.4.2: each of the two sources contributes exactly 4 000 training-split frames."""

N_BOOTSTRAP_MINIBATCHES = 200
"""10.4.2: 200 minibatches, bootstrapped."""

CHECKPOINT_FRACTIONS: tuple[float, ...] = (0.25, 0.50, 1.00)
"""The final checkpoint plus one at 25% and one at 50% of training (10.4.2)."""

NONTRIVIAL_POOL_CLEAN_FAMILIES = 10
"""The non-trivial pool is 10 clean morphology families with a defect injected in each."""

P7_MIN_GAIN_PP = 3.0
"""The P7 positive control: J(B3) - J(B2) >= 3 pp on a real morphology difference."""

BASELINE_EG_IMPROVEMENT = 0.34
"""IQL+EG over IQL on the 70% suboptimal datasets (arXiv 2602.18025), for contrast."""

BASELINE_SUBOPTIMAL_SHARE = 0.70
"""The dataset mix that 0.34 was measured on. Carried with it; the pair is the fact."""

# 10.4.2's cost line: 18 runs = non-trivial pool B2/B3 at 3 seeds (6) + the P7
# positive control B2/B3 at 3 seeds (6) + B3's PCGrad and GradNorm variants at 3
# seeds (6). At 9.8.1's unit training cost that is 90 GPU-h, which is what the
# experiment config declares.
E2PRIME_SEEDS_PER_ARM = 3
GPU_HOURS_PER_RUN = 5.0
"""9.8.1's unit training cost u_tr, the single conservative value for every arm."""

# 10.4.2's transcription of the same-DoF-different-reach level. These are
# manufacturer specification figures quoted in 10.4.2 and 9.3.1, not quantities
# this study measures, so they are constants here rather than lock-file entries
# (CONTRACT 3.1). The original Panda vs FR3 pairing is void: the FR3 asset is not
# obtainable (9.3.1), and Kinova3 replaces it.
SAME_DOF_REACH_M: dict[str, float] = {"Panda": 0.855, "Kinova3": 0.902}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DegenerateGradientTensorError(ValueError):
    """A parameter tensor's gradient has zero norm, so its cosine is undefined."""


class GradientStructureMismatch(ValueError):
    """Two gradients do not carry the same parameter tensors."""


class MissingCheckpointError(ValueError):
    """A reading was assembled without all three checkpoints of 10.4.2."""


class MissingUpperAnchorError(ValueError):
    """A checkpoint has no upper-anchor samples, so it has no null to be read against."""


class UnequalMinibatchCountError(ValueError):
    """Two levels were read on different numbers of minibatches."""


class MixedMinibatchPlanError(ValueError):
    """Samples taken under different minibatch plans were pooled."""


class BaselineIsNotAComparisonBaseline(ValueError):
    """The published r = 0.63 was used as a number to test ours against."""


class PositiveControlFailed(ValueError):
    """P7 did not clear 3 pp, so the twin-pair reading cannot be attributed."""


class DegenerateRegressorError(ValueError):
    """The morphology-distance axis has no spread, so no line can be fitted on it."""


# ---------------------------------------------------------------------------
# The measurement protocol (10.4.2, "all preregistered")
# ---------------------------------------------------------------------------


class Checkpoint(Enum):
    """Where in training the gradients are read.

    Three points, not one: 10.4.2 makes gradient conflict a process quantity and
    says in as many words that a single reading is not enough.
    """

    QUARTER = 0.25
    HALF = 0.50
    FINAL = 1.00

    @property
    def fraction(self) -> float:
        return float(self.value)

    def __str__(self) -> str:
        return "final" if self is Checkpoint.FINAL else f"{self.fraction:.0%} of training"


class SourcePairLevel(Enum):
    """10.4.2's independent variable, plus the null it is read against.

    The three morphology-distance levels are a ladder in FGW distance: zero on a
    structurally aligned twin pair (9.3.4 asserts it), the smallest non-zero step
    between two arms of the same DoF, and the largest step in the registry's core
    six. `UPPER_ANCHOR` is not a fourth level -- it is the no-conflict control,
    two disjoint minibatches drawn from one and the same source, whose cosine has
    no morphology difference in it at all.
    """

    TWIN_PAIR = "twin pair, FGW = 0"
    SAME_DOF_DIFFERENT_REACH = "same DoF, different reach"
    CROSS_DOF = "cross DoF"
    UPPER_ANCHOR = "no-conflict upper anchor: two disjoint minibatches of one source"

    @property
    def is_distance_level(self) -> bool:
        return self is not SourcePairLevel.UPPER_ANCHOR

    @property
    def families(self) -> tuple[str, str] | None:
        """The family pair 10.4.2 names for this level, when it names one.

        The twin-pair level and the anchor are defined by a relation rather than
        by a pair of names -- any family against its own structurally aligned
        twin, and any source against itself -- so they return `None` instead of a
        pair the proposal does not fix.
        """
        if self is SourcePairLevel.SAME_DOF_DIFFERENT_REACH:
            return ("Panda", "Kinova3")
        if self is SourcePairLevel.CROSS_DOF:
            return ("Panda", "UR5e")
        return None


DISTANCE_LEVELS: tuple[SourcePairLevel, ...] = tuple(
    level for level in SourcePairLevel if level.is_distance_level
)


@dataclass(frozen=True)
class MinibatchPlan:
    """How 10.4.2's 4 000 frames per source are cut into minibatches.

    Both sides of a pair get *strictly equal* frame counts split into the same
    number of minibatches, and the upper anchor uses the same batch size. The
    reason is not symmetry for its own sake: the cosine between two minibatch
    gradients rises towards one with the batch size, because both estimates
    converge on the same full-batch gradient. A level read at a larger batch than
    another would sit higher for that reason alone, and the whole endpoint is a
    comparison between levels.
    """

    frames_per_source: int = FRAMES_PER_SOURCE
    n_minibatches: int = N_BOOTSTRAP_MINIBATCHES

    def __post_init__(self) -> None:
        if self.frames_per_source <= 0 or self.n_minibatches <= 0:
            raise ValueError(
                f"frames_per_source={self.frames_per_source}, "
                f"n_minibatches={self.n_minibatches}: both must be positive"
            )
        if self.frames_per_source % self.n_minibatches:
            raise ValueError(
                f"{self.frames_per_source} frames do not divide into "
                f"{self.n_minibatches} minibatches. 10.4.2 fixes equal batch frame "
                "counts on both sides of every pair, and a remainder would make the "
                "last minibatch smaller than the rest -- the cosine is sensitive to "
                "the effective batch size, so an odd batch is a different estimator"
            )

    @property
    def minibatch_frames(self) -> int:
        return self.frames_per_source // self.n_minibatches

    def describe(self) -> str:
        return (
            f"{self.frames_per_source} training-split frames per source, "
            f"{self.n_minibatches} minibatches of {self.minibatch_frames} frames; "
            "the upper anchor uses the same batch size (10.4.2)"
        )


# ---------------------------------------------------------------------------
# The published claim, as a citation and not as a yardstick
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineGradientCosineClaim:
    """arXiv 2602.18025's r = 0.63 -- quotable, not comparable.

    10.4.2 is explicit on both halves. The claim is quoted to establish that a
    similarity-gradient bridge is asserted in the literature at all. It is *not*
    a comparison baseline: it is fitted on legged platforms, offline RL, another
    node feature set and another gradient-cosine caliber, so its x axis is on a
    different scale from ours and a difference between the two r values would
    measure the scales, not the bridge.

    The p value carries a mandatory caveat: 16 platforms give 120 platform pairs
    and those pairs are not independent, so 1.26e-14 may not be read as an
    independent-sample p value. `independent_sample_p` therefore raises instead
    of returning it.
    """

    pearson_r: float = 0.63
    p_value: float = 1.26e-14
    n_platforms: int = 16
    citation: str = "arXiv:2602.18025 (ICLR 2026), locomotion, 16 platforms, offline RL"

    @property
    def n_platform_pairs(self) -> int:
        """The 120 pairs of 10.4.2, recomputed from the platform count."""
        return self.n_platforms * (self.n_platforms - 1) // 2

    @property
    def caveat(self) -> str:
        return (
            f"the {self.n_platform_pairs} platform pairs from {self.n_platforms} "
            f"platforms are not independent, so p = {self.p_value:.3g} may not be read "
            "as an independent-sample p value (10.4.2)"
        )

    def independent_sample_p(self) -> float:
        raise BaselineIsNotAComparisonBaseline(self.caveat)

    def compare_with(self, our_r: float) -> float:
        raise BaselineIsNotAComparisonBaseline(
            f"our r = {our_r:.4g} may not be tested against the published "
            f"{self.pearson_r}: 10.4.2 records that the published line is fitted on "
            "legged platforms, offline RL, another node feature set and another "
            "gradient-cosine caliber, so the two x axes are not on one scale. E2's "
            "own line is fitted on our own non-trivial pool and reported with its "
            "own confidence band; the published number is cited, not compared"
        )


BASELINE_CLAIM = BaselineGradientCosineClaim()


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------


def weighted_tensor_cosine(
    a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray]
) -> float:
    """cos(g_k, g_k') per parameter tensor, weighted by parameter count (10.4.2).

    Not the cosine of the two flattened gradients: 10.4.2 says "compute the
    cosine per parameter tensor, then weight by parameter count", and the two are
    different numbers. A flattened cosine is dominated by whichever tensor has
    the largest gradient norm, typically the vision trunk's first convolution,
    whereas the per-tensor form asks the question once of every tensor and then
    weights by how much of the model each one is.

    A tensor whose gradient vanishes on one side has no direction, so its cosine
    does not exist. That raises rather than contributing a zero: a zero is the
    value of "orthogonal", which is a strong claim about a tensor that in fact
    said nothing, and it would drag the weighted mean towards zero in proportion
    to the dead tensor's size.
    """
    if set(a) != set(b):
        only_a = sorted(set(a) - set(b))
        only_b = sorted(set(b) - set(a))
        raise GradientStructureMismatch(
            f"the two gradients carry different parameter tensors; only in the first "
            f"{only_a}, only in the second {only_b}. A per-tensor cosine is defined "
            "on one and the same parameterisation"
        )
    if not a:
        raise GradientStructureMismatch("no parameter tensors: there is nothing to align")

    total_weight = 0.0
    total = 0.0
    for name in sorted(a):
        ga = np.asarray(a[name], dtype=np.float64).reshape(-1)
        gb = np.asarray(b[name], dtype=np.float64).reshape(-1)
        if ga.shape != gb.shape:
            raise GradientStructureMismatch(
                f"tensor {name!r} has {ga.size} parameters on one side and {gb.size} "
                "on the other"
            )
        na = float(np.linalg.norm(ga))
        nb = float(np.linalg.norm(gb))
        if na == 0.0 or nb == 0.0:
            raise DegenerateGradientTensorError(
                f"tensor {name!r} has a zero-norm gradient on "
                f"{'both sides' if na == nb else 'one side'}, so cos(g_k, g_k') is "
                "undefined there. It is not counted as an orthogonal pair: zero is "
                "the value of 'orthogonal', and weighting a dead tensor by its "
                "parameter count would pull the weighted mean towards zero in "
                "proportion to its size"
            )
        weight = float(ga.size)
        total += weight * float(np.dot(ga, gb) / (na * nb))
        total_weight += weight
    return total / total_weight


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CosineSample:
    """One minibatch's weighted gradient cosine, at one level and one checkpoint."""

    level: SourcePairLevel
    checkpoint: Checkpoint
    minibatch_index: int
    cosine: float
    plan: MinibatchPlan
    loss: str = "bc"
    split: str = "train"

    def __post_init__(self) -> None:
        if not math.isfinite(self.cosine):
            raise ValueError(
                f"{self.level.name}/{self.checkpoint.name}/mb{self.minibatch_index}: "
                "the cosine is not finite"
            )
        if not -1.0 - 1e-9 <= self.cosine <= 1.0 + 1e-9:
            raise ValueError(
                f"{self.level.name}/{self.checkpoint.name}/mb{self.minibatch_index}: "
                f"cosine {self.cosine} is outside [-1, 1]"
            )
        if self.loss != "bc":
            raise ValueError(
                f"loss {self.loss!r}: 10.4.2 fixes the loss at the BC loss. A cosine "
                "taken on a different objective is a different quantity and cannot "
                "share a table with this one"
            )
        if self.split != "train":
            raise ValueError(
                f"split {self.split!r}: 10.4.2 fixes the frames at the training "
                "split, which is the split the gradients under test were taken on"
            )


class GradientCosineObservations:
    """The samples of one E2' reading, checked for the protocol of 10.4.2."""

    def __init__(self, samples: Sequence[CosineSample]) -> None:
        if not samples:
            raise ValueError("E2' has no gradient-cosine samples")
        plans = {s.plan for s in samples}
        if len(plans) > 1:
            raise MixedMinibatchPlanError(
                f"the samples were taken under {len(plans)} different minibatch plans "
                f"{[p.describe() for p in sorted(plans, key=lambda p: p.minibatch_frames)]}. "
                "The gradient cosine rises with the effective batch size, so two "
                "levels read at different batch sizes differ for that reason alone "
                "(10.4.2 fixes equal batches on both sides and the same size for the "
                "anchor)"
            )
        self._plan = next(iter(plans))
        self._samples = tuple(samples)

        present = {s.checkpoint for s in self._samples}
        missing = [c for c in Checkpoint if c not in present]
        if missing:
            raise MissingCheckpointError(
                f"checkpoints {[c.name for c in missing]} have no samples. 10.4.2 "
                "reads the final checkpoint plus one at 25% and one at 50% of "
                "training, because gradient conflict is a training-process quantity "
                "and a single reading is not enough"
            )
        for checkpoint in Checkpoint:
            for level in SourcePairLevel:
                if not self._at(level, checkpoint):
                    if level is SourcePairLevel.UPPER_ANCHOR:
                        raise MissingUpperAnchorError(
                            f"{checkpoint.name} has no upper-anchor samples. The "
                            "no-conflict anchor -- two disjoint minibatches from the "
                            "same source -- is E2's null, and T12 requires both "
                            "distributions to sit significantly below it"
                        )
                    raise MissingCheckpointError(
                        f"{checkpoint.name} has no samples at level {level.name}. "
                        "10.4.2 reads all three morphology-distance levels at every "
                        "checkpoint; a level dropped at one checkpoint and kept at "
                        "another would make the three points incomparable"
                    )
        counts = {
            (level, checkpoint): len(self._at(level, checkpoint))
            for level in SourcePairLevel
            for checkpoint in Checkpoint
        }
        if len(set(counts.values())) > 1:
            raise UnequalMinibatchCountError(
                f"the levels were read on different numbers of minibatches: "
                f"{ {f'{k[0].name}@{k[1].name}': v for k, v in sorted(counts.items(), key=str)} }. "
                "10.4.2 splits both sources into the same number of minibatches, and "
                "the bootstrap resamples one shared minibatch index across the levels"
            )
        self._n_minibatches = next(iter(counts.values()))
        if self._n_minibatches != self._plan.n_minibatches:
            raise UnequalMinibatchCountError(
                f"{self._n_minibatches} samples per level against a plan of "
                f"{self._plan.n_minibatches} minibatches"
            )

    @property
    def plan(self) -> MinibatchPlan:
        return self._plan

    @property
    def n_minibatches(self) -> int:
        return self._n_minibatches

    def _at(
        self, level: SourcePairLevel, checkpoint: Checkpoint
    ) -> tuple[CosineSample, ...]:
        return tuple(
            s for s in self._samples if s.level is level and s.checkpoint is checkpoint
        )

    def vector(self, level: SourcePairLevel, checkpoint: Checkpoint) -> np.ndarray:
        """The level's cosines at one checkpoint, ordered by minibatch index.

        Ordered rather than as-collected, because the bootstrap draws one index
        vector and applies it to every level: the pairing is the minibatch
        index, and an unordered array would pair minibatch 3 of one level with
        minibatch 17 of another.
        """
        rows = sorted(self._at(level, checkpoint), key=lambda s: s.minibatch_index)
        indices = [s.minibatch_index for s in rows]
        if len(set(indices)) != len(indices):
            raise ValueError(
                f"{level.name}@{checkpoint.name}: duplicate minibatch indices "
                f"{sorted({i for i in indices if indices.count(i) > 1})}"
            )
        return np.array([s.cosine for s in rows], dtype=float)


# ---------------------------------------------------------------------------
# The two-sample reading (10.4.2's primary criterion, T12)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TwoSampleReading:
    """One checkpoint's T12 reading: delta cos, the two levels, and the anchor."""

    checkpoint: Checkpoint
    twin: BootstrapResult
    cross_dof: BootstrapResult
    same_dof: BootstrapResult
    anchor: BootstrapResult
    delta_cos: BootstrapResult
    twin_minus_anchor: BootstrapResult
    cross_minus_anchor: BootstrapResult
    plan: MinibatchPlan

    @property
    def indistinguishable(self) -> bool:
        """|delta cos| < 0.15 -- the first half of T12."""
        return abs(self.delta_cos.point) < DELTA_COS_INDISTINGUISHABLE

    @property
    def twin_ci_upper(self) -> float:
        return self.twin.ci[1]

    @property
    def cross_ci_upper(self) -> float:
        return self.cross_dof.ci[1]

    @property
    def below_ceiling(self) -> bool:
        """Both 95% CI upper bounds below 0.85 -- the second half of T12.

        10.4.2 names the twin pair's bound explicitly and asks for "both
        significantly below the upper anchor"; the ceiling is applied to both
        levels, because a cross-DoF distribution sitting above it while the twin
        pair sat below would be the bridge holding, not breaking.
        """
        return (
            self.twin_ci_upper < TWIN_PAIR_CI_UPPER_CEILING
            and self.cross_ci_upper < TWIN_PAIR_CI_UPPER_CEILING
        )

    @property
    def significantly_below_anchor(self) -> bool:
        """Both level-minus-anchor CIs lie strictly below zero."""
        return self.twin_minus_anchor.ci[1] < 0.0 and self.cross_minus_anchor.ci[1] < 0.0

    @property
    def t12_holds(self) -> bool:
        return self.indistinguishable and self.below_ceiling

    @property
    def anchor_meets_expectation(self) -> bool:
        return self.anchor.point > UPPER_ANCHOR_EXPECTED_FLOOR

    @property
    def f3_triggered(self) -> bool:
        """10.4.2's F3: the gradient signal did separate the twin pair.

        Both halves are required, as written: the twin pair's CI upper bound
        reaches the 0.85 ceiling *and* the twin pair sits significantly above the
        cross-DoF pair. Either one alone is a different reading -- a high twin
        cosine with no separation is an apparatus that cannot resolve anything.
        """
        return (
            self.twin_ci_upper >= TWIN_PAIR_CI_UPPER_CEILING
            and self.delta_cos.ci[0] > 0.0
        )

    def as_row(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint),
            "delta_cos": self.delta_cos.point,
            "delta_cos_ci": list(self.delta_cos.ci),
            "twin_pair_mean": self.twin.point,
            "twin_pair_ci": list(self.twin.ci),
            "same_dof_mean": self.same_dof.point,
            "same_dof_ci": list(self.same_dof.ci),
            "cross_dof_mean": self.cross_dof.point,
            "cross_dof_ci": list(self.cross_dof.ci),
            "upper_anchor_mean": self.anchor.point,
            "upper_anchor_ci": list(self.anchor.ci),
            "twin_minus_anchor_ci": list(self.twin_minus_anchor.ci),
            "cross_minus_anchor_ci": list(self.cross_minus_anchor.ci),
            "indistinguishable": self.indistinguishable,
            "below_ceiling": self.below_ceiling,
            "significantly_below_anchor": self.significantly_below_anchor,
            "t12_holds": self.t12_holds,
            "f3_triggered": self.f3_triggered,
            "anchor_meets_expectation": self.anchor_meets_expectation,
            "n_minibatches": self.plan.n_minibatches,
            "minibatch_frames": self.plan.minibatch_frames,
        }


def two_sample_reading(
    observations: GradientCosineObservations,
    checkpoint: Checkpoint,
    *,
    rng: np.random.Generator,
    n_resamples: int,
    level: float = 1.0 - ALPHA_PRIMARY,
) -> TwoSampleReading:
    """T12 at one checkpoint, bootstrapped over the minibatches.

    One index vector is drawn per resample and applied to every level, which is
    the discipline of 10.1.3 transplanted from initial states to minibatches: the
    minibatch partition of the training split is common to the levels, so the
    partition layer is perfectly correlated across them and resampling each level
    independently would inflate the variance of delta cos -- the very quantity
    the criterion is a threshold on.
    """
    arms = {
        "twin": observations.vector(SourcePairLevel.TWIN_PAIR, checkpoint),
        "same_dof": observations.vector(
            SourcePairLevel.SAME_DOF_DIFFERENT_REACH, checkpoint
        ),
        "cross_dof": observations.vector(SourcePairLevel.CROSS_DOF, checkpoint),
        "anchor": observations.vector(SourcePairLevel.UPPER_ANCHOR, checkpoint),
    }
    statistics = {
        "twin": lambda a: float(a["twin"].mean()),
        "same_dof": lambda a: float(a["same_dof"].mean()),
        "cross_dof": lambda a: float(a["cross_dof"].mean()),
        "anchor": lambda a: float(a["anchor"].mean()),
        "delta_cos": lambda a: float(a["twin"].mean() - a["cross_dof"].mean()),
        "twin_minus_anchor": lambda a: float(a["twin"].mean() - a["anchor"].mean()),
        "cross_minus_anchor": lambda a: float(
            a["cross_dof"].mean() - a["anchor"].mean()
        ),
    }
    boot = paired_block_bootstrap_multi(
        arms, statistics, n_resamples=n_resamples, rng=rng, level=level
    )
    return TwoSampleReading(
        checkpoint=checkpoint,
        twin=boot["twin"],
        same_dof=boot["same_dof"],
        cross_dof=boot["cross_dof"],
        anchor=boot["anchor"],
        delta_cos=boot["delta_cos"],
        twin_minus_anchor=boot["twin_minus_anchor"],
        cross_minus_anchor=boot["cross_minus_anchor"],
        plan=observations.plan,
    )


# ---------------------------------------------------------------------------
# Our own regression line (10.4.2, "our own regression line")
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OwnRegressionLine:
    """The morphology-distance / gradient-cosine line refitted on our own pool.

    10.4.2 asks for our own r with its confidence band, fitted on the non-trivial
    pool (10 clean families with a defect injected in each), and for the twin
    pair to be reported as this line's residual at the origin -- the twin pair
    sits at distance zero by construction, so the line's intercept is what it
    predicts there and the residual is a number about our own line rather than
    about anybody else's.

    The band is a Fisher z interval on r. `n_pairs` is the number of source pairs
    the line was fitted on, and it is carried because the pairs are not
    independent: they are built from a smaller number of sources, so the interval
    is optimistic in exactly the way 10.4.2 warns the published p value is.
    """

    slope: float
    intercept: float
    pearson_r: float
    r_ci: tuple[float, float]
    n_pairs: int
    n_sources: int
    twin_pair_cosine: float
    level: float

    @property
    def predicted_at_zero(self) -> float:
        return self.intercept

    @property
    def twin_pair_residual(self) -> float:
        """The twin pair as this line's origin residual (10.4.2)."""
        return self.twin_pair_cosine - self.predicted_at_zero

    @property
    def r_ci_excludes_zero(self) -> bool:
        lo, hi = self.r_ci
        return lo > 0.0 or hi < 0.0

    @property
    def pairs_are_independent(self) -> bool:
        """Always False, and reported: C(n, 2) pairs come from n sources.

        The same non-independence 10.4.2 attaches to the published p value
        attaches to ours. Stating it as a property rather than as a comment is
        what puts it in the results table.
        """
        return False

    def as_row(self) -> dict[str, Any]:
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "pearson_r": self.pearson_r,
            "r_ci": list(self.r_ci),
            "r_ci_excludes_zero": self.r_ci_excludes_zero,
            "n_sources": self.n_sources,
            "n_pairs": self.n_pairs,
            "pairs_are_independent": self.pairs_are_independent,
            "twin_pair_cosine": self.twin_pair_cosine,
            "predicted_at_distance_zero": self.predicted_at_zero,
            "twin_pair_origin_residual": self.twin_pair_residual,
            "comparison_note": (
                "the published r is cited to establish that the claim exists and is "
                "never used as a comparison baseline (10.4.2)"
            ),
        }


def fit_own_regression_line(
    *,
    distances: Sequence[float],
    cosines: Sequence[float],
    n_sources: int,
    twin_pair_cosine: float,
    level: float = 1.0 - ALPHA_PRIMARY,
) -> OwnRegressionLine:
    """Least squares of gradient cosine on morphology distance, with a Fisher band."""
    x = np.asarray(distances, dtype=float)
    y = np.asarray(cosines, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(
            f"distances {x.shape} and cosines {y.shape} must be one aligned vector"
        )
    if x.size < 4:
        raise ValueError(
            f"{x.size} source pairs: a Fisher interval on r needs n > 3, and 10.4.2's "
            "non-trivial pool supplies far more than that"
        )
    if float(np.std(x)) <= 0.0:
        raise DegenerateRegressorError(
            "every source pair has the same morphology distance, so no line can be "
            "fitted on it and 'morphology distance predicts gradient cosine' has no "
            "content on this pool. The non-trivial pool of 10.4.2 exists precisely to "
            "give the clustering a partition it can act on"
        )
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    n = int(x.size)
    z = math.atanh(max(min(r, 1.0 - 1e-12), -1.0 + 1e-12))
    se = 1.0 / math.sqrt(n - 3)
    crit = float(sps.norm.ppf(1.0 - (1.0 - level) / 2.0))
    lo, hi = math.tanh(z - crit * se), math.tanh(z + crit * se)
    expected_pairs = n_sources * (n_sources - 1) // 2
    if n > expected_pairs:
        raise ValueError(
            f"{n} pairs from {n_sources} sources, but {n_sources} sources give at most "
            f"{expected_pairs} distinct pairs"
        )
    return OwnRegressionLine(
        slope=float(slope),
        intercept=float(intercept),
        pearson_r=r,
        r_ci=(lo, hi),
        n_pairs=n,
        n_sources=int(n_sources),
        twin_pair_cosine=float(twin_pair_cosine),
        level=level,
    )


# ---------------------------------------------------------------------------
# The non-trivial pool's variance decomposition and the P7 positive control
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VarianceDecomposition:
    """J(B3) - J(B2) split by the source of the pair (10.4.2's non-trivial pool).

    On the non-trivial pool the clustering produces a real partition, so the
    question becomes empirical: of the negative transfer it recovers, what share
    comes from the defect axis? The decomposition is over two disjoint sources of
    variance in the per-pair gains -- pairs that cross a morphology family, and
    pairs inside one family that cross a defect -- and the reported number is the
    second one's share.

    Both source lists are required and both must be non-empty: a decomposition
    with one source is not a decomposition, and 10.4.2's whole quantity is the
    share of the second.
    """

    cross_family_gains: tuple[float, ...]
    same_family_cross_defect_gains: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("cross_family_gains", "same_family_cross_defect_gains"):
            values = getattr(self, name)
            if len(values) < 2:
                raise ValueError(
                    f"{name} has {len(values)} pair(s); a variance needs at least two, "
                    "and a decomposition needs both sources present. 10.4.2's reading "
                    "is the defect-axis share, which is undefined without the "
                    "cross-family part to read it against"
                )
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"{name} contains a non-finite gain")

    @staticmethod
    def _var(values: tuple[float, ...]) -> float:
        return float(np.var(np.asarray(values, dtype=float), ddof=1))

    @property
    def cross_family_variance(self) -> float:
        return self._var(self.cross_family_gains)

    @property
    def defect_axis_variance(self) -> float:
        return self._var(self.same_family_cross_defect_gains)

    @property
    def defect_axis_share(self) -> float:
        total = self.cross_family_variance + self.defect_axis_variance
        if total <= 0.0:
            raise DegenerateRegressorError(
                "both sources have zero variance in the per-pair gain, so there is no "
                "recovered negative transfer to apportion"
            )
        return self.defect_axis_variance / total

    @property
    def mean_gain_pp(self) -> float:
        """The pooled mean of J(B3) - J(B2) over every pair, in percentage points."""
        pooled = list(self.cross_family_gains) + list(
            self.same_family_cross_defect_gains
        )
        return float(np.mean(pooled))

    def as_row(self) -> dict[str, Any]:
        return {
            "n_cross_family_pairs": len(self.cross_family_gains),
            "n_same_family_cross_defect_pairs": len(self.same_family_cross_defect_gains),
            "cross_family_variance": self.cross_family_variance,
            "defect_axis_variance": self.defect_axis_variance,
            "defect_axis_share": self.defect_axis_share,
            "mean_gain_pp": self.mean_gain_pp,
            "contrast_number": (
                f"for contrast, {BASELINE_CLAIM.citation} reports "
                f"{BASELINE_EG_IMPROVEMENT:.0%} for IQL+EG over IQL on its "
                f"{BASELINE_SUBOPTIMAL_SHARE:.0%}-suboptimal datasets; a different "
                "domain and a different metric, quoted and not compared"
            ),
        }


@dataclass(frozen=True)
class P7PositiveControl:
    """Morphology clustering does gain on a real morphology difference (10.4.2).

    Mandatory. Without it, "it does nothing on the twin pairs" is consistent with
    the baseline simply having been run badly, and the twin-pair reading could
    not be attributed to the bridge. The preregistered bar is J(B3) - J(B2) >= 3
    pp on Panda vs UR5e.
    """

    gain_pp: float
    families: tuple[str, str] = ("Panda", "UR5e")

    def __post_init__(self) -> None:
        if not math.isfinite(self.gain_pp):
            raise ValueError("the P7 gain is not finite")

    @property
    def passes(self) -> bool:
        return self.gain_pp >= P7_MIN_GAIN_PP

    def require_pass(self) -> None:
        if not self.passes:
            raise PositiveControlFailed(
                f"P7 gives J(B3) - J(B2) = {self.gain_pp:.4g} pp on "
                f"{self.families[0]} vs {self.families[1]}, below the preregistered "
                f"{P7_MIN_GAIN_PP} pp. 10.4.2 makes this control mandatory: without "
                "it, a null twin-pair reading is equally consistent with the baseline "
                "having been run badly, so the twin-pair result cannot be attributed "
                "to the similarity-gradient bridge and the verdict is inconclusive"
            )


# ---------------------------------------------------------------------------
# The run budget of 10.4.2, recomputed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class E2PrimeRunBudget:
    """10.4.2's 18 runs and 90 GPU-h, as an arithmetic rather than as two numbers."""

    seeds_per_arm: int = E2PRIME_SEEDS_PER_ARM
    gpu_hours_per_run: float = GPU_HOURS_PER_RUN

    @property
    def components(self) -> dict[str, int]:
        """The three blocks 10.4.2 adds up, each in runs."""
        return {
            "non-trivial pool, B2 and B3": 2 * self.seeds_per_arm,
            "P7 positive control, B2 and B3": 2 * self.seeds_per_arm,
            "B3's PCGrad and GradNorm variants": 2 * self.seeds_per_arm,
        }

    @property
    def n_runs(self) -> int:
        return sum(self.components.values())

    @property
    def gpu_hours(self) -> float:
        return self.n_runs * self.gpu_hours_per_run

    @property
    def primary_criterion_new_runs(self) -> int:
        """Zero: T12 is read on the already-trained pooled model (10.4.2)."""
        return 0

    def as_row(self) -> dict[str, Any]:
        return {
            "components": self.components,
            "n_runs": self.n_runs,
            "gpu_hours": self.gpu_hours,
            "gpu_hours_per_run": self.gpu_hours_per_run,
            "primary_criterion_new_runs": self.primary_criterion_new_runs,
        }


E2PRIME_RUN_BUDGET = E2PrimeRunBudget()
E2PRIME_RUNS = E2PRIME_RUN_BUDGET.n_runs
E2PRIME_GPU_HOURS = E2PRIME_RUN_BUDGET.gpu_hours


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

WORDING = (
    "morphology distance has no predictive power for gradient cosine along the "
    "defect axis; the similarity-gradient bridge breaks at this point"
)
"""10.4.2's wording discipline, verbatim. Not "morphology clustering does not work"."""

CONSTRUCTIVE_FACTS: tuple[str, ...] = (
    "the FGW numerical matrix on the structurally aligned twin pairs, one under "
    "FEATSET_OURS and one under FEATSET_BASELINE, produced by "
    "portal_trace.registry.twins.fgw_matrix",
    "the actual clustering partition: on a twin pair the FGW distance is zero, so "
    "hierarchical clustering returns a single group and B3's group-wise update rule "
    "is literally B2's",
)
"""10.4.2's two items reported *beside* the test and never tested (appendix A-2).

Both are constructive: the first is a matrix the registry computes, the second is
an algebraic consequence of a zero distance. Appendix A-2 names running a
statistical test on the second as a mistake -- the true difference is zero by
construction, so a paired difference contains nothing but the seed noise of two
independent trainings, and testing it spends compute proving a definition.
"""


@dataclass(frozen=True)
class E2PrimeReport:
    """The `gradient_cosine_two_sample` row, over all three checkpoints."""

    readings: tuple[TwoSampleReading, ...]
    positive_control: P7PositiveControl
    own_line: OwnRegressionLine | None
    decomposition: VarianceDecomposition | None
    budget: E2PrimeRunBudget = E2PRIME_RUN_BUDGET

    def __post_init__(self) -> None:
        seen = [r.checkpoint for r in self.readings]
        if sorted(c.fraction for c in seen) != sorted(CHECKPOINT_FRACTIONS):
            raise MissingCheckpointError(
                f"the report carries checkpoints {[str(c) for c in seen]}; 10.4.2 "
                f"fixes {list(CHECKPOINT_FRACTIONS)} of training"
            )

    def reading(self, checkpoint: Checkpoint) -> TwoSampleReading:
        for r in self.readings:
            if r.checkpoint is checkpoint:
                return r
        raise MissingCheckpointError(f"no reading at {checkpoint.name}")

    @property
    def t12_holds_at_every_checkpoint(self) -> bool:
        return all(r.t12_holds for r in self.readings)

    @property
    def f3_triggered(self) -> bool:
        """F3 fires if the gradient signal separated the twin pair at any checkpoint."""
        return any(r.f3_triggered for r in self.readings)

    @property
    def verdict(self) -> str:
        """The three-way verdict of 11.3.

        `inconclusive` when the P7 positive control fails: the twin-pair reading
        is then not attributable, whichever way it came out. Otherwise T12 is
        supported only if it holds at all three checkpoints -- 10.4.2 makes the
        cosine a process quantity and reads three points for that reason, so a
        bridge that breaks at one of them and holds at another has not produced
        the indistinguishability the claim is about.
        """
        if not self.positive_control.passes:
            return "inconclusive"
        if self.f3_triggered:
            return "F3: the gradient signal separates the twin pair"
        return "T12 supported" if self.t12_holds_at_every_checkpoint else "not significant"

    @property
    def verdict_kind(self) -> Verdict:
        """The same reading in 11.3's three-valued vocabulary.

        F3 maps to `NOT_SIGNIFICANT` rather than to a fourth value: the endpoint
        under test is T12, and a twin pair the gradient signal separates is T12
        failing. F3 is the narrative that a failure opens, carried beside the
        verdict in `f3_triggered`, and 11.3 allows exactly three verdicts.
        """
        if not self.positive_control.passes:
            return Verdict.INCONCLUSIVE
        if self.t12_holds_at_every_checkpoint and not self.f3_triggered:
            return Verdict.SIGNIFICANT
        return Verdict.NOT_SIGNIFICANT

    @property
    def wording(self) -> str:
        return WORDING

    def as_row(self) -> dict[str, Any]:
        return {
            "endpoint": "gradient_cosine_two_sample",
            "verdict": self.verdict,
            "verdict_kind": self.verdict_kind.value,
            "wording": self.wording,
            "t12_holds_at_every_checkpoint": self.t12_holds_at_every_checkpoint,
            "f3_triggered": self.f3_triggered,
            "delta_cos_threshold": DELTA_COS_INDISTINGUISHABLE,
            "twin_pair_ci_ceiling": TWIN_PAIR_CI_UPPER_CEILING,
            "checkpoints": [r.as_row() for r in self.readings],
            "positive_control": {
                "families": list(self.positive_control.families),
                "gain_pp": self.positive_control.gain_pp,
                "threshold_pp": P7_MIN_GAIN_PP,
                "passes": self.positive_control.passes,
            },
            "own_regression_line": None if self.own_line is None else self.own_line.as_row(),
            "non_trivial_pool_decomposition": (
                None if self.decomposition is None else self.decomposition.as_row()
            ),
            "baseline_claim": {
                "pearson_r": BASELINE_CLAIM.pearson_r,
                "p_value": BASELINE_CLAIM.p_value,
                "n_platforms": BASELINE_CLAIM.n_platforms,
                "n_platform_pairs": BASELINE_CLAIM.n_platform_pairs,
                "caveat": BASELINE_CLAIM.caveat,
                "used_as_comparison_baseline": False,
            },
            "constructive_facts_reported_not_tested": list(CONSTRUCTIVE_FACTS),
            "budget": self.budget.as_row(),
        }


def e2prime_report(
    observations: GradientCosineObservations,
    *,
    positive_control: P7PositiveControl,
    rng: np.random.Generator,
    n_resamples: int,
    own_line: OwnRegressionLine | None = None,
    decomposition: VarianceDecomposition | None = None,
    level: float = 1.0 - ALPHA_PRIMARY,
) -> E2PrimeReport:
    """Read T12 at all three checkpoints and assemble the row."""
    readings = tuple(
        two_sample_reading(
            observations, checkpoint, rng=rng, n_resamples=n_resamples, level=level
        )
        for checkpoint in Checkpoint
    )
    return E2PrimeReport(
        readings=readings,
        positive_control=positive_control,
        own_line=own_line,
        decomposition=decomposition,
    )


def n_source_pairs(n_sources: int) -> int:
    """C(n, 2), the pair count a pool of `n_sources` gives a pairwise regression."""
    if n_sources < 2:
        raise ValueError(f"{n_sources} sources give no pairs")
    return len(list(combinations(range(n_sources), 2)))


def _self_check() -> None:
    """Every printed number of 10.4.2 that has inputs, recomputed from them."""
    plan = MinibatchPlan()
    if plan.minibatch_frames * plan.n_minibatches != FRAMES_PER_SOURCE:
        raise AssertionError(
            f"{plan.n_minibatches} minibatches of {plan.minibatch_frames} frames do "
            f"not make up 10.4.2's {FRAMES_PER_SOURCE} frames per source"
        )
    if BASELINE_CLAIM.n_platform_pairs != 120:
        raise AssertionError(
            f"10.4.2 says 16 platforms give 120 platform pairs; C(16, 2) here is "
            f"{BASELINE_CLAIM.n_platform_pairs}"
        )
    if n_source_pairs(BASELINE_CLAIM.n_platforms) != BASELINE_CLAIM.n_platform_pairs:
        raise AssertionError("the pair count and the platform count disagree")
    if E2PRIME_RUN_BUDGET.n_runs != 18:
        raise AssertionError(
            f"10.4.2's cost line adds 6 + 6 + 6 = 18 runs; the components here give "
            f"{E2PRIME_RUN_BUDGET.n_runs} ({E2PRIME_RUN_BUDGET.components})"
        )
    if E2PRIME_RUN_BUDGET.gpu_hours != 90.0:
        raise AssertionError(
            f"18 runs at 9.8.1's {GPU_HOURS_PER_RUN} GPU-h/run is 90 GPU-h; this "
            f"gives {E2PRIME_RUN_BUDGET.gpu_hours}"
        )
    if tuple(c.fraction for c in Checkpoint) != CHECKPOINT_FRACTIONS:
        raise AssertionError(
            f"the checkpoint ladder is {[c.fraction for c in Checkpoint]}, 10.4.2 "
            f"fixes {list(CHECKPOINT_FRACTIONS)}"
        )
    reaches = sorted(SAME_DOF_REACH_M.values())
    if reaches[0] >= reaches[-1]:
        raise AssertionError(
            "the same-DoF-different-reach level needs two different reaches; "
            f"{SAME_DOF_REACH_M} gives one"
        )
    if set(SAME_DOF_REACH_M) != set(SourcePairLevel.SAME_DOF_DIFFERENT_REACH.families):
        raise AssertionError(
            f"the reach table names {sorted(SAME_DOF_REACH_M)} and the level names "
            f"{sorted(SourcePairLevel.SAME_DOF_DIFFERENT_REACH.families)}"
        )


_self_check()
