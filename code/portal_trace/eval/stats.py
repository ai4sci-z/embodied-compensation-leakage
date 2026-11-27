"""Inference for the preregistered endpoints: proposal 10.7 and 11.3.

Five things live here, and they are the five the proposal names:

* the **crossed random effects mixed model** of 10.7.2, morphology family and
  defect type both random, seeds nested underneath;
* the **within-family restricted permutation test**, the co-primary test, which
  wins when the two disagree (10.7.2);
* the **paired block bootstrap** of 10.1.3 and 11.3 -- initial-state indices are
  resampled once and *shared by every arm*, because the evaluation initial-state
  set is common across arms and independent resampling would put the shared
  layer back into a paired contrast;
* **Holm-Bonferroni** over the five members of the primary hypothesis family
  (10.7.6);
* the **variance model** of 10.7.3/11.3 and the two MDE formulas (paired
  difference, and the slope form of 10.7.5).

Two design decisions are worth stating because they are load bearing.

The first: an `Mde` never travels as a bare float. 10.7.4 fixes three calibers
of the same quantity -- excluding the seed term (the hard lower bound of
14.5 E-13), including it, and including it with the t correction, which is the
main-table caliber -- and requires that any quotation of an MDE say which one it
is. Carrying the caliber in the object is how a report gets that for free.

The second: `fit_crossed_random_effects` *refuses* to run at M <= 3 rather than
running and returning a zero variance component. 10.7.2 says the degradation to
a fixed morphology effect with within-family cluster-robust SE "is not
optional" at M <= 3, and that every row of the summary table must state which
inference it used. A silent choice would satisfy the arithmetic and break the
reporting rule, so the caller has to name the fallback and the result carries
the `InferenceKind` that `report.py` then prints.

Sections cited inline. Every number that comes from the proposal is a named
constant next to its section number (contract 3.1: stated values are
specification, measured ones go through `portal_trace.lockfile`).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import stats as sps

__all__ = [
    "ALPHA_PRIMARY",
    "ALPHA_SECONDARY_FDR",
    "TARGET_POWER",
    "MDE_COEFFICIENT_NORMAL",
    "MDE_COEFFICIENT_T",
    "N_FIXED_INITIAL_STATES",
    "CROWN_UNDERPOWER_BAND",
    "PRIMARY_HYPOTHESIS_FAMILY",
    "Design",
    "DefectLayer",
    "VarianceComponents",
    "MdeCaliber",
    "Mde",
    "var_paired_mean",
    "paired_difference_mde",
    "slope_mde",
    "slope_mde_degenerate_m1",
    "LAMBDA_PART_COMPONENTS",
    "SlopeMdeRow",
    "SLOPE_MDE_TABLE",
    "slope_degenerate_inflation",
    "InferenceKind",
    "preregistered_inference",
    "ModelDiagnostics",
    "CellObservations",
    "EffectEstimate",
    "fit_crossed_random_effects",
    "fit_family_fixed_cluster_robust",
    "PermutationUnit",
    "PermutationResult",
    "within_family_restricted_permutation",
    "BootstrapResult",
    "paired_block_bootstrap",
    "paired_block_bootstrap_multi",
    "HolmResult",
    "holm_bonferroni",
    "intersection_union_p",
    "benjamini_hochberg",
    "AbsoluteRateInterval",
    "absolute_success_rate_ci",
    "DegenerateModelError",
    "DesignTooSmallError",
]


# --- preregistered constants (10.7.4, 10.7.6, 10.1.1) ----------------------

ALPHA_PRIMARY = 0.05
"""Family-wise error rate for the primary hypothesis family (10.7.6)."""

ALPHA_SECONDARY_FDR = 0.10
"""Benjamini-Hochberg FDR level for the secondary endpoints (10.7.6)."""

TARGET_POWER = 0.80
"""Preregistered power target (10.7.6)."""

MDE_COEFFICIENT_NORMAL = 2.80
"""z_{1-alpha/2} + z_{power} = 1.960 + 0.842 (10.7.4, "quick estimate only")."""

MDE_COEFFICIENT_T = 2.95
"""t-corrected coefficient, the main-table caliber (10.7.4; x1.053 on 2.80).

10.7.4 derives 2.95 at n = 18 and then prints it for every row of the table,
including the (6,6,3) rows where n = 36. Reproduced as printed; the discrepancy
is recorded rather than silently corrected, because the MDE table is a frozen
preregistration artefact (10.7.8 item 4).
"""

N_FIXED_INITIAL_STATES = 50
"""Evaluation initial states, shared by every arm (10.1.1, 10.4.0)."""

CROWN_UNDERPOWER_BAND = (0.70, 1.00)
"""A crown whose measured effect lands here is inconclusive (10.7.5, 10.7.6)."""

PRIMARY_HYPOTHESIS_FAMILY: tuple[str, ...] = (
    "success_rate_paired_difference",   # E1, primary decision endpoint
    "lambda_part_excess",               # E1, Lambda^exc
    "lambda_emp_slope_pair",            # E7, the res/con slope pair as one hypothesis
    "connectivity_slope",               # E8 crown, Lambda^part against ln R_eff
    "isolation_slope",                  # E9 crown, Lambda^part against I
)
"""The five members of the primary family, in the order 10.7.6 lists them.

The names are the ones the experiment configs declare, not a second set of
labels: `configs.loader` checks this tuple against the declared endpoints and
against `prereg.yaml`, so a family that drifts on one side cannot load. An
earlier revision spelled them differently here, which made the two halves of
the repository disagree about what the family was while both passed their own
checks.

Holm's realised ordering is data driven; what 11.3 freezes is the membership.
`holm_bonferroni` checks membership against this tuple and reports the realised
rank, so both readings of "their Holm-Bonferroni positions are fixed" are
available to the reader.
"""

_ZERO_VARIANCE_TOL = 1e-8


class DegenerateModelError(RuntimeError):
    """The mixed model did not converge, or returned a zero variance component.

    11.3 lists both as preregistered triggers for an `inconclusive` verdict, so
    this is raised rather than papered over; `report.py` turns it into a verdict.
    """


class DesignTooSmallError(ValueError):
    """REML was asked for on a design where 10.7.2 forbids it."""


# --- design and variance model (10.7.3, 11.3) ------------------------------


@dataclass(frozen=True)
class Design:
    """The (M, D, S) triple every summary row has to carry (10.7.10, 11.4)."""

    m_families: int
    d_defects: int
    s_seeds: int
    label: str

    def __post_init__(self) -> None:
        for name in ("m_families", "d_defects", "s_seeds"):
            v = getattr(self, name)
            if v < 1:
                raise ValueError(f"{self.label}: {name} must be >= 1, got {v}")

    @property
    def n_cells(self) -> int:
        return self.m_families * self.d_defects

    def __str__(self) -> str:
        return f"(M={self.m_families}, D={self.d_defects}, S={self.s_seeds})"


class DefectLayer(Enum):
    """Whether the defect layer is a random effect the design can generalise over.

    At D = 1 the defect random intercept has no replication: sigma_d cannot be
    estimated, and sigma_d^2/D is not an average over anything. 10.7.3 writes the
    term unconditionally, but the 10.7.4 row for E4b at (4,1,3) prints 9.4 pp,
    which is only reachable with the term dropped (with it, the same inputs give
    15.2 pp). So the term is dropped exactly when the arm declares that it makes
    no cross-defect generalisation claim -- which is what D = 1 forces anyway --
    and the declaration is explicit rather than inferred.
    """

    RANDOM = "random"
    FIXED = "fixed"


@dataclass(frozen=True)
class VarianceComponents:
    """The four layers of 10.7.3, as standard deviations in endpoint units.

    sigma_config^2 = sigma_m^2 + sigma_d^2 + sigma_e^2 and *excludes* the seed
    layer, which is the fourth term and is divided by S on its own.
    """

    sigma_m: float
    sigma_d: float
    sigma_e: float
    sigma_seed: float

    def __post_init__(self) -> None:
        for name in ("sigma_m", "sigma_d", "sigma_e", "sigma_seed"):
            v = getattr(self, name)
            if v < 0 or not math.isfinite(v):
                raise ValueError(f"{name} must be finite and non-negative, got {v}")

    @property
    def sigma_config(self) -> float:
        """sqrt(sigma_m^2 + sigma_d^2 + sigma_e^2); the seed layer is not in it."""
        return math.sqrt(self.sigma_m**2 + self.sigma_d**2 + self.sigma_e**2)

    @classmethod
    def equal_three_way_split(cls, *, sigma_config: float, sigma_seed: float) -> VarianceComponents:
        """The neutral split of 10.7.4: sigma_config^2 shared equally by three layers.

        10.7.7 records that this has no prior evidence behind it and sits on the
        optimistic side; `all_morphology` and `all_residual` are the two corners
        that bracket it.
        """
        each = sigma_config / math.sqrt(3.0)
        return cls(sigma_m=each, sigma_d=each, sigma_e=each, sigma_seed=sigma_seed)

    @classmethod
    def all_morphology(cls, *, sigma_config: float, sigma_seed: float) -> VarianceComponents:
        """Pessimistic corner: the morphology main effect is divided by M alone."""
        return cls(sigma_m=sigma_config, sigma_d=0.0, sigma_e=0.0, sigma_seed=sigma_seed)

    @classmethod
    def all_residual(cls, *, sigma_config: float, sigma_seed: float) -> VarianceComponents:
        """Optimistic corner: everything lands in the cell residual, divided by MD."""
        return cls(sigma_m=0.0, sigma_d=0.0, sigma_e=sigma_config, sigma_seed=sigma_seed)


def var_paired_mean(
    components: VarianceComponents,
    design: Design,
    *,
    include_seed: bool,
    defect_layer: DefectLayer = DefectLayer.RANDOM,
) -> float:
    """Var(D_bar) = sigma_m^2/M + sigma_d^2/D + sigma_e^2/(MD) + sigma_seed^2/(MDS).

    Proposal 10.7.3 and 11.3. The morphology main effect is divided by M only --
    it is shared across the D cells of a family -- so the effective sample size
    is driven by min(M, D), not by MD.
    """
    m, d, s = design.m_families, design.d_defects, design.s_seeds
    if defect_layer is DefectLayer.RANDOM and d == 1:
        raise ValueError(
            f"{design.label}: D = 1 cannot carry a random defect layer -- sigma_d has no "
            "replication to be estimated from. Declare DefectLayer.FIXED, which drops "
            "sigma_d^2/D and forbids any cross-defect generalisation claim for this arm "
            "(this is what reproduces the 9.4 pp printed for E4b in 10.7.4)."
        )
    v = components.sigma_m**2 / m + components.sigma_e**2 / (m * d)
    if defect_layer is DefectLayer.RANDOM:
        v += components.sigma_d**2 / d
    if include_seed:
        v += components.sigma_seed**2 / (m * d * s)
    return v


class MdeCaliber(Enum):
    """The three calibers 10.7.4 distinguishes for the same quantity.

    Quoting an MDE without saying which caliber it is is the failure 10.7.4
    legislates against ("wherever an MDE is quoted, use this table's caliber;
    when quoting the hard lower bound, say that it excludes the seed term").
    """

    TABLE_T_CORRECTED = "table (with seed, t-corrected)"
    WITH_SEED_NORMAL = "with seed, normal coefficient"
    HARD_LOWER_BOUND_NO_SEED = "hard lower bound (no seed term, 14.5 E-13)"

    @property
    def coefficient(self) -> float:
        if self is MdeCaliber.TABLE_T_CORRECTED:
            return MDE_COEFFICIENT_T
        return MDE_COEFFICIENT_NORMAL

    @property
    def includes_seed(self) -> bool:
        return self is not MdeCaliber.HARD_LOWER_BOUND_NO_SEED


@dataclass(frozen=True)
class Mde:
    """A minimum detectable effect that knows its own caliber, design and unit."""

    value: float
    unit: str
    caliber: MdeCaliber
    design: Design
    components: VarianceComponents
    standard_error: float
    kind: str  # "paired_difference" | "slope" | "slope_m1_degenerate"
    defect_layer: DefectLayer = DefectLayer.RANDOM

    def __float__(self) -> float:
        return float(self.value)

    def describe(self) -> str:
        return (
            f"MDE {self.value:.4g} {self.unit} [{self.kind}, {self.caliber.value}, "
            f"coefficient {self.caliber.coefficient}, {self.design}]"
        )


def paired_difference_mde(
    components: VarianceComponents,
    design: Design,
    *,
    unit: str,
    caliber: MdeCaliber = MdeCaliber.TABLE_T_CORRECTED,
    defect_layer: DefectLayer = DefectLayer.RANDOM,
) -> Mde:
    """MDE for a paired-difference endpoint (10.7.3, 10.7.4)."""
    v = var_paired_mean(
        components, design, include_seed=caliber.includes_seed, defect_layer=defect_layer
    )
    se = math.sqrt(v)
    return Mde(
        value=caliber.coefficient * se,
        unit=unit,
        caliber=caliber,
        design=design,
        components=components,
        standard_error=se,
        kind="paired_difference",
        defect_layer=defect_layer,
    )


def slope_mde(
    components: VarianceComponents,
    design: Design,
    *,
    sum_squares_x: float,
    unit: str,
    caliber: MdeCaliber = MdeCaliber.TABLE_T_CORRECTED,
) -> Mde:
    """Var(beta_hat) = (sigma_e^2 + sigma_seed^2/S) / (M sum_i (x_i - x_bar)^2).

    Proposal 10.7.5. The morphology random intercept does not appear: the slope
    is estimated inside a family and then averaged across families, so sigma_m
    differences out. `design.d_defects` is unused here and is carried only so the
    row can print the design it belongs to.
    """
    if sum_squares_x <= 0:
        raise ValueError(f"sum_squares_x must be positive, got {sum_squares_x}")
    seed_term = (
        components.sigma_seed**2 / design.s_seeds if caliber.includes_seed else 0.0
    )
    v = (components.sigma_e**2 + seed_term) / (design.m_families * sum_squares_x)
    se = math.sqrt(v)
    return Mde(
        value=caliber.coefficient * se,
        unit=unit,
        caliber=caliber,
        design=design,
        components=components,
        standard_error=se,
        kind="slope",
    )


def slope_mde_degenerate_m1(
    components: VarianceComponents,
    design: Design,
    *,
    sum_squares_x: float,
    unit: str,
    caliber: MdeCaliber = MdeCaliber.TABLE_T_CORRECTED,
) -> Mde:
    """The M = 1 degenerate form, sigma_config^2 / (S sum_i (x_i - x_bar)^2).

    Provided only so that the two can be printed side by side. 11.3 warns that
    none of the three crown slope designs is M = 1 and that using this form
    misstates the MDE by about sqrt(M); `slope_degenerate_inflation` computes the
    factor for a given design instead of asserting it.
    """
    if sum_squares_x <= 0:
        raise ValueError(f"sum_squares_x must be positive, got {sum_squares_x}")
    v = components.sigma_config**2 / (design.s_seeds * sum_squares_x)
    se = math.sqrt(v)
    return Mde(
        value=caliber.coefficient * se,
        unit=unit,
        caliber=caliber,
        design=design,
        components=components,
        standard_error=se,
        kind="slope_m1_degenerate",
    )


LAMBDA_PART_COMPONENTS = VarianceComponents.equal_three_way_split(
    sigma_config=0.060, sigma_seed=0.030
)
"""The Lambda^part variance layers of the 10.7.4 table, in endpoint units.

sigma_config = 0.060 and sigma_seed = 0.030, split three ways as 10.7.4's
neutral assumption requires. The four slope rows of 10.7.5 are reproduced from
these two numbers alone, which is what makes `SLOPE_MDE_TABLE` checkable rather
than transcribed.
"""


@dataclass(frozen=True)
class SlopeMdeRow:
    """One row of 10.7.5's slope MDE table, with everything it was computed from.

    10.7.8 item 4 registers this table, and a registered MDE is only meaningful
    beside the design and the regressor geometry that produced it: the same
    endpoint at a different (M, S) or a different level spread has a different
    MDE, and a bare number cannot be checked against anything. `recompute`
    rebuilds the value from the variance layers so that a change to sigma, to
    the design or to the level spread cannot leave a stale MDE standing in the
    configuration -- `check` is what turns that into a refusal.
    """

    endpoint: str
    experiment: str
    regressor: str
    levels: str
    spread: float
    sum_squares_x: float
    design: Design
    registered_mde: float
    components: VarianceComponents = LAMBDA_PART_COMPONENTS
    note: str = ""

    def recompute(self) -> Mde:
        return slope_mde(
            self.components,
            self.design,
            sum_squares_x=self.sum_squares_x,
            unit="1",
        )

    def check(self, *, tolerance: float = 5e-4) -> None:
        """Raise unless the registered MDE is the one the inputs give.

        The tolerance is half of the last printed digit: 10.7.5 prints three
        decimals, so a row that reproduces to better than 0.0005 is the printed
        number and anything else is a different number wearing its name.
        """
        recomputed = self.recompute().value
        if abs(recomputed - self.registered_mde) > tolerance:
            raise ValueError(
                f"{self.experiment}.{self.endpoint}: 10.7.5 registers MDE "
                f"{self.registered_mde:.4g}, but {self.design} with sum of squares "
                f"{self.sum_squares_x} and {self.components} gives "
                f"{recomputed:.4g}. One of the two moved; the registered table is "
                "frozen under 10.7.8 item 4, so the inputs are what have to be "
                "explained"
            )


SLOPE_MDE_TABLE: tuple[SlopeMdeRow, ...] = (
    SlopeMdeRow(
        endpoint="lambda_emp_res",
        experiment="E7",
        regressor="log2 K",
        levels="K in {2, 4, 6, 8}",
        spread=2.00,
        sum_squares_x=2.26,
        design=Design(m_families=6, d_defects=2, s_seeds=2, label="E7"),
        registered_mde=0.033,
        note=(
            "Predicted flat, so the CI must sit inside +-0.033. The registered "
            "alternative is the K^(-1/2) decay at beta about -0.075, 2.3x the MDE, "
            "and it is anchored on lambda_emp_res(K=2) = 0.30 -- if the anchor moves "
            "the alternative is recomputed (10.4.7)"
        ),
    ),
    SlopeMdeRow(
        endpoint="lambda_emp_con",
        experiment="E7",
        regressor="log2 K",
        levels="K in {2, 4, 6, 8}",
        spread=2.00,
        sum_squares_x=2.26,
        design=Design(m_families=6, d_defects=2, s_seeds=2, label="E7"),
        registered_mde=0.033,
        note="Predicted to fall, beta_hat at most -0.08, which is 2.4x the MDE",
    ),
    SlopeMdeRow(
        endpoint="connectivity_slope",
        experiment="E8",
        regressor="ln R_eff",
        levels="rungs A and C1-C4",
        spread=1.79,
        sum_squares_x=1.88,
        design=Design(m_families=4, d_defects=1, s_seeds=3, label="E8"),
        registered_mde=0.042,
        note=(
            "The registered effect 0.045 is 1.07x the MDE, so the margin is 7%. "
            "10.7.5 forbids recovering it with a one-sided test or a widened alpha: "
            "a measured slope inside 0.7-1.0x the MDE is inconclusive and says so"
        ),
    ),
    SlopeMdeRow(
        endpoint="isolation_slope",
        experiment="E9",
        regressor="I",
        levels="I0-I3 at the measured 0 / 0.28 / 0.42 / 1.00",
        spread=1.00,
        sum_squares_x=0.532,
        design=Design(m_families=4, d_defects=1, s_seeds=3, label="E9"),
        registered_mde=0.078,
        note=(
            "I is measured, not a design constant, so this row moves with the "
            "ladder: 10.4.9's own formula caps I2 at 0.42 against the nominal 0.71, "
            "which takes the sum of squares from 0.593 to 0.532 and the MDE from "
            "0.074 to 0.078. The registered effect 0.12 is 1.5x the measured-ladder "
            "MDE and 1.6x the nominal one, so the row survives the correction -- but "
            "0.074 is the MDE of a ladder this experiment does not run"
        ),
    ),
)
"""10.7.5's slope MDE table, frozen (10.7.8 item 4).

E3's dose slope is deliberately absent: 10.7.5 gives it no single MDE because
the rank test, not the slope, is its criterion (Spearman rho <= -0.7) and its
sum of squares is computed per defect.
"""


def slope_degenerate_inflation(
    components: VarianceComponents, design: Design, *, sum_squares_x: float
) -> float:
    """Ratio of the degenerate MDE to the correct one for this design.

    Both MDEs share the coefficient and the x spread, so the ratio is
    sqrt( (sigma_config^2 / S) * M / (sigma_e^2 + sigma_seed^2/S) ), which is
    approximately sqrt(M) for the sigma_seed = sigma_config/2 ratio the Lambda
    rows of 10.7.4 use.
    """
    correct = slope_mde(components, design, sum_squares_x=sum_squares_x, unit="1")
    degenerate = slope_mde_degenerate_m1(
        components, design, sum_squares_x=sum_squares_x, unit="1"
    )
    return degenerate.value / correct.value


# --- which inference this arm is allowed to use (10.7.2) -------------------


class InferenceKind(Enum):
    """The 10.7.2 degradation ladder. Every summary row prints which one it used."""

    CROSSED_RANDOM_EFFECTS_REML = "crossed random effects REML (M=6)"
    FAMILY_FIXED_CLUSTER_ROBUST = "family fixed effect + within-family cluster-robust SE"
    FAMILY_FIXED_EXACT_PERMUTATION = "family fixed effect + within-family exact permutation"
    BAYES_WEAK_PRIOR = "Bayesian, half-t(3,0,sigma_Y) prior on sigma, with convergence diagnostics"
    WITHIN_FAMILY_PAIRED_ONLY = "within-family paired differences only, no cross-family claim"


def preregistered_inference(design: Design) -> tuple[InferenceKind, str]:
    """The inference 10.7.2 assigns to this M, plus the note the row must carry.

    Returns the kind and a human-readable flag. The flag is not optional: 10.7.2
    ends with "every row of the summary table must state which inference that
    row used", and `report.ResultRow` refuses to render without it.
    """
    m = design.m_families
    if m >= 5:
        return (
            InferenceKind.CROSSED_RANDOM_EFFECTS_REML,
            "crossed random effects REML; sigma_m, sigma_d, sigma_seed reported",
        )
    if m == 4:
        return (
            InferenceKind.FAMILY_FIXED_CLUSTER_ROBUST,
            "M=4: morphology family is a FIXED effect with within-family cluster-robust "
            "SE; sigma_m is NOT reported (10.7.2)",
        )
    if m == 3:
        return (
            InferenceKind.FAMILY_FIXED_CLUSTER_ROBUST,
            "M=3: morphology family is a FIXED effect with within-family cluster-robust "
            "SE; REML would almost certainly return a zero variance component, so this "
            "is not optional (10.7.2). sigma_m is NOT reported",
        )
    return (
        InferenceKind.WITHIN_FAMILY_PAIRED_ONLY,
        "M<=2: morphology family is a FIXED effect; only within-family paired "
        "differences and cross-family consistency are reported, no cross-family "
        "generalisation statement (10.7.2)",
    )


# --- the crossed random effects model (10.7.2) -----------------------------


@dataclass(frozen=True)
class ModelDiagnostics:
    """What `report.py` needs in order to apply the 11.3 inconclusive triggers."""

    converged: bool
    zero_variance_components: tuple[str, ...]
    message: str = ""

    @property
    def degenerate(self) -> bool:
        return (not self.converged) or bool(self.zero_variance_components)


@dataclass(frozen=True)
class CellObservations:
    """One row per (family, defect, method, seed); the statistical unit is the cell.

    10.7.1: rollouts are not the sample size. `y` is already a per-(cell, method,
    seed) summary -- a success rate over the shared initial states, a Lambda
    reading, a peak force -- and nothing below the cell enters here.
    """

    family: np.ndarray
    defect: np.ndarray
    method: np.ndarray
    seed: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.y)
        for name in ("family", "defect", "method", "seed"):
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(f"{name} has {len(arr)} rows, y has {n}")
        if len(np.unique(self.method)) != 2:
            raise ValueError(
                "the 10.7.2 model tests one method contrast; pass exactly two method "
                f"labels, got {sorted(np.unique(self.method).tolist())}"
            )

    @property
    def design(self) -> Design:
        return Design(
            m_families=int(len(np.unique(self.family))),
            d_defects=int(len(np.unique(self.defect))),
            s_seeds=int(len(np.unique(self.seed))),
            label="observed",
        )

    def paired_differences(self, *, reference: object, comparator: object) -> tuple[
        np.ndarray, np.ndarray, np.ndarray
    ]:
        """Cell-level (comparator - reference) means over seeds.

        Returns (family, defect, difference), aligned. Seeds are averaged inside
        the cell before differencing so the returned array is one value per cell,
        which is the statistical unit of 10.7.1.
        """
        keys = sorted(
            {(str(f), str(d)) for f, d in zip(self.family, self.defect, strict=True)}
        )
        fam_out, def_out, diff_out = [], [], []
        for fam, dfc in keys:
            in_cell = (self.family.astype(str) == fam) & (self.defect.astype(str) == dfc)
            ref = self.y[in_cell & (self.method == reference)]
            cmp_ = self.y[in_cell & (self.method == comparator)]
            if ref.size == 0 or cmp_.size == 0:
                raise ValueError(
                    f"cell ({fam}, {dfc}) is missing one arm; the paired contrast of "
                    "10.1.1 is only defined on complete cells"
                )
            fam_out.append(fam)
            def_out.append(dfc)
            diff_out.append(float(cmp_.mean() - ref.mean()))
        return np.array(fam_out), np.array(def_out), np.array(diff_out)


@dataclass(frozen=True)
class EffectEstimate:
    """A method contrast with everything a summary row needs."""

    estimate: float
    standard_error: float
    ci: tuple[float, float]
    p_value: float
    df: float
    inference: InferenceKind
    inference_flag: str
    design: Design
    n_cells: int
    sigma_m: float | None
    sigma_d: float | None
    sigma_e: float | None
    sigma_seed: float | None
    diagnostics: ModelDiagnostics


def fit_crossed_random_effects(
    obs: CellObservations,
    *,
    reference: object,
    comparator: object,
    alpha: float = ALPHA_PRIMARY,
) -> EffectEstimate:
    """The 10.7.2 primary test: Y_mds = b0 + b1 method + a_m + b_d + e_md + u_mds.

    Family and defect are both random intercepts (crossed, not nested), the
    (family, defect) cell carries e_md, and the residual is the seed layer
    u_mds. Fitted by REML through statsmodels' variance-component machinery with
    a single trivial group, which is the standard way to write a crossed model
    there.

    Raises `DesignTooSmallError` at M <= 3. 10.7.2 is explicit that REML on those
    arms "almost certainly returns a zero variance component, so this is not an
    option"; returning a zero sigma_m from here would look like an estimate.
    Call `fit_family_fixed_cluster_robust` instead and let the row carry the flag.
    """
    import pandas as pd
    import statsmodels.formula.api as smf

    design = obs.design
    if design.m_families <= 3:
        kind, flag = preregistered_inference(design)
        raise DesignTooSmallError(
            f"M = {design.m_families}: 10.7.2 forbids crossed REML here. Use "
            f"fit_family_fixed_cluster_robust and flag the row: {flag} "
            f"(inference kind {kind.name})"
        )

    frame = pd.DataFrame(
        {
            "y": np.asarray(obs.y, dtype=float),
            "family": obs.family.astype(str),
            "defect": obs.defect.astype(str),
            "seed": obs.seed.astype(str),
            "method": np.where(obs.method == comparator, 1.0, 0.0),
        }
    )
    if not np.isin(obs.method, [reference, comparator]).all():
        raise ValueError(
            f"method labels must be exactly {{{reference!r}, {comparator!r}}}, got "
            f"{sorted(set(obs.method.tolist()))}"
        )
    frame["cell"] = frame["family"] + "|" + frame["defect"]
    frame["_g"] = 1

    vc = {
        "family": "0 + C(family)",
        "defect": "0 + C(defect)",
        "cell": "0 + C(cell)",
    }
    model = smf.mixedlm("y ~ method", frame, groups=frame["_g"], vc_formula=vc, re_formula="0")
    with np.errstate(all="ignore"):
        fit = model.fit(reml=True, method=["lbfgs"])

    converged = bool(getattr(fit, "converged", False))
    vcomp = dict(zip(model.exog_vc.names, np.atleast_1d(fit.vcomp), strict=True))
    scale = float(fit.scale)
    zeros = tuple(
        name for name, v in vcomp.items() if float(v) <= _ZERO_VARIANCE_TOL
    ) + (("residual",) if scale <= _ZERO_VARIANCE_TOL else ())
    diagnostics = ModelDiagnostics(
        converged=converged,
        zero_variance_components=zeros,
        message="" if converged else "REML did not converge (11.3 inconclusive trigger)",
    )

    beta = float(fit.fe_params["method"])
    se = float(fit.bse["method"])
    # Configuration-layer df, not observation-layer: the unit is the cell (10.7.1).
    df = max(design.n_cells - 2, 1)
    tcrit = float(sps.t.ppf(1.0 - alpha / 2.0, df))
    p = float(2.0 * sps.t.sf(abs(beta / se), df)) if se > 0 else float("nan")

    return EffectEstimate(
        estimate=beta,
        standard_error=se,
        ci=(beta - tcrit * se, beta + tcrit * se),
        p_value=p,
        df=float(df),
        inference=InferenceKind.CROSSED_RANDOM_EFFECTS_REML,
        inference_flag=preregistered_inference(design)[1],
        design=design,
        n_cells=design.n_cells,
        sigma_m=math.sqrt(max(float(vcomp.get("family", 0.0)), 0.0)),
        sigma_d=math.sqrt(max(float(vcomp.get("defect", 0.0)), 0.0)),
        sigma_e=math.sqrt(max(float(vcomp.get("cell", 0.0)), 0.0)),
        sigma_seed=math.sqrt(max(scale, 0.0)),
        diagnostics=diagnostics,
    )


def fit_family_fixed_cluster_robust(
    obs: CellObservations,
    *,
    reference: object,
    comparator: object,
    alpha: float = ALPHA_PRIMARY,
) -> EffectEstimate:
    """The M <= 3 (and M = 4) caliber: family fixed, SE clustered within family.

    10.7.2. sigma_m is deliberately `None` in the result -- a fixed morphology
    effect does not estimate a morphology variance, and printing one would be
    the exact confusion the degradation table exists to prevent.

    With three or four clusters the cluster-robust variance estimator is
    strongly downward biased; the t reference distribution therefore uses
    G - 1 degrees of freedom, and the returned flag says so.
    """
    import pandas as pd
    import statsmodels.formula.api as smf

    design = obs.design
    frame = pd.DataFrame(
        {
            "y": np.asarray(obs.y, dtype=float),
            "family": obs.family.astype(str),
            "defect": obs.defect.astype(str),
            "method": np.where(obs.method == comparator, 1.0, 0.0),
        }
    )
    formula = "y ~ method + C(family)"
    if design.d_defects > 1:
        formula += " + C(defect)"
    fit = smf.ols(formula, frame).fit(
        cov_type="cluster", cov_kwds={"groups": frame["family"]}, use_t=True
    )

    beta = float(fit.params["method"])
    se = float(fit.bse["method"])
    n_clusters = design.m_families
    df = max(n_clusters - 1, 1)
    tcrit = float(sps.t.ppf(1.0 - alpha / 2.0, df))
    p = float(2.0 * sps.t.sf(abs(beta / se), df)) if se > 0 else float("nan")

    kind, flag = preregistered_inference(design)
    flag = f"{flag}; cluster-robust SE on G = {n_clusters} clusters, t reference with {df} df"
    return EffectEstimate(
        estimate=beta,
        standard_error=se,
        ci=(beta - tcrit * se, beta + tcrit * se),
        p_value=p,
        df=float(df),
        inference=kind,
        inference_flag=flag,
        design=design,
        n_cells=design.n_cells,
        sigma_m=None,
        sigma_d=None,
        sigma_e=None,
        sigma_seed=None,
        diagnostics=ModelDiagnostics(converged=True, zero_variance_components=()),
    )


# --- within-family restricted permutation (10.7.2) -------------------------


class PermutationUnit(Enum):
    """Which label swaps the restricted randomisation admits.

    10.7.2 says "permute the method labels inside a morphology family, keeping
    the within-family correlation structure", which admits two readings, and the
    proposal does not choose between them. Both are implemented; the caller
    names one and the report prints it.

    CELL swaps the two method labels independently in each (family, defect)
    cell. It holds a_m and b_d exactly fixed and is the finest randomisation
    consistent with the paired design.

    FAMILY swaps all D cells of a family together. It additionally preserves the
    within-family correlation of the differences, which is the conservative
    reading, at the cost of only 2^M distinct assignments.
    """

    CELL = "cell"
    FAMILY = "family"


@dataclass(frozen=True)
class PermutationResult:
    """A permutation test that hands back its whole null distribution.

    11.3 requires the full null distribution to be published for every
    Lambda-family conclusion, not just the p value, so it is a field rather than
    something the caller has to ask for.
    """

    observed: float
    null_distribution: np.ndarray
    p_two_sided: float
    exact: bool
    unit: PermutationUnit
    n_assignments: int

    def quantile_of_observed(self) -> float:
        return float(np.mean(self.null_distribution <= self.observed))


def _family_averaging_matrix(family: np.ndarray) -> np.ndarray:
    """(n_cells,) weights whose dot product with the cells is the family-balanced mean.

    Families are weighted equally rather than cells, which matters when an arm
    runs an unbalanced subset (10.7.10); on a balanced 6 x 6 panel the two
    coincide. Returned as a weight vector rather than computed per permutation so
    that the whole null distribution is one matrix product -- 2^18 assignments is
    a routine exact enumeration here and a Python loop over it is not.
    """
    fams, inverse = np.unique(family, return_inverse=True)
    counts = np.bincount(inverse, minlength=fams.size).astype(float)
    return 1.0 / (counts[inverse] * fams.size)


def _family_balanced_mean(family: np.ndarray, values: np.ndarray) -> float:
    return float(_family_averaging_matrix(family) @ values)


def within_family_restricted_permutation(
    *,
    family: np.ndarray,
    paired_difference: np.ndarray,
    unit: PermutationUnit,
    rng: np.random.Generator,
    n_permutations: int | None = None,
    max_exact: int = 1 << 18,
) -> PermutationResult:
    """Restricted permutation of the method label inside each morphology family.

    Under H0 the two method labels are exchangeable within their randomisation
    unit, so permuting them flips the sign of the corresponding paired
    differences. Enumerated exactly when the number of assignments is at most
    `max_exact` (10.7.2 names the exact within-family test for M = 4), Monte
    Carlo otherwise, in which case `n_permutations` is required -- there is no
    default draw count, because the resolution of the p value depends on it.
    """
    family = np.asarray(family)
    diffs = np.asarray(paired_difference, dtype=float)
    if family.shape != diffs.shape:
        raise ValueError(f"family {family.shape} and differences {diffs.shape} disagree")
    if diffs.size == 0:
        raise ValueError("no cells to permute")

    if unit is PermutationUnit.CELL:
        block_of_cell = np.arange(diffs.size)
    else:
        _, block_of_cell = np.unique(family, return_inverse=True)
    n_blocks = int(block_of_cell.max()) + 1
    n_assignments = 1 << n_blocks

    weights = _family_averaging_matrix(family)
    observed = float(weights @ diffs)
    # weights * diffs, so a sign matrix over cells contracts straight to the
    # statistic: null = signs_over_cells @ weighted.
    weighted = weights * diffs

    if n_assignments <= max_exact:
        signs = np.array(
            list(itertools.product((1.0, -1.0), repeat=n_blocks)), dtype=float
        )
        exact = True
    else:
        if n_permutations is None:
            raise ValueError(
                f"{n_blocks} randomisation units give 2^{n_blocks} assignments, above "
                f"max_exact={max_exact}; pass n_permutations explicitly so the p-value "
                "resolution is on the record"
            )
        signs = rng.choice((1.0, -1.0), size=(n_permutations, n_blocks))
        exact = False
    null = signs[:, block_of_cell] @ weighted

    # +1 in numerator and denominator: the observed assignment is a member of the
    # randomisation set, and omitting it gives an anti-conservative p at small n.
    p = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (null.size + 1))
    return PermutationResult(
        observed=observed,
        null_distribution=null,
        p_two_sided=min(p, 1.0),
        exact=exact,
        unit=unit,
        n_assignments=n_assignments if exact else int(null.size),
    )


# --- paired block bootstrap (10.1.3, 11.3) ---------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    replicates: np.ndarray
    level: float
    n_initial_states: int

    @property
    def ci(self) -> tuple[float, float]:
        lo = float(np.quantile(self.replicates, (1.0 - self.level) / 2.0))
        hi = float(np.quantile(self.replicates, 1.0 - (1.0 - self.level) / 2.0))
        return lo, hi

    @property
    def standard_error(self) -> float:
        return float(np.std(self.replicates, ddof=1))


def paired_block_bootstrap_multi(
    per_state: Mapping[str, np.ndarray],
    statistics: Mapping[str, Callable[[Mapping[str, np.ndarray]], float]],
    *,
    n_resamples: int,
    rng: np.random.Generator,
    level: float = 0.95,
) -> dict[str, BootstrapResult]:
    """Trajectory-level block bootstrap with initial-state indices shared by all arms.

    Proposal 10.1.3 and 11.3: the evaluation initial-state set is common to every
    arm, which makes the shared-initial-state layer perfectly correlated across
    arms. One index vector is drawn per resample and applied to every arm *and*
    to every statistic, so the layer cancels in a paired contrast exactly as it
    does in the point estimate, and two statistics computed here stay correlated
    the way they are in the data. That second property is what Leak_beh's Fieller
    interval needs: its numerator and denominator share J(Oracle).

    There is deliberately no switch for independent per-arm resampling. It would
    inflate the variance of every cross-arm comparison, and the API should not
    make it reachable.

    Each row of `per_state[arm]` is one whole trajectory's summary, so resampling
    rows *is* the block bootstrap with block length equal to the full horizon T
    (10.1.3 fixes the block at the whole trajectory).
    """
    if not per_state:
        raise ValueError("no arms given")
    if not statistics:
        raise ValueError("no statistics given")
    lengths = {name: len(arr) for name, arr in per_state.items()}
    n_states = next(iter(lengths.values()))
    if any(v != n_states for v in lengths.values()):
        raise ValueError(
            f"all arms must be evaluated on the same initial-state set; got {lengths}. "
            "Paired resampling is only defined on a shared index space (11.3)."
        )
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")

    replicates = {name: np.empty(n_resamples, dtype=float) for name in statistics}
    for b in range(n_resamples):
        idx = rng.integers(0, n_states, size=n_states)
        resampled = {name: arr[idx] for name, arr in per_state.items()}
        for name, fn in statistics.items():
            replicates[name][b] = fn(resampled)
    return {
        name: BootstrapResult(
            point=float(fn(per_state)),
            replicates=replicates[name],
            level=level,
            n_initial_states=n_states,
        )
        for name, fn in statistics.items()
    }


def paired_block_bootstrap(
    per_state: Mapping[str, np.ndarray],
    statistic: Callable[[Mapping[str, np.ndarray]], float],
    *,
    n_resamples: int,
    rng: np.random.Generator,
    level: float = 0.95,
) -> BootstrapResult:
    """Single-statistic form of `paired_block_bootstrap_multi`."""
    return paired_block_bootstrap_multi(
        per_state,
        {"statistic": statistic},
        n_resamples=n_resamples,
        rng=rng,
        level=level,
    )["statistic"]


# --- multiplicity (10.7.6) --------------------------------------------------


def intersection_union_p(components: Mapping[str, float]) -> float:
    """The p value of a conjunction of directional claims: the largest component.

    T9 is "lambda_emp_res is flat in log2 K *and* lambda_emp_con falls in K",
    and 10.7.6 counts the pair as one member of the primary family rather than
    two. The intersection-union test is what turns two component tests into the
    one number Holm then ranks: rejecting the conjunction requires every
    component to be rejected on its own, so the size of the combined test is at
    most alpha with no correction between the components -- the conjunction is
    harder to declare than either part, not easier.

    Taking the maximum is therefore the whole test, and the reason it needs no
    weighting is that the components are not being combined into an average.
    Anything that pooled them (Fisher's method, a mean of the two slopes) would
    let one strongly significant component carry a null one, which is the
    opposite of what a conjunction claims.
    """
    if len(components) < 2:
        raise ValueError(
            f"an intersection-union test needs at least two components; got "
            f"{sorted(components)}. A conjunction of one claim is that claim, and "
            "recording it as a pair would misreport how it was tested"
        )
    values = np.array(list(components.values()), dtype=float)
    if np.any((values < 0) | (values > 1) | ~np.isfinite(values)):
        raise ValueError(f"component p values must lie in [0, 1]: {dict(components)}")
    return float(values.max())


@dataclass(frozen=True)
class HolmResult:
    labels: tuple[str, ...]
    p_raw: tuple[float, ...]
    p_adjusted: tuple[float, ...]
    rejected: tuple[bool, ...]
    holm_rank: tuple[int, ...]
    alpha: float


def holm_bonferroni(
    p_values: Mapping[str, float],
    *,
    alpha: float = ALPHA_PRIMARY,
    expected_family: Sequence[str] | None = PRIMARY_HYPOTHESIS_FAMILY,
) -> HolmResult:
    """Holm-Bonferroni at alpha over the primary family (10.7.6).

    `expected_family` is checked rather than assumed: 11.3 freezes the five
    members, and quietly running Holm over four of them (or over six) changes
    the family-wise level. Pass `expected_family=None` for any other family.
    """
    if expected_family is not None and set(p_values) != set(expected_family):
        missing = sorted(set(expected_family) - set(p_values))
        extra = sorted(set(p_values) - set(expected_family))
        raise ValueError(
            "the primary family is frozen at the five members of 10.7.6; "
            f"missing={missing} unexpected={extra}. Running Holm over a different "
            "set silently changes the family-wise error rate."
        )
    labels = tuple(p_values)
    raw = np.array([p_values[k] for k in labels], dtype=float)
    if np.any((raw < 0) | (raw > 1) | ~np.isfinite(raw)):
        raise ValueError(f"p values must lie in [0, 1]: {dict(zip(labels, raw, strict=True))}")

    order = np.argsort(raw, kind="stable")
    m = raw.size
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * raw[idx])
        adjusted[idx] = min(running, 1.0)
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)
    return HolmResult(
        labels=labels,
        p_raw=tuple(raw.tolist()),
        p_adjusted=tuple(adjusted.tolist()),
        rejected=tuple((adjusted < alpha).tolist()),
        holm_rank=tuple(ranks.tolist()),
        alpha=alpha,
    )


def benjamini_hochberg(
    p_values: Mapping[str, float], *, q: float = ALPHA_SECONDARY_FDR
) -> HolmResult:
    """BH at FDR q over the secondary endpoints (10.7.6).

    Shares `HolmResult` because a summary row needs the same four columns; the
    `alpha` field carries q.
    """
    labels = tuple(p_values)
    raw = np.array([p_values[k] for k in labels], dtype=float)
    order = np.argsort(raw, kind="stable")
    m = raw.size
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        running = min(running, m * raw[idx] / (rank + 1))
        adjusted[idx] = min(running, 1.0)
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)
    return HolmResult(
        labels=labels,
        p_raw=tuple(raw.tolist()),
        p_adjusted=tuple(adjusted.tolist()),
        rejected=tuple((adjusted <= q).tolist()),
        holm_rank=tuple(ranks.tolist()),
        alpha=q,
    )


# --- absolute success rates (10.1.1, 11.3) ---------------------------------


@dataclass(frozen=True)
class AbsoluteRateInterval:
    """Two intervals for one absolute rate, because they answer different questions.

    11.3: "when reporting absolute success rates, attach the binomial plus
    finite-population CI (50 fixed initial states, about 7 pp of common offset at
    p = 0.5, which enters no paired CI)".

    `rollout_lo/hi` is a Wilson interval over the rollouts and is the right
    interval for "what is this arm's rate *on these 50 states*". `population_lo/hi`
    adds the state-sampling term sqrt(p(1-p)/50) and is the right interval for
    "what is this arm's rate over initial states in general". The second term is
    a common offset shared by every arm, so `enters_paired_ci` is False and
    `endpoints.py` never adds it to a paired contrast.
    """

    point: float
    n_success: int
    n_rollouts: int
    n_initial_states: int
    level: float
    rollout_lo: float
    rollout_hi: float
    state_sampling_se: float
    population_lo: float
    population_hi: float
    enters_paired_ci: bool = False

    def describe(self) -> str:
        return (
            f"{self.point:.4f} "
            f"[{self.rollout_lo:.4f}, {self.rollout_hi:.4f}] over {self.n_initial_states} "
            f"fixed states; [{self.population_lo:.4f}, {self.population_hi:.4f}] with the "
            f"state-sampling term ({self.state_sampling_se * 100:.1f} pp, common to all "
            "arms, excluded from every paired CI)"
        )


def absolute_success_rate_ci(
    *,
    n_success: int,
    n_rollouts: int,
    n_initial_states: int,
    level: float = 0.95,
) -> AbsoluteRateInterval:
    """Binomial (Wilson) plus finite-population interval for an absolute rate.

    The state-sampling variance is bounded above by p(1-p)/N, attained when each
    initial state is deterministically a success or a failure. At p = 0.5 and
    N = 50 that is 7.07 pp, the number 10.1.1 quotes; the bound is used rather
    than an estimate of the between-state variance because the proposal quotes
    the bound and because a per-state variance estimate needs the per-state
    breakdown, which is not always retained.
    """
    if not 0 <= n_success <= n_rollouts:
        raise ValueError(f"n_success={n_success} outside [0, {n_rollouts}]")
    if n_rollouts < 1 or n_initial_states < 1:
        raise ValueError("n_rollouts and n_initial_states must be positive")

    p = n_success / n_rollouts
    z = float(sps.norm.ppf(1.0 - (1.0 - level) / 2.0))

    denom = 1.0 + z**2 / n_rollouts
    centre = (p + z**2 / (2 * n_rollouts)) / denom
    half = z * math.sqrt(p * (1 - p) / n_rollouts + z**2 / (4 * n_rollouts**2)) / denom
    rollout_lo, rollout_hi = max(0.0, centre - half), min(1.0, centre + half)

    state_se = math.sqrt(p * (1 - p) / n_initial_states)
    total_se = math.sqrt(p * (1 - p) / n_rollouts + p * (1 - p) / n_initial_states)
    return AbsoluteRateInterval(
        point=p,
        n_success=n_success,
        n_rollouts=n_rollouts,
        n_initial_states=n_initial_states,
        level=level,
        rollout_lo=rollout_lo,
        rollout_hi=rollout_hi,
        state_sampling_se=state_se,
        population_lo=max(0.0, p - z * total_se),
        population_hi=min(1.0, p + z * total_se),
    )
