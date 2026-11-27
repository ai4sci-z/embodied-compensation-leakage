"""The three endpoints, the four safety quantities, and Leak_beh (proposal 10.1).

Proposal 10.1 puts three endpoints in a table and then spends four paragraphs on
the discipline that keeps them apart. What that discipline turns into here:

* **the primary decision endpoint is the paired success-rate difference**
  J(Oracle) - J(pooled), MDE 7.3 pp (10.1.1, 10.7.4). It is paired on the shared
  initial-state index set, and the roughly 7 pp state-sampling offset of an
  absolute rate is deliberately unreachable from a paired CI -- 10.1.1 says the
  offset is common to every arm and enters no paired interval, so
  `PairedEndpointResult` has no field it could be added to.
* **the primary leakage endpoint is Lambda^part**, the nuisance-basis partial
  regression coefficient, with the naive Lambda printed in the same row
  (10.1 rule 4, 11.3). Their difference is the overall nuisance adjustment.
  A separate nested contrast, changing only the shrinkage column, identifies
  the shrinkage-specific adjustment.
* **the Lambda^orc three-term decomposition** of 10.1.5, whose third term has
  unknown sign; the class refuses to hand back a lemma-1 statistic without it,
  because 11.3 requires the sign of Lambda - E_rho[w] to be read only after the
  third term has been taken off.
* **the four safety quantities**: peak cfrc_ext (collision transient, not
  quasi-static force -- 10.1 rule 3), torque saturation time share, terminal
  overshoot, constraint violation rate.
* **Leak_beh**, with a Fieller interval and the denominator rule of 10.1.2 that
  sends a configuration to `inconclusive` when the denominator is below the
  preregistered floor.

Amplitude and energy are kept apart by `diagnostics.units`: the Lambda family is
`Amplitude`, Phi is an `Energy`, and Phi is only ever obtained through
`Amplitude.squared()` (A-14 forbids treating Phi and Lambda as two independent
pieces of evidence).

Upstream dependency, noted rather than imported: the per-cell Lambda readings
that `LeakageReading` carries are produced by
`portal_trace.diagnostics.leakage` (naive and partial-regression fits),
`portal_trace.diagnostics.common_mode` (the E7 empirical pair) and
`portal_trace.diagnostics.design_graph` (R_eff for the E8 regressor), which are
written concurrently. This module takes the readings as typed values rather than
recomputing them, so the estimator lives in exactly one place.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import numpy as np
from scipy import stats as sps

from portal_trace.cards import DataCard, ModelCard, require_all_admissible
from portal_trace.diagnostics.units import Amplitude, Energy
from portal_trace.eval.stats import (
    AbsoluteRateInterval,
    BootstrapResult,
    DefectLayer,
    Design,
    Mde,
    MdeCaliber,
    VarianceComponents,
    absolute_success_rate_ci,
    paired_block_bootstrap,
    paired_block_bootstrap_multi,
    paired_difference_mde,
)

__all__ = [
    "EndpointRole",
    "StateDistribution",
    "ReferencePoint",
    "EndpointSpec",
    "ENDPOINT_SPECS",
    "PRINTED_MDE_10_7_4",
    "EndpointProvenance",
    "PairedEndpointResult",
    "paired_success_rate_difference",
    "SafetyQuantity",
    "SAFETY_QUANTITIES",
    "safety_paired_difference",
    "LeakageReading",
    "phi_from_lambda",
    "lambda_exc",
    "SURROGATE_FAMILY_SIZE",
    "SURROGATE_MIN_CELL_CARDINALITY",
    "DEFAULT_LAMBDA_MIN_DENOMINATOR_ENERGY",
    "LambdaOrcDecomposition",
    "DistributionSensitivity",
    "FiellerInterval",
    "fieller_ratio_interval",
    "DenominatorRule",
    "LeakBehResult",
    "leak_beh",
    "LEAK_BEH_PREREGISTERED_EFFECT_SIZE",
    "LEAK_BEH_DENOMINATOR_FLOOR_PP",
    "LEAK_BEH_DENOMINATOR_CI_LOWER_FLOOR_PP",
    "FIELLER_MDE_WIDENING",
]


# --- preregistered effect sizes and floors ---------------------------------

SUCCESS_RATE_PREREGISTERED_EFFECT_PP = 10.0
"""J(Oracle) - J(pooled) >= 10 pp on at least 4/6 defect classes (10.1.1)."""

PEAK_CFRC_PREREGISTERED_EFFECT_N = 4.0
"""Pooled arm's peak cfrc_ext exceeds the Oracle arm's by >= 4 N (10.1.1)."""

LAMBDA_EXC_PREREGISTERED_EFFECT = 0.10
"""Lambda^exc >= 0.10 on at least 4/6 defect classes (10.4.1)."""

LEAK_BEH_PREREGISTERED_EFFECT_SIZE = 0.35
"""Leakage eats about 35% of the pooling gain (10.1.2 rule 3).

Present so that a "not significant" verdict on a descriptive quantity has a
boundary to be read against; 11.3 makes carrying it a condition of reporting
Leak_beh at all.
"""

LEAK_BEH_DENOMINATOR_FLOOR_PP = 10.0
"""Denominator floor, = 1.4 x the success-rate MDE (10.1.2 rule 2)."""

LEAK_BEH_DENOMINATOR_CI_LOWER_FLOOR_PP = 5.0
"""...and the 95% bootstrap CI lower bound of the denominator (10.1.2 rule 2)."""

FIELLER_MDE_WIDENING = 1.30
"""Widening applied to Leak_beh's MDE for the Fieller interval (10.7.4).

Back-computed from the table, which prints 0.36 -> 0.47 at (6,6,3) and
0.45 -> 0.58 at (3,6,3): those imply 1.299 and 1.288. The proposal states the
widened numbers but not the factor, so it is fixed here at 1.30 and the (3,6,3)
row is reproduced to 0.59 against a printed 0.58 -- recorded rather than tuned
to fit both rows.
"""

SURROGATE_FAMILY_SIZE = 8
"""|S| >= 8 preregistered surrogate defects (10.4.1, 11.3)."""

SURROGATE_MIN_CELL_CARDINALITY = 6
"""Below this per-cell cardinality, Lambda^exc for that cell is exploratory only."""

DEFAULT_LAMBDA_MIN_DENOMINATOR_ENERGY = 1e-12
"""Fail-closed floor for energy denominators in Lambda-family summaries."""

MIN_RAW_FRAMES_FOR_REPORTING = 2_000
"""Regime admission: m_min = 0.02 AND n_min = 2000 raw frames (10.7.8 item 7)."""


class EndpointRole(Enum):
    """10.7.6's three multiplicity tiers, which are also reporting tiers."""

    PRIMARY = "primary (Holm-Bonferroni at alpha = 0.05)"
    SECONDARY = "secondary (Benjamini-Hochberg at FDR 0.10)"
    EXPLORATORY = "exploratory (no multiplicity control; excluded from abstract and conclusions)"


class StateDistribution(Enum):
    """Which measure a reading was taken on (6.2, 10.1.5)."""

    RHO_OFF = "rho^off, the clean reference policy's occupancy on the evaluation embodiment"
    RHO_ON = "rho^on, the evaluated policy's own occupancy"
    POOLED_MARGINAL = "the pooled state marginal"


class ReferencePoint(Enum):
    """Lambda's two reference points (6.2). Both are always reported."""

    TEACHER_CLEAN = "mu_tau, the teacher's clean policy (Lambda^ref)"
    ORACLE_B7 = "the B7 Oracle policy (Lambda^orc)"


# --- the 10.7.4 MDE table, computed rather than transcribed ----------------


@dataclass(frozen=True)
class EndpointSpec:
    """One row of 10.7.4: the sigmas, the design it belongs to, and its role."""

    name: str
    unit: str
    sigma_config: float
    sigma_seed: float
    design: Design
    role: EndpointRole
    defect_layer: DefectLayer = DefectLayer.RANDOM
    preregistered_effect_size: float | None = None
    practical_relevance_threshold: float | None = None
    fieller_widened: bool = False

    @property
    def components(self) -> VarianceComponents:
        return VarianceComponents.equal_three_way_split(
            sigma_config=self.sigma_config, sigma_seed=self.sigma_seed
        )

    def mde(self, *, caliber: MdeCaliber = MdeCaliber.TABLE_T_CORRECTED) -> Mde:
        base = paired_difference_mde(
            self.components,
            self.design,
            unit=self.unit,
            caliber=caliber,
            defect_layer=self.defect_layer,
        )
        if not self.fieller_widened:
            return base
        return Mde(
            value=base.value * FIELLER_MDE_WIDENING,
            unit=base.unit,
            caliber=base.caliber,
            design=base.design,
            components=base.components,
            standard_error=base.standard_error,
            kind="paired_difference_fieller_widened",
            defect_layer=base.defect_layer,
        )


_D663 = Design(6, 6, 3, "36 cells, 3 seeds (B2 vs B7)")
_D663_S5 = Design(6, 6, 5, "36 cells, 5 seeds")
_D363 = Design(3, 6, 3, "S_18 arm")
_D413 = Design(4, 1, 3, "E4b, T3")
_D332 = Design(3, 3, 2, "E4")

ENDPOINT_SPECS: dict[str, EndpointSpec] = {
    spec.name: spec
    for spec in (
        EndpointSpec(
            name="success_rate_paired_difference",
            unit="pp",
            sigma_config=7.0,
            sigma_seed=5.0,
            design=_D663,
            role=EndpointRole.PRIMARY,
            preregistered_effect_size=SUCCESS_RATE_PREREGISTERED_EFFECT_PP,
            practical_relevance_threshold=SUCCESS_RATE_PREREGISTERED_EFFECT_PP,
        ),
        EndpointSpec(
            name="success_rate_paired_difference@S18",
            unit="pp",
            sigma_config=7.0,
            sigma_seed=5.0,
            design=_D363,
            role=EndpointRole.SECONDARY,
        ),
        EndpointSpec(
            # D = 1: 10.7.3 writes sigma_d^2/D unconditionally, but the 9.4 pp this
            # row prints is only reachable with the term dropped. See DefectLayer.
            name="success_rate_paired_difference@E4b_T3",
            unit="pp",
            sigma_config=7.0,
            sigma_seed=5.0,
            design=_D413,
            role=EndpointRole.EXPLORATORY,
            defect_layer=DefectLayer.FIXED,
            preregistered_effect_size=9.5,
        ),
        EndpointSpec(
            name="lambda_part",
            unit="dimensionless",
            sigma_config=0.060,
            sigma_seed=0.030,
            design=_D663,
            role=EndpointRole.PRIMARY,
        ),
        EndpointSpec(
            name="lambda_part@S18",
            unit="dimensionless",
            sigma_config=0.060,
            sigma_seed=0.030,
            design=_D363,
            role=EndpointRole.SECONDARY,
        ),
        EndpointSpec(
            name="lambda_exc",
            unit="dimensionless",
            sigma_config=0.050,
            sigma_seed=0.025,
            design=_D663,
            role=EndpointRole.PRIMARY,
            preregistered_effect_size=LAMBDA_EXC_PREREGISTERED_EFFECT,
            practical_relevance_threshold=LAMBDA_EXC_PREREGISTERED_EFFECT,
        ),
        EndpointSpec(
            name="peak_cfrc_ext",
            unit="N",
            sigma_config=2.4,
            sigma_seed=1.2,
            design=_D663,
            role=EndpointRole.SECONDARY,
            preregistered_effect_size=PEAK_CFRC_PREREGISTERED_EFFECT_N,
            practical_relevance_threshold=PEAK_CFRC_PREREGISTERED_EFFECT_N,
        ),
        EndpointSpec(
            name="unexpected_collisions_per_rollout",
            unit="count",
            sigma_config=0.18,
            sigma_seed=0.09,
            design=_D663,
            role=EndpointRole.SECONDARY,
        ),
        EndpointSpec(
            name="torque_saturation_time_share",
            unit="pp",
            sigma_config=1.6,
            sigma_seed=0.8,
            design=_D663,
            role=EndpointRole.SECONDARY,
        ),
        EndpointSpec(
            name="terminal_overshoot",
            unit="mm",
            sigma_config=0.9,
            sigma_seed=0.45,
            design=_D663,
            role=EndpointRole.SECONDARY,
        ),
        EndpointSpec(
            name="cos_gphi_ctrue",
            unit="dimensionless",
            sigma_config=0.070,
            sigma_seed=0.035,
            design=_D332,
            role=EndpointRole.SECONDARY,
        ),
        EndpointSpec(
            name="leak_beh",
            unit="ratio",
            sigma_config=0.35,
            sigma_seed=0.18,
            design=_D663,
            role=EndpointRole.EXPLORATORY,
            preregistered_effect_size=LEAK_BEH_PREREGISTERED_EFFECT_SIZE,
            fieller_widened=True,
        ),
        EndpointSpec(
            name="leak_beh@S18",
            unit="ratio",
            sigma_config=0.35,
            sigma_seed=0.18,
            design=_D363,
            role=EndpointRole.EXPLORATORY,
            preregistered_effect_size=LEAK_BEH_PREREGISTERED_EFFECT_SIZE,
            fieller_widened=True,
        ),
    )
}

PRINTED_MDE_10_7_4: dict[str, tuple[float, int]] = {
    # endpoint -> (value printed in 10.7.4, decimals it is printed to)
    "success_rate_paired_difference": (7.3, 1),
    "success_rate_paired_difference@S18": (9.1, 1),
    "success_rate_paired_difference@E4b_T3": (9.4, 1),
    "lambda_part": (0.062, 3),
    "lambda_part@S18": (0.077, 3),
    "lambda_exc": (0.052, 3),
    "peak_cfrc_ext": (2.5, 1),
    "unexpected_collisions_per_rollout": (0.19, 2),
    "torque_saturation_time_share": (1.7, 1),
    "terminal_overshoot": (0.93, 2),
    "cos_gphi_ctrue": (0.11, 2),
    "leak_beh": (0.47, 2),
    "leak_beh@S18": (0.58, 2),
}
"""The values 10.7.4 prints, for the import-time cross-check below.

Keeping the printed values next to the computed ones is what turns "the table
was transcribed correctly" into something a reviewer can see fail.
"""


def _self_check() -> None:
    """Every computed MDE must reproduce the printed 10.7.4 value.

    Runs at import, like `defects.params._self_check`. The one row that does not
    round to its printed value is `leak_beh@S18`, whose Fieller widening factor
    is not stated in the proposal (see `FIELLER_MDE_WIDENING`); it is allowed one
    unit in the last printed place and nothing more.
    """
    tolerant = {"leak_beh@S18"}
    for name, (printed, decimals) in PRINTED_MDE_10_7_4.items():
        computed = ENDPOINT_SPECS[name].mde().value
        slack = 10.0**-decimals * (1.0 if name in tolerant else 0.5)
        if abs(round(computed, decimals) - printed) > slack + 1e-12:
            raise AssertionError(
                f"{name}: 10.7.4 prints {printed} but the variance model of 10.7.3 "
                f"gives {computed:.6g}. The MDE table is a frozen preregistration "
                "artefact (10.7.8 item 4); reconcile it, do not adjust the tolerance."
            )


_self_check()


# --- the 11.4 admissibility gate on the way into an endpoint ---------------


@dataclass(frozen=True)
class EndpointProvenance:
    """The cards an endpoint computation stands on (11.4).

    Constructing this validates every card, so an endpoint that takes a
    `EndpointProvenance` cannot be computed from an object that is missing an
    11.4 minimum field. That is 11.4's rule -- "missing any one of them and the
    object may not be used in the computation of any preregistered endpoint" --
    placed on the path in rather than in a checklist beside it.
    """

    endpoint: str
    data_cards: tuple[DataCard, ...]
    model_cards: tuple[ModelCard, ...]

    def __post_init__(self) -> None:
        if not self.model_cards:
            raise ValueError(
                f"{self.endpoint}: at least one model card is required; 11.4 makes the "
                "arm's (M, D, S) a property of its card"
            )
        require_all_admissible(
            (*self.data_cards, *self.model_cards), endpoint=self.endpoint
        )

    @property
    def design(self) -> Design:
        """The common design of the arms; a cross-arm endpoint needs one design."""
        designs = {(d.m_families, d.d_defects, d.s_seeds): d for d in
                   (mc.design for mc in self.model_cards)}
        if len(designs) != 1:
            raise ValueError(
                f"{self.endpoint}: the arms declare different designs "
                f"{sorted(str(d) for d in designs.values())}; a paired cross-arm "
                "endpoint has one (M, D, S), and 10.7.10 forbids a subset arm "
                "borrowing the full design's MDE"
            )
        return next(iter(designs.values()))


# --- the primary decision endpoint (10.1.1) --------------------------------


@dataclass(frozen=True)
class PairedEndpointResult:
    """A paired cross-arm contrast, with the pieces a summary row needs.

    There is no field for the state-sampling offset. 10.1.1: the absolute-rate
    state-sampling error is a *common* offset across arms and "enters no paired
    CI", so it is reachable only through `absolute_reference` /
    `absolute_comparator`, which are labelled as absolute quantities.
    """

    spec: EndpointSpec
    reference_arm: str
    comparator_arm: str
    point: float
    ci: tuple[float, float]
    bootstrap: BootstrapResult
    n_initial_states: int
    provenance: EndpointProvenance
    absolute_reference: AbsoluteRateInterval | None = None
    absolute_comparator: AbsoluteRateInterval | None = None

    @property
    def mde(self) -> Mde:
        return self.spec.mde()

    @property
    def unit(self) -> str:
        return self.spec.unit

    def describe(self) -> str:
        return (
            f"{self.spec.name}: {self.comparator_arm} - {self.reference_arm} = "
            f"{self.point:.4g} {self.unit}, 95% paired CI "
            f"[{self.ci[0]:.4g}, {self.ci[1]:.4g}], {self.mde.describe()}"
        )


def _as_state_vector(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one value per initial state, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    return arr


def paired_success_rate_difference(
    *,
    oracle_success: np.ndarray,
    pooled_success: np.ndarray,
    provenance: EndpointProvenance,
    rng: np.random.Generator,
    n_resamples: int,
    spec_name: str = "success_rate_paired_difference",
    level: float = 0.95,
    oracle_arm: str = "B7 Oracle",
    pooled_arm: str = "B2 naive pooling",
) -> PairedEndpointResult:
    """J(Oracle) - J(pooled) in percentage points, MDE 7.3 pp (10.1.1, 10.7.4).

    `oracle_success` and `pooled_success` are per-initial-state success
    indicators on the *same* index set: 10.4.0 shares the evaluation initial
    states across arms, and 11.3 requires the bootstrap to resample those indices
    once and give every arm the same draw. Passing arrays of different lengths is
    an error rather than something to be aligned here, because a mismatch means
    the arms did not see the same states and the contrast is not paired.

    The absolute rates come back with their own binomial + finite-population
    intervals, which is 11.3's reporting rule; the roughly 7 pp state-sampling
    term in those intervals is not added to the paired CI.
    """
    oracle = _as_state_vector(oracle_success, "oracle_success")
    pooled = _as_state_vector(pooled_success, "pooled_success")
    if oracle.shape != pooled.shape:
        raise ValueError(
            f"the two arms were evaluated on {oracle.size} and {pooled.size} initial "
            "states; the paired difference of 10.1.1 is defined only on the shared set"
        )
    for name, arr in (("oracle_success", oracle), ("pooled_success", pooled)):
        if not np.all((arr == 0) | (arr == 1)):
            raise ValueError(f"{name} must be 0/1 success indicators per initial state")

    spec = ENDPOINT_SPECS[spec_name]

    def statistic(arms: Mapping[str, np.ndarray]) -> float:
        return 100.0 * float(arms["oracle"].mean() - arms["pooled"].mean())

    boot = paired_block_bootstrap(
        {"oracle": oracle, "pooled": pooled},
        statistic,
        n_resamples=n_resamples,
        rng=rng,
        level=level,
    )
    n_states = oracle.size
    return PairedEndpointResult(
        spec=spec,
        reference_arm=pooled_arm,
        comparator_arm=oracle_arm,
        point=boot.point,
        ci=boot.ci,
        bootstrap=boot,
        n_initial_states=n_states,
        provenance=provenance,
        absolute_reference=absolute_success_rate_ci(
            n_success=int(pooled.sum()),
            n_rollouts=n_states,
            n_initial_states=n_states,
            level=level,
        ),
        absolute_comparator=absolute_success_rate_ci(
            n_success=int(oracle.sum()),
            n_rollouts=n_states,
            n_initial_states=n_states,
            level=level,
        ),
    )


# --- the four safety quantities (10.1, 10.1.1) -----------------------------


class SafetyQuantity(Enum):
    """The physical safety endpoints of 10.1.

    Peak cfrc_ext is a *collision transient*, dominated by approach velocity, not
    a quasi-static force: 10.1 rule 3 records that controller and environment
    stiffness are in series under robosuite's OSC_POSE default k_p = 150, so
    0.5 mm of penetration gives 0.075 N, far below the noise floor. A
    quasi-static force endpoint would be measuring nothing.
    """

    PEAK_CFRC_EXT = "peak_cfrc_ext"
    TORQUE_SATURATION_TIME_SHARE = "torque_saturation_time_share"
    TERMINAL_OVERSHOOT = "terminal_overshoot"
    CONSTRAINT_VIOLATION_RATE = "constraint_violation_rate"

    @property
    def spec(self) -> EndpointSpec:
        if self is SafetyQuantity.CONSTRAINT_VIOLATION_RATE:
            # 10.7.4 prints no sigma row for the violation rate; it shares the
            # unexpected-collision-count row, which is the same absorbing-state
            # event counted per rollout (8: violations are not retryable, task
            # failures are, which is why the rate is reported at all).
            return ENDPOINT_SPECS["unexpected_collisions_per_rollout"]
        return ENDPOINT_SPECS[self.value]


SAFETY_QUANTITIES: tuple[SafetyQuantity, ...] = tuple(SafetyQuantity)


def safety_paired_difference(
    quantity: SafetyQuantity,
    *,
    reference: np.ndarray,
    comparator: np.ndarray,
    provenance: EndpointProvenance,
    rng: np.random.Generator,
    n_resamples: int,
    level: float = 0.95,
    reference_arm: str = "B7 Oracle",
    comparator_arm: str = "B2 naive pooling",
) -> PairedEndpointResult:
    """Paired difference of a safety quantity on the shared initial-state set.

    `reference` and `comparator` are one value per initial state: the peak of
    ||cfrc_ext|| over the rollout, the fraction of control steps at the torque
    limit, the terminal overshoot in mm, or the 0/1 constraint-violation
    indicator. Reducing a rollout to one number before this point is what makes
    the block bootstrap's block the whole trajectory (10.1.3).

    The sign convention is comparator - reference, so the preregistered
    "the pooled arm's peak cfrc_ext exceeds the Oracle arm's by at least 4 N"
    (10.1.1) is a positive effect.
    """
    ref = _as_state_vector(reference, "reference")
    cmp_ = _as_state_vector(comparator, "comparator")
    if ref.shape != cmp_.shape:
        raise ValueError(
            f"{quantity.value}: arms evaluated on {ref.size} and {cmp_.size} initial "
            "states; a paired safety contrast needs the shared set (10.4.0)"
        )
    if quantity is SafetyQuantity.CONSTRAINT_VIOLATION_RATE and not np.all(
        (ref == 0) | (ref == 1) | (cmp_ == 0) | (cmp_ == 1)
    ):
        raise ValueError("the constraint violation rate takes 0/1 indicators per rollout")

    def statistic(arms: Mapping[str, np.ndarray]) -> float:
        return float(arms["comparator"].mean() - arms["reference"].mean())

    boot = paired_block_bootstrap(
        {"reference": ref, "comparator": cmp_},
        statistic,
        n_resamples=n_resamples,
        rng=rng,
        level=level,
    )
    return PairedEndpointResult(
        spec=quantity.spec,
        reference_arm=reference_arm,
        comparator_arm=comparator_arm,
        point=boot.point,
        ci=boot.ci,
        bootstrap=boot,
        n_initial_states=ref.size,
        provenance=provenance,
    )


# --- the leakage endpoints (10.1, 10.4.1, 11.3) ----------------------------


def phi_from_lambda(
    lambda_reading: Amplitude,
    *,
    e_c_squared: float,
    e_eps_squared: float,
    min_denominator_energy: float = DEFAULT_LAMBDA_MIN_DENOMINATOR_ENERGY,
) -> Energy:
    """Phi = Lambda^2 E||c||^2 / E||eps||^2 (6.1), through the one legal crossing.

    A-14 forbids reading Phi and Lambda as two independent pieces of evidence, so
    Phi is never computed from a formula of its own: it goes through
    `Amplitude.squared()`, which is where the dimension change is visible in a
    diff.
    """
    if not math.isfinite(min_denominator_energy) or min_denominator_energy <= 0.0:
        raise ValueError("min_denominator_energy must be finite and positive")
    if not math.isfinite(e_eps_squared) or e_eps_squared <= min_denominator_energy:
        raise ValueError(
            "E||eps||^2 is at or below the preregistered denominator floor "
            f"({e_eps_squared:.6g} <= {min_denominator_energy:.6g}); Phi is undefined"
        )
    if not math.isfinite(e_c_squared) or e_c_squared < 0:
        raise ValueError(f"E||c||^2 must be non-negative, got {e_c_squared}")
    aligned = lambda_reading.squared() * e_c_squared
    return Energy(aligned.value / e_eps_squared)


@dataclass(frozen=True)
class LeakageReading:
    """One cell's leakage readings: Lambda^part primary, naive Lambda alongside.

    10.1 rule 4 and 11.3: the primary endpoint is always the nuisance-basis
    partial regression coefficient, with the naive Lambda in the same row and no
    conditional branch that switches caliber when the two differ. Naive minus
    full is the overall nuisance adjustment. The shrinkage-specific effect is a
    nested contrast with all other nuisance columns held fixed.
    """

    cell: str
    lambda_part: Amplitude
    lambda_naive: Amplitude
    lambda_without_shrinkage: Amplitude
    lambda_ref: Amplitude
    lambda_orc: Amplitude
    phi_leakage_share: Energy
    state_distribution: StateDistribution
    nuisance_basis: tuple[str, ...]
    clip_saturation_share: float
    n_raw_frames: int
    target_residual_energy_share: float
    target_vif: float
    target_residual_share_floor: float

    #  10.4.1 makes these two mandatory members of the nuisance basis.
    REQUIRED_NUISANCE_TERMS: ClassVar[tuple[str, ...]] = (
        "shrinkage_field",
        "temporal_smoothing_field",
    )

    def __post_init__(self) -> None:
        missing = [t for t in self.REQUIRED_NUISANCE_TERMS if t not in self.nuisance_basis]
        if missing:
            raise ValueError(
                f"{self.cell}: the nuisance basis of 10.4.1 must contain {missing}. The "
                "shrinkage field is mandatory, not a conditional upgrade: the two arms' "
                "shrinkage targets differ exactly along c_bar, so B7 does not difference "
                "the term out and the surrogate null does not cover it"
            )
        if self.n_raw_frames < MIN_RAW_FRAMES_FOR_REPORTING:
            raise ValueError(
                f"{self.cell}: {self.n_raw_frames} raw frames is below the reporting "
                f"admission threshold n_min = {MIN_RAW_FRAMES_FOR_REPORTING} (10.7.8 "
                "item 7)"
            )
        if not math.isfinite(self.target_residual_share_floor) or not (
            0.0 < self.target_residual_share_floor <= 1.0
        ):
            raise ValueError(
                f"{self.cell}: target_residual_share_floor must lie in (0, 1]"
            )
        if (
            not math.isfinite(self.target_residual_energy_share)
            or self.target_residual_energy_share < self.target_residual_share_floor
            or self.target_residual_energy_share > 1.0 + 1e-12
        ):
            raise ValueError(
                f"{self.cell}: c_delta residual energy share "
                f"{self.target_residual_energy_share:.6g} is below preregistered floor "
                f"{self.target_residual_share_floor:.6g}; Lambda^part is unidentified"
            )
        expected_vif = 1.0 / self.target_residual_energy_share
        if not math.isfinite(self.target_vif) or not math.isclose(
            self.target_vif, expected_vif, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValueError(
                f"{self.cell}: target_vif must equal 1 / residual energy share "
                f"({expected_vif:.6g}), got {self.target_vif:.6g}"
            )

    @property
    def overall_nuisance_adjustment(self) -> Amplitude:
        """Naive minus full; an aggregate adjustment, not a mechanism share."""
        return self.lambda_naive - self.lambda_part

    @property
    def nested_shrinkage_effect(self) -> Amplitude:
        """No-shrinkage fit minus full fit, holding all other nuisance fixed."""
        return self.lambda_without_shrinkage - self.lambda_part

    @property
    def needs_clipping_annotation(self) -> bool:
        return self.clip_saturation_share > 0.05


def lambda_exc(
    *,
    lambda_part: Amplitude,
    surrogate_lambda_parts: Sequence[Amplitude],
    cell: str,
) -> tuple[Amplitude, bool]:
    """Lambda^exc = Lambda^part - E_{delta~S}[Lambda^part] (10.4.1, 11.3).

    Returns the excess leakage and whether the cell may be more than exploratory:
    11.3 allows the per-cell surrogate cardinality to fall below 8 but not below
    6, and a cell under that floor reports Lambda^exc as exploratory only.

    Note that the surrogates enter through their *partial* regression readings,
    matching 10.4.1's definition; using naive Lambda on one side and partial on
    the other would put the shrinkage channel into the excess.
    """
    if not surrogate_lambda_parts:
        raise ValueError(f"{cell}: Lambda^exc needs a surrogate null distribution")
    values = np.array([float(s.value) for s in surrogate_lambda_parts], dtype=float)
    admissible = values.size >= SURROGATE_MIN_CELL_CARDINALITY
    return Amplitude(float(lambda_part.value) - float(values.mean())), admissible


@dataclass(frozen=True)
class DistributionSensitivity:
    """Posterior components on another distribution; never part of the sum."""

    distribution: StateDistribution
    alpha_bar_rho: float
    covariance_term: float


@dataclass(frozen=True)
class LambdaOrcDecomposition:
    """Lambda^orc = alpha_bar_rho + Cov_rho(w,||c||^2)/E||c||^2 - E<u_orc,c>/E||c||^2.

    Proposal 10.1.5 and 11.3. Three properties of this decomposition are load
    bearing and are enforced here rather than described:

    All three terms and Lambda^orc use `distribution`. A pooled-state result can
    be attached only as `pooled_sensitivity`; it is not substituted into this
    identity. The third term has unknown sign and is always reported explicitly.
    """

    cell: str
    distribution: StateDistribution
    alpha_bar_rho: float
    covariance_term: float
    reference_alignment_term: float
    lambda_orc: Amplitude
    pooled_sensitivity: DistributionSensitivity | None = None

    def __post_init__(self) -> None:
        if (
            self.pooled_sensitivity is not None
            and self.pooled_sensitivity.distribution is not StateDistribution.POOLED_MARGINAL
        ):
            raise ValueError(
                f"{self.cell}: pooled_sensitivity must be labelled POOLED_MARGINAL, got "
                f"{self.pooled_sensitivity.distribution.name}"
            )

    @property
    def total(self) -> Amplitude:
        """The reconstructed Lambda^orc, for the identity check against the fit."""
        return Amplitude(
            self.alpha_bar_rho + self.covariance_term - self.reference_alignment_term
        )

    @property
    def covariance_from_reading(self) -> Amplitude:
        """Covariance implied by Lambda and the other same-distribution terms."""
        return Amplitude(
            float(self.lambda_orc.value) - self.alpha_bar_rho + self.reference_alignment_term
        )

    @property
    def identity_residual(self) -> float:
        """Observed Lambda minus the same-distribution reconstruction."""
        return float(self.lambda_orc.value) - float(self.total.value)

    def terms(self) -> dict[str, float]:
        """The three terms, for the term-by-term reporting 10.1.5 requires."""
        return {
            "alpha_bar_rho": self.alpha_bar_rho,
            "covariance_term": self.covariance_term,
            "reference_alignment_term_sign_unknown": self.reference_alignment_term,
        }


# --- Leak_beh: Fieller interval and the denominator rule (10.1.2) ----------


@dataclass(frozen=True)
class FiellerInterval:
    """An exact confidence set for the ratio of two correlated normal estimates.

    Fieller's theorem solves (n - theta d)^2 = t^2 Var(n - theta d) for theta.
    When g = t^2 Var(d) / d^2 >= 1 the quadratic's leading coefficient changes
    sign and the confidence set is not an interval: it is either the whole line
    or the complement of a bounded interval. That is not a numerical nuisance to
    be clipped away -- it is the statement that the denominator is not
    distinguishable from zero, which is exactly the situation 10.1.2 says
    Leak_beh lives in, and it is reported as such.
    """

    point: float
    lower: float
    upper: float
    bounded: bool
    kind: str  # "bounded" | "complement_of_interval" | "entire_line"
    g: float
    level: float

    def describe(self) -> str:
        if self.kind == "bounded":
            return f"{self.point:.4g} [Fieller {self.lower:.4g}, {self.upper:.4g}]"
        if self.kind == "entire_line":
            return (
                f"{self.point:.4g} [Fieller set is the entire real line; the denominator "
                f"is not separated from zero, g = {self.g:.3g} >= 1]"
            )
        return (
            f"{self.point:.4g} [Fieller set is the complement of "
            f"({self.lower:.4g}, {self.upper:.4g}); unbounded, g = {self.g:.3g} >= 1]"
        )


def fieller_ratio_interval(
    *,
    numerator: float,
    denominator: float,
    var_numerator: float,
    var_denominator: float,
    cov: float,
    df: float,
    level: float = 0.95,
) -> FiellerInterval:
    """Fieller confidence set for numerator / denominator (10.1.2 rule 1)."""
    if var_numerator < 0 or var_denominator < 0:
        raise ValueError("variances must be non-negative")
    if cov * cov > var_numerator * var_denominator * (1.0 + 1e-9):
        raise ValueError(
            f"Cov = {cov:.6g} exceeds sqrt(Var_n Var_d) = "
            f"{math.sqrt(var_numerator * var_denominator):.6g}, so the 2x2 covariance "
            "matrix is not positive semi-definite. Fieller's discriminant carries "
            "Var_n - Cov^2 / Var_d, which such an input drives negative and turns into "
            "a spuriously wide interval rather than an error"
        )
    if df <= 0:
        raise ValueError(f"the t reference distribution needs df > 0, got {df}")
    if denominator == 0:
        raise ZeroDivisionError(
            "the Leak_beh denominator J(Oracle) - J(single-embodiment) is exactly zero; "
            "the denominator rule of 10.1.2 should have removed this configuration "
            "before a ratio was formed"
        )
    t = float(sps.t.ppf(1.0 - (1.0 - level) / 2.0, df))
    point = numerator / denominator
    g = t**2 * var_denominator / denominator**2

    disc = (
        var_numerator
        - 2.0 * point * cov
        + point**2 * var_denominator
        - g * (var_numerator - cov**2 / var_denominator if var_denominator > 0 else 0.0)
    )
    if disc < 0:
        return FiellerInterval(
            point=point,
            lower=-math.inf,
            upper=math.inf,
            bounded=False,
            kind="entire_line",
            g=g,
            level=level,
        )
    root = t / abs(denominator) * math.sqrt(disc)
    centre = point - g * cov / var_denominator if var_denominator > 0 else point
    if g < 1.0:
        return FiellerInterval(
            point=point,
            lower=(centre - root) / (1.0 - g),
            upper=(centre + root) / (1.0 - g),
            bounded=True,
            kind="bounded",
            g=g,
            level=level,
        )
    # g >= 1: the two roots bound the *excluded* region.
    lo = (centre + root) / (1.0 - g)
    hi = (centre - root) / (1.0 - g)
    return FiellerInterval(
        point=point,
        lower=min(lo, hi),
        upper=max(lo, hi),
        bounded=False,
        kind="complement_of_interval",
        g=g,
        level=level,
    )


class DenominatorRule(Enum):
    """Outcome of 10.1.2's denominator threshold rule.

    Either failure sends the configuration out of the summary table and into the
    appendix, where only the raw numerator and denominator are printed;
    `eval.report` turns both into an `inconclusive` verdict, which is the third
    of 11.3's preregistered inconclusive triggers.
    """

    PASSES = "denominator above the floor"
    BELOW_POINT_FLOOR = "point estimate of the denominator below 10 pp"
    CI_LOWER_BELOW_FLOOR = "95% bootstrap CI lower bound of the denominator below 5 pp"


@dataclass(frozen=True)
class LeakBehResult:
    """Leak_beh = (J(Oracle) - J(pooled)) / (J(Oracle) - J(single-embodiment)).

    Descriptive, appended, exploratory (10.1.2, 10.7.6). Its MDE of 0.47 covers
    the whole plausible range of the quantity, which is why it cannot carry a
    decision and why it must arrive with a preregistered effect size of 0.35 --
    without one, "not significant" has no boundary to be read against (11.3).
    """

    cell: str
    numerator_pp: float
    denominator_pp: float
    denominator_ci_pp: tuple[float, float]
    fieller: FiellerInterval | None
    bootstrap_percentile_ci: tuple[float, float]
    denominator_rule: DenominatorRule
    preregistered_effect_size: float
    provenance: EndpointProvenance

    @property
    def admissible_to_summary(self) -> bool:
        return self.denominator_rule is DenominatorRule.PASSES

    @property
    def mde(self) -> Mde:
        return ENDPOINT_SPECS["leak_beh"].mde()

    @property
    def intervals_disagree(self) -> bool:
        """10.1.2 rule 1: when Fieller and the percentile bootstrap disagree, say so.

        Fieller wins, and the disagreement is written down rather than resolved
        silently.
        """
        if self.fieller is None or not self.fieller.bounded:
            return True
        lo_b, hi_b = self.bootstrap_percentile_ci
        return not (
            math.isclose(self.fieller.lower, lo_b, rel_tol=0.10, abs_tol=0.02)
            and math.isclose(self.fieller.upper, hi_b, rel_tol=0.10, abs_tol=0.02)
        )

    def describe(self) -> str:
        interval = (
            "no ratio: the denominator J(Oracle) - J(single-embodiment) is exactly zero"
            if self.fieller is None
            else self.fieller.describe()
        )
        head = (
            f"Leak_beh[{self.cell}] = {self.numerator_pp:.3g} pp / "
            f"{self.denominator_pp:.3g} pp = {interval}"
        )
        if not self.admissible_to_summary:
            head += f"  -- EXCLUDED FROM THE SUMMARY: {self.denominator_rule.value}"
        return head


def leak_beh(
    *,
    cell: str,
    oracle_success: np.ndarray,
    pooled_success: np.ndarray,
    single_embodiment_success: np.ndarray,
    provenance: EndpointProvenance,
    rng: np.random.Generator,
    n_resamples: int,
    level: float = 0.95,
) -> LeakBehResult:
    """Leak_beh with a Fieller interval and 10.1.2's denominator rule.

    All three arms are evaluated on the shared initial-state set and resampled
    with one shared index vector (11.3). The Fieller inputs -- the two variances
    and their covariance -- come from that same paired bootstrap, so the
    correlation between numerator and denominator (they share J(Oracle)) is
    carried rather than assumed away; ignoring it would be the main reason a
    normal-approximation interval for this ratio is wrong.
    """
    oracle = _as_state_vector(oracle_success, "oracle_success")
    pooled = _as_state_vector(pooled_success, "pooled_success")
    single = _as_state_vector(single_embodiment_success, "single_embodiment_success")
    if not (oracle.shape == pooled.shape == single.shape):
        raise ValueError(
            f"{cell}: the three arms cover {oracle.size}, {pooled.size} and "
            f"{single.size} initial states; Leak_beh is a ratio of two paired "
            "differences on one shared set"
        )

    # Both differences must be read off the *same* index draws: they share
    # J(Oracle), and Fieller needs that correlation.
    boots = paired_block_bootstrap_multi(
        {"oracle": oracle, "pooled": pooled, "single": single},
        {
            "numerator": lambda a: 100.0 * float(a["oracle"].mean() - a["pooled"].mean()),
            "denominator": lambda a: 100.0 * float(a["oracle"].mean() - a["single"].mean()),
        },
        n_resamples=n_resamples,
        rng=rng,
        level=level,
    )
    num_rep = boots["numerator"].replicates
    den_rep = boots["denominator"].replicates
    numerator = boots["numerator"].point
    denominator = boots["denominator"].point
    alpha = 1.0 - level
    den_ci = (
        float(np.quantile(den_rep, alpha / 2.0)),
        float(np.quantile(den_rep, 1.0 - alpha / 2.0)),
    )

    if denominator < LEAK_BEH_DENOMINATOR_FLOOR_PP:
        rule = DenominatorRule.BELOW_POINT_FLOOR
    elif den_ci[0] < LEAK_BEH_DENOMINATOR_CI_LOWER_FLOOR_PP:
        rule = DenominatorRule.CI_LOWER_BELOW_FLOOR
    else:
        rule = DenominatorRule.PASSES

    # 3374 says outright that the denominator is a random quantity that can cross
    # zero, so the exactly-zero draw is a case rather than an accident. The ratio
    # does not exist there and no interval is manufactured for it; the rule above
    # has already sent the configuration to the appendix, where 10.1.2 prints the
    # numerator and the denominator on their own.
    fieller: FiellerInterval | None = None
    if denominator != 0.0:
        # The variances entering Fieller come from the trajectory-level block
        # bootstrap, whose resampling unit is the shared initial state; the t
        # reference distribution therefore carries that unit's degrees of
        # freedom. The (M, D, S) design is the unit of the 10.7.4 MDE table and
        # not of this variance.
        fieller = fieller_ratio_interval(
            numerator=numerator,
            denominator=denominator,
            var_numerator=float(np.var(num_rep, ddof=1)),
            var_denominator=float(np.var(den_rep, ddof=1)),
            cov=float(np.cov(num_rep, den_rep, ddof=1)[0, 1]),
            df=float(boots["denominator"].n_initial_states - 1),
            level=level,
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_rep = num_rep / den_rep
    finite = ratio_rep[np.isfinite(ratio_rep)]
    percentile_ci = (
        (float(np.quantile(finite, alpha / 2.0)), float(np.quantile(finite, 1.0 - alpha / 2.0)))
        if finite.size
        else (-math.inf, math.inf)
    )

    return LeakBehResult(
        cell=cell,
        numerator_pp=numerator,
        denominator_pp=denominator,
        denominator_ci_pp=den_ci,
        fieller=fieller,
        bootstrap_percentile_ci=percentile_ci,
        denominator_rule=rule,
        preregistered_effect_size=LEAK_BEH_PREREGISTERED_EFFECT_SIZE,
        provenance=provenance,
    )
