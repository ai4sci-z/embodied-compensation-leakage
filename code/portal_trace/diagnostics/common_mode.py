"""Common mode: the injected-truth side and the two T9 empirical readings.

Proposal 7.5 (proposition 1(c)), 7.10 (proposition 4), 10.4.7 (E7) and 11.3.

Two sides that must not be confused, which is why they live in one module with
separate types:

Injected-truth side -- gamma_{delta,r}, gamma_bar_r, Q_div, Lambda_res. These
are deterministic functions of the injected compensation fields and can be
computed before any training starts. 7.5 scope note (4) is explicit that
Lambda_res is therefore a *geometric* quantity and **not a preregistered test
statistic**; `ResidualGeometry.is_test_statistic` exists to say so in code.

Empirical side -- c_bar, c_tilde_k, Lambda^emp_res, Lambda^emp_con. These are
the two preregistered T9 quantities (11.3), read off the deployment error of
the strongest conditioning arm. They are policy dependent and only exist after
training.

Type discipline (14.5 E-1): Q_div is an energy share, the Lambda family is
amplitude valued, and the two may not be subtracted or divided. `Energy` and
`Amplitude` make that a runtime error rather than a review comment. The only
bridge between them is the Cauchy-Schwarz inequality of 7.5 scope note (3),
which this module computes explicitly.

Weighting (10.4.7): the occupancy-weighted caliber weights numerator and
denominator separately and takes the ratio last. An occupancy-weighted mean of
per-regime slopes is a different number and is not a slope.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from portal_trace.diagnostics.units import Amplitude, Energy
from portal_trace.types import Regime

__all__ = [
    "T9_FLAT_BAND",
    "T9_DECLINE_SLOPE",
    "T9_ALTERNATIVE_SLOPE",
    "RegimeWeights",
    "CompensationEffects",
    "ResidualGeometry",
    "EmpiricalCommonModeReadings",
    "KScanPoint",
    "KScanVerdict",
    "KScanAnalysis",
    "contrast_fields",
    "common_mode_field",
]

# 11.3: T9's two criteria. Specification values, not measurements.
T9_FLAT_BAND = 0.033
T9_DECLINE_SLOPE = -0.08
# 10.4.7: the distinguishable alternative, Lambda^emp_res ~ K^{-1/2}, which over
# the K in {2,4,6,8} design gives this slope against log2 K.
T9_ALTERNATIVE_SLOPE = -0.075


@dataclass(frozen=True)
class RegimeWeights:
    """m_r, the occupancy mass used to weight the per-regime readings (10.4.7).

    Supplied explicitly rather than derived from the frames at hand: E7 weights
    by rho^off(r), which is a property of the evaluation occupancy and not of
    whichever cell's frames happen to be in the accumulator.
    """

    weights: Mapping[Regime, float]

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("no occupancy weights supplied")
        values = np.array(list(self.weights.values()), dtype=np.float64)
        if np.any(values < 0.0):
            raise ValueError("occupancy mass cannot be negative")
        total = float(values.sum())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"occupancy mass over the reported regimes sums to {total}, not 1. "
                "R is a partition (9.7.1); a cover would double count frames"
            )

    def of(self, regime: Regime) -> float:
        return float(self.weights.get(regime, 0.0))

    @property
    def regimes(self) -> tuple[Regime, ...]:
        return tuple(r for r in Regime if r in self.weights)


def common_mode_field(fields: Sequence[np.ndarray]) -> np.ndarray:
    """c_bar = (1/K) sum_k c_{delta_k} (7.10).

    Uniform mean, always. 7.5 scope note (2): under a mixture-weighted mean the
    projection onto the unidentifiable subspace is no longer orthogonal and the
    energy decomposition stops being one.
    """
    if len(fields) < 2:
        raise ValueError(
            f"the common mode is a mean over K >= 2 sources, got K={len(fields)}"
        )
    stacked = np.stack([np.asarray(f, dtype=np.float64) for f in fields], axis=0)
    return stacked.mean(axis=0)


def contrast_fields(fields: Sequence[np.ndarray]) -> list[np.ndarray]:
    """c_tilde_k = c_{delta_k} - c_bar, which sum to zero by construction."""
    c_bar = common_mode_field(fields)
    return [np.asarray(f, dtype=np.float64) - c_bar for f in fields]


@dataclass(frozen=True)
class CompensationEffects:
    """gamma_{delta,r} for one regime, over the K sources, plus Q_div.

    gamma_{delta,r} = (1/n_T) sum_tau E_{rho_{tau delta}|r}[c_delta] (7.1). The
    average over tasks is uniform and must run over *all* tasks: dropping an
    unobserved task changes the normalisation and makes gamma depend on the
    design, which is exactly what the two-step definition of 7.1 exists to
    prevent.
    """

    regime: Regime
    gamma: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if len(self.gamma) < 2:
            raise ValueError(
                f"Q_div compares K >= 2 sources (7.5), got K={len(self.gamma)}"
            )
        widths = {np.shape(v) for v in self.gamma.values()}
        if len(widths) != 1:
            raise ValueError(f"gamma vectors disagree on shape: {widths}")

    @classmethod
    def from_task_means(
        cls,
        regime: Regime,
        per_source_task_means: Mapping[str, Mapping[str, np.ndarray]],
        tasks: Sequence[str],
    ) -> CompensationEffects:
        """Build gamma_{delta,r} from E_{rho_{tau delta}|r}[c_delta] per task."""
        task_list = tuple(tasks)
        if len(set(task_list)) != len(task_list):
            raise ValueError(f"the task list has duplicates: {task_list}")
        gamma: dict[str, np.ndarray] = {}
        for source, means in per_source_task_means.items():
            missing = [t for t in task_list if t not in means]
            if missing:
                raise ValueError(
                    f"source {source!r} has no regime-{regime.value} mean for tasks "
                    f"{missing}. gamma is a uniform average over all n_T tasks (7.1); "
                    "averaging over the observed subset makes it design dependent"
                )

            stacked = np.stack(
                [np.asarray(means[t], dtype=np.float64) for t in task_list], axis=0
            )
            gamma[source] = stacked.mean(axis=0)
        return cls(regime=regime, gamma=gamma)

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(sorted(self.gamma))

    @property
    def n_sources(self) -> int:
        return len(self.gamma)

    @property
    def gamma_bar(self) -> np.ndarray:
        """gamma_bar_r, the uniform mean over sources (7.5)."""
        return np.stack([self.gamma[k] for k in self.sources], axis=0).mean(axis=0)

    @property
    def mean_energy(self) -> Energy:
        """E_k||gamma_{delta_k,r}||^2."""
        values = np.stack([self.gamma[k] for k in self.sources], axis=0)
        return Energy(float(np.mean(np.sum(np.square(values), axis=1))))

    @property
    def common_mode_energy(self) -> Energy:
        """||gamma_bar_r||^2."""
        return Energy(float(np.sum(np.square(self.gamma_bar))))

    @property
    def q_div(self) -> Energy:
        """Q_div = 1 - ||gamma_bar_r||^2 / E_k||gamma_{delta_k,r}||^2 (7.5).

        An energy share in [0, 1]. Comparable with Phi, never with Lambda.
        """
        denominator = float(self.mean_energy.value)
        if denominator == 0.0:
            raise ValueError(
                "E_k||gamma||^2 is zero: no source has any regime-"
                f"{self.regime.value} compensation, so the removable share is undefined"
            )
        value = 1.0 - float(self.common_mode_energy.value) / denominator
        # Orthogonality of the uniform mean puts this in [0,1] exactly; a
        # violation means the gamma set was not built with a uniform mean.
        if not -1e-9 <= value <= 1.0 + 1e-9:
            raise ValueError(
                f"Q_div came out at {value}, outside [0, 1]. The energy decomposition of "
                "7.5 holds only for the uniform mean"
            )
        return Energy(min(max(value, 0.0), 1.0))


@dataclass(frozen=True)
class ResidualGeometry:
    """Lambda_res and the Cauchy-Schwarz bound that links it to Q_div (7.5).

    Lambda_res = <gamma_bar_r, c_delta> / ||c_delta||^2, both readings taken in
    L^2(rho): the numerator is E_rho<gamma_bar_r, c_delta(x)> and the
    denominator E_rho||c_delta(x)||^2.

    **Not a test statistic.** 7.5 scope note (4): gamma_bar_r and c_delta are
    both injected ground truth, so Lambda_res is fixed before training and its
    behaviour in K is an arithmetic consequence of the injection design. It is
    reported to connect the energy share Q_div to an amplitude, and for nothing
    else. Its denominator is also ill-defined for K > 1 (which delta?), which
    is the second reason T9's quantities are the empirical pair instead.
    """

    effects: CompensationEffects
    inner_product: float
    energy_c_delta: Energy
    delta_label: str

    is_test_statistic: bool = False

    @property
    def lambda_res(self) -> Amplitude:
        denominator = float(self.energy_c_delta.value)
        if denominator == 0.0:
            raise ValueError("E||c_delta||^2 is zero; Lambda_res has no denominator")
        return Amplitude(self.inner_product / denominator)

    @property
    def cauchy_schwarz_bound(self) -> float:
        """sqrt(1 - Q_div) * sqrt(E_k||gamma||^2) / ||c_delta|| (7.5 note 3)."""
        q = float(self.effects.q_div.value)
        norm_c = float(self.energy_c_delta.sqrt().value)
        if norm_c == 0.0:
            raise ValueError("||c_delta|| is zero; the bound has no denominator")
        return math.sqrt(max(1.0 - q, 0.0)) * float(
            self.effects.mean_energy.sqrt().value
        ) / norm_c

    @property
    def bound_is_respected(self) -> bool:
        return abs(float(self.lambda_res.value)) <= self.cauchy_schwarz_bound * (1 + 1e-9)

    def as_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "quantity": "q_div",
                "kind": "energy",
                "value": float(self.effects.q_div.value),
                "role": "by construction",
            },
            {
                "quantity": "lambda_res",
                "kind": "amplitude",
                "value": float(self.lambda_res.value),
                "role": "geometric, not a test statistic",
                "delta": self.delta_label,
            },
            {
                "quantity": "cauchy_schwarz_bound",
                "kind": "amplitude",
                "value": self.cauchy_schwarz_bound,
                "role": "upper bound on |lambda_res|",
            },
        ]


@dataclass(frozen=True)
class EmpiricalCommonModeReadings:
    """Lambda^emp_res and Lambda^emp_con for one K (11.3, 10.4.7).

    Both are computed from the occupancy-weighted L^2(rho^off) inner products
    of the conditioning arm's deployment error against c_bar and against the
    orthogonalised contrast subspace.

    Ambiguity that the proposal leaves open: 7.5 says the contrast reading is
    "the projection norm normalised by the subspace dimension" without fixing
    whether the norm or its square carries the division. Squared norms are
    additive across dimensions, so `lambda_emp_con` divides the squared norm by
    the dimension and takes the root, which keeps the reading an amplitude and
    comparable across K. `lambda_emp_con_linear_norm` is the other reading,
    reported alongside so the choice is visible rather than buried.
    """

    k: int
    lambda_emp_res: Amplitude
    projection_energy: Energy
    subspace_dimension: int
    weights: RegimeWeights

    def __post_init__(self) -> None:
        if self.k < 2:
            raise ValueError(f"K is the number of source embodiments and is >= 2, got {self.k}")
        if self.subspace_dimension < 1:
            raise ValueError(
                "the contrast subspace is empty; with K >= 2 and distinct compensation "
                "fields it has dimension at least 1"
            )
        if self.subspace_dimension > self.k - 1:
            raise ValueError(
                f"the contrast fields sum to zero, so their span has dimension at most "
                f"K-1 = {self.k - 1}, got {self.subspace_dimension}"
            )

    @property
    def lambda_emp_con(self) -> Amplitude:
        per_dimension = Energy(
            float(self.projection_energy.value) / self.subspace_dimension
        )
        return per_dimension.sqrt()

    @property
    def lambda_emp_con_linear_norm(self) -> Amplitude:
        return Amplitude(
            float(self.projection_energy.sqrt().value) / self.subspace_dimension
        )

    @classmethod
    def from_weighted_gram(
        cls,
        *,
        k: int,
        weights: RegimeWeights,
        per_regime_gram: Mapping[Regime, np.ndarray],
        names: Sequence[str],
        error_name: str,
        common_mode_name: str,
        contrast_names: Sequence[str],
        rank_tolerance: float = 1e-10,
    ) -> EmpiricalCommonModeReadings:
        """Assemble both readings from per-regime L^2 Gram matrices.

        `per_regime_gram[r][i, j]` is E_{rho^off|r}<f_i, f_j> with `names`
        giving the row order -- exactly what
        `streaming.L2GramAccumulator.gram(regime)` returns.
        """
        index = {name: i for i, name in enumerate(names)}
        for required in (error_name, common_mode_name, *contrast_names):
            if required not in index:
                raise ValueError(f"field {required!r} is not in the Gram, which has {names}")
        weighted = np.zeros((len(names), len(names)), dtype=np.float64)
        for regime in weights.regimes:
            gram = per_regime_gram.get(regime)
            if gram is None:
                raise ValueError(
                    f"occupancy weight {weights.of(regime)} was given for "
                    f"{regime.value} but no Gram; the weighted numerator and denominator "
                    "would be taken over different regime sets"
                )
            weighted += weights.of(regime) * np.asarray(gram, dtype=np.float64)

        i_eps = index[error_name]
        i_bar = index[common_mode_name]
        denominator = float(weighted[i_bar, i_bar])
        if denominator <= 0.0:
            raise ValueError(
                "the occupancy-weighted E||c_bar||^2 is zero: the sources have no common "
                "mode, so the common-mode reading has no denominator"
            )
        lambda_res = Amplitude(float(weighted[i_eps, i_bar]) / denominator)

        # Gram-Schmidt in the weighted L^2 inner product. Working on the Gram
        # rather than on frames keeps this streaming-compatible.
        contrast_idx = [index[n] for n in contrast_names]
        basis: list[np.ndarray] = []
        for column in contrast_idx:
            coord = np.zeros(len(names), dtype=np.float64)
            coord[column] = 1.0
            for existing in basis:
                coord = coord - float(existing @ weighted @ coord) * existing
            norm_sq = float(coord @ weighted @ coord)
            if norm_sq <= rank_tolerance * max(float(weighted[column, column]), 1.0):
                # Deliberately dropped: the contrast fields sum to zero, so one
                # direction is always dependent, and duplicated sources add more.
                continue
            basis.append(coord / math.sqrt(norm_sq))
        if not basis:
            raise ValueError(
                "the contrast fields span nothing: all K sources carry the same "
                "compensation, which is Q_div = 0 and the degenerate case of 7.5"
            )
        eps_coord = np.zeros(len(names), dtype=np.float64)
        eps_coord[i_eps] = 1.0
        projection = sum(float(u @ weighted @ eps_coord) ** 2 for u in basis)
        return cls(
            k=k,
            lambda_emp_res=lambda_res,
            projection_energy=Energy(projection),
            subspace_dimension=len(basis),
            weights=weights,
        )


@dataclass(frozen=True)
class KScanPoint:
    """One (K, reading) observation in the E7 scan."""

    k: int
    lambda_emp_res: Amplitude
    lambda_emp_con: Amplitude

    def __post_init__(self) -> None:
        if self.k < 2:
            raise ValueError(f"K >= 2 in the E7 scan, got {self.k}")


@dataclass(frozen=True)
class KScanVerdict:
    """Slope of one reading against log2 K, with the T9 criterion applied."""

    quantity: str
    slope: float
    standard_error: float
    ci_low: float
    ci_high: float
    n_points: int
    criterion: str
    passes: bool

    def as_row(self) -> dict[str, Any]:
        return {
            "quantity": self.quantity,
            "slope": self.slope,
            "se": self.standard_error,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_points": self.n_points,
            "criterion": self.criterion,
            "passes": self.passes,
        }


class KScanAnalysis:
    """The T9 pair as a function of K (11.3, 10.4.7).

    Lambda^emp_res must be **flat** in log2 K: the criterion is that the
    confidence interval of the slope lies inside +-0.033. Lambda^emp_con must
    **decline**: the criterion is a point slope of at most -0.08. Both criteria
    are specification values from 11.3, not calibrated thresholds, so they are
    named constants here rather than lock-file entries.

    The distinguishable alternative is registered as well: if the learner
    departs from the pooled conditional mean, Lambda^emp_res decays like
    K^{-1/2}, which over this design is a slope of about -0.075 -- more than
    twice the flat band, so the two hypotheses separate on power.

    Inference here is the ordinary-least-squares slope with a t interval. It is
    the within-arm reading; the cross random-effects model over
    (family x defect) that carries the final verdict lives in `eval/stats.py`.
    """

    def __init__(self, points: Sequence[KScanPoint], *, confidence: float = 0.95) -> None:
        if len(points) < 3:
            raise ValueError(
                f"a slope with a usable interval needs at least three K levels, got "
                f"{len(points)}; E7 runs K in {{2, 4, 6, 8}} (10.4.7)"
            )
        if len({p.k for p in points}) < 3:
            raise ValueError("at least three distinct K levels are needed for a slope")
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
        self._points = tuple(sorted(points, key=lambda p: p.k))
        self._confidence = confidence

    @property
    def log2_k(self) -> np.ndarray:
        return np.array([math.log2(p.k) for p in self._points], dtype=np.float64)

    def _slope(self, quantity: str, values: np.ndarray) -> tuple[float, float, float, float]:
        x = self.log2_k
        n = x.size
        x_centred = x - x.mean()
        sxx = float(np.dot(x_centred, x_centred))
        if sxx == 0.0:
            raise ValueError("all K levels are equal; the slope is not identified")
        slope = float(np.dot(x_centred, values - values.mean()) / sxx)
        intercept = float(values.mean() - slope * x.mean())
        residuals = values - (intercept + slope * x)
        dof = n - 2
        if dof <= 0:
            raise ValueError(f"{quantity}: {n} points leave no residual degrees of freedom")
        sigma2 = float(np.dot(residuals, residuals)) / dof
        se = math.sqrt(sigma2 / sxx)
        half = float(stats.t.ppf(0.5 + self._confidence / 2.0, dof)) * se
        return slope, se, slope - half, slope + half

    def residual_verdict(self) -> KScanVerdict:
        values = np.array(
            [float(p.lambda_emp_res.value) for p in self._points], dtype=np.float64
        )
        slope, se, low, high = self._slope("lambda_emp_res", values)
        return KScanVerdict(
            quantity="lambda_emp_res",
            slope=slope,
            standard_error=se,
            ci_low=low,
            ci_high=high,
            n_points=len(self._points),
            criterion=f"flat: CI within +-{T9_FLAT_BAND}",
            passes=low >= -T9_FLAT_BAND and high <= T9_FLAT_BAND,
        )

    def contrast_verdict(self) -> KScanVerdict:
        values = np.array(
            [float(p.lambda_emp_con.value) for p in self._points], dtype=np.float64
        )
        slope, se, low, high = self._slope("lambda_emp_con", values)
        return KScanVerdict(
            quantity="lambda_emp_con",
            slope=slope,
            standard_error=se,
            ci_low=low,
            ci_high=high,
            n_points=len(self._points),
            criterion=f"declines: slope <= {T9_DECLINE_SLOPE}",
            passes=slope <= T9_DECLINE_SLOPE,
        )

    def alternative_curve(self) -> dict[str, Any]:
        """The registered alternative: Lambda^emp_res(K) = A K^{-1/2} (10.4.7).

        A is anchored on the smallest K in the scan, so the curve is not fitted
        to the data it is being distinguished from.
        """
        anchor = self._points[0]
        amplitude = float(anchor.lambda_emp_res.value) * math.sqrt(anchor.k)
        predicted = np.array(
            [amplitude / math.sqrt(p.k) for p in self._points], dtype=np.float64
        )
        slope, _, _, _ = self._slope("alternative", predicted)
        return {
            "form": "A * K^(-1/2)",
            "amplitude": amplitude,
            "anchor_k": anchor.k,
            "predicted": {p.k: float(v) for p, v in zip(self._points, predicted, strict=True)},
            "slope_vs_log2_k": slope,
            "registered_slope": T9_ALTERNATIVE_SLOPE,
        }

    def t9_verdict(self) -> dict[str, Any]:
        """T9 needs both legs; either alone is not the prediction (7.5)."""
        residual = self.residual_verdict()
        contrast = self.contrast_verdict()
        return {
            "lambda_emp_res": residual.as_row(),
            "lambda_emp_con": contrast.as_row(),
            "alternative": self.alternative_curve(),
            "t9_supported": residual.passes and contrast.passes,
            "note": (
                "flat alone does not support T9: without the contrast leg declining, a "
                "flat reading cannot be told from a reading that is flat in everything "
                "(7.5)"
            ),
        }
