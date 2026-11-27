"""The Lambda family: Lambda^ref, Lambda^orc, Lambda^part, Phi, rho_dir.

Implements proposal 6.1-6.5 and the reporting discipline of 10.1.4-10.1.5 and
11.3. Four things in here are decisions rather than transcription, and each is
forced by a clause of the proposal:

1. Every expectation is taken over frames first and the ratio second. 6.3 gives
   the counterexample: read as a per-state cosine, rho_dir is not even sign
   consistent with the L^2(rho) quantity. `Lambda` is therefore assembled from
   the entries of one `L2GramAccumulator`, never from per-frame ratios.

2. Phi is obtained only as `rho_dir.squared()`. Appendix A-14 forbids treating
   Phi and rho_dir as two pieces of evidence; they are the same condition
   written twice, so there is no second code path that could disagree.

3. Lambda^part is the primary endpoint and naive Lambda is reported next to it
   in the same table, unconditionally (11.3, 14.5 E-10). Their difference is
   the overall nuisance adjustment. The shrinkage contribution is identified
   separately by a nested fit that changes only the shrinkage column.

4. The deterministic nuisance basis is mandatory: the shrinkage field
   -(pi_{e',0,tau} - a_bar), the time-smoothing field (S_h pi - pi), and the
   gain-error field pi itself. The real compensation field and each surrogate
   negative control are estimated in separate regressions against this same
   deterministic basis. They never compete as simultaneous regressors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from portal_trace.diagnostics.streaming import (
    L2GramAccumulator,
    NeumaierSum,
    StreamingLeastSquares,
    smoothing_field,
)
from portal_trace.diagnostics.units import Amplitude, Energy
from portal_trace.types import Regime

__all__ = [
    "FIELD_EPS_ORC",
    "FIELD_EPS_REF",
    "FIELD_C_DELTA",
    "FIELD_U_ORC",
    "NU_SHRINKAGE",
    "NU_TIME_SMOOTHING",
    "NU_GAIN_ERROR",
    "SURROGATE_PREFIX",
    "DEFAULT_MIN_DENOMINATOR_ENERGY",
    "InterferenceBasis",
    "LeakageChunk",
    "LeakageAccumulator",
    "LeakageReport",
    "PartialRegressionReading",
    "ThreeTermDecomposition",
    "PosteriorCovarianceAccumulator",
    "PosteriorSensitivity",
    "shrinkage_field",
    "time_smoothing_field",
    "gain_error_field",
    "surrogate_column",
]

FIELD_EPS_ORC = "eps_orc"
FIELD_EPS_REF = "eps_ref"
FIELD_C_DELTA = "c_delta"
FIELD_U_ORC = "u_orc"
NU_SHRINKAGE = "nu_shrinkage"
NU_TIME_SMOOTHING = "nu_time_smoothing"
NU_GAIN_ERROR = "nu_gain_error"
SURROGATE_PREFIX = "nu_surrogate:"

# 6.4 layer 2 classes (i)-(iii). Class (iv) is the surrogate set, whose size is
# a property of the preregistered surrogate list rather than a constant.
_DETERMINISTIC_NUISANCE = (NU_SHRINKAGE, NU_TIME_SMOOTHING, NU_GAIN_ERROR)

DEFAULT_MIN_DENOMINATOR_ENERGY = 1e-12
_NUMERICAL_RESIDUAL_SHARE_FLOOR = 64.0 * np.finfo(np.float64).eps


def surrogate_column(name: str) -> str:
    """Column name of a surrogate compensation field in the interference basis."""
    return f"{SURROGATE_PREFIX}{name}"


def shrinkage_field(pi_clean: np.ndarray, a_bar: np.ndarray) -> np.ndarray:
    """nu_1 = -(pi_{e',0,tau}(x) - a_bar), the shrinkage field of 6.4 layer 2.

    `a_bar` is the mean action of the *pooled training set*, which contains
    compensation (a_bar = mu_bar + c_bar, 6.2). Passing the evaluation-set mean
    instead would remove exactly the term this column exists to absorb.
    """
    pi = np.asarray(pi_clean, dtype=np.float64)
    mean = np.asarray(a_bar, dtype=np.float64)
    if mean.shape != (pi.shape[1],):
        raise ValueError(f"a_bar must have shape ({pi.shape[1]},), got {mean.shape}")
    return -(pi - mean)


def time_smoothing_field(
    pi_clean: np.ndarray, episode_index: np.ndarray, window: int
) -> np.ndarray:
    """nu_2 = (S_h pi_{e',0,tau} - pi_{e',0,tau})(x)."""
    return smoothing_field(pi_clean, episode_index, window)


def gain_error_field(pi_clean: np.ndarray) -> np.ndarray:
    """nu_3 = pi_{e',0,tau}(x) itself."""
    return np.asarray(pi_clean, dtype=np.float64)


@dataclass(frozen=True)
class InterferenceBasis:
    """One chunk's deterministic nuisance fields and surrogate controls.

    The three deterministic fields are the common adjustment set for the real
    target and every surrogate. Surrogates are required negative controls, but
    are not placed together in one regression design.
    """

    shrinkage: np.ndarray
    time_smoothing: np.ndarray
    gain_error: np.ndarray
    surrogates: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if not self.surrogates:
            raise ValueError(
                "class (iv) of the interference basis is the surrogate compensation "
                "fields (6.4 layer 2); an empty surrogate set is not a basis"
            )
        shapes = {
            "shrinkage": np.shape(self.shrinkage),
            "time_smoothing": np.shape(self.time_smoothing),
            "gain_error": np.shape(self.gain_error),
        }
        for name, arr in self.surrogates.items():
            shapes[f"surrogate {name}"] = np.shape(arr)
        distinct = set(shapes.values())
        if len(distinct) != 1:
            raise ValueError(f"interference fields disagree on shape: {shapes}")

    @property
    def surrogate_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.surrogates))

    def as_columns(self) -> dict[str, np.ndarray]:
        columns = {
            NU_SHRINKAGE: np.asarray(self.shrinkage, dtype=np.float64),
            NU_TIME_SMOOTHING: np.asarray(self.time_smoothing, dtype=np.float64),
            NU_GAIN_ERROR: np.asarray(self.gain_error, dtype=np.float64),
        }
        for name in self.surrogate_names:
            columns[surrogate_column(name)] = np.asarray(
                self.surrogates[name], dtype=np.float64
            )
        return columns


@dataclass(frozen=True)
class LeakageChunk:
    """One chunk of frames evaluated on rho^off (6.6, 14.5 E-8).

    `eps_orc` and `eps_ref` are the same deployment error against the two
    reference points of 6.2; both are required because 6.2 makes reporting
    both compulsory and their difference informative. `u_orc` is the Oracle
    arm's own underfitting error, needed for the third term of 10.1.5; it is
    optional only because it comes from a separate, zero-new-training pass.
    """

    eps_orc: np.ndarray
    eps_ref: np.ndarray
    c_delta: np.ndarray
    basis: InterferenceBasis
    regime: np.ndarray
    u_orc: np.ndarray | None = None

    def columns(self) -> dict[str, np.ndarray]:
        columns = {
            FIELD_EPS_ORC: np.asarray(self.eps_orc, dtype=np.float64),
            FIELD_EPS_REF: np.asarray(self.eps_ref, dtype=np.float64),
            FIELD_C_DELTA: np.asarray(self.c_delta, dtype=np.float64),
        }
        columns.update(self.basis.as_columns())
        if self.u_orc is not None:
            columns[FIELD_U_ORC] = np.asarray(self.u_orc, dtype=np.float64)
        return columns


@dataclass(frozen=True)
class PartialRegressionReading:
    """Lambda^part for one regressor of interest, with its nuisance coefficients."""

    target: str
    lambda_part: Amplitude
    nuisance_coefficients: dict[str, float]
    condition_number: float
    residual_share: float
    n_rows: int

    @property
    def vif(self) -> float:
        if self.residual_share <= 0.0:
            raise ValueError("VIF is undefined when target residual energy is zero")
        return 1.0 / self.residual_share


class PosteriorCovarianceAccumulator:
    """Cov_rho(w, ||c_delta||^2) and E_rho[w] for the three-term decomposition.

    w(x) is the source posterior from the discriminator (lemma 1). One instance
    is fed per distribution. Only an instance matching Lambda's distribution
    may enter its decomposition; other distributions are sensitivity analyses.
    """

    def __init__(
        self,
        distribution: str,
        *,
        min_denominator_energy: float = DEFAULT_MIN_DENOMINATOR_ENERGY,
    ) -> None:
        if not distribution.strip():
            raise ValueError("distribution must be named")
        if not np.isfinite(min_denominator_energy) or min_denominator_energy <= 0.0:
            raise ValueError("min_denominator_energy must be finite and positive")
        self.distribution = distribution
        self.min_denominator_energy = float(min_denominator_energy)
        self._n = 0
        self._sum_w = NeumaierSum(())
        self._sum_e = NeumaierSum(())
        self._sum_we = NeumaierSum(())

    def update(self, posterior_w: np.ndarray, c_delta: np.ndarray) -> None:
        w = np.asarray(posterior_w, dtype=np.float64).reshape(-1)
        c = np.asarray(c_delta, dtype=np.float64)
        if c.ndim != 2 or c.shape[0] != w.shape[0]:
            raise ValueError(f"c_delta must be ({w.shape[0]}, d), got {c.shape}")
        if not np.all(np.isfinite(w)) or not np.all(np.isfinite(c)):
            raise ValueError("posterior_w and c_delta must be finite")
        if np.any(w < 0.0) or np.any(w > 1.0):
            raise ValueError("w(x) is a posterior probability and must lie in [0, 1]")
        energy_per_frame = np.einsum("nd,nd->n", c, c)
        self._n += w.shape[0]
        self._sum_w.add(float(w.sum()))
        self._sum_e.add(float(energy_per_frame.sum()))
        self._sum_we.add(float(np.dot(w, energy_per_frame)))

    @property
    def bar_alpha_rho(self) -> float:
        """E_rho[w] -- lemma 1's alpha only when rho is the pooled marginal."""
        if self._n == 0:
            raise ValueError("no frames accumulated")
        return float(self._sum_w.value) / self._n

    @property
    def mean_energy(self) -> Energy:
        if self._n == 0:
            raise ValueError("no frames accumulated")
        return Energy(float(self._sum_e.value) / self._n)

    @property
    def covariance_term(self) -> Amplitude:
        """Cov_rho(w, ||c_delta||^2) / E_rho||c_delta||^2 (lemma 1)."""
        if self._n == 0:
            raise ValueError("no frames accumulated")
        mean_w = self.bar_alpha_rho
        mean_e = float(self.mean_energy.value)
        mean_we = float(self._sum_we.value) / self._n
        if mean_e <= self.min_denominator_energy:
            raise ValueError(
                "E||c_delta||^2 is at or below the preregistered denominator floor "
                f"({mean_e:.6g} <= {self.min_denominator_energy:.6g}); the covariance "
                "term is undefined and must not be reported"
            )
        return Amplitude((mean_we - mean_w * mean_e) / mean_e)


@dataclass(frozen=True)
class PosteriorSensitivity:
    """Posterior terms on a non-primary distribution, reported as sensitivity."""

    distribution: str
    bar_alpha_rho: float
    covariance_term: Amplitude


@dataclass(frozen=True)
class ThreeTermDecomposition:
    """Lambda^orc = bar_alpha_rho + Cov/E||c||^2 - E<u_orc,c>/E||c||^2 (10.1.5).

    The third term is the reference-arm alignment term and **its sign is
    unknown** (6.5 scope note ii). `Lambda - bar_alpha_rho` therefore may not
    be read on its own; `covariance_from_reading` is what the first two terms
    imply once the third has been subtracted, and it is the number T1 compares
    against the directly measured covariance term.

    Every term in the identity is evaluated on `distribution`. A pooled-state
    calculation may be attached as `pooled_sensitivity`, but it is never added
    to a rho^off term or presented as part of the identity.
    """

    lambda_orc: Amplitude
    distribution: str
    bar_alpha_rho: float
    covariance_term: Amplitude
    reference_arm_alignment: Amplitude
    pooled_sensitivity: PosteriorSensitivity | None = None

    @property
    def covariance_from_reading(self) -> Amplitude:
        """(Lambda^orc - bar_alpha_rho) + reference-arm alignment."""
        return Amplitude(
            float(self.lambda_orc.value)
            - self.bar_alpha_rho
            + float(self.reference_arm_alignment.value)
        )

    @property
    def identity_residual(self) -> float:
        """Observed Lambda minus the same-distribution three-term reconstruction."""
        reconstructed = (
            self.bar_alpha_rho
            + float(self.covariance_term.value)
            - float(self.reference_arm_alignment.value)
        )
        return float(self.lambda_orc.value) - reconstructed

    def as_rows(self) -> list[dict[str, Any]]:
        rows = [
            {
                "term": "bar_alpha_rho",
                "value": self.bar_alpha_rho,
                "distribution": self.distribution,
                "sign_known": True,
            },
            {
                "term": "covariance",
                "value": float(self.covariance_term.value),
                "distribution": self.distribution,
                "sign_known": True,
            },
            {
                "term": "reference_arm_alignment",
                "value": float(self.reference_arm_alignment.value),
                "distribution": self.distribution,
                "sign_known": False,
            },
        ]
        if self.pooled_sensitivity is not None:
            rows.extend(
                [
                    {
                        "term": "bar_alpha_rho",
                        "value": self.pooled_sensitivity.bar_alpha_rho,
                        "distribution": self.pooled_sensitivity.distribution,
                        "role": "sensitivity_only",
                    },
                    {
                        "term": "covariance",
                        "value": float(self.pooled_sensitivity.covariance_term.value),
                        "distribution": self.pooled_sensitivity.distribution,
                        "role": "sensitivity_only",
                    },
                ]
            )
        return rows


@dataclass(frozen=True)
class LeakageReport:
    """Everything 11.4's model card asks for from this module, in one object."""

    lambda_ref: Amplitude
    lambda_orc: Amplitude
    lambda_part: Amplitude
    rho_dir_orc: Amplitude
    rho_dir_ref: Amplitude
    phi_leakage_share: Energy
    energy_c_delta: Energy
    energy_eps_orc: Energy
    energy_u_orc: Energy | None
    oracle_residual_energy_ratio: float | None
    surrogate_partial: dict[str, Amplitude]
    surrogate_naive: dict[str, Amplitude]
    nuisance_coefficients: dict[str, float]
    lambda_without_shrinkage: Amplitude
    target_residual_energy_share: float
    target_vif: float
    target_residual_share_floor: float | None
    per_regime_lambda_orc: dict[Regime, Amplitude]
    occupancy: dict[Regime, float]
    condition_number: float
    n_frames: int
    excluded_surrogates: tuple[str, ...]
    three_term: ThreeTermDecomposition | None = None

    @property
    def overall_nuisance_adjustment(self) -> Amplitude:
        """Naive minus fully adjusted Lambda; not attributable to one nuisance."""
        return self.lambda_orc - self.lambda_part

    @property
    def nested_shrinkage_effect(self) -> Amplitude:
        """Change from adding shrinkage to an otherwise identical nuisance fit."""
        return self.lambda_without_shrinkage - self.lambda_part

    def as_table(self) -> list[dict[str, Any]]:
        """The primary and companion readings, always in the same table (11.3)."""
        rows = [
            {"quantity": "lambda_part", "role": "primary", "value": float(self.lambda_part.value)},
            {"quantity": "lambda_orc", "role": "companion", "value": float(self.lambda_orc.value)},
            {"quantity": "lambda_ref", "role": "companion", "value": float(self.lambda_ref.value)},
            {
                "quantity": "overall_nuisance_adjustment",
                "role": "derived",
                "value": float(self.overall_nuisance_adjustment.value),
            },
            {
                "quantity": "nested_shrinkage_effect",
                "role": "derived_nested_contrast",
                "value": float(self.nested_shrinkage_effect.value),
            },
            {
                "quantity": "target_residual_energy_share",
                "role": "identifiability_diagnostic",
                "value": self.target_residual_energy_share,
            },
            {
                "quantity": "target_vif",
                "role": "identifiability_diagnostic",
                "value": self.target_vif,
            },
            {
                "quantity": "rho_dir",
                "role": "companion",
                "value": float(self.rho_dir_orc.value),
            },
            {
                "quantity": "phi_leakage_share",
                "role": "companion",
                "value": float(self.phi_leakage_share.value),
            },
        ]
        if self.oracle_residual_energy_ratio is not None:
            rows.append(
                {
                    "quantity": "oracle_residual_energy_ratio",
                    "role": "reference_arm_fidelity_diagnostic",
                    "value": self.oracle_residual_energy_ratio,
                }
            )
        return rows


class LeakageAccumulator:
    """One streaming pass produces every reading in `LeakageReport`.

    A separate partial regression is kept for c_delta and for each surrogate.
    Each uses exactly the same deterministic nuisance columns; the real target
    and surrogate targets never enter one another's design.
    """

    def __init__(
        self,
        surrogate_names: Sequence[str],
        *,
        redundant_surrogates: Sequence[str] = (),
        excluded_surrogates: Sequence[str] = (),
        with_reference_arm: bool = False,
        distribution: str = "rho_off",
        target_residual_share_floor: float | None = None,
        min_denominator_energy: float = DEFAULT_MIN_DENOMINATOR_ENERGY,
    ) -> None:
        names = tuple(sorted(surrogate_names))
        if not names:
            raise ValueError("the interference basis needs at least one surrogate field")
        if redundant_surrogates:
            raise ValueError(
                "redundant_surrogates is obsolete: real and surrogate coefficients are "
                "estimated separately against the same deterministic nuisance basis"
            )
        if not distribution.strip():
            raise ValueError("distribution must be named")
        if target_residual_share_floor is not None and (
            not np.isfinite(target_residual_share_floor)
            or not 0.0 < target_residual_share_floor <= 1.0
        ):
            raise ValueError("target_residual_share_floor must lie in (0, 1]")
        if not np.isfinite(min_denominator_energy) or min_denominator_energy <= 0.0:
            raise ValueError("min_denominator_energy must be finite and positive")
        excluded = tuple(sorted(excluded_surrogates))
        if set(names) & set(excluded):
            raise ValueError("a surrogate cannot be both admitted and excluded")
        if len(set(excluded)) != len(excluded):
            raise ValueError("excluded surrogate names must be unique")
        self._surrogates = names
        self._excluded = excluded
        self._with_reference_arm = with_reference_arm
        self._distribution = distribution
        self._target_residual_share_floor = target_residual_share_floor
        self._min_denominator_energy = float(min_denominator_energy)

        gram_names = [FIELD_EPS_ORC, FIELD_EPS_REF, FIELD_C_DELTA, *_DETERMINISTIC_NUISANCE]
        gram_names += [surrogate_column(n) for n in names]
        if with_reference_arm:
            gram_names.append(FIELD_U_ORC)
        self._gram = L2GramAccumulator(gram_names)

        self._targets = (FIELD_C_DELTA, *(surrogate_column(n) for n in names))
        self._fits = {
            target: StreamingLeastSquares((target, *_DETERMINISTIC_NUISANCE))
            for target in self._targets
        }
        self._without_shrinkage_fit = StreamingLeastSquares(
            (FIELD_C_DELTA, NU_TIME_SMOOTHING, NU_GAIN_ERROR)
        )
        self._target_projection = StreamingLeastSquares(_DETERMINISTIC_NUISANCE)
        self._n_frames = 0

    @property
    def excluded_surrogates(self) -> tuple[str, ...]:
        return self._excluded

    def update(self, chunk: LeakageChunk) -> None:
        columns = chunk.columns()
        if self._with_reference_arm and FIELD_U_ORC not in columns:
            raise ValueError(
                "this accumulator was built with with_reference_arm=True, so every chunk "
                "must carry u_orc for the third term of 10.1.5"
            )
        if not self._with_reference_arm and FIELD_U_ORC in columns:
            raise ValueError(
                "u_orc was supplied but the accumulator was not built for the three-term "
                "decomposition; build it with with_reference_arm=True"
            )
        supplied = set(chunk.basis.surrogate_names)
        if supplied != set(self._surrogates):
            raise ValueError(
                f"surrogate set changed mid-stream: declared {sorted(self._surrogates)}, "
                f"chunk carries {sorted(supplied)}"
            )
        self._gram.update(columns, chunk.regime)
        target_values = np.asarray(chunk.eps_orc, dtype=np.float64)
        for fit in self._fits.values():
            design = {name: columns[name] for name in fit.names}
            fit.update_fields(design, target_values)
        self._without_shrinkage_fit.update_fields(
            {name: columns[name] for name in self._without_shrinkage_fit.names},
            target_values,
        )
        self._target_projection.update_fields(
            {name: columns[name] for name in self._target_projection.names},
            columns[FIELD_C_DELTA],
        )
        self._n_frames += target_values.shape[0]

    def _require_energy(self, value: Energy, label: str) -> float:
        energy = float(value.value)
        if not np.isfinite(energy) or energy <= self._min_denominator_energy:
            raise ValueError(
                f"{label}={energy:.6g} is at or below the preregistered denominator "
                f"floor {self._min_denominator_energy:.6g}; the reading is undefined"
            )
        return energy

    def _naive_lambda(self, target: str, regime: Regime | None = None) -> Amplitude:
        numerator = self._gram.cross(FIELD_EPS_ORC, target, regime)
        denominator = self._gram.energy(target, regime)
        self._require_energy(
            denominator,
            f"E||{target}||^2 on {self._distribution if regime is None else regime.value}",
        )
        # Same-kind division cancels to a plain ratio; Lambda is an amplitude.
        return Amplitude(float(numerator / denominator))

    def finalise(
        self,
        *,
        posterior_off: PosteriorCovarianceAccumulator | None = None,
        posterior_pooled: PosteriorCovarianceAccumulator | None = None,
    ) -> LeakageReport:
        if self._n_frames == 0:
            raise ValueError("no frames accumulated; there is nothing to report")
        lambda_orc = self._naive_lambda(FIELD_C_DELTA)
        numerator_ref = self._gram.cross(FIELD_EPS_REF, FIELD_C_DELTA)
        energy_c = self._gram.energy(FIELD_C_DELTA)
        self._require_energy(energy_c, "E||c_delta||^2")
        lambda_ref = Amplitude(float(numerator_ref / energy_c))

        energy_eps_orc = self._gram.energy(FIELD_EPS_ORC)
        energy_eps_ref = self._gram.energy(FIELD_EPS_REF)
        energy_u_orc = (
            self._gram.energy(FIELD_U_ORC) if self._with_reference_arm else None
        )
        # rho_dir = E<eps,c> / (||eps||_{L2} ||c||_{L2}); the two norms are the
        # only legal energy -> amplitude crossings in this function.
        self._require_energy(energy_eps_orc, "E||eps_orc||^2")
        self._require_energy(energy_eps_ref, "E||eps_ref||^2")
        if energy_u_orc is not None and (
            not np.isfinite(float(energy_u_orc.value))
            or float(energy_u_orc.value) < 0.0
        ):
            raise ValueError("E||u_orc||^2 is invalid")
        norm_c = energy_c.sqrt()
        rho_dir_orc = Amplitude(
            float(self._gram.cross(FIELD_EPS_ORC, FIELD_C_DELTA).value)
            / (float(energy_eps_orc.sqrt().value) * float(norm_c.value))
        )
        rho_dir_ref = Amplitude(
            float(numerator_ref.value)
            / (float(energy_eps_ref.sqrt().value) * float(norm_c.value))
        )
        # A-14: Phi has no formula of its own.
        phi = rho_dir_orc.squared()

        solutions = {target: fit.solve() for target, fit in self._fits.items()}
        primary = solutions[FIELD_C_DELTA]
        lambda_part = Amplitude(primary.coefficient(FIELD_C_DELTA))
        without_shrinkage = self._without_shrinkage_fit.solve()
        lambda_without_shrinkage = Amplitude(
            without_shrinkage.coefficient(FIELD_C_DELTA)
        )
        projection = self._target_projection.solve()
        if projection.total_sum_of_squares <= 0.0:
            raise ValueError("c_delta has zero total energy; residual share and VIF are undefined")
        residual_share = projection.residual_sum_of_squares / projection.total_sum_of_squares
        floor = max(
            _NUMERICAL_RESIDUAL_SHARE_FLOOR,
            self._target_residual_share_floor or 0.0,
        )
        if not np.isfinite(residual_share) or residual_share < floor:
            raise ValueError(
                "c_delta residual energy after deterministic nuisance projection is "
                f"{residual_share:.6g}, below the required floor {floor:.6g}; Lambda^part "
                "is not identifiable and reporting fails closed"
            )
        target_vif = 1.0 / residual_share
        nuisance = {
            name: primary.coefficient(name)
            for name in primary.names
            if name != FIELD_C_DELTA
        }

        surrogate_partial = {
            name: Amplitude(solutions[surrogate_column(name)].coefficient(surrogate_column(name)))
            for name in self._surrogates
        }
        surrogate_naive = {
            name: self._naive_lambda(surrogate_column(name)) for name in self._surrogates
        }

        per_regime = {}
        occupancy = self._gram.occupancy()
        for r in Regime:
            if self._gram.count(r) == 0:
                continue
            per_regime[r] = self._naive_lambda(FIELD_C_DELTA, regime=r)

        three_term = None
        if posterior_off is not None or posterior_pooled is not None:
            if posterior_off is None:
                raise ValueError(
                    "the three-term decomposition needs a posterior accumulator on the "
                    "same distribution as Lambda; pooled-only input is sensitivity, not "
                    "an identity"
                )
            if posterior_off.distribution != self._distribution:
                raise ValueError(
                    "all terms in the three-term decomposition must use one distribution: "
                    f"Lambda uses {self._distribution!r}, posterior uses "
                    f"{posterior_off.distribution!r}"
                )
            if (
                posterior_pooled is not None
                and posterior_pooled.distribution == self._distribution
            ):
                raise ValueError(
                    "posterior_pooled duplicates the primary distribution; omit it or "
                    "provide a genuinely different pooled sensitivity distribution"
                )
            if not self._with_reference_arm:
                raise ValueError(
                    "the reference-arm alignment term needs u_orc; build the accumulator "
                    "with with_reference_arm=True"
                )
            alignment = Amplitude(
                float(self._gram.cross(FIELD_U_ORC, FIELD_C_DELTA) / energy_c)
            )
            three_term = ThreeTermDecomposition(
                lambda_orc=lambda_orc,
                distribution=self._distribution,
                bar_alpha_rho=posterior_off.bar_alpha_rho,
                covariance_term=posterior_off.covariance_term,
                reference_arm_alignment=alignment,
                pooled_sensitivity=(
                    None
                    if posterior_pooled is None
                    else PosteriorSensitivity(
                        distribution=posterior_pooled.distribution,
                        bar_alpha_rho=posterior_pooled.bar_alpha_rho,
                        covariance_term=posterior_pooled.covariance_term,
                    )
                ),
            )

        return LeakageReport(
            lambda_ref=lambda_ref,
            lambda_orc=lambda_orc,
            lambda_part=lambda_part,
            rho_dir_orc=rho_dir_orc,
            rho_dir_ref=rho_dir_ref,
            phi_leakage_share=phi,
            energy_c_delta=energy_c,
            energy_eps_orc=energy_eps_orc,
            energy_u_orc=energy_u_orc,
            oracle_residual_energy_ratio=(
                None if energy_u_orc is None else float(energy_u_orc / energy_c)
            ),
            surrogate_partial=surrogate_partial,
            surrogate_naive=surrogate_naive,
            nuisance_coefficients=nuisance,
            lambda_without_shrinkage=lambda_without_shrinkage,
            target_residual_energy_share=residual_share,
            target_vif=target_vif,
            target_residual_share_floor=self._target_residual_share_floor,
            per_regime_lambda_orc=per_regime,
            occupancy=occupancy,
            condition_number=primary.condition_number,
            n_frames=self._n_frames,
            excluded_surrogates=self._excluded,
            three_term=three_term,
        )
