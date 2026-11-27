"""Excess leakage Lambda^exc against surrogate negative controls.

Implements proposal 6.4 layer 1, 9.2.11 and 10.4.1:

    Lambda^exc_delta = Lambda_delta - E_{delta_tilde ~ S}[Lambda_delta_tilde]

with S the preregistered surrogate set. Three clauses shape the code:

* The comparison is a descriptive rank of the real reading among the matched
  negative controls. The controls are not exchangeable random permutations,
  so the rank is not a p value or an exact permutation test.
* 11.3 requires the full negative-control distribution to be reported. The
  reading therefore carries every surrogate coefficient, and `as_rows`
  serialises them.
* A cell may lose surrogates to the four-dimensional matching tolerances of
  9.2.11. Down to 6 the cell still counts; below 6 its Lambda^exc is
  **exploratory only** (11.3) and this module marks it as such rather than
  suppressing or silently reporting it.

Both readings enter as Lambda^part (10.4.1): the real defect and each
surrogate take their turn as the regressor of interest against the same
interference basis. Feeding naive Lambda values in here would compare a
partial-regression number against a set of unadjusted ones.

The random-rotation null of 6.4 is available as a zero-cost first screen. It is
never the primary criterion: when c_delta concentrates on a low-dimensional
subspace of action space, rotation moves it into directions where the policy
error is near zero and the null becomes optimistic. Every reading that carries
one is flagged accordingly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from portal_trace.diagnostics.units import Amplitude

__all__ = [
    "PREREGISTERED_SURROGATE_COUNT",
    "MIN_EFFECTIVE_SURROGATES",
    "SurrogateNull",
    "ExcessLeakageReading",
    "RandomRotationScreen",
    "excess_leakage",
]

# 6.4 layer 1 and 9.2.11: the preregistered set has eight members.
PREREGISTERED_SURROGATE_COUNT = 8
# 11.3: a cell may drop below eight, but below six the reading is exploratory.
MIN_EFFECTIVE_SURROGATES = 6


@dataclass(frozen=True)
class SurrogateNull:
    """The surrogate negative-control distribution, reported in full (11.3).

    `admitted` is the per-cell effective cardinality |S| after the 9.2.11
    admission check; surrogates rejected by the four-dimensional matching
    tolerances are absent from `readings` and named in `rejected`.
    """

    readings: Mapping[str, Amplitude]
    rejected: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.readings:
            raise ValueError(
                "the surrogate negative-control distribution is empty; Lambda^exc is a difference "
                "against it and has no value (6.4 layer 1)"
            )
        overlap = set(self.readings) & set(self.rejected)
        if overlap:
            raise ValueError(f"surrogates both admitted and rejected: {sorted(overlap)}")

    @property
    def admitted(self) -> int:
        return len(self.readings)

    @property
    def values(self) -> np.ndarray:
        return np.array(
            [float(self.readings[name].value) for name in sorted(self.readings)],
            dtype=np.float64,
        )

    @property
    def mean(self) -> Amplitude:
        return Amplitude(float(self.values.mean()))

    @property
    def spread(self) -> float:
        """Sample standard deviation of the controls, for the results table only.

        A set of six to eight controls is worth showing with its scale attached.
        """
        if self.admitted < 2:
            raise ValueError("a spread needs at least two admitted surrogates")
        return float(self.values.std(ddof=1))

    def as_rows(self) -> list[dict[str, Any]]:
        rows = [
            {"surrogate": name, "admitted": True, "lambda_part": float(value.value)}
            for name, value in sorted(self.readings.items())
        ]
        rows += [
            {"surrogate": name, "admitted": False, "lambda_part": None}
            for name in sorted(self.rejected)
        ]
        return rows


@dataclass(frozen=True)
class RandomRotationScreen:
    """The zero-cost first screen of 6.4: c_delta rotated by one orthogonal Q.

    Norm preserving and state-alignment preserving, so it isolates directional
    alignment -- and precisely for that reason it is optimistic when c_delta is
    concentrated in action space. Carried alongside the primary reading and
    always labelled.
    """

    readings: Mapping[str, Amplitude]
    action_space_participation: float

    def __post_init__(self) -> None:
        if not self.readings:
            raise ValueError("a rotation screen with no rotations is not a screen")
        if not 0.0 < self.action_space_participation <= 1.0:
            raise ValueError(
                "action_space_participation is the share of action dimensions carrying "
                f"c_delta's energy and must lie in (0, 1], got "
                f"{self.action_space_participation}"
            )

    @property
    def mean(self) -> Amplitude:
        return Amplitude(
            float(np.mean([float(v.value) for v in self.readings.values()]))
        )

    @property
    def optimism_warning(self) -> str:
        return (
            "random-rotation null; optimistic when c_delta concentrates in action space "
            f"(participation {self.action_space_participation:.2f}). First screen only, "
            "never the primary criterion (6.4, 14.5 E-11)"
        )


@dataclass(frozen=True)
class ExcessLeakageReading:
    """Lambda^exc for one cell, with the whole null behind it."""

    lambda_part: Amplitude
    null: SurrogateNull
    rotation_screen: RandomRotationScreen | None = None

    @property
    def lambda_exc(self) -> Amplitude:
        return self.lambda_part - self.null.mean

    @property
    def rank(self) -> int:
        """Descriptive 1-based position among negative controls + real, ascending.

        The preregistered prediction of 10.4.1 is that Lambda^part_delta lands
        in the last position of a nine-point ordering (eight surrogates plus
        the real defect).
        """
        return int(np.sum(self.null.values < float(self.lambda_part.value))) + 1

    @property
    def negative_control_rank_fraction(self) -> float:
        """Descriptive rank divided by the number of controls plus the real item."""
        return self.rank / (self.null.admitted + 1)

    @property
    def is_confirmatory(self) -> bool:
        return self.null.admitted >= MIN_EFFECTIVE_SURROGATES

    @property
    def status(self) -> str:
        return "confirmatory" if self.is_confirmatory else "exploratory"

    @property
    def rank_resolution(self) -> float:
        """Spacing between adjacent descriptive rank fractions."""
        return 1.0 / (self.null.admitted + 1)

    def as_rows(self) -> list[dict[str, Any]]:
        """Point estimate, status and every negative-control reading (11.3)."""
        rows: list[dict[str, Any]] = [
            {
                "quantity": "lambda_exc",
                "value": float(self.lambda_exc.value),
                "status": self.status,
                "n_surrogates_effective": self.null.admitted,
                "negative_control_rank": self.rank,
                "negative_control_rank_fraction": self.negative_control_rank_fraction,
                "rank_resolution": self.rank_resolution,
                "rank_interpretation": "descriptive_only_not_a_p_value",
            },
            {
                "quantity": "lambda_part",
                "value": float(self.lambda_part.value),
                "status": self.status,
            },
            {
                "quantity": "null_mean",
                "value": float(self.null.mean.value),
                "status": self.status,
            },
        ]
        rows += [{"quantity": "null_member", **row} for row in self.null.as_rows()]
        if self.rotation_screen is not None:
            rows.append(
                {
                    "quantity": "rotation_screen_mean",
                    "value": float(self.rotation_screen.mean.value),
                    "status": "screen_only",
                    "note": self.rotation_screen.optimism_warning,
                }
            )
        return rows


def excess_leakage(
    lambda_part: Amplitude,
    surrogate_readings: Mapping[str, Amplitude],
    *,
    rejected_surrogates: tuple[str, ...] = (),
    rotation_screen: RandomRotationScreen | None = None,
) -> ExcessLeakageReading:
    """Assemble Lambda^exc for one cell from its Lambda^part readings.

    `surrogate_readings` are Lambda^part values with the surrogate as the
    regressor of interest (10.4.1), which is what
    `leakage.LeakageReport.surrogate_partial` returns.
    """
    if not isinstance(lambda_part, Amplitude):
        raise TypeError(
            f"lambda_part must be an Amplitude, got {type(lambda_part).__name__}; the "
            "Lambda family is amplitude valued (14.5 E-1)"
        )
    declared = len(surrogate_readings) + len(rejected_surrogates)
    if declared != PREREGISTERED_SURROGATE_COUNT:
        raise ValueError(
            f"the preregistered surrogate set has {PREREGISTERED_SURROGATE_COUNT} members "
            f"(9.2.11); {declared} were accounted for. Every rejection must be named so "
            "the effective cardinality can be deducted honestly"
        )
    return ExcessLeakageReading(
        lambda_part=lambda_part,
        null=SurrogateNull(readings=dict(surrogate_readings), rejected=rejected_surrogates),
        rotation_screen=rotation_screen,
    )
