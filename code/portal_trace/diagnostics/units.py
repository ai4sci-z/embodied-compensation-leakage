"""Amplitude and energy, kept apart by the type system.

Proposal 14.5 E-1 and 11.2 require that amplitude-valued diagnostics (the Lambda
family, rho_dir, kappa) and energy-valued ones (Phi, Q_div, expected squared
norms) cannot be mixed: appendix A-6 lists "compare Q_div against Lambda
directly" as an excluded path, because the two differ by a square. Prose cannot
enforce that; these wrappers can.

Crossing between the two is legal in exactly two places, and both are spelled
out as named methods so that they show up in a diff: `Amplitude.squared()` (this
is how Phi = rho_dir**2 is obtained) and `Energy.sqrt()`.

`EffectiveSampleCount` exists for the same reason. Appendix A-8 and 11.4 forbid
using raw frame counts as design-graph edge weights, since that overstates
precision by about an order of magnitude. The only constructors are the ones
that go through n_traj * (T / tau_corr).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "Amplitude",
    "Energy",
    "EffectiveSampleCount",
    "DimensionError",
    "amp",
    "energy",
]

Numeric = float | int | np.ndarray


class DimensionError(TypeError):
    """An amplitude was combined with an energy."""


def _describe(x: Any) -> str:
    return type(x).__name__


def _as_value(x: Numeric) -> float | np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    if isinstance(x, (int, float, np.floating, np.integer)):
        return float(x)
    raise TypeError(f"expected a float or ndarray, got {_describe(x)}")


@dataclass(frozen=True)
class _Quantity:
    """Shared arithmetic. Subclasses only differ by which type they are."""

    value: float | np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _as_value(self.value))

    # -- guards ---------------------------------------------------------

    def _same_kind(self, other: Any, op: str) -> _Quantity:
        if isinstance(other, _Quantity) and type(other) is not type(self):
            raise DimensionError(
                f"cannot {op} {type(other).__name__} and {type(self).__name__}: "
                "amplitude and energy differ by a square (proposal 14.5 E-1, "
                "appendix A-6). If you meant to compare them, square the "
                "amplitude with .squared() or take .sqrt() of the energy, and say "
                "in the report which one you did."
            )
        if not isinstance(other, _Quantity):
            raise DimensionError(
                f"cannot {op} a bare {_describe(other)} with {type(self).__name__}; "
                "wrap it so the dimension is explicit"
            )
        return other

    # -- within-kind arithmetic ------------------------------------------

    def __add__(self, other: Any) -> _Quantity:
        other = self._same_kind(other, "add")
        return type(self)(self.value + other.value)

    def __sub__(self, other: Any) -> _Quantity:
        other = self._same_kind(other, "subtract")
        return type(self)(self.value - other.value)

    def __neg__(self) -> _Quantity:
        return type(self)(-self.value)

    def __abs__(self) -> _Quantity:
        return type(self)(abs(self.value))

    def __mul__(self, scalar: Numeric) -> _Quantity:
        if isinstance(scalar, _Quantity):
            raise DimensionError(
                f"multiplying two {type(self).__name__} values changes the dimension; "
                "if you want an energy from amplitudes, use .squared() explicitly"
            )
        return type(self)(self.value * _as_value(scalar))

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> float | np.ndarray | _Quantity:
        if isinstance(other, _Quantity):
            other = self._same_kind(other, "divide")
            return self.value / other.value  # same kind cancels: plain ratio
        return type(self)(self.value / _as_value(other))

    # -- comparison (within kind only) ------------------------------------

    def __lt__(self, other: Any) -> Any:
        return self.value < self._same_kind(other, "compare").value

    def __le__(self, other: Any) -> Any:
        return self.value <= self._same_kind(other, "compare").value

    def __gt__(self, other: Any) -> Any:
        return self.value > self._same_kind(other, "compare").value

    def __ge__(self, other: Any) -> Any:
        return self.value >= self._same_kind(other, "compare").value

    def __float__(self) -> float:
        if isinstance(self.value, np.ndarray):
            raise TypeError(f"{type(self).__name__} holds an array, not a scalar")
        return float(self.value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.value!r})"


@dataclass(frozen=True, repr=False)
class Amplitude(_Quantity):
    """A first-order quantity: Lambda and its relatives, rho_dir, kappa."""

    def squared(self) -> Energy:
        """The one legal amplitude -> energy crossing.

        Phi = rho_dir**2 (proposal 6.3) is obtained this way and no other way;
        appendix A-14 forbids treating Phi and rho_dir as two independent
        pieces of evidence, so Phi is never computed from its own formula.
        """
        return Energy(np.square(self.value) if isinstance(self.value, np.ndarray)
                      else self.value ** 2)


@dataclass(frozen=True, repr=False)
class Energy(_Quantity):
    """A second-order quantity: Phi, Q_div, expected squared norms."""

    def sqrt(self) -> Amplitude:
        """The one legal energy -> amplitude crossing."""
        if isinstance(self.value, np.ndarray):
            if np.any(self.value < 0):
                raise ValueError("cannot take sqrt of a negative energy")
            return Amplitude(np.sqrt(self.value))
        if self.value < 0:
            raise ValueError("cannot take sqrt of a negative energy")
        return Amplitude(math.sqrt(self.value))


def amp(x: Numeric) -> Amplitude:
    return Amplitude(x)


def energy(x: Numeric) -> Energy:
    return Energy(x)


@dataclass(frozen=True)
class EffectiveSampleCount:
    """n_traj * (T / tau_corr), the only admissible design-graph edge weight.

    There is deliberately no public constructor that takes a frame count. The
    proposal (11.4, appendix A-8) is explicit that using raw frames overstates
    precision by roughly an order of magnitude, and `design_graph` accepts only
    this type, so the mistake cannot be made by passing an int.
    """

    value: float
    n_traj: int
    horizon: int
    tau_corr: float
    sigma2: float | None = None

    def __post_init__(self) -> None:
        if self.n_traj <= 0:
            raise ValueError(f"n_traj must be positive, got {self.n_traj}")
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        if not (self.tau_corr > 0):
            raise ValueError(
                f"tau_corr must be positive, got {self.tau_corr}; it is measured "
                "per cell and per regime, never assumed"
            )
        if self.sigma2 is not None and not (self.sigma2 > 0):
            raise ValueError(f"sigma2 must be positive, got {self.sigma2}")

    @classmethod
    def from_trajectories(
        cls, *, n_traj: int, horizon: int, tau_corr: float
    ) -> EffectiveSampleCount:
        """Homoscedastic edge weight: w = n_traj * (T / tau_corr)."""
        return cls(
            value=n_traj * (horizon / tau_corr),
            n_traj=n_traj,
            horizon=horizon,
            tau_corr=tau_corr,
        )

    @classmethod
    def from_trajectories_heteroscedastic(
        cls, *, n_traj: int, horizon: int, tau_corr: float, sigma2: float
    ) -> EffectiveSampleCount:
        """Heteroscedastic conductance: w = n_eff / sigma^2 (proposal 7.9).

        Both calibers are reported, so both are constructible; which one a
        result used is recorded alongside it rather than inferred.
        """
        n_eff = n_traj * (horizon / tau_corr)
        return cls(
            value=n_eff / sigma2,
            n_traj=n_traj,
            horizon=horizon,
            tau_corr=tau_corr,
            sigma2=sigma2,
        )

    @property
    def is_heteroscedastic(self) -> bool:
        return self.sigma2 is not None

    @property
    def raw_frames(self) -> int:
        """The frame count this was derived from, for reporting only.

        Present so that data cards can show both numbers side by side. It is
        not accepted anywhere as a weight.
        """
        return self.n_traj * self.horizon

    def __float__(self) -> float:
        return float(self.value)

    def __repr__(self) -> str:
        kind = "heteroscedastic" if self.is_heteroscedastic else "homoscedastic"
        return (
            f"EffectiveSampleCount({self.value:.4g}, {kind}, "
            f"n_traj={self.n_traj}, T={self.horizon}, tau_corr={self.tau_corr:.4g})"
        )
