"""`Delta`: the analytic representation of one defect instance.

This is the object that flows through the whole injection stack. It carries no
simulator state and no family-specific index: it is (defect code, level, the
resolved injection parameters, a per-joint scaling), and nothing else. The
injectors in `inject.py`, the closed-form fields in `fields.py` and the
surrogate matching in `surrogate.py` all read the same `Delta`.

Two decisions are recorded here rather than spread over the injectors.

**delta = 0 is not "all parameters zero".** CONTRACT 3.3 makes the clean twin
`apply(delta.zero())` on the same code path, so `zero()` has to produce a
parameter set that is the *identity of that defect's operator*, not a vector of
zeros. For an additive defect that is 0; for the multiplicative D8 it is g = 1;
for a compliance defect it is the rigid limit, which 9.2.3 fixes at
k0 = 10^6 N m/rad; for D4hi it is the +-10^-5 rad shadow band that 9.3.4
requires the clean twin to carry so that the two kinematic graphs match node for
node. `identity_values` states this per defect, and `zero()` reads it. Parameters
that are structural rather than dose-carrying (D9's pole count and phase, D10's
time constant) are preserved by `zero()`, because zeroing them would change the
shape of the operator rather than its magnitude.

**Per-joint scaling interpolates from the identity, not from zero.** The
effective value on joint j is `identity + scale_j * (value - identity)`. That
single rule is correct for additive and multiplicative parameters alike: a
one-hot scale gives 9.2.1's single-joint offset, and it gives g_j = 1 on the
untouched joints for D8 rather than g_j = 0, which would be a torque cut rather
than an unscaled joint.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from portal_trace.defects.params import DEFECTS, DefectSpec, defect
from portal_trace.lockfile import REQUIRED_MEASUREMENTS, LockFile, default_lock_path

__all__ = [
    "FrozenParams",
    "Delta",
    "IDENTITY_PARAMS",
    "CLEAN_TRANSMISSION_STIFFNESS",
    "CLEAN_SHADOW_HALF_BAND",
    "clean_compliance_ratio",
    "identity_values",
    "DEFAULT_TARGET_JOINT_NUMBER",
    "SURROGATE_SWAP_JOINT_NUMBER",
    "single_joint_scale",
    "all_joint_scale",
    "default_joint_scale",
]

# 9.2.3: the clean twin of a series-elastic joint carries the same extra joint
# with the stiffness set to 10^6 N m/rad, which deflects by less than 10^-5 rad
# under rated load.
CLEAN_TRANSMISSION_STIFFNESS = 1.0e6

# 9.3.4 / 9.2.4: the clean twin of D4hi carries the same shadow joint with
# range="-1e-5 1e-5". The dose parameter `backlash` is the total band b, so the
# identity band is twice the half range.
CLEAN_SHADOW_HALF_BAND = 1.0e-5

# D7 has no dose axis in the 9.2.10 table (both parameter columns are "—"), so
# `link_compliance` is defined as the compliance of the first bending mode as a
# fraction of the compliance that puts that mode on the control loop's Nyquist
# frequency (`shadow_joint.LINK_FLEX_REFERENCE_OMEGA`). Unlike D3a's 10^-6, the
# rigid limit cannot be taken arbitrarily far: MuJoCo integrates joint stiffness
# explicitly, so the clean twin's mode is capped by the 2 ms step. Where that cap
# falls is an empirical property of the integrator, so it is measured rather than
# written here -- `scripts/measure_constants.py --what d7_stability` sweeps the
# ratio to the divergence boundary and records it.
_DECL_D7_CLEAN = REQUIRED_MEASUREMENTS["defect.d7_clean_compliance"]


@lru_cache(maxsize=8)
def _read_d7_clean(path: Path, mtime_ns: int) -> float:
    return float(LockFile(path).get(_DECL_D7_CLEAN))


def clean_compliance_ratio(*, lock: LockFile | None = None) -> float:
    """D7's delta = 0 compliance ratio, from the lock file.

    Memoised on the lock's mtime rather than read once at import: `zero()` and
    `joint_values` run on every reset, and re-parsing the json each time would
    put a file read inside the injection path.
    """
    if lock is not None:
        return float(lock.get(_DECL_D7_CLEAN))
    path = default_lock_path()
    if not path.exists():
        return float(LockFile(path).get(_DECL_D7_CLEAN))  # raises MeasurementMissingError
    return _read_d7_clean(path, path.stat().st_mtime_ns)

# 9.2.11 fixes the surrogate delta_1 construction as "joint 3 -> joint 5", which
# also fixes which joint the single-joint defects act on by default. These are
# MJCF joint numbers (joint1 .. jointN), i.e. one-based.
DEFAULT_TARGET_JOINT_NUMBER = 3
SURROGATE_SWAP_JOINT_NUMBER = 5


# The delta = 0 value of every dose-carrying parameter, per defect. A parameter
# absent from a defect's entry is structural: `zero()` preserves it.
# A value of None here is not "no identity": it is an identity the proposal does
# not fix, which the lock file supplies. The distinction matters because
# `joint_values` reads a *missing key* as "structural, broadcast unchanged", and
# silently treating an unmeasured identity that way would scale D7's dose from
# the wrong origin.
_IDENTITY: dict[str, dict[str, float | None]] = {
    "D1": {"delta_q": 0.0},
    "D2": {"n_control_steps": 0.0},
    "D3a": {"k_transmission": CLEAN_TRANSMISSION_STIFFNESS},
    "D3b": {"sigma_g": 0.0},
    "D4hi": {"backlash": 2.0 * CLEAN_SHADOW_HALF_BAND},
    "D4lo": {"backlash": 0.0},
    "D5": {"w_over_tau_max": 0.0},
    "D6": {"delta_f": 0.0},
    "D7": {"link_compliance": None},
    "D8": {"gain": 1.0},
    "D9": {"amplitude": 0.0},
    "D10": {"delta_q_inf": 0.0},
}

# Which parameters each defect's identity covers. Static, so that constructing a
# Delta and validating its parameter names never needs the lock file -- the
# calibration sweep that measures D7's identity has to build D7 deltas first.
IDENTITY_PARAMS: dict[str, frozenset[str]] = {
    code: frozenset(vals) for code, vals in _IDENTITY.items()
}

_MEASURED_IDENTITY = {
    ("D7", "link_compliance"): clean_compliance_ratio,
}


def identity_values(defect_code: str, *, lock: LockFile | None = None) -> dict[str, float]:
    """The delta = 0 value of every dose-carrying parameter of this defect."""
    out: dict[str, float] = {}
    for name, value in _IDENTITY[defect_code].items():
        if value is None:
            out[name] = _MEASURED_IDENTITY[(defect_code, name)](lock=lock)
        else:
            out[name] = value
    return out


if set(_IDENTITY) != set(DEFECTS):
    raise AssertionError(
        "every catalogue entry needs an identity parameter set, or its clean twin "
        f"cannot be built: missing {sorted(set(DEFECTS) - set(_IDENTITY))}, "
        f"unknown {sorted(set(_IDENTITY) - set(DEFECTS))}"
    )

_unresolvable = [
    f"{code}.{name}"
    for code, vals in _IDENTITY.items()
    for name, value in vals.items()
    if value is None and (code, name) not in _MEASURED_IDENTITY
]
if _unresolvable:
    raise AssertionError(
        f"{_unresolvable} have no identity value and no lock reader; their clean "
        "twin would silently be built at the defective dose"
    )


class FrozenParams(Mapping[str, float]):
    """An immutable, hashable, order-independent float mapping.

    `Delta` has to be hashable (it keys the twin-pair registry and the surrogate
    cache) and JSON-serialisable, which a plain dict field would break.
    """

    __slots__ = ("_items", "_hash")

    def __init__(self, items: Mapping[str, float] | tuple[tuple[str, float], ...]) -> None:
        pairs = tuple(sorted((str(k), float(v)) for k, v in dict(items).items()))
        for name, value in pairs:
            if not math.isfinite(value):
                raise ValueError(f"parameter {name!r} is {value}, which is not a finite dose")
        object.__setattr__(self, "_items", pairs)
        object.__setattr__(self, "_hash", hash(pairs))

    def __getitem__(self, key: str) -> float:
        for name, value in self._items:
            if name == key:
                return value
        raise KeyError(f"{key!r} is not an injection parameter here; have {sorted(self)}")

    def __iter__(self) -> Iterator[str]:
        return (name for name, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v!r}" for k, v in self._items)
        return f"FrozenParams({body})"

    def to_json(self) -> dict[str, float]:
        return {k: v for k, v in self._items}


@dataclass(frozen=True)
class Delta:
    """One defect instance: what `inject.apply` writes and what `fields` reads.

    `joint_scale` is per *arm* joint, in the order the family's MJCF declares
    them. Its length is checked against the compiled model at injection time
    rather than here, because a `Delta` is family-independent by design: the same
    dose has to be expressible against a 5, 6 or 7 DoF arm (9.3.1).
    """

    defect_code: str
    level: int
    params: FrozenParams
    joint_scale: tuple[float, ...]

    def __post_init__(self) -> None:
        spec = defect(self.defect_code)
        declared = spec.level(self.level)
        if not isinstance(self.params, FrozenParams):
            object.__setattr__(self, "params", FrozenParams(self.params))
        expected = set(declared.params) | IDENTITY_PARAMS[self.defect_code]
        got = set(self.params)
        if got != expected:
            raise ValueError(
                f"{self.defect_code} L{self.level} takes parameters {sorted(expected)}, "
                f"got {sorted(got)}"
            )
        if not self.joint_scale:
            raise ValueError(f"{self.defect_code}: a defect acting on no joint is not a defect")
        scale = tuple(float(s) for s in self.joint_scale)
        for s in scale:
            if not math.isfinite(s):
                raise ValueError(f"{self.defect_code}: non-finite joint scale {scale}")
        object.__setattr__(self, "joint_scale", scale)

    # -- construction ----------------------------------------------------

    @classmethod
    def from_level(
        cls,
        defect_code: str,
        level: int,
        *,
        joint_scale: tuple[float, ...] | np.ndarray,
    ) -> Delta:
        """Build from the 9.2.10 table. The level's parameters are copied verbatim."""
        spec = defect(defect_code)
        return cls(
            defect_code=defect_code,
            level=level,
            params=FrozenParams(spec.level(level).params),
            joint_scale=tuple(float(s) for s in joint_scale),
        )

    @classmethod
    def from_json(cls, blob: Mapping[str, Any]) -> Delta:
        return cls(
            defect_code=str(blob["defect_code"]),
            level=int(blob["level"]),
            params=FrozenParams(blob["params"]),
            joint_scale=tuple(float(s) for s in blob["joint_scale"]),
        )

    # -- the clean twin --------------------------------------------------

    def zero(self) -> Delta:
        """The identity of this defect's operator, with its structure preserved.

        This is the clean twin (CONTRACT 3.3). It keeps `defect_code`, `level`
        and `joint_scale` so that the twin pair runs through exactly the same
        branch of `inject.apply` and, for the graph-changing defects, compiles to
        the same node set.
        """
        identity = identity_values(self.defect_code)
        return replace(
            self,
            params=FrozenParams({k: identity.get(k, v) for k, v in self.params.items()}),
        )

    @property
    def is_zero(self) -> bool:
        identity = identity_values(self.defect_code)
        return all(self.params[k] == v for k, v in identity.items())

    def scaled(self, factor: float) -> Delta:
        """Interpolate the dose towards the identity: factor 0 is clean, 1 is self.

        The second-order verification hook of `fields` uses this to read off the
        O(delta^2) content of a closed form by comparing c(2 eps) with 2 c(eps).
        """
        if not math.isfinite(factor):
            raise ValueError(f"dose factor {factor} is not finite")
        identity = identity_values(self.defect_code)
        return replace(
            self,
            params=FrozenParams(
                {
                    k: (identity[k] + factor * (v - identity[k])) if k in identity else v
                    for k, v in self.params.items()
                }
            ),
        )

    # -- reading ---------------------------------------------------------

    @property
    def spec(self) -> DefectSpec:
        return defect(self.defect_code)

    @property
    def n_joints(self) -> int:
        return len(self.joint_scale)

    def joint_values(self, name: str) -> np.ndarray:
        """The per-joint effective value of parameter `name`.

        `identity + scale_j * (value - identity)`. Structural parameters have no
        identity and are broadcast unchanged, since scaling a pole count or a
        thermal time constant per joint is not a dose.

        A uniform scale of 1 leaves the identity out of the answer exactly, so it
        is not looked up. That is arithmetic, not a fallback: the identity cancels
        for every parameter, and elising the subtract-then-add is if anything more
        accurate than performing it. It also breaks a circularity -- D7's identity
        is measured by sweeping D7 itself, and the sweep runs at uniform scale.
        """
        value = self.params[name]
        scale = np.asarray(self.joint_scale, dtype=np.float64)
        if np.all(scale == 1.0):
            return np.full(scale.shape, value, dtype=np.float64)
        identity = identity_values(self.defect_code).get(name)
        if identity is None:
            return np.full(scale.shape, value, dtype=np.float64)
        return identity + scale * (value - identity)

    @property
    def magnitude(self) -> float:
        """L2 distance of the dose-carrying parameters from the identity.

        The dose axes of E1/E3 order levels by this. It is not a physical unit:
        9.2.10 requires the per-family observable for that, and it comes from the
        lock file.
        """
        identity = identity_values(self.defect_code)
        return float(
            np.sqrt(sum((self.params[k] - v) ** 2 for k, v in identity.items()))
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "defect_code": self.defect_code,
            "level": self.level,
            "params": self.params.to_json(),
            "joint_scale": list(self.joint_scale),
        }

    def dumps(self) -> str:
        return json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))

    def __str__(self) -> str:
        body = ",".join(f"{k}={v:g}" for k, v in sorted(self.params.items()))
        active = ",".join(str(i) for i, s in enumerate(self.joint_scale) if s != 0.0)
        return f"{self.defect_code}/L{self.level}[{body}]@j{{{active}}}"


def single_joint_scale(n_joints: int, joint_number: int) -> tuple[float, ...]:
    """One-hot scale for the single-joint doses of 9.2.1 / 9.2.4.

    `joint_number` is one-based, matching the MJCF names joint1 .. jointN.
    """
    if not 1 <= joint_number <= n_joints:
        raise ValueError(
            f"joint{joint_number} does not exist on a {n_joints} DoF arm; "
            "MJCF joint numbering is one-based"
        )
    return tuple(1.0 if i == joint_number - 1 else 0.0 for i in range(n_joints))


def all_joint_scale(n_joints: int) -> tuple[float, ...]:
    """Uniform scale, for the defects 9.2.0 states act on every arm joint."""
    if n_joints < 1:
        raise ValueError(f"an arm with {n_joints} joints cannot carry a defect")
    return (1.0,) * n_joints


def default_joint_scale(defect_code: str, n_joints: int) -> tuple[float, ...]:
    """The preregistered acting-joint pattern for a production defect cell.

    D1 and both D4 calibers are the single-joint interventions of 9.2.1 and
    9.2.4. Section 9.2.11 fixes joint 3 as their primary location and joint 5 as
    the acting-joint surrogate. The other catalogue entries act on every arm
    joint unless an ablation explicitly supplies a different scale.
    """
    defect(defect_code)
    if defect_code in {"D1", "D4hi", "D4lo"}:
        return single_joint_scale(n_joints, DEFAULT_TARGET_JOINT_NUMBER)
    return all_joint_scale(n_joints)
