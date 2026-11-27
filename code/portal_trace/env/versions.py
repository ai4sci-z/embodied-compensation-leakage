"""Version and backend checks.

Proposal 11.3 pins MuJoCo, robosuite and robomimic and requires the hashes to
travel with every result; 9.1.2 forbids mixing backends within a quantity that
reaches an endpoint. Both are enforced here rather than documented: a mismatch
raises, and the backend that produced a number is recorded next to it instead of
being inferred later.
"""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass, field
from enum import Enum

from portal_trace import PINNED_VERSIONS

__all__ = [
    "Backend",
    "VersionMismatch",
    "BackendUnavailable",
    "EnvironmentFingerprint",
    "installed_versions",
    "check_pinned_versions",
    "fingerprint",
]


class Backend(str, Enum):
    """Physics backends. 9.1.2 makes CPU MuJoCo the primary caliber.

    MJX is degradation branch 0 and Isaac Lab is the third-party cross-check.
    All three are implemented; none of them is a fallback for another. A backend
    that cannot be probed raises, because silently switching would mix calibers
    inside one endpoint.
    """

    CPU_MUJOCO = "cpu_mujoco"
    MJX = "mjx"
    ISAAC_LAB = "isaac_lab"


class VersionMismatch(RuntimeError):
    """An installed package differs from the pinned version."""


class BackendUnavailable(RuntimeError):
    """A backend was requested and is not usable here.

    Deliberately fatal. 9.1.2 forbids feeding numbers from two backends into one
    endpoint, so falling back to a different backend would corrupt the result in
    a way that is invisible downstream.
    """


def installed_versions() -> dict[str, str | None]:
    """Report the versions actually importable, `None` where absent."""
    out: dict[str, str | None] = {}
    for name in PINNED_VERSIONS:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            out[name] = None
            continue
        out[name] = getattr(mod, "__version__", None) or "unknown"
    return out


def check_pinned_versions(*, required: tuple[str, ...] | None = None) -> None:
    """Raise unless the named packages match `PINNED_VERSIONS` exactly.

    `required` selects which pins matter for the caller; a script that only
    touches MuJoCo should not be blocked by a missing robomimic. Callers that
    produce numbers for a preregistered endpoint pass all three.
    """
    names = required if required is not None else tuple(PINNED_VERSIONS)
    found = installed_versions()
    problems = []
    for name in names:
        want = PINNED_VERSIONS[name]
        have = found.get(name)
        if have is None:
            problems.append(f"  {name}: not installed, pinned at {want}")
        elif have != want:
            problems.append(f"  {name}: {have} installed, pinned at {want}")
    if problems:
        raise VersionMismatch(
            "installed versions disagree with the pins in proposal 9.1.1:\n"
            + "\n".join(problems)
            + "\nThe pins exist because bit-exact reproduction is a stated "
            "requirement; loosening them invalidates the reproducibility claim "
            "rather than fixing anything."
        )


@dataclass(frozen=True)
class EnvironmentFingerprint:
    """What has to travel with every recorded number (11.3)."""

    backend: Backend
    versions: dict[str, str | None]
    python: str
    platform: str
    lite_physics: bool
    extra: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "backend": self.backend.value,
            "versions": dict(self.versions),
            "python": self.python,
            "platform": self.platform,
            "lite_physics": self.lite_physics,
            **({"extra": dict(self.extra)} if self.extra else {}),
        }


def fingerprint(backend: Backend, *, lite_physics: bool = True) -> EnvironmentFingerprint:
    """Snapshot the environment.

    `lite_physics` defaults to True to match 9.1.5: injection has to land
    between `step1` and `step2`, and the non-lite branch runs a full forward
    twice per substep. When it is switched off, every pure-physics CPU line in
    the budget doubles, and 9.1.5 forbids absorbing that factor into the
    contingency margin -- so the flag is recorded, not assumed.
    """
    return EnvironmentFingerprint(
        backend=backend,
        versions=installed_versions(),
        python=sys.version.split()[0],
        platform=platform.platform(),
        lite_physics=lite_physics,
    )
