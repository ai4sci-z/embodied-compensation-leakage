"""Identifiers and array containers shared across the package.

Defined once, here, so that a cell index means the same thing in the teacher, in
the diagnostics and in the statistics. Two of these carry decisions rather than
just data:

`Regime` fixes the priority order r4 > r1 > r2 > r3 > r0. Proposition 1(e')(i)
needs R to be a partition rather than a cover -- if a frame could belong to two
regimes, the per-regime conditional means would be correlated and the block
structure of the design matrix would not hold.

`Cell` is the statistical unit. 11.3 fixes n = 36 = 6 families x 6 core defects
for the primary endpoint arms, and every secondary arm has to declare which
subset of those 36 it ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

__all__ = [
    "Regime",
    "REGIME_PRIORITY",
    "REGIME_ORDINAL",
    "REGIME_BY_ORDINAL",
    "Cell",
    "EmbodimentId",
    "Phase",
    "Episode",
    "FrameTable",
    "SeedTriple",
]


class Regime(str, Enum):
    """The five regimes of 9.7.1."""

    R0_OTHER = "r0"
    R1_REVERSAL = "r1"        # a joint is reversing direction
    R2_QUASISTATIC = "r2"     # slow, carrying gravity load
    R3_FREESPACE = "r3"       # fast, out of contact
    R4_CONTACT = "r4"         # wrist force above threshold


# Exclusivity order (9.7.1). Earlier wins.
REGIME_PRIORITY: tuple[Regime, ...] = (
    Regime.R4_CONTACT,
    Regime.R1_REVERSAL,
    Regime.R2_QUASISTATIC,
    Regime.R3_FREESPACE,
    Regime.R0_OTHER,
)

# Storage encoding, independent of the priority order so that reordering
# priorities never silently reinterprets an array on disk.
REGIME_ORDINAL: dict[Regime, int] = {r: i for i, r in enumerate(Regime)}
REGIME_BY_ORDINAL: dict[int, Regime] = {i: r for r, i in REGIME_ORDINAL.items()}


class Phase(str, Enum):
    """Trajectory phases used by the phase-split residual report (9.6.4).

    Distinct from `Regime`: a phase is a stretch of the reference trajectory,
    known before the rollout, while a regime is decided per frame from the
    realised state.
    """

    APPROACH = "approach"
    GRASP = "grasp"
    TRANSPORT = "transport"
    TERMINAL = "terminal"


@dataclass(frozen=True, order=True)
class EmbodimentId:
    """A concrete robot: a clean family plus a defect at a level.

    `defect_code=None` with `level=None` is the clean twin. It is spelled as an
    absent defect rather than as a separate class because the clean twin is
    produced by the same injection path evaluated at delta = 0.
    """

    family: str
    defect_code: str | None = None
    level: int | None = None

    def __post_init__(self) -> None:
        if (self.defect_code is None) != (self.level is None):
            raise ValueError(
                f"{self!r}: a defect and its level go together; a defect without a "
                "level is not a robot"
            )

    @property
    def is_clean(self) -> bool:
        return self.defect_code is None

    @property
    def clean_twin(self) -> EmbodimentId:
        return EmbodimentId(self.family)

    def __str__(self) -> str:
        if self.is_clean:
            return f"{self.family}/clean"
        return f"{self.family}/{self.defect_code}/L{self.level}"


@dataclass(frozen=True, order=True)
class Cell:
    """One (task, family, defect) cell -- the statistical unit of 11.3."""

    task: str
    family: str
    defect_code: str

    def __str__(self) -> str:
        return f"{self.task}|{self.family}|{self.defect_code}"


@dataclass(frozen=True)
class SeedTriple:
    """Three independent seeds (11.3).

    Cross-arm comparisons align `env` and `eval` and vary `train`, so that arm
    differences carry no initial-state difference and the training seed's
    variance lands in the third layer of the variance decomposition.
    """

    env: int
    train: int
    eval: int


@dataclass
class Episode:
    """One demonstration or rollout, at fixed length T = 400 (9.6.1).

    Early stopping is off: after success the teacher holds the terminal pose to
    the horizon, so `num_samples` is always 400 and frame-level mixing weights
    stay comparable across cells.
    """

    embodiment: EmbodimentId
    task: str
    seeds: SeedTriple
    states: np.ndarray        # (T, nq + nv) MuJoCo state, for bit-exact replay
    actions: np.ndarray       # (T, action_dim) what the wrapper received
    observations: dict[str, np.ndarray]
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        T = len(self.states)
        if len(self.actions) != T:
            raise ValueError(f"states has {T} frames, actions has {len(self.actions)}")
        for k, v in self.observations.items():
            if len(v) != T:
                raise ValueError(f"observation {k!r} has {len(v)} frames, expected {T}")

    @property
    def horizon(self) -> int:
        return len(self.states)


@dataclass
class FrameTable:
    """Frames of one cell, laid out column-wise for the streaming statistics.

    The diagnostics never materialise 45 M frames at once; they accumulate
    sufficient statistics over tables like this one, which is why the fields are
    plain arrays rather than a list of `Episode`.
    """

    cell: Cell
    episode_index: np.ndarray   # (N,) which episode each frame came from
    time_index: np.ndarray      # (N,) frame position within its episode
    regime: np.ndarray          # (N,) uint8, Regime ordinal
    actions_defective: np.ndarray   # (N, A) pi_{e,delta,tau}(x)
    actions_clean: np.ndarray       # (N, A) pi_{e,0,tau}(x), from query_clean
    phi: np.ndarray | None = None   # (N, P) state features
    extras: dict[str, np.ndarray] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.episode_index)
        for name in ("time_index", "regime", "actions_defective", "actions_clean"):
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(f"{name} has {len(arr)} rows, expected {n}")
        if self.phi is not None and len(self.phi) != n:
            raise ValueError(f"phi has {len(self.phi)} rows, expected {n}")

    @property
    def n_frames(self) -> int:
        return len(self.episode_index)

    @property
    def c_true(self) -> np.ndarray:
        """The per-frame compensation field, c_delta(x) = pi_{e,delta,tau} - pi_{e,0,tau}.

        Definition 3.1. Both sides are already clipped to the action range
        individually by `teacher.query`, which matters: clipping the difference
        instead would misreport saturated frames.
        """
        return self.actions_defective - self.actions_clean

    def mask(self, regime: Regime) -> np.ndarray:
        return self.regime == REGIME_ORDINAL[regime]
