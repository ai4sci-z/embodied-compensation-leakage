"""The isolation measure I and the measured parameter counts per level.

Implements proposal 10.4.9 and the model-card fields of 11.4. The measure is

    I = 1 - sum_p n_p [ C(K,2)^{-1} sum_{k<k'} min(s_{k,p}, s_{k',p}) ]
            / sum_p n_p s_bar_p ,      s_bar_p = (1/K) sum_k s_{k,p}

where s_{k,p} is the **expected share of routing weight** that embodiment k
puts on parameter block p (1 for a dense block, 0/1 for hard routing, the
expected softmax weight for soft routing) and n_p is the block's parameter
count. Fully shared dense gives I = 0, fully disjoint gives I = 1.

**There is no threshold anywhere in this module.** 10.4.9 and 11.5 are explicit
that defining "are these parameters shared" by thresholding a soft routing
weight makes the regressor of E9's main regression irreproducible. I is
continuous and computed from the routing weights as they are.

Two further requirements from 10.4.9 and appendix A-25 are enforced here rather
than left to the results table:

* Capacity is matched on the **active** parameter count only, to +-10%. Total
  and active cannot both be matched -- a dense level has active = total, a
  K-way disjoint level has one or the other but not both -- and log(total) goes
  into the regression as a covariate.
* An isolation level has to be reported next to the decisive fragmentation
  control I3' (K disjoint modules split by a random pooled subset, identical
  parameter count and per-module data volume). Without it, a slope in I is not
  attributable to embodiment identity.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "ACTIVE_PARAM_MATCH_TOLERANCE",
    "ParameterBlock",
    "IsolationLevel",
    "IsolationLadder",
    "isolation_measure",
    "routing_entropy_complement",
]

# 10.4.9: the preregistered capacity-matching tolerance on active parameters.
ACTIVE_PARAM_MATCH_TOLERANCE = 0.10


@dataclass(frozen=True)
class ParameterBlock:
    """One expert / head / layer, with each embodiment's expected routing share."""

    name: str
    n_params: int
    routing_share: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.n_params <= 0:
            raise ValueError(f"block {self.name!r} has {self.n_params} parameters")
        if not self.routing_share:
            raise ValueError(f"block {self.name!r} has no routing shares")
        for source, share in self.routing_share.items():
            if not 0.0 <= share <= 1.0:
                raise ValueError(
                    f"block {self.name!r}, embodiment {source!r}: the expected share of "
                    f"routing weight lies in [0, 1], got {share}"
                )


def isolation_measure(blocks: Sequence[ParameterBlock], sources: Sequence[str]) -> float:
    """I, the parameter-disjointness measure of 10.4.9."""
    if len(sources) < 2:
        raise ValueError(
            f"I compares pairs of embodiments and needs K >= 2, got K={len(sources)}"
        )
    if len(set(sources)) != len(sources):
        raise ValueError(f"the embodiment list has duplicates: {sources}")
    if not blocks:
        raise ValueError("a model with no parameter blocks has no isolation measure")
    n_pairs = math.comb(len(sources), 2)
    shared = 0.0
    total = 0.0
    for block in blocks:
        missing = [s for s in sources if s not in block.routing_share]
        if missing:
            raise ValueError(
                f"block {block.name!r} has no routing share for {missing}; a missing "
                "share is not a zero share, and I would silently read it as disjointness"
            )
        shares = np.array([block.routing_share[s] for s in sources], dtype=np.float64)
        pairwise_min = sum(
            min(shares[i], shares[j])
            for i in range(len(sources))
            for j in range(i + 1, len(sources))
        )
        shared += block.n_params * (pairwise_min / n_pairs)
        total += block.n_params * float(shares.mean())
    if total <= 0.0:
        raise ValueError(
            "no embodiment routes any weight to any block; the model produces no action "
            "and I is undefined"
        )
    return 1.0 - shared / total


def routing_entropy_complement(
    blocks: Sequence[ParameterBlock], sources: Sequence[str]
) -> float:
    """The complement of the normalised HB-MoE routing entropy (10.4.9).

    10.4.9 makes this the **second, attached** regressor and I the main one.
    Reported separately rather than blended in: entropy is blind to block size,
    so a model that routes uniformly over one huge block and one tiny one looks
    maximally shared to the entropy and barely shared to I.

    Per embodiment the routing shares are normalised into a distribution over
    blocks and their Shannon entropy is divided by log(#blocks); the value
    returned is one minus the mean over embodiments, so it runs the same way as
    I (0 for fully shared, 1 for fully concentrated).
    """
    if len(blocks) < 2:
        raise ValueError(
            f"routing entropy over {len(blocks)} block(s) has no scale; log(#blocks) is zero"
        )
    per_source = []
    for source in sources:
        shares = np.array(
            [block.routing_share[source] for block in blocks], dtype=np.float64
        )
        total = shares.sum()
        if total <= 0.0:
            raise ValueError(f"embodiment {source!r} routes no weight to any block")
        p = shares / total
        nonzero = p[p > 0.0]
        per_source.append(-float(np.sum(nonzero * np.log(nonzero))) / math.log(len(blocks)))
    return 1.0 - float(np.mean(per_source))


@dataclass(frozen=True)
class IsolationLevel:
    """One rung of the E9 ladder, with its measured parameter counts.

    `measured_active_params` and `measured_total_params` are **measured**
    numbers from the trained model (11.4), not derived from the block table.
    `implied_active_params` recomputes them from the routing shares so the two
    can be printed side by side; a large gap means the block table does not
    describe the model that was actually trained.
    """

    label: str
    blocks: tuple[ParameterBlock, ...]
    sources: tuple[str, ...]
    measured_active_params: int
    measured_total_params: int
    is_fragmentation_control: bool = False
    wall_clock_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.measured_active_params <= 0 or self.measured_total_params <= 0:
            raise ValueError(f"level {self.label!r} has non-positive parameter counts")
        if self.measured_active_params > self.measured_total_params:
            raise ValueError(
                f"level {self.label!r} reports more active than total parameters "
                f"({self.measured_active_params} > {self.measured_total_params})"
            )
        if self.wall_clock_seconds is not None and self.wall_clock_seconds <= 0.0:
            raise ValueError(
                f"level {self.label!r} reports a wall clock of {self.wall_clock_seconds} s"
            )

    @property
    def isolation(self) -> float:
        return isolation_measure(self.blocks, self.sources)

    @property
    def routing_entropy_complement(self) -> float | None:
        """10.4.9's attached second regressor; undefined for a single block."""
        if len(self.blocks) < 2:
            return None
        return routing_entropy_complement(self.blocks, self.sources)

    @property
    def implied_active_params(self) -> float:
        """sum_p n_p s_bar_p: the active count the routing table implies."""
        return float(
            sum(
                block.n_params
                * float(np.mean([block.routing_share[s] for s in self.sources]))
                for block in self.blocks
            )
        )

    @property
    def log_total_params(self) -> float:
        """log(total parameters), the covariate 10.4.9 puts in the main regression."""
        return math.log(self.measured_total_params)

    def as_row(self) -> dict[str, Any]:
        return {
            "level": self.label,
            "isolation_I": self.isolation,
            "routing_entropy_complement": self.routing_entropy_complement,
            "active_params_measured": self.measured_active_params,
            "active_params_implied": self.implied_active_params,
            "total_params_measured": self.measured_total_params,
            "log_total_params": self.log_total_params,
            "wall_clock_seconds": self.wall_clock_seconds,
            "fragmentation_control": self.is_fragmentation_control,
        }


class IsolationLadder:
    """The E9 ladder, with the two structural checks of 10.4.9 and A-25."""

    def __init__(self, levels: Sequence[IsolationLevel]) -> None:
        if len(levels) < 2:
            raise ValueError(f"a ladder needs at least two levels, got {len(levels)}")
        labels = [level.label for level in levels]
        if len(set(labels)) != len(labels):
            raise ValueError(f"level labels must be unique, got {labels}")
        source_sets = {level.sources for level in levels}
        if len(source_sets) != 1:
            raise ValueError(
                "the ladder's levels disagree on the source embodiment set; I is a "
                "comparison across the same K embodiments"
            )
        self._levels = tuple(levels)

    @property
    def levels(self) -> tuple[IsolationLevel, ...]:
        return self._levels

    def activation_matching_report(self) -> dict[str, Any]:
        """Capacity matching, checked on active parameters only (10.4.9)."""
        active = np.array([lv.measured_active_params for lv in self._levels], dtype=np.float64)
        reference = float(active.mean())
        deviations = {
            lv.label: float(lv.measured_active_params / reference - 1.0)
            for lv in self._levels
        }
        worst = max(abs(v) for v in deviations.values())
        return {
            "reference_active_params": reference,
            "relative_deviation": deviations,
            "worst_relative_deviation": worst,
            "tolerance": ACTIVE_PARAM_MATCH_TOLERANCE,
            "matched": worst <= ACTIVE_PARAM_MATCH_TOLERANCE,
            "note": (
                "total and active parameter counts cannot both be matched (appendix "
                "A-25); active is the preregistered caliber and log(total) is a covariate"
            ),
        }

    def require_fragmentation_control(self) -> IsolationLevel:
        """A-25: the ladder is uninterpretable without the I3' control."""
        controls = [lv for lv in self._levels if lv.is_fragmentation_control]
        if not controls:
            raise ValueError(
                "no level is marked as the fragmentation control. Every isolation level "
                "splits the data K ways as well, so a slope in I can be explained by "
                "per-module data volume alone; I3' (random pooled-subset split, identical "
                "parameter count and per-module data) is what separates the two (10.4.9, "
                "appendix A-25)"
            )
        if len(controls) > 1:
            raise ValueError(
                f"more than one fragmentation control: {[c.label for c in controls]}"
            )
        return controls[0]

    def as_rows(self) -> list[dict[str, Any]]:
        return [lv.as_row() for lv in self._levels]

    def isolation_axis(self) -> dict[str, float]:
        """I per level, the regressor of E9's main regression."""
        return {lv.label: lv.isolation for lv in self._levels}
