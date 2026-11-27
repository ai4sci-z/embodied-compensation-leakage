"""A6: regime granularity, 3 / 5 / 8 regimes over the labels of 9.7.

Proposition 1(e')'s stratified estimate is computed per regime, and 10.6.1's A6
row asks how much it depends on how finely the state space is cut: the finer the
partition, the harder per-regime connectivity is to satisfy, because every cell
needs its own N_0 = 200 admitted frames (11.3 item 9). The row is CPU
post-processing -- it re-partitions labels that are already computed and trains
nothing -- so what it needs is a partition object, not a second labeller.

`diagnostics.regimes` produces the five labels of `types.Regime` and owns the
thresholds. This module is the map from those five onto a coarser or a finer
label set, and it refuses anything that is not a partition: a coarsening whose
groups overlap, or that drops a base regime, would silently move frames out of
the denominator of the occupancy mass and make two granularity levels
incomparable.

The three levels
----------------

*3 regimes* merges the five into transient / free / contact. The merge is stated
in the configuration, not here, because which base regimes go together is a
preregistered choice and the code must not be able to make it.

*5 regimes* is the identity and is the main caliber.

*8 regimes* refines by crossing three of the base regimes with the sign of the
tool's velocity along the task's insertion axis: approaching and departing are
different contact histories at the same label, and the sign is a property of a
frame that needs no new threshold and therefore no new measurement. Splitting
three of the five gives 5 - 3 + 6 = 8. The reversal regime r1 is not among the
three and cannot be: it is defined by a velocity zero crossing, so a split by
the sign of that same velocity would put its two halves on either side of the
event that defines it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from portal_trace.configs.schema import ConfigError
from portal_trace.types import Regime

__all__ = ["RegimePartition", "APPROACH_SIGN_LABELS"]

# The two halves a refined regime is split into, in the order the sign takes:
# index 0 is a negative along-axis velocity component.
APPROACH_SIGN_LABELS: tuple[str, str] = ("approaching", "departing")


@dataclass(frozen=True)
class RegimePartition:
    """One A6 level: a partition of the five base regimes of 9.7.

    `merge` maps a new label to the base regimes it absorbs; `refine` names the
    base regimes that are split by the approach sign. A level uses one of the
    two, never both: a partition that coarsens and refines at once is two
    interventions, and the row asks about granularity along one axis.
    """

    level_id: str
    merge: dict[str, tuple[Regime, ...]]
    refine: tuple[Regime, ...]

    def __post_init__(self) -> None:
        if self.merge and self.refine:
            raise ConfigError(
                f"{self.level_id}: a granularity level either coarsens or refines. "
                "Doing both makes the level's cell count reachable by two different "
                "routes, and the sensitivity of proposition 1(e')'s estimate could "
                "not be attributed to either"
            )
        if Regime.R1_REVERSAL in self.refine:
            raise ConfigError(
                f"{self.level_id}: r1 is the velocity zero crossing, so splitting it "
                "by the sign of that velocity puts its two halves on either side of "
                "the event that defines the regime"
            )
        if self.merge:
            seen: list[Regime] = []
            for group in self.merge.values():
                if not group:
                    raise ConfigError(f"{self.level_id}: an empty merge group")
                seen.extend(group)
            if sorted(r.value for r in seen) != sorted(r.value for r in Regime):
                raise ConfigError(
                    f"{self.level_id}: the merge groups are "
                    f"{sorted(r.value for r in seen)} and the base partition is "
                    f"{sorted(r.value for r in Regime)}. A coarsening that overlaps or "
                    "omits a base regime is not a partition, and the occupancy masses "
                    "would no longer sum to one"
                )

    @classmethod
    def from_parameters(cls, parameters: dict[str, object], *, where: str) -> RegimePartition:
        """Build from an ablation level's `parameters` block.

        Both keys are required and one of them is empty on every level, so that
        a level states which of the two operations it is not doing rather than
        leaving it to be inferred from an absent key.
        """
        missing = sorted({"merge", "refine"} - set(parameters))
        if missing:
            raise ConfigError(f"{where}: A6 level is missing {missing}")
        merge_blob = parameters["merge"] or {}
        if not isinstance(merge_blob, dict):
            raise ConfigError(f"{where}: `merge` is a map from new label to base regimes")
        refine_blob = parameters["refine"] or ()
        return cls(
            level_id=where,
            merge={
                str(label): tuple(Regime(str(r)) for r in group)
                for label, group in merge_blob.items()
            },
            refine=tuple(Regime(str(r)) for r in refine_blob),
        )

    @property
    def n_regimes(self) -> int:
        if self.merge:
            return len(self.merge)
        return len(Regime) + len(self.refine)

    def labels(
        self, base: np.ndarray, *, approach_sign: np.ndarray | None = None
    ) -> np.ndarray:
        """Re-label frames under this partition.

        `base` holds the `Regime` values of 9.7 as strings, one per frame.
        `approach_sign` is the signed along-axis velocity component and is
        required exactly when the level refines: it is refused rather than
        defaulted to zero, because a zero sign would collapse a refined level
        back onto the main one without saying so.
        """
        values = np.asarray(base, dtype=object)
        if self.merge:
            out = np.empty(values.shape, dtype=object)
            for label, group in self.merge.items():
                member = np.isin(values, [r.value for r in group])
                out[member] = label
            unlabelled = out == None  # noqa: E711 - object array, `is None` is not vectorised
            if unlabelled.any():
                raise ConfigError(
                    f"{self.level_id}: {int(unlabelled.sum())} frames carry a label "
                    f"outside {sorted(r.value for r in Regime)}"
                )
            return out
        if not self.refine:
            return values.copy()
        if approach_sign is None:
            raise ConfigError(
                f"{self.level_id}: refining by the approach sign needs the signed "
                "along-axis velocity component. It is a per-frame quantity of the "
                "rollout, and this partition does not invent one"
            )
        sign = np.asarray(approach_sign, dtype=np.float64)
        if sign.shape != values.shape:
            raise ConfigError(
                f"{self.level_id}: {values.shape} labels against {sign.shape} signs"
            )
        out = values.copy()
        half = np.where(sign < 0.0, APPROACH_SIGN_LABELS[0], APPROACH_SIGN_LABELS[1])
        for regime in self.refine:
            member = values == regime.value
            out[member] = np.char.add(f"{regime.value}_", half[member].astype(str))
        return out
