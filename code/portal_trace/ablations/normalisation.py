"""A3: the action-normalisation caliber, which is confounder X7's own ablation.

9.6.2 fixes one global caliber: the normalisation statistics are computed once
over the whole pool and belong to the data pipeline, never to the policy --
`policies.interface` says so in as many words, and `policies.obs.NormKind` is
the proprioception side of the same rule. A3 asks what happens when that scope
moves, because X7 is the possibility that the normalisation *absorbs* the
defect: statistics fitted per embodiment subtract exactly the per-embodiment
mean offset a constant compensation field produces, so a leakage number
computed under a per-embodiment caliber would be smaller for a reason that has
nothing to do with what the learner did. 10.6.1 marks the result to be reported
as a finding rather than as a correction, which is why the main caliber stays
global whatever A3 returns.

What a caliber is, concretely: the grouping key of the statistics. Under the
global caliber there is one (mean, scale) pair for the whole pool; under the
per-embodiment caliber there is one per (family, defect); under the per-task
caliber one per task; under no normalisation the transform is the identity and
the statistics do not exist at all. This module is that mapping and its two
consequences -- how many statistics blocks a pool needs, and which defects
become unidentifiable from the action labels alone -- and nothing else. It does
not compute statistics: those are measurements over collected demonstrations
and belong to the data pipeline that owns them.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["ActionNormCaliber", "grouping_keys", "absorbs_constant_offset"]


class ActionNormCaliber(str, Enum):
    """The four levels of the A3 row. The value is what the config writes."""

    GLOBAL = "global"                  # 9.6.2, the main caliber
    PER_EMBODIMENT = "per_embodiment"
    PER_TASK = "per_task"
    NONE = "none"


# The scope tuple of each caliber, in the same shape the seed table uses for its
# three layers: a caliber is the list of fields the statistics are keyed by, and
# the empty tuple is the global one rather than a missing entry.
_SCOPES: dict[ActionNormCaliber, tuple[str, ...]] = {
    ActionNormCaliber.GLOBAL: (),
    ActionNormCaliber.PER_EMBODIMENT: ("family", "defect"),
    ActionNormCaliber.PER_TASK: ("task",),
    ActionNormCaliber.NONE: (),
}


def grouping_keys(caliber: ActionNormCaliber) -> tuple[str, ...]:
    """The fields the statistics are keyed by under this caliber.

    `NONE` and `GLOBAL` both return the empty tuple and are still different
    levels: one has a single pair of statistics and the other has none. Callers
    that need to tell them apart ask the caliber, not the key list, which is why
    this function is not the only interface to the enum.
    """
    return _SCOPES[caliber]


def absorbs_constant_offset(caliber: ActionNormCaliber) -> bool:
    """Whether this caliber removes a per-embodiment constant from the labels.

    This is X7 stated as a predicate. A caliber keyed by the embodiment
    subtracts the mean of that embodiment's actions, and a compensation field
    that is constant over the trajectory is exactly such a mean; the resulting
    labels then carry no trace of it, and every leakage estimator downstream is
    reading a quantity the pipeline already removed. The per-task caliber does
    not: its groups pool all embodiments, so a per-embodiment offset survives as
    a within-group difference.
    """
    return "family" in _SCOPES[caliber]
