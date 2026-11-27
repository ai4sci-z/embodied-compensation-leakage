"""B1: "don't use other people's data".

**Proposal item.** Baseline B1 of 10.5. Behaviour cloning on the target embodiment's
own 200 demonstrations per (config, core task) and nothing else. Subset `S9`, two
seeds, 18 runs.

**What it controls for.** The lower anchor of the leakage axis. Its pool contains no
other embodiment, so there is no compensation to inherit and `Lambda` goes to zero by
construction; what it pays is data volume, one seventh of the pooled arms', and its
success rate is expected to be correspondingly lower. Without it, a pooled arm's
success rate has no floor to be read against, and "pooling hurts" cannot be separated
from "pooling helps less than more data would".

**Provenance.** Built on the installed `robomimic` 0.5.0 via the shared 10.4.0
architecture (`portal_policy.PortalTracePolicy`, whose visual trunk and MLP blocks are
robomimic's). The arm differs from B2 in its pool alone, which is why it is a pool
declaration plus a subclass rather than a second network: any architectural difference
between B1 and B2 would land inside the paired difference the endpoint reads.
"""

from __future__ import annotations

from portal_trace.policies.pools import PoolSpec, SubsetName, pool
from portal_trace.policies.portal_policy import PortalTracePolicy

__all__ = ["POOL", "SUBSET", "SEEDS", "B1TargetOnlyPolicy"]

POOL: PoolSpec = pool("B1")
SUBSET = SubsetName.S9
SEEDS = 2


class B1TargetOnlyPolicy(PortalTracePolicy):
    """The 10.4.0 architecture trained on the target embodiment's own data."""

    arm = "B1-target-only"
