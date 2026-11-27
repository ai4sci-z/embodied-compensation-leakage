"""B2: "more data is better".

**Proposal item.** Baseline B2 of 10.5, and one of the two arms that carry the primary
endpoint. Naive pooling with no embodiment conditioning of any kind: the target's 200
demonstrations plus six sources at 200 each, 1400 per (config, core task), nominal
mixing weight `alpha = 1200/1400 = 0.857`. Subset `S36`, three seeds, 108 runs.

**What it controls for.** It is the arm being evaluated, not a control: E1's endpoint
is the paired difference between the Oracle arm B7 and this one, and every leakage
reading is taken on a policy trained exactly this way. Its role in the design is to be
the thing current practice would do.

The pool contains no `delta = 0` cell anywhere. 10.4.0 spells out the consequence and
it has to be repeated wherever a B2 number is quoted: route A's anchoring does not hold
by construction, so the correct reading of a leakage value measured here is
"insufficient information", not "no leakage". E3's `||delta||` axis exists to supply the
missing anchor, and the difference between the two pools is the reading of whether the
learner uses it.

**Provenance.** Built on the installed `robomimic` 0.5.0 through the shared 10.4.0
architecture. Nothing is added to it: this arm is the architecture.
"""

from __future__ import annotations

from portal_trace.policies.pools import PoolSpec, SubsetName, pool
from portal_trace.policies.portal_policy import PortalTracePolicy

__all__ = ["POOL", "SUBSET", "SEEDS", "B2NaivePoolingPolicy"]

POOL: PoolSpec = pool("B2")
SUBSET = SubsetName.S36
SEEDS = 3


class B2NaivePoolingPolicy(PortalTracePolicy):
    """The 10.4.0 architecture on the pooled data, unconditioned."""

    arm = "B2-naive-pooling"
