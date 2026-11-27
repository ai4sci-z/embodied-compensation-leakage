"""B7: the Oracle.

**Proposal item.** Baseline B7 of 10.5, with its implementation discipline in 10.5.5.
Trained on the **defect-free twin of this defective embodiment**, on the same reference
trajectories, with the same number of demonstrations. Subset `S36`, three seeds, 108
runs -- one of the two full-36-config arms, and 540 GPU-h that 10.5.5 marks
incompressible.

**What it controls for.** It is the reference point of the primary endpoint. Matched to
the pooled arm on data volume, reference trajectories, teacher operator and training
budget, differing **only** in the defect, so the data-volume-induced shrinkage
difference cancels in the paired difference. It carries three roles at once: the
reference caliber for `Lambda^orc`, the upper anchor of the paired success-rate
difference and of `Leak_beh`, and E1's control arm.

**The limit that must travel with every B7 number.** It fixes data volume; it does not
fix the shrinkage target. The pooled arm shrinks towards `mu_bar + c_bar`, the Oracle
towards `mu_bar`, and the difference is exactly along `c_bar`. The two arms' underfitting
is therefore not isomorphic, and `Lambda^orc` carries a systematic positive bias that
**layer 2's shrinkage-field partial regression** has to absorb (6.4). The pseudo-defect
null distribution cannot absorb it, because a pseudo field appears in no dataset mean.
`BIAS_NOTE` is that sentence, attached to the arm so a results table can quote it.

**What B7 is not.** It is not a transfer target. A-1 in the appendix rejects any harm
definition that needs a `delta = 0` deployment target, because no real robot can be
declared defect-free. B7 is a reference arm and nothing else.

**Provenance.** Built on the installed `robomimic` 0.5.0 through the shared 10.4.0
architecture. Identical network to B2 by construction -- `check_matched` enforces that
together with the other three matched items of 10.5.5, because an Oracle that differs
architecturally is not an Oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

from portal_trace.policies.interface import PolicyConfig
from portal_trace.policies.pools import PoolSpec, SubsetName, pool
from portal_trace.policies.portal_policy import PortalTracePolicy
from portal_trace.policies.schedule import OptimSpec

__all__ = [
    "POOL",
    "SUBSET",
    "SEEDS",
    "BIAS_NOTE",
    "MatchedItem",
    "MATCHED_ITEMS",
    "check_matched",
    "B7OraclePolicy",
]

POOL: PoolSpec = pool("B7")
SUBSET = SubsetName.S36
SEEDS = 3

BIAS_NOTE = (
    "Lambda^orc carries a systematic positive bias of +lambda * c_bar: the pooled arm's "
    "shrinkage target is mu_bar + c_bar and the Oracle's is mu_bar, so their shrinkage "
    "components differ exactly along c_bar, which is co-directional with c_delta. The "
    "term is absorbed by the layer-2 shrinkage-field partial regression (6.4), not by "
    "B7 and not by the pseudo-defect null distribution."
)


@dataclass(frozen=True)
class MatchedItem:
    """One row of 10.5.5's matching table, with the consequence of not matching it."""

    name: str
    how: str
    if_unmatched: str
    in_degradation_ladder: bool


MATCHED_ITEMS: tuple[MatchedItem, ...] = (
    MatchedItem(
        name="same reference trajectories",
        how=(
            "the embodiment-independent task-space reference set X_tau is reused "
            "demonstration by demonstration, including the initial-state and goal-pose "
            "seeds, which `seed_env` guarantees by not containing defect_id"
        ),
        if_unmatched="different state distribution, so X8 enters the difference",
        in_degradation_ladder=True,
    ),
    MatchedItem(
        name="same demonstration count",
        how="1 400 per (config, core task), matched one for one with the pooled arm",
        if_unmatched=(
            "shrinkage contamination along +(mu_tau - a_bar), a direction the "
            "pseudo-defect null does not match (X2)"
        ),
        # 10.5.5 is explicit: this row does not enter the degradation ladder.
        in_degradation_ladder=False,
    ),
    MatchedItem(
        name="same teacher operator",
        how="same ILC configuration, same k_max, same ilc_contact_variant, same tracker gains",
        if_unmatched="measures a teacher difference rather than a defect difference (X3)",
        in_degradation_ladder=True,
    ),
    MatchedItem(
        name="same training budget",
        how="same architecture, parameter count, epochs, batch size and augmentation",
        if_unmatched="different underfitting levels, so the upper anchor drifts",
        in_degradation_ladder=True,
    ),
)


def check_matched(
    oracle: PolicyConfig,
    pooled: PolicyConfig,
    *,
    oracle_optim: OptimSpec,
    pooled_optim: OptimSpec,
) -> None:
    """Enforce the two matched items that are visible from inside this package.

    Architecture and training budget are checkable here; reference trajectories and the
    teacher operator are properties of the collection pipeline and are asserted there.
    `PolicyConfig` compares by value and excludes `options`, so this is a genuine
    architectural comparison and not an identity test.
    """
    if oracle.evolve(arm=pooled.arm) != pooled:
        raise ValueError(
            "B7 and the pooled arm must be the same architecture (10.5.5, 'same "
            "training budget'): an Oracle with a different network puts an "
            "architectural difference inside the primary endpoint's paired difference"
        )
    if oracle_optim != pooled_optim:
        raise ValueError(
            "B7 and the pooled arm must share the optimisation budget: same epochs, "
            "batch size and augmentation (10.5.5). Different budgets move the upper "
            "anchor of Leak_beh"
        )
    if POOL.total_demos != pool("B2").total_demos:
        raise AssertionError(
            f"B7 holds {POOL.total_demos} demonstrations and the pooled arm "
            f"{pool('B2').total_demos}; 10.5.5 requires 1 400 on both sides and marks "
            "this row as not part of the degradation ladder"
        )


class B7OraclePolicy(PortalTracePolicy):
    """B2's network on the clean twin's data. The difference is the pool, only."""

    arm = "B7-oracle"
