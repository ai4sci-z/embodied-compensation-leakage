"""B8: counterfactual decompensation.

**Proposal item.** Baseline B8 of 10.5, specified in 10.5.3. The same defective
demonstrations with the action label replaced by `a - c_true_delta`. Subset `S18`, three
seeds, 54 runs. One of the five things 10.9.4 forbids dropping when the budget is cut.

**What it controls for.** Two of the three most dangerous alternative explanations, and
it *eliminates* them rather than hardening against them, because the two arms are the
same trajectories through the same states with only the action label differing:

* X3, teacher quality -- demonstrations on a defective body might simply be worse
  (slower, shakier), so the policy would be learning "bad demonstrations" rather than
  "compensation".
* X8, state-distribution shift -- trajectories on a defective body visit different
  states, so the leakage might be covariate shift.

Hence `J(B8) - J(naive pooling)` is the causal effect of the compensation. **No
simulator ground truth, no such control**; this is the paradigm's own killer comparison
and no published work can run it.

**Implementation discipline (10.5.3).** `c_true` is evaluated on the *clean reference
policy's* actions per the 14.5 E-4 definition and differenced at the same control-step
index. The variant evaluated on the policy under test is a sensitivity analysis only
(A12) and is selected by `CTrueCaliber`, never silently. The per-cell fraction of frames
where the relabelled action saturates the clipping range is reported:
`clip_saturation_fraction` is that number, and it is a property of the label transform,
so it lives here.

**Provenance.** Implemented directly on top of the shared 10.4.0 architecture -- the
network is B2's, and only the label transform differs, which is what "paired
demonstration for demonstration" means.
"""

from __future__ import annotations

from enum import Enum

import torch

from portal_trace.policies.interface import Batch, PolicyConfig
from portal_trace.policies.pools import PoolSpec, SubsetName, pool
from portal_trace.policies.portal_policy import PortalTracePolicy

__all__ = [
    "POOL",
    "SUBSET",
    "SEEDS",
    "ACTION_LIMIT",
    "CTrueCaliber",
    "decompensated_actions",
    "clip_saturation_fraction",
    "B8CounterfactualPolicy",
]

POOL: PoolSpec = pool("B8")
SUBSET = SubsetName.S18
SEEDS = 3

# 9.6.1: every action dimension lives in [-1, 1]. Subtracting the compensation can
# leave that box, and where it does the label is clipped, which is a real loss of
# information rather than a formatting detail -- hence it is measured and reported.
ACTION_LIMIT = 1.0


class CTrueCaliber(str, Enum):
    """Which actions `c_true` was evaluated on (A12).

    `CLEAN_REFERENCE_POLICY` is the main caliber of 14.5 E-4. `POLICY_UNDER_TEST` is the
    sensitivity variant and may not be used for the primary reading. The enum exists so
    that a run records which one it used instead of the choice being implicit in a data
    path.
    """

    CLEAN_REFERENCE_POLICY = "clean_reference_policy"
    POLICY_UNDER_TEST = "policy_under_test"


def decompensated_actions(batch: Batch) -> torch.Tensor:
    """`a - c_true`, clipped back into the action box of 9.6.1."""
    actions = batch.require_actions()
    c_true = batch.require_c_true()
    if actions.shape != c_true.shape:
        raise ValueError(
            f"c_true {tuple(c_true.shape)} does not match the action chunk "
            f"{tuple(actions.shape)}; both are (B, CHUNK, 7) on the same control-step "
            "index (10.1.4)"
        )
    return (actions - c_true).clamp(-ACTION_LIMIT, ACTION_LIMIT)


def clip_saturation_fraction(batch: Batch) -> float:
    """The fraction of relabelled action entries that hit the clipping range.

    10.5.3 requires this per cell. A large value means the counterfactual label is
    partly unrepresentable in the action space, and the arm's difference from B2 is then
    an underestimate of the compensation's causal effect.
    """
    actions = batch.require_actions()
    raw = actions - batch.require_c_true()
    return float((raw.abs() > ACTION_LIMIT).to(torch.float32).mean())


class B8CounterfactualPolicy(PortalTracePolicy):
    """B2's network trained on `a - c_true`.

    The whole of B8 is the one overridden hook. Anything else that differed from B2
    would break the demonstration-for-demonstration pairing that eliminates X3 and X8.
    """

    arm = "B8-counterfactual-decompensation"

    def __init__(
        self,
        cfg: PolicyConfig,
        *,
        caliber: CTrueCaliber = CTrueCaliber.CLEAN_REFERENCE_POLICY,
    ) -> None:
        super().__init__(cfg)
        self.caliber = caliber

    def action_target(self, batch: Batch) -> torch.Tensor:
        return decompensated_actions(batch)
