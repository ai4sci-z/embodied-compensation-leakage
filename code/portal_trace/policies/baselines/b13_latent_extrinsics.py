"""B13: "learn the embodiment representation from the data".

**Proposal item.** Baseline B13 of 10.5, specified in 10.5.6. `K_hist` frames of
proprioception go into a low-dimensional continuous latent, the latent conditions the
action head, and **no embodiment label is given**. Two latent widths x `S6` x two seeds,
24 runs.

**What it controls for.** The rung that was missing from the baseline table, and 2.1
calls it the strongest gap in it. On the axis of how the embodiment is represented, B6 is
discrete-and-labelled and B4 is continuous-but-morphological; B13 is the missing
middle -- continuous, unlabelled, and with a coordinate on the defect axis, because that
is precisely what a latent inferred from execution history is designed to capture
(friction, gain, delay, load). It is also the correct low end of E9's isolation axis: a
shared trunk that is nonetheless conditioned on a hidden embodiment quantity, without
which the sweep's low end is only "fully shared dense".

10.4.8 requires it to be run at its strongest configuration -- `K_hist` reported jointly
with A2's history-window ablation, the latent width swept over two levels, and the
conditioning entering at the same place as B6's token -- and it stands *alongside* B6 as
the "strongest conditioning arm" candidate in T9, so `epsilon_cond` may be taken from
here.

**The moat, stated by this project rather than left to a reviewer (10.5.6).** Latent
extrinsics still cannot fix the common mode. The latent can only express what *differs*
between embodiments; a bias shared by every source is a constant on any such latent and
therefore has nowhere to be absorbed. That single structural argument is what makes T9
immune to this whole family, and it is why E7 rather than B13 is the crown.

**Provenance.** Reimplemented from the literature -- RMA-style latent extrinsics, online
system identification and context/adaptation-module conditioning are a family of methods
with no single canonical implementation, and J-PARC's regime-inference half belongs to the
same family. The encoder is `conditioning.LatentExtrinsicsEncoder`, an RMA-style temporal
convolution over the recent execution history, and the conditioning path is the decoder's
FiLM layers, shared with B6 by requirement.
"""

from __future__ import annotations

import torch

from portal_trace.policies.conditioning import LATENT_DIMS, LatentExtrinsicsEncoder
from portal_trace.policies.interface import (
    Batch,
    ParameterBlock,
    PolicyConfig,
    dense_blocks,
)
from portal_trace.policies.pools import PoolSpec, SubsetName, pool
from portal_trace.policies.portal_policy import PortalTracePolicy

__all__ = [
    "POOL",
    "SUBSET",
    "SEEDS",
    "LATENT_DIMS",
    "COMMON_MODE_MOAT",
    "B13LatentExtrinsicsPolicy",
]

POOL: PoolSpec = pool("B2-like")
SUBSET = SubsetName.S6
SEEDS = 2

COMMON_MODE_MOAT = (
    "A latent inferred from proprioception can only represent what differs between "
    "embodiments. A bias that every source embodiment shares is a constant function of "
    "any such latent, so the conditioning channel cannot absorb it. B13 therefore lowers "
    "the contrastive component and leaves the common mode where it is (10.5.6)."
)


class B13LatentExtrinsicsPolicy(PortalTracePolicy):
    """The 10.4.0 architecture conditioned on a latent inferred from the history.

    No embodiment label enters anywhere: `Observation.embodiment_index` is not read by
    this arm, which is the property that distinguishes it from B6 and the reason it is
    the correct control for "soft conditioning" in E9.
    """

    arm = "B13-latent-extrinsics"

    def __init__(
        self,
        cfg: PolicyConfig,
        *,
        latent_dim: int = LATENT_DIMS[0],
        use_action_history: bool = True,
    ) -> None:
        if latent_dim not in LATENT_DIMS:
            raise ValueError(
                f"latent_dim={latent_dim} is not one of the two swept levels "
                f"{LATENT_DIMS} (10.5.6); a third level would not be comparable with "
                "the reported pair"
            )
        if not cfg.obs.use_proprio:
            raise ValueError(
                "the latent is inferred from proprioception; with no proprioception "
                "pathway this arm has nothing to infer from"
            )
        super().__init__(cfg, cond_dim=latent_dim)
        self.latent_encoder = LatentExtrinsicsEncoder(
            cfg.obs.k_hist, latent_dim, use_action_history=use_action_history
        )

    def conditioning(self, batch: Batch) -> torch.Tensor:
        proprio = batch.obs.proprio
        if proprio is None:
            raise ValueError(
                "B13 infers its latent from the proprioception history; "
                "Observation.proprio is None"
            )
        return self.latent_encoder(proprio, batch.extras.get("action_history"))

    def parameter_blocks(self) -> tuple[ParameterBlock, ...]:
        return dense_blocks(self, self.cfg.n_pool_embodiments)
