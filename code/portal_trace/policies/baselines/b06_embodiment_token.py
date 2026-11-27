"""B6: "one token per embodiment is enough".

**Proposal item.** Baseline B6 of 10.5, specified in 10.5.2. One-hot and learned
embedding forms, plus an inference-time token permutation ablation. Subset `S18`, two
seeds, 36 runs. Scheduled early, with B4, because gate G5 turns on both.

**What it controls for.** The strongest *existing* practice, and therefore the arm that
decides how much of this project can claim novelty. Embodiment and dataset tokens are
already standard in RT-X, Octo and OpenVLA, so "B6 is enough" is not a friendly result:
it would mean no algorithmic novelty and that the practice already exists. 10.5.2
preregisters the reading -- if B6 pushes `Lambda^part` to within 0.05 of B7 Oracle,
every claim about the contrastive component reduces to "already covered by existing
practice", and E7's two policy-side readings become the only positive result.

The same reasoning is what makes E7 the crown: a per-embodiment learnable bias is
exactly the right shape to absorb the *contrastive* part of the compensation, and
exactly the wrong shape for the common mode, which is same-signed everywhere and has no
contrast for a token to key on.

**The ablation.** At inference, feed embodiment `i` the token of `pi(i)`. If the leakage
rides on the token, permuting moves it; if it rides on the trunk, nothing happens. It
costs no training and is applied through `EmbodimentToken.set_permutation`, so the
weights are untouched.

**Provenance.** Implemented directly -- a conditioning token has no upstream to reuse.
The conditioning enters through the decoder's FiLM layers, which is the same attachment
point B4's style token and B13's latent use, because 10.5.6 requires those three arms
to condition at the same place for them to be comparable.
"""

from __future__ import annotations

import torch

from portal_trace.policies.conditioning import EmbodimentToken, EmbodimentTokenKind
from portal_trace.policies.interface import (
    Batch,
    ParameterBlock,
    PolicyConfig,
    RoutingKey,
    dense_blocks,
)
from portal_trace.policies.pools import PoolSpec, SubsetName, pool
from portal_trace.policies.portal_policy import PortalTracePolicy

__all__ = ["POOL", "SUBSET", "SEEDS", "B6EmbodimentTokenPolicy"]

POOL: PoolSpec = pool("B2-like")
SUBSET = SubsetName.S18
SEEDS = 2


class B6EmbodimentTokenPolicy(PortalTracePolicy):
    """The 10.4.0 architecture conditioned on the embodiment identity."""

    arm = "B6-embodiment-token"

    def __init__(
        self,
        cfg: PolicyConfig,
        *,
        kind: EmbodimentTokenKind = EmbodimentTokenKind.EMBEDDING,
        token_dim: int = 16,
    ) -> None:
        token = EmbodimentToken(cfg.n_pool_embodiments, kind, dim=token_dim)
        super().__init__(cfg, cond_dim=token.dim)
        self.token = token

    def conditioning(self, batch: Batch) -> torch.Tensor:
        return self.token(batch.obs.routing_index(RoutingKey.EMBODIMENT))

    def permute_tokens(self, permutation: torch.Tensor | None) -> None:
        """Install the inference-time permutation of 10.5. Training is unaffected."""
        self.token.set_permutation(permutation)

    def parameter_blocks(self) -> tuple[ParameterBlock, ...]:
        # Dense in the sense 10.4.9 cares about: every embodiment routes through the
        # same FiLM weights and the same decoder. Only one row of the embedding table is
        # read per sample, but that table is a few hundred parameters against fourteen
        # million, so billing it per embodiment would claim a structural isolation the
        # arm does not have and move the index far below its reported precision.
        return dense_blocks(self, self.cfg.n_pool_embodiments)
