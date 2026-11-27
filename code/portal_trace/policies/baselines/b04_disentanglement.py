"""B4: "task-embodiment disentanglement is already solved".

**Proposal item.** Baseline B4 of 10.5, specified in 10.5.1. Four variants, the most
dangerous collision in the whole plan, scheduled for week 5 so that a failure to beat
it leaves time to re-position (gate G5, exit F4):

* `PORTED_MI` -- CLUB mutual-information minimisation between the head's two branches
  plus InfoNCE with the embodiment labels as the negative source, the whole objective
  moved from the pixel layer to the action layer. Ported from `Disentangled
  Cross-Embodiment Video Editing`.
* `DISDP` -- the shared / private split with its reconstruction constraint kept and the
  private branch reinterpreted as the embodiment branch. Ported from `DisDP` (RLC 2025,
  Vanjani et al.).
* `DOMAIN_ADVERSARIAL` -- an embodiment discriminator behind a gradient-reversal layer,
  `lambda` swept over three levels. Implemented directly.
* `STYLE_TOKEN` -- a learned low-dimensional embodiment token multiplied into the action
  head's FiLM layer. Implemented directly.

**What it controls for.** Whether an off-the-shelf disentangling objective already
removes the compensation. The preregistered expectation is that all four lower the
**contrastive component** -- it has embodiment labels to key on -- and that none of them
touches the **common mode**, which is same-signed across every embodiment and therefore
has no gradient under any objective whose discriminative signal is embodiment identity.
E7 is the direct test of that expectation, which is why E7 and not B4 is the crown.

**Provenance.** Reimplemented from the two papers. `DisDP`'s repository is empty
("coming soon"), so there is nothing to vendor; the ported disentangling objective is
published for video pixels and its transfer to the action layer is the work 10.5.1
asks for, so no upstream applies either. The estimators themselves
(`CLUBEstimator`, `MINEEstimator`, `info_nce`, `gradient_reversal`) live in
`policies/losses.py` because 10.5.1's four variants share them and A14 sweeps which
one is used.

Two disciplines worth stating.

The task branch is what produces the action together with the embodiment branch, not a
decorative side head: 10.5.1 says the estimator is mounted *on the action head's two
branches*, so a variant whose branches did not carry the action would be measuring the
disentanglement of something the policy does not use.

The MI terms run their two objectives in one optimiser step through
`frozen_upper_bound` / `frozen_estimate`: the variational critic is trained on
detached inputs and the representations see a detached critic. Without that split the
critic would minimise the very bound it exists to tighten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn.functional as F
from robomimic.models.base_nets import MLP
from torch import nn

from portal_trace.policies.conditioning import EmbodimentToken, EmbodimentTokenKind
from portal_trace.policies.interface import (
    Batch,
    ParameterBlock,
    PolicyConfig,
    PolicyOutput,
    RoutingKey,
    dense_blocks,
)
from portal_trace.policies.losses import (
    CLUBEstimator,
    MINEEstimator,
    chunk_bc_loss,
    gradient_reversal,
)
from portal_trace.policies.pools import PoolSpec, SubsetName, pool
from portal_trace.policies.portal_policy import PortalTracePolicy

__all__ = [
    "POOL",
    "SUBSET",
    "SEEDS",
    "B4Variant",
    "MIEstimatorKind",
    "LAMBDA_LEVELS",
    "TEMPERATURE_LEVELS",
    "B4Settings",
    "B4DisentanglementPolicy",
]

POOL: PoolSpec = pool("B2-like")
SUBSET = SubsetName.S6
SEEDS = 2

# 10.5.7: three levels per variant, each variant scored at its own best.
LAMBDA_LEVELS: tuple[float, ...] = (0.1, 0.3, 1.0)
TEMPERATURE_LEVELS: tuple[float, ...] = (0.05, 0.1, 0.5)

_BRANCH_DIM = 128


class B4Variant(str, Enum):
    PORTED_MI = "ported_mi"
    DISDP = "disdp"
    DOMAIN_ADVERSARIAL = "domain_adversarial"
    STYLE_TOKEN = "style_token"


class MIEstimatorKind(str, Enum):
    """A14's estimator axis: the disentangling result must not depend on the bound."""

    CLUB = "club"
    MINE = "mine"


@dataclass(frozen=True)
class B4Settings:
    """One point of the per-variant sweep."""

    variant: B4Variant
    lambd: float = LAMBDA_LEVELS[1]
    temperature: float = TEMPERATURE_LEVELS[1]
    estimator: MIEstimatorKind = MIEstimatorKind.CLUB
    token_dim: int = 16
    branch_dim: int = _BRANCH_DIM
    extras: dict[str, float] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.lambd < 0.0:
            raise ValueError("lambda is a magnitude")
        if self.temperature <= 0.0:
            raise ValueError("the InfoNCE temperature is positive")


class _TwoBranchHead(nn.Module):
    """The action head 10.5.1 mounts the estimators on.

    Both branches feed the action projection, so the disentanglement is a property of the
    representation the policy actually acts through. They are deliberately the same shape
    and carry the same regularisation: an asymmetry between them would decide which
    branch ends up holding the embodiment information before any objective did.
    """

    def __init__(self, d_model: int, branch_dim: int, action_dim: int) -> None:
        super().__init__()
        self.task = MLP(
            input_dim=d_model,
            output_dim=branch_dim,
            layer_dims=(d_model,),
            activation=nn.GELU,
        )
        self.embodiment = MLP(
            input_dim=d_model,
            output_dim=branch_dim,
            layer_dims=(d_model,),
            activation=nn.GELU,
        )
        self.to_action = nn.Linear(2 * branch_dim, action_dim)

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        task = self.task(features)
        embodiment = self.embodiment(features)
        action = self.to_action(torch.cat((task, embodiment), dim=-1))
        return action, task, embodiment


class B4DisentanglementPolicy(PortalTracePolicy):
    """The 10.4.0 trunk with one of the four disentangling heads of 10.5.1."""

    arm = "B4-disentanglement"

    def __init__(self, cfg: PolicyConfig, *, settings: B4Settings) -> None:
        style = settings.variant is B4Variant.STYLE_TOKEN
        token = (
            EmbodimentToken(
                cfg.n_pool_embodiments,
                EmbodimentTokenKind.EMBEDDING,
                dim=settings.token_dim,
            )
            if style
            else None
        )
        super().__init__(cfg, cond_dim=token.dim if token is not None else 0)
        self.settings = settings
        self.token = token
        self.head = (
            None
            if style
            else _TwoBranchHead(cfg.d_model, settings.branch_dim, cfg.action_dim)
        )
        k = cfg.n_pool_embodiments
        d = settings.branch_dim

        if settings.variant is B4Variant.PORTED_MI:
            # The InfoNCE half: the embodiment branch is pulled onto its own label's
            # vector against every other label, which is what "the embodiment labels are
            # the negative source" means (10.5.1). It is also what stops the MI term from
            # being minimised trivially by collapsing the embodiment branch.
            self.label_embedding = nn.Embedding(k, d)
            self.club = CLUBEstimator(d, d) if settings.estimator is MIEstimatorKind.CLUB else None
            self.mine = MINEEstimator(d, d) if settings.estimator is MIEstimatorKind.MINE else None
        elif settings.variant is B4Variant.DISDP:
            # DisDP keeps a reconstruction constraint on the concatenated
            # shared/private pair; without it the private branch is free to be empty.
            self.reconstruct = MLP(
                input_dim=2 * d,
                output_dim=cfg.d_model,
                layer_dims=(cfg.d_model,),
                activation=nn.GELU,
            )
            self.private_classifier = nn.Linear(d, k)
        elif settings.variant is B4Variant.DOMAIN_ADVERSARIAL:
            self.discriminator = MLP(
                input_dim=d, output_dim=k, layer_dims=(d,), activation=nn.GELU
            )

    def conditioning(self, batch: Batch) -> torch.Tensor | None:
        if self.token is None:
            return None
        return self.token(batch.obs.routing_index(RoutingKey.EMBODIMENT))

    def forward(self, batch: Batch) -> PolicyOutput:
        memory = self.encode(batch.obs)
        features = self.decoder.features(memory, cond=self.conditioning(batch))
        if self.head is None:
            prediction = self.decoder.to_action(features)
            task = embodiment = None
        else:
            prediction, task, embodiment = self.head(features)

        if batch.actions is None:
            return PolicyOutput(action_chunk=prediction)
        bc = chunk_bc_loss(
            prediction,
            self.action_target(batch),
            pad_mask=batch.action_pad_mask,
            sample_weight=batch.sample_weight,
        )
        aux = self._variant_losses(batch, features, task, embodiment)
        total = bc
        for term in aux.values():
            total = total + term
        return PolicyOutput(
            action_chunk=prediction,
            loss=total,
            aux_losses={"bc": bc.detach(), **{k: v.detach() for k, v in aux.items()}},
            extras={"features": features},
        )

    def _variant_losses(
        self,
        batch: Batch,
        features: torch.Tensor,
        task: torch.Tensor | None,
        embodiment: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        variant = self.settings.variant
        if variant is B4Variant.STYLE_TOKEN:
            # The token enters through FiLM and carries no extra objective: 10.5.1
            # lists it as the cheapest of the four for exactly that reason.
            return {}
        assert task is not None and embodiment is not None  # the three other variants
        labels = batch.obs.routing_index(RoutingKey.EMBODIMENT)
        # Pool the chunk axis: the disentanglement is a statement about the sample's
        # representation, and treating the eight chunk steps as separate samples would
        # put eight copies of the same embodiment label in every InfoNCE denominator.
        task_flat = task.mean(dim=1)
        emb_flat = embodiment.mean(dim=1)
        lambd = self.settings.lambd

        if variant is B4Variant.PORTED_MI:
            if self.club is not None:
                bound = self.club.frozen_upper_bound(task_flat, emb_flat)
                critic = -self.club.log_likelihood(task_flat.detach(), emb_flat.detach())
            else:
                assert self.mine is not None
                bound = self.mine.frozen_estimate(task_flat, emb_flat)
                critic = -self.mine(task_flat.detach(), emb_flat.detach())
            # InfoNCE against the label prototypes: cosine similarities over a
            # temperature, the positive being the sample's own embodiment.
            logits = (
                F.normalize(emb_flat, dim=-1)
                @ F.normalize(self.label_embedding.weight, dim=-1).t()
            ) / self.settings.temperature
            return {
                "mi_bound": lambd * bound,
                "mi_critic": critic,
                "info_nce": lambd * F.cross_entropy(logits, labels),
            }

        if variant is B4Variant.DISDP:
            joined = torch.cat((task, embodiment), dim=-1)
            reconstruction = F.mse_loss(self.reconstruct(joined), features.detach())
            per_step = self.private_classifier(embodiment)
            private = F.cross_entropy(
                per_step.reshape(-1, per_step.shape[-1]),
                labels.repeat_interleave(per_step.shape[1]),
            )
            return {"reconstruction": lambd * reconstruction, "private_embodiment": lambd * private}

        logits = self.discriminator(gradient_reversal(task_flat, lambd))
        return {"domain_adversarial": F.cross_entropy(logits, labels)}

    def parameter_blocks(self) -> tuple[ParameterBlock, ...]:
        return dense_blocks(self, self.cfg.n_pool_embodiments)
