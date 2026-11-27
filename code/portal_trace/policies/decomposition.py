"""E4, the two-way decomposition head, in all three forms (10.4.4).

E4 is an instrument, not an algorithmic contribution. Its one endpoint is the
ground-truth cosine `cos(g_phi, c_true)` where `g_phi` is the compensation
component the head splits out, evaluated in `L^2(rho^off)`. The reason it exists is
that nobody else can compute that number, because nobody else has a per-frame
compensation ground truth.

The independent variable is the functional form of the head, and 10.4.4 lists three:

| Form | `c(x)` | Preregistered prediction |
| --- | --- | --- |
| `LOW_RANK` | `W_delta phi(x)`, `W` low rank | between the other two |
| `PHYSICAL_BASIS` | `sum_j delta_j psi_j(x)` | cosine >= 0.6 |
| `UNCONSTRAINED_MLP` | an MLP on the trunk features | cosine <= 0.35 |

The gap of 0.25 between the last two is the finding the design is powered for: it
says identifiability is not a question of fitting capacity.

Two implementation disciplines the section forces:

The compensation branch is **never supervised with `c_true`**. The head is fitted by
behaviour cloning on the pooled data and the cosine is measured afterwards.
Regressing on `c_true` would make the endpoint a training objective and the number
meaningless.

The trunk is **frozen**. 10.4.4 costs the experiment at zero full training runs plus
54 head fits, which is only true if the pooled backbone is reused as-is.
"""

from __future__ import annotations

from enum import Enum

import torch
from robomimic.models.base_nets import MLP
from torch import nn

from portal_trace.policies.decoder import FeedForwardFactory
from portal_trace.policies.interface import (
    Batch,
    ParameterBlock,
    PolicyConfig,
    PolicyOutput,
    dense_blocks,
)
from portal_trace.policies.losses import chunk_bc_loss, cosine_alignment
from portal_trace.policies.obs import ACTION_DIM, ARM_ACTION_DIM, CHUNK
from portal_trace.policies.physics_basis import (
    DEFAULT_INJECTED_JOINTS,
    psi_column_count,
    psi_group_slices,
    validate_psi_basis,
)
from portal_trace.policies.portal_policy import PortalTracePolicy

__all__ = [
    "DEFAULT_LOW_RANK",
    "DecompositionForm",
    "TwoWayDecompositionHead",
    "E4DecompositionPolicy",
]

# `W_delta` is `d_a x d_phi` with `d_a = 7`, so its rank cannot exceed 7 whatever is
# asked for: the shape already caps it. A form called low rank therefore has to sit
# strictly below that cap, and 4 is the largest value that still constrains the map.
# 10.4.4 fixes the three functional forms but not this number, so it is part of the
# form's definition and is reported with the results rather than tuned.
DEFAULT_LOW_RANK = 4


class DecompositionForm(str, Enum):
    """The three forms of 10.4.4, and the function class each one restricts to."""

    LOW_RANK = "low_rank"
    PHYSICAL_BASIS = "physical_basis"
    UNCONSTRAINED_MLP = "unconstrained_mlp"


class TwoWayDecompositionHead(nn.Module):
    """Splits the predicted action into task intent and compensation.

    The task-intent branch is the same unrestricted MLP in all three forms; only the
    compensation branch changes. That is deliberate and is the point of 7.3: if the
    task-intent class is unrestricted, a physical basis on the compensation side can
    be absorbed by it, so restricting only the compensation branch is not expected to
    identify anything on its own. E4 measures how far that goes; E10 is the arm that
    adds the (ER) input-channel excision on top.
    """

    def __init__(
        self,
        form: DecompositionForm,
        d_model: int,
        *,
        action_dim: int = ACTION_DIM,
        hidden: int = 256,
        phi_dim: int | None = None,
        low_rank: int = DEFAULT_LOW_RANK,
        n_injected_joints: int = DEFAULT_INJECTED_JOINTS,
    ) -> None:
        super().__init__()
        self.form = form
        self.action_dim = action_dim
        self.n_injected_joints = n_injected_joints
        self.task_intent = MLP(
            input_dim=d_model,
            output_dim=action_dim,
            layer_dims=(hidden,),
            activation=nn.GELU,
        )

        if form is DecompositionForm.LOW_RANK:
            if phi_dim is None:
                raise ValueError(
                    "the low-rank form needs phi_dim, the effective rank of the 9.4.4 "
                    "feature basis after the truncated SVD. It is a per-cell "
                    "measurement (W1 reports the rank histogram), so there is no "
                    "default: pass the width of the phi_features you are about to feed"
                )
            if low_rank < 1 or low_rank > min(action_dim, phi_dim):
                raise ValueError(
                    f"low_rank={low_rank} must lie in [1, min(action_dim, phi_dim)] = "
                    f"[1, {min(action_dim, phi_dim)}]"
                )
            # W_delta = left @ right, rank <= low_rank. Proposition 1(e')(ii) turns the
            # scalar coefficient of the leakage argument into this matrix; the
            # null-space argument of 1(a) then holds column by column.
            self.right = nn.Parameter(torch.zeros(low_rank, phi_dim))
            self.left = nn.Parameter(torch.zeros(action_dim, low_rank))
            nn.init.normal_(self.right, std=phi_dim**-0.5)
            nn.init.normal_(self.left, std=low_rank**-0.5)
            self.phi_dim = phi_dim
            self.low_rank = low_rank
        elif form is DecompositionForm.PHYSICAL_BASIS:
            n_psi = psi_column_count(n_injected_joints)
            # One learnable coefficient per basis column: this is delta_hat, and E10
            # reports how close it lands to the true delta.
            self.delta_hat = nn.Parameter(torch.zeros(n_psi))
            nn.init.normal_(self.delta_hat, std=n_psi**-0.5)
            self.n_psi = n_psi
        else:
            self.compensation_mlp = MLP(
                input_dim=d_model,
                output_dim=action_dim,
                layer_dims=(hidden, hidden),
                activation=nn.GELU,
            )

    def delta_coefficients(self) -> dict[str, torch.Tensor]:
        """`delta_hat` split per defect group; only the physical-basis form has one."""
        if self.form is not DecompositionForm.PHYSICAL_BASIS:
            raise TypeError(
                f"the {self.form.value} form has no physical coefficients; only "
                "PHYSICAL_BASIS parameterises c by delta_hat"
            )
        return {
            name: self.delta_hat[sl]
            for name, sl in psi_group_slices(self.n_injected_joints).items()
        }

    def compensation(
        self,
        features: torch.Tensor,
        *,
        phi_features: torch.Tensor | None,
        psi_basis: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.form is DecompositionForm.LOW_RANK:
            if phi_features is None:
                raise ValueError("the low-rank form needs Batch.phi_features")
            if phi_features.shape[-1] != self.phi_dim:
                raise ValueError(
                    f"phi_features has width {phi_features.shape[-1]}, the head was "
                    f"built for {self.phi_dim}"
                )
            projected = torch.einsum("rp,btp->btr", self.right, phi_features)
            return torch.einsum("ar,btr->bta", self.left, projected)
        if self.form is DecompositionForm.PHYSICAL_BASIS:
            if psi_basis is None:
                raise ValueError("the physical-basis form needs Batch.psi_basis")
            psi = validate_psi_basis(
                psi_basis, chunk=CHUNK, n_injected_joints=self.n_injected_joints
            )
            arm = torch.einsum("j,btja->bta", self.delta_hat, psi)
            if self.action_dim == ARM_ACTION_DIM:
                return arm
            # The physical basis spans the task-space block only; there is no closed
            # form for a gripper compensation column, so it is exactly zero rather
            # than a free parameter that would make the form less restricted than
            # 10.4.4 says it is.
            pad = arm.new_zeros(*arm.shape[:-1], self.action_dim - ARM_ACTION_DIM)
            return torch.cat((arm, pad), dim=-1)
        return self.compensation_mlp(features)

    def forward(
        self,
        features: torch.Tensor,
        *,
        phi_features: torch.Tensor | None = None,
        psi_basis: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns `(task_intent, compensation)`; their sum is the predicted action."""
        return (
            self.task_intent(features),
            self.compensation(features, phi_features=phi_features, psi_basis=psi_basis),
        )


class E4DecompositionPolicy(PortalTracePolicy):
    """A frozen pooled trunk with a two-way decomposition head on top (10.4.4).

    Reimplemented from the proposal: the decomposition head is this project's own
    instrument and has no upstream implementation. The trunk is the 10.4.0
    architecture built by `PortalTracePolicy`, and `load_trunk` is how a trained
    pooled checkpoint gets in; the head is the only thing that trains.
    """

    arm = "E4-decomposition"

    def __init__(
        self,
        cfg: PolicyConfig,
        *,
        form: DecompositionForm,
        phi_dim: int | None = None,
        low_rank: int = DEFAULT_LOW_RANK,
        n_injected_joints: int = DEFAULT_INJECTED_JOINTS,
        feed_forward_factory: FeedForwardFactory | None = None,
    ) -> None:
        super().__init__(cfg, feed_forward_factory=feed_forward_factory)
        self.form = form
        self.head = TwoWayDecompositionHead(
            form,
            cfg.d_model,
            action_dim=cfg.action_dim,
            phi_dim=phi_dim,
            low_rank=low_rank,
            n_injected_joints=n_injected_joints,
        )
        # The trunk's own action projection is unused: the head produces the action
        # as intent + compensation. It stays in the module so that a pooled
        # checkpoint loads without key surgery, and it is frozen with the rest.
        self.freeze_trunk()

    def freeze_trunk(self) -> None:
        for module in (self.encoder, self.decoder):
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True) -> E4DecompositionPolicy:
        super().train(mode)
        # Keep the trunk in eval mode whatever the caller does: its dropout and the
        # image augmentation would otherwise make the frozen features stochastic and
        # add variance to a cosine that is being read to two decimal places.
        self.encoder.eval()
        self.decoder.eval()
        return self

    def forward(self, batch: Batch) -> PolicyOutput:
        with torch.no_grad():
            memory = self.encode(batch.obs)
            features = self.decoder.features(
                memory,
                cond=self.conditioning(batch),
                data_mask=self.action_space_descriptor(batch),
            )
        intent, compensation = self.head(
            features, phi_features=batch.phi_features, psi_basis=batch.psi_basis
        )
        prediction = intent + compensation

        extras: dict[str, torch.Tensor] = {
            "task_intent": intent,
            "compensation": compensation,
        }
        if batch.c_true is not None:
            # The endpoint, as a metric. Detached on purpose: 10.4.4 reads the cosine
            # off a head fitted by behaviour cloning, and letting it into the loss
            # would turn the instrument into its own answer.
            extras["cosine_to_c_true"] = cosine_alignment(
                compensation.detach(), batch.c_true
            ).reshape(1)
        if batch.actions is None:
            return PolicyOutput(action_chunk=prediction, extras=extras)

        bc = chunk_bc_loss(
            prediction,
            self.action_target(batch),
            pad_mask=batch.action_pad_mask,
            sample_weight=batch.sample_weight,
        )
        return PolicyOutput(
            action_chunk=prediction,
            loss=bc,
            aux_losses={"bc": bc.detach()},
            extras=extras,
        )

    def parameter_blocks(self) -> tuple[ParameterBlock, ...]:
        return dense_blocks(self, self.cfg.n_pool_embodiments)
