"""The 10.4.0 architecture: the arm every experiment is run on, and the base class.

    ResNet-18 visual tower on two 84x84 cameras
    -> 4-layer transformer decoder head, action chunk 8
    -> 6-dimensional task-space incremental pose + 1 gripper dimension

with a 70-dimensional proprioception input (9.6.1 as corrected in `obs.py`).

10.4.0 says the action space is literally identical across the arms of the main
setting -- a fixed nominal embodiment with only `delta` varying -- which is what
makes assumption (A6) hold by construction. That is also why the class is written
as a base class rather than as one of thirteen siblings: nine of the thirteen
baselines are this architecture with a different pool, a different label transform,
or one extra conditioning channel, and the moment any of them changed the trunk,
the paired differences the endpoints are built from would stop being paired.

The extension points are four small hooks. A subclass overrides only what its
proposal item actually changes, so that a reader of `b06_embodiment_token.py` can
see the whole of what B6 is:

* `conditioning(batch)` -- the FiLM vector, if the arm has one.
* `action_space_descriptor(batch)` -- what AS-MoE routes on, for E9's I1.
* `action_target(batch)` -- the supervision, which B8 replaces with `a - c_true`.
* `auxiliary_losses(...)` -- extra terms, which B4, B12 and B13 add.

Early stopping cannot be switched on from here: see `schedule.py`. The policy has
no termination output, so a rollout runs the fixed 400 steps of 9.6.1.
"""

from __future__ import annotations

import torch

from portal_trace.policies.decoder import ActionChunkDecoder, FeedForwardFactory
from portal_trace.policies.encoders import ObservationEncoder
from portal_trace.policies.interface import (
    Batch,
    Observation,
    ParameterBlock,
    Policy,
    PolicyConfig,
    PolicyOutput,
    dense_blocks,
)
from portal_trace.policies.losses import chunk_bc_loss
from portal_trace.policies.obs import ACTION_DIM, PROPRIO_GROUPS

__all__ = ["PortalTracePolicy"]


class PortalTracePolicy(Policy):
    """The shared architecture of 10.4.0.

    On its own it is the naive-pooling arm's network; B2 subclasses it only to
    attach the pool composition, because "which demonstrations went in" is the
    independent variable and belongs next to the arm, not here.
    """

    arm = "portal-10.4.0"

    def __init__(
        self,
        cfg: PolicyConfig,
        *,
        cond_dim: int = 0,
        action_dim: int = ACTION_DIM,
        feed_forward_factory: FeedForwardFactory | None = None,
    ) -> None:
        super().__init__(cfg)
        masked_groups = cfg.options.get("masked_proprio_groups", [])
        if not isinstance(masked_groups, (list, tuple)) or any(
            not isinstance(name, str) for name in masked_groups
        ):
            raise ValueError("masked_proprio_groups must be a list or tuple of group names")
        if len(set(masked_groups)) != len(masked_groups):
            raise ValueError("masked_proprio_groups must not contain duplicate names")
        groups_by_name = {group.name: group for group in PROPRIO_GROUPS}
        unknown_groups = set(masked_groups) - groups_by_name.keys()
        if unknown_groups:
            raise ValueError(f"unknown masked proprioception groups: {sorted(unknown_groups)}")
        if masked_groups and not cfg.obs.use_proprio:
            raise ValueError("masked_proprio_groups requires ObsSpec.use_proprio=True")
        # Plain slice metadata adds no parameters, buffers, or random draws.
        self._masked_proprio_slices = tuple(
            groups_by_name[name].slice for name in masked_groups
        )
        self.encoder = ObservationEncoder(
            cfg.obs, cfg.d_model, cfg.n_tasks, augment=cfg.augment
        )
        self.decoder = ActionChunkDecoder(
            cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.dropout,
            cond_dim=cond_dim,
            action_dim=action_dim,
            feed_forward_factory=feed_forward_factory,
        )

    # --- hooks ------------------------------------------------------------

    def conditioning(self, batch: Batch) -> torch.Tensor | None:
        """The FiLM vector for this arm. `None` for the unconditioned architecture."""
        del batch
        return None

    def action_space_descriptor(self, batch: Batch) -> torch.Tensor | None:
        """What AS-MoE routes on (E9's I1). `None` when there is no mixture."""
        del batch
        return None

    def action_target(self, batch: Batch) -> torch.Tensor:
        """The action label. B8 is the only arm that changes it."""
        return batch.require_actions()

    def auxiliary_losses(
        self, batch: Batch, features: torch.Tensor, prediction: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Terms added to the behaviour-cloning loss."""
        del batch, features, prediction
        return {}

    # --- forward ----------------------------------------------------------

    def encode(self, obs: Observation) -> torch.Tensor:
        proprio = obs.proprio
        if self._masked_proprio_slices:
            if proprio is None:
                raise ValueError("masked normalized proprioception requires Observation.proprio")
            # Mask normalized inputs, not raw sensor values. Training and
            # deployment share this path; the caller's observation is intact.
            proprio = proprio.clone()
            for group_slice in self._masked_proprio_slices:
                proprio[..., group_slice] = 0
        return self.encoder(obs.images, proprio, obs.task_index)

    def forward(self, batch: Batch) -> PolicyOutput:
        memory = self.encode(batch.obs)
        prediction, features = self.decoder(
            memory,
            cond=self.conditioning(batch),
            data_mask=self.action_space_descriptor(batch),
        )
        if batch.actions is None:
            return PolicyOutput(action_chunk=prediction)

        bc = chunk_bc_loss(
            prediction,
            self.action_target(batch),
            pad_mask=batch.action_pad_mask,
            sample_weight=batch.sample_weight,
        )
        aux = self.auxiliary_losses(batch, features, prediction)
        total = bc
        for term in aux.values():
            total = total + term
        return PolicyOutput(
            action_chunk=prediction,
            loss=total,
            aux_losses={"bc": bc.detach(), **{k: v.detach() for k, v in aux.items()}},
            extras={"features": features},
        )

    # --- capacity ---------------------------------------------------------

    def parameter_blocks(self) -> tuple[ParameterBlock, ...]:
        return dense_blocks(self, self.cfg.n_pool_embodiments)
