"""A1: the encoder each observation-modality level is built with.

`portal_trace.ablations.modality` says what a level is; this module builds it.
The split is so that reading `configs/experiments/ablations.yaml` does not pull
in torch and robomimic -- see that module's docstring.

The main level and the RGB-only level are the stock `ObservationEncoder`: the
first is 10.4.0's arm verbatim, and the second is what `ObsSpec(use_proprio=
False)` already produces, which is also how B9 is built. Only the
proprioception-only level needs anything new, and what it needs is an encoder
with no visual pathway -- a token sequence of `k_hist` proprioception tokens plus
the task token, positionally embedded over that shorter length.
"""

from __future__ import annotations

import torch
from torch import nn

from portal_trace.ablations.modality import ModalitySpec
from portal_trace.policies.encoders import ObservationEncoder, ProprioEncoder
from portal_trace.policies.interface import ParameterBlock, Policy, PolicyConfig, dense_blocks
from portal_trace.policies.obs import ACTION_DIM, ObsSpec
from portal_trace.policies.portal_policy import PortalTracePolicy

__all__ = [
    "A1ModalityPolicy",
    "ProprioOnlyEncoder",
    "build_modality_policy",
    "obs_spec_for",
]


def obs_spec_for(spec: ModalitySpec, base: ObsSpec) -> ObsSpec:
    """`base` with the level's proprioception decision applied.

    The camera tuple is carried through unchanged even when the level drops
    images: `ObsSpec` refuses an empty one, and the names remain the streams the
    collection stage recorded, which is what the data card reports.
    """
    return ObsSpec(
        cameras=base.cameras,
        image_size=base.image_size,
        k_hist=base.k_hist,
        use_proprio=spec.use_proprio,
        use_task_token=base.use_task_token,
    )


class ProprioOnlyEncoder(ObservationEncoder):
    """`ObservationEncoder` with the visual trunk removed rather than blinded.

    `nn.Module.__init__` is called instead of the base class's, which would build
    two ResNet-18 trunks this level exists to do without. The inherited `forward`
    starts from `self.visual(images)` unconditionally, so it is overridden here;
    everything else -- the proprioception encoder, the task embedding, the
    positional embedding and the final norm -- is the base class's, so the level
    differs from the main arm in the image pathway and in nothing else.
    """

    def __init__(self, spec: ObsSpec, d_model: int, n_tasks: int) -> None:
        nn.Module.__init__(self)
        if not spec.use_proprio:
            raise ValueError(
                "a proprioception-only encoder without proprioception would leave "
                "the task token as the whole input"
            )
        self.spec = spec
        self.d_model = d_model
        self.visual = None
        self.proprio = ProprioEncoder(d_model)
        self.task_embedding = nn.Embedding(n_tasks, d_model) if spec.use_task_token else None
        self.n_tokens = spec.k_hist + (1 if spec.use_task_token else 0)
        self.position = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        images: dict[str, torch.Tensor],
        proprio: torch.Tensor | None,
        task_index: torch.Tensor | None,
    ) -> torch.Tensor:
        if images:
            raise ValueError(
                "this arm is A1's proprioception-only level and has no image "
                "pathway; passing frames would silently discard them, and the "
                "level's whole content is that they are not there"
            )
        if proprio is None:
            raise ValueError("A1's proprioception-only level needs Observation.proprio")
        parts = [self.proprio(proprio)]
        if self.task_embedding is not None:
            if task_index is None:
                raise ValueError(
                    "this arm conditions on the task instruction; "
                    "Observation.task_index is None"
                )
            parts.append(self.task_embedding(task_index).unsqueeze(1))
        return self.norm(torch.cat(parts, dim=1) + self.position)


class A1ModalityPolicy(PortalTracePolicy):
    """The 10.4.0 architecture on one of A1's three sensor configurations."""

    arm = "A1-modality"

    def __init__(self, cfg: PolicyConfig, *, modality: ModalitySpec) -> None:
        if modality.use_images:
            # `use_proprio` already lives in the ObsSpec, so the two
            # image-carrying levels are the stock policy and share its code path.
            super().__init__(cfg)
            self.modality = modality
            return
        Policy.__init__(self, cfg)
        self.modality = modality
        self.encoder = ProprioOnlyEncoder(cfg.obs, cfg.d_model, cfg.n_tasks)
        from portal_trace.policies.decoder import ActionChunkDecoder

        self.decoder = ActionChunkDecoder(
            cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.dropout,
            cond_dim=0,
            action_dim=ACTION_DIM,
        )

    def parameter_blocks(self) -> tuple[ParameterBlock, ...]:
        return dense_blocks(self, self.cfg.n_pool_embodiments)


def build_modality_policy(
    modality: ModalitySpec, *, base: PolicyConfig | None = None
) -> A1ModalityPolicy:
    """Build the arm for one A1 level, holding everything else at 10.4.0."""
    cfg = base or PolicyConfig(arm=f"A1-{modality.level_id}")
    return A1ModalityPolicy(
        cfg.evolve(obs=obs_spec_for(modality, cfg.obs)), modality=modality
    )
