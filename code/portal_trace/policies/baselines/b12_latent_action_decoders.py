"""B12: "one decoder per embodiment".

**Proposal item.** Baseline B12 of 10.5: `Latent Action Diffusion for Cross-Embodiment
Manipulation` (arXiv 2506.14608), an embodiment-agnostic latent action plus a
per-embodiment decoder. 10.5 gives it no runs of its own -- its training runs are counted
inside E9's I3 rung -- so this module is both the baseline and the second implementation
of that rung, and it reports its parameters on the same caliber
`diagnostics/isolation.py` consumes.

**What it controls for.** It is the empirical strike instance of Corollary 1. The
corollary says that if the action offset is produced by parameter-disjoint submodules
routed deterministically by robot instance, then each submodule's design graph collapses
to a single-embodiment star and the compensation inside it is completely unidentifiable.
A per-embodiment decoder is per-instance parameter isolation, literally. So the arm is
expected to be *good* at the task and *maximally* leaky, which is the prediction T6 puts
on the isolation axis -- and it is the reason E9 needs I3', because the same routing also
gives each decoder only `1/K` of the data.

**Provenance.** Reimplemented from the paper; the two pieces that have installed
implementations use them: the denoising network is `robomimic.models.diffusion_policy_nets
.ConditionalUnet1D` and the noise schedule is `diffusers`' `DDPMScheduler`, which is the
same pair `robomimic.algo.diffusion_policy` is built on. The latent action space, the
per-embodiment decoders and the routing are from the paper.

Two implementation notes.

The UNet is built with `down_dims=(128, 256)` rather than robomimic's `(256, 512, 1024)`.
That default is sized for a 7-dimensional action chunk of length 16 in a policy with no
other head; here it would be 65 M parameters on top of a 14 M trunk and would make every
capacity comparison in 10.4.9 meaningless. The chunk is 8 steps, so two downsampling
levels are also as many as the temporal axis supports.

The paper pretrains the latent action space and then freezes it. Both orders are
available: `freeze_latent_space` gives the two-stage recipe, and joint training is the
default because with one action space across the pool (10.4.0) the alignment stage has
nothing to align.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from robomimic.models.base_nets import MLP
from robomimic.models.diffusion_policy_nets import ConditionalUnet1D
from torch import nn

from portal_trace.policies.interface import (
    Batch,
    ParameterBlock,
    PolicyConfig,
    PolicyOutput,
    RoutingKey,
    dense_blocks,
    hard_routed_blocks,
)
from portal_trace.policies.pools import PoolSpec, SubsetName, pool
from portal_trace.policies.portal_policy import PortalTracePolicy

__all__ = [
    "POOL",
    "SUBSET",
    "SEEDS",
    "LATENT_ACTION_DIM",
    "N_TRAIN_TIMESTEPS",
    "N_INFERENCE_STEPS",
    "B12LatentActionPolicy",
]

POOL: PoolSpec = pool("B2-like")
# E9's I3 rung is run on the isolation ladder's subset, not on a baseline subset of its
# own; 10.5 lists zero runs for B12 for exactly this reason. That subset is S^(4), which
# is what E9 and `arms.yaml` both say -- this line read S6 while the config read S4fam,
# and an arm with zero runs is exactly where such a disagreement survives unnoticed,
# since nothing is ever launched from it to contradict it.
SUBSET = SubsetName.S4fam
# Per-run seed count, not `seeds.yaml`'s allocation: that file gives B12 an empty list
# because its runs are counted as E9's I3 rung, while this is what one run would cost.
SEEDS = 1

LATENT_ACTION_DIM = 16
N_TRAIN_TIMESTEPS = 100
# DDPM sampling is the rollout cost of this arm; the rollout is 400 fixed steps per
# episode (9.6.1), so the step count is a real budget item and not a free parameter.
N_INFERENCE_STEPS = 10


class B12LatentActionPolicy(PortalTracePolicy):
    """Embodiment-agnostic latent action, diffusion head, per-embodiment decoders."""

    arm = "B12-latent-action-decoders"

    def __init__(
        self,
        cfg: PolicyConfig,
        *,
        latent_dim: int = LATENT_ACTION_DIM,
        n_inference_steps: int = N_INFERENCE_STEPS,
        reconstruction_weight: float = 1.0,
    ) -> None:
        super().__init__(cfg)
        self.latent_dim = latent_dim
        self.n_inference_steps = n_inference_steps
        self.reconstruction_weight = reconstruction_weight
        k = cfg.n_pool_embodiments

        # The shared half: actions in, latent actions out. One encoder, because 10.4.0
        # fixes one action space across the pool -- what is per embodiment is the decoder,
        # which is the object Corollary 1 is about.
        self.latent_encoder = MLP(
            input_dim=cfg.action_dim,
            output_dim=latent_dim,
            layer_dims=(256,),
            activation=nn.GELU,
        )
        self.decoders = nn.ModuleList(
            MLP(
                input_dim=latent_dim,
                output_dim=cfg.action_dim,
                layer_dims=(256,),
                activation=nn.GELU,
            )
            for _ in range(k)
        )
        self.unet = ConditionalUnet1D(
            input_dim=latent_dim, global_cond_dim=cfg.d_model, down_dims=[128, 256]
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=N_TRAIN_TIMESTEPS,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def freeze_latent_space(self) -> None:
        """The paper's two-stage recipe: fit the latent space, then the policy."""
        for p in self.latent_encoder.parameters():
            p.requires_grad_(False)
        for p in self.decoders.parameters():
            p.requires_grad_(False)

    def _condition(self, batch: Batch) -> torch.Tensor:
        memory = self.encode(batch.obs)
        features = self.decoder.features(memory)
        # One global conditioning vector, which is what ConditionalUnet1D's FiLM takes;
        # the per-step structure of the chunk is carried by the UNet's temporal axis.
        return features.mean(dim=1)

    def _decode(self, latent: torch.Tensor, batch: Batch) -> torch.Tensor:
        """Route each sample's latent chunk to its embodiment's decoder."""
        index = batch.obs.routing_index(RoutingKey.EMBODIMENT)
        if int(index.max()) >= len(self.decoders):
            raise ValueError(
                f"embodiment index {int(index.max())} exceeds the "
                f"{len(self.decoders)} decoders"
            )
        out = torch.zeros(
            latent.shape[0], latent.shape[1], self.cfg.action_dim, device=latent.device
        )
        for key in index.unique().tolist():
            mask = index == key
            out[mask] = self.decoders[key](latent[mask])
        return out

    @torch.no_grad()
    def _sample_latent(self, condition: torch.Tensor) -> torch.Tensor:
        latent = torch.randn(
            condition.shape[0], self.cfg.chunk, self.latent_dim, device=condition.device
        )
        self.noise_scheduler.set_timesteps(self.n_inference_steps)
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.unet(latent, t, global_cond=condition)
            latent = self.noise_scheduler.step(noise_pred, t, latent).prev_sample
        return latent

    def forward(self, batch: Batch) -> PolicyOutput:
        condition = self._condition(batch)
        if batch.actions is None:
            latent = self._sample_latent(condition)
            return PolicyOutput(action_chunk=self._decode(latent, batch))

        target = self.action_target(batch)
        latent_target = self.latent_encoder(target)
        reconstruction = self._decode(latent_target, batch)

        noise = torch.randn_like(latent_target)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (latent_target.shape[0],),
            device=latent_target.device,
        )
        # The diffusion objective is on the latent, so the latent must not drift under
        # it: the target is detached and only the reconstruction term shapes the latent
        # space, which is the paper's separation of the two stages expressed as two
        # gradients rather than two runs.
        noisy = self.noise_scheduler.add_noise(latent_target.detach(), noise, timesteps)
        noise_pred = self.unet(noisy, timesteps, global_cond=condition)
        diffusion = F.mse_loss(noise_pred, noise)
        recon = F.mse_loss(reconstruction, target)
        total = diffusion + self.reconstruction_weight * recon
        return PolicyOutput(
            # The reconstruction, not a sample: 10.4.0's training loop never needs the
            # sampled action, and paying for a 10-step DDPM chain on every training step
            # would make the arm's step time incomparable with every other arm's.
            action_chunk=reconstruction,
            loss=total,
            aux_losses={"diffusion": diffusion.detach(), "reconstruction": recon.detach()},
        )

    def parameter_blocks(self) -> tuple[ParameterBlock, ...]:
        k = self.cfg.n_pool_embodiments
        blocks = list(dense_blocks(self, k, exclude=("decoders",)))
        blocks.extend(hard_routed_blocks("latent_decoder", self.decoders, k))
        return tuple(blocks)
