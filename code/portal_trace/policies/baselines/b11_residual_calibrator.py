"""B11: "just add a residual head at deployment".

**Proposal item.** Baseline B11 of 10.5, the J-PARC-style arm: a frozen trunk plus an
implicit joint-fault regime inferred from recent joint dynamics and execution history,
conditioning a single shared residual calibrator, with the residual added **only to the
translational arm dimensions**. Five levels of `N` target demonstrations x `S9` x two
seeds, head training only (10.5 lists zero full runs for it and 18 GPU-h of head
training).

**What it controls for.** It instantiates route (I) of Proposition 1(d) -- access to the
target embodiment's own interaction or return signal -- and it is expected to work
clearly once `N >= 20`. That is not a refutation of anything: route (I) is one of the
five branches the proposition itself lists, and 2.1 restates the scope, namely that
Proposition 1(b) is about offline datasets in which `delta` is constant, while this arm
uses target-embodiment signal and is therefore outside that scope to begin with. The
`N in {0, 5, 20, 50, 100}` curve is mandatory because the reading is *where* the branch
starts working, not whether it does.

Its second role is external support for assumption (A5)(ii): the regime posterior is
computed from a short window of the policy input, and J-PARC showed that this works on a
real VLA stack -- OpenVLA-OFT 42.3% to 46.8%, pi-0.5 48.4% to 55.2% under joint lock.
That is why `FaultRegimeHead` posteriors are over the five regimes of 9.7.1 rather than
over a private label set.

**Provenance.** Reimplemented from `Uncovering Vulnerability of VLA Models under
Joint-Level Physical Faults` (arXiv 2606.10501); no public code exists. Three mechanisms
are taken from the paper and are the reason the arm is what it is: the frozen backbone,
the implicit regime inference from a short history, and one shared calibrator conditioned
on that regime rather than one calibrator per fault. The restriction of the residual to
the translational dimensions is also the paper's, and 2.1 reads it as the paper confining
itself to the subspace where a residual is most easily absorbed -- so it is kept exactly,
not generalised.

Head capacity and head training length are chosen at this arm's own optimum and are
deliberately *not* aligned with the main arm (10.5.7): a baseline is only worth beating
at its best.
"""

from __future__ import annotations

import torch
from robomimic.models.base_nets import MLP
from torch import nn

from portal_trace.policies.conditioning import N_REGIMES, FaultRegimeHead
from portal_trace.policies.interface import (
    Batch,
    ParameterBlock,
    PolicyConfig,
    PolicyOutput,
    dense_blocks,
)
from portal_trace.policies.losses import chunk_bc_loss
from portal_trace.policies.obs import TRANSLATION_DIMS
from portal_trace.policies.pools import PoolSpec, SubsetName, pool
from portal_trace.policies.portal_policy import PortalTracePolicy

__all__ = [
    "POOL",
    "SUBSET",
    "SEEDS",
    "TARGET_DEMO_LEVELS",
    "EFFECTIVE_FROM",
    "ResidualCalibrator",
    "B11ResidualCalibratorPolicy",
]

# The pool the frozen trunk came from; the head itself is fitted on N target
# demonstrations, which is the arm's independent variable rather than a pool.
POOL: PoolSpec = pool("B2-like")
SUBSET = SubsetName.S9
SEEDS = 2

TARGET_DEMO_LEVELS: tuple[int, ...] = (0, 5, 20, 50, 100)
# 14.3's preregistered reading for route (I): clearly effective from this level up.
EFFECTIVE_FROM = 20


class ResidualCalibrator(nn.Module):
    """One shared calibrator, conditioned on the regime posterior.

    Shared rather than per regime: that is the paper's mechanism and the reason the arm
    needs so few target demonstrations, since every regime's data trains the same
    weights and the posterior only selects the behaviour.

    The output projection is zero-initialised, which makes the `N = 0` point of the
    mandatory curve exactly the frozen trunk rather than the frozen trunk plus noise.
    """

    def __init__(
        self, d_model: int, *, hidden: int = 256, regime_dim: int = N_REGIMES
    ) -> None:
        super().__init__()
        self.net = MLP(
            input_dim=d_model + regime_dim,
            output_dim=hidden,
            layer_dims=(hidden,),
            activation=nn.GELU,
        )
        self.to_residual = nn.Linear(hidden, len(TRANSLATION_DIMS))
        nn.init.zeros_(self.to_residual.weight)
        nn.init.zeros_(self.to_residual.bias)

    def forward(
        self, features: torch.Tensor, regime_posterior: torch.Tensor
    ) -> torch.Tensor:
        """Returns the translational residual, shape (B, CHUNK, 3)."""
        chunk = features.shape[1]
        conditioned = torch.cat(
            (features, regime_posterior.unsqueeze(1).expand(-1, chunk, -1)), dim=-1
        )
        return self.to_residual(self.net(conditioned))


class B11ResidualCalibratorPolicy(PortalTracePolicy):
    """Frozen 10.4.0 trunk, regime inference, shared residual on the translation axes."""

    arm = "B11-residual-calibrator"

    def __init__(
        self,
        cfg: PolicyConfig,
        *,
        n_target_demos: int = EFFECTIVE_FROM,
        hidden: int = 256,
        use_action_history: bool = True,
    ) -> None:
        super().__init__(cfg)
        if n_target_demos not in TARGET_DEMO_LEVELS:
            raise ValueError(
                f"n_target_demos={n_target_demos} is not one of the mandatory levels "
                f"{TARGET_DEMO_LEVELS}; 10.5 requires the whole curve to be reported, "
                "so a run has to sit on one of its points"
            )
        if not cfg.obs.use_proprio:
            raise ValueError(
                "the regime is inferred from recent joint dynamics; an arm with no "
                "proprioception pathway cannot carry J-PARC's mechanism"
            )
        self.n_target_demos = n_target_demos
        self.regime_head = FaultRegimeHead(
            cfg.obs.k_hist, hidden=hidden, use_action_history=use_action_history
        )
        self.calibrator = ResidualCalibrator(cfg.d_model, hidden=hidden)
        self.freeze_trunk()

    def freeze_trunk(self) -> None:
        """Freeze everything the pooled checkpoint provided (the paper's setting)."""
        for module in (self.encoder, self.decoder):
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True) -> B11ResidualCalibratorPolicy:
        super().train(mode)
        # The trunk stays in eval mode however the caller drives the module: its
        # dropout and image augmentation would make the frozen features stochastic and
        # the residual would spend its capacity on that noise.
        self.encoder.eval()
        self.decoder.eval()
        return self

    def forward(self, batch: Batch) -> PolicyOutput:
        proprio = batch.obs.proprio
        if proprio is None:
            raise ValueError("B11 reads the proprioception history; Observation.proprio is None")
        action_history = batch.extras.get("action_history")
        posterior = self.regime_head(proprio, action_history)

        with torch.no_grad():
            memory = self.encode(batch.obs)
            features = self.decoder.features(memory)
            base = self.decoder.to_action(features)

        residual = self.calibrator(features, posterior)
        prediction = base.clone()
        # Only the translational arm dimensions; rotation and gripper commands are left
        # exactly as the frozen policy issued them (the paper's restriction).
        prediction[..., list(TRANSLATION_DIMS)] = (
            prediction[..., list(TRANSLATION_DIMS)] + residual
        )

        extras = {"regime_posterior": posterior, "residual": residual}
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
