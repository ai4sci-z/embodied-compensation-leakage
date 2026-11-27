"""B5: "just randomise it away".

**Proposal item.** Baseline B5 of 10.5: training-time domain randomisation over the
defect parameters. Subset `S6`, one seed, 6 runs.

**What it controls for.** The reflex answer from the sim2real literature. It wins on
robustness and loses on the thing under study, and 10.5 states why in one line: what it
randomises is the *environment*, not the demonstrator's body. The demonstrations are
collected once, on a defective body, by a teacher that compensated; randomising what
the policy sees at training time cannot remove a compensation that is already inside
the action label. The arm is therefore expected to have a leakage indistinguishable
from B2 while being more robust to everything else -- and that combination is itself an
informative reading.

**Provenance.** Implemented directly; domain randomisation is a technique, not a
codebase. The randomisation ranges are not invented here: they are read off the level
tables in `defects/params.py`, which transcribe 9.2.10.

**Where each axis is applied, and why it is split in two.** The policy input is
normalised under the global caliber of 9.6.2, whose statistics belong to the data
pipeline. An injection parameter in radians or newton-metres therefore has no
well-defined image in the policy's input space, so the magnitude-valued axes (D1's
`delta_q`, D6's `delta_f`, D5's dead zone, D3b's `sigma_g`) are randomised where they
are defined -- at collection, by the injector, from the distribution
`DEFECT_RANDOMISATION` publishes. What this module randomises is the part that *is*
exactly expressible on the policy input and nothing else:

* **latency**, by shifting the history window by `n` control steps. `n` is literally
  D2's injection parameter and its levels are literally 0-4 control steps (9.2.10), so
  this axis needs no unit conversion at all.
* **the visual channel**, by the strong setting of the same crop-and-jitter
  augmentation the main arm uses at its nominal setting.
* **sensor availability**, by dropping the wrist force/torque block, which is a
  zero-write on a channel group and exact in any normalisation.

Nothing here touches the action label. An arm that perturbed the label would no longer
be domain randomisation; it would be B8.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from portal_trace.defects.params import DEFECTS, DefectSpec
from portal_trace.policies.encoders import ImageAugmentation
from portal_trace.policies.interface import (
    Batch,
    Observation,
    ParameterBlock,
    PolicyConfig,
    PolicyOutput,
    dense_blocks,
)
from portal_trace.policies.obs import proprio_group
from portal_trace.policies.pools import PoolSpec, SubsetName, pool
from portal_trace.policies.portal_policy import PortalTracePolicy

__all__ = [
    "POOL",
    "SUBSET",
    "SEEDS",
    "DefectRandomisation",
    "DEFECT_RANDOMISATION",
    "MAX_LATENCY_STEPS",
    "FT_DROPOUT_PROBABILITY",
    "InputRandomisation",
    "B5DomainRandomisationPolicy",
]

POOL: PoolSpec = pool("B2-like")
SUBSET = SubsetName.S6
SEEDS = 1

# D2's own levels: 0, 1, 2, 3, 4 control steps at 20 Hz (9.2.10). The randomisation is
# uniform over them, which is the widest range the catalogue defines.
MAX_LATENCY_STEPS = 4
FT_DROPOUT_PROBABILITY = 0.1


@dataclass(frozen=True)
class DefectRandomisation:
    """The distribution the injector samples a defect parameter vector from.

    Published from here because the arm owns the choice, consumed at collection
    because that is where an injection parameter has a meaning. `ranges` is per defect
    code and per parameter name, taken as the hull of that defect's five levels: the
    levels are a dose ladder, and randomising over their hull is the widest
    randomisation the catalogue supports without extrapolating past a validated dose.
    """

    codes: tuple[str, ...]

    @property
    def ranges(self) -> dict[str, dict[str, tuple[float, float]]]:
        out: dict[str, dict[str, tuple[float, float]]] = {}
        for code in self.codes:
            spec: DefectSpec = DEFECTS[code]
            per_param: dict[str, list[float]] = {}
            for level in spec.levels:
                for name, value in level.params.items():
                    per_param.setdefault(name, []).append(value)
            out[code] = {n: (min(v), max(v)) for n, v in per_param.items()}
        return out

    def sample(self, generator: torch.Generator | None = None) -> dict[str, dict[str, float]]:
        """One defect parameter vector per defect code, uniform over the hull."""
        out: dict[str, dict[str, float]] = {}
        for code, params in self.ranges.items():
            drawn: dict[str, float] = {}
            for name, (lo, hi) in params.items():
                u = torch.rand((), generator=generator).item()
                drawn[name] = lo + u * (hi - lo)
            out[code] = drawn
        return out


# The six core defects of 9.3.3; randomising over anything else would put the arm on a
# different defect set from every other arm in the table.
DEFECT_RANDOMISATION = DefectRandomisation(codes=("D1", "D2", "D3b", "D4hi", "D5", "D6"))


class InputRandomisation(torch.nn.Module):
    """The part of B5 that acts on the policy input, training only."""

    def __init__(self, *, max_latency: int = MAX_LATENCY_STEPS) -> None:
        super().__init__()
        if max_latency < 0:
            raise ValueError("a latency randomisation is over non-negative delays")
        self.max_latency = max_latency
        self.ft_slice = proprio_group("wrist_ft").slice

    def _shift(self, x: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        """Delay each sample's history by its own number of control steps.

        A delayed window repeats its oldest frame, which is what a first-order hold on
        a not-yet-arrived observation does; dropping the sample instead would make the
        randomisation change the batch composition.
        """
        k = x.shape[1]
        index = torch.arange(k, device=x.device).unsqueeze(0) - steps.unsqueeze(1)
        index = index.clamp_min(0)
        flat = index.reshape(-1)
        batch_offset = torch.arange(x.shape[0], device=x.device).repeat_interleave(k) * k
        gathered = x.reshape(x.shape[0] * k, *x.shape[2:])[batch_offset + flat]
        return gathered.reshape(x.shape)

    def forward(self, obs: Observation) -> Observation:
        if not self.training:
            return obs
        b = obs.batch_size
        device = obs.device
        steps = torch.randint(0, self.max_latency + 1, (b,), device=device)
        images = {k: self._shift(v, steps) for k, v in obs.images.items()}
        proprio = obs.proprio
        if proprio is not None:
            proprio = self._shift(proprio, steps).clone()
            drop = torch.rand(b, device=device) < FT_DROPOUT_PROBABILITY
            proprio[drop, :, self.ft_slice] = 0.0
        return Observation(
            images=images,
            proprio=proprio,
            task_index=obs.task_index,
            embodiment_index=obs.embodiment_index,
            partition_index=obs.partition_index,
        )


class B5DomainRandomisationPolicy(PortalTracePolicy):
    """B2's network, trained under randomised inputs.

    The network is deliberately identical to B2's: 10.5 puts B5's intervention on the
    data, and an architectural difference would confound the comparison it exists for.
    The strong augmentation setting replaces the nominal one rather than being added to
    it, so that the arm has one visual randomisation and not two.
    """

    arm = "B5-domain-randomisation"

    def __init__(self, cfg: PolicyConfig, *, max_latency: int = MAX_LATENCY_STEPS) -> None:
        super().__init__(cfg)
        if not cfg.augment:
            raise ValueError(
                "B5 is the domain-randomisation arm; building it with augment=False "
                "would leave it randomising latency and sensor availability only, "
                "which is not the baseline 10.5 lists"
            )
        self.randomise = InputRandomisation(max_latency=max_latency)
        # Twice the nominal shift budget; the jitter strengths are the augmentation's
        # own, which are already at the upper end of the DrQ-v2 recipe.
        self.encoder.visual.augment = ImageAugmentation(pad=8)

    def forward(self, batch: Batch) -> PolicyOutput:
        randomised = Batch(
            obs=self.randomise(batch.obs),
            actions=batch.actions,
            action_pad_mask=batch.action_pad_mask,
            c_true=batch.c_true,
            sample_weight=batch.sample_weight,
            phi_features=batch.phi_features,
            psi_basis=batch.psi_basis,
            extras=batch.extras,
        )
        return super().forward(randomised)

    def parameter_blocks(self) -> tuple[ParameterBlock, ...]:
        return dense_blocks(self, self.cfg.n_pool_embodiments)
