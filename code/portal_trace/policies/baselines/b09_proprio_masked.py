"""B9: "the leakage rides on the embodiment cues".

**Proposal item.** Baseline B9 of 10.5. Mask the proprioceptive channels, keeping RGB
and the task instruction, as the interventional test of `Lambda -> E_rho[w]`. Subset
`S9`, two seeds, 18 runs. It is one of the two arms the 14.3 falsification table uses
for "is the policy fitting the pooled conditional mean", with the preregistered reading
that masking must shrink the statistic by at least 60%.

**What it controls for.** The channel the leakage enters through. If flattening the
source posterior collapses `Lambda`, the leakage was carried by the embodiment-revealing
input; if it does not, the leakage is in the trunk's shared structure. It is expected to
lose success rate, because proprioception is genuinely useful, and that loss is not a
problem for its role: the reading is about the statistic, not about the return.
Complementary to ablation A1, which sweeps the modality rather than intervening on it.

**Why the intervention is structural and not a mask of zeros.** With `use_proprio=False`
the encoder has no proprioception pathway at all: no first layer, no bias. Feeding zeros
into an existing pathway would leave that layer's bias free to carry a per-embodiment
offset learned during training, which is exactly the channel the intervention is
supposed to remove. `ObsSpec` keeps `k_hist` at its main value so the image history is
untouched and B9 remains one intervention rather than two.

**Provenance.** Implemented directly; the intervention is a configuration of the shared
10.4.0 encoder.
"""

from __future__ import annotations

from dataclasses import replace

from portal_trace.policies.interface import PolicyConfig
from portal_trace.policies.pools import PoolSpec, SubsetName, pool
from portal_trace.policies.portal_policy import PortalTracePolicy

__all__ = [
    "POOL",
    "SUBSET",
    "SEEDS",
    "SHRINKAGE_TARGET",
    "masked_config",
    "B9ProprioMaskedPolicy",
]

POOL: PoolSpec = pool("B2-like")
SUBSET = SubsetName.S9
SEEDS = 2

# 14.3's preregistered reading: masking must collapse the statistic by at least 60%.
SHRINKAGE_TARGET = 0.60


def masked_config(cfg: PolicyConfig) -> PolicyConfig:
    """The same config with the proprioception pathway removed."""
    return cfg.evolve(obs=replace(cfg.obs, use_proprio=False))


class B9ProprioMaskedPolicy(PortalTracePolicy):
    """B2's network with no proprioception pathway.

    The config is rewritten in the constructor rather than expected from the caller: an
    arm named B9 that was handed a proprioception-carrying config would build the arm it
    is the control for.
    """

    arm = "B9-proprio-masked"

    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__(masked_config(cfg))
