"""B10: `Lambda`-guided data curation.

**Proposal item.** Baseline B10 of 10.5, specified in 10.5.4. Three levels of
source weighting: down-weight by per-source `Lambda^part`, drop the frames with the
largest `||c_delta||`, or drop the offending source outright. Three levels x `S6` x one
seed, 18 runs.

**What it controls for.** The only data-side baseline, and the first thing a practitioner
actually does: not a two-way decomposition, but "leave it out / weight it down / drop the
high-`||c||` frames". That is the whole data-curation direction -- influence functions,
data selection -- and 10.5.4 expects it to be strong. It is also the most natural *use*
of the diagnostic: it turns `Lambda` from a measurement into a curation tool, which is a
good outcome for the paper rather than a threat, and if B10 does not lose on any endpoint
the text has to recommend it.

**Preregistered reading (10.5.4).** Frame-level curation beats B2 by at least 4
percentage points of success rate **and** is indistinguishable from B2 on
`Lambda^emp_res`: dropping frames cannot remove the common mode, because the common mode
is present in every frame at once.

**Provenance.** Implemented directly. The curation weights enter through
`Batch.sample_weight`, which `chunk_bc_loss` already honours; the frames are reweighted
rather than deleted so that the three levels differ only in their weight function and
the state distribution stays the pooled one where a weight is non-zero.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

import torch

from portal_trace.policies.interface import (
    Batch,
    PolicyConfig,
    PolicyOutput,
    RoutingKey,
)
from portal_trace.policies.pools import PoolSpec, SubsetName, pool
from portal_trace.policies.portal_policy import PortalTracePolicy

__all__ = [
    "POOL",
    "SUBSET",
    "SEEDS",
    "CurationLevel",
    "FRAME_DROP_QUANTILE",
    "SUCCESS_MARGIN_TARGET",
    "CurationRule",
    "fit_frame_threshold",
    "B10CuratedPoolingPolicy",
]

POOL: PoolSpec = pool("B2-like")
SUBSET = SubsetName.S6
SEEDS = 1

# 10.5.4: the top 20% of frames by ||c_delta(x)||.
FRAME_DROP_QUANTILE = 0.80
# The preregistered margin over B2, in percentage points of success rate.
SUCCESS_MARGIN_TARGET = 4.0


class CurationLevel(str, Enum):
    """The three levels of 10.5."""

    SOURCE_DOWNWEIGHT = "source_downweight"
    FRAME_DROP = "frame_drop"
    SOURCE_DROP = "source_drop"


@dataclass(frozen=True)
class CurationRule:
    """One level, with the numbers it needs to be applied.

    `source_lambda` is the per-source `Lambda^part` measured on a previous run -- B10 is
    downstream of the diagnostic by construction, and the values are a measurement, so
    there is no default. Index 0 is the target embodiment, whose own data is never
    curated: the arm curates *other people's* data, which is the practitioner's question.
    """

    level: CurationLevel
    source_lambda: tuple[float, ...]
    beta: float = 4.0
    drop_above: float = 0.5
    frame_threshold: float | None = None

    def __post_init__(self) -> None:
        if len(self.source_lambda) < 2:
            raise ValueError(
                "the pool has a target and at least one source; source_lambda covers "
                "every pool embodiment, target first"
            )
        if self.beta < 0.0:
            raise ValueError("beta is a non-negative sharpness")
        if self.level is CurationLevel.FRAME_DROP and self.frame_threshold is None:
            raise ValueError(
                "frame-level curation needs the ||c_delta|| threshold of the 80th "
                "percentile; fit it once over the training split with "
                "`fit_frame_threshold` and pass it, so that the same cut is applied to "
                "every batch"
            )

    def embodiment_weights(self, device: torch.device) -> torch.Tensor:
        """Weight per pool embodiment, target first."""
        lam = torch.tensor(self.source_lambda, dtype=torch.float32, device=device)
        if self.level is CurationLevel.SOURCE_DROP:
            weights = (lam <= self.drop_above).to(torch.float32)
        elif self.level is CurationLevel.SOURCE_DOWNWEIGHT:
            weights = torch.exp(-self.beta * lam.clamp_min(0.0))
        else:
            weights = torch.ones_like(lam)
        # The target's own data is never curated.
        weights[0] = 1.0
        return weights

    def weights(self, batch: Batch) -> torch.Tensor:
        """The per-sample weight this level assigns."""
        index = batch.obs.routing_index(RoutingKey.EMBODIMENT)
        table = self.embodiment_weights(index.device)
        if int(index.max()) >= table.numel():
            raise ValueError(
                f"embodiment index {int(index.max())} exceeds the "
                f"{table.numel()} entries of source_lambda"
            )
        weights = table[index]
        if self.level is CurationLevel.FRAME_DROP:
            assert self.frame_threshold is not None  # enforced in __post_init__
            magnitude = _chunk_magnitude(batch)
            weights = weights * (magnitude <= self.frame_threshold).to(weights.dtype)
        if batch.sample_weight is not None:
            weights = weights * batch.sample_weight.to(weights.dtype)
        return weights


def _chunk_magnitude(batch: Batch) -> torch.Tensor:
    """`||c_delta(x)||` per sample, over the chunk the sample supervises.

    The norm is taken over the whole chunk rather than the first step: the sample's
    supervision is the chunk, so a cut on the first step's magnitude would keep frames
    whose remaining seven steps carry most of the compensation.
    """
    c_true = batch.require_c_true()
    return c_true.flatten(1).norm(dim=1)


def fit_frame_threshold(
    magnitudes: Iterable[torch.Tensor], *, quantile: float = FRAME_DROP_QUANTILE
) -> float:
    """The `||c_delta||` cut, fitted once over the training split.

    Takes an iterable of per-batch magnitude vectors so that the pass over the pool does
    not have to fit in memory. The quantile is computed on the concatenation, which is
    the pooled marginal 10.5.4's "top 20% of frames" refers to.
    """
    collected = [m.detach().reshape(-1).to(torch.float32) for m in magnitudes]
    if not collected:
        raise ValueError("no magnitudes given; the threshold is a property of the pool")
    values = torch.cat(collected)
    if not 0.0 < quantile < 1.0:
        raise ValueError("the quantile lies strictly inside (0, 1)")
    return float(torch.quantile(values, quantile))


class B10CuratedPoolingPolicy(PortalTracePolicy):
    """B2's network on a reweighted pool.

    The reweighting is applied inside `forward` rather than by the sampler because the
    frame-level level needs `c_true`, which rides on the batch. A batch whose every
    sample is dropped raises from `chunk_bc_loss`: with a 20% cut that means the sampler
    handed over a batch drawn entirely from the tail, and silently skipping it would make
    the realised curation rate differ from the declared one.
    """

    arm = "B10-curated-pooling"

    def __init__(self, cfg: PolicyConfig, *, rule: CurationRule) -> None:
        super().__init__(cfg)
        if len(rule.source_lambda) != cfg.n_pool_embodiments:
            raise ValueError(
                f"source_lambda has {len(rule.source_lambda)} entries for "
                f"{cfg.n_pool_embodiments} pool embodiments"
            )
        self.rule = rule

    def forward(self, batch: Batch) -> PolicyOutput:
        if batch.actions is None:
            return super().forward(batch)
        curated = Batch(
            obs=batch.obs,
            actions=batch.actions,
            action_pad_mask=batch.action_pad_mask,
            c_true=batch.c_true,
            sample_weight=self.rule.weights(batch),
            phi_features=batch.phi_features,
            psi_basis=batch.psi_basis,
            extras=batch.extras,
        )
        return super().forward(curated)
