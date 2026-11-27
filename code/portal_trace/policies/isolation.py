"""E9's isolation ladder I0-I3 and the decisive control I3' (10.4.9).

The independent variable is the **parameter disjointness** `I`, written without a
threshold so that the main regression's covariate is reproducible. For a parameter
block `p` with `n_p` parameters, let `s_{k,p}` be the expected routing share of pool
embodiment `k` on that block (1 for a dense block, 0/1 for a hard router, the mean
gate weight for a soft one):

    I = 1 - [ sum_p n_p * ( C(K,2)^-1 * sum_{k<k'} min(s_kp, s_k'p) ) ]
            / [ sum_p n_p * mean_k s_kp ]

Fully shared gives 0, fully disjoint gives 1.

| Level | Implementation | Expected `I` |
| --- | --- | --- |
| I0 | fully shared dense | 0.00 |
| I1 | HB-MoE soft routing, vendored HiMoE | 0.28 |
| I2 | hard-routed disjoint action heads | 0.71 |
| I3 | per-embodiment fine-tuning | 1.00 |
| I3' | K disjoint modules split by a random pooled subset | 1.00 |

**I3' is the precondition for the crown.** I2 and I3 give each module only `1/K` of
the data, so the isolation axis also moves per-module data volume, and a rising
`Lambda` is fully explainable by low-data shrinkage whose direction is not
orthogonal to `c_delta` -- a confound pointing the same way as the prediction under
test. I3' matches I3 in parameter count, activated count and per-module data volume,
and differs only in whether the split key is embodiment identity. In this module that
difference is literally one enum value, `RoutingKey`. If the two read the same, E9
measured data fragmentation and T6 has no support.

**Capacity is matched on activated parameters only**, to +-10%, with the total
carried as a covariate. The two cannot be matched at once, and activated count is
what determines fitting ability. The match is exact by construction here rather than
searched for: every level keeps the same widths, and a hard-routed replica only
changes how many copies exist, not how many run.

`ladder_parameter_counts` is the function `diagnostics/isolation.py` consumes.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
from torch import nn

from portal_trace.policies.decoder import ActionChunkDecoder
from portal_trace.policies.encoders import ObservationEncoder
from portal_trace.policies.himoe_adapter import (
    MixtureFeedForward,
    action_space_descriptor,
    build_mixture_factory,
)
from portal_trace.policies.interface import (
    Batch,
    Observation,
    ParameterBlock,
    Policy,
    PolicyConfig,
    PolicyOutput,
    RoutingKey,
    dense_blocks,
)
from portal_trace.policies.losses import chunk_bc_loss
from portal_trace.policies.portal_policy import PortalTracePolicy

__all__ = [
    "IsolationLevel",
    "EXPECTED_ISOLATION_INDEX",
    "isolation_index",
    "measure_route_shares",
    "with_measured_shares",
    "SoftRoutedPolicy",
    "IsolationScope",
    "ParameterIsolatedPolicy",
    "LevelReport",
    "IsolationLadder",
    "ladder_parameter_counts",
]


class IsolationLevel(str, Enum):
    I0_DENSE = "I0"
    I1_SOFT_ROUTED = "I1"
    I2_HARD_ROUTED_HEADS = "I2"
    I3_PER_EMBODIMENT = "I3"
    I3_PRIME_RANDOM_PARTITION = "I3prime"


# 10.4.9's table. Preregistered expectations, not measurements: the realised index
# is computed from the built model and reported next to these.
EXPECTED_ISOLATION_INDEX: dict[IsolationLevel, float] = {
    IsolationLevel.I0_DENSE: 0.00,
    IsolationLevel.I1_SOFT_ROUTED: 0.28,
    IsolationLevel.I2_HARD_ROUTED_HEADS: 0.71,
    IsolationLevel.I3_PER_EMBODIMENT: 1.00,
    IsolationLevel.I3_PRIME_RANDOM_PARTITION: 1.00,
}

# 10.4.9 matches activated parameters to this tolerance.
ACTIVE_MATCH_TOLERANCE = 0.10


# --- the index ------------------------------------------------------------


def isolation_index(blocks: tuple[ParameterBlock, ...]) -> float:
    """`I` from a block decomposition, exactly as 10.4.9 writes it."""
    if not blocks:
        raise ValueError("the isolation index needs at least one parameter block")
    unmeasured = [b.name for b in blocks if b.route_shares is None]
    if unmeasured:
        raise ValueError(
            f"{len(unmeasured)} block(s) have no routing shares, first is "
            f"{unmeasured[0]!r}. These sit behind a stochastic router, so their "
            "s_{k,p} is what the router does on real per-embodiment data; call "
            "`measure_route_shares` with one batch per pool embodiment and pass the "
            "result through `with_measured_shares`. This package does not put a "
            "plausible number here"
        )
    widths = {len(b.route_shares) for b in blocks}  # type: ignore[arg-type]
    if len(widths) != 1:
        raise ValueError(f"blocks disagree on the number of pool embodiments: {widths}")
    k = widths.pop()
    if k < 2:
        raise ValueError(
            "the index compares pairs of embodiments; a pool of one has no pairs and "
            "10.4.9's covariate is undefined there"
        )
    n_p = np.array([b.total_params for b in blocks], dtype=np.float64)
    shares = np.array([b.route_shares for b in blocks], dtype=np.float64)  # (P, K)
    pairs = list(itertools.combinations(range(k), 2))
    overlap = np.mean(
        [np.minimum(shares[:, a], shares[:, b]) for a, b in pairs], axis=0
    )
    numerator = float(np.sum(n_p * overlap))
    denominator = float(np.sum(n_p * shares.mean(axis=1)))
    if denominator <= 0.0:
        raise ValueError(
            "every block has zero routing share on every embodiment; nothing in the "
            "model is on any forward path"
        )
    return 1.0 - numerator / denominator


def measure_route_shares(
    policy: nn.Module, batches: Mapping[int, Batch]
) -> dict[str, tuple[float, ...]]:
    """Run one batch per pool embodiment and read the routers' shares off the hooks.

    `batches` maps the pool embodiment index to a batch drawn from that embodiment's
    own frames -- `s_{k,p}` is an expectation under `rho_k`, so a mixed batch would
    measure something else. Evaluation mode, so that dropout does not add noise to a
    quantity that goes into a regression as a covariate.
    """
    mixtures = {
        name: module
        for name, module in policy.named_modules()
        if isinstance(module, MixtureFeedForward)
    }
    if not mixtures:
        raise ValueError(
            "this arm has no soft router, so its shares are known by construction and "
            "do not need measuring"
        )
    keys = sorted(batches)
    if not keys:
        raise ValueError("no batches given; the share table needs one per embodiment")
    per_expert: dict[str, list[np.ndarray]] = {name: [] for name in mixtures}
    was_training = policy.training
    policy.eval()
    try:
        with torch.no_grad():
            for k in keys:
                policy(batches[k])
                for name, module in mixtures.items():
                    shares = module.last_shares
                    if shares is None:
                        raise AssertionError(
                            f"{name} did not route on embodiment {k}; a mixture that is "
                            "never reached cannot be given a share"
                        )
                    per_expert[name].append(shares.cpu().numpy().copy())
    finally:
        policy.train(was_training)

    out: dict[str, tuple[float, ...]] = {}
    for name, rows in per_expert.items():
        table = np.stack(rows, axis=0)  # (K, n_experts)
        for expert in range(table.shape[1]):
            out[f"{name}.expert[{expert}]"] = tuple(float(v) for v in table[:, expert])
    return out


def with_measured_shares(
    blocks: tuple[ParameterBlock, ...], shares: Mapping[str, tuple[float, ...]]
) -> tuple[ParameterBlock, ...]:
    """Fill in the blocks whose shares had to be measured."""
    out: list[ParameterBlock] = []
    for block in blocks:
        if block.route_shares is not None:
            out.append(block)
            continue
        if block.name not in shares:
            raise KeyError(
                f"no measured routing shares for block {block.name!r}; "
                f"`measure_route_shares` returned {sorted(shares)[:4]}..."
            )
        out.append(
            ParameterBlock(
                name=block.name,
                total_params=block.total_params,
                active_params=block.active_params,
                route_shares=shares[block.name],
            )
        )
    return tuple(out)


# --- block decomposition of a decoder ------------------------------------


def decoder_blocks(
    decoder: ActionChunkDecoder, n_embodiments: int, *, prefix: str
) -> tuple[ParameterBlock, ...]:
    """Partition a decoder, delegating mixture layers to their own accounting."""
    blocks: list[ParameterBlock] = list(
        dense_blocks(decoder, n_embodiments, prefix=prefix, exclude=("layers",))
    )
    for i, layer in enumerate(decoder.layers):
        layer_prefix = f"{prefix}layers.{i}."
        blocks.extend(
            dense_blocks(
                layer, n_embodiments, prefix=layer_prefix, exclude=("feed_forward",)
            )
        )
        ffn = layer.feed_forward
        if isinstance(ffn, MixtureFeedForward):
            blocks.extend(
                ffn.parameter_blocks(f"{layer_prefix}feed_forward", n_embodiments)
            )
        else:
            blocks.extend(
                dense_blocks(
                    ffn, n_embodiments, prefix=f"{layer_prefix}feed_forward."
                )
            )
    return tuple(blocks)


# --- I1 -------------------------------------------------------------------


class SoftRoutedPolicy(PortalTracePolicy):
    """I1: the vendored HB-MoE and AS-MoE in the decoder's feed-forward blocks.

    Built on the vendored HiMoE-VLA action module (`third_party/himoe`), not
    reimplemented. AS-MoE sits at the boundary layers and HB-MoE at the adjacent
    inner ones, per the upstream placement rule; in this setting the action space is
    literally identical across the pool, so AS-MoE routes on a constant and the whole
    isolation comes from HB-MoE, which is what 10.4.9 says it should.
    """

    arm = "E9-I1-soft-routed"

    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__(
            cfg,
            feed_forward_factory=build_mixture_factory(
                d_model=cfg.d_model,
                ffn_dim=cfg.ffn_dim,
                dropout=cfg.dropout,
                n_layers=cfg.n_layers,
            ),
        )

    def action_space_descriptor(self, batch: Batch) -> torch.Tensor:
        return action_space_descriptor(
            batch.obs.batch_size, device=batch.obs.device
        )

    def parameter_blocks(self) -> tuple[ParameterBlock, ...]:
        blocks = list(dense_blocks(self, self.cfg.n_pool_embodiments, exclude=("decoder",)))
        blocks.extend(
            decoder_blocks(self.decoder, self.cfg.n_pool_embodiments, prefix="decoder.")
        )
        return tuple(blocks)


# --- I2, I3, I3' ----------------------------------------------------------


class IsolationScope(str, Enum):
    """How much of the network is replicated per routing key.

    `DECODER` is I2's disjoint action heads: the visual tower stays shared, the whole
    4-layer decoder head is replicated. `FULL` is I3's per-embodiment fine-tuning,
    where nothing is shared.
    """

    DECODER = "decoder"
    FULL = "full"


def _select(obs: Observation, mask: torch.Tensor) -> Observation:
    """The sub-batch of one routing key."""
    return Observation(
        images={k: v[mask] for k, v in obs.images.items()},
        proprio=None if obs.proprio is None else obs.proprio[mask],
        task_index=None if obs.task_index is None else obs.task_index[mask],
        embodiment_index=(
            None if obs.embodiment_index is None else obs.embodiment_index[mask]
        ),
        partition_index=(
            None if obs.partition_index is None else obs.partition_index[mask]
        ),
    )


class _Trunk(nn.Module):
    """Encoder plus decoder, the unit that gets replicated."""

    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__()
        self.encoder = ObservationEncoder(
            cfg.obs, cfg.d_model, cfg.n_tasks, augment=cfg.augment
        )
        self.decoder = ActionChunkDecoder(
            cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.dropout,
            action_dim=cfg.action_dim,
        )

    def forward(self, obs: Observation) -> tuple[torch.Tensor, torch.Tensor]:
        memory = self.encoder(obs.images, obs.proprio, obs.task_index)
        return self.decoder(memory)


class ParameterIsolatedPolicy(Policy):
    """Hard-routed disjoint modules: I2, I3 and I3'.

    Reimplemented from the proposal; there is no upstream to reuse, because the point
    of the level is the routing discipline rather than any published architecture.
    The three levels differ in exactly two enum values, which is the property 10.4.9
    depends on: I3 and I3' must be identical in every respect except the split key.

    Routing is deterministic and exclusive, so each module sees `1/K` of the pool and
    the confound this ladder is designed to separate is present by construction --
    which is why I3' exists.
    """

    def __init__(
        self,
        cfg: PolicyConfig,
        *,
        scope: IsolationScope,
        routing_key: RoutingKey,
        arm: str,
    ) -> None:
        self.arm = arm
        super().__init__(cfg)
        self.scope = scope
        self.routing_key = routing_key
        k = cfg.n_pool_embodiments
        if scope is IsolationScope.FULL:
            self.shared_encoder = None
            self.modules_per_key = nn.ModuleList(_Trunk(cfg) for _ in range(k))
        else:
            self.shared_encoder = ObservationEncoder(
                cfg.obs, cfg.d_model, cfg.n_tasks, augment=cfg.augment
            )
            self.modules_per_key = nn.ModuleList(
                ActionChunkDecoder(
                    cfg.d_model,
                    n_layers=cfg.n_layers,
                    n_heads=cfg.n_heads,
                    ffn_dim=cfg.ffn_dim,
                    dropout=cfg.dropout,
                    action_dim=cfg.action_dim,
                )
                for _ in range(k)
            )

    def _route(self, batch: Batch) -> torch.Tensor:
        idx = batch.obs.routing_index(self.routing_key)
        k = self.cfg.n_pool_embodiments
        if int(idx.min()) < 0 or int(idx.max()) >= k:
            raise ValueError(
                f"routing index out of range for {k} modules: saw "
                f"[{int(idx.min())}, {int(idx.max())}]"
            )
        return idx

    def forward(self, batch: Batch) -> PolicyOutput:
        idx = self._route(batch)
        obs = batch.obs
        memory = (
            None
            if self.shared_encoder is None
            else self.shared_encoder(obs.images, obs.proprio, obs.task_index)
        )
        prediction = torch.zeros(
            obs.batch_size, self.cfg.chunk, self.cfg.action_dim, device=obs.device
        )
        for key in idx.unique().tolist():
            mask = idx == key
            if self.scope is IsolationScope.FULL:
                chunk, _ = self.modules_per_key[key](_select(obs, mask))
            else:
                assert memory is not None
                chunk, _ = self.modules_per_key[key](memory[mask])
            prediction[mask] = chunk
        if batch.actions is None:
            return PolicyOutput(action_chunk=prediction)
        loss = chunk_bc_loss(
            prediction,
            batch.require_actions(),
            pad_mask=batch.action_pad_mask,
            sample_weight=batch.sample_weight,
        )
        return PolicyOutput(
            action_chunk=prediction, loss=loss, aux_losses={"bc": loss.detach()}
        )

    def parameter_blocks(self) -> tuple[ParameterBlock, ...]:
        k = self.cfg.n_pool_embodiments
        blocks: list[ParameterBlock] = list(
            dense_blocks(self, k, exclude=("modules_per_key",))
        )
        for i, module in enumerate(self.modules_per_key):
            n = sum(p.numel() for p in module.parameters())
            blocks.append(
                ParameterBlock(
                    name=f"module[{i}]",
                    total_params=n,
                    # Exactly one replica runs per sample; billing the first keeps the
                    # activated total right and the choice is free because the replicas
                    # are the same size.
                    active_params=n if i == 0 else 0,
                    route_shares=tuple(1.0 if j == i else 0.0 for j in range(k)),
                )
            )
        return tuple(blocks)


# --- the ladder -----------------------------------------------------------


@dataclass(frozen=True)
class LevelReport:
    """One row of the per-level table 10.4.9 requires to be published."""

    level: IsolationLevel
    arm: str
    active_params: int
    total_params: int
    expected_index: float
    measured_index: float | None
    active_ratio_to_i0: float

    @property
    def capacity_matched(self) -> bool:
        return abs(self.active_ratio_to_i0 - 1.0) <= ACTIVE_MATCH_TOLERANCE

    def as_row(self) -> dict[str, object]:
        return {
            "level": self.level.value,
            "arm": self.arm,
            "active_params": self.active_params,
            "total_params": self.total_params,
            "expected_I": self.expected_index,
            "measured_I": self.measured_index,
            "active_ratio_to_I0": round(self.active_ratio_to_i0, 4),
            "capacity_matched": self.capacity_matched,
        }


class IsolationLadder:
    """Builds the five levels from one config and reports their capacity."""

    def __init__(self, cfg: PolicyConfig) -> None:
        if cfg.n_pool_embodiments < 2:
            raise ValueError(
                "the ladder routes on pool embodiment identity; a pool of one has no "
                "isolation axis"
            )
        self.cfg = cfg

    def build(self, level: IsolationLevel) -> Policy:
        cfg = self.cfg.evolve(arm=f"E9-{level.value}")
        if level is IsolationLevel.I0_DENSE:
            return PortalTracePolicy(cfg)
        if level is IsolationLevel.I1_SOFT_ROUTED:
            return SoftRoutedPolicy(cfg)
        if level is IsolationLevel.I2_HARD_ROUTED_HEADS:
            return ParameterIsolatedPolicy(
                cfg,
                scope=IsolationScope.DECODER,
                routing_key=RoutingKey.EMBODIMENT,
                arm="E9-I2-hard-routed-heads",
            )
        if level is IsolationLevel.I3_PER_EMBODIMENT:
            return ParameterIsolatedPolicy(
                cfg,
                scope=IsolationScope.FULL,
                routing_key=RoutingKey.EMBODIMENT,
                arm="E9-I3-per-embodiment",
            )
        return ParameterIsolatedPolicy(
            cfg,
            scope=IsolationScope.FULL,
            routing_key=RoutingKey.RANDOM_PARTITION,
            arm="E9-I3prime-random-partition",
        )

    def report(
        self, *, measured_shares: Mapping[str, tuple[float, ...]] | None = None
    ) -> tuple[LevelReport, ...]:
        """Per-level activated and total counts, plus the realised index.

        `measured_index` is `None` only for I1 without measured routing shares: a soft
        router's index is a property of what it does on data, and 10.4.9 uses it as a
        regression covariate.
        """
        rows: list[LevelReport] = []
        i0_active: int | None = None
        for level in IsolationLevel:
            policy = self.build(level)
            report = policy.parameter_report()
            blocks = report.blocks
            if measured_shares is not None:
                blocks = with_measured_shares(blocks, measured_shares)
            # None for a soft router whose shares have not been measured; the index is
            # then a property of data this function was not given, and 10.4.9's
            # covariate is not allowed to be filled in with a plausible value.
            index = (
                None
                if any(b.route_shares is None for b in blocks)
                else isolation_index(blocks)
            )
            if i0_active is None:
                i0_active = report.active
            rows.append(
                LevelReport(
                    level=level,
                    arm=report.arm,
                    active_params=report.active,
                    total_params=report.total,
                    expected_index=EXPECTED_ISOLATION_INDEX[level],
                    measured_index=index,
                    active_ratio_to_i0=report.active / i0_active,
                )
            )
            # I3 and I3' hold K full replicas; dropping each level before building the
            # next keeps one of them in memory at a time rather than two.
            del policy
        return tuple(rows)

    def check_capacity_match(self, rows: tuple[LevelReport, ...]) -> None:
        """Raise if a level breaks the +-10% activated-parameter match of 10.4.9."""
        bad = [r for r in rows if not r.capacity_matched]
        if bad:
            listed = ", ".join(
                f"{r.level.value} at {r.active_ratio_to_i0:.3f} of I0" for r in bad
            )
            raise ValueError(
                f"activated parameters are not matched to +-{ACTIVE_MATCH_TOLERANCE:.0%}: "
                f"{listed}. 10.4.9 preregisters that caliber because activated count "
                "determines fitting ability, and an unmatched level makes the "
                "Lambda-vs-I slope confounded with capacity"
            )


def ladder_parameter_counts(cfg: PolicyConfig) -> dict[str, dict[str, int]]:
    """Per-level activated and total parameter counts.

    The entry point `diagnostics/isolation.py` consumes: 10.4.9 puts the activated
    count in the capacity-matching check and `log(total)` in the mixed model as a
    covariate, so both are returned per level, keyed by the level name.
    """
    ladder = IsolationLadder(cfg)
    return {
        row.level.value: {
            "active_params": row.active_params,
            "total_params": row.total_params,
        }
        for row in ladder.report()
    }
