"""The one interface `runners/` and `eval/` see.

Four members, per the integration contract: build from a config, `forward`,
`act`, `parameter_report`. Everything an arm needs beyond that -- pool
composition, gradient surgery, mixture routing -- is declared on the config or
carried on the batch, so that a runner can drive B1 and E9's I3' through the same
loop.

Two design points are load bearing.

`ParameterBlock` exists because 10.4.9 defines the isolation covariate as a
parameter-weighted routing statistic: `I` needs the parameter count `n_p` of
every block together with the per-embodiment routing share `s_{k,p}`. Making
every arm publish that decomposition is what lets `diagnostics/isolation.py`
compute the covariate without knowing which arm it is looking at.

`Policy` has no termination output. 9.6.1 fixes `horizon=400` with early
stopping off and `num_samples` always 400, so a policy that could say "done"
would be a way to reintroduce it; see `schedule.py`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn

from portal_trace.policies.obs import ACTION_DIM, CHUNK, ObsSpec

__all__ = [
    "Observation",
    "Batch",
    "PolicyOutput",
    "PolicyConfig",
    "ParameterBlock",
    "ParameterReport",
    "Policy",
    "RoutingKey",
    "dense_blocks",
    "dense_block",
    "hard_routed_block",
    "hard_routed_blocks",
]


class RoutingKey(str, Enum):
    """What a hard-routed module keys on.

    The two members are the whole content of E9's decisive control: I3 routes on
    embodiment identity, I3' routes on a random partition of the same pool with
    the same module count and the same per-module data volume (10.4.9). If the
    two arms read the same, E9 measured data fragmentation and T6 has no support.
    """

    EMBODIMENT = "embodiment"
    RANDOM_PARTITION = "random_partition"


@dataclass
class Observation:
    """One batch of policy inputs.

    Images arrive as float tensors in [0, 1], shape (B, K, 3, 84, 84) per camera;
    proprioception as (B, K, 70). Both are already normalised under the 9.6.2
    global caliber by the data pipeline -- the policy never owns the statistics,
    because they are computed across every family, defect and task and would
    otherwise be recomputed per arm.
    """

    images: dict[str, torch.Tensor]
    proprio: torch.Tensor | None = None
    task_index: torch.Tensor | None = None
    embodiment_index: torch.Tensor | None = None
    partition_index: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return next(iter(self.images.values())).shape[0]

    @property
    def device(self) -> torch.device:
        return next(iter(self.images.values())).device

    def routing_index(self, key: RoutingKey) -> torch.Tensor:
        """The integer each hard-routed block routes on."""
        if key is RoutingKey.EMBODIMENT:
            idx = self.embodiment_index
            what = "embodiment_index"
        else:
            idx = self.partition_index
            what = "partition_index"
        if idx is None:
            raise ValueError(
                f"routing on {key.value} needs Observation.{what}; the runner must "
                "supply it for every frame in the pool"
            )
        return idx


@dataclass
class Batch:
    """Inputs plus whatever supervision the arm consumes.

    `c_true` is the per-frame compensation ground truth of Definition 3.1, and it
    is the reason two of these fields exist at all: B8 subtracts it from the
    action label, and E4's endpoint is the cosine between a head output and it.
    `phi_features` and `psi_basis` are the 9.4.4 state feature basis and the
    physical basis; they are computed from the simulator state by the teacher, so
    they ride on the batch rather than being recomputed inside a head. Both carry
    a chunk axis: the compensation E4 measures is a field on the eight states the
    chunk covers, not on the current one.
    """

    obs: Observation
    actions: torch.Tensor | None = None            # (B, CHUNK, 7), normalised
    action_pad_mask: torch.Tensor | None = None    # (B, CHUNK) bool, True = real frame
    c_true: torch.Tensor | None = None             # (B, CHUNK, 7)
    sample_weight: torch.Tensor | None = None      # (B,), B10's curation weights
    phi_features: torch.Tensor | None = None       # (B, CHUNK, P)
    psi_basis: torch.Tensor | None = None          # (B, CHUNK, n_psi, 6)
    extras: dict[str, torch.Tensor] = field(default_factory=dict)

    def require_actions(self) -> torch.Tensor:
        if self.actions is None:
            raise ValueError("this arm's loss needs Batch.actions")
        return self.actions

    def require_c_true(self) -> torch.Tensor:
        if self.c_true is None:
            raise ValueError(
                "this arm needs Batch.c_true, the per-frame compensation ground "
                "truth from the hdf5 `c_true` dataset (9.6.3)"
            )
        return self.c_true


@dataclass
class PolicyOutput:
    """What every arm returns.

    `loss` is the scalar to call `.backward()` on and already contains every
    auxiliary term; `aux_losses` reports the terms separately so a run can log the
    load-balancing loss, the mutual-information loss and the reconstruction loss
    without recomputing them.
    """

    action_chunk: torch.Tensor                      # (B, CHUNK, 7)
    loss: torch.Tensor | None = None
    aux_losses: dict[str, torch.Tensor] = field(default_factory=dict)
    extras: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyConfig:
    """Everything the 10.4.0 architecture and its variants are built from.

    Defaults are the values 10.4.0 writes down, except `n_tasks`, which follows
    from the registry, and `n_pool_embodiments`, which follows from the 10.4.0
    data-composition table (one target plus K = 6 sources).
    """

    arm: str
    obs: ObsSpec = field(default_factory=ObsSpec)
    n_tasks: int = 6                    # the six core tasks a policy is evaluated on
    n_pool_embodiments: int = 7         # 1 target + K = 6 sources, 10.4.0
    d_model: int = 512
    n_layers: int = 4                   # 10.4.0: 4-layer transformer decoder head
    n_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1
    augment: bool = True                # random crop + colour jitter, 10.4.0
    options: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.n_layers != 4:
            # A4 sweeps depth; every other section reads 4 layers as part of the
            # endpoint definition, so a non-default depth has to be deliberate.
            if self.n_layers < 1:
                raise ValueError("a transformer head needs at least one layer")
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model={self.d_model} is not divisible by n_heads={self.n_heads}"
            )
        if self.n_tasks < 1:
            raise ValueError("a policy conditions on at least one task")
        if self.n_pool_embodiments < 1:
            raise ValueError("a pool holds at least one embodiment")

    @property
    def chunk(self) -> int:
        return CHUNK

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    def evolve(self, **changes: Any) -> PolicyConfig:
        return replace(self, **changes)


@dataclass(frozen=True)
class ParameterBlock:
    """One block `p` of the 10.4.9 isolation index.

    `route_shares` is `s_{k,p}`, the expected routing weight of pool embodiment
    `k` on this block: 1 for a dense block, an indicator for a hard router, and
    the mean gate weight for a soft router. It is `None` exactly when the router
    is stochastic and the shares have to be measured on real per-embodiment
    batches; nothing in this package substitutes a guess for them.
    """

    name: str
    total_params: int
    active_params: int
    route_shares: tuple[float, ...] | None

    def __post_init__(self) -> None:
        if self.active_params > self.total_params:
            raise ValueError(
                f"block {self.name}: {self.active_params} active exceeds "
                f"{self.total_params} total"
            )
        if self.route_shares is not None and any(
            s < 0.0 or s > 1.0 for s in self.route_shares
        ):
            raise ValueError(f"block {self.name}: routing shares must lie in [0, 1]")


@dataclass(frozen=True)
class ParameterReport:
    """Capacity accounting, in the caliber 10.4.9 preregisters.

    E9 matches *activated* parameters to +-10% and carries the total as a
    covariate, because the two cannot be matched at once. Both numbers are
    published here and go into the results table per that section.
    """

    arm: str
    total: int
    trainable: int
    frozen: int
    active: int
    blocks: tuple[ParameterBlock, ...]

    def as_row(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "total": self.total,
            "trainable": self.trainable,
            "frozen": self.frozen,
            "active": self.active,
            "n_blocks": len(self.blocks),
        }


class Policy(nn.Module, abc.ABC):
    """Base class for every arm in 10.4-10.6.

    Subclasses implement `forward` and `parameter_blocks`. `act` and
    `parameter_report` are derived and should not be overridden unless the arm
    genuinely changes what they mean -- B11 overrides neither, it only freezes
    part of itself.

    There is deliberately no `reset`, no `is_terminal` and no `done`: the fixed
    400-step horizon of 9.6.1 is a property of the rollout, and an arm that could
    signal termination would be a way to put early stopping back.
    """

    arm: ClassVar[str] = ""

    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__()
        if not self.arm:
            raise TypeError(f"{type(self).__name__} must declare a class-level `arm`")
        self.cfg = cfg

    @classmethod
    def from_config(cls, cfg: PolicyConfig) -> Policy:
        return cls(cfg)

    @abc.abstractmethod
    def forward(self, batch: Batch) -> PolicyOutput:
        """Predict the action chunk and, when the batch carries labels, the loss."""

    @abc.abstractmethod
    def parameter_blocks(self) -> tuple[ParameterBlock, ...]:
        """The block decomposition 10.4.9's isolation index is computed from."""

    @torch.no_grad()
    def act(self, obs: Observation) -> np.ndarray:
        """One action chunk, shape (B, CHUNK, 7), in normalised action units.

        Temporal ensembling and the choice of how many of the eight steps to
        execute belong to the rollout loop, not here: E9 and E4b compare arms on
        the same execution policy and it has to be defined in one place.
        """
        was_training = self.training
        self.eval()
        try:
            out = self.forward(Batch(obs=obs))
        finally:
            self.train(was_training)
        return out.action_chunk.detach().cpu().numpy()

    def parameter_report(self) -> ParameterReport:
        blocks = self.parameter_blocks()
        counted = sum(b.total_params for b in blocks)
        total = sum(p.numel() for p in self.parameters())
        if counted != total:
            raise AssertionError(
                f"{self.arm}: parameter blocks sum to {counted} but the module holds "
                f"{total}; the block decomposition must partition the parameters, "
                "otherwise the isolation index of 10.4.9 is computed on the wrong "
                "denominator"
            )
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return ParameterReport(
            arm=self.arm,
            total=total,
            trainable=trainable,
            frozen=total - trainable,
            active=sum(b.active_params for b in blocks),
            blocks=blocks,
        )


def dense_blocks(
    module: nn.Module,
    n_embodiments: int,
    *,
    prefix: str = "",
    exclude: tuple[str, ...] = (),
) -> tuple[ParameterBlock, ...]:
    """One dense block per direct child, plus one for parameters held directly.

    Used so that `parameter_report`'s partition check is satisfied by construction:
    a subclass that routes some child hard lists it in `exclude` and appends its
    own routed blocks, and forgetting to do so raises instead of quietly
    misreporting the isolation denominator.
    """
    blocks: list[ParameterBlock] = []
    direct = sum(p.numel() for p in module.parameters(recurse=False))
    if direct:
        blocks.append(
            ParameterBlock(f"{prefix}<direct>", direct, direct, (1.0,) * n_embodiments)
        )
    for name, child in module.named_children():
        if name in exclude:
            continue
        n = sum(p.numel() for p in child.parameters())
        if n:
            blocks.append(
                ParameterBlock(f"{prefix}{name}", n, n, (1.0,) * n_embodiments)
            )
    return tuple(blocks)


def dense_block(name: str, module: nn.Module, n_embodiments: int) -> ParameterBlock:
    """A block every embodiment routes through: `s_{k,p} = 1` for all k."""
    n = sum(p.numel() for p in module.parameters())
    return ParameterBlock(
        name=name,
        total_params=n,
        active_params=n,
        route_shares=(1.0,) * n_embodiments,
    )


def hard_routed_block(
    name: str, modules: nn.ModuleList, n_embodiments: int, module_index: int
) -> ParameterBlock:
    """One replica of a hard-routed block: `s_{k,p} = 1[k == module_index]`.

    The replica is billed its own parameters as total and as active, and the
    caller is expected to emit one block per replica; summing the replicas gives
    the total, while only the routed one contributes to the active count, which is
    handled by `hard_routed_blocks`.
    """
    n = sum(p.numel() for p in modules[module_index].parameters())
    shares = tuple(1.0 if k == module_index else 0.0 for k in range(n_embodiments))
    return ParameterBlock(name=name, total_params=n, active_params=0, route_shares=shares)


def hard_routed_blocks(
    name: str, modules: nn.ModuleList, n_embodiments: int
) -> tuple[ParameterBlock, ...]:
    """All replicas of a hard-routed block, with exactly one billed as active.

    Which replica is billed is arbitrary -- they are the same size -- and the
    choice does not affect the isolation index, which reads `route_shares`.
    """
    if len(modules) != n_embodiments:
        raise ValueError(
            f"{name}: {len(modules)} replicas for {n_embodiments} pool embodiments; "
            "a hard router needs one module per routing key"
        )
    out: list[ParameterBlock] = []
    for i in range(n_embodiments):
        blk = hard_routed_block(f"{name}[{i}]", modules, n_embodiments, i)
        if i == 0:
            blk = ParameterBlock(
                name=blk.name,
                total_params=blk.total_params,
                active_params=blk.total_params,
                route_shares=blk.route_shares,
            )
        out.append(blk)
    return tuple(out)
