"""Thin adapter over the vendored HiMoE-VLA action module (E9's level I1).

The mixture modules themselves are not reimplemented: `third_party/himoe` holds
`ASMoE`, `HBMoE` and their two routers verbatim from
<https://github.com/ZhiyingDu/HiMoE-VLA> at commit
`27a2c46932d8b6373ca0074eb997f299bcd4f6f5`, with provenance in
`third_party/README.md`. 10.4.9 uses the HB-MoE router as the continuous isolation
covariate, so reimplementing it would put E9's regressor on a different router from
the published one.

What this adapter does, all of it from the outside:

1. Puts the vendored tree on `sys.path`, because `himoe.py` imports
   `moevla.models.modeling_moe` absolutely and editing that import would fork the
   file.
2. Supplies router configurations sized so that the *activated* parameter count of
   a mixture block matches the dense feed-forward block it replaces, which is the
   capacity-matching caliber 10.4.9 preregisters (activated matched to +-10%, total
   carried as a covariate).
3. Applies the upstream layer-placement rule, read off `HiMoEDecoderLayer.__init__`
   in the vendored `himoe.py`.
4. Records per-expert routing shares through a forward hook on the gate, because the
   isolation index of 10.4.9 needs the whole share table `s_{k,p}` and the vendored
   gates return only the top-k selection.

**One dtype accommodation.** `MoEGate_mutual_info`, the AS-Reg router, casts its
input to bfloat16 unconditionally, so its weight has to be bfloat16 too or
`F.linear` raises. The adapter casts that one submodule and leaves everything else
in float32. The cost is a low-precision router in the AS branch, which in this
setting routes on a constant (see `ACTION_SPACE_DESCRIPTOR`) and therefore has
nothing to resolve.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
from torch import nn

from portal_trace.policies.interface import ParameterBlock
from portal_trace.policies.obs import ACTION_DIM, ARM_ACTION_DIM
from portal_trace.registry.tasks import CONTROL_FREQ

_VENDOR_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "himoe"
_VENDOR_MODULE = _VENDOR_ROOT / "moevla" / "models" / "modeling_moe.py"

if not _VENDOR_MODULE.is_file():
    raise ImportError(
        f"vendored HiMoE-VLA action module not found at {_VENDOR_MODULE}. It is "
        "checked in under third_party/himoe; see third_party/README.md for the "
        "upstream URL and commit if it has to be re-fetched. E9's I1 level cannot be "
        "built without it"
    )
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from moevla.models.modeling_moe import ASMoE, HBMoE  # noqa: E402

__all__ = [
    "MixtureKind",
    "ASMoESettings",
    "HBMoESettings",
    "MixtureFeedForward",
    "ACTION_SPACE_DESCRIPTOR_DIM",
    "action_space_descriptor",
    "himoe_layer_plan",
    "build_mixture_factory",
]


class MixtureKind(str, Enum):
    """Which of the two HiMoE mixtures a layer carries.

    `AS` handles action-space heterogeneity and carries the contrastive AS-Reg;
    `HB` handles the remaining embodiment heterogeneity and carries the
    load-balancing HB-Reg; `DENSE` is the shared transformer block between them.
    """

    AS = "as_moe"
    HB = "hb_moe"
    DENSE = "dense"


# --- the action-space descriptor AS-MoE routes on -------------------------

# 10.4.0 fixes one action space for the whole main setting: OSC_POSE's six
# incremental pose dimensions plus one gripper dimension, all in [-1, 1], at 20 Hz,
# with output_max = [0.05 m, 0.5 rad]. The descriptor below is therefore a constant
# across the pool, which is exactly what 10.4.9 says: "the action space is literally
# identical, so the AS-MoE branch carries no action-space difference and the whole
# isolation comes from HB-MoE and the head structure". Keeping the branch with a
# constant descriptor rather than deleting it preserves the published structure and
# makes the degeneracy a measured routing share instead of an assumption.
ACTION_SPACE_DESCRIPTOR: tuple[float, ...] = (
    1.0,                       # task-space control (as opposed to joint-space)
    0.0,                       # joint-space control
    ARM_ACTION_DIM / 7.0,      # arm action width, normalised by the maximum DoF
    (ACTION_DIM - ARM_ACTION_DIM) / 7.0,   # gripper width
    CONTROL_FREQ / 100.0,      # control frequency
    0.05,                      # output_max, translation, metres per action unit
    0.5,                       # output_max, rotation, radians per action unit
    1.0,                       # input range is symmetric [-1, 1]
)
ACTION_SPACE_DESCRIPTOR_DIM = len(ACTION_SPACE_DESCRIPTOR)


def action_space_descriptor(
    batch_size: int, *, device: torch.device | None = None
) -> torch.Tensor:
    """The (B, D) descriptor AS-MoE's router reads.

    Materialised rather than expanded: the vendored router broadcasts it over the
    sequence and then flattens with `view`, which needs real strides.
    """
    return torch.tensor(
        ACTION_SPACE_DESCRIPTOR, dtype=torch.float32, device=device
    ).repeat(batch_size, 1)


# --- router settings, sized against the dense block ----------------------


def _dense_ffn_params(d_model: int, ffn_dim: int) -> int:
    """Parameters of the block `DenseFeedForward` puts in a layer, biases included."""
    return 2 * d_model * ffn_dim + ffn_dim + d_model


def _expert_width(
    *, target_params: int, d_model: int, n_active_experts: int, gate_params: int
) -> int:
    """Expert intermediate width that matches `target_params` when active.

    An expert is three bias-free `d_model x m` projections, so the activated count is
    `3 * d_model * m * n_active_experts + gate_params`. Solving for `m` is exact; no
    search is needed, and the caller asserts the +-10% of 10.4.9 afterwards.
    """
    per_expert = 3 * d_model * n_active_experts
    width = (target_params - gate_params) // per_expert
    if width < 1:
        raise ValueError(
            f"cannot match {target_params} activated parameters with "
            f"{n_active_experts} active experts of width >= 1 at d_model={d_model}"
        )
    return width


@dataclass
class HBMoESettings:
    """Configuration object for the vendored `HBMoE`.

    Duck-typed against upstream's `HBMoEConfig`: the vendored module reads these
    attribute names off whatever it is handed. Written as a dataclass here so that
    the sizing arithmetic is visible and the defaults are ours rather than upstream's
    1024-wide Gemma ones.
    """

    hidden_size: int
    moe_intermediate_size: int
    n_routed_experts: int = 8
    num_experts_per_tok: int = 2
    n_shared_experts: int | None = 1
    hidden_act: str = "silu"
    pretraining_tp: int = 1
    scoring_func: str = "softmax"
    aux_loss_alpha: float = 0.001    # HB-Reg strength, upstream's value
    seq_aux: bool = True
    norm_topk_prob: bool = True

    @classmethod
    def matched_to_dense(
        cls, *, d_model: int, ffn_dim: int, n_routed_experts: int = 8, top_k: int = 2
    ) -> HBMoESettings:
        gate = n_routed_experts * d_model
        width = _expert_width(
            target_params=_dense_ffn_params(d_model, ffn_dim),
            d_model=d_model,
            n_active_experts=top_k + 1,   # the shared expert is always on the path
            gate_params=gate,
        )
        return cls(
            hidden_size=d_model,
            moe_intermediate_size=width,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=top_k,
        )

    @property
    def n_active_experts(self) -> int:
        return self.num_experts_per_tok + (self.n_shared_experts or 0)


@dataclass
class ASMoESettings:
    """Configuration object for the vendored `ASMoE`.

    `condition_dim` is the width of the action-space descriptor its router gates on,
    and upstream's default of no shared expert is kept: AS-MoE is a pure top-1 router
    over action spaces.
    """

    hidden_size: int
    moe_intermediate_size: int
    condition_dim: int = ACTION_SPACE_DESCRIPTOR_DIM
    n_routed_experts: int = 3
    num_experts_per_tok: int = 1
    n_shared_experts: int | None = None
    hidden_act: str = "silu"
    pretraining_tp: int = 1
    scoring_func: str = "softmax"
    aux_loss_alpha: float = 0.002    # AS-Reg strength, upstream's value
    seq_aux: bool = True
    norm_topk_prob: bool = True

    @classmethod
    def matched_to_dense(
        cls, *, d_model: int, ffn_dim: int, n_routed_experts: int = 3
    ) -> ASMoESettings:
        gate = n_routed_experts * ACTION_SPACE_DESCRIPTOR_DIM
        width = _expert_width(
            target_params=_dense_ffn_params(d_model, ffn_dim),
            d_model=d_model,
            n_active_experts=1,
            gate_params=gate,
        )
        return cls(
            hidden_size=d_model,
            moe_intermediate_size=width,
            n_routed_experts=n_routed_experts,
        )

    @property
    def n_active_experts(self) -> int:
        return self.num_experts_per_tok


# --- layer placement ------------------------------------------------------


def himoe_layer_plan(n_layers: int) -> tuple[MixtureKind, ...]:
    """Which mixture each decoder layer carries.

    The upstream rule, verbatim from `HiMoEDecoderLayer.__init__` in the vendored
    `himoe.py`: AS-MoE where `layer_idx <= 1 or layer_idx >= L - 2`, HB-MoE where
    `layer_idx <= 5 or layer_idx >= L - 6`, dense in between. Written for the deep
    Gemma action expert it was published on.

    10.4.0's head is four layers, and there the predicate marks every layer AS-MoE:
    the two boundary bands meet in the middle and HB-MoE gets nothing. That is not a
    faithful I1 -- 10.4.9 requires the isolation to come from HB-MoE -- so for stacks
    too shallow for the upstream bands the rule is applied at its intent: one AS-MoE
    layer at each boundary, HB-MoE on every layer between them. The two regimes are
    both returned by this function so that the depth at which the plan changes is
    visible and reported with the results rather than buried in a config.
    """
    if n_layers < 3:
        raise ValueError(
            f"n_layers={n_layers} leaves no inner layer for HB-MoE; E9's I1 needs at "
            "least one boundary layer at each end and one layer between them"
        )
    upstream = tuple(
        MixtureKind.AS
        if (i <= 1 or i >= n_layers - 2)
        else MixtureKind.HB
        if (i <= 5 or i >= n_layers - 6)
        else MixtureKind.DENSE
        for i in range(n_layers)
    )
    if MixtureKind.HB in upstream:
        return upstream
    return tuple(
        MixtureKind.AS if i in (0, n_layers - 1) else MixtureKind.HB
        for i in range(n_layers)
    )


# --- the wrapper ----------------------------------------------------------


class MixtureFeedForward(nn.Module):
    """A vendored mixture behind the decoder's feed-forward interface.

    Also the measurement point for `s_{k,p}`: the hook on the gate turns the top-k
    selection the vendored router returns into a full per-expert share vector, which
    is what the isolation index of 10.4.9 integrates over. Shares are recorded, never
    assumed; `last_shares` is `None` until a forward pass has happened.
    """

    def __init__(self, kind: MixtureKind, settings: ASMoESettings | HBMoESettings) -> None:
        super().__init__()
        if kind is MixtureKind.DENSE:
            raise ValueError("a dense layer does not need this wrapper")
        self.kind = kind
        self.settings = settings
        self.n_experts = settings.n_routed_experts
        if kind is MixtureKind.AS:
            self.mixture: nn.Module = ASMoE(settings)
            # See the module docstring: MoEGate_mutual_info casts its input to
            # bfloat16, so its weight must follow or F.linear raises.
            self.mixture.gate.to(torch.bfloat16)
        else:
            self.mixture = HBMoE(settings)
        self._last_shares: torch.Tensor | None = None
        self.mixture.gate.register_forward_hook(self._record)

    def _record(self, module: nn.Module, args: tuple, output: tuple) -> None:  # noqa: ANN001
        del module, args
        topk_idx, topk_weight, _aux = output
        idx = topk_idx.reshape(-1)
        weight = topk_weight.reshape(-1).to(torch.float32)
        shares = torch.zeros(self.n_experts, dtype=torch.float32, device=weight.device)
        shares.scatter_add_(0, idx, weight)
        n_tokens = topk_idx.shape[0]
        self._last_shares = (shares / max(n_tokens, 1)).detach()

    @property
    def last_shares(self) -> torch.Tensor | None:
        """Mean routing weight per routed expert on the most recent forward pass."""
        return self._last_shares

    def forward(
        self, hidden_states: torch.Tensor, data_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.kind is MixtureKind.AS:
            if data_mask is None:
                raise ValueError(
                    "AS-MoE routes on the action-space descriptor; the arm must return "
                    "it from `action_space_descriptor`"
                )
            # Handed over in float32 on purpose. The vendored router casts to bfloat16
            # itself, and that cast is also what makes the broadcast descriptor
            # contiguous before it is flattened with `view`; pre-casting here turns the
            # cast into a no-op and the flatten then raises.
            return self.mixture(hidden_states, data_mask.to(torch.float32))
        # HB-MoE gates on the hidden states; the vendored signature still takes the
        # descriptor and ignores it.
        return self.mixture(hidden_states, data_mask)

    def parameter_blocks(self, prefix: str, n_embodiments: int) -> tuple[ParameterBlock, ...]:
        """One block per routed expert, plus dense blocks for gate and shared expert.

        Routed experts get `route_shares=None`: their shares are what a soft router
        decides on real data, and 10.4.9's covariate is not allowed to be a guess.
        """
        blocks: list[ParameterBlock] = []
        experts = self.mixture.experts
        for i, expert in enumerate(experts):
            n = sum(p.numel() for p in expert.parameters())
            active = n if i < self.settings.num_experts_per_tok else 0
            blocks.append(
                ParameterBlock(
                    name=f"{prefix}.expert[{i}]",
                    total_params=n,
                    active_params=active,
                    route_shares=None,
                )
            )
        n_gate = sum(p.numel() for p in self.mixture.gate.parameters())
        blocks.append(
            ParameterBlock(
                name=f"{prefix}.gate",
                total_params=n_gate,
                active_params=n_gate,
                route_shares=(1.0,) * n_embodiments,
            )
        )
        shared = getattr(self.mixture, "shared_experts", None)
        if shared is not None:
            n_shared = sum(p.numel() for p in shared.parameters())
            blocks.append(
                ParameterBlock(
                    name=f"{prefix}.shared_expert",
                    total_params=n_shared,
                    active_params=n_shared,
                    route_shares=(1.0,) * n_embodiments,
                )
            )
        return tuple(blocks)


def build_mixture_factory(
    *, d_model: int, ffn_dim: int, dropout: float, n_layers: int
):
    """A `FeedForwardFactory` implementing the HiMoE plan for a decoder of `n_layers`.

    Layers the plan marks dense get the same `DenseFeedForward` the rest of the
    package uses, so I0 and I1 differ only where the plan says they differ.
    """
    from portal_trace.policies.decoder import DenseFeedForward

    plan = himoe_layer_plan(n_layers)
    as_settings = ASMoESettings.matched_to_dense(d_model=d_model, ffn_dim=ffn_dim)
    hb_settings = HBMoESettings.matched_to_dense(d_model=d_model, ffn_dim=ffn_dim)

    def factory(layer_index: int) -> nn.Module:
        kind = plan[layer_index]
        if kind is MixtureKind.DENSE:
            return DenseFeedForward(d_model, ffn_dim, dropout)
        settings = as_settings if kind is MixtureKind.AS else hb_settings
        return MixtureFeedForward(kind, settings)

    return factory
