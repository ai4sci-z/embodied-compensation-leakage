"""Embodiment conditioning channels shared by B4, B6, B11 and B13.

The four baselines that condition on the embodiment sit at different points on one
axis, and 10.5.6 requires them to enter the network at the same place so that they are
comparable:

* B6 gives an instance-level label -- discrete, needs the label, can catch the
  contrastive component.
* B4's style-token variant gives a learned low-dimensional token multiplied into the
  action head's FiLM layer.
* B13 gives a continuous latent inferred from `K_hist` frames of proprioception --
  no label at all. This is the rung between B6 and B4, and 2.1 calls it the strongest
  gap in the baseline table.
* B11 infers a discrete fault regime from recent joint dynamics and execution history
  and conditions a shared residual calibrator on it.

All four produce a vector that the decoder's FiLM layers consume, which is why they
are here and not each inside its own arm.

The moat that has to be stated rather than left to a reader (10.5.6): none of these
can absorb the common mode. A latent inferred from proprioception can only express
what differs between embodiments, and a bias every source shares is a constant on any
such latent, so it has nowhere to go. That is why E7 and not B13 is the crown.
"""

from __future__ import annotations

from enum import Enum

import torch
from robomimic.models.base_nets import MLP
from torch import nn

from portal_trace.policies.obs import ACTION_DIM, PROPRIO_DIM, proprio_group
from portal_trace.types import Regime

__all__ = [
    "EmbodimentTokenKind",
    "EmbodimentToken",
    "LatentExtrinsicsEncoder",
    "LATENT_DIMS",
    "FaultRegimeHead",
    "N_REGIMES",
    "dynamics_channels",
]

# 9.7.1's five regimes; B11's regime posterior is over these, and the shared caliber
# is what makes it evidence for (A5)(ii) rather than a private construction.
N_REGIMES = len(Regime)

# B13 sweeps two latent widths (10.5, "latent dimension swept over two levels").
LATENT_DIMS: tuple[int, ...] = (8, 32)


class EmbodimentTokenKind(str, Enum):
    ONE_HOT = "one_hot"
    EMBEDDING = "embedding"


class EmbodimentToken(nn.Module):
    """B6's conditioning channel, in both of its forms.

    `permute` implements the inference-time ablation 10.5 asks for: feed embodiment
    `i` the token of embodiment `pi(i)`. If the arm's leakage is carried by the token,
    permuting it at inference moves the leakage; if it is carried by the trunk, nothing
    happens. Because the permutation is applied at lookup time it costs nothing and
    cannot change training.
    """

    def __init__(
        self,
        n_embodiments: int,
        kind: EmbodimentTokenKind = EmbodimentTokenKind.EMBEDDING,
        *,
        dim: int = 16,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.n_embodiments = n_embodiments
        if kind is EmbodimentTokenKind.ONE_HOT:
            self.dim = n_embodiments
            self.table: nn.Module | None = None
        else:
            self.dim = dim
            self.table = nn.Embedding(n_embodiments, dim)
            nn.init.trunc_normal_(self.table.weight, std=0.02)
        self.register_buffer(
            "permutation", torch.arange(n_embodiments), persistent=False
        )

    def set_permutation(self, permutation: torch.Tensor | None) -> None:
        """Install or clear the inference-time token permutation."""
        if permutation is None:
            self.permutation = torch.arange(self.n_embodiments, device=self.permutation.device)
            return
        if permutation.shape != (self.n_embodiments,):
            raise ValueError(
                f"permutation must have {self.n_embodiments} entries, got "
                f"{tuple(permutation.shape)}"
            )
        if sorted(permutation.tolist()) != list(range(self.n_embodiments)):
            raise ValueError("the ablation needs a permutation, not an arbitrary map")
        self.permutation = permutation.to(self.permutation.device)

    def forward(self, embodiment_index: torch.Tensor) -> torch.Tensor:
        idx = self.permutation[embodiment_index]
        if self.table is None:
            return torch.nn.functional.one_hot(idx, self.n_embodiments).to(
                dtype=torch.float32
            )
        return self.table(idx)


def dynamics_channels(proprio: torch.Tensor) -> torch.Tensor:
    """The execution-chain channels a latent or a regime is inferred from.

    Joint velocity, joint acceleration, the gravity-torque channels and the wrist
    force/torque: the quantities that carry friction, gain, delay and load. Slicing
    them out rather than handing over all 70 dimensions keeps the end-effector pose out
    of the inference path, so that the latent cannot become a task-progress feature --
    which would make B13 look like it captures the embodiment when it captures the
    phase.
    """
    if proprio.shape[-1] != PROPRIO_DIM:
        raise ValueError(
            f"expected the {PROPRIO_DIM}-dimensional layout, got {proprio.shape[-1]}"
        )
    names = ("joint_vel", "joint_acc", "tau_g", "wrist_ft", "dof_mask")
    return torch.cat([proprio[..., proprio_group(n).slice] for n in names], dim=-1)


_DYNAMICS_DIM = 7 + 7 + 7 + 6 + 7


class LatentExtrinsicsEncoder(nn.Module):
    """B13: a low-dimensional continuous latent from `K_hist` frames, no labels.

    The RMA-style adaptation module: a temporal 1-D convolution over the recent
    execution history followed by a projection. Reimplemented from the literature
    rather than vendored -- this is a family of methods (RMA, online system
    identification, context conditioning) with no single canonical implementation, and
    10.5.7 asks for the strongest configuration rather than a particular repository.

    The optional action history is what makes the latent able to see delay and gain:
    with proprioception alone, a commanded-versus-achieved discrepancy is not
    observable within the window.
    """

    def __init__(
        self,
        k_hist: int,
        latent_dim: int = LATENT_DIMS[0],
        *,
        hidden: int = 128,
        use_action_history: bool = True,
    ) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError("the latent needs at least one dimension")
        self.latent_dim = latent_dim
        self.use_action_history = use_action_history
        in_dim = _DYNAMICS_DIM + (ACTION_DIM if use_action_history else 0)
        kernel = min(3, k_hist)
        self.temporal = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=kernel, padding=kernel // 2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=kernel, padding=kernel // 2),
            nn.GELU(),
        )
        self.project = MLP(
            input_dim=hidden, output_dim=latent_dim, layer_dims=(hidden,), activation=nn.GELU
        )

    def forward(
        self, proprio: torch.Tensor, action_history: torch.Tensor | None = None
    ) -> torch.Tensor:
        features = dynamics_channels(proprio)
        if self.use_action_history:
            if action_history is None:
                raise ValueError(
                    "this encoder was built to read the action history; the arm must "
                    "put it on the batch as extras['action_history'], shape (B, K, 7)"
                )
            if action_history.shape[:2] != proprio.shape[:2]:
                raise ValueError(
                    "action history and proprioception must cover the same frames, got "
                    f"{tuple(action_history.shape[:2])} and {tuple(proprio.shape[:2])}"
                )
            features = torch.cat((features, action_history), dim=-1)
        # (B, K, C) -> (B, C, K) for the temporal convolution, then pool over time.
        hidden = self.temporal(features.transpose(1, 2)).mean(dim=-1)
        return self.project(hidden)


class FaultRegimeHead(nn.Module):
    """B11: an implicit joint-fault regime inferred from a short history (J-PARC).

    Reimplemented from `Uncovering Vulnerability of VLA Models under Joint-Level
    Physical Faults` (arXiv 2606.10501); there is no public code. The mechanism taken
    from the paper is the one the proposal cites twice: infer a latent fault regime
    from recent joint dynamics and execution history, then condition a single shared
    residual calibrator on it, rather than learning one calibrator per fault.

    The posterior is over the five regimes of 9.7.1 rather than over a private label
    set, which is what makes the arm double as published support for (A5)(ii): regime
    labels are constructible from a short window of the policy input.
    """

    def __init__(self, k_hist: int, *, hidden: int = 128, use_action_history: bool = True) -> None:
        super().__init__()
        self.use_action_history = use_action_history
        in_dim = (_DYNAMICS_DIM + (ACTION_DIM if use_action_history else 0)) * k_hist
        self.net = MLP(
            input_dim=in_dim,
            output_dim=N_REGIMES,
            layer_dims=(hidden, hidden),
            activation=nn.GELU,
        )

    def forward(
        self, proprio: torch.Tensor, action_history: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Returns the regime posterior, shape (B, N_REGIMES)."""
        features = dynamics_channels(proprio)
        if self.use_action_history:
            if action_history is None:
                raise ValueError(
                    "the regime head reads the execution history; the arm must supply "
                    "extras['action_history'], shape (B, K, 7)"
                )
            features = torch.cat((features, action_history), dim=-1)
        logits = self.net(features.flatten(1))
        return torch.softmax(logits, dim=-1)
