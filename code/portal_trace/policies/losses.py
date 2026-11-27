"""Behaviour-cloning loss over an action chunk, and the pieces baselines add to it.

The main loss is squared error on normalised actions, which is robomimic's default
BC objective (`l2_weight=1.0`) and the setting 9.6.1 says the hyper-parameter
search starts from. It is masked and weighted because two baselines need it to be:
the last seven frames of a fixed-length 400-step episode have no full chunk ahead
of them, and B10 curates by reweighting samples rather than by deleting them from
the pool.

`gradient_reversal` and the mutual-information estimators live here rather than in
the arm that uses them because B4 runs four variants that share them (10.5.1).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "chunk_bc_loss",
    "gradient_reversal",
    "GradientReversal",
    "CLUBEstimator",
    "MINEEstimator",
    "info_nce",
    "cosine_alignment",
    "routing_entropy",
    "max_routing_entropy",
]


def chunk_bc_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    pad_mask: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared error over (B, CHUNK, action_dim), masked and weighted.

    `pad_mask` is True on frames that exist. Padded chunk steps carry no
    supervision, so they are dropped from both numerator and denominator; folding
    them in with a repeated last action would put a spurious "hold still" signal at
    the end of every episode, which is the terminal-pose hold of 9.6.1 duplicated.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction {tuple(prediction.shape)} does not match target "
            f"{tuple(target.shape)}"
        )
    per_element = (prediction - target).pow(2)
    weights = torch.ones_like(per_element)
    if pad_mask is not None:
        if pad_mask.shape != prediction.shape[:2]:
            raise ValueError(
                f"pad_mask {tuple(pad_mask.shape)} does not match (B, CHUNK) "
                f"{tuple(prediction.shape[:2])}"
            )
        weights = weights * pad_mask.to(per_element.dtype).unsqueeze(-1)
    if sample_weight is not None:
        if sample_weight.shape != prediction.shape[:1]:
            raise ValueError(
                f"sample_weight {tuple(sample_weight.shape)} does not match batch "
                f"{prediction.shape[0]}"
            )
        weights = weights * sample_weight.to(per_element.dtype).view(-1, 1, 1)
    denominator = weights.sum()
    if denominator <= 0:
        raise ValueError(
            "every element of the batch is masked or zero-weighted; a curation or "
            "padding rule has removed the whole batch"
        )
    return (per_element * weights).sum() / denominator


class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:  # noqa: ANN001
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # noqa: ANN001, ANN205
        return -ctx.lambd * grad_output, None


def gradient_reversal(x: torch.Tensor, lambd: float) -> torch.Tensor:
    """Identity forward, negated-and-scaled gradient backward (B4, DANN variant)."""
    return _GradientReversalFn.apply(x, lambd)


class GradientReversal(nn.Module):
    """Module form, so that `lambd` can be swept as a config value (B4, three levels)."""

    def __init__(self, lambd: float) -> None:
        super().__init__()
        if lambd < 0.0:
            raise ValueError("the reversal strength is a magnitude; pass lambd >= 0")
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gradient_reversal(x, self.lambd)


class CLUBEstimator(nn.Module):
    """Contrastive log-ratio upper bound on I(x; y), the estimator B4 ports.

    `Disentangled Cross-Embodiment Video Editing` minimises mutual information
    between a task representation and an embodiment representation with CLUB; 10.5.1
    moves the objective from the pixel layer to the action layer, so the two inputs
    here are the two branches of the action head rather than two image encoders.

    CLUB needs its variational approximation `q(y|x)` trained by maximum likelihood
    while the representations are trained to minimise the bound, so the two losses
    are returned separately and the caller must not sum them.
    """

    def __init__(self, x_dim: int, y_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.mu = nn.Sequential(
            nn.Linear(x_dim, hidden), nn.ReLU(), nn.Linear(hidden, y_dim)
        )
        self.log_var = nn.Sequential(
            nn.Linear(x_dim, hidden), nn.ReLU(), nn.Linear(hidden, y_dim), nn.Tanh()
        )

    def _params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mu(x), self.log_var(x)

    def log_likelihood(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """The term that trains `q(y|x)`; maximise it, i.e. minimise its negation."""
        mu, log_var = self._params(x)
        return (-((mu - y) ** 2) / log_var.exp() - log_var).sum(dim=-1).mean()

    def frozen_upper_bound(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """`upper_bound` with `q`'s own parameters detached.

        This is what makes the two objectives runnable in one optimiser step without
        interfering: the bound then moves only the representations, and
        `log_likelihood` on detached inputs moves only `q`. Summing `upper_bound` and
        `log_likelihood` directly would have `q` minimise the bound it is supposed to
        tighten, which is the failure mode CLUB's alternating scheme exists to avoid.
        """
        mu, log_var = self._frozen_params(x)
        return self._bound(mu, log_var, y)

    def _frozen_params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        detach = lambda m: {n: p.detach() for n, p in m.named_parameters()}  # noqa: E731
        return (
            torch.func.functional_call(self.mu, detach(self.mu), (x,)),
            torch.func.functional_call(self.log_var, detach(self.log_var), (x,)),
        )

    def _bound(
        self, mu: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        positive = -((mu - y) ** 2) / log_var.exp()
        # Negative pairs come from shuffling y within the batch, which is the
        # marginal sample CLUB's bound is defined against.
        shuffled = y[torch.randperm(y.shape[0], device=y.device)]
        negative = -((mu - shuffled) ** 2) / log_var.exp()
        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def upper_bound(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """The bound on I(x; y); this is what the representations minimise."""
        mu, log_var = self._params(x)
        return self._bound(mu, log_var, y)


class MINEEstimator(nn.Module):
    """Donsker-Varadhan mutual-information estimator, A14's second estimator.

    Reported alongside CLUB rather than instead of it: A14 sweeps the estimator
    because the disentangling result should not depend on which bound was used.
    """

    def __init__(self, x_dim: int, y_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        # Starts at one, not zero: the ratio below divides by it on the first step.
        self.register_buffer("running_mean", torch.ones(()))
        self.momentum = 0.9

    def frozen_estimate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """The estimate with the critic detached, and without touching the EMA.

        The counterpart of `CLUBEstimator.frozen_upper_bound`: the critic maximises
        the estimate through `forward`, the representations minimise it through this,
        and the moving average is updated exactly once per step.
        """
        detach = {n: p.detach() for n, p in self.net.named_parameters()}
        joint = torch.func.functional_call(
            self.net, detach, (torch.cat((x, y), dim=-1),)
        ).mean()
        shuffled = y[torch.randperm(y.shape[0], device=y.device)]
        marginal = torch.func.functional_call(
            self.net, detach, (torch.cat((x, shuffled), dim=-1),)
        )
        log_mean = marginal.exp().mean() / self.running_mean.clamp_min(1e-8)
        return joint - log_mean.log() - self.running_mean.clamp_min(1e-8).log()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        joint = self.net(torch.cat((x, y), dim=-1)).mean()
        shuffled = y[torch.randperm(y.shape[0], device=y.device)]
        marginal = self.net(torch.cat((x, shuffled), dim=-1))
        # The log-sum-exp term is biased when estimated per batch; the standard fix
        # is an exponential moving average in the denominator of its gradient.
        with torch.no_grad():
            self.running_mean.mul_(self.momentum).add_(
                (1.0 - self.momentum) * marginal.exp().mean()
            )
        log_mean = marginal.exp().mean() / self.running_mean.clamp_min(1e-8)
        return joint - log_mean.log() - self.running_mean.clamp_min(1e-8).log()


def info_nce(
    anchor: torch.Tensor, positive: torch.Tensor, *, temperature: float = 0.1
) -> torch.Tensor:
    """InfoNCE over a batch, with the diagonal as the positive pair (B4).

    Paired with CLUB in the ported objective: CLUB pushes the two branches apart,
    InfoNCE keeps the task branch aligned with its own augmented view, and without
    the second term the first is satisfied by collapsing one branch.
    """
    if anchor.shape != positive.shape:
        raise ValueError("InfoNCE needs matched shapes for anchor and positive")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    logits = a @ p.transpose(0, 1) / temperature
    labels = torch.arange(a.shape[0], device=a.device)
    return F.cross_entropy(logits, labels)


def cosine_alignment(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The L2 cosine E4's endpoint is, computed over a batch of vector fields.

    Flattens chunk and action dimensions before the inner product, because the
    endpoint of 10.4.4 is the cosine in `L^2(rho^off)` between two vector fields and
    not the mean of per-frame cosines: the latter weights near-zero frames equally
    with the frames that carry the compensation.
    """
    if a.shape != b.shape:
        raise ValueError(f"cosine needs matched shapes, got {a.shape} and {b.shape}")
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    denominator = flat_a.norm() * flat_b.norm()
    if denominator.item() == 0.0:
        raise ValueError(
            "one of the fields is identically zero; the cosine of 10.4.4 is undefined "
            "there and the frame set has to be reported as degenerate rather than "
            "scored as 0"
        )
    return torch.dot(flat_a, flat_b) / denominator


def routing_entropy(shares: torch.Tensor) -> torch.Tensor:
    """Entropy of a routing distribution, in nats, E9's secondary covariate.

    Reported alongside the parameter-disjointness index; 10.4.9 makes the index the
    main regressor and the entropy complement a companion reading.
    """
    p = shares.clamp_min(1e-12)
    p = p / p.sum(dim=-1, keepdim=True)
    return -(p * p.log()).sum(dim=-1)


def max_routing_entropy(n_experts: int) -> float:
    """The entropy of a uniform router, the normaliser for the complement."""
    if n_experts < 1:
        raise ValueError("n_experts must be positive")
    return math.log(n_experts)
