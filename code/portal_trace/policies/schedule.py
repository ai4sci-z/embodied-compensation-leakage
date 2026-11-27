"""Optimisation and rollout length, with early stopping made unrepresentable.

9.6.1 fixes `horizon=400` and turns early stopping off; 10.4.0 fixes the step
budget and repeats the ban. The reason is not tidiness: with early stopping the
frame-level mixing weight `alpha_frame` drifts away from the nominal demo-level
`alpha_demo`, the drift is monotone in `||delta||`, and Lemma 1's test statistic
`Lambda^part - E_rho[w]` sits directly on that drift, in the same direction as the
effect being tested. On the training side, a stopping rule would let two arms see
different amounts of data and reintroduce confounder X2.

So this module does not default early stopping to off. It makes it impossible:

* `EpisodeLength` is the only way to state how long an episode is, and it has no
  field to set -- `num_samples` is 400 for every instance.
* `validate_episode_length` rejects anything else, with the section number.
* `OptimSpec.from_mapping` rejects a config that so much as mentions a stopping
  rule, by key name, before any value is read.

There is no code path in `portal_trace.policies` that consumes a patience, a
monitored metric or a plateau rule, so a config carrying one now fails loudly
rather than being ignored.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from portal_trace.registry.tasks import CONTROL_FREQ, HORIZON

__all__ = [
    "FIXED_HORIZON",
    "EarlyStoppingForbidden",
    "EpisodeLength",
    "validate_episode_length",
    "OptimSpec",
    "MIN_EPOCHS",
    "MAIN_BATCH_SIZE",
    "MAIN_LR",
]

# Imported rather than restated so that the two cannot drift apart.
FIXED_HORIZON: int = HORIZON

if FIXED_HORIZON != 400:
    raise AssertionError(
        f"registry.tasks.HORIZON is {FIXED_HORIZON}; 9.6.1 fixes it at 400 and the "
        "alpha_frame = alpha_demo identity of X14 holds by construction only there"
    )

# 10.4.0's optimiser row.
MAIN_LR = 3e-4
MAIN_BATCH_SIZE = 64
MIN_EPOCHS = 60

# Any of these in a training config means someone is trying to stop early.
_STOPPING_KEYS = frozenset(
    {
        "early_stop",
        "early_stopping",
        "earlystopping",
        "patience",
        "es_patience",
        "min_delta",
        "monitor",
        "monitor_metric",
        "stop_on_plateau",
        "plateau_patience",
        "max_steps_without_improvement",
        "best_metric",
        "save_best_only",
        "terminate_on_success",
        "stop_on_success",
        "done_mode",
    }
)


class EarlyStoppingForbidden(ValueError):
    """Raised when a config mentions a stopping rule.

    9.6.1 and 10.4.0 both forbid it. The message names the section rather than
    just refusing, because the natural next move -- deleting the key -- is right.
    """

    def __init__(self, keys: frozenset[str]) -> None:
        listed = ", ".join(sorted(keys))
        super().__init__(
            f"training config carries stopping rule keys [{listed}]. 10.4.0 fixes the "
            "step count and disables early stopping, and 9.6.1 fixes horizon=400 with "
            "num_samples always 400: a stopping rule makes alpha_frame drift from "
            "alpha_demo in a direction correlated with ||delta||, which is the same "
            "direction as the effect under test. Remove the keys; there is no setting "
            "of them that this package will honour."
        )


@dataclass(frozen=True)
class EpisodeLength:
    """The length of every episode, demonstration or rollout alike.

    Carries no state. It exists so that "how long is an episode" has a type whose
    only possible answer is 400, and so that a function needing the number takes
    this instead of an `int` a caller could compute.
    """

    @property
    def num_samples(self) -> int:
        return FIXED_HORIZON

    @property
    def seconds(self) -> float:
        return FIXED_HORIZON / CONTROL_FREQ

    def __len__(self) -> int:
        return FIXED_HORIZON

    def __str__(self) -> str:
        return f"{FIXED_HORIZON} steps ({self.seconds:g} s at {CONTROL_FREQ} Hz)"


def validate_episode_length(num_samples: int, *, what: str) -> int:
    """Check a length read off disk or returned by a rollout.

    `what` names the thing being checked so that the message points at the demo or
    the rollout that is short, not at this function.
    """
    if num_samples != FIXED_HORIZON:
        raise ValueError(
            f"{what} has num_samples={num_samples}, expected {FIXED_HORIZON}. 9.6.1 "
            "collects fixed length: after success the teacher holds the terminal pose "
            "to the horizon. A short episode means early stopping was left on "
            "somewhere in the collection pipeline, and the alpha_demo/alpha_frame pair "
            "stored per config is the end-to-end check for exactly this -- any nonzero "
            "difference between them is a bug, not a tolerable deviation"
        )
    return num_samples


@dataclass(frozen=True)
class OptimSpec:
    """10.4.0's optimiser row: AdamW, cosine annealed, fixed step count.

    `epochs` is part of the endpoint definition, which is why the floor of 60 is
    enforced rather than documented: an arm trained for fewer epochs is not
    comparable to the Oracle arm it is differenced against (10.5.5, "same training
    budget").
    """

    lr: float = MAIN_LR
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    batch_size: int = MAIN_BATCH_SIZE
    epochs: int = MIN_EPOCHS
    warmup_steps: int = 2_000
    grad_clip: float = 1.0
    min_lr_ratio: float = 0.01

    def __post_init__(self) -> None:
        if self.epochs < MIN_EPOCHS:
            raise ValueError(
                f"epochs={self.epochs} is below the {MIN_EPOCHS} of 10.4.0, which "
                "fixes the budget per config; a shorter budget breaks the "
                "same-training-budget match of 10.5.5"
            )
        if self.lr <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("lr must be positive and weight decay non-negative")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not 0.0 <= self.min_lr_ratio < 1.0:
            raise ValueError("min_lr_ratio must lie in [0, 1)")

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> OptimSpec:
        """Build from a declarative config, refusing stopping rules by key name."""
        offenders = frozenset(k for k in cfg if k.lower() in _STOPPING_KEYS)
        if offenders:
            raise EarlyStoppingForbidden(offenders)
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(cfg) - known)
        if unknown:
            raise ValueError(
                f"unknown optimiser keys {unknown}; OptimSpec accepts {sorted(known)}"
            )
        return cls(**dict(cfg))

    def total_steps(self, *, n_frames: int) -> int:
        """The fixed step count, from the dataset size and the epoch budget.

        Fixed, not a bound: with the 9.6.1 length caliber the number of frames per
        epoch is exactly `n_demos * 400`, so the schedule is known before training
        starts and the cosine annealing has a definite end.
        """
        if n_frames < 1:
            raise ValueError("n_frames must be positive")
        return self.epochs * math.ceil(n_frames / self.batch_size)

    def lr_at(self, step: int, *, total_steps: int) -> float:
        """Linear warmup then cosine annealing to `min_lr_ratio * lr`."""
        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        if step < 0:
            raise ValueError("step must be non-negative")
        if step < self.warmup_steps:
            return self.lr * (step + 1) / max(self.warmup_steps, 1)
        span = max(total_steps - self.warmup_steps, 1)
        progress = min((step - self.warmup_steps) / span, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine)
