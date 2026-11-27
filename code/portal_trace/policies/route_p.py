"""E10 route (P): the four function-class rungs, and the (ER) channel excision.

Proposal 10.4.10. Route (P) is the branch of proposition 1(d) that buys
identifiability from a physical prior instead of from diversity, and E10 is its
only test: a single-embodiment pool, K = 1, with no clean twin in it, because
with the clean twin the question would be route (A) again.

The four rungs differ only in the function class of the two heads.

* **(i)** unrestricted mu head, compensation restricted to span psi_j with a
  learnable delta_hat. No (ER). Predicted to *fail* at a relative recovery error
  of 0.80 or worse, level with the unconstrained MLP.
* **(ii)** the same compensation class plus the (ER) input-channel excision: the
  mu head sees end-effector pose, object state and the goal and nothing else,
  while the recent joint-velocity sign, the zero-crossing flag and the
  load-torque history are reserved for the compensation pathway. Predicted to
  succeed at 0.25 or better.
* **(iii)** unconstrained MLP compensation head. The unidentifiability control,
  predicted at 0.80 or worse.
* **(iv)** the defect parameters handed to the compensation head directly. An
  upper-bound control on the success rate and **not** route (P), which requires
  delta to be learned and is given only the function form.

Why the excision is the whole of rung (ii)
------------------------------------------
Under `DecompositionForm.PHYSICAL_BASIS` the compensation branch is
`delta_hat . psi(x, t)` with delta_hat a free coefficient vector and no
observation input at all -- which is the right shape, since delta is a property
of the body and not a function of the state. So the "exclusive" ownership 10.4.10
gives the compensation head is not a second input pathway: those three signals
are exactly what 9.4.4 builds psi out of (`sign(qdot_j) J e_j`,
`J^{-T} tau_g(q)`, `d/dt a_nom`), and reserving them means *removing them from
the mu head*. Rung (ii) is therefore a change to one input mask, which is why
10.4.10 calls it configuration-layer and zero extra training cost, and it is the
only form in which (ER) can be implemented rather than assumed.

What the proposal leaves open, and the choices taken
----------------------------------------------------
10.4.10 names the mu head's admitted inputs as "end-effector pose + object state
+ goal" without mapping them onto the 70-dimensional layout of 9.6.1. Object
state and the goal are not proprioceptive at all -- they arrive as the two camera
streams and the task token -- so the mapping only has to be made for the pose,
and it is taken in robomimic's sense of the `eef` observation group: position,
orientation and gripper joint position. Gripper *velocity* is not in it, which
keeps every velocity channel out of the mu head; that is the direction the
exclusion restriction cares about.

The layout's remaining groups -- raw joint angles and their cos/sin, joint
acceleration, wrist force/torque and the DoF mask -- are in neither list, and
rung (ii) drops them rather than routing them to a default destination. Sending
them to the mu head would breach "only"; sending them to the compensation
pathway would add columns psi does not have. The DoF mask is the one that costs
nothing to drop: E10's pool is a single embodiment, so the mask is constant
across the whole dataset.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch
from robomimic.models.base_nets import MLP
from torch import nn

from portal_trace.policies.decomposition import DecompositionForm
from portal_trace.policies.obs import PROPRIO_GROUPS, ProprioGroup, proprio_group

__all__ = [
    "MU_HEAD_POSE_GROUPS",
    "COMPENSATION_EXCLUSIVE_GROUPS",
    "RELATIVE_ERROR_SUCCESS",
    "RELATIVE_ERROR_FAILURE",
    "MU_HEAD_LAMBDA_REF_MAX",
    "SUCCESS_RATE_GAP_MAX_PP",
    "FAILURE_RUNG_SPREAD_MAX",
    "ChannelExcision",
    "RungLabel",
    "RoutePRung",
    "ROUTE_P_RUNGS",
    "ExcisedProprioEncoder",
    "RungReading",
    "RoutePReport",
    "ChannelOverlapError",
    "NoExcisionError",
    "MissingRungReadingError",
    "GroundTruthRungNotRouteP",
]

MU_HEAD_POSE_GROUPS: tuple[str, ...] = (
    "eef_pos",
    "eef_quat",
    "eef_quat_site",
    "gripper_qpos",
)
"""The proprioceptive part of "end-effector pose" for rung (ii)'s mu head."""

COMPENSATION_EXCLUSIVE_GROUPS: tuple[str, ...] = ("joint_vel", "tau_g")
"""Recent joint-velocity sign and zero crossings, and the load-torque history.

Both are read over the K_hist window rather than per frame, which is what makes
them a *sign* and a *history*: `joint_vel` over the window gives the sign and the
zero crossings, `tau_g` over the window gives the load-torque history.
"""

RELATIVE_ERROR_SUCCESS = 0.25
"""10.4.10's preregistered prediction for rung (ii)."""

RELATIVE_ERROR_FAILURE = 0.80
"""...and for rungs (i) and (iii), which are predicted not to recover delta."""

MU_HEAD_LAMBDA_REF_MAX = 0.20
"""The mu head's residual cleanliness at rung (ii), read as Lambda^ref."""

SUCCESS_RATE_GAP_MAX_PP = 5.0
"""Rung (ii) must come within 5 pp of the ground-truth rung's success rate."""

FAILURE_RUNG_SPREAD_MAX = 0.10
"""Rungs (i) and (iii) must land within 0.10 of each other to read as level."""


class ChannelOverlapError(ValueError):
    """The mu head and the compensation pathway claim the same channel."""


class NoExcisionError(ValueError):
    """A rung declaring (ER) admits every channel, so nothing was excluded."""


class MissingRungReadingError(ValueError):
    """A rung the verdict rule quotes has no reading."""


class GroundTruthRungNotRouteP(ValueError):
    """The ground-truth rung was used as route (P) evidence."""


@dataclass(frozen=True)
class ChannelExcision:
    """Which of the 70 proprioception channels each pathway may see.

    Held as group names and resolved to indices on demand: 9.6.1's layout is the
    single source of the offsets, and a rung that stored indices would go stale
    the moment a group's width changed.
    """

    mu_head_groups: tuple[str, ...]
    compensation_groups: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (*self.mu_head_groups, *self.compensation_groups):
            proprio_group(name)
        overlap = sorted(set(self.mu_head_groups) & set(self.compensation_groups))
        if overlap:
            raise ChannelOverlapError(
                f"groups {overlap} are claimed by both pathways. 10.4.10 gives the "
                "compensation head *exclusive* use of its channels, and a channel the "
                "mu head can also read is not excluded from it"
            )

    @staticmethod
    def unrestricted() -> ChannelExcision:
        """Every channel to the mu head: the main architecture, rungs (i)/(iii)/(iv)."""
        return ChannelExcision(
            mu_head_groups=tuple(g.name for g in PROPRIO_GROUPS),
            compensation_groups=(),
        )

    @staticmethod
    def exclusion_restriction() -> ChannelExcision:
        """Rung (ii): pose to the mu head, the defect signals to the compensation side."""
        return ChannelExcision(
            mu_head_groups=MU_HEAD_POSE_GROUPS,
            compensation_groups=COMPENSATION_EXCLUSIVE_GROUPS,
        )

    def mu_head_channels(self) -> tuple[int, ...]:
        out: list[int] = []
        for name in self.mu_head_groups:
            group = proprio_group(name)
            out.extend(range(group.start, group.stop))
        return tuple(sorted(out))

    def dropped_groups(self) -> tuple[str, ...]:
        claimed = set(self.mu_head_groups) | set(self.compensation_groups)
        return tuple(g.name for g in PROPRIO_GROUPS if g.name not in claimed)

    @property
    def is_restricted(self) -> bool:
        return len(self.mu_head_channels()) < sum(g.width for g in PROPRIO_GROUPS)

    def as_row(self) -> dict[str, Any]:
        return {
            "mu_head_groups": list(self.mu_head_groups),
            "mu_head_width": len(self.mu_head_channels()),
            "compensation_exclusive_groups": list(self.compensation_groups),
            "dropped_groups": list(self.dropped_groups()),
        }


class RungLabel(str, Enum):
    """10.4.10's four rungs, by the numerals the table uses."""

    NO_ER_PHYSICAL_BASIS = "i"
    ER_PHYSICAL_BASIS = "ii"
    UNCONSTRAINED_MLP = "iii"
    GROUND_TRUTH = "iv"


@dataclass(frozen=True)
class RoutePRung:
    """One rung: the two function classes, the excision and the prediction.

    `predicted_relative_error` is a bound with a direction, not a point: the
    table predicts "at most 0.25" for rung (ii) and "at least 0.80" for the two
    failures, and `predicted_at_most` says which.
    """

    label: RungLabel
    compensation_form: DecompositionForm | None
    excision: ChannelExcision
    predicted_relative_error: float | None
    predicted_at_most: bool
    is_route_p: bool
    description: str

    def __post_init__(self) -> None:
        if self.label is RungLabel.ER_PHYSICAL_BASIS and not self.excision.is_restricted:
            raise NoExcisionError(
                "rung (ii) is defined by the (ER) input-channel excision; an excision "
                "that admits all 70 channels makes it a copy of rung (i)"
            )
        if self.compensation_form is None and self.is_route_p:
            raise GroundTruthRungNotRouteP(
                "a rung whose compensation head is handed the defect parameters is not "
                "route (P): 10.4.10 requires delta to be learned and only the function "
                "form to be given"
            )
        if self.predicted_relative_error is not None and not (
            0.0 <= self.predicted_relative_error <= 1.0
        ):
            raise ValueError(
                f"rung ({self.label.value}): a relative recovery error prediction of "
                f"{self.predicted_relative_error} is outside [0, 1]"
            )

    def meets_prediction(self, relative_error: float) -> bool:
        if self.predicted_relative_error is None:
            raise ValueError(
                f"rung ({self.label.value}) has no registered relative-error "
                "prediction; it is the upper-bound control and is read on the success "
                "rate"
            )
        if self.predicted_at_most:
            return relative_error <= self.predicted_relative_error
        return relative_error >= self.predicted_relative_error

    def as_row(self) -> dict[str, Any]:
        return {
            "rung": self.label.value,
            "compensation_form": (
                "ground-truth defect parameters"
                if self.compensation_form is None
                else self.compensation_form.value
            ),
            "excision": self.excision.as_row(),
            "predicted_relative_error": self.predicted_relative_error,
            "prediction_direction": "at most" if self.predicted_at_most else "at least",
            "is_route_p": self.is_route_p,
            "description": self.description,
        }


ROUTE_P_RUNGS: tuple[RoutePRung, ...] = (
    RoutePRung(
        label=RungLabel.NO_ER_PHYSICAL_BASIS,
        compensation_form=DecompositionForm.PHYSICAL_BASIS,
        excision=ChannelExcision.unrestricted(),
        predicted_relative_error=RELATIVE_ERROR_FAILURE,
        predicted_at_most=False,
        is_route_p=True,
        description=(
            "span psi_j with a learnable delta_hat, unrestricted mu head, no (ER)"
        ),
    ),
    RoutePRung(
        label=RungLabel.ER_PHYSICAL_BASIS,
        compensation_form=DecompositionForm.PHYSICAL_BASIS,
        excision=ChannelExcision.exclusion_restriction(),
        predicted_relative_error=RELATIVE_ERROR_SUCCESS,
        predicted_at_most=True,
        is_route_p=True,
        description=(
            "span psi_j with a learnable delta_hat and the (ER) input-channel excision"
        ),
    ),
    RoutePRung(
        label=RungLabel.UNCONSTRAINED_MLP,
        compensation_form=DecompositionForm.UNCONSTRAINED_MLP,
        excision=ChannelExcision.unrestricted(),
        predicted_relative_error=RELATIVE_ERROR_FAILURE,
        predicted_at_most=False,
        is_route_p=True,
        description="unconstrained MLP compensation head, the unidentifiability control",
    ),
    RoutePRung(
        label=RungLabel.GROUND_TRUTH,
        compensation_form=None,
        excision=ChannelExcision.unrestricted(),
        predicted_relative_error=None,
        predicted_at_most=True,
        is_route_p=False,
        description="the defect parameters handed over directly; a success-rate upper bound",
    ),
)


class ExcisedProprioEncoder(nn.Module):
    """The 10.4.0 proprioception pathway over an admitted subset of the channels.

    Selection rather than masking. A zeroed channel still occupies an input
    column whose first-layer bias can carry a constant offset, which is the same
    objection 10.6.1 raises against implementing B9's RGB-only level by feeding
    zeros; under (ER) that offset would be a channel the mu head is supposed not
    to have.
    """

    def __init__(
        self, channels: Sequence[int], d_model: int, *, hidden: int = 512
    ) -> None:
        super().__init__()
        index = torch.as_tensor(sorted(set(int(c) for c in channels)), dtype=torch.long)
        if index.numel() != len(channels):
            raise ValueError(f"duplicate channel indices in {list(channels)}")
        if index.numel() == 0:
            raise ValueError("an excised pathway with no admitted channel has no input")
        width = sum(g.width for g in PROPRIO_GROUPS)
        if int(index.max()) >= width or int(index.min()) < 0:
            raise ValueError(
                f"channel indices must lie in [0, {width}); got "
                f"[{int(index.min())}, {int(index.max())}]"
            )
        self.register_buffer("index", index, persistent=True)
        self.input_dim = int(index.numel())
        self.net = MLP(
            input_dim=self.input_dim,
            output_dim=d_model,
            layer_dims=(hidden,),
            activation=nn.GELU,
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        width = sum(g.width for g in PROPRIO_GROUPS)
        if proprio.ndim != 3 or proprio.shape[-1] != width:
            raise ValueError(
                f"expected proprioception of shape (B, K, {width}), got "
                f"{tuple(proprio.shape)}"
            )
        b, k, _ = proprio.shape
        admitted = proprio.index_select(-1, self.index)
        return self.net(admitted.reshape(b * k, -1)).reshape(b, k, -1)


def relative_recovery_error(
    delta_hat: np.ndarray | torch.Tensor, delta: np.ndarray | torch.Tensor
) -> float:
    """||delta_hat - delta|| / ||delta||, E10's endpoint (i).

    The denominator is the true defect, so a zero delta has no relative error --
    and E10's pool is a defective body by construction, so a zero here means the
    ground truth was not stored rather than that the body is clean.
    """
    a = np.asarray(delta_hat, dtype=np.float64).reshape(-1)
    b = np.asarray(delta, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(
            f"delta_hat has {a.size} coefficients and delta has {b.size}. The "
            "comparison is per psi group (9.4.4's layout), so the two must be written "
            "in the same basis"
        )
    denominator = float(np.linalg.norm(b))
    if denominator == 0.0:
        raise ValueError(
            "||delta|| is zero, so the relative recovery error is undefined. E10's "
            "pool is a single *defective* embodiment; a zero ground truth means the "
            "stored per-trajectory delta is missing, which is the one thing this "
            "endpoint cannot be computed without (it has no real-data counterpart)"
        )
    return float(np.linalg.norm(a - b) / denominator)


@dataclass(frozen=True)
class RungReading:
    """One rung's three endpoints (10.4.10).

    `mu_head_lambda_ref` is the mu head evaluated as a policy on the clean twin;
    `success_rate_pp` is on the target body. The ground-truth rung has no
    recovery error because nothing was recovered.
    """

    label: RungLabel
    relative_error: float | None
    mu_head_lambda_ref: float
    success_rate_pp: float
    surrogate_relative_error: float | None = None

    def __post_init__(self) -> None:
        if self.label is RungLabel.GROUND_TRUTH:
            if self.relative_error is not None:
                raise GroundTruthRungNotRouteP(
                    "rung (iv) is handed delta, so a recovery error for it would be "
                    "zero by construction and would print as a route (P) result"
                )
        elif self.relative_error is None:
            raise ValueError(f"rung ({self.label.value}) has no relative recovery error")
        for name, value in (
            ("mu_head_lambda_ref", self.mu_head_lambda_ref),
            ("success_rate_pp", self.success_rate_pp),
        ):
            if not math.isfinite(value):
                raise ValueError(f"rung ({self.label.value}): {name} is not finite")
        if not 0.0 <= self.success_rate_pp <= 100.0:
            raise ValueError(
                f"rung ({self.label.value}): success rate {self.success_rate_pp} is not "
                "a percentage"
            )


@dataclass(frozen=True)
class RoutePReport:
    """10.4.10's preregistered predictions, evaluated together."""

    readings: dict[RungLabel, RungReading]

    @property
    def er_rung(self) -> RungReading:
        return self._require(RungLabel.ER_PHYSICAL_BASIS)

    def _require(self, label: RungLabel) -> RungReading:
        try:
            return self.readings[label]
        except KeyError:
            raise MissingRungReadingError(
                f"rung ({label.value}) has no reading. 10.4.10's prediction is a joint "
                "statement over the four rungs: rung (ii) succeeding means nothing "
                "without rungs (i) and (iii) failing at the same time, and the success "
                "rate is read against rung (iv)"
            ) from None

    @property
    def route_p_supported(self) -> bool:
        """Rung (ii) at or under 0.25, clean mu head, within 5 pp of the anchor."""
        er = self.er_rung
        anchor = self._require(RungLabel.GROUND_TRUTH)
        return (
            er.relative_error is not None
            and er.relative_error <= RELATIVE_ERROR_SUCCESS
            and er.mu_head_lambda_ref <= MU_HEAD_LAMBDA_REF_MAX
            and abs(anchor.success_rate_pp - er.success_rate_pp)
            <= SUCCESS_RATE_GAP_MAX_PP
        )

    @property
    def controls_behave(self) -> bool:
        """Rungs (i) and (iii) both at 0.80 or worse, and within 0.10 of each other."""
        no_er = self._require(RungLabel.NO_ER_PHYSICAL_BASIS)
        mlp = self._require(RungLabel.UNCONSTRAINED_MLP)
        assert no_er.relative_error is not None
        assert mlp.relative_error is not None
        return (
            no_er.relative_error >= RELATIVE_ERROR_FAILURE
            and mlp.relative_error >= RELATIVE_ERROR_FAILURE
            and abs(no_er.relative_error - mlp.relative_error) < FAILURE_RUNG_SPREAD_MAX
        )

    @property
    def rung_i_unexpectedly_succeeded(self) -> bool:
        no_er = self._require(RungLabel.NO_ER_PHYSICAL_BASIS)
        assert no_er.relative_error is not None
        return no_er.relative_error <= RELATIVE_ERROR_SUCCESS

    @property
    def verdict(self) -> str:
        if self.rung_i_unexpectedly_succeeded:
            return (
                "algorithm-layer finding: rung (i) recovered delta without the (ER) "
                "excision, so what worked is the inductive bias of span psi_j and not "
                "the exclusion restriction. 10.4.10 forbids counting this as "
                "information-layer evidence for route (P)"
            )
        if self.route_p_supported and self.controls_behave:
            return (
                "route (P) supported: rung (ii) recovers delta under the only "
                "implementable form of (ER), with both unidentifiability controls level "
                "at 0.80 or worse"
            )
        if not self.route_p_supported:
            return (
                "route (P) not instantiable at this scale: even with an implementable "
                "(ER), rung (ii) does not recover delta. Proposition 1(d)'s five-way "
                "choice keeps experimental support for four branches only"
            )
        return (
            "inconclusive: rung (ii) met its prediction but the controls did not "
            "separate, so the reading cannot be attributed to the exclusion restriction"
        )

    def as_report(self) -> dict[str, Any]:
        rows = []
        for rung in ROUTE_P_RUNGS:
            row = rung.as_row()
            reading = self.readings.get(rung.label)
            if reading is not None:
                row["relative_error"] = reading.relative_error
                row["mu_head_lambda_ref"] = reading.mu_head_lambda_ref
                row["success_rate_pp"] = reading.success_rate_pp
                row["surrogate_relative_error"] = reading.surrogate_relative_error
                row["meets_prediction"] = (
                    None
                    if reading.relative_error is None
                    else rung.meets_prediction(reading.relative_error)
                )
            rows.append(row)
        return {
            "experiment": "E10",
            "endpoint": "delta_recovery_error",
            "ground_truth_only": (
                "the comparison object is the stored per-trajectory delta, which exists "
                "only because the simulator provides it; this endpoint has no real-data "
                "counterpart"
            ),
            "rungs": rows,
            "route_p_supported": self.route_p_supported,
            "controls_behave": self.controls_behave,
            "rung_i_unexpectedly_succeeded": self.rung_i_unexpectedly_succeeded,
            "verdict": self.verdict,
        }


def rung(label: RungLabel) -> RoutePRung:
    for candidate in ROUTE_P_RUNGS:
        if candidate.label is label:
            return candidate
    raise KeyError(f"no route (P) rung {label!r}")


def _self_check() -> None:
    """(ER) really removes the psi-carrying channels from the mu head.

    The rung is a table of group names, and a rename or a re-ordering of the
    9.6.1 layout would leave the table looking right while admitting a channel
    it means to exclude. This checks the resolved indices instead.
    """
    excision = ChannelExcision.exclusion_restriction()
    admitted = set(excision.mu_head_channels())
    for name in COMPENSATION_EXCLUSIVE_GROUPS:
        group: ProprioGroup = proprio_group(name)
        if admitted & set(range(group.start, group.stop)):
            raise AssertionError(
                f"the (ER) mu head can still read {name}, which 10.4.10 reserves for "
                "the compensation pathway"
            )
    if not admitted:
        raise AssertionError("the (ER) mu head admits no channel at all")


_self_check()
