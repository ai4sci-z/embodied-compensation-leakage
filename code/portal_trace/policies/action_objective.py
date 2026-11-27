"""Action objectives whose training and deployment semantics stay explicit."""

from __future__ import annotations

import torch
import torch.nn.functional as F

MSE_ACTION_LOSS = "mse"
STRUCTURED_ACTION_LOSS = "continuous-rms-mse+gripper-bce-v1"
STANDARDIZED_ACTION_LOSS = "continuous-standardized-mse+gripper-bce-v2"
ACTION_LOSS_KINDS = frozenset(
    {MSE_ACTION_LOSS, STRUCTURED_ACTION_LOSS, STANDARDIZED_ACTION_LOSS}
)
CONTINUOUS_ACTION_DIMS = 6
GRIPPER_ACTION_DIM = 6


def structured_action_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    continuous_rms: torch.Tensor,
    pad_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    continuous_weight: float = 1.0,
    gripper_weight: float = 1.0,
    loss_kind: str = STRUCTURED_ACTION_LOSS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Six-dimensional normalized regression plus binary gripper classification."""
    if prediction.shape != target.shape or prediction.shape[-1] != 7:
        raise ValueError("structured action loss requires matching (..., 7) tensors")
    if pad_mask.shape != prediction.shape[:2]:
        raise ValueError("structured action loss pad mask does not match action chunk")
    if tuple(continuous_rms.shape) != (CONTINUOUS_ACTION_DIMS,):
        raise ValueError("structured action loss requires six continuous RMS scales")
    if not torch.isfinite(continuous_rms).all() or bool((continuous_rms <= 0).any()):
        raise ValueError("structured action RMS scales must be finite and positive")

    weights = pad_mask.to(prediction.dtype)
    if sample_weight is not None:
        if sample_weight.shape != prediction.shape[:1]:
            raise ValueError("structured action sample weights do not match batch")
        weights = weights * sample_weight.to(prediction.dtype).view(-1, 1)
    denominator = weights.sum()
    if denominator <= 0:
        raise ValueError("structured action loss has no positive-weight action rows")

    scales = continuous_rms.view(1, 1, -1)
    if loss_kind == STRUCTURED_ACTION_LOSS:
        standardized_error = (
            (prediction[..., :CONTINUOUS_ACTION_DIMS] - target[..., :CONTINUOUS_ACTION_DIMS])
            / scales
        ).square()
    elif loss_kind == STANDARDIZED_ACTION_LOSS:
        standardized_target = target[..., :CONTINUOUS_ACTION_DIMS] / scales
        standardized_error = (
            prediction[..., :CONTINUOUS_ACTION_DIMS] - standardized_target
        ).square()
    else:
        raise ValueError(f"unsupported structured action loss kind {loss_kind!r}")
    continuous = (standardized_error * weights.unsqueeze(-1)).sum() / (
        denominator * CONTINUOUS_ACTION_DIMS
    )
    gripper_target = (target[..., GRIPPER_ACTION_DIM] + 1.0) * 0.5
    if bool(((gripper_target < 0) | (gripper_target > 1)).any()):
        raise ValueError("structured gripper targets must lie in [-1, 1]")
    gripper_elements = F.binary_cross_entropy_with_logits(
        prediction[..., GRIPPER_ACTION_DIM], gripper_target, reduction="none"
    )
    gripper = (gripper_elements * weights).sum() / denominator
    total = continuous * continuous_weight + gripper * gripper_weight
    return total, continuous, gripper


def deployment_action_chunk(
    prediction: torch.Tensor,
    *,
    loss_kind: str,
    continuous_rms: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map objective-specific model outputs into canonical action units."""
    if loss_kind == MSE_ACTION_LOSS:
        return prediction
    if loss_kind not in {STRUCTURED_ACTION_LOSS, STANDARDIZED_ACTION_LOSS}:
        raise ValueError(f"unsupported action loss kind {loss_kind!r}")
    if prediction.shape[-1] != 7:
        raise ValueError("structured action output must end in seven action dimensions")
    action = prediction.clone()
    if loss_kind == STANDARDIZED_ACTION_LOSS:
        if continuous_rms is None or tuple(continuous_rms.shape) != (
            CONTINUOUS_ACTION_DIMS,
        ):
            raise ValueError("standardized action deployment requires six RMS scales")
        action[..., :CONTINUOUS_ACTION_DIMS] *= continuous_rms.to(
            dtype=prediction.dtype, device=prediction.device
        )
    action[..., GRIPPER_ACTION_DIM] = torch.where(
        prediction[..., GRIPPER_ACTION_DIM] >= 0,
        torch.ones_like(prediction[..., GRIPPER_ACTION_DIM]),
        -torch.ones_like(prediction[..., GRIPPER_ACTION_DIM]),
    )
    return action
