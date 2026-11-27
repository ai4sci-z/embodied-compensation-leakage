"""The visual tower and the proprioception pathway of 10.4.0.

One ResNet-18 per camera, not one shared across both: `agentview` is world-fixed
and `robot0_eye_in_hand` rides the gripper, so their statistics have nothing in
common and sharing the trunk would be a capacity decision disguised as a symmetry.
This follows robomimic, which builds a separate visual core per image key, and the
backbone itself is `robomimic.models.base_nets.ResNet18Conv` rather than a private
copy.

Each 84x84 frame leaves the trunk as a 512x3x3 map, and all nine cells are kept as
tokens after a 1x1 projection. The alternative -- spatial-softmax keypoint pooling
to a single vector per frame, robomimic's own default for its BC recipes -- would
leave the 4-layer decoder head with `2 * K_hist` keys to cross-attend over and
throw away exactly the spatial precision the tight-tolerance tasks need: the
adopted tolerance on `NutAssemblySquareTight-v0` is 6.75 mm, which is 0.135 action
units on the 9.6.1 scale.

Weights are randomly initialised. ImageNet initialisation is the usual robomimic
choice, but 9.1.1 requires bit-exact reproduction and pretrained weights are an
unpinned downloaded artefact.

Augmentation is random crop plus colour jitter (10.4.0) and nothing else. There is
no action-side augmentation anywhere in this package: it would be confounded with
the compensation field by construction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from robomimic.models.base_nets import MLP, ResNet18Conv
from torch import nn

from portal_trace.policies.obs import PROPRIO_DIM, ObsSpec

__all__ = [
    "ImageAugmentation",
    "VisualTower",
    "ProprioEncoder",
    "ObservationEncoder",
]

# DrQ-style shift augmentation: pad by 4 px and crop back to 84x84, so the token
# grid keeps its shape and the positional embedding stays meaningful.
_SHIFT_PAD = 4
_BRIGHTNESS = 0.3
_CONTRAST = 0.3
_SATURATION = 0.3

_GREY = (0.299, 0.587, 0.114)


class ImageAugmentation(nn.Module):
    """Random shift and colour jitter, applied per sample, training only.

    Per sample rather than per batch: a single jitter factor for the whole batch
    correlates the perturbation with the gradient step and buys much less
    invariance for the same cost.
    """

    def __init__(self, pad: int = _SHIFT_PAD) -> None:
        super().__init__()
        self.pad = pad
        grey = torch.tensor(_GREY).view(1, 3, 1, 1)
        self.register_buffer("grey", grey, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        return self._jitter(self._shift(x))

    def _shift(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        padded = F.pad(x, (self.pad,) * 4, mode="replicate")
        # Sample the crop as a translation of the identity sampling grid, which
        # keeps the whole operation one batched grid_sample instead of N slices.
        eps_h = self.pad / (h + 2 * self.pad)
        eps_w = self.pad / (w + 2 * self.pad)
        base_y = torch.linspace(-1.0 + eps_h, 1.0 - eps_h, h, device=x.device, dtype=x.dtype)
        base_x = torch.linspace(-1.0 + eps_w, 1.0 - eps_w, w, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(base_y, base_x, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(n, h, w, 2)
        offset = torch.empty(n, 1, 1, 2, device=x.device, dtype=x.dtype).uniform_(-1.0, 1.0)
        offset = offset * torch.tensor([eps_w, eps_h], device=x.device, dtype=x.dtype)
        return F.grid_sample(padded, grid + offset, align_corners=False, padding_mode="border")

    def _jitter(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]

        def factor(strength: float) -> torch.Tensor:
            return torch.empty(n, 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(
                1.0 - strength, 1.0 + strength
            )

        x = x * factor(_BRIGHTNESS)
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = mean + (x - mean) * factor(_CONTRAST)
        grey = (x * self.grey.to(dtype=x.dtype)).sum(dim=1, keepdim=True)
        x = grey + (x - grey) * factor(_SATURATION)
        return x.clamp_(0.0, 1.0)


class VisualTower(nn.Module):
    """ResNet-18 per camera, spatial cells projected to `d_model` tokens."""

    def __init__(self, spec: ObsSpec, d_model: int, *, augment: bool = True) -> None:
        super().__init__()
        self.spec = spec
        self.d_model = d_model
        self.trunks = nn.ModuleDict(
            {
                self._key(cam): ResNet18Conv(input_channel=3, pretrained=False)
                for cam in spec.cameras
            }
        )
        trunk = next(iter(self.trunks.values()))
        self.feature_channels, self.grid_h, self.grid_w = trunk.output_shape(
            [3, spec.image_size, spec.image_size]
        )
        self.cells = self.grid_h * self.grid_w
        self.projections = nn.ModuleDict(
            {
                self._key(cam): nn.Conv2d(self.feature_channels, d_model, kernel_size=1)
                for cam in spec.cameras
            }
        )
        self.augment = ImageAugmentation() if augment else None

    @staticmethod
    def _key(camera: str) -> str:
        # ModuleDict keys cannot contain a dot; camera names do not, but the
        # robosuite prefix convention makes that an accident rather than a rule.
        return camera.replace(".", "_")

    @property
    def n_tokens(self) -> int:
        return self.spec.n_cameras * self.spec.k_hist * self.cells

    def forward(self, images: dict[str, torch.Tensor]) -> torch.Tensor:
        missing = [c for c in self.spec.cameras if c not in images]
        if missing:
            raise ValueError(
                f"missing camera streams {missing}; 10.4.0 fixes both "
                f"{list(self.spec.cameras)} and an arm cannot silently drop one"
            )
        tokens: list[torch.Tensor] = []
        for cam in self.spec.cameras:
            x = images[cam]
            if x.ndim != 5:
                raise ValueError(
                    f"{cam}: expected (B, K, 3, H, W), got {tuple(x.shape)}"
                )
            b, k = x.shape[:2]
            if k != self.spec.k_hist:
                raise ValueError(
                    f"{cam}: {k} history frames, ObsSpec declares {self.spec.k_hist}"
                )
            flat = x.reshape(b * k, *x.shape[2:])
            if self.augment is not None:
                flat = self.augment(flat)
            key = self._key(cam)
            feat = self.projections[key](self.trunks[key](flat))
            tokens.append(feat.flatten(2).transpose(1, 2).reshape(b, k * self.cells, self.d_model))
        return torch.cat(tokens, dim=1)


class ProprioEncoder(nn.Module):
    """One token per history frame from the 70-dimensional vector."""

    def __init__(self, d_model: int, *, hidden: int = 512) -> None:
        super().__init__()
        self.net = MLP(
            input_dim=PROPRIO_DIM,
            output_dim=d_model,
            layer_dims=(hidden,),
            activation=nn.GELU,
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        if proprio.ndim != 3 or proprio.shape[-1] != PROPRIO_DIM:
            raise ValueError(
                f"expected proprioception of shape (B, K, {PROPRIO_DIM}), got "
                f"{tuple(proprio.shape)}; the width is 50 native + 7 tau_g + 6 wrist "
                "F/T + 7 DoF mask"
            )
        b, k, _ = proprio.shape
        return self.net(proprio.reshape(b * k, -1)).reshape(b, k, -1)


class ObservationEncoder(nn.Module):
    """Images, proprioception and the task instruction as one token sequence.

    The task token is the "task instruction" channel that B9 keeps when it strips
    proprioception, and it is what makes one policy per config cover the six core
    tasks it is evaluated on (10.4.0's evaluation row).
    """

    def __init__(self, spec: ObsSpec, d_model: int, n_tasks: int, *, augment: bool = True) -> None:
        super().__init__()
        self.spec = spec
        self.d_model = d_model
        self.visual = VisualTower(spec, d_model, augment=augment)
        self.proprio = ProprioEncoder(d_model) if spec.use_proprio else None
        self.task_embedding = nn.Embedding(n_tasks, d_model) if spec.use_task_token else None
        self.n_tokens = (
            self.visual.n_tokens
            + (spec.k_hist if spec.use_proprio else 0)
            + (1 if spec.use_task_token else 0)
        )
        self.position = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        images: dict[str, torch.Tensor],
        proprio: torch.Tensor | None,
        task_index: torch.Tensor | None,
    ) -> torch.Tensor:
        parts = [self.visual(images)]
        if self.proprio is not None:
            if proprio is None:
                raise ValueError(
                    "this arm's ObsSpec declares proprioception; Observation.proprio "
                    "is None"
                )
            parts.append(self.proprio(proprio))
        elif proprio is not None:
            raise ValueError(
                "this arm has no proprioception pathway (B9); passing "
                "Observation.proprio would silently discard it"
            )
        if self.task_embedding is not None:
            if task_index is None:
                raise ValueError(
                    "this arm conditions on the task instruction; "
                    "Observation.task_index is None"
                )
            parts.append(self.task_embedding(task_index).unsqueeze(1))
        tokens = torch.cat(parts, dim=1)
        return self.norm(tokens + self.position)
