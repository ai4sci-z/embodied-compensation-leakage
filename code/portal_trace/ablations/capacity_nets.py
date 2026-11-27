"""A4: the three networks the capacity levels of 10.6.1 name.

`portal_trace.ablations.capacity` says what a level *is*; this module builds it.
The split is so that reading `configs/experiments/ablations.yaml` does not pull
in torch and robomimic -- see that module's docstring.

============  =====================================================  ==========
level         visual trunk                                           head depth
============  =====================================================  ==========
resnet18+4    `robomimic.models.base_nets.ResNet18Conv`, the main     4
              arm's trunk, 512x3x3 on an 84x84 frame
resnet34+4    torchvision `resnet34` truncated at layer4, the same    4
              512x3x3 output shape and about 1.8x the parameters
vits+6        a ViT-S/12 written here: 84 = 7 x 12 exactly, so the    6
              patch grid is 7x7 with no padding, 384 wide, 12 deep,
              6 heads
============  =====================================================  ==========

The ResNet-34 trunk is torchvision's rather than robomimic's because robomimic
0.5.0 ships `ResNet18Conv` and `ResNet50Conv` and nothing between them, and
ResNet-50 is a bottleneck architecture whose 2048-channel output would change
the projection width as well as the depth -- two variables where the table asks
for one. The ViT is written out here for the same reason the ResNets are not:
neither torchvision nor robomimic has a ViT-S at 84x84, and the ones they do
have assume 224x224 with a patch grid that does not divide 84.

Weights are randomly initialised on all three levels, as 10.4.0 requires of the
main arm: 9.1.1 wants bit-exact reproduction and pretrained weights are an
unpinned downloaded artefact. `pretrained=True` therefore never appears here,
and a level that asked for it would make the capacity comparison a comparison
of initialisations.
"""

from __future__ import annotations

import torch
from robomimic.models.base_nets import ResNet18Conv
from torch import nn

from portal_trace.ablations.capacity import FRAME_PIXELS, CapacitySpec, VisualBackbone
from portal_trace.policies.decoder import ActionChunkDecoder, FeedForwardFactory
from portal_trace.policies.encoders import (
    ImageAugmentation,
    ObservationEncoder,
    ProprioEncoder,
    VisualTower,
)
from portal_trace.policies.interface import (
    ParameterBlock,
    Policy,
    PolicyConfig,
    dense_blocks,
)
from portal_trace.policies.obs import ACTION_DIM, IMAGE_SIZE, ObsSpec
from portal_trace.policies.portal_policy import PortalTracePolicy

__all__ = [
    "A4CapacityPolicy",
    "CapacityObservationEncoder",
    "CapacityVisualTower",
    "PatchEmbedding",
    "VitTrunk",
    "build_capacity_policy",
]

# `capacity.FRAME_PIXELS` exists so that a level block can be validated without
# importing this stack. It is the same number as 9.6.1's frame width or the
# early check is checking the wrong thing.
assert FRAME_PIXELS == IMAGE_SIZE, (
    f"capacity.FRAME_PIXELS={FRAME_PIXELS} but policies.obs.IMAGE_SIZE={IMAGE_SIZE}"
)

# 10.6.1 names the level "ViT-S"; the small configuration is 384 wide, 12 deep
# and 6 heads, with an MLP ratio of 4. Those four numbers are the definition of
# the size class rather than a tuning choice, so they are constants here and
# the level block states only the patch size, which is the one thing 84x84
# forces a decision about.
VIT_S_WIDTH = 384
VIT_S_DEPTH = 12
VIT_S_HEADS = 6
VIT_S_MLP_RATIO = 4


class PatchEmbedding(nn.Module):
    """Non-overlapping patch projection, stride equal to the kernel."""

    def __init__(self, patch_size: int, width: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _VitBlock(nn.Module):
    """Pre-norm self-attention and MLP, the residual form ViT is defined with."""

    def __init__(self, width: int, heads: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * mlp_ratio),
            nn.GELU(),
            nn.Linear(width * mlp_ratio, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm_attn(x)
        x = x + self.dropout(self.attn(h, h, h, need_weights=False)[0])
        return x + self.dropout(self.mlp(self.norm_mlp(x)))


class VitTrunk(nn.Module):
    """ViT-S over an 84x84 frame, returning a feature map, not a vector.

    The output is (B, 384, gh, gw) rather than a pooled embedding because the
    rest of the encoder treats a trunk's output as a grid of cells and projects
    it with a 1x1 convolution (10.4.0). Pooling to one vector per frame would
    throw away the spatial precision the tight-tolerance tasks need -- the
    adopted tolerance on `NutAssemblySquareTight-v0` is 1.0 mm -- and would make
    the ViT level differ from the ResNet levels in the token layout as well as
    in the trunk.

    There is no class token. It would be a 197th token with no counterpart in
    the two ResNet levels, and the encoder's positional embedding is sized from
    the cell count.
    """

    def __init__(self, patch_size: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        if IMAGE_SIZE % patch_size:
            raise ValueError(
                f"patch size {patch_size} does not divide the {IMAGE_SIZE}x"
                f"{IMAGE_SIZE} frame of 9.6.1"
            )
        self.grid = IMAGE_SIZE // patch_size
        self.width = VIT_S_WIDTH
        self.patch = PatchEmbedding(patch_size, VIT_S_WIDTH)
        self.position = nn.Parameter(torch.zeros(1, self.grid * self.grid, VIT_S_WIDTH))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList(
            _VitBlock(VIT_S_WIDTH, VIT_S_HEADS, VIT_S_MLP_RATIO, dropout)
            for _ in range(VIT_S_DEPTH)
        )
        self.norm = nn.LayerNorm(VIT_S_WIDTH)

    def output_shape(self, input_shape: list[int]) -> list[int]:
        """Channels, height, width -- the signature robomimic's trunks expose."""
        del input_shape
        return [self.width, self.grid, self.grid]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(x).flatten(2).transpose(1, 2) + self.position
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        b = tokens.shape[0]
        return tokens.transpose(1, 2).reshape(b, self.width, self.grid, self.grid)


class _ResNet34Trunk(nn.Module):
    """torchvision `resnet34` up to and including layer4.

    Truncated exactly where robomimic truncates its ResNet-18: the average pool
    and the classifier are dropped, so an 84x84 input leaves a 512x3x3 map and
    the ResNet-34 level differs from the ResNet-18 level in depth alone.
    """

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import resnet34

        # 9.1.1: no downloaded weights anywhere in the package.
        net = resnet34(weights=None)
        self.net = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
        )

    def output_shape(self, input_shape: list[int]) -> list[int]:
        c, h, w = input_shape
        with torch.no_grad():
            out = self.net(torch.zeros(1, c, h, w))
        return [int(out.shape[1]), int(out.shape[2]), int(out.shape[3])]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_trunk(backbone: VisualBackbone, patch_size: int | None) -> nn.Module:
    if backbone is VisualBackbone.RESNET18:
        return ResNet18Conv(input_channel=3, pretrained=False)
    if backbone is VisualBackbone.RESNET34:
        return _ResNet34Trunk()
    if patch_size is None:
        raise ValueError("the ViT trunk needs a patch size")
    return VitTrunk(patch_size)


# ---------------------------------------------------------------------------
# the encoder and the policy
# ---------------------------------------------------------------------------


class CapacityVisualTower(VisualTower):
    """`VisualTower` with the trunk chosen by the A4 level.

    `nn.Module.__init__` is called directly instead of the base class's, because
    `VisualTower.__init__` constructs one ResNet-18 per camera unconditionally
    and building the largest module of the arm only to discard it would double
    the initialisation cost of every A4 run. Every attribute the inherited
    `forward` reads is set below, and the trunk still exposes `output_shape`, so
    the two classes stay interchangeable.
    """

    def __init__(
        self,
        spec: ObsSpec,
        d_model: int,
        *,
        backbone: VisualBackbone,
        patch_size: int | None = None,
        augment: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        self.spec = spec
        self.d_model = d_model
        self.backbone = backbone
        self.trunks = nn.ModuleDict(
            {self._key(cam): _build_trunk(backbone, patch_size) for cam in spec.cameras}
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


class CapacityObservationEncoder(ObservationEncoder):
    """`ObservationEncoder` over a `CapacityVisualTower`.

    Same reason as above for bypassing the base constructor. The token count
    changes with the trunk -- a ViT-S/12 grid is 7x7 against the ResNets' 3x3 --
    so the positional embedding is sized here rather than inherited.
    """

    def __init__(
        self,
        spec: ObsSpec,
        d_model: int,
        n_tasks: int,
        *,
        backbone: VisualBackbone,
        patch_size: int | None = None,
        augment: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        self.spec = spec
        self.d_model = d_model
        self.visual = CapacityVisualTower(
            spec, d_model, backbone=backbone, patch_size=patch_size, augment=augment
        )
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


class A4CapacityPolicy(PortalTracePolicy):
    """The 10.4.0 architecture at one of A4's three capacities.

    It is a subclass rather than a sibling for the reason 10.4.0 gives for the
    baselines: the four hooks (`conditioning`, `action_space_descriptor`,
    `action_target`, `auxiliary_losses`) are untouched, so a reader can see that
    A4 changes the trunk and the head depth and nothing about what is learned.
    """

    arm = "A4-capacity"

    def __init__(
        self,
        cfg: PolicyConfig,
        *,
        capacity: CapacitySpec,
        cond_dim: int = 0,
        action_dim: int = ACTION_DIM,
        feed_forward_factory: FeedForwardFactory | None = None,
    ) -> None:
        # `Policy.__init__`, not `PortalTracePolicy.__init__`: the latter builds
        # the fixed ResNet-18 encoder, which is exactly the object this arm
        # replaces.
        Policy.__init__(self, cfg)
        if cfg.n_layers != capacity.n_layers or cfg.d_model != capacity.d_model:
            raise ValueError(
                f"{capacity.architecture}: the PolicyConfig says "
                f"({cfg.d_model}, {cfg.n_layers}) and the level says "
                f"({capacity.d_model}, {capacity.n_layers}). Build the config with "
                "CapacitySpec.policy_config so that the model card and the network "
                "cannot disagree"
            )
        self.capacity = capacity
        self.encoder = CapacityObservationEncoder(
            cfg.obs,
            cfg.d_model,
            cfg.n_tasks,
            backbone=capacity.backbone,
            patch_size=capacity.patch_size,
            augment=cfg.augment,
        )
        self.decoder = ActionChunkDecoder(
            cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.dropout,
            cond_dim=cond_dim,
            action_dim=action_dim,
            feed_forward_factory=feed_forward_factory,
        )

    def parameter_blocks(self) -> tuple[ParameterBlock, ...]:
        return dense_blocks(self, self.cfg.n_pool_embodiments)


def build_capacity_policy(
    capacity: CapacitySpec, *, base: PolicyConfig | None = None
) -> A4CapacityPolicy:
    """Build the arm for one A4 level.

    `base` defaults to the 10.4.0 configuration under this level's architecture
    name, which is what makes the three levels differ in the four capacity
    fields and in nothing else.
    """
    cfg = capacity.policy_config(base or PolicyConfig(arm=capacity.architecture))
    return A4CapacityPolicy(cfg, capacity=capacity)
