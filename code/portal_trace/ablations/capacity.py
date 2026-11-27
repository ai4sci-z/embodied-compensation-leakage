"""A4: what one level of the capacity sweep of 10.6.1 says.

A4 asks one question: is the leakage an artefact of underfitting? The answer is
only worth anything if the larger arms are genuinely larger *on the pathway the
leakage is supposed to travel down*, which 9.7.3 and 10.5's B9 both put on the
observation side. Widening the decoder head alone would leave the visual trunk
-- the majority of the parameters of the 10.4.0 arm -- untouched, so the sweep
moves the trunk and the head together, exactly as the proposal's three level
labels do: ResNet-18+4 layers, ResNet-34+4, ViT-S+6.

Everything else is held: `d_model`, the token layout of `ObservationEncoder`,
the augmentation, the action space and the pool. 10.4.0 makes the action space
literally identical across arms so that assumption (A6) holds by construction,
and a capacity ablation that also changed the action head would stop being
paired with the main arm.

This module is the half of A4 that the configuration loader needs, and it is
kept free of `torch` deliberately. `portal_trace.policies` pulls in torch and
robomimic at package-import time, and `load_bundle` runs in every CLI
invocation and every test session; a level block is text, and checking that it
is well formed must not cost a deep-learning stack. The networks the levels
name are in `portal_trace.ablations.capacity_nets`, which imports this module.

No level is a default. `CapacitySpec.from_parameters` reads the numbers out of
the preregistered level block of `configs/experiments/ablations.yaml`; there is
no capacity table in this module, because a second copy of the levels is a way
for the config and the code to disagree about what was run.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # `policies.interface` costs a torch import; the annotation does not.
    from portal_trace.policies.interface import PolicyConfig

__all__ = ["FRAME_PIXELS", "CapacitySpec", "VisualBackbone"]

# The frame width of 9.6.1. Duplicated from `policies.obs.IMAGE_SIZE` so that a
# level block can be checked without importing the policy package;
# `capacity_nets` asserts the two agree at import time, so the copy cannot rot.
FRAME_PIXELS = 84


class VisualBackbone(str, Enum):
    """The three trunks of the A4 row. The value is what the config writes."""

    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    VIT_S = "vit_s"


@dataclass(frozen=True)
class CapacitySpec:
    """One A4 level, as the preregistered level block states it.

    `architecture` is the string `runners.train` puts in the model card and
    passes to `build_robomimic_config`, so it is the name under which this level
    appears in every results table. The loader cross-checks it against the
    `architecture` field of the level's arm in `arms.yaml`: the two are the same
    fact written in two files, and 11.4 makes the model card's architecture
    field part of what a summary row means.
    """

    architecture: str
    backbone: VisualBackbone
    n_layers: int
    d_model: int
    n_heads: int
    ffn_dim: int
    patch_size: int | None = None

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError(
                f"{self.architecture}: a transformer head needs at least one layer"
            )
        if self.d_model % self.n_heads:
            raise ValueError(
                f"{self.architecture}: d_model={self.d_model} is not divisible by "
                f"n_heads={self.n_heads}"
            )
        if self.backbone is VisualBackbone.VIT_S:
            if self.patch_size is None:
                raise ValueError(
                    f"{self.architecture}: a ViT level has to state its patch size. "
                    "The patch grid decides the token count of the whole encoder, "
                    "which is what the positional embedding is sized from"
                )
            if FRAME_PIXELS % self.patch_size:
                raise ValueError(
                    f"{self.architecture}: patch size {self.patch_size} does not "
                    f"divide the {FRAME_PIXELS}x{FRAME_PIXELS} frame of 9.6.1. "
                    "Padding the frame would put a learned border token in the grid "
                    "and the trunks would no longer see the same pixels"
                )
        elif self.patch_size is not None:
            raise ValueError(
                f"{self.architecture}: patch_size is a ViT parameter and the level "
                f"declares the {self.backbone.value} trunk"
            )

    @classmethod
    def from_parameters(cls, parameters: dict[str, Any], *, where: str) -> CapacitySpec:
        """Build from an ablation level's `parameters` block.

        Every key is required. A capacity level that inherited `d_model` from
        the main arm's default would be a level whose capacity is decided
        somewhere the reader of the ablation table cannot see.
        """
        missing = sorted(
            {"architecture", "backbone", "n_layers", "d_model", "n_heads", "ffn_dim"}
            - set(parameters)
        )
        if missing:
            raise ValueError(f"{where}: A4 level is missing {missing}")
        backbone = VisualBackbone(str(parameters["backbone"]))
        patch = parameters.get("patch_size")
        return cls(
            architecture=str(parameters["architecture"]),
            backbone=backbone,
            n_layers=int(parameters["n_layers"]),
            d_model=int(parameters["d_model"]),
            n_heads=int(parameters["n_heads"]),
            ffn_dim=int(parameters["ffn_dim"]),
            patch_size=None if patch is None else int(patch),
        )

    def policy_config(self, base: PolicyConfig) -> PolicyConfig:
        """The `PolicyConfig` this level trains under.

        `base` supplies everything the level does not move -- the observation
        spec, the task count, the pool width, the dropout -- so that a
        difference between two A4 arms is the capacity and nothing else.
        """
        return base.evolve(
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            ffn_dim=self.ffn_dim,
            options={**base.options, "capacity": self.architecture},
        )
