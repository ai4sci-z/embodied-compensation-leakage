"""A1: what one level of the observation-modality ablation of 10.6.1 says.

A1 is registered as exploratory by preregistration item 16 (hypothesis T3): it
asks which channel the leakage travels down, and 10.6.3's second reading rule
makes its answer a finding to be reported as it comes out rather than a cleanup
step. Its three levels are RGB only, RGB with proprioception (the main setting)
and proprioception only.

Both non-main levels remove a pathway *structurally*, not by zeroing an input.
`ObsSpec` already states the reason for the proprioception side, which B9 shares:
a zeroed input still lets the first layer's bias carry a per-embodiment offset,
so a masked arm and an arm without the pathway are different interventions and
only the second one answers the question. The same argument applies to the image
side, so the proprioception-only level has no visual trunk at all rather than a
trunk fed with black frames.

The task token stays on all three levels. It is not a modality: it is what makes
one policy cover the six core tasks of 10.4.0's evaluation row, and removing it
along with a modality would confound A1 with a task-conditioning ablation.

`ObsSpec` has no image switch -- 10.4.0 fixes two RGB streams and its constructor
refuses an empty camera tuple -- so the level's `use_images` flag lives here and
is consumed by `portal_trace.ablations.modality_nets`. The camera names are still
declared on the proprioception-only level, exactly as `ObsSpec`'s docstring says,
because they name the streams the dataset would have carried and the data card
reports them either way.

Kept free of `torch` for the reason `ablations.capacity` gives: the configuration
loader validates every level block and must not pay for a deep-learning stack to
read text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["ModalitySpec"]


@dataclass(frozen=True)
class ModalitySpec:
    """One A1 level: which of the two sensor pathways the policy is built with."""

    level_id: str
    use_images: bool
    use_proprio: bool

    def __post_init__(self) -> None:
        if not (self.use_images or self.use_proprio):
            raise ValueError(
                f"{self.level_id}: neither images nor proprioception. A policy with "
                "no sensory pathway sees only the task token, which is a "
                "task-prior arm and not a level of the modality ablation"
            )

    @classmethod
    def from_parameters(cls, parameters: dict[str, Any], *, where: str) -> ModalitySpec:
        missing = sorted({"use_images", "use_proprio"} - set(parameters))
        if missing:
            raise ValueError(f"{where}: A1 level is missing {missing}")
        return cls(
            level_id=where,
            use_images=bool(parameters["use_images"]),
            use_proprio=bool(parameters["use_proprio"]),
        )

    @property
    def is_main(self) -> bool:
        """The 10.4.0 setting: both pathways present."""
        return self.use_images and self.use_proprio
