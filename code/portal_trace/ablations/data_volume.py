"""A5: the data-volume sweep of 10.6.1, 200 / 400 / 800 demonstrations per cell.

The prediction the row exists to test is a negative one: leakage is not an
estimation error, so it should not fall as the pool grows. That prediction is
only readable if the *composition* of the pool is held while its size moves.
The 10.4.0 data ratio table has two independent columns -- `n_target_demos` and
`n_source_demos` -- and scaling only one of them would change alpha, the source
share, which is E3's second dose axis. A5 would then be measuring E3's axis
under a different name.

So the invariant this module enforces is: every A5 level scales both columns by
the same integer factor, keeps `k_sources` at the main arm's 6, and therefore
reproduces the main arm's alpha_demo of 6*200 / 1400 = 0.857 exactly. 9.6.1
fixes the horizon at 400 with early stopping off, so alpha_frame equals
alpha_demo bit for bit and the invariant covers both calibers at once.

The budget is expressed where `runners.train` reads it. `resolve_pool` turns
`PoolComposition.n_target_demos` and `.n_source_demos` into a per-cell file
list, and `TrainingJob.pool.n_demos` is what the 11.4 data card publishes; a
level is therefore a pool block in `subsets.yaml` plus the arm that names it,
and this module is what checks that the two say the same thing as the level
block in `ablations.yaml`.

Sharing with W1. The proposal's A5 row says the levels are shared with W1's
"are 200 demonstrations enough" learning curve, which is why the ladder starts
at the main arm's 200 rather than below it: the main setting has to be one of
the levels, or the curve has no anchor at the value every other chapter used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from portal_trace.configs.schema import ConfigError, PoolComposition

__all__ = ["DataVolumeSpec", "MAIN_ARM_DEMOS_PER_CELL", "check_pool_matches_level"]

# 10.4.0's data ratio table: 200 target demonstrations and 200 per source, six
# sources, 1400 in the pool. A5's factors are stated against this number.
MAIN_ARM_DEMOS_PER_CELL = 200


@dataclass(frozen=True)
class DataVolumeSpec:
    """One A5 level: a demonstration budget per (config, task).

    `factor` is carried rather than derived so that the configuration states it
    and the code checks it, instead of the code deciding what the level meant.
    """

    level_id: str
    n_target_demos: int
    n_source_demos: int
    k_sources: int
    factor: int

    def __post_init__(self) -> None:
        if self.factor < 1:
            raise ConfigError(f"{self.level_id}: a data-volume factor is at least 1")
        expected = MAIN_ARM_DEMOS_PER_CELL * self.factor
        if self.n_target_demos != expected or self.n_source_demos != expected:
            raise ConfigError(
                f"{self.level_id}: declares factor {self.factor} against the 10.4.0 "
                f"table's {MAIN_ARM_DEMOS_PER_CELL} demonstrations per cell, which is "
                f"{expected}, but writes target={self.n_target_demos} and "
                f"source={self.n_source_demos}. A5 scales both columns together; "
                "scaling one of them moves alpha, which is E3's second dose axis and "
                "not this one"
            )
        if self.k_sources != 6:
            raise ConfigError(
                f"{self.level_id}: k_sources is {self.k_sources}. A5 moves the volume, "
                "not the number of sources; K is E4b's and E7's independent variable"
            )

    @classmethod
    def from_parameters(cls, parameters: dict[str, Any], *, where: str) -> DataVolumeSpec:
        missing = sorted(
            {"n_target_demos", "n_source_demos", "k_sources", "factor"} - set(parameters)
        )
        if missing:
            raise ConfigError(f"{where}: A5 level is missing {missing}")
        return cls(
            level_id=where,
            n_target_demos=int(parameters["n_target_demos"]),
            n_source_demos=int(parameters["n_source_demos"]),
            k_sources=int(parameters["k_sources"]),
            factor=int(parameters["factor"]),
        )

    @property
    def pool_size(self) -> int:
        return self.n_target_demos + self.k_sources * self.n_source_demos

    @property
    def alpha_demo(self) -> float:
        return (self.k_sources * self.n_source_demos) / self.pool_size


def check_pool_matches_level(pool: PoolComposition, spec: DataVolumeSpec) -> None:
    """Refuse an A5 arm whose pool block disagrees with its level block.

    The level block is what the ablation table publishes and the pool block is
    what `runners.train.resolve_pool` actually reads. A disagreement would put
    one number in the paper and a different one in the run, which is the failure
    11.3 item 10 makes the pool table part of the preregistration to prevent.
    """
    if (
        pool.n_target_demos != spec.n_target_demos
        or pool.n_source_demos != spec.n_source_demos
        or pool.k_sources != spec.k_sources
    ):
        raise ConfigError(
            f"A5 level {spec.level_id}: the level declares "
            f"({spec.n_target_demos}, {spec.n_source_demos}, {spec.k_sources}) and the "
            f"pool block for arm {pool.arm} declares "
            f"({pool.n_target_demos}, {pool.n_source_demos}, {pool.k_sources}). "
            "runners.train reads the pool block, so the ablation table would report a "
            "budget that was not trained on"
        )
    # Held to the tolerance the data card rounds alpha to (10.4.0 prints 0.857).
    main_alpha = (6 * MAIN_ARM_DEMOS_PER_CELL) / (7 * MAIN_ARM_DEMOS_PER_CELL)
    if abs(pool.alpha_demo - main_alpha) > 5e-4:
        raise ConfigError(
            f"A5 level {spec.level_id}: alpha_demo is {pool.alpha_demo:.4f} against the "
            f"main arm's {main_alpha:.4f}. The volume sweep has to hold the source "
            "share, or its slope is confounded with E3's alpha axis"
        )
