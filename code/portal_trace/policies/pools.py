"""Data composition per arm: the two tables of 10.4.0, written as objects.

10.4.0 fixes the pool literally -- "a newcomer can write the first training config
straight off this table" -- and 10.4.0's per-arm pool table says which arm's pool
contains a clean twin, which contains the target embodiment's own data, and what each
of those facts does to how a leakage reading may be interpreted. Those two tables are
the whole content of B1, B2 and B7, so they live here as declarations that the arm
modules point at, and the runner turns into a sampler.

Two of the fields are there to be checked rather than read:

`alpha_demo` and `alpha_frame` must be bit-identical. Under the fixed-length
collection of 9.6.1 every demonstration is 400 frames, so the frame-level mixing
weight equals the nominal demo-level one by construction (X14). The registry stores
both per config and the results table reports both; any difference at all is a bug in
the collection pipeline, not a tolerable deviation, and `check_alpha_identity` is
where that is enforced.

`contains_clean_twin` is what decides how a leakage number reads. B2's pool has no
`delta = 0` anywhere, so route A's anchoring does not hold by construction and the
correct reading of its leakage is "insufficient information" rather than "no
leakage". E3's `||delta||` axis swaps the sixth source for a same-family clean twin
for exactly this reason, and the difference between that arm and B2 is itself the
reading of whether the algorithm uses route A's anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "N_SOURCES",
    "N_TARGET_DEMOS",
    "N_SOURCE_DEMOS",
    "POOLED_DEMOS",
    "POOL_EMBODIMENTS",
    "SubsetName",
    "SUBSET_DESIGN",
    "PoolSpec",
    "POOLS",
    "pool",
    "check_alpha_identity",
]

# 10.4.0's main-arm data table. K is the number of source embodiments; 14.5 E-2
# forbids confusing it with K_hist, the history window length.
N_SOURCES = 6
N_TARGET_DEMOS = 200
N_SOURCE_DEMOS = 200
POOLED_DEMOS = N_TARGET_DEMOS + N_SOURCES * N_SOURCE_DEMOS   # 1400 per (config, task)
# The pool holds the target embodiment plus its K sources; this is the K that the
# hard routers of E9 replicate over.
POOL_EMBODIMENTS = 1 + N_SOURCES


class SubsetName(str, Enum):
    """The five named subsets of 10.4.0.

    Which families and defects each one contains belongs to the registry; what is
    fixed here is the design each implies, because every summary row has to carry its
    own `(M, D, S)` and a subset arm may not borrow the `(6, 6, 3)` numbers.
    """

    S36 = "S36"
    S18 = "S18"
    S9 = "S9"
    S6 = "S6"
    # Spelled the way the configuration spells it. S^(4) is the one subset whose
    # name is not just a cell count -- it is a *family*-layer subset, and the
    # policy layer used to call it "S4" while every YAML file called it "S4fam".
    # Two spellings of one subset is how an arm comes to be filed under a subset
    # that no config declares, so there is one spelling and it is this one.
    S4fam = "S4fam"


# (M families, D defects) per subset; S, the seed count, is per arm.
SUBSET_DESIGN: dict[SubsetName, tuple[int, int]] = {
    SubsetName.S36: (6, 6),
    SubsetName.S18: (3, 6),
    SubsetName.S9: (3, 3),
    SubsetName.S6: (2, 3),
    # S^(4) is a family-layer subset: four families, defect axis set by the experiment.
    SubsetName.S4fam: (4, 1),
}


@dataclass(frozen=True)
class PoolSpec:
    """One row of 10.4.0's per-arm pool table."""

    name: str
    n_target_demos: int
    n_sources: int
    n_source_demos: int
    contains_clean_twin: bool
    target_in_pool: bool
    sources_are_clean_twins: bool
    reading: str

    def __post_init__(self) -> None:
        if self.n_target_demos < 0 or self.n_source_demos < 0 or self.n_sources < 0:
            raise ValueError(f"{self.name}: demonstration counts are non-negative")
        if not self.target_in_pool and self.n_target_demos:
            raise ValueError(
                f"{self.name}: target demonstrations counted but the target is declared "
                "absent from the pool"
            )
        if self.sources_are_clean_twins and not self.contains_clean_twin:
            raise ValueError(
                f"{self.name}: sources are clean twins, so the pool does contain "
                "delta = 0"
            )

    @property
    def total_demos(self) -> int:
        return self.n_target_demos + self.n_sources * self.n_source_demos

    @property
    def alpha_demo(self) -> float:
        """Nominal, demonstration-level mixing weight."""
        total = self.total_demos
        if total == 0:
            raise ValueError(f"{self.name}: an empty pool has no mixing weight")
        return (self.n_sources * self.n_source_demos) / total

    @property
    def alpha_frame(self) -> float:
        """Realised, frame-level mixing weight.

        Equal to `alpha_demo` by construction: every episode is exactly 400 frames
        (9.6.1), so frame counts are demo counts times a constant. Computed from frame
        counts rather than returned as `alpha_demo` so that the identity is exercised
        rather than assumed.
        """
        source_frames = self.n_sources * self.n_source_demos * 400
        target_frames = self.n_target_demos * 400
        total = source_frames + target_frames
        if total == 0:
            raise ValueError(f"{self.name}: an empty pool has no mixing weight")
        return source_frames / total


def check_alpha_identity(spec: PoolSpec) -> None:
    """Enforce `alpha_frame == alpha_demo` bit for bit (X14, 9.6.1 item 4)."""
    if spec.alpha_frame != spec.alpha_demo:
        raise ValueError(
            f"{spec.name}: alpha_frame={spec.alpha_frame!r} differs from "
            f"alpha_demo={spec.alpha_demo!r}. Under the fixed-length collection of "
            "9.6.1 the two are equal by construction, so a difference means early "
            "stopping was left on during collection"
        )


_POOLS: tuple[PoolSpec, ...] = (
    PoolSpec(
        name="B1",
        n_target_demos=N_TARGET_DEMOS,
        n_sources=0,
        n_source_demos=0,
        contains_clean_twin=False,
        target_in_pool=True,
        sources_are_clean_twins=False,
        reading="no-leakage baseline; one seventh of the pooled arms' data",
    ),
    PoolSpec(
        name="B2",
        n_target_demos=N_TARGET_DEMOS,
        n_sources=N_SOURCES,
        n_source_demos=N_SOURCE_DEMOS,
        contains_clean_twin=False,
        target_in_pool=True,
        sources_are_clean_twins=False,
        reading=(
            "route A's anchoring does not hold by construction, so the correct reading "
            "of the leakage number is 'insufficient information'"
        ),
    ),
    PoolSpec(
        name="B7",
        # One cell, not seven: 10.5.5 and the data table of 10.4.0 both say the Oracle's
        # data comes entirely from *this* defective embodiment's clean twin, and that the
        # single clean-twin cell therefore holds 1 400 demonstrations rather than 200.
        # Encoding it as 200 + 6 x 200 would give the same total and the wrong arm.
        n_target_demos=0,
        n_sources=1,
        n_source_demos=POOLED_DEMOS,
        contains_clean_twin=True,
        target_in_pool=False,
        sources_are_clean_twins=True,
        reading=(
            "the primary endpoint's reference point, and not a transfer target: the "
            "clean twin of the evaluated embodiment, matched on reference trajectories, "
            "demonstration count, teacher operator and training budget, differing only "
            "in the defect (10.5.5)"
        ),
    ),
    PoolSpec(
        name="B8",
        n_target_demos=N_TARGET_DEMOS,
        n_sources=N_SOURCES,
        n_source_demos=N_SOURCE_DEMOS,
        contains_clean_twin=False,
        target_in_pool=True,
        sources_are_clean_twins=False,
        reading=(
            "paired demonstration for demonstration with B2; same state distribution, "
            "same data volume, only the action label differs"
        ),
    ),
    PoolSpec(
        name="B2-like",
        n_target_demos=N_TARGET_DEMOS,
        n_sources=N_SOURCES,
        n_source_demos=N_SOURCE_DEMOS,
        contains_clean_twin=False,
        target_in_pool=True,
        sources_are_clean_twins=False,
        reading="B4 / B6 / B9 / B10 / B13 all pool exactly as B2 does",
    ),
    PoolSpec(
        name="E3-delta-axis",
        n_target_demos=N_TARGET_DEMOS,
        n_sources=N_SOURCES,
        n_source_demos=N_SOURCE_DEMOS,
        contains_clean_twin=True,
        target_in_pool=True,
        sources_are_clean_twins=False,
        reading=(
            "the sixth source is a same-family clean twin, because T2's descending "
            "segment needs clean-embodiment data covering rho^off in the pool; the "
            "difference from B2 is itself the reading of whether route A's anchor is "
            "used"
        ),
    ),
    PoolSpec(
        name="E7",
        n_target_demos=N_TARGET_DEMOS,
        n_sources=N_SOURCES,
        n_source_demos=N_SOURCE_DEMOS,
        contains_clean_twin=False,
        target_in_pool=True,
        sources_are_clean_twins=False,
        reading="sources are the reserved new delta instances; the common mode has no anchor",
    ),
    PoolSpec(
        name="E8",
        n_target_demos=N_TARGET_DEMOS,
        n_sources=N_SOURCES,
        n_source_demos=N_SOURCE_DEMOS,
        contains_clean_twin=False,
        target_in_pool=True,
        sources_are_clean_twins=False,
        reading=(
            "rearranged by design-graph level; no delta = 0, so a disconnected level's "
            "null space is not reconnected by a clean embodiment as an extra node"
        ),
    ),
)

POOLS: dict[str, PoolSpec] = {p.name: p for p in _POOLS}


def pool(name: str) -> PoolSpec:
    try:
        return POOLS[name]
    except KeyError:
        raise KeyError(f"no pool spec {name!r}; 10.4.0 defines {sorted(POOLS)}") from None


for _spec in _POOLS:
    check_alpha_identity(_spec)
del _spec
