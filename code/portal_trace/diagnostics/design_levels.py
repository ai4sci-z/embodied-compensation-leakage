"""E8's six design-graph rungs: the data ratio table, as a graph per rung.

Proposal 10.4.8. E8 is one of the three crowns and the only carrier of the
empirical bridge T11, and its independent variable is unusual in that it costs
nothing to produce: the six rungs are six ways of writing the same 1 200
demonstrations per morphology family into a (task, defect) allocation table, so
the whole sweep is a data-subset choice over demonstrations that already exist.

The rungs sit on 6 tasks x 2 defects (D4 and D3b). A **bridge task** carries
demonstrations on both defects and is the only kind of task that connects the
two defect nodes; a non-bridge task hangs off one defect and contributes
nothing to the contrast between them. The rung is named by the bridge count b,
and the whole R_eff column of 10.4.8's table follows from b alone: b parallel
paths of two unit resistors each give R_eff(D4, D3b) = 2 / b, which is 0.333 at
b = 6 and 2.0 at b = 1, and is infinite at b = 0 because the graph then has two
components. `_self_check` recomputes all six from the allocation at import, so
the table is checked rather than transcribed.

Edge weights and the normalisation
----------------------------------
`design_graph.DesignGraph` accepts `EffectiveSampleCount` and nothing else,
because 11.4 and appendix A-8 forbid a raw frame count as a conductance. 10.4.8
prints its R_eff column "at normalised edge weights (a bridge task's 100
demonstrations per defect is w = 1)" and says the absolute values are recomputed
per cell from W2's measured tau_corr.

Both readings are here and they are different objects. `r_eff_normalised`
multiplies the graph's R_eff by the weight of the 100-demonstration reference
edge, which cancels tau_corr and the horizon exactly -- that is what makes it a
*relative* number computable before any data exists -- and it is available only
on a graph whose cells share one tau_corr, because with per-cell tau_corr there
is no single reference edge to divide by. `r_eff_absolute` is the graph's own
R_eff in inverse effective samples and is the reading that needs the lock file.
Asking for the normalised value on a measured graph raises rather than picking
one cell's tau_corr to normalise by.

What is left to choose, and how
-------------------------------
10.4.8 fixes b, the per-task demonstration counts and rung B's 3 + 3 split, and
leaves two things open. Which tasks bridge at C1-C4 is one: it does not move
R_eff, lambda_2 or Xi_G at all, since the task nodes are interchangeable at
equal weight, but the allocation table has to name tasks, so bridges are taken
alternately from the two halves 9.5.4 preregisters for rung B, which keeps both
halves represented at every rung. Which defect a non-bridge task hangs off is
the other: it does move the defect marginals, so non-bridge tasks alternate
between the two defects, which is the allocation that keeps the marginals as
close to equal as a 200-demonstration granularity allows. Both choices are
recorded on the rung and printed with it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from portal_trace.diagnostics.design_graph import DesignGraph
from portal_trace.diagnostics.units import EffectiveSampleCount
from portal_trace.registry.tasks import CORE_TASKS, HORIZON

__all__ = [
    "E8_DEFECTS",
    "E8_TASK_HALF_A",
    "E8_TASK_HALF_B",
    "E8_TASKS",
    "DEMOS_PER_FAMILY",
    "BRIDGE_DEMOS_PER_DEFECT",
    "NON_BRIDGE_DEMOS",
    "NORMALISED_UNIT_DEMOS",
    "NOMINAL_TAU_CORR",
    "DesignRung",
    "RungGraph",
    "E8_RUNGS",
    "DesignGraphLadder",
    "NonUniformTauCorrError",
    "DisconnectedRungError",
    "MarginalMismatchError",
]

# 10.4.8: the design graph is 6 tasks x 2 defects, and the two defects are D4
# (backlash) and D3b (gravity sag), in the order the table's R_eff column names
# them.
E8_DEFECTS: tuple[str, ...] = ("D4", "D3b")

# 9.5.4 preregisters the 3 + 3 split that rung B's disconnection is built from:
# each half carries at least one kappa >= 0.8 approach-contact task and one that
# is not, so neither half degenerates into a single task type. That section names
# the nut tasks by their unmodified environments; the arms train on the tight
# variants of 9.5.4, which is what 10.1's core six lists, so the split is
# transcribed under the ids the arms actually run.
E8_TASK_HALF_A: tuple[str, ...] = ("NutAssemblySquareTight-v0", "Stack", "Door")
E8_TASK_HALF_B: tuple[str, ...] = ("NutAssemblyRoundTight-v0", "ToolHang", "Wipe")
E8_TASKS: tuple[str, ...] = (*E8_TASK_HALF_A, *E8_TASK_HALF_B)

# 10.4.8's demonstration budget, per morphology family, identical on every rung.
DEMOS_PER_FAMILY = 1_200
BRIDGE_DEMOS_PER_DEFECT = 100
NON_BRIDGE_DEMOS = 200

# The anchor of the table's normalised column: "a bridge task's 100
# demonstrations per defect corresponds to w = 1".
NORMALISED_UNIT_DEMOS = BRIDGE_DEMOS_PER_DEFECT

# A design-time graph has no measured tau_corr, and `EffectiveSampleCount`
# refuses to be built without one. This value is what a uniform-tau_corr graph
# is built at; it cancels out of every quantity this module normalises, and
# `_self_check` verifies the cancellation against a second value rather than
# asserting it in prose.
NOMINAL_TAU_CORR = 1.0


class NonUniformTauCorrError(ValueError):
    """A normalised reading was asked for on a graph with per-cell tau_corr."""


class DisconnectedRungError(ValueError):
    """A quantity that needs a finite R_eff was asked for on rung B."""


class MarginalMismatchError(ValueError):
    """Two rungs that 10.4.8 matches by construction do not match."""


def _self_check_tasks() -> None:
    if len(set(E8_TASKS)) != len(E8_TASKS):
        raise AssertionError(f"the two halves of 9.5.4 overlap: {E8_TASKS}")
    if set(E8_TASKS) != set(CORE_TASKS):
        raise AssertionError(
            f"10.4.8 builds E8's design graph on all six core tasks; the split names "
            f"{sorted(E8_TASKS)} against a core set of {sorted(CORE_TASKS)}"
        )


_self_check_tasks()


@dataclass(frozen=True)
class DesignRung:
    """One row of 10.4.8's table, with the allocation it is realised by.

    `registered_r_eff` and `registered_ln_r_eff` are the printed values; the
    rung computes its own from the allocation and `DesignGraphLadder.check`
    refuses a disagreement. `is_upper_anchor` marks rung B, which 10.4.8 demotes
    from the primary criterion to an anchor reported as approximately true by
    construction.
    """

    label: str
    n_bridge_tasks: int
    registered_r_eff: float
    registered_ln_r_eff: float | None
    connected: bool
    is_upper_anchor: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.n_bridge_tasks <= len(E8_TASKS):
            raise ValueError(
                f"rung {self.label}: b = {self.n_bridge_tasks} outside "
                f"[0, {len(E8_TASKS)}]"
            )
        if self.connected != (self.n_bridge_tasks > 0):
            raise ValueError(
                f"rung {self.label}: connectivity of the two defect nodes is exactly "
                "'there is at least one bridge task'; b = "
                f"{self.n_bridge_tasks} with connected={self.connected}"
            )
        if self.connected == math.isinf(self.registered_r_eff):
            raise ValueError(
                f"rung {self.label}: a connected rung has a finite R_eff and a "
                "disconnected one has R_eff = infinity"
            )

    @property
    def bridge_tasks(self) -> tuple[str, ...]:
        """Bridges taken alternately from the two halves of 9.5.4."""
        interleaved = [
            task
            for pair in zip(E8_TASK_HALF_A, E8_TASK_HALF_B, strict=True)
            for task in pair
        ]
        return tuple(sorted(interleaved[: self.n_bridge_tasks], key=E8_TASKS.index))

    @property
    def non_bridge_tasks(self) -> tuple[str, ...]:
        bridges = set(self.bridge_tasks)
        return tuple(t for t in E8_TASKS if t not in bridges)

    def allocation(self) -> dict[tuple[str, str], int]:
        """Demonstrations per (task, defect) for one morphology family.

        Rung B is the 3 + 3 split of 9.5.4 verbatim -- half A only on D4, half B
        only on D3b -- rather than the alternating assignment the C rungs use,
        because its two components are the point of the rung.
        """
        out: dict[tuple[str, str], int] = {}
        if self.n_bridge_tasks == 0:
            for task in E8_TASK_HALF_A:
                out[(task, E8_DEFECTS[0])] = NON_BRIDGE_DEMOS
            for task in E8_TASK_HALF_B:
                out[(task, E8_DEFECTS[1])] = NON_BRIDGE_DEMOS
            return out
        for task in self.bridge_tasks:
            for defect in E8_DEFECTS:
                out[(task, defect)] = BRIDGE_DEMOS_PER_DEFECT
        for i, task in enumerate(self.non_bridge_tasks):
            out[(task, E8_DEFECTS[i % len(E8_DEFECTS)])] = NON_BRIDGE_DEMOS
        return out

    @property
    def n_demos(self) -> int:
        return sum(self.allocation().values())

    def task_marginals(self) -> dict[str, int]:
        out = dict.fromkeys(E8_TASKS, 0)
        for (task, _), n in self.allocation().items():
            out[task] += n
        return out

    def defect_marginals(self) -> dict[str, int]:
        out = dict.fromkeys(E8_DEFECTS, 0)
        for (_, defect), n in self.allocation().items():
            out[defect] += n
        return out

    def graph(
        self,
        *,
        tau_corr: float | Mapping[tuple[str, str], float] = NOMINAL_TAU_CORR,
        horizon: int = HORIZON,
    ) -> RungGraph:
        """Build this rung's design graph at a uniform or a per-cell tau_corr.

        A scalar `tau_corr` gives the design-time graph the normalised column of
        10.4.8 is read off; a mapping gives the measured graph, whose R_eff is
        absolute and whose normalised reading is refused.
        """
        allocation = self.allocation()
        uniform = not isinstance(tau_corr, Mapping)
        edges: dict[tuple[str, str], EffectiveSampleCount] = {}
        for cell, n_demos in allocation.items():
            if uniform:
                tc = float(tau_corr)  # type: ignore[arg-type]
            else:
                if cell not in tau_corr:
                    raise KeyError(
                        f"rung {self.label}: no measured tau_corr for cell {cell}. "
                        "Edge weights are effective frame counts n / tau_corr (11.4), "
                        "and tau_corr is per cell and per regime -- produce it with "
                        "scripts/measure_constants.py --what tau_corr"
                    )
                tc = float(tau_corr[cell])
            edges[cell] = EffectiveSampleCount.from_trajectories(
                n_traj=n_demos, horizon=horizon, tau_corr=tc
            )
        reference = (
            EffectiveSampleCount.from_trajectories(
                n_traj=NORMALISED_UNIT_DEMOS,
                horizon=horizon,
                tau_corr=float(tau_corr),  # type: ignore[arg-type]
            )
            if uniform
            else None
        )
        return RungGraph(rung=self, graph=DesignGraph(edges), reference_weight=reference)


@dataclass(frozen=True)
class RungGraph:
    """One rung's graph, with the two R_eff calibers kept apart."""

    rung: DesignRung
    graph: DesignGraph
    reference_weight: EffectiveSampleCount | None

    @property
    def uniform_tau_corr(self) -> bool:
        return self.reference_weight is not None

    def r_eff_absolute(self) -> float:
        """R_eff(D4, D3b) in inverse effective samples, the reading of 7.9."""
        return self.graph.r_eff(*E8_DEFECTS, side="defect")

    def _require_reference(self, what: str) -> float:
        if self.reference_weight is None:
            raise NonUniformTauCorrError(
                f"rung {self.rung.label}: {what} is 10.4.8's normalised column, whose "
                "unit is the 100-demonstration bridge edge. With a per-cell tau_corr "
                "there is no single reference edge, and normalising by one cell's "
                "value would make the number depend on which cell was picked. Report "
                "the absolute reading instead; 10.4.8 says the absolute values are "
                "recomputed per cell from W2's measured tau_corr"
            )
        return float(self.reference_weight)

    def r_eff_normalised(self) -> float:
        """R_eff at the table's caliber: w = 1 for a 100-demonstration bridge edge.

        Multiplying the absolute R_eff by the reference edge weight cancels the
        horizon and tau_corr exactly, which is why this number is computable
        before a single trajectory has been collected.
        """
        reference = self._require_reference("R_eff")
        return self.r_eff_absolute() * reference

    def ln_r_eff(self) -> float:
        """The regressor `connectivity_slope` reads (10.4.8, 10.7.5)."""
        value = self.r_eff_normalised()
        if math.isinf(value):
            raise DisconnectedRungError(
                f"rung {self.rung.label} has two components, so R_eff(D4, D3b) is "
                "infinite and ln R_eff does not exist. 10.4.8 estimates the slope on "
                "the five connected rungs A and C1-C4; rung B is the upper anchor and "
                "is reported beside the slope, not on it"
            )
        return math.log(value)

    def lambda2_normalised(self) -> float:
        """lambda_2(G) at the same caliber; 10.4.8 reports it alongside R_eff."""
        return self.graph.lambda2() / self._require_reference("lambda_2")

    def xi_g(self) -> float:
        """Xi_G, which is invariant under a global rescaling of the edge weights.

        L_G^+ picks up 1/c when every conductance is scaled by c and W picks up
        c, so the bias map L_G^+ M^T W is unchanged: Xi_G needs no normalisation
        and is the same number on the design-time and the measured graph as long
        as the measured tau_corr is uniform.
        """
        return self.graph.xi_g()

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "rung": self.rung.label,
            "b": self.rung.n_bridge_tasks,
            "bridge_tasks": list(self.rung.bridge_tasks),
            "non_bridge_tasks": list(self.rung.non_bridge_tasks),
            "n_demos": self.rung.n_demos,
            "task_marginals": self.rung.task_marginals(),
            "defect_marginals": self.rung.defect_marginals(),
            "connected": self.graph.is_connected,
            "n_components": self.graph.n_components,
            "xi_g": self.xi_g(),
            "upper_anchor": self.rung.is_upper_anchor,
        }
        if self.uniform_tau_corr:
            r_eff = self.r_eff_normalised()
            row["r_eff_normalised"] = r_eff
            row["lambda2_normalised"] = self.lambda2_normalised()
            row["ln_r_eff"] = None if math.isinf(r_eff) else self.ln_r_eff()
        else:
            row["r_eff_absolute"] = self.r_eff_absolute()
            row["lambda2_absolute"] = self.graph.lambda2()
        return row


E8_RUNGS: tuple[DesignRung, ...] = (
    DesignRung("A", 6, 1.0 / 3.0, -1.099, True),
    DesignRung("C1", 4, 0.5, -0.693, True),
    DesignRung("C2", 3, 2.0 / 3.0, -0.405, True),
    DesignRung("C3", 2, 1.0, 0.0, True),
    DesignRung("C4", 1, 2.0, 0.693, True),
    DesignRung("B", 0, math.inf, None, False, is_upper_anchor=True),
)
"""10.4.8's six rungs, in the order the table prints them."""


class DesignGraphLadder:
    """The six rungs as one object, with 10.4.8's structural rules enforced.

    Three of those rules are the reason this is a class and not a list.

    The slope is estimated on **five** rungs, not six: 10.4.8 makes rung B an
    upper anchor reported as approximately true by construction, excluded from
    the primary criterion and from the Holm-Bonferroni family. `slope_rungs`
    is what feeds `connectivity_slope`, and `upper_anchor` is separate.

    Rung B's task and defect marginals match rung A's word for word, so that the
    only difference between them is the support structure of the ratio table and
    task difficulty is matched by construction. That is checkable arithmetic,
    and `require_anchor_marginals_match` checks it.

    Every rung's demonstration budget is the same 1 200 per morphology family.
    A rung that spent more would be a data-volume arm wearing a connectivity
    label, which is confounder X2 reintroduced through the back door.
    """

    def __init__(
        self,
        rungs: Sequence[DesignRung] = E8_RUNGS,
        *,
        tau_corr: float | Mapping[tuple[str, str], float] = NOMINAL_TAU_CORR,
        horizon: int = HORIZON,
    ) -> None:
        if len(rungs) < 2:
            raise ValueError(f"a ladder needs at least two rungs, got {len(rungs)}")
        labels = [r.label for r in rungs]
        if len(set(labels)) != len(labels):
            raise ValueError(f"rung labels must be unique, got {labels}")
        self._rungs = tuple(rungs)
        self._graphs = {r.label: r.graph(tau_corr=tau_corr, horizon=horizon) for r in rungs}
        budgets = {r.label: r.n_demos for r in rungs}
        off_budget = {k: v for k, v in budgets.items() if v != DEMOS_PER_FAMILY}
        if off_budget:
            raise ValueError(
                f"rungs {off_budget} do not spend the {DEMOS_PER_FAMILY} "
                "demonstrations per morphology family that 10.4.8 fixes for every "
                "rung. E8's independent variable is the support structure of the "
                "ratio table; a rung with a different budget varies data volume too"
            )

    @property
    def rungs(self) -> tuple[DesignRung, ...]:
        return self._rungs

    def graph(self, label: str) -> RungGraph:
        try:
            return self._graphs[label]
        except KeyError:
            raise KeyError(
                f"no rung {label!r}; this ladder has {sorted(self._graphs)}"
            ) from None

    @property
    def slope_rungs(self) -> tuple[DesignRung, ...]:
        """The connected rungs the primary criterion is estimated on (10.4.8)."""
        return tuple(r for r in self._rungs if not r.is_upper_anchor)

    @property
    def upper_anchor(self) -> DesignRung:
        anchors = [r for r in self._rungs if r.is_upper_anchor]
        if len(anchors) != 1:
            raise ValueError(
                f"10.4.8 has exactly one upper anchor, rung B; this ladder marks "
                f"{[a.label for a in anchors]}"
            )
        return anchors[0]

    def ln_r_eff_axis(self) -> dict[str, float]:
        """The regressor per slope rung, which is what E8's slope is fitted on."""
        return {r.label: self._graphs[r.label].ln_r_eff() for r in self.slope_rungs}

    def sum_squares_x(self) -> float:
        """sum_i (x_i - x_bar)^2 over the slope rungs; 10.7.5 registers 1.88."""
        values = list(self.ln_r_eff_axis().values())
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values)

    def spread_x(self) -> float:
        values = list(self.ln_r_eff_axis().values())
        return max(values) - min(values)

    def require_anchor_marginals_match(self, reference_label: str = "A") -> None:
        """10.4.8's construction: the anchor matches the reference rung's marginals."""
        anchor = self.upper_anchor
        reference = next(r for r in self._rungs if r.label == reference_label)
        for what, left, right in (
            ("task", anchor.task_marginals(), reference.task_marginals()),
            ("defect", anchor.defect_marginals(), reference.defect_marginals()),
        ):
            if left != right:
                raise MarginalMismatchError(
                    f"rung {anchor.label}'s {what} marginals {left} differ from rung "
                    f"{reference.label}'s {right}. 10.4.8 matches the two word for word "
                    "so that the only difference between them is the support structure "
                    "of the ratio table; unmatched marginals put a data-volume "
                    "difference into the anchor's gap"
                )

    def check_registered_values(self, *, tolerance: float = 1e-3) -> None:
        """Every printed R_eff and ln R_eff must follow from the allocation.

        The tolerance is half of the last printed digit of 10.4.8's ln R_eff
        column, which is given to three decimals.
        """
        for rung in self._rungs:
            graph = self._graphs[rung.label]
            computed = graph.r_eff_normalised()
            if math.isinf(rung.registered_r_eff) != math.isinf(computed):
                raise ValueError(
                    f"rung {rung.label}: 10.4.8 registers R_eff "
                    f"{rung.registered_r_eff} but the allocation gives {computed}"
                )
            if math.isinf(computed):
                continue
            if abs(computed - rung.registered_r_eff) > tolerance:
                raise ValueError(
                    f"rung {rung.label}: 10.4.8 registers R_eff "
                    f"{rung.registered_r_eff:.4g}, the allocation gives {computed:.4g}. "
                    "The table is a frozen preregistration artefact (10.7.8 item 4); "
                    "the allocation is what has to be explained"
                )
            if rung.registered_ln_r_eff is None:
                raise ValueError(
                    f"rung {rung.label} is connected but registers no ln R_eff"
                )
            if abs(graph.ln_r_eff() - rung.registered_ln_r_eff) > tolerance:
                raise ValueError(
                    f"rung {rung.label}: 10.4.8 registers ln R_eff "
                    f"{rung.registered_ln_r_eff}, the allocation gives "
                    f"{graph.ln_r_eff():.4g}"
                )

    def as_rows(self) -> list[dict[str, Any]]:
        return [self._graphs[r.label].as_row() for r in self._rungs]


def _self_check() -> None:
    """The six rungs reproduce 10.4.8, and the normalisation really cancels.

    The second half is the load-bearing one. `r_eff_normalised` claims that
    multiplying by the reference edge weight removes tau_corr and the horizon, so
    that a column printed as "relative" is genuinely independent of two numbers
    W2 has not measured yet. Building the ladder twice at different values and
    comparing is the check; without it the claim is a comment.
    """
    ladder = DesignGraphLadder()
    ladder.check_registered_values()
    ladder.require_anchor_marginals_match()

    other = DesignGraphLadder(tau_corr=7.5, horizon=120)
    for rung in E8_RUNGS:
        a = ladder.graph(rung.label)
        b = other.graph(rung.label)
        for name, left, right in (
            ("R_eff", a.r_eff_normalised(), b.r_eff_normalised()),
            ("lambda_2", a.lambda2_normalised(), b.lambda2_normalised()),
            ("Xi_G", a.xi_g(), b.xi_g()),
        ):
            if math.isinf(left) and math.isinf(right):
                continue
            if not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12):
                raise AssertionError(
                    f"rung {rung.label}: normalised {name} moved from {left} to "
                    f"{right} when tau_corr and the horizon changed. 10.4.8's "
                    "normalised column is supposed to be independent of both"
                )

    registered_sum_squares = 1.88
    if abs(ladder.sum_squares_x() - registered_sum_squares) > 5e-3:
        raise AssertionError(
            f"10.7.5 registers sum (x - x_bar)^2 = {registered_sum_squares} for "
            f"connectivity_slope, the ladder gives {ladder.sum_squares_x():.4g}"
        )


_self_check()
