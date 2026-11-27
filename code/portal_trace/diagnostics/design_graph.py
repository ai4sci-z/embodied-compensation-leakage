"""The design graph G, its Laplacian, R_eff, lambda_2(G) and Xi_G.

Implements proposal 7.2 (proposition 1(a)), 7.7 (the bias map and Xi_G), 7.9
(proposition 3), 9.7.5 (the per-regime connectivity procedure) and 11.4.

This module computes **design quantities only**. It contains no estimator:
theta_hat and gamma_hat come from `two_way_fit` and nothing else (11.2). What
it provides is the geometry those estimates are read against.

Edge weights
------------
`EffectiveSampleCount` is the only admissible weight type and it can only be
built from n_traj * (T / tau_corr). Appendix A-8 and 11.4 are explicit that a
raw frame count overstates precision by roughly an order of magnitude, because
frames inside one trajectory are strongly autocorrelated. Passing an int is a
`TypeError` here, not a warning, and the two calibers of 7.9 -- homoscedastic
n_eff and heteroscedastic n_eff / sigma^2 -- may not be mixed inside one graph.

Reparameterisation
------------------
Proposition 3's proof writes the model row of cell (tau, delta) as the signed
incidence vector e_tau - f_delta of beta = [theta; -gamma], which makes
X^T W X the weighted Laplacian exactly. This module uses that parameterisation
throughout, so `two_way_fit` can read gamma off the second block with a sign
flip and nothing else has to know about it.

Reporting rule
--------------
9.7.5 and 11.3: **if a regime is disconnected its conclusion is not reported.**
Not reported with a warning -- not reported. `RegimeRestriction.require_
reportable()` raises, because in a disconnected regime the numbers a solver
returns are a point the regulariser picked out of the null space.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from portal_trace.diagnostics.units import EffectiveSampleCount

__all__ = [
    "N0_PRIMARY_FRAMES",
    "N0_SENSITIVITY_FRAMES",
    "M_MIN_OCCUPANCY",
    "N_MIN_RAW_FRAMES",
    "DesignGraph",
    "RegimeRestriction",
    "CellObservation",
    "DisconnectedRegime",
    "frame_count_admission",
    "occupancy_admission",
]

# 7.8 and 11.3: the per-regime observed cell set uses N_0 = 200 frames as the
# primary caliber, with a three-level sensitivity sweep.
N0_PRIMARY_FRAMES = 200
N0_SENSITIVITY_FRAMES = (50, 200, 800)

# 9.7.5 states a second, stricter admission rule for the same set: occupancy
# mass at least 0.02 *and* at least 2000 raw frames. The two calibers disagree
# (see the module note in the report); both are implemented and neither is
# silently preferred.
M_MIN_OCCUPANCY = 0.02
N_MIN_RAW_FRAMES = 2000


class DisconnectedRegime(ValueError):
    """A conclusion was requested from a regime whose design graph is disconnected."""


@dataclass(frozen=True)
class CellObservation:
    """One (task, defect) cell inside one regime, as 9.7.4's long table sees it."""

    task: str
    defect: str
    n_frames: int
    occupancy: float
    tau_corr: float
    n_traj: int
    horizon: int
    sigma2: float | None = None

    def __post_init__(self) -> None:
        if self.n_frames < 0:
            raise ValueError(f"n_frames cannot be negative, got {self.n_frames}")
        if not 0.0 <= self.occupancy <= 1.0:
            raise ValueError(f"occupancy mass must lie in [0, 1], got {self.occupancy}")

    def weight(self, *, heteroscedastic: bool) -> EffectiveSampleCount:
        if not heteroscedastic:
            return EffectiveSampleCount.from_trajectories(
                n_traj=self.n_traj, horizon=self.horizon, tau_corr=self.tau_corr
            )
        if self.sigma2 is None:
            raise ValueError(
                f"cell ({self.task}, {self.defect}) has no sigma^2, so the "
                "heteroscedastic conductance n/sigma^2 of 7.9 cannot be built"
            )
        return EffectiveSampleCount.from_trajectories_heteroscedastic(
            n_traj=self.n_traj,
            horizon=self.horizon,
            tau_corr=self.tau_corr,
            sigma2=self.sigma2,
        )


def frame_count_admission(cells: Sequence[CellObservation], n0: int) -> list[CellObservation]:
    """E_r = {(tau, delta) : N_{tau delta}(r) >= N_0} (7.8, 11.3)."""
    if n0 <= 0:
        raise ValueError(f"N_0 counts frames and must be positive, got {n0}")
    return [c for c in cells if c.n_frames >= n0]


def occupancy_admission(
    cells: Sequence[CellObservation],
    *,
    m_min: float = M_MIN_OCCUPANCY,
    n_min: int = N_MIN_RAW_FRAMES,
) -> list[CellObservation]:
    """The 9.7.5 rule: occupancy mass >= m_min *and* raw frames >= n_min."""
    return [c for c in cells if c.occupancy >= m_min and c.n_frames >= n_min]


class DesignGraph:
    """G = (T u D, E) with `EffectiveSampleCount` edge weights."""

    def __init__(self, edges: Mapping[tuple[str, str], EffectiveSampleCount]) -> None:
        if not edges:
            raise ValueError("a design graph with no observed cells has no content")
        calibers = set()
        for key, weight in edges.items():
            if not isinstance(weight, EffectiveSampleCount):
                raise TypeError(
                    f"edge {key} has weight {weight!r} of type {type(weight).__name__}. "
                    "Design-graph edge weights are EffectiveSampleCount and nothing else "
                    "(11.4, appendix A-8): a raw frame count overstates precision by about "
                    "an order of magnitude. Build it with "
                    "EffectiveSampleCount.from_trajectories(n_traj=..., horizon=..., "
                    "tau_corr=...)"
                )
            if float(weight) <= 0.0:
                raise ValueError(f"edge {key} has non-positive conductance {float(weight)}")
            calibers.add(weight.is_heteroscedastic)
        if len(calibers) > 1:
            raise ValueError(
                "this graph mixes homoscedastic and heteroscedastic edge weights. 7.9 "
                "reports the two calibers side by side, as two graphs, because R_eff "
                "means a different thing in each"
            )
        self._edges = dict(edges)
        self._heteroscedastic = calibers.pop()
        self._tasks = tuple(sorted({t for t, _ in self._edges}))
        self._defects = tuple(sorted({d for _, d in self._edges}))
        self._task_index = {t: i for i, t in enumerate(self._tasks)}
        self._defect_index = {
            d: len(self._tasks) + i for i, d in enumerate(self._defects)
        }
        self._laplacian: np.ndarray | None = None
        self._pinv: np.ndarray | None = None
        self._components: tuple[frozenset[str], ...] | None = None

    # -- structure --------------------------------------------------------

    @property
    def tasks(self) -> tuple[str, ...]:
        return self._tasks

    @property
    def defects(self) -> tuple[str, ...]:
        return self._defects

    @property
    def edges(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._edges))

    @property
    def is_heteroscedastic(self) -> bool:
        return self._heteroscedastic

    @property
    def n_nodes(self) -> int:
        return len(self._tasks) + len(self._defects)

    @property
    def is_balanced(self) -> bool:
        """E = T x D, the condition proposition 1(e) needs beyond connectivity."""
        return len(self._edges) == len(self._tasks) * len(self._defects)

    def node_index(self, name: str, *, side: str) -> int:
        if side == "task":
            if name not in self._task_index:
                raise KeyError(f"task {name!r} is not a node of this graph")
            return self._task_index[name]
        if side == "defect":
            if name not in self._defect_index:
                raise KeyError(f"defect {name!r} is not a node of this graph")
            return self._defect_index[name]
        raise ValueError(f"side must be 'task' or 'defect', got {side!r}")

    def raw_frames(self) -> int:
        """Total raw frames behind the graph, for the data card only (11.4)."""
        return sum(w.raw_frames for w in self._edges.values())

    def effective_samples(self) -> float:
        return float(sum(float(w) for w in self._edges.values()))

    # -- linear algebra ---------------------------------------------------

    def incidence(self) -> np.ndarray:
        """M with rows e_tau - f_delta, the reparameterisation of 7.9 step 2."""
        m = np.zeros((len(self._edges), self.n_nodes), dtype=np.float64)
        for row, (task, defect) in enumerate(self.edges):
            m[row, self._task_index[task]] = 1.0
            m[row, self._defect_index[defect]] = -1.0
        return m

    def weight_matrix(self) -> np.ndarray:
        return np.diag([float(self._edges[e]) for e in self.edges])

    def laplacian(self) -> np.ndarray:
        """L_G = M^T W M = sum_e w_e (e_u - e_v)(e_u - e_v)^T."""
        if self._laplacian is None:
            m = self.incidence()
            w = np.array([float(self._edges[e]) for e in self.edges], dtype=np.float64)
            self._laplacian = m.T @ (w[:, None] * m)
        return self._laplacian.copy()

    def laplacian_pinv(self) -> np.ndarray:
        if self._pinv is None:
            self._pinv = np.linalg.pinv(self.laplacian(), hermitian=True)
        return self._pinv.copy()

    def components(self) -> tuple[frozenset[str], ...]:
        """Connected components, as sets of ``task:``/``defect:`` labels."""
        if self._components is None:
            parent: dict[str, str] = {}

            def find(x: str) -> str:
                parent.setdefault(x, x)
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a: str, b: str) -> None:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

            for task, defect in self.edges:
                union(f"task:{task}", f"defect:{defect}")
            groups: dict[str, set[str]] = {}
            for node in list(parent):
                groups.setdefault(find(node), set()).add(node)
            self._components = tuple(
                frozenset(g) for g in sorted(groups.values(), key=lambda s: sorted(s))
            )
        return self._components

    @property
    def n_components(self) -> int:
        return len(self.components())

    @property
    def is_connected(self) -> bool:
        return self.n_components == 1

    def kernel_dimension(self, action_dim: int) -> int:
        """dim ker M = d_a * #components (proposition 1(a)).

        The d_a factor is not optional: theta and gamma are vector parameters
        and the null direction of definition 1 is a vector translation.
        """
        if action_dim < 1:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        return action_dim * self.n_components

    def same_component(self, a: str, b: str, *, side: str = "task") -> bool:
        label_a = f"{'task' if side == 'task' else 'defect'}:{a}"
        label_b = f"{'task' if side == 'task' else 'defect'}:{b}"
        for component in self.components():
            if label_a in component:
                return label_b in component
        raise KeyError(f"{label_a} is not a node of this graph")

    def r_eff(self, a: str, b: str, *, side: str = "task") -> float:
        """R_eff(a, b) = (e_a - e_b)^T L^+ (e_a - e_b) (proposition 3).

        Infinite across components: corollary (i) reads disconnection as the
        limit of vanishing edge conductance, and proposition 1(a)'s binary
        verdict is that degenerate case.
        """
        if a == b:
            return 0.0
        if not self.same_component(a, b, side=side):
            return math.inf
        u = np.zeros(self.n_nodes, dtype=np.float64)
        u[self.node_index(a, side=side)] = 1.0
        u[self.node_index(b, side=side)] = -1.0
        return float(u @ self.laplacian_pinv() @ u)

    def lambda2(self) -> float:
        """Algebraic connectivity: the second smallest Laplacian eigenvalue.

        Zero when the graph is disconnected, which is how 9.7.5 reports it.
        """
        eigenvalues = np.linalg.eigvalsh(self.laplacian())
        return float(max(eigenvalues[1], 0.0))

    def r_eff_bounds(self) -> tuple[float, float]:
        """2/lambda_n <= R_eff <= 2/lambda_2 (proposition 3, corollary ii)."""
        eigenvalues = np.linalg.eigvalsh(self.laplacian())
        lambda_2 = float(max(eigenvalues[1], 0.0))
        lambda_n = float(eigenvalues[-1])
        if lambda_2 <= 0.0:
            return 2.0 / lambda_n, math.inf
        return 2.0 / lambda_n, 2.0 / lambda_2

    def bias_map(self) -> np.ndarray:
        """L_G^+ M^T W: the bias map whose task rows carry the 7.7 multiplier."""
        return self.laplacian_pinv() @ self.incidence().T @ self.weight_matrix()

    def xi_g(self) -> float:
        """Xi_G = max_tau ||h_tau||_1 / (2 (1 - 1/n_T)) (7.7, 14.5 E-16).

        A pure design quantity, computed numerically per design level. No
        "Xi_G >= 1" claim is made or checked: the L_1 norm of a pseudo-inverse
        row is not monotone in the edge weights.
        """
        n_t = len(self._tasks)
        if n_t < 2:
            raise ValueError(
                f"Xi_G normalises by the balanced-design value 2(1 - 1/n_T), which "
                f"vanishes at n_T = 1; this graph has {n_t} task node(s)"
            )
        h = self.bias_map()[: n_t, :]
        max_row_l1 = float(np.max(np.sum(np.abs(h), axis=1)))
        return max_row_l1 / (2.0 * (1.0 - 1.0 / n_t))

    def r_eff_gradient(self, a: str, b: str, *, side: str = "task") -> dict[
        tuple[str, str], float
    ]:
        """dR_eff(a,b)/dw_e = -(V_u - V_v)^2 (proposition 3, corollary ii).

        V is the node potential of the unit-current field injected at a and
        extracted at b. The reading is operational: adding data pays most on
        the edge with the largest potential difference.
        """
        if not self.same_component(a, b, side=side):
            raise DisconnectedRegime(
                f"{a} and {b} are in different components; R_eff is infinite and its "
                "gradient is not defined"
            )
        u = np.zeros(self.n_nodes, dtype=np.float64)
        u[self.node_index(a, side=side)] = 1.0
        u[self.node_index(b, side=side)] = -1.0
        potentials = self.laplacian_pinv() @ u
        out: dict[tuple[str, str], float] = {}
        for task, defect in self.edges:
            drop = potentials[self._task_index[task]] - potentials[self._defect_index[defect]]
            out[(task, defect)] = -float(drop * drop)
        return out

    def predicted_contrast_variance(
        self, a: str, b: str, sigma2: float | None = None, *, side: str = "task"
    ) -> float:
        """Var(theta_a - theta_b) predicted by proposition 3.

        Homoscedastic caliber: sigma^2 R_eff, and sigma^2 must be supplied.
        Heteroscedastic caliber: R_eff, with the noise already inside the
        conductance, and supplying sigma^2 is an error rather than a scaling.
        """
        r = self.r_eff(a, b, side=side)
        if self._heteroscedastic:
            if sigma2 is not None:
                raise ValueError(
                    "the heteroscedastic caliber absorbs the noise scale into the "
                    "conductance n/sigma^2; multiplying by sigma^2 again double counts it"
                )
            return r
        if sigma2 is None:
            raise ValueError(
                "the homoscedastic caliber predicts sigma^2 R_eff; sigma^2 is estimated "
                "from the within-cell residual and must be passed explicitly"
            )
        if sigma2 <= 0.0:
            raise ValueError(f"sigma^2 must be positive, got {sigma2}")
        return sigma2 * r

    def summary(self, key_pairs: Sequence[tuple[str, str]] = ()) -> dict[str, Any]:
        """The three things 9.7.5 requires to be reported together."""
        return {
            "n_tasks": len(self._tasks),
            "n_defects": len(self._defects),
            "n_edges": len(self._edges),
            "balanced": self.is_balanced,
            "caliber": "heteroscedastic" if self._heteroscedastic else "homoscedastic",
            "n_components": self.n_components,
            "connected": self.is_connected,
            "lambda2": self.lambda2(),
            "xi_g": self.xi_g() if len(self._tasks) >= 2 else None,
            "effective_samples": self.effective_samples(),
            "raw_frames": self.raw_frames(),
            "r_eff": {f"{a}|{b}": self.r_eff(a, b) for a, b in key_pairs},
        }


@dataclass(frozen=True)
class RegimeRestriction:
    """G|_r for one regime under one admission caliber (9.7.5).

    Carries the excluded cells as well as the graph, because 9.7.5 requires the
    list of excluded cells with their (m, n) to be reported next to the
    connectivity verdict.
    """

    regime_label: str
    caliber: str
    graph: DesignGraph | None
    admitted: tuple[CellObservation, ...]
    excluded: tuple[CellObservation, ...]

    @property
    def status(self) -> str:
        if self.graph is None:
            return "empty"
        return "connected" if self.graph.is_connected else "disconnected"

    @property
    def reportable(self) -> bool:
        """9.7.5: a disconnected regime's conclusions are not reported."""
        return self.graph is not None and self.graph.is_connected

    def require_reportable(self) -> DesignGraph:
        if self.graph is None:
            raise DisconnectedRegime(
                f"regime {self.regime_label} has no admitted cells under the "
                f"{self.caliber} caliber; there is nothing to report"
            )
        if not self.graph.is_connected:
            raise DisconnectedRegime(
                f"regime {self.regime_label} has {self.graph.n_components} components "
                f"under the {self.caliber} caliber. 9.7.5 preregisters that a "
                "disconnected regime's conclusions are not reported: the compensation "
                "contrasts are unidentifiable at the information level (proposition "
                "1(a)), so any number a solver returns is a point the regulariser chose "
                "out of the null space"
            )
        return self.graph

    def as_row(self, key_pairs: Sequence[tuple[str, str]] = ()) -> dict[str, Any]:
        """The verdict row, with R_eff on the key pairs 9.7.5 asks to be reported.

        A key pair whose task is not a node of the restriction reads infinite,
        the same as a pair split across components: the admission rule kept no
        cell for that task in this regime, so there is no path and the contrast
        is unidentifiable for the same reason (proposition 1(a)).
        """
        row: dict[str, Any] = {
            "regime": self.regime_label,
            "caliber": self.caliber,
            "status": self.status,
            "reportable": self.reportable,
            "n_admitted": len(self.admitted),
            "n_excluded": len(self.excluded),
            "excluded_cells": [
                {
                    "task": c.task,
                    "defect": c.defect,
                    "m": c.occupancy,
                    "n_frames": c.n_frames,
                }
                for c in self.excluded
            ],
        }
        if self.graph is not None:
            nodes = set(self.graph.tasks)
            row.update(
                {
                    "n_components": self.graph.n_components,
                    "lambda2": self.graph.lambda2(),
                    "xi_g": self.graph.xi_g() if len(self.graph.tasks) >= 2 else None,
                    "r_eff": {
                        f"{a}|{b}": (
                            self.graph.r_eff(a, b)
                            if {a, b} <= nodes
                            else math.inf
                        )
                        for a, b in key_pairs
                    },
                }
            )
        elif key_pairs:
            row["r_eff"] = {f"{a}|{b}": math.inf for a, b in key_pairs}
        return row

    @classmethod
    def build(
        cls,
        regime_label: str,
        cells: Sequence[CellObservation],
        *,
        admission: str = "n0",
        n0: int = N0_PRIMARY_FRAMES,
        heteroscedastic: bool = False,
    ) -> RegimeRestriction:
        """Restrict to one regime and decide connectivity.

        `admission` selects between the two calibers the proposal states for
        the same set: ``"n0"`` is 7.8/11.3's N_{tau delta}(r) >= N_0, and
        ``"occupancy"`` is 9.7.5's m >= 0.02 and n >= 2000 raw frames.
        """
        if admission == "n0":
            admitted = frame_count_admission(cells, n0)
            caliber = f"N_0={n0} frames"
        elif admission == "occupancy":
            admitted = occupancy_admission(cells)
            caliber = f"m>={M_MIN_OCCUPANCY} and n>={N_MIN_RAW_FRAMES} raw frames"
        else:
            raise ValueError(f"admission must be 'n0' or 'occupancy', got {admission!r}")
        caliber += "; heteroscedastic" if heteroscedastic else "; homoscedastic"
        admitted_set = {(c.task, c.defect) for c in admitted}
        excluded = tuple(c for c in cells if (c.task, c.defect) not in admitted_set)
        if not admitted:
            return cls(regime_label, caliber, None, (), excluded)
        graph = DesignGraph(
            {
                (c.task, c.defect): c.weight(heteroscedastic=heteroscedastic)
                for c in admitted
            }
        )
        return cls(regime_label, caliber, graph, tuple(admitted), excluded)
