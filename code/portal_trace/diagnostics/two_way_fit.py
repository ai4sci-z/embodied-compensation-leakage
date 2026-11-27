"""Weighted least squares for the two-way additive model (E12).

Implements proposal 7.1, 7.7, 7.9 and 10.4.12. **This is the only module in
`diagnostics` that produces estimates** (11.2): `design_graph` computes design
quantities and `common_mode` reads gamma off the injected ground truth, and
neither is an estimator. Propositions 1(a), 1(c), 1(e), 1(e') and 3 have no
measured counterpart without what is fitted here.

The model is eta_{tau delta, r} = theta_{tau,r} + gamma_{delta,r}, fitted per
regime on the within-regime cell means with the effective sample count as the
weight. Three outputs are mandatory (10.4.12, 11.2):

(i)   Both fitting calibers. The **direct** caliber fits eta_{tau delta}. The
      **twin-paired-difference** caliber fits eta_{tau delta} - eta_{tau 0} at
      the same seed_env and the same reference trajectory, which removes the
      mu-side interaction term A_{tau delta} to first order -- the term that
      proposition 1(e) says is usually the largest, because it carries task
      scale rather than compensation scale. Each caliber reports theta_hat,
      gamma_hat and the residual interaction share.

(ii)  The **measured** Var(theta_hat_tau - theta_hat_tau') from resampling
      seed_env, against proposition 3's predicted sigma^2 R_eff, pair by pair.
      This is proposition 3's only empirical check; everything else that cites
      it (E8's slope) tests the empirical bridge instead (7.12 item 1).

(iii) Xi_G computed from the actual per-level numbers, never asserted to be
      at least 1 (7.7, 14.5 E-16).

Gauge
-----
The kernel is a d_a-dimensional translation (proposition 1(a)), so a solution
is only defined up to it. 7.5 scope note (1) fixes the sum-to-zero convention
sum_k gamma_hat = 0, and that is the convention applied here, per connected
component. It is an implementation choice made visible, not a theorem: any
statement about the level of gamma is a statement about this gauge.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from portal_trace.diagnostics.design_graph import DesignGraph, DisconnectedRegime
from portal_trace.diagnostics.units import EffectiveSampleCount

__all__ = [
    "PROP3_RATIO_WINDOW",
    "CellMean",
    "SeedEnvCell",
    "SeedEnvPanel",
    "TwoWayFitResult",
    "ContrastVarianceCheck",
    "TwoWayReport",
    "fit_two_way",
    "fit_both_calibers",
    "check_proposition_three",
]

# 10.4.12: the preregistered acceptance window for the measured/predicted
# variance ratio. Specification value, not a calibrated threshold.
PROP3_RATIO_WINDOW = (0.7, 1.4)


@dataclass(frozen=True)
class CellMean:
    """eta_{tau delta, r}: one cell's within-regime mean action, with its weight."""

    task: str
    defect: str
    mean: np.ndarray
    weight: EffectiveSampleCount
    n_frames: int
    sigma2: float | None = None

    def __post_init__(self) -> None:
        arr = np.asarray(self.mean, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"a cell mean is a d_a vector, got shape {arr.shape}")
        if not isinstance(self.weight, EffectiveSampleCount):
            raise TypeError(
                f"cell ({self.task}, {self.defect}) has weight of type "
                f"{type(self.weight).__name__}; the WLS weight is the effective sample "
                "count n_traj * (T / tau_corr) (11.4, appendix A-8)"
            )
        object.__setattr__(self, "mean", arr)


@dataclass(frozen=True)
class SeedEnvCell:
    """One cell's contribution from one environment seed.

    The seed_env layer is the resampling unit for output (ii): 11.3 aligns
    seed_env across arms, so resampling it is what isolates the variance
    component proposition 3 predicts.
    """

    seed_env: int
    task: str
    defect: str
    mean: np.ndarray
    n_traj: int
    horizon: int
    tau_corr: float
    n_frames: int
    sigma2: float | None = None

    def __post_init__(self) -> None:
        if np.ndim(self.mean) != 1:
            raise ValueError(f"a cell mean is a d_a vector, got shape {np.shape(self.mean)}")
        if self.n_frames <= 0 or self.n_traj <= 0:
            raise ValueError(
                f"cell ({self.task}, {self.defect}) at seed_env {self.seed_env} carries "
                f"{self.n_frames} frames over {self.n_traj} trajectories; an empty cell "
                "is an absent edge of the design graph, not an edge of weight zero"
            )
        if self.horizon <= 0 or not (self.tau_corr > 0.0):
            raise ValueError(
                f"cell ({self.task}, {self.defect}) has horizon {self.horizon} and "
                f"tau_corr {self.tau_corr}; the effective sample count n_traj * "
                "(T / tau_corr) needs both positive (11.4, appendix A-8)"
            )


class SeedEnvPanel:
    """Per-seed cell means, poolable and resamplable.

    The panel is held as a dense (seed, cell) grid rather than as a list, and
    pooling is a matrix product against the multiplicity of each seed in the
    draw. Output (ii) of 10.4.12 refits the model on a few thousand replicates
    per regime and per caliber, and walking the cells in Python once per
    replicate is roughly five times the cost of everything else in the loop put
    together (`reports/verification/bench_prop3.py`).

    The horizon and action-width agreement that pooling needs is therefore
    checked once here, over the whole panel, instead of once per replicate over
    whichever seeds were drawn. That is strictly earlier: a panel that mixes
    horizons within a cell is now rejected when it is built rather than when a
    replicate happens to draw both seeds.
    """

    def __init__(self, cells: Sequence[SeedEnvCell]) -> None:
        if not cells:
            raise ValueError("an empty panel has no cell means")
        widths = {np.shape(c.mean) for c in cells}
        if len(widths) != 1:
            raise ValueError(f"cell means disagree on action width: {widths}")
        self._width = int(next(iter(widths))[0])

        self._seeds = tuple(sorted({c.seed_env for c in cells}))
        self._keys = tuple(sorted({(c.task, c.defect) for c in cells}))
        self._seed_at = {s: i for i, s in enumerate(self._seeds)}
        key_at = {k: j for j, k in enumerate(self._keys)}
        n_s, n_c = len(self._seeds), len(self._keys)

        self._mean = np.zeros((n_s, n_c, self._width), dtype=np.float64)
        self._frames = np.zeros((n_s, n_c), dtype=np.float64)
        self._tau_corr = np.zeros((n_s, n_c), dtype=np.float64)
        self._n_traj = np.zeros((n_s, n_c), dtype=np.float64)
        self._sigma2 = np.zeros((n_s, n_c), dtype=np.float64)
        self._has_sigma2 = np.zeros((n_s, n_c), dtype=bool)
        self._present = np.zeros((n_s, n_c), dtype=bool)
        horizon = np.zeros((n_s, n_c), dtype=np.int64)

        for cell in cells:
            i, j = self._seed_at[cell.seed_env], key_at[(cell.task, cell.defect)]
            if self._present[i, j]:
                raise ValueError(
                    f"cell ({cell.task}, {cell.defect}) appears twice at seed_env "
                    f"{cell.seed_env}; pool it before building the panel, or the "
                    "duplicate silently doubles that seed's weight in every replicate"
                )
            self._present[i, j] = True
            self._mean[i, j] = cell.mean
            self._frames[i, j] = cell.n_frames
            self._tau_corr[i, j] = cell.tau_corr
            self._n_traj[i, j] = cell.n_traj
            horizon[i, j] = cell.horizon
            if cell.sigma2 is not None:
                self._sigma2[i, j] = cell.sigma2
                self._has_sigma2[i, j] = True

        self._horizon = np.zeros(n_c, dtype=np.int64)
        for j, (task, defect) in enumerate(self._keys):
            seen = np.unique(horizon[self._present[:, j], j])
            if seen.size != 1:
                raise ValueError(
                    f"cell ({task}, {defect}) mixes horizons {sorted(int(h) for h in seen)}; "
                    "9.6.1 collects at a fixed T = 400 and the effective sample count "
                    "assumes it"
                )
            self._horizon[j] = seen[0]

        # Pooling divides by the frame total, so the mean and tau_corr are
        # carried pre-weighted: the replicate loop then needs one matrix product
        # each rather than a multiply and a reduction.
        self._frame_mean = self._frames[:, :, None] * self._mean
        self._frame_tau = self._frames * self._tau_corr
        self._frame_sigma2 = self._frames * self._sigma2

    @property
    def seeds(self) -> tuple[int, ...]:
        return self._seeds

    @property
    def action_width(self) -> int:
        return self._width

    def pooled(self, seeds: Sequence[int] | None = None) -> list[CellMean]:
        """Frame-weighted pooling of the selected seeds into cell means.

        `seeds` may repeat, which is what makes a bootstrap replicate; a seed
        drawn twice contributes twice, exactly as resampling intends.
        """
        if seeds is None:
            multiplicity = np.ones(len(self._seeds), dtype=np.float64)
        else:
            chosen = tuple(seeds)
            missing = [s for s in set(chosen) if s not in self._seed_at]
            if missing:
                raise KeyError(f"unknown seed_env values: {sorted(missing)}")
            multiplicity = np.bincount(
                [self._seed_at[s] for s in chosen], minlength=len(self._seeds)
            ).astype(np.float64)
        return self._pool(multiplicity)

    def _pool(self, multiplicity: np.ndarray) -> list[CellMean]:
        """Pool the panel with each seed counted `multiplicity` times."""
        m = multiplicity
        total = m @ self._frames
        active = np.flatnonzero(total > 0.0)
        if active.size == 0:
            raise ValueError("the selected seeds carry no frames")

        mean = np.einsum("s,scd->cd", m, self._frame_mean)[active] / total[active, None]
        tau_corr = (m @ self._frame_tau)[active] / total[active]
        n_traj = (m @ self._n_traj)[active]

        # A cell whose sigma^2 is present at some of its drawn contributions and
        # absent at others cannot be pooled: the heteroscedastic conductance of
        # 7.9 is n_eff / sigma^2, and averaging over the ones that have it would
        # weight the cell as if the rest were noiseless.
        occurrences = (m @ self._present)[active]
        with_sigma2 = (m @ (self._has_sigma2 & self._present))[active]
        partial = np.flatnonzero((with_sigma2 > 0.0) & (with_sigma2 != occurrences))
        if partial.size:
            j = active[partial[0]]
            raise ValueError(
                f"cell {self._keys[j]} has sigma^2 on {with_sigma2[partial[0]]:.0f} of "
                f"{occurrences[partial[0]]:.0f} seed_env contributions. The "
                "heteroscedastic conductance of 7.9 is n_eff / sigma^2 and pooling it "
                "from a subset would weight the cell as if the rest were noiseless"
            )
        sigma2 = (m @ self._frame_sigma2)[active] / total[active]

        out: list[CellMean] = []
        for row, j in enumerate(active):
            task, defect = self._keys[j]
            out.append(
                CellMean(
                    task=task,
                    defect=defect,
                    mean=mean[row],
                    weight=EffectiveSampleCount.from_trajectories(
                        n_traj=int(n_traj[row]),
                        horizon=int(self._horizon[j]),
                        tau_corr=float(tau_corr[row]),
                    ),
                    n_frames=int(total[j]),
                    sigma2=float(sigma2[row]) if with_sigma2[row] > 0.0 else None,
                )
            )
        return out

    def resample(self, rng: np.random.Generator) -> list[CellMean]:
        drawn = rng.choice(len(self._seeds), size=len(self._seeds), replace=True)
        return self._pool(np.bincount(drawn, minlength=len(self._seeds)).astype(np.float64))


@dataclass(frozen=True)
class TwoWayFitResult:
    """theta_hat and gamma_hat for one caliber, in the sum-to-zero gauge."""

    caliber: str
    theta: dict[str, np.ndarray]
    gamma: dict[str, np.ndarray]
    graph: DesignGraph
    weighted_residual: float
    weighted_total: float
    sigma2_within: float | None

    @property
    def residual_interaction_share(self) -> float:
        """Weighted RSS over weighted TSS: what additivity fails to reproduce.

        This is the empirical size of A_{tau delta} + B_{tau delta} of 7.7,
        expressed as a share. The paired caliber should come out smaller,
        because the differencing removes A to first order.
        """
        if self.weighted_total <= 0.0:
            raise ValueError(
                "the cell means carry no weighted variation around the grand mean; a "
                "residual share has no denominator"
            )
        return self.weighted_residual / self.weighted_total

    def contrast(self, a: str, b: str) -> np.ndarray:
        for name in (a, b):
            if name not in self.theta:
                raise KeyError(f"task {name!r} is not in the fit")
        if not self.graph.same_component(a, b):
            raise DisconnectedRegime(
                f"tasks {a} and {b} are in different components of the design graph; "
                "their contrast is not estimable (proposition 1(a))"
            )
        return self.theta[a] - self.theta[b]

    def defect_contrast(self, a: str, b: str) -> np.ndarray:
        if not self.graph.same_component(a, b, side="defect"):
            raise DisconnectedRegime(
                f"defects {a} and {b} are in different components; their compensation "
                "contrast is not identifiable (proposition 1(c))"
            )
        return self.gamma[a] - self.gamma[b]

    def as_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for task, value in sorted(self.theta.items()):
            rows.append(
                {"caliber": self.caliber, "side": "theta", "node": task, "value": value.tolist()}
            )
        for defect, value in sorted(self.gamma.items()):
            rows.append(
                {"caliber": self.caliber, "side": "gamma", "node": defect, "value": value.tolist()}
            )
        rows.append(
            {
                "caliber": self.caliber,
                "side": "fit",
                "node": "residual_interaction_share",
                "value": self.residual_interaction_share,
            }
        )
        return rows


def fit_two_way(
    cells: Sequence[CellMean],
    *,
    caliber: str,
    heteroscedastic: bool = False,
) -> TwoWayFitResult:
    """WLS fit of eta = theta_tau + gamma_delta on the cell means.

    The normal-equation matrix is the weighted Laplacian exactly (7.9 step 2),
    so the fit and the design quantities come out of the same object and cannot
    drift apart.
    """
    if not cells:
        raise ValueError("no cell means supplied")
    widths = {c.mean.shape[0] for c in cells}
    if len(widths) != 1:
        raise ValueError(f"cell means disagree on action width: {widths}")
    d_a = widths.pop()

    weights: dict[tuple[str, str], EffectiveSampleCount] = {}
    for cell in cells:
        key = (cell.task, cell.defect)
        if key in weights:
            raise ValueError(f"cell {key} appears twice; pool it before fitting")
        if heteroscedastic:
            if cell.sigma2 is None:
                raise ValueError(
                    f"cell {key} has no sigma^2; the heteroscedastic conductance of 7.9 "
                    "needs a per-cell within-regime residual variance"
                )
            weights[key] = EffectiveSampleCount.from_trajectories_heteroscedastic(
                n_traj=cell.weight.n_traj,
                horizon=cell.weight.horizon,
                tau_corr=cell.weight.tau_corr,
                sigma2=cell.sigma2,
            )
        else:
            weights[key] = cell.weight
    graph = DesignGraph(weights)

    order = graph.edges
    lookup = {(c.task, c.defect): c for c in cells}
    y = np.stack([lookup[e].mean for e in order], axis=0)
    w = np.array([float(weights[e]) for e in order], dtype=np.float64)
    m = graph.incidence()
    beta = graph.laplacian_pinv() @ (m.T @ (w[:, None] * y))

    n_t = len(graph.tasks)
    theta = {task: beta[i].copy() for task, i in zip(graph.tasks, range(n_t), strict=True)}
    gamma = {
        defect: -beta[n_t + i].copy()
        for i, defect in enumerate(graph.defects)
    }

    # 7.5 scope note (1): pin the d_a-dimensional gauge with sum_k gamma = 0,
    # separately on each component, since the kernel has one copy per component.
    for component in graph.components():
        defects = [n.split(":", 1)[1] for n in component if n.startswith("defect:")]
        tasks = [n.split(":", 1)[1] for n in component if n.startswith("task:")]
        if not defects:
            continue
        shift = np.mean(np.stack([gamma[d] for d in defects], axis=0), axis=0)
        for d in defects:
            gamma[d] = gamma[d] - shift
        for t in tasks:
            theta[t] = theta[t] + shift

    fitted = np.stack([theta[t] + gamma[d] for t, d in order], axis=0)
    residual = float(np.sum(w[:, None] * np.square(y - fitted)))
    grand = (w[:, None] * y).sum(axis=0) / w.sum()
    total = float(np.sum(w[:, None] * np.square(y - grand)))

    # rank(L_G) = n_nodes - #components (proposition 1(a)); the -1 that a
    # connected design would use undercounts the free parameters of a
    # disconnected one by d_a per extra component and inflates sigma^2 with it.
    rank = graph.n_nodes - graph.n_components
    dof = (len(order) - rank) * d_a
    sigma2_within = residual / dof if dof > 0 else None

    return TwoWayFitResult(
        caliber=caliber,
        theta=theta,
        gamma=gamma,
        graph=graph,
        weighted_residual=residual,
        weighted_total=total,
        sigma2_within=sigma2_within,
    )


def fit_both_calibers(
    direct_cells: Sequence[CellMean],
    paired_cells: Sequence[CellMean],
    *,
    heteroscedastic: bool = False,
) -> dict[str, TwoWayFitResult]:
    """Output (i) of 10.4.12: the direct and twin-paired-difference calibers.

    `paired_cells` must hold eta_{tau delta} - eta_{tau 0} computed at the same
    seed_env and the same reference trajectory. On that caliber the true task
    effect is zero (proposition 1(e)(ii)), so ||theta_hat^pair|| is directly
    the measured identification bias.
    """
    direct_keys = {(c.task, c.defect) for c in direct_cells}
    paired_keys = {(c.task, c.defect) for c in paired_cells}
    if direct_keys != paired_keys:
        raise ValueError(
            "the two calibers must cover the same cells, otherwise their bias ratios are "
            f"not comparable: only direct {sorted(direct_keys - paired_keys)}, only "
            f"paired {sorted(paired_keys - direct_keys)}"
        )
    return {
        "direct": fit_two_way(direct_cells, caliber="direct", heteroscedastic=heteroscedastic),
        "paired": fit_two_way(
            paired_cells, caliber="twin_paired_difference", heteroscedastic=heteroscedastic
        ),
    }


@dataclass(frozen=True)
class ContrastVarianceCheck:
    """Output (ii): measured versus predicted Var(theta_hat_tau - theta_hat_tau').

    The measured value is the seed_env resampling variance of the contrast,
    averaged over action components -- proposition 3 predicts an isotropic
    covariance sigma^2 R_eff I_{d_a}, so a per-component average is the like-
    for-like scalar.
    """

    task_a: str
    task_b: str
    measured_variance: float
    predicted_variance: float
    r_eff: float
    sigma2: float | None
    caliber: str
    n_resamples: int
    n_skipped: int

    @property
    def ratio(self) -> float:
        if self.predicted_variance <= 0.0:
            raise ValueError(
                f"proposition 3 predicts a variance of {self.predicted_variance} for "
                f"({self.task_a}, {self.task_b}); the ratio has no denominator"
            )
        return self.measured_variance / self.predicted_variance

    @property
    def within_preregistered_window(self) -> bool:
        low, high = PROP3_RATIO_WINDOW
        return low <= self.ratio <= high

    def as_row(self) -> dict[str, Any]:
        return {
            "task_a": self.task_a,
            "task_b": self.task_b,
            "caliber": self.caliber,
            "r_eff": self.r_eff,
            "sigma2": self.sigma2,
            "measured_variance": self.measured_variance,
            "predicted_variance": self.predicted_variance,
            "ratio": self.ratio,
            "in_window": self.within_preregistered_window,
            "n_resamples": self.n_resamples,
            "n_skipped": self.n_skipped,
        }


def check_proposition_three(
    panel: SeedEnvPanel,
    pairs: Sequence[tuple[str, str]],
    *,
    n_resamples: int,
    rng: np.random.Generator,
    heteroscedastic: bool = False,
    sigma2: float | None = None,
) -> list[ContrastVarianceCheck]:
    """Output (ii) of 10.4.12, pair by pair.

    Resamples seed_env with replacement, refits the two-way model on each
    replicate and compares the empirical contrast variance against
    sigma^2 R_eff. Replicates in which the pair falls into different components
    are counted and excluded rather than filled in -- an unidentifiable
    contrast has no value to average.
    """
    if n_resamples < 2:
        raise ValueError(f"a variance needs at least two replicates, got {n_resamples}")
    reference = fit_two_way(
        panel.pooled(), caliber="direct", heteroscedastic=heteroscedastic
    )
    if heteroscedastic and sigma2 is not None:
        raise ValueError(
            "under the heteroscedastic caliber the noise scale is inside the conductance; "
            "passing sigma^2 as well double counts it (7.9)"
        )
    if not heteroscedastic and sigma2 is None:
        if reference.sigma2_within is None:
            raise ValueError(
                "sigma^2 was not supplied and the fit has no residual degrees of freedom "
                "to estimate it from"
            )
        sigma2 = reference.sigma2_within

    samples: dict[tuple[str, str], list[np.ndarray]] = {p: [] for p in pairs}
    skipped: dict[tuple[str, str], int] = dict.fromkeys(pairs, 0)
    for _ in range(n_resamples):
        cells = panel.resample(rng)
        try:
            fit = fit_two_way(cells, caliber="bootstrap", heteroscedastic=heteroscedastic)
        except (ValueError, TypeError):
            for p in pairs:
                skipped[p] += 1
            continue
        for pair in pairs:
            try:
                samples[pair].append(fit.contrast(*pair))
            except (KeyError, DisconnectedRegime):
                skipped[pair] += 1

    out: list[ContrastVarianceCheck] = []
    for pair in pairs:
        drawn = samples[pair]
        if len(drawn) < 2:
            raise DisconnectedRegime(
                f"pair {pair} was estimable in {len(drawn)} of {n_resamples} replicates; "
                "there is no measured variance to compare against proposition 3"
            )
        stacked = np.stack(drawn, axis=0)
        measured = float(np.mean(stacked.var(axis=0, ddof=1)))
        r_eff = reference.graph.r_eff(*pair)
        if not math.isfinite(r_eff):
            raise DisconnectedRegime(
                f"pair {pair} has infinite effective resistance in the pooled design"
            )
        predicted = reference.graph.predicted_contrast_variance(
            *pair, sigma2=None if heteroscedastic else sigma2
        )
        out.append(
            ContrastVarianceCheck(
                task_a=pair[0],
                task_b=pair[1],
                measured_variance=measured,
                predicted_variance=predicted,
                r_eff=r_eff,
                sigma2=None if heteroscedastic else sigma2,
                caliber="heteroscedastic" if heteroscedastic else "homoscedastic",
                n_resamples=len(drawn),
                n_skipped=skipped[pair],
            )
        )
    return out


@dataclass(frozen=True)
class TwoWayReport:
    """The three mandatory outputs of 10.4.12 in one object."""

    fits: Mapping[str, TwoWayFitResult]
    variance_checks: Sequence[ContrastVarianceCheck]
    xi_g: float

    @property
    def paired_removes_task_scale(self) -> bool:
        """10.4.12's prediction: the paired caliber's bias ratio is smaller."""
        return (
            self.fits["paired"].residual_interaction_share
            < self.fits["direct"].residual_interaction_share
        )

    def as_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for fit in self.fits.values():
            rows.extend(fit.as_rows())
        rows.extend(check.as_row() for check in self.variance_checks)
        rows.append({"caliber": "design", "side": "xi_g", "node": "-", "value": self.xi_g})
        return rows
