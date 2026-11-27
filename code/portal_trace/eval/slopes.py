"""Slope endpoints: the regression 10.7.5 registers an MDE for, as a fit.

`stats.py` owns the *paired contrast* estimators and the two MDE formulas.
What it does not own is the fit behind the four slope rows of 10.7.5 -- E7's two
readings against log2 K, E8's Lambda^part against ln R_eff, E9's against I -- and
E3's per-defect dose slope, which 10.7.5 deliberately gives no single MDE. This
module is that fit, written once so the crown slopes and the dose slopes cannot
diverge into two conventions.

Three things it is careful about, all of them from 10.7.5 and 10.7.2.

**The slope is estimated inside a morphology family and then averaged across
families.** That sentence is what removes sigma_m from the slope variance
formula, so a fit that pooled the families into one regression line would be
tested against an MDE derived for a different estimator. Every estimate here
therefore carries `per_family_slopes` as well as the combined number, and the
combined number comes from a family fixed effect -- never from a pooled line.

**Which inference runs is decided by M and printed, not chosen.** The dispatch
is `stats.preregistered_inference`, the same one the paired contrasts use, so
E7 at M = 6 gets crossed REML and E8/E9 at M = 4 get the family fixed effect
with within-family cluster-robust standard errors. At M <= 2 -- E3's dose axes
-- the returned flag says no cross-family generalisation is being claimed, and
`SlopeEstimate.cross_family_claim_admissible` is False so a caller cannot make
one by accident.

**The realised regressor geometry travels with the estimate.** 10.7.5's MDE is
a function of sum_i (x_i - x_bar)^2 over the *levels*, and the registered value
was computed on a nominal ladder. `SlopeEstimate.sum_squares_x` is the realised
one, computed from the distinct level positions actually fitted, so the row can
print the registered MDE next to the MDE of the ladder that ran -- which is
exactly what the E9 row of 10.7.5 says has to happen when the regressor is
measured rather than assigned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

from portal_trace.eval.stats import (
    ALPHA_PRIMARY,
    Design,
    InferenceKind,
    Mde,
    MdeCaliber,
    ModelDiagnostics,
    VarianceComponents,
    preregistered_inference,
    slope_mde,
)

__all__ = [
    "SlopeObservations",
    "SlopeEstimate",
    "DegenerateRegressorError",
    "fit_slope",
    "realised_slope_mde",
]

_ZERO_VARIANCE_TOL = 1e-8


class DegenerateRegressorError(ValueError):
    """The regressor has no spread, inside a family or overall.

    Raised rather than returning an infinite standard error: a slope against a
    constant is not a wide estimate of a slope, it is the absence of one, and
    10.7.5's MDE has sum_i (x_i - x_bar)^2 in its denominator.
    """


@dataclass(frozen=True)
class SlopeObservations:
    """One row per (family, defect, level, seed): the unit of 10.7.1 is the cell.

    `level` is the rung label -- ``"C2"``, ``"K=4"``, dose level ``"3"`` -- and
    `x` is that rung's regressor value. They are separate columns because the
    two are not interchangeable: E8's rungs are named by the data ratio table
    while their x is ln R_eff, and E9's x is measured off the trained routers,
    so a rung can keep its identity while its x moves.
    """

    family: np.ndarray
    defect: np.ndarray
    seed: np.ndarray
    level: np.ndarray
    x: np.ndarray
    y: np.ndarray
    label: str

    def __post_init__(self) -> None:
        n = len(self.y)
        for name in ("family", "defect", "seed", "level", "x"):
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(
                    f"{self.label}: column {name} has {len(arr)} rows, y has {n}"
                )
        if n == 0:
            raise ValueError(f"{self.label}: no observations")
        x = np.asarray(self.x, dtype=float)
        y = np.asarray(self.y, dtype=float)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError(
                f"{self.label}: x and y must be finite. A disconnected design rung has "
                "R_eff = infinity and ln R_eff undefined; it is an anchor reported "
                "beside the slope, not a point on it (10.4.8)"
            )
        levels = np.asarray(self.level).astype(str)
        for lv in np.unique(levels):
            positions = np.unique(x[levels == lv])
            if positions.size != 1:
                raise ValueError(
                    f"{self.label}: level {lv!r} carries {positions.size} distinct "
                    "regressor values. A rung has one position on the axis; two means "
                    "either two rungs were merged or the measured regressor was read "
                    "per cell, in which case give each cell its own level label"
                )

    @property
    def design(self) -> Design:
        return Design(
            m_families=int(np.unique(np.asarray(self.family).astype(str)).size),
            d_defects=int(np.unique(np.asarray(self.defect).astype(str)).size),
            s_seeds=int(np.unique(np.asarray(self.seed).astype(str)).size),
            label=self.label,
        )

    @property
    def levels(self) -> tuple[str, ...]:
        """Rung labels in ascending regressor order, which is the reporting order."""
        levels = np.asarray(self.level).astype(str)
        x = np.asarray(self.x, dtype=float)
        seen = {lv: float(x[levels == lv][0]) for lv in np.unique(levels)}
        return tuple(sorted(seen, key=lambda lv: seen[lv]))

    @property
    def level_positions(self) -> dict[str, float]:
        levels = np.asarray(self.level).astype(str)
        x = np.asarray(self.x, dtype=float)
        return {lv: float(x[levels == lv][0]) for lv in self.levels}

    @property
    def sum_squares_x(self) -> float:
        """sum_i (x_i - x_bar)^2 over the distinct levels (10.7.5).

        Over levels, not over rows. 10.7.5's variance formula already divides by
        M and by S; counting each level once per cell would divide by them twice
        and understate the MDE by roughly sqrt(M S).
        """
        positions = np.array(list(self.level_positions.values()), dtype=float)
        if positions.size < 2:
            raise DegenerateRegressorError(
                f"{self.label}: {positions.size} distinct level(s); a slope needs at "
                "least two positions on the axis"
            )
        centred = positions - positions.mean()
        return float(np.dot(centred, centred))

    @property
    def spread_x(self) -> float:
        positions = np.array(list(self.level_positions.values()), dtype=float)
        return float(positions.max() - positions.min())

    def per_family_slopes(self) -> dict[str, float]:
        """The within-family ordinary least squares slope, family by family.

        10.7.5's estimator in its own words. Cells are averaged within a level
        first so that a family with an unbalanced number of seeds at one rung
        does not weight that rung more heavily than the design does.
        """
        families = np.asarray(self.family).astype(str)
        levels = np.asarray(self.level).astype(str)
        x = np.asarray(self.x, dtype=float)
        y = np.asarray(self.y, dtype=float)
        out: dict[str, float] = {}
        for fam in sorted(np.unique(families)):
            in_family = families == fam
            fam_levels = np.unique(levels[in_family])
            if fam_levels.size < 2:
                raise DegenerateRegressorError(
                    f"{self.label}: family {fam!r} has {fam_levels.size} level(s). "
                    "10.7.5 estimates the slope inside a family and averages across "
                    "families, which a single-level family cannot contribute to"
                )
            xs = np.array(
                [float(x[in_family & (levels == lv)][0]) for lv in fam_levels]
            )
            ys = np.array(
                [float(np.mean(y[in_family & (levels == lv)])) for lv in fam_levels]
            )
            centred = xs - xs.mean()
            sxx = float(np.dot(centred, centred))
            if sxx <= 0.0:
                raise DegenerateRegressorError(
                    f"{self.label}: family {fam!r} has no regressor spread"
                )
            out[fam] = float(np.dot(centred, ys - ys.mean()) / sxx)
        return out


@dataclass(frozen=True)
class SlopeEstimate:
    """A fitted slope with the columns 11.4 makes mandatory for a summary row."""

    label: str
    slope: float
    standard_error: float
    ci: tuple[float, float]
    p_value: float
    df: float
    inference: InferenceKind
    inference_flag: str
    design: Design
    n_cells: int
    n_levels: int
    sum_squares_x: float
    spread_x: float
    per_family_slopes: dict[str, float]
    diagnostics: ModelDiagnostics
    alpha: float

    @property
    def cross_family_claim_admissible(self) -> bool:
        """10.7.2: at M <= 2 only within-family statements may be made."""
        return self.inference is not InferenceKind.WITHIN_FAMILY_PAIRED_ONLY

    @property
    def family_slopes_agree_in_sign(self) -> bool:
        signs = {math.copysign(1.0, s) for s in self.per_family_slopes.values() if s != 0.0}
        return len(signs) <= 1

    def describe(self) -> str:
        lo, hi = self.ci
        return (
            f"{self.label}: beta_hat = {self.slope:.4g} "
            f"[{lo:.4g}, {hi:.4g}], SE {self.standard_error:.4g}, "
            f"{self.design} over {self.n_levels} levels, "
            f"sum (x - x_bar)^2 = {self.sum_squares_x:.4g}"
        )


def fit_slope(
    obs: SlopeObservations, *, alpha: float = ALPHA_PRIMARY
) -> SlopeEstimate:
    """Fit y on x under the inference 10.7.2 assigns to this design's M.

    At M >= 5 this is the crossed random-effects model of 10.7.2 with the
    regressor in the fixed part; at M <= 4 the morphology family becomes a fixed
    effect with within-family cluster-robust standard errors, which 10.7.2 calls
    not optional rather than a fallback. Either way the returned flag says which
    ran, because every row of the summary table has to state it.
    """
    import pandas as pd
    import statsmodels.formula.api as smf

    design = obs.design
    kind, flag = preregistered_inference(design)
    per_family = obs.per_family_slopes()
    frame = pd.DataFrame(
        {
            "y": np.asarray(obs.y, dtype=float),
            "x": np.asarray(obs.x, dtype=float),
            "family": np.asarray(obs.family).astype(str),
            "defect": np.asarray(obs.defect).astype(str),
            "seed": np.asarray(obs.seed).astype(str),
        }
    )
    frame["cell"] = frame["family"] + "|" + frame["defect"]

    if kind is InferenceKind.CROSSED_RANDOM_EFFECTS_REML:
        frame["_g"] = 1
        vc = {"family": "0 + C(family)", "defect": "0 + C(defect)", "cell": "0 + C(cell)"}
        model = smf.mixedlm("y ~ x", frame, groups=frame["_g"], vc_formula=vc, re_formula="0")
        with np.errstate(all="ignore"):
            fit = model.fit(reml=True, method=["lbfgs"])
        vcomp = dict(zip(model.exog_vc.names, np.atleast_1d(fit.vcomp), strict=True))
        zeros = tuple(n for n, v in vcomp.items() if float(v) <= _ZERO_VARIANCE_TOL)
        if float(fit.scale) <= _ZERO_VARIANCE_TOL:
            zeros = (*zeros, "residual")
        diagnostics = ModelDiagnostics(
            converged=bool(getattr(fit, "converged", False)),
            zero_variance_components=zeros,
            message="" if getattr(fit, "converged", False) else "REML did not converge",
        )
        beta = float(fit.fe_params["x"])
        se = float(fit.bse["x"])
        df = float(max(design.n_cells - 2, 1))
    else:
        formula = "y ~ x + C(family)"
        if design.d_defects > 1:
            formula += " + C(defect)"
        fit = smf.ols(formula, frame).fit(
            cov_type="cluster", cov_kwds={"groups": frame["family"]}, use_t=True
        )
        beta = float(fit.params["x"])
        se = float(fit.bse["x"])
        df = float(max(design.m_families - 1, 1))
        flag = (
            f"{flag}; cluster-robust SE on G = {design.m_families} clusters, "
            f"t reference with {df:.0f} df"
        )
        diagnostics = ModelDiagnostics(converged=True, zero_variance_components=())

    tcrit = float(sps.t.ppf(1.0 - alpha / 2.0, df))
    p = float(2.0 * sps.t.sf(abs(beta / se), df)) if se > 0 else float("nan")
    return SlopeEstimate(
        label=obs.label,
        slope=beta,
        standard_error=se,
        ci=(beta - tcrit * se, beta + tcrit * se),
        p_value=p,
        df=df,
        inference=kind,
        inference_flag=flag,
        design=design,
        n_cells=design.n_cells,
        n_levels=len(obs.levels),
        sum_squares_x=obs.sum_squares_x,
        spread_x=obs.spread_x,
        per_family_slopes=per_family,
        diagnostics=diagnostics,
        alpha=alpha,
    )


def realised_slope_mde(
    obs: SlopeObservations,
    components: VarianceComponents,
    *,
    unit: str = "1",
    caliber: MdeCaliber = MdeCaliber.TABLE_T_CORRECTED,
) -> Mde:
    """10.7.5's MDE recomputed on the ladder that actually ran.

    Reported *beside* the registered value, never instead of it: 10.7.8 item 4
    freezes the table, and the E9 row of 10.7.5 spells out that the recomputation
    on the realised ladder is an addition to the registered number.
    """
    return slope_mde(
        components,
        obs.design,
        sum_squares_x=obs.sum_squares_x,
        unit=unit,
        caliber=caliber,
    )
