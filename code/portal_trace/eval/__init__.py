"""Endpoints, statistics, verdict rendering and the P0 / P1 tables (10.1-10.3, 10.7, 11.3).

Three modules, in the order a result moves through them:

* `endpoints` -- what is measured. The primary decision endpoint (the paired
  success-rate difference), the primary leakage endpoint (Lambda^part, with the
  naive Lambda alongside), the Lambda^orc three-term decomposition, the four
  safety quantities, and Leak_beh with its Fieller interval and denominator rule.
* `stats` -- how it is tested. The crossed random effects model, the
  within-family restricted permutation test, the paired block bootstrap,
  Holm-Bonferroni, the variance model and both MDE formulas.
* `report` -- what may be said about it. Significant, not significant,
  inconclusive; inconclusive is a first-class outcome with its own preregistered
  triggers, and the renderer refuses to soften it.

`endpoints` performs a cross-check at import: every MDE in the 10.7.4 table is
recomputed from the 10.7.3 variance model and compared with the printed value.

Two more modules produce the published tables of the calibration phases, which
are results rather than tests and so live next to the endpoints they are read
with: `calibration_tables` for P0's nine tables (10.2.5) and `regime_tables` for
P1's three outputs (10.3.2). The gate verdicts they feed are in `runners.gates`,
with the ledger that orders them.
"""

from __future__ import annotations

from portal_trace.eval.calibration_tables import (
    P0_TABLE_NAMES,
    CalibrationTable,
    MissingCalibrationInput,
    build_p0_tables,
)
from portal_trace.eval.endpoints import (
    ENDPOINT_SPECS,
    DenominatorRule,
    DistributionSensitivity,
    EndpointProvenance,
    EndpointRole,
    EndpointSpec,
    LambdaOrcDecomposition,
    LeakageReading,
    LeakBehResult,
    PairedEndpointResult,
    ReferencePoint,
    SafetyQuantity,
    StateDistribution,
    lambda_exc,
    leak_beh,
    paired_success_rate_difference,
    phi_from_lambda,
    safety_paired_difference,
)
from portal_trace.eval.regime_tables import (
    NOMINAL_OCCUPANCY_ROWS,
    MissingRegimeInput,
    OccupancyCell,
    build_connectivity_report,
    build_cross_embodiment_tv_table,
    build_occupancy_table,
)
from portal_trace.eval.report import (
    Decision,
    ForbiddenPhrasingError,
    IncompleteRowError,
    InconclusiveReason,
    NullDistributionDigest,
    ResultRow,
    Verdict,
    attach_narrative,
    decide,
    render_row,
    render_summary_table,
    render_verdict,
)
from portal_trace.eval.stats import (
    ALPHA_PRIMARY,
    MDE_COEFFICIENT_NORMAL,
    MDE_COEFFICIENT_T,
    PRIMARY_HYPOTHESIS_FAMILY,
    CellObservations,
    DefectLayer,
    Design,
    EffectEstimate,
    InferenceKind,
    Mde,
    MdeCaliber,
    ModelDiagnostics,
    PermutationUnit,
    VarianceComponents,
    absolute_success_rate_ci,
    benjamini_hochberg,
    fit_crossed_random_effects,
    fit_family_fixed_cluster_robust,
    holm_bonferroni,
    paired_block_bootstrap,
    paired_difference_mde,
    preregistered_inference,
    slope_degenerate_inflation,
    slope_mde,
    slope_mde_degenerate_m1,
    var_paired_mean,
    within_family_restricted_permutation,
)

__all__ = [
    "P0_TABLE_NAMES",
    "CalibrationTable",
    "MissingCalibrationInput",
    "build_p0_tables",
    "NOMINAL_OCCUPANCY_ROWS",
    "MissingRegimeInput",
    "OccupancyCell",
    "build_connectivity_report",
    "build_cross_embodiment_tv_table",
    "build_occupancy_table",
    "ENDPOINT_SPECS",
    "DenominatorRule",
    "DistributionSensitivity",
    "EndpointProvenance",
    "EndpointRole",
    "EndpointSpec",
    "LambdaOrcDecomposition",
    "LeakageReading",
    "LeakBehResult",
    "PairedEndpointResult",
    "ReferencePoint",
    "SafetyQuantity",
    "StateDistribution",
    "lambda_exc",
    "leak_beh",
    "paired_success_rate_difference",
    "phi_from_lambda",
    "safety_paired_difference",
    "Decision",
    "ForbiddenPhrasingError",
    "InconclusiveReason",
    "IncompleteRowError",
    "NullDistributionDigest",
    "ResultRow",
    "Verdict",
    "attach_narrative",
    "decide",
    "render_row",
    "render_summary_table",
    "render_verdict",
    "ALPHA_PRIMARY",
    "MDE_COEFFICIENT_NORMAL",
    "MDE_COEFFICIENT_T",
    "PRIMARY_HYPOTHESIS_FAMILY",
    "CellObservations",
    "DefectLayer",
    "Design",
    "EffectEstimate",
    "InferenceKind",
    "Mde",
    "MdeCaliber",
    "ModelDiagnostics",
    "PermutationUnit",
    "VarianceComponents",
    "absolute_success_rate_ci",
    "benjamini_hochberg",
    "fit_crossed_random_effects",
    "fit_family_fixed_cluster_robust",
    "holm_bonferroni",
    "paired_block_bootstrap",
    "paired_difference_mde",
    "preregistered_inference",
    "slope_degenerate_inflation",
    "slope_mde",
    "slope_mde_degenerate_m1",
    "within_family_restricted_permutation",
    "var_paired_mean",
]
