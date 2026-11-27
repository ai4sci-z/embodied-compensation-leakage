"""E5, the DROID real-data observation arm (proposal 10.4.6).

Five modules, in the order the data moves through them:

* `pair` -- the cmd/ach pose pairing (`action_dict/cartesian_position` against
  `observation/cartesian_position`, both poses), time alignment, and the frozen
  preregistration that supplies the filter and the difference format.
* `segments` -- zero-crossing segmentation of motion segments and the 200 ms
  pre-latency window.
* `transfer_ratio` -- endpoint A, T_comp, in all four readings.
* `anchor_S` -- endpoint B, per-institution delta_hat, psi_j(delta_hat)
  subtraction and the held-out-institution test.
* `arm` -- the two endpoints assembled into the arm, with 10.4.6's three-tier F5
  exit and the budget that says why a zero-training arm still costs 9 GPU-h.

The filter, the difference format, the zero-crossing rule and the window length
are loaded from the preregistration file rather than passed as arguments with
defaults. T_comp moves when any of them moves, so each would otherwise be a
researcher degree of freedom inside the statistic; 10.7.8 item 16 freezes all
four, and `E5Preregistration.load` is the only way into them.

E5's conclusions are observational associations. (A2) excludes human
teleoperation and DROID is human teleoperation; the causal chain is carried by
E1 and E7-E9, not here.
"""

from __future__ import annotations

from portal_trace.real.anchor_S import (
    N_HELD_OUT_LABS,
    PREREGISTERED_VARIANCE_DROP,
    PSI_BASIS_DROID,
    DeltaHat,
    EndpointBResult,
    HeldOutFoldResult,
    LabSplit,
    PhysicalBasis,
    fit_delta_hat,
    held_out_lab_test,
    institution_folds,
    subtract_compensation_model,
)
from portal_trace.real.arm import (
    DROID_ROSTER,
    E5_BUDGET,
    E5_COMPLETE_RUNS,
    E5_ENDPOINT_A_GPU_HOURS,
    E5_ENDPOINT_B_GPU_HOURS,
    E5_GPU_HOURS,
    ENDPOINT_A_NAMES,
    ENDPOINT_B_NAME,
    CausalClaimRefused,
    DroidRosterMismatch,
    E5ArmReport,
    E5Budget,
    EndpointBMissing,
    F5Tier,
    droid_roster,
    e5_report,
)
from portal_trace.real.pair import (
    ACH_FIELD,
    CMD_FIELD,
    N_DROID_LABS,
    DifferenceFormat,
    DroidLabId,
    E5Preregistration,
    FilterSpec,
    PairedPoseSeries,
    PreregistrationMissingError,
    pair_cmd_ach,
)
from portal_trace.real.segments import MotionSegment, SegmentSet, segment_by_zero_crossing
from portal_trace.real.transfer_ratio import (
    BETWEEN_LAB_ICC_FLOOR,
    PREREGISTERED_T_COMP_MIN_LABS,
    PREREGISTERED_T_COMP_THRESHOLD,
    PoseComponentGroup,
    SegmentObservations,
    SegmentReferencing,
    TCompReading,
    TCompReport,
    TCompResult,
    VarianceComponentFit,
    between_lab_variance_component,
    collect_observations,
    t_comp_all_readings,
    t_comp_reading,
)

__all__ = [
    "ACH_FIELD",
    "CMD_FIELD",
    "N_DROID_LABS",
    "DifferenceFormat",
    "DroidLabId",
    "E5Preregistration",
    "FilterSpec",
    "PairedPoseSeries",
    "PreregistrationMissingError",
    "pair_cmd_ach",
    "MotionSegment",
    "SegmentSet",
    "segment_by_zero_crossing",
    "BETWEEN_LAB_ICC_FLOOR",
    "PREREGISTERED_T_COMP_MIN_LABS",
    "PREREGISTERED_T_COMP_THRESHOLD",
    "PoseComponentGroup",
    "SegmentObservations",
    "SegmentReferencing",
    "TCompReading",
    "TCompReport",
    "TCompResult",
    "VarianceComponentFit",
    "between_lab_variance_component",
    "collect_observations",
    "t_comp_all_readings",
    "t_comp_reading",
    "N_HELD_OUT_LABS",
    "PREREGISTERED_VARIANCE_DROP",
    "PSI_BASIS_DROID",
    "DeltaHat",
    "EndpointBResult",
    "HeldOutFoldResult",
    "LabSplit",
    "PhysicalBasis",
    "fit_delta_hat",
    "held_out_lab_test",
    "institution_folds",
    "subtract_compensation_model",
    "E5_COMPLETE_RUNS",
    "E5_GPU_HOURS",
    "E5_ENDPOINT_A_GPU_HOURS",
    "E5_ENDPOINT_B_GPU_HOURS",
    "ENDPOINT_A_NAMES",
    "ENDPOINT_B_NAME",
    "DROID_ROSTER",
    "droid_roster",
    "F5Tier",
    "E5Budget",
    "E5_BUDGET",
    "E5ArmReport",
    "e5_report",
    "DroidRosterMismatch",
    "CausalClaimRefused",
    "EndpointBMissing",
]
