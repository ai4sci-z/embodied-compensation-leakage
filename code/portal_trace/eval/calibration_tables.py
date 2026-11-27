"""P0's nine calibration tables (proposal 10.2.5).

10.2.5 closes the pre-calibration phase with a deliverable list: "the Goldilocks
success-rate matrix, the kappa table (with `tol_kind`), the cl_pm / A / two
classification angles table, the force calibration curves (one per solver-
parameter group), D3's two-caliber equivalent-delay table, the level-to-real-
machine reference table, the per-level three-way feasibility result table, the
`shadow_joint_offset` table and the D5 dead-zone duty table. These nine tables
cost 0 GPU-h and 21 CPU-h between them. They remain descriptive inputs to the
decision: the supplied preregistration does not define how eight task results
select one level per (family, defect), nor how ties are resolved, so this module
must not manufacture that reduction or fire the five-defect exit from it.

This module is the producer for that list, and it is only an assembler: every
number in every cell arrives as a measured record, either through an input
dataclass the calibration scripts fill in or through `LockFile`. Nothing here
has a default and nothing here estimates a quantity it was not given -- a table
whose inputs are absent raises `MissingCalibrationInput`, which names the table,
the rows that are missing and what produces them.

What the module *does* own is the derived columns, because those are the part
10.2.5 freezes and the part a reader has to be able to check against the
proposal: whether a cell bites (25-65% uncompensated, clean twin at or above
90%), whether kappa was evaluated in the phase 10.2.3 pins it to, whether a
level's measured angle admits it to T8's sign prediction, whether D3b is below
1 ms at every level, and whether the three-way feasible set of 10.2.3 is empty.
Those rules are transcribed with their section numbers next to them.

Three refusals are worth naming, because each closes a way of producing a table
that would look right and mean something else.

* `KappaPhase.WHOLE_TRAJECTORY_PEAK` exists so that it can be rejected. 10.2.3
  shows the same D2 level reading 15 mm at 0.30 m/s and 1.0 mm at 0.020 m/s;
  a kappa table built off the trajectory peak is a table about approach speed.
* `ForceWindow.QUASI_STATIC` likewise. Appendix A-11: controller and environment
  stiffness are in series, so the quasi-static force saturates at k_p * d and a
  0.5 mm penetration at k_p = 150 is 0.075 N, under the noise floor. The safety
  endpoint is defined on the collision transient, and a curve that mixes the two
  windows is not the curve 10.2.3 asks for.
* A saturated clearance sweep is a lower bound, not a measurement. It is
  harmless when it lands on the wide side of the intermediate band and fatal
  when it lands on the narrow side, so the feasibility builder refuses exactly
  the second case rather than carrying a flag nobody reads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from portal_trace.defects.params import DEG, DefectSpec, defect
from portal_trace.eval.endpoints import ENDPOINT_SPECS
from portal_trace.lockfile import REQUIRED_MEASUREMENTS, LockFile

__all__ = [
    "GOLDILOCKS_WINDOW_PCT",
    "CLEAN_TWIN_CEILING_PCT",
    "GOLDILOCKS_ROLLOUTS_PER_CELL",
    "GOLDILOCKS_MIN_PASSING_DEFECTS",
    "KAPPA_PHASE_DISTANCE_M",
    "KAPPA_PHASE_SPEED_MS",
    "SIGN_PREDICTION_MAX_ANGLE_DEG",
    "D3B_MAX_EQUIVALENT_DELAY_S",
    "D5_DUTY_FLOOR",
    "SOLVER_GROUP_IDS",
    "P0_TABLE_NAMES",
    "REAL_MACHINE_REFERENCE",
    "MissingCalibrationInput",
    "DecisionRuleUnavailable",
    "CalibrationTable",
    "KappaPhase",
    "ForceWindow",
    "D5Caliber",
    "ReferenceAxis",
    "RealMachineReference",
    "GoldilocksCell",
    "KappaCell",
    "DirectionalClearanceCell",
    "ForceCalibrationPoint",
    "D3DelayRow",
    "ShadowJointOffsetCell",
    "DeadZoneDutyCell",
    "defects_meeting_goldilocks",
    "build_goldilocks_table",
    "build_kappa_table",
    "build_directional_clearance_table",
    "build_force_calibration_table",
    "build_d3_equivalent_delay_table",
    "build_level_reference_table",
    "build_feasibility_table",
    "build_shadow_joint_offset_table",
    "build_d5_dead_zone_duty_table",
    "build_p0_tables",
]

# 10.2.2: the Goldilocks criterion. The window is on the *uncompensated* tracker
# (ILC k = 0) and the ceiling is on the same family's clean twin; 10.2.4's
# closing note insists both are read as a defect-over-task-tolerance criterion
# rather than a defect-amplitude one.
GOLDILOCKS_WINDOW_PCT = (25.0, 65.0)
CLEAN_TWIN_CEILING_PCT = 90.0
GOLDILOCKS_ROLLOUTS_PER_CELL = 20
# 10.2.2 names an exit at fewer than five defects, but the supplied materials do
# not define the cross-task level-selection rule needed to count those defects.
# The threshold is retained as a reported preregistration constant; applying it
# remains blocked by `DecisionRuleUnavailable`.
GOLDILOCKS_MIN_PASSING_DEFECTS = 5

# 10.2.3, frozen by 10.2.5: kappa is evaluated on the terminal fine-approach
# frames and a whole-trajectory peak is not admissible.
KAPPA_PHASE_DISTANCE_M = 15e-3
KAPPA_PHASE_SPEED_MS = 0.05

# 10.2.3 and 11.3's eleventh item: only levels whose measured angle(v, u) or
# angle(v, n) is under 30 degrees enter proposition 2(c)'s sign prediction.
SIGN_PREDICTION_MAX_ANGLE_DEG = 30.0

# 10.2.3 requirement 4: T8's carrier must be D3b, which is purely static and
# below 1 ms at every level. D3a's omega_n sits inside the control bandwidth.
D3B_MAX_EQUIVALENT_DELAY_S = 1e-3

# 10.2.5: after the caliber is unified, a family whose every joint spends less
# than 5% of the time inside the dead zone takes D5 out of the primary test.
D5_DUTY_FLOOR = 0.05

# 10.2.5 freezes three solref/solimp groups, "default / hard / soft"; A8's level
# block in `configs/experiments/ablations.yaml` carries the scales themselves.
SOLVER_GROUP_IDS: tuple[str, ...] = ("default", "hard", "soft")

# The nine tables of 10.2.5, in the order that section lists them.
P0_TABLE_NAMES: tuple[str, ...] = (
    "goldilocks_success_rate",
    "kappa",
    "directional_clearance",
    "force_calibration",
    "d3_equivalent_delay",
    "level_real_machine_reference",
    "t8_three_way_feasibility",
    "shadow_joint_offset",
    "d5_dead_zone_duty",
)


class MissingCalibrationInput(LookupError):
    """A P0 table was requested and the measurements behind it are not there.

    Distinct from `MeasurementMissingError`, which covers a single declared
    constant. This one covers a whole table's worth of rollout-level records --
    the Goldilocks sweep, the clearance queries, the force curve -- which are
    produced by a calibration run rather than written into the lock file.
    """

    def __init__(self, table: str, what: str, produced_by: str) -> None:
        self.table = table
        self.what = what
        self.produced_by = produced_by
        super().__init__(
            f"P0 table {table!r} cannot be built: {what}. It is produced by "
            f"{produced_by}. 10.2.5 publishes all nine tables, and a table with an "
            "invented row is not one of them"
        )


class DecisionRuleUnavailable(RuntimeError):
    """A requested P0 decision has no explicit preregistered reduction rule."""


@dataclass(frozen=True)
class CalibrationTable:
    """One of 10.2.5's nine tables: named columns, units, and rows.

    Rows are plain dictionaries rather than a per-table dataclass because these
    are published artefacts -- long-format CSV and JSON -- and the column list is
    the part that has to be checkable against the proposal. `__post_init__`
    enforces that every row carries exactly the declared columns, which is what
    stops a builder from quietly dropping a column the section requires.
    """

    name: str
    title: str
    proposal_ref: str
    columns: tuple[str, ...]
    units: Mapping[str, str]
    rows: tuple[Mapping[str, Any], ...]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.columns:
            raise ValueError(f"{self.name}: a table with no columns is not a table")
        unknown = sorted(set(self.units) - set(self.columns))
        if unknown:
            raise ValueError(f"{self.name}: units declared for absent columns {unknown}")
        expected = set(self.columns)
        for index, row in enumerate(self.rows):
            got = set(row)
            if got != expected:
                raise ValueError(
                    f"{self.name} row {index}: columns {sorted(got)} do not match the "
                    f"declared {sorted(expected)}"
                )

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    def column(self, name: str) -> tuple[Any, ...]:
        if name not in self.columns:
            raise KeyError(f"{self.name} has no column {name!r}; it has {list(self.columns)}")
        return tuple(row[name] for row in self.rows)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "proposal_ref": self.proposal_ref,
            "columns": list(self.columns),
            "units": dict(self.units),
            "n_rows": self.n_rows,
            "rows": [dict(row) for row in self.rows],
            "notes": list(self.notes),
        }

    def render(self) -> str:
        header = " | ".join(
            f"{c}[{self.units[c]}]" if c in self.units else c for c in self.columns
        )
        lines = [f"{self.name} -- {self.title} ({self.proposal_ref})", f"  {header}"]
        for row in self.rows:
            lines.append("  " + " | ".join(_render_cell(row[c]) for c in self.columns))
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def _render_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _require(rows: Sequence[Any], table: str, what: str, produced_by: str) -> None:
    if not rows:
        raise MissingCalibrationInput(table, f"no {what} were supplied", produced_by)


# ---------------------------------------------------------------------------
# Table 1: the Goldilocks success-rate matrix (10.2.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldilocksCell:
    """One (defect, level, family, task) cell of the 10.2.2 sweep.

    Both sides are counts rather than rates, because the criterion is read at
    20 rollouts and a rate on its own hides how many rollouts produced it.
    """

    defect: str
    level: int
    family: str
    task: str
    n_rollouts: int
    n_success: int
    clean_twin_n_rollouts: int
    clean_twin_n_success: int

    def __post_init__(self) -> None:
        if self.n_rollouts != GOLDILOCKS_ROLLOUTS_PER_CELL:
            raise ValueError(
                f"{self.defect} L{self.level} {self.family}/{self.task}: 10.2.2 fixes "
                f"the sweep at {GOLDILOCKS_ROLLOUTS_PER_CELL} rollouts per cell and this "
                f"cell has {self.n_rollouts}. The 25-65% window is a binomial statement "
                "at that count; read at another count it is a different criterion"
            )
        if self.clean_twin_n_rollouts <= 0:
            raise ValueError(
                f"{self.family}/{self.task}: the ceiling condition is a rate on the clean "
                "twin and there are no clean-twin rollouts to take it from"
            )
        for name in ("n_success", "clean_twin_n_success"):
            count = getattr(self, name)
            total = self.n_rollouts if name == "n_success" else self.clean_twin_n_rollouts
            if not 0 <= count <= total:
                raise ValueError(f"{name}={count} is not in [0, {total}]")

    @property
    def success_rate_pct(self) -> float:
        return 100.0 * self.n_success / self.n_rollouts

    @property
    def clean_twin_rate_pct(self) -> float:
        return 100.0 * self.clean_twin_n_success / self.clean_twin_n_rollouts

    @property
    def bites(self) -> bool:
        low, high = GOLDILOCKS_WINDOW_PCT
        return low <= self.success_rate_pct <= high

    @property
    def ceiling_holds(self) -> bool:
        return self.clean_twin_rate_pct >= CLEAN_TWIN_CEILING_PCT

    @property
    def passes(self) -> bool:
        return self.bites and self.ceiling_holds


def defects_meeting_goldilocks(cells: Sequence[GoldilocksCell]) -> tuple[str, ...]:
    """Refuse a defect-level claim until its cross-task rule is preregistered."""
    del cells
    raise DecisionRuleUnavailable(
        "Goldilocks defect-level decision rule unavailable: `passes` is defined only "
        "for each (defect, level, family, task) cell. No explicit preregistered rule "
        "reduces the eight task results to one level per (family, defect), or resolves "
        "ties. Refusing to name passing defects or execute the five-defect exit"
    )


def build_goldilocks_table(cells: Sequence[GoldilocksCell]) -> CalibrationTable:
    """10.2.5 table 1: 6 defects x 5 levels x 6 families x 8 tasks."""
    _require(cells, "goldilocks_success_rate", "sweep cells",
             "the 10.2.2 uncompensated-tracker sweep (ILC k = 0)")
    rows = [
        {
            "defect": c.defect,
            "level": c.level,
            "family": c.family,
            "task": c.task,
            "n_rollouts": c.n_rollouts,
            "success_rate": c.success_rate_pct,
            "clean_twin_rate": c.clean_twin_rate_pct,
            "bites": c.bites,
            "ceiling_holds": c.ceiling_holds,
            "passes": c.passes,
        }
        for c in cells
    ]
    notes = [
        f"window {GOLDILOCKS_WINDOW_PCT[0]:g}-{GOLDILOCKS_WINDOW_PCT[1]:g}% on the "
        f"uncompensated tracker, clean twin at or above {CLEAN_TWIN_CEILING_PCT:g}% (10.2.2)",
        "passes is a per-cell descriptive predicate only; it neither establishes a "
        "defect-level pass nor selects a dose level",
        "defect-level selection and the five-defect exit are decision-rule-unavailable "
        "until an explicit preregistered cross-task reduction and tie-break rule exists",
    ]
    return CalibrationTable(
        name="goldilocks_success_rate",
        title="Goldilocks success-rate matrix",
        proposal_ref="10.2.2, 10.2.5",
        columns=(
            "defect", "level", "family", "task", "n_rollouts", "success_rate",
            "clean_twin_rate", "bites", "ceiling_holds", "passes",
        ),
        units={"success_rate": "%", "clean_twin_rate": "%"},
        rows=tuple(rows),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Table 2: kappa, with its tol_kind column (10.2.3 quantity 1)
# ---------------------------------------------------------------------------


class KappaPhase(str, Enum):
    """Which frames a `KappaCell` was evaluated on.

    Declared by the producer rather than inferred, because the two readings are
    both computable from the same rollout and only one of them is kappa.
    """

    TERMINAL_FINE_APPROACH = "terminal_fine_approach"
    WHOLE_TRAJECTORY_PEAK = "whole_trajectory_peak"


@dataclass(frozen=True)
class KappaCell:
    """||c_delta(x*)|| for one cell, with the phase it was taken in."""

    task: str
    geom_variant: str
    defect: str
    level: int
    family: str
    c_norm_m: float
    n_frames_in_phase: int
    phase: KappaPhase

    def __post_init__(self) -> None:
        if self.phase is not KappaPhase.TERMINAL_FINE_APPROACH:
            raise ValueError(
                f"{self.defect} L{self.level} on {self.task}: kappa was evaluated on "
                f"{self.phase.value}. 10.2.5 freezes the evaluation phase at end-effector "
                f"distance below {KAPPA_PHASE_DISTANCE_M * 1e3:g} mm and speed below "
                f"{KAPPA_PHASE_SPEED_MS:g} m/s; at 0.30 m/s the same D2 level reads 15 mm "
                "instead of 1.0 mm, so a whole-trajectory peak measures the approach speed"
            )
        if self.n_frames_in_phase <= 0:
            raise ValueError(
                f"{self.defect} L{self.level} on {self.task}: no frames fell in the "
                "terminal fine-approach phase, so this cell has no kappa"
            )
        if not self.c_norm_m >= 0.0:
            raise ValueError(f"||c_delta(x*)|| must be non-negative, got {self.c_norm_m}")


def build_kappa_table(cells: Sequence[KappaCell], *, lock: LockFile) -> CalibrationTable:
    """10.2.5 table 2: kappa = ||c_delta(x*)|| / tol(tau), carrying `tol_kind`.

    The tolerance and its kind come from the lock file per (task, geometry
    variant), because 9.5.4 measures them and 10.2.3 forbids ranking a
    `radial_clearance` kappa against a `marker_halfwidth` one. The kind travels
    in its own column for exactly that reason.
    """
    _require(cells, "kappa", "terminal-phase readings of ||c_delta(x*)||",
             "the 10.2.3 kappa sweep over the frozen evaluation phase")
    tol_decl = REQUIRED_MEASUREMENTS["task.tol.{task}.{geom_variant}"]
    kind_decl = REQUIRED_MEASUREMENTS["task.tol_kind.{task}.{geom_variant}"]
    rows = []
    for c in cells:
        tol = float(lock.get(tol_decl, task=c.task, geom_variant=c.geom_variant))
        kind = str(lock.get(kind_decl, task=c.task, geom_variant=c.geom_variant))
        if tol <= 0.0:
            raise ValueError(
                f"{c.task}/{c.geom_variant}: the measured tolerance is {tol}, and kappa "
                "divides by it"
            )
        rows.append(
            {
                "task": c.task,
                "geom_variant": c.geom_variant,
                "defect": c.defect,
                "level": c.level,
                "family": c.family,
                "c_norm": c.c_norm_m,
                "tol": tol,
                "tol_kind": kind,
                "kappa": c.c_norm_m / tol,
                "n_frames_in_phase": c.n_frames_in_phase,
            }
        )
    return CalibrationTable(
        name="kappa",
        title="kappa = ||c_delta(x*)|| / tol, with tol_kind",
        proposal_ref="10.2.3 quantity 1, 10.2.5",
        columns=(
            "task", "geom_variant", "defect", "level", "family", "c_norm", "tol",
            "tol_kind", "kappa", "n_frames_in_phase",
        ),
        units={"c_norm": "m", "tol": "m", "kappa": "dimensionless"},
        rows=tuple(rows),
        notes=(
            f"evaluated on frames with end-effector distance below "
            f"{KAPPA_PHASE_DISTANCE_M * 1e3:g} mm and speed below {KAPPA_PHASE_SPEED_MS:g} "
            "m/s (10.2.3, frozen by 10.2.5)",
            "kappa is only ordered within one tol_kind: radial_clearance, "
            "marker_halfwidth and centroid_projection are three semantics (9.5.4)",
        ),
    )


# ---------------------------------------------------------------------------
# Table 3: cl_pm, A and the two classification angles (10.2.3 quantity 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectionalClearanceCell:
    """One stepped-collision query, as `registry.geometry.measure_clearance` gives it.

    Floats rather than a `ClearanceResult` so that this module does not pull
    MuJoCo into `eval/`; the caller passes the fields off the result it measured.
    `saturated_*` says the sweep ran out of range, which makes that side a lower
    bound rather than a clearance.
    """

    defect: str
    level: int
    task: str
    geom_variant: str
    cl_plus_m: float
    cl_minus_m: float
    angle_v_u_deg: float
    angle_v_n_deg: float
    saturated_plus: bool = False
    saturated_minus: bool = False

    def __post_init__(self) -> None:
        for name in ("cl_plus_m", "cl_minus_m"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} is a distance and cannot be negative")
        for name in ("angle_v_u_deg", "angle_v_n_deg"):
            angle = getattr(self, name)
            if not 0.0 <= angle <= 180.0:
                raise ValueError(f"{name} must lie in [0, 180] degrees, got {angle}")

    @property
    def asymmetry_m(self) -> float:
        """A = cl_- - cl_+ (8.1, 9.5.5)."""
        return self.cl_minus_m - self.cl_plus_m

    @property
    def enters_sign_prediction(self) -> bool:
        """11.3 item 11: angle(v, u) or angle(v, n) below 30 degrees."""
        return min(self.angle_v_u_deg, self.angle_v_n_deg) < SIGN_PREDICTION_MAX_ANGLE_DEG

    @property
    def predicted_danger_direction(self) -> str | None:
        """The direction proposition 2(b) predicts, where a prediction is made.

        Only the sign of A decides it, and only inside the intermediate band --
        which the feasibility table checks, not this one. Outside the admitted
        angle the level is observed and not predicted (10.2.3).
        """
        if not self.enters_sign_prediction or self.asymmetry_m == 0.0:
            return None
        return "D->C" if self.asymmetry_m > 0.0 else "C->D"


def build_directional_clearance_table(
    cells: Sequence[DirectionalClearanceCell],
) -> CalibrationTable:
    """10.2.5 table 3: cl_+, cl_-, A and the two classification angles."""
    _require(cells, "directional_clearance", "stepped collision queries",
             "registry.geometry.measure_clearance along v = c_delta(x*)/||c_delta(x*)||")
    rows = [
        {
            "defect": c.defect,
            "level": c.level,
            "task": c.task,
            "geom_variant": c.geom_variant,
            "cl_plus": c.cl_plus_m,
            "cl_minus": c.cl_minus_m,
            "asymmetry": c.asymmetry_m,
            "angle_v_u": c.angle_v_u_deg,
            "angle_v_n": c.angle_v_n_deg,
            "saturated_plus": c.saturated_plus,
            "saturated_minus": c.saturated_minus,
            "enters_sign_prediction": c.enters_sign_prediction,
            "predicted_danger_direction": c.predicted_danger_direction,
        }
        for c in cells
    ]
    return CalibrationTable(
        name="directional_clearance",
        title="cl_pm, the asymmetry A and the two classification angles",
        proposal_ref="10.2.3 quantity 2, 9.5.5, 10.2.5",
        columns=(
            "defect", "level", "task", "geom_variant", "cl_plus", "cl_minus", "asymmetry",
            "angle_v_u", "angle_v_n", "saturated_plus", "saturated_minus",
            "enters_sign_prediction", "predicted_danger_direction",
        ),
        units={
            "cl_plus": "m", "cl_minus": "m", "asymmetry": "m",
            "angle_v_u": "deg", "angle_v_n": "deg",
        },
        rows=tuple(rows),
        notes=(
            f"only levels with a measured angle below {SIGN_PREDICTION_MAX_ANGLE_DEG:g} "
            "degrees enter T8's sign prediction; the rest are observed and not predicted "
            "(10.2.3, 11.3 item 11)",
            "D1 and D6 get no prior classification -- it is decided per level (10.2.3)",
            "a saturated sweep is a lower bound on that side, not a clearance",
        ),
    )


# ---------------------------------------------------------------------------
# Table 4: the force calibration curves (10.2.3 quantity 3)
# ---------------------------------------------------------------------------


class ForceWindow(str, Enum):
    """Which contact window a peak `cfrc_ext` reading was taken in (A-11)."""

    COLLISION_TRANSIENT = "collision_transient"
    QUASI_STATIC = "quasi_static"


@dataclass(frozen=True)
class ForceCalibrationPoint:
    """One point of the "||c|| -> peak cfrc_ext" curve at one solver group."""

    solver_group: str
    terminal_offset_m: float
    approach_speed_ms: float
    peak_cfrc_ext_n: float
    quasi_static_kp_d_n: float
    window: ForceWindow

    def __post_init__(self) -> None:
        if self.window is not ForceWindow.COLLISION_TRANSIENT:
            raise ValueError(
                f"solver group {self.solver_group}, d = {self.terminal_offset_m * 1e3:g} mm: "
                f"the reading is from the {self.window.value} window. Appendix A-11 excludes "
                "quasi-static contact force from the safety endpoint: controller and "
                "environment stiffness are in series, F -> k_p d saturates, and at "
                "k_p = 150 a 0.5 mm penetration is 0.075 N, under the noise floor"
            )
        if self.approach_speed_ms <= 0.0:
            raise ValueError(
                "a collision transient needs a positive approach speed; the curve is "
                "read as dominated by v rather than by d (10.2.3)"
            )
        if self.quasi_static_kp_d_n <= 0.0:
            raise ValueError(
                "the analytic quasi-static prediction k_p d is the curve's control column "
                "and has to be positive"
            )


def build_force_calibration_table(
    points: Sequence[ForceCalibrationPoint],
    *,
    solver_group_ids: Sequence[str] = SOLVER_GROUP_IDS,
) -> CalibrationTable:
    """10.2.5 table 4: one curve per solver-parameter group, all three present.

    A8 is the gate on the safety endpoint and MuJoCo's contact is a soft
    constraint, so "environment stiffness" is a solver parameter rather than a
    physical constant (10.2.3). A curve measured on one group is not the
    deliverable, and this refuses to call it one.
    """
    _require(points, "force_calibration", "curve points",
             "the 10.2.3 sweep of terminal offset d and approach speed v")
    present = {p.solver_group for p in points}
    missing = [g for g in solver_group_ids if g not in present]
    if missing:
        raise MissingCalibrationInput(
            "force_calibration",
            f"no points for solver group(s) {missing}",
            "the same sweep run under A8's solref/solimp level block",
        )
    unknown = sorted(present - set(solver_group_ids))
    if unknown:
        raise ValueError(
            f"points carry solver group(s) {unknown}; 10.2.5 freezes the three groups at "
            f"{list(solver_group_ids)}"
        )
    rows = [
        {
            "solver_group": p.solver_group,
            "d": p.terminal_offset_m,
            "v": p.approach_speed_ms,
            "peak_cfrc_ext": p.peak_cfrc_ext_n,
            "quasi_static_kp_d": p.quasi_static_kp_d_n,
            "transient_over_quasi_static": p.peak_cfrc_ext_n / p.quasi_static_kp_d_n,
        }
        for p in points
    ]
    return CalibrationTable(
        name="force_calibration",
        title="peak cfrc_ext against terminal offset and approach speed",
        proposal_ref="10.2.3 quantity 3, appendix A-11, 10.2.5",
        columns=(
            "solver_group", "d", "v", "peak_cfrc_ext", "quasi_static_kp_d",
            "transient_over_quasi_static",
        ),
        units={
            "d": "m", "v": "m/s", "peak_cfrc_ext": "N", "quasi_static_kp_d": "N",
            "transient_over_quasi_static": "dimensionless",
        },
        rows=tuple(rows),
        notes=(
            "collision-transient readings only; the quasi-static column is the analytic "
            "k_p d control, not a second measurement",
            f"one curve per solver group {list(solver_group_ids)} (A8, 10.2.5)",
        ),
    )


# ---------------------------------------------------------------------------
# Table 5: D3's two calibers (10.2.3 quantity 4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class D3DelayRow:
    """Equivalent delay of one D3 caliber at one level.

    `omega_n_hz` is D3a's natural frequency and is absent on D3b, which adds no
    dynamic term at all -- that absence is the property T8 needs, so it is a
    `None` and not a zero.
    """

    caliber: str
    level: int
    equivalent_delay_s: float
    omega_n_hz: float | None = None

    def __post_init__(self) -> None:
        if self.caliber not in ("D3a", "D3b"):
            raise ValueError(f"D3 has two calibers, D3a and D3b; got {self.caliber!r}")
        if self.caliber == "D3b" and self.omega_n_hz is not None:
            raise ValueError(
                "D3b is the purely static caliber (a perturbed gravity-compensation "
                "coefficient) and has no natural frequency; reporting one would make it "
                "look like a dynamic term and undo its orthogonality to D2 (A-16)"
            )
        if self.equivalent_delay_s < 0.0:
            raise ValueError(
                f"an equivalent delay cannot be negative, got {self.equivalent_delay_s}"
            )


def build_d3_equivalent_delay_table(rows: Sequence[D3DelayRow]) -> CalibrationTable:
    """10.2.5 table 5: both D3 calibers at every level.

    The table exists to show that T8's carrier is orthogonal to D2. That is a
    statement about all five levels of D3b at once, so a partial table cannot
    make it and the builder requires the full ladder of both calibers.
    """
    _require(rows, "d3_equivalent_delay", "delay readings",
             "the 10.2.3 cross-correlation time shift on both D3 calibers")
    for caliber in ("D3a", "D3b"):
        levels = sorted(r.level for r in rows if r.caliber == caliber)
        expected = sorted(lv.index for lv in defect(caliber).levels)
        if levels != expected:
            raise MissingCalibrationInput(
                "d3_equivalent_delay",
                f"{caliber} has levels {levels}, the 9.2.10 ladder is {expected}",
                "the 10.2.3 cross-correlation time shift on both D3 calibers",
            )
    table_rows = [
        {
            "caliber": r.caliber,
            "level": r.level,
            "equivalent_delay": r.equivalent_delay_s,
            "omega_n": r.omega_n_hz,
            "below_1ms": r.equivalent_delay_s < D3B_MAX_EQUIVALENT_DELAY_S,
        }
        for r in rows
    ]
    carrier_admissible = all(
        r.equivalent_delay_s < D3B_MAX_EQUIVALENT_DELAY_S for r in rows if r.caliber == "D3b"
    )
    notes = [
        f"T8's carrier must be D3b and below {D3B_MAX_EQUIVALENT_DELAY_S * 1e3:g} ms at "
        "every level (10.2.3 quantity 4)",
        "D3a is not orthogonal to D2 -- its omega_n sits inside the control bandwidth -- "
        "so it enters the leakage-existence and dose arms only (A-16)",
    ]
    if not carrier_admissible:
        notes.append(
            "D3b exceeds the 1 ms bound at some level, so it is not a purely static "
            "caliber on this build and T8's contrast carrier is contaminated by a "
            "dynamic term shared with D2"
        )
    return CalibrationTable(
        name="d3_equivalent_delay",
        title="equivalent delay of both D3 calibers, per level",
        proposal_ref="10.2.3 quantity 4, 10.2.5",
        columns=("caliber", "level", "equivalent_delay", "omega_n", "below_1ms"),
        units={"equivalent_delay": "s", "omega_n": "Hz"},
        rows=tuple(table_rows),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Table 6: level against the real-machine reference (10.2.4)
# ---------------------------------------------------------------------------


class ReferenceAxis(str, Enum):
    """Which column of 10.2.4 the real-machine range is stated against.

    Not uniform across the table, and the difference matters: D1's "1-2 mm" is
    an end-effector deviation and belongs next to the observable, while D4's
    "under 1 arcmin" is a backlash angle and belongs next to the injected
    parameter. Comparing a range against the wrong column would answer a
    different question in the same cell.
    """

    PARAMETER = "injection_parameter"
    OBSERVABLE = "observable"


@dataclass(frozen=True)
class RealMachineReference:
    """One row of 10.2.4's real-machine column, transcribed with its axis."""

    low: float
    high: float
    unit: str
    axis: ReferenceAxis
    parameter_key: str | None
    source: str

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"reference range [{self.low}, {self.high}] is inverted")
        if (self.axis is ReferenceAxis.PARAMETER) != (self.parameter_key is not None):
            raise ValueError(
                "a reference stated against the injection parameter has to name which "
                "parameter, and one stated against the observable must not"
            )


# 10.2.4's table, transcribed. Values are specification, not measurements
# (contract 3.1): the proposal states them, and the verdict column below is
# computed against the measured level rather than transcribed with them.
REAL_MACHINE_REFERENCE: dict[str, RealMachineReference] = {
    "D1": RealMachineReference(
        1e-3, 2e-3, "m", ReferenceAxis.OBSERVABLE, None,
        "10.2.4: end-effector equivalent deviation, 1-2 mm",
    ),
    "D2": RealMachineReference(
        20e-3, 100e-3, "s", ReferenceAxis.OBSERVABLE, None,
        "10.2.4: 20-100 ms over a whole teleoperation chain",
    ),
    "D3a": RealMachineReference(
        0.5e-3, 2e-3, "m", ReferenceAxis.OBSERVABLE, None,
        "10.2.4: rated-load end-effector droop, 0.5-2 mm",
    ),
    "D3b": RealMachineReference(
        0.5e-3, 2e-3, "m", ReferenceAxis.OBSERVABLE, None,
        "10.2.4: rated-load end-effector droop, 0.5-2 mm",
    ),
    # 10.2.4 states D4's reference as an angle and the ladder is stored in
    # radians (`defects.params.DEG`), so the range is converted rather than the
    # level: converting the level would put a second version of 9.2.10's table
    # in this module.
    "D4hi": RealMachineReference(
        0.0, DEG / 60.0, "rad", ReferenceAxis.PARAMETER, "backlash",
        "10.2.4: under 1 arcmin of backlash, i.e. 0.017 degrees",
    ),
    "D4lo": RealMachineReference(
        0.0, DEG / 60.0, "rad", ReferenceAxis.PARAMETER, "backlash",
        "10.2.4: under 1 arcmin of backlash, i.e. 0.017 degrees",
    ),
    "D5": RealMachineReference(
        0.005, 0.02, "share of tau_max", ReferenceAxis.PARAMETER, "w_over_tau_max",
        "10.2.4: dead band 0.5-2% of the torque limit",
    ),
    "D6": RealMachineReference(
        0.1, 0.5, "N m", ReferenceAxis.PARAMETER, "delta_f",
        "10.2.4: friction asymmetry 0.1-0.5 N m",
    ),
}


def _level_parameter(spec: DefectSpec, level_index: int, key: str) -> float:
    params = spec.level(level_index).params
    if key not in params:
        raise KeyError(
            f"{spec.code} L{level_index} has parameters {sorted(params)} and 10.2.4 states "
            f"its real-machine range against {key!r}"
        )
    return float(params[key])


def build_level_reference_table(
    defect_codes: Sequence[str], *, family: str, lock: LockFile
) -> CalibrationTable:
    """10.2.5 table 6: injected parameter, observable, real-machine reference.

    The observable is the per-family measurement of 9.2.10 (`defect.observable`)
    and is read from the lock file, because one injection parameter means six
    different doses across six families. D8 is the exception 9.2.10 does not
    cover -- its parameter is a dimensionless torque ratio and is its own
    observable on every family -- and `DefectSpec.observable_is_parameter`
    records that.

    A defect with no row in 10.2.4 gets a row here with an empty reference and
    `reference_status` saying so. That is the case for D8, which was priority C
    when 10.2.4 was written and which 9.2.6 later moved into the core six; a
    range invented for it would be the one number in this table that no section
    of the proposal stands behind.
    """
    _require(defect_codes, "level_real_machine_reference", "defect codes", "10.2.4")
    obs_decl = REQUIRED_MEASUREMENTS["defect.observable.{defect}.{level}.{family}"]
    rows = []
    for code in defect_codes:
        spec = defect(code)
        reference = REAL_MACHINE_REFERENCE.get(code)
        for level in spec.levels:
            if spec.observable_is_parameter:
                (observable,) = level.params.values()
            else:
                observable = float(
                    lock.get(obs_decl, defect=code, level=str(level.index), family=family)
                )
            if reference is None:
                compared: float | None = None
                within: bool | None = None
                status = (
                    "10.2.4 states no real-machine range for this defect; it was priority C "
                    "when that table was written and 9.2.6 moved it into the core six"
                )
            else:
                compared = (
                    observable
                    if reference.axis is ReferenceAxis.OBSERVABLE
                    else _level_parameter(spec, level.index, str(reference.parameter_key))
                )
                within = reference.low <= compared <= reference.high
                status = reference.source
            rows.append(
                {
                    "defect": code,
                    "level": level.index,
                    "family": family,
                    "parameters": dict(level.params),
                    "observable": observable,
                    "observable_unit": level.observable_unit,
                    "reference_low": None if reference is None else reference.low,
                    "reference_high": None if reference is None else reference.high,
                    "reference_unit": None if reference is None else reference.unit,
                    "reference_axis": None if reference is None else reference.axis.value,
                    "compared_value": compared,
                    "within_real_machine_scale": within,
                    "reference_status": status,
                }
            )
    return CalibrationTable(
        name="level_real_machine_reference",
        title="level against the real-machine reference magnitude",
        proposal_ref="10.2.4, 9.2.10, 10.2.5",
        columns=(
            "defect", "level", "family", "parameters", "observable", "observable_unit",
            "reference_low", "reference_high", "reference_unit", "reference_axis",
            "compared_value", "within_real_machine_scale", "reference_status",
        ),
        units={},
        rows=tuple(rows),
        notes=(
            "9.2.10 is the single source of the level ladder; this table quotes it and "
            "never restates a level",
            "the observable is measured per family, because one parameter reaches the end "
            "effector through J, Lambda_eff and k_p (9.2.10)",
        ),
    )


# ---------------------------------------------------------------------------
# Table 7: the three-way feasibility check (10.2.3, 10.8.1's G3)
# ---------------------------------------------------------------------------


def _intermediate_band(cell: DirectionalClearanceCell) -> tuple[float, float]:
    """(min(cl_+, cl_-), max(cl_+, cl_-)], proposition 2(b)'s band."""
    return min(cell.cl_plus_m, cell.cl_minus_m), max(cell.cl_plus_m, cell.cl_minus_m)


def build_feasibility_table(
    goldilocks: Sequence[GoldilocksCell],
    clearance: Sequence[DirectionalClearanceCell],
    kappa: Sequence[KappaCell],
    *,
    family: str,
) -> CalibrationTable:
    """10.2.5 table 7: the per-level three-way feasible set.

    One row per (defect, level, task, geometry variant) that the clearance sweep
    covers, with the three conditions of 10.2.3 evaluated side by side:

    1. P0 bite -- uncompensated success rate in [25, 65];
    2. ceiling -- the clean twin at or above 90;
    3. intermediate band -- min(cl_+, cl_-) < ||c_delta(x*)|| <= max(cl_+, cl_-).

    `family` is required and is not an axis of the quadruple: 10.2.3 writes the
    check over (defect, level, task, geometry variant), while the bite and the
    ceiling are per family. Fixing the carrier family makes the quadruple
    well-defined and puts the choice in the table rather than in a silent
    reduction over families.
    """
    _require(clearance, "t8_three_way_feasibility", "clearance queries",
             "10.2.3 quantity 2, the stepped collision query along v")
    bite_by_key = {
        (c.defect, c.level, c.task): c for c in goldilocks if c.family == family
    }
    kappa_by_key = {
        (k.defect, k.level, k.task, k.geom_variant): k for k in kappa if k.family == family
    }

    rows = []
    for cell in clearance:
        bite_cell = bite_by_key.get((cell.defect, cell.level, cell.task))
        if bite_cell is None:
            raise MissingCalibrationInput(
                "t8_three_way_feasibility",
                f"no Goldilocks cell for ({cell.defect}, L{cell.level}, {cell.task}) on "
                f"{family}, so conditions 1 and 2 cannot be evaluated there",
                "the 10.2.2 uncompensated-tracker sweep",
            )
        kappa_cell = kappa_by_key.get((cell.defect, cell.level, cell.task, cell.geom_variant))
        if kappa_cell is None:
            raise MissingCalibrationInput(
                "t8_three_way_feasibility",
                f"no terminal-phase ||c_delta(x*)|| for ({cell.defect}, L{cell.level}, "
                f"{cell.task}, {cell.geom_variant}) on {family}, so the intermediate band "
                "cannot be evaluated there",
                "the 10.2.3 kappa sweep",
            )
        low, high = _intermediate_band(cell)
        saturated_low = (
            cell.saturated_plus if cell.cl_plus_m <= cell.cl_minus_m else cell.saturated_minus
        )
        if saturated_low:
            raise ValueError(
                f"({cell.defect}, L{cell.level}, {cell.task}, {cell.geom_variant}): the "
                "sweep that produced the *lower* end of the intermediate band saturated, "
                "so that end is a lower bound and the band condition is not decidable. "
                "Re-run measure_clearance with more steps rather than reading the bound "
                "as a clearance"
            )
        c_norm = kappa_cell.c_norm_m
        in_band = low < c_norm <= high
        rows.append(
            {
                "defect": cell.defect,
                "level": cell.level,
                "task": cell.task,
                "geom_variant": cell.geom_variant,
                "family": family,
                "success_rate": bite_cell.success_rate_pct,
                "clean_twin_rate": bite_cell.clean_twin_rate_pct,
                "c_norm": c_norm,
                "band_low": low,
                "band_high": high,
                "bites": bite_cell.bites,
                "ceiling_holds": bite_cell.ceiling_holds,
                "in_band": in_band,
                "angle_admits_sign_prediction": cell.enters_sign_prediction,
                "feasible": bite_cell.bites and bite_cell.ceiling_holds and in_band,
            }
        )
    feasible = [r for r in rows if r["feasible"]]
    variants = sorted({str(r["geom_variant"]) for r in rows})
    empty_variants = sorted(
        {v for v in variants if not any(r["feasible"] for r in rows if r["geom_variant"] == v)}
    )
    notes = [
        "conditions: P0 bite in [25, 65], clean twin at or above 90, and "
        "min(cl_+, cl_-) < ||c_delta(x*)|| <= max(cl_+, cl_-) (10.2.3)",
        f"feasible quadruples: {len(feasible)} of {len(rows)}",
        f"geometry variants with an empty feasible set: {empty_variants}",
    ]
    if empty_variants == variants:
        notes.append(
            "every geometry variant's feasible set is empty, which is exit F8: T8 is not "
            "measurable and proposition 2 falls back to (a)'s type dichotomy plus the "
            "controlled sign flip as a qualitative observation (10.9.3)"
        )
    return CalibrationTable(
        name="t8_three_way_feasibility",
        title="per-level three-way feasibility check",
        proposal_ref="10.2.3, 10.2.5, 10.8.1 G3",
        columns=(
            "defect", "level", "task", "geom_variant", "family", "success_rate",
            "clean_twin_rate", "c_norm", "band_low", "band_high", "bites", "ceiling_holds",
            "in_band", "angle_admits_sign_prediction", "feasible",
        ),
        units={
            "success_rate": "%", "clean_twin_rate": "%", "c_norm": "m",
            "band_low": "m", "band_high": "m",
        },
        rows=tuple(rows),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Table 8: shadow_joint_offset (10.2.3 quantity 5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShadowJointOffsetCell:
    """The clean twin carrying the D4 shadow joint against the plain clean body."""

    family: str
    task: str
    tracking_rms_m_with_shadow: float
    tracking_rms_m_plain: float
    peak_cfrc_ext_n_with_shadow: float
    peak_cfrc_ext_n_plain: float

    @property
    def tracking_rms_offset_m(self) -> float:
        return self.tracking_rms_m_with_shadow - self.tracking_rms_m_plain

    @property
    def peak_cfrc_ext_offset_n(self) -> float:
        return self.peak_cfrc_ext_n_with_shadow - self.peak_cfrc_ext_n_plain


def build_shadow_joint_offset_table(
    cells: Sequence[ShadowJointOffsetCell],
) -> CalibrationTable:
    """10.2.5 table 8: the offset the D4 high-fidelity caliber leaves behind.

    Lambda is barely affected -- both sides of a twin pair carry the shadow joint
    and the mismatch cancels in the difference -- but T8 and the safety endpoint
    are absolute quantities and inherit it directly. The comparison is therefore
    against the safety endpoint's own MDE from 10.7.4, which is where the
    consequence 10.2.5 preregisters is triggered.
    """
    _require(cells, "shadow_joint_offset", "paired clean-twin rollouts",
             "the 10.2.3 shadow-joint comparison on the clean bodies")
    safety_mde = float(ENDPOINT_SPECS["peak_cfrc_ext"].mde().value)
    rows = [
        {
            "family": c.family,
            "task": c.task,
            "tracking_rms_offset": c.tracking_rms_offset_m,
            "peak_cfrc_ext_offset": c.peak_cfrc_ext_offset_n,
            "safety_mde": safety_mde,
            "exceeds_safety_mde": abs(c.peak_cfrc_ext_offset_n) > safety_mde,
        }
        for c in cells
    ]
    notes = [
        f"the peak cfrc_ext offset is read against the safety endpoint's MDE, "
        f"{safety_mde:g} N (10.7.4)",
        "this offset is unrelated to Lambda and the report has to say so (10.2.5)",
    ]
    if any(row["exceeds_safety_mde"] for row in rows):
        notes.append(
            "the offset exceeds the safety endpoint's MDE somewhere, so 10.2.5's frozen "
            "choice fires: T8 either moves to the D4 low-fidelity caliber, accepting that "
            "there is no true meshing impact, or its safety endpoint becomes the "
            "failure-type profile"
        )
    return CalibrationTable(
        name="shadow_joint_offset",
        title="shadow-joint clean twin minus plain clean body",
        proposal_ref="10.2.3 quantity 5, 9.2.4, 10.2.5",
        columns=(
            "family", "task", "tracking_rms_offset", "peak_cfrc_ext_offset", "safety_mde",
            "exceeds_safety_mde",
        ),
        units={"tracking_rms_offset": "m", "peak_cfrc_ext_offset": "N", "safety_mde": "N"},
        rows=tuple(rows),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Table 9: the D5 dead-zone duty (10.2.3 quantity 6)
# ---------------------------------------------------------------------------


class D5Caliber(str, Enum):
    """The two calibers 10.2.5 allows for the dead-zone duty share."""

    TOTAL_CTRL = "total_ctrl"
    TASK_ONLY = "task_only"


@dataclass(frozen=True)
class DeadZoneDutyCell:
    """Share of time one joint of one family spends inside D5's dead zone."""

    family: str
    joint: int
    duty: float
    caliber: D5Caliber

    def __post_init__(self) -> None:
        if not 0.0 <= self.duty <= 1.0:
            raise ValueError(f"a duty share must lie in [0, 1], got {self.duty}")


def build_d5_dead_zone_duty_table(
    cells: Sequence[DeadZoneDutyCell], *, families: Sequence[str]
) -> CalibrationTable:
    """10.2.5 table 9: per (family, joint) dead-zone duty, on one caliber.

    The caliber has to be the same on all six families or the six cells are not
    the same physical quantity and gamma_{D5,r} is not comparable across them.
    That is a refusal rather than a note, because a mixed-caliber table reads
    exactly like a unified one.
    """
    _require(cells, "d5_dead_zone_duty", "per-joint duty measurements",
             "the 10.2.3 dead-zone occupancy sweep")
    calibers = {c.caliber for c in cells}
    if len(calibers) > 1:
        raise ValueError(
            f"the dead-zone duty is reported under calibers {sorted(c.value for c in calibers)}. "
            "10.2.5 freezes one caliber -- total_ctrl or task_only -- across all six "
            "families, because otherwise the six cells are not the same physical quantity "
            "and gamma_{D5,r} cannot be compared between them"
        )
    caliber = calibers.pop()
    covered = {c.family for c in cells}
    missing = [f for f in families if f not in covered]
    if missing:
        raise MissingCalibrationInput(
            "d5_dead_zone_duty",
            f"no joints measured on famil(y/ies) {missing}",
            "the 10.2.3 dead-zone occupancy sweep",
        )
    rows = [
        {
            "family": c.family,
            "joint": c.joint,
            "duty": c.duty,
            "caliber": caliber.value,
            "above_floor": c.duty >= D5_DUTY_FLOOR,
        }
        for c in cells
    ]
    starved = sorted(
        f for f in covered
        if all(c.duty < D5_DUTY_FLOOR for c in cells if c.family == f)
    )
    notes = [
        f"one caliber across all six families: {caliber.value} (10.2.5)",
        f"floor {D5_DUTY_FLOOR:.0%} of the time inside the dead zone",
        f"families with every joint below the floor: {starved}",
    ]
    if starved:
        notes.append(
            "10.2.5's preregistered consequence fires: D5 leaves the primary test and a "
            "seventh defect takes its place -- D9 in an analytically writable form first, "
            "otherwise D6 back on the qfrc_applied caliber with its 0.64-3.52 N m offset "
            "annotated"
        )
    return CalibrationTable(
        name="d5_dead_zone_duty",
        title="per (embodiment, joint) dead-zone duty share",
        proposal_ref="10.2.3 quantity 6, 10.2.5",
        columns=("family", "joint", "duty", "caliber", "above_floor"),
        units={"duty": "share"},
        rows=tuple(rows),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# The nine together
# ---------------------------------------------------------------------------


def build_p0_tables(
    *,
    goldilocks: Sequence[GoldilocksCell] | None = None,
    kappa: Sequence[KappaCell] | None = None,
    clearance: Sequence[DirectionalClearanceCell] | None = None,
    force_curve: Sequence[ForceCalibrationPoint] | None = None,
    d3_delays: Sequence[D3DelayRow] | None = None,
    shadow_joint: Sequence[ShadowJointOffsetCell] | None = None,
    dead_zone: Sequence[DeadZoneDutyCell] | None = None,
    defect_codes: Sequence[str],
    families: Sequence[str],
    t8_carrier_family: str,
    lock: LockFile,
) -> dict[str, CalibrationTable]:
    """All nine tables of 10.2.5, or a refusal naming every input that is absent.

    The nine are published together, so a partial set is reported as missing
    rather than returned. Each builder is public and can be called on its own
    while a calibration run is still in progress. This assembler proves only
    that the descriptive tables are complete; it does not select Goldilocks
    levels or execute the defect-count exit while that decision rule is absent.
    """
    absent = [
        name
        for name, value in (
            ("goldilocks", goldilocks),
            ("kappa", kappa),
            ("clearance", clearance),
            ("force_curve", force_curve),
            ("d3_delays", d3_delays),
            ("shadow_joint", shadow_joint),
            ("dead_zone", dead_zone),
        )
        if not value
    ]
    if absent:
        raise MissingCalibrationInput(
            "p0 (all nine)",
            f"these inputs were not supplied: {absent}",
            "the W1 calibration run of 10.2.1, in the order that section fixes",
        )
    tables = {
        "goldilocks_success_rate": build_goldilocks_table(goldilocks),
        "kappa": build_kappa_table(kappa, lock=lock),
        "directional_clearance": build_directional_clearance_table(clearance),
        "force_calibration": build_force_calibration_table(force_curve),
        "d3_equivalent_delay": build_d3_equivalent_delay_table(d3_delays),
        "level_real_machine_reference": build_level_reference_table(
            defect_codes, family=t8_carrier_family, lock=lock
        ),
        "t8_three_way_feasibility": build_feasibility_table(
            goldilocks, clearance, kappa, family=t8_carrier_family
        ),
        "shadow_joint_offset": build_shadow_joint_offset_table(shadow_joint),
        "d5_dead_zone_duty": build_d5_dead_zone_duty_table(dead_zone, families=families),
    }
    if tuple(tables) != P0_TABLE_NAMES:
        raise AssertionError(
            f"10.2.5 lists nine tables in a fixed order {P0_TABLE_NAMES}; this assembler "
            f"produced {tuple(tables)}"
        )
    return tables
