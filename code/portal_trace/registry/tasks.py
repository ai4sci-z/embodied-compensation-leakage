"""The eight tasks of 9.5.2 and the geometry variants of 9.5.5.

Ten environments are registered for those eight tasks. Six are core and two are
held out; the remaining two are the unmodified robosuite nut environments that
9.5.4 compiles the tight-fit core pair out of, kept because P0 measures the
native clearance on them. `TaskRole` says which is which.

Environment identifiers are robosuite 1.5.2 natives. `horizon` is overridden to
400 everywhere (9.5.2): robosuite defaults to 1000, and keeping the default
would move the arithmetic base of the entire budget chapter.

What is *not* here is any tolerance value. tol(tau) is the denominator of kappa
and 9.5.4 requires it to come out of `scripts/compute_task_tol.py`; this module
only says which bodies the script should measure and along which direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "TolKind",
    "TolRule",
    "TerminalPoseRule",
    "TaskRole",
    "GeomVariant",
    "TaskSpec",
    "TASKS",
    "task",
    "CORE_TASKS",
    "HELDOUT_TASKS",
    "BASE_TASKS",
    "HORIZON",
    "CONTROL_FREQ",
    "SIMULATION_TIMESTEP",
]

# 9.1.1 / 9.6.1. These three fix the 500 Hz injection hook and the budget base.
HORIZON = 400
CONTROL_FREQ = 20
SIMULATION_TIMESTEP = 0.002


class TolKind(str, Enum):
    """9.5.4: three semantics that must never share a column.

    A kappa computed against a marker half-width and one computed against a
    radial clearance are not comparable, so every kappa table carries this.
    """

    RADIAL_CLEARANCE = "radial_clearance"
    MARKER_HALFWIDTH = "marker_halfwidth"
    GRASP_WINDOW = "grasp_window"


class TaskRole(str, Enum):
    """Which of 9.5.2's three standings a registered environment holds.

    `CORE` and `HELDOUT` are the eight tasks of 9.5.2: six that every training
    arm sees and two that none does. `BASE` is neither. 9.5.4 compiles the two
    nut tasks into tight-fit environments under their own registered names, and
    the unmodified robosuite originals stay in the registry because P0 measures
    the native clearance on them and 9.5.4's comparability note is stated
    against them. They are never trained on and never counted among the eight,
    so giving them `CORE` would put eight tasks in a six-task set and silently
    change E8's 3 + 3 split and (A4)'s six-fold leave-one-task-out.
    """

    CORE = "core"
    HELDOUT = "heldout"
    BASE = "base"


class TerminalPoseRule(str, Enum):
    """How compute_task_tol reaches x*.

    `AXIS_COINCIDENCE` applies when the terminal pose is defined by the geometry
    itself: a nut over its peg, a cube centred on the one below it. The script
    can then impose it analytically and needs nothing else.

    `FROM_REFERENCE` applies when it is not, and 9.5.4 says so -- x* comes from
    the end of the reference trajectory. ToolHang's frame does not sit with its
    origin on the stand's origin, and a can dropped into a bin has no canonical
    lateral position at all. For these the script loads the reference and refuses
    to guess.
    """

    AXIS_COINCIDENCE = "axis_coincidence"
    FROM_REFERENCE = "from_reference"


class TolRule(str, Enum):
    """How tol is read off the geometry.

    `CLEARANCE` is the 9.5.4 definition: hole inner radius minus inserted outer
    radius in the normal plane. `CENTROID_CONTAINMENT` is what Stack needs --
    9.5.4 derives its 25 mm from cubeA's centroid having to project inside
    cubeB's top face, so the inserted part contributes a point rather than an
    outline. They are both radial clearances in metres and both live in the
    kappa table, but conflating the formulas would turn Stack's 25 mm into 5 mm
    and destroy the low-kappa control.
    """

    CLEARANCE = "clearance"
    CENTROID_CONTAINMENT = "centroid_containment"


@dataclass(frozen=True)
class GeomVariant:
    """A geometry level: G0 baseline, G1 recessed well, G2 well plus baffle (9.5.5).

    `mjcf_edits` describes the change declaratively; `registry.geometry` applies
    it. The well depth d and baffle height h are deliberately not defaults --
    9.5.5 derives them backwards from the three-way feasibility check, so P0 has
    to measure cl_pm and ||c_delta(x*)|| before they can be set.
    """

    name: str
    description: str
    mjcf_edits: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    env_name: str
    env_kwargs: dict[str, Any]
    tol_kind: TolKind
    role: TaskRole
    dominant_regimes: tuple[str, ...]
    approach_contact: bool
    # Only set for insertion-type tasks: which body goes into which, and along
    # what world direction at the terminal pose.
    inserted_body: str | None = None
    hole_body: str | None = None
    insertion_axis: tuple[float, float, float] | None = None
    # Which of the two bodies is free at the terminal pose. compute_task_tol has
    # to place it on the other one's axis before measuring: at reset the nut sits
    # somewhere else on the table, and measuring there answers a different
    # question.
    moving_body: str | None = None
    terminal_yaw_rad: float = 0.0
    tol_rule: TolRule = TolRule.CLEARANCE
    terminal_pose_rule: TerminalPoseRule = TerminalPoseRule.AXIS_COINCIDENCE
    geom_variants: tuple[GeomVariant, ...] = ()
    tight_variant_of: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        # G0 is the identity edit, so every task has it whether or not it was
        # listed. Leaving it implicit made the lock key `task.tol.<task>.G0`
        # resolvable for tasks whose `geom_variants` was empty while
        # `geom_variant(task, "G0")` raised on the same pair, which is a
        # disagreement between two readings of the same registry.
        if not any(v.name == "G0" for v in self.geom_variants):
            object.__setattr__(self, "geom_variants", (_G0, *self.geom_variants))
        if self.tol_kind is TolKind.RADIAL_CLEARANCE:
            if not (self.inserted_body and self.hole_body and self.insertion_axis):
                raise ValueError(
                    f"{self.task_id}: a radial_clearance task needs inserted_body, "
                    "hole_body and insertion_axis for compute_task_tol"
                )
            if self.moving_body not in (self.inserted_body, self.hole_body):
                raise ValueError(
                    f"{self.task_id}: moving_body must be one of the two measured "
                    f"bodies, got {self.moving_body!r}"
                )
        if self.env_kwargs.get("horizon", HORIZON) != HORIZON:
            raise ValueError(f"{self.task_id}: horizon is fixed at {HORIZON} (9.5.2)")


_Z_DOWN = (0.0, 0.0, -1.0)

# 9.5.5: the peg-top lead-in chamfer, and the well/baffle of G1/G2. Only the
# chamfer parameters are fixed in the proposal (4 mm high, 0.01775 top half
# width); d and h carry no value here on purpose.
_LEAD_IN = {
    "kind": "peg_top_chamfer",
    "chamfer_height_m": 0.004,
    "top_halfwidth_m": 0.01775,
    "entry_lateral_allowance_m": 0.005,
}

# 9.5.5 sizes the well off the nut's outer half width, 0.04375 m, giving an
# inner half width of 0.04475. Measured against the real MJCF that is too narrow:
# robosuite's SquareNut is not just the ring, it carries a handle geom reaching to
# x = +0.07925 m from the body origin, and a 0.04475 well catches the handle on
# the outside of the slab. `registry.geometry` measured cl_+ = 0.00 mm for both G1
# and G2 at the proposal's number, so G1 loses its 2.5 mm and G2's sign never
# flips -- T8's controlled sign flip is dead at 0.04475. Sizing off the handle
# instead (0.07925 + 1 mm wall gap) reproduces the intent: cl_+ = 2.50 mm on both,
# and G2 comes out at A = -1.70 mm against G0's +199.90 mm.
_NUT_HANDLE_REACH_M = 0.07925
_WELL_WALL_GAP_M = 0.001
_WELL_INNER_HALFWIDTH_M = _NUT_HANDLE_REACH_M + _WELL_WALL_GAP_M       # 0.08025
_WELL_MOUTH_HALFWIDTH_M = _WELL_INNER_HALFWIDTH_M + 0.003              # 0.08325
_WELL_LEAD_IN_HEIGHT_M = 0.0015

_G0 = GeomVariant("G0", "baseline geometry, no MJCF change")
_G1 = GeomVariant(
    "G1",
    f"recessed well around peg1: four boxes, inner half width "
    f"{_WELL_INNER_HALFWIDTH_M} m, 45 deg lead-in {_WELL_LEAD_IN_HEIGHT_M * 1e3} mm "
    f"high with mouth half width {_WELL_MOUTH_HALFWIDTH_M} m. Depth d comes from P0",
    {"kind": "recessed_well", "inner_halfwidth_m": _WELL_INNER_HALFWIDTH_M,
     "mouth_halfwidth_m": _WELL_MOUTH_HALFWIDTH_M,
     "lead_in_height_m": _WELL_LEAD_IN_HEIGHT_M},
)
_G2 = GeomVariant(
    "G2",
    "G1 plus a baffle above the nut; height h comes from P0. This is the variant "
    "where sign(A) flips relative to G0 with the defect held fixed",
    {"kind": "recessed_well_with_baffle", "inner_halfwidth_m": _WELL_INNER_HALFWIDTH_M,
     "mouth_halfwidth_m": _WELL_MOUTH_HALFWIDTH_M,
     "lead_in_height_m": _WELL_LEAD_IN_HEIGHT_M,
     "baffle_halfsize_m": (0.06, 0.06, 0.005), "baffle_xy_m": (0.23, 0.1)},
)

_TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(
        task_id="NutAssemblySquare",
        env_name="NutAssembly",
        env_kwargs={"single_object_mode": 2, "nut_type": "square", "horizon": HORIZON},
        tol_kind=TolKind.RADIAL_CLEARANCE,
        role=TaskRole.BASE,
        dominant_regimes=("r1", "r4"),
        approach_contact=True,
        inserted_body="peg1",
        hole_body="SquareNut_main",
        insertion_axis=_Z_DOWN,
        moving_body="SquareNut_main",
        geom_variants=(_G0,),
        notes="native clearance 6.75 mm; P0 measures it here, no arm trains on it",
    ),
    TaskSpec(
        task_id="NutAssemblyRound",
        env_name="NutAssembly",
        env_kwargs={"single_object_mode": 2, "nut_type": "round", "horizon": HORIZON},
        tol_kind=TolKind.RADIAL_CLEARANCE,
        role=TaskRole.BASE,
        dominant_regimes=("r1", "r4"),
        approach_contact=True,
        inserted_body="peg2",
        hole_body="RoundNut_main",
        insertion_axis=_Z_DOWN,
        moving_body="RoundNut_main",
        geom_variants=(_G0,),
        notes="native clearance 11.2 mm; P0 measures it here, no arm trains on it",
    ),
    # 9.5.4 is emphatic that these are different environments, not a harder
    # setting of the same one: arena geoms and the success criterion both change,
    # so results on them are not comparable with published robomimic/MimicGen
    # numbers and the registered name has to differ.
    TaskSpec(
        task_id="NutAssemblySquareTight-v0",
        env_name="NutAssemblySquareTight-v0",
        env_kwargs={"single_object_mode": 2, "nut_type": "square", "horizon": HORIZON},
        tol_kind=TolKind.RADIAL_CLEARANCE,
        role=TaskRole.CORE,
        dominant_regimes=("r1", "r4"),
        approach_contact=True,
        inserted_body="peg1",
        hole_body="SquareNut_main",
        insertion_axis=_Z_DOWN,
        moving_body="SquareNut_main",
        geom_variants=(_G0, _G1, _G2),
        tight_variant_of="NutAssemblySquare",
        notes=(
            "peg1 size 0.016 0.016 0.1 -> 0.02175 0.02175 0.03, giving tol = 1.00 mm; "
            f"lead-in chamfer {_LEAD_IN}; success_fn overridden with the three "
            "conditions of 9.5.4 (radial < tol, descent >= 15 mm, r_reach < 0.6)"
        ),
    ),
    TaskSpec(
        task_id="NutAssemblyRoundTight-v0",
        env_name="NutAssemblyRoundTight-v0",
        env_kwargs={"single_object_mode": 2, "nut_type": "round", "horizon": HORIZON},
        tol_kind=TolKind.RADIAL_CLEARANCE,
        role=TaskRole.CORE,
        dominant_regimes=("r1", "r4"),
        approach_contact=True,
        inserted_body="peg2",
        hole_body="RoundNut_main",
        insertion_axis=_Z_DOWN,
        moving_body="RoundNut_main",
        geom_variants=(_G0,),
        tight_variant_of="NutAssemblyRound",
        notes="peg2 size 0.02 0.1 -> 0.0302 0.03, giving tol = 1.00 mm",
    ),
    TaskSpec(
        task_id="ToolHang",
        env_name="ToolHang",
        env_kwargs={"horizon": HORIZON},
        tol_kind=TolKind.RADIAL_CLEARANCE,
        role=TaskRole.CORE,
        dominant_regimes=("r1", "r4"),
        approach_contact=True,
        inserted_body="frame_root",
        hole_body="stand_root",
        insertion_axis=_Z_DOWN,
        moving_body="frame_root",
        terminal_pose_rule=TerminalPoseRule.FROM_REFERENCE,
        notes=(
            "two-stage insertion; tol is measured on the first stage, the frame "
            "into the stand's hole. Mesh geometry, so tol only comes from the "
            "script, and the frame body carries 112 geoms of which only the base "
            "post is in the fit -- the script's axial window selects them. Its "
            "median episode length is a W1 item: if it exceeds 400 steps the "
            "core/heldout roles swap (9.5.2 note 2)"
        ),
    ),
    TaskSpec(
        task_id="Stack",
        env_name="Stack",
        env_kwargs={"horizon": HORIZON},
        tol_kind=TolKind.RADIAL_CLEARANCE,
        role=TaskRole.CORE,
        dominant_regimes=("r3", "r4"),
        approach_contact=True,
        inserted_body="cubeA_main",
        hole_body="cubeB_main",
        insertion_axis=_Z_DOWN,
        moving_body="cubeA_main",
        tol_rule=TolRule.CENTROID_CONTAINMENT,
        notes=(
            "the low-kappa control. Stable placement needs cubeA's centroid inside "
            "cubeB's top face, so tol is cubeB's half width rather than a clearance "
            "between walls; the script reports it through the same support-function "
            "path with the roles of the two bodies swapped"
        ),
    ),
    TaskSpec(
        task_id="Door",
        env_name="Door",
        env_kwargs={"horizon": HORIZON, "use_latch": False},
        tol_kind=TolKind.GRASP_WINDOW,
        role=TaskRole.CORE,
        dominant_regimes=("r2",),
        approach_contact=False,
        notes=(
            "unlocked quasi-static load regime r2; tol is the handle grasp margin. "
            "The spring-latch subtask is disabled so Door isolates hinge loading "
            "rather than coupling grasp, latch release, and hinge motion"
        ),
    ),
    TaskSpec(
        task_id="Wipe",
        env_name="Wipe",
        env_kwargs={"horizon": HORIZON},
        tol_kind=TolKind.MARKER_HALFWIDTH,
        role=TaskRole.CORE,
        dominant_regimes=("r4",),
        approach_contact=False,
        notes="no crisp terminal pose; tol is the marker half width from line_width",
    ),
    TaskSpec(
        task_id="PickPlaceCan",
        env_name="PickPlace",
        env_kwargs={"single_object_mode": 2, "object_type": "can", "horizon": HORIZON},
        tol_kind=TolKind.RADIAL_CLEARANCE,
        role=TaskRole.HELDOUT,
        dominant_regimes=("r3",),
        approach_contact=False,
        inserted_body="Can_main",
        hole_body="bin2",
        insertion_axis=_Z_DOWN,
        moving_body="Can_main",
        terminal_pose_rule=TerminalPoseRule.FROM_REFERENCE,
        notes="held out; used only for the (A4) hold-out R^2",
    ),
    TaskSpec(
        task_id="Lift",
        env_name="Lift",
        env_kwargs={"horizon": HORIZON},
        tol_kind=TolKind.GRASP_WINDOW,
        role=TaskRole.HELDOUT,
        dominant_regimes=("r3",),
        approach_contact=False,
        notes="no fit at all, only a 0.04 m lift; the low end of defect-active occupancy",
    ),
)

TASKS: dict[str, TaskSpec] = {t.task_id: t for t in _TASKS}
CORE_TASKS: tuple[str, ...] = tuple(t.task_id for t in _TASKS if t.role is TaskRole.CORE)
HELDOUT_TASKS: tuple[str, ...] = tuple(t.task_id for t in _TASKS if t.role is TaskRole.HELDOUT)
BASE_TASKS: tuple[str, ...] = tuple(t.task_id for t in _TASKS if t.role is TaskRole.BASE)


def task(task_id: str) -> TaskSpec:
    try:
        return TASKS[task_id]
    except KeyError:
        raise KeyError(f"unknown task {task_id!r}; known: {sorted(TASKS)}") from None


if (len(CORE_TASKS), len(HELDOUT_TASKS), len(BASE_TASKS)) != (6, 2, 2):
    raise AssertionError(
        "9.5.2 lists eight tasks -- six core and two held out -- and 9.5.4 adds the "
        "two unmodified nut environments the tight variants were compiled from; "
        f"found {len(CORE_TASKS)} core, {len(HELDOUT_TASKS)} held out, "
        f"{len(BASE_TASKS)} base"
    )
_orphan_tight = [
    t.task_id
    for t in _TASKS
    if t.tight_variant_of is not None and t.tight_variant_of not in TASKS
]
if _orphan_tight:
    raise AssertionError(
        f"{_orphan_tight} name a parent environment the registry does not hold; "
        "9.5.4's comparability note is stated against that parent"
    )
