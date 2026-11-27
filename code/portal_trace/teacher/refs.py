"""Embodiment-independent task-space reference trajectories X_tau (9.4.2).

9.4.2 fixes the construction: task-space waypoints (pre-grasp pose, grasp pose,
lift, align, insert), quintic time parameterisation with zero endpoint velocity
and acceleration, sampled at 20 Hz to T = 400 steps. The reference contains no
joint quantity, so one X_tau per task drives all ten morphology families, which
is how the "same set of reference trajectories" requirement of 14.5 E-9 is
satisfied by construction rather than by bookkeeping.

Two decisions that the proposal leaves open and that are forced here.

**Frame.** 9.4.2 says the waypoints are given in the base frame. Taken
literally, the same numbers in each family's own base frame would place the
waypoints at different world positions, because the families do not share a base
pose -- the trajectory would stop being the same task. The reference is therefore
stored and resolved in the world (task) frame, which is what makes it shared, and
`osc` maps it into whichever base frame the tracker runs in. That map is a rigid
transform and carries no joint quantity, so the embodiment-independence claim is
unaffected.

**Anchors, not coordinates.** `seed_env` randomises object placement (9.1.6), so
a reference frozen as 400 numeric poses would only be valid for one reset. What
is frozen in `refs/<task>.json` is the *geometric recipe*: which task frames the
waypoints hang off, which offsets are taken from the compiled geometry, the
segment durations, the gripper command and the phase label. `resolve` evaluates
that recipe against the model and data of a reset environment. Nothing in the
recipe was obtained by recording a robot; every offset either comes from the task
geometry (peg top height, body bounding extents, the handle-to-centre vector of a
nut) or is a stated design parameter of the demonstrator, declared as a named
constant below.

Reference quality -- how often the tracker following X_tau actually succeeds, per
task and per family -- is a P0 measurement, not something this module asserts.
ILC improves tracking, not task success, and 9.6.7 item 4 requires unsuccessful
demonstrations to be kept and evaluated anyway.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from portal_trace.registry.tasks import CONTROL_FREQ, HORIZON, TASKS, TaskSpec
from portal_trace.registry.tasks import task as task_spec
from portal_trace.types import Phase

__all__ = [
    "TermKind",
    "Term",
    "YawKind",
    "YawRule",
    "Expansion",
    "Waypoint",
    "ReferenceSpec",
    "ReferenceTrajectory",
    "PHASE_ORDER",
    "TOOL_DOWN",
    "GRIPPER_OPEN",
    "GRIPPER_CLOSED",
    "REFS_DIR",
    "build_all_specs",
    "load_spec",
    "resolve",
    "terminal_pose_of",
    "write_specs",
    "main",
]

REFS_DIR = Path(__file__).resolve().parent / "refs"

# 9.6.1: the gripper channel of the 7-dim action, robosuite GRIP convention.
GRIPPER_OPEN = -1.0
GRIPPER_CLOSED = 1.0

# The tool frame with its approach axis along world -z. Determinant +1, so it is
# a rotation: x -> x, y -> -y, z -> -z.
TOOL_DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])

# Centre of the jaw-yaw fold band, i.e. the yaw the wrist is asked to stay near.
#
# A parallel jaw is symmetric under a half turn, so a grasp fixes the yaw only
# modulo pi and a representative has to be picked. The choice matters because the
# tightest wrist in the 9.5.1 family set (Panda, +-2.897 rad) has only about
# +-pi/2 of slack around the pose the workcell puts it in, so a band centred on
# the wrong yaw makes half of all object placements demand a rotation the arm
# cannot make -- which shows up as a saturated, blind rollout rather than as an
# error.
#
# pi/2 is a property of the workcell, not of any arm: 9.5.3 mounts every family
# at the -x edge of the table facing +x, and reaching along +x with the jaws
# across the reach direction puts the tool yaw at pi/2. Measured at reset, five
# of the six families sit within 0.04 rad of it (Panda 1.590, Sawyer 1.761, IIWA
# 1.584, Kinova3 1.557, UR5e 1.533) and XArm7, whose gripper is mounted a quarter
# turn round, sits at 0.016 with a +-6.283 rad wrist that covers the band twice
# over. So the datum stays a single task-frame constant and 9.4.6's ban on
# per-embodiment tuning is not touched.
YAW_DATUM_RAD = math.pi / 2.0

PHASE_ORDER: tuple[Phase, ...] = (Phase.APPROACH, Phase.GRASP, Phase.TRANSPORT, Phase.TERMINAL)
_PHASE_ORDINAL: dict[Phase, int] = {p: i for i, p in enumerate(PHASE_ORDER)}

# ---------------------------------------------------------------------------
# Design parameters of the demonstrator.
#
# These are not measurements and they are not quoted from the proposal: they are
# the free parameters of the reference path, chosen once and frozen so that every
# family sees the same trajectory. They are collected here rather than sprinkled
# through the task table so that a change to any of them is a one-line diff that
# invalidates every stored reference at once.
# ---------------------------------------------------------------------------

STANDOFF_M = 0.10          # pre-grasp height above a grasp site
TRANSPORT_CLEARANCE_M = 0.07   # height above the tallest obstacle while carrying
FINAL_APPROACH_M = 0.02    # height above the seated pose where the slow descent starts
SEATING_MARGIN_M = 0.004   # residual gap at the terminal pose, below every tol(tau)
# How far the carried part is pushed past the top of the feature it engages. The
# nut is 20 mm thick, so 20 mm below the peg top leaves it threaded on the peg
# with the hole fully engaged; it then slides to the table when the jaws open.
# Releasing above the peg instead would drop the nut from a height and turn the
# terminal contact -- the phase kappa is evaluated at (14.5 E-6) -- into an impact.
THREADING_DEPTH_M = 0.02
WIPE_CONTACT_DEPTH_M = 0.004   # keeps the wiper plane below the marker centres in contact
DOOR_STANDOFF_M = 0.09     # handle approach offset along the door's outward normal
DOOR_SWING_RAD = 0.35      # inside the 0.4 rad joint limit; success needs 0.3
# HookFrame's grasp box is 33.75 mm below the intersection site along the
# mount-to-intersection line (robosuite's fixed frame_height/grip_location).
TOOLHANG_GRIP_FRACTION = 0.03375 / 0.17625
# A shallow tilt keeps the frame hook between the stand walls while the tip
# remains within the native 50 mm insertion tolerance.  The stand's wall
# centre is 45 mm from its base geom; the native predicate intentionally uses
# that base geom as its distance reference.
TOOLHANG_FRAME_TILT_RAD = 0.0
# The stand's actual insertion pocket is centred between the opposing walls.
# Keep the frame tip at that physical support point; the base geom origin is
# offset by 45 mm in y and is only used for the insertion-distance predicate.
TOOLHANG_MOUNT_CENTER_Y_M = 0.045
TOOLHANG_FRAME_TRANSPORT_TILT_RAD = math.pi / 2.0
# The wrench hole-to-grip offset is horizontal in the compiled object frame;
# keeping the tool transport tilt at zero leaves the hole centre on the
# horizontal frame hook instead of rotating that offset into world z.
TOOLHANG_TOOL_TILT_RAD = -0.09
TOOLHANG_TOOL_COMMAND_YAW_OFFSET_RAD = 0.10
# Panda's finger pads sit about 20 mm above the EEF site in the ToolHang grasp
# orientation.  Command the wrist below the tool grip site so the pads close on
# the grip body instead of touching its upper edge.  The same rigid offset is
# carried into the placement waypoints below.
TOOLHANG_TOOL_GRASP_Z_OFFSET_M = -0.020
# Keep the hole just inside the hook while leaving the long tool handle and
# gripper outside the stand, so release can clear the frame without pushing it.
# Insert past the hole midpoint without driving the hook into the far wall.  A
# single-frame diagnostic at 0.6 increased the normal residual, so the frozen
# recipe uses the geometrically stable 0.4 fraction.
TOOLHANG_HOOK_INSERT_FRACTION = 0.4
# Apply the measured clean-rollout normal correction at the closest hook
# contact.  Its along-hook fraction was already admissible (0.527); removing
# the remaining 14.0 mm orthogonal residual centers the hole without changing
# insertion depth.  These are world-frame terminal-placement offsets, not
# reset-position coordinates.
TOOLHANG_TOOL_SEATING_OFFSET_M = -0.06754
TOOLHANG_TOOL_HOLE_SHIFT = (-0.01115, 0.00477)


class TermKind(str, Enum):
    """The closed set of geometry lookups a waypoint offset may perform.

    Closed for the same reason `defects.params.WritableField` is: a reference
    that could evaluate an arbitrary expression against the live environment
    would no longer be reviewable as a geometric recipe.
    """

    SITE_POS = "site_pos"            # world position of a site
    BODY_POS = "body_pos"            # world position of a body frame
    SITE_DELTA = "site_delta"        # site(a) - site(b), the grasp offset of a carried part
    BODY_TOP_Z = "body_top_z"        # highest point of a body's geoms, world z
    BODY_BBOX_HALF = "body_bbox_half"  # half of a body's geom bounding extent on one axis
    GEOM_POS = "geom_pos"              # world position of a compiled geom
    ARC = "arc"                      # site(a) rotated about the world z axis through body(b)
    JOINT_ARC = "joint_arc"          # site(a) rotated about the axis and anchor of joint(b)


@dataclass(frozen=True)
class Term:
    """One additive contribution to a waypoint position, in world coordinates.

    `rotate` marks a term that turns with the waypoint's `AlignRule`. The grasp
    offset of a carried part is the only thing that needs it: once the wrist
    rotates to square the part up with its receptacle, the vector from the grip
    to the part's centre has rotated by the same angle, and using the un-rotated
    one puts the part's centre off the target by the whole offset.
    """

    kind: TermKind
    a: str = ""
    b: str = ""
    mask: tuple[float, float, float] = (1.0, 1.0, 1.0)
    axis: int = 2
    scale: float = 1.0
    angle_rad: float = 0.0
    rotate: bool = False
    frame: str = "term"

    def __post_init__(self) -> None:
        if self.frame not in ("term", "command"):
            raise ValueError(
                f"term frame must be 'term' or 'command', got {self.frame!r}"
            )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind.value}
        if self.a:
            out["a"] = self.a
        if self.b:
            out["b"] = self.b
        out["mask"] = list(self.mask)
        if self.kind in (TermKind.BODY_BBOX_HALF,):
            out["axis"] = self.axis
        if self.scale != 1.0:
            out["scale"] = self.scale
        if self.angle_rad:
            out["angle_rad"] = self.angle_rad
        if self.rotate:
            out["rotate"] = True
        if self.frame != "term":
            out["frame"] = self.frame
        return out

    @classmethod
    def from_json(cls, blob: dict[str, Any]) -> Term:
        return cls(
            kind=TermKind(blob["kind"]),
            a=blob.get("a", ""),
            b=blob.get("b", ""),
            mask=tuple(float(v) for v in blob.get("mask", (1.0, 1.0, 1.0))),  # type: ignore[arg-type]
            axis=int(blob.get("axis", 2)),
            scale=float(blob.get("scale", 1.0)),
            angle_rad=float(blob.get("angle_rad", 0.0)),
            rotate=bool(blob.get("rotate", False)),
            frame=str(blob.get("frame", "term")),
        )


@dataclass(frozen=True)
class AlignRule:
    """Square a carried part up with the feature it is about to enter.

    Object placement is randomised in yaw and the receptacles are not, so a part
    picked up at its reset yaw arrives at the hole rotated. For a square nut over
    a square peg that is fatal well before the 6.75 mm radial tolerance is
    reached: the clearance a rotated square sees shrinks with the sine of the
    misalignment, and the descent jams a few millimetres in.

    `symmetry_fold` is the rotational symmetry order of the mating pair: 4 for a
    square nut on a square peg, 8 for robosuite's octagonal round nut. It leaves
    that many equivalent orientations to mate in, and which one is picked decides
    where the wrist has to be, because a part gripped away from its mating axis
    swings its grip point around as it turns. `grip_site` and `axis_site` are
    that lever arm -- the point the jaws hold and the point that goes over the
    feature -- and the branch is chosen to swing the grip toward the robot edge
    of the workcell.

    That is not a nicety either. The square nut's handle is 54 mm out from the
    hole and peg1 sits 844 mm from the base along the reach direction: with the
    handle swung outward the wrist is asked for 857 mm, past the 855 mm the
    longest arm in the family set has, and the elbow saturates against its limit
    for the rest of the episode. Swinging it inward instead costs nothing, since
    every branch mates equally well. Like `YAW_DATUM_RAD` the criterion is stated
    against the workcell (9.5.3 puts every base at the -x edge), not against any
    one arm, so 9.4.6 is untouched.

    `engaged` says whether this waypoint is one of the ones that holds the
    correction. It has to be carried by the rule rather than inferred from the
    gripper channel, because the waypoints before the grasp need the rule too:
    they have to agree with the transport ones on which branch was chosen, and a
    pre-grasp waypoint that did not know the rule could pick another and leave
    the wrist to unwind the difference while the part is in the jaws.
    """

    target_body: str
    source_body: str
    grip_site: str
    axis_site: str
    symmetry_fold: int
    engaged: bool = True

    def __post_init__(self) -> None:
        if self.symmetry_fold < 2:
            raise ValueError(
                f"symmetry_fold must be at least 2, got {self.symmetry_fold}; a "
                "one-fold part has a single mating orientation and pi/fold exceeds "
                "the wrist's own half turn, so no branch is guaranteed to exist"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "target_body": self.target_body,
            "source_body": self.source_body,
            "grip_site": self.grip_site,
            "axis_site": self.axis_site,
            "symmetry_fold": self.symmetry_fold,
            "engaged": self.engaged,
        }

    @classmethod
    def from_json(cls, blob: dict[str, Any]) -> AlignRule:
        return cls(
            target_body=blob["target_body"],
            source_body=blob["source_body"],
            grip_site=blob["grip_site"],
            axis_site=blob["axis_site"],
            symmetry_fold=int(blob["symmetry_fold"]),
            engaged=bool(blob["engaged"]),
        )


class YawKind(str, Enum):
    FIXED = "fixed"          # a constant world yaw
    BODY_YAW = "body_yaw"    # the yaw of a body frame
    RADIAL = "radial"        # the yaw of the vector from site(center) to site(anchor)


@dataclass(frozen=True)
class YawRule:
    kind: YawKind = YawKind.FIXED
    anchor: str = ""
    center: str = ""
    offset_rad: float = 0.0

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind.value}
        if self.anchor:
            out["anchor"] = self.anchor
        if self.center:
            out["center"] = self.center
        if self.offset_rad:
            out["offset_rad"] = self.offset_rad
        return out

    @classmethod
    def from_json(cls, blob: dict[str, Any]) -> YawRule:
        return cls(
            kind=YawKind(blob["kind"]),
            anchor=blob.get("anchor", ""),
            center=blob.get("center", ""),
            offset_rad=float(blob.get("offset_rad", 0.0)),
        )


@dataclass(frozen=True)
class Expansion:
    """A waypoint that stands for a run of same-shaped waypoints.

    `Wipe` has a marker path whose length is an environment parameter, so its
    reference cannot be a fixed waypoint list. The expansion names a site pattern
    and the resolver instantiates one waypoint per site that exists in the model,
    splitting the segment budget evenly.
    """

    site_pattern: str
    max_count: int

    def to_json(self) -> dict[str, Any]:
        return {"site_pattern": self.site_pattern, "max_count": self.max_count}

    @classmethod
    def from_json(cls, blob: dict[str, Any]) -> Expansion:
        return cls(site_pattern=blob["site_pattern"], max_count=int(blob["max_count"]))


@dataclass(frozen=True)
class Waypoint:
    """One task-space pose the reference passes through.

    Position is `anchor * anchor_mask + sum(terms) + Rz(yaw) @ const`. The
    constant is applied in the resolved yaw frame so that lateral standoffs
    (the door approach) follow the object; the pure-z constants that dominate the
    table are yaw-invariant either way.

    `steps` is the number of control steps of the segment *ending* at this
    waypoint. The segments of a spec sum to exactly `HORIZON`, which is what
    makes 9.6.1's fixed-length collection exact rather than an upper bound.
    """

    name: str
    phase: Phase
    steps: int
    anchor: str = ""
    anchor_kind: str = "site"       # "site" or "body"
    anchor_mask: tuple[float, float, float] = (1.0, 1.0, 1.0)
    terms: tuple[Term, ...] = ()
    const: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: YawRule = YawRule()
    align: AlignRule | None = None
    tilt_rad: float = 0.0           # rotation about the tool x axis, after TOOL_DOWN
    gripper: float = GRIPPER_OPEN
    rotate_from_yaw: YawRule | None = None
    # Position terms can be derived from a carried object's planned pose while
    # the commanded wrist pose follows a different object (ToolHang's tool is
    # placed from the already assembled frame).  These optional fields keep
    # that rigid transform explicit instead of baking reset-time coordinates
    # into a later waypoint.
    term_yaw: YawRule | None = None
    term_tilt_rad: float | None = None
    command_rotate_from_yaw: YawRule | None = None
    command_rotate_from_body: str | None = None
    command_rotate_from_site: str | None = None
    expand: Expansion | None = None
    derived_from: str = ""

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError(f"waypoint {self.name!r}: steps must be positive, got {self.steps}")
        if self.anchor_kind not in ("site", "body"):
            raise ValueError(f"waypoint {self.name!r}: anchor_kind must be site or body")
        if self.gripper not in (GRIPPER_OPEN, GRIPPER_CLOSED):
            raise ValueError(
                f"waypoint {self.name!r}: the gripper channel is binary in the "
                f"demonstrator, got {self.gripper}"
            )
        if sum(value is not None for value in (
            self.command_rotate_from_yaw,
            self.command_rotate_from_body,
            self.command_rotate_from_site,
        )) > 1:
            raise ValueError(
                f"waypoint {self.name!r}: command rotation may reference either "
                "a yaw rule or a body, not both"
            )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "phase": self.phase.value,
            "steps": self.steps,
            "gripper": self.gripper,
        }
        if self.anchor:
            out["anchor"] = self.anchor
            out["anchor_kind"] = self.anchor_kind
            out["anchor_mask"] = list(self.anchor_mask)
        if self.terms:
            out["terms"] = [t.to_json() for t in self.terms]
        if any(self.const):
            out["const"] = list(self.const)
        if self.yaw != YawRule():
            out["yaw"] = self.yaw.to_json()
        if self.align is not None:
            out["align"] = self.align.to_json()
        if self.rotate_from_yaw is not None:
            out["rotate_from_yaw"] = self.rotate_from_yaw.to_json()
        if self.term_yaw is not None:
            out["term_yaw"] = self.term_yaw.to_json()
        if self.term_tilt_rad is not None:
            out["term_tilt_rad"] = self.term_tilt_rad
        if self.command_rotate_from_yaw is not None:
            out["command_rotate_from_yaw"] = self.command_rotate_from_yaw.to_json()
        if self.command_rotate_from_body is not None:
            out["command_rotate_from_body"] = self.command_rotate_from_body
        if self.command_rotate_from_site is not None:
            out["command_rotate_from_site"] = self.command_rotate_from_site
        if self.tilt_rad:
            out["tilt_rad"] = self.tilt_rad
        if self.expand is not None:
            out["expand"] = self.expand.to_json()
        if self.derived_from:
            out["derived_from"] = self.derived_from
        return out

    @classmethod
    def from_json(cls, blob: dict[str, Any]) -> Waypoint:
        return cls(
            name=blob["name"],
            phase=Phase(blob["phase"]),
            steps=int(blob["steps"]),
            anchor=blob.get("anchor", ""),
            anchor_kind=blob.get("anchor_kind", "site"),
            anchor_mask=tuple(float(v) for v in blob.get("anchor_mask", (1.0, 1.0, 1.0))),  # type: ignore[arg-type]
            terms=tuple(Term.from_json(t) for t in blob.get("terms", ())),
            const=tuple(float(v) for v in blob.get("const", (0.0, 0.0, 0.0))),  # type: ignore[arg-type]
            yaw=YawRule.from_json(blob["yaw"]) if "yaw" in blob else YawRule(),
            align=AlignRule.from_json(blob["align"]) if "align" in blob else None,
            tilt_rad=float(blob.get("tilt_rad", 0.0)),
            gripper=float(blob["gripper"]),
            rotate_from_yaw=(
                YawRule.from_json(blob["rotate_from_yaw"])
                if "rotate_from_yaw" in blob else None
            ),
            term_yaw=(
                YawRule.from_json(blob["term_yaw"])
                if "term_yaw" in blob else None
            ),
            term_tilt_rad=(
                float(blob["term_tilt_rad"])
                if "term_tilt_rad" in blob else None
            ),
            command_rotate_from_yaw=(
                YawRule.from_json(blob["command_rotate_from_yaw"])
                if "command_rotate_from_yaw" in blob else None
            ),
            command_rotate_from_body=blob.get("command_rotate_from_body"),
            command_rotate_from_site=blob.get("command_rotate_from_site"),
            expand=Expansion.from_json(blob["expand"]) if "expand" in blob else None,
            derived_from=blob.get("derived_from", ""),
        )


@dataclass(frozen=True)
class ReferenceSpec:
    """The frozen recipe for one task's X_tau."""

    task_id: str
    waypoints: tuple[Waypoint, ...]
    horizon: int = HORIZON
    control_freq: int = CONTROL_FREQ
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.waypoints:
            raise ValueError(f"{self.task_id}: a reference with no waypoints is not a reference")
        total = sum(w.steps for w in self.waypoints)
        if total != self.horizon:
            raise ValueError(
                f"{self.task_id}: segments sum to {total} control steps, horizon is "
                f"{self.horizon}. 9.6.1 makes T = 400 exact, so this is an error "
                "rather than something to pad"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "horizon": self.horizon,
            "control_freq": self.control_freq,
            "notes": self.notes,
            "waypoints": [w.to_json() for w in self.waypoints],
        }

    @classmethod
    def from_json(cls, blob: dict[str, Any]) -> ReferenceSpec:
        return cls(
            task_id=blob["task_id"],
            waypoints=tuple(Waypoint.from_json(w) for w in blob["waypoints"]),
            horizon=int(blob["horizon"]),
            control_freq=int(blob["control_freq"]),
            notes=blob.get("notes", ""),
        )


@dataclass(frozen=True)
class ReferenceTrajectory:
    """X_tau resolved against one reset state, in the world frame.

    `positions` and `rotations` are the commanded end-effector pose per control
    step; `gripper` is the seventh action channel; `phase` is the 9.6.4 phase
    label that the phase-split residual report keys on.
    """

    task_id: str
    positions: np.ndarray       # (T, 3) float64
    rotations: np.ndarray       # (T, 3, 3) float64
    gripper: np.ndarray         # (T,) float64
    phase: np.ndarray           # (T,) uint8, index into PHASE_ORDER
    waypoint_index: np.ndarray  # (T,) int32
    waypoint_names: tuple[str, ...]

    def __post_init__(self) -> None:
        T = len(self.positions)
        for name in ("rotations", "gripper", "phase", "waypoint_index"):
            arr = getattr(self, name)
            if len(arr) != T:
                raise ValueError(f"{name} has {len(arr)} frames, positions has {T}")

    @property
    def horizon(self) -> int:
        return len(self.positions)

    def phase_mask(self, phase: Phase) -> np.ndarray:
        return self.phase == _PHASE_ORDINAL[phase]


# ---------------------------------------------------------------------------
# Resolution against a live model
# ---------------------------------------------------------------------------


def _site_pos(sim: Any, name: str) -> np.ndarray:
    return np.asarray(sim.data.site_xpos[sim.model.site_name2id(name)], dtype=np.float64)


def _body_pos(sim: Any, name: str) -> np.ndarray:
    return np.asarray(sim.data.body_xpos[sim.model.body_name2id(name)], dtype=np.float64)


def _body_extent(sim: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    """World-axis-aligned bounds of a body's collision geoms, as (low, high).

    Built from `geom_aabb`, MuJoCo's per-geom local bounding box, rotated into
    the world by the geom's own frame. The alternative -- treating a geom's
    largest half-size as its reach on every axis -- is a bounding sphere, and it
    is wildly wrong for the flat and thin shapes that receptacles are made of: a
    bin wall 10 mm thick and 500 mm long reports a 250 mm vertical extent and
    lifts every clearance hung off that bin by a quarter of a metre.

    Visual geoms are excluded because a clearance is about what the arm can hit.
    Keeping them would also drag in decoration that reaches far outside the
    physical body -- the PickPlace bins hang 800 mm of cosmetic leg below the
    table.
    """
    model = sim.model
    bid = model.body_name2id(name)
    adr = int(model.body_geomadr[bid])
    ids = [
        g for g in range(adr, adr + int(model.body_geomnum[bid]))
        if model.geom_contype[g] or model.geom_conaffinity[g]
    ]
    if not ids:
        raise ValueError(
            f"body {name!r} carries no collision geom, so its extent is undefined; "
            "the reference cannot hang a clearance off it"
        )
    aabb = np.asarray(model._model.geom_aabb, dtype=np.float64)[ids].reshape(-1, 2, 3)
    rot = np.asarray(sim.data.geom_xmat[ids], dtype=np.float64).reshape(-1, 3, 3)
    centre = (
        np.asarray(sim.data.geom_xpos[ids], dtype=np.float64)
        + np.einsum("gij,gj->gi", rot, aabb[:, 0])
    )
    half = np.einsum("gij,gj->gi", np.abs(rot), aabb[:, 1])
    return (centre - half).min(axis=0), (centre + half).max(axis=0)


def _body_top_z(sim: Any, name: str) -> float:
    return float(_body_extent(sim, name)[1][2])


def _body_bbox_half(sim: Any, name: str, axis: int) -> float:
    lo, hi = _body_extent(sim, name)
    return 0.5 * float(hi[axis] - lo[axis])


def _eval_term(term: Term, sim: Any) -> np.ndarray:
    mask = np.asarray(term.mask, dtype=np.float64)
    if term.kind is TermKind.SITE_POS:
        vec = _site_pos(sim, term.a)
    elif term.kind is TermKind.BODY_POS:
        vec = _body_pos(sim, term.a)
    elif term.kind is TermKind.SITE_DELTA:
        vec = _site_pos(sim, term.a) - _site_pos(sim, term.b)
    elif term.kind is TermKind.BODY_TOP_Z:
        vec = np.array([0.0, 0.0, _body_top_z(sim, term.a)])
    elif term.kind is TermKind.BODY_BBOX_HALF:
        vec = np.zeros(3)
        vec[term.axis] = _body_bbox_half(sim, term.a, term.axis)
    elif term.kind is TermKind.GEOM_POS:
        vec = np.asarray(sim.data.geom_xpos[sim.model.geom_name2id(term.a)], dtype=np.float64)
    elif term.kind is TermKind.ARC:
        pivot = _body_pos(sim, term.b)
        rel = _site_pos(sim, term.a) - pivot
        c, s = math.cos(term.angle_rad), math.sin(term.angle_rad)
        rot = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1], rel[2]])
        vec = pivot + rot
    elif term.kind is TermKind.JOINT_ARC:
        vec = _rotate_about_joint(_site_pos(sim, term.a), term.b, term.angle_rad, sim)
    else:
        raise ValueError(f"unhandled term kind {term.kind!r}")
    return term.scale * mask * vec


def _rotate_about_joint(point: np.ndarray, joint: str, angle: float, sim: Any) -> np.ndarray:
    """Rotate a world point about a compiled joint's world anchor and axis."""
    model = sim.model._model
    jid = sim.model.joint_name2id(joint)
    bid = int(model.jnt_bodyid[jid])
    body_rot = np.asarray(sim.data.body_xmat[bid], dtype=np.float64).reshape(3, 3)
    pivot = np.asarray(sim.data.body_xpos[bid], dtype=np.float64) + body_rot @ model.jnt_pos[jid]
    axis = body_rot @ np.asarray(model.jnt_axis[jid], dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    rel = np.asarray(point, dtype=np.float64) - pivot
    c, s = math.cos(angle), math.sin(angle)
    rot = c * rel + s * np.cross(axis, rel) + (1.0 - c) * axis * np.dot(axis, rel)
    return pivot + rot


def _wrap_jaw_yaw(yaw: float) -> float:
    """Fold a yaw into the half-open band of width pi centred on `YAW_DATUM_RAD`.

    The two representations of a parallel-jaw grasp are the same grasp, so the
    fold costs nothing and buys the wrist headroom argued for at the datum.
    """
    return YAW_DATUM_RAD + _fold(yaw - YAW_DATUM_RAD, math.pi)


def _resolve_yaw(rule: YawRule, sim: Any) -> float:
    if rule.kind is YawKind.FIXED:
        raw = rule.offset_rad
    elif rule.kind is YawKind.BODY_YAW:
        mat = np.asarray(
            sim.data.body_xmat[sim.model.body_name2id(rule.anchor)], dtype=np.float64
        ).reshape(3, 3)
        raw = math.atan2(mat[1, 0], mat[0, 0]) + rule.offset_rad
    elif rule.kind is YawKind.RADIAL:
        d = _site_pos(sim, rule.anchor) - _site_pos(sim, rule.center)
        raw = math.atan2(d[1], d[0]) + rule.offset_rad
    else:
        raise ValueError(f"unhandled yaw kind {rule.kind!r}")
    return _wrap_jaw_yaw(raw)


def _rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _body_yaw(sim: Any, name: str) -> float:
    mat = np.asarray(sim.data.body_xmat[sim.model.body_name2id(name)],
                     dtype=np.float64).reshape(3, 3)
    return math.atan2(mat[1, 0], mat[0, 0])


def _fold(angle: float, period: float) -> float:
    """Fold into (-period/2, period/2]."""
    a = (angle + period / 2.0) % period - period / 2.0
    return period / 2.0 if a <= -period / 2.0 else a


def _align_turn(rule: AlignRule, grasp_yaw: float, sim: Any) -> float:
    """How far the wrist turns between the grasp and the mated pose.

    Both ends are held strictly inside the jaw-yaw band, so the wrist is always
    reachable and the turn is whatever the two band-resident yaws differ by. The
    branches that mate are spaced 2*pi/fold apart, the band is pi wide, so at
    least one lands in it for any fold of 2 or more and usually two do; of those
    the one that swings the grip toward the robot edge wins, ties going to the
    smaller turn so a part already square with its feature is left alone.

    Rotating the grip offset by exactly this turn, rather than by the nominal
    correction, is what keeps the arithmetic honest: a branch a half turn away
    mates just as well but puts the grip on the opposite side, and a waypoint
    that assumed the nominal correction would miss by twice the lever arm.
    """
    period = 2.0 * math.pi / rule.symmetry_fold
    nominal = _fold(
        _body_yaw(sim, rule.target_body) - _body_yaw(sim, rule.source_body), period
    )
    lever = _site_pos(sim, rule.grip_site) - _site_pos(sim, rule.axis_site)
    radius = float(math.hypot(lever[0], lever[1]))
    bearing = math.atan2(lever[1], lever[0])

    lo, hi = YAW_DATUM_RAD - math.pi / 2.0, YAW_DATUM_RAD + math.pi / 2.0
    best: tuple[float, float] | None = None
    turn: float | None = None
    for k in range(-rule.symmetry_fold, rule.symmetry_fold + 1):
        candidate = nominal + k * period
        if not lo < grasp_yaw + candidate <= hi:
            continue
        score = (radius * math.cos(bearing + candidate), abs(candidate))
        if best is None or score < best:
            best, turn = score, candidate
    if turn is None:
        raise ValueError(
            f"no mating branch of {rule.source_body!r} onto {rule.target_body!r} "
            f"puts the wrist inside the jaw-yaw band at grasp yaw {grasp_yaw:.3f}"
        )
    return turn


def _waypoint_yaw(wp: Waypoint, sim: Any) -> tuple[float, float]:
    """The wrist yaw at this waypoint, and how far it has turned since the grasp."""
    grasp_yaw = _resolve_yaw(wp.yaw, sim)
    if wp.align is None:
        return grasp_yaw, 0.0
    turn = _align_turn(wp.align, grasp_yaw, sim)
    return (grasp_yaw + turn, turn) if wp.align.engaged else (grasp_yaw, 0.0)


def _waypoint_pose(wp: Waypoint, sim: Any) -> tuple[np.ndarray, np.ndarray]:
    yaw, align_turn = _waypoint_yaw(wp, sim)
    tilt = _rot_x(wp.tilt_rad)
    command_rot = _rot_z(yaw) @ TOOL_DOWN @ tilt
    term_yaw = _resolve_yaw(wp.term_yaw, sim) if wp.term_yaw is not None else yaw
    term_tilt = _rot_x(wp.term_tilt_rad if wp.term_tilt_rad is not None else wp.tilt_rad)
    term_rot = _rot_z(term_yaw) @ TOOL_DOWN @ term_tilt
    if (
        wp.command_rotate_from_body is None
        and wp.command_rotate_from_site is None
        and wp.command_rotate_from_yaw is None
    ):
        command_terms_rot = command_rot
    elif wp.command_rotate_from_site is not None:
        site_id = sim.model.site_name2id(wp.command_rotate_from_site)
        command_reference_rot = np.asarray(
            sim.data.site_xmat[site_id], dtype=np.float64
        ).reshape(3, 3)
        # The hole offset is measured in the reset grasp-site frame. Move it
        # into the terminal command frame with the same rigid transform used
        # by the held tool: R_terminal @ R_grasp(reset)^T.
        command_terms_rot = command_rot @ command_reference_rot.T
    elif wp.command_rotate_from_body is not None:
        body_id = sim.model.body_name2id(wp.command_rotate_from_body)
        command_reference_rot = np.asarray(
            sim.data.body_xmat[body_id], dtype=np.float64
        ).reshape(3, 3)
        # Terms measured in a carried object's reset frame must follow the
        # terminal command pose via R_terminal @ R_object(reset)^T.
        command_terms_rot = command_rot @ command_reference_rot.T
    else:
        command_reference_yaw = _resolve_yaw(wp.command_rotate_from_yaw, sim)
        command_reference_rot = _rot_z(command_reference_yaw) @ TOOL_DOWN
        command_terms_rot = command_rot @ command_reference_rot.T
    if wp.rotate_from_yaw is None:
        # SITE_DELTA values are already world-frame vectors at reset.  They
        # therefore need only the relative alignment turn, not the absolute
        # tool-down command rotation.  Applying term_rot here rotates the lever
        # arm twice and places a carried nut several centimetres off its peg.
        r_terms = _rot_z(align_turn) if wp.align is not None else np.eye(3)
    else:
        reference_yaw = _resolve_yaw(wp.rotate_from_yaw, sim)
        reference_rot = _rot_z(reference_yaw) @ TOOL_DOWN
        # Object-relative terms must follow the same rigid transform as the
        # held object.  World-frame constants remain separate below.
        r_terms = term_rot @ reference_rot.T
    pos = np.zeros(3)
    if wp.anchor:
        base = _site_pos(sim, wp.anchor) if wp.anchor_kind == "site" else _body_pos(sim, wp.anchor)
        pos = pos + np.asarray(wp.anchor_mask, dtype=np.float64) * base
    for term in wp.terms:
        value = _eval_term(term, sim)
        if not term.rotate:
            transformed = value
        elif term.frame == "command":
            transformed = command_terms_rot @ value
        else:
            transformed = r_terms @ value
        pos = pos + transformed
    rz = _rot_z(yaw)
    pos = pos + rz @ np.asarray(wp.const, dtype=np.float64)
    return pos, command_rot


def _quintic(u: np.ndarray) -> np.ndarray:
    """s(u) = 10u^3 - 15u^4 + 6u^5, with s'(0) = s'(1) = s''(0) = s''(1) = 0.

    Applied per segment, so the trajectory comes to rest at every waypoint. The
    alternative -- one quintic across the whole path -- satisfies 9.4.2's
    endpoint condition too but sweeps through the grasp pose at speed, and the
    grasp has to be stationary for the gripper command to mean anything.
    """
    return u ** 3 * (10.0 + u * (-15.0 + 6.0 * u))


def _slerp(r0: np.ndarray, r1: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Constant-axis interpolation from r0 to r1 at scalar fractions `s`."""
    rel = r1 @ r0.T
    rotvec = Rotation.from_matrix(rel).as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.repeat(r0[None, :, :], len(s), axis=0)
    rots = Rotation.from_rotvec(s[:, None] * rotvec[None, :]).as_matrix()
    return rots @ r0[None, :, :]


def _expand(spec: ReferenceSpec, sim: Any) -> tuple[Waypoint, ...]:
    """Instantiate `Expansion` waypoints against the sites present in the model."""
    out: list[Waypoint] = []
    for wp in spec.waypoints:
        if wp.expand is None:
            out.append(wp)
            continue
        names = [
            wp.expand.site_pattern.format(i=i)
            for i in range(wp.expand.max_count)
        ]
        present = [n for n in names if _has_site(sim, n)]
        if not present:
            raise ValueError(
                f"{spec.task_id}: waypoint {wp.name!r} expands over "
                f"{wp.expand.site_pattern!r} and the model has none of them"
            )
        shares = _allocate(tuple(1 for _ in present), wp.steps)
        for k, (site, steps) in enumerate(zip(present, shares, strict=True)):
            out.append(
                replace(
                    wp,
                    name=f"{wp.name}_{k}",
                    steps=steps,
                    anchor=site,
                    anchor_kind="site",
                    expand=None,
                )
            )
    return tuple(out)


def _has_site(sim: Any, name: str) -> bool:
    try:
        sim.model.site_name2id(name)
    except (ValueError, KeyError):
        return False
    return True


def resolve(spec: ReferenceSpec, sim: Any) -> ReferenceTrajectory:
    """Evaluate a spec against a reset environment, giving X_tau in world coordinates.

    `sim` is a robosuite `MjSim` (or anything exposing the same `model`/`data`
    name lookups). The environment must have been reset and forwarded: the
    resolver reads `site_xpos`, `body_xpos`, `body_xmat` and `geom_xpos`, all of
    which are position-stage quantities.
    """
    waypoints = _expand(spec, sim)
    poses = [_waypoint_pose(wp, sim) for wp in waypoints]

    # The first segment starts from the pose the arm is already in, so that the
    # reference is continuous with the reset state rather than commanding a jump
    # the tracker would have to absorb at output_max.
    start_pos, start_rot = _tool_pose(sim)

    positions = np.empty((spec.horizon, 3), dtype=np.float64)
    rotations = np.empty((spec.horizon, 3, 3), dtype=np.float64)
    gripper = np.empty(spec.horizon, dtype=np.float64)
    phase = np.empty(spec.horizon, dtype=np.uint8)
    windex = np.empty(spec.horizon, dtype=np.int32)

    cursor = 0
    prev_pos, prev_rot = start_pos, start_rot
    for j, (wp, (pos, rot)) in enumerate(zip(waypoints, poses, strict=True)):
        n = wp.steps
        u = (np.arange(1, n + 1, dtype=np.float64)) / n
        s = _quintic(u)
        positions[cursor:cursor + n] = prev_pos[None, :] + s[:, None] * (pos - prev_pos)[None, :]
        rotations[cursor:cursor + n] = _slerp(prev_rot, rot, s)
        gripper[cursor:cursor + n] = wp.gripper
        phase[cursor:cursor + n] = _PHASE_ORDINAL[wp.phase]
        windex[cursor:cursor + n] = j
        cursor += n
        prev_pos, prev_rot = pos, rot

    return ReferenceTrajectory(
        task_id=spec.task_id,
        positions=positions,
        rotations=rotations,
        gripper=gripper,
        phase=phase,
        waypoint_index=windex,
        waypoint_names=tuple(w.name for w in waypoints),
    )


_TERMINAL_OBJECT_WAYPOINTS: dict[str, tuple[str, str]] = {
    "ToolHang": ("frame_close", "frame_insert"),
    "PickPlaceCan": ("close", "descend"),
}


def terminal_pose_of(spec: TaskSpec, sim: Any) -> Any:
    """Return the moving object's pose at the reference-defined terminal state.

    The reference controls the gripper, not the free object directly.  Propagate
    the object's reset-time transform relative to the grasp waypoint through to
    the terminal gripper waypoint.  This gives geometry calibration the same
    ``x*`` as the demonstrator without running contact dynamics or recording a
    family-specific robot trajectory.
    """
    from portal_trace.registry.geometry import TerminalPose

    if spec.moving_body is None:
        raise ValueError(f"{spec.task_id}: FROM_REFERENCE needs a moving_body")
    try:
        grasp_name, terminal_name = _TERMINAL_OBJECT_WAYPOINTS[spec.task_id]
    except KeyError as exc:
        raise ValueError(
            f"{spec.task_id}: no reference waypoint pair declares the moving "
            "object's grasp and terminal poses"
        ) from exc

    reference = load_spec(spec.task_id)
    waypoints = {wp.name: wp for wp in _expand(reference, sim)}
    missing = [name for name in (grasp_name, terminal_name) if name not in waypoints]
    if missing:
        raise ValueError(f"{spec.task_id}: reference lacks terminal-pose waypoints {missing}")
    grasp_pos, grasp_rot = _waypoint_pose(waypoints[grasp_name], sim)
    terminal_pos, terminal_rot = _waypoint_pose(waypoints[terminal_name], sim)

    body_id = sim.model.body_name2id(spec.moving_body)
    body_pos = np.asarray(sim.data.body_xpos[body_id], dtype=np.float64)
    body_rot = np.asarray(sim.data.body_xmat[body_id], dtype=np.float64).reshape(3, 3)
    carried_rot = terminal_rot @ grasp_rot.T
    target_rot = carried_rot @ body_rot
    target_pos = terminal_pos + carried_rot @ (body_pos - grasp_pos)
    quat_xyzw = Rotation.from_matrix(target_rot).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    return TerminalPose(
        body=spec.moving_body,
        position=target_pos,
        quaternion=quat_wxyz,
    )


def _tool_pose(sim: Any) -> tuple[np.ndarray, np.ndarray]:
    """The controlled site's world pose, read the way the OSC controller reads it.

    Uses `robot0_grip_site`, which is the `ref_name` robosuite hands the arm
    controller for a single-arm robot. Reading the same site keeps the reference
    and the tracker in one frame; taking the eef body pose instead would offset
    the whole trajectory by the grip-site offset of whichever family is running.
    """
    for name in ("gripper0_right_grip_site", "gripper0_grip_site", "robot0_grip_site"):
        if _has_site(sim, name):
            sid = sim.model.site_name2id(name)
            pos = np.asarray(sim.data.site_xpos[sid], dtype=np.float64).copy()
            rot = np.asarray(sim.data.site_xmat[sid], dtype=np.float64).reshape(3, 3).copy()
            return pos, rot
    raise ValueError(
        "no grip site found on this model; the reference resolver needs the same "
        "controlled site the OSC controller tracks"
    )


# ---------------------------------------------------------------------------
# The specs
# ---------------------------------------------------------------------------


def _allocate(weights: Sequence[int], total: int) -> tuple[int, ...]:
    """Largest-remainder allocation of `total` control steps over `weights`.

    Deterministic and exact: the segments sum to `total` with no rounding slack,
    which is what `ReferenceSpec.__post_init__` checks.
    """
    w = np.asarray(weights, dtype=np.float64)
    if np.any(w <= 0):
        raise ValueError(f"segment weights must be positive, got {list(weights)}")
    exact = w / w.sum() * total
    base = np.floor(exact).astype(int)
    base = np.maximum(base, 1)
    short = total - int(base.sum())
    if short > 0:
        order = np.argsort(-(exact - np.floor(exact)), kind="stable")
        for i in range(short):
            base[order[i % len(base)]] += 1
    elif short < 0:
        order = np.argsort(exact - np.floor(exact), kind="stable")
        i = 0
        while short < 0:
            j = order[i % len(base)]
            if base[j] > 1:
                base[j] -= 1
                short += 1
            i += 1
    return tuple(int(v) for v in base)


def _spec(task_id: str, rows: Sequence[dict[str, Any]], *, notes: str = "") -> ReferenceSpec:
    steps = _allocate([int(r.pop("w")) for r in rows], HORIZON)
    waypoints = tuple(Waypoint(steps=s, **r) for s, r in zip(steps, rows, strict=True))
    return ReferenceSpec(task_id=task_id, waypoints=waypoints, notes=notes)


def _nut_assembly_spec(
    task_id: str, *, nut: str, peg: str, symmetry_fold: int
) -> ReferenceSpec:
    handle = f"{nut}_handle_site"
    center = f"{nut}_center_site"
    body = f"{nut}_main"
    # The jaws straddle the handle bar across its width, so the jaw axis is the
    # quarter turn off the radial vector from the nut centre to the handle site;
    # the radial vector itself is read off the model rather than tabulated, so
    # this stays a statement about nut geometry.
    #
    # The quarter turn is not cosmetic. Closing along the radial direction puts
    # the inboard finger one finger-width from the nut centre, and the nut centre
    # is exactly where the peg is: the finger lands on the peg top and the nut
    # stops 12 cm above the table with the hole still empty. Closing across the
    # bar keeps both fingers out at the handle's own radius, 54 mm, clear of a
    # peg 16 mm wide. The bar is 31.8 mm across, well inside the jaw stroke of
    # every gripper in the family set.
    yaw = YawRule(kind=YawKind.RADIAL, anchor=handle, center=center,
                  offset_rad=math.pi / 2.0)
    align = AlignRule(target_body=peg, source_body=body, grip_site=handle,
                      axis_site=center, symmetry_fold=symmetry_fold)
    held = replace(align, engaged=False)
    # Carrying the nut by its handle means the eef has to sit at
    # peg_xy + (handle_xy - centre_xy) for the nut's hole to be over the peg,
    # with the offset turned by whatever the squaring-up rotation was.
    over_peg = (
        Term(kind=TermKind.BODY_POS, a=peg, mask=(1.0, 1.0, 0.0)),
        Term(kind=TermKind.SITE_DELTA, a=handle, b=center, mask=(1.0, 1.0, 0.0), rotate=True),
    )
    peg_top = Term(kind=TermKind.BODY_TOP_Z, a=peg, mask=(0.0, 0.0, 1.0))
    # The seated height is the handle's own height with the nut resting on the
    # table, read after the scene has settled. Stopping partway down the peg
    # instead leaves the nut above the success band of `on_peg` and, worse, leaves
    # the terminal contact in whatever partially-engaged state the descent
    # happened to jam in.
    seated_z = Term(kind=TermKind.SITE_POS, a=handle, mask=(0.0, 0.0, 1.0))
    return _spec(
        task_id,
        [
            dict(name="pre_grasp", phase=Phase.APPROACH, w=15, anchor=handle, yaw=yaw,
                 align=held, const=(0.0, 0.0, STANDOFF_M), gripper=GRIPPER_OPEN,
                 derived_from="handle site + STANDOFF_M"),
            dict(name="grasp", phase=Phase.APPROACH, w=11, anchor=handle, yaw=yaw,
                 align=held, gripper=GRIPPER_OPEN, derived_from="handle site"),
            dict(name="close", phase=Phase.GRASP, w=5, anchor=handle, yaw=yaw,
                 align=held, gripper=GRIPPER_CLOSED,
                 derived_from="handle site, jaws closing"),
            dict(name="lift", phase=Phase.TRANSPORT, w=10, anchor=handle,
                 anchor_mask=(1.0, 1.0, 0.0), terms=(peg_top,), yaw=yaw, align=align,
                 const=(0.0, 0.0, TRANSPORT_CLEARANCE_M), gripper=GRIPPER_CLOSED,
                 derived_from="peg top + TRANSPORT_CLEARANCE_M, above the nut"),
            dict(name="align", phase=Phase.TRANSPORT, w=17, terms=(*over_peg, peg_top),
                 yaw=yaw, align=align, const=(0.0, 0.0, TRANSPORT_CLEARANCE_M),
                 gripper=GRIPPER_CLOSED,
                 derived_from="peg xy + grasp offset, peg top + TRANSPORT_CLEARANCE_M"),
            dict(name="descend", phase=Phase.TRANSPORT, w=12, terms=(*over_peg, peg_top),
                 yaw=yaw, align=align, const=(0.0, 0.0, FINAL_APPROACH_M),
                 gripper=GRIPPER_CLOSED, derived_from="peg top + FINAL_APPROACH_M"),
            dict(name="insert", phase=Phase.TERMINAL, w=14, terms=(*over_peg, seated_z),
                 yaw=yaw, align=align, const=(0.0, 0.0, SEATING_MARGIN_M),
                 gripper=GRIPPER_CLOSED,
                 derived_from="peg xy + grasp offset, settled handle z + SEATING_MARGIN_M"),
            dict(name="release", phase=Phase.TERMINAL, w=5, terms=(*over_peg, seated_z),
                 yaw=yaw, align=align, const=(0.0, 0.0, SEATING_MARGIN_M),
                 gripper=GRIPPER_OPEN, derived_from="same pose, jaws opening"),
            dict(name="retreat", phase=Phase.TERMINAL, w=11, terms=(*over_peg, peg_top),
                 yaw=yaw, align=align, const=(0.0, 0.0, TRANSPORT_CLEARANCE_M),
                 gripper=GRIPPER_OPEN, derived_from="peg top + TRANSPORT_CLEARANCE_M"),
        ],
        notes=(
            f"nut={nut}, peg={peg}. Grasp yaw is the handle's radial direction; the "
            "align waypoint carries the handle-to-centre offset so that the hole, "
            f"not the grip, ends up over the peg. Body {body} is the moving body of "
            "the 9.5.2 table"
        ),
    )


def _stack_spec() -> ReferenceSpec:
    a_site, b_site = "cubeA_default_site", "cubeB_default_site"
    yaw = YawRule(kind=YawKind.BODY_YAW, anchor="cubeA_main")
    over_b = (
        Term(kind=TermKind.SITE_POS, a=b_site, mask=(1.0, 1.0, 0.0)),
        Term(kind=TermKind.BODY_TOP_Z, a="cubeB_main", mask=(0.0, 0.0, 1.0)),
    )
    a_half = Term(kind=TermKind.BODY_BBOX_HALF, a="cubeA_main", axis=2, mask=(0.0, 0.0, 1.0))
    return _spec(
        "Stack",
        [
            dict(name="pre_grasp", phase=Phase.APPROACH, w=15, anchor=a_site, yaw=yaw,
                 const=(0.0, 0.0, STANDOFF_M), gripper=GRIPPER_OPEN),
            dict(name="grasp", phase=Phase.APPROACH, w=11, anchor=a_site, yaw=yaw,
                 gripper=GRIPPER_OPEN),
            dict(name="close", phase=Phase.GRASP, w=5, anchor=a_site, yaw=yaw,
                 gripper=GRIPPER_CLOSED),
            dict(name="lift", phase=Phase.TRANSPORT, w=10, anchor=a_site,
                 yaw=yaw, const=(0.0, 0.0, STANDOFF_M + TRANSPORT_CLEARANCE_M),
                 gripper=GRIPPER_CLOSED),
            dict(name="align", phase=Phase.TRANSPORT, w=17, terms=over_b, yaw=yaw,
                 const=(0.0, 0.0, STANDOFF_M), gripper=GRIPPER_CLOSED,
                 derived_from="cubeB top + STANDOFF_M"),
            dict(name="descend", phase=Phase.TRANSPORT, w=12, terms=(*over_b, a_half),
                 yaw=yaw, const=(0.0, 0.0, FINAL_APPROACH_M), gripper=GRIPPER_CLOSED),
            dict(name="place", phase=Phase.TERMINAL, w=14, terms=(*over_b, a_half),
                 yaw=yaw, const=(0.0, 0.0, SEATING_MARGIN_M), gripper=GRIPPER_CLOSED,
                 derived_from="cubeB top + cubeA half height + SEATING_MARGIN_M"),
            dict(name="release", phase=Phase.TERMINAL, w=5, terms=(*over_b, a_half),
                 yaw=yaw, const=(0.0, 0.0, SEATING_MARGIN_M), gripper=GRIPPER_OPEN),
            dict(name="retreat", phase=Phase.TERMINAL, w=11, terms=over_b, yaw=yaw,
                 const=(0.0, 0.0, STANDOFF_M), gripper=GRIPPER_OPEN),
        ],
        notes=(
            "the low-kappa control of 9.5.2. tol is a centroid-containment radius "
            "(TolRule.CENTROID_CONTAINMENT), so the terminal pose only has to put "
            "cubeA's centre inside cubeB's top face"
        ),
    )


def _lift_spec() -> ReferenceSpec:
    site = "cube_default_site"
    yaw = YawRule(kind=YawKind.BODY_YAW, anchor="cube_main")
    return _spec(
        "Lift",
        [
            dict(name="pre_grasp", phase=Phase.APPROACH, w=18, anchor=site, yaw=yaw,
                 const=(0.0, 0.0, STANDOFF_M), gripper=GRIPPER_OPEN),
            dict(name="grasp", phase=Phase.APPROACH, w=12, anchor=site, yaw=yaw,
                 gripper=GRIPPER_OPEN),
            dict(name="close", phase=Phase.GRASP, w=6, anchor=site, yaw=yaw,
                 gripper=GRIPPER_CLOSED),
            dict(name="lift", phase=Phase.TRANSPORT, w=14, anchor=site, yaw=yaw,
                 const=(0.0, 0.0, STANDOFF_M), gripper=GRIPPER_CLOSED,
                 derived_from="0.10 m, well above the 0.04 m success threshold"),
            dict(name="hold", phase=Phase.TERMINAL, w=50, anchor=site, yaw=yaw,
                 const=(0.0, 0.0, STANDOFF_M), gripper=GRIPPER_CLOSED,
                 derived_from="9.6.1 fixed-length collection: hold to the horizon"),
        ],
        notes="held out; no fit at all, the low end of defect-active occupancy (9.5.2)",
    )


def _pick_place_can_spec() -> ReferenceSpec:
    site = "Can_default_site"
    yaw = YawRule(kind=YawKind.BODY_YAW, anchor="Can_main")
    # PickPlace sends object i to the quadrant of bin2 selected by the two low
    # bits of i; the can is object 3, so both offsets are positive quarter-widths
    # of the bin. Read off the bin's bounding extent rather than tabulated.
    over_target = (
        Term(kind=TermKind.BODY_POS, a="bin2", mask=(1.0, 1.0, 0.0)),
        Term(kind=TermKind.BODY_BBOX_HALF, a="bin2", axis=0, mask=(1.0, 0.0, 0.0), scale=0.5),
        Term(kind=TermKind.BODY_BBOX_HALF, a="bin2", axis=1, mask=(0.0, 1.0, 0.0), scale=0.5),
        Term(kind=TermKind.BODY_TOP_Z, a="bin2", mask=(0.0, 0.0, 1.0)),
    )
    return _spec(
        "PickPlaceCan",
        [
            dict(name="pre_grasp", phase=Phase.APPROACH, w=14, anchor=site, yaw=yaw,
                 const=(0.0, 0.0, STANDOFF_M), gripper=GRIPPER_OPEN),
            dict(name="grasp", phase=Phase.APPROACH, w=10, anchor=site, yaw=yaw,
                 gripper=GRIPPER_OPEN),
            dict(name="close", phase=Phase.GRASP, w=5, anchor=site, yaw=yaw,
                 gripper=GRIPPER_CLOSED),
            dict(name="lift", phase=Phase.TRANSPORT, w=11, anchor=site, yaw=yaw,
                 const=(0.0, 0.0, STANDOFF_M + TRANSPORT_CLEARANCE_M),
                 gripper=GRIPPER_CLOSED),
            dict(name="transit", phase=Phase.TRANSPORT, w=20, terms=over_target, yaw=yaw,
                 const=(0.0, 0.0, STANDOFF_M + TRANSPORT_CLEARANCE_M),
                 gripper=GRIPPER_CLOSED,
                 derived_from="bin2 centre + quarter bounding extent on x and y"),
            dict(name="descend", phase=Phase.TERMINAL, w=15, terms=over_target, yaw=yaw,
                 const=(0.0, 0.0, FINAL_APPROACH_M), gripper=GRIPPER_CLOSED),
            dict(name="release", phase=Phase.TERMINAL, w=6, terms=over_target, yaw=yaw,
                 const=(0.0, 0.0, FINAL_APPROACH_M), gripper=GRIPPER_OPEN),
            dict(name="retreat", phase=Phase.TERMINAL, w=19, terms=over_target, yaw=yaw,
                 const=(0.0, 0.0, STANDOFF_M + TRANSPORT_CLEARANCE_M),
                 gripper=GRIPPER_OPEN),
        ],
        notes="held out; used only for the (A4) hold-out R^2 (9.6.6)",
    )


def _door_spec() -> ReferenceSpec:
    handle = "Door_handle"
    # The hinge anchor is not Door_frame's body origin: in the shipped model the
    # two differ by more than 30 cm. The pull follows Door_hinge's compiled joint
    # anchor and positive axis, exactly matching hinge_qpos. The wrist yaw stays
    # fixed so the parallel jaws retain force closure across the handle while it
    # moves; rotating the wrist with the panel made the handle slip and the door
    # close again during the terminal hold.
    yaw = YawRule(kind=YawKind.BODY_YAW, anchor="Door_door")

    def arc(frac: float) -> tuple[Term, ...]:
        return (Term(kind=TermKind.JOINT_ARC, a=handle, b="Door_hinge",
                     angle_rad=DOOR_SWING_RAD * frac),)

    return _spec(
        "Door",
        [
            dict(name="pre_grasp", phase=Phase.APPROACH, w=16, anchor=handle, yaw=yaw,
                 const=(0.0, DOOR_STANDOFF_M, DOOR_STANDOFF_M), gripper=GRIPPER_OPEN,
                 derived_from="handle site + DOOR_STANDOFF_M along the approach diagonal"),
            dict(name="reach", phase=Phase.APPROACH, w=11, anchor=handle, yaw=yaw,
                 gripper=GRIPPER_OPEN),
            dict(name="close", phase=Phase.GRASP, w=6, anchor=handle, yaw=yaw,
                 gripper=GRIPPER_CLOSED),
            dict(name="grasp_hold", phase=Phase.TRANSPORT, w=10, anchor=handle, yaw=yaw,
                 gripper=GRIPPER_CLOSED),
            dict(name="pull_1", phase=Phase.TRANSPORT, w=13, terms=arc(0.34), yaw=yaw,
                 gripper=GRIPPER_CLOSED),
            dict(name="pull_2", phase=Phase.TRANSPORT, w=13, terms=arc(0.67), yaw=yaw,
                 gripper=GRIPPER_CLOSED),
            dict(name="pull_3", phase=Phase.TERMINAL, w=13, terms=arc(1.0), yaw=yaw,
                 gripper=GRIPPER_CLOSED),
            dict(name="hold", phase=Phase.TERMINAL, w=18, terms=arc(1.0), yaw=yaw,
                 gripper=GRIPPER_CLOSED),
        ],
        notes=(
            "unlocked r2 quasi-static carrier. The swing is an arc about the "
            f"Door_hinge joint pivot, {DOOR_SWING_RAD} rad inside the 0.4 rad "
            "joint limit, against a success criterion of 0.3 rad"
        ),
    )


def _wipe_spec() -> ReferenceSpec:
    first = "contact0_default_site"
    return _spec(
        "Wipe",
        [
            dict(name="pre_touch", phase=Phase.APPROACH, w=10, anchor=first,
                 const=(0.0, 0.0, STANDOFF_M), gripper=GRIPPER_CLOSED,
                 derived_from="first marker + STANDOFF_M"),
            dict(name="touch", phase=Phase.GRASP, w=6, anchor=first,
                 const=(0.0, 0.002, -WIPE_CONTACT_DEPTH_M), gripper=GRIPPER_CLOSED,
                 derived_from="first marker, pressed WIPE_CONTACT_DEPTH_M into the surface"),
            dict(name="sweep", phase=Phase.TERMINAL, w=74,
                 const=(0.0, 0.002, -WIPE_CONTACT_DEPTH_M), gripper=GRIPPER_CLOSED,
                 expand=Expansion(site_pattern="contact{i}_default_site", max_count=100),
                 derived_from="marker path shifted 2 mm in world +y to centre the wiper footprint"),
            dict(name="sweep_hold", phase=Phase.TERMINAL, w=4,
                 anchor="contact99_default_site",
                 const=(0.0, 0.002, -WIPE_CONTACT_DEPTH_M), gripper=GRIPPER_CLOSED,
                 derived_from="terminal dwell on the final marker before lift-off"),
            dict(name="lift_off", phase=Phase.TERMINAL, w=6,
                 anchor="contact0_default_site", anchor_mask=(0.0, 0.0, 0.0),
                 terms=(Term(kind=TermKind.SITE_POS, a="table_top"),),
                 const=(0.0, 0.0, STANDOFF_M), gripper=GRIPPER_CLOSED),
        ],
        notes=(
            "no crisp terminal pose (9.5.2); the marker count is an environment "
            "parameter, so the sweep is an expansion rather than a fixed list"
        ),
    )


def _tool_hang_spec() -> ReferenceSpec:
    grip = "frame_intersection_site"
    frame_mount, frame_tip, stand_base = "frame_mount_site", "frame_tip_site", "stand_base"
    # The gripper must first match the frame's reset yaw to establish force
    # closure. Once closed, switching to a fixed world yaw rotates the held frame
    # so its hook enters the stand; using the fixed yaw during approach misses
    # the small grip box and leaves the frame sitting in place.
    frame_grasp_yaw = YawRule(kind=YawKind.BODY_YAW, anchor="frame_root")
    frame_transport_yaw = YawRule(kind=YawKind.FIXED, offset_rad=1.0e-3)
    tool_grip = "tool_grip_default_site"
    tool_grasp_yaw = YawRule(kind=YawKind.BODY_YAW, anchor="tool_root")
    tool_terminal_yaw = YawRule(
        kind=YawKind.BODY_YAW,
        anchor="tool_root",
        offset_rad=TOOLHANG_TOOL_COMMAND_YAW_OFFSET_RAD,
    )
    grip_delta = Term(
        kind=TermKind.SITE_DELTA,
        a=frame_mount,
        b=grip,
        scale=TOOLHANG_GRIP_FRACTION,
        rotate=True,
    )
    grip_to_tip = (
        Term(kind=TermKind.SITE_DELTA, a=grip, b=frame_tip, rotate=True),
        grip_delta,
    )
    grip_to_tip_xy = tuple(replace(term, mask=(1.0, 1.0, 0.0)) for term in grip_to_tip)
    grip_to_tip_z = tuple(replace(term, mask=(0.0, 0.0, 1.0)) for term in grip_to_tip)
    over_stand = (
        Term(kind=TermKind.GEOM_POS, a=stand_base, mask=(1.0, 1.0, 0.0)),
        *grip_to_tip_xy,
    )
    stand_top = Term(kind=TermKind.BODY_TOP_Z, a="stand_root", mask=(0.0, 0.0, 1.0))
    seated_frame_z = (
        Term(kind=TermKind.GEOM_POS, a=stand_base, mask=(0.0, 0.0, 1.0)),
        *grip_to_tip_z,
    )
    frame_clear_z = (
        Term(kind=TermKind.BODY_TOP_Z, a="stand_root", mask=(0.0, 0.0, 1.0)),
        *grip_to_tip_z,
    )
    frame_clear = (*over_stand, *frame_clear_z)
    # The frame is moved before the tool is picked up.  Derive its planned
    # hook pose from the target frame tip and rigid site offsets, rather than
    # reusing reset-time frame_hang_site coordinates.  The two wall geoms are
    # opposite sides of the square mount, so their mean is the support centre.
    wall_center_xy = (
        Term(kind=TermKind.GEOM_POS, a="stand_wall0", mask=(1.0, 1.0, 0.0), scale=0.5),
        Term(kind=TermKind.GEOM_POS, a="stand_wall2", mask=(1.0, 1.0, 0.0), scale=0.5),
    )
    frame_hang_delta = Term(
        kind=TermKind.SITE_DELTA, a="frame_hang_site", b=frame_tip,
        rotate=True,
    )
    hook_insert_delta = Term(
        kind=TermKind.SITE_DELTA, a="frame_intersection_site", b="frame_hang_site",
        scale=TOOLHANG_HOOK_INSERT_FRACTION, rotate=True,
    )
    tool_hole_delta = Term(
        kind=TermKind.SITE_DELTA, a=tool_grip, b="tool_hole1_center",
        rotate=True, frame="command",
    )
    frame_hang_xy = replace(frame_hang_delta, mask=(1.0, 1.0, 0.0))
    frame_hang_z = replace(frame_hang_delta, mask=(0.0, 0.0, 1.0))
    hook_insert_xy = replace(hook_insert_delta, mask=(1.0, 1.0, 0.0))
    hook_insert_z = replace(hook_insert_delta, mask=(0.0, 0.0, 1.0))
    tool_hole_xy = replace(tool_hole_delta, mask=(1.0, 1.0, 0.0))
    tool_hole_z = replace(tool_hole_delta, mask=(0.0, 0.0, 1.0))
    over_hook = (*wall_center_xy, frame_hang_xy, hook_insert_xy,
                 tool_hole_xy)
    hook_z = (
        Term(kind=TermKind.GEOM_POS, a=stand_base, mask=(0.0, 0.0, 1.0)),
        frame_hang_z, hook_insert_z, tool_hole_z,
    )
    hook_clear_z = (
        Term(kind=TermKind.GEOM_POS, a=stand_base, mask=(0.0, 0.0, 1.0)),
        frame_hang_z, hook_insert_z, tool_hole_z,
    )
    return _spec(
        "ToolHang",
        [
            dict(name="frame_pre_grasp", phase=Phase.APPROACH, w=8, anchor=grip,
                 terms=(grip_delta,), yaw=frame_grasp_yaw,
                 const=(0.0, 0.0, STANDOFF_M), gripper=GRIPPER_OPEN),
            dict(name="frame_grasp", phase=Phase.APPROACH, w=6, anchor=grip,
                 terms=(grip_delta,), yaw=frame_grasp_yaw,
                 gripper=GRIPPER_OPEN),
            dict(name="frame_close", phase=Phase.GRASP, w=3, anchor=grip,
                 terms=(grip_delta,), yaw=frame_grasp_yaw,
                 gripper=GRIPPER_CLOSED),
            dict(name="frame_lift", phase=Phase.TRANSPORT, w=6, anchor=grip,
                 anchor_mask=(1.0, 1.0, 0.0), terms=(grip_delta, stand_top), yaw=frame_transport_yaw,
                 rotate_from_yaw=frame_grasp_yaw,
                 const=(0.0, 0.0, TRANSPORT_CLEARANCE_M), tilt_rad=TOOLHANG_FRAME_TILT_RAD,
                 gripper=GRIPPER_CLOSED),
            dict(name="frame_align", phase=Phase.TRANSPORT, w=15,
                 terms=frame_clear, yaw=frame_transport_yaw,
                 rotate_from_yaw=frame_grasp_yaw,
                 const=(0.0, TOOLHANG_MOUNT_CENTER_Y_M, TRANSPORT_CLEARANCE_M),
                 tilt_rad=TOOLHANG_FRAME_TRANSPORT_TILT_RAD,
                 gripper=GRIPPER_CLOSED,
                 derived_from="frame-tip xy over stand, stand top + clearance"),
            dict(name="frame_insert", phase=Phase.TERMINAL, w=12,
                 terms=(*over_stand, *seated_frame_z), yaw=frame_transport_yaw,
                 rotate_from_yaw=frame_grasp_yaw,
                 const=(0.0, TOOLHANG_MOUNT_CENTER_Y_M, 0.0),
                 tilt_rad=TOOLHANG_FRAME_TRANSPORT_TILT_RAD, gripper=GRIPPER_CLOSED),
            dict(name="frame_release", phase=Phase.TERMINAL, w=3,
                 terms=(*over_stand, *seated_frame_z), yaw=frame_transport_yaw,
                 rotate_from_yaw=frame_grasp_yaw,
                 const=(0.0, TOOLHANG_MOUNT_CENTER_Y_M, 0.0),
                 tilt_rad=TOOLHANG_FRAME_TRANSPORT_TILT_RAD, gripper=GRIPPER_OPEN),
            dict(name="frame_retreat", phase=Phase.TRANSPORT, w=3,
                 terms=frame_clear, yaw=frame_transport_yaw,
                 rotate_from_yaw=frame_grasp_yaw,
                 const=(0.0, TOOLHANG_MOUNT_CENTER_Y_M, 0.0),
                 tilt_rad=TOOLHANG_FRAME_TRANSPORT_TILT_RAD, gripper=GRIPPER_OPEN),
            dict(name="tool_pre_grasp", phase=Phase.APPROACH, w=6, anchor=tool_grip,
                 yaw=tool_grasp_yaw,
                 const=(0.0, 0.0, STANDOFF_M + TOOLHANG_TOOL_GRASP_Z_OFFSET_M),
                 gripper=GRIPPER_OPEN),
            dict(name="tool_grasp", phase=Phase.APPROACH, w=6, anchor=tool_grip,
                 yaw=tool_grasp_yaw,
                 const=(0.0, 0.0, TOOLHANG_TOOL_GRASP_Z_OFFSET_M),
                 gripper=GRIPPER_OPEN),
            dict(name="tool_close", phase=Phase.GRASP, w=3, anchor=tool_grip, yaw=tool_grasp_yaw,
                 const=(0.0, 0.0, TOOLHANG_TOOL_GRASP_Z_OFFSET_M),
                 gripper=GRIPPER_CLOSED),
            dict(name="tool_lift", phase=Phase.TRANSPORT, w=6, anchor=tool_grip,
                 anchor_mask=(1.0, 1.0, 0.0), terms=(stand_top,), yaw=tool_grasp_yaw,
                 const=(0.0, 0.0, TRANSPORT_CLEARANCE_M), gripper=GRIPPER_CLOSED),
            dict(name="tool_align", phase=Phase.TRANSPORT, w=9,
                 terms=(*over_hook, *hook_clear_z), yaw=tool_terminal_yaw,
                 rotate_from_yaw=frame_grasp_yaw,
                 term_yaw=frame_transport_yaw,
                 term_tilt_rad=TOOLHANG_FRAME_TRANSPORT_TILT_RAD,
                 command_rotate_from_yaw=tool_grasp_yaw,
                 tilt_rad=TOOLHANG_TOOL_TILT_RAD,
                 const=(*TOOLHANG_TOOL_HOLE_SHIFT,
                        TRANSPORT_CLEARANCE_M + TOOLHANG_TOOL_GRASP_Z_OFFSET_M),
                 gripper=GRIPPER_CLOSED,
                 derived_from="planned frame hook midpoint + hole-to-grip offset"),
            dict(name="tool_hang", phase=Phase.TERMINAL, w=7, terms=(*over_hook, *hook_z),
                 yaw=tool_terminal_yaw, rotate_from_yaw=frame_grasp_yaw,
                 term_yaw=frame_transport_yaw,
                 term_tilt_rad=TOOLHANG_FRAME_TRANSPORT_TILT_RAD,
                 command_rotate_from_yaw=tool_grasp_yaw,
                 tilt_rad=TOOLHANG_TOOL_TILT_RAD,
                 const=(*TOOLHANG_TOOL_HOLE_SHIFT,
                        TOOLHANG_TOOL_SEATING_OFFSET_M + TOOLHANG_TOOL_GRASP_Z_OFFSET_M),
                 gripper=GRIPPER_CLOSED),
            dict(name="tool_release", phase=Phase.TERMINAL, w=3,
                 terms=(*over_hook, *hook_z), yaw=tool_terminal_yaw,
                 rotate_from_yaw=frame_grasp_yaw,
                 term_yaw=frame_transport_yaw,
                 term_tilt_rad=TOOLHANG_FRAME_TRANSPORT_TILT_RAD,
                 command_rotate_from_yaw=tool_grasp_yaw,
                 tilt_rad=TOOLHANG_TOOL_TILT_RAD,
                 const=(*TOOLHANG_TOOL_HOLE_SHIFT,
                        TOOLHANG_TOOL_SEATING_OFFSET_M + TOOLHANG_TOOL_GRASP_Z_OFFSET_M),
                 gripper=GRIPPER_OPEN),
            dict(name="tool_retreat", phase=Phase.TERMINAL, w=4,
                 terms=(*over_hook, *hook_clear_z), yaw=tool_terminal_yaw,
                 rotate_from_yaw=frame_grasp_yaw,
                 term_yaw=frame_transport_yaw,
                 term_tilt_rad=TOOLHANG_FRAME_TRANSPORT_TILT_RAD,
                 command_rotate_from_yaw=tool_grasp_yaw,
                 tilt_rad=TOOLHANG_TOOL_TILT_RAD,
                 const=(*TOOLHANG_TOOL_HOLE_SHIFT,
                        TRANSPORT_CLEARANCE_M + TOOLHANG_TOOL_SEATING_OFFSET_M
                        + TOOLHANG_TOOL_GRASP_Z_OFFSET_M),
                 gripper=GRIPPER_OPEN),
        ],
        notes=(
            "two insertions inside one horizon. 9.5.2 note 2 makes ToolHang's median "
            "episode length a W1 item: if the reference needs more than 400 steps to "
            "complete both stages, the core/heldout roles swap"
        ),
    )


def build_all_specs() -> dict[str, ReferenceSpec]:
    """One spec per task in `registry.tasks.TASKS`.

    The two tight variants share their parent's recipe verbatim. 9.5.4 shrinks
    the peg and rewrites the success criterion; neither changes which frames the
    waypoints hang off, and the peg-top clearance is resolved from the compiled
    geometry, so the recipe transfers without a numeric edit.
    """
    square = _nut_assembly_spec(
        "NutAssemblySquare", nut="SquareNut", peg="peg1", symmetry_fold=4
    )
    # robosuite's round nut is an octagonal prism, so eight-fold: the residual
    # correction is under 22.5 degrees and the hole is round anyway.
    rnd = _nut_assembly_spec(
        "NutAssemblyRound", nut="RoundNut", peg="peg2", symmetry_fold=8
    )
    specs = {
        square.task_id: square,
        rnd.task_id: rnd,
        "NutAssemblySquareTight-v0": replace(
            square,
            task_id="NutAssemblySquareTight-v0",
            notes=square.notes + "; inherited by NutAssemblySquareTight-v0 (9.5.4)",
        ),
        "NutAssemblyRoundTight-v0": replace(
            rnd,
            task_id="NutAssemblyRoundTight-v0",
            notes=rnd.notes + "; inherited by NutAssemblyRoundTight-v0 (9.5.4)",
        ),
    }
    for spec in (_stack_spec(), _lift_spec(), _pick_place_can_spec(),
                 _door_spec(), _wipe_spec(), _tool_hang_spec()):
        specs[spec.task_id] = spec
    missing = sorted(set(TASKS) - set(specs))
    if missing:
        raise AssertionError(
            f"no reference trajectory for {missing}; every task in the 9.5.2 table "
            "needs one, since X_tau is what makes the families comparable"
        )
    return specs


def spec_path(task_id: str) -> Path:
    return REFS_DIR / f"{task_id}.json"


def load_spec(task_id: str) -> ReferenceSpec:
    path = spec_path(task_id)
    if not path.exists():
        raise FileNotFoundError(
            f"no reference trajectory for {task_id!r} at {path}. Generate the set "
            "with `python -m portal_trace.teacher.refs --write`; it is checked in "
            "because X_tau is frozen across every family and every arm (14.5 E-9)"
        )
    with path.open("r", encoding="utf-8") as fh:
        spec = ReferenceSpec.from_json(json.load(fh))
    if spec.task_id != task_id:
        raise ValueError(f"{path} holds a spec for {spec.task_id!r}, not {task_id!r}")
    return spec


def write_specs(specs: dict[str, ReferenceSpec] | None = None) -> list[Path]:
    specs = specs if specs is not None else build_all_specs()
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for task_id, spec in sorted(specs.items()):
        path = spec_path(task_id)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(spec.to_json(), fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        written.append(path)
    return written


def _validate_against_model(spec: ReferenceSpec, sim: Any) -> None:
    """Every anchor a spec names has to exist in the compiled model."""
    for wp in _expand(spec, sim):
        if wp.anchor:
            probe = _has_site if wp.anchor_kind == "site" else _has_body
            if not probe(sim, wp.anchor):
                raise ValueError(f"{spec.task_id}/{wp.name}: no {wp.anchor_kind} {wp.anchor!r}")
        for term in wp.terms:
            for name, kind in ((term.a, _term_a_kind(term)), (term.b, _term_b_kind(term))):
                if not name:
                    continue
                if kind == "site":
                    ok = _has_site(sim, name)
                elif kind == "body":
                    ok = _has_body(sim, name)
                elif kind == "geom":
                    ok = _has_geom(sim, name)
                else:
                    ok = _has_joint(sim, name)
                if not ok:
                    raise ValueError(f"{spec.task_id}/{wp.name}: no {kind} {name!r}")
        if wp.align is not None:
            for name in (wp.align.target_body, wp.align.source_body):
                if not _has_body(sim, name):
                    raise ValueError(f"{spec.task_id}/{wp.name}: no body {name!r}")
            for name in (wp.align.grip_site, wp.align.axis_site):
                if not _has_site(sim, name):
                    raise ValueError(f"{spec.task_id}/{wp.name}: no site {name!r}")


def _term_a_kind(term: Term) -> str:
    if term.kind is TermKind.GEOM_POS:
        return "geom"
    return "body" if term.kind in (TermKind.BODY_POS, TermKind.BODY_TOP_Z,
                                   TermKind.BODY_BBOX_HALF) else "site"


def _term_b_kind(term: Term) -> str:
    if term.kind is TermKind.ARC:
        return "body"
    if term.kind is TermKind.JOINT_ARC:
        return "joint"
    return "site"


def _has_body(sim: Any, name: str) -> bool:
    try:
        sim.model.body_name2id(name)
    except (ValueError, KeyError):
        return False
    return True


def _has_joint(sim: Any, name: str) -> bool:
    try:
        sim.model.joint_name2id(name)
    except (ValueError, KeyError):
        return False
    return True


def _has_geom(sim: Any, name: str) -> bool:
    try:
        sim.model.geom_name2id(name)
    except (ValueError, KeyError):
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", action="store_true", help="regenerate teacher/refs/*.json")
    p.add_argument(
        "--validate",
        action="store_true",
        help="build each task's environment and check that every anchor exists",
    )
    p.add_argument("--task", action="append", default=None, help="restrict to these tasks")
    args = p.parse_args(argv)

    specs = build_all_specs()
    if args.task:
        specs = {k: v for k, v in specs.items() if k in set(args.task)}

    if args.write:
        for path in write_specs(specs):
            print(f"[refs] wrote {path}")

    if args.validate:
        from portal_trace.teacher.osc import make_env  # local: pulls in robosuite

        for task_id, spec in sorted(specs.items()):
            ts = task_spec(task_id)
            if ts.tight_variant_of is not None:
                print(f"[refs] {task_id}: skipped, its environment is registered by registry/")
                continue
            env = make_env(task_id, family="Panda", seed_env=0)
            try:
                _validate_against_model(spec, env.sim)
                traj = resolve(spec, env.sim)
                span = np.linalg.norm(np.diff(traj.positions, axis=0), axis=1)
                print(
                    f"[refs] {task_id}: {len(spec.waypoints)} waypoints, "
                    f"path {span.sum() * 100:.1f} cm, max step {span.max() * 1000:.2f} mm"
                )
            finally:
                env.close()

    if not (args.write or args.validate):
        p.error("nothing to do: pass --write, --validate or both")
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
