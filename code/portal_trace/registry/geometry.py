"""The G0/G1/G2 environment geometry of 9.5.5, and the cl_pm measurement.

T8's strong form needs the sign of the harm asymmetry
`asymmetry = cl_minus - cl_plus` to flip with the defect, the teacher, the
reference trajectory and the tolerance all held fixed. 9.5.5 achieves that with
three geometry levels that touch nothing but the arena MJCF:

G0  baseline `pegs_arena`. The nut lands on the table, so cl_plus is about zero
    and cl_minus is the height of the workspace: asymmetry >> 0.
G1  a recessed well around `peg1`. The table collision geom drops by the well
    depth d and four boxes restore the surface outside a square opening, which
    is 9.5.5's "four boxes forming a square well" and its "well floor geom,
    flush with the table, moved down by d" -- the same two sentences, in that
    order. The well mouth gets a lead-in for the reason 9.5.4 gives the peg its
    chamfer: a 1 mm wall clearance with a square edge would make "put the nut in
    the well" a second measure-zero alignment event in series with the
    insertion, and G1 is supposed to move cl_pm without moving task difficulty.
G2  G1 plus a baffle above the nut, which caps cl_minus at the baffle height h:
    asymmetry < 0, with the defect unchanged.

d and h have no defaults anywhere in this module. 9.5.5 derives them backwards
from F8's three-way feasibility check -- the intermediate band is (h, d] and it
has to intersect the Goldilocks window that the tight-fit tolerance defines -- so
P0 measures cl_pm and ||c_delta(x*)|| first and only then are d and h fixed.
`well_depth_m` and `baffle_height_m` read them from the lock file and raise when
they are not there.

`measure_clearance` implements the stepped `mj_collision` of 9.5.5 and does not
use `mj_ray`. The reason is in the section and is not a matter of taste: a square
nut is four boxes forming a hollow ring, and a ray cast down the hole axis goes
straight through the hole and reports the table, or nothing. cl_pm is how far the
*whole rigid body* translates before its first contact, which is a minimum over
every geom pair, and only a swept collision query answers that.

One measured constraint on the well opening, found with `measure_clearance` and
not derivable from 9.5.5. `SquareNut` is not the ring alone: robosuite gives it a
handle, geom `SquareNut_g4`, reaching to x = +0.07925 m from the body origin
while the ring's own half extent is 0.04375 m. A well sized to the ring plus 1 mm
of wall clearance therefore lands the handle on the slab outside the mouth and
cl_plus comes out 0.00 mm instead of d, which defeats the whole point of G1.
`inner_halfwidth_m` has to cover the handle -- 0.08025 m for the square nut --
and at that value the three levels reproduce 9.5.5's intent exactly. The
verification numbers are in the report; the value itself lives in
`registry.tasks`, which owns the variant parameters.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from portal_trace.lockfile import REQUIRED_MEASUREMENTS, LockFile
from portal_trace.registry.tasks import GeomVariant, TaskSpec, task

__all__ = [
    "TerminalPose",
    "ClearanceResult",
    "CLEARANCE_STEP_M",
    "CLEARANCE_MAX_STEPS",
    "apply_variant",
    "build_arena",
    "make_env",
    "well_depth_m",
    "baffle_height_m",
    "measure_clearance",
    "body_extent_along",
    "geom_variant",
    "env_class_with_geom_variant",
    "main",
]

# 9.5.5: 1e-4 m gives 0.1 mm resolution and 2000 steps cover 200 mm. Both are
# specification, not tuning: the reported cl_pm is quantised to the step, and the
# 0.8 mm / 2.5 mm band the whole design hangs on needs that resolution.
CLEARANCE_STEP_M = 1e-4
CLEARANCE_MAX_STEPS = 2000

_DECL_CLEARANCE = REQUIRED_MEASUREMENTS["geom.clearance.{task}.{geom_variant}.{direction}"]


# ---------------------------------------------------------------------------
# d and h come from the lock file
# ---------------------------------------------------------------------------


def well_depth_m(task_id: str, *, lock: LockFile | None = None) -> float:
    """The G1 well depth d.

    d *is* the downward clearance G1 is built to produce, so it is read from
    `geom.clearance.<task>.G1.plus` rather than from a second declaration that
    could drift away from it. The direction of the derivation is 9.5.5's: P0
    fixes the target clearance from the feasibility check, the MJCF is built to
    realise it, and `measure_clearance` then reports what the *actual* geometry
    gives -- which differs from the nominal, because the lead-in eats into the
    top of the wall and the stepped query sees that.
    """
    lock = lock if lock is not None else LockFile()
    return float(lock.get(_DECL_CLEARANCE, task=task_id, geom_variant="G1", direction="plus"))


def baffle_height_m(task_id: str, *, lock: LockFile | None = None) -> float:
    """The G2 baffle height h, i.e. the upward clearance the baffle leaves."""
    lock = lock if lock is not None else LockFile()
    return float(lock.get(_DECL_CLEARANCE, task=task_id, geom_variant="G2", direction="minus"))


def geom_variant(task_id: str, name: str) -> GeomVariant:
    spec: TaskSpec = task(task_id)
    for variant in spec.geom_variants:
        if variant.name == name:
            return variant
    raise KeyError(
        f"{task_id} has no geometry variant {name!r}; it declares "
        f"{[v.name for v in spec.geom_variants]}"
    )


# ---------------------------------------------------------------------------
# MJCF editing
# ---------------------------------------------------------------------------


def _find(root: ET.Element, path: str, what: str) -> ET.Element:
    element = root.find(path)
    if element is None:
        raise KeyError(f"the MJCF has no {what} ({path!r}); this is not a pegs arena")
    return element


def _floats(text: str | None, n: int, what: str) -> np.ndarray:
    if text is None:
        raise KeyError(f"{what} has no value in the MJCF")
    values = np.asarray([float(v) for v in text.split()], dtype=np.float64)
    if values.size != n:
        raise ValueError(f"{what} has {values.size} components, expected {n}")
    return values


def _box(
    name: str,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    collision: bool,
    friction: str | None,
    material: str | None,
) -> ET.Element:
    size = 0.5 * (hi - lo)
    if np.any(size <= 0.0):
        raise ValueError(
            f"{name}: degenerate box with half sizes {size.tolist()}. The well "
            "parameters do not describe a solid; check d against the lead-in height"
        )
    attrib = {
        "name": name,
        "type": "box",
        "pos": " ".join(f"{v:.17g}" for v in 0.5 * (lo + hi)),
        "size": " ".join(f"{v:.17g}" for v in size),
        "group": "0" if collision else "1",
    }
    if not collision:
        attrib["contype"] = "0"
        attrib["conaffinity"] = "0"
    if collision and friction is not None:
        attrib["friction"] = friction
    if not collision and material is not None:
        attrib["material"] = material
    return ET.Element("geom", attrib)


def _ramp(
    name: str,
    *,
    axis: int,
    sign: float,
    centre: np.ndarray,
    inner: float,
    mouth: float,
    lead_in: float,
    top_z: float,
    half_width: float,
    collision: bool,
    friction: str | None,
    material: str | None,
) -> ET.Element:
    """One side of the well-mouth lead-in, as a rotated box.

    The sloped face runs from (inner, top_z - lead_in) to (mouth, top_z). Its
    inclination is fixed by those three numbers, which come from
    `tasks.GeomVariant.mjcf_edits`; 9.5.5 also calls the lead-in 45 degrees, and
    with a 3 mm widening over a 1.5 mm rise it is 63.4 degrees from vertical
    instead. The three lengths are the authority here, because they are what
    `cl_plus` and the nut-to-wall clearance are computed from, and the angle is
    a description of them.

    The box is thick enough to fill the wedge and no thicker than the solid it
    sits in: its inner face is exactly the slope, and it extends outward and
    downward into the wall and the well floor, both of which are already solid.
    """
    widening = mouth - inner
    if widening <= 0.0 or lead_in <= 0.0:
        raise ValueError(f"{name}: the lead-in needs a positive rise and a positive widening")
    length = float(np.hypot(widening, lead_in))

    other = 1 - axis
    inward = np.zeros(3)
    inward[axis] = -sign
    up = np.array([0.0, 0.0, 1.0])

    normal = (lead_in / length) * inward + (widening / length) * up
    along = (-widening / length) * inward + (lead_in / length) * up
    binormal = np.cross(along, normal)

    rot = np.column_stack([normal, binormal, along])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.reshape(-1))

    mid = np.zeros(3)
    mid[axis] = centre[axis] + sign * 0.5 * (inner + mouth)
    mid[other] = centre[other]
    mid[2] = top_z - 0.5 * lead_in
    thickness = widening

    attrib = {
        "name": name,
        "type": "box",
        "pos": " ".join(f"{v:.17g}" for v in mid - thickness * normal),
        "quat": " ".join(f"{v:.17g}" for v in quat),
        "size": f"{thickness:.17g} {half_width:.17g} {0.5 * length:.17g}",
        "group": "0" if collision else "1",
    }
    if not collision:
        attrib["contype"] = "0"
        attrib["conaffinity"] = "0"
    if collision and friction is not None:
        attrib["friction"] = friction
    if not collision and material is not None:
        attrib["material"] = material
    return ET.Element("geom", attrib)


def _apply_recessed_well(
    root: ET.Element, edits: dict[str, Any], *, depth: float, peg_body: str
) -> None:
    inner = float(edits["inner_halfwidth_m"])
    mouth = float(edits["mouth_halfwidth_m"])
    lead_in = float(edits["lead_in_height_m"])
    if not 0.0 < lead_in < depth:
        raise ValueError(
            f"the lead-in is {lead_in * 1e3:.2f} mm and the well is {depth * 1e3:.2f} mm "
            "deep; 9.5.5 requires the chamfer not to consume the depth that defines "
            "cl_plus, so a lead-in at or above d is a contradiction rather than a "
            "tight fit"
        )
    if mouth <= inner:
        raise ValueError("the well mouth must be wider than the well interior")

    worldbody = _find(root, "worldbody", "worldbody")
    table_body = _find(worldbody, "./body[@name='table']", "table body")
    peg = _find(worldbody, f"./body[@name='{peg_body}']", f"{peg_body} body")
    collision = _find(table_body, "./geom[@name='table_collision']", "table_collision geom")
    visual = _find(table_body, "./geom[@name='table_visual']", "table_visual geom")

    table_pos = _floats(table_body.get("pos"), 3, "table body pos")
    half = _floats(collision.get("size"), 3, "table_collision size")
    geom_pos = _floats(collision.get("pos") or "0 0 0", 3, "table_collision pos")
    peg_pos = _floats(peg.get("pos"), 3, f"{peg_body} pos")
    friction = collision.get("friction")
    material = visual.get("material")

    # Everything below is in the table body's frame, which is where the two
    # geoms already live.
    centre = np.array([peg_pos[0] - table_pos[0], peg_pos[1] - table_pos[1], 0.0])
    top = geom_pos[2] + half[2]
    lo_corner = geom_pos[:2] - half[:2]
    hi_corner = geom_pos[:2] + half[:2]
    if np.any(centre[:2] - mouth <= lo_corner) or np.any(centre[:2] + mouth >= hi_corner):
        raise ValueError(
            f"the well mouth around {peg_body} reaches the table edge; the four boxes "
            "that restore the surface would be degenerate"
        )

    # 9.5.5, sentence two: the well floor geom is the table, moved down by d.
    collision.set("pos", " ".join(f"{v:.17g}" for v in geom_pos - np.array([0.0, 0.0, depth])))
    visual_pos = _floats(visual.get("pos") or "0 0 0", 3, "table_visual pos")
    visual.set("pos", " ".join(f"{v:.17g}" for v in visual_pos - np.array([0.0, 0.0, depth])))

    def emit(name: str, lo: np.ndarray, hi: np.ndarray) -> None:
        table_body.append(_box(name, lo, hi, collision=True, friction=friction, material=None))
        table_body.append(
            _box(f"{name}_visual", lo, hi, collision=False, friction=None, material=material)
        )

    # 9.5.5, sentence one: four boxes around the peg forming a square well. They
    # are the table surface restored everywhere except the opening.
    z_lo, z_hi = top - depth, top
    emit(
        "well_slab_xlo",
        np.array([lo_corner[0], lo_corner[1], z_lo]),
        np.array([centre[0] - mouth, hi_corner[1], z_hi]),
    )
    emit(
        "well_slab_xhi",
        np.array([centre[0] + mouth, lo_corner[1], z_lo]),
        np.array([hi_corner[0], hi_corner[1], z_hi]),
    )
    emit(
        "well_slab_ylo",
        np.array([centre[0] - mouth, lo_corner[1], z_lo]),
        np.array([centre[0] + mouth, centre[1] - mouth, z_hi]),
    )
    emit(
        "well_slab_yhi",
        np.array([centre[0] - mouth, centre[1] + mouth, z_lo]),
        np.array([centre[0] + mouth, hi_corner[1], z_hi]),
    )

    # The straight part of the wall, from the floor up to the foot of the
    # lead-in. It narrows the opening from the mouth half width to the inner
    # half width, which is what sets the 1 mm wall clearance G1 is specified by.
    wall_lo, wall_hi = top - depth, top - lead_in
    emit(
        "well_wall_xlo",
        np.array([centre[0] - mouth, centre[1] - mouth, wall_lo]),
        np.array([centre[0] - inner, centre[1] + mouth, wall_hi]),
    )
    emit(
        "well_wall_xhi",
        np.array([centre[0] + inner, centre[1] - mouth, wall_lo]),
        np.array([centre[0] + mouth, centre[1] + mouth, wall_hi]),
    )
    emit(
        "well_wall_ylo",
        np.array([centre[0] - inner, centre[1] - mouth, wall_lo]),
        np.array([centre[0] + inner, centre[1] - inner, wall_hi]),
    )
    emit(
        "well_wall_yhi",
        np.array([centre[0] - inner, centre[1] + inner, wall_lo]),
        np.array([centre[0] + inner, centre[1] + mouth, wall_hi]),
    )

    for axis, sign, tag, half_width in (
        (0, -1.0, "xlo", mouth),
        (0, +1.0, "xhi", mouth),
        (1, -1.0, "ylo", inner),
        (1, +1.0, "yhi", inner),
    ):
        for is_collision in (True, False):
            suffix = "" if is_collision else "_visual"
            table_body.append(
                _ramp(
                    f"well_leadin_{tag}{suffix}",
                    axis=axis,
                    sign=sign,
                    centre=centre,
                    inner=inner,
                    mouth=mouth,
                    lead_in=lead_in,
                    top_z=top,
                    half_width=half_width,
                    collision=is_collision,
                    friction=friction,
                    material=material,
                )
            )


def _apply_baffle(
    root: ET.Element, edits: dict[str, Any], *, height: float, nut_terminal_z_m: float
) -> None:
    """G2's baffle, at 9.5.5's Z = z*_nut + 0.01 + h + 0.005."""
    half = np.asarray(edits["baffle_halfsize_m"], dtype=np.float64)
    xy = np.asarray(edits["baffle_xy_m"], dtype=np.float64)
    z = nut_terminal_z_m + 0.01 + height + float(half[2])

    worldbody = _find(root, "worldbody", "worldbody")
    body = ET.SubElement(
        worldbody, "body", {"name": "baffle1", "pos": f"{xy[0]:.17g} {xy[1]:.17g} {z:.17g}"}
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": "baffle1_collision",
            "type": "box",
            "size": " ".join(f"{v:.17g}" for v in half),
            "group": "0",
        },
    )


def apply_variant(
    root: ET.Element,
    variant: GeomVariant,
    *,
    well_depth: float | None = None,
    baffle_height: float | None = None,
    nut_terminal_z_m: float | None = None,
    peg_body: str = "peg1",
) -> None:
    """Apply a 9.5.5 geometry level to an MJCF tree in place.

    `root` is any tree that contains the `pegs_arena` worldbody -- the arena on
    its own, or a robosuite `MujocoWorldBase` after the arena has been merged
    into it. G0 is a no-op, and it goes through this function rather than around
    it so that the three levels differ only in their edits.

    `well_depth` and `baffle_height` are required by the variants that use them
    and have no defaults; `well_depth_m` / `baffle_height_m` fetch them from the
    lock file.
    """
    edits = variant.mjcf_edits
    kind = edits.get("kind")
    if kind is None:
        if edits:
            raise ValueError(f"{variant.name}: edits without a `kind` are not applicable")
        return

    if kind not in ("recessed_well", "recessed_well_with_baffle"):
        raise NotImplementedError(f"{variant.name}: no editor for edit kind {kind!r}")

    if well_depth is None:
        raise ValueError(
            f"{variant.name} needs the well depth d and it has no default; 9.5.5 "
            "derives it from the F8 feasibility check, so pass `well_depth_m(task)`"
        )
    _apply_recessed_well(root, edits, depth=well_depth, peg_body=peg_body)

    if kind == "recessed_well_with_baffle":
        if baffle_height is None or nut_terminal_z_m is None:
            raise ValueError(
                f"{variant.name} needs the baffle height h and the nut's terminal "
                "height z*_nut, neither of which has a default (9.5.5)"
            )
        _apply_baffle(root, edits, height=baffle_height, nut_terminal_z_m=nut_terminal_z_m)


def build_arena(
    variant: GeomVariant,
    *,
    well_depth: float | None = None,
    baffle_height: float | None = None,
    nut_terminal_z_m: float | None = None,
    table_full_size: tuple[float, float, float] = (0.8, 0.8, 0.05),
    table_offset: tuple[float, float, float] = (0.0, 0.0, 0.82),
    peg_body: str = "peg1",
):
    """A `PegsArena` carrying the variant, for a standalone compile check.

    The table dimensions are `NutAssembly`'s, because the arena's own defaults
    put the table top at z = 0 and compile to a degenerate leg cylinder. Passing
    them here keeps the standalone check on the same geometry the task uses.
    """
    from robosuite.models.arenas import PegsArena

    arena = PegsArena(table_full_size=table_full_size, table_offset=table_offset)
    apply_variant(
        arena.root,
        variant,
        well_depth=well_depth,
        baffle_height=baffle_height,
        nut_terminal_z_m=nut_terminal_z_m,
        peg_body=peg_body,
    )
    return arena


def env_class_with_geom_variant(
    env_cls: type,
    variant: GeomVariant,
    *,
    well_depth: float | None = None,
    baffle_height: float | None = None,
    nut_terminal_z_m: float | None = None,
    peg_body: str = "peg1",
) -> type:
    """A subclass of a robosuite env whose arena carries the variant.

    The edit is applied in `_load_model`, after the arena has been merged and
    before `_initialize_sim` compiles anything. robosuite's `hard_reset` default
    is `True`, so `_load_model` runs again on every reset and the edit is
    re-applied to a freshly built tree -- which is why this is a model-build hook
    and not a write into a live `mjModel`.
    """

    class _WithGeomVariant(env_cls):  # type: ignore[valid-type, misc]
        def _load_model(self) -> None:
            super()._load_model()
            apply_variant(
                self.model.root,
                variant,
                well_depth=well_depth,
                baffle_height=baffle_height,
                nut_terminal_z_m=nut_terminal_z_m,
                peg_body=peg_body,
            )

    _WithGeomVariant.__name__ = f"{env_cls.__name__}_{variant.name}"
    _WithGeomVariant.__qualname__ = _WithGeomVariant.__name__
    return _WithGeomVariant


def make_env(
    spec: TaskSpec,
    family: str = "Panda",
    *,
    variant: str = "G0",
    well_depth: float | None = None,
    baffle_height: float | None = None,
    nut_terminal_z_m: float | None = None,
):
    """A live env for one (task, geometry variant), through both registries.

    `robosuite.make` cannot reach either layer on its own: 9.5.4's tight-fit
    environments are registered by `registry.tight`, and the G1/G2 arenas are
    subclasses built per call because they carry d and h, which are numbers that
    P0 measures rather than names that can be registered ahead of time. The class
    is instantiated directly, which is what `robosuite.make` does once it has
    looked the name up.
    """
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config
    from robosuite.environments import base as suite_base

    from portal_trace.registry.tasks import CONTROL_FREQ, HORIZON

    if spec.tight_variant_of is not None:
        from portal_trace.registry.tight import ensure_registered

        env_cls = ensure_registered(spec)
    else:
        env_cls = suite_base.REGISTERED_ENVS.get(spec.env_name)
        if env_cls is None:
            raise KeyError(
                f"robosuite {suite.__version__} does not register {spec.env_name!r}"
            )

    chosen = geom_variant(spec.task_id, variant)
    if chosen.mjcf_edits:
        peg = spec.inserted_body or "peg1"
        env_cls = env_class_with_geom_variant(
            env_cls,
            chosen,
            well_depth=well_depth,
            baffle_height=baffle_height,
            nut_terminal_z_m=nut_terminal_z_m,
            peg_body=peg,
        )

    kwargs = dict(spec.env_kwargs)
    kwargs.pop("horizon", None)
    return env_cls(
        robots=family,
        controller_configs=load_composite_controller_config(controller=None, robot=family),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=CONTROL_FREQ,
        horizon=HORIZON,
        ignore_done=True,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# cl_pm by stepped mj_collision (9.5.5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TerminalPose:
    """The pose x* at which cl_pm is evaluated.

    A pose and not just a position, because the harm asymmetry is a property of
    the terminal configuration: a square nut rotated by 45 degrees over a square
    well has a different downward clearance from one that is aligned.
    """

    body: str
    position: np.ndarray   # (3,) world
    quaternion: np.ndarray  # (4,) world, w x y z

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", np.asarray(self.position, dtype=np.float64))
        object.__setattr__(self, "quaternion", np.asarray(self.quaternion, dtype=np.float64))
        if self.position.shape != (3,):
            raise ValueError(f"position must be (3,), got {self.position.shape}")
        if self.quaternion.shape != (4,):
            raise ValueError(f"quaternion must be (4,), got {self.quaternion.shape}")
        norm = float(np.linalg.norm(self.quaternion))
        if not np.isfinite(norm) or abs(norm - 1.0) > 1e-9:
            raise ValueError(f"quaternion is not unit, |q| = {norm}")


@dataclass(frozen=True)
class ClearanceResult:
    """cl_+, cl_- and the asymmetry A of 8.1 / 9.5.5, in metres."""

    body: str
    direction: tuple[float, float, float]
    cl_plus: float
    cl_minus: float
    contact_at_pose: bool
    saturated_plus: bool
    saturated_minus: bool
    first_contact_plus: tuple[str, str] | None
    first_contact_minus: tuple[str, str] | None
    step_m: float
    max_steps: int

    @property
    def asymmetry(self) -> float:
        return self.cl_minus - self.cl_plus

    def to_json(self) -> dict[str, Any]:
        return {
            "body": self.body,
            "direction": list(self.direction),
            "cl_plus_m": self.cl_plus,
            "cl_minus_m": self.cl_minus,
            "asymmetry_m": self.asymmetry,
            "contact_at_pose": self.contact_at_pose,
            "saturated_plus": self.saturated_plus,
            "saturated_minus": self.saturated_minus,
            "first_contact_plus": list(self.first_contact_plus or ()),
            "first_contact_minus": list(self.first_contact_minus or ()),
            "step_m": self.step_m,
            "max_steps": self.max_steps,
        }


def _body_geom_ids(model: mujoco.MjModel, body_id: int) -> set[int]:
    start = int(model.body_geomadr[body_id])
    return set(range(start, start + int(model.body_geomnum[body_id])))


def _geom_support(model: mujoco.MjModel, data: mujoco.MjData, gid: int, v: np.ndarray) -> float:
    """Half extent of one geom along `v`, from the primitive's support function.

    Exact for the shapes robosuite's receptacles and parts are made of, and
    conservative for the rest. The alternative, `geom_rbound`, is the radius of
    the bounding sphere: on the square nut's ring that is 55 mm, which would
    swamp every millimetre-scale quantity computed from it.
    """
    rot = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
    size = np.asarray(model.geom_size[gid], dtype=np.float64)
    w = rot.T @ v  # v in the geom's own axes
    kind = int(model.geom_type[gid])
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return float(np.abs(w) @ size[:3])
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return float(size[0] * np.hypot(w[0], w[1]) + size[1] * abs(w[2]))
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return float(size[1] * abs(w[2]) + size[0])
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(size[0])
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return float(np.linalg.norm(w * size[:3]))
    # Meshes and height fields have no closed form here. The bounding sphere is
    # an over-estimate, which is the safe direction for every caller: it lifts a
    # body further clear than needed and never leaves it interpenetrating.
    return float(model.geom_rbound[gid])


def body_extent_along(
    model: mujoco.MjModel, data: mujoco.MjData, body_name: str, v: np.ndarray
) -> tuple[float, float]:
    """`(min, max)` of `x . v` over a body's *collision* geoms, in world units.

    Visual geoms are excluded: an extent is about what can touch something, and
    robosuite's decoration reaches well outside the physical body.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise KeyError(f"no body named {body_name!r} in this model")
    v = np.asarray(v, dtype=np.float64)
    v = v / np.linalg.norm(v)
    lo, hi = np.inf, -np.inf
    for gid in sorted(_body_geom_ids(model, body_id)):
        if int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0:
            continue
        centre = float(np.asarray(data.geom_xpos[gid], dtype=np.float64) @ v)
        half = _geom_support(model, data, gid, v)
        lo, hi = min(lo, centre - half), max(hi, centre + half)
    if not np.isfinite(lo):
        raise ValueError(f"{body_name} has no collision geoms, so it has no extent")
    return lo, hi


def _free_joint_qpos_adr(model: mujoco.MjModel, body_id: int, body_name: str) -> int:
    adr = int(model.body_jntadr[body_id])
    if adr < 0 or int(model.jnt_type[adr]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError(
            f"{body_name} has no free joint, so it cannot be translated to probe the "
            "clearance. cl_pm is defined for the part that moves at the terminal pose"
        )
    return int(model.jnt_qposadr[adr])


def _sweep(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    qadr: int,
    origin: np.ndarray,
    direction: np.ndarray,
    moving: set[int],
    ignore: set[int],
    step: float,
    max_steps: int,
) -> tuple[float, bool, tuple[str, str] | None]:
    """Translate along `direction` until the first contact; 9.5.5's algorithm.

    Returns the last penetration-free offset, whether the sweep ran out of range,
    and the geom pair that stopped it. `mj_kinematics` before every
    `mj_collision` is mandatory: the broad phase reads `geom_xpos`, and skipping
    it would collide the geometry at the previous pose.
    """
    for k in range(1, max_steps + 1):
        data.qpos[qadr : qadr + 3] = origin + k * step * direction
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        hit = _first_contact(model, data, moving, ignore)
        if hit is not None:
            return (k - 1) * step, False, hit
    return max_steps * step, True, None


def _first_contact(
    model: mujoco.MjModel, data: mujoco.MjData, moving: set[int], ignore: set[int]
) -> tuple[str, str] | None:
    for i in range(int(data.ncon)):
        contact = data.contact[i]
        if float(contact.dist) > 0.0:
            continue
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if (g1 in moving) == (g2 in moving):
            continue  # neither geom is ours, or both are (a self contact)
        other = g2 if g1 in moving else g1
        if other in ignore:
            continue
        return (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"geom{g1}",
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"geom{g2}",
        )
    return None


def measure_clearance(
    env: Any,
    terminal_pose: TerminalPose,
    v: np.ndarray,
    *,
    step: float = CLEARANCE_STEP_M,
    max_steps: int = CLEARANCE_MAX_STEPS,
    ignore_bodies: tuple[str, ...] = (),
) -> ClearanceResult:
    """cl_+ and cl_- at the terminal pose, along +-v (9.5.5).

    The body named by `terminal_pose` is placed at x*, then translated in steps
    of `step` along +v and along -v until `mj_collision` reports its first
    contact with anything outside itself. The answer is the last offset with no
    contact, minimised over every geom of the body, which is why this is a swept
    collision query and not a ray cast:

    `mj_ray(m, d, pnt, vec, geomgroup, flg_static, bodyexclude, geomid)`
    intersects a *single line* with the scene and returns the parameter of the
    nearest visible surface. The square nut is four boxes forming a hollow ring,
    so a ray from the hole axis passes straight through the hole and answers a
    question about the table underneath rather than about the nut's freedom to
    descend. 9.5.5 keeps `mj_ray` only as a cross-check for a convex single geom
    (Stack's cubeA), and notes that MuJoCo 3.3.2 still has its eight-argument
    signature -- the `normal` output was added in 3.5.0.

    `env.sim` is left exactly as it was found: the probe writes `qpos`, and the
    original is restored before returning.
    """
    v = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        raise ValueError("the terminal compensation direction v must be non-degenerate")
    v = v / norm
    if step <= 0.0 or max_steps <= 0:
        raise ValueError("the sweep needs a positive step and at least one step")

    model = env.sim.model._model
    data = env.sim.data._data

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, terminal_pose.body)
    if body_id < 0:
        raise KeyError(f"no body named {terminal_pose.body!r} in this environment")
    moving = _body_geom_ids(model, body_id)
    if not moving:
        raise ValueError(f"{terminal_pose.body} has no geoms, so it has no clearance")

    ignore: set[int] = set()
    for name in ignore_bodies:
        other_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if other_id < 0:
            raise KeyError(f"no body named {name!r} to ignore")
        ignore |= _body_geom_ids(model, other_id)

    qadr = _free_joint_qpos_adr(model, body_id, terminal_pose.body)
    saved = np.array(data.qpos, dtype=np.float64, copy=True)
    try:
        data.qpos[qadr : qadr + 3] = terminal_pose.position
        data.qpos[qadr + 3 : qadr + 7] = terminal_pose.quaternion
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        contact_at_pose = _first_contact(model, data, moving, ignore) is not None

        origin = np.asarray(terminal_pose.position, dtype=np.float64)
        cl_plus, sat_plus, hit_plus = _sweep(
            model,
            data,
            qadr=qadr,
            origin=origin,
            direction=v,
            moving=moving,
            ignore=ignore,
            step=step,
            max_steps=max_steps,
        )
        cl_minus, sat_minus, hit_minus = _sweep(
            model,
            data,
            qadr=qadr,
            origin=origin,
            direction=-v,
            moving=moving,
            ignore=ignore,
            step=step,
            max_steps=max_steps,
        )
    finally:
        data.qpos[:] = saved
        mujoco.mj_forward(model, data)

    return ClearanceResult(
        body=terminal_pose.body,
        direction=(float(v[0]), float(v[1]), float(v[2])),
        cl_plus=cl_plus,
        cl_minus=cl_minus,
        contact_at_pose=contact_at_pose,
        saturated_plus=sat_plus,
        saturated_minus=sat_minus,
        first_contact_plus=hit_plus,
        first_contact_minus=hit_minus,
        step_m=step,
        max_steps=max_steps,
    )


# ---------------------------------------------------------------------------
# Compile check
# ---------------------------------------------------------------------------


def compile_and_step(
    variant: GeomVariant,
    *,
    well_depth: float | None = None,
    baffle_height: float | None = None,
    nut_terminal_z_m: float | None = None,
    n_steps: int = 100,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Build the arena at this level, compile it and integrate it."""
    arena = build_arena(
        variant,
        well_depth=well_depth,
        baffle_height=baffle_height,
        nut_terminal_z_m=nut_terminal_z_m,
    )
    model = arena.get_model(mode="mujoco")
    data = mujoco.MjData(model)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise AssertionError(f"{variant.name}: the model diverged within {n_steps} steps")
    return model, data


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="NutAssemblySquareTight-v0")
    p.add_argument(
        "--variants", nargs="*", default=None, help="default: every variant the task has"
    )
    p.add_argument(
        "--well-depth-m",
        type=float,
        default=None,
        help="override d instead of reading geom.clearance.<task>.G1.plus from the lock",
    )
    p.add_argument(
        "--baffle-height-m",
        type=float,
        default=None,
        help="override h instead of reading geom.clearance.<task>.G2.minus from the lock",
    )
    p.add_argument(
        "--nut-terminal-z",
        type=float,
        default=None,
        help="z*_nut, the nut's height at the terminal pose, needed by G2",
    )
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    spec = task(args.task)
    names = args.variants or [v.name for v in spec.geom_variants]
    depth = args.well_depth_m if args.well_depth_m is not None else None
    height = args.baffle_height_m if args.baffle_height_m is not None else None

    report: dict[str, Any] = {"task": args.task, "variants": {}}
    for name in names:
        variant = geom_variant(args.task, name)
        needs_d = variant.mjcf_edits.get("kind") is not None
        d = depth if depth is not None else (well_depth_m(args.task) if needs_d else None)
        needs_h = variant.mjcf_edits.get("kind") == "recessed_well_with_baffle"
        h = height if height is not None else (baffle_height_m(args.task) if needs_h else None)
        model, data = compile_and_step(
            variant,
            well_depth=d,
            baffle_height=h,
            nut_terminal_z_m=args.nut_terminal_z,
            n_steps=args.steps,
        )
        report["variants"][name] = {
            "nbody": int(model.nbody),
            "ngeom": int(model.ngeom),
            "steps": args.steps,
            "sim_time_s": float(data.time),
            "well_depth_m": d,
            "baffle_height_m": h,
        }
        print(
            f"[{name}] compiled: nbody={model.nbody} ngeom={model.ngeom}; "
            f"stepped {args.steps} times to t={data.time:.3f} s; "
            f"d={d if d is None else f'{d * 1e3:.3f} mm'} "
            f"h={h if h is None else f'{h * 1e3:.3f} mm'}"
        )

    if args.out is not None:
        import json

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[write] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
