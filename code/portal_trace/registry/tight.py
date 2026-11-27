"""The tight-fit environments of 9.5.4: `NutAssembly{Square,Round}Tight-v0`.

Why they exist. The native task's effective tolerance is not its fit clearance.
`NutAssembly._check_success` reads

    self.objects_on_pegs[i] = int(self.on_peg(obj_pos, i) and r_reach < 0.6)

with `on_peg` accepting `abs(dx) < 0.03 and abs(dy) < 0.03 and z < table + 0.05`,
so the native criterion is a 30 mm box, thirty times the 1 mm clearance the
geometry actually has. kappa = ||c_delta(x*)|| / tol would then be quoted against
a number that no part of the physics uses, and the Goldilocks window would force
defect amplitudes an order of magnitude larger than the ones the proposal wants
to make claims about.

9.5.4 makes four changes, and each one fails differently if left out:

1. The peg shrinks the clearance to 1.00 mm. `peg1`'s two box geoms go from
   `0.016 0.016 0.1` to `0.02175 0.02175 0.03` against the square nut's 0.02275
   hole half width; `peg2`'s two cylinders go from `0.02 0.1` to `0.0302 0.03`
   against the round nut's 0.0312 hole radius. The *peg* moves and not the nut,
   because the nut is the grasped object: changing its size would move the grasp
   geometry and the inertia at the same time, and the tolerance change would come
   out confounded with a demonstration-difficulty change.
2. A 45 degree lead-in on the peg top turns the hit from a measure-zero alignment
   event into a process with a capture basin: 5 mm of lateral allowance at the
   entry, converging linearly to the fit clearance over the 4 mm of chamfer. The
   fit clearance itself is untouched, so kappa's denominator does not move.
3. The success criterion is replaced by the three conditions of 9.5.4, so that
   what counts as success and what divides kappa are the same number.
4. The registered name changes. This is not bookkeeping: arena geoms and the
   success function both differ from the published environment, so success rates
   here are not comparable with robomimic or MimicGen figures, and the "200
   demonstrations per cell is the same scale as proficient-human" justification
   of 9.6.1 does not survive the change either.

The chamfer is a convex mesh with inline vertices rather than a stack of plates.
MuJoCo has no cone or frustum primitive -- 9.5.4's `type="cone"` does not exist
in the format -- and a frustum is convex, so the convex hull of its eight (square
peg) or 2 x 32 (round peg) vertices is the exact solid rather than an
approximation of it.

Specification arithmetic that does not survive contact with the MJCF. 9.5.4 says
the shortened peg stands about 30 mm above the table, of which 4 mm is chamfer
and 26 mm parallel. `peg1`'s body sits at z = 0.85 and the table surface is at
0.82, so a half length of 0.03 puts the peg top at 0.88 and leaves 60 mm exposed,
not 30. The 26 mm figure inherits the error. Nothing downstream breaks -- more
parallel section is strictly better for the 15 mm descent the success criterion
asks for -- but the number is wrong where it is printed, and the realised
exposure is reported by `describe` rather than assumed.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import threading
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from portal_trace.registry.tasks import TASKS, TaskSpec, task

__all__ = [
    "TightFit",
    "TIGHT_FITS",
    "CHAMFER_HEIGHT_M",
    "ENTRY_LATERAL_ALLOWANCE_M",
    "REQUIRED_DESCENT_M",
    "R_REACH_MAX",
    "tight_fit",
    "apply_tight_fit",
    "tight_env_class",
    "ensure_registered",
    "mjcf_sha256",
    "success_fn_sha256",
    "describe",
    "main",
]

# 9.5.4, fixed by the proposal and frozen once the go/no-go passes: the chamfer is
# part of the task definition, not a tunable. The escalation ladder for a failed
# go/no-go (4 mm -> 6 mm, then the entry allowance 5 mm -> 8 mm) is in 9.5.4 and
# is applied by editing these, never at call sites.
CHAMFER_HEIGHT_M = 0.004
ENTRY_LATERAL_ALLOWANCE_M = 0.005

# 9.5.4's success conditions 2 and 3.
REQUIRED_DESCENT_M = 0.015
R_REACH_MAX = 0.6

# The number of sides the round peg's frustum hull is built from. A circle has no
# exact polyhedral representation, so this is the one place the geometry is an
# approximation; 32 sides put the inscribed-circle error at 1.5 um against a
# 1.00 mm clearance, four orders of magnitude below the quantity being measured.
_CONE_SIDES = 32

# `preflight_p0 --workers N` builds the same tight-fit class from concurrent
# cells. Keep the check/create/publish sequence atomic so a first-use race cannot
# leave `suite.make` observing a partially registered task.
_REGISTRATION_LOCK = threading.Lock()


@dataclass(frozen=True)
class TightFit:
    """One tight-fit environment, field by field from 9.5.4's table."""

    task_id: str
    base_env: str
    peg_body: str
    peg_index: int              # 0 for peg1 / square, 1 for peg2 / round
    nut_body: str
    native_size: tuple[float, ...]
    tight_size: tuple[float, ...]
    hole_halfwidth_m: float     # the nut's inner half width or radius

    def __post_init__(self) -> None:
        if len(self.native_size) != len(self.tight_size):
            raise ValueError(
                f"{self.task_id}: the tight size has to have the same arity as the "
                f"native one, got {self.tight_size} against {self.native_size}"
            )
        if self.tol <= 0.0:
            raise ValueError(
                f"{self.task_id}: the peg half width {self.peg_halfwidth} is at or "
                f"above the hole's {self.hole_halfwidth_m}; the nut cannot go on"
            )

    @property
    def peg_halfwidth(self) -> float:
        return float(self.tight_size[0])

    @property
    def peg_halflength(self) -> float:
        return float(self.tight_size[-1])

    @property
    def tol(self) -> float:
        """9.5.4's fit tolerance, the denominator of kappa."""
        return self.hole_halfwidth_m - self.peg_halfwidth

    @property
    def chamfer_top_halfwidth(self) -> float:
        """45 degrees: the taper loses exactly its own height in half width."""
        return self.peg_halfwidth - CHAMFER_HEIGHT_M

    @property
    def is_round(self) -> bool:
        return len(self.tight_size) == 2

    def check_entry_allowance(self) -> None:
        """The chamfer top must clear the hole by 9.5.4's lateral allowance."""
        got = self.hole_halfwidth_m - self.chamfer_top_halfwidth
        if abs(got - ENTRY_LATERAL_ALLOWANCE_M) > 1e-9:
            raise ValueError(
                f"{self.task_id}: the chamfer top leaves {got * 1e3:.3f} mm of lateral "
                f"allowance, 9.5.4 fixes it at {ENTRY_LATERAL_ALLOWANCE_M * 1e3:.1f} mm. "
                "The allowance, the chamfer height and the fit clearance are one "
                "system: at 45 degrees the first two determine the third"
            )


TIGHT_FITS: dict[str, TightFit] = {
    fit.task_id: fit
    for fit in (
        TightFit(
            task_id="NutAssemblySquareTight-v0",
            base_env="NutAssembly",
            peg_body="peg1",
            peg_index=0,
            nut_body="SquareNut_main",
            native_size=(0.016, 0.016, 0.1),
            tight_size=(0.02175, 0.02175, 0.03),
            hole_halfwidth_m=0.02275,
        ),
        TightFit(
            task_id="NutAssemblyRoundTight-v0",
            base_env="NutAssembly",
            peg_body="peg2",
            peg_index=1,
            nut_body="RoundNut_main",
            native_size=(0.02, 0.1),
            tight_size=(0.0302, 0.03),
            hole_halfwidth_m=0.0312,
        ),
    )
}


def tight_fit(task_id: str) -> TightFit:
    try:
        return TIGHT_FITS[task_id]
    except KeyError:
        raise KeyError(
            f"{task_id!r} is not a tight-fit environment; 9.5.4 defines "
            f"{sorted(TIGHT_FITS)}"
        ) from None


# ---------------------------------------------------------------------------
# MJCF
# ---------------------------------------------------------------------------


def _fmt(values: Any) -> str:
    return " ".join(f"{float(v):.17g}" for v in np.atleast_1d(np.asarray(values)))


def _frustum_vertices(fit: TightFit) -> np.ndarray:
    """The chamfer solid, in the geom's own frame, base at z = 0.

    Returned as vertices rather than as a primitive because MuJoCo has none that
    tapers. The hull of these points is the frustum exactly, for the square peg;
    for the round one it is the inscribed prismatic approximation described in
    `_CONE_SIDES`.
    """
    a, b, h = fit.peg_halfwidth, fit.chamfer_top_halfwidth, CHAMFER_HEIGHT_M
    if fit.is_round:
        angles = np.arange(_CONE_SIDES) * (2.0 * np.pi / _CONE_SIDES)
        ring = np.column_stack([np.cos(angles), np.sin(angles)])
        return np.vstack(
            [
                np.column_stack([a * ring, np.zeros(_CONE_SIDES)]),
                np.column_stack([b * ring, np.full(_CONE_SIDES, h)]),
            ]
        )
    corners = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])
    return np.vstack(
        [
            np.column_stack([a * corners, np.zeros(4)]),
            np.column_stack([b * corners, np.full(4, h)]),
        ]
    )


def _peg_geoms(root: ET.Element, fit: TightFit) -> tuple[ET.Element, list[ET.Element]]:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("the arena has no worldbody")
    body = worldbody.find(f"./body[@name='{fit.peg_body}']")
    if body is None:
        present = sorted(b.get("name", "?") for b in worldbody.findall("./body"))
        raise KeyError(f"no body named {fit.peg_body!r} in this arena; it has {present}")
    geoms = [g for g in body.findall("./geom") if g.get("type") in ("box", "cylinder")]
    if len(geoms) != 2:
        raise ValueError(
            f"{fit.peg_body} has {len(geoms)} box/cylinder geoms, 9.5.4 expects the "
            "collision/visual pair that ships in pegs_arena.xml. The arena has "
            "changed shape and the tight-fit edit cannot be applied blind"
        )
    return body, geoms


def apply_tight_fit(root: ET.Element, fit: TightFit) -> None:
    """Resize the peg and add its lead-in, in place, on a parsed arena tree.

    Idempotent by refusing rather than by skipping: applying the edit twice would
    put two chamfers on one peg, and a silent second application is exactly the
    kind of thing that produces a geometry nobody can reconstruct from the config.
    """
    fit.check_entry_allowance()
    body, geoms = _peg_geoms(root, fit)

    for geom in geoms:
        size = tuple(float(v) for v in (geom.get("size") or "").split())
        if size == tuple(fit.tight_size):
            raise ValueError(
                f"{fit.peg_body} is already at the tight size {fit.tight_size}; "
                "applying 9.5.4's edit again would stack a second chamfer on the peg"
            )
        if size != tuple(fit.native_size):
            raise ValueError(
                f"{fit.peg_body} has size {size}, and 9.5.4's edit is specified "
                f"against the shipped {fit.native_size}. The arena has changed and "
                "the 1.00 mm clearance cannot be reached by this substitution"
            )
        geom.set("size", _fmt(fit.tight_size))

    collision, visual = geoms
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    mesh_name = f"{fit.peg_body}_leadin"
    if asset.find(f"./mesh[@name='{mesh_name}']") is not None:
        raise ValueError(f"{mesh_name} is already in the arena assets")
    ET.SubElement(
        asset, "mesh", {"name": mesh_name, "vertex": _fmt(_frustum_vertices(fit).reshape(-1))}
    )

    # The chamfer sits on the peg's top face. Its base half width equals the
    # shaft's, so the two solids meet with no lip for the nut to catch on.
    base_z = fit.peg_halflength
    shared = {"type": "mesh", "mesh": mesh_name, "pos": _fmt((0.0, 0.0, base_z))}
    lead_collision = {**shared, "name": f"{fit.peg_body}_leadin_collision", "group": "0"}
    if (friction := collision.get("friction")) is not None:
        lead_collision["friction"] = friction
    lead_visual = {
        **shared,
        "name": f"{fit.peg_body}_leadin_visual",
        "group": "1",
        "contype": "0",
        "conaffinity": "0",
    }
    if (material := visual.get("material")) is not None:
        lead_visual["material"] = material
    body.append(ET.Element("geom", lead_collision))
    body.append(ET.Element("geom", lead_visual))


# ---------------------------------------------------------------------------
# The success criterion
# ---------------------------------------------------------------------------


_WORLD_Z = np.array([0.0, 0.0, 1.0])


def _lowest_point_z(model: Any, data: Any, body_name: str) -> float:
    """World z of the lowest point of a body's collision geoms.

    The support function of each primitive along -z, not a bounding sphere: the
    descent condition is a 15 mm threshold and `geom_rbound` on the square nut's
    ring is 55 mm, which would swamp it.
    """
    from portal_trace.registry.geometry import body_extent_along

    return body_extent_along(model, data, body_name, _WORLD_Z)[0]


def tight_env_class(base_cls: type, fit: TightFit) -> type:
    """A subclass of the robosuite env carrying 9.5.4's geometry and criterion."""

    class _Tight(base_cls):  # type: ignore[valid-type, misc]
        """9.5.4's tight-fit variant. Not comparable with the published task."""

        tight_fit_spec = fit

        def _load_model(self) -> None:
            super()._load_model()
            apply_tight_fit(self.model.root, fit)

        def _reset_internal(self) -> None:
            super()._reset_internal()
            # The descent datum of condition 2 is the underside height at the
            # *first* entry into the chamfer, so it is episode state and has to be
            # cleared here. robosuite rebuilds the sim on a hard reset, so the geom
            # ids are re-resolved on the same pass.
            self._entry_underside_z: float | None = None

        def tight_fit_state(self) -> dict[str, float | bool | None]:
            """The three conditions, as numbers, so a failure says which one failed."""
            model, data = self.sim.model._model, self.sim.data._data
            peg_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, fit.peg_body)
            nut_id = self.obj_body_id[self.nuts[fit.peg_index].name]
            peg_pos = np.asarray(data.xpos[peg_id], dtype=np.float64)
            nut_pos = np.asarray(self.sim.data.body_xpos[nut_id], dtype=np.float64)

            radial = float(np.hypot(*(nut_pos[:2] - peg_pos[:2])))
            underside = _lowest_point_z(model, data, fit.nut_body)
            chamfer_top = float(peg_pos[2]) + fit.peg_halflength + CHAMFER_HEIGHT_M

            # Entry: the underside has passed the top of the chamfer while the nut
            # is inside the capture basin the chamfer provides.
            if (
                self._entry_underside_z is None
                and underside <= chamfer_top
                and radial <= ENTRY_LATERAL_ALLOWANCE_M + fit.tol
            ):
                self._entry_underside_z = underside

            descent = (
                None if self._entry_underside_z is None else self._entry_underside_z - underside
            )
            dist = min(
                float(
                    np.linalg.norm(
                        self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]] - nut_pos
                    )
                )
                for arm in self.robots[0].arms
            )
            r_reach = float(1.0 - np.tanh(10.0 * dist))
            return {
                "radial_m": radial,
                "tol_m": fit.tol,
                "radial_ok": radial < fit.tol,
                "underside_z": underside,
                "chamfer_top_z": chamfer_top,
                "entry_underside_z": self._entry_underside_z,
                "descent_m": descent,
                "descent_ok": descent is not None and descent >= REQUIRED_DESCENT_M,
                "r_reach": r_reach,
                "released_ok": r_reach < R_REACH_MAX,
            }

        def _check_success(self) -> bool:
            """9.5.4's three conditions, all required.

            Condition 3 is not a formality. Without it a nut held in the gripper at
            the right place scores as success, and that state involves no insertion
            at all -- while the terminal contact phase is precisely where kappa and
            the harm asymmetry are evaluated.
            """
            state = self.tight_fit_state()
            success = bool(state["radial_ok"] and state["descent_ok"] and state["released_ok"])
            self.objects_on_pegs[fit.peg_index] = int(success)
            return success

    _Tight.__name__ = fit.task_id.replace("-", "_").replace(".", "_")
    _Tight.__qualname__ = _Tight.__name__
    return _Tight


# ---------------------------------------------------------------------------
# Registration and provenance
# ---------------------------------------------------------------------------


def ensure_registered(spec: TaskSpec) -> type:
    """Register `spec.env_name` with robosuite, once, and return the class.

    9.5.4 requires the name to differ from the published one, so `suite.make`
    cannot find these until this has run. Callers wire it explicitly rather than
    relying on an import side effect: a registration that happens because a module
    was imported is a registration that can silently not happen.
    """
    import robosuite as suite
    from robosuite.environments import base as suite_base

    fit = tight_fit(spec.task_id)
    registry = suite_base.REGISTERED_ENVS
    with _REGISTRATION_LOCK:
        existing = registry.get(fit.task_id)
        if existing is not None:
            return existing

        base_cls = registry.get(fit.base_env)
        if base_cls is None:
            raise KeyError(
                f"robosuite {suite.__version__} does not register {fit.base_env!r}, which "
                f"{fit.task_id} is built from"
            )
        # robosuite registers by class name through EnvMeta, so the subclass has to
        # carry the registered name before the metaclass sees it.
        cls = tight_env_class(base_cls, fit)
        cls.__name__ = fit.task_id
        registry[fit.task_id] = cls
        return cls


def mjcf_sha256(spec: TaskSpec) -> str:
    """The hash 9.5.4 puts in the registry, taken on the edited arena.

    Computed on the canonicalised tree rather than on a serialisation of the live
    `mjModel`: the model carries compiler defaults and pointer-order artefacts
    that differ between MuJoCo builds, and the object 9.5.4 wants pinned is the
    geometry description.
    """
    import robosuite
    from robosuite.models.arenas import PegsArena

    fit = tight_fit(spec.task_id)
    del robosuite  # imported for the side effect of resolving the asset root
    arena = PegsArena()
    apply_tight_fit(arena.root, fit)
    canonical = ET.tostring(arena.root, encoding="utf-8")
    return hashlib.sha256(canonical).hexdigest()


def success_fn_sha256(spec: TaskSpec) -> str:
    """The hash of the replaced criterion, over its source and its constants."""
    fit = tight_fit(spec.task_id)
    payload = json.dumps(
        {
            "source": inspect.getsource(tight_env_class),
            "lowest_point": inspect.getsource(_lowest_point_z),
            "tol_m": fit.tol,
            "required_descent_m": REQUIRED_DESCENT_M,
            "r_reach_max": R_REACH_MAX,
            "chamfer_height_m": CHAMFER_HEIGHT_M,
            "entry_lateral_allowance_m": ENTRY_LATERAL_ALLOWANCE_M,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def describe(spec: TaskSpec) -> dict[str, Any]:
    """Everything about one tight-fit environment that 9.5.4 asks to be recorded."""
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config

    fit = tight_fit(spec.task_id)
    ensure_registered(spec)
    kwargs = dict(spec.env_kwargs)
    kwargs.pop("horizon", None)
    env = suite.make(
        spec.env_name,
        robots="Panda",
        controller_configs=load_composite_controller_config(controller=None, robot="Panda"),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=400,
        ignore_done=True,
        **kwargs,
    )
    try:
        env.reset()
        model, data = env.sim.model._model, env.sim.data._data
        peg_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, fit.peg_body)
        peg_z = float(data.xpos[peg_id][2])
        table_z = float(env.table_offset[2])
        shaft_top = peg_z + fit.peg_halflength
        state = env.tight_fit_state()
        return {
            "task_id": fit.task_id,
            "peg_body": fit.peg_body,
            "native_size": list(fit.native_size),
            "tight_size": list(fit.tight_size),
            "tol_m": fit.tol,
            "hole_halfwidth_m": fit.hole_halfwidth_m,
            "chamfer_height_m": CHAMFER_HEIGHT_M,
            "chamfer_top_halfwidth_m": fit.chamfer_top_halfwidth,
            "entry_lateral_allowance_m": fit.hole_halfwidth_m - fit.chamfer_top_halfwidth,
            "table_surface_z": table_z,
            "shaft_top_z": shaft_top,
            "chamfer_top_z": shaft_top + CHAMFER_HEIGHT_M,
            "exposed_above_table_m": shaft_top + CHAMFER_HEIGHT_M - table_z,
            "parallel_section_above_table_m": shaft_top - table_z,
            "required_descent_m": REQUIRED_DESCENT_M,
            "mjcf_sha256": mjcf_sha256(spec),
            "success_fn_sha256": success_fn_sha256(spec),
            "state_at_reset": state,
            "success_at_reset": bool(env._check_success()),
        }
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", nargs="*", default=sorted(TIGHT_FITS))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    report: dict[str, Any] = {}
    for task_id in args.task:
        spec = task(task_id)
        info = describe(spec)
        report[task_id] = info
        print(f"\n=== {task_id} ===")
        for key, value in info.items():
            if key == "state_at_reset":
                continue
            if isinstance(value, float) and key.endswith("_m"):
                print(f"  {key:36s} {value * 1e3:.4f} mm")
            else:
                print(f"  {key:36s} {value}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\n[write] {args.out}")
    return 0


if _unlisted := set(TIGHT_FITS) - set(TASKS):
    raise AssertionError(
        "a tight-fit environment is defined here that the task registry does not "
        f"list: {sorted(_unlisted)}"
    )

if _unbuilt := {t for t, s in TASKS.items() if s.tight_variant_of is not None} - set(TIGHT_FITS):
    raise AssertionError(
        "the task registry declares a tight-fit variant with no construction here: "
        f"{sorted(_unbuilt)}. 9.5.4 defines these by their MJCF edit, so a task that "
        "claims to be one and has no edit would silently run the native geometry "
        "under the tight-fit name"
    )


if __name__ == "__main__":
    raise SystemExit(main())
