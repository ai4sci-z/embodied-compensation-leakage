"""W1 item 1: the fit tolerance tol(tau) of every task (9.5.4).

tol(tau) is the minimum radial clearance between the inserted part and the hole
wall, in the plane normal to the insertion direction at the terminal pose,
computed from the compiled model's geom sizes and poses. It is the denominator
of kappa, so every dose statement, the Goldilocks criterion and the T8 decision
set are downstream of this script -- which is why 9.8.5 puts it first in the W1
order, ahead of any defect level sweep.

The correctness check is fixed by the proposal: the script must reproduce the
hand computation of 9.5.4 on the two nut tasks, 6.75 mm and 11.2 mm, to within
0.01 mm. If it does not, the four mesh tolerances it produces may not be used.

`mj_ray` is deliberately not used (9.5.5): a hollow ring lets a ray from the
hole axis pass straight through the opening, so the clearance is computed per
geom analytically and minimised, which is what the ring's inner boundary
actually is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from portal_trace.env.versions import Backend, check_pinned_versions, fingerprint
from portal_trace.lockfile import REQUIRED_MEASUREMENTS, LockFile
from portal_trace.registry.tasks import (
    TASKS,
    TaskSpec,
    TerminalPoseRule,
    TolKind,
    TolRule,
    task,
)
from portal_trace.scripts.geometry2d import (
    CrossSection,
    cross_section_of_geom,
    first_contact_distance,
    min_contact_distance,
    plane_basis,
)

__all__ = ["TolResult", "compute_tol", "main", "HAND_COMPUTED_MM", "HAND_CHECK_TOL_MM"]

# 9.5.4, hand computed from the MJCF and used as the script's only correctness
# check. Both are per-axis values: for the square nut the axis direction is also
# the minimum, for the round nut the octagon's diagonal wall sits 0.024 mm
# closer, which the script reports separately.
HAND_COMPUTED_MM: dict[str, float] = {
    "NutAssemblySquare": 6.75,
    "NutAssemblyRound": 11.2,
}
HAND_CHECK_TOL_MM = 0.01

# Directions sampled in the normal plane. The four axes and four diagonals are
# what 9.5.4 asks to report; the dense sweep exists because a mesh cross-section
# can have its minimum anywhere, and reporting only eight directions on ToolHang
# would overstate the clearance.
_N_DENSE = 720


@dataclass(frozen=True)
class TolResult:
    task_id: str
    tol_kind: TolKind
    tol_axis_m: float
    tol_diagonal_m: float
    tol_min_m: float
    argmin_direction_deg: float
    r_hole_axis_m: float
    r_inserted_axis_m: float
    inserted_geoms: tuple[str, ...]
    hole_geoms: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "tol_kind": self.tol_kind.value,
            "tol_mm": self.tol_min_m * 1e3,
            "tol_axis_mm": self.tol_axis_m * 1e3,
            "tol_diagonal_mm": self.tol_diagonal_m * 1e3,
            "tol_argmin_deg": self.argmin_direction_deg,
            "r_hole_axis_mm": self.r_hole_axis_m * 1e3,
            "r_inserted_axis_mm": self.r_inserted_axis_m * 1e3,
            "inserted_geoms": list(self.inserted_geoms),
            "hole_geoms": list(self.hole_geoms),
        }


def _body_geoms(model: mujoco.MjModel, body_name: str) -> tuple[int, ...]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)
        ]
        raise KeyError(f"no body named {body_name!r}; model has {sorted(n for n in names if n)}")
    start = int(model.body_geomadr[body_id])
    count = int(model.body_geomnum[body_id])
    ids = tuple(range(start, start + count))
    # Visual-only geoms (contype == conaffinity == 0) do not participate in the
    # fit and are excluded; in pegs_arena every peg carries one collision geom
    # and one identical visual twin, and counting both changes nothing but makes
    # the report confusing.
    return tuple(
        g for g in ids if int(model.geom_contype[g]) or int(model.geom_conaffinity[g])
    )


def _axial_span(
    model: mujoco.MjModel, data: mujoco.MjData, geom_ids, basis
) -> tuple[float, float]:
    """Extent of the given geoms along the insertion axis, world frame.

    Uses MuJoCo's own bounding radius, which is exact enough for a containment
    test and does not depend on the geom type.
    """
    lo, hi = np.inf, -np.inf
    for g in geom_ids:
        centre = float((np.asarray(data.geom_xpos[g]) - basis.origin) @ basis.axis)
        r = float(model.geom_rbound[g])
        lo = min(lo, centre - r)
        hi = max(hi, centre + r)
    return lo, hi


def _spans_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def place_at_terminal_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec: TaskSpec,
    *,
    sim: Any | None = None,
) -> None:
    """Step 1 of the algorithm: mj_forward to x*.

    At reset the manipulated object sits at a random spot on the table, so the
    clearance measured there is not the fit tolerance. The terminal pose of an
    insertion task is defined by the two axes coinciding, which is what this
    imposes: the free body's origin goes onto the static body's axis at the
    prescribed terminal yaw.
    """
    assert spec.moving_body and spec.inserted_body and spec.hole_body

    if spec.terminal_pose_rule is TerminalPoseRule.FROM_REFERENCE:
        # 9.5.4 takes x* from the end of the reference trajectory. Imposing axis
        # coincidence instead would silently answer a different question: for
        # ToolHang the frame's origin is nowhere near the stand's, and the two
        # bodies come out interpenetrating, which reads as a clearance of zero.
        from portal_trace.teacher.refs import terminal_pose_of

        if sim is None:
            raise ValueError(
                f"{spec.task_id}: FROM_REFERENCE terminal placement needs the "
                "robosuite simulation wrapper used by the reference resolver"
            )
        pose = terminal_pose_of(spec, sim)
        _write_free_joint(model, spec.moving_body, data, pose.position, pose.quaternion)
        mujoco.mj_forward(model, data)
        return

    static_name = spec.hole_body if spec.moving_body == spec.inserted_body else spec.inserted_body

    moving_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.moving_body)
    static_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, static_name)
    if moving_id < 0 or static_id < 0:
        present = sorted(
            n for i in range(model.nbody)
            if (n := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i))
        )
        raise KeyError(
            f"{spec.task_id}: cannot resolve {spec.moving_body!r} / {static_name!r} "
            f"in the compiled model. Bodies present: {present}"
        )

    # Height along the insertion axis is irrelevant to a normal-plane clearance,
    # but keeping the body at the static one's height puts the two shapes in the
    # configuration a reader would picture.
    target = np.asarray(data.xpos[static_id], dtype=np.float64).copy()
    half_yaw = 0.5 * spec.terminal_yaw_rad
    quat = np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)])
    _write_free_joint(model, spec.moving_body, data, target, quat)
    mujoco.mj_forward(model, data)


def _write_free_joint(
    model: mujoco.MjModel,
    body_name: str,
    data: mujoco.MjData,
    position: np.ndarray,
    quaternion: np.ndarray,
) -> None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jnt_adr = int(model.body_jntadr[body_id])
    if jnt_adr < 0 or int(model.jnt_type[jnt_adr]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError(
            f"{body_name!r} has no free joint, so its terminal pose cannot be "
            "imposed by writing qpos"
        )
    qadr = int(model.jnt_qposadr[jnt_adr])
    dadr = int(model.jnt_dofadr[jnt_adr])
    data.qpos[qadr : qadr + 3] = position
    data.qpos[qadr + 3 : qadr + 7] = quaternion
    data.qvel[dadr : dadr + 6] = 0.0


def compute_tol(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec: TaskSpec,
    *,
    hole_axis_point: np.ndarray | None = None,
) -> TolResult:
    """Steps 2-5 of the 9.5.4 algorithm on a model already at the terminal pose."""
    if spec.tol_kind is not TolKind.RADIAL_CLEARANCE:
        raise ValueError(
            f"{spec.task_id} has tol_kind {spec.tol_kind.value}; "
            "only radial_clearance goes through the geometric algorithm"
        )
    assert spec.inserted_body and spec.hole_body and spec.insertion_axis

    inserted_ids = _body_geoms(model, spec.inserted_body)
    hole_ids = _body_geoms(model, spec.hole_body)
    if spec.task_id == "ToolHang":
        # ToolHang's task fit is the frame tip entering the four-wall pocket.
        # The full bodies also contain the horizontal/vertical frame bars and
        # the solid stand base; including those normal support contacts makes
        # the reference-defined pose report a spurious zero clearance.
        def geom_name(g: int) -> str:
            return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""

        inserted_ids = [
            g for g in inserted_ids
            if geom_name(g) == "frame_tip_cylinder"
            or geom_name(g).startswith("frame_tip_cone_")
        ]
        hole_ids = [
            g for g in hole_ids
            if geom_name(g) in {f"stand_wall{i}" for i in range(4)}
        ]
        if not inserted_ids or not hole_ids:
            raise ValueError(
                "ToolHang: the compiled model lacks the frame-tip / stand-wall "
                "collision geoms required for its pocket fit"
            )
    if not inserted_ids or not hole_ids:
        raise ValueError(f"{spec.task_id}: one side has no collision geoms")

    # The hole axis: the centre of the hole body, unless the caller pins it.
    if hole_axis_point is None:
        if spec.task_id == "ToolHang":
            # The stand body origin is at the rear of the arena, while the
            # four pocket walls are centred at the actual insertion axis.  Use
            # the wall-centre centroid so the radial queries are taken from
            # the physical pocket axis rather than an arbitrary body origin.
            hole_axis_point = np.mean(
                [np.asarray(data.geom_xpos[g], dtype=np.float64) for g in hole_ids],
                axis=0,
            )
        else:
            hole_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.hole_body)
            hole_axis_point = np.asarray(data.xpos[hole_body_id], dtype=np.float64)

    basis = plane_basis(hole_axis_point, np.asarray(spec.insertion_axis, dtype=np.float64))
    hole = [cross_section_of_geom(model, data, g, basis) for g in hole_ids]

    # Only the part of the moving body that is actually inside the fit region can
    # touch a wall. Without this an object whose gripped end is wider than its
    # inserted end -- ToolHang's frame, a nut's handle -- reports a clearance for
    # a part of itself that is nowhere near the hole. The window comes from the
    # hole's own axial extent rather than from a parameter.
    axial_lo, axial_hi = _axial_span(model, data, hole_ids, basis)
    inserted_in_window = [
        g for g in inserted_ids
        if _spans_overlap(_axial_span(model, data, (g,), basis), (axial_lo, axial_hi))
    ]
    if not inserted_in_window:
        raise ValueError(
            f"{spec.task_id}: no geom of {spec.inserted_body!r} overlaps the fit "
            f"region of {spec.hole_body!r} along the insertion axis "
            f"([{axial_lo:.4g}, {axial_hi:.4g}] m). The terminal pose being measured "
            "does not have the two parts engaged, so there is no fit here"
        )
    inserted = [cross_section_of_geom(model, data, g, basis) for g in inserted_in_window]

    if spec.tol_rule is TolRule.CENTROID_CONTAINMENT:
        # Stack: the moving part contributes a point, because the criterion is
        # that cubeA's centroid project inside cubeB's supporting face (9.5.4).
        if len(hole) != 1:
            raise ValueError(
                f"{spec.task_id}: the centroid containment rule reads the boundary "
                f"of one supporting face, and {spec.hole_body!r} has {len(hole)} "
                "collision geoms. The distance to the boundary of a union is not "
                "the minimum of the per-geom distances, so this needs an explicit "
                "decision rather than a silent answer"
            )
        moving = [CrossSection(np.zeros((1, 2)), 0.0, "centroid", "point")]
    else:
        moving = inserted

    def clearance(theta: float) -> float:
        return first_contact_distance(moving, hole, basis.direction(theta))

    axes = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
    diagonals = [0.25 * np.pi, 0.75 * np.pi, 1.25 * np.pi, 1.75 * np.pi]

    tol_axis = min(clearance(t) for t in axes)
    tol_diag = min(clearance(t) for t in diagonals)

    if spec.tol_rule is TolRule.CENTROID_CONTAINMENT:
        # The origin lies inside the face, so the minimum over directions is the
        # inner distance to its boundary.
        tol_min = hole[0].inner_distance_from(np.zeros(2))
        argmin_dir = _argmin_direction_by_sweep(clearance)
    else:
        tol_min, best_dir = min_contact_distance(moving, hole)
        argmin_dir = float(np.degrees(np.arctan2(best_dir[1], best_dir[0])) % 360.0)

    if not math.isfinite(tol_axis) or not math.isfinite(tol_diag):
        raise ValueError(
            f"{spec.task_id}: no wall lies along an axis or diagonal direction, so "
            "there is no fit to measure at this pose. Check that the terminal pose "
            "really puts the two parts in contact configuration"
        )
    if tol_min <= 0.0:
        raise ValueError(
            f"{spec.task_id}: measured tol is {tol_min * 1e3:.4g} mm, i.e. the two "
            "parts interpenetrate at the pose that was measured. A fit tolerance "
            "cannot be non-positive; the terminal pose is wrong, not the geometry"
        )
    if tol_min > min(tol_axis, tol_diag) + 1e-12:
        raise AssertionError(
            f"{spec.task_id}: closed-form minimum {tol_min:.9g} exceeds the sampled "
            f"directions {min(tol_axis, tol_diag):.9g}; the two computations disagree"
        )

    return TolResult(
        task_id=spec.task_id,
        tol_kind=spec.tol_kind,
        tol_axis_m=float(tol_axis),
        tol_diagonal_m=float(tol_diag),
        tol_min_m=float(tol_min),
        argmin_direction_deg=argmin_dir,
        r_hole_axis_m=float(min(t for cs in hole if (t := cs.ray_boundary(basis.direction(0.0)))
                                is not None)),
        r_inserted_axis_m=float(max(cs.support(basis.direction(0.0)) for cs in moving)),
        inserted_geoms=tuple(cs.geom_name for cs in inserted),
        hole_geoms=tuple(cs.geom_name for cs in hole),
    )


def _argmin_direction_by_sweep(clearance) -> float:
    dense = np.linspace(0.0, 2.0 * np.pi, _N_DENSE, endpoint=False)
    vals = np.array([clearance(float(t)) for t in dense])
    return float(np.degrees(dense[int(np.argmin(vals))]))


def _make_env(spec: TaskSpec):
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config

    if spec.tight_variant_of is not None:
        # The tight-fit environments are not robosuite natives: 9.5.4 changes arena
        # geoms and the success criterion, which is why they carry their own
        # registered names. registry.tight builds and registers them.
        from portal_trace.registry.tight import ensure_registered

        ensure_registered(spec)

    kwargs = dict(spec.env_kwargs)
    kwargs.pop("horizon", None)
    controller = load_composite_controller_config(controller=None, robot="Panda")
    return suite.make(
        spec.env_name,
        robots="Panda",
        controller_configs=controller,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=400,
        ignore_done=True,
        **kwargs,
    )


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "not-a-git-checkout"


def _script_sha256() -> str:
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.read_bytes())
    return h.hexdigest()


def run(task_ids: list[str], *, write_lock: bool, out_path: Path | None) -> dict[str, TolResult]:
    check_pinned_versions(required=("mujoco", "robosuite"))
    results: dict[str, TolResult] = {}
    for task_id in task_ids:
        spec = task(task_id)
        if spec.tol_kind is not TolKind.RADIAL_CLEARANCE:
            print(f"[skip] {task_id}: tol_kind={spec.tol_kind.value}, not a geometric fit")
            continue
        env = _make_env(spec)
        try:
            env.reset()
            model = env.sim.model._model
            data = env.sim.data._data
            mujoco.mj_forward(model, data)
            place_at_terminal_pose(model, data, spec, sim=env.sim)
            res = compute_tol(model, data, spec)
        finally:
            env.close()
        results[task_id] = res
        print(
            f"[tol] {task_id:32s} axis={res.tol_axis_m * 1e3:8.4f} mm  "
            f"diag={res.tol_diagonal_m * 1e3:8.4f} mm  "
            f"min={res.tol_min_m * 1e3:8.4f} mm @ {res.argmin_direction_deg:6.1f} deg"
        )

    failures = []
    for task_id, want_mm in HAND_COMPUTED_MM.items():
        if task_id not in results:
            continue
        got_mm = results[task_id].tol_axis_m * 1e3
        if abs(got_mm - want_mm) > HAND_CHECK_TOL_MM:
            failures.append(f"  {task_id}: script {got_mm:.4f} mm vs 9.5.4 hand {want_mm} mm")
    if failures:
        raise SystemExit(
            "the 9.5.4 correctness check failed:\n"
            + "\n".join(failures)
            + "\nThe mesh tolerances this run produced may not be used (9.5.4)."
        )
    if set(HAND_COMPUTED_MM) <= set(results):
        print("[check] reproduces the 9.5.4 hand computation on both nut tasks")

    if out_path is not None:
        doc = {
            "tol_source": f"scripts/compute_task_tol.py@{_script_sha256()}",
            "environment": fingerprint(Backend.CPU_MUJOCO).to_json(),
            "tasks": {k: v.to_json() for k, v in results.items()},
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"[write] {out_path}")

    if write_lock:
        lock = LockFile()
        decl_tol = REQUIRED_MEASUREMENTS["task.tol.{task}.{geom_variant}"]
        decl_kind = REQUIRED_MEASUREMENTS["task.tol_kind.{task}.{geom_variant}"]
        commit = _git_commit()
        for task_id, res in results.items():
            lock.put(
                decl_tol,
                res.tol_min_m,
                source_commit=commit,
                notes=(
                    f"min over {_N_DENSE} directions; axis={res.tol_axis_m:.6g} m, "
                    f"diagonal={res.tol_diagonal_m:.6g} m"
                ),
                extra=res.to_json(),
                task=task_id,
                geom_variant="G0",
            )
            lock.put(
                decl_kind,
                res.tol_kind.value,
                source_commit=commit,
                task=task_id,
                geom_variant="G0",
            )
        lock.save()
        print(f"[lock] wrote {2 * len(results)} entries to {lock.path}")
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="task ids; default is every radial_clearance task in the registry",
    )
    p.add_argument("--write-lock", action="store_true", help="record results in registry/lock.json")
    p.add_argument("--out", type=Path, default=None, help="also write a standalone json report")
    args = p.parse_args(argv)

    task_ids = args.tasks or [
        t for t, s in TASKS.items() if s.tol_kind is TolKind.RADIAL_CLEARANCE
    ]
    run(task_ids, write_lock=args.write_lock, out_path=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
