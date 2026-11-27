"""Probe one real registry task against the MJX degradation backend.

This is an engineering preflight, not a timing benchmark or a paper result. It
stops after one unactuated MJX step and records blockers before any collection
or training data can be produced on a partially compatible backend.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import mujoco

from portal_trace.registry.tasks import task
from portal_trace.runners.backends import MjxBackend, RolloutSpec


RESULT_KIND = "engineering_preflight_not_paper_result"


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str | None:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)


def _body_name(model: mujoco.MjModel, geom_id: int) -> str | None:
    body_id = int(model.geom_bodyid[geom_id])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)


def _unsupported_collision_pairs(model: mujoco.MjModel) -> list[dict[str, Any]]:
    # MJX exposes no public compatibility report. This private module is used
    # only by the preflight and is versioned with the pinned mujoco-mjx wheel.
    from mujoco.mjx._src import collision_driver

    rows: list[dict[str, Any]] = []
    for geom1, geom2, pair_index in collision_driver.geom_pairs(model):
        type1, type2 = model.geom_type[[geom1, geom2]]
        if collision_driver.has_collision_fn(type1, type2):
            continue
        rows.append(
            {
                "geom1": int(geom1),
                "name1": _geom_name(model, int(geom1)),
                "type1": mujoco.mjtGeom(int(type1)).name,
                "body1": _body_name(model, int(geom1)),
                "geom2": int(geom2),
                "name2": _geom_name(model, int(geom2)),
                "type2": mujoco.mjtGeom(int(type2)).name,
                "body2": _body_name(model, int(geom2)),
                "explicit_pair": pair_index != -1,
            }
        )
    return rows


def _robosuite_action_dim(spec: RolloutSpec) -> int:
    import robosuite

    env = robosuite.make(
        spec.env_name,
        robots=spec.family,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=spec.control_freq,
        horizon=spec.horizon,
        ignore_done=True,
        **spec.robosuite_env_kwargs,
    )
    try:
        return int(env.action_dim)
    finally:
        env.close()


def _jax_wrapper_candidates() -> list[str]:
    from portal_trace.defects import inject

    required = ("init_carry", "pre_action", "pre_substep")
    return sorted(
        name
        for name, value in vars(inject).items()
        if isinstance(value, type)
        and all(callable(getattr(value, method, None)) for method in required)
    )


def probe(task_id: str, family: str, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "result_kind": RESULT_KIND,
        "task": task_id,
        "family": family,
        "seed": seed,
        "usable_for_collection": False,
    }
    try:
        import jax
        from mujoco import mjx

        backend = MjxBackend()
        backend.probe()
        spec_data = task(task_id)
        spec = RolloutSpec(
            task_id=spec_data.task_id,
            env_name=spec_data.env_name,
            env_kwargs=dict(spec_data.env_kwargs),
            family=family,
            env_seeds=(seed,),
        )

        t0 = time.perf_counter()
        model = backend._compile_model(spec)
        t1 = time.perf_counter()
        action_dim = _robosuite_action_dim(spec)
        unsupported = _unsupported_collision_pairs(model)
        jax_wrapper_candidates = _jax_wrapper_candidates()
        out.update(
            {
                "jax_version": jax.__version__,
                "mujoco_version": mujoco.__version__,
                "jax_backend": jax.default_backend(),
                "jax_devices": [str(device) for device in jax.devices()],
                "model": {
                    "nq": int(model.nq),
                    "nv": int(model.nv),
                    "nu": int(model.nu),
                    "nbody": int(model.nbody),
                    "ngeom": int(model.ngeom),
                },
                "robosuite_action_dim": action_dim,
                "direct_ctrl_shape_compatible": action_dim == int(model.nu),
                "jax_wrapper_ops_candidates": jax_wrapper_candidates,
                "jax_wrapper_ops_implementation_present": bool(jax_wrapper_candidates),
                "unsupported_collision_pair_count": len(unsupported),
                "unsupported_collision_pairs": unsupported,
                "robosuite_compile_s": t1 - t0,
            }
        )

        t2 = time.perf_counter()
        mjx_model = mjx.put_model(model)
        data = mjx.make_data(mjx_model)
        stepped = mjx.step(mjx_model, data)
        jax.block_until_ready(stepped.qpos)
        t3 = time.perf_counter()
        out.update(
            {
                "mjx_model_and_eager_step_s": t3 - t2,
                "finite_after_step": bool(jax.numpy.all(jax.numpy.isfinite(stepped.qpos))),
                "model_step_ok": True,
            }
        )

        # A raw MJX step is necessary but not sufficient. The current backend
        # still needs a JAX implementation of the robosuite controller and the
        # defect wrapper before it can preserve CPU rollout semantics.
        if action_dim != int(model.nu):
            out["blocker"] = (
                "robosuite policy actions are high-level controller commands, but "
                "MjxBackend assigns them directly to raw actuator ctrl with a different width"
            )
        else:
            out["blocker"] = (
                "raw model stepping works, but the robosuite controller and JaxWrapperOps "
                "ports remain required before collection"
            )
    except Exception as exc:
        out.update(
            {
                "model_step_ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--family", default="Panda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.output.exists():
        parser.error(f"refusing to overwrite existing preflight output: {args.output}")
    result = probe(args.task, args.family, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("usable_for_collection") else 2


if __name__ == "__main__":
    raise SystemExit(main())
