"""Demonstration collection (proposal 9.6.3, 9.6.5).

Writes `raw/<task>/<family>/<defect>/demo.hdf5`: states and actions only, no
images. 9.1.3's first consequence is that rendering costs eight times what
physics does and the same states are reused by several arms, so images are
materialised on demand by `runners.materialise` instead of being produced here.

The ground-truth and replay extensions to robomimic's native layout are written
here too, because they only exist at collection time: `delta_true`, the per-frame
`c_true`, `defect_state`, `actuation_state`, `tau_g`, `wrist_ft`, and the regime
label sequence. Per-frame c_true is what distinguishes this dataset from every other
cross-embodiment collection -- OXE, DROID and RT-X store (observation, action)
and cannot store c_delta, because that needs the same teacher's action at the
same state on a defect-free twin, and in the real world the twin does not exist.
It is also nearly free: 400 x 7 x 4 = 11 kB per trajectory against 16.9 MB of
uncompressed images.

Ordering constraint from 9.8.4: if the measured overrun factor exceeds the
preregistered threshold, degradation branch 0 -- porting the pure-physics load
to MJX -- has to happen before this stage, not after. Collecting first and
porting later throws the collection away, because 9.1.2 forbids mixing the two
backends' data inside one endpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from portal_trace.configs.validator import RunRequest
from portal_trace.lockfile import REQUIRED_MEASUREMENTS, source_commit
from portal_trace.registry.families import family as family_spec
from portal_trace.registry.tasks import CONTROL_FREQ, CORE_TASKS, HORIZON
from portal_trace.registry.tasks import task as task_spec
from portal_trace.runners.backends import (
    degradation_branch_zero,
    get_backend,
)
from portal_trace.runners.common import RunContext, dataset_paths, require_module
from portal_trace.runners.dataset_contract import (
    PAIRED_TWIN_SCHEMA_VERSION,
    SURROGATE_CONTEXT_SCHEMA_VERSION,
    require_analysis_trace_contract,
    require_current_defect_state_contract,
    require_dataset_identity,
    require_paired_twin_contract,
    stamp_analysis_trace_contract,
    stamp_dataset_identity,
    stamp_defect_state_contract,
    stamp_paired_twin_contract,
)
from portal_trace.types import EmbodimentId, Episode, SeedTriple

__all__ = [
    "B7_CLEAN_DEMOS_PER_TASK",
    "CollectionPlan",
    "plan_collection",
    "collect",
    "collect_b7_clean",
    "PROPRIO_LAYOUT",
]

# CONTRACT section 6. robosuite 1.5.2's native robot0_proprio-state is 50 wide
# on a 7-DoF arm, not the 32 that 9.6.1 assumes, so the stored vector is 70
# rather than the 52 printed in 9.6.3 and 11.4. The width is measured, not
# assumed, and the correction is recorded in the amendment log.
PROPRIO_LAYOUT: tuple[tuple[str, int], ...] = (
    ("robot0_proprio-state (native)", 50),
    ("tau_g (gravity torque)", 7),
    ("wrist force/torque", 6),
    ("DoF mask", 7),
)
PROPRIO_DIM = sum(width for _, width in PROPRIO_LAYOUT)
PROPRIO_PHYSICAL_CHANNELS = PROPRIO_DIM - 7   # 63; the mask is not a physical channel

# Collection consumes only the environment layer. A sentinel is more truthful
# than borrowing one training arm's seed and thereby making the shared
# demonstrations appear to belong to that arm.
_UNUSED_TRAIN_SEED = -1
B7_CLEAN_DEMOS_PER_TASK = 1_400
_PAIR_TRANSACTION_SCHEMA = PAIRED_TWIN_SCHEMA_VERSION
_B7_TRANSACTION_SCHEMA = 1
_SURROGATE_CONTEXT_SCHEMA = SURROGATE_CONTEXT_SCHEMA_VERSION


@dataclass(frozen=True)
class CollectionPlan:
    """One (task, family, defect) cell's collection job."""

    task: str
    family: str
    defect: str
    level: int
    n_demos: int
    env_seeds: tuple[int, ...]
    geom_variant: str
    output: Path
    source_validation_reservation_sha256: str | None = None
    source_validation_collection_spec: Mapping[str, Any] | None = None

    @property
    def n_frames(self) -> int:
        return self.n_demos * HORIZON

    @property
    def clean_output(self) -> Path:
        """The B7 Oracle source produced by this same twin collection."""
        return (
            self.output.parents[2]
            / f"clean-{self.defect}"
            / f"L{self.level}"
            / self.output.name
        )

    @property
    def transaction_marker(self) -> Path:
        return (
            self.output.parents[2]
            / ".collection-transactions"
            / f"{self.defect}-L{self.level}-n{self.n_demos}.json"
        )

    @property
    def b7_parts_dir(self) -> Path:
        return self.clean_output.parent / ".b7-parts"

    @property
    def b7_completion_marker(self) -> Path:
        return self.clean_output.parent / "b7-clean-complete.json"


def plan_collection(
    ctx: RunContext,
    *,
    n_demos: int,
    level: int,
    env_seeds: tuple[int, ...] | None = None,
    source_validation_reservation_sha256: str | None = None,
    source_validation_collection_spec: Mapping[str, Any] | None = None,
) -> CollectionPlan:
    """Resolve the cell and its environment seeds.

    The seeds come from the three-layer table's environment layer, which is not
    scoped by defect (9.1.6). That is what makes the same (task, family, demo
    index) produce bit-identical initial states across the six core defects, so
    one clean-twin demonstration serves all six and the eval_init list can be
    reused on defective targets.
    """
    req = ctx.request
    if req.task is None or req.family is None or req.defect is None:
        raise ValueError(
            "collection needs --task, --family and --defect: the cell is the "
            "statistical unit and the output path is keyed by it"
        )
    if isinstance(n_demos, bool) or not isinstance(n_demos, int) or n_demos < 1:
        raise ValueError(f"n_demos must be a positive integer, got {n_demos!r}")
    spec = task_spec(req.task)
    from portal_trace.defects.params import defect

    defect(req.defect).level(level)   # raises if the level is absent from the table
    seeds = (
        tuple(env_seeds)
        if env_seeds is not None
        else tuple(
            ctx.bundle.seeds.env_seed(task=req.task, family=req.family, demo_index=i)
            for i in range(n_demos)
        )
    )
    if len(seeds) != n_demos:
        raise ValueError("explicit collection seeds must match n_demos exactly")
    if len(set(seeds)) != len(seeds):
        raise ValueError(
            f"{req.task}/{req.family}: environment seed allocation collides within "
            f"demo indices 0..{n_demos - 1}"
        )
    paths = dataset_paths(
        ctx.dataset_root,
        task=req.task,
        family=req.family,
        defect=req.defect,
        level=level,
    )
    return CollectionPlan(
        task=spec.task_id,
        family=req.family,
        defect=req.defect,
        level=level,
        n_demos=n_demos,
        env_seeds=seeds,
        geom_variant=req.geom_variant,
        output=paths["raw"],
        source_validation_reservation_sha256=source_validation_reservation_sha256,
        source_validation_collection_spec=source_validation_collection_spec,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_sha256(seeds: tuple[int, ...]) -> str:
    payload = json.dumps(list(seeds), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _paired_binding_id(plan: CollectionPlan) -> str:
    return _canonical_sha256(
        {
            "schema_version": PAIRED_TWIN_SCHEMA_VERSION,
            "task": plan.task,
            "family": plan.family,
            "defect": plan.defect,
            "level": plan.level,
            "geom_variant": plan.geom_variant,
        }
    )


def _reference_trajectory_sha256(reference: Any) -> str:
    return _canonical_sha256(
        {
            "task_id": reference.task_id,
            "positions": _array_sha256(reference.positions),
            "rotations": _array_sha256(reference.rotations),
            "gripper": _array_sha256(reference.gripper),
            "phase": _array_sha256(reference.phase),
            "waypoint_index": _array_sha256(reference.waypoint_index),
            "waypoint_names": list(reference.waypoint_names),
        }
    )


def _paired_episode_provenance(
    plan: CollectionPlan,
    pair: Any,
    reference_spec: Any,
    reference: Any,
    *,
    seed_env: int,
    initial_physics: np.ndarray,
    initial_actuation: np.ndarray,
    config: Any,
    controller_sha256: str,
    clean_controller_sha256: str,
    run_source_commit: str,
    runtime_json: str,
) -> dict[str, str]:
    physics_sha = _array_sha256(initial_physics)
    actuation_sha = _array_sha256(initial_actuation)
    initial_frame_sha = _canonical_sha256(
        {"physics_sha256": physics_sha, "actuation_sha256": actuation_sha}
    )
    spec_sha = _canonical_sha256(reference_spec.to_json())
    trajectory_sha = _reference_trajectory_sha256(reference)
    binding_id = _paired_binding_id(plan)
    pair_id = _canonical_sha256(
        {
            "paired_binding_id": binding_id,
            "seed_env": seed_env,
            "initial_frame_sha256": initial_frame_sha,
            "reference_spec_sha256": spec_sha,
            "reference_trajectory_sha256": trajectory_sha,
        }
    )
    config_identity = {
        "implementation": "portal_trace.teacher.ilc.run_ilc",
        "lead_steps": int(config.lead_steps),
        "q_cutoff_hz": float(config.q_cutoff_hz),
        "k_max": int(config.k_max),
        "contact_variant": config.contact_variant.value,
        "source_commit": run_source_commit,
    }
    defective_teacher = _canonical_sha256(
        config_identity | {"role": "defective", "delta": pair.delta.to_json()}
    )
    clean_teacher = _canonical_sha256(
        config_identity | {"role": "clean", "delta": pair.delta.zero().to_json()}
    )
    return {
        "pair_id": pair_id,
        "paired_binding_id": binding_id,
        "initial_physics_sha256": physics_sha,
        "initial_actuation_sha256": actuation_sha,
        "initial_frame_sha256": initial_frame_sha,
        "reference_spec_sha256": spec_sha,
        "reference_trajectory_sha256": trajectory_sha,
        "controller_identity_defective_sha256": controller_sha256,
        "controller_identity_clean_sha256": clean_controller_sha256,
        "teacher_identity_defective_sha256": defective_teacher,
        "teacher_identity_clean_sha256": clean_teacher,
        "collection_source_commit": run_source_commit,
        "collection_runtime_json": runtime_json,
        **(
            {
                "source_validation_reservation_sha256": (
                    plan.source_validation_reservation_sha256
                )
            }
            if plan.source_validation_reservation_sha256 is not None
            else {}
        ),
    }


def _write_json_immutable(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite collection marker {path}")
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite collection marker {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _validate_collected_file(
    plan: CollectionPlan,
    path: Path,
    *,
    clean: bool,
    allow_seed_superset: bool = False,
) -> int:
    """Validate identity, exact seed ordering and the current state semantics."""
    import h5py

    expected_count = plan.n_demos
    count = require_dataset_identity(
        path,
        task=plan.task,
        family=plan.family,
        defect="clean" if clean else plan.defect,
        level=plan.level,
        clean_of=plan.defect if clean else None,
        min_demos=expected_count,
    )
    if not allow_seed_superset and count != expected_count:
        raise ValueError(
            f"{path}: contains {count} demonstrations, expected exactly {expected_count}"
        )
    require_current_defect_state_contract(path, requested_defect=plan.defect)
    require_analysis_trace_contract(path)
    require_paired_twin_contract(path)
    with h5py.File(path, "r") as dataset:
        if plan.source_validation_reservation_sha256 is not None:
            observed_reservation = str(
                dataset.attrs.get("source_validation_reservation_sha256", "")
            )
            if observed_reservation != plan.source_validation_reservation_sha256:
                raise ValueError(
                    f"{path}: dataset is not bound to the requested validation reservation"
                )
            expected_collection = plan.source_validation_collection_spec or {}
            for observed_name, expected_name in (
                ("collection_source_commit", "source_commit"),
                ("collection_runtime_json", "environment_fingerprint"),
                ("paired_binding_id", "paired_binding_id"),
            ):
                if str(dataset.attrs.get(observed_name, "")) != str(
                    expected_collection.get(expected_name, "")
                ):
                    raise ValueError(
                        f"{path}: collected {observed_name} differs from the reservation"
                    )
        keys = sorted(dataset["data"], key=lambda name: int(name.split("_")[1]))
        expected_keys = [f"demo_{index}" for index in range(count)]
        if keys != expected_keys:
            raise ValueError(f"{path}: demonstrations are not contiguous and ordered")
        seeds = tuple(int(dataset[f"data/{key}"].attrs["seed_env"]) for key in keys)
        if plan.source_validation_reservation_sha256 is not None and any(
            str(dataset[f"data/{key}"].attrs.get(
                "source_validation_reservation_sha256", ""
            ))
            != plan.source_validation_reservation_sha256
            for key in keys[:expected_count]
        ):
            raise ValueError(f"{path}: episode reservation bindings are incomplete")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{path}: contains duplicate environment seeds")
    if seeds[:expected_count] != plan.env_seeds:
        raise ValueError(
            f"{path}: stored environment seeds do not match demo indices "
            f"0..{expected_count - 1}"
        )
    return count


def _pair_marker_payload(plan: CollectionPlan) -> dict[str, Any]:
    defective_count = _validate_collected_file(plan, plan.output, clean=False)
    clean_count = _validate_collected_file(
        plan,
        plan.clean_output,
        clean=True,
        allow_seed_superset=plan.source_validation_reservation_sha256 is None,
    )
    import h5py

    with h5py.File(plan.output, "r") as defective_file, h5py.File(
        plan.clean_output, "r"
    ) as clean_file:
        defective_binding = str(defective_file.attrs["paired_binding_id"])
        clean_binding = str(clean_file.attrs["paired_binding_id"])
        defective_pair_ids = json.loads(
            str(defective_file.attrs["paired_episode_ids_json"])
        )
        clean_pair_ids = json.loads(str(clean_file.attrs["paired_episode_ids_json"]))
    if (
        defective_binding != clean_binding
        or clean_pair_ids[:defective_count] != defective_pair_ids
    ):
        raise ValueError("clean and defective raw files do not carry the same pair binding")
    defective_sha = _sha256(plan.output)
    clean_sha = _sha256(plan.clean_output)
    raw_binding = {
        "paired_binding_id": defective_binding,
        "pair_ids": defective_pair_ids,
        "defective_sha256": defective_sha,
        "clean_sha256": clean_sha,
    }
    return {
        "schema_version": _PAIR_TRANSACTION_SCHEMA,
        "producer": "portal_trace.runners.collect",
        "status": "complete",
        "task": plan.task,
        "family": plan.family,
        "defect": plan.defect,
        "level": plan.level,
        "requested_demos": plan.n_demos,
        "seed_sha256": _seed_sha256(plan.env_seeds),
        **(
            {
                "source_validation_reservation_sha256": (
                    plan.source_validation_reservation_sha256
                )
            }
            if plan.source_validation_reservation_sha256 is not None
            else {}
        ),
        "paired_binding_id": defective_binding,
        "pair_ids": defective_pair_ids,
        "raw_pair_sha256": _canonical_sha256(raw_binding),
        "raw": {
            "defective": {
                "path": str(plan.output.resolve()),
                "n_demos": defective_count,
                "sha256": defective_sha,
            },
            "clean": {
                "path": str(plan.clean_output.resolve()),
                "clean_of": plan.defect,
                "n_demos": clean_count,
                "sha256": clean_sha,
            },
        },
    }


def _validate_pair_marker(plan: CollectionPlan) -> None:
    path = plan.transaction_marker
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid collection completion marker") from exc
    expected = _pair_marker_payload(plan)
    if marker != expected:
        raise ValueError(
            f"{path}: collection marker disagrees with the immutable datasets"
        )


def _complete_pair_if_recoverable(plan: CollectionPlan) -> bool:
    if plan.transaction_marker.exists():
        _validate_pair_marker(plan)
        return True
    if plan.output.exists():
        _validate_collected_file(plan, plan.output, clean=False)
    if plan.clean_output.exists():
        _validate_collected_file(
            plan,
            plan.clean_output,
            clean=True,
            allow_seed_superset=plan.source_validation_reservation_sha256 is None,
        )
    if plan.output.exists() and plan.clean_output.exists():
        _publish_pair_marker(plan)
        return True
    return False


def _publish_pair_marker(plan: CollectionPlan) -> None:
    payload = _pair_marker_payload(plan)
    try:
        _write_json_immutable(plan.transaction_marker, payload)
    except FileExistsError:
        _validate_pair_marker(plan)


def collect(
    ctx: RunContext,
    *,
    n_demos: int,
    level: int,
    overrun_step: float,
    overrun_frame: float,
    source_validation_reservation_path: Path | None = None,
) -> CollectionPlan:
    """Run the ILC teacher over one cell and write the hdf5.

    `overrun_step` and `overrun_frame` are the measured-over-assumed ratios of
    9.8.4 and have no defaults: they come from `sim.tau_step` and
    `sim.tau_frame` in the lock file, and the whole degradation ladder is keyed
    on them.
    """
    reservation = None
    if source_validation_reservation_path is not None:
        from portal_trace.runners.source_validation import (
            load_source_validation_reservation,
        )

        reservation = load_source_validation_reservation(
            source_validation_reservation_path
        )
        collection_spec = reservation.collection_spec
        requested_cell = (
            ctx.request.task,
            ctx.request.family,
            ctx.request.defect,
            level,
            ctx.request.geom_variant,
        )
        reserved_cell = tuple(
            collection_spec[name]
            for name in ("task", "family", "defect", "level", "geom_variant")
        )
        if requested_cell != reserved_cell:
            raise ValueError(
                "collection request differs from the source-validation reservation cell"
            )
        if n_demos != len(reservation.reserved_seed_envs):
            raise ValueError(
                "validation collection n_demos must equal the number of reserved seeds"
            )
    plan = plan_collection(
        ctx,
        n_demos=n_demos,
        level=level,
        env_seeds=(None if reservation is None else reservation.reserved_seed_envs),
        source_validation_reservation_sha256=(
            None if reservation is None else reservation.artifact_sha256
        ),
        source_validation_collection_spec=(
            None if reservation is None else reservation.collection_spec
        ),
    )
    if _complete_pair_if_recoverable(plan):
        return plan

    degradation_branch_zero(
        overrun_step=overrun_step,
        overrun_frame=overrun_frame,
        threshold=ctx.threshold("overrun_factor_branch_zero"),
        stage="collect",
    )

    if plan.geom_variant != "G0":
        raise NotImplementedError(
            f"collection currently builds the registry task's G0 model, not "
            f"{plan.geom_variant}. Running it under that label would silently erase "
            "the geometry intervention, so non-G0 collection is blocked until the "
            "geometry builder is connected to defects.inject.build_twin_pair"
        )

    backend = get_backend(ctx.backend)
    backend.probe()
    if backend.backend.value != "cpu_mujoco":
        raise NotImplementedError(
            f"the executable twin/ILC collection path currently targets CPU MuJoCo, "
            f"not {backend.backend.value}. The MJX wrapper port must be completed as "
            "one calibrated backend before it can produce demonstrations"
        )
    run_source_commit = source_commit()
    runtime_json = json.dumps(
        backend.fingerprint(lite_physics=True).to_json(),
        sort_keys=True,
        separators=(",", ":"),
    )
    if reservation is not None:
        from portal_trace.runners.formal_training import _git_training_producer_state

        if _git_training_producer_state() != reservation.producer_git_state:
            raise RuntimeError(
                "collection producer state differs from the source-validation reservation"
            )
        if run_source_commit != reservation.collection_spec["source_commit"]:
            raise RuntimeError(
                "collection HEAD differs from the source-validation reservation"
            )
        if runtime_json != reservation.collection_spec["environment_fingerprint"]:
            raise RuntimeError(
                "collection backend fingerprint differs from the source-validation reservation"
            )

    delta_mod = require_module("portal_trace.defects.delta", needed_for="collection")
    inject = require_module("portal_trace.defects.inject", needed_for="collection")
    families = require_module("portal_trace.registry.families", needed_for="collection")
    ilc = require_module("portal_trace.teacher.ilc", needed_for="collection")
    osc = require_module("portal_trace.teacher.osc", needed_for="collection")
    query = require_module("portal_trace.teacher.query", needed_for="collection")
    refs = require_module("portal_trace.teacher.refs", needed_for="collection")
    regimes = require_module("portal_trace.diagnostics.regimes", needed_for="collection")

    thresholds = regimes.RegimeThresholds.from_lock(
        ctx.lock, task=plan.task, family=plan.family
    )
    spec = task_spec(plan.task)
    n_dof = families.descriptor(plan.family).dof
    delta = delta_mod.Delta.from_level(
        plan.defect,
        plan.level,
        joint_scale=delta_mod.default_joint_scale(plan.defect, n_dof),
    )
    tol = float(
        ctx.lock.get(
            REQUIRED_MEASUREMENTS["task.tol.{task}.{geom_variant}"],
            task=plan.task,
            geom_variant=plan.geom_variant,
        )
    )
    reference_spec = refs.load_spec(plan.task)
    episodes = []
    clean_episodes = []
    for seed_env in plan.env_seeds:
        seeds = SeedTriple(
            env=seed_env,
            train=_UNUSED_TRAIN_SEED,
            eval=ctx.bundle.seeds.eval_seed(task=plan.task, family=plan.family),
        )
        pair = inject.build_twin_pair(spec, plan.family, delta, seeds)
        try:
            def reset_pair(pair=pair, seed_env=seed_env) -> np.ndarray:
                return osc.reset_twin_pair(pair, seed_env)

            reset = {side: reset_pair for side in pair}
            initial_physics = reset_pair()
            initial_actuation = pair.clean.actuation_state().copy()
            reference = refs.resolve(reference_spec, pair.defective.sim)
            config = ilc.ILCConfig.for_delta(delta)
            defective_result = ilc.run_ilc(
                pair.defective,
                reference,
                family=plan.family,
                seed_env=seed_env,
                tol=tol,
                config=config,
                reset=reset[pair.defective],
            )
            clean_result = ilc.run_ilc(
                pair.clean,
                reference,
                family=plan.family,
                seed_env=seed_env,
                tol=tol,
                config=config,
                reset=reset[pair.clean],
            )
            if defective_result.final_tracking is None or clean_result.final_tracking is None:
                raise ValueError("ILC returned no final rollout; pair provenance is unavailable")
            paired_provenance = _paired_episode_provenance(
                plan,
                pair,
                reference_spec,
                reference,
                seed_env=seed_env,
                initial_physics=initial_physics,
                initial_actuation=initial_actuation,
                config=config,
                controller_sha256=defective_result.final_tracking.controller_config_sha256,
                clean_controller_sha256=clean_result.final_tracking.controller_config_sha256,
                run_source_commit=run_source_commit,
                runtime_json=runtime_json,
            )
            episodes.append(
                _episode_from_ilc(
                    plan,
                    pair,
                    reference,
                    defective_result,
                    clean_result,
                    seeds=seeds,
                    thresholds=thresholds,
                    regimes=regimes,
                    query=query,
                    paired_provenance=paired_provenance,
                )
            )
            clean_episodes.append(
                _clean_episode_from_ilc(
                    plan,
                    pair,
                    reference,
                    clean_result,
                    seeds=seeds,
                    thresholds=thresholds,
                    regimes=regimes,
                    paired_provenance=paired_provenance,
                )
            )
        finally:
            pair.defective.close()
            pair.clean.close()

    if not plan.output.exists():
        try:
            _write_hdf5(plan, episodes, delta=delta, ctx=ctx, clean=False)
        except FileExistsError:
            _validate_collected_file(plan, plan.output, clean=False)
    clean_plan = replace(plan, output=plan.clean_output)
    if not clean_plan.output.exists():
        try:
            _write_hdf5(
                clean_plan,
                clean_episodes,
                delta=delta.zero(),
                ctx=ctx,
                clean=True,
            )
        except FileExistsError:
            _validate_collected_file(
                plan, plan.clean_output, clean=True, allow_seed_superset=True
            )
    _publish_pair_marker(plan)
    return plan


def _b7_marker_payload(plan: CollectionPlan) -> dict[str, Any]:
    count = _validate_collected_file(plan, plan.clean_output, clean=True)
    parts_sha256 = _b7_parts_sha256(plan)
    return {
        "schema_version": _B7_TRANSACTION_SCHEMA,
        "producer": "portal_trace.runners.collect.collect_b7_clean",
        "status": "complete",
        "task": plan.task,
        "family": plan.family,
        "owner_defect": plan.defect,
        "clean_of": plan.defect,
        "level": plan.level,
        "n_demos": count,
        "seed_sha256": _seed_sha256(plan.env_seeds),
        "dataset": {
            "path": str(plan.clean_output.resolve()),
            "sha256": _sha256(plan.clean_output),
        },
        "parts_dir": str(plan.b7_parts_dir.resolve()),
        "parts_sha256": parts_sha256,
    }


def _validate_b7_marker(plan: CollectionPlan) -> None:
    path = plan.b7_completion_marker
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid B7 completion marker") from exc
    if marker != _b7_marker_payload(plan):
        raise ValueError(f"{path}: B7 marker disagrees with the immutable clean dataset")


def _publish_b7_marker(plan: CollectionPlan) -> None:
    payload = _b7_marker_payload(plan)
    try:
        _write_json_immutable(plan.b7_completion_marker, payload)
    except FileExistsError:
        _validate_b7_marker(plan)


def _b7_parts_sha256(plan: CollectionPlan) -> str:
    digest = hashlib.sha256()
    for index, seed in enumerate(plan.env_seeds):
        part = plan.b7_parts_dir / f"{index:06d}.hdf5"
        if not part.exists():
            raise FileNotFoundError(
                f"{part}: B7 completion requires all {plan.n_demos} immutable parts"
            )
        part_plan = replace(plan, n_demos=1, env_seeds=(seed,), output=part)
        _validate_collected_file(part_plan, part, clean=True)
        digest.update(f"{index}:".encode("ascii"))
        digest.update(_sha256(part).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _assemble_b7_parts(plan: CollectionPlan, *, delta: Any, ctx: RunContext) -> None:
    """Assemble 1,400 immutable one-demo parts without loading them into memory."""
    import h5py

    if plan.clean_output.exists():
        _validate_collected_file(plan, plan.clean_output, clean=True)
        return
    plan.clean_output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=plan.clean_output.parent,
        prefix=f".{plan.clean_output.stem}.b7.",
        suffix=plan.clean_output.suffix,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        _write_hdf5_contents(
            replace(plan, output=temporary),
            [],
            delta=delta.zero(),
            ctx=ctx,
            clean=True,
        )
        with h5py.File(temporary, "a") as destination:
            data = destination["data"]
            for index, seed in enumerate(plan.env_seeds):
                part = plan.b7_parts_dir / f"{index:06d}.hdf5"
                part_plan = replace(
                    plan, n_demos=1, env_seeds=(seed,), output=part
                )
                _validate_collected_file(part_plan, part, clean=True)
                with h5py.File(part, "r") as source:
                    source.copy("data/demo_0", data, name=f"demo_{index}")
            data.attrs["total"] = plan.n_demos * HORIZON
            demos = [data[f"demo_{index}"] for index in range(plan.n_demos)]
            binding_ids = {str(demo.attrs["paired_binding_id"]) for demo in demos}
            source_commits = {str(demo.attrs["collection_source_commit"]) for demo in demos}
            runtime_jsons = {str(demo.attrs["collection_runtime_json"]) for demo in demos}
            if len(binding_ids) != 1 or len(source_commits) != 1 or len(runtime_jsons) != 1:
                raise ValueError("B7 parts disagree on paired provenance")
            stamp_paired_twin_contract(
                destination.attrs,
                role="clean",
                binding_id=binding_ids.pop(),
                pair_ids=tuple(str(demo.attrs["pair_id"]) for demo in demos),
                transaction_marker=str(plan.transaction_marker.resolve()),
                source_commit=source_commits.pop(),
                runtime_json=runtime_jsons.pop(),
            )
        with temporary.open("rb") as fh:
            os.fsync(fh.fileno())
        try:
            os.link(temporary, plan.clean_output)
        except FileExistsError:
            _validate_collected_file(plan, plan.clean_output, clean=True)
    finally:
        temporary.unlink(missing_ok=True)


def collect_b7_clean(
    ctx: RunContext,
    *,
    level: int,
    overrun_step: float,
    overrun_frame: float,
) -> CollectionPlan:
    """Produce B7's exact 1,400-demo clean twin for one core task."""
    plan = plan_collection(ctx, n_demos=B7_CLEAN_DEMOS_PER_TASK, level=level)
    if plan.task not in CORE_TASKS:
        raise ValueError(
            f"B7 clean production is defined only for core tasks, got {plan.task!r}"
        )
    b7_cells = set(ctx.bundle.cells("B7"))
    if (plan.family, plan.defect) not in b7_cells:
        raise ValueError(
            f"B7 clean production requires an S36 target cell, got "
            f"({plan.family}, {plan.defect})"
        )
    if plan.b7_completion_marker.exists():
        _validate_b7_marker(plan)
        return plan
    if plan.clean_output.exists():
        _validate_collected_file(plan, plan.clean_output, clean=True)
        _publish_b7_marker(plan)
        return plan

    degradation_branch_zero(
        overrun_step=overrun_step,
        overrun_frame=overrun_frame,
        threshold=ctx.threshold("overrun_factor_branch_zero"),
        stage="collect",
    )
    if plan.geom_variant != "G0":
        raise NotImplementedError("B7 clean production currently supports only G0")
    backend = get_backend(ctx.backend)
    backend.probe()
    if backend.backend.value != "cpu_mujoco":
        raise NotImplementedError("B7 clean production currently requires CPU MuJoCo")
    run_source_commit = source_commit()
    runtime_json = json.dumps(
        backend.fingerprint(lite_physics=True).to_json(),
        sort_keys=True,
        separators=(",", ":"),
    )

    delta_mod = require_module("portal_trace.defects.delta", needed_for="B7 collection")
    inject = require_module("portal_trace.defects.inject", needed_for="B7 collection")
    families = require_module("portal_trace.registry.families", needed_for="B7 collection")
    ilc = require_module("portal_trace.teacher.ilc", needed_for="B7 collection")
    osc = require_module("portal_trace.teacher.osc", needed_for="B7 collection")
    refs = require_module("portal_trace.teacher.refs", needed_for="B7 collection")
    regimes = require_module("portal_trace.diagnostics.regimes", needed_for="B7 collection")

    thresholds = regimes.RegimeThresholds.from_lock(
        ctx.lock, task=plan.task, family=plan.family
    )
    spec = task_spec(plan.task)
    n_dof = families.descriptor(plan.family).dof
    delta = delta_mod.Delta.from_level(
        plan.defect,
        plan.level,
        joint_scale=delta_mod.default_joint_scale(plan.defect, n_dof),
    )
    tol = float(
        ctx.lock.get(
            REQUIRED_MEASUREMENTS["task.tol.{task}.{geom_variant}"],
            task=plan.task,
            geom_variant=plan.geom_variant,
        )
    )
    plan.b7_parts_dir.mkdir(parents=True, exist_ok=True)
    eval_seed = ctx.bundle.seeds.eval_seed(task=plan.task, family=plan.family)
    config = ilc.ILCConfig.for_delta(delta)
    reference_spec = refs.load_spec(plan.task)

    for index, seed_env in enumerate(plan.env_seeds):
        part = plan.b7_parts_dir / f"{index:06d}.hdf5"
        part_plan = replace(plan, n_demos=1, env_seeds=(seed_env,), output=part)
        if part.exists():
            _validate_collected_file(part_plan, part, clean=True)
            continue
        seeds = SeedTriple(env=seed_env, train=_UNUSED_TRAIN_SEED, eval=eval_seed)
        pair = inject.build_twin_pair(spec, plan.family, delta, seeds)
        try:
            def reset_pair(pair=pair, seed_env=seed_env) -> np.ndarray:
                return osc.reset_twin_pair(pair, seed_env)

            initial_physics = reset_pair()
            initial_actuation = pair.clean.actuation_state().copy()
            reference = refs.resolve(reference_spec, pair.defective.sim)
            clean_result = ilc.run_ilc(
                pair.clean,
                reference,
                family=plan.family,
                seed_env=seed_env,
                tol=tol,
                config=config,
                reset=reset_pair,
            )
            if clean_result.final_tracking is None:
                raise ValueError(
                    "clean ILC returned no final rollout; pair provenance is unavailable"
                )
            paired_provenance = _paired_episode_provenance(
                plan,
                pair,
                reference_spec,
                reference,
                seed_env=seed_env,
                initial_physics=initial_physics,
                initial_actuation=initial_actuation,
                config=config,
                controller_sha256=clean_result.final_tracking.controller_config_sha256,
                clean_controller_sha256=clean_result.final_tracking.controller_config_sha256,
                run_source_commit=run_source_commit,
                runtime_json=runtime_json,
            )
            episode = _clean_episode_from_ilc(
                plan,
                pair,
                reference,
                clean_result,
                seeds=seeds,
                thresholds=thresholds,
                regimes=regimes,
                paired_provenance=paired_provenance,
            )
        finally:
            pair.defective.close()
            pair.clean.close()
        try:
            _write_hdf5(
                part_plan,
                [episode],
                delta=delta.zero(),
                ctx=ctx,
                clean=True,
            )
        except FileExistsError:
            _validate_collected_file(part_plan, part, clean=True)

    _assemble_b7_parts(plan, delta=delta, ctx=ctx)
    _publish_b7_marker(plan)
    return plan


def _clean_episode_from_ilc(
    plan: CollectionPlan,
    pair: Any,
    reference: Any,
    clean_result: Any,
    *,
    seeds: SeedTriple,
    thresholds: Any,
    regimes: Any,
    paired_provenance: Mapping[str, str],
    allow_calibration: bool = False,
) -> Episode:
    """Materialise the clean side of the pair for B7's all-clean pool."""
    tracking = clean_result.final_tracking
    if tracking is None:
        raise ValueError("clean ILC returned no final rollout; no B7 source can be stored")
    labels = _label_tracking(
        tracking, pair.clean, thresholds, regimes, allow_calibration=allow_calibration
    )
    actions = np.asarray(tracking.actions, dtype=np.float64)
    metadata = {
        "rewards": tracking.rewards,
        "dones": tracking.dones,
        "c_true": np.zeros_like(actions),
        "actions_defective": actions,
        "actions_clean": actions,
        "c_true_clipped": np.zeros(tracking.horizon, dtype=bool),
        "tau_g": tracking.tau_g,
        "wrist_ft": tracking.wrist_ft,
        "defect_state": tracking.defect_state,
        "actuation_state": tracking.actuation,
        "regime": labels,
        "u_inf": np.asarray(clean_result.feedforward, dtype=np.float64),
        "u_inf_clean": np.asarray(clean_result.feedforward, dtype=np.float64),
        "phase": np.asarray(reference.phase, dtype=np.int8),
        "contact": np.asarray(tracking.task_contact, dtype=bool),
        "eef_pos_cmd": tracking.eef_pos_cmd,
        "eef_pos_ach": tracking.eef_pos_ach,
        "tracking_error": tracking.error,
        "joint_torque": tracking.joint_torque,
        "model_file": pair.clean.model.get_xml(),
        "ilc_iters": clean_result.iterations,
        "ilc_status": clean_result.outcome.value,
        "ilc_contact_variant": clean_result.config.contact_variant.value,
        "clean_ilc_iters": clean_result.iterations,
        "clean_ilc_status": clean_result.outcome.value,
        "t_success": tracking.t_success,
        "demo_noise_scale": 0.0,
        "enable_numba": False,
        "assets_commit": paired_provenance["collection_source_commit"],
        "mode_hash": int(tracking.mode_hash),
        "defect_impl_variant": "clean_twin",
        "controller_config_sha256": tracking.controller_config_sha256,
        "c_true_clip_saturation_share": 0.0,
        **paired_provenance,
    }
    return Episode(
        embodiment=EmbodimentId(plan.family),
        task=plan.task,
        seeds=seeds,
        states=tracking.states,
        actions=actions,
        observations={},
        success=tracking.success,
        metadata=metadata,
    )


def _episode_from_ilc(
    plan: CollectionPlan,
    pair: Any,
    reference: Any,
    defective_result: Any,
    clean_result: Any,
    *,
    seeds: SeedTriple,
    thresholds: Any,
    regimes: Any,
    query: Any,
    paired_provenance: Mapping[str, str],
    allow_calibration: bool = False,
) -> Episode:
    """Turn the final real twin rollouts into one stored demonstration."""
    tracking = defective_result.final_tracking
    if tracking is None or clean_result.final_tracking is None:
        raise ValueError("ILC returned no final rollout; no demonstration can be stored")
    if tracking.tau_g.shape[1] != pair.defective.index.n_joints:
        raise ValueError(
            f"recorded tau_g is {tracking.tau_g.shape}, expected one column per "
            f"arm joint ({pair.defective.index.n_joints})"
        )
    if tracking.defect_state.shape[1] == 0:
        raise ValueError("the final defective rollout did not record wrapper memory state")

    twin = query.twin_query(
        pair,
        reference,
        u_inf_defective=defective_result.feedforward,
        u_inf_clean=clean_result.feedforward,
    )
    steps = np.arange(tracking.horizon, dtype=np.int64)
    truth = twin.c_true(tracking.states, steps, tracking.defect_state)
    if not np.allclose(truth.actions_defective, tracking.actions, atol=1e-10, rtol=0.0):
        error = float(np.max(np.abs(truth.actions_defective - tracking.actions)))
        raise AssertionError(
            f"off-branch query does not reproduce the stored defective teacher "
            f"actions (max absolute error {error:.3e})"
        )
    if not np.array_equal(truth.actions_defective - truth.actions_clean, truth.c):
        raise AssertionError(
            "same-state twin query returned an inconsistent action decomposition"
        )

    labels = _label_tracking(
        tracking, pair.defective, thresholds, regimes,
        allow_calibration=allow_calibration,
    )
    from portal_trace.defects.adot import DelayCaliber
    from portal_trace.defects.fields import FieldContext

    derivative_horizon = tracking.horizon - 1
    a_dot_nominal = query.a_dot_nominal(
        twin,
        tracking.states[:derivative_horizon],
        steps[:derivative_horizon],
        caliber=DelayCaliber.ACTION_SIDE,
        actuation=tracking.actuation[:derivative_horizon],
    )
    field_context = FieldContext.from_wrapper(pair.defective)
    injected_joints = tuple(
        index for index, scale in enumerate(pair.delta.joint_scale) if scale != 0.0
    )
    if not injected_joints:
        raise ValueError("defective collection has no active joint for surrogate context")
    metadata = {
        "rewards": tracking.rewards,
        "dones": tracking.dones,
        "c_true": truth.c,
        "actions_defective": truth.actions_defective,
        "actions_clean": truth.actions_clean,
        "c_true_clipped": truth.clipped,
        "tau_g": tracking.tau_g,
        "wrist_ft": tracking.wrist_ft,
        "defect_state": tracking.defect_state,
        "actuation_state": tracking.actuation,
        "regime": labels,
        "u_inf": np.asarray(defective_result.feedforward, dtype=np.float64),
        "u_inf_clean": np.asarray(clean_result.feedforward, dtype=np.float64),
        "phase": np.asarray(reference.phase, dtype=np.int8),
        "contact": np.asarray(tracking.task_contact, dtype=bool),
        "eef_pos_cmd": tracking.eef_pos_cmd,
        "eef_pos_ach": tracking.eef_pos_ach,
        "tracking_error": tracking.error,
        "joint_torque": tracking.joint_torque,
        "surrogate_context": {
            "a_dot_nominal": a_dot_nominal,
            "qpos_index": np.asarray(field_context.qpos_index, dtype=np.int64),
            "qvel_index": np.asarray(field_context.qvel_index, dtype=np.int64),
            "site_id": field_context.site_id,
            "kp": field_context.kp,
            "output_max": field_context.output_max,
            "action_dim": field_context.action_dim,
            "injected_joints": np.asarray(injected_joints, dtype=np.int64),
        },
        "model_file": pair.defective.model.get_xml(),
        "ilc_iters": defective_result.iterations,
        "ilc_status": defective_result.outcome.value,
        "ilc_contact_variant": defective_result.config.contact_variant.value,
        "clean_ilc_iters": clean_result.iterations,
        "clean_ilc_status": clean_result.outcome.value,
        "t_success": tracking.t_success,
        "demo_noise_scale": 0.0,
        "enable_numba": False,
        "assets_commit": paired_provenance["collection_source_commit"],
        "mode_hash": int(tracking.mode_hash),
        "defect_impl_variant": require_module(
            "portal_trace.registry.embodiments", needed_for="collection metadata"
        ).IMPL_VARIANTS[plan.defect],
        "controller_config_sha256": tracking.controller_config_sha256,
        "c_true_clip_saturation_share": truth.saturated_share,
        **paired_provenance,
    }
    return Episode(
        embodiment=EmbodimentId(plan.family, plan.defect, plan.level),
        task=plan.task,
        seeds=seeds,
        states=tracking.states,
        actions=truth.actions_defective,
        observations={},
        success=tracking.success,
        metadata=metadata,
    )


def _label_tracking(
    tracking: Any,
    wrapper: Any,
    thresholds: Any,
    regimes: Any,
    *,
    allow_calibration: bool = False,
) -> np.ndarray:
    """Apply the frozen short-history regime partition to one final rollout."""
    qvel = tracking.states[
        :, int(wrapper.sim.model.nq) + np.asarray(wrapper.index.qvel_index, dtype=np.intp)
    ]
    k_hist = regimes.K_HIST_PRIMARY
    padded = np.concatenate([np.repeat(qvel[:1], k_hist - 1, axis=0), qvel], axis=0)
    history = np.lib.stride_tricks.sliding_window_view(
        padded, window_shape=k_hist, axis=0
    ).transpose(0, 2, 1)
    ee_velocity = np.zeros_like(tracking.eef_pos_cmd)
    ee_velocity[1:] = np.diff(tracking.eef_pos_cmd, axis=0) * CONTROL_FREQ
    signals = regimes.RegimeSignals(
        qdot_history=history,
        tau_g=tracking.tau_g,
        wrist_ft=tracking.wrist_ft,
        ee_velocity=ee_velocity,
    )
    return regimes.label_frames(
        regimes.RegimeFeatures.from_signals(signals), thresholds,
        for_calibration=allow_calibration,
    )


def _write_hdf5(
    plan: CollectionPlan,
    episodes: list[Any],
    *,
    delta: Any,
    ctx: RunContext,
    clean: bool = False,
    root_attrs: Mapping[str, Any] | None = None,
) -> None:
    """Write a complete HDF5 file, then publish it without replacing a cell."""
    output = plan.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite collected dataset {output}")

    fd, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.stem}.",
        suffix=output.suffix,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        _write_hdf5_contents(
            replace(plan, output=temporary),
            episodes,
            delta=delta,
            ctx=ctx,
            clean=clean,
            root_attrs=root_attrs,
        )
        try:
            os.link(temporary, output)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite collected dataset {output}"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _write_hdf5_contents(
    plan: CollectionPlan,
    episodes: list[Any],
    *,
    delta: Any,
    ctx: RunContext,
    clean: bool = False,
    root_attrs: Mapping[str, Any] | None = None,
) -> None:
    """Write the 9.6.3 payload to a private path prior to publication."""
    import h5py

    with h5py.File(plan.output, "w") as fh:
        # Zero-valued dose anchors (notably D2/L0) are defective-design cells,
        # even though their numerical delta matches the clean twin.  Identity
        # therefore comes from the caller's dataset role, never delta.is_zero.
        is_clean = clean
        identity = {
            "task": plan.task,
            "family": plan.family,
            "defect": "clean" if is_clean else plan.defect,
            "level": plan.level,
            "clean_of": plan.defect if is_clean else None,
        }
        stamp_dataset_identity(fh.attrs, **identity)
        stamp_defect_state_contract(fh.attrs, owner_defect=plan.defect)
        stamp_analysis_trace_contract(fh.attrs)
        if episodes and all("surrogate_context" in ep.metadata for ep in episodes):
            fh.attrs["surrogate_context_schema_version"] = _SURROGATE_CONTEXT_SCHEMA
        if episodes:
            binding_ids = {str(ep.metadata.get("paired_binding_id", "")) for ep in episodes}
            source_commits = {
                str(ep.metadata.get("collection_source_commit", "")) for ep in episodes
            }
            runtime_jsons = {
                str(ep.metadata.get("collection_runtime_json", "")) for ep in episodes
            }
            if len(binding_ids) != 1 or len(source_commits) != 1 or len(runtime_jsons) != 1:
                raise ValueError("all episodes in one raw file must share pair provenance")
            stamp_paired_twin_contract(
                fh.attrs,
                role="clean" if is_clean else "defective",
                binding_id=binding_ids.pop(),
                pair_ids=tuple(str(ep.metadata.get("pair_id", "")) for ep in episodes),
                transaction_marker=str(plan.transaction_marker.resolve()),
                source_commit=source_commits.pop(),
                runtime_json=runtime_jsons.pop(),
            )
        if plan.source_validation_reservation_sha256 is not None:
            fh.attrs["source_validation_reservation_sha256"] = (
                plan.source_validation_reservation_sha256
            )
        for key, value in sorted((root_attrs or {}).items()):
            if key in fh.attrs:
                raise ValueError(f"root provenance attribute {key!r} is reserved")
            if not isinstance(value, (str, int, float, bool)):
                raise TypeError(
                    f"root provenance attribute {key!r} must be an HDF5 scalar"
                )
            fh.attrs[key] = value
        data = fh.create_group("data")
        data.attrs["total"] = sum(ep.horizon for ep in episodes)
        data.attrs["env_args"] = _env_args_json(plan)
        for i, ep in enumerate(episodes):
            g = data.create_group(f"demo_{i}")
            if ep.horizon != HORIZON:
                raise ValueError(
                    f"{plan.task}/{plan.family}/{plan.defect} demo {i} has "
                    f"{ep.horizon} frames; 9.6.1 fixes the horizon at {HORIZON} with "
                    "early stopping off, so that alpha_frame equals alpha_demo by "
                    "construction"
                )
            g.attrs["num_samples"] = ep.horizon
            g.attrs["seed_env"] = ep.seeds.env
            g.attrs["defect_id"] = "clean" if is_clean else plan.defect
            g.attrs["defect_level"] = -1 if is_clean else plan.level
            g.attrs["family_id"] = plan.family
            g.attrs["geom_variant"] = plan.geom_variant
            stamp_dataset_identity(g.attrs, **identity)
            stamp_defect_state_contract(g.attrs, owner_defect=plan.defect)
            stamp_analysis_trace_contract(g.attrs)
            g.attrs["paired_twin_schema_version"] = PAIRED_TWIN_SCHEMA_VERSION
            g.attrs["pair_role"] = "clean" if is_clean else "defective"
            for key in (
                "pair_id",
                "paired_binding_id",
                "initial_physics_sha256",
                "initial_actuation_sha256",
                "initial_frame_sha256",
                "reference_spec_sha256",
                "reference_trajectory_sha256",
                "controller_identity_defective_sha256",
                "controller_identity_clean_sha256",
                "teacher_identity_defective_sha256",
                "teacher_identity_clean_sha256",
                "collection_source_commit",
                "collection_runtime_json",
                *(
                    ("source_validation_reservation_sha256",)
                    if plan.source_validation_reservation_sha256 is not None
                    else ()
                ),
            ):
                value = ep.metadata.get(key)
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"{plan.task}/{plan.family}/{plan.defect} demo {i} is missing "
                        f"paired provenance field {key!r}; pilot data is not promoted"
                    )
                g.attrs[key] = value
            for key in (
                "model_file", "ilc_iters", "ilc_status", "ilc_contact_variant",
                "t_success", "demo_noise_scale", "enable_numba", "assets_commit",
                "mode_hash", "defect_impl_variant", "clean_ilc_iters",
                "clean_ilc_status", "controller_config_sha256",
                "c_true_clip_saturation_share",
            ):
                if key not in ep.metadata:
                    raise ValueError(
                        f"{plan.task}/{plan.family}/{plan.defect} demo {i} is missing "
                        f"the per-trajectory metadata field {key!r}. 11.4 makes the "
                        "data card's minimum field set mandatory: an object missing "
                        "any of them may not be used for a preregistered endpoint"
                    )
                g.attrs[key] = ep.metadata[key]
            # Dose pilots replay a converged parent at fixed iteration rungs.
            # Keep that distinction in the trajectory itself so a downstream
            # analysis cannot mistake k=0/2/5 for independently converged ILC
            # runs.  Ordinary collection episodes do not carry these optional
            # fields and retain the existing contract unchanged.
            for key in (
                "ilc_rollout_kind",
                "ilc_parent_status",
                "clean_ilc_parent_status",
                "ilc_source_iteration",
                "clean_ilc_source_iteration",
            ):
                if key in ep.metadata:
                    g.attrs[key] = ep.metadata[key]

            g.create_dataset("states", data=ep.states.astype(np.float64))
            actions_defective = np.asarray(
                ep.metadata.get("actions_defective"), dtype=np.float32
            )
            actions_clean = np.asarray(ep.metadata.get("actions_clean"), dtype=np.float32)
            if (
                actions_defective.shape != ep.actions.shape
                or actions_clean.shape != ep.actions.shape
            ):
                raise ValueError(
                    f"demo {i} paired actions have shapes {actions_defective.shape} and "
                    f"{actions_clean.shape}, expected {ep.actions.shape}"
                )
            stored_actions = np.asarray(ep.actions, dtype=np.float32)
            if not np.array_equal(stored_actions, actions_defective):
                raise ValueError(f"demo {i} Episode.actions is not actions_defective")
            stored_c_true = actions_defective - actions_clean
            claimed_c_true = np.asarray(ep.metadata["c_true"], dtype=np.float32)
            if claimed_c_true.shape != stored_c_true.shape or not np.allclose(
                claimed_c_true, stored_c_true, rtol=0.0, atol=2 * np.finfo(np.float32).eps
            ):
                raise ValueError(
                    f"demo {i} c_true is inconsistent with its same-state twin actions"
                )
            if not np.array_equal(actions_defective - actions_clean, stored_c_true):
                raise AssertionError("stored action decomposition is not exact")
            g.create_dataset("actions", data=actions_defective)
            g.create_dataset("actions_defective", data=actions_defective)
            g.create_dataset("actions_clean", data=actions_clean)
            g.create_dataset("rewards", data=ep.metadata["rewards"].astype(np.float32))
            g.create_dataset("dones", data=ep.metadata["dones"].astype(np.int64))
            from portal_trace.registry.embodiments import PARAMETER_UNITS

            component_names = tuple(sorted(delta.params))
            missing_units = [name for name in component_names if name not in PARAMETER_UNITS]
            if missing_units:
                raise KeyError(
                    f"{delta.defect_code}: injection parameters {missing_units} have no "
                    "unit in PARAMETER_UNITS"
                )
            g.create_dataset(
                "delta_true",
                data=np.asarray(
                    [delta.params[name] for name in component_names], dtype=np.float64
                ),
            )
            g.attrs["delta_fields"] = json.dumps(
                component_names, separators=(",", ":")
            )
            g.attrs["delta_units"] = json.dumps(
                [PARAMETER_UNITS[name] for name in component_names], separators=(",", ":")
            )
            g.attrs["delta_json"] = delta.dumps()
            g.create_dataset("c_true", data=stored_c_true)
            analysis_fields = {
                "c_true_clipped": (None, np.bool_),
                "u_inf": (6, np.float32),
                "u_inf_clean": (6, np.float32),
                "phase": (None, np.int8),
                "contact": (None, np.int8),
                "eef_pos_cmd": (3, np.float32),
                "eef_pos_ach": (3, np.float32),
                "tracking_error": (6, np.float32),
                "joint_torque": (None, np.float32),
            }
            for name, (width, dtype) in analysis_fields.items():
                if name not in ep.metadata:
                    raise ValueError(
                        f"{plan.task}/{plan.family}/{plan.defect} demo {i} is missing "
                        f"{name}; the analysis-trace contract requires the exact "
                        "collection-time control boundary"
                    )
                values = np.asarray(ep.metadata[name])
                if values.ndim == 1:
                    valid = values.shape[0] == ep.horizon and width is None
                else:
                    valid = (
                        values.ndim == 2
                        and values.shape[0] == ep.horizon
                        and (width is None or values.shape[1] == width)
                    )
                if not valid:
                    expected = "(T,)" if width is None else f"(T, {width})"
                    raise ValueError(
                        f"demo {i} {name} has shape {values.shape}, expected {expected}"
                    )
                g.create_dataset(name, data=values.astype(dtype))
            for signal, width in (("tau_g", None), ("wrist_ft", 6)):
                if signal not in ep.metadata:
                    raise ValueError(
                        f"{plan.task}/{plan.family}/{plan.defect} demo {i} is missing "
                        f"{signal}; the 70-wide policy input cannot be reconstructed "
                        "during materialisation"
                    )
                values = np.asarray(ep.metadata[signal], dtype=np.float32)
                valid = (
                    values.ndim == 2
                    and values.shape[0] == ep.horizon
                    and (
                        1 <= values.shape[1] <= 7
                        if width is None
                        else values.shape[1] == width
                    )
                )
                if not valid:
                    expected = "1..7" if width is None else str(width)
                    raise ValueError(
                        f"demo {i} {signal} has shape {values.shape}, expected "
                        f"({ep.horizon}, {expected})"
                    )
                g.create_dataset(signal, data=values)
            if "surrogate_context" in ep.metadata:
                context = ep.metadata["surrogate_context"]
                required_context = {
                    "a_dot_nominal", "qpos_index", "qvel_index", "site_id", "kp",
                    "output_max", "action_dim", "injected_joints",
                }
                if not isinstance(context, Mapping) or set(context) != required_context:
                    raise ValueError(f"demo {i} surrogate context is incomplete")
                a_dot = np.asarray(context["a_dot_nominal"], dtype=np.float64)
                if a_dot.shape != (ep.horizon - 1, 6) or not np.isfinite(a_dot).all():
                    raise ValueError(
                        f"demo {i} a_dot_nominal has shape {a_dot.shape}, expected "
                        f"({ep.horizon - 1}, 6)"
                    )
                surrogate = g.create_group("surrogate_context")
                surrogate.attrs["schema_version"] = _SURROGATE_CONTEXT_SCHEMA
                for name in (
                    "a_dot_nominal", "qpos_index", "qvel_index", "kp", "output_max",
                    "injected_joints",
                ):
                    surrogate.create_dataset(name, data=context[name])
                surrogate.attrs["site_id"] = int(context["site_id"])
                surrogate.attrs["action_dim"] = int(context["action_dim"])
            actuation = np.asarray(ep.metadata["actuation_state"], dtype=np.float32)
            if actuation.ndim != 2 or actuation.shape[0] != ep.horizon:
                raise ValueError(
                    f"demo {i} actuation_state has shape {actuation.shape}, expected "
                    f"({ep.horizon}, d)"
                )
            g.create_dataset("actuation_state", data=actuation)
            g.create_dataset(
                "defect_state", data=ep.metadata["defect_state"].astype(np.float32)
            )
            g.create_dataset(
                "regime", data=ep.metadata["regime"].astype(np.int8)
            )
        fh.attrs["environment"] = str(ctx.report.backend.value)
        fh.attrs["proprio_dim"] = PROPRIO_DIM


def _env_args_json(plan: CollectionPlan) -> str:
    spec = task_spec(plan.task)
    env_kwargs = dict(spec.env_kwargs)
    env_kwargs.update(
        robots=family_spec(plan.family).robosuite_robot,
        control_freq=CONTROL_FREQ,
        horizon=HORIZON,
        lite_physics=True,
        hard_reset=True,
    )
    return json.dumps(
        {
            "env_name": spec.env_name,
            "env_version": "1.5.2",
            # robomimic 0.5.0 reads this key literally; `env_type` is not an alias.
            "type": 1,
            "env_kwargs": env_kwargs,
        },
        sort_keys=True,
    )


def collect_request(
    *, experiment_id: str, task: str, family: str, defect: str, backend: Any, **kw: Any
) -> RunRequest:
    """Build the request a collection run validates against."""
    return RunRequest(
        experiment_id=experiment_id,
        stage="collect",
        backend=backend,
        task=task,
        family=family,
        defect=defect,
        **kw,
    )
