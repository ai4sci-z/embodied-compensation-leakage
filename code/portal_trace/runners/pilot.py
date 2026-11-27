"""Small real-data ILC pilot runner.

This is deliberately separate from the preregistered collection stage.  It
executes the same twin/ILC/episode/HDF5 path for one explicitly chosen cell,
without pretending that one seed is a calibration result.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from portal_trace.defects.delta import Delta, default_joint_scale
from portal_trace.defects.inject import build_twin_pair
from portal_trace.lockfile import source_commit
from portal_trace.registry.families import descriptor as family_descriptor
from portal_trace.registry.tasks import task as task_spec
from portal_trace.runners.backends import get_backend
from portal_trace.types import SeedTriple

DOSE_PILOT_MARKER = "PORTAL_TRACE_ILC_DOSE_PILOT_NOT_FORMAL"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_ilc_pilot(
    *,
    task: str,
    family: str,
    defect: str,
    level: int,
    seed_env: int,
    output: Path,
    thresholds: Any,
    tol: float,
    k_max: int = 8,
    threshold_provenance: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Run one real twin pair and publish defective/clean raw HDF5 files."""
    from portal_trace.diagnostics import regimes
    from portal_trace.runners.collect import (
        CollectionPlan,
        _clean_episode_from_ilc,
        _episode_from_ilc,
        _paired_episode_provenance,
        _publish_pair_marker,
        _write_hdf5,
    )
    from portal_trace.teacher import ilc, osc, query, refs

    spec = task_spec(task)
    delta = Delta.from_level(
        defect,
        level,
        joint_scale=default_joint_scale(defect, family_descriptor(family).dof),
    )
    seeds = SeedTriple(env=seed_env, train=-1, eval=0)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pair = build_twin_pair(spec, family, delta, seeds)
    plan = CollectionPlan(
        task=task,
        family=family,
        defect=defect,
        level=level,
        n_demos=1,
        env_seeds=(seed_env,),
        geom_variant="G0",
        output=output,
    )
    # Match the production level-aware tree: clean companions live beside the
    # defective cell under ``clean-<defect>/L<level>``.
    clean_output = plan.clean_output
    # _write_hdf5 only needs the backend value for provenance.  Formal
    # preflight is intentionally not invoked by this engineering-only runner.
    ctx = SimpleNamespace(report=SimpleNamespace(backend=SimpleNamespace(value="cpu_mujoco")))
    run_source_commit = source_commit()
    runtime_json = json.dumps(
        get_backend("cpu_mujoco").fingerprint(lite_physics=True).to_json(),
        sort_keys=True,
        separators=(",", ":"),
    )
    try:
        initial_physics = osc.reset_twin_pair(pair, seed_env)
        initial_actuation = pair.clean.actuation_state().copy()
        reference_spec = refs.load_spec(task)
        reference = refs.resolve(reference_spec, pair.defective.sim)
        # Match production collection, including D2's delay-aligned lead.  The
        # pilot is only useful when it exercises the same teacher update law.
        config = ilc.ILCConfig.for_delta(delta, k_max=k_max)

        def reset_pair() -> None:
            osc.reset_twin_pair(pair, seed_env)

        defective = ilc.run_ilc(
            pair.defective, reference, family=family, seed_env=seed_env,
            tol=tol, config=config, reset=reset_pair,
        )
        clean = ilc.run_ilc(
            pair.clean, reference, family=family, seed_env=seed_env,
            tol=tol, config=config, reset=reset_pair,
        )
        if defective.final_tracking is None or clean.final_tracking is None:
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
            controller_sha256=defective.final_tracking.controller_config_sha256,
            clean_controller_sha256=clean.final_tracking.controller_config_sha256,
            run_source_commit=run_source_commit,
            runtime_json=runtime_json,
        )
        episode = _episode_from_ilc(
            plan, pair, reference, defective, clean, seeds=seeds,
            thresholds=thresholds, regimes=regimes, query=query,
            paired_provenance=paired_provenance,
            allow_calibration=True,
        )
        clean_episode = _clean_episode_from_ilc(
            plan, pair, reference, clean, seeds=seeds,
            thresholds=thresholds, regimes=regimes, allow_calibration=True,
            paired_provenance=paired_provenance,
        )
        clean_plan = replace(plan, output=clean_output)
        _write_hdf5(
            plan,
            [episode],
            delta=delta,
            ctx=ctx,
            clean=False,
            root_attrs=threshold_provenance,
        )
        _write_hdf5(
            clean_plan,
            [clean_episode],
            delta=delta.zero(),
            ctx=ctx,
            clean=True,
            root_attrs=threshold_provenance,
        )
        _publish_pair_marker(plan)
        return output, clean_output
    finally:
        pair.defective.close()
        pair.clean.close()


def run_ilc_dose_pilot(
    *,
    task: str,
    family: str,
    defect: str,
    level: int,
    seed_env: int,
    output_root: Path,
    thresholds: Any,
    tol: float,
    k_max: int = 20,
    threshold_provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Run one ILC pair, then publish full demonstrations for A9's five doses.

    This is an engineering pilot, not a formal E3/A9 result.  It makes the
    preregistered iteration axis executable without rerunning ILC independently
    at every rung: each stored episode is a fresh deterministic replay using
    the feedforward produced by the shared parent run.
    """
    from portal_trace.diagnostics import regimes
    from portal_trace.runners.collect import (
        CollectionPlan,
        _clean_episode_from_ilc,
        _episode_from_ilc,
        _paired_episode_provenance,
        _publish_pair_marker,
        _write_hdf5,
    )
    from portal_trace.teacher import ilc, osc, query, refs

    if k_max < max(ilc.DOSE_LADDER):
        raise ValueError(
            f"k_max={k_max} cannot observe the fixed dose ladder {ilc.DOSE_LADDER}"
        )

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite dose manifest {manifest_path}")

    spec = task_spec(task)
    delta = Delta.from_level(
        defect,
        level,
        joint_scale=default_joint_scale(defect, family_descriptor(family).dof),
    )
    seeds = SeedTriple(env=seed_env, train=-1, eval=0)
    pair = build_twin_pair(spec, family, delta, seeds)
    ctx = SimpleNamespace(report=SimpleNamespace(backend=SimpleNamespace(value="cpu_mujoco")))
    run_source_commit = source_commit()
    runtime_json = json.dumps(
        get_backend("cpu_mujoco").fingerprint(lite_physics=True).to_json(),
        sort_keys=True,
        separators=(",", ":"),
    )
    provenance = dict(threshold_provenance or {})
    provenance.update(
        pilot_data_marker=DOSE_PILOT_MARKER,
        pilot_source_commit=run_source_commit,
    )

    try:
        initial_physics = osc.reset_twin_pair(pair, seed_env)
        initial_actuation = pair.clean.actuation_state().copy()
        reference_spec = refs.load_spec(task)
        reference = refs.resolve(reference_spec, pair.defective.sim)
        config = ilc.ILCConfig.for_delta(delta, k_max=k_max)

        def reset_pair() -> None:
            osc.reset_twin_pair(pair, seed_env)

        defective = ilc.run_ilc(
            pair.defective,
            reference,
            family=family,
            seed_env=seed_env,
            tol=tol,
            config=config,
            reset=reset_pair,
        )
        clean = ilc.run_ilc(
            pair.clean,
            reference,
            family=family,
            seed_env=seed_env,
            tol=tol,
            config=config,
            reset=reset_pair,
        )
        if not defective.converged or not clean.converged:
            raise RuntimeError(
                "the five-level dose pilot requires converged defective and clean "
                f"parents; got {defective.outcome.value}/{clean.outcome.value}"
            )

        records: list[dict[str, Any]] = []
        requested_doses: tuple[tuple[str, int | None], ...] = (
            *((f"k{k}", k) for k in ilc.DOSE_LADDER),
            ("k_conv", None),
        )
        tracker = osc.OSCTracker()
        for label, requested in requested_doses:
            if requested is None:
                assert defective.k_conv is not None and clean.k_conv is not None
                u_defective = defective.feedforward
                u_clean = clean.feedforward
                def_source = defective.k_conv
                clean_source = clean.k_conv
            else:
                u_defective = defective.dose_feedforward(requested)
                u_clean = clean.dose_feedforward(requested)
                def_source = defective.dose_source_iteration(requested)
                clean_source = clean.dose_source_iteration(requested)

            reset_pair()
            defective_tracking = tracker.track(
                pair.defective,
                reference,
                family=family,
                seed_env=seed_env,
                feedforward=u_defective,
            )
            reset_pair()
            clean_tracking = tracker.track(
                pair.clean,
                reference,
                family=family,
                seed_env=seed_env,
                feedforward=u_clean,
            )

            defective_view = SimpleNamespace(
                final_tracking=defective_tracking,
                feedforward=u_defective,
                iterations=def_source,
                outcome=defective.outcome,
                config=defective.config,
            )
            clean_view = SimpleNamespace(
                final_tracking=clean_tracking,
                feedforward=u_clean,
                iterations=clean_source,
                outcome=clean.outcome,
                config=clean.config,
            )
            output = (
                output_root
                / label
                / task
                / family
                / defect
                / f"L{level}"
                / "demo.hdf5"
            )
            plan = CollectionPlan(
                task=task,
                family=family,
                defect=defect,
                level=level,
                n_demos=1,
                env_seeds=(seed_env,),
                geom_variant="G0",
                output=output,
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
                controller_sha256=defective_tracking.controller_config_sha256,
                clean_controller_sha256=clean_tracking.controller_config_sha256,
                run_source_commit=run_source_commit,
                runtime_json=runtime_json,
            )
            episode = _episode_from_ilc(
                plan,
                pair,
                reference,
                defective_view,
                clean_view,
                seeds=seeds,
                thresholds=thresholds,
                regimes=regimes,
                query=query,
                paired_provenance=paired_provenance,
                allow_calibration=True,
            )
            clean_episode = _clean_episode_from_ilc(
                plan,
                pair,
                reference,
                clean_view,
                seeds=seeds,
                thresholds=thresholds,
                regimes=regimes,
                paired_provenance=paired_provenance,
                allow_calibration=True,
            )
            episode.metadata.update(
                ilc_rollout_kind="dose_replay",
                ilc_parent_status=defective.outcome.value,
                clean_ilc_parent_status=clean.outcome.value,
                ilc_source_iteration=def_source,
                clean_ilc_source_iteration=clean_source,
            )
            clean_episode.metadata.update(
                ilc_rollout_kind="dose_replay",
                ilc_parent_status=defective.outcome.value,
                clean_ilc_parent_status=clean.outcome.value,
                ilc_source_iteration=def_source,
                clean_ilc_source_iteration=clean_source,
            )
            dose_attrs = dict(provenance)
            dose_attrs.update(
                ilc_dose_label=label,
                ilc_dose_requested_iteration=-1 if requested is None else requested,
                ilc_dose_defective_source_iteration=def_source,
                ilc_dose_clean_source_iteration=clean_source,
                ilc_parent_defective_k_conv=defective.k_conv,
                ilc_parent_clean_k_conv=clean.k_conv,
            )
            clean_output = plan.clean_output
            _write_hdf5(
                plan,
                [episode],
                delta=delta,
                ctx=ctx,
                clean=False,
                root_attrs=dose_attrs,
            )
            _write_hdf5(
                replace(plan, output=clean_output),
                [clean_episode],
                delta=delta.zero(),
                ctx=ctx,
                clean=True,
                root_attrs=dose_attrs,
            )
            _publish_pair_marker(plan)
            records.append(
                {
                    "label": label,
                    "requested_iteration": requested,
                    "defective_source_iteration": def_source,
                    "clean_source_iteration": clean_source,
                    "defective_success": bool(defective_tracking.success),
                    "clean_success": bool(clean_tracking.success),
                    "defective_t_success": int(defective_tracking.t_success),
                    "clean_t_success": int(clean_tracking.t_success),
                    "defective_rms": float(defective_tracking.rms()),
                    "clean_rms": float(clean_tracking.rms()),
                    "defective_path": str(output.resolve()),
                    "clean_path": str(clean_output.resolve()),
                    "defective_sha256": _sha256(output),
                    "clean_sha256": _sha256(clean_output),
                }
            )

        manifest = {
            "schema": "portal_trace.ilc_dose_pilot/1",
            "marker": DOSE_PILOT_MARKER,
            "source_commit": run_source_commit,
            "task": task,
            "family": family,
            "defect": defect,
            "level": level,
            "seed_env": seed_env,
            "tol_m": tol,
            "parent_defective": defective.summary(),
            "parent_clean": clean.summary(),
            "doses": records,
        }
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return manifest_path
    finally:
        pair.defective.close()
        pair.clean.close()


def pilot_thresholds(report: dict[str, Any]) -> Any:
    """Build thresholds from a regime calibration JSON artifact."""
    from portal_trace.diagnostics.regimes import Provenance, RegimeThresholds

    values = report.get("thresholds", report)
    return RegimeThresholds(
        f_th=float(values["f_th"]),
        eps_v=float(values["eps_v"]),
        eps_qs=float(values["eps_qs"]),
        tau_min=float(values["tau_min"]),
        eps_fs=float(values["eps_fs"]),
        provenance=Provenance.CALIBRATION_ITERATE,
    )
