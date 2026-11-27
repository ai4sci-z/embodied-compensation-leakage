"""Bounded engineering producer for a real B7 clean-twin data slice.

The formal B7 Oracle requires 1,400 clean demonstrations for every core task.
This producer intentionally creates a smaller single-task slice so the real
ILC -> materialisation -> formal-loader path can be exercised before that full
collection is affordable.  Its HDF5 and manifest carry fail-closed partial-
budget markers; they are never formal B7 or paper evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from portal_trace.lockfile import source_commit
from portal_trace.registry.tasks import CORE_TASKS, TaskRole, task as task_spec
from portal_trace.runners.collect import B7_CLEAN_DEMOS_PER_TASK
from portal_trace.runners.formal_adapter_pilot import formal_pool_entries_from_datasets
from portal_trace.runners.formal_data import (
    build_formal_data_plan,
    fit_formal_normalization,
    load_formal_normalization,
)
from portal_trace.runners.materialise import MaterialisationSpec, _materialise_one
from portal_trace.runners.multiseed_expert import _sha256, merge_expert_sources
from portal_trace.runners.pilot import pilot_thresholds, run_ilc_pilot

B7_CLEAN_SLICE_SCHEMA = "portal_trace.engineering_b7_clean_slice/1"
B7_CLEAN_SLICE_MARKER = (
    "PORTAL_TRACE_ENGINEERING_B7_CLEAN_SLICE_PARTIAL_BUDGET_NOT_PAPER_RESULT"
)
B7_CLEAN_SLICE_RELEASE_STATUS = "engineering_only"
B7_CLEAN_SLICE_BUDGET_STATUS = "partial"
B7_CLEAN_SLICE_MIN_SEEDS = 5


@dataclass(frozen=True)
class B7CleanSliceArtifacts:
    dataset: Path
    plan: Path
    normalization: Path
    manifest: Path
    train_seeds: tuple[int, ...]
    heldout_seeds: tuple[int, ...]


def _publish_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite B7 clean-slice manifest {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def require_clean_tracked_worktree(repository: Path | None = None) -> None:
    repository = (
        Path(repository).resolve()
        if repository is not None
        else Path(__file__).resolve().parents[2]
    )
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    )
    if result.stdout.strip():
        raise RuntimeError(
            "B7 clean-slice provenance requires a clean tracked worktree; "
            "commit tracked producer changes before collection"
        )


def _threshold_inputs(path: Path) -> tuple[Any, dict[str, Any]]:
    report_bytes = Path(path).read_bytes()
    report = json.loads(report_bytes)
    thresholds = pilot_thresholds(report)
    provenance = {
        "pilot_data_marker": B7_CLEAN_SLICE_MARKER,
        "pilot_source_commit": source_commit(),
        "threshold_report_path": str(Path(path).resolve()),
        "threshold_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "threshold_report_source_commit": str(report.get("source_commit", "unknown")),
        "threshold_report_status": str(report.get("status", "legacy_unspecified")),
        "thresholds_json": json.dumps(
            thresholds.as_dict(), sort_keys=True, separators=(",", ":")
        ),
    }
    return thresholds, provenance


def _seed_split(plan: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    by_id = {row["id"]: int(row["seed_env"]) for row in plan.payload["pool"]}
    train = tuple(sorted(by_id[item] for item in plan.train_episode_ids))
    heldout = tuple(sorted(by_id[item] for item in plan.heldout_episode_ids))
    if set(train) & set(heldout):
        raise RuntimeError("B7 clean-slice plan leaked an environment seed across splits")
    return train, heldout


def build_b7_clean_slice(
    *,
    task: str,
    family: str,
    owner_defect: str,
    level: int,
    seed_envs: tuple[int, ...],
    output_root: Path,
    threshold_report: Path,
    tol: float,
    environment_fingerprint: str,
    split_seed: int = 20260919,
    train_fraction: float = 0.8,
    k_max: int = 8,
) -> B7CleanSliceArtifacts:
    """Collect and publish one provenance-bound, partial-budget clean twin."""
    spec = task_spec(task)
    if spec.role is TaskRole.BASE:
        raise ValueError(f"B7 clean slice requires a non-base task, got {task!r}")
    if len(seed_envs) < B7_CLEAN_SLICE_MIN_SEEDS:
        raise ValueError(
            f"B7 clean slice requires at least {B7_CLEAN_SLICE_MIN_SEEDS} "
            "independent seeds for a useful 4/1 split"
        )
    if len(set(seed_envs)) != len(seed_envs):
        raise ValueError("B7 clean-slice environment seeds must be unique")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seed_envs):
        raise ValueError("B7 clean-slice environment seeds must be non-negative integers")
    if not family or not owner_defect or not environment_fingerprint:
        raise ValueError("family, owner_defect and environment_fingerprint are required")
    if len(seed_envs) >= B7_CLEAN_DEMOS_PER_TASK:
        raise ValueError(
            "engineering B7 clean slice must remain below the 1,400-demo formal budget"
        )

    require_clean_tracked_worktree()
    output_root = Path(output_root).resolve()
    dataset_path = output_root / "b7-clean-slice.hdf5"
    plan_path = output_root / "formal-plan.json"
    normalization_path = output_root / "normalization.json"
    manifest_path = output_root / "manifest.json"
    for path in (dataset_path, plan_path, normalization_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite B7 clean-slice artifact {path}")

    producer_commit = source_commit()
    thresholds, threshold_provenance = _threshold_inputs(threshold_report)
    clean_obs_sources: list[Path] = []
    clean_raw_sources: list[Path] = []
    for seed_env in seed_envs:
        source_root = output_root / "sources" / f"seed-{seed_env}"
        defective_raw = (
            source_root
            / "raw"
            / task
            / family
            / owner_defect
            / f"L{level}"
            / "demo.hdf5"
        )
        clean_raw = (
            source_root
            / "raw"
            / task
            / family
            / f"clean-{owner_defect}"
            / f"L{level}"
            / "demo.hdf5"
        )
        clean_obs = (
            source_root
            / "obs"
            / task
            / family
            / f"clean-{owner_defect}"
            / f"L{level}"
            / "obs_84_ag_eih.hdf5"
        )
        if clean_obs.exists():
            if not clean_raw.exists() or not defective_raw.exists():
                raise RuntimeError(
                    f"seed {seed_env}: materialised clean source exists without its "
                    "complete raw twin pair"
                )
        else:
            if clean_raw.exists() or defective_raw.exists():
                if not clean_raw.exists() or not defective_raw.exists():
                    raise RuntimeError(
                        f"seed {seed_env}: incomplete raw twin pair cannot be resumed"
                    )
            else:
                _defective, produced_clean_raw = run_ilc_pilot(
                    task=task,
                    family=family,
                    defect=owner_defect,
                    level=level,
                    seed_env=seed_env,
                    output=defective_raw,
                    thresholds=thresholds,
                    tol=tol,
                    k_max=k_max,
                    threshold_provenance=threshold_provenance,
                )
                if produced_clean_raw.resolve() != clean_raw.resolve():
                    raise RuntimeError("ILC pilot returned an unexpected clean-twin path")
            _materialise_one(
                MaterialisationSpec(raw=clean_raw, output=clean_obs, level=level)
            )
        clean_raw_sources.append(clean_raw.resolve())
        clean_obs_sources.append(clean_obs.resolve())

    records = merge_expert_sources(
        tuple(clean_obs_sources),
        output=dataset_path,
        producer_source_commit=producer_commit,
        raw_sources=tuple(clean_raw_sources),
        root_provenance={
            "b7_clean_slice_schema": B7_CLEAN_SLICE_SCHEMA,
            "b7_clean_slice_marker": B7_CLEAN_SLICE_MARKER,
            "b7_clean_slice_release_status": B7_CLEAN_SLICE_RELEASE_STATUS,
            "b7_clean_slice_budget_status": B7_CLEAN_SLICE_BUDGET_STATUS,
            "b7_clean_slice_formal_demos_per_task": B7_CLEAN_DEMOS_PER_TASK,
            "b7_clean_slice_observed_demos": len(seed_envs),
            "b7_clean_slice_task": task,
            "b7_clean_slice_target_family": family,
            "b7_clean_slice_owner_defect": owner_defect,
            "b7_clean_slice_level": level,
            "b7_clean_slice_producer_commit": producer_commit,
        },
    )
    expected_identity = (task, family, "clean", level, owner_defect)
    actual_identities = {
        (row.task, row.family, row.defect, row.level, row.clean_of) for row in records
    }
    if actual_identities != {expected_identity}:
        raise RuntimeError(
            f"B7 clean-slice identity changed during assembly: {sorted(actual_identities)}"
        )

    entries = formal_pool_entries_from_datasets((dataset_path,))
    plan = build_formal_data_plan(
        entries,
        seed=split_seed,
        train_fraction=train_fraction,
        source_commit=producer_commit,
        environment_fingerprint=environment_fingerprint,
    )
    plan.write(plan_path)
    normalization = fit_formal_normalization(plan)
    normalization.write(normalization_path)
    normalization = load_formal_normalization(normalization_path, plan=plan)
    train_seeds, heldout_seeds = _seed_split(plan)

    payload = {
        "schema": B7_CLEAN_SLICE_SCHEMA,
        "marker": B7_CLEAN_SLICE_MARKER,
        "release_status": B7_CLEAN_SLICE_RELEASE_STATUS,
        "producer_source_commit": producer_commit,
        "worktree_dirty": False,
        "worktree_dirty_scope": "tracked_files_only",
        "identity": {
            "task": task,
            "task_role": spec.role.value,
            "family": family,
            "defect": "clean",
            "clean_of": owner_defect,
            "level": level,
        },
        "budget": {
            "status": B7_CLEAN_SLICE_BUDGET_STATUS,
            "observed_demos": len(records),
            "formal_demos_per_task": B7_CLEAN_DEMOS_PER_TASK,
            "formal_task_count": len(CORE_TASKS),
            "formal_tasks": list(CORE_TASKS),
        },
        "admitted_for": ["B7_oracle_engineering_loader_probe"],
        "forbidden_for": [
            "formal_B7_oracle",
            "paper_result",
            "full_demo_budget_claim",
        ],
        "sources": [row.manifest_row() for row in records],
        "split": {
            "seed": split_seed,
            "train_fraction": train_fraction,
            "train_seeds": list(train_seeds),
            "heldout_seeds": list(heldout_seeds),
            "seed_overlap": [],
        },
        "artifacts": {
            "dataset": str(dataset_path),
            "dataset_sha256": _sha256(dataset_path),
            "plan": str(plan_path),
            "plan_sha256": plan.plan_sha256,
            "plan_file_sha256": _sha256(plan_path),
            "normalization": str(normalization_path),
            "normalization_file_sha256": _sha256(normalization_path),
            "normalization_artifact_sha256": normalization.artifact_sha256,
        },
    }
    _publish_json(manifest_path, payload)
    return B7CleanSliceArtifacts(
        dataset=dataset_path,
        plan=plan_path,
        normalization=normalization_path,
        manifest=manifest_path,
        train_seeds=train_seeds,
        heldout_seeds=heldout_seeds,
    )
