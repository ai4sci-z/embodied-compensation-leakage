"""Collect and calibrate the frozen 9.7.1 regime thresholds.

The command is deliberately split in two. ``collect`` runs the real
uncompensated OSC tracker (ILC k=0) and writes an immutable feature artifact.
``calibrate`` combines explicitly named artifacts, writes a JSON report, and
only modifies the lock file when ``--freeze`` is supplied and the coordinate
descent has converged. Freezing additionally requires an explicit sampling
protocol fixed after P0. A pilot artifact is therefore useful for validating
the pipeline but cannot silently become a formal measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from portal_trace.configs.loader import ConfigBundle, load_bundle
from portal_trace.diagnostics.regimes import (
    OCCUPANCY_BAND,
    K_HIST_PRIMARY,
    RegimeFeatures,
    RegimeSignals,
    calibrate_regime_thresholds,
    label_frames,
    occupancy_mass,
)
from portal_trace.lockfile import NO_CHECKOUT, LockFile, source_commit
from portal_trace.registry.tasks import CONTROL_FREQ, HORIZON
from portal_trace.registry.tasks import task as task_spec
from portal_trace.types import Regime, SeedTriple

__all__ = [
    "features_from_tracking",
    "collect_feature_artifact",
    "load_feature_artifacts",
    "calibrate_artifacts",
    "cross_validate_artifacts",
    "main",
]

_FEATURE_NAMES = (
    "wrist_force_norm",
    "reversal_strength",
    "qdot_norm",
    "tau_g_inf",
    "ee_speed",
)
_ARTIFACT_SCHEMA = 2
_PROTOCOL_SCHEMA = 1


def features_from_tracking(tracking: Any, wrapper: Any) -> RegimeFeatures:
    """Build the causal five-frame regime features from one real rollout."""
    nq = int(wrapper.sim.model.nq)
    qvel_index = np.asarray(wrapper.index.qvel_index, dtype=np.intp)
    qvel = np.asarray(tracking.states, dtype=np.float64)[:, nq + qvel_index]
    k_hist = K_HIST_PRIMARY
    padded = np.concatenate([np.repeat(qvel[:1], k_hist - 1, axis=0), qvel], axis=0)
    history = np.lib.stride_tricks.sliding_window_view(
        padded, window_shape=k_hist, axis=0
    ).transpose(0, 2, 1)
    ee_velocity = np.zeros_like(tracking.eef_pos_cmd, dtype=np.float64)
    ee_velocity[1:] = np.diff(tracking.eef_pos_cmd, axis=0) * CONTROL_FREQ
    return RegimeFeatures.from_signals(
        RegimeSignals(
            qdot_history=history,
            tau_g=np.asarray(tracking.tau_g, dtype=np.float64),
            wrist_ft=np.asarray(tracking.wrist_ft, dtype=np.float64),
            ee_velocity=ee_velocity,
        )
    )


def _concatenate(features: Sequence[RegimeFeatures]) -> RegimeFeatures:
    if not features:
        raise ValueError("no regime feature artifacts were supplied")
    k_hist = {item.k_hist for item in features}
    crossing = {item.steps_since_crossing for item in features}
    if len(k_hist) != 1 or len(crossing) != 1:
        raise ValueError(
            "regime artifacts disagree on K_hist or steps_since_crossing; "
            "thresholds calibrated on different definitions cannot be pooled"
        )
    return RegimeFeatures(
        **{
            name: np.concatenate(
                [np.asarray(getattr(item, name), dtype=np.float64) for item in features]
            )
            for name in _FEATURE_NAMES
        },
        k_hist=next(iter(k_hist)),
        steps_since_crossing=next(iter(crossing)),
    )


def _parse_cell(value: str) -> tuple[str, int]:
    try:
        defect, raw_level = value.split(":", 1)
        return defect, int(raw_level)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            f"cell {value!r} must be DEFECT:LEVEL, for example D2:1"
        ) from exc


def _worktree_dirty() -> bool | None:
    """Whether tracked or untracked source differs from the recorded commit."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return bool(result.stdout.strip())


def _save_artifact(
    output: Path,
    features: RegimeFeatures,
    *,
    metadata: dict[str, Any],
    cell_index: np.ndarray,
    rollout_index: np.ndarray,
) -> None:
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable regime feature artifact {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **{name: np.asarray(getattr(features, name), dtype=np.float64) for name in _FEATURE_NAMES},
        k_hist=np.asarray(features.k_hist, dtype=np.int64),
        steps_since_crossing=np.asarray(features.steps_since_crossing, dtype=np.int64),
        cell_index=np.asarray(cell_index, dtype=np.int32),
        rollout_index=np.asarray(rollout_index, dtype=np.int32),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
    )


def collect_feature_artifact(
    *,
    task: str,
    family: str,
    cells: Sequence[tuple[str, int]],
    n_rollouts: int,
    start_index: int,
    output: Path,
    bundle: ConfigBundle | None = None,
) -> Path:
    """Run explicit (defect, level) cells with the uncompensated tracker."""
    if not cells:
        raise ValueError("at least one --cell DEFECT:LEVEL is required")
    if n_rollouts < 1:
        raise ValueError(f"n_rollouts must be positive, got {n_rollouts}")
    if start_index < 0:
        raise ValueError(f"start_index must be non-negative, got {start_index}")

    from portal_trace.defects.delta import Delta, default_joint_scale
    from portal_trace.defects.inject import build_twin_pair
    from portal_trace.registry.families import descriptor
    from portal_trace.teacher import osc, refs

    resolved_bundle = bundle or load_bundle()
    spec = task_spec(task)
    n_dof = descriptor(family).dof
    reference_spec = refs.load_spec(task)
    eval_seed = resolved_bundle.seeds.eval_seed(task=task, family=family)
    all_features: list[RegimeFeatures] = []
    cell_rows: list[np.ndarray] = []
    rollout_rows: list[np.ndarray] = []
    seed_envs: list[int] = []

    for cell_id, (defect, level) in enumerate(cells):
        delta = Delta.from_level(defect, level, joint_scale=default_joint_scale(defect, n_dof))
        for rollout_offset in range(n_rollouts):
            demo_index = start_index + rollout_offset
            seed_env = resolved_bundle.seeds.env_seed(
                task=task, family=family, demo_index=demo_index
            )
            seeds = SeedTriple(env=seed_env, train=-1, eval=eval_seed)
            pair = build_twin_pair(spec, family, delta, seeds)
            try:
                osc.reset_twin_pair(pair, seed_env)
                reference = refs.resolve(reference_spec, pair.defective.sim)
                tracking = osc.OSCTracker().track(
                    pair.defective,
                    reference,
                    family=family,
                    seed_env=seed_env,
                    feedforward=None,
                )
                features = features_from_tracking(tracking, pair.defective)
                all_features.append(features)
                cell_rows.append(np.full(features.n_frames, cell_id, dtype=np.int32))
                rollout_rows.append(
                    np.full(features.n_frames, rollout_offset, dtype=np.int32)
                )
                seed_envs.append(seed_env)
            finally:
                pair.defective.close()
                pair.clean.close()

    combined = _concatenate(all_features)
    metadata = {
        "schema_version": _ARTIFACT_SCHEMA,
        "producer": "portal_trace.scripts.calibrate_regimes collect",
        "source_commit": source_commit(),
        "worktree_dirty": _worktree_dirty(),
        "task": task,
        "family": family,
        "cells": [{"defect": defect, "level": level} for defect, level in cells],
        "n_rollouts_per_cell": n_rollouts,
        "start_index": start_index,
        "seed_envs": seed_envs,
        "tracker": "OSCTracker, feedforward=None (ILC k=0)",
        "n_frames": combined.n_frames,
    }
    _save_artifact(
        output,
        combined,
        metadata=metadata,
        cell_index=np.concatenate(cell_rows),
        rollout_index=np.concatenate(rollout_rows),
    )
    return output


def _load_artifact(path: Path) -> tuple[RegimeFeatures, dict[str, Any]]:
    features, metadata, _cell_index, _rollout_index = _load_partitioned_artifact(path)
    return features, metadata


def _load_partitioned_artifact(
    path: Path,
) -> tuple[RegimeFeatures, dict[str, Any], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        if metadata.get("schema_version") != _ARTIFACT_SCHEMA:
            raise ValueError(
                f"{path}: artifact schema {metadata.get('schema_version')}, "
                f"expected {_ARTIFACT_SCHEMA}"
            )
        features = RegimeFeatures(
            **{name: np.asarray(data[name], dtype=np.float64) for name in _FEATURE_NAMES},
            k_hist=int(data["k_hist"]),
            steps_since_crossing=int(data["steps_since_crossing"]),
        )
        cell_index = np.asarray(data["cell_index"])
        rollout_index = np.asarray(data["rollout_index"])
    required = {
        "task",
        "family",
        "cells",
        "n_rollouts_per_cell",
        "start_index",
        "seed_envs",
        "source_commit",
        "n_frames",
    }
    if missing := sorted(required - set(metadata)):
        raise ValueError(f"{path}: artifact metadata is missing fields {missing}")
    n_cells = len(metadata["cells"])
    n_rollouts = int(metadata["n_rollouts_per_cell"])
    expected_frames = n_cells * n_rollouts * HORIZON
    if n_cells < 1 or n_rollouts < 1:
        raise ValueError(f"{path}: artifact cells and rollout count must be positive")
    if features.n_frames != int(metadata["n_frames"]):
        raise ValueError(
            f"{path}: metadata says {metadata['n_frames']} frames, artifact has "
            f"{features.n_frames}"
        )
    if features.n_frames != expected_frames:
        raise ValueError(
            f"{path}: {n_cells} cells x {n_rollouts} rollouts require "
            f"{expected_frames} frames, artifact has {features.n_frames}"
        )
    if len(metadata["seed_envs"]) != n_cells * n_rollouts:
        raise ValueError(
            f"{path}: seed_envs must contain one seed per cell rollout"
        )
    for name, index, upper in (
        ("cell_index", cell_index, n_cells),
        ("rollout_index", rollout_index, n_rollouts),
    ):
        if index.shape != (features.n_frames,) or index.dtype.kind not in "iu":
            raise ValueError(
                f"{path}: {name} must be an integer vector of length {features.n_frames}"
            )
        if np.any(index < 0) or np.any(index >= upper):
            raise ValueError(f"{path}: {name} contains an out-of-range index")
    counts = np.bincount(
        cell_index.astype(np.int64) * n_rollouts + rollout_index.astype(np.int64),
        minlength=n_cells * n_rollouts,
    )
    if not np.all(counts == HORIZON):
        raise ValueError(
            f"{path}: every (cell, rollout) must contain exactly {HORIZON} frames, "
            f"got counts {counts.tolist()}"
        )
    return features, metadata, cell_index, rollout_index


def load_feature_artifacts(
    paths: Sequence[Path],
) -> tuple[RegimeFeatures, list[dict[str, Any]]]:
    loaded = [_load_artifact(Path(path)) for path in paths]
    if not loaded:
        raise ValueError("at least one --input artifact is required")
    metadata = [item[1] for item in loaded]
    task_family = {(item["task"], item["family"]) for item in metadata}
    if len(task_family) != 1:
        raise ValueError(
            f"regime thresholds are scoped per (task, family), got {sorted(task_family)}"
        )
    return _concatenate([item[0] for item in loaded]), metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_freeze_protocol(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        protocol = json.load(fh)
    if protocol.get("schema_version") != _PROTOCOL_SCHEMA:
        raise ValueError(
            f"{path}: protocol schema {protocol.get('schema_version')}, "
            f"expected {_PROTOCOL_SCHEMA}"
        )
    required = {
        "task",
        "family",
        "cells",
        "start_index",
        "n_rollouts_per_cell",
    }
    if missing := sorted(required - set(protocol)):
        raise ValueError(f"{path}: freeze protocol is missing fields {missing}")
    cells = [
        {"defect": str(item["defect"]), "level": int(item["level"])}
        for item in protocol["cells"]
    ]
    if not cells or len({(item["defect"], item["level"]) for item in cells}) != len(cells):
        raise ValueError(f"{path}: protocol cells must be non-empty and unique")
    start_index = int(protocol["start_index"])
    n_rollouts = int(protocol["n_rollouts_per_cell"])
    if start_index < 0 or n_rollouts < 1:
        raise ValueError(
            f"{path}: start_index must be non-negative and n_rollouts_per_cell positive"
        )
    return {
        **protocol,
        "task": str(protocol["task"]),
        "family": str(protocol["family"]),
        "cells": cells,
        "start_index": start_index,
        "n_rollouts_per_cell": n_rollouts,
    }


def _validate_freeze_protocol(
    protocol: dict[str, Any], metadata: Sequence[dict[str, Any]]
) -> None:
    expected_scope = (protocol["task"], protocol["family"])
    expected_cells = protocol["cells"]
    ranges: list[tuple[int, int]] = []
    commits: set[str] = set()
    for item in metadata:
        scope = (item.get("task"), item.get("family"))
        if scope != expected_scope:
            raise ValueError(
                f"freeze protocol is scoped to {expected_scope}, artifact is scoped to {scope}"
            )
        if item.get("cells") != expected_cells:
            raise ValueError(
                "every formal artifact must contain the protocol's complete ordered cell set"
            )
        start = int(item.get("start_index", -1))
        count = int(item.get("n_rollouts_per_cell", 0))
        if start < 0 or count < 1:
            raise ValueError("artifact has invalid start_index or n_rollouts_per_cell")
        expected_frames = len(expected_cells) * count * HORIZON
        if int(item.get("n_frames", -1)) != expected_frames:
            raise ValueError(
                f"artifact declares {item.get('n_frames')} frames; its cells and rollout "
                f"count require {expected_frames}"
            )
        ranges.append((start, start + count))
        commits.add(str(item.get("source_commit")))

    ranges.sort()
    expected_start = protocol["start_index"]
    expected_stop = expected_start + protocol["n_rollouts_per_cell"]
    cursor = expected_start
    for start, stop in ranges:
        if start != cursor:
            kind = "overlap" if start < cursor else "gap"
            raise ValueError(
                f"formal artifact rollout ranges have a {kind}: expected next index "
                f"{cursor}, got [{start}, {stop})"
            )
        cursor = stop
    if cursor != expected_stop:
        raise ValueError(
            f"formal artifacts cover rollout indices [{expected_start}, {cursor}), "
            f"protocol requires [{expected_start}, {expected_stop})"
        )
    if len(commits) != 1:
        raise ValueError(
            "formal artifacts were produced by different source commits and cannot be pooled"
        )


def calibrate_artifacts(
    inputs: Sequence[Path],
    *,
    report_path: Path,
    freeze: bool = False,
    protocol_path: Path | None = None,
    lock: LockFile | None = None,
) -> dict[str, Any]:
    features, metadata = load_feature_artifacts(inputs)
    task, family = metadata[0]["task"], metadata[0]["family"]
    calibration = calibrate_regime_thresholds(features, task=task, family=family)
    artifacts = [
        {"path": str(Path(path)), "sha256": _sha256(Path(path))} for path in inputs
    ]
    report = {
        "task": task,
        "family": family,
        "n_frames": features.n_frames,
        "thresholds": calibration.thresholds.as_dict(),
        "occupancy": {regime.value: value for regime, value in calibration.occupancy.items()},
        "merged_into_r0": [regime.value for regime in calibration.merged_into_r0],
        "under_occupied": [regime.value for regime in calibration.under_occupied],
        "rounds": calibration.rounds,
        "converged": calibration.converged,
        "all_active_regimes_in_band": calibration.all_in_band,
        "artifacts": artifacts,
        "artifact_metadata": metadata,
        "protocol": None,
        "frozen": False,
    }
    if freeze:
        if protocol_path is None:
            raise ValueError(
                "refusing to freeze regime thresholds without an explicit sampling "
                "protocol; pass --protocol after P0 has frozen the cell set"
            )
        protocol = _load_freeze_protocol(protocol_path)
        _validate_freeze_protocol(protocol, metadata)
        if not calibration.converged or not calibration.all_in_band:
            raise ValueError(
                "refusing to freeze regime thresholds: coordinate descent must "
                "converge and every active regime must lie in the admissible band"
            )
        inadmissible = [
            item
            for item in metadata
            if item.get("source_commit") in (None, NO_CHECKOUT)
            or item.get("worktree_dirty") is not False
        ]
        if inadmissible:
            raise ValueError(
                "refusing to freeze regime thresholds: every input artifact must "
                "record a clean Git commit. Pilot artifacts from a dirty or "
                "non-Git worktree remain valid for pipeline testing only"
            )
        if _worktree_dirty() is not False:
            raise ValueError(
                "refusing to freeze regime thresholds from a dirty or non-Git "
                "calibration worktree"
            )
        destination = lock or LockFile()
        notes = "inputs=" + ",".join(
            f"{item['path']}@sha256:{item['sha256']}" for item in artifacts
        )
        calibration.freeze(destination, source_commit=source_commit(), notes=notes)
        destination.save()
        report["protocol"] = {
            "path": str(protocol_path),
            "sha256": _sha256(protocol_path),
            "contents": protocol,
        }
        report["frozen"] = True
        report["lock_path"] = str(destination.path)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _feature_subset(features: RegimeFeatures, mask: np.ndarray) -> RegimeFeatures:
    """Select frames without changing the regime feature definition."""
    return RegimeFeatures(
        **{
            name: np.asarray(getattr(features, name), dtype=np.float64)[mask]
            for name in _FEATURE_NAMES
        },
        k_hist=features.k_hist,
        steps_since_crossing=features.steps_since_crossing,
    )


def _occupancy_payload(
    features: RegimeFeatures,
    *,
    thresholds: Any,
    active_regimes: Sequence[Regime],
) -> dict[str, Any]:
    labels = label_frames(
        features,
        thresholds,
        active_regimes=active_regimes,
        for_calibration=True,
    )
    occupancy = occupancy_mass(labels)
    low, high = OCCUPANCY_BAND
    in_band = {
        regime.value: low <= occupancy[regime] <= high
        for regime in active_regimes
    }
    return {
        "n_frames": features.n_frames,
        "occupancy": {regime.value: occupancy[regime] for regime in Regime},
        "in_band": in_band,
        "all_active_regimes_in_band": all(in_band.values()),
    }


def cross_validate_artifacts(
    calibration_inputs: Sequence[Path],
    validation_inputs: Sequence[Path],
    *,
    report_path: Path,
) -> dict[str, Any]:
    """Fit once, then measure fixed-threshold occupancy on held-out rollouts.

    Calibration occupancy is an optimisation target, so it is not evidence of
    generalisation. This report keeps the held-out frames out of the coordinate
    descent and exposes every cell/rollout result instead of hiding failures in
    a pooled average.
    """
    calibration_features, calibration_metadata = load_feature_artifacts(
        calibration_inputs
    )
    calibration = calibrate_regime_thresholds(
        calibration_features,
        task=calibration_metadata[0]["task"],
        family=calibration_metadata[0]["family"],
    )
    if not calibration.converged:
        raise ValueError("calibration split did not converge; held-out validation is undefined")

    expected_scope = (calibration.task, calibration.family)
    validation_parts = [
        _load_partitioned_artifact(Path(path)) for path in validation_inputs
    ]
    if not validation_parts:
        raise ValueError("at least one --validation-input artifact is required")
    for _features, metadata, _cell_index, _rollout_index in validation_parts:
        scope = (metadata["task"], metadata["family"])
        if scope != expected_scope:
            raise ValueError(
                f"held-out artifact is scoped to {scope}, calibration is {expected_scope}"
            )
        if (
            _features.k_hist != calibration_features.k_hist
            or _features.steps_since_crossing
            != calibration_features.steps_since_crossing
        ):
            raise ValueError(
                "held-out artifact changes K_hist or steps_since_crossing; fixed "
                "threshold validation requires the same regime definition"
            )

    all_metadata = [*calibration_metadata, *(part[1] for part in validation_parts)]
    commits = {str(metadata.get("source_commit")) for metadata in all_metadata}
    if len(commits) != 1 or NO_CHECKOUT in commits:
        raise ValueError(
            "calibration and held-out artifacts must share one recorded source commit"
        )
    dirty = [metadata for metadata in all_metadata if metadata.get("worktree_dirty") is not False]
    if dirty:
        raise ValueError(
            "cross-validation requires artifacts collected from clean worktrees"
        )

    active = calibration.active_regimes
    pooled = _concatenate([part[0] for part in validation_parts])
    units: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    calibration_seeds = {
        int(seed)
        for metadata in calibration_metadata
        for seed in metadata["seed_envs"]
    }
    for path, (features, metadata, cell_index, rollout_index) in zip(
        validation_inputs, validation_parts, strict=True
    ):
        n_rollouts = int(metadata["n_rollouts_per_cell"])
        start_index = int(metadata["start_index"])
        seed_envs = [int(seed) for seed in metadata["seed_envs"]]
        for cell_id, cell in enumerate(metadata["cells"]):
            for rollout_offset in range(n_rollouts):
                mask = (cell_index == cell_id) & (rollout_index == rollout_offset)
                seed_position = cell_id * n_rollouts + rollout_offset
                seed_env = seed_envs[seed_position]
                if seed_env in calibration_seeds:
                    raise ValueError(
                        f"held-out seed {seed_env} also occurs in calibration inputs"
                    )
                if seed_env in seen_seeds and len(metadata["cells"]) == 1:
                    raise ValueError(f"held-out rollout seed {seed_env} is duplicated")
                seen_seeds.add(seed_env)
                units.append(
                    {
                        "artifact": str(Path(path)),
                        "artifact_sha256": _sha256(Path(path)),
                        "cell": {
                            "defect": str(cell["defect"]),
                            "level": int(cell["level"]),
                        },
                        "rollout_index": start_index + rollout_offset,
                        "seed_env": seed_env,
                        **_occupancy_payload(
                            _feature_subset(features, mask),
                            thresholds=calibration.thresholds,
                            active_regimes=active,
                        ),
                    }
                )

    report = {
        "schema_version": 1,
        "producer": "portal_trace.scripts.calibrate_regimes cross-validate",
        "status": "engineering_calibration_validation_not_frozen",
        "task": calibration.task,
        "family": calibration.family,
        "source_commit": next(iter(commits)),
        "thresholds": calibration.thresholds.as_dict(),
        "active_regimes": [regime.value for regime in active],
        "merged_into_r0": [regime.value for regime in calibration.merged_into_r0],
        "occupancy_band": list(OCCUPANCY_BAND),
        "calibration": {
            "inputs": [
                {"path": str(Path(path)), "sha256": _sha256(Path(path))}
                for path in calibration_inputs
            ],
            "artifact_metadata": calibration_metadata,
            "n_frames": calibration.n_frames,
            "occupancy": {
                regime.value: calibration.occupancy[regime] for regime in Regime
            },
            "converged": calibration.converged,
            "rounds": calibration.rounds,
        },
        "heldout": {
            "inputs": [
                {"path": str(Path(path)), "sha256": _sha256(Path(path))}
                for path in validation_inputs
            ],
            "artifact_metadata": [part[1] for part in validation_parts],
            "pooled": _occupancy_payload(
                pooled, thresholds=calibration.thresholds, active_regimes=active
            ),
            "units": units,
            "all_units_in_band": all(
                unit["all_active_regimes_in_band"] for unit in units
            ),
        },
        "frozen": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        raise FileExistsError(
            f"refusing to overwrite calibration validation report {report_path}"
        )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="run real uncompensated tracker rollouts")
    collect.add_argument("--task", required=True)
    collect.add_argument("--family", required=True)
    collect.add_argument("--cell", action="append", type=_parse_cell, required=True)
    collect.add_argument("--n-rollouts", type=int, required=True)
    collect.add_argument("--start-index", type=int, default=0)
    collect.add_argument("--output", type=Path, required=True)

    calibrate = sub.add_parser("calibrate", help="fit thresholds from feature artifacts")
    calibrate.add_argument("--input", type=Path, action="append", required=True)
    calibrate.add_argument("--report", type=Path, required=True)
    calibrate.add_argument("--freeze", action="store_true")
    calibrate.add_argument(
        "--protocol",
        type=Path,
        default=None,
        help="sampling protocol required by --freeze; fixed after P0 selects cells",
    )
    calibrate.add_argument("--lock", type=Path, default=None)

    cross_validate = sub.add_parser(
        "cross-validate",
        help="fit thresholds on calibration artifacts and validate fixed values held out",
    )
    cross_validate.add_argument(
        "--calibration-input", type=Path, action="append", required=True
    )
    cross_validate.add_argument(
        "--validation-input", type=Path, action="append", required=True
    )
    cross_validate.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "collect":
        path = collect_feature_artifact(
            task=args.task,
            family=args.family,
            cells=args.cell,
            n_rollouts=args.n_rollouts,
            start_index=args.start_index,
            output=args.output,
        )
        print(path)
        return 0

    if args.command == "cross-validate":
        report = cross_validate_artifacts(
            args.calibration_input,
            args.validation_input,
            report_path=args.report,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    lock = LockFile(args.lock) if args.lock is not None else None
    report = calibrate_artifacts(
        args.input,
        report_path=args.report,
        freeze=args.freeze,
        protocol_path=args.protocol,
        lock=lock,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
