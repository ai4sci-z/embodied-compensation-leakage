"""Train and evaluate the bounded policy on real pilot observations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

from portal_trace.policies.architecture import (
    SMOKE_CHECKPOINT_SCHEMA,
    SMOKE_ONLY_MARKER,
    SmokeSplit,
    SmokeTrainingConfig,
    run_smoke_training,
)
from portal_trace.policies.obs import CHUNK, K_HIST_MAIN
from portal_trace.runners.dataset_contract import require_dataset_identity
from portal_trace.runners.policy_dataset import PolicyDatasetEntry
from portal_trace.runners.smoke_evaluate import (
    SmokeEvaluationConfig,
    run_smoke_evaluation,
)

POLICY_PILOT_SCHEMA = "portal_trace.policy_pilot/2"
POLICY_PILOT_MARKER = "PORTAL_TRACE_POLICY_PILOT_NOT_FORMAL"


@dataclass(frozen=True)
class PolicyPilotResult:
    checkpoint_path: Path
    evaluation_path: Path
    summary_path: Path
    final_loss: float
    parameter_delta_l2: float
    training_windows: int
    evaluation_windows: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _entry(path: Path, *, embodiment_index: int, task_indices: dict[str, int]) -> tuple[PolicyDatasetEntry, int]:
    import h5py

    path = Path(path).resolve()
    with h5py.File(path, "r") as dataset:
        def text_attr(name: str) -> str:
            value = dataset.attrs.get(name)
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            if not isinstance(value, str) or not value:
                raise ValueError(f"{path}: missing or empty dataset identity attribute {name}")
            return value

        task = text_attr("dataset_task")
        family = text_attr("dataset_family")
        defect = text_attr("dataset_defect")
        raw_level = dataset.attrs.get("dataset_level")
        if isinstance(raw_level, bool) or not isinstance(raw_level, Integral):
            raise ValueError(f"{path}: dataset_level must be a non-negative integer")
        level = int(raw_level)
        if level < 0:
            raise ValueError(f"{path}: dataset_level must be a non-negative integer")
        clean_of = dataset.attrs.get("dataset_clean_of", "")
        if isinstance(clean_of, bytes):
            clean_of = clean_of.decode("utf-8")
        if not isinstance(clean_of, str):
            raise ValueError(f"{path}: dataset_clean_of must be text")
        demos = sorted(dataset["data"], key=lambda name: int(name.split("_")[1]))
        if len(demos) != 1:
            raise ValueError(f"policy pilot requires one demo per file, got {len(demos)} in {path}")
        horizon = int(dataset[f"data/{demos[0]}/actions"].shape[0])
    require_dataset_identity(
        path,
        task=task,
        family=family,
        defect=defect,
        level=level,
        clean_of=clean_of,
        min_demos=1,
    )
    if task not in task_indices:
        task_indices[task] = len(task_indices)
    return (
        PolicyDatasetEntry(
            path=path,
            task_index=task_indices[task],
            embodiment_index=embodiment_index,
            n_demos=1,
            family=family,
            defect=defect,
            level=level,
        ),
        horizon,
    )


def _publish_summary(path: Path, payload: dict[str, object]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite policy pilot summary {path}")
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


def run_policy_pilot(
    *,
    datasets: tuple[Path, ...],
    checkpoint_path: Path,
    evaluation_path: Path,
    summary_path: Path,
    train_windows_per_entry: int = 32,
    eval_windows_per_entry: int = 16,
    optimizer_steps: int = 4,
    batch_size: int = 8,
    seed: int = 20260916,
) -> PolicyPilotResult:
    """Run real optimizer updates and held-window inference on pilot HDF5 files."""
    if not datasets:
        raise ValueError("policy pilot requires at least one materialised dataset")
    if train_windows_per_entry < 1 or eval_windows_per_entry < 1:
        raise ValueError("policy pilot window counts must be positive")

    task_indices: dict[str, int] = {}
    entries: list[PolicyDatasetEntry] = []
    horizons: list[int] = []
    for embodiment_index, path in enumerate(datasets):
        entry, horizon = _entry(
            Path(path), embodiment_index=embodiment_index, task_indices=task_indices
        )
        entries.append(entry)
        horizons.append(horizon)

    training_indices: list[int] = []
    evaluation_indices: list[int] = []
    offset = 0
    for path, horizon in zip(datasets, horizons, strict=True):
        eval_start = horizon - eval_windows_per_entry
        latest_training_target = train_windows_per_entry - 1 + CHUNK - 1
        earliest_eval_history = eval_start - K_HIST_MAIN + 1
        if latest_training_target >= earliest_eval_history:
            raise ValueError(
                f"{path}: train/eval windows overlap after accounting for "
                f"K_hist={K_HIST_MAIN} and chunk={CHUNK}"
            )
        training_indices.extend(offset + index for index in range(train_windows_per_entry))
        evaluation_indices.extend(offset + index for index in range(eval_start, horizon))
        offset += horizon

    entry_tuple = tuple(entries)
    training = run_smoke_training(
        SmokeTrainingConfig(
            schema=SMOKE_CHECKPOINT_SCHEMA,
            marker=SMOKE_ONLY_MARKER,
            entries=entry_tuple,
            split=SmokeSplit(tuple(training_indices)),
            seed=seed,
            checkpoint_path=Path(checkpoint_path),
            optimizer_steps=optimizer_steps,
            batch_size=batch_size,
        )
    )
    evaluation = run_smoke_evaluation(
        SmokeEvaluationConfig(
            checkpoint_path=Path(checkpoint_path),
            entries=entry_tuple,
            eval_indices=tuple(evaluation_indices),
            artifact_path=Path(evaluation_path),
            batch_size=batch_size,
        )
    )
    payload: dict[str, object] = {
        "schema": POLICY_PILOT_SCHEMA,
        "marker": POLICY_PILOT_MARKER,
        "datasets": [str(entry.path) for entry in entries],
        "dataset_sha256": [_sha256(entry.path) for entry in entries],
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": _sha256(Path(checkpoint_path)),
        "evaluation": str(Path(evaluation_path).resolve()),
        "evaluation_sha256": _sha256(Path(evaluation_path)),
        "training_windows": len(training_indices),
        "evaluation_windows": len(evaluation_indices),
        "optimizer_steps": training.optimizer_steps,
        "final_loss": training.final_loss,
        "parameter_delta_l2": training.parameter_delta_l2,
        "evaluation_output_sha256": evaluation.output_sha256,
    }
    _publish_summary(Path(summary_path), payload)
    return PolicyPilotResult(
        checkpoint_path=Path(checkpoint_path),
        evaluation_path=Path(evaluation_path),
        summary_path=Path(summary_path),
        final_loss=training.final_loss,
        parameter_delta_l2=training.parameter_delta_l2,
        training_windows=len(training_indices),
        evaluation_windows=len(evaluation_indices),
    )
