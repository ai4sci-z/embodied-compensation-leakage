"""Build a hash-bound pilot data plan and run the bounded adapter probe.

This is an engineering bridge for real materialised HDF5 files.  It exercises
the formal data-plan, train-only normalization, registered policy, optimizer
update, held-out inference, and checkpoint reload contracts without entering
the formal training stage or publishing a paper result.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

from portal_trace.policies.interface import PolicyConfig
from portal_trace.runners.dataset_contract import require_dataset_identity
from portal_trace.runners.formal_adapter import (
    FORMAL_ADAPTER_PROBE_MARKER,
    FormalAdapterConfig,
    run_formal_adapter_probe,
)
from portal_trace.runners.formal_data import (
    FormalPoolEntry,
    build_formal_data_plan,
    fit_formal_normalization,
)

PILOT_SCHEMA = "portal_trace.formal_adapter_pilot/1"
PILOT_MARKER = "PORTAL_TRACE_FORMAL_ADAPTER_PILOT_NOT_FORMAL"

__all__ = [
    "FormalAdapterPilotResult",
    "FormalDatasetDescriptor",
    "describe_formal_dataset",
    "formal_pool_entries_from_datasets",
    "run_formal_adapter_pilot",
]


@dataclass(frozen=True)
class FormalAdapterPilotResult:
    plan_path: Path
    normalization_path: Path
    checkpoint_path: Path
    summary_path: Path
    plan_sha256: str
    final_loss: float
    parameter_delta_l2: float
    train_windows: int
    heldout_windows: int


@dataclass(frozen=True)
class FormalDatasetDescriptor:
    path: Path
    task: str
    family: str
    defect: str
    level: int
    clean_of: str
    n_demos: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_json(path: Path, payload: dict[str, object]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable pilot summary {path}")
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


def describe_formal_dataset(path: Path) -> FormalDatasetDescriptor:
    import h5py

    path = Path(path).resolve()
    with h5py.File(path, "r") as dataset:
        if "data" not in dataset or not dataset["data"]:
            raise ValueError(f"{path}: materialised dataset has no demos")
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
        n_demos = len(dataset["data"])
    require_dataset_identity(
        path,
        task=task,
        family=family,
        defect=defect,
        level=level,
        clean_of=clean_of,
        min_demos=1,
    )
    return FormalDatasetDescriptor(
        path=path,
        task=task,
        family=family,
        defect=defect,
        level=level,
        clean_of=clean_of,
        n_demos=n_demos,
    )


def formal_pool_entries_from_datasets(
    datasets: tuple[Path, ...],
) -> tuple[FormalPoolEntry, ...]:
    """Infer stable task and embodiment indices from materialised datasets."""
    if not datasets:
        raise ValueError("formal data planning requires at least one dataset")
    descriptors = tuple(describe_formal_dataset(path) for path in datasets)
    task_indices = {
        task: index for index, task in enumerate(sorted({item.task for item in descriptors}))
    }
    embodiment_identities = sorted(
        {(item.family, item.defect, item.level, item.clean_of) for item in descriptors}
    )
    embodiment_indices = {
        identity: index for index, identity in enumerate(embodiment_identities)
    }
    return tuple(
        FormalPoolEntry(
            path=item.path,
            task_index=task_indices[item.task],
            embodiment_index=embodiment_indices[
                (item.family, item.defect, item.level, item.clean_of)
            ],
            n_demos=item.n_demos,
            task=item.task,
            family=item.family,
            defect=item.defect,
            level=item.level,
            clean_of=item.clean_of,
        )
        for item in descriptors
    )


def run_formal_adapter_pilot(
    *,
    datasets: tuple[Path, ...],
    plan_path: Path,
    normalization_path: Path,
    checkpoint_path: Path,
    summary_path: Path,
    source_commit: str,
    environment_fingerprint: str,
    seed: int = 20260916,
    train_fraction: float = 0.8,
    optimizer_steps: int = 2,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    d_model: int = 32,
    n_layers: int = 1,
    n_heads: int = 2,
    ffn_dim: int = 64,
) -> FormalAdapterPilotResult:
    """Run a bounded adapter probe over real materialised pilot datasets."""
    if len(datasets) < 2:
        raise ValueError("formal adapter pilot needs at least two datasets for a held-out split")
    if (
        len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit.lower())
    ):
        raise ValueError("source_commit must be a complete 40-character Git SHA-1")
    entries = formal_pool_entries_from_datasets(datasets)
    task_indices = {entry.task: entry.task_index for entry in entries}
    embodiment_indices = {
        (entry.family, entry.defect, entry.level, entry.clean_of): entry.embodiment_index
        for entry in entries
    }
    plan = build_formal_data_plan(
        entries,
        seed=seed,
        train_fraction=train_fraction,
        source_commit=source_commit,
        environment_fingerprint=environment_fingerprint,
    )
    plan.write(Path(plan_path))
    normalization = fit_formal_normalization(plan)
    normalization.write(Path(normalization_path))
    config = PolicyConfig(
        arm="portal-10.4.0",
        n_tasks=len(task_indices),
        n_pool_embodiments=len(embodiment_indices),
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        augment=False,
    )
    result = run_formal_adapter_probe(
        FormalAdapterConfig(
            plan=plan,
            normalization=normalization,
            policy_config=config,
            seed=seed,
            optimizer_steps=optimizer_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            checkpoint_path=Path(checkpoint_path),
        )
    )
    payload: dict[str, object] = {
        "schema": PILOT_SCHEMA,
        "marker": PILOT_MARKER,
        "adapter_marker": FORMAL_ADAPTER_PROBE_MARKER,
        "datasets": [str(path.resolve()) for path in datasets],
        "dataset_sha256": [_sha256(path) for path in datasets],
        "plan": str(Path(plan_path).resolve()),
        "plan_sha256": plan.plan_sha256,
        "normalization": str(Path(normalization_path).resolve()),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "optimizer_steps": result.optimizer_steps,
        "final_loss": result.final_loss,
        "parameter_delta_l2": result.parameter_delta_l2,
        "train_windows": result.train_windows,
        "heldout_windows": result.heldout_windows,
        "heldout_action_mse": result.heldout_action_mse,
        "heldout_decompensated_action_mse": result.heldout_decompensated_action_mse,
        "heldout_c_true_rms": result.heldout_c_true_rms,
        "output_shapes": {
            "train_reference": list(result.reference_output.shape),
            "heldout": list(result.heldout_output.shape),
        },
    }
    _publish_json(Path(summary_path), payload)
    return FormalAdapterPilotResult(
        plan_path=Path(plan_path),
        normalization_path=Path(normalization_path),
        checkpoint_path=Path(checkpoint_path),
        summary_path=Path(summary_path),
        plan_sha256=plan.plan_sha256,
        final_loss=result.final_loss,
        parameter_delta_l2=result.parameter_delta_l2,
        train_windows=result.train_windows,
        heldout_windows=result.heldout_windows,
    )
