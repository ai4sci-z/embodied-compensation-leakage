"""Offline deterministic evaluation probe for the smoke checkpoint only.

This module is deliberately separate from :mod:`runners.evaluate`.  It consumes
materialised policy windows and checks that the tiny smoke checkpoint can be
loaded, normalised and run deterministically.  Its JSON output is an engineering
artifact, not a rollout result: it has no success, return, or paper endpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from portal_trace.policies.architecture import load_smoke_checkpoint
from portal_trace.policies.losses import chunk_bc_loss
from portal_trace.runners.policy_dataset import (
    PolicyDatasetEntry,
    RawPolicyWindowDataset,
    collate_policy_samples,
)

__all__ = [
    "SMOKE_EVAL_MARKER",
    "SMOKE_EVAL_SCHEMA",
    "SmokeEvaluationConfig",
    "SmokeEvaluationResult",
    "run_smoke_evaluation",
    "validate_smoke_evaluation_artifact",
]


SMOKE_EVAL_SCHEMA = "portal_trace.smoke_evaluation/1"
SMOKE_EVAL_MARKER = "PORTAL_TRACE_SMOKE_EVAL_NOT_FORMAL"


@dataclass(frozen=True)
class SmokeEvaluationConfig:
    """Explicit non-training windows to use for the offline probe."""

    checkpoint_path: Path
    entries: tuple[PolicyDatasetEntry, ...]
    eval_indices: tuple[int, ...]
    artifact_path: Path
    batch_size: int = 1

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("smoke evaluation requires explicit materialised entries")
        if not self.eval_indices:
            raise ValueError("smoke evaluation requires an explicit non-empty eval split")
        if any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in self.eval_indices
        ):
            raise TypeError("smoke evaluation indices must be integers")
        if any(index < 0 for index in self.eval_indices):
            raise ValueError("smoke evaluation indices must be non-negative")
        if len(set(self.eval_indices)) != len(self.eval_indices):
            raise ValueError("smoke evaluation indices must be unique")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise ValueError("smoke evaluation batch_size must be a positive integer")
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        object.__setattr__(self, "artifact_path", Path(self.artifact_path))


@dataclass(frozen=True)
class SmokeEvaluationResult:
    """The saved probe artifact and its non-semantic output summary."""

    artifact_path: Path
    n_windows: int
    action_shape: tuple[int, int, int]
    output_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _entry_payload(entry: PolicyDatasetEntry) -> dict[str, Any]:
    path = Path(entry.path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"smoke evaluation dataset does not exist: {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "task_index": entry.task_index,
        "embodiment_index": entry.embodiment_index,
        "n_demos": entry.n_demos,
    }


def _checkpoint_bindings(path: Path) -> tuple[dict[str, Any], ...]:
    """Read metadata after the architecture loader has validated the checkpoint."""
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"{path}: unable to read smoke checkpoint metadata") from exc
    entries = raw.get("dataset_entries") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("smoke checkpoint has no dataset entry bindings")
    bindings: list[dict[str, Any]] = []
    for index, item in enumerate(entries):
        if not isinstance(item, dict) or set(item) != {
            "path",
            "sha256",
            "task_index",
            "embodiment_index",
            "n_demos",
        }:
            raise ValueError(f"smoke checkpoint dataset_entries[{index}] is malformed")
        bound_path = Path(item["path"]).resolve()
        if not bound_path.is_file():
            raise FileNotFoundError(f"smoke checkpoint dataset is missing: {bound_path}")
        if _sha256(bound_path) != item["sha256"]:
            raise ValueError(f"smoke checkpoint dataset SHA-256 no longer matches: {bound_path}")
        bindings.append(
            {
                "path": str(bound_path),
                "sha256": item["sha256"],
                "task_index": item["task_index"],
                "embodiment_index": item["embodiment_index"],
                "n_demos": item["n_demos"],
            }
        )
    return tuple(bindings)


def _immutable_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite smoke evaluation artifact {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite smoke evaluation artifact {path}"
            ) from exc
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_body(payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body.pop("artifact_sha256", None)
    return body


def validate_smoke_evaluation_artifact(
    path: Path | str,
) -> dict[str, Any]:
    """Validate the marker, finite probe fields and tamper-evident artifact hash."""
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: unreadable smoke evaluation artifact") from exc
    if not isinstance(document, dict):
        raise ValueError("smoke evaluation artifact must be a JSON object")
    expected = {
        "schema",
        "marker",
        "checkpoint",
        "dataset_entries",
        "split",
        "probe",
        "artifact_sha256",
    }
    if set(document) != expected:
        raise ValueError(
            "smoke evaluation artifact fields mismatch; "
            f"missing={sorted(expected - set(document))}, "
            f"extra={sorted(set(document) - expected)}"
        )
    if document["schema"] != SMOKE_EVAL_SCHEMA:
        raise ValueError("artifact is not the smoke evaluation schema")
    if document["marker"] != SMOKE_EVAL_MARKER:
        raise ValueError("artifact lacks the required non-formal smoke marker")
    digest = document["artifact_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or _canonical_sha256(_artifact_body(document)) != digest
    ):
        raise ValueError("smoke evaluation artifact SHA-256 does not match")
    probe = document["probe"]
    if not isinstance(probe, dict) or set(probe) != {
        "n_windows",
        "action_shape",
        "finite",
        "action_mean",
        "action_std",
        "masked_bc_mse",
        "output_sha256",
    }:
        raise ValueError("smoke evaluation probe is incomplete")
    if probe["finite"] is not True or probe["n_windows"] < 1:
        raise ValueError("smoke evaluation probe is not a finite non-empty output")
    if not all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in (
            probe["action_mean"],
            probe["action_std"],
            probe["masked_bc_mse"],
        )
    ):
        raise ValueError("smoke evaluation probe has non-finite summary values")
    return document


def run_smoke_evaluation(config: SmokeEvaluationConfig) -> SmokeEvaluationResult:
    """Run deterministic forward passes on an explicit non-training split."""
    checkpoint = Path(config.checkpoint_path)
    loaded = load_smoke_checkpoint(checkpoint)
    bindings = _checkpoint_bindings(checkpoint)
    supplied = tuple(_entry_payload(entry) for entry in config.entries)
    if supplied != bindings:
        raise ValueError(
            "smoke evaluation entries do not exactly match the materialised files bound "
            "by the checkpoint"
        )
    overlap = set(config.eval_indices) & set(loaded.split.training_indices)
    if overlap:
        raise ValueError(
            f"smoke evaluation split overlaps training indices {sorted(overlap)}; "
            "evaluation must be non-training"
        )
    dataset = RawPolicyWindowDataset(config.entries)
    invalid = [index for index in config.eval_indices if index >= len(dataset)]
    if invalid:
        raise ValueError(
            f"smoke evaluation indices {invalid} exceed dataset length {len(dataset)}"
        )

    outputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    loaded.model.eval()
    for start in range(0, len(config.eval_indices), config.batch_size):
        selected = config.eval_indices[start : start + config.batch_size]
        batch = collate_policy_samples([dataset[index] for index in selected])
        outputs.append(loaded.inference(batch))
        targets.append(batch.actions)
        masks.append(batch.action_pad_mask)
    actions = torch.cat(outputs, dim=0)
    target_actions = torch.cat(targets, dim=0)
    target_masks = torch.cat(masks, dim=0)
    if actions.ndim != 3 or not torch.isfinite(actions).all():
        raise RuntimeError("smoke evaluation forward produced an invalid action tensor")
    repeat: list[torch.Tensor] = []
    for start in range(0, len(config.eval_indices), config.batch_size):
        selected = config.eval_indices[start : start + config.batch_size]
        repeat.append(
            loaded.inference(
                collate_policy_samples([dataset[index] for index in selected])
            )
        )
    repeated_actions = torch.cat(repeat, dim=0)
    if not torch.equal(actions, repeated_actions):
        raise RuntimeError("smoke evaluation forward is not deterministic")

    output_bytes = actions.numpy().tobytes()
    output_sha = hashlib.sha256(output_bytes).hexdigest()
    body: dict[str, Any] = {
        "schema": SMOKE_EVAL_SCHEMA,
        "marker": SMOKE_EVAL_MARKER,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": _sha256(checkpoint),
        },
        "dataset_entries": list(supplied),
        "split": {
            "eval_indices": list(config.eval_indices),
            "training_overlap": False,
        },
        "probe": {
            "n_windows": int(actions.shape[0]),
            "action_shape": list(actions.shape),
            "finite": True,
            "action_mean": float(actions.mean()),
            "action_std": float(actions.std(unbiased=False)),
            "masked_bc_mse": float(
                chunk_bc_loss(actions, target_actions, pad_mask=target_masks)
            ),
            "output_sha256": output_sha,
        },
    }
    body["artifact_sha256"] = _canonical_sha256(body)
    _immutable_json(config.artifact_path, body)
    validate_smoke_evaluation_artifact(config.artifact_path)
    return SmokeEvaluationResult(
        artifact_path=config.artifact_path,
        n_windows=int(actions.shape[0]),
        action_shape=tuple(int(value) for value in actions.shape),
        output_sha256=output_sha,
    )
