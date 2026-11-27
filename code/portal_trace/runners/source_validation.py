"""Pre-collection seed reservation for prospective source validation.

The reservation is created before any validation trajectory exists. A later
formal source-discriminator run must consume exactly these environment seeds,
so failed or inconvenient pairs cannot be silently replaced after inspection.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from portal_trace.runners.dataset_contract import PAIRED_TWIN_SCHEMA_VERSION
from portal_trace.runners.formal_data import FormalDataPlan

__all__ = [
    "SOURCE_VALIDATION_RESERVATION_SCHEMA",
    "SOURCE_VALIDATION_RESERVATION_MARKER",
    "SourceValidationReservation",
    "source_validation_collection_spec",
    "source_validation_run_spec",
    "create_source_validation_reservation",
    "load_source_validation_reservation",
]

SOURCE_VALIDATION_RESERVATION_SCHEMA = "portal_trace.source_validation_reservation/3"
SOURCE_VALIDATION_RESERVATION_MARKER = "PORTAL_TRACE_PRECOLLECTION_SEED_RESERVATION"
_METHOD_SCHEMA = "portal_trace.source_discriminator/8"
_SELECTION_ALGORITHM = "sha256-reservation-seed-counter-mod-2^31-v1"
# FormalDataPlan keeps at least one pair on its unused train side; reserving
# seven is therefore the minimum that can yield six prospective held-out pairs.
_MIN_FORMAL_PAIRS = 7


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _immutable_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite validation reservation {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class SourceValidationReservation:
    development_plan_sha256: str
    method_schema: str
    reservation_seed: int
    reserved_seed_envs: tuple[int, ...]
    sacrificial_seed_env: int
    heldout_seed_envs: tuple[int, ...]
    run_spec: dict[str, Any]
    collection_spec: dict[str, Any]
    producer_git_state: dict[str, Any]
    release_status: str
    artifact_sha256: str

    def as_body(self) -> dict[str, Any]:
        return {
            "schema": SOURCE_VALIDATION_RESERVATION_SCHEMA,
            "marker": SOURCE_VALIDATION_RESERVATION_MARKER,
            "development_plan_sha256": self.development_plan_sha256,
            "method_schema": self.method_schema,
            "selection_algorithm": _SELECTION_ALGORITHM,
            "reservation_seed": self.reservation_seed,
            "reserved_seed_envs": list(self.reserved_seed_envs),
            "sacrificial_seed_env": self.sacrificial_seed_env,
            "heldout_seed_envs": list(self.heldout_seed_envs),
            "run_spec": self.run_spec,
            "collection_spec": self.collection_spec,
            "producer_git_state": self.producer_git_state,
            "release_status": self.release_status,
        }

    def write(self, path: Path | str) -> None:
        _immutable_json(
            Path(path), self.as_body() | {"artifact_sha256": self.artifact_sha256}
        )


def _validate_git_state(value: Any, *, formal: bool) -> dict[str, Any]:
    expected = {"head", "tracked_worktree_dirty", "tracked_diff_sha256"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("validation reservation Git state is incomplete")
    if (
        not isinstance(value["head"], str)
        or len(value["head"]) != 40
        or any(character not in "0123456789abcdef" for character in value["head"])
        or not isinstance(value["tracked_worktree_dirty"], bool)
    ):
        raise ValueError("validation reservation Git state is invalid")
    diff_sha = value["tracked_diff_sha256"]
    if value["tracked_worktree_dirty"]:
        if not isinstance(diff_sha, str) or len(diff_sha) != 64:
            raise ValueError("dirty validation reservation lacks its diff SHA-256")
    elif diff_sha is not None:
        raise ValueError("clean validation reservation unexpectedly carries a diff SHA-256")
    if formal and value["tracked_worktree_dirty"]:
        raise ValueError("formal validation seeds cannot be reserved from a dirty HEAD")
    return dict(value)


def _validate_reservation(reservation: SourceValidationReservation) -> None:
    if (
        len(reservation.development_plan_sha256) != 64
        or any(c not in "0123456789abcdef" for c in reservation.development_plan_sha256)
    ):
        raise ValueError("validation reservation development plan SHA-256 is invalid")
    if reservation.method_schema != _METHOD_SCHEMA:
        raise ValueError("validation reservation method schema is stale")
    if isinstance(reservation.reservation_seed, bool) or not isinstance(
        reservation.reservation_seed, int
    ):
        raise ValueError("validation reservation seed must be an integer")
    seeds = reservation.reserved_seed_envs
    if (
        not seeds
        or tuple(sorted(seeds)) != seeds
        or len(set(seeds)) != len(seeds)
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ValueError("validation reservation environment seeds are invalid")
    if (
        isinstance(reservation.sacrificial_seed_env, bool)
        or not isinstance(reservation.sacrificial_seed_env, int)
        or reservation.sacrificial_seed_env not in seeds
    ):
        raise ValueError("validation reservation sacrificial seed is invalid")
    if (
        tuple(sorted(reservation.heldout_seed_envs)) != reservation.heldout_seed_envs
        or len(set(reservation.heldout_seed_envs)) != len(reservation.heldout_seed_envs)
        or set(reservation.heldout_seed_envs)
        != set(seeds) - {reservation.sacrificial_seed_env}
    ):
        raise ValueError("validation reservation held-out seed roles are invalid")
    if reservation.release_status not in {"formal", "engineering_only"}:
        raise ValueError("validation reservation release status is invalid")
    if (
        reservation.release_status == "formal"
        and len(reservation.reserved_seed_envs) < _MIN_FORMAL_PAIRS
    ):
        raise ValueError(
            f"formal validation reservation needs at least {_MIN_FORMAL_PAIRS} pairs"
        )
    if reservation.run_spec != source_validation_run_spec(**reservation.run_spec):
        raise ValueError("validation reservation run spec is invalid")
    if reservation.collection_spec != source_validation_collection_spec(
        **reservation.collection_spec
    ):
        raise ValueError("validation reservation collection spec is invalid")
    for name in ("task", "family", "defect", "level"):
        if reservation.collection_spec[name] != reservation.run_spec[name]:
            raise ValueError(
                f"validation reservation collection and run specs disagree on {name}"
            )
    git_state = _validate_git_state(
        reservation.producer_git_state,
        formal=reservation.release_status == "formal",
    )
    if (
        reservation.release_status == "formal"
        and reservation.collection_spec["source_commit"] != git_state["head"]
    ):
        raise ValueError(
            "formal validation collection commit must equal the clean reservation HEAD"
        )
    if _canonical_sha256(reservation.as_body()) != reservation.artifact_sha256:
        raise ValueError("validation reservation content SHA-256 changed")


def _reserved_seeds(
    *, reservation_seed: int, count: int, excluded: set[int]
) -> tuple[int, ...]:
    selected: set[int] = set()
    counter = 0
    while len(selected) < count:
        digest = hashlib.sha256(
            f"{reservation_seed}\0{counter}".encode("ascii")
        ).digest()
        candidate = int.from_bytes(digest[:8], "big") % (2**31)
        if candidate not in excluded:
            selected.add(candidate)
        counter += 1
    return tuple(sorted(selected))


def _seed_roles(
    seeds: tuple[int, ...], *, reservation_seed: int
) -> tuple[int, tuple[int, ...]]:
    ranked = sorted(
        seeds,
        key=lambda seed: hashlib.sha256(
            f"{reservation_seed}\0{seed}".encode("ascii")
        ).hexdigest(),
    )
    return ranked[0], tuple(sorted(ranked[1:]))


def source_validation_collection_spec(
    *,
    source_commit: str,
    environment_fingerprint: str,
    task: str,
    family: str,
    defect: str,
    level: int,
    geom_variant: str,
    paired_binding_id: str | None = None,
) -> dict[str, Any]:
    """Canonical data-generation identity frozen before validation collection."""
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError("source-validation collection commit must be a lowercase Git SHA-1")
    try:
        runtime = json.loads(environment_fingerprint)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("source-validation environment fingerprint must be JSON") from exc
    if not isinstance(runtime, dict) or not runtime:
        raise ValueError("source-validation environment fingerprint must be a JSON object")
    canonical_runtime = json.dumps(
        runtime, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    for name, value in (
        ("task", task),
        ("family", family),
        ("defect", defect),
        ("geom_variant", geom_variant),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"source-validation collection {name} is invalid")
    if isinstance(level, bool) or not isinstance(level, int) or level < 0:
        raise ValueError("source-validation collection level is invalid")
    binding = {
        "schema_version": PAIRED_TWIN_SCHEMA_VERSION,
        "task": task,
        "family": family,
        "defect": defect,
        "level": level,
        "geom_variant": geom_variant,
    }
    computed_binding_id = _canonical_sha256(binding)
    if paired_binding_id is not None and paired_binding_id != computed_binding_id:
        raise ValueError("source-validation paired binding does not match its cell spec")
    return {
        "source_commit": source_commit,
        "environment_fingerprint": canonical_runtime,
        "task": task,
        "family": family,
        "defect": defect,
        "level": level,
        "geom_variant": geom_variant,
        "paired_binding_id": computed_binding_id,
    }


def source_validation_run_spec(
    *,
    normalization_sha256: str,
    source_embodiment_indices: list[int] | tuple[int, ...],
    target_embodiment_indices: list[int] | tuple[int, ...],
    task: str,
    family: str,
    defect: str,
    level: int,
    training_seed: int,
    epochs: int,
    optimizer_steps: int | None,
    batch_size: int,
    learning_rate: float,
    device: str,
    n_tasks: int,
    d_model: int,
    augment: bool,
    selection_interval: int,
    observation: dict[str, Any],
    fusion_mode: str = "modality-mean",
) -> dict[str, Any]:
    """Canonical configuration frozen before prospective data collection."""
    if (
        not isinstance(normalization_sha256, str)
        or len(normalization_sha256) != 64
        or any(c not in "0123456789abcdef" for c in normalization_sha256)
    ):
        raise ValueError("source-validation normalization SHA-256 is invalid")
    source = tuple(sorted(source_embodiment_indices))
    target = tuple(sorted(target_embodiment_indices))
    if (
        not source
        or not target
        or len(set(source)) != len(source)
        or len(set(target)) != len(target)
        or set(source) & set(target)
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in source + target)
    ):
        raise ValueError("source-validation embodiment partition is invalid")
    for name, value in (("task", task), ("family", family), ("defect", defect)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"source-validation {name} is invalid")
    integer_fields = {
        "level": level,
        "training_seed": training_seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "n_tasks": n_tasks,
        "d_model": d_model,
        "selection_interval": selection_interval,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for name, value in integer_fields.items()
        if name != "level"
    ) or isinstance(level, bool) or not isinstance(level, int) or level < 0:
        raise ValueError("source-validation integer run settings are invalid")
    if optimizer_steps is not None and (
        isinstance(optimizer_steps, bool)
        or not isinstance(optimizer_steps, int)
        or optimizer_steps < 1
    ):
        raise ValueError("source-validation optimizer_steps is invalid")
    if (
        not isinstance(learning_rate, (int, float))
        or isinstance(learning_rate, bool)
        or not 0.0 < float(learning_rate) < float("inf")
    ):
        raise ValueError("source-validation learning rate is invalid")
    if device not in {"cpu", "cuda"}:
        raise ValueError("source-validation device must be CPU or CUDA")
    if not isinstance(augment, bool):
        raise ValueError("source-validation augment flag must be boolean")
    if not isinstance(observation, dict) or not observation:
        raise ValueError("source-validation observation spec is invalid")
    if fusion_mode not in {"modality-mean", "camera-time", "legacy-mean"}:
        raise ValueError("source-validation fusion mode is invalid")
    # The JSON round trip rejects non-serializable values and normalizes tuples.
    normalized_observation = json.loads(
        json.dumps(observation, sort_keys=True, allow_nan=False)
    )
    run_spec = {
        "normalization_sha256": normalization_sha256,
        "source_embodiment_indices": list(source),
        "target_embodiment_indices": list(target),
        "task": task,
        "family": family,
        "defect": defect,
        "level": level,
        "training_seed": training_seed,
        "epochs": epochs,
        "optimizer_steps": optimizer_steps,
        "batch_size": batch_size,
        "learning_rate": float(learning_rate),
        "device": device,
        "n_tasks": n_tasks,
        "d_model": d_model,
        "augment": augment,
        "selection_interval": selection_interval,
        "observation": normalized_observation,
    }
    # Preserve existing reservations verbatim when the default is unchanged.
    if fusion_mode != "modality-mean":
        run_spec["fusion_mode"] = fusion_mode
    return run_spec


def create_source_validation_reservation(
    development_plan: FormalDataPlan,
    *,
    reservation_seed: int,
    count: int,
    run_spec: dict[str, Any],
    collection_spec: dict[str, Any],
    producer_git_state: dict[str, Any],
    allow_engineering: bool = False,
) -> SourceValidationReservation:
    if isinstance(reservation_seed, bool) or not isinstance(reservation_seed, int):
        raise TypeError("validation reservation seed must be an integer")
    if isinstance(count, bool) or not isinstance(count, int) or count < 3:
        raise ValueError("validation reservation needs at least three twin-pair seeds")
    if not allow_engineering and count < _MIN_FORMAL_PAIRS:
        raise ValueError(
            f"formal validation reservation needs at least {_MIN_FORMAL_PAIRS} pairs"
        )
    state = _validate_git_state(producer_git_state, formal=not allow_engineering)
    canonical_run_spec = source_validation_run_spec(**run_spec)
    canonical_collection_spec = source_validation_collection_spec(**collection_spec)
    if not allow_engineering and canonical_collection_spec["source_commit"] != state["head"]:
        raise ValueError(
            "formal validation collection commit must equal the clean reservation HEAD"
        )
    for name in ("task", "family", "defect", "level"):
        if canonical_collection_spec[name] != canonical_run_spec[name]:
            raise ValueError(f"validation collection and run specs disagree on {name}")
    development_rows = tuple(development_plan.payload["pool"])
    for row in development_rows:
        if (
            row["task"] != canonical_run_spec["task"]
            or row["family"] != canonical_run_spec["family"]
            or int(row["level"]) != canonical_run_spec["level"]
            or not (
                row["defect"] == canonical_run_spec["defect"]
                or (
                    row["defect"] == "clean"
                    and row["clean_of"] == canonical_run_spec["defect"]
                )
            )
        ):
            raise ValueError(
                "source-validation reservation requires a cell-specific development plan"
            )
    observed_indices = {int(row["embodiment_index"]) for row in development_rows}
    declared_indices = set(canonical_run_spec["source_embodiment_indices"]) | set(
        canonical_run_spec["target_embodiment_indices"]
    )
    if observed_indices != declared_indices:
        raise ValueError(
            "source/target indices must partition the cell-specific development plan"
        )
    observed_bindings = {str(row["paired_binding_id"]) for row in development_rows}
    if observed_bindings != {canonical_collection_spec["paired_binding_id"]}:
        raise ValueError(
            "validation collection geometry binding differs from the development cell"
        )
    excluded = {int(row["seed_env"]) for row in development_plan.payload["pool"]}
    reserved = _reserved_seeds(
        reservation_seed=reservation_seed,
        count=count,
        excluded=excluded,
    )
    sacrificial, heldout = _seed_roles(reserved, reservation_seed=reservation_seed)
    body = {
        "schema": SOURCE_VALIDATION_RESERVATION_SCHEMA,
        "marker": SOURCE_VALIDATION_RESERVATION_MARKER,
        "development_plan_sha256": development_plan.plan_sha256,
        "method_schema": _METHOD_SCHEMA,
        "selection_algorithm": _SELECTION_ALGORITHM,
        "reservation_seed": reservation_seed,
        "reserved_seed_envs": list(reserved),
        "sacrificial_seed_env": sacrificial,
        "heldout_seed_envs": list(heldout),
        "run_spec": canonical_run_spec,
        "collection_spec": canonical_collection_spec,
        "producer_git_state": state,
        "release_status": "engineering_only" if allow_engineering else "formal",
    }
    reservation = SourceValidationReservation(
        development_plan_sha256=body["development_plan_sha256"],
        method_schema=body["method_schema"],
        reservation_seed=body["reservation_seed"],
        reserved_seed_envs=tuple(body["reserved_seed_envs"]),
        sacrificial_seed_env=body["sacrificial_seed_env"],
        heldout_seed_envs=tuple(body["heldout_seed_envs"]),
        run_spec=body["run_spec"],
        collection_spec=body["collection_spec"],
        producer_git_state=body["producer_git_state"],
        release_status=body["release_status"],
        artifact_sha256=_canonical_sha256(body),
    )
    _validate_reservation(reservation)
    return reservation


def load_source_validation_reservation(
    path: Path | str,
    *,
    development_plan: FormalDataPlan | None = None,
) -> SourceValidationReservation:
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read source-validation reservation {source}") from exc
    expected = {
        "schema",
        "marker",
        "development_plan_sha256",
        "method_schema",
        "selection_algorithm",
        "reservation_seed",
        "reserved_seed_envs",
        "sacrificial_seed_env",
        "heldout_seed_envs",
        "run_spec",
        "collection_spec",
        "producer_git_state",
        "release_status",
        "artifact_sha256",
    }
    if not isinstance(document, dict) or set(document) != expected:
        raise ValueError("source-validation reservation fields are incomplete")
    if (
        document["schema"] != SOURCE_VALIDATION_RESERVATION_SCHEMA
        or document["marker"] != SOURCE_VALIDATION_RESERVATION_MARKER
        or document["selection_algorithm"] != _SELECTION_ALGORITHM
    ):
        raise ValueError("source-validation reservation schema or marker is invalid")
    reservation = SourceValidationReservation(
        development_plan_sha256=document["development_plan_sha256"],
        method_schema=document["method_schema"],
        reservation_seed=document["reservation_seed"],
        reserved_seed_envs=tuple(document["reserved_seed_envs"]),
        sacrificial_seed_env=document["sacrificial_seed_env"],
        heldout_seed_envs=tuple(document["heldout_seed_envs"]),
        run_spec=document["run_spec"],
        collection_spec=document["collection_spec"],
        producer_git_state=document["producer_git_state"],
        release_status=document["release_status"],
        artifact_sha256=document["artifact_sha256"],
    )
    _validate_reservation(reservation)
    if (
        development_plan is not None
        and reservation.development_plan_sha256 != development_plan.plan_sha256
    ):
        raise ValueError("validation reservation is bound to another development plan")
    return reservation
