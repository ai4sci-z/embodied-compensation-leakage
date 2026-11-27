"""Hash-bound data planning for the future formal policy adapter.

This module is deliberately independent of ``runners.train``.  It creates an
immutable description of a materialised pool, performs an episode-level split,
and fits proprioception statistics using the training episodes only.  It does
not train a model or create a paper result.  The marker and schema are kept
separate from the smoke contracts so a smoke artifact cannot be accepted by
this path accidentally.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch

from portal_trace.policies.obs import (
    CAMERAS,
    IMAGE_SIZE,
    PROPRIO_DIM,
    PROPRIO_GROUPS,
    PROPRIO_ACCELERATION_CAUSAL,
    PROPRIO_ACCELERATION_LEGACY,
    NormKind,
)
from portal_trace.runners.dataset_contract import (
    require_dataset_identity,
    require_delta_value_contract,
    require_paired_twin_contract,
    require_surrogate_context_contract,
)
from portal_trace.runners.policy_dataset import PolicyDatasetEntry

__all__ = [
    "FORMAL_DATA_PLAN_SCHEMA",
    "FORMAL_DATA_PLAN_MARKER",
    "FORMAL_NORMALIZATION_SCHEMA",
    "FORMAL_NORMALIZATION_MARKER",
    "PRECOLLECTION_SEED_SPLIT_STRATEGY",
    "FormalPoolEntry",
    "FormalDataPlan",
    "FormalProprioNormalization",
    "build_formal_data_plan",
    "derive_target_only_data_plan",
    "fit_formal_normalization",
    "heldout_cell_episode_ids",
    "state_manifest_sha256",
    "load_formal_data_plan",
    "load_formal_normalization",
    "proprio_acceleration_semantics",
]

FORMAL_DATA_PLAN_SCHEMA = "portal_trace.formal_data_plan/4"
_LEGACY_ENGINEERING_DATA_PLAN_SCHEMA = "portal_trace.formal_data_plan/3"
_LEGACY_ENGINEERING_SPLIT_STRATEGY = "stratified_condition_episode_sha256_rank"
FORMAL_DATA_PLAN_MARKER = "PORTAL_TRACE_FORMAL_DATA_PLAN_NOT_SMOKE"
FORMAL_NORMALIZATION_SCHEMA = "portal_trace.formal_normalization/1"
FORMAL_NORMALIZATION_MARKER = "PORTAL_TRACE_FORMAL_NORMALIZATION_NOT_SMOKE"


def proprio_acceleration_semantics(plan: FormalDataPlan) -> str:
    """Read the observation convention from the plan's bound HDF5 files."""
    import h5py

    conventions = set()
    for path in sorted({str(row["path"]) for row in plan.payload["pool"]}):
        with h5py.File(path, "r") as handle:
            value = handle.attrs.get(
                "proprio_acceleration_semantics", PROPRIO_ACCELERATION_LEGACY
            )
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            conventions.add(str(value))
    if len(conventions) != 1 or not conventions.issubset(
        {PROPRIO_ACCELERATION_LEGACY, PROPRIO_ACCELERATION_CAUSAL}
    ):
        raise ValueError(f"inconsistent or unsupported acceleration semantics: {sorted(conventions)}")
    return next(iter(conventions))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite formal data artifact {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class FormalPoolEntry:
    """One materialised file and its stable categorical identity."""

    path: Path
    task_index: int
    embodiment_index: int
    n_demos: int
    task: str = ""
    family: str = ""
    defect: str = ""
    level: int | None = None
    clean_of: str = ""

    @classmethod
    def from_policy_entry(cls, entry: PolicyDatasetEntry, **metadata: Any) -> FormalPoolEntry:
        return cls(
            Path(entry.path),
            entry.task_index,
            entry.embodiment_index,
            entry.n_demos,
            task=str(metadata.get("task", "")),
            family=str(metadata.get("family", "")),
            defect=str(metadata.get("defect", "")),
            level=metadata.get("level"),
            clean_of=str(metadata.get("clean_of", "")),
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).resolve())
        if self.task_index < 0 or self.embodiment_index < 0:
            raise ValueError("formal pool indices must be non-negative")
        if self.n_demos < 1:
            raise ValueError("formal pool entries must request at least one demo")
        for field in ("task", "family", "defect"):
            if not isinstance(getattr(self, field), str):
                raise TypeError(f"formal pool {field} must be text")
        if not isinstance(self.clean_of, str):
            raise TypeError("formal pool clean_of must be text")
        if self.level is not None and (
            isinstance(self.level, bool)
            or not isinstance(self.level, int)
            or self.level < 0
        ):
            raise ValueError("formal pool level must be a non-negative integer")


_CONDITION_FIELDS = ("task", "family", "defect", "level", "clean_of")


def _condition_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in _CONDITION_FIELDS)


def _require_unique_condition_seeds(rows: list[dict[str, Any]]) -> None:
    seen: dict[tuple[Any, ...], str] = {}
    duplicates: list[tuple[tuple[Any, ...], str, str]] = []
    for row in rows:
        key = (*_condition_identity(row), row.get("seed_env"))
        episode_id = str(row.get("id"))
        previous = seen.setdefault(key, episode_id)
        if previous != episode_id:
            duplicates.append((key, previous, episode_id))
    if duplicates:
        details = [
            {
                "condition_seed": list(key),
                "first_episode": first,
                "duplicate_episode": duplicate,
            }
            for key, first, duplicate in duplicates
        ]
        raise ValueError(
            "formal data plan refuses repeated condition/seed episodes; copied files "
            f"are not independent samples: {details}"
        )


def _required_text(attrs: Any, name: str, *, location: str) -> str:
    value = attrs.get(name)
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{location}: {name} is not valid UTF-8") from exc
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location}: {name} must be non-empty text")
    return value


def _required_integer(attrs: Any, name: str, *, location: str) -> int:
    value = attrs.get(name)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{location}: {name} must be an integer")
    return int(value)


def _episode_quality(attrs: Any, *, location: str) -> dict[str, Any]:
    quality = {
        "seed_env": _required_integer(attrs, "seed_env", location=location),
        "ilc_status": _required_text(attrs, "ilc_status", location=location),
        "clean_ilc_status": _required_text(attrs, "clean_ilc_status", location=location),
        "t_success": _required_integer(attrs, "t_success", location=location),
    }
    if quality["seed_env"] < 0:
        raise ValueError(f"{location}: seed_env must be non-negative")
    if quality["ilc_status"] != "converged" or quality["clean_ilc_status"] != "converged":
        raise ValueError(
            f"{location}: formal training requires both ILC twins to converge; "
            f"got ilc_status={quality['ilc_status']!r}, "
            f"clean_ilc_status={quality['clean_ilc_status']!r}"
        )
    if quality["t_success"] < 0:
        raise ValueError(
            f"{location}: formal training requires persistent task success (t_success >= 0)"
        )
    return quality


_SPLIT_STRATEGY = "paired_twin_group_sha256_rank"
PRECOLLECTION_SEED_SPLIT_STRATEGY = "paired_twin_seed_sha256_rank"
_SPLIT_STRATEGIES = {_SPLIT_STRATEGY, PRECOLLECTION_SEED_SPLIT_STRATEGY}


def _legacy_engineering_split(
    rows: list[dict[str, Any]], *, seed: int, train_fraction: float
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Reproduce schema 3's historical split without claiming twin isolation."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_condition_identity(row), []).append(row)
    train: list[str] = []
    heldout: list[str] = []
    for identity, group in sorted(groups.items(), key=lambda item: repr(item[0])):
        if len(group) < 2:
            raise ValueError(
                "legacy engineering data plan needs at least two episodes per condition; "
                f"{identity!r} has {len(group)}"
            )
        identity_token = json.dumps(identity, separators=(",", ":"), ensure_ascii=False)
        ranked = sorted(
            group,
            key=lambda row: hashlib.sha256(
                f"{seed}\0{identity_token}\0{row['id']}".encode("utf-8")
            ).hexdigest(),
        )
        n_train = min(len(ranked) - 1, max(1, round(len(ranked) * train_fraction)))
        train.extend(row["id"] for row in ranked[:n_train])
        heldout.extend(row["id"] for row in ranked[n_train:])
    return tuple(sorted(train)), tuple(sorted(heldout))


def _paired_stratified_split(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    train_fraction: float,
    strategy: str = _SPLIT_STRATEGY,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split complete clean/defective twin groups, never individual episodes."""
    if strategy not in _SPLIT_STRATEGIES:
        raise ValueError(f"unknown formal split strategy {strategy!r}")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        binding = row.get("paired_binding_id")
        pair_id = row.get("pair_id")
        if not isinstance(binding, str) or not binding:
            raise ValueError("formal data split requires paired_binding_id on every row")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError("formal data split requires pair_id on every row")
        groups.setdefault(binding, []).append(row)
    train: list[str] = []
    heldout: list[str] = []
    for binding, group in sorted(groups.items()):
        pair_groups: dict[str, list[dict[str, Any]]] = {}
        for row in group:
            pair_groups.setdefault(row["pair_id"], []).append(row)
        if len(pair_groups) < 2:
            raise ValueError(
                "formal data plan needs at least two admitted twin pairs per paired "
                f"binding for train/held-out coverage; {binding!r} has "
                f"{len(pair_groups)}"
            )
        pair_seeds: dict[str, int] = {}
        for pair_id, pair_rows in pair_groups.items():
            seeds = {row.get("seed_env") for row in pair_rows}
            if len(seeds) != 1:
                raise ValueError(
                    f"paired twin {pair_id!r} spans multiple environment seeds: "
                    f"{sorted(seeds, key=str)}"
                )
            pair_seeds[pair_id] = int(next(iter(seeds)))
        if len(set(pair_seeds.values())) != len(pair_seeds):
            raise ValueError(
                f"paired binding {binding!r} reuses an environment seed across twin pairs"
            )
        ranked_pair_ids = sorted(
            pair_groups,
            key=lambda pair_id: hashlib.sha256(
                (
                    f"{seed}\0{binding}\0{pair_id}"
                    if strategy == _SPLIT_STRATEGY
                    else f"{seed}\0{pair_seeds[pair_id]}"
                ).encode()
            ).hexdigest(),
        )
        n_train = min(
            len(ranked_pair_ids) - 1,
            max(1, round(len(ranked_pair_ids) * train_fraction)),
        )
        train_pairs = set(ranked_pair_ids[:n_train])
        for pair_id, pair_rows in pair_groups.items():
            destination = train if pair_id in train_pairs else heldout
            destination.extend(row["id"] for row in pair_rows)
    return tuple(sorted(train)), tuple(sorted(heldout))


@dataclass(frozen=True)
class FormalDataPlan:
    """Immutable, hash-bound episode split and pool manifest."""

    payload: dict[str, Any]
    artifact_sha256: str

    @property
    def plan_sha256(self) -> str:
        return self.artifact_sha256

    @property
    def train_episode_ids(self) -> tuple[str, ...]:
        return tuple(self.payload["split"]["train_episode_ids"])

    @property
    def heldout_episode_ids(self) -> tuple[str, ...]:
        return tuple(self.payload["split"]["heldout_episode_ids"])

    def write(self, path: Path) -> None:
        _immutable_json(Path(path), self.payload | {"artifact_sha256": self.artifact_sha256})


def heldout_cell_episode_ids(
    plan: FormalDataPlan,
    *,
    task: str,
    family: str,
    defect: str,
    level: int,
) -> tuple[str, ...]:
    """Return the exact held-out episodes for one physical evaluation cell."""
    heldout = set(plan.heldout_episode_ids)
    selected = tuple(
        sorted(
            row["id"]
            for row in plan.payload["pool"]
            if row["id"] in heldout
            and row["task"] == task
            and row["family"] == family
            and row["defect"] == defect
            and row["level"] == level
        )
    )
    if not selected:
        raise ValueError(
            "formal plan has no held-out episode for "
            f"{task}/{family}/{defect}/L{level}"
        )
    return selected


def state_manifest_sha256(
    plan: FormalDataPlan, episode_ids: tuple[str, ...]
) -> str:
    """Bind an analysis artifact to the selected same-state episode rows."""
    selected = set(episode_ids)
    rows = [
        {
            key: row[key]
            for key in (
                "id",
                "sha256",
                "pair_id",
                "paired_binding_id",
                "collection_source_commit",
                "collection_runtime_json",
            )
        }
        for row in plan.payload["pool"]
        if row["id"] in selected
    ]
    if {row["id"] for row in rows} != selected:
        raise ValueError("state manifest contains an episode outside the formal plan")
    encoded = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FormalProprioNormalization:
    """Training-only 70-channel statistics bound to one data-plan hash."""

    means: tuple[float, ...]
    scales: tuple[float, ...]
    training_episode_ids: tuple[str, ...]
    fitted_rows: int
    plan_sha256: str
    artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        if len(self.means) != PROPRIO_DIM or len(self.scales) != PROPRIO_DIM:
            raise ValueError(f"normalization must describe exactly {PROPRIO_DIM} channels")
        if not all(math.isfinite(v) for v in (*self.means, *self.scales)):
            raise ValueError("normalization contains a non-finite statistic")
        if any(v <= 0 for v in self.scales):
            raise ValueError("normalization scales must be positive")
        if not self.training_episode_ids or self.fitted_rows < 1:
            raise ValueError("normalization needs a non-empty training split")
        if len(self.plan_sha256) != 64:
            raise ValueError("normalization must bind a plan SHA-256")

    def apply(self, proprio: torch.Tensor) -> torch.Tensor:
        if proprio.shape[-1] != PROPRIO_DIM or proprio.ndim < 2:
            raise ValueError(f"normalization expects a final dimension of {PROPRIO_DIM}")
        if not torch.isfinite(proprio).all():
            raise ValueError("formal proprioception contains non-finite values")
        means = torch.tensor(self.means, dtype=proprio.dtype, device=proprio.device)
        scales = torch.tensor(self.scales, dtype=proprio.dtype, device=proprio.device)
        normalized = (proprio - means) / scales
        for group in PROPRIO_GROUPS:
            if group.norm is NormKind.UNIT_NORM:
                values = proprio[..., group.slice]
                normalized[..., group.slice] = values / torch.linalg.vector_norm(
                    values, dim=-1, keepdim=True
                ).clamp_min(1e-12)
            elif group.norm is NormKind.NONE:
                normalized[..., group.slice] = proprio[..., group.slice]
        if not torch.isfinite(normalized).all():
            raise ValueError("formal normalization produced non-finite values")
        return normalized

    def _body(self) -> dict[str, Any]:
        return {
            "schema": FORMAL_NORMALIZATION_SCHEMA,
            "marker": FORMAL_NORMALIZATION_MARKER,
            "plan_sha256": self.plan_sha256,
            "training_episode_ids": list(self.training_episode_ids),
            "fitted_rows": self.fitted_rows,
            "means": list(self.means),
            "scales": list(self.scales),
        }

    def write(self, path: Path) -> None:
        body = self._body()
        artifact = _canonical_sha256(body)
        _immutable_json(Path(path), body | {"artifact_sha256": artifact})


def _coerce_entries(
    entries: tuple[FormalPoolEntry | PolicyDatasetEntry, ...],
) -> tuple[FormalPoolEntry, ...]:
    out: list[FormalPoolEntry] = []
    for item in entries:
        if isinstance(item, FormalPoolEntry):
            out.append(item)
        elif isinstance(item, PolicyDatasetEntry):
            out.append(FormalPoolEntry.from_policy_entry(item))
        else:
            raise TypeError("formal pool entries must be FormalPoolEntry or PolicyDatasetEntry")
    if not out:
        raise ValueError("formal data plan requires at least one pool entry")
    paths = [str(entry.path) for entry in out]
    if len(set(paths)) != len(paths):
        raise ValueError("formal data plan refuses duplicate pool file paths")
    return tuple(sorted(out, key=lambda entry: str(entry.path)))


def _inspect_entry(entry: FormalPoolEntry) -> tuple[dict[str, Any], ...]:
    import h5py

    if not entry.path.is_file():
        raise FileNotFoundError(f"formal pool dataset does not exist: {entry.path}")
    if not entry.task or not entry.family or not entry.defect or entry.level is None:
        raise ValueError(
            f"{entry.path}: formal pool entry must declare task/family/defect/level"
        )
    require_dataset_identity(
        entry.path,
        task=entry.task,
        family=entry.family,
        defect=entry.defect,
        level=entry.level,
        clean_of=entry.clean_of,
        min_demos=entry.n_demos,
    )
    require_delta_value_contract(entry.path)
    require_paired_twin_contract(entry.path, verify_pair_marker=True)
    if entry.defect != "clean":
        require_surrogate_context_contract(entry.path)
    episodes: list[dict[str, Any]] = []
    try:
        with h5py.File(entry.path, "r") as dataset:
            if "data" not in dataset:
                raise ValueError(f"{entry.path}: materialised dataset has no data group")
            names = sorted(dataset["data"], key=lambda name: int(name.split("_")[-1]))
            if len(names) < entry.n_demos:
                raise ValueError(
                    f"{entry.path}: requested {entry.n_demos} demos but contains only {len(names)}"
                )
            root_reservation = None
            if "source_validation_reservation_sha256" in dataset.attrs:
                root_reservation = _required_text(
                    dataset.attrs,
                    "source_validation_reservation_sha256",
                    location=str(entry.path),
                )
            for demo in names[: entry.n_demos]:
                group = dataset["data"][demo]
                location = f"{entry.path}:data/{demo}"
                quality = _episode_quality(group.attrs, location=location)
                reservation_binding: dict[str, str] = {}
                if "source_validation_reservation_sha256" in group.attrs:
                    reservation_binding["source_validation_reservation_sha256"] = (
                        _required_text(
                            group.attrs,
                            "source_validation_reservation_sha256",
                            location=location,
                        )
                    )
                episode_reservation = reservation_binding.get(
                    "source_validation_reservation_sha256"
                )
                if episode_reservation != root_reservation:
                    raise ValueError(
                        f"{location}: root and episode validation reservation bindings differ"
                    )
                if any(
                    field not in group
                    for field in (
                        "actions",
                        "actions_defective",
                        "actions_clean",
                        "c_true",
                        "obs",
                    )
                ):
                    raise ValueError(f"{entry.path}:data/{demo}: missing policy datasets")
                horizon = len(group["actions"])
                if horizon < 1 or group["actions"].shape != (horizon, 7):
                    raise ValueError(f"{entry.path}:data/{demo}: actions have invalid shape")
                if group["c_true"].shape != (horizon, 7):
                    raise ValueError(f"{entry.path}:data/{demo}: c_true has invalid shape")
                actions = np.asarray(group["actions"], dtype=np.float32)
                defective = np.asarray(group["actions_defective"], dtype=np.float32)
                clean = np.asarray(group["actions_clean"], dtype=np.float32)
                truth = np.asarray(group["c_true"], dtype=np.float32)
                if (
                    defective.shape != (horizon, 7)
                    or clean.shape != (horizon, 7)
                    or not np.array_equal(actions, defective)
                    or not np.array_equal(defective - clean, truth)
                ):
                    raise ValueError(
                        f"{entry.path}:data/{demo}: paired action identity is invalid"
                    )
                proprio = group["obs"].get("robot0_proprio-state")
                if proprio is None or proprio.shape != (horizon, PROPRIO_DIM):
                    raise ValueError(f"{entry.path}:data/{demo}: proprio has invalid shape")
                for camera in CAMERAS:
                    image = group["obs"].get(camera)
                    if image is None or image.shape != (horizon, IMAGE_SIZE, IMAGE_SIZE, 3):
                        raise ValueError(
                            f"{entry.path}:data/{demo}: camera {camera} has invalid shape"
                        )
                episode_id = f"{entry.path}::data/{demo}"
                episodes.append(
                    {
                        "id": episode_id,
                        "demo": demo,
                        "horizon": horizon,
                        "window_ids": list(range(horizon)),
                        "window_count": horizon,
                        "pair_id": _required_text(group.attrs, "pair_id", location=location),
                        "paired_binding_id": _required_text(
                            group.attrs, "paired_binding_id", location=location
                        ),
                        "collection_source_commit": _required_text(
                            group.attrs, "collection_source_commit", location=location
                        ),
                        "collection_runtime_json": _required_text(
                            group.attrs, "collection_runtime_json", location=location
                        ),
                        **reservation_binding,
                        **quality,
                    }
                )
    except (OSError, KeyError) as exc:
        raise ValueError(f"{entry.path}: unreadable materialised policy dataset") from exc
    return tuple(episodes)


def build_formal_data_plan(
    entries: tuple[FormalPoolEntry | PolicyDatasetEntry, ...],
    *,
    seed: int,
    train_fraction: float = 0.8,
    source_commit: str,
    environment_fingerprint: str,
    split_strategy: str = _SPLIT_STRATEGY,
) -> FormalDataPlan:
    """Build a deterministic episode split with SHA-256 bindings.

    Complete clean/defective twin pairs are ranked as one unit; no pair can
    appear in both sets. The ranking is independent of input order. Prospective
    validation may rank by its pre-reserved environment seed so the role is
    fixed before trajectory contents exist.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("formal split seed must be an integer")
    if not math.isfinite(train_fraction) or not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit.lower())
    ):
        raise ValueError("formal data plan source_commit must be a complete Git SHA-1")
    if not environment_fingerprint:
        raise ValueError("formal data plan requires an environment_fingerprint")
    if split_strategy not in _SPLIT_STRATEGIES:
        raise ValueError(f"unknown formal split strategy {split_strategy!r}")
    pool = _coerce_entries(entries)
    task_to_index: dict[str, int] = {}
    index_to_task: dict[int, str] = {}
    embodiment_to_index: dict[tuple[str, str, int, str], int] = {}
    index_to_embodiment: dict[int, tuple[str, str, int, str]] = {}
    for entry in pool:
        if not entry.task or not entry.family or not entry.defect or entry.level is None:
            raise ValueError(
                f"{entry.path}: formal pool entry must declare task/family/defect/level"
            )
        task_previous = task_to_index.setdefault(entry.task, entry.task_index)
        index_previous = index_to_task.setdefault(entry.task_index, entry.task)
        if task_previous != entry.task_index or index_previous != entry.task:
            raise ValueError("formal task names and task_index values must form a bijection")
        embodiment = (entry.family, entry.defect, entry.level, entry.clean_of)
        embodiment_previous = embodiment_to_index.setdefault(
            embodiment, entry.embodiment_index
        )
        index_previous_embodiment = index_to_embodiment.setdefault(
            entry.embodiment_index, embodiment
        )
        if (
            embodiment_previous != entry.embodiment_index
            or index_previous_embodiment != embodiment
        ):
            raise ValueError(
                "physical embodiment identities and embodiment_index values must form a "
                "bijection; environment seed/file identity must not create an embodiment"
            )
    if set(task_to_index.values()) != set(range(len(task_to_index))):
        raise ValueError("formal task_index values must be contiguous from zero")
    if set(embodiment_to_index.values()) != set(range(len(embodiment_to_index))):
        raise ValueError("formal embodiment_index values must be contiguous from zero")
    rows: list[dict[str, Any]] = []
    for entry in pool:
        for episode in _inspect_entry(entry):
            rows.append(
                {
                    **episode,
                    "path": str(entry.path),
                    "sha256": _sha256(entry.path),
                    "task_index": entry.task_index,
                    "embodiment_index": entry.embodiment_index,
                    "task": entry.task,
                    "family": entry.family,
                    "defect": entry.defect,
                    "level": entry.level,
                    "clean_of": entry.clean_of,
                }
            )
    if len(rows) < 2:
        raise ValueError("formal data plan requires at least two episodes for a held-out split")
    _require_unique_condition_seeds(rows)
    observed_commits = {row["collection_source_commit"] for row in rows}
    observed_runtimes = {row["collection_runtime_json"] for row in rows}
    if observed_commits != {source_commit}:
        raise ValueError(
            "formal data plan source_commit disagrees with collected episode provenance"
        )
    if observed_runtimes != {environment_fingerprint}:
        raise ValueError(
            "formal data plan environment_fingerprint disagrees with collected runtime"
        )
    train, heldout = _paired_stratified_split(
        rows, seed=seed, train_fraction=train_fraction, strategy=split_strategy
    )
    body: dict[str, Any] = {
        "schema": FORMAL_DATA_PLAN_SCHEMA,
        "marker": FORMAL_DATA_PLAN_MARKER,
        "seed": seed,
        "split": {
            "strategy": split_strategy,
            "train_fraction": train_fraction,
            "train_episode_ids": list(train),
            "heldout_episode_ids": list(heldout),
        },
        "pool": rows,
        "counts": {
            "episodes": len(rows),
            "train_episodes": len(train),
            "heldout_episodes": len(heldout),
            "train_windows": sum(row["window_count"] for row in rows if row["id"] in train),
            "heldout_windows": sum(row["window_count"] for row in rows if row["id"] in heldout),
        },
        "provenance": {
            "source_commit": source_commit,
            "environment_fingerprint": environment_fingerprint,
        },
    }
    plan = FormalDataPlan(body, _canonical_sha256(body))
    _verify_plan(plan)
    return plan


def derive_target_only_data_plan(
    plan: FormalDataPlan,
    *,
    target_embodiment_index: int | None = None,
    target_embodiment_indices: tuple[int, ...] | None = None,
) -> FormalDataPlan:
    """Derive a hash-bound B1 plan containing one target embodiment only.

    The source plan is verified before rows are selected, and the selected
    episodes are re-split with the source plan's deterministic seed and train
    fraction.  This keeps B1's train/held-out boundary episode-disjoint while
    retaining every dataset SHA-256 binding.  A target subset with fewer than
    two episodes is rejected rather than silently becoming a train-only pilot.
    """
    _verify_plan(plan)
    if target_embodiment_indices is None:
        if (
            isinstance(target_embodiment_index, bool)
            or not isinstance(target_embodiment_index, int)
            or target_embodiment_index < 0
        ):
            raise ValueError("target_embodiment_index must be a non-negative integer")
        target_indices = (target_embodiment_index,)
    else:
        if target_embodiment_index is not None:
            raise ValueError(
                "provide target_embodiment_index or target_embodiment_indices, not both"
            )
        if not target_embodiment_indices:
            raise ValueError("target_embodiment_indices must not be empty")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in target_embodiment_indices
        ):
            raise ValueError("target_embodiment_indices must contain non-negative integers")
        target_indices = tuple(sorted(set(target_embodiment_indices)))

    selected = [
        {**dict(row), "embodiment_index": 0}
        for row in plan.payload["pool"]
        if int(row["embodiment_index"]) in target_indices
    ]
    if len(selected) < 2:
        raise ValueError(
            "target-only formal plan requires at least two target episodes for a "
            f"held-out split; embodiment_indices={list(target_indices)} has "
            f"{len(selected)}"
        )
    # A target identity may span environment seeds, but it may not span
    # different task/family/defect/level conditions.  Without this check the
    # helper could create a deceptively clean B1 plan from a mixed subset.
    identity_fields = (
        "task",
        "task_index",
        "family",
        "defect",
        "level",
        "clean_of",
    )
    identities = {
        tuple(row.get(field) for field in identity_fields) for row in selected
    }
    if len(identities) != 1:
        raise ValueError(
            "target-only source embodiment group mixes physical identities "
            f"identities: {sorted(identities, key=str)}"
        )

    split = plan.payload["split"]
    train_fraction = float(split["train_fraction"])
    train, heldout = _paired_stratified_split(
        selected,
        seed=int(plan.payload["seed"]),
        train_fraction=train_fraction,
        strategy=split["strategy"],
    )
    body: dict[str, Any] = {
        "schema": FORMAL_DATA_PLAN_SCHEMA,
        "marker": FORMAL_DATA_PLAN_MARKER,
        "seed": plan.payload["seed"],
        "split": {
            "strategy": split["strategy"],
            "train_fraction": train_fraction,
            "train_episode_ids": list(train),
            "heldout_episode_ids": list(heldout),
        },
        "pool": sorted(selected, key=lambda row: row["id"]),
        "counts": {
            "episodes": len(selected),
            "train_episodes": len(train),
            "heldout_episodes": len(heldout),
            "train_windows": sum(row["window_count"] for row in selected if row["id"] in train),
            "heldout_windows": sum(
                row["window_count"] for row in selected if row["id"] in heldout
            ),
        },
        "provenance": {
            **plan.payload["provenance"],
            "derived_from_plan_sha256": plan.plan_sha256,
            "selection": {
                "kind": "embodiment_index",
                "target_embodiment_indices": list(target_indices),
            },
        },
    }
    derived = FormalDataPlan(body, _canonical_sha256(body))
    _verify_plan(derived)
    return derived


def _verify_plan(
    plan: FormalDataPlan, *, allow_legacy_engineering: bool = False
) -> None:
    legacy_engineering = (
        allow_legacy_engineering
        and plan.payload.get("schema") == _LEGACY_ENGINEERING_DATA_PLAN_SCHEMA
    )
    if (
        (plan.payload.get("schema") != FORMAL_DATA_PLAN_SCHEMA and not legacy_engineering)
        or plan.payload.get("marker") != FORMAL_DATA_PLAN_MARKER
    ):
        raise ValueError("formal data plan schema or marker is invalid")
    body = dict(plan.payload)
    if _canonical_sha256(body) != plan.artifact_sha256:
        raise ValueError("formal data plan content SHA-256 does not match")
    if isinstance(plan.payload.get("seed"), bool) or not isinstance(
        plan.payload.get("seed"), int
    ):
        raise ValueError("formal data plan seed must be an integer")
    split = plan.payload.get("split")
    if not isinstance(split, dict) or set(split) != {
        "strategy",
        "train_fraction",
        "train_episode_ids",
        "heldout_episode_ids",
    }:
        raise ValueError("formal data plan split is incomplete")
    train_fraction = split["train_fraction"]
    allowed_strategies = (
        {_LEGACY_ENGINEERING_SPLIT_STRATEGY} if legacy_engineering else _SPLIT_STRATEGIES
    )
    if (
        split["strategy"] not in allowed_strategies
        or not isinstance(train_fraction, (int, float))
        or isinstance(train_fraction, bool)
        or not math.isfinite(float(train_fraction))
        or not 0.0 < float(train_fraction) < 1.0
    ):
        raise ValueError("formal data plan split strategy or train_fraction is invalid")
    train = set(plan.train_episode_ids)
    heldout = set(plan.heldout_episode_ids)
    if not train or not heldout or train & heldout:
        raise ValueError("formal data plan split must be non-empty and disjoint")
    if len(train) != len(plan.train_episode_ids) or len(heldout) != len(
        plan.heldout_episode_ids
    ):
        raise ValueError("formal data plan split IDs must be unique")
    if list(plan.train_episode_ids) != sorted(train) or list(plan.heldout_episode_ids) != sorted(
        heldout
    ):
        raise ValueError("formal data plan split IDs must be canonically ordered")
    pool = plan.payload.get("pool")
    if not isinstance(pool, list) or not pool:
        raise ValueError("formal data plan pool is missing or empty")
    ids = [row.get("id") for row in pool if isinstance(row, dict)]
    if len(ids) != len(pool) or any(not isinstance(value, str) for value in ids):
        raise ValueError("formal data plan pool contains malformed episode rows")
    if len(set(ids)) != len(ids) or set(ids) != train | heldout:
        raise ValueError("formal data plan split IDs must exactly cover pool episodes")
    expected_row_fields = {
        "id",
        "demo",
        "horizon",
        "window_ids",
        "window_count",
        "seed_env",
        "ilc_status",
        "clean_ilc_status",
        "t_success",
        "path",
        "sha256",
        "task_index",
        "embodiment_index",
        "task",
        "family",
        "defect",
        "level",
        "clean_of",
        "pair_id",
        "paired_binding_id",
        "collection_source_commit",
        "collection_runtime_json",
    }
    optional_row_fields = {"source_validation_reservation_sha256"}
    if any(
        set(row) not in (expected_row_fields, expected_row_fields | optional_row_fields)
        for row in pool
    ):
        raise ValueError("formal data plan episode fields are incomplete or unexpected")
    reservation_presence = {
        "source_validation_reservation_sha256" in row for row in pool
    }
    if len(reservation_presence) != 1:
        raise ValueError("formal data plan mixes reserved and unreserved collection rows")
    _require_unique_condition_seeds(pool)
    if legacy_engineering:
        expected_train, expected_heldout = _legacy_engineering_split(
            pool, seed=plan.payload["seed"], train_fraction=float(train_fraction)
        )
    else:
        expected_train, expected_heldout = _paired_stratified_split(
            pool,
            seed=plan.payload["seed"],
            train_fraction=float(train_fraction),
            strategy=split["strategy"],
        )
    if tuple(plan.train_episode_ids) != expected_train or tuple(
        plan.heldout_episode_ids
    ) != expected_heldout:
        raise ValueError("formal data plan split does not match its declared split contract")
    train_pairs = {row["pair_id"] for row in pool if row["id"] in train}
    heldout_pairs = {row["pair_id"] for row in pool if row["id"] in heldout}
    if train_pairs & heldout_pairs and not legacy_engineering:
        raise ValueError("formal data plan splits a paired twin across train and held-out")

    task_to_index: dict[str, int] = {}
    index_to_task: dict[int, str] = {}
    embodiment_to_index: dict[tuple[str, str, int, str], int] = {}
    index_to_embodiment: dict[int, tuple[str, str, int, str]] = {}
    for row in pool:
        if (
            not isinstance(row["task"], str)
            or not row["task"]
            or not isinstance(row["family"], str)
            or not row["family"]
            or not isinstance(row["defect"], str)
            or not row["defect"]
            or not isinstance(row["clean_of"], str)
            or isinstance(row["level"], bool)
            or not isinstance(row["level"], int)
            or row["level"] < 0
            or isinstance(row["task_index"], bool)
            or not isinstance(row["task_index"], int)
            or row["task_index"] < 0
            or isinstance(row["embodiment_index"], bool)
            or not isinstance(row["embodiment_index"], int)
            or row["embodiment_index"] < 0
        ):
            raise ValueError("formal data plan contains an invalid physical identity")
        task_previous = task_to_index.setdefault(row["task"], row["task_index"])
        index_previous = index_to_task.setdefault(row["task_index"], row["task"])
        if task_previous != row["task_index"] or index_previous != row["task"]:
            raise ValueError("formal task names and task_index values must form a bijection")
        embodiment = (
            row["family"],
            row["defect"],
            row["level"],
            row["clean_of"],
        )
        embodiment_previous = embodiment_to_index.setdefault(
            embodiment, row["embodiment_index"]
        )
        index_previous_embodiment = index_to_embodiment.setdefault(
            row["embodiment_index"], embodiment
        )
        if (
            embodiment_previous != row["embodiment_index"]
            or index_previous_embodiment != embodiment
        ):
            raise ValueError(
                "physical embodiment identities and embodiment_index values must form a "
                "bijection"
            )
    if set(task_to_index.values()) != set(range(len(task_to_index))):
        raise ValueError("formal task_index values must be contiguous from zero")
    if set(embodiment_to_index.values()) != set(range(len(embodiment_to_index))):
        raise ValueError("formal embodiment_index values must be contiguous from zero")
    counts = plan.payload.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("formal data plan counts are missing")
    expected_counts = {
        "episodes": len(pool),
        "train_episodes": len(train),
        "heldout_episodes": len(heldout),
        "train_windows": sum(row["window_count"] for row in pool if row["id"] in train),
        "heldout_windows": sum(row["window_count"] for row in pool if row["id"] in heldout),
    }
    if counts != expected_counts:
        raise ValueError("formal data plan counts disagree with its episode split")
    provenance = plan.payload.get("provenance")
    if (
        not isinstance(provenance, dict)
        or not isinstance(provenance.get("source_commit"), str)
        or not provenance["source_commit"]
        or not isinstance(provenance.get("environment_fingerprint"), str)
        or not provenance["environment_fingerprint"]
    ):
        raise ValueError("formal data plan provenance is incomplete")
    import h5py

    verified_files: set[tuple[str, str, str, str, int, str]] = set()
    file_hashes: dict[Path, str] = {}
    for row in pool:
        if (
            not isinstance(row.get("sha256"), str)
            or len(row["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in row["sha256"])
            or isinstance(row.get("window_count"), bool)
            or not isinstance(row.get("window_count"), int)
            or row["window_count"] < 1
            or row.get("window_ids") != list(range(row["window_count"]))
            or row.get("horizon") != row["window_count"]
            or not isinstance(row.get("demo"), str)
            or not row["demo"]
            or not isinstance(row.get("path"), str)
            or not row["path"]
            or isinstance(row.get("seed_env"), bool)
            or not isinstance(row.get("seed_env"), int)
            or row["seed_env"] < 0
            or not isinstance(row.get("ilc_status"), str)
            or row["ilc_status"] != "converged"
            or not isinstance(row.get("clean_ilc_status"), str)
            or row["clean_ilc_status"] != "converged"
            or isinstance(row.get("t_success"), bool)
            or not isinstance(row.get("t_success"), int)
            or row["t_success"] < 0
            or not isinstance(row.get("pair_id"), str)
            or len(row["pair_id"]) != 64
            or not isinstance(row.get("paired_binding_id"), str)
            or len(row["paired_binding_id"]) != 64
            or row.get("collection_source_commit") != provenance["source_commit"]
            or row.get("collection_runtime_json") != provenance["environment_fingerprint"]
            or (
                "source_validation_reservation_sha256" in row
                and (
                    len(row["source_validation_reservation_sha256"]) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in row["source_validation_reservation_sha256"]
                    )
                )
            )
        ):
            raise ValueError("formal data plan contains an invalid episode binding")
        path = Path(row["path"])
        if not path.is_absolute():
            raise ValueError("formal data plan dataset paths must be absolute")
        if not path.is_file():
            raise ValueError(f"formal data plan dataset is unavailable: {path}")
        observed_hash = file_hashes.get(path)
        if observed_hash is None:
            observed_hash = _sha256(path)
            file_hashes[path] = observed_hash
        if observed_hash != row["sha256"]:
            raise ValueError(f"formal data plan dataset SHA-256 no longer matches: {path}")
        if row["id"] != f"{path}::data/{row['demo']}":
            raise ValueError("formal data plan episode id disagrees with path/demo binding")
        identity_key = (
            str(path),
            row["task"],
            row["family"],
            row["defect"],
            row["level"],
            row["clean_of"],
        )
        if identity_key not in verified_files:
            require_dataset_identity(
                path,
                task=row["task"],
                family=row["family"],
                defect=row["defect"],
                level=row["level"],
                clean_of=row["clean_of"],
                min_demos=1,
            )
            require_paired_twin_contract(path, verify_pair_marker=True)
            if row["defect"] != "clean":
                require_surrogate_context_contract(path)
            verified_files.add(identity_key)
        try:
            with h5py.File(path, "r") as dataset:
                group = dataset["data"][row["demo"]]
                if len(group["actions"]) != row["horizon"]:
                    raise ValueError(
                        f"{row['id']}: action horizon changed since the plan was created"
                    )
                quality = _episode_quality(group.attrs, location=row["id"])
                if "source_validation_reservation_sha256" in row:
                    observed_reservation = _required_text(
                        group.attrs,
                        "source_validation_reservation_sha256",
                        location=row["id"],
                    )
                    if observed_reservation != row["source_validation_reservation_sha256"]:
                        raise ValueError(
                            f"{row['id']}: validation reservation binding changed"
                        )
        except (OSError, KeyError) as exc:
            raise ValueError(f"{row['id']}: bound episode is unavailable") from exc
        for field in ("seed_env", "ilc_status", "clean_ilc_status", "t_success"):
            if row[field] != quality[field]:
                raise ValueError(f"{row['id']}: stored {field} quality binding changed")


def load_formal_data_plan(
    path: Path, *, allow_legacy_engineering: bool = False
) -> FormalDataPlan:
    """Read a bound plan; schema 3 requires explicit engineering-only admission."""
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read formal data plan {path}") from exc
    if not isinstance(document, dict):
        raise ValueError("formal data plan must be a JSON object")
    artifact = document.pop("artifact_sha256", None)
    if not isinstance(artifact, str) or len(artifact) != 64:
        raise ValueError("formal data plan has no artifact SHA-256")
    expected_fields = {
        "schema",
        "marker",
        "seed",
        "split",
        "pool",
        "counts",
        "provenance",
    }
    if set(document) != expected_fields:
        raise ValueError("formal data plan fields are incomplete or unexpected")
    plan = FormalDataPlan(document, artifact)
    _verify_plan(plan, allow_legacy_engineering=allow_legacy_engineering)
    return plan


def fit_formal_normalization(plan: FormalDataPlan) -> FormalProprioNormalization:
    """Fit 9.6.2 statistics using rows from the plan's training episodes only."""
    _verify_plan(plan)
    import h5py

    train_ids = set(plan.train_episode_ids)
    rows: list[np.ndarray] = []
    for item in plan.payload["pool"]:
        if item["id"] not in train_ids:
            continue
        with h5py.File(item["path"], "r") as dataset:
            values = np.asarray(
                dataset["data"][item["demo"]]["obs"]["robot0_proprio-state"],
                dtype=np.float64,
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{item['id']}: training proprioception contains non-finite values")
        rows.append(values)
    if not rows:
        raise ValueError("formal normalization has no training rows")
    matrix = np.concatenate(rows, axis=0)
    means = np.zeros(PROPRIO_DIM, dtype=np.float64)
    scales = np.ones(PROPRIO_DIM, dtype=np.float64)
    for group in PROPRIO_GROUPS:
        values = matrix[:, group.slice]
        if group.norm in {NormKind.STANDARDISE, NormKind.ZERO_MEAN}:
            means[group.slice] = values.mean(axis=0)
        if group.norm is NormKind.STANDARDISE:
            scales[group.slice] = np.maximum(values.std(axis=0), 1e-3)
    return FormalProprioNormalization(
        tuple(float(v) for v in means),
        tuple(float(v) for v in scales),
        tuple(sorted(train_ids)),
        int(matrix.shape[0]),
        plan.plan_sha256,
    )


def load_formal_normalization(
    path: Path,
    *,
    plan: FormalDataPlan,
    allow_legacy_engineering: bool = False,
) -> FormalProprioNormalization:
    _verify_plan(plan, allow_legacy_engineering=allow_legacy_engineering)
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read formal normalization {path}") from exc
    if not isinstance(document, dict):
        raise ValueError("formal normalization must be a JSON object")
    artifact = document.pop("artifact_sha256", None)
    if not isinstance(artifact, str) or _canonical_sha256(document) != artifact:
        raise ValueError("formal normalization content SHA-256 does not match")
    expected_fields = {
        "schema",
        "marker",
        "plan_sha256",
        "training_episode_ids",
        "fitted_rows",
        "means",
        "scales",
    }
    if set(document) != expected_fields:
        raise ValueError("formal normalization fields are incomplete or unexpected")
    if (
        document.get("schema") != FORMAL_NORMALIZATION_SCHEMA
        or document.get("marker") != FORMAL_NORMALIZATION_MARKER
    ):
        raise ValueError("formal normalization schema or marker is invalid")
    if document.get("plan_sha256") != plan.plan_sha256:
        raise ValueError("formal normalization is bound to a different data plan")
    result = FormalProprioNormalization(
        tuple(document.get("means", ())),
        tuple(document.get("scales", ())),
        tuple(document.get("training_episode_ids", ())),
        int(document.get("fitted_rows", 0)),
        str(document["plan_sha256"]),
        artifact,
    )
    if tuple(result.training_episode_ids) != tuple(plan.train_episode_ids):
        raise ValueError("formal normalization training split disagrees with data plan")
    return result
