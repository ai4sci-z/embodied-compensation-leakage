"""Immutable matched-surrogate fields for one held-out evaluation cell."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "SURROGATE_INPUT_SCHEMA",
    "SurrogateInputProvenance",
    "SurrogateInputEpisode",
    "SurrogateInputHeader",
    "write_surrogate_input",
    "load_surrogate_input",
]

SURROGATE_INPUT_SCHEMA = "portal_trace.surrogate_input/1"
_MARKER = "PORTAL_TRACE_HASH_BOUND_MATCHED_SURROGATES"


def _sha(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 string")
    return value


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    )


@dataclass(frozen=True)
class SurrogateInputProvenance:
    evaluation_plan_sha256: str
    source_state_manifest_sha256: str
    construction_sha256: str
    task: str
    family: str
    defect: str
    level: int
    seed: int
    admitted_surrogates: tuple[str, ...]
    struck_surrogates: tuple[str, ...]
    match_reports: Mapping[str, Mapping[str, Any]]
    release_status: str

    def __post_init__(self) -> None:
        for field in (
            "evaluation_plan_sha256", "source_state_manifest_sha256",
            "construction_sha256",
        ):
            _sha(getattr(self, field), field=field)
        for field in ("task", "family", "defect"):
            if not str(getattr(self, field)).strip():
                raise ValueError(f"surrogate {field} must be named")
        if isinstance(self.level, bool) or not isinstance(self.level, int) or self.level < 0:
            raise ValueError("surrogate level must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("surrogate seed must be an integer")
        admitted = tuple(self.admitted_surrogates)
        struck = tuple(self.struck_surrogates)
        if not admitted or set(admitted) & set(struck):
            raise ValueError("surrogate admitted/struck sets must be non-empty and disjoint")
        if set(self.match_reports) != set(admitted) | set(struck):
            raise ValueError("surrogate match reports do not cover the declared set")
        if self.release_status not in {"formal", "engineering_only"}:
            raise ValueError("release_status must be formal or engineering_only")

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluation_plan_sha256": self.evaluation_plan_sha256,
            "source_state_manifest_sha256": self.source_state_manifest_sha256,
            "construction_sha256": self.construction_sha256,
            "task": self.task,
            "family": self.family,
            "defect": self.defect,
            "level": self.level,
            "seed": self.seed,
            "admitted_surrogates": list(self.admitted_surrogates),
            "struck_surrogates": list(self.struck_surrogates),
            "match_reports": {
                name: dict(report) for name, report in self.match_reports.items()
            },
            "release_status": self.release_status,
        }


@dataclass(frozen=True)
class SurrogateInputEpisode:
    episode_id: str
    fields: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("surrogate episode_id must be named")
        if not self.fields:
            raise ValueError("surrogate episode must contain admitted fields")
        shape: tuple[int, int] | None = None
        for name, values in self.fields.items():
            if not name or "/" in name or "\\" in name:
                raise ValueError("surrogate names must be plain non-empty identifiers")
            array = np.asarray(values, dtype=np.float64)
            if array.ndim != 2 or min(array.shape) < 1 or not np.isfinite(array).all():
                raise ValueError(f"surrogate {name} must be a finite (frames, action_dim) array")
            if shape is None:
                shape = array.shape
            elif array.shape != shape:
                raise ValueError("surrogate fields changed shape within one episode")


@dataclass(frozen=True)
class SurrogateInputHeader:
    provenance: SurrogateInputProvenance
    episode_ids: tuple[str, ...]
    surrogate_names: tuple[str, ...]
    n_frames: int
    action_dim: int
    artifact_sha256: str


def _artifact_sha(
    provenance: SurrogateInputProvenance,
    episodes: Sequence[SurrogateInputEpisode],
) -> str:
    digest = hashlib.sha256()
    digest.update(SURROGATE_INPUT_SCHEMA.encode("ascii"))
    digest.update(_canonical_json(provenance.as_dict()).encode("ascii"))
    for episode in episodes:
        digest.update(episode.episode_id.encode("utf-8"))
        for name in sorted(episode.fields):
            values = np.ascontiguousarray(episode.fields[name], dtype=np.float64)
            digest.update(name.encode("utf-8"))
            digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
            digest.update(values.tobytes())
    return digest.hexdigest()


def write_surrogate_input(
    path: Path | str,
    *,
    provenance: SurrogateInputProvenance,
    episodes: Sequence[SurrogateInputEpisode],
) -> SurrogateInputHeader:
    import h5py

    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite surrogate input {output}")
    rows = tuple(episodes)
    if not rows:
        raise ValueError("a surrogate input needs at least one episode")
    episode_ids = tuple(row.episode_id for row in rows)
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("surrogate input episode IDs must be unique")
    names = tuple(sorted(rows[0].fields))
    if names != tuple(sorted(provenance.admitted_surrogates)):
        raise ValueError("surrogate fields disagree with the admitted provenance set")
    if any(tuple(sorted(row.fields)) != names for row in rows):
        raise ValueError("surrogate field set changed between episodes")
    shape = np.shape(rows[0].fields[names[0]])
    if any(
        np.shape(values) != (np.shape(values)[0], shape[1])
        for row in rows
        for values in row.fields.values()
    ):
        raise ValueError("surrogate action width changed between episodes")
    artifact_sha = _artifact_sha(provenance, rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        with h5py.File(temporary, "w") as root:
            root.attrs["schema"] = SURROGATE_INPUT_SCHEMA
            root.attrs["marker"] = _MARKER
            root.attrs["provenance_json"] = _canonical_json(provenance.as_dict())
            root.attrs["artifact_sha256"] = artifact_sha
            data = root.create_group("data")
            for index, episode in enumerate(rows):
                group = data.create_group(f"episode_{index:06d}")
                group.attrs["episode_id"] = episode.episode_id
                for name in names:
                    group.create_dataset(
                        name, data=episode.fields[name], compression="gzip"
                    )
            root.flush()
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return SurrogateInputHeader(
        provenance=provenance,
        episode_ids=episode_ids,
        surrogate_names=names,
        n_frames=sum(np.shape(row.fields[names[0]])[0] for row in rows),
        action_dim=int(shape[1]),
        artifact_sha256=artifact_sha,
    )


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_surrogate_input(
    path: Path | str,
) -> tuple[SurrogateInputHeader, dict[str, dict[str, np.ndarray]]]:
    import h5py

    source = Path(path)
    with h5py.File(source, "r") as root:
        if _decode(root.attrs.get("schema", "")) != SURROGATE_INPUT_SCHEMA:
            raise ValueError(f"{source}: unsupported surrogate-input schema")
        if _decode(root.attrs.get("marker", "")) != _MARKER:
            raise ValueError(f"{source}: surrogate-input marker is missing")
        try:
            raw = json.loads(_decode(root.attrs["provenance_json"]))
            declared_sha = _sha(
                _decode(root.attrs["artifact_sha256"]), field="artifact_sha256"
            )
            groups = root["data"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{source}: surrogate-input header is incomplete") from exc
        expected = {
            "evaluation_plan_sha256", "source_state_manifest_sha256",
            "construction_sha256", "task", "family", "defect", "level", "seed",
            "admitted_surrogates", "struck_surrogates", "match_reports",
            "release_status",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError(f"{source}: surrogate-input provenance is incomplete")
        provenance = SurrogateInputProvenance(
            **raw
            | {
                "admitted_surrogates": tuple(raw["admitted_surrogates"]),
                "struck_surrogates": tuple(raw["struck_surrogates"]),
            }
        )
        episodes: list[SurrogateInputEpisode] = []
        for name in sorted(groups):
            group = groups[name]
            episodes.append(
                SurrogateInputEpisode(
                    episode_id=_decode(group.attrs["episode_id"]),
                    fields={field: group[field][()] for field in group},
                )
            )
    if not episodes:
        raise ValueError(f"{source}: surrogate-input artifact contains no episodes")
    if len({row.episode_id for row in episodes}) != len(episodes):
        raise ValueError(f"{source}: surrogate-input episode IDs are not unique")
    observed_sha = _artifact_sha(provenance, episodes)
    if observed_sha != declared_sha:
        raise ValueError(f"{source}: surrogate-input content SHA-256 changed")
    names = tuple(sorted(episodes[0].fields))
    if any(tuple(sorted(row.fields)) != names for row in episodes):
        raise ValueError(f"{source}: surrogate set changes between episodes")
    rows = {
        row.episode_id: {
            name: np.asarray(values, dtype=np.float64)
            for name, values in row.fields.items()
        }
        for row in episodes
    }
    return (
        SurrogateInputHeader(
            provenance=provenance,
            episode_ids=tuple(row.episode_id for row in episodes),
            surrogate_names=names,
            n_frames=sum(np.shape(row.fields[names[0]])[0] for row in episodes),
            action_dim=int(np.shape(episodes[0].fields[names[0]])[1]),
            artifact_sha256=declared_sha,
        ),
        rows,
    )
