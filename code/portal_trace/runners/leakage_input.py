"""Hash-bound inputs for the offline Lambda diagnostic.

The leakage endpoint is evaluated on one shared set of ``rho_off`` frames.  A
valid input therefore has to bind the evaluated policy, B7 Oracle, teacher
counterfactual and nuisance fields to exactly the same frame rows.  This module
owns that boundary.  It does not infer a missing field and it does not write
into the demonstration dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from portal_trace.diagnostics.leakage import InterferenceBasis, LeakageChunk

__all__ = [
    "LEAKAGE_INPUT_SCHEMA",
    "LEAKAGE_INPUT_MARKER",
    "LeakageInputProvenance",
    "LeakageInputEpisode",
    "LeakageInputHeader",
    "write_leakage_input",
    "load_leakage_input_header",
    "stream_leakage_input",
]

LEAKAGE_INPUT_SCHEMA = "portal_trace.leakage_input/3"
LEAKAGE_INPUT_MARKER = "PORTAL_TRACE_HASH_BOUND_LEAKAGE_INPUT"
_IDENTITY_ATOL = 1e-10
_STORED_ACTION_IDENTITY_ATOL = 4.0 * float(np.finfo(np.float32).eps)


def _require_sha256(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 string")
    return value


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _action_field(value: Any, *, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape (frames, action_dim), got {array.shape}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _assert_identity(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    statement: str,
    atol: float = _IDENTITY_ATOL,
) -> None:
    if not np.allclose(lhs, rhs, rtol=0.0, atol=atol):
        error = float(np.max(np.abs(lhs - rhs)))
        raise ValueError(
            f"{statement} failed (max absolute residual {error:.6g}, "
            f"tolerance {atol:.1e})"
        )


@dataclass(frozen=True)
class LeakageInputProvenance:
    """Immutable identities of the policies and shared state set."""

    evaluated_arm: str
    oracle_arm: str
    evaluated_checkpoint_sha256: str
    oracle_checkpoint_sha256: str
    evaluated_task_index: int
    evaluated_embodiment_index: int
    oracle_task_index: int
    oracle_embodiment_index: int
    evaluation_plan_sha256: str
    source_state_manifest_sha256: str
    task: str
    family: str
    defect: str
    level: int
    training_seed: int
    time_smoothing_window: int
    release_status: str
    source_posterior_artifact_sha256: str | None = None
    surrogate_artifact_sha256: str | None = None
    excluded_surrogates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.evaluated_arm.strip():
            raise ValueError("evaluated_arm must be named")
        if self.oracle_arm != "B7-oracle":
            raise ValueError("the Lambda reference arm must be B7-oracle")
        for field in (
            "evaluated_checkpoint_sha256",
            "oracle_checkpoint_sha256",
            "evaluation_plan_sha256",
            "source_state_manifest_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        for field in (
            "evaluated_task_index",
            "evaluated_embodiment_index",
            "oracle_task_index",
            "oracle_embodiment_index",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        for field in ("task", "family", "defect"):
            if not getattr(self, field).strip():
                raise ValueError(f"{field} must be named")
        if isinstance(self.level, bool) or not isinstance(self.level, int) or self.level < 0:
            raise ValueError("level must be a non-negative integer")
        if isinstance(self.training_seed, bool) or not isinstance(self.training_seed, int):
            raise ValueError("training_seed must be an integer")
        if (
            isinstance(self.time_smoothing_window, bool)
            or not isinstance(self.time_smoothing_window, int)
            or self.time_smoothing_window < 3
            or self.time_smoothing_window % 2 != 1
        ):
            raise ValueError("time_smoothing_window must be an odd integer >= 3")
        if self.release_status not in {"formal", "engineering_only"}:
            raise ValueError("release_status must be formal or engineering_only")
        if self.source_posterior_artifact_sha256 is not None:
            _require_sha256(
                self.source_posterior_artifact_sha256,
                field="source_posterior_artifact_sha256",
            )
        if self.surrogate_artifact_sha256 is not None:
            _require_sha256(
                self.surrogate_artifact_sha256, field="surrogate_artifact_sha256"
            )
        if self.release_status == "formal" and (
            self.source_posterior_artifact_sha256 is None
            or self.surrogate_artifact_sha256 is None
        ):
            raise ValueError(
                "formal leakage provenance requires source-posterior and surrogate bindings"
            )
        if any(not name or "/" in name or "\\" in name for name in self.excluded_surrogates):
            raise ValueError("excluded surrogate names must be plain identifiers")

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluated_arm": self.evaluated_arm,
            "oracle_arm": self.oracle_arm,
            "evaluated_checkpoint_sha256": self.evaluated_checkpoint_sha256,
            "oracle_checkpoint_sha256": self.oracle_checkpoint_sha256,
            "evaluated_task_index": self.evaluated_task_index,
            "evaluated_embodiment_index": self.evaluated_embodiment_index,
            "oracle_task_index": self.oracle_task_index,
            "oracle_embodiment_index": self.oracle_embodiment_index,
            "evaluation_plan_sha256": self.evaluation_plan_sha256,
            "source_state_manifest_sha256": self.source_state_manifest_sha256,
            "task": self.task,
            "family": self.family,
            "defect": self.defect,
            "level": self.level,
            "training_seed": self.training_seed,
            "time_smoothing_window": self.time_smoothing_window,
            "release_status": self.release_status,
            "source_posterior_artifact_sha256": self.source_posterior_artifact_sha256,
            "surrogate_artifact_sha256": self.surrogate_artifact_sha256,
            "excluded_surrogates": list(self.excluded_surrogates),
        }


@dataclass(frozen=True)
class LeakageInputEpisode:
    """All action-valued fields for one shared-state episode."""

    episode_id: str
    evaluated_action: np.ndarray
    oracle_action: np.ndarray
    clean_teacher_action: np.ndarray
    defective_teacher_action: np.ndarray
    c_true: np.ndarray
    regime: np.ndarray
    c_true_clipped: np.ndarray
    shrinkage: np.ndarray
    time_smoothing: np.ndarray
    gain_error: np.ndarray
    surrogates: Mapping[str, np.ndarray]
    posterior_w: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.episode_id.strip():
            raise ValueError("episode_id must be named")
        evaluated = _action_field(self.evaluated_action, name="evaluated_action")
        shape = evaluated.shape
        oracle = _action_field(self.oracle_action, name="oracle_action", shape=shape)
        clean = _action_field(
            self.clean_teacher_action, name="clean_teacher_action", shape=shape
        )
        defective = _action_field(
            self.defective_teacher_action, name="defective_teacher_action", shape=shape
        )
        truth = _action_field(self.c_true, name="c_true", shape=shape)
        shrinkage = _action_field(self.shrinkage, name="shrinkage", shape=shape)
        smoothing = _action_field(
            self.time_smoothing, name="time_smoothing", shape=shape
        )
        gain = _action_field(self.gain_error, name="gain_error", shape=shape)
        del shrinkage, smoothing, gain

        regime = np.asarray(self.regime)
        clipped = np.asarray(self.c_true_clipped)
        if regime.shape != (shape[0],) or not np.issubdtype(regime.dtype, np.integer):
            raise ValueError("regime must be an integer vector with one row per frame")
        if np.any((regime < 0) | (regime > 4)):
            raise ValueError("regime ordinals must lie in [0, 4]")
        if clipped.shape != (shape[0],) or clipped.dtype != np.bool_:
            raise ValueError("c_true_clipped must be a bool vector with one row per frame")
        if not self.surrogates:
            raise ValueError("at least one descriptive surrogate field is required")
        for name, values in self.surrogates.items():
            if (
                not name.strip()
                or name.startswith("nu_surrogate:")
                or "/" in name
                or "\\" in name
            ):
                raise ValueError("surrogate names must be non-empty unprefixed identifiers")
            _action_field(values, name=f"surrogate {name}", shape=shape)
        if self.posterior_w is not None:
            posterior = np.asarray(self.posterior_w, dtype=np.float64)
            if posterior.shape != (shape[0],):
                raise ValueError(
                    f"posterior_w has shape {posterior.shape}, expected ({shape[0]},)"
                )
            if not np.isfinite(posterior).all() or np.any(
                (posterior < 0.0) | (posterior > 1.0)
            ):
                raise ValueError("posterior_w must contain finite probabilities in [0, 1]")

        _assert_identity(
            defective - clean,
            truth,
            statement="c_true == defective_teacher_action - clean_teacher_action",
            atol=_STORED_ACTION_IDENTITY_ATOL,
        )
        eps_orc = evaluated - oracle
        eps_ref = evaluated - clean
        u_orc = oracle - clean
        _assert_identity(
            eps_ref,
            eps_orc + u_orc,
            statement="eps_ref == eps_orc + u_orc",
        )

    @property
    def n_frames(self) -> int:
        return int(np.shape(self.evaluated_action)[0])

    @property
    def action_dim(self) -> int:
        return int(np.shape(self.evaluated_action)[1])

    @property
    def eps_orc(self) -> np.ndarray:
        return np.asarray(self.evaluated_action, dtype=np.float64) - np.asarray(
            self.oracle_action, dtype=np.float64
        )

    @property
    def eps_ref(self) -> np.ndarray:
        return np.asarray(self.evaluated_action, dtype=np.float64) - np.asarray(
            self.clean_teacher_action, dtype=np.float64
        )

    @property
    def u_orc(self) -> np.ndarray:
        return np.asarray(self.oracle_action, dtype=np.float64) - np.asarray(
            self.clean_teacher_action, dtype=np.float64
        )

    def as_chunk(self) -> LeakageChunk:
        return LeakageChunk(
            eps_orc=self.eps_orc,
            eps_ref=self.eps_ref,
            c_delta=np.asarray(self.c_true, dtype=np.float64),
            u_orc=self.u_orc,
            regime=np.asarray(self.regime, dtype=np.uint8),
            basis=InterferenceBasis(
                shrinkage=np.asarray(self.shrinkage, dtype=np.float64),
                time_smoothing=np.asarray(self.time_smoothing, dtype=np.float64),
                gain_error=np.asarray(self.gain_error, dtype=np.float64),
                surrogates={
                    name: np.asarray(values, dtype=np.float64)
                    for name, values in self.surrogates.items()
                },
            ),
        )


@dataclass(frozen=True)
class LeakageInputHeader:
    provenance: LeakageInputProvenance
    surrogate_names: tuple[str, ...]
    episode_ids: tuple[str, ...]
    n_frames: int
    action_dim: int
    artifact_sha256: str
    has_source_posterior: bool


def _update_digest_array(digest: Any, name: str, value: Any) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def _artifact_sha256(
    provenance: LeakageInputProvenance, episodes: Sequence[LeakageInputEpisode]
) -> str:
    digest = hashlib.sha256()
    digest.update(LEAKAGE_INPUT_SCHEMA.encode("ascii"))
    digest.update(_canonical_json(provenance.as_dict()).encode("ascii"))
    for episode in episodes:
        digest.update(episode.episode_id.encode("utf-8"))
        for name in (
            "evaluated_action",
            "oracle_action",
            "clean_teacher_action",
            "defective_teacher_action",
            "c_true",
            "regime",
            "c_true_clipped",
            "shrinkage",
            "time_smoothing",
            "gain_error",
        ):
            _update_digest_array(digest, name, getattr(episode, name))
        for name in sorted(episode.surrogates):
            _update_digest_array(digest, f"surrogate:{name}", episode.surrogates[name])
        if episode.posterior_w is not None:
            _update_digest_array(digest, "posterior_w", episode.posterior_w)
    return digest.hexdigest()


def write_leakage_input(
    path: Path | str,
    *,
    provenance: LeakageInputProvenance,
    episodes: Sequence[LeakageInputEpisode],
) -> LeakageInputHeader:
    """Write one immutable diagnostic input with an all-content SHA-256 binding."""
    import h5py

    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite leakage input {output}")
    rows = tuple(episodes)
    if not rows:
        raise ValueError("a leakage input needs at least one episode")
    episode_ids = tuple(row.episode_id for row in rows)
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("leakage input episode_id values must be unique")
    surrogate_names = tuple(sorted(rows[0].surrogates))
    if any(tuple(sorted(row.surrogates)) != surrogate_names for row in rows):
        raise ValueError("the surrogate set must be identical in every episode")
    posterior_presence = tuple(row.posterior_w is not None for row in rows)
    if any(posterior_presence) and not all(posterior_presence):
        raise ValueError("source posterior must be present for every episode or none")
    has_source_posterior = all(posterior_presence)
    if has_source_posterior != (
        provenance.source_posterior_artifact_sha256 is not None
    ):
        raise ValueError(
            "source-posterior rows and provenance artifact binding must be present together"
        )
    action_dim = rows[0].action_dim
    if any(row.action_dim != action_dim for row in rows):
        raise ValueError("action width changed between leakage input episodes")
    artifact_sha = _artifact_sha256(provenance, rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        with h5py.File(temporary, "w") as root:
            root.attrs["schema"] = LEAKAGE_INPUT_SCHEMA
            root.attrs["marker"] = LEAKAGE_INPUT_MARKER
            root.attrs["provenance_json"] = _canonical_json(provenance.as_dict())
            root.attrs["surrogate_names_json"] = _canonical_json(
                {"names": list(surrogate_names)}
            )
            root.attrs["artifact_sha256"] = artifact_sha
            root.attrs["has_source_posterior"] = has_source_posterior
            data = root.create_group("data")
            for index, episode in enumerate(rows):
                group = data.create_group(f"episode_{index:06d}")
                group.attrs["episode_id"] = episode.episode_id
                for name in (
                    "evaluated_action",
                    "oracle_action",
                    "clean_teacher_action",
                    "defective_teacher_action",
                    "c_true",
                    "regime",
                    "c_true_clipped",
                    "shrinkage",
                    "time_smoothing",
                    "gain_error",
                ):
                    group.create_dataset(name, data=getattr(episode, name), compression="gzip")
                surrogate_group = group.create_group("surrogates")
                for name in surrogate_names:
                    surrogate_group.create_dataset(
                        name, data=episode.surrogates[name], compression="gzip"
                    )
                if episode.posterior_w is not None:
                    group.create_dataset(
                        "posterior_w", data=episode.posterior_w, compression="gzip"
                    )
            root.flush()
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return LeakageInputHeader(
        provenance=provenance,
        surrogate_names=surrogate_names,
        episode_ids=episode_ids,
        n_frames=sum(row.n_frames for row in rows),
        action_dim=action_dim,
        artifact_sha256=artifact_sha,
        has_source_posterior=has_source_posterior,
    )


def _decode_attr(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _provenance_from_dict(value: Mapping[str, Any]) -> LeakageInputProvenance:
    expected = {
        "evaluated_arm",
        "oracle_arm",
        "evaluated_checkpoint_sha256",
        "oracle_checkpoint_sha256",
        "evaluated_task_index",
        "evaluated_embodiment_index",
        "oracle_task_index",
        "oracle_embodiment_index",
        "evaluation_plan_sha256",
        "source_state_manifest_sha256",
        "task",
        "family",
        "defect",
        "level",
        "training_seed",
        "time_smoothing_window",
        "release_status",
        "source_posterior_artifact_sha256",
        "surrogate_artifact_sha256",
        "excluded_surrogates",
    }
    if set(value) != expected:
        raise ValueError("leakage input provenance fields are incomplete or unexpected")
    return LeakageInputProvenance(
        **dict(value) | {"excluded_surrogates": tuple(value["excluded_surrogates"])}
    )


def _read_episodes(
    path: Path,
) -> tuple[LeakageInputProvenance, tuple[LeakageInputEpisode, ...], str]:
    import h5py

    with h5py.File(path, "r") as root:
        if _decode_attr(root.attrs.get("schema", "")) != LEAKAGE_INPUT_SCHEMA:
            raise ValueError(f"{path}: unsupported leakage input schema")
        if _decode_attr(root.attrs.get("marker", "")) != LEAKAGE_INPUT_MARKER:
            raise ValueError(f"{path}: leakage input marker is missing")
        try:
            provenance_raw = json.loads(_decode_attr(root.attrs["provenance_json"]))
            declared_surrogates = tuple(
                json.loads(_decode_attr(root.attrs["surrogate_names_json"]))["names"]
            )
            declared_sha = _require_sha256(
                _decode_attr(root.attrs["artifact_sha256"]), field="artifact_sha256"
            )
            groups = root["data"]
            has_source_posterior = bool(root.attrs["has_source_posterior"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}: leakage input header is incomplete") from exc
        provenance = _provenance_from_dict(provenance_raw)
        episodes: list[LeakageInputEpisode] = []
        for key in sorted(groups):
            group = groups[key]
            surrogate_group = group.get("surrogates")
            if surrogate_group is None:
                raise ValueError(f"{path}:{key}: surrogate group is missing")
            surrogates = {name: surrogate_group[name][()] for name in surrogate_group}
            if tuple(sorted(surrogates)) != declared_surrogates:
                raise ValueError(f"{path}:{key}: surrogate set disagrees with header")
            try:
                episodes.append(
                    LeakageInputEpisode(
                        episode_id=_decode_attr(group.attrs["episode_id"]),
                        evaluated_action=group["evaluated_action"][()],
                        oracle_action=group["oracle_action"][()],
                        clean_teacher_action=group["clean_teacher_action"][()],
                        defective_teacher_action=group["defective_teacher_action"][()],
                        c_true=group["c_true"][()],
                        regime=group["regime"][()],
                        c_true_clipped=group["c_true_clipped"][()],
                        shrinkage=group["shrinkage"][()],
                        time_smoothing=group["time_smoothing"][()],
                        gain_error=group["gain_error"][()],
                        surrogates=surrogates,
                        posterior_w=(
                            group["posterior_w"][()]
                            if has_source_posterior and "posterior_w" in group
                            else None
                        ),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{key}: required leakage field is missing") from exc
    rows = tuple(episodes)
    if not rows:
        raise ValueError(f"{path}: leakage input contains no episodes")
    posterior_presence = tuple(row.posterior_w is not None for row in rows)
    if any(posterior_presence) and not all(posterior_presence):
        raise ValueError(f"{path}: source posterior is missing from some episodes")
    observed_posterior = all(posterior_presence)
    if observed_posterior != has_source_posterior:
        raise ValueError(f"{path}: source-posterior header disagrees with episode data")
    if observed_posterior != (
        provenance.source_posterior_artifact_sha256 is not None
    ):
        raise ValueError(f"{path}: source-posterior provenance binding is inconsistent")
    observed_sha = _artifact_sha256(provenance, rows)
    if observed_sha != declared_sha:
        raise ValueError(
            f"{path}: leakage input content SHA-256 changed "
            f"({observed_sha} != {declared_sha})"
        )
    return provenance, rows, declared_sha


def load_leakage_input_header(path: Path | str) -> LeakageInputHeader:
    source = Path(path)
    provenance, rows, artifact_sha = _read_episodes(source)
    return LeakageInputHeader(
        provenance=provenance,
        surrogate_names=tuple(sorted(rows[0].surrogates)),
        episode_ids=tuple(row.episode_id for row in rows),
        n_frames=sum(row.n_frames for row in rows),
        action_dim=rows[0].action_dim,
        artifact_sha256=artifact_sha,
        has_source_posterior=rows[0].posterior_w is not None,
    )


def stream_leakage_input(path: Path | str) -> Iterator[tuple[LeakageInputEpisode, LeakageChunk]]:
    """Yield validated episodes and their exact ``LeakageAccumulator`` chunks."""
    _provenance, rows, _artifact_sha = _read_episodes(Path(path))
    for episode in rows:
        yield episode, episode.as_chunk()
