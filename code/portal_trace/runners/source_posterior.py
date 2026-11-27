"""Immutable per-frame source-posterior artifacts for Lambda decomposition.

The discriminator is trained separately from the policies.  Its output is
joined to the exact held-out ``rho_off`` episode rows only through this
hash-bound artifact, so ``w(x)`` cannot silently come from another split or
another classifier checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "SOURCE_POSTERIOR_SCHEMA",
    "SourcePosteriorProvenance",
    "SourcePosteriorEpisode",
    "SourcePosteriorHeader",
    "write_source_posterior",
    "load_source_posterior",
]

SOURCE_POSTERIOR_SCHEMA = "portal_trace.source_posterior/2"
_MARKER = "PORTAL_TRACE_HASH_BOUND_SOURCE_POSTERIOR"


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
class SourcePosteriorProvenance:
    plan_sha256: str
    model_sha256: str
    feature_spec_sha256: str
    training_split_sha256: str
    source_embodiment_indices: tuple[int, ...]
    target_embodiment_indices: tuple[int, ...]
    validation_metrics: Mapping[str, float]
    release_status: str

    def __post_init__(self) -> None:
        for field in (
            "plan_sha256", "model_sha256", "feature_spec_sha256",
            "training_split_sha256",
        ):
            _sha(getattr(self, field), field=field)
        source = tuple(self.source_embodiment_indices)
        target = tuple(self.target_embodiment_indices)
        if not source or not target:
            raise ValueError("source and target embodiment sets must both be non-empty")
        if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in source + target):
            raise ValueError("embodiment indices must be non-negative integers")
        if len(set(source)) != len(source) or len(set(target)) != len(target):
            raise ValueError("embodiment index sets must not contain duplicates")
        if set(source) & set(target):
            raise ValueError("source and target embodiment index sets must be disjoint")
        expected_metrics = {
            "auc",
            "ece",
            "alpha_recovery_abs_error",
            "validation_bce",
            "actual_source_fraction",
            "mean_posterior",
            "w_ece",
            "w_alpha_recovery",
            "w_support_gap",
            "calibration_off_support_fraction",
            "off_support_fraction",
            "in_support_fraction",
            "support_distance_threshold",
            "error_budget_total",
            "error_budget_limit",
        }
        informativeness_metrics = {
            "information_gain_bits",
            "information_gain_floor_bits",
        }
        robust_metrics = {
            "worst_pair_auc",
            "worst_pair_ece",
            "worst_pair_alpha_recovery_abs_error",
            "worst_pair_support_gap",
            "worst_pair_information_gain_bits",
            "validation_pair_count",
        }
        pooled_metrics = {
            "pooled_ece",
            "pooled_alpha_recovery_abs_error",
            "pooled_information_gain_bits",
            "pooled_w_support_gap",
        }
        optional_metrics = informativeness_metrics | robust_metrics | pooled_metrics
        observed_metrics = set(self.validation_metrics)
        if not expected_metrics <= observed_metrics or not observed_metrics <= (
            expected_metrics | optional_metrics
        ):
            raise ValueError("source-posterior validation metrics are incomplete")
        for name, value in self.validation_metrics.items():
            if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
                raise ValueError(f"source-posterior validation metric {name} is invalid")
        for name in (
            "auc", "ece", "alpha_recovery_abs_error",
            "actual_source_fraction", "mean_posterior",
            "w_ece", "w_alpha_recovery", "w_support_gap",
            "calibration_off_support_fraction", "off_support_fraction",
            "in_support_fraction", "error_budget_limit",
            "worst_pair_auc", "worst_pair_ece",
            "worst_pair_alpha_recovery_abs_error", "worst_pair_support_gap",
            "pooled_ece", "pooled_w_support_gap",
            "pooled_alpha_recovery_abs_error",
        ):
            if name not in self.validation_metrics:
                continue
            value = float(self.validation_metrics[name])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"source-posterior validation metric {name} is outside [0, 1]")
        if float(self.validation_metrics["validation_bce"]) < 0.0:
            raise ValueError("source-posterior validation BCE must be non-negative")
        if float(self.validation_metrics["support_distance_threshold"]) < 0.0:
            raise ValueError("source-posterior support threshold must be non-negative")
        if "information_gain_floor_bits" in self.validation_metrics and float(
            self.validation_metrics["information_gain_floor_bits"]
        ) <= 0.0:
            raise ValueError("source-posterior information-gain floor must be positive")
        if "validation_pair_count" in self.validation_metrics and (
            float(self.validation_metrics["validation_pair_count"]) < 1.0
            or not float(self.validation_metrics["validation_pair_count"]).is_integer()
        ):
            raise ValueError("source-posterior validation pair count is invalid")
        budget_total = float(self.validation_metrics["error_budget_total"])
        reconstructed_budget = sum(
            float(self.validation_metrics[name])
            for name in ("w_ece", "w_alpha_recovery", "w_support_gap")
        )
        if not math.isclose(budget_total, reconstructed_budget, abs_tol=1.0e-12):
            raise ValueError("source-posterior error budget does not equal its three terms")
        if self.release_status == "formal" and budget_total > float(
            self.validation_metrics["error_budget_limit"]
        ):
            raise ValueError("formal source posterior exceeds its discriminator error budget")
        if self.release_status == "formal":
            required_formal_metrics = informativeness_metrics | robust_metrics
            if not required_formal_metrics <= observed_metrics:
                raise ValueError(
                    "formal source posterior lacks pair-robust validation metrics"
                )
            if float(self.validation_metrics["information_gain_bits"]) < float(
                self.validation_metrics["information_gain_floor_bits"]
            ):
                raise ValueError("formal source posterior is not informative")
            if float(self.validation_metrics["worst_pair_auc"]) < 0.5:
                raise ValueError("formal source posterior has an orientation-failed pair")
        if float(self.validation_metrics["auc"]) < 0.5:
            raise ValueError(
                "source-posterior AUC is below 0.5; orientation-failed posteriors "
                "cannot enter the Lambda decomposition"
            )
        if self.release_status not in {"formal", "engineering_only"}:
            raise ValueError("release_status must be formal or engineering_only")

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_sha256": self.plan_sha256,
            "model_sha256": self.model_sha256,
            "feature_spec_sha256": self.feature_spec_sha256,
            "training_split_sha256": self.training_split_sha256,
            "source_embodiment_indices": list(self.source_embodiment_indices),
            "target_embodiment_indices": list(self.target_embodiment_indices),
            "validation_metrics": {
                name: float(value) for name, value in self.validation_metrics.items()
            },
            "release_status": self.release_status,
        }


@dataclass(frozen=True)
class SourcePosteriorEpisode:
    episode_id: str
    posterior_w: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("episode_id must be named")
        values = np.asarray(self.posterior_w, dtype=np.float64)
        if values.ndim != 1 or values.size < 1:
            raise ValueError("posterior_w must be a non-empty vector")
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("posterior_w must contain finite probabilities in [0, 1]")


@dataclass(frozen=True)
class SourcePosteriorHeader:
    provenance: SourcePosteriorProvenance
    episode_ids: tuple[str, ...]
    n_frames: int
    artifact_sha256: str


def _artifact_sha256(
    provenance: SourcePosteriorProvenance,
    episodes: Sequence[SourcePosteriorEpisode],
) -> str:
    digest = hashlib.sha256()
    digest.update(SOURCE_POSTERIOR_SCHEMA.encode("ascii"))
    digest.update(_canonical_json(provenance.as_dict()).encode("ascii"))
    for episode in episodes:
        values = np.ascontiguousarray(episode.posterior_w, dtype=np.float64)
        digest.update(episode.episode_id.encode("utf-8"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def write_source_posterior(
    path: Path | str,
    *,
    provenance: SourcePosteriorProvenance,
    episodes: Sequence[SourcePosteriorEpisode],
) -> SourcePosteriorHeader:
    import h5py

    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite source posterior {output}")
    rows = tuple(episodes)
    if not rows:
        raise ValueError("a source-posterior artifact needs at least one episode")
    episode_ids = tuple(row.episode_id for row in rows)
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("source-posterior episode IDs must be unique")
    artifact_sha = _artifact_sha256(provenance, rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        with h5py.File(temporary, "w") as root:
            root.attrs["schema"] = SOURCE_POSTERIOR_SCHEMA
            root.attrs["marker"] = _MARKER
            root.attrs["provenance_json"] = _canonical_json(provenance.as_dict())
            root.attrs["artifact_sha256"] = artifact_sha
            data = root.create_group("data")
            for index, episode in enumerate(rows):
                group = data.create_group(f"episode_{index:06d}")
                group.attrs["episode_id"] = episode.episode_id
                group.create_dataset(
                    "posterior_w",
                    data=np.asarray(episode.posterior_w, dtype=np.float64),
                    compression="gzip",
                )
            root.flush()
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return SourcePosteriorHeader(
        provenance=provenance,
        episode_ids=episode_ids,
        n_frames=sum(len(row.posterior_w) for row in rows),
        artifact_sha256=artifact_sha,
    )


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_source_posterior(
    path: Path | str,
) -> tuple[SourcePosteriorHeader, dict[str, np.ndarray]]:
    import h5py

    source = Path(path)
    with h5py.File(source, "r") as root:
        if _decode(root.attrs.get("schema", "")) != SOURCE_POSTERIOR_SCHEMA:
            raise ValueError(f"{source}: unsupported source-posterior schema")
        if _decode(root.attrs.get("marker", "")) != _MARKER:
            raise ValueError(f"{source}: source-posterior marker is missing")
        try:
            raw = json.loads(_decode(root.attrs["provenance_json"]))
            declared_sha = _sha(
                _decode(root.attrs["artifact_sha256"]), field="artifact_sha256"
            )
            groups = root["data"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{source}: source-posterior header is incomplete") from exc
        expected = {
            "plan_sha256", "model_sha256", "feature_spec_sha256",
            "training_split_sha256", "source_embodiment_indices",
            "target_embodiment_indices", "release_status",
            "validation_metrics",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError(f"{source}: source-posterior provenance is incomplete")
        provenance = SourcePosteriorProvenance(
            **raw
            | {
                "source_embodiment_indices": tuple(raw["source_embodiment_indices"]),
                "target_embodiment_indices": tuple(raw["target_embodiment_indices"]),
            }
        )
        rows: list[SourcePosteriorEpisode] = []
        for name in sorted(groups):
            group = groups[name]
            try:
                rows.append(
                    SourcePosteriorEpisode(
                        episode_id=_decode(group.attrs["episode_id"]),
                        posterior_w=group["posterior_w"][()],
                    )
                )
            except KeyError as exc:
                raise ValueError(f"{source}:{name}: posterior fields are incomplete") from exc
    if not rows:
        raise ValueError(f"{source}: source-posterior artifact contains no episodes")
    if len({row.episode_id for row in rows}) != len(rows):
        raise ValueError(f"{source}: source-posterior episode IDs are not unique")
    observed_sha = _artifact_sha256(provenance, rows)
    if observed_sha != declared_sha:
        raise ValueError(f"{source}: source-posterior content SHA-256 changed")
    values = {
        row.episode_id: np.asarray(row.posterior_w, dtype=np.float64) for row in rows
    }
    return (
        SourcePosteriorHeader(
            provenance=provenance,
            episode_ids=tuple(row.episode_id for row in rows),
            n_frames=sum(len(row.posterior_w) for row in rows),
            artifact_sha256=declared_sha,
        ),
        values,
    )
