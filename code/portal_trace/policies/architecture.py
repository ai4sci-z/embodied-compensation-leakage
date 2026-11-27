"""Training-architecture boundary for the custom PORTAL-Trace policies.

The policy modules in this package implement a ResNet/Transformer model with a
different observation, history, chunking and checkpoint contract from stock
robomimic BC. This module canonicalises the human-readable arm declarations and
validates a pool without routing the custom policy through an incompatible
robomimic algorithm. It also owns a deliberately tiny, explicitly marked smoke
path that exercises real policy windows, training-split normalization, optimizer
steps and checkpoint round trips without claiming to train a paper architecture.

The historical ``run_training`` adapter still raises a named error because its
signature cannot carry a formal plan or normalization artifact. The formal
runner uses ``runners.formal_training`` directly; smoke checkpoints remain
inadmissible to formal training or evaluation.
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
from torch import nn

from portal_trace.policies.interface import Batch, Observation, PolicyOutput
from portal_trace.policies.losses import chunk_bc_loss
from portal_trace.policies.obs import (
    ACTION_DIM,
    CAMERAS,
    CHUNK,
    IMAGE_SIZE,
    K_HIST_MAIN,
    PROPRIO_DIM,
    PROPRIO_GROUPS,
    NormKind,
)
from portal_trace.runners.policy_dataset import (
    PolicyDatasetEntry,
    RawPolicyWindowDataset,
    collate_policy_samples,
)

__all__ = [
    "ArchitectureUnavailable",
    "FrozenProprioNormalization",
    "LoadedSmokeCheckpoint",
    "SMOKE_CHECKPOINT_SCHEMA",
    "SMOKE_ONLY_MARKER",
    "SmokeModelConfig",
    "SmokeSplit",
    "SmokeTrainingConfig",
    "SmokeTrainingResult",
    "TrainingConfig",
    "canonical_architecture",
    "build_robomimic_config",
    "fit_smoke_normalization",
    "load_smoke_checkpoint",
    "run_smoke_training",
    "run_training",
]


class ArchitectureUnavailable(RuntimeError):
    """The custom policy-to-checkpoint training adapter is not complete."""


_DEFAULT_DESCRIPTION = "resnet18_transformer4"
SMOKE_CHECKPOINT_SCHEMA = "portal_trace.smoke_checkpoint/1"
SMOKE_ONLY_MARKER = "PORTAL_TRACE_SMOKE_ONLY_NOT_FORMAL"
_SMOKE_MODEL_KIND = "tiny_causal_policy_v1"
_MAX_SMOKE_STEPS = 4


def canonical_architecture(value: str | None) -> str:
    """Map config prose to the stable architecture key used by lock files."""
    if value is None or not str(value).strip():
        return _DEFAULT_DESCRIPTION
    text = " ".join(str(value).strip().lower().split())
    if text in {_DEFAULT_DESCRIPTION, "resnet18_transformer4"}:
        return _DEFAULT_DESCRIPTION
    if text == "resnet-18 vision tower x2 + 4-layer transformer decoder head":
        return _DEFAULT_DESCRIPTION
    if text == (
        "resnet-18 vision tower x2 + 4-layer transformer decoder head, no proprioception pathway"
    ):
        return "resnet18_transformer4_no_proprio"
    if text == "no vision tower + 4-layer transformer decoder head over 70-dim proprioception":
        return "proprio_transformer4"
    for k_hist in (1, 2, 5, 10):
        if text == (
            f"resnet-18 vision tower x2 + 4-layer transformer decoder head, k_hist = {k_hist}"
        ):
            return f"resnet18_transformer4_khist{k_hist}"
    if text == "resnet-34 vision tower x2 + 4-layer transformer decoder head":
        return "resnet34_transformer4"
    if text == "vit-s/12 vision tower x2 + 6-layer transformer decoder head":
        return "vits_transformer6"
    raise ValueError(
        f"unknown policy architecture declaration {value!r}; use a canonical key "
        "or one of the declared arm descriptions"
    )


@dataclass(frozen=True)
class TrainingConfig:
    """Validated inputs for the future custom training loop."""

    architecture: str
    pool: tuple[tuple[Path, int], ...]
    seed: int
    output_dir: Path

    @property
    def n_demos(self) -> int:
        return sum(n for _, n in self.pool)


@dataclass(frozen=True)
class SmokeSplit:
    """Explicit window indices used by the smoke-only optimizer."""

    training_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.training_indices:
            raise ValueError("smoke training requires an explicit non-empty split")
        if any(
            isinstance(index, bool) or not isinstance(index, int) for index in self.training_indices
        ):
            raise TypeError("smoke training indices must be integers")
        if any(index < 0 for index in self.training_indices):
            raise ValueError("smoke training indices must be non-negative")
        if len(set(self.training_indices)) != len(self.training_indices):
            raise ValueError("smoke training indices must be unique")


@dataclass(frozen=True)
class SmokeModelConfig:
    """Small model used only to exercise the adapter, never a paper arm."""

    kind: str = _SMOKE_MODEL_KIND
    hidden_dim: int = 16

    def __post_init__(self) -> None:
        if self.kind != _SMOKE_MODEL_KIND:
            raise ValueError(f"smoke model kind must be {_SMOKE_MODEL_KIND!r}")
        if isinstance(self.hidden_dim, bool) or not isinstance(self.hidden_dim, int):
            raise TypeError("smoke hidden_dim must be an integer")
        if self.hidden_dim < 4 or self.hidden_dim > 256:
            raise ValueError("smoke hidden_dim must lie in [4, 256]")


@dataclass(frozen=True)
class SmokeTrainingConfig:
    """Opt-in contract for a bounded, explicitly non-formal training probe."""

    schema: str
    marker: str
    entries: tuple[PolicyDatasetEntry, ...]
    split: SmokeSplit
    seed: int
    checkpoint_path: Path
    optimizer_steps: int = 1
    batch_size: int = 1
    learning_rate: float = 1e-3
    model: SmokeModelConfig = SmokeModelConfig()

    def __post_init__(self) -> None:
        if self.schema != SMOKE_CHECKPOINT_SCHEMA:
            raise ValueError(f"smoke schema must be {SMOKE_CHECKPOINT_SCHEMA!r}")
        if self.marker != SMOKE_ONLY_MARKER:
            raise ValueError(f"smoke marker must be {SMOKE_ONLY_MARKER!r}")
        if not self.entries or any(
            not isinstance(entry, PolicyDatasetEntry) for entry in self.entries
        ):
            raise ValueError("smoke training requires explicit PolicyDatasetEntry values")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("smoke seed must be an integer")
        if (
            isinstance(self.optimizer_steps, bool)
            or not isinstance(self.optimizer_steps, int)
            or self.optimizer_steps not in range(1, _MAX_SMOKE_STEPS + 1)
        ):
            raise ValueError(f"smoke optimizer_steps must lie in [1, {_MAX_SMOKE_STEPS}]")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise ValueError("smoke batch_size must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("smoke learning_rate must be finite and positive")
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))


@dataclass(frozen=True)
class FrozenProprioNormalization:
    """Immutable, training-split-bound statistics for the 70 input channels."""

    means: tuple[float, ...]
    scales: tuple[float, ...]
    training_indices: tuple[int, ...]
    fitted_rows: int

    def __post_init__(self) -> None:
        if len(self.means) != PROPRIO_DIM or len(self.scales) != PROPRIO_DIM:
            raise ValueError(f"normalization must describe exactly {PROPRIO_DIM} channels")
        if not all(math.isfinite(value) for value in (*self.means, *self.scales)):
            raise ValueError("normalization contains a non-finite statistic")
        if any(scale <= 0.0 for scale in self.scales):
            raise ValueError("normalization scales must be positive")
        SmokeSplit(self.training_indices)
        if self.fitted_rows < 1:
            raise ValueError("normalization must be fitted on at least one row")

    def apply(self, proprio: torch.Tensor) -> torch.Tensor:
        if proprio.ndim != 3 or proprio.shape[-1] != PROPRIO_DIM:
            raise ValueError(
                f"smoke normalization expects (B, K, {PROPRIO_DIM}), got {tuple(proprio.shape)}"
            )
        if not torch.isfinite(proprio).all():
            raise ValueError("smoke proprioception contains non-finite values")
        mean = torch.tensor(self.means, dtype=proprio.dtype, device=proprio.device)
        scale = torch.tensor(self.scales, dtype=proprio.dtype, device=proprio.device)
        normalized = (proprio - mean) / scale
        for group in PROPRIO_GROUPS:
            if group.norm is NormKind.UNIT_NORM:
                values = proprio[..., group.slice]
                denominator = torch.linalg.vector_norm(values, dim=-1, keepdim=True).clamp_min(
                    1e-12
                )
                normalized[..., group.slice] = values / denominator
            elif group.norm is NormKind.NONE:
                normalized[..., group.slice] = proprio[..., group.slice]
        if not torch.isfinite(normalized).all():
            raise ValueError("smoke normalization produced non-finite values")
        return normalized

    def as_payload(self) -> dict[str, Any]:
        return {
            "means": list(self.means),
            "scales": list(self.scales),
            "training_indices": list(self.training_indices),
            "fitted_rows": self.fitted_rows,
        }


@dataclass(frozen=True)
class SmokeTrainingResult:
    checkpoint_path: Path
    final_loss: float
    optimizer_steps: int
    parameter_delta_l2: float
    reference_output: torch.Tensor


@dataclass(frozen=True)
class LoadedSmokeCheckpoint:
    """Validated smoke model plus its frozen input transform."""

    model: nn.Module
    normalization: FrozenProprioNormalization
    split: SmokeSplit
    dataset_entries: tuple[PolicyDatasetEntry, ...]
    seed: int
    optimizer_steps: int

    @torch.no_grad()
    def inference(self, batch: Batch) -> torch.Tensor:
        inference_batch = _normalise_batch(batch, self.normalization, labels=False)
        self.model.eval()
        output = self.model(inference_batch).action_chunk
        expected = (inference_batch.obs.batch_size, CHUNK, ACTION_DIM)
        if tuple(output.shape) != expected:
            raise ValueError(f"smoke inference produced {tuple(output.shape)}, expected {expected}")
        if not torch.isfinite(output).all():
            raise ValueError("smoke inference produced non-finite actions")
        return output.detach().cpu()


class _TinySmokePolicy(nn.Module):
    """Contract probe over the real Batch layout, intentionally not a paper model."""

    def __init__(self, *, hidden_dim: int, n_tasks: int) -> None:
        super().__init__()
        if n_tasks < 1:
            raise ValueError("smoke model requires at least one task")
        self.hidden_dim = hidden_dim
        self.n_tasks = n_tasks
        self.network = nn.Sequential(
            nn.Linear(PROPRIO_DIM + len(CAMERAS) + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, CHUNK * ACTION_DIM),
        )

    def forward(self, batch: Batch) -> PolicyOutput:
        obs = batch.obs
        if obs.proprio is None:
            raise ValueError("smoke policy requires normalized proprioception")
        if obs.proprio.ndim != 3 or obs.proprio.shape[1:] != (K_HIST_MAIN, PROPRIO_DIM):
            raise ValueError(
                f"smoke policy requires proprioception shaped (B, {K_HIST_MAIN}, {PROPRIO_DIM})"
            )
        image_means: list[torch.Tensor] = []
        for camera in CAMERAS:
            if camera not in obs.images:
                raise ValueError(f"smoke policy is missing camera {camera!r}")
            image = obs.images[camera]
            expected = (obs.batch_size, K_HIST_MAIN, 3, IMAGE_SIZE, IMAGE_SIZE)
            if tuple(image.shape) != expected:
                raise ValueError(
                    f"{camera}: smoke image shape {tuple(image.shape)}, expected {expected}"
                )
            if not torch.isfinite(image).all():
                raise ValueError(f"{camera}: smoke image contains non-finite values")
            image_means.append(image.mean(dim=(1, 2, 3, 4)))
        if obs.task_index is None or obs.task_index.shape != (obs.batch_size,):
            raise ValueError("smoke policy requires one task index per sample")
        if torch.any(obs.task_index < 0) or torch.any(obs.task_index >= self.n_tasks):
            raise ValueError("smoke task index is outside the configured task range")
        task_scale = float(max(1, self.n_tasks - 1))
        features = torch.cat(
            (
                obs.proprio.mean(dim=1),
                torch.stack(image_means, dim=-1),
                obs.task_index.to(obs.proprio.dtype).unsqueeze(-1) / task_scale,
            ),
            dim=-1,
        )
        prediction = self.network(features).reshape(-1, CHUNK, ACTION_DIM)
        if batch.actions is None:
            return PolicyOutput(action_chunk=prediction)
        loss = chunk_bc_loss(
            prediction,
            batch.actions,
            pad_mask=batch.action_pad_mask,
            sample_weight=batch.sample_weight,
        )
        return PolicyOutput(
            action_chunk=prediction,
            loss=loss,
            aux_losses={"bc": loss.detach()},
        )


def _validate_split(dataset: RawPolicyWindowDataset, split: SmokeSplit) -> None:
    invalid = [index for index in split.training_indices if index >= len(dataset)]
    if invalid:
        raise ValueError(
            f"smoke training split indices {invalid} exceed dataset length {len(dataset)}"
        )


def fit_smoke_normalization(
    dataset: RawPolicyWindowDataset,
    split: SmokeSplit,
) -> FrozenProprioNormalization:
    """Fit group-aware proprio statistics on exactly the declared smoke split."""
    _validate_split(dataset, split)
    rows = torch.cat(
        [dataset[index]["proprio"].to(torch.float64) for index in split.training_indices],
        dim=0,
    )
    if rows.ndim != 2 or rows.shape[1] != PROPRIO_DIM or not torch.isfinite(rows).all():
        raise ValueError("smoke training split has invalid proprioception")
    means = torch.zeros(PROPRIO_DIM, dtype=torch.float64)
    scales = torch.ones(PROPRIO_DIM, dtype=torch.float64)
    for group in PROPRIO_GROUPS:
        values = rows[:, group.slice]
        if group.norm in {NormKind.STANDARDISE, NormKind.ZERO_MEAN}:
            means[group.slice] = values.mean(dim=0)
        if group.norm is NormKind.STANDARDISE:
            scales[group.slice] = values.std(dim=0, correction=0).clamp_min(1e-3)
    return FrozenProprioNormalization(
        means=tuple(float(value) for value in means.tolist()),
        scales=tuple(float(value) for value in scales.tolist()),
        training_indices=split.training_indices,
        fitted_rows=int(rows.shape[0]),
    )


def _normalise_batch(
    batch: Batch,
    normalization: FrozenProprioNormalization,
    *,
    labels: bool,
) -> Batch:
    if batch.obs.proprio is None:
        raise ValueError("smoke adapter requires proprioception")
    obs = Observation(
        images=batch.obs.images,
        proprio=normalization.apply(batch.obs.proprio),
        task_index=batch.obs.task_index,
        embodiment_index=batch.obs.embodiment_index,
        partition_index=batch.obs.partition_index,
    )
    if not labels:
        return Batch(obs=obs)
    return Batch(
        obs=obs,
        actions=batch.actions,
        action_pad_mask=batch.action_pad_mask,
        c_true=batch.c_true,
        sample_weight=batch.sample_weight,
        phi_features=batch.phi_features,
        psi_basis=batch.psi_basis,
        extras=batch.extras,
    )


def _entry_payload(entry: PolicyDatasetEntry) -> dict[str, Any]:
    return {
        "path": str(Path(entry.path).resolve()),
        "sha256": _file_sha256(Path(entry.path)),
        "task_index": entry.task_index,
        "embodiment_index": entry.embodiment_index,
        "n_demos": entry.n_demos,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_content_sha256(payload: dict[str, Any]) -> str:
    """Hash metadata and tensors independently of torch's container bytes."""
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("smoke checkpoint model_state is invalid")
    metadata = {
        key: value for key, value in payload.items() if key not in {"model_state", "content_sha256"}
    }
    digest = hashlib.sha256(
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("smoke checkpoint model_state is invalid")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _atomic_publish_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite smoke checkpoint {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite smoke checkpoint {path}") from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _parameter_delta_l2(before: dict[str, torch.Tensor], model: nn.Module) -> float:
    squared = 0.0
    for name, parameter in model.named_parameters():
        difference = parameter.detach().cpu().to(torch.float64) - before[name]
        squared += float(difference.square().sum())
    return math.sqrt(squared)


def run_smoke_training(config: SmokeTrainingConfig) -> SmokeTrainingResult:
    """Run a bounded adapter probe and publish a smoke-marked checkpoint."""
    dataset = RawPolicyWindowDataset(config.entries)
    _validate_split(dataset, config.split)
    if config.batch_size > len(config.split.training_indices):
        raise ValueError("smoke batch_size cannot exceed the explicit training split")
    normalization = fit_smoke_normalization(dataset, config.split)
    n_tasks = max(entry.task_index for entry in config.entries) + 1
    reference_raw = collate_policy_samples([dataset[config.split.training_indices[0]]])

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = _TinySmokePolicy(hidden_dim=config.model.hidden_dim, n_tasks=n_tasks)
        before = {
            name: parameter.detach().cpu().to(torch.float64).clone()
            for name, parameter in model.named_parameters()
        }
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        model.train()
        final_loss = float("nan")
        indices = config.split.training_indices
        for step in range(config.optimizer_steps):
            start = (step * config.batch_size) % len(indices)
            chosen = tuple(
                indices[(start + offset) % len(indices)] for offset in range(config.batch_size)
            )
            raw_batch = collate_policy_samples([dataset[index] for index in chosen])
            batch = _normalise_batch(raw_batch, normalization, labels=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            if output.loss is None or not torch.isfinite(output.loss):
                raise RuntimeError("smoke optimizer received a missing or non-finite loss")
            output.loss.backward()
            optimizer.step()
            final_loss = float(output.loss.detach().cpu())

        parameter_delta = _parameter_delta_l2(before, model)
        if not math.isfinite(parameter_delta) or parameter_delta <= 0.0:
            raise RuntimeError("smoke optimizer did not update any model parameter")
        model.eval()
        reference_output = (
            model(_normalise_batch(reference_raw, normalization, labels=False))
            .action_chunk.detach()
            .cpu()
        )

        payload: dict[str, Any] = {
            "schema": SMOKE_CHECKPOINT_SCHEMA,
            "marker": SMOKE_ONLY_MARKER,
            "seed": config.seed,
            "optimizer_steps": config.optimizer_steps,
            "split": {"training_indices": list(config.split.training_indices)},
            "dataset_entries": [_entry_payload(entry) for entry in config.entries],
            "model": {
                "kind": config.model.kind,
                "hidden_dim": config.model.hidden_dim,
                "n_tasks": n_tasks,
            },
            "normalization": normalization.as_payload(),
            "model_state": model.state_dict(),
        }
        payload["content_sha256"] = _checkpoint_content_sha256(payload)
        _atomic_publish_checkpoint(config.checkpoint_path, payload)

        loaded = load_smoke_checkpoint(config.checkpoint_path)
        reloaded_output = loaded.inference(reference_raw)
        if not torch.allclose(reference_output, reloaded_output, rtol=0.0, atol=1e-7):
            raise RuntimeError("smoke checkpoint changed inference across save/load")
    return SmokeTrainingResult(
        checkpoint_path=config.checkpoint_path,
        final_loss=final_loss,
        optimizer_steps=config.optimizer_steps,
        parameter_delta_l2=parameter_delta,
        reference_output=reference_output,
    )


def _require_exact_keys(value: Any, expected: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    if set(value) != expected:
        raise ValueError(
            f"{where} fields mismatch; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )
    return value


def load_smoke_checkpoint(path: Path) -> LoadedSmokeCheckpoint:
    """Strictly load only the dedicated smoke schema and validate every binding."""
    path = Path(path)
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"{path}: unreadable smoke checkpoint") from exc
    document = _require_exact_keys(
        raw,
        {
            "schema",
            "marker",
            "seed",
            "optimizer_steps",
            "split",
            "dataset_entries",
            "model",
            "normalization",
            "model_state",
            "content_sha256",
        },
        where="smoke checkpoint",
    )
    if document["schema"] != SMOKE_CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint is not the smoke-only schema")
    if document["marker"] != SMOKE_ONLY_MARKER:
        raise ValueError("checkpoint lacks the required smoke-only marker")

    split_payload = _require_exact_keys(
        document["split"], {"training_indices"}, where="smoke checkpoint split"
    )
    raw_indices = split_payload["training_indices"]
    if not isinstance(raw_indices, list):
        raise ValueError("smoke checkpoint training_indices must be a list")
    split = SmokeSplit(tuple(raw_indices))

    entries = document["dataset_entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("smoke checkpoint must bind non-empty dataset entries")
    parsed_entries: list[PolicyDatasetEntry] = []
    for index, entry in enumerate(entries):
        item = _require_exact_keys(
            entry,
            {"path", "sha256", "task_index", "embodiment_index", "n_demos"},
            where=f"smoke checkpoint dataset_entries[{index}]",
        )
        if not isinstance(item["path"], str) or not item["path"]:
            raise ValueError("smoke checkpoint dataset path must be non-empty")
        for name in ("task_index", "embodiment_index", "n_demos"):
            if isinstance(item[name], bool) or not isinstance(item[name], int):
                raise ValueError(f"smoke checkpoint dataset {name} must be an integer")
        if item["task_index"] < 0 or item["embodiment_index"] < 0 or item["n_demos"] < 1:
            raise ValueError("smoke checkpoint contains an invalid dataset entry")
        if (
            not isinstance(item["sha256"], str)
            or len(item["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in item["sha256"])
        ):
            raise ValueError("smoke checkpoint dataset SHA-256 is invalid")
        dataset_path = Path(item["path"])
        if not dataset_path.is_file():
            raise ValueError(f"smoke checkpoint dataset does not exist: {dataset_path}")
        if _file_sha256(dataset_path) != item["sha256"]:
            raise ValueError(f"smoke checkpoint dataset SHA-256 does not match: {dataset_path}")
        parsed_entries.append(
            PolicyDatasetEntry(
                dataset_path,
                task_index=item["task_index"],
                embodiment_index=item["embodiment_index"],
                n_demos=item["n_demos"],
            )
        )

    model_payload = _require_exact_keys(
        document["model"], {"kind", "hidden_dim", "n_tasks"}, where="smoke checkpoint model"
    )
    model_config = SmokeModelConfig(
        kind=model_payload["kind"], hidden_dim=model_payload["hidden_dim"]
    )
    n_tasks = model_payload["n_tasks"]
    if isinstance(n_tasks, bool) or not isinstance(n_tasks, int) or n_tasks < 1:
        raise ValueError("smoke checkpoint n_tasks must be a positive integer")

    normalization_payload = _require_exact_keys(
        document["normalization"],
        {"means", "scales", "training_indices", "fitted_rows"},
        where="smoke checkpoint normalization",
    )
    means = normalization_payload["means"]
    scales = normalization_payload["scales"]
    normalization_indices = normalization_payload["training_indices"]
    if not isinstance(means, list) or not isinstance(scales, list):
        raise ValueError("smoke checkpoint normalization statistics must be lists")
    if not isinstance(normalization_indices, list):
        raise ValueError("smoke checkpoint normalization split must be a list")
    normalization = FrozenProprioNormalization(
        means=tuple(means),
        scales=tuple(scales),
        training_indices=tuple(normalization_indices),
        fitted_rows=normalization_payload["fitted_rows"],
    )
    if normalization.training_indices != split.training_indices:
        raise ValueError("smoke checkpoint normalization is not bound to its training split")

    seed = document["seed"]
    optimizer_steps = document["optimizer_steps"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("smoke checkpoint seed must be an integer")
    if (
        isinstance(optimizer_steps, bool)
        or not isinstance(optimizer_steps, int)
        or optimizer_steps not in range(1, _MAX_SMOKE_STEPS + 1)
    ):
        raise ValueError("smoke checkpoint has an invalid optimizer step count")

    state = document["model_state"]
    if (
        not isinstance(state, dict)
        or not state
        or any(
            not isinstance(name, str) or not isinstance(value, torch.Tensor)
            for name, value in state.items()
        )
    ):
        raise ValueError("smoke checkpoint model_state is invalid")
    if any(
        (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all()
        for value in state.values()
    ):
        raise ValueError("smoke checkpoint model_state contains non-finite values")
    content_sha256 = document["content_sha256"]
    if (
        not isinstance(content_sha256, str)
        or len(content_sha256) != 64
        or any(char not in "0123456789abcdef" for char in content_sha256)
        or _checkpoint_content_sha256(document) != content_sha256
    ):
        raise ValueError("smoke checkpoint content SHA-256 does not match")
    model = _TinySmokePolicy(hidden_dim=model_config.hidden_dim, n_tasks=n_tasks)
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("smoke checkpoint model_state does not match its model config") from exc
    model.eval()
    return LoadedSmokeCheckpoint(
        model=model,
        normalization=normalization,
        split=split,
        dataset_entries=tuple(parsed_entries),
        seed=seed,
        optimizer_steps=optimizer_steps,
    )


def build_robomimic_config(
    *,
    architecture: str,
    pool: list[tuple[Path, int]],
    seed: int,
    output_dir: Path,
) -> TrainingConfig:
    """Validate a pool and return an explicit adapter config.

    The historical function name is kept for the runner API, but the returned
    object is not a stock robomimic config. That distinction prevents accidental
    use of robomimic's single-frame BC code for this package's 5-frame, 8-step
    custom policy.
    """
    canonical = canonical_architecture(architecture)
    if canonical != _DEFAULT_DESCRIPTION:
        raise ArchitectureUnavailable(
            f"architecture {canonical!r} is declared but its custom dataset/checkpoint "
            "adapter is not implemented yet; no stock robomimic substitution is allowed"
        )
    entries: list[tuple[Path, int]] = []
    for path, n_demos in pool:
        path = Path(path)
        if n_demos < 1:
            raise ValueError(f"training pool entry {path} has n_demos={n_demos}")
        if not path.exists():
            raise FileNotFoundError(
                f"training pool file {path} is missing; materialise the collection "
                "before constructing a checkpoint job"
            )
        import h5py

        try:
            with h5py.File(path, "r") as dataset:
                actual_demos = len(dataset["data"])
        except (OSError, KeyError) as exc:
            raise ValueError(
                f"training pool file {path} is not a readable materialised dataset"
            ) from exc
        if actual_demos < n_demos:
            raise ValueError(
                f"training pool entry {path} declares {n_demos} demonstrations but "
                f"the materialised file contains only {actual_demos}; refusing to "
                "inflate its sampling weight"
            )
        entries.append((path, int(n_demos)))
    if not entries:
        raise ValueError("training pool is empty")
    return TrainingConfig(
        architecture=canonical,
        pool=tuple(entries),
        seed=int(seed),
        output_dir=Path(output_dir),
    )


def run_training(config: TrainingConfig) -> None:
    """Refuse formal training until the custom adapter and checkpoint are wired."""
    del config
    raise ArchitectureUnavailable(
        "formal training is intentionally blocked: the PORTAL-Trace adapter still "
        "needs the declared ResNet/Transformer optimizer, formal pool-wide split and "
        "normalization provenance, B8 c_true transforms, and a gated formal "
        "checkpoint publication/evaluation path. The engineering producer and "
        "tiny smoke adapter cannot authorize paper runs"
    )
