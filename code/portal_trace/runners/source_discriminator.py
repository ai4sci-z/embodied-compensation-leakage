"""Train the binary source discriminator used by the Lambda decomposition.

This is an independent estimator of ``w(x) = P(source | observation)``.  It
uses a deterministic paired validation seed inside the formal plan's training
split for checkpoint selection, evaluates calibration once on the untouched
held-out episodes, and publishes posterior values only for an explicitly
selected held-out cell.  It does not reuse the B4 adversarial head or policy
labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler

from portal_trace.cards.model_card import (
    DISCRIMINATOR_ERROR_BUDGET,
    DISCRIMINATOR_INFORMATION_GAIN_FLOOR_BITS,
)
from portal_trace.policies.encoders import ObservationEncoder
from portal_trace.policies.interface import Batch, Observation, ObsSpec
from portal_trace.runners.formal_adapter import entries_from_formal_plan
from portal_trace.runners.formal_data import (
    FormalDataPlan,
    FormalProprioNormalization,
    load_formal_data_plan,
    load_formal_normalization,
)
from portal_trace.runners.policy_dataset import (
    RawPolicyWindowDataset,
    collate_policy_samples,
)
from portal_trace.runners.source_posterior import (
    SourcePosteriorEpisode,
    SourcePosteriorHeader,
    SourcePosteriorProvenance,
    write_source_posterior,
)
from portal_trace.runners.source_validation import (
    load_source_validation_reservation,
    source_validation_run_spec,
)

__all__ = [
    "SOURCE_DISCRIMINATOR_SCHEMA",
    "SourceDiscriminatorConfig",
    "SourceDiscriminatorResult",
    "train_source_discriminator",
]

SOURCE_DISCRIMINATOR_SCHEMA = "portal_trace.source_discriminator/8"
_MARKER = "PORTAL_TRACE_SOURCE_DISCRIMINATOR_NOT_POLICY"
_PLATT_REGULARIZATION_CANDIDATES = (0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0)
_GROUP_ROBUSTNESS_CANDIDATES = (0.0, 1.0, 4.0, 16.0)
_GROUP_DRO_STEP_SIZE = 0.01
_MIN_PROSPECTIVE_VALIDATION_PAIRS = 6


@dataclass(frozen=True)
class SourceDiscriminatorConfig:
    plan_path: Path
    normalization_path: Path
    checkpoint_path: Path
    posterior_path: Path
    source_embodiment_indices: tuple[int, ...]
    target_embodiment_indices: tuple[int, ...]
    task: str
    family: str
    defect: str
    level: int
    seed: int
    epochs: int = 10
    optimizer_steps: int | None = None
    batch_size: int = 64
    learning_rate: float = 3e-4
    device: str = "cuda"
    n_tasks: int = 6
    d_model: int = 256
    augment: bool = True
    selection_interval: int = 25
    observation: ObsSpec = ObsSpec()
    validation_plan_path: Path | None = None
    validation_reservation_path: Path | None = None
    fusion_mode: str = "modality-mean"

    def __post_init__(self) -> None:
        for field in (
            "plan_path", "normalization_path", "checkpoint_path", "posterior_path"
        ):
            object.__setattr__(self, field, Path(getattr(self, field)).resolve())
        if self.validation_plan_path is not None:
            object.__setattr__(
                self,
                "validation_plan_path",
                Path(self.validation_plan_path).resolve(),
            )
        if self.validation_reservation_path is not None:
            object.__setattr__(
                self,
                "validation_reservation_path",
                Path(self.validation_reservation_path).resolve(),
            )
        if self.checkpoint_path == self.posterior_path:
            raise ValueError("source discriminator checkpoint and posterior paths must differ")
        source = tuple(self.source_embodiment_indices)
        target = tuple(self.target_embodiment_indices)
        if (
            not source
            or not target
            or len(set(source)) != len(source)
            or len(set(target)) != len(target)
            or set(source) & set(target)
        ):
            raise ValueError("source and target embodiment sets must be non-empty and disjoint")
        if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in source + target):
            raise ValueError("source discriminator embodiment indices must be non-negative")
        object.__setattr__(self, "source_embodiment_indices", tuple(sorted(source)))
        object.__setattr__(self, "target_embodiment_indices", tuple(sorted(target)))
        for field in ("task", "family", "defect"):
            if not str(getattr(self, field)).strip():
                raise ValueError(f"source discriminator {field} must be named")
        for field in (
            "level", "seed", "epochs", "batch_size", "n_tasks", "d_model",
            "selection_interval",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"source discriminator {field} must be an integer")
        if self.level < 0 or min(
            self.epochs,
            self.batch_size,
            self.n_tasks,
            self.d_model,
            self.selection_interval,
        ) < 1:
            raise ValueError("source discriminator sizes and epochs must be positive")
        if self.optimizer_steps is not None and (
            isinstance(self.optimizer_steps, bool)
            or not isinstance(self.optimizer_steps, int)
            or self.optimizer_steps < 1
        ):
            raise ValueError("source discriminator optimizer_steps must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("source discriminator learning_rate must be finite and positive")
        if self.fusion_mode not in {"modality-mean", "camera-time", "legacy-mean"}:
            raise ValueError("unsupported source discriminator fusion mode")
        target_device = torch.device(self.device)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA source discriminator but CUDA is unavailable")
        if target_device.type not in {"cpu", "cuda"}:
            raise ValueError("source discriminator device must be CPU or CUDA")


@dataclass(frozen=True)
class SourceDiscriminatorResult:
    checkpoint_path: Path
    posterior_path: Path
    posterior_header: SourcePosteriorHeader
    release_status: str
    epochs: int
    optimizer_steps: int
    final_loss: float
    best_step: int
    selection_metrics: dict[str, float]
    validation_metrics: dict[str, float]


class _SourceDiscriminator(nn.Module):
    def __init__(self, config: SourceDiscriminatorConfig) -> None:
        super().__init__()
        self.encoder = ObservationEncoder(
            config.observation,
            config.d_model,
            config.n_tasks,
            augment=config.augment,
        )
        self.visual_tokens = self.encoder.visual.n_tokens
        self.proprio_tokens = config.observation.k_hist if config.observation.use_proprio else 0
        self.task_tokens = 1 if config.observation.use_task_token else 0
        self.fusion_mode = config.fusion_mode
        if self.fusion_mode == "legacy-mean":
            # Checkpoints through /7 pooled all tokens before the unchanged head.
            self.modality_fusion = nn.Identity()
        else:
            fusion_tokens = (
                config.observation.n_cameras * config.observation.k_hist
                + self.proprio_tokens
                + self.task_tokens
                if self.fusion_mode == "camera-time"
                else self.n_modalities
            )
            self.modality_fusion = nn.Sequential(
                nn.LayerNorm(config.d_model * fusion_tokens),
                nn.Linear(config.d_model * fusion_tokens, config.d_model),
                nn.GELU(),
            )
        self.head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
        )

    @property
    def n_modalities(self) -> int:
        return 1 + int(self.proprio_tokens > 0) + int(self.task_tokens > 0)

    def features(self, observation: Observation) -> torch.Tensor:
        memory = self.encoder(
            observation.images, observation.proprio, observation.task_index
        )
        if self.fusion_mode == "legacy-mean":
            return memory.mean(dim=1)
        offset = 0
        visual = memory[:, : self.visual_tokens]
        if self.fusion_mode == "camera-time":
            # VisualTower emits camera, history, spatial-cell order. Pool only
            # spatial cells; each camera/time position keeps its own input slot.
            spec = self.encoder.spec
            visual = visual.reshape(
                memory.shape[0], spec.n_cameras, spec.k_hist,
                self.encoder.visual.cells, memory.shape[-1],
            ).mean(dim=3).flatten(start_dim=1)
        else:
            visual = visual.mean(dim=1)
        modalities = [visual]
        offset += self.visual_tokens
        if self.proprio_tokens:
            proprio = memory[:, offset : offset + self.proprio_tokens]
            modalities.append(
                proprio.flatten(start_dim=1)
                if self.fusion_mode == "camera-time"
                else proprio.mean(dim=1)
            )
            offset += self.proprio_tokens
        if self.task_tokens:
            modalities.append(memory[:, offset : offset + self.task_tokens].mean(dim=1))
            offset += self.task_tokens
        if offset != memory.shape[1]:
            raise RuntimeError("source discriminator token partition disagrees with encoder")
        return self.modality_fusion(torch.cat(modalities, dim=-1))

    def forward(self, observation: Observation) -> torch.Tensor:
        return self.head(self.features(observation)).squeeze(-1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _selected_episode_ids(
    plan: FormalDataPlan,
    config: SourceDiscriminatorConfig,
    *,
    eligible_episode_ids: Sequence[str],
) -> tuple[str, ...]:
    eligible = set(eligible_episode_ids)
    selected = tuple(
        sorted(
            row["id"]
            for row in plan.payload["pool"]
            if row["id"] in eligible
            and row["task"] == config.task
            and row["family"] == config.family
            and row["defect"] == config.defect
            and row["level"] == config.level
        )
    )
    if not selected:
        raise ValueError("source discriminator found no held-out episode for the requested cell")
    return selected


def _validate_prospective_validation_plan(
    development: FormalDataPlan,
    validation: FormalDataPlan,
    config: SourceDiscriminatorConfig,
    *,
    allow_engineering: bool,
) -> dict[str, Any]:
    """Require a genuinely new complete-pair population for final validation."""
    if development.plan_sha256 == validation.plan_sha256:
        raise ValueError("prospective validation plan must differ from the development plan")

    development_rows = tuple(development.payload["pool"])
    validation_rows = tuple(validation.payload["pool"])
    development_seeds = {int(row["seed_env"]) for row in development_rows}
    validation_seeds = {int(row["seed_env"]) for row in validation_rows}
    seed_overlap = development_seeds & validation_seeds
    if seed_overlap:
        raise ValueError(
            "prospective validation reuses development environment seeds: "
            f"{sorted(seed_overlap)}"
        )
    development_pairs = {str(row["pair_id"]) for row in development_rows}
    validation_pairs = {str(row["pair_id"]) for row in validation_rows}
    if development_pairs & validation_pairs:
        raise ValueError("prospective validation reuses a development twin pair")
    development_hashes = {str(row["sha256"]) for row in development_rows}
    if development_hashes & {str(row["sha256"]) for row in validation_rows}:
        raise ValueError("prospective validation reuses a development dataset file")

    def task_map(plan: FormalDataPlan) -> dict[int, str]:
        return {
            int(row["task_index"]): str(row["task"])
            for row in plan.payload["pool"]
        }

    def embodiment_map(plan: FormalDataPlan) -> dict[int, tuple[str, str, int, str]]:
        return {
            int(row["embodiment_index"]): (
                str(row["family"]),
                str(row["defect"]),
                int(row["level"]),
                str(row["clean_of"]),
            )
            for row in plan.payload["pool"]
        }

    if task_map(development) != task_map(validation):
        raise ValueError("prospective validation task-index mapping changed")
    if embodiment_map(development) != embodiment_map(validation):
        raise ValueError("prospective validation embodiment-index mapping changed")

    rows_by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in validation_rows:
        if (
            row["task"] != config.task
            or row["family"] != config.family
            or int(row["level"]) != config.level
            or not (
                row["defect"] == config.defect
                or (row["defect"] == "clean" and row["clean_of"] == config.defect)
            )
        ):
            raise ValueError(
                "prospective validation plan contains a row outside the requested cell"
            )
        rows_by_pair.setdefault(str(row["pair_id"]), []).append(row)

    source_indices = set(config.source_embodiment_indices)
    target_indices = set(config.target_embodiment_indices)
    for pair_id, rows in rows_by_pair.items():
        if len(rows) != 2:
            raise ValueError(f"prospective twin pair {pair_id} is not exactly two episodes")
        if len({int(row["seed_env"]) for row in rows}) != 1:
            raise ValueError(f"prospective twin pair {pair_id} spans multiple seeds")
        indices = {int(row["embodiment_index"]) for row in rows}
        if not (indices & source_indices and indices & target_indices):
            raise ValueError(
                f"prospective twin pair {pair_id} lacks a source or target episode"
            )
    heldout_ids = set(validation.heldout_episode_ids)
    heldout_rows = [row for row in validation_rows if row["id"] in heldout_ids]
    heldout_pairs = {str(row["pair_id"]) for row in heldout_rows}
    if any(
        {str(row["id"]) for row in rows_by_pair[pair_id]} - heldout_ids
        for pair_id in heldout_pairs
    ):
        raise ValueError("prospective validation split divided a twin pair")
    minimum_pairs = 2 if allow_engineering else _MIN_PROSPECTIVE_VALIDATION_PAIRS
    if len(heldout_pairs) < minimum_pairs:
        raise ValueError(
            "prospective validation has too few complete twin pairs: "
            f"{len(heldout_pairs)} < {minimum_pairs}"
        )
    return {
        "mode": "prospective-separate-plan",
        "plan_sha256": validation.plan_sha256,
        "development_plan_sha256": development.plan_sha256,
        "episode_ids": sorted(str(row["id"]) for row in heldout_rows),
        "pair_ids": sorted(heldout_pairs),
        "seed_envs": sorted({int(row["seed_env"]) for row in heldout_rows}),
        "reserved_pool_seed_envs": sorted(validation_seeds),
        "minimum_pairs": minimum_pairs,
        "all_pairs_complete": True,
        "development_seed_overlap": [],
    }


def _validate_label_partition(
    plan: FormalDataPlan, config: SourceDiscriminatorConfig
) -> None:
    observed = {int(row["embodiment_index"]) for row in plan.payload["pool"]}
    declared = set(config.source_embodiment_indices) | set(
        config.target_embodiment_indices
    )
    if observed != declared:
        raise ValueError(
            "source/target embodiment sets must partition the complete formal plan; "
            f"missing={sorted(observed - declared)}, unexpected={sorted(declared - observed)}"
        )
    max_task = max(int(row["task_index"]) for row in plan.payload["pool"])
    if max_task >= config.n_tasks:
        raise ValueError("source discriminator n_tasks cannot represent the formal plan")


def _selection_split_ids(
    plan: FormalDataPlan, config: SourceDiscriminatorConfig
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    """Reserve disjoint complete twin pairs for selection and calibration.

    The formal held-out split is the downstream evaluation population, so using
    it for early stopping would tune the source posterior on the same episodes
    later consumed by the leakage estimator. Checkpoint selection and posterior
    calibration are also different statistical operations, so they receive
    separate complete twin pairs from ``plan.train_episode_ids``. Both are
    chosen by a stable hash of pair identity and the configured training seed.
    """

    training = set(plan.train_episode_ids)
    rows = [row for row in plan.payload["pool"] if row["id"] in training]
    source_indices = set(config.source_embodiment_indices)
    target_indices = set(config.target_embodiment_indices)
    rows_by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_pair.setdefault(str(row["pair_id"]), []).append(row)
    eligible_pairs = []
    for pair_id, pair_rows in rows_by_pair.items():
        indices = {int(row["embodiment_index"]) for row in pair_rows}
        if indices & source_indices and indices & target_indices:
            eligible_pairs.append(pair_id)
    if len(eligible_pairs) < 4:
        raise ValueError(
            "source discriminator needs at least four complete source/target twin "
            "pairs inside the training split: one for checkpoint selection, at least "
            "two for posterior calibration, and at least one for fitting"
        )

    def rank(pair_id: str) -> str:
        return hashlib.sha256(
            f"{config.seed}\0source-discriminator-selection\0{pair_id}".encode("ascii")
        ).hexdigest()

    ranked_pairs = sorted(eligible_pairs, key=rank)
    n_selection_pairs = min(
        len(ranked_pairs) - 3,
        max(1, math.ceil(len(ranked_pairs) * 0.20)),
    )
    n_calibration_pairs = min(
        len(ranked_pairs) - n_selection_pairs - 1,
        max(2, math.ceil(len(ranked_pairs) * 0.25)),
    )
    selection_pair_ids = tuple(ranked_pairs[:n_selection_pairs])
    calibration_pair_ids = tuple(
        ranked_pairs[n_selection_pairs : n_selection_pairs + n_calibration_pairs]
    )
    selection_ids = tuple(
        sorted(
            row["id"]
            for pair_id in selection_pair_ids
            for row in rows_by_pair[pair_id]
        )
    )
    calibration_ids = tuple(
        sorted(
            row["id"]
            for pair_id in calibration_pair_ids
            for row in rows_by_pair[pair_id]
        )
    )
    fit_ids = tuple(sorted(training - set(selection_ids) - set(calibration_ids)))
    if not fit_ids or not selection_ids or not calibration_ids:
        raise ValueError("source discriminator fit/selection/calibration split is empty")

    def represented(episode_ids: tuple[str, ...]) -> tuple[bool, bool]:
        selected = set(episode_ids)
        indices = {
            int(row["embodiment_index"])
            for row in rows
            if row["id"] in selected
        }
        return bool(indices & source_indices), bool(indices & target_indices)

    if any(
        represented(episode_ids) != (True, True)
        for episode_ids in (fit_ids, selection_ids, calibration_ids)
    ):
        raise ValueError(
            "source discriminator fit, selection and calibration splits must each "
            "contain source and target episodes"
        )
    fit_pairs = {
        str(row["pair_id"]) for row in rows if row["id"] in set(fit_ids)
    }
    if (set(selection_pair_ids) | set(calibration_pair_ids)) & fit_pairs:
        raise RuntimeError(
            "source discriminator split leaked a reserved twin pair into fitting"
        )

    def pair_seed(pair_id: str, *, role: str) -> int:
        seeds = {int(row["seed_env"]) for row in rows_by_pair[pair_id]}
        if len(seeds) != 1:
            raise ValueError(f"source discriminator {role} twin spans multiple seeds")
        return next(iter(seeds))

    return (
        fit_ids,
        selection_ids,
        calibration_ids,
        selection_pair_ids,
        calibration_pair_ids,
        tuple(pair_seed(pair_id, role="selection") for pair_id in selection_pair_ids),
        tuple(pair_seed(pair_id, role="calibration") for pair_id in calibration_pair_ids),
    )


def _normalised_observation(
    batch: Batch,
    normalization: FormalProprioNormalization,
    device: torch.device,
) -> Observation:
    if batch.obs.proprio is None or batch.obs.task_index is None:
        raise ValueError("source discriminator requires proprioception and task identity")
    return Observation(
        images={name: values.to(device) for name, values in batch.obs.images.items()},
        proprio=normalization.apply(batch.obs.proprio).to(device),
        task_index=batch.obs.task_index.to(device),
        embodiment_index=(
            None
            if batch.obs.embodiment_index is None
            else batch.obs.embodiment_index.to(device)
        ),
        partition_index=None,
    )


def _labels(
    batch: Batch,
    config: SourceDiscriminatorConfig,
    device: torch.device,
) -> torch.Tensor:
    indices = batch.obs.embodiment_index
    if indices is None:
        raise ValueError("source discriminator requires embodiment identity labels")
    source = torch.zeros_like(indices, dtype=torch.bool)
    target = torch.zeros_like(indices, dtype=torch.bool)
    for index in config.source_embodiment_indices:
        source |= indices == index
    for index in config.target_embodiment_indices:
        target |= indices == index
    if not bool(torch.all(source | target)) or bool(torch.any(source & target)):
        raise ValueError("batch contains an embodiment outside the source/target partition")
    return source.to(device=device, dtype=torch.float32)


def _training_pair_metadata(
    plan: FormalDataPlan,
    config: SourceDiscriminatorConfig,
    *,
    fit_ids: Sequence[str],
) -> tuple[dict[str, int], dict[str, int], tuple[str, ...]]:
    """Bind every fitting episode to one complete pair and one class.

    Pair identity is carried only as loss metadata. ``_normalised_observation``
    deliberately removes it before the encoder is called, so inference remains
    a function of the observation alone.
    """
    selected = set(fit_ids)
    rows_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in plan.payload["pool"]:
        if row["id"] in selected:
            rows_by_pair[str(row["pair_id"])].append(row)
    pair_ids = tuple(sorted(rows_by_pair))
    if not pair_ids:
        raise ValueError("source discriminator fitting split has no twin pairs")
    source_indices = set(config.source_embodiment_indices)
    target_indices = set(config.target_embodiment_indices)
    pair_index_by_episode: dict[str, int] = {}
    class_by_episode: dict[str, int] = {}
    for pair_index, pair_id in enumerate(pair_ids):
        rows = rows_by_pair[pair_id]
        if len(rows) != 2:
            raise ValueError(f"fitting twin pair {pair_id} is not exactly two episodes")
        classes: set[int] = set()
        for row in rows:
            embodiment_index = int(row["embodiment_index"])
            if embodiment_index in source_indices:
                label = 1
            elif embodiment_index in target_indices:
                label = 0
            else:
                raise ValueError("fitting pair contains an undeclared embodiment")
            episode_id = str(row["id"])
            pair_index_by_episode[episode_id] = pair_index
            class_by_episode[episode_id] = label
            classes.add(label)
        if classes != {0, 1}:
            raise ValueError(f"fitting twin pair {pair_id} lacks one source/target class")
    if set(pair_index_by_episode) != selected:
        raise RuntimeError("fitting episode-to-pair metadata is incomplete")
    return pair_index_by_episode, class_by_episode, pair_ids


class _PairClassBalancedSampler(Sampler[int]):
    """Sample every complete pair and both labels at equal frequency."""

    def __init__(
        self,
        dataset: RawPolicyWindowDataset,
        *,
        pair_index_by_episode: Mapping[str, int],
        class_by_episode: Mapping[str, int],
        num_samples: int,
        seed: int,
    ) -> None:
        if num_samples < 1:
            raise ValueError("pair-balanced sampler needs at least one sample")
        strata: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index in range(len(dataset)):
            episode_id, _time_index = dataset.window_identity(index)
            try:
                stratum = (
                    int(pair_index_by_episode[episode_id]),
                    int(class_by_episode[episode_id]),
                )
            except KeyError as exc:
                raise ValueError(
                    "pair-balanced sampler encountered an unbound fitting episode"
                ) from exc
            strata[stratum].append(index)
        pair_indices = {pair_index for pair_index, _label in strata}
        expected = {(pair_index, label) for pair_index in pair_indices for label in (0, 1)}
        if set(strata) != expected or any(not indices for indices in strata.values()):
            raise ValueError("pair-balanced sampler requires both classes for every pair")
        self._strata = tuple((key, tuple(strata[key])) for key in sorted(strata))
        self._num_samples = int(num_samples)
        self._generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        produced = 0
        while produced < self._num_samples:
            order = torch.randperm(len(self._strata), generator=self._generator).tolist()
            for stratum_index in order:
                _key, candidates = self._strata[stratum_index]
                selected = int(
                    torch.randint(len(candidates), (1,), generator=self._generator).item()
                )
                yield candidates[selected]
                produced += 1
                if produced == self._num_samples:
                    return


def _pair_group_dro_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pair_indices: torch.Tensor,
    log_group_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[int, float]]:
    """Return GroupDRO risk over pair-wise BCE losses for one balanced batch."""
    if logits.shape != labels.shape or pair_indices.shape != labels.shape:
        raise ValueError("pair-aware source loss inputs must have identical shapes")
    per_sample = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    observed = torch.unique(pair_indices, sorted=True)
    if observed.numel() == 0:
        raise ValueError("pair-aware source batch contains no pair identity")
    group_losses = torch.stack(
        [per_sample[pair_indices == index].mean() for index in observed]
    )
    with torch.no_grad():
        log_group_weights[observed] += _GROUP_DRO_STEP_SIZE * group_losses.detach()
        log_group_weights -= torch.logsumexp(log_group_weights, dim=0)
    observed_weights = torch.softmax(log_group_weights, dim=0)[observed]
    observed_weights = observed_weights / observed_weights.sum()
    loss = torch.sum(observed_weights * group_losses)
    resolved = {
        int(index.detach().cpu()): float(value.detach().cpu())
        for index, value in zip(observed, group_losses, strict=True)
    }
    return loss, resolved


def _auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positives = probabilities[labels == 1]
    negatives = probabilities[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        raise ValueError("source discriminator validation AUC needs both classes")
    comparisons = positives[:, None] - negatives[None, :]
    wins = np.count_nonzero(comparisons > 0)
    ties = np.count_nonzero(comparisons == 0)
    return float((wins + 0.5 * ties) / comparisons.size)


def _ece(labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> float:
    total = probabilities.size
    value = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (probabilities >= lower) & (
            probabilities <= upper if upper == 1.0 else probabilities < upper
        )
        if np.any(selected):
            value += (np.count_nonzero(selected) / total) * abs(
                float(probabilities[selected].mean() - labels[selected].mean())
            )
    return float(value)


def _classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    clipped = np.clip(probabilities, 1.0e-7, 1.0 - 1.0e-7)
    actual_fraction = float(labels.mean())
    mean_posterior = float(probabilities.mean())
    return {
        "auc": _auc(labels.astype(np.int8), probabilities),
        "ece": _ece(labels, probabilities),
        "alpha_recovery_abs_error": abs(mean_posterior - actual_fraction),
        "validation_bce": float(
            np.mean(-(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)))
        ),
        "actual_source_fraction": actual_fraction,
        "mean_posterior": mean_posterior,
    }


def _information_gain_bits(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Proper-score gain over the constant empirical-prior predictor."""
    truth = np.asarray(labels, dtype=np.float64)
    predicted = np.clip(
        np.asarray(probabilities, dtype=np.float64), 1.0e-7, 1.0 - 1.0e-7
    )
    if truth.ndim != 1 or predicted.shape != truth.shape or truth.size < 2:
        raise ValueError(
            "information gain needs aligned non-empty label/probability vectors"
        )
    prior = float(truth.mean())
    if not 0.0 < prior < 1.0:
        raise ValueError("information gain needs both source and target labels")
    prior_bce = -(prior * math.log(prior) + (1.0 - prior) * math.log(1.0 - prior))
    model_bce = float(
        np.mean(
            -(truth * np.log(predicted) + (1.0 - truth) * np.log(1.0 - predicted))
        )
    )
    return float((prior_bce - model_bce) / math.log(2.0))


def _pair_selection_metrics(
    pair_outputs: Sequence[tuple[str, np.ndarray, np.ndarray]],
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    """Resolve checkpoint quality by complete pair, never pooled frame count."""
    rows: list[dict[str, float | str]] = []
    for pair_id, logits, labels in pair_outputs:
        probabilities = _calibrated_probabilities(logits, scale=1.0, bias=0.0)
        metrics = _classification_metrics(labels, probabilities)
        metrics["calibration_burden"] = (
            metrics["ece"] + metrics["alpha_recovery_abs_error"]
        )
        metrics["information_gain_bits"] = _information_gain_bits(labels, probabilities)
        rows.append({"pair_id": pair_id, **metrics})
    if not rows:
        raise ValueError("checkpoint selection needs at least one complete twin pair")
    aucs = [float(row["auc"]) for row in rows]
    bces = [float(row["validation_bce"]) for row in rows]
    burdens = [float(row["calibration_burden"]) for row in rows]
    information = [float(row["information_gain_bits"]) for row in rows]
    aggregate = {
        "orientation_failure_count": float(sum(auc < 0.5 for auc in aucs)),
        "worst_pair_auc": min(aucs),
        "worst_pair_validation_bce": max(bces),
        "worst_pair_calibration_burden": max(burdens),
        "worst_pair_information_gain_bits": min(information),
        "mean_pair_validation_bce": float(np.mean(bces)),
        "mean_pair_calibration_burden": float(np.mean(burdens)),
        "mean_pair_information_gain_bits": float(np.mean(information)),
    }
    return aggregate, rows


def _checkpoint_selection_key(metrics: Mapping[str, float]) -> tuple[float, ...]:
    """Rank orientation-safe checkpoints by worst-pair score and burden."""
    failures = metrics["orientation_failure_count"]
    failed = float(failures > 0.0)
    return (
        failed,
        failures if failed else 0.0,
        -metrics["worst_pair_auc"] if failed else 0.0,
        metrics["worst_pair_validation_bce"],
        metrics["worst_pair_calibration_burden"],
        metrics["mean_pair_validation_bce"],
    )


@torch.no_grad()
def _infer_raw(
    model: _SourceDiscriminator,
    dataset: RawPolicyWindowDataset,
    normalization: FormalProprioNormalization,
    config: SourceDiscriminatorConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_policy_samples,
    )
    model.eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    features: list[np.ndarray] = []
    for batch in loader:
        observation = _normalised_observation(batch, normalization, device)
        encoded = model.features(observation)
        logits.append(model.head(encoded).squeeze(-1).cpu().numpy())
        labels.append(_labels(batch, config, device).cpu().numpy())
        features.append(encoded.cpu().numpy())
    return np.concatenate(logits), np.concatenate(labels), np.concatenate(features)


def _calibrated_probabilities(
    logits: np.ndarray, *, scale: float, bias: float
) -> np.ndarray:
    calibrated = np.asarray(logits, dtype=np.float64) * scale + bias
    positive = calibrated >= 0.0
    probabilities = np.empty_like(calibrated)
    probabilities[positive] = 1.0 / (1.0 + np.exp(-calibrated[positive]))
    exponential = np.exp(calibrated[~positive])
    probabilities[~positive] = exponential / (1.0 + exponential)
    return probabilities


def _fit_platt_calibrator(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    regularization: float = 0.0,
) -> tuple[float, float]:
    """Fit a monotone affine calibration, shrinking toward the raw logits."""
    raw = torch.as_tensor(logits, dtype=torch.float64)
    truth = torch.as_tensor(labels, dtype=torch.float64)
    if raw.ndim != 1 or truth.shape != raw.shape or raw.numel() < 2:
        raise ValueError("Platt calibration needs aligned non-empty logit and label vectors")
    if torch.unique(truth).numel() != 2:
        raise ValueError("Platt calibration needs both source and target labels")
    if not math.isfinite(regularization) or regularization < 0.0:
        raise ValueError("Platt regularization must be finite and non-negative")
    log_scale = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_scale, bias],
        max_iter=100,
        tolerance_grad=1.0e-10,
        tolerance_change=1.0e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(log_scale.exp() * raw + bias, truth)
        if regularization:
            loss = loss + regularization * (log_scale.square() + bias.square())
        loss.backward()
        return loss

    optimizer.step(closure)
    scale = float(log_scale.detach().exp())
    intercept = float(bias.detach())
    if not math.isfinite(scale) or not math.isfinite(intercept) or scale <= 0.0:
        raise RuntimeError("Platt calibration produced invalid affine parameters")
    return scale, intercept


def _fit_pair_robust_platt(
    pair_outputs: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    regularization: float,
    robustness: float,
) -> tuple[float, float]:
    """Fit one affine calibrator with equal-pair entropic-risk weighting.

    ``robustness=0`` is the pair-balanced mean BCE. Positive values smoothly
    approach the worst-pair BCE, so a long or easy trajectory cannot dominate
    calibration. The transform remains global and monotone; pair identity is
    never available when scoring a new episode.
    """
    folds = tuple(pair_outputs)
    if len(folds) < 2:
        raise ValueError("pair-robust Platt calibration needs at least two twin pairs")
    if not math.isfinite(regularization) or regularization < 0.0:
        raise ValueError("Platt regularization must be finite and non-negative")
    if not math.isfinite(robustness) or robustness < 0.0:
        raise ValueError("group robustness must be finite and non-negative")

    tensors: list[tuple[torch.Tensor, torch.Tensor]] = []
    for logits, labels in folds:
        raw = torch.as_tensor(logits, dtype=torch.float64)
        truth = torch.as_tensor(labels, dtype=torch.float64)
        if raw.ndim != 1 or truth.shape != raw.shape or raw.numel() < 2:
            raise ValueError(
                "each calibration pair needs aligned logit and label vectors"
            )
        if torch.unique(truth).numel() != 2:
            raise ValueError(
                "each calibration pair needs both source and target labels"
            )
        tensors.append((raw, truth))

    log_scale = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_scale, bias],
        max_iter=100,
        tolerance_grad=1.0e-10,
        tolerance_change=1.0e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        losses = torch.stack(
            [
                F.binary_cross_entropy_with_logits(
                    log_scale.exp() * raw + bias, truth
                )
                for raw, truth in tensors
            ]
        )
        if robustness == 0.0:
            loss = losses.mean()
        else:
            loss = (
                torch.logsumexp(robustness * losses, dim=0)
                - math.log(len(tensors))
            ) / robustness
        if regularization:
            loss = loss + regularization * (log_scale.square() + bias.square())
        loss.backward()
        return loss

    optimizer.step(closure)
    scale = float(log_scale.detach().exp())
    intercept = float(bias.detach())
    if not math.isfinite(scale) or not math.isfinite(intercept) or scale <= 0.0:
        raise RuntimeError("pair-robust Platt calibration produced invalid parameters")
    return scale, intercept


def _select_pair_regularized_platt(
    pair_outputs: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    candidates: Sequence[float] = _PLATT_REGULARIZATION_CANDIDATES,
) -> tuple[float, list[dict[str, Any]]]:
    """Choose Platt shrinkage by leave-one-complete-pair-out BCE."""
    folds = tuple(pair_outputs)
    if len(folds) < 2:
        raise ValueError("pair-aware Platt selection needs at least two twin pairs")
    candidate_values = tuple(float(value) for value in candidates)
    if (
        not candidate_values
        or len(set(candidate_values)) != len(candidate_values)
        or any(not math.isfinite(value) or value < 0.0 for value in candidate_values)
    ):
        raise ValueError("Platt regularization candidates must be unique and non-negative")

    rows: list[dict[str, Any]] = []
    for regularization in candidate_values:
        fold_metrics = []
        for heldout_index, (heldout_logits, heldout_labels) in enumerate(folds):
            fit_logits = np.concatenate(
                [
                    values
                    for index, (values, _labels) in enumerate(folds)
                    if index != heldout_index
                ]
            )
            fit_labels = np.concatenate(
                [
                    values
                    for index, (_logits, values) in enumerate(folds)
                    if index != heldout_index
                ]
            )
            scale, bias = _fit_platt_calibrator(
                fit_logits,
                fit_labels,
                regularization=regularization,
            )
            metrics = _classification_metrics(
                heldout_labels,
                _calibrated_probabilities(heldout_logits, scale=scale, bias=bias),
            )
            fold_metrics.append(
                {
                    "fold": heldout_index,
                    "validation_bce": metrics["validation_bce"],
                    "ece": metrics["ece"],
                    "alpha_recovery_abs_error": metrics[
                        "alpha_recovery_abs_error"
                    ],
                }
            )
        rows.append(
            {
                "regularization": regularization,
                "mean_pair_validation_bce": float(
                    np.mean([row["validation_bce"] for row in fold_metrics])
                ),
                "mean_pair_ece": float(
                    np.mean([row["ece"] for row in fold_metrics])
                ),
                "mean_pair_alpha_recovery_abs_error": float(
                    np.mean(
                        [row["alpha_recovery_abs_error"] for row in fold_metrics]
                    )
                ),
                "folds": fold_metrics,
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            row["mean_pair_validation_bce"],
            row["mean_pair_ece"],
            -row["regularization"],
        ),
    )
    return float(selected["regularization"]), rows


def _select_pair_robust_platt(
    pair_outputs: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    regularization_candidates: Sequence[float] = _PLATT_REGULARIZATION_CANDIDATES,
    robustness_candidates: Sequence[float] = _GROUP_ROBUSTNESS_CANDIDATES,
    information_gain_floor_bits: float = DISCRIMINATOR_INFORMATION_GAIN_FLOOR_BITS,
    allow_uninformative: bool = False,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Select group robustness by leave-one-complete-pair-out validation.

    A candidate is informative only when every held-out fold clears the fixed
    proper-score floor; a strong average cannot hide a collapsed pair.
    """
    folds = tuple(pair_outputs)
    if len(folds) < 3:
        raise ValueError("pair-robust selection needs at least three twin pairs")
    regularizations = tuple(float(value) for value in regularization_candidates)
    robustness_values = tuple(float(value) for value in robustness_candidates)
    if (
        not regularizations
        or len(set(regularizations)) != len(regularizations)
        or any(not math.isfinite(value) or value < 0.0 for value in regularizations)
    ):
        raise ValueError(
            "Platt regularization candidates must be unique and non-negative"
        )
    if (
        not robustness_values
        or len(set(robustness_values)) != len(robustness_values)
        or any(not math.isfinite(value) or value < 0.0 for value in robustness_values)
    ):
        raise ValueError(
            "group-robustness candidates must be unique and non-negative"
        )
    if (
        not math.isfinite(information_gain_floor_bits)
        or information_gain_floor_bits <= 0.0
    ):
        raise ValueError("information-gain floor must be finite and positive")

    rows: list[dict[str, Any]] = []
    for robustness in robustness_values:
        for regularization in regularizations:
            fold_metrics = []
            for heldout_index, (heldout_logits, heldout_labels) in enumerate(folds):
                fit_pairs = tuple(
                    pair for index, pair in enumerate(folds) if index != heldout_index
                )
                scale, bias = _fit_pair_robust_platt(
                    fit_pairs,
                    regularization=regularization,
                    robustness=robustness,
                )
                probabilities = _calibrated_probabilities(
                    heldout_logits, scale=scale, bias=bias
                )
                metrics = _classification_metrics(heldout_labels, probabilities)
                fold_metrics.append(
                    {
                        "fold": heldout_index,
                        "validation_bce": metrics["validation_bce"],
                        "ece": metrics["ece"],
                        "alpha_recovery_abs_error": metrics[
                            "alpha_recovery_abs_error"
                        ],
                        "calibration_burden": metrics["ece"]
                        + metrics["alpha_recovery_abs_error"],
                        "information_gain_bits": _information_gain_bits(
                            heldout_labels, probabilities
                        ),
                    }
                )
            mean_information_gain = float(
                np.mean([row["information_gain_bits"] for row in fold_metrics])
            )
            worst_information_gain = float(
                min(row["information_gain_bits"] for row in fold_metrics)
            )
            rows.append(
                {
                    "robustness": robustness,
                    "regularization": regularization,
                    "worst_pair_calibration_burden": float(
                        max(row["calibration_burden"] for row in fold_metrics)
                    ),
                    "mean_pair_calibration_burden": float(
                        np.mean(
                            [row["calibration_burden"] for row in fold_metrics]
                        )
                    ),
                    "mean_pair_validation_bce": float(
                        np.mean([row["validation_bce"] for row in fold_metrics])
                    ),
                    "mean_information_gain_bits": mean_information_gain,
                    "worst_pair_information_gain_bits": worst_information_gain,
                    "informative": worst_information_gain
                    >= information_gain_floor_bits,
                    "folds": fold_metrics,
                }
            )
    eligible = [row for row in rows if row["informative"]]
    if not eligible:
        if not allow_uninformative:
            raise RuntimeError(
                "no pair-robust calibrator retained the preregistered information gain"
            )
        selected = min(
            rows,
            key=lambda row: (
                -row["worst_pair_information_gain_bits"],
                row["worst_pair_calibration_burden"],
                row["mean_pair_calibration_burden"],
                row["mean_pair_validation_bce"],
                row["robustness"],
                row["regularization"],
            ),
        )
    else:
        selected = min(
            eligible,
            key=lambda row: (
                row["worst_pair_calibration_burden"],
                row["mean_pair_calibration_burden"],
                row["mean_pair_validation_bce"],
                row["robustness"],
                row["regularization"],
            ),
        )
    selected["selected"] = True
    return (
        float(selected["regularization"]),
        float(selected["robustness"]),
        rows,
    )


def _same_class_nearest_distances(
    reference_features: np.ndarray,
    reference_labels: np.ndarray,
    query_features: np.ndarray,
    query_labels: np.ndarray,
) -> np.ndarray:
    distances = np.empty(len(query_features), dtype=np.float64)
    for label in (0.0, 1.0):
        reference = reference_features[reference_labels == label]
        query_mask = query_labels == label
        query = query_features[query_mask]
        if reference.size == 0 or query.size == 0:
            raise ValueError("support diagnostics need both classes in every split")
        class_distances = np.empty(len(query), dtype=np.float64)
        for query_start in range(0, len(query), 64):
            query_block = query[query_start : query_start + 64]
            best = np.full(len(query_block), np.inf, dtype=np.float64)
            for reference_start in range(0, len(reference), 512):
                reference_block = reference[reference_start : reference_start + 512]
                squared = np.sum(
                    (query_block[:, None, :] - reference_block[None, :, :]) ** 2,
                    axis=2,
                )
                best = np.minimum(best, squared.min(axis=1))
            class_distances[query_start : query_start + len(query_block)] = np.sqrt(best)
        distances[query_mask] = class_distances
    return distances


def _support_diagnostics(
    *,
    fit_features: np.ndarray,
    fit_labels: np.ndarray,
    calibration_features: np.ndarray,
    calibration_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
) -> dict[str, float]:
    """Measure excess held-out mass outside the fitted representation support.

    Distances are class-conditional in the frozen discriminator representation.
    The calibration pair fixes a 95th-percentile reference radius. The budgeted
    gap is only the held-out off-support mass above the observed calibration
    baseline, while both raw fractions remain available for interpretation.
    """
    center = fit_features.mean(axis=0)
    feature_scale = fit_features.std(axis=0)
    feature_scale = np.where(feature_scale > 1.0e-8, feature_scale, 1.0)

    def standardize(values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - center) / feature_scale

    fit = standardize(fit_features)
    calibration = standardize(calibration_features)
    validation = standardize(validation_features)
    calibration_distances = _same_class_nearest_distances(
        fit, fit_labels, calibration, calibration_labels
    )
    validation_distances = _same_class_nearest_distances(
        fit, fit_labels, validation, validation_labels
    )
    threshold = float(np.quantile(calibration_distances, 0.95, method="higher"))
    calibration_off = float(np.mean(calibration_distances > threshold))
    off_support = validation_distances > threshold
    validation_off = float(np.mean(off_support))
    in_support = ~off_support
    return {
        "w_support_gap": max(0.0, validation_off - calibration_off),
        "calibration_off_support_fraction": calibration_off,
        "off_support_fraction": validation_off,
        "in_support_fraction": float(np.mean(in_support)),
        "support_distance_threshold": threshold,
    }


def _posterior_episodes(
    model: _SourceDiscriminator,
    dataset: RawPolicyWindowDataset,
    normalization: FormalProprioNormalization,
    config: SourceDiscriminatorConfig,
    device: torch.device,
    *,
    calibration_scale: float,
    calibration_bias: float,
) -> tuple[SourcePosteriorEpisode, ...]:
    logits, _labels_unused, _features_unused = _infer_raw(
        model, dataset, normalization, config, device
    )
    probabilities = _calibrated_probabilities(
        logits, scale=calibration_scale, bias=calibration_bias
    )
    rows: dict[str, list[tuple[int, float]]] = {}
    for index, probability in enumerate(probabilities):
        episode_id, time_index = dataset.window_identity(index)
        rows.setdefault(episode_id, []).append((time_index, float(probability)))
    episodes = []
    for episode_id in sorted(rows):
        ordered = sorted(rows[episode_id])
        if [time for time, _value in ordered] != list(range(len(ordered))):
            raise RuntimeError(f"source-posterior frames are not contiguous for {episode_id}")
        episodes.append(
            SourcePosteriorEpisode(
                episode_id=episode_id,
                posterior_w=np.asarray([value for _time, value in ordered]),
            )
        )
    return tuple(episodes)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite source discriminator {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train_source_discriminator(
    config: SourceDiscriminatorConfig,
    *,
    allow_engineering: bool = False,
) -> SourceDiscriminatorResult:
    """Train, validate and publish the posterior for one held-out cell."""
    from portal_trace.runners.formal_training import _git_training_producer_state

    if config.checkpoint_path.exists() or config.posterior_path.exists():
        raise FileExistsError("source discriminator outputs are immutable and already exist")
    plan = load_formal_data_plan(config.plan_path)
    normalization = load_formal_normalization(config.normalization_path, plan=plan)
    _validate_label_partition(plan, config)
    frozen_run_spec = source_validation_run_spec(
        normalization_sha256=normalization.artifact_sha256,
        source_embodiment_indices=config.source_embodiment_indices,
        target_embodiment_indices=config.target_embodiment_indices,
        task=config.task,
        family=config.family,
        defect=config.defect,
        level=config.level,
        training_seed=config.seed,
        epochs=config.epochs,
        optimizer_steps=config.optimizer_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        device=torch.device(config.device).type,
        n_tasks=config.n_tasks,
        d_model=config.d_model,
        augment=config.augment,
        selection_interval=config.selection_interval,
        observation=asdict(config.observation),
        fusion_mode=config.fusion_mode,
    )
    validation_reservation = None
    if config.validation_plan_path is None:
        if not allow_engineering:
            raise ValueError(
                "formal source-discriminator validation requires a separate prospective "
                "--validation-plan; the development plan held-out may only be used for "
                "engineering diagnostics"
            )
        validation_plan = plan
        validation_ids = tuple(plan.heldout_episode_ids)
        validation_contract = {
            "mode": "development-heldout-engineering-only",
            "plan_sha256": plan.plan_sha256,
            "development_plan_sha256": plan.plan_sha256,
            "episode_ids": list(validation_ids),
            "pair_ids": sorted(
                {
                    str(row["pair_id"])
                    for row in plan.payload["pool"]
                    if row["id"] in set(validation_ids)
                }
            ),
            "seed_envs": sorted(
                {
                    int(row["seed_env"])
                    for row in plan.payload["pool"]
                    if row["id"] in set(validation_ids)
                }
            ),
            "minimum_pairs": 0,
            "all_pairs_complete": True,
            "development_seed_overlap": "expected-engineering-reuse",
        }
    else:
        validation_plan = load_formal_data_plan(config.validation_plan_path)
        _validate_label_partition(validation_plan, config)
        validation_contract = _validate_prospective_validation_plan(
            plan,
            validation_plan,
            config,
            allow_engineering=allow_engineering,
        )
        validation_ids = tuple(validation_plan.heldout_episode_ids)
        if config.validation_reservation_path is None:
            if not allow_engineering:
                raise ValueError(
                    "formal prospective validation requires a pre-collection "
                    "--validation-reservation"
                )
        else:
            validation_reservation = load_source_validation_reservation(
                config.validation_reservation_path,
                development_plan=plan,
            )
            observed_validation_seeds = tuple(
                validation_contract["reserved_pool_seed_envs"]
            )
            if observed_validation_seeds != validation_reservation.reserved_seed_envs:
                raise ValueError(
                    "prospective validation seeds do not exactly match their "
                    "pre-collection reservation"
                )
            if validation_reservation.method_schema != SOURCE_DISCRIMINATOR_SCHEMA:
                raise ValueError(
                    "prospective validation reservation targets another method schema"
                )
            if validation_reservation.run_spec != frozen_run_spec:
                raise ValueError(
                    "formal source-discriminator run settings differ from the "
                    "pre-collection reservation"
                )
            collection_spec = validation_reservation.collection_spec
            if validation_plan.payload["provenance"] != {
                "source_commit": collection_spec["source_commit"],
                "environment_fingerprint": collection_spec["environment_fingerprint"],
            }:
                raise ValueError(
                    "prospective validation plan provenance differs from its reservation"
                )
            reservation_hashes = {
                row.get("source_validation_reservation_sha256")
                for row in validation_plan.payload["pool"]
            }
            if reservation_hashes != {validation_reservation.artifact_sha256}:
                raise ValueError(
                    "prospective validation data were not collected under this reservation"
                )
            observed_bindings = {
                str(row["paired_binding_id"])
                for row in validation_plan.payload["pool"]
            }
            if observed_bindings != {collection_spec["paired_binding_id"]}:
                raise ValueError(
                    "prospective validation physical binding differs from its reservation"
                )
            heldout_id_set = set(validation_plan.heldout_episode_ids)
            observed_heldout_seeds = tuple(
                sorted(
                    {
                        int(row["seed_env"])
                        for row in validation_plan.payload["pool"]
                        if row["id"] in heldout_id_set
                    }
                )
            )
            observed_train_seeds = {
                int(row["seed_env"])
                for row in validation_plan.payload["pool"]
                if row["id"] in set(validation_plan.train_episode_ids)
            }
            if observed_heldout_seeds != validation_reservation.heldout_seed_envs:
                raise ValueError("prospective held-out seeds differ from their frozen roles")
            if observed_train_seeds != {validation_reservation.sacrificial_seed_env}:
                raise ValueError("prospective sacrificial seed differs from its frozen role")
            if (
                not allow_engineering
                and validation_reservation.release_status != "formal"
            ):
                raise ValueError(
                    "formal validation cannot consume an engineering-only seed reservation"
                )
            validation_contract["reservation_sha256"] = (
                validation_reservation.artifact_sha256
            )
            validation_contract["reservation_release_status"] = (
                validation_reservation.release_status
            )
            validation_contract["reserved_run_spec"] = (
                validation_reservation.run_spec
            )
    selected_ids = _selected_episode_ids(
        validation_plan,
        config,
        eligible_episode_ids=validation_ids,
    )
    (
        fit_ids,
        selection_ids,
        calibration_ids,
        selection_pair_ids,
        calibration_pair_ids,
        selection_seeds,
        calibration_seeds,
    ) = _selection_split_ids(plan, config)
    entries = entries_from_formal_plan(plan)
    validation_entries = entries_from_formal_plan(validation_plan)
    pair_index_by_episode, class_by_episode, fit_pair_ids = _training_pair_metadata(
        plan, config, fit_ids=fit_ids
    )
    train_dataset = RawPolicyWindowDataset(
        entries,
        k_hist=config.observation.k_hist,
        episode_ids=fit_ids,
        partition_indices=pair_index_by_episode,
    )
    selection_id_set = set(selection_ids)
    selection_datasets: list[tuple[str, RawPolicyWindowDataset]] = []
    for pair_id in selection_pair_ids:
        pair_episode_ids = tuple(
            sorted(
                str(row["id"])
                for row in plan.payload["pool"]
                if row["id"] in selection_id_set and str(row["pair_id"]) == pair_id
            )
        )
        if len(pair_episode_ids) != 2:
            raise RuntimeError("selection twin pair is incomplete")
        selection_datasets.append(
            (
                pair_id,
                RawPolicyWindowDataset(
                    entries,
                    k_hist=config.observation.k_hist,
                    episode_ids=pair_episode_ids,
                ),
            )
        )
    selected_dataset = RawPolicyWindowDataset(
        validation_entries,
        k_hist=config.observation.k_hist,
        episode_ids=selected_ids,
    )
    steps_per_epoch = config.optimizer_steps or math.ceil(
        len(train_dataset) / config.batch_size
    )
    start_state = _git_training_producer_state()
    if (
        not allow_engineering
        and validation_reservation is not None
        and validation_reservation.producer_git_state["head"] != start_state["head"]
    ):
        raise RuntimeError(
            "formal source-discriminator code changed after validation seeds were reserved"
        )
    if not allow_engineering and start_state["tracked_worktree_dirty"]:
        raise RuntimeError("formal source-discriminator training requires a clean tracked HEAD")

    device = torch.device(config.device)
    sampler = _PairClassBalancedSampler(
        train_dataset,
        pair_index_by_episode=pair_index_by_episode,
        class_by_episode=class_by_episode,
        num_samples=steps_per_epoch * config.batch_size,
        seed=config.seed,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_policy_samples,
    )
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model = _SourceDiscriminator(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_curve: list[float] = []
    selection_curve: list[dict[str, Any]] = []
    best_step = 0
    best_selection_metrics: dict[str, float] | None = None
    best_selection_key: tuple[float, ...] | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    log_group_weights = torch.zeros(len(fit_pair_ids), device=device)
    final_training_pair_losses: dict[int, float] = {}
    model.train()
    global_step = 0
    total_steps = config.epochs * steps_per_epoch
    print(
        f"Source estimator {config.fusion_mode}: {total_steps} updates, "
        f"{len(fit_pair_ids)} fit pairs, {len(selection_datasets)} selection pairs",
        flush=True,
    )
    for _epoch in range(config.epochs):
        iterator = iter(loader)
        for _step in range(steps_per_epoch):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            logits = model(_normalised_observation(batch, normalization, device))
            if batch.obs.partition_index is None:
                raise RuntimeError("pair-aware training batch lost its pair identity")
            loss, final_training_pair_losses = _pair_group_dro_loss(
                logits,
                _labels(batch, config, device),
                batch.obs.partition_index.to(device),
                log_group_weights,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("source discriminator produced a non-finite loss")
            loss.backward()
            optimizer.step()
            loss_curve.append(float(loss.detach().cpu()))
            global_step += 1
            if (
                global_step % config.selection_interval == 0
                or global_step == total_steps
            ):
                pair_outputs = []
                for pair_id, pair_dataset in selection_datasets:
                    pair_logits, pair_labels, _selection_features = _infer_raw(
                        model, pair_dataset, normalization, config, device
                    )
                    pair_outputs.append((pair_id, pair_logits, pair_labels))
                selection_metrics, selection_pair_metrics = _pair_selection_metrics(
                    pair_outputs
                )
                selection_key = _checkpoint_selection_key(selection_metrics)
                print(
                    f"Source estimator {config.fusion_mode}: step={global_step}/{total_steps}, "
                    f"loss={loss_curve[-1]:.5f}, selection={selection_metrics}",
                    flush=True,
                )
                selection_curve.append(
                    {
                        "step": global_step,
                        **selection_metrics,
                        "pair_metrics": selection_pair_metrics,
                    }
                )
                if best_selection_key is None or selection_key < best_selection_key:
                    best_step = global_step
                    best_selection_metrics = selection_metrics
                    best_selection_key = selection_key
                    best_model_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
                model.train()

    if best_model_state is None or best_selection_metrics is None or best_step < 1:
        raise RuntimeError("source discriminator produced no selection checkpoint")
    model.load_state_dict(best_model_state)

    calibration_id_set = set(calibration_ids)
    calibration_outputs = []
    for pair_id in calibration_pair_ids:
        pair_episode_ids = tuple(
            sorted(
                row["id"]
                for row in plan.payload["pool"]
                if row["id"] in calibration_id_set
                and str(row["pair_id"]) == pair_id
            )
        )
        if len(pair_episode_ids) != 2:
            raise RuntimeError("calibration twin pair is incomplete")
        pair_dataset = RawPolicyWindowDataset(
            entries,
            k_hist=config.observation.k_hist,
            episode_ids=pair_episode_ids,
        )
        calibration_outputs.append(
            _infer_raw(model, pair_dataset, normalization, config, device)
        )
    calibration_logits = np.concatenate([row[0] for row in calibration_outputs])
    calibration_labels = np.concatenate([row[1] for row in calibration_outputs])
    calibration_features = np.concatenate([row[2] for row in calibration_outputs])
    (
        calibration_regularization,
        calibration_robustness,
        calibration_cv,
    ) = _select_pair_robust_platt(
        tuple((row[0], row[1]) for row in calibration_outputs),
        allow_uninformative=allow_engineering,
    )
    selected_calibration_cv = next(
        row for row in calibration_cv if row.get("selected") is True
    )
    calibration_scale, calibration_bias = _fit_pair_robust_platt(
        tuple((row[0], row[1]) for row in calibration_outputs),
        regularization=calibration_regularization,
        robustness=calibration_robustness,
    )
    calibrated_selection_metrics = _classification_metrics(
        calibration_labels,
        _calibrated_probabilities(
            calibration_logits, scale=calibration_scale, bias=calibration_bias
        ),
    )
    validation_rows_by_pair: dict[str, list[dict[str, Any]]] = {}
    validation_id_set = set(validation_ids)
    for row in validation_plan.payload["pool"]:
        if row["id"] in validation_id_set:
            validation_rows_by_pair.setdefault(str(row["pair_id"]), []).append(row)
    pair_validation_outputs: list[
        tuple[str, int, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    for pair_id in sorted(validation_rows_by_pair):
        pair_rows = validation_rows_by_pair[pair_id]
        pair_ids = tuple(sorted(str(row["id"]) for row in pair_rows))
        pair_seeds = {int(row["seed_env"]) for row in pair_rows}
        if len(pair_rows) != 2 or len(pair_seeds) != 1:
            raise ValueError(
                f"validation pair {pair_id} is not one complete source/target twin"
            )
        pair_dataset = RawPolicyWindowDataset(
            validation_entries,
            k_hist=config.observation.k_hist,
            episode_ids=pair_ids,
        )
        pair_logits, pair_labels, pair_features = _infer_raw(
            model, pair_dataset, normalization, config, device
        )
        pair_validation_outputs.append(
            (pair_id, next(iter(pair_seeds)), pair_logits, pair_labels, pair_features)
        )
    validation_logits = np.concatenate([row[2] for row in pair_validation_outputs])
    labels = np.concatenate([row[3] for row in pair_validation_outputs])
    validation_features = np.concatenate([row[4] for row in pair_validation_outputs])
    probabilities = _calibrated_probabilities(
        validation_logits, scale=calibration_scale, bias=calibration_bias
    )
    metrics = _classification_metrics(labels, probabilities)
    _fit_logits, fit_labels, fit_features = _infer_raw(
        model, train_dataset, normalization, config, device
    )
    pair_metrics: list[dict[str, Any]] = []
    for pair_id, seed_env, pair_logits, pair_labels, pair_features in pair_validation_outputs:
        pair_probabilities = _calibrated_probabilities(
            pair_logits, scale=calibration_scale, bias=calibration_bias
        )
        pair_metric = _classification_metrics(pair_labels, pair_probabilities)
        pair_metric["information_gain_bits"] = _information_gain_bits(
            pair_labels, pair_probabilities
        )
        pair_metric.update(
            _support_diagnostics(
                fit_features=fit_features,
                fit_labels=fit_labels,
                calibration_features=calibration_features,
                calibration_labels=calibration_labels,
                validation_features=pair_features,
                validation_labels=pair_labels,
            )
        )
        pair_metric["error_budget_total"] = (
            pair_metric["ece"]
            + pair_metric["alpha_recovery_abs_error"]
            + pair_metric["w_support_gap"]
        )
        pair_metrics.append(
            {"pair_id": pair_id, "seed_env": seed_env, **pair_metric}
        )
    metrics.update(
        _support_diagnostics(
            fit_features=fit_features,
            fit_labels=fit_labels,
            calibration_features=calibration_features,
            calibration_labels=calibration_labels,
            validation_features=validation_features,
            validation_labels=labels,
        )
    )
    pooled_ece = metrics["ece"]
    pooled_alpha_error = metrics["alpha_recovery_abs_error"]
    pooled_information_gain = _information_gain_bits(labels, probabilities)
    pooled_support = {
        name: metrics[name]
        for name in (
            "w_support_gap",
            "calibration_off_support_fraction",
            "off_support_fraction",
            "in_support_fraction",
            "support_distance_threshold",
        )
    }
    worst_pair = max(pair_metrics, key=lambda row: row["error_budget_total"])
    metrics["pooled_ece"] = pooled_ece
    metrics["pooled_alpha_recovery_abs_error"] = pooled_alpha_error
    metrics["pooled_information_gain_bits"] = pooled_information_gain
    metrics["worst_pair_auc"] = min(row["auc"] for row in pair_metrics)
    metrics["worst_pair_ece"] = worst_pair["ece"]
    metrics["worst_pair_alpha_recovery_abs_error"] = worst_pair[
        "alpha_recovery_abs_error"
    ]
    metrics["worst_pair_support_gap"] = worst_pair["w_support_gap"]
    metrics["worst_pair_information_gain_bits"] = min(
        row["information_gain_bits"] for row in pair_metrics
    )
    metrics["validation_pair_count"] = len(pair_metrics)
    metrics["w_ece"] = metrics["worst_pair_ece"]
    metrics["w_alpha_recovery"] = metrics[
        "worst_pair_alpha_recovery_abs_error"
    ]
    metrics["w_support_gap"] = metrics["worst_pair_support_gap"]
    metrics["pooled_w_support_gap"] = pooled_support["w_support_gap"]
    metrics["error_budget_limit"] = DISCRIMINATOR_ERROR_BUDGET
    metrics["error_budget_total"] = (
        metrics["w_ece"]
        + metrics["w_alpha_recovery"]
        + metrics["w_support_gap"]
    )
    metrics["information_gain_bits"] = metrics["worst_pair_information_gain_bits"]
    metrics["information_gain_floor_bits"] = (
        DISCRIMINATOR_INFORMATION_GAIN_FLOOR_BITS
    )
    if metrics["auc"] < 0.5:
        raise RuntimeError(
            "source discriminator held-out AUC is below 0.5, indicating an orientation "
            "failure or split/domain confound; refusing to publish a source posterior"
        )
    if not allow_engineering and metrics["worst_pair_auc"] < 0.5:
        raise RuntimeError(
            "source discriminator has a prospective twin pair with AUC below 0.5"
        )
    if not allow_engineering and (
        metrics["information_gain_bits"] < metrics["information_gain_floor_bits"]
    ):
        raise RuntimeError(
            "source posterior is calibrated but not informative enough: "
            f"{metrics['information_gain_bits']:.6g} < "
            f"{metrics['information_gain_floor_bits']:.6g} bits over the constant prior"
        )
    if not allow_engineering and metrics["error_budget_total"] > metrics["error_budget_limit"]:
        raise RuntimeError(
            "source discriminator exceeds the preregistered absolute error budget: "
            f"{metrics['error_budget_total']:.6g} > {metrics['error_budget_limit']:.6g}"
        )
    posterior_episodes = _posterior_episodes(
        model,
        selected_dataset,
        normalization,
        config,
        device,
        calibration_scale=calibration_scale,
        calibration_bias=calibration_bias,
    )
    end_state = _git_training_producer_state()
    stable = start_state == end_state
    if not stable:
        raise RuntimeError("tracked source changed during source-discriminator training")
    release_status = "engineering_only" if allow_engineering else "formal"
    if release_status == "formal" and end_state["tracked_worktree_dirty"]:
        raise RuntimeError("formal source-discriminator output cannot bind a dirty HEAD")

    feature_spec = {
        "observation": asdict(config.observation),
        "d_model": config.d_model,
        "n_tasks": config.n_tasks,
        "architecture": "ObservationEncoder-modality-balanced-pooling-MLP-binary",
        "pooling": {
            "strategy": "mean-within-modality-then-learned-fusion",
            "visual_tokens": model.visual_tokens,
            "proprio_tokens": model.proprio_tokens,
            "task_tokens": model.task_tokens,
        },
    }
    if config.fusion_mode != "modality-mean":
        feature_spec["fusion_mode"] = config.fusion_mode
        feature_spec["architecture"] = (
            "ObservationEncoder-camera-time-fusion-MLP-binary"
            if config.fusion_mode == "camera-time"
            else "ObservationEncoder-meanpool-MLP-binary"
        )
        feature_spec["pooling"]["strategy"] = (
            "spatial-mean-per-camera-frame-then-ordered-fusion"
            if config.fusion_mode == "camera-time"
            else "mean-all-tokens"
        )
    training_split_sha = _json_sha(list(fit_ids))
    checkpoint = {
        "schema": SOURCE_DISCRIMINATOR_SCHEMA,
        "marker": _MARKER,
        "release_status": release_status,
        "plan_sha256": plan.plan_sha256,
        "validation_plan_sha256": validation_plan.plan_sha256,
        "normalization_sha256": normalization.artifact_sha256,
        "feature_spec": feature_spec,
        "feature_spec_sha256": _json_sha(feature_spec),
        "training_split_sha256": training_split_sha,
        "source_embodiment_indices": list(config.source_embodiment_indices),
        "target_embodiment_indices": list(config.target_embodiment_indices),
        "seed": config.seed,
        "epochs": config.epochs,
        "optimizer_steps_per_epoch": steps_per_epoch,
        "loss_curve": loss_curve,
        "training_objective": {
            "sampler": "equal-probability-complete-pair-and-source-target-class",
            "risk": "group-dro-over-pair-bce",
            "group_dro_step_size": _GROUP_DRO_STEP_SIZE,
            "pair_ids": list(fit_pair_ids),
            "final_group_weights": torch.softmax(
                log_group_weights.detach().cpu(), dim=0
            ).tolist(),
            "last_batch_pair_losses": {
                fit_pair_ids[index]: value
                for index, value in sorted(final_training_pair_losses.items())
            },
        },
        "selection": {
            "strategy": "disjoint-complete-twin-pairs-sha256-rank",
            "checkpoint_metric": (
                "orientation-admissible-then-worst-pair-bce-calibration-burden"
            ),
            "interval": config.selection_interval,
            "pair_ids": list(selection_pair_ids),
            "seed_envs": list(selection_seeds),
            "fit_episode_ids": list(fit_ids),
            "selection_episode_ids": list(selection_ids),
            "best_step": best_step,
            "best_metrics": best_selection_metrics,
            "curve": selection_curve,
        },
        "calibration": {
            "strategy": "pair-loo-selected-group-robust-platt-scaling",
            "pair_ids": list(calibration_pair_ids),
            "seed_envs": list(calibration_seeds),
            "episode_ids": list(calibration_ids),
            "selection_metric": (
                "worst_pair_ece_plus_alpha_error_subject_to_information_gain"
            ),
            "regularization": calibration_regularization,
            "robustness": calibration_robustness,
            "regularization_candidates": list(_PLATT_REGULARIZATION_CANDIDATES),
            "robustness_candidates": list(_GROUP_ROBUSTNESS_CANDIDATES),
            "information_gain_floor_bits": (
                DISCRIMINATOR_INFORMATION_GAIN_FLOOR_BITS
            ),
            "information_gate_passed": bool(selected_calibration_cv["informative"]),
            "pair_loo_cv": calibration_cv,
            "scale": calibration_scale,
            "bias": calibration_bias,
            "metrics": calibrated_selection_metrics,
        },
        "validation": validation_contract | {"pair_metrics": pair_metrics},
        "validation_metrics": metrics,
        "producer_git_state": end_state,
        "model_state": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
    }
    _atomic_torch_save(config.checkpoint_path, checkpoint)
    model_sha = _sha256(config.checkpoint_path)
    provenance = SourcePosteriorProvenance(
        plan_sha256=validation_plan.plan_sha256,
        model_sha256=model_sha,
        feature_spec_sha256=checkpoint["feature_spec_sha256"],
        training_split_sha256=training_split_sha,
        source_embodiment_indices=config.source_embodiment_indices,
        target_embodiment_indices=config.target_embodiment_indices,
        validation_metrics=metrics,
        release_status=release_status,
    )
    posterior_header = write_source_posterior(
        config.posterior_path,
        provenance=provenance,
        episodes=posterior_episodes,
    )
    return SourceDiscriminatorResult(
        checkpoint_path=config.checkpoint_path,
        posterior_path=config.posterior_path,
        posterior_header=posterior_header,
        release_status=release_status,
        epochs=config.epochs,
        optimizer_steps=config.epochs * steps_per_epoch,
        final_loss=loss_curve[best_step - 1],
        best_step=best_step,
        selection_metrics=best_selection_metrics,
        validation_metrics=metrics,
    )
