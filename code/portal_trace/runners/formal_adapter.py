"""Bounded execution adapter for the custom PORTAL-Trace policy.

This module is the bridge between a hash-bound :class:`FormalDataPlan` and the
policy registry.  It deliberately exposes a bounded probe rather than the
formal experiment runner: the probe proves that real policy windows, the
training-only transform, the registered arm and a masked BC update compose
without publishing a paper result.  ``runners.train`` remains gated until the
P0/data/calibration contracts are complete and a deployment checkpoint protocol
is implemented.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from portal_trace.policies.interface import Batch, Observation, Policy, PolicyConfig
from portal_trace.policies.registry import build
from portal_trace.runners.formal_data import (
    FormalDataPlan,
    FormalProprioNormalization,
    _verify_plan,
)
from portal_trace.runners.policy_dataset import (
    PolicyDatasetEntry,
    RawPolicyWindowDataset,
    collate_policy_samples,
)

__all__ = [
    "FORMAL_ADAPTER_PROBE_SCHEMA",
    "FORMAL_ADAPTER_PROBE_MARKER",
    "FormalAdapterConfig",
    "FormalAdapterResult",
    "entries_from_formal_plan",
    "measure_formal_adapter_throughput",
    "run_formal_adapter_probe",
    "load_formal_adapter_probe",
]

FORMAL_ADAPTER_PROBE_SCHEMA = "portal_trace.formal_adapter_probe/1"
FORMAL_ADAPTER_PROBE_MARKER = "PORTAL_TRACE_ADAPTER_PROBE_NOT_FORMAL_RESULT"
_MAX_PROBE_STEPS = 4


@dataclass(frozen=True)
class FormalAdapterConfig:
    """Inputs for one explicitly bounded, non-publication adapter probe."""

    plan: FormalDataPlan
    normalization: FormalProprioNormalization
    policy_config: PolicyConfig
    seed: int
    optimizer_steps: int = 1
    batch_size: int = 1
    learning_rate: float = 1e-3
    checkpoint_path: Path | None = None

    def __post_init__(self) -> None:
        _verify_plan(self.plan)
        if self.normalization.plan_sha256 != self.plan.plan_sha256:
            raise ValueError("formal adapter normalization is bound to a different data plan")
        if tuple(self.normalization.training_episode_ids) != tuple(self.plan.train_episode_ids):
            raise ValueError(
                "formal adapter normalization training split disagrees with the data plan"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("formal adapter seed must be an integer")
        if (
            isinstance(self.optimizer_steps, bool)
            or not isinstance(self.optimizer_steps, int)
            or self.optimizer_steps not in range(1, _MAX_PROBE_STEPS + 1)
        ):
            raise ValueError(f"formal adapter optimizer_steps must lie in [1, {_MAX_PROBE_STEPS}]")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise ValueError("formal adapter batch_size must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("formal adapter learning_rate must be finite and positive")
        if self.checkpoint_path is not None:
            object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))


@dataclass(frozen=True)
class FormalAdapterResult:
    checkpoint_path: Path | None
    final_loss: float
    optimizer_steps: int
    parameter_delta_l2: float
    train_windows: int
    heldout_windows: int
    heldout_action_mse: float
    heldout_decompensated_action_mse: float
    heldout_c_true_rms: float
    reference_output: torch.Tensor
    heldout_output: torch.Tensor


def entries_from_formal_plan(plan: FormalDataPlan) -> tuple[PolicyDatasetEntry, ...]:
    """Convert the plan's episode rows into validated file-level entries.

    ``FormalDataPlan`` rejects duplicate paths, so every path has one task and
    embodiment identity.  The largest declared demo index determines the
    prefix accepted by ``RawPolicyWindowDataset``; split selection below then
    removes held-out episodes exactly by their canonical IDs.
    """
    _verify_plan(plan)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in plan.payload["pool"]:
        grouped[str(Path(row["path"]).resolve())].append(row)
    entries: list[PolicyDatasetEntry] = []
    for path_text, rows in sorted(grouped.items()):
        identities = {(int(row["task_index"]), int(row["embodiment_index"])) for row in rows}
        if len(identities) != 1:
            raise ValueError(
                "formal plan assigns multiple task/embodiment identities to "
                f"{path_text}"
            )
        metadata = {
            (str(row.get("family", "")), str(row.get("defect", "")), row.get("level"))
            for row in rows
        }
        if len(metadata) != 1:
            raise ValueError(f"formal plan metadata disagrees across episodes for {path_text}")
        task_index, embodiment_index = next(iter(identities))
        demo_indices = [int(str(row["demo"]).rsplit("_", 1)[1]) for row in rows]
        entries.append(
            PolicyDatasetEntry(
                Path(path_text),
                task_index=task_index,
                embodiment_index=embodiment_index,
                n_demos=max(demo_indices) + 1,
                family=str(rows[0].get("family", "")),
                defect=str(rows[0].get("defect", "")),
                level=rows[0].get("level"),
            )
        )
    return tuple(entries)


def _partition_indices(plan: FormalDataPlan, n_partitions: int) -> dict[str, int]:
    if n_partitions < 1:
        raise ValueError("formal adapter partition count must be positive")
    return {
        row["id"]: int.from_bytes(
            hashlib.sha256(
                f"{plan.payload['seed']}\0{row['id']}".encode()
            ).digest()[:8],
            "big",
        )
        % n_partitions
        for row in plan.payload["pool"]
    }


def _normalise_batch(
    batch: Batch,
    normalization: FormalProprioNormalization,
    *,
    policy: Policy,
) -> Batch:
    proprio = batch.obs.proprio
    if proprio is None:
        raise ValueError("formal adapter dataset batch has no proprioception")
    obs = Observation(
        images=batch.obs.images,
        proprio=normalization.apply(proprio) if policy.cfg.obs.use_proprio else None,
        task_index=batch.obs.task_index,
        embodiment_index=batch.obs.embodiment_index,
        partition_index=batch.obs.partition_index,
    )
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


def _move_batch(batch: Batch, device: torch.device) -> Batch:
    """Move every tensor carried by a batch, including optional endpoint fields."""
    obs = Observation(
        images={name: value.to(device) for name, value in batch.obs.images.items()},
        proprio=None if batch.obs.proprio is None else batch.obs.proprio.to(device),
        task_index=None if batch.obs.task_index is None else batch.obs.task_index.to(device),
        embodiment_index=(
            None
            if batch.obs.embodiment_index is None
            else batch.obs.embodiment_index.to(device)
        ),
        partition_index=(
            None
            if batch.obs.partition_index is None
            else batch.obs.partition_index.to(device)
        ),
    )
    return Batch(
        obs=obs,
        actions=None if batch.actions is None else batch.actions.to(device),
        action_pad_mask=(
            None if batch.action_pad_mask is None else batch.action_pad_mask.to(device)
        ),
        c_true=None if batch.c_true is None else batch.c_true.to(device),
        sample_weight=None if batch.sample_weight is None else batch.sample_weight.to(device),
        phi_features=None if batch.phi_features is None else batch.phi_features.to(device),
        psi_basis=None if batch.psi_basis is None else batch.psi_basis.to(device),
        extras={name: value.to(device) for name, value in batch.extras.items()},
    )


def _formal_datasets(
    plan: FormalDataPlan, policy_config: PolicyConfig, *, episode_cache_bytes: int = 0
) -> tuple[RawPolicyWindowDataset, RawPolicyWindowDataset]:
    entries = entries_from_formal_plan(plan)
    partition_map = _partition_indices(plan, policy_config.n_pool_embodiments)
    common = {
        "entries": entries,
        "k_hist": policy_config.obs.k_hist,
        "partition_indices": partition_map,
        "episode_cache_bytes": episode_cache_bytes,
    }
    return (
        RawPolicyWindowDataset(episode_ids=plan.train_episode_ids, **common),
        RawPolicyWindowDataset(episode_ids=plan.heldout_episode_ids, **common),
    )


def measure_formal_adapter_throughput(
    plan: FormalDataPlan,
    normalization: FormalProprioNormalization,
    policy_config: PolicyConfig,
    *,
    seed: int,
    batch_size: int = 1,
    n_steps: int = 20,
    n_warmup: int = 5,
    device: str = "cuda",
) -> dict[str, Any]:
    """Measure optimizer steps/s on real plan-bound windows.

    This is a calibration measurement, not a training routine.  The caller must
    provide a real formal data plan and its matching normalization artifact;
    synthetic fixtures are suitable for tests but must never be used to write the
    repository lock.  Warm-up and measured steps are deterministic and no model
    checkpoint is published.
    """
    if normalization.plan_sha256 != plan.plan_sha256:
        raise ValueError("throughput normalization is bound to a different data plan")
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("throughput n_steps must be a positive integer")
    if isinstance(n_warmup, bool) or not isinstance(n_warmup, int) or n_warmup < 0:
        raise ValueError("throughput n_warmup must be a non-negative integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("throughput batch_size must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("throughput seed must be an integer")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA throughput measurement but CUDA is unavailable")
    train_dataset, _heldout_dataset = _formal_datasets(plan, policy_config)
    if batch_size > len(train_dataset):
        raise ValueError("throughput batch_size cannot exceed training windows")
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_policy_samples,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        policy = build(policy_config.arm, policy_config).to(target)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        iterator = iter(loader)

        def next_batch() -> Batch:
            nonlocal iterator
            try:
                raw = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw = next(iterator)
            return _move_batch(
                _normalise_batch(raw, normalization, policy=policy), target
            )

        policy.train()
        for _ in range(n_warmup):
            optimizer.zero_grad(set_to_none=True)
            output = policy(next_batch())
            if output.loss is None or not torch.isfinite(output.loss):
                raise RuntimeError("throughput warm-up received a missing or non-finite loss")
            output.loss.backward()
            optimizer.step()
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        started = time.perf_counter()
        for _ in range(n_steps):
            optimizer.zero_grad(set_to_none=True)
            output = policy(next_batch())
            if output.loss is None or not torch.isfinite(output.loss):
                raise RuntimeError("throughput run received a missing or non-finite loss")
            output.loss.backward()
            optimizer.step()
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        elapsed = time.perf_counter() - started
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("throughput timer returned an invalid duration")
    return {
        "steps_per_second": n_steps / elapsed,
        "seconds_per_step": elapsed / n_steps,
        "n_steps": n_steps,
        "n_warmup": n_warmup,
        "batch_size": batch_size,
        "device": str(target),
        "plan_sha256": plan.plan_sha256,
        "policy_config": _config_payload(policy_config),
        "train_windows": len(train_dataset),
    }


def _state_hash(payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"model_state", "content_sha256"}
    }
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    for name in sorted(payload["model_state"]):
        tensor = payload["model_state"][name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _atomic_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite formal adapter probe {path}")
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _config_payload(config: PolicyConfig) -> dict[str, Any]:
    return {
        "arm": config.arm,
        "n_tasks": config.n_tasks,
        "n_pool_embodiments": config.n_pool_embodiments,
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "ffn_dim": config.ffn_dim,
        "dropout": config.dropout,
        "augment": config.augment,
        "options": dict(config.options),
        "obs": {
            "cameras": list(config.obs.cameras),
            "image_size": config.obs.image_size,
            "k_hist": config.obs.k_hist,
            "use_proprio": config.obs.use_proprio,
            "use_task_token": config.obs.use_task_token,
        },
    }


def run_formal_adapter_probe(config: FormalAdapterConfig) -> FormalAdapterResult:
    """Run a real registered policy for at most four masked BC updates."""
    entries = entries_from_formal_plan(config.plan)
    partition_map = _partition_indices(config.plan, config.policy_config.n_pool_embodiments)
    train_dataset = RawPolicyWindowDataset(
        entries,
        k_hist=config.policy_config.obs.k_hist,
        episode_ids=config.plan.train_episode_ids,
        partition_indices=partition_map,
    )
    heldout_dataset = RawPolicyWindowDataset(
        entries,
        k_hist=config.policy_config.obs.k_hist,
        episode_ids=config.plan.heldout_episode_ids,
        partition_indices=partition_map,
    )
    if config.batch_size > len(train_dataset):
        raise ValueError("formal adapter batch_size cannot exceed training windows")
    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_policy_samples,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        policy = build(config.policy_config.arm, config.policy_config)
        optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
        before = {
            name: parameter.detach().cpu().to(torch.float64).clone()
            for name, parameter in policy.named_parameters()
        }
        iterator = iter(loader)
        final_loss = float("nan")
        policy.train()
        for _step in range(config.optimizer_steps):
            try:
                raw = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw = next(iterator)
            batch = _normalise_batch(raw, config.normalization, policy=policy)
            optimizer.zero_grad(set_to_none=True)
            output = policy(batch)
            if output.loss is None or not torch.isfinite(output.loss):
                raise RuntimeError("formal adapter policy returned a missing or non-finite loss")
            output.loss.backward()
            optimizer.step()
            final_loss = float(output.loss.detach().cpu())
        delta = math.sqrt(
            sum(
                float(
                    (parameter.detach().cpu().to(torch.float64) - before[name])
                    .square()
                    .sum()
                )
                for name, parameter in policy.named_parameters()
            )
        )
        if not math.isfinite(delta) or delta <= 0:
            raise RuntimeError("formal adapter optimizer did not update model parameters")
        policy.eval()
        reference_raw = collate_policy_samples([train_dataset[0]])
        reference_output = policy(
            _normalise_batch(reference_raw, config.normalization, policy=policy)
        ).action_chunk.detach().cpu()
        heldout_raw = collate_policy_samples([heldout_dataset[0]])
        heldout_output = policy(
            _normalise_batch(heldout_raw, config.normalization, policy=policy)
        ).action_chunk.detach().cpu()
        if (
            tuple(reference_output.shape)
            != (1, config.policy_config.chunk, config.policy_config.action_dim)
            or tuple(heldout_output.shape)
            != (1, config.policy_config.chunk, config.policy_config.action_dim)
            or not torch.isfinite(reference_output).all()
            or not torch.isfinite(heldout_output).all()
        ):
            raise RuntimeError(
                "formal adapter probe produced a non-finite or malformed evaluation output"
            )
        # Keep the adapter probe diagnostically useful without turning it into a
        # paper endpoint: evaluate every held-out window against both the stored
        # defective-teacher action and the clipped clean/decompensated target.
        # These are normalized-action MSEs and are explicitly marked non-formal by
        # the producer that publishes them.
        heldout_loader = DataLoader(
            heldout_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_policy_samples,
        )
        action_error = 0.0
        decomp_error = 0.0
        c_true_squared = 0.0
        valid_elements = 0
        for raw in heldout_loader:
            batch = _normalise_batch(raw, config.normalization, policy=policy)
            output = policy(batch).action_chunk
            if batch.actions is None or batch.c_true is None or batch.action_pad_mask is None:
                raise RuntimeError("formal adapter held-out batch is missing evaluation fields")
            mask = batch.action_pad_mask.unsqueeze(-1).expand_as(output)
            target = batch.actions
            decompensated = (target - batch.c_true).clamp(-1.0, 1.0)
            if not torch.isfinite(output).all():
                raise RuntimeError("formal adapter held-out output is non-finite")
            action_error += float(((output - target).square() * mask).sum().cpu())
            decomp_error += float(((output - decompensated).square() * mask).sum().cpu())
            c_true_squared += float((batch.c_true.square() * mask).sum().cpu())
            valid_elements += int(mask.sum().item())
        if valid_elements < 1:
            raise RuntimeError("formal adapter held-out split has no valid action elements")
        heldout_action_mse = action_error / valid_elements
        heldout_decompensated_action_mse = decomp_error / valid_elements
        heldout_c_true_rms = math.sqrt(c_true_squared / valid_elements)
        if not all(
            math.isfinite(value)
            for value in (
                heldout_action_mse,
                heldout_decompensated_action_mse,
                heldout_c_true_rms,
            )
        ):
            raise RuntimeError("formal adapter held-out metrics are non-finite")
        checkpoint_path = config.checkpoint_path
        if checkpoint_path is not None:
            payload: dict[str, Any] = {
                "schema": FORMAL_ADAPTER_PROBE_SCHEMA,
                "marker": FORMAL_ADAPTER_PROBE_MARKER,
                "seed": config.seed,
                "optimizer_steps": config.optimizer_steps,
                "plan_sha256": config.plan.plan_sha256,
                "normalization_plan_sha256": config.normalization.plan_sha256,
                "policy_config": _config_payload(config.policy_config),
                "model_state": policy.state_dict(),
            }
            payload["content_sha256"] = _state_hash(payload)
            _atomic_save(checkpoint_path, payload)
            loaded = load_formal_adapter_probe(
                checkpoint_path,
                policy_config=config.policy_config,
                expected_plan_sha256=config.plan.plan_sha256,
            )
            loaded.eval()
            reloaded = loaded(
                _normalise_batch(reference_raw, config.normalization, policy=loaded)
            ).action_chunk.detach().cpu()
            if not torch.allclose(reference_output, reloaded, rtol=0, atol=1e-6):
                raise RuntimeError(
                    "formal adapter probe checkpoint changed inference across reload"
                )
    return FormalAdapterResult(
        checkpoint_path=checkpoint_path,
        final_loss=final_loss,
        optimizer_steps=config.optimizer_steps,
        parameter_delta_l2=delta,
        train_windows=len(train_dataset),
        heldout_windows=len(heldout_dataset),
        heldout_action_mse=heldout_action_mse,
        heldout_decompensated_action_mse=heldout_decompensated_action_mse,
        heldout_c_true_rms=heldout_c_true_rms,
        reference_output=reference_output,
        heldout_output=heldout_output,
    )


def load_formal_adapter_probe(
    path: Path,
    *,
    policy_config: PolicyConfig,
    expected_plan_sha256: str | None = None,
) -> Policy:
    """Load only the bounded probe schema and reconstruct the registered policy."""
    try:
        document = torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"{path}: unreadable formal adapter probe") from exc
    if not isinstance(document, dict) or set(document) != {
        "schema", "marker", "seed", "optimizer_steps", "plan_sha256",
        "normalization_plan_sha256", "policy_config", "model_state", "content_sha256",
    }:
        raise ValueError("formal adapter probe fields are incomplete or unexpected")
    if (
        document["schema"] != FORMAL_ADAPTER_PROBE_SCHEMA
        or document["marker"] != FORMAL_ADAPTER_PROBE_MARKER
    ):
        raise ValueError("checkpoint is not the non-formal adapter probe schema")
    if expected_plan_sha256 is not None and document["plan_sha256"] != expected_plan_sha256:
        raise ValueError("formal adapter probe is bound to a different data plan")
    if document["normalization_plan_sha256"] != document["plan_sha256"]:
        raise ValueError("formal adapter probe normalization is bound to a different data plan")
    if (
        isinstance(document["seed"], bool)
        or not isinstance(document["seed"], int)
        or isinstance(document["optimizer_steps"], bool)
        or not isinstance(document["optimizer_steps"], int)
        or document["optimizer_steps"] not in range(1, _MAX_PROBE_STEPS + 1)
    ):
        raise ValueError("formal adapter probe seed or optimizer step count is invalid")
    if document["policy_config"] != _config_payload(policy_config):
        raise ValueError(
            "formal adapter probe policy config does not match requested loader config"
        )
    if document["content_sha256"] != _state_hash(document):
        raise ValueError("formal adapter probe content SHA-256 does not match")
    state = document["model_state"]
    if not isinstance(state, dict) or not state:
        raise ValueError("formal adapter probe model state is empty or invalid")
    if any(
        (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    ):
        raise ValueError("formal adapter probe model state contains non-finite values")
    policy = build(policy_config.arm, policy_config)
    try:
        policy.load_state_dict(document["model_state"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("formal adapter probe model state does not match policy config") from exc
    return policy
