"""Shared producer for engineering and formal registry-policy checkpoints.

Both paths consume a hash-bound formal data plan, train-only proprioception
normalisation, a registry policy and the same behaviour-cloning loop. The
engineering API remains explicitly opted in and writes ``engineering_only``;
the formal API is used by :mod:`runners.train` and additionally requires a
clean, stable producer HEAD. The data-plan source commit remains separately
preserved in checkpoint provenance.

``optimizer_steps`` is the number of updates *per epoch*.  The producer cycles
the training loader when necessary, reshuffling windows on every pass using an
independent seeded generator. Sequential sampling is an explicit reproduction
option. The actual total (``epochs * optimizer_steps``) and sampling settings
are recorded in the checkpoint.
"""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from portal_trace.policies.action_objective import (
    ACTION_LOSS_KINDS,
    CONTINUOUS_ACTION_DIMS,
    MSE_ACTION_LOSS,
    deployment_action_chunk,
    structured_action_loss,
)
from portal_trace.policies.interface import Batch, PolicyConfig
from portal_trace.policies.registry import build
from portal_trace.runners.formal_adapter import (
    _config_payload,
    _formal_datasets,
    _move_batch,
    _normalise_batch,
)
from portal_trace.runners.formal_checkpoint import (
    FORMAL_CHECKPOINT_MARKER,
    FORMAL_CHECKPOINT_SCHEMA,
    _content_sha256,
    _pool_bindings,
    _sha256,
    load_formal_checkpoint,
)
from portal_trace.runners.formal_data import (
    FormalDataPlan,
    FormalProprioNormalization,
    _verify_plan,
    load_formal_data_plan,
    load_formal_normalization,
)
from portal_trace.runners.policy_dataset import collate_policy_samples

__all__ = [
    "ENGINEERING_FORMAL_PRODUCER_MARKER",
    "FORMAL_RELEASE_PRODUCER_MARKER",
    "FormalTrainingConfig",
    "FormalTrainingResult",
    "produce_formal_checkpoint",
    "run_formal_release_training",
    "run_formal_training",
    "evaluate_formal_checkpoint",
]


# This marker is returned by the producer API, while the on-disk marker remains
# the stable admission value required by ``formal_checkpoint/1``.  Keeping the
# two separate makes it impossible to mistake a loaded artifact for a paper
# result merely because it passed the deployment loader.
ENGINEERING_FORMAL_PRODUCER_MARKER = (
    "PORTAL_TRACE_ENGINEERING_FORMAL_PRODUCER_NOT_PAPER_RESULT"
)
FORMAL_RELEASE_PRODUCER_MARKER = "PORTAL_TRACE_FORMAL_RELEASE_PRODUCER"


def _cpu_state(value: Any) -> Any:
    """Copy optimizer state to CPU so checkpoints are portable across devices."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(item) for item in value)
    return value


def _git_training_producer_state() -> dict[str, Any]:
    """Capture tracked changes plus untracked executable project sources."""
    repository = Path(__file__).resolve().parents[2]

    def git(*args: str) -> bytes:
        try:
            completed = subprocess.run(
                ("git", "-C", str(repository), *args),
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError("cannot capture formal training producer Git state") from exc
        return completed.stdout

    head = git("rev-parse", "HEAD").decode("ascii").strip()
    status = git("status", "--porcelain=v1", "--untracked-files=no")
    tracked_diff = git("diff", "--binary", "HEAD", "--")
    untracked_names = tuple(
        name
        for name in git(
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "portal_trace",
            "scripts",
        ).split(b"\0")
        if name
    )
    untracked_digest = hashlib.sha256()
    for encoded_name in sorted(untracked_names):
        relative = encoded_name.decode("utf-8")
        path = repository / relative
        if not path.is_file():
            raise RuntimeError(f"untracked producer source is unavailable: {relative}")
        untracked_digest.update(encoded_name)
        untracked_digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                untracked_digest.update(block)
        untracked_digest.update(b"\0")
    dirty = bool(status or untracked_names)
    return {
        "head": head,
        "tracked_worktree_dirty": dirty,
        "tracked_diff_sha256": (
            hashlib.sha256(
                status
                + b"\0"
                + tracked_diff
                + b"\0untracked-project-sources\0"
                + untracked_digest.digest()
            ).hexdigest()
            if dirty
            else None
        ),
    }


def _training_producer_provenance(
    start: Mapping[str, Any], end: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "capture": "git-head+tracked-and-untracked-project-sources-v2",
        "start": dict(start),
        "end": dict(end),
        "stable": dict(start) == dict(end),
    }


def _atomic_save(path: Path, payload: dict[str, Any]) -> None:
    """Publish one checkpoint without replacing an existing path."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite formal checkpoint {path}")
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # link(2) is an atomic no-replace operation on the same filesystem.
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_file_binding(path: Path, expected: Any, *, kind: str) -> None:
    """Ensure the supplied in-memory artifact is exactly the file being bound."""
    if not path.is_file():
        raise ValueError(f"formal training {kind} artifact does not exist: {path}")
    if kind == "plan":
        loaded = load_formal_data_plan(path)
        if loaded != expected:
            raise ValueError("formal training plan object does not match plan_path")
    elif kind == "normalization":
        loaded = load_formal_normalization(path, plan=expected[0])
        if loaded != expected[1]:
            raise ValueError(
                "formal training normalization object does not match normalization_path"
            )
    else:  # pragma: no cover - private helper is called with fixed kinds
        raise AssertionError(f"unknown formal training binding kind {kind!r}")


def _normalization_payload_equal(
    left: FormalProprioNormalization, right: FormalProprioNormalization
) -> bool:
    """Compare normalization data while allowing a pre-write artifact hash of ``None``."""
    return (
        left.means == right.means
        and left.scales == right.scales
        and left.training_episode_ids == right.training_episode_ids
        and left.fitted_rows == right.fitted_rows
        and left.plan_sha256 == right.plan_sha256
    )


@dataclass(frozen=True)
class FormalTrainingConfig:
    """Inputs for an explicitly opted-in engineering formal training run."""

    plan: FormalDataPlan
    normalization: FormalProprioNormalization
    plan_path: Path
    normalization_path: Path
    checkpoint_path: Path
    policy_config: PolicyConfig
    seed: int
    epochs: int = 1
    optimizer_steps: int = 1
    batch_size: int = 1
    learning_rate: float = 1e-3
    device: str = "cpu"
    loss_kind: str = MSE_ACTION_LOSS
    continuous_loss_weight: float = 1.0
    gripper_loss_weight: float = 1.0
    window_sampling: str = "shuffled"
    sampling_seed: int | None = None
    episode_cache_bytes: int = 0

    def __post_init__(self) -> None:
        _verify_plan(self.plan)
        if self.normalization.plan_sha256 != self.plan.plan_sha256:
            raise ValueError("formal training normalization is bound to a different data plan")
        if tuple(self.normalization.training_episode_ids) != tuple(self.plan.train_episode_ids):
            raise ValueError(
                "formal training normalization training split disagrees with the data plan"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("formal training seed must be an integer")
        if self.window_sampling not in {"shuffled", "sequential"}:
            raise ValueError("window_sampling must be shuffled or sequential")
        sampling_seed = self.seed % (2 ** 64) if self.sampling_seed is None else self.sampling_seed
        if (
            isinstance(sampling_seed, bool)
            or not isinstance(sampling_seed, int)
            or not 0 <= sampling_seed < 2 ** 64
        ):
            raise ValueError("sampling_seed must be an integer in [0, 2**64)")
        object.__setattr__(self, "sampling_seed", sampling_seed)
        if (
            isinstance(self.episode_cache_bytes, bool)
            or not isinstance(self.episode_cache_bytes, int)
            or self.episode_cache_bytes < 0
        ):
            raise ValueError("episode_cache_bytes must be a non-negative integer")
        for name, value in (("epochs", self.epochs), ("optimizer_steps", self.optimizer_steps)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"formal training {name} must be a positive integer")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise TypeError("formal training batch_size must be an integer")
        if self.batch_size < 1:
            raise ValueError("formal training batch_size must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("formal training learning_rate must be finite and positive")
        if self.loss_kind not in ACTION_LOSS_KINDS:
            raise ValueError(f"unsupported formal training loss kind {self.loss_kind!r}")
        if self.loss_kind != MSE_ACTION_LOSS and self.policy_config.arm not in {
            "B1-target-only",
            "B2-naive-pooling",
            "B7-oracle",
        }:
            raise ValueError(
                "structured action loss is currently admitted only for B1-target-only, "
                "B2-naive-pooling, and the matched B7-oracle engineering comparison"
            )
        for name, value in (
            ("continuous_loss_weight", self.continuous_loss_weight),
            ("gripper_loss_weight", self.gripper_loss_weight),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"formal training {name} must be finite and positive")
        try:
            target = torch.device(self.device)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"formal training device is invalid: {self.device!r}") from exc
        if target.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA formal training but CUDA is unavailable")
        if target.type not in {"cpu", "cuda"}:
            raise ValueError("formal training device must be CPU or CUDA")
        object.__setattr__(self, "plan_path", Path(self.plan_path).resolve())
        object.__setattr__(self, "normalization_path", Path(self.normalization_path).resolve())
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path).resolve())
        _validate_file_binding(self.plan_path, self.plan, kind="plan")
        # ``fit_formal_normalization`` intentionally returns an in-memory value
        # before its artifact hash exists.  Re-load the immutable file here so
        # the producer always records the exact on-disk normalization hash.
        loaded_normalization = load_formal_normalization(self.normalization_path, plan=self.plan)
        if not _normalization_payload_equal(self.normalization, loaded_normalization):
            raise ValueError(
                "formal training normalization object does not match normalization_path"
            )
        object.__setattr__(self, "normalization", loaded_normalization)
        if (
            self.checkpoint_path == self.plan_path
            or self.checkpoint_path == self.normalization_path
        ):
            raise ValueError("formal training checkpoint path must differ from input artifacts")
        max_task = max(int(row["task_index"]) for row in self.plan.payload["pool"])
        max_embodiment = max(int(row["embodiment_index"]) for row in self.plan.payload["pool"])
        if max_task >= self.policy_config.n_tasks:
            raise ValueError("formal training policy config cannot represent all plan tasks")
        if max_embodiment >= self.policy_config.n_pool_embodiments:
            raise ValueError(
                "formal training policy config cannot represent all plan embodiments"
            )


@dataclass(frozen=True)
class FormalTrainingResult:
    """Engineering evidence returned after a checkpoint round trip."""

    checkpoint_path: Path
    marker: str
    epochs: int
    optimizer_steps: int
    final_loss: float
    loss_curve: tuple[float, ...]
    train_windows: int
    heldout_windows: int
    reference_output: torch.Tensor
    heldout_output: torch.Tensor
    heldout_action_mse: float
    heldout_action_mse_per_dim: tuple[float, ...]
    heldout_decompensated_action_mse: float
    heldout_c_true_rms: float
    heldout_c_true_rms_per_dim: tuple[float, ...]
    heldout_first_step_action_mse_per_dim: tuple[float, ...]
    heldout_first_step_prediction_rms_per_dim: tuple[float, ...]
    heldout_first_step_target_rms_per_dim: tuple[float, ...]
    heldout_first_step_gripper_sign_accuracy: float
    action_loss: Mapping[str, Any]


def _fit_continuous_action_rms(
    train_dataset: Any,
) -> tuple[tuple[float, ...], int]:
    """Fit six action RMS scales from the admitted training split only."""
    squared, valid_rows = train_dataset.action_squared_sum(CONTINUOUS_ACTION_DIMS)
    if valid_rows < 1:
        raise RuntimeError("training split has no valid actions for RMS scale fitting")
    rms = tuple(math.sqrt(float(value) / valid_rows) for value in squared)
    if not all(math.isfinite(value) and value > 0.0 for value in rms):
        raise RuntimeError("training-only continuous action RMS scales are not positive")
    return rms, valid_rows


def _action_loss_metadata(
    config: FormalTrainingConfig,
    *,
    continuous_rms: tuple[float, ...] | None,
    fitted_rows: int,
) -> dict[str, Any]:
    return {
        "kind": config.loss_kind,
        "continuous_dims": list(range(CONTINUOUS_ACTION_DIMS)),
        "gripper_dim": 6,
        "continuous_weight": config.continuous_loss_weight,
        "gripper_weight": config.gripper_loss_weight,
        "continuous_rms": list(continuous_rms) if continuous_rms is not None else None,
        "deployment_gripper": (
            "sign_logit_to_pm1"
            if config.loss_kind != MSE_ACTION_LOSS
            else "identity"
        ),
        "scale_provenance": (
            {
                "split": "train",
                "plan_sha256": config.plan.plan_sha256,
                "training_episode_ids": list(config.plan.train_episode_ids),
                "fitted_action_rows": fitted_rows,
            }
            if continuous_rms is not None
            else None
        ),
    }


def evaluate_formal_checkpoint(
    *,
    checkpoint_path: Path,
    plan_path: Path,
    normalization_path: Path,
    policy_config: PolicyConfig,
    expected_arm_id: str | None = None,
    device: str = "cpu",
    batch_size: int = 32,
) -> dict[str, Any]:
    """Measure one admitted engineering checkpoint on the complete held-out split.

    This is intentionally a measurement primitive, not a deployment evaluator:
    it has no simulator rollout and its returned values remain engineering-only.
    The checkpoint loader still enforces the exact plan, normalization, policy,
    and source provenance bindings before any window is evaluated.
    """
    plan = load_formal_data_plan(plan_path)
    normalization = load_formal_normalization(normalization_path, plan=plan)
    try:
        target = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"checkpoint evaluation device is invalid: {device!r}") from exc
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA checkpoint evaluation but CUDA is unavailable")
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("checkpoint evaluation device must be CPU or CUDA")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("checkpoint evaluation batch_size must be a positive integer")
    loaded = load_formal_checkpoint(
        checkpoint_path,
        plan_path=plan_path,
        normalization_path=normalization_path,
        policy_config=policy_config,
        expected_arm_id=expected_arm_id,
        expected_source_commit=plan.payload["provenance"]["source_commit"],
        allow_engineering=True,
    )
    action_loss = loaded.metadata["training"]["loss"]
    loss_kind = str(action_loss["kind"])
    continuous_rms = (
        None
        if loss_kind == MSE_ACTION_LOSS
        else torch.as_tensor(action_loss["continuous_rms"], device=target)
    )
    policy = loaded.policy.to(target)
    _train_dataset, heldout_dataset = _formal_datasets(plan, policy_config)
    heldout_ids = frozenset(plan.heldout_episode_ids)
    group_by_episode = {
        str(row["id"]): str(row.get("defect", "unspecified"))
        for row in plan.payload["pool"]
        if str(row["id"]) in heldout_ids
    }
    heldout_window_groups = tuple(
        group_by_episode[heldout_dataset.window_identity(index)[0]]
        for index in range(len(heldout_dataset))
    )
    heldout_group_stats: dict[str, dict[str, Any]] = {
        group: {
            "windows": 0,
            "valid_action_rows": 0,
            "action_error_per_dim": torch.zeros(
                policy_config.action_dim, dtype=torch.float64
            ),
            "c_true_squared_per_dim": torch.zeros(
                policy_config.action_dim, dtype=torch.float64
            ),
            "first_step_rows": 0,
            "gripper_sign_correct": 0,
        }
        for group in sorted(set(heldout_window_groups))
    }
    window_offset = 0
    loader = DataLoader(
        heldout_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_policy_samples,
    )
    action_error = 0.0
    action_error_per_dim = torch.zeros(
        policy_config.action_dim, dtype=torch.float64
    )
    decomp_error = 0.0
    c_true_squared = 0.0
    c_true_squared_per_dim = torch.zeros_like(action_error_per_dim)
    valid_elements = 0
    first_step_error_per_dim = torch.zeros_like(action_error_per_dim)
    first_step_prediction_squared_per_dim = torch.zeros_like(action_error_per_dim)
    first_step_target_squared_per_dim = torch.zeros_like(action_error_per_dim)
    first_step_rows = 0
    gripper_sign_correct = 0
    policy.eval()
    with torch.no_grad():
        for raw in loader:
            batch = _move_batch(
                _normalise_batch(raw, normalization, policy=policy), target
            )
            output = policy(batch)
            prediction = deployment_action_chunk(
                output.action_chunk,
                loss_kind=loss_kind,
                continuous_rms=continuous_rms,
            )
            if (
                batch.actions is None
                or batch.c_true is None
                or batch.action_pad_mask is None
                or prediction.shape != batch.actions.shape
            ):
                raise RuntimeError("checkpoint held-out batch is missing evaluation fields")
            mask = batch.action_pad_mask.unsqueeze(-1).expand_as(prediction)
            decompensated = (batch.actions - batch.c_true).clamp(-1.0, 1.0)
            if not torch.isfinite(prediction).all():
                raise RuntimeError("checkpoint held-out output is non-finite")
            action_error += float(
                ((prediction - batch.actions).square() * mask).sum()
            )
            action_error_per_dim += (
                ((prediction - batch.actions).square() * mask)
                .sum(dim=(0, 1))
                .detach()
                .cpu()
                .to(torch.float64)
            )
            decomp_error += float(
                ((prediction - decompensated).square() * mask).sum()
            )
            c_true_squared += float((batch.c_true.square() * mask).sum())
            c_true_squared_per_dim += (
                (batch.c_true.square() * mask)
                .sum(dim=(0, 1))
                .detach()
                .cpu()
                .to(torch.float64)
            )
            batch_size_actual = int(prediction.shape[0])
            batch_groups = heldout_window_groups[
                window_offset : window_offset + batch_size_actual
            ]
            if len(batch_groups) != batch_size_actual:
                raise RuntimeError("checkpoint held-out group alignment is incomplete")
            for group in set(batch_groups):
                row_selector = torch.as_tensor(
                    [label == group for label in batch_groups],
                    dtype=torch.bool,
                    device=prediction.device,
                )
                group_prediction = prediction[row_selector]
                group_actions = batch.actions[row_selector]
                group_c_true = batch.c_true[row_selector]
                group_pad_mask = batch.action_pad_mask[row_selector]
                group_mask = group_pad_mask.unsqueeze(-1).expand_as(group_prediction)
                stats = heldout_group_stats[group]
                stats["windows"] += int(row_selector.sum().item())
                stats["valid_action_rows"] += int(group_pad_mask.sum().item())
                stats["action_error_per_dim"] += (
                    ((group_prediction - group_actions).square() * group_mask)
                    .sum(dim=(0, 1))
                    .detach()
                    .cpu()
                    .to(torch.float64)
                )
                stats["c_true_squared_per_dim"] += (
                    (group_c_true.square() * group_mask)
                    .sum(dim=(0, 1))
                    .detach()
                    .cpu()
                    .to(torch.float64)
                )
                group_first_mask = group_pad_mask[:, 0]
                group_first_prediction = group_prediction[group_first_mask, 0, :]
                group_first_target = group_actions[group_first_mask, 0, :]
                stats["first_step_rows"] += int(group_first_mask.sum().item())
                stats["gripper_sign_correct"] += int(
                    (
                        (group_first_prediction[:, -1] >= 0)
                        == (group_first_target[:, -1] >= 0)
                    )
                    .sum()
                    .item()
                )
            window_offset += batch_size_actual
            valid_elements += int(mask.sum().item())
            first_mask = batch.action_pad_mask[:, 0]
            first_prediction = prediction[first_mask, 0, :]
            first_target = batch.actions[first_mask, 0, :]
            first_step_error_per_dim += (
                (first_prediction - first_target)
                .square()
                .sum(dim=0)
                .detach()
                .cpu()
                .to(torch.float64)
            )
            first_step_prediction_squared_per_dim += (
                first_prediction.square().sum(dim=0).detach().cpu().to(torch.float64)
            )
            first_step_target_squared_per_dim += (
                first_target.square().sum(dim=0).detach().cpu().to(torch.float64)
            )
            first_step_rows += int(first_mask.sum().item())
            gripper_sign_correct += int(
                ((first_prediction[:, -1] >= 0) == (first_target[:, -1] >= 0))
                .sum()
                .item()
            )
    if valid_elements < 1:
        raise RuntimeError("checkpoint held-out split has no valid action elements")
    if first_step_rows < 1:
        raise RuntimeError("checkpoint held-out split has no first-step actions")
    if window_offset != len(heldout_dataset):
        raise RuntimeError("checkpoint held-out group alignment did not consume all windows")
    valid_action_rows = valid_elements // policy_config.action_dim
    heldout_action_mse_per_dim = [
        float(value) for value in action_error_per_dim / valid_action_rows
    ]
    heldout_first_step_action_mse_per_dim = [
        float(value) for value in first_step_error_per_dim / first_step_rows
    ]
    heldout_first_step_prediction_rms_per_dim = [
        math.sqrt(float(value) / first_step_rows)
        for value in first_step_prediction_squared_per_dim
    ]
    heldout_first_step_target_rms_per_dim = [
        math.sqrt(float(value) / first_step_rows)
        for value in first_step_target_squared_per_dim
    ]
    heldout_first_step_gripper_sign_accuracy = (
        gripper_sign_correct / first_step_rows
    )
    heldout_c_true_rms_per_dim = [
        math.sqrt(float(value) / valid_action_rows)
        for value in c_true_squared_per_dim
    ]
    heldout_by_defect_group: dict[str, dict[str, Any]] = {}
    for group, stats in heldout_group_stats.items():
        group_valid_rows = int(stats["valid_action_rows"])
        group_first_rows = int(stats["first_step_rows"])
        if group_valid_rows < 1 or group_first_rows < 1:
            raise RuntimeError(f"checkpoint held-out group {group!r} is empty")
        group_action_mse_per_dim = [
            float(value) / group_valid_rows
            for value in stats["action_error_per_dim"]
        ]
        group_c_true_rms_per_dim = [
            math.sqrt(float(value) / group_valid_rows)
            for value in stats["c_true_squared_per_dim"]
        ]
        heldout_by_defect_group[group] = {
            "windows": int(stats["windows"]),
            "valid_action_rows": group_valid_rows,
            "action_mse": sum(group_action_mse_per_dim) / policy_config.action_dim,
            "action_mse_per_dim": group_action_mse_per_dim,
            "continuous_action_rmse": math.sqrt(
                sum(group_action_mse_per_dim[:CONTINUOUS_ACTION_DIMS])
                / CONTINUOUS_ACTION_DIMS
            ),
            "c_true_rms": math.sqrt(
                sum(value * value for value in group_c_true_rms_per_dim)
                / policy_config.action_dim
            ),
            "c_true_rms_per_dim": group_c_true_rms_per_dim,
            "first_step_gripper_sign_accuracy": (
                int(stats["gripper_sign_correct"]) / group_first_rows
            ),
        }
    values = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "plan_sha256": loaded.plan.plan_sha256,
        "plan_file_sha256": _sha256(Path(plan_path)),
        "normalization_sha256": _sha256(Path(normalization_path)),
        "normalization_artifact_sha256": loaded.normalization.artifact_sha256,
        "seed": int(loaded.metadata["seed"]),
        "epochs": int(loaded.metadata["training"]["epochs"]),
        "optimizer_steps": int(loaded.metadata["training"]["optimizer_steps"]),
        "final_loss": float(loaded.metadata["training"]["final_loss"]),
        "action_loss": dict(action_loss),
        "heldout_windows": len(heldout_dataset),
        "heldout_action_mse": action_error / valid_elements,
        "heldout_action_mse_per_dim": heldout_action_mse_per_dim,
        "heldout_decompensated_action_mse": decomp_error / valid_elements,
        "heldout_c_true_rms": math.sqrt(c_true_squared / valid_elements),
        "heldout_c_true_rms_per_dim": heldout_c_true_rms_per_dim,
        "heldout_by_defect_group": heldout_by_defect_group,
        "heldout_first_step_action_mse_per_dim": (
            heldout_first_step_action_mse_per_dim
        ),
        "heldout_first_step_prediction_rms_per_dim": (
            heldout_first_step_prediction_rms_per_dim
        ),
        "heldout_first_step_target_rms_per_dim": (
            heldout_first_step_target_rms_per_dim
        ),
        "heldout_first_step_gripper_sign_accuracy": (
            heldout_first_step_gripper_sign_accuracy
        ),
    }
    finite_fields = (
        "final_loss",
        "heldout_action_mse",
        "heldout_decompensated_action_mse",
        "heldout_c_true_rms",
    )
    if not all(math.isfinite(float(values[field])) for field in finite_fields):
        raise RuntimeError("checkpoint held-out metrics are non-finite")
    diagnostic_values = (
        *heldout_action_mse_per_dim,
        *heldout_first_step_action_mse_per_dim,
        *heldout_first_step_prediction_rms_per_dim,
        *heldout_first_step_target_rms_per_dim,
        *heldout_c_true_rms_per_dim,
        *(
            value
            for stats in heldout_by_defect_group.values()
            for field in (
                "action_mse",
                "continuous_action_rmse",
                "c_true_rms",
                "first_step_gripper_sign_accuracy",
            )
            for value in (float(stats[field]),)
        ),
        heldout_first_step_gripper_sign_accuracy,
    )
    if not all(math.isfinite(value) for value in diagnostic_values):
        raise RuntimeError("checkpoint held-out diagnostics are non-finite")
    return values


def run_formal_training(
    config: FormalTrainingConfig,
    *,
    allow_engineering_formal: bool = False,
) -> FormalTrainingResult:
    """Train a registry policy and atomically publish an engineering checkpoint.

    The explicit keyword is required even for temporary fixtures.  This API is
    intentionally absent from ``runners.train`` and the CLI, and it never
    publishes result tables or modifies the calibration registry.
    """
    if allow_engineering_formal is not True:
        raise PermissionError(
            "formal producer is engineering-only; pass allow_engineering_formal=True "
            "explicitly and do not treat its output as a paper result"
        )
    return _run_formal_training(
        config,
        release_status="engineering_only",
        result_marker=ENGINEERING_FORMAL_PRODUCER_MARKER,
    )


def run_formal_release_training(config: FormalTrainingConfig) -> FormalTrainingResult:
    """Publish a deployment-admissible checkpoint from one clean, stable HEAD.

    This is intentionally separate from the engineering opt-in.  A formal run
    cannot inherit the engineering-only release status or producer marker, and
    it refuses a dirty tracked worktree before allocating the model or reading
    batches.  The checkpoint loader repeats the same provenance admission check
    after publication.
    """
    producer_start = _git_training_producer_state()
    if producer_start["tracked_worktree_dirty"]:
        raise RuntimeError(
            "formal checkpoint publication requires a clean tracked worktree"
        )
    return _run_formal_training(
        config,
        release_status="formal",
        result_marker=FORMAL_RELEASE_PRODUCER_MARKER,
        producer_start=producer_start,
    )


def _run_formal_training(
    config: FormalTrainingConfig,
    *,
    release_status: str,
    result_marker: str,
    producer_start: Mapping[str, Any] | None = None,
) -> FormalTrainingResult:
    """Shared registry-policy optimizer used by engineering and formal producers."""
    if release_status not in {"engineering_only", "formal"}:
        raise ValueError(f"invalid formal training release status {release_status!r}")
    if config.checkpoint_path.exists():
        raise FileExistsError(f"refusing to overwrite formal checkpoint {config.checkpoint_path}")
    if producer_start is None:
        producer_start = _git_training_producer_state()

    train_dataset, heldout_dataset = _formal_datasets(
        config.plan, config.policy_config, episode_cache_bytes=config.episode_cache_bytes
    )
    if config.batch_size > len(train_dataset):
        raise ValueError("formal training batch_size cannot exceed training windows")
    continuous_rms: tuple[float, ...] | None = None
    fitted_action_rows = 0
    if config.loss_kind != MSE_ACTION_LOSS:
        continuous_rms, fitted_action_rows = _fit_continuous_action_rms(train_dataset)
    action_loss = _action_loss_metadata(
        config,
        continuous_rms=continuous_rms,
        fitted_rows=fitted_action_rows,
    )
    sampling_generator = torch.Generator()
    sampling_generator.manual_seed(config.sampling_seed)
    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.window_sampling == "shuffled",
        generator=sampling_generator,
        num_workers=0,
        collate_fn=collate_policy_samples,
    )
    target = torch.device(config.device)
    total_steps = config.epochs * config.optimizer_steps
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        if target.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        policy = build(config.policy_config.arm, config.policy_config).to(target)
        optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
        policy.train()
        iterator = iter(loader)
        final_loss = float("nan")
        loss_curve: list[float] = []
        training_start = time.monotonic()
        print(
            f"Training {config.policy_config.arm}: updates={total_steps}, "
            f"batch_size={config.batch_size}, sampling={config.window_sampling}, "
            f"sampling_seed={config.sampling_seed}",
            flush=True,
        )
        for _epoch in range(config.epochs):
            for _step in range(config.optimizer_steps):
                try:
                    raw = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    raw = next(iterator)
                batch: Batch = _move_batch(
                    _normalise_batch(raw, config.normalization, policy=policy), target
                )
                optimizer.zero_grad(set_to_none=True)
                output = policy(batch)
                if output.loss is None or not torch.isfinite(output.loss):
                    raise RuntimeError(
                        "formal training policy returned a missing or non-finite loss"
                    )
                loss = output.loss
                if config.loss_kind != MSE_ACTION_LOSS:
                    if (
                        continuous_rms is None
                        or batch.actions is None
                        or batch.action_pad_mask is None
                    ):
                        raise RuntimeError("structured action loss inputs are unavailable")
                    loss, continuous_loss, gripper_loss = structured_action_loss(
                        output.action_chunk,
                        batch.actions,
                        continuous_rms=torch.as_tensor(
                            continuous_rms,
                            dtype=output.action_chunk.dtype,
                            device=output.action_chunk.device,
                        ),
                        pad_mask=batch.action_pad_mask,
                        sample_weight=batch.sample_weight,
                        continuous_weight=config.continuous_loss_weight,
                        gripper_weight=config.gripper_loss_weight,
                        loss_kind=config.loss_kind,
                    )
                    if not all(
                        torch.isfinite(term)
                        for term in (loss, continuous_loss, gripper_loss)
                    ):
                        raise RuntimeError("structured action loss is non-finite")
                loss.backward()
                optimizer.step()
                final_loss = float(loss.detach().cpu())
                loss_curve.append(final_loss)
                completed_steps = _epoch * config.optimizer_steps + _step + 1
                if completed_steps % 100 == 0 or completed_steps == total_steps:
                    print(
                        f"{config.policy_config.arm}: {completed_steps}/{total_steps} updates, "
                        f"loss={final_loss:.6g}, "
                        f"elapsed={time.monotonic() - training_start:.1f}s",
                        flush=True,
                    )
        if not math.isfinite(final_loss):  # pragma: no cover - guarded by loop and config
            raise RuntimeError("formal training completed without a finite loss")

        policy.eval()
        reference_raw = collate_policy_samples([train_dataset[0]])
        heldout_raw = collate_policy_samples([heldout_dataset[0]])
        reference_batch = _move_batch(
            _normalise_batch(reference_raw, config.normalization, policy=policy), target
        )
        heldout_batch = _move_batch(
            _normalise_batch(heldout_raw, config.normalization, policy=policy), target
        )
        with torch.no_grad():
            reference_output = deployment_action_chunk(
                policy(reference_batch).action_chunk,
                loss_kind=config.loss_kind,
                continuous_rms=(
                    None
                    if continuous_rms is None
                    else torch.as_tensor(continuous_rms, device=target)
                ),
            ).detach().cpu()
            heldout_output = deployment_action_chunk(
                policy(heldout_batch).action_chunk,
                loss_kind=config.loss_kind,
                continuous_rms=(
                    None
                    if continuous_rms is None
                    else torch.as_tensor(continuous_rms, device=target)
                ),
            ).detach().cpu()
        expected_shape = (1, config.policy_config.chunk, config.policy_config.action_dim)
        if (
            tuple(reference_output.shape) != expected_shape
            or tuple(heldout_output.shape) != expected_shape
        ):
            raise RuntimeError("formal training produced a malformed action chunk")
        if not torch.isfinite(reference_output).all() or not torch.isfinite(heldout_output).all():
            raise RuntimeError("formal training produced non-finite action chunks")

        # Evaluate the complete held-out split against its stored labels.  The
        # mask excludes right-padded chunk elements, so this is a real action
        # error rather than the norm of the prediction itself.
        heldout_loader = DataLoader(
            heldout_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_policy_samples,
        )
        action_error = 0.0
        action_error_per_dim = torch.zeros(
            config.policy_config.action_dim, dtype=torch.float64
        )
        decomp_error = 0.0
        c_true_squared = 0.0
        c_true_squared_per_dim = torch.zeros_like(action_error_per_dim)
        valid_elements = 0
        first_step_error_per_dim = torch.zeros_like(action_error_per_dim)
        first_step_prediction_squared_per_dim = torch.zeros_like(action_error_per_dim)
        first_step_target_squared_per_dim = torch.zeros_like(action_error_per_dim)
        first_step_rows = 0
        gripper_sign_correct = 0
        for raw in heldout_loader:
            batch = _move_batch(
                _normalise_batch(raw, config.normalization, policy=policy), target
            )
            with torch.no_grad():
                output = policy(batch)
            prediction = deployment_action_chunk(
                output.action_chunk,
                loss_kind=config.loss_kind,
                continuous_rms=(
                    None
                    if continuous_rms is None
                    else torch.as_tensor(continuous_rms, device=target)
                ),
            )
            if (
                batch.actions is None
                or batch.c_true is None
                or batch.action_pad_mask is None
                or prediction.shape != batch.actions.shape
            ):
                raise RuntimeError("formal training held-out batch is missing evaluation fields")
            mask = batch.action_pad_mask.unsqueeze(-1).expand_as(prediction)
            decompensated = (batch.actions - batch.c_true).clamp(-1.0, 1.0)
            if not torch.isfinite(prediction).all():
                raise RuntimeError("formal training held-out output is non-finite")
            action_error += float(
                ((prediction - batch.actions).square() * mask).sum().cpu()
            )
            action_error_per_dim += (
                ((prediction - batch.actions).square() * mask)
                .sum(dim=(0, 1))
                .detach()
                .cpu()
                .to(torch.float64)
            )
            decomp_error += float(
                ((prediction - decompensated).square() * mask).sum().cpu()
            )
            c_true_squared += float((batch.c_true.square() * mask).sum().cpu())
            c_true_squared_per_dim += (
                (batch.c_true.square() * mask)
                .sum(dim=(0, 1))
                .detach()
                .cpu()
                .to(torch.float64)
            )
            valid_elements += int(mask.sum().item())
            first_mask = batch.action_pad_mask[:, 0]
            first_prediction = prediction[first_mask, 0, :]
            first_target = batch.actions[first_mask, 0, :]
            first_step_error_per_dim += (
                (first_prediction - first_target)
                .square()
                .sum(dim=0)
                .detach()
                .cpu()
                .to(torch.float64)
            )
            first_step_prediction_squared_per_dim += (
                first_prediction.square().sum(dim=0).detach().cpu().to(torch.float64)
            )
            first_step_target_squared_per_dim += (
                first_target.square().sum(dim=0).detach().cpu().to(torch.float64)
            )
            first_step_rows += int(first_mask.sum().item())
            gripper_sign_correct += int(
                ((first_prediction[:, -1] >= 0) == (first_target[:, -1] >= 0))
                .sum()
                .item()
            )
        if valid_elements < 1:
            raise RuntimeError("formal training held-out split has no valid action elements")
        if first_step_rows < 1:
            raise RuntimeError("formal training held-out split has no first-step actions")
        valid_action_rows = valid_elements // config.policy_config.action_dim
        heldout_action_mse = action_error / valid_elements
        heldout_action_mse_per_dim = tuple(
            float(value) for value in action_error_per_dim / valid_action_rows
        )
        heldout_decompensated_action_mse = decomp_error / valid_elements
        heldout_c_true_rms = math.sqrt(c_true_squared / valid_elements)
        heldout_c_true_rms_per_dim = tuple(
            math.sqrt(float(value) / valid_action_rows)
            for value in c_true_squared_per_dim
        )
        heldout_first_step_action_mse_per_dim = tuple(
            float(value) for value in first_step_error_per_dim / first_step_rows
        )
        heldout_first_step_prediction_rms_per_dim = tuple(
            math.sqrt(float(value) / first_step_rows)
            for value in first_step_prediction_squared_per_dim
        )
        heldout_first_step_target_rms_per_dim = tuple(
            math.sqrt(float(value) / first_step_rows)
            for value in first_step_target_squared_per_dim
        )
        heldout_first_step_gripper_sign_accuracy = gripper_sign_correct / first_step_rows
        if not all(
            math.isfinite(value)
            for value in (
                heldout_action_mse,
                heldout_decompensated_action_mse,
                heldout_c_true_rms,
                *heldout_action_mse_per_dim,
                *heldout_first_step_action_mse_per_dim,
                *heldout_first_step_prediction_rms_per_dim,
                *heldout_first_step_target_rms_per_dim,
                *heldout_c_true_rms_per_dim,
                heldout_first_step_gripper_sign_accuracy,
            )
        ):
            raise RuntimeError("formal training held-out metrics are non-finite")

        producer_end = _git_training_producer_state()
        producer_provenance = _training_producer_provenance(
            producer_start, producer_end
        )
        if release_status == "formal" and (
            not producer_provenance["stable"]
            or producer_end["tracked_worktree_dirty"]
        ):
            raise RuntimeError(
                "formal checkpoint publication requires one stable clean training "
                "producer HEAD"
            )
        payload: dict[str, Any] = {
            "schema": FORMAL_CHECKPOINT_SCHEMA,
            "marker": FORMAL_CHECKPOINT_MARKER,
            "release_status": release_status,
            "arm_id": config.policy_config.arm,
            "policy_config": _config_payload(config.policy_config),
            "seed": config.seed,
            "formal_plan": {
                "path": str(config.plan_path),
                "sha256": _sha256(config.plan_path),
                "artifact_sha256": config.plan.plan_sha256,
            },
            "normalization": {
                "path": str(config.normalization_path),
                "sha256": _sha256(config.normalization_path),
                "artifact_sha256": config.normalization.artifact_sha256,
                "plan_sha256": config.plan.plan_sha256,
                "training_episode_ids": list(config.plan.train_episode_ids),
            },
            "pool_bindings": _pool_bindings(config.plan),
            "training": {
                "epochs": config.epochs,
                "optimizer_steps": total_steps,
                "final_loss": final_loss,
                "loss": action_loss,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "window_sampling": config.window_sampling,
                "sampling_seed": config.sampling_seed,
            },
            "data_plan_provenance": dict(config.plan.payload["provenance"]),
            "training_producer_provenance": producer_provenance,
            "model_state": _cpu_state(policy.state_dict()),
            "optimizer_state": _cpu_state(optimizer.state_dict()),
        }
        payload["content_sha256"] = _content_sha256(payload)
        _atomic_save(config.checkpoint_path, payload)

    loaded = load_formal_checkpoint(
        config.checkpoint_path,
        plan_path=config.plan_path,
        normalization_path=config.normalization_path,
        policy_config=config.policy_config,
        expected_arm_id=config.policy_config.arm,
        expected_source_commit=config.plan.payload["provenance"]["source_commit"],
        allow_engineering=release_status == "engineering_only",
    )
    loaded.policy.to(target)
    reloaded_reference = loaded.act(_move_batch(reference_raw, target).obs)
    reloaded_heldout = loaded.act(_move_batch(heldout_raw, target).obs)
    if not torch.allclose(
        reference_output, torch.as_tensor(reloaded_reference), rtol=0.0, atol=1e-6
    ) or not torch.allclose(
        heldout_output, torch.as_tensor(reloaded_heldout), rtol=0.0, atol=1e-6
    ):
        raise RuntimeError("formal checkpoint changed inference across save/load")
    return FormalTrainingResult(
        checkpoint_path=config.checkpoint_path,
        marker=result_marker,
        epochs=config.epochs,
        optimizer_steps=total_steps,
        final_loss=final_loss,
        loss_curve=tuple(loss_curve),
        train_windows=len(train_dataset),
        heldout_windows=len(heldout_dataset),
        reference_output=reference_output,
        heldout_output=heldout_output,
        heldout_action_mse=heldout_action_mse,
        heldout_action_mse_per_dim=heldout_action_mse_per_dim,
        heldout_decompensated_action_mse=heldout_decompensated_action_mse,
        heldout_c_true_rms=heldout_c_true_rms,
        heldout_c_true_rms_per_dim=heldout_c_true_rms_per_dim,
        heldout_first_step_action_mse_per_dim=heldout_first_step_action_mse_per_dim,
        heldout_first_step_prediction_rms_per_dim=(
            heldout_first_step_prediction_rms_per_dim
        ),
        heldout_first_step_target_rms_per_dim=heldout_first_step_target_rms_per_dim,
        heldout_first_step_gripper_sign_accuracy=(
            heldout_first_step_gripper_sign_accuracy
        ),
        action_loss=action_loss,
    )


# A descriptive alias keeps call sites readable without creating a second path.
produce_formal_checkpoint = run_formal_training
