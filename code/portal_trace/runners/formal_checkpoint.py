"""Admission checks for formal deployment checkpoints.

The bounded adapter probe has its own deliberately non-formal checkpoint
schema.  This module defines the stricter consumer contract that a real
training runner must satisfy before :mod:`runners.evaluate` may roll a policy
out. The shared producer lives in :mod:`runners.formal_training`; this module
remains the independent loader/validator used after publication.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from portal_trace.policies.action_objective import (
    CONTINUOUS_ACTION_DIMS,
    MSE_ACTION_LOSS,
    STANDARDIZED_ACTION_LOSS,
    STRUCTURED_ACTION_LOSS,
    deployment_action_chunk,
)
from portal_trace.policies.interface import Observation, Policy, PolicyConfig
from portal_trace.policies.registry import build
from portal_trace.runners.formal_data import (
    FormalDataPlan,
    FormalProprioNormalization,
    _verify_plan,
    load_formal_data_plan,
    load_formal_normalization,
)

__all__ = [
    "FORMAL_CHECKPOINT_SCHEMA",
    "LEGACY_FORMAL_CHECKPOINT_SCHEMA",
    "FORMAL_CHECKPOINT_MARKER",
    "LoadedFormalCheckpoint",
    "load_deployment_checkpoint",
    "load_formal_checkpoint",
]

LEGACY_FORMAL_CHECKPOINT_SCHEMA = "portal_trace.formal_checkpoint/2"
FORMAL_CHECKPOINT_SCHEMA = "portal_trace.formal_checkpoint/3"
FORMAL_CHECKPOINT_MARKER = "PORTAL_TRACE_FORMAL_CHECKPOINT_NOT_SMOKE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _hash_value(value: Any, digest: hashlib._Hash) -> None:
    """Hash nested checkpoint values with tensor dtype/shape boundaries."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(tensor.numpy().tobytes())
        return
    if value is None:
        digest.update(b"none\0")
        return
    if isinstance(value, bool):
        digest.update(f"bool:{value}\0".encode())
        return
    if isinstance(value, (int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("formal checkpoint contains a non-finite scalar")
        digest.update(f"scalar:{type(value).__name__}:".encode())
        digest.update(str(value).encode())
        digest.update(b"\0")
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=str):
            digest.update(str(key).encode())
            digest.update(b"\0")
            _hash_value(value[key], digest)
        return
    if isinstance(value, (tuple, list)):
        digest.update(b"sequence\0")
        for item in value:
            _hash_value(item, digest)
        digest.update(b"end\0")
        return
    raise TypeError(f"formal checkpoint contains unsupported value {type(value).__name__}")


def _content_sha256(document: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(document):
        if key in {"model_state", "optimizer_state", "content_sha256"}:
            continue
        digest.update(key.encode())
        digest.update(b"\0")
        _hash_value(document[key], digest)
    _hash_value(document["model_state"], digest)
    _hash_value(document["optimizer_state"], digest)
    return digest.hexdigest()


def _finite_tensors(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return not (value.is_floating_point() or value.is_complex()) or bool(
            torch.isfinite(value).all()
        )
    if isinstance(value, Mapping):
        return all(_finite_tensors(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_tensors(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _require_sha(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"formal checkpoint {field} must be a lowercase SHA-256")
    return value


def _legacy_mse_loss_metadata() -> dict[str, Any]:
    return {
        "kind": MSE_ACTION_LOSS,
        "continuous_dims": list(range(CONTINUOUS_ACTION_DIMS)),
        "gripper_dim": 6,
        "continuous_weight": 1.0,
        "gripper_weight": 1.0,
        "continuous_rms": None,
        "deployment_gripper": "identity",
        "scale_provenance": None,
    }


def _validate_action_loss_metadata(
    value: Any, *, plan: FormalDataPlan
) -> dict[str, Any]:
    expected = set(_legacy_mse_loss_metadata())
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("formal checkpoint action loss metadata is incomplete")
    kind = value["kind"]
    if kind not in {MSE_ACTION_LOSS, STRUCTURED_ACTION_LOSS, STANDARDIZED_ACTION_LOSS}:
        raise ValueError("formal checkpoint action loss kind is invalid")
    if value["continuous_dims"] != list(range(CONTINUOUS_ACTION_DIMS)):
        raise ValueError("formal checkpoint continuous action dimensions are invalid")
    if value["gripper_dim"] != 6:
        raise ValueError("formal checkpoint gripper action dimension is invalid")
    for field in ("continuous_weight", "gripper_weight"):
        field_value = value[field]
        if (
            not isinstance(field_value, (int, float))
            or isinstance(field_value, bool)
            or not math.isfinite(float(field_value))
            or float(field_value) <= 0.0
        ):
            raise ValueError(f"formal checkpoint {field} is invalid")
    if kind == MSE_ACTION_LOSS:
        if (
            value["continuous_rms"] is not None
            or value["deployment_gripper"] != "identity"
            or value["scale_provenance"] is not None
        ):
            raise ValueError("formal checkpoint MSE loss metadata is inconsistent")
    else:
        scales = value["continuous_rms"]
        if (
            not isinstance(scales, list)
            or len(scales) != CONTINUOUS_ACTION_DIMS
            or not all(
                isinstance(scale, (int, float))
                and not isinstance(scale, bool)
                and math.isfinite(float(scale))
                and float(scale) > 0.0
                for scale in scales
            )
        ):
            raise ValueError("formal checkpoint continuous RMS scales are invalid")
        scale_provenance = value["scale_provenance"]
        if not isinstance(scale_provenance, dict) or set(scale_provenance) != {
            "split",
            "plan_sha256",
            "training_episode_ids",
            "fitted_action_rows",
        }:
            raise ValueError("formal checkpoint action scale provenance is incomplete")
        if (
            value["deployment_gripper"] != "sign_logit_to_pm1"
            or scale_provenance["split"] != "train"
            or scale_provenance["plan_sha256"] != plan.plan_sha256
            or tuple(scale_provenance["training_episode_ids"])
            != tuple(plan.train_episode_ids)
            or isinstance(scale_provenance["fitted_action_rows"], bool)
            or not isinstance(scale_provenance["fitted_action_rows"], int)
            or scale_provenance["fitted_action_rows"] < 1
        ):
            raise ValueError("formal checkpoint action scale provenance is invalid")
    return dict(value)


def _validate_training_producer_provenance(
    value: Any, *, release_status: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "capture",
        "start",
        "end",
        "stable",
    }:
        raise ValueError("formal checkpoint training producer provenance is incomplete")
    capture = value["capture"]
    if capture not in {
        "git-head+tracked-worktree-v1",
        "git-head+tracked-and-untracked-project-sources-v2",
    }:
        raise ValueError("formal checkpoint training producer capture kind is invalid")
    if release_status == "formal" and capture != (
        "git-head+tracked-and-untracked-project-sources-v2"
    ):
        raise ValueError(
            "formal checkpoint must capture untracked executable project sources"
        )

    states: list[dict[str, Any]] = []
    for phase in ("start", "end"):
        state = value[phase]
        if not isinstance(state, dict) or set(state) != {
            "head",
            "tracked_worktree_dirty",
            "tracked_diff_sha256",
        }:
            raise ValueError(
                f"formal checkpoint training producer {phase} state is incomplete"
            )
        head = state["head"]
        if (
            not isinstance(head, str)
            or len(head) not in {40, 64}
            or any(char not in "0123456789abcdef" for char in head)
        ):
            raise ValueError(
                f"formal checkpoint training producer {phase} HEAD is invalid"
            )
        dirty = state["tracked_worktree_dirty"]
        if not isinstance(dirty, bool):
            raise ValueError(
                f"formal checkpoint training producer {phase} dirty flag is invalid"
            )
        diff_sha = state["tracked_diff_sha256"]
        if dirty:
            _require_sha(diff_sha, field=f"training_producer.{phase}.tracked_diff_sha256")
        elif diff_sha is not None:
            raise ValueError(
                f"formal checkpoint clean training producer {phase} has a diff hash"
            )
        states.append(dict(state))

    stable = value["stable"]
    if not isinstance(stable, bool) or stable != (states[0] == states[1]):
        raise ValueError("formal checkpoint training producer stability flag is invalid")
    if release_status == "formal" and (
        not stable
        or states[0]["tracked_worktree_dirty"]
        or states[1]["tracked_worktree_dirty"]
        or states[0]["head"] != states[1]["head"]
    ):
        raise ValueError(
            "formal checkpoint admission requires one stable clean training producer HEAD"
        )
    return {
        "capture": capture,
        "start": states[0],
        "end": states[1],
        "stable": stable,
    }


def _pool_bindings(plan: FormalDataPlan) -> list[dict[str, Any]]:
    return [
        {
            "id": row["id"],
            "path": row["path"],
            "sha256": row["sha256"],
            "task_index": row["task_index"],
            "embodiment_index": row["embodiment_index"],
            "task": row["task"],
            "family": row.get("family", ""),
            "defect": row.get("defect", ""),
            "level": row.get("level"),
            "clean_of": row["clean_of"],
            "demo": row["demo"],
            "window_count": row["window_count"],
            "seed_env": row["seed_env"],
            "ilc_status": row["ilc_status"],
            "clean_ilc_status": row["clean_ilc_status"],
            "t_success": row["t_success"],
        }
        for row in plan.payload["pool"]
    ]


@dataclass(frozen=True)
class LoadedFormalCheckpoint:
    """A validated policy plus the exact train-only normalization it requires."""

    policy: Policy
    normalization: FormalProprioNormalization
    plan: FormalDataPlan
    metadata: Mapping[str, Any]
    checkpoint_sha256: str

    @torch.no_grad()
    def act(self, observation: Observation) -> np.ndarray:
        """Run inference with the exact train-only proprioception transform."""
        if observation.proprio is not None and self.policy.cfg.obs.use_proprio:
            observation = Observation(
                images=observation.images,
                proprio=self.normalization.apply(observation.proprio),
                task_index=observation.task_index,
                embodiment_index=observation.embodiment_index,
                partition_index=observation.partition_index,
            )
        actions = self.policy.act(observation)
        action_loss = self.metadata["training"]["loss"]
        loss_kind = str(action_loss["kind"])
        if loss_kind != MSE_ACTION_LOSS:
            actions = deployment_action_chunk(
                torch.as_tensor(actions),
                loss_kind=loss_kind,
                continuous_rms=torch.as_tensor(action_loss["continuous_rms"]),
            ).numpy()
        expected = (observation.batch_size, self.policy.cfg.chunk, self.policy.cfg.action_dim)
        if tuple(actions.shape) != expected:
            raise ValueError(
                f"formal deployment policy returned {tuple(actions.shape)}, expected {expected}"
            )
        if not torch.isfinite(torch.as_tensor(actions)).all():
            raise ValueError("formal deployment policy returned non-finite actions")
        return actions


def load_deployment_checkpoint(
    path: Path | str,
    *,
    plan_path: Path | str,
    normalization_path: Path | str,
    policy_config: PolicyConfig,
    expected_arm_id: str | None = None,
    expected_source_commit: str | None = None,
) -> LoadedFormalCheckpoint:
    """Admit a checkpoint for deployment, never for engineering-only probes.

    This is the narrow deployment boundary.  It deliberately does not expose
    ``allow_engineering``: callers that need to inspect an engineering
    checkpoint must use :func:`load_formal_checkpoint` explicitly, while every
    deployment caller is fail-closed on ``release_status=engineering_only``.
    The returned object still only performs local model loading and normalized
    inference; the rollout backend and scientific evaluation gates remain
    outside this function.
    """
    return load_formal_checkpoint(
        path,
        plan_path=plan_path,
        normalization_path=normalization_path,
        policy_config=policy_config,
        expected_arm_id=expected_arm_id,
        expected_source_commit=expected_source_commit,
        allow_engineering=False,
    )


def load_formal_checkpoint(
    path: Path | str,
    *,
    plan_path: Path | str,
    normalization_path: Path | str,
    policy_config: PolicyConfig,
    expected_arm_id: str | None = None,
    expected_source_commit: str | None = None,
    allow_engineering: bool = False,
) -> LoadedFormalCheckpoint:
    """Load a formal checkpoint only when every provenance binding agrees.

    The caller supplies the expected policy configuration and artifact paths so
    a checkpoint cannot silently select a different arm or recompute statistics.
    This function does not authorize an evaluation run by itself; the formal
    training runner and the P0/evaluation gates still own that decision.
    """
    checkpoint_path = Path(path)
    plan_path = Path(plan_path)
    normalization_path = Path(normalization_path)
    try:
        document = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
        raise ValueError(f"{checkpoint_path}: unreadable formal checkpoint") from exc
    if not isinstance(document, dict):
        raise ValueError("formal checkpoint must be a mapping")
    common_fields = {
        "schema",
        "marker",
        "arm_id",
        "release_status",
        "policy_config",
        "seed",
        "formal_plan",
        "normalization",
        "pool_bindings",
        "training",
        "model_state",
        "optimizer_state",
        "content_sha256",
    }
    legacy_fields = common_fields | {"provenance"}
    current_fields = common_fields | {
        "data_plan_provenance",
        "training_producer_provenance",
    }
    schema = document.get("schema")
    document_fields = set(document)
    if schema == LEGACY_FORMAL_CHECKPOINT_SCHEMA:
        if document_fields != legacy_fields:
            raise ValueError("legacy formal checkpoint fields are incomplete or unexpected")
        legacy_layout = True
    elif schema == FORMAL_CHECKPOINT_SCHEMA:
        if document_fields != current_fields:
            raise ValueError("formal checkpoint fields are incomplete or unexpected")
        legacy_layout = False
    else:
        raise ValueError("checkpoint is not the formal checkpoint schema")
    if document["marker"] != FORMAL_CHECKPOINT_MARKER:
        raise ValueError("checkpoint lacks the formal non-smoke marker")
    if document["release_status"] not in {"formal", "engineering_only"}:
        raise ValueError("formal checkpoint release_status is invalid")
    if legacy_layout and document["release_status"] == "formal":
        raise ValueError(
            "legacy checkpoint lacks training producer provenance and cannot be "
            "admitted as formal"
        )
    if document["release_status"] == "engineering_only" and not allow_engineering:
        raise ValueError(
            "formal checkpoint is engineering-only; deployment admission requires "
            "release_status=formal"
        )
    if not isinstance(document["arm_id"], str) or document["arm_id"] != policy_config.arm:
        raise ValueError("formal checkpoint arm_id does not match policy configuration")
    if expected_arm_id is not None and document["arm_id"] != expected_arm_id:
        raise ValueError("formal checkpoint arm_id does not match requested arm")
    if document["policy_config"] != _config_payload(policy_config):
        raise ValueError("formal checkpoint policy configuration does not match request")
    if isinstance(document["seed"], bool) or not isinstance(document["seed"], int):
        raise ValueError("formal checkpoint seed must be an integer")
    if not _finite_tensors(document["model_state"]) or not _finite_tensors(
        document["optimizer_state"]
    ):
        raise ValueError("formal checkpoint contains non-finite state")
    if not isinstance(document["model_state"], dict) or not document["model_state"]:
        raise ValueError("formal checkpoint model_state is empty or invalid")
    if document["optimizer_state"] is not None and not isinstance(
        document["optimizer_state"], dict
    ):
        raise ValueError("formal checkpoint optimizer_state must be a mapping or null")
    _require_sha(document["content_sha256"], field="content_sha256")
    if _content_sha256(document) != document["content_sha256"]:
        raise ValueError("formal checkpoint content SHA-256 does not match")

    allow_legacy_engineering = (
        allow_engineering and document["release_status"] == "engineering_only"
    )
    plan = load_formal_data_plan(
        plan_path, allow_legacy_engineering=allow_legacy_engineering
    )
    _verify_plan(plan, allow_legacy_engineering=allow_legacy_engineering)
    plan_binding = document["formal_plan"]
    if not isinstance(plan_binding, dict) or set(plan_binding) != {
        "path",
        "sha256",
        "artifact_sha256",
    }:
        raise ValueError("formal checkpoint formal_plan binding is incomplete")
    if (
        Path(plan_binding["path"]).resolve() != plan_path.resolve()
        or _require_sha(plan_binding["sha256"], field="formal_plan.sha256")
        != _sha256(plan_path)
        or _require_sha(plan_binding["artifact_sha256"], field="formal_plan.artifact_sha256")
        != plan.plan_sha256
    ):
        raise ValueError("formal checkpoint formal data plan binding does not match")

    normalization = load_formal_normalization(
        normalization_path,
        plan=plan,
        allow_legacy_engineering=allow_legacy_engineering,
    )
    norm_binding = document["normalization"]
    if not isinstance(norm_binding, dict) or set(norm_binding) != {
        "path",
        "sha256",
        "artifact_sha256",
        "plan_sha256",
        "training_episode_ids",
    }:
        raise ValueError("formal checkpoint normalization binding is incomplete")
    if (
        Path(norm_binding["path"]).resolve() != normalization_path.resolve()
        or _require_sha(norm_binding["sha256"], field="normalization.sha256")
        != _sha256(normalization_path)
        or _require_sha(norm_binding["artifact_sha256"], field="normalization.artifact_sha256")
        != normalization.artifact_sha256
        or norm_binding["plan_sha256"] != plan.plan_sha256
        or tuple(norm_binding["training_episode_ids"])
        != tuple(plan.train_episode_ids)
    ):
        raise ValueError("formal checkpoint normalization binding does not match")
    if document["pool_bindings"] != _pool_bindings(plan):
        raise ValueError("formal checkpoint pool bindings do not match formal data plan")
    provenance = (
        document["provenance"]
        if legacy_layout
        else document["data_plan_provenance"]
    )
    plan_provenance = plan.payload["provenance"]
    # Derived plans may carry additional identity bindings (for example the
    # target-only embodiment selection).  The checkpoint must preserve the
    # complete plan provenance exactly; requiring the historical two-field set
    # here would reject valid derived plans after training has already bound to
    # them.
    if not isinstance(provenance, dict) or not isinstance(plan_provenance, dict):
        raise ValueError("formal checkpoint provenance is incomplete")
    if set(provenance) != set(plan_provenance) or provenance != plan_provenance:
        raise ValueError("formal checkpoint provenance does not match formal data plan")
    if expected_source_commit is not None and provenance["source_commit"] != expected_source_commit:
        raise ValueError("formal checkpoint source commit does not match request")
    training_producer_provenance = (
        {
            "capture": "legacy_unrecorded",
            "start": None,
            "end": None,
            "stable": False,
        }
        if legacy_layout
        else _validate_training_producer_provenance(
            document["training_producer_provenance"],
            release_status=document["release_status"],
        )
    )
    training = document["training"]
    sampling_fields = {
        "batch_size", "learning_rate", "window_sampling", "sampling_seed"
    }
    if not isinstance(training, dict) or frozenset(training) not in {
        frozenset({
        "epochs",
        "optimizer_steps",
        "final_loss",
        }),
        frozenset({
            "epochs",
            "optimizer_steps",
            "final_loss",
            "loss",
        }),
        frozenset({"epochs", "optimizer_steps", "final_loss", "loss"} | sampling_fields),
    }:
        raise ValueError("formal checkpoint training metadata is incomplete")
    if (
        isinstance(training["epochs"], bool)
        or not isinstance(training["epochs"], int)
        or training["epochs"] < 1
        or isinstance(training["optimizer_steps"], bool)
        or not isinstance(training["optimizer_steps"], int)
        or training["optimizer_steps"] < 1
        or not isinstance(training["final_loss"], (int, float))
        or not math.isfinite(float(training["final_loss"]))
    ):
        raise ValueError("formal checkpoint training metadata is invalid")
    if sampling_fields.issubset(training):
        if (
            isinstance(training["batch_size"], bool)
            or not isinstance(training["batch_size"], int)
            or training["batch_size"] < 1
            or isinstance(training["learning_rate"], bool)
            or not isinstance(training["learning_rate"], (int, float))
            or not math.isfinite(float(training["learning_rate"]))
            or training["learning_rate"] <= 0
            or training["window_sampling"] not in {"shuffled", "sequential"}
            or isinstance(training["sampling_seed"], bool)
            or not isinstance(training["sampling_seed"], int)
            or not 0 <= training["sampling_seed"] < 2 ** 64
        ):
            raise ValueError("formal checkpoint training sampling metadata is invalid")
    action_loss = (
        _validate_action_loss_metadata(training["loss"], plan=plan)
        if "loss" in training
        else _legacy_mse_loss_metadata()
    )
    normalized_training = dict(training)
    normalized_training["loss"] = action_loss

    policy = build(policy_config.arm, policy_config)
    try:
        policy.load_state_dict(document["model_state"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("formal checkpoint model_state does not match policy config") from exc
    policy.eval()
    train_ids = set(plan.train_episode_ids)
    heldout_ids = set(plan.heldout_episode_ids)
    train_pairs = {row["pair_id"] for row in plan.payload["pool"] if row["id"] in train_ids}
    heldout_pairs = {
        row["pair_id"] for row in plan.payload["pool"] if row["id"] in heldout_ids
    }
    from portal_trace.runners.formal_data import proprio_acceleration_semantics

    return LoadedFormalCheckpoint(
        policy=policy,
        normalization=normalization,
        plan=plan,
        metadata={
            "arm_id": document["arm_id"],
            "release_status": document["release_status"],
            "seed": document["seed"],
            "training": normalized_training,
            "data_plan_provenance": provenance,
            "data_plan_schema": plan.payload["schema"],
            "proprio_acceleration_semantics": proprio_acceleration_semantics(plan),
            "train_heldout_overlap_pair_ids": sorted(train_pairs & heldout_pairs),
            "training_producer_provenance": training_producer_provenance,
            "checkpoint_path": str(checkpoint_path.resolve()),
        },
        checkpoint_sha256=_sha256(checkpoint_path),
    )
