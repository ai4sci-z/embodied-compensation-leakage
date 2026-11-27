"""Offline policy/Oracle evaluation on one shared ``rho_off`` frame set."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from portal_trace.diagnostics.leakage import (
    gain_error_field,
    shrinkage_field,
    time_smoothing_field,
)
from portal_trace.policies.interface import Observation, PolicyConfig
from portal_trace.runners.formal_adapter import (
    _partition_indices,
    entries_from_formal_plan,
)
from portal_trace.runners.formal_checkpoint import (
    LoadedFormalCheckpoint,
    load_deployment_checkpoint,
    load_formal_checkpoint,
)
from portal_trace.runners.formal_data import (
    FormalDataPlan,
    heldout_cell_episode_ids,
    load_formal_data_plan,
    state_manifest_sha256,
)
from portal_trace.runners.leakage_input import (
    LeakageInputEpisode,
    LeakageInputHeader,
    LeakageInputProvenance,
    write_leakage_input,
)
from portal_trace.runners.policy_dataset import (
    RawPolicyWindowDataset,
    collate_policy_samples,
)
from portal_trace.runners.source_posterior import load_source_posterior
from portal_trace.runners.surrogate_input import load_surrogate_input

__all__ = [
    "PolicyCheckpointSpec",
    "materialize_leakage_input",
]


@dataclass(frozen=True)
class PolicyCheckpointSpec:
    """One checkpoint plus the plan-bound categorical identity used at inference."""

    checkpoint: Path
    plan: Path
    normalization: Path
    policy_config: PolicyConfig
    task_index: int
    embodiment_index: int

    def __post_init__(self) -> None:
        for field in ("checkpoint", "plan", "normalization"):
            object.__setattr__(self, field, Path(getattr(self, field)))
        if self.task_index not in range(self.policy_config.n_tasks):
            raise ValueError("task_index is outside the checkpoint task vocabulary")
        if self.embodiment_index not in range(self.policy_config.n_pool_embodiments):
            raise ValueError(
                "embodiment_index is outside the checkpoint embodiment vocabulary"
            )


def _load_checkpoint(
    spec: PolicyCheckpointSpec, *, allow_engineering: bool
) -> LoadedFormalCheckpoint:
    loader = load_formal_checkpoint if allow_engineering else load_deployment_checkpoint
    kwargs: dict[str, Any] = {
        "plan_path": spec.plan,
        "normalization_path": spec.normalization,
        "policy_config": spec.policy_config,
        "expected_arm_id": spec.policy_config.arm,
    }
    if allow_engineering:
        kwargs["allow_engineering"] = True
    return loader(spec.checkpoint, **kwargs)


def _observation_for(
    observation: Observation, spec: PolicyCheckpointSpec
) -> Observation:
    import torch

    batch_size = observation.batch_size
    device = observation.device
    return Observation(
        images=observation.images,
        proprio=observation.proprio,
        task_index=torch.full(
            (batch_size,), spec.task_index, dtype=torch.long, device=device
        ),
        embodiment_index=torch.full(
            (batch_size,), spec.embodiment_index, dtype=torch.long, device=device
        ),
        partition_index=observation.partition_index,
    )


def _validate_inference_identity(
    loaded: LoadedFormalCheckpoint,
    spec: PolicyCheckpointSpec,
    *,
    task: str,
    family: str,
    defect: str,
    level: int,
    oracle: bool,
) -> None:
    rows = tuple(loaded.plan.payload["pool"])
    task_names = {
        str(row["task"])
        for row in rows
        if int(row["task_index"]) == spec.task_index
    }
    if task_names != {task}:
        raise ValueError(
            f"checkpoint task_index {spec.task_index} maps to {sorted(task_names)}, "
            f"not requested task {task!r}"
        )
    embodiment_rows = tuple(
        row for row in rows if int(row["embodiment_index"]) == spec.embodiment_index
    )
    if not embodiment_rows:
        raise ValueError(
            f"checkpoint embodiment_index {spec.embodiment_index} is absent from its "
            "training plan"
        )
    identities = {
        (
            str(row["family"]),
            str(row["defect"]),
            int(row["level"]),
            str(row["clean_of"]),
        )
        for row in embodiment_rows
    }
    if any(identity[0] != family for identity in identities):
        raise ValueError(
            f"checkpoint embodiment_index {spec.embodiment_index} is not family {family!r}"
        )
    if oracle and identities != {(family, "clean", level, defect)}:
        raise ValueError(
            "B7 Oracle embodiment token must bind the clean twin for the requested "
            f"cell; observed identities={sorted(identities)}"
        )


def _selected_episode_ids(
    plan: FormalDataPlan,
    *,
    task: str,
    family: str,
    defect: str,
    level: int,
) -> tuple[str, ...]:
    return heldout_cell_episode_ids(
        plan, task=task, family=family, defect=defect, level=level
    )


def _state_manifest_sha256(plan: FormalDataPlan, episode_ids: tuple[str, ...]) -> str:
    return state_manifest_sha256(plan, episode_ids)


def _training_action_mean(plan: FormalDataPlan) -> np.ndarray:
    import h5py

    training = set(plan.train_episode_ids)
    total: np.ndarray | None = None
    n_rows = 0
    for row in plan.payload["pool"]:
        if row["id"] not in training:
            continue
        with h5py.File(row["path"], "r") as root:
            actions = np.asarray(
                root["data"][row["demo"]]["actions"], dtype=np.float64
            )
        if actions.ndim != 2 or not np.isfinite(actions).all():
            raise ValueError(f"{row['id']}: training actions are malformed")
        value = actions.sum(axis=0)
        total = value if total is None else total + value
        n_rows += actions.shape[0]
    if total is None or n_rows == 0:
        raise ValueError("evaluated checkpoint plan contains no training actions")
    return total / n_rows


def _predict_shared_windows(
    dataset: RawPolicyWindowDataset,
    *,
    evaluated: LoadedFormalCheckpoint,
    evaluated_spec: PolicyCheckpointSpec,
    oracle: LoadedFormalCheckpoint,
    oracle_spec: PolicyCheckpointSpec,
    batch_size: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        samples = [dataset[index] for index in range(start, stop)]
        batch = collate_policy_samples(samples)
        evaluated_chunk = evaluated.act(_observation_for(batch.obs, evaluated_spec))
        oracle_chunk = oracle.act(_observation_for(batch.obs, oracle_spec))
        expected = (
            stop - start,
            evaluated_spec.policy_config.chunk,
            evaluated_spec.policy_config.action_dim,
        )
        if evaluated_chunk.shape != expected or oracle_chunk.shape != expected:
            raise ValueError(
                "policy or Oracle returned a chunk shape inconsistent with the shared windows"
            )
        for offset, index in enumerate(range(start, stop)):
            episode_id, time_index = dataset.window_identity(index)
            key = f"{episode_id}::t={time_index}"
            if key in predictions:
                raise ValueError(f"duplicate shared evaluation window {key}")
            predictions[key] = (
                np.asarray(evaluated_chunk[offset, 0], dtype=np.float64),
                np.asarray(oracle_chunk[offset, 0], dtype=np.float64),
            )
    return predictions


def _episode_from_source(
    episode_id: str,
    *,
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    action_mean: np.ndarray,
    smoothing_window: int,
    posterior_w: np.ndarray | None,
    surrogate_fields: dict[str, np.ndarray],
) -> LeakageInputEpisode:
    import h5py

    try:
        path_text, demo = episode_id.rsplit("::data/", 1)
    except ValueError as exc:
        raise ValueError(f"malformed formal episode ID {episode_id!r}") from exc
    path = Path(path_text)
    with h5py.File(path, "r") as root:
        group = root["data"][demo]
        required = (
            "actions_defective",
            "actions_clean",
            "c_true",
            "c_true_clipped",
            "regime",
        )
        missing = [name for name in required if name not in group]
        if missing:
            raise ValueError(f"{episode_id}: missing leakage source fields {missing}")
        defective = np.asarray(group["actions_defective"], dtype=np.float64)
        clean = np.asarray(group["actions_clean"], dtype=np.float64)
        c_true = np.asarray(group["c_true"], dtype=np.float64)
        clipped = np.asarray(group["c_true_clipped"], dtype=bool)
        regime = np.asarray(group["regime"])
        surrogates = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in surrogate_fields.items()
        }
    if defective.ndim != 2 or defective.shape[1] != action_mean.shape[0]:
        raise ValueError(
            f"{episode_id}: teacher action shape {defective.shape} disagrees with "
            f"training action mean width {action_mean.shape}"
        )
    for name, values in {"actions_clean": clean, "c_true": c_true}.items():
        if values.shape != defective.shape:
            raise ValueError(
                f"{episode_id}: {name} has shape {values.shape}, expected {defective.shape}"
            )
    if clipped.shape != (defective.shape[0],) or regime.shape != (defective.shape[0],):
        raise ValueError(f"{episode_id}: frame labels disagree with the action horizon")
    if not surrogates:
        raise ValueError(f"{episode_id}: no admitted matched surrogate fields")
    surrogate_shapes = {name: values.shape for name, values in surrogates.items()}
    if len(set(surrogate_shapes.values())) != 1:
        raise ValueError(f"{episode_id}: matched surrogates change shape: {surrogate_shapes}")
    analysis_shape = next(iter(surrogate_shapes.values()))
    expected_analysis_shape = (defective.shape[0] - 1, defective.shape[1])
    if analysis_shape != expected_analysis_shape:
        raise ValueError(
            f"{episode_id}: matched surrogates have shape {analysis_shape}, expected "
            f"{expected_analysis_shape}; a_dot_nominal defines only the first T-1 frames"
        )
    if posterior_w is not None and np.shape(posterior_w) != (defective.shape[0],):
        raise ValueError(
            f"{episode_id}: posterior has shape {np.shape(posterior_w)}, expected "
            f"({defective.shape[0]},)"
        )
    horizon = analysis_shape[0]
    defective = defective[:horizon]
    clean = clean[:horizon]
    c_true = c_true[:horizon]
    clipped = clipped[:horizon]
    regime = regime[:horizon]
    if posterior_w is not None:
        posterior_w = np.asarray(posterior_w, dtype=np.float64)[:horizon]
    evaluated_action = np.empty_like(defective)
    oracle_action = np.empty_like(defective)
    for time_index in range(horizon):
        key = f"{episode_id}::t={time_index}"
        try:
            evaluated_action[time_index], oracle_action[time_index] = predictions[key]
        except KeyError as exc:
            raise ValueError(f"prediction is missing for shared window {key}") from exc
    episode_index = np.zeros(horizon, dtype=np.int64)
    return LeakageInputEpisode(
        episode_id=episode_id,
        evaluated_action=evaluated_action,
        oracle_action=oracle_action,
        clean_teacher_action=clean,
        defective_teacher_action=defective,
        c_true=c_true,
        regime=regime,
        c_true_clipped=clipped,
        shrinkage=shrinkage_field(oracle_action, action_mean),
        time_smoothing=time_smoothing_field(
            oracle_action, episode_index, smoothing_window
        ),
        gain_error=gain_error_field(oracle_action),
        surrogates=surrogates,
        posterior_w=posterior_w,
    )


def materialize_leakage_input(
    *,
    output: Path,
    evaluation_plan_path: Path,
    evaluated_spec: PolicyCheckpointSpec,
    oracle_spec: PolicyCheckpointSpec,
    task: str,
    family: str,
    defect: str,
    level: int,
    smoothing_window: int,
    batch_size: int = 32,
    allow_engineering: bool = False,
    source_posterior_path: Path | None = None,
    surrogate_input_path: Path,
) -> LeakageInputHeader:
    """Run both checkpoints on identical held-out frames and publish one artifact."""
    evaluation_plan = load_formal_data_plan(evaluation_plan_path)
    evaluated = _load_checkpoint(evaluated_spec, allow_engineering=allow_engineering)
    oracle = _load_checkpoint(oracle_spec, allow_engineering=allow_engineering)
    _validate_inference_identity(
        evaluated,
        evaluated_spec,
        task=task,
        family=family,
        defect=defect,
        level=level,
        oracle=False,
    )
    _validate_inference_identity(
        oracle,
        oracle_spec,
        task=task,
        family=family,
        defect=defect,
        level=level,
        oracle=True,
    )
    if evaluated.metadata["seed"] != oracle.metadata["seed"]:
        raise ValueError("evaluated policy and B7 must use the same training seed")
    evaluated_architecture = asdict(evaluated_spec.policy_config)
    oracle_architecture = asdict(oracle_spec.policy_config)
    evaluated_architecture.pop("arm")
    oracle_architecture.pop("arm")
    if evaluated_architecture != oracle_architecture:
        raise ValueError("evaluated policy and B7 architecture configurations differ")
    episode_ids = _selected_episode_ids(
        evaluation_plan,
        task=task,
        family=family,
        defect=defect,
        level=level,
    )
    surrogate_header, surrogate_rows = load_surrogate_input(surrogate_input_path)
    surrogate_provenance = surrogate_header.provenance
    expected_cell = (task, family, defect, level)
    surrogate_cell = (
        surrogate_provenance.task,
        surrogate_provenance.family,
        surrogate_provenance.defect,
        surrogate_provenance.level,
    )
    if surrogate_cell != expected_cell:
        raise ValueError("matched-surrogate artifact identifies a different cell")
    if surrogate_provenance.evaluation_plan_sha256 != evaluation_plan.plan_sha256:
        raise ValueError("matched-surrogate artifact is bound to a different plan")
    expected_manifest_sha = _state_manifest_sha256(evaluation_plan, episode_ids)
    if surrogate_provenance.source_state_manifest_sha256 != expected_manifest_sha:
        raise ValueError("matched-surrogate artifact is bound to different state rows")
    if set(surrogate_rows) != set(episode_ids):
        raise ValueError("matched-surrogate episode set does not match the held-out cell")
    dataset = RawPolicyWindowDataset(
        entries_from_formal_plan(evaluation_plan),
        k_hist=evaluated_spec.policy_config.obs.k_hist,
        episode_ids=episode_ids,
        partition_indices=_partition_indices(
            evaluation_plan, evaluated_spec.policy_config.n_pool_embodiments
        ),
    )
    predictions = _predict_shared_windows(
        dataset,
        evaluated=evaluated,
        evaluated_spec=evaluated_spec,
        oracle=oracle,
        oracle_spec=oracle_spec,
        batch_size=batch_size,
    )
    action_mean = _training_action_mean(evaluated.plan)
    if source_posterior_path is None and not allow_engineering:
        raise ValueError(
            "formal leakage materialisation requires a formal source posterior; "
            "omitting --source-posterior is permitted only for engineering diagnostics"
        )
    posterior_artifact_sha256 = None
    posterior_rows: dict[str, np.ndarray] = {}
    posterior_release_status = "engineering_only"
    if source_posterior_path is not None:
        posterior_header, posterior_rows = load_source_posterior(source_posterior_path)
        if posterior_header.provenance.plan_sha256 != evaluation_plan.plan_sha256:
            raise ValueError("source posterior is bound to a different evaluation plan")
        if set(posterior_rows) != set(episode_ids):
            raise ValueError(
                "source posterior episode set does not exactly match the held-out cell"
            )
        posterior_artifact_sha256 = posterior_header.artifact_sha256
        posterior_release_status = posterior_header.provenance.release_status
    episodes = tuple(
        _episode_from_source(
            episode_id,
            predictions=predictions,
            action_mean=action_mean,
            smoothing_window=smoothing_window,
            posterior_w=posterior_rows.get(episode_id),
            surrogate_fields=surrogate_rows[episode_id],
        )
        for episode_id in episode_ids
    )
    release_status = (
        "formal"
        if (
            evaluated.metadata["release_status"]
            == oracle.metadata["release_status"]
            == posterior_release_status
            == surrogate_provenance.release_status
            == "formal"
        )
        else "engineering_only"
    )
    if release_status != "formal" and not allow_engineering:
        raise ValueError("engineering-only checkpoints cannot produce a formal leakage input")
    provenance = LeakageInputProvenance(
        evaluated_arm=evaluated_spec.policy_config.arm,
        oracle_arm=oracle_spec.policy_config.arm,
        evaluated_checkpoint_sha256=evaluated.checkpoint_sha256,
        oracle_checkpoint_sha256=oracle.checkpoint_sha256,
        evaluated_task_index=evaluated_spec.task_index,
        evaluated_embodiment_index=evaluated_spec.embodiment_index,
        oracle_task_index=oracle_spec.task_index,
        oracle_embodiment_index=oracle_spec.embodiment_index,
        evaluation_plan_sha256=evaluation_plan.plan_sha256,
        source_state_manifest_sha256=expected_manifest_sha,
        task=task,
        family=family,
        defect=defect,
        level=level,
        training_seed=int(evaluated.metadata["seed"]),
        time_smoothing_window=smoothing_window,
        release_status=release_status,
        source_posterior_artifact_sha256=posterior_artifact_sha256,
        surrogate_artifact_sha256=surrogate_header.artifact_sha256,
        excluded_surrogates=surrogate_provenance.struck_surrogates,
    )
    return write_leakage_input(output, provenance=provenance, episodes=episodes)
