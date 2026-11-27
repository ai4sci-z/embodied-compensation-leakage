#!/usr/bin/env python3
"""Train matched shared policies on three physical compensation sources.

Commands:
  python -m course_tools.train_three_source_pool --manifest episodes.json --statistics stats.json --fit-statistics
  python -m course_tools.train_three_source_pool --manifest episodes.json --statistics stats.json --config run.json --output RUN

All labels remain canonical OSC actions. Only the six continuous outputs are
standardized internally; deployment_action_chunk converts them back exactly
once. The checkpoint loader takes raw Observation tensors and returns canonical
action chunks, so existing rollout observation adapters can be reused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from course_tools.three_source_data import (
    EpisodeManifest, PoolDataset, PairedSourceSampler, SOURCES, LABEL_MODES,
    fit_statistics, validate_statistics, normalize_observation, prepare_batch,
    sha256, json_hash,
)
from portal_trace.policies.action_objective import (
    STANDARDIZED_ACTION_LOSS, structured_action_loss, deployment_action_chunk,
)
from portal_trace.policies.interface import Batch, Observation, PolicyConfig
from portal_trace.policies.obs import ObsSpec
from portal_trace.policies.portal_policy import PortalTracePolicy

CHECKPOINT_SCHEMA = "dl_course.three_source_policy/1"
CONFIG_SCHEMA = "dl_course.three_source_training/1"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_torch(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def read_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    if cfg.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"config schema must be {CONFIG_SCHEMA}")
    required = ("seed", "sampling_seed", "updates", "batch_size", "source_weights", "label_mode")
    for key in required:
        if key not in cfg:
            raise ValueError(f"training config must specify {key}")
    defaults = {"learning_rate": 3e-4, "d_model": 512, "n_layers": 4, "n_heads": 8,
                "ffn_dim": 2048, "dropout": 0.1, "augment": False, "k_hist": 5,
                "checkpoint_every": 5000, "evaluate_every": 5000, "log_every": 100,
                "episode_cache_mib": 2048, "torch_threads": 2, "level": None,
                "gradient_clip_norm": None, "validation_stride": 1}
    cfg = {**defaults, **cfg}
    if cfg["label_mode"] not in LABEL_MODES:
        raise ValueError("unknown label intervention")
    for key in ("updates", "batch_size", "checkpoint_every", "evaluate_every", "log_every",
                "k_hist", "torch_threads", "validation_stride"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("seed", "sampling_seed"):
        if not isinstance(cfg[key], int) or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if not np.isfinite(cfg["learning_rate"]) or cfg["learning_rate"] <= 0:
        raise ValueError("learning rate must be positive")
    weights = cfg["source_weights"]
    if set(weights) != set(SOURCES):
        raise ValueError(f"source weights must contain {SOURCES}")
    values = np.asarray([weights[s] for s in SOURCES], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError("invalid source weights")
    cfg["source_weights"] = dict(zip(SOURCES, (values / values.sum()).tolist()))
    return cfg


def policy_config(cfg: dict) -> PolicyConfig:
    return PolicyConfig(arm="three-source-unconditioned", obs=ObsSpec(k_hist=cfg["k_hist"]),
                        n_tasks=1, n_pool_embodiments=4,
                        d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
                        ffn_dim=cfg["ffn_dim"], dropout=cfg["dropout"], augment=cfg["augment"])


class LoadedThreeSourcePolicy:
    """Shared-policy deployment interface; observation normalization is internal."""
    def __init__(self, policy, metadata, statistics, device):
        self.policy, self.metadata, self.statistics = policy, metadata, statistics
        self.device = torch.device(device)
        self.history = policy.cfg.obs.k_hist
        self.rms = torch.tensor(statistics["action_rms"], dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def predict(self, observation: Observation, *, clip: bool = True) -> np.ndarray:
        self.policy.eval()
        obs = normalize_observation(observation, self.statistics, self.device)
        out = self.policy(Batch(obs=obs)).action_chunk
        action = deployment_action_chunk(out, loss_kind=STANDARDIZED_ACTION_LOSS,
                                         continuous_rms=self.rms)
        return (action.clamp(-1, 1) if clip else action).cpu().numpy()

    def act(self, observation: Observation) -> np.ndarray:
        return self.predict(observation, clip=True)

    def reset_episode(self):
        """Temporal history is supplied by the existing observation adapter."""


def load_three_source_checkpoint(path: str | Path, *, device="cpu") -> LoadedThreeSourcePolicy:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("not a three-source checkpoint")
    validate_statistics(payload["statistics"])
    if json_hash(payload["run_identity"]) != payload["run_sha256"]:
        raise ValueError("checkpoint run identity mismatch")
    policy = PortalTracePolicy(policy_config(payload["config"]))
    policy.load_state_dict(payload["model_state"], strict=True)
    policy.to(device).eval()
    metadata = {k: v for k, v in payload.items()
                if k not in {"model_state", "optimizer_state", "rng", "statistics"}}
    metadata["checkpoint_sha256"] = sha256(path)
    return LoadedThreeSourcePolicy(policy, metadata, payload["statistics"], device)


@torch.no_grad()
def evaluate(policy, dataset: PoolDataset, statistics: dict, config: dict, device) -> dict:
    """Per-source first-action errors on a fixed validation scene split.

    No source posterior or pseudo-leakage label is inferred here. Full directional
    leakage analysis consumes stored predictions and counterfactual references.
    """
    was_training = policy.training
    policy.eval()
    rms = torch.tensor(statistics["action_rms"], dtype=torch.float32, device=device)
    totals = {s: {"n": 0, "target_squared": 0., "raw_squared": 0., "clean_squared": 0.,
                  "gripper_correct": 0, "clip_elements": 0} for s in SOURCES}
    try:
        indices = np.arange(0, len(dataset), config["validation_stride"], dtype=np.int64)
        for offset in range(0, len(indices), config["batch_size"]):
            batch, source_ids = dataset.batch(indices[offset:offset + config["batch_size"]],
                                               mode=config["label_mode"])
            batch = prepare_batch(batch, statistics, device)
            prediction = policy(Batch(obs=batch.obs)).action_chunk
            action = deployment_action_chunk(prediction, loss_kind=STANDARDIZED_ACTION_LOSS,
                                             continuous_rms=rms)[:, 0]
            raw = batch.extras["raw_actions"][:, 0]
            clean = raw[:, :6] - batch.c_true[:, 0, :6]
            target = batch.actions[:, 0]
            for source_id, source in enumerate(SOURCES):
                chosen = (source_ids == source_id).to(device)
                n = int(chosen.sum())
                if not n:
                    continue
                cell = totals[source]
                cell["n"] += n
                cell["target_squared"] += float((action[chosen, :6] - target[chosen, :6]).square().sum())
                cell["raw_squared"] += float((action[chosen, :6] - raw[chosen, :6]).square().sum())
                cell["clean_squared"] += float((action[chosen, :6] - clean[chosen]).square().sum())
                cell["gripper_correct"] += int(((action[chosen, 6] > 0) == (raw[chosen, 6] > 0)).sum())
                cell["clip_elements"] += int((action[chosen, :6].abs() > 1).sum())
    finally:
        policy.train(was_training)
    report = {}
    for source, cell in totals.items():
        n = cell["n"]
        report[source] = {**cell,
            "target_rmse": float(np.sqrt(cell["target_squared"] / (n * 6))) if n else None,
            "raw_reference_rmse": float(np.sqrt(cell["raw_squared"] / (n * 6))) if n else None,
            "clean_reference_rmse": float(np.sqrt(cell["clean_squared"] / (n * 6))) if n else None,
            "gripper_accuracy": cell["gripper_correct"] / n if n else None}
    return {"by_source": report, "prediction": "unclipped canonical first action",
            "split": "validation", "stride": config["validation_stride"],
            "unique_scenes": sorted({r["seed_env"] for r in dataset.rows})}


def train(args) -> dict:
    config = read_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(config["torch_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = EpisodeManifest.load(args.manifest)
    statistics = json.loads(args.statistics.read_text())
    validate_statistics(statistics, manifest)
    training = PoolDataset(manifest, "train", k_hist=config["k_hist"],
                           cache_mib=config["episode_cache_mib"], level=config["level"])
    validation = PoolDataset(manifest, "validation", k_hist=config["k_hist"],
                             cache_mib=min(config["episode_cache_mib"], 512), level=config["level"])
    sampler = PairedSourceSampler(training, config["source_weights"], config["batch_size"],
                                 config["sampling_seed"])
    code_hashes = {"trainer": sha256(__file__),
                   "data": sha256(Path(__file__).with_name("three_source_data.py"))}
    run_identity = {"config": config, "manifest_sha256": manifest.manifest_sha256,
                    "data_file_hashes": manifest.file_hashes,
                    "statistics_sha256": statistics["statistics_sha256"], "code_hashes": code_hashes}
    run_hash = json_hash(run_identity)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "latest.pt"
    run_path = output / "run.json"
    if run_path.exists():
        if json.loads(run_path.read_text())["run_sha256"] != run_hash:
            raise ValueError("output belongs to a different experiment")
        if not args.resume:
            raise FileExistsError("output run already exists; use --resume for this same run")
    elif args.resume:
        raise FileNotFoundError("no matching run exists to resume")
    else:
        atomic_json(run_path, {"run_sha256": run_hash, **run_identity})

    random.seed(config["seed"])
    np.random.seed(config["seed"] % (2**32))
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    policy = PortalTracePolicy(policy_config(config)).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config["learning_rate"])
    parameter_count = sum(p.numel() for p in policy.parameters())
    counts = {s: 0 for s in SOURCES}
    source_frames = {s: 0 for s in SOURCES}
    clip_counts = {"intervened_continuous_elements_outside_box": 0}
    previous_chain = "0" * 64
    start_step, elapsed_before = 0, 0.
    curve = []
    if args.resume:
        payload = torch.load(latest, map_location="cpu", weights_only=True)
        if payload["run_sha256"] != run_hash:
            raise ValueError("resume checkpoint belongs to another experiment")
        policy.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        start_step, elapsed_before = payload["step"], payload["elapsed_seconds"]
        counts, source_frames = payload["source_sample_counts"], payload["source_action_frame_counts"]
        previous_chain, curve = payload["sampling_chain_sha256"], payload["training_curve"]
        clip_counts = payload["label_range_counts"]
        torch.set_rng_state(payload["rng"]["torch"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
    rms = torch.tensor(statistics["action_rms"], dtype=torch.float32, device=device)
    started = time.monotonic()
    interval = {"loss": 0., "continuous": 0., "gripper": 0., "count": 0}
    policy.train()

    def checkpoint(step, validation_report):
        payload = {"schema": CHECKPOINT_SCHEMA, "run_identity": run_identity,
                   "run_sha256": run_hash, "config": config, "statistics": statistics,
                   "step": step, "parameter_count": parameter_count,
                   "elapsed_seconds": elapsed_before + time.monotonic() - started,
                   "model_state": cpu_tree(policy.state_dict()),
                   "optimizer_state": cpu_tree(optimizer.state_dict()),
                   "source_sample_counts": counts.copy(), "source_action_frame_counts": source_frames.copy(),
                   "sampling_chain_sha256": previous_chain, "training_curve": curve.copy(),
                   "validation": validation_report, "label_range_counts": clip_counts.copy(),
                   "rng": {"torch": torch.get_rng_state(),
                           "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}}
        atomic_torch(latest, payload)
        destination = output / f"policy-step-{step}.pt"
        # One Adam state is enough for exact resumption. Scheduled snapshots
        # retain all learned weights, configuration and evidence, without six
        # duplicate optimizer histories per model.
        snapshot = {k: v for k, v in payload.items() if k not in {"optimizer_state", "rng"}}
        atomic_torch(destination, snapshot)
        return destination

    last_validation = payload.get("validation") if args.resume else None
    for step0 in range(start_step, config["updates"]):
        indices = sampler.indices(step0)
        previous_chain = hashlib.sha256(bytes.fromhex(previous_chain) + indices.astype("<i8").tobytes()).hexdigest()
        batch, source_ids = training.batch(indices, mode=config["label_mode"])
        for source_id, source in enumerate(SOURCES):
            selected = source_ids == source_id
            counts[source] += int(selected.sum())
            source_frames[source] += int(batch.action_pad_mask[selected].sum())
        clip_counts["intervened_continuous_elements_outside_box"] += int(
            ((batch.actions[..., :6].abs() > 1.0001) & batch.action_pad_mask[..., None]).sum())
        batch = prepare_batch(batch, statistics, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = policy(Batch(obs=batch.obs)).action_chunk
        loss, continuous, gripper = structured_action_loss(
            prediction, batch.actions, continuous_rms=rms, pad_mask=batch.action_pad_mask,
            loss_kind=STANDARDIZED_ACTION_LOSS)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step0 + 1}")
        loss.backward()
        if config["gradient_clip_norm"] is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True)
        optimizer.step()
        interval["loss"] += float(loss.detach())
        interval["continuous"] += float(continuous.detach())
        interval["gripper"] += float(gripper.detach())
        interval["count"] += 1
        step = step0 + 1
        if step % config["log_every"] == 0 or step == config["updates"]:
            row = {"step": step, "elapsed_seconds": elapsed_before + time.monotonic() - started,
                   **{k: interval[k] / interval["count"] for k in ("loss", "continuous", "gripper")},
                   "source_samples": counts.copy(), "sampling_chain_sha256": previous_chain}
            curve.append(row)
            atomic_json(output / "progress.json", {"status": "training", "run_sha256": run_hash,
                        "total_updates": config["updates"], "parameter_count": parameter_count, **row})
            print(json.dumps(row), flush=True)
            interval = {"loss": 0., "continuous": 0., "gripper": 0., "count": 0}
        if step % config["evaluate_every"] == 0 or step == config["updates"]:
            last_validation = evaluate(policy, validation, statistics, config, device)
            last_validation["step"] = step
            atomic_json(output / f"validation-step-{step}.json", last_validation)
        if step % config["checkpoint_every"] == 0 or step == config["updates"]:
            checkpoint(step, last_validation)
    if start_step >= config["updates"]:
        final = output / f"policy-step-{start_step}.pt"
    else:
        final = output / f"policy-step-{config['updates']}.pt"
    summary = {"status": "completed", "run_sha256": run_hash, "config": config,
               "step": config["updates"], "checkpoint": str(final), "checkpoint_sha256": sha256(final),
               "source_sample_counts": counts, "source_action_frame_counts": source_frames,
               "sampling_chain_sha256": previous_chain, "parameter_count": parameter_count,
               "elapsed_seconds": elapsed_before + time.monotonic() - started,
               "validation": last_validation, "label_range_counts": clip_counts}
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "progress.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--fit-statistics", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.fit_statistics:
        if args.statistics.exists():
            raise FileExistsError("statistics file already exists")
        manifest = EpisodeManifest.load(args.manifest)
        result = fit_statistics(manifest)
        atomic_json(args.statistics, result)
        print(json.dumps({"statistics": str(args.statistics), "sha256": result["statistics_sha256"]}))
        return
    if args.config is None or args.output is None:
        parser.error("training requires --config and --output")
    print(json.dumps(train(args)), flush=True)


if __name__ == "__main__":
    main()
