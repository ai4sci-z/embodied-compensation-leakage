"""Shared data, sampling and normalization for the compensation-pooling study.

The source name is supervision metadata; it is never passed to the policy.
Existing PORTAL-Trace temporal indexing, padding and observation layouts are
used unchanged. Statistics always come from the complete manifest training
split, not from intervention labels or a run's source mixture.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from portal_trace.policies.interface import Batch, Observation
from portal_trace.policies.obs import (
    CHUNK, PROPRIO_DIM, PROPRIO_GROUPS, NormKind, PROPRIO_ACCELERATION_CAUSAL,
)
from portal_trace.runners.policy_dataset import (
    PolicyDatasetEntry, RawPolicyWindowDataset, collate_policy_samples,
)

SOURCES = ("clean", "delay", "backlash", "droop")
LABEL_MODES = ("raw", "remove-delay", "remove-backlash", "remove-droop", "remove-all")
MANIFEST_SCHEMA = "dl_course.three_source_episodes/1"
STATS_SCHEMA = "dl_course.three_source_statistics/1"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass
class EpisodeManifest:
    path: Path
    payload: dict
    rows: list[dict]
    manifest_sha256: str
    file_hashes: dict[str, str]

    @classmethod
    def load(cls, path: str | Path, *, verify_hashes: bool = True) -> "EpisodeManifest":
        path = Path(path).resolve()
        payload = json.loads(path.read_text())
        if payload.get("schema") != MANIFEST_SCHEMA:
            raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA}")
        rows = []
        seen = set()
        seed_splits = defaultdict(set)
        for original in payload.get("episodes", []):
            row = dict(original)
            for key in ("path", "demo", "source", "seed_env", "split", "level"):
                if key not in row:
                    raise ValueError(f"episode is missing {key}: {original}")
            p = Path(row["path"])
            row["path"] = str((p if p.is_absolute() else path.parent / p).resolve())
            row["id"] = f"{row['path']}::data/{row['demo']}"
            if row["id"] in seen:
                raise ValueError(f"duplicate episode {row['id']}")
            seen.add(row["id"])
            if row["source"] not in SOURCES or row["split"] not in ("train", "validation", "test"):
                raise ValueError(f"invalid source/split: {row}")
            if not isinstance(row["seed_env"], int) or not isinstance(row["level"], int):
                raise ValueError("seed_env and level must be integers")
            seed_splits[row["seed_env"]].add(row["split"])
            rows.append(row)
        if not rows or any(len(splits) > 1 for splits in seed_splits.values()):
            raise ValueError("manifest is empty or a scene appears in multiple splits")
        if not all(any(r["split"] == split for r in rows) for split in ("train", "validation")):
            raise ValueError("both train and validation splits are required")
        tasks = {row.get("task", payload.get("task", "Stack")) for row in rows}
        if len(tasks) != 1:
            raise ValueError("this shared experiment uses one fixed task")
        hashes = {}
        for p in sorted({r["path"] for r in rows}):
            hashes[p] = sha256(p) if verify_hashes else "not_checked"
            with h5py.File(p, "r") as handle:
                semantics = handle.attrs.get("proprio_acceleration_semantics",
                                             payload.get("proprio_acceleration_semantics"))
                if isinstance(semantics, bytes):
                    semantics = semantics.decode()
                if semantics != PROPRIO_ACCELERATION_CAUSAL:
                    raise ValueError(f"{p}: missing or non-causal acceleration convention")
                for row in (r for r in rows if r["path"] == p):
                    if verify_hashes and row.get("sha256") and row["sha256"] != hashes[p]:
                        raise ValueError(f"dataset hash mismatch: {p}")
                    group = handle[f"data/{row['demo']}"]
                    RawPolicyWindowDataset._validate_demo(Path(p), row["demo"], group)
                    for key in ("actions", "c_true"):
                        values = np.asarray(group[key])
                        if not np.isfinite(values).all():
                            raise ValueError(f"non-finite {p}/{row['demo']}/{key}")
                    c = np.asarray(group["c_true"])
                    if not np.allclose(c[:, 6], 0, atol=1e-7):
                        raise ValueError("this study intervenes on six motion dimensions; gripper c_true must be zero")
                    if row["source"] == "clean" and not np.allclose(c, 0, atol=1e-7):
                        raise ValueError("clean-reference episodes must have c_true=0")
                    actions = np.asarray(group["actions"])
                    if np.max(np.abs(actions)) > 1.00001:
                        raise ValueError("canonical action labels must be in [-1,1]")
                    # c_true is a difference of two clipped teacher actions.
                    clean = actions[:, :6] - c[:, :6]
                    if np.max(np.abs(clean)) > 1.0001:
                        raise ValueError("actions-c_true does not reconstruct a valid clean teacher action")
                    row["horizon"] = len(actions)
        return cls(path, payload, sorted(rows, key=lambda r: (r["split"], r["source"],
                       r["seed_env"], r["level"], r["id"])), sha256(path), hashes)


class PoolDataset:
    def __init__(self, manifest: EpisodeManifest, split: str, *, k_hist: int = 5,
                 cache_mib: int = 1024, level: int | None = None):
        self.manifest, self.split = manifest, split
        self.rows = [r for r in manifest.rows if r["split"] == split
                     and (level is None or r["source"] == "clean" or r["level"] == level)]
        if not self.rows:
            raise ValueError(f"no episodes for {split}, level={level}")
        entries = []
        for p in sorted({r["path"] for r in self.rows}):
            with h5py.File(p, "r") as handle:
                n_demos = len(handle["data"])
            entries.append(PolicyDatasetEntry(Path(p), 0, 0, n_demos))
        self.raw = RawPolicyWindowDataset(tuple(entries), k_hist=k_hist,
                    episode_ids=tuple(r["id"] for r in self.rows),
                    episode_cache_bytes=cache_mib * 1024 ** 2)
        lookup = {r["id"]: r for r in self.rows}
        self.source_ids = np.empty(len(self.raw), dtype=np.int64)
        self.by_source = {s: [] for s in SOURCES}
        self.by_source_scene = {s: defaultdict(list) for s in SOURCES}
        for idx in range(len(self.raw)):
            episode, _ = self.raw.window_identity(idx)
            row = lookup[episode]
            self.source_ids[idx] = SOURCES.index(row["source"])
            self.by_source[row["source"]].append(idx)
            self.by_source_scene[row["source"]][row["seed_env"]].append(idx)
        self.by_source = {s: np.asarray(v, dtype=np.int64) for s, v in self.by_source.items()}
        self.by_source_scene = {s: {seed: np.asarray(v, dtype=np.int64) for seed, v in cells.items()}
                                for s, cells in self.by_source_scene.items()}

    def __len__(self):
        return len(self.raw)

    def batch(self, indices, *, mode: str = "raw") -> tuple[Batch, torch.Tensor]:
        if mode not in LABEL_MODES:
            raise ValueError(f"unknown label mode {mode}")
        indices = np.asarray(indices, dtype=np.int64)
        batch = collate_policy_samples([self.raw[int(i)] for i in indices])
        source_ids = torch.from_numpy(self.source_ids[indices].copy())
        # The unconditioned policy is never allowed to see source identities.
        batch.obs.embodiment_index = None
        batch.obs.partition_index = None
        batch.extras["raw_actions"] = batch.actions.clone()
        if mode != "raw":
            selected = source_ids != 0 if mode == "remove-all" else source_ids == SOURCES.index(mode[7:])
            target = batch.actions.clone()
            target[selected, :, :6] -= batch.c_true[selected, :, :6]
            batch.actions = target
        return batch, source_ids


class PairedSourceSampler:
    """A step-addressable sampler independent of labels and model RNG.

    Within each batch the requested source counts use randomized systematic
    allocation; integral proportions are exact. Scenes are uniform within
    source, followed by a uniform control step. All sources must cover the same
    training scenes so a source-weight intervention does not change scene support.
    """
    def __init__(self, dataset: PoolDataset, weights: dict[str, float], batch_size: int, seed: int):
        if set(weights) != set(SOURCES):
            raise ValueError(f"weights must name {SOURCES}")
        w = np.asarray([weights[s] for s in SOURCES], dtype=np.float64)
        if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
            raise ValueError("invalid source weights")
        self.weights = w / w.sum()
        self.dataset, self.batch_size, self.seed = dataset, int(batch_size), int(seed)
        if batch_size < 1 or seed < 0:
            raise ValueError("invalid batch size or sampling seed")
        active = [s for s, weight in zip(SOURCES, w) if weight > 0]
        scene_sets = [set(dataset.by_source_scene[s]) for s in active]
        if not scene_sets or not scene_sets[0] or any(x != scene_sets[0] for x in scene_sets):
            raise ValueError("active sources must contain the same set of scenes")
        self.scenes = np.asarray(sorted(scene_sets[0]), dtype=np.int64)

    def indices(self, step: int) -> np.ndarray:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(step)]))
        u = (np.arange(self.batch_size) + rng.random()) / self.batch_size
        ids = np.searchsorted(np.cumsum(self.weights), u, side="right").clip(max=3)
        rng.shuffle(ids)
        result = []
        for source_id in ids:
            scene = int(rng.choice(self.scenes))
            choices = self.dataset.by_source_scene[SOURCES[int(source_id)]][scene]
            result.append(int(rng.choice(choices)))
        return np.asarray(result, dtype=np.int64)


def fit_statistics(manifest: EpisodeManifest) -> dict:
    """Global statistics, fitted once to all training rows before label removal."""
    total = np.zeros(PROPRIO_DIM, dtype=np.float64)
    second = total.copy()
    action_second = np.zeros(6, dtype=np.float64)
    n, action_n = 0, 0
    for row in (r for r in manifest.rows if r["split"] == "train"):
        with h5py.File(row["path"], "r") as handle:
            group = handle[f"data/{row['demo']}"]
            prop = np.asarray(group["obs/robot0_proprio-state"], dtype=np.float64)
            actions = np.asarray(group["actions"][:, :6], dtype=np.float64)
        if not np.isfinite(prop).all():
            raise ValueError("non-finite training proprioception")
        total += prop.sum(0)
        second += np.square(prop).sum(0)
        n += len(prop)
        multiplicity = np.minimum(np.arange(len(actions)) + 1, CHUNK)
        action_second += (np.square(actions) * multiplicity[:, None]).sum(0)
        action_n += int(multiplicity.sum())
    means = total / n
    scales = np.sqrt(np.maximum(second / n - np.square(means), 0)).clip(min=1e-3)
    for group in PROPRIO_GROUPS:
        if group.norm in (NormKind.NONE, NormKind.UNIT_NORM):
            means[group.slice] = 0
        if group.norm is not NormKind.STANDARDISE:
            scales[group.slice] = 1
    result = {"schema": STATS_SCHEMA, "manifest_sha256": manifest.manifest_sha256,
              "file_hashes": manifest.file_hashes, "fitted_rows": n,
              "action_rows_chunk_weighted": action_n, "means": means.tolist(),
              "scales": scales.tolist(),
              "action_rms": np.sqrt(action_second / action_n).clip(min=1e-3).tolist(),
              "fit_rule": "all manifest training episodes; original labels; identical across mixtures and interventions"}
    result["statistics_sha256"] = json_hash(result)
    return result


def validate_statistics(statistics: dict, manifest: EpisodeManifest | None = None) -> None:
    if statistics.get("schema") != STATS_SCHEMA:
        raise ValueError("unknown statistics schema")
    unsigned = {k: v for k, v in statistics.items() if k != "statistics_sha256"}
    if json_hash(unsigned) != statistics.get("statistics_sha256"):
        raise ValueError("statistics checksum mismatch")
    if manifest is not None and (statistics["manifest_sha256"] != manifest.manifest_sha256
                                or statistics["file_hashes"] != manifest.file_hashes):
        raise ValueError("statistics are bound to different data")


def normalize_observation(obs: Observation, statistics: dict, device) -> Observation:
    if obs.proprio is None:
        raise ValueError("shared policy needs proprioception")
    prop = obs.proprio.to(device=device, dtype=torch.float32)
    means = torch.as_tensor(statistics["means"], dtype=prop.dtype, device=device)
    scales = torch.as_tensor(statistics["scales"], dtype=prop.dtype, device=device)
    normalized = (prop - means) / scales
    for group in PROPRIO_GROUPS:
        if group.norm is NormKind.UNIT_NORM:
            q = prop[..., group.slice]
            normalized[..., group.slice] = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
        elif group.norm is NormKind.NONE:
            normalized[..., group.slice] = prop[..., group.slice]
    return Observation(images={k: v.to(device=device, dtype=torch.float32) for k, v in obs.images.items()},
                       proprio=normalized,
                       task_index=torch.zeros(obs.batch_size, dtype=torch.long, device=device),
                       embodiment_index=None, partition_index=None)


def prepare_batch(batch: Batch, statistics: dict, device) -> Batch:
    return Batch(obs=normalize_observation(batch.obs, statistics, device),
                 actions=batch.actions.to(device), action_pad_mask=batch.action_pad_mask.to(device),
                 c_true=batch.c_true.to(device),
                 extras={k: v.to(device) for k, v in batch.extras.items()})
