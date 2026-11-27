"""Read-only HDF5 window adapter for the PORTAL-Trace policy interface.

This module owns temporal indexing and tensor layout only. Proprioception is
returned in stored units; the formal adapter applies the hash-bound global
training-split normalisation before a policy update. The formal
``runners.train`` publication path remains blocked until that adapter and its
deployment checkpoint contract are admitted there.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from portal_trace.policies.interface import Batch, Observation
from portal_trace.policies.obs import (
    ACTION_DIM,
    CAMERAS,
    CHUNK,
    IMAGE_SIZE,
    K_HIST_MAIN,
    PROPRIO_DIM,
)

__all__ = [
    "PolicyDatasetEntry",
    "RawPolicyWindowDataset",
    "collate_policy_samples",
]


@dataclass(frozen=True)
class PolicyDatasetEntry:
    """One validated materialised file and its categorical policy indices."""

    path: Path
    task_index: int
    embodiment_index: int
    n_demos: int
    partition_index: int | None = None
    family: str = ""
    defect: str = ""
    level: int | None = None

    def __post_init__(self) -> None:
        if self.task_index < 0 or self.embodiment_index < 0:
            raise ValueError("task and embodiment indices must be non-negative")
        if self.n_demos < 1:
            raise ValueError("a policy dataset entry must admit at least one demo")
        if self.partition_index is not None and (
            isinstance(self.partition_index, bool)
            or not isinstance(self.partition_index, int)
            or self.partition_index < 0
        ):
            raise ValueError("partition index must be a non-negative integer when provided")
        if not isinstance(self.family, str) or not isinstance(self.defect, str):
            raise TypeError("policy dataset family and defect metadata must be strings")
        if self.level is not None and (
            isinstance(self.level, bool) or not isinstance(self.level, int) or self.level < 0
        ):
            raise ValueError("policy dataset level must be a non-negative integer when provided")


@dataclass(frozen=True)
class _Window:
    entry: PolicyDatasetEntry
    demo: str
    time_index: int
    partition_index: int | None = None


class RawPolicyWindowDataset(Dataset[dict[str, Any]]):
    """Map materialised episodes to causal histories and action chunks.

    History is left-padded by repeating frame zero. Supervision is right-padded
    with zeros and accompanied by ``action_pad_mask``; padded values therefore
    cannot enter the behaviour-cloning loss. Files are opened inside
    ``__getitem__`` so DataLoader workers never inherit live h5py handles. The
    optional episode cache retains raw arrays (including uint8 RGB), bounded by
    their total array bytes. Its budget is per dataset instance / worker, not a
    process-wide memory cap; zero keeps the original uncached path.
    """

    def __init__(
        self,
        entries: tuple[PolicyDatasetEntry, ...],
        *,
        k_hist: int = K_HIST_MAIN,
        chunk: int = CHUNK,
        episode_ids: tuple[str, ...] | None = None,
        partition_indices: Mapping[str, int] | None = None,
        episode_cache_bytes: int = 0,
    ) -> None:
        if not entries:
            raise ValueError("policy dataset has no entries")
        if k_hist < 1:
            raise ValueError("k_hist must be positive")
        if chunk != CHUNK:
            raise ValueError(f"policy action chunk is fixed at {CHUNK}, got {chunk}")
        self.k_hist = int(k_hist)
        self.chunk = int(chunk)
        if (
            isinstance(episode_cache_bytes, bool)
            or not isinstance(episode_cache_bytes, int)
            or episode_cache_bytes < 0
        ):
            raise ValueError("episode_cache_bytes must be a non-negative integer")
        self.episode_cache_bytes = episode_cache_bytes
        self._episode_cache: OrderedDict[
            tuple[Path, str], tuple[dict[str, Any], int]
        ] = OrderedDict()
        self._episode_cache_size = 0
        selected_episodes = None if episode_ids is None else frozenset(episode_ids)
        if selected_episodes is not None and not selected_episodes:
            raise ValueError("policy dataset episode_ids must be non-empty")
        if partition_indices is not None:
            partition_indices = dict(partition_indices)
            if any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                for index in partition_indices.values()
            ):
                raise ValueError("partition indices must be non-negative integers")
        self._windows: list[_Window] = []

        import h5py

        for entry in entries:
            path = Path(entry.path)
            try:
                with h5py.File(path, "r") as dataset:
                    demos = sorted(
                        dataset["data"], key=lambda name: int(name.split("_")[1])
                    )
                    if len(demos) < entry.n_demos:
                        raise ValueError(
                            f"{path}: admits {entry.n_demos} demos but contains "
                            f"only {len(demos)}"
                        )
                    for demo in demos[: entry.n_demos]:
                        episode_id = f"{path.resolve()}::data/{demo}"
                        if selected_episodes is not None and episode_id not in selected_episodes:
                            continue
                        if partition_indices is not None and episode_id not in partition_indices:
                            raise ValueError(
                                f"partition_indices has no entry for selected episode {episode_id}"
                            )
                        group = dataset["data"][demo]
                        self._validate_demo(path, demo, group)
                        self._windows.extend(
                            _Window(
                                entry,
                                demo,
                                time_index,
                                (
                                    getattr(entry, "partition_index", None)
                                    if partition_indices is None
                                    else partition_indices.get(episode_id)
                                ),
                            )
                            for time_index in range(len(group["actions"]))
                        )
            except (OSError, KeyError) as exc:
                raise ValueError(f"{path}: unreadable materialised policy dataset") from exc
        if not self._windows:
            if selected_episodes is None:
                raise ValueError("policy dataset contains no windows")
            raise ValueError(
                "policy dataset episode_ids select no windows; check the plan paths and demos"
            )

    @staticmethod
    def _validate_demo(path: Path, demo: str, group: Any) -> None:
        where = f"{path}:data/{demo}"
        if "obs" not in group:
            raise ValueError(f"{where}: no materialised obs group")
        obs = group["obs"]
        required = (*CAMERAS, "robot0_proprio-state")
        missing = [name for name in required if name not in obs]
        if missing:
            raise ValueError(f"{where}: missing policy observations {missing}")
        horizon = len(group["actions"])
        if group["actions"].shape != (horizon, ACTION_DIM):
            raise ValueError(f"{where}: actions must have shape (T, {ACTION_DIM})")
        if "c_true" not in group or group["c_true"].shape != (horizon, ACTION_DIM):
            raise ValueError(f"{where}: c_true must have shape (T, {ACTION_DIM})")
        if obs["robot0_proprio-state"].shape != (horizon, PROPRIO_DIM):
            raise ValueError(f"{where}: proprio must have shape (T, {PROPRIO_DIM})")
        for camera in CAMERAS:
            if obs[camera].shape != (horizon, IMAGE_SIZE, IMAGE_SIZE, 3):
                raise ValueError(
                    f"{where}: {camera} must have shape (T, {IMAGE_SIZE}, "
                    f"{IMAGE_SIZE}, 3)"
                )

    def __len__(self) -> int:
        return len(self._windows)

    def window_identity(self, index: int) -> tuple[str, int]:
        """Return the immutable episode ID and control-step index for one sample."""
        window = self._windows[index]
        episode_id = f"{Path(window.entry.path).resolve()}::data/{window.demo}"
        return episode_id, window.time_index

    def action_squared_sum(self, n_dims: int) -> tuple[torch.Tensor, int]:
        """Aggregate chunk-weighted action squares without loading observations."""
        if (
            isinstance(n_dims, bool)
            or not isinstance(n_dims, int)
            or not 1 <= n_dims <= ACTION_DIM
        ):
            raise ValueError(f"action statistic dimensions must lie in [1, {ACTION_DIM}]")
        grouped: dict[tuple[Path, str], list[int]] = defaultdict(list)
        for window in self._windows:
            grouped[(Path(window.entry.path), window.demo)].append(window.time_index)

        import h5py

        squared = np.zeros(n_dims, dtype=np.float64)
        valid_rows = 0
        for (path, demo), time_indices in grouped.items():
            with h5py.File(path, "r") as dataset:
                actions = np.asarray(
                    dataset[f"data/{demo}/actions"][:, :n_dims], dtype=np.float64
                )
            counts = np.zeros(len(actions), dtype=np.float64)
            for time_index in time_indices:
                counts[time_index : min(time_index + self.chunk, len(actions))] += 1.0
            squared += (np.square(actions) * counts[:, None]).sum(axis=0)
            valid_rows += int(counts.sum())
        return torch.from_numpy(squared), valid_rows

    @contextmanager
    def _episode_group(self, window: _Window):
        """Yield an episode without retaining an HDF5 handle in the cache.

        Oversize episodes follow the existing window-read path. Eviction takes
        place before loading an admissible episode, so resident cached arrays
        never exceed the configured byte budget. Dataset files must remain
        immutable for the lifetime of a dataset, as required by its data plan.
        """
        import h5py

        key = (Path(window.entry.path), window.demo)
        cached = self._episode_cache.get(key)
        if cached is not None:
            self._episode_cache.move_to_end(key)
            yield cached[0]
            return
        with h5py.File(window.entry.path, "r") as dataset:
            group = dataset["data"][window.demo]
            if self.episode_cache_bytes == 0:
                yield group
                return
            names = (*CAMERAS, "robot0_proprio-state")
            sources = [group["obs"][name] for name in names]
            sources.extend((group["actions"], group["c_true"]))
            required_bytes = sum(int(value.size) * value.dtype.itemsize for value in sources)
            if required_bytes > self.episode_cache_bytes:
                yield group
                return
            while self._episode_cache_size + required_bytes > self.episode_cache_bytes:
                _, (_, evicted_bytes) = self._episode_cache.popitem(last=False)
                self._episode_cache_size -= evicted_bytes
            payload = {
                "obs": {name: group["obs"][name][()] for name in names},
                "actions": group["actions"][()],
                "c_true": group["c_true"][()],
            }
        self._episode_cache[key] = (payload, required_bytes)
        self._episode_cache_size += required_bytes
        yield payload

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self._windows[index]
        with self._episode_group(window) as group:
            t = window.time_index

            def causal_history(source: Any, *, dtype: Any | None = None) -> np.ndarray:
                start = max(0, t - self.k_hist + 1)
                values = np.asarray(source[start : t + 1], dtype=dtype)
                if isinstance(source, np.ndarray):
                    # A cached slice can share memory with the whole episode.
                    # Give every returned tensor private, writable storage;
                    # callers may transform it in place without corrupting a
                    # future sample, and torch sees no read-only NumPy array.
                    values = values.copy()
                n_pad = self.k_hist - len(values)
                if n_pad:
                    values = np.concatenate(
                        (np.repeat(values[:1], n_pad, axis=0), values), axis=0
                    )
                return values

            images = {
                camera: torch.from_numpy(
                    causal_history(group["obs"][camera]).transpose(0, 3, 1, 2)
                ).to(torch.float32)
                / 255.0
                for camera in CAMERAS
            }
            proprio = torch.from_numpy(
                causal_history(
                    group["obs"]["robot0_proprio-state"], dtype=np.float32
                )
            )
            horizon = len(group["actions"])
            stop = min(t + self.chunk, horizon)
            n_real = stop - t
            actions = torch.zeros((self.chunk, ACTION_DIM), dtype=torch.float32)
            c_true = torch.zeros_like(actions)
            actions[:n_real] = torch.from_numpy(
                np.asarray(group["actions"][t:stop], dtype=np.float32)
            )
            c_true[:n_real] = torch.from_numpy(
                np.asarray(group["c_true"][t:stop], dtype=np.float32)
            )
            pad_mask = torch.zeros(self.chunk, dtype=torch.bool)
            pad_mask[:n_real] = True
        return {
            "images": images,
            "proprio": proprio,
            "actions": actions,
            "c_true": c_true,
            "action_pad_mask": pad_mask,
            "task_index": torch.tensor(window.entry.task_index, dtype=torch.long),
            "embodiment_index": torch.tensor(
                window.entry.embodiment_index, dtype=torch.long
            ),
            "partition_index": (
                None
                if window.partition_index is None
                else torch.tensor(window.partition_index, dtype=torch.long)
            ),
        }


def collate_policy_samples(samples: list[dict[str, Any]]) -> Batch:
    """Stack raw window samples into the policy's typed batch carrier."""
    if not samples:
        raise ValueError("cannot collate an empty policy batch")
    if any(sample["partition_index"] is None for sample in samples) and any(
        sample["partition_index"] is not None for sample in samples
    ):
        raise ValueError("partition_index must be present for every sample or none")
    images = {
        camera: torch.stack([sample["images"][camera] for sample in samples])
        for camera in CAMERAS
    }
    obs = Observation(
        images=images,
        proprio=torch.stack([sample["proprio"] for sample in samples]),
        task_index=torch.stack([sample["task_index"] for sample in samples]),
        embodiment_index=torch.stack(
            [sample["embodiment_index"] for sample in samples]
        ),
        partition_index=(
            None
            if all(sample["partition_index"] is None for sample in samples)
            else torch.stack(
                [sample["partition_index"] for sample in samples]
            )
        ),
    )
    return Batch(
        obs=obs,
        actions=torch.stack([sample["actions"] for sample in samples]),
        action_pad_mask=torch.stack(
            [sample["action_pad_mask"] for sample in samples]
        ),
        c_true=torch.stack([sample["c_true"] for sample in samples]),
    )
