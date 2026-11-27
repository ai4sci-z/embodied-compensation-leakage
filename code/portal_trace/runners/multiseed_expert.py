"""Engineering-only assembly of independent expert episodes across env seeds.

The expensive part of expert generation remains in :mod:`runners.pilot` and
image rendering remains in :mod:`runners.materialise`.  This module only joins
their one-episode materialised artifacts while retaining a per-episode binding
to the exact source bytes, seed, code revision, and collection configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

from portal_trace.runners.dataset_contract import (
    require_analysis_trace_contract,
    require_current_defect_state_contract,
    require_dataset_identity,
    require_delta_value_contract,
)
from portal_trace.runners.formal_adapter_pilot import describe_formal_dataset

MULTISEED_EXPERT_SCHEMA = "portal_trace.engineering_multiseed_expert/1"
MULTISEED_EXPERT_MARKER = "PORTAL_TRACE_ENGINEERING_MULTISEED_EXPERT_NOT_PAPER_RESULT"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _json_value(value: Any) -> Any:
    """Convert HDF5 scalar metadata to canonical JSON-compatible values."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"unsupported HDF5 metadata value {type(value).__name__}")


@dataclass(frozen=True)
class ExpertSource:
    path: Path
    sha256: str
    raw_path: Path | None
    raw_sha256: str | None
    source_demo: str
    seed_env: int
    source_commit: str
    config: dict[str, Any]
    config_sha256: str
    horizon: int
    task: str
    family: str
    defect: str
    level: int
    clean_of: str

    def manifest_row(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "raw_path": None if self.raw_path is None else str(self.raw_path),
            "raw_sha256": self.raw_sha256,
            "source_demo": self.source_demo,
            "seed_env": self.seed_env,
            "source_commit": self.source_commit,
            "config": self.config,
            "config_sha256": self.config_sha256,
            "horizon": self.horizon,
        }


def inspect_expert_source(path: Path, *, raw_path: Path | None = None) -> ExpertSource:
    """Validate one materialised, one-episode expert source and bind its metadata."""
    import h5py

    path = Path(path).resolve()
    descriptor = describe_formal_dataset(path)
    if descriptor.n_demos != 1:
        raise ValueError(
            f"{path}: each multi-seed source must contain exactly one episode, "
            f"found {descriptor.n_demos}"
        )
    require_dataset_identity(
        path,
        task=descriptor.task,
        family=descriptor.family,
        defect=descriptor.defect,
        level=descriptor.level,
        clean_of=descriptor.clean_of,
        min_demos=1,
    )
    require_current_defect_state_contract(
        path, requested_defect=descriptor.clean_of or descriptor.defect
    )
    require_analysis_trace_contract(path)
    require_delta_value_contract(path)
    with h5py.File(path, "r") as dataset:
        names = sorted(dataset["data"], key=lambda name: int(name.split("_")[-1]))
        demo_name = names[0]
        demo = dataset["data"][demo_name]
        raw_seed = demo.attrs.get("seed_env")
        if isinstance(raw_seed, bool) or not isinstance(raw_seed, Integral):
            raise ValueError(f"{path}:data/{demo_name}: seed_env must be an integer")
        seed_env = int(raw_seed)
        if seed_env < 0:
            raise ValueError(f"{path}:data/{demo_name}: seed_env must be non-negative")
        horizon = int(demo.attrs.get("num_samples", len(demo["actions"])))
        if horizon < 1 or len(demo["actions"]) != horizon:
            raise ValueError(f"{path}:data/{demo_name}: inconsistent episode horizon")
        ilc_status = _json_value(demo.attrs.get("ilc_status", ""))
        clean_ilc_status = _json_value(demo.attrs.get("clean_ilc_status", ""))
        t_success = _json_value(demo.attrs.get("t_success", -1))
        if ilc_status != "converged" or clean_ilc_status != "converged":
            raise ValueError(
                f"{path}:data/{demo_name}: expert source requires converged defective "
                f"and clean ILC, got {ilc_status!r}/{clean_ilc_status!r}"
            )
        if isinstance(t_success, bool) or not isinstance(t_success, int) or t_success < 0:
            raise ValueError(
                f"{path}:data/{demo_name}: expert source requires persistent success"
            )
        source_commit = _json_value(demo.attrs.get("assets_commit", "unknown"))
        if not isinstance(source_commit, str) or not source_commit:
            raise ValueError(f"{path}:data/{demo_name}: assets_commit must be text")
        config = {
            "dataset_identity": {
                key: _json_value(dataset.attrs[key])
                for key in sorted(dataset.attrs)
                if key.startswith("dataset_")
            },
            "environment": _json_value(dataset.attrs.get("environment", "unknown")),
            "env_args": _json_value(dataset["data"].attrs.get("env_args", "")),
            "controller_config_sha256": _json_value(
                demo.attrs.get("controller_config_sha256", "unknown")
            ),
            "defect_impl_variant": _json_value(
                demo.attrs.get("defect_impl_variant", "unknown")
            ),
            "ilc_contact_variant": _json_value(
                demo.attrs.get("ilc_contact_variant", "unknown")
            ),
            "delta_json": _json_value(demo.attrs.get("delta_json", "unknown")),
            "demo_noise_scale": _json_value(demo.attrs.get("demo_noise_scale", 0.0)),
            "enable_numba": _json_value(demo.attrs.get("enable_numba", False)),
        }
    resolved_raw: Path | None = None
    raw_sha256: str | None = None
    if raw_path is not None:
        resolved_raw = Path(raw_path).resolve()
        require_dataset_identity(
            resolved_raw,
            task=descriptor.task,
            family=descriptor.family,
            defect=descriptor.defect,
            level=descriptor.level,
            clean_of=descriptor.clean_of,
            min_demos=1,
        )
        require_current_defect_state_contract(
            resolved_raw, requested_defect=descriptor.clean_of or descriptor.defect
        )
        require_analysis_trace_contract(resolved_raw)
        require_delta_value_contract(resolved_raw)
        with h5py.File(resolved_raw, "r") as raw:
            raw_names = sorted(raw["data"], key=lambda name: int(name.split("_")[-1]))
            if len(raw_names) != 1:
                raise ValueError(
                    f"{resolved_raw}: each raw source must contain exactly one episode"
                )
            raw_seed = raw["data"][raw_names[0]].attrs.get("seed_env")
            if isinstance(raw_seed, bool) or not isinstance(raw_seed, Integral):
                raise ValueError(f"{resolved_raw}: seed_env must be an integer")
            if int(raw_seed) != seed_env:
                raise ValueError(
                    f"raw/materialised seed mismatch: {int(raw_seed)} != {seed_env}"
                )
        raw_sha256 = _sha256(resolved_raw)
    config_json = _canonical_json(config)
    return ExpertSource(
        path=path,
        sha256=_sha256(path),
        raw_path=resolved_raw,
        raw_sha256=raw_sha256,
        source_demo=demo_name,
        seed_env=seed_env,
        source_commit=source_commit,
        config=config,
        config_sha256=hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        horizon=horizon,
        task=descriptor.task,
        family=descriptor.family,
        defect=descriptor.defect,
        level=descriptor.level,
        clean_of=descriptor.clean_of,
    )


def merge_expert_sources(
    sources: tuple[Path, ...],
    *,
    output: Path,
    producer_source_commit: str,
    raw_sources: tuple[Path, ...] | None = None,
    root_provenance: Mapping[str, str | int | float | bool] | None = None,
) -> tuple[ExpertSource, ...]:
    """Publish one episode-preserving HDF5 file from independent seed artifacts."""
    import h5py

    if len(sources) < 2:
        raise ValueError("multi-seed expert assembly requires at least two sources")
    if raw_sources is not None and len(raw_sources) != len(sources):
        raise ValueError("raw_sources must have one path for every materialised source")
    raw_paths = raw_sources if raw_sources is not None else (None,) * len(sources)
    inspected = tuple(
        sorted(
            (
                inspect_expert_source(path, raw_path=raw_path)
                for path, raw_path in zip(sources, raw_paths, strict=True)
            ),
            key=lambda item: item.seed_env,
        )
    )
    seeds = [item.seed_env for item in inspected]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"multi-seed expert sources contain duplicate seeds: {seeds}")
    identities = {
        (item.task, item.family, item.defect, item.level, item.clean_of)
        for item in inspected
    }
    if len(identities) != 1:
        raise ValueError(f"multi-seed expert sources mix dataset identities: {sorted(identities)}")
    config_hashes = {item.config_sha256 for item in inspected}
    if len(config_hashes) != 1:
        raise ValueError(
            "multi-seed expert sources use different collection/materialisation configs: "
            f"{sorted(config_hashes)}"
        )
    if (
        len(producer_source_commit) != 40
        or any(char not in "0123456789abcdef" for char in producer_source_commit.lower())
    ):
        raise ValueError("producer_source_commit must be a complete Git SHA-1")

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite multi-seed expert dataset {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        with h5py.File(inspected[0].path, "r") as first, h5py.File(temporary, "w") as target:
            for key, value in first.attrs.items():
                target.attrs[key] = value
            target.attrs["multiseed_expert_schema"] = MULTISEED_EXPERT_SCHEMA
            target.attrs["multiseed_expert_marker"] = MULTISEED_EXPERT_MARKER
            target.attrs["multiseed_expert_producer_commit"] = producer_source_commit
            target.attrs["multiseed_expert_seed_order_json"] = _canonical_json(seeds)
            for key, value in sorted((root_provenance or {}).items()):
                if not isinstance(key, str) or not key:
                    raise ValueError("multi-seed root provenance keys must be non-empty text")
                if key in target.attrs:
                    raise ValueError(f"multi-seed root provenance attribute {key!r} is reserved")
                if not isinstance(value, (str, int, float, bool)):
                    raise TypeError(
                        f"multi-seed root provenance attribute {key!r} must be an HDF5 scalar"
                    )
                target.attrs[key] = value
            data = target.create_group("data")
            data.attrs["total"] = sum(item.horizon for item in inspected)
            data.attrs["env_args"] = first["data"].attrs["env_args"]

            for index, item in enumerate(inspected):
                with h5py.File(item.path, "r") as source:
                    source.copy(source[f"data/{item.source_demo}"], data, name=f"demo_{index}")
                demo = data[f"demo_{index}"]
                demo.attrs["expert_source_path"] = str(item.path)
                demo.attrs["expert_source_sha256"] = item.sha256
                demo.attrs["expert_source_demo"] = item.source_demo
                demo.attrs["expert_source_commit"] = item.source_commit
                demo.attrs["expert_source_config_sha256"] = item.config_sha256
                demo.attrs["expert_seed_index"] = index
                if item.raw_path is not None:
                    demo.attrs["expert_raw_source_path"] = str(item.raw_path)
                    demo.attrs["expert_raw_source_sha256"] = item.raw_sha256
        try:
            os.link(temporary, output)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite multi-seed expert dataset {output}"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)
    return inspected
