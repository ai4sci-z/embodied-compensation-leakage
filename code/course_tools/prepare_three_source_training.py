#!/usr/bin/env python3
"""Bind the complete three-source dataset to 33 matched training configurations.

This command prepares immutable files. It never launches training.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

QUEUE_SCHEMA = "dl_course.three_source_training_queue/1"
ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("clean", "delay", "backlash", "droop")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def json_bytes(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def immutable_write(path: Path, content: bytes):
    """An interrupted preparation can continue, but existing bytes cannot change."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"existing immutable file differs: {path}")
        return
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("xb") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != content:
            raise
    finally:
        temporary.unlink(missing_ok=True)


def conditions():
    balanced = dict.fromkeys(SOURCES, 0.25)
    out = []
    for label in ("raw", "remove-delay", "remove-backlash", "remove-droop", "remove-all"):
        out.append((f"L1-balanced-{label}", 1, label, balanced.copy()))
    for source in SOURCES[1:]:
        weights = dict.fromkeys(SOURCES, 0.125)
        weights[source] = 0.625
        out.append((f"L1-emphasize-{source}-raw", 1, "raw", weights))
    for label in ("raw", "remove-all"):
        out.append((f"L2-balanced-{label}", 2, label, balanced.copy()))
    out.append(("clean-only", 1, "raw", {s: float(s == "clean") for s in SOURCES}))
    return out


def training_config(seed, sampling_seed, level, label, weights):
    return {
        "schema": "dl_course.three_source_training/1", "seed": seed,
        "sampling_seed": sampling_seed, "updates": 30000, "batch_size": 16,
        "source_weights": weights, "label_mode": label, "level": level,
        "learning_rate": 3e-4, "d_model": 512, "n_layers": 4, "n_heads": 8,
        "ffn_dim": 2048, "dropout": 0.1, "augment": False, "k_hist": 5,
        "checkpoint_every": 10000, "evaluate_every": 10000, "log_every": 100,
        "episode_cache_mib": 6144, "torch_threads": 4,
        "gradient_clip_norm": None, "validation_stride": 1,
    }


def validate_complete_manifest(payload, protocol, summary):
    """Match each intended scene/source/level, including one normal per scene."""
    if payload.get("schema") != "dl_course.three_source_episodes/1":
        raise ValueError("wrong episode manifest schema")
    if payload.get("errors"):
        raise ValueError("manifest contains failed materialisations")
    if summary.get("status") != "complete" or summary.get("errors", 0) != 0:
        raise ValueError("materialisation has not completed without execution errors")
    splits = protocol["splits"]
    if {k: len(v) for k, v in splits.items()} != {"train": 48, "validation": 12, "test": 24}:
        raise ValueError("expected the declared 48/12/24 scene protocol")
    all_seeds = [seed for values in splits.values() for seed in values]
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("protocol scene seeds overlap")
    cells = Counter((cell["source"], cell["level"]) for cell in protocol["cells"])
    intended = Counter((s, level) for s in SOURCES[1:] for level in (1, 2))
    if cells != intended:
        raise ValueError("expected all three sources at both levels, once each")
    expected = Counter((split, seed, source, level)
        for split, seeds in splits.items() for seed in seeds for source, level in cells)
    observed = Counter()
    clean = Counter()
    for row in payload.get("episodes", []):
        if row.get("status", "complete") != "complete":
            raise ValueError("manifest includes unfinished episodes")
        if row["source"] == "clean":
            clean[(row["split"], row["seed_env"])] += 1
        else:
            observed[(row["split"], row["seed_env"], row["source"], row["level"])] += 1
    expected_clean = Counter((split, seed) for split, seeds in splits.items() for seed in seeds)
    if observed != expected or clean != expected_clean:
        raise ValueError("manifest is incomplete or duplicates a planned source/scene")
    count = sum(observed.values()) + sum(clean.values())
    if summary.get("rendered") != count or summary.get("expected_episodes") != count:
        raise ValueError("materialisation summary does not match the complete manifest")
    if count != 588:
        raise ValueError("expected 588 rendered episodes")
    return count


def prepare(args):
    # Import the actual data/normalisation implementation only for preparation.
    from course_tools.three_source_data import EpisodeManifest, validate_statistics
    from course_tools.train_three_source_pool import read_config
    from portal_trace.policies.obs import CHUNK

    if CHUNK != 8:
        raise ValueError("the agreed policy uses eight-action prediction chunks")

    manifest_path = args.manifest.resolve()
    stats_path = args.statistics.resolve()
    protocol_path = args.collection_protocol.resolve()
    summary_path = args.materialise_summary.resolve()
    output = args.output.resolve()
    payload = json.loads(manifest_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    summary = json.loads(summary_path.read_text())
    if payload.get("collection_protocol_sha256") != sha256(protocol_path):
        raise ValueError("manifest is bound to another collection protocol")
    count = validate_complete_manifest(payload, protocol, summary)
    manifest = EpisodeManifest.load(manifest_path)
    statistics = json.loads(stats_path.read_text())
    validate_statistics(statistics, manifest)
    if any(row["horizon"] != protocol["horizon"] for row in manifest.rows):
        raise ValueError("episode horizon differs from the complete-task protocol")
    if len(args.seeds) != 3 or len(set(args.seeds)) != 3 or min(args.seeds) < 0:
        raise ValueError("provide exactly three distinct nonnegative training seeds")
    code_hashes = {"trainer": sha256(ROOT / "course_tools/train_three_source_pool.py"),
                   "data": sha256(ROOT / "course_tools/three_source_data.py")}
    bindings = {
        "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "statistics": {"path": str(stats_path), "sha256": sha256(stats_path)},
        "collection_protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path)},
        "materialise_summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
    }
    identity_shared = {"manifest_sha256": manifest.manifest_sha256,
        "data_file_hashes": manifest.file_hashes,
        "statistics_sha256": statistics["statistics_sha256"], "code_hashes": code_hashes}
    runs = []
    # Each seed completes the full scientific comparison before moving to the next.
    for index, seed in enumerate(args.seeds):
        sampling_seed = args.sampling_seed_base + index
        for condition, level, label, weights in conditions():
            run_id = f"{condition}-seed-{seed}"
            cfg_path = output / "configs" / f"{run_id}.json"
            immutable_write(cfg_path, json_bytes(training_config(seed, sampling_seed, level, label, weights)))
            cfg = read_config(cfg_path)
            runs.append({"id": run_id, "condition": condition, "config": str(cfg_path),
                "config_sha256": sha256(cfg_path), "output": str(output / "runs" / run_id),
                "run_sha256": json_hash({"config": cfg, **identity_shared}),
                "seed": seed, "sampling_seed": sampling_seed, "updates": cfg["updates"],
                "sampling_group": f"balanced-L{level}-seed-{seed}" if "balanced" in condition else run_id})
    for name, binding in bindings.items():
        immutable_write(output / "frozen" / f"{name}.json", Path(binding["path"]).read_bytes())
    queue = {"schema": QUEUE_SCHEMA, "project_root": str(ROOT), "bindings": bindings,
             "shared_run_identity": identity_shared, "episode_count": count,
             "runs": runs, "action_chunk": 8,
             "data_policy": "complete manifest; fixed files and original-label training statistics",
             "run_order": "all 11 conditions per seed; final step 30000"}
    queue["queue_sha256"] = json_hash(queue)
    immutable_write(output / "queue.json", json_bytes(queue))
    return {"status": "prepared", "queue": str(output / "queue.json"),
            "queue_sha256": queue["queue_sha256"], "runs": len(runs), "training_started": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--statistics", type=Path, required=True)
    p.add_argument("--collection-protocol", type=Path, required=True)
    p.add_argument("--materialise-summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs=3, default=[2026111801, 2026111802, 2026111803])
    p.add_argument("--sampling-seed-base", type=int, default=20261118)
    args = p.parse_args()
    if args.sampling_seed_base < 0:
        p.error("sampling-seed-base must be nonnegative")
    print(json.dumps(prepare(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
