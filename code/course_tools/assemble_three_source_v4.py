"""Assemble fixed-scene three-source data with only corrected backlash physics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from course_tools.collect_limit_backlash import atomic_json, sha256
from course_tools.prepare_three_source_training import validate_complete_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=Path("course_results/three-source-v2"))
    parser.add_argument("--backlash", type=Path, default=Path("course_results/three-source-backlash-v4"))
    parser.add_argument("--output", type=Path, default=Path("course_results/three-source-v4"))
    args = parser.parse_args()
    old, backlash, out = args.old.resolve(), args.backlash.resolve(), args.output.resolve()
    protocol_path = old / "collection_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    summary_path = backlash / "materialise_summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "complete" or summary.get("rendered") != 168 or summary.get("errors"):
        raise ValueError("corrected backlash dataset is not fully rendered")
    rows = []
    import h5py
    for split, seeds in protocol["splits"].items():
        for seed in seeds:
            for source, defect, level in (("clean", "D2", 1), ("delay", "D2", 1),
                    ("delay", "D2", 2), ("droop", "D3b", 1), ("droop", "D3b", 2),
                    ("backlash", "D4hi", 1), ("backlash", "D4hi", 2)):
                base = backlash if source == "backlash" else old
                cell = base / "data/Stack/Panda" / f"{defect}-L{level}" / f"seed-{seed}"
                result_path = cell / "observations" / f"{source}-result.json"
                row = json.loads(result_path.read_text())
                if row.get("status") != "complete":
                    raise ValueError(f"unfinished episode: {result_path}")
                if (row["source"], row["level"], row["seed_env"], row["split"]) != (source, level, seed, split):
                    raise ValueError(f"episode identity differs: {result_path}")
                if sha256(row["path"]) != row["sha256"] or sha256(row["raw_path"]) != row["raw_sha256"]:
                    raise ValueError(f"dataset changed: {result_path}")
                if source == "backlash":
                    with h5py.File(row["path"]) as h:
                        if h.attrs["course_collection_schema"] != "dl_course.limit_backlash_collection/2":
                            raise ValueError("old backlash recording is excluded")
                        if h.attrs["course_physics_version"] != "dl_course.backlash.unilateral-limit-v1":
                            raise ValueError("wrong backlash physics")
                        if "transmission" not in h["data/demo_0"]:
                            raise ValueError("missing recorded transmission history")
                    receipt = json.loads(Path(row["path"] + ".alignment.json").read_text())
                    if receipt["schema"] != "dl_course.physics_episode_materialisation/2" or not receipt["actual_rendering_models"]:
                        raise ValueError("missing actual rendering model audit")
                row["result_record_sha256"] = sha256(result_path)
                rows.append(row)
    payload = {"schema": "dl_course.three_source_episodes/1", "task": "Stack",
               "collection_protocol_sha256": sha256(protocol_path), "episodes": rows, "errors": [],
               "backlash_protocol_sha256": sha256(backlash / "batch_protocol.json"),
               "backlash_materialise_summary_sha256": sha256(summary_path),
               "normal_source_selection": "one D2-L1 normal reference per scene",
               "all_attempts_retained": True, "old_backlash_excluded": True}
    complete = {"status": "complete", "errors": 0, "expected_episodes": 588, "rendered": len(rows)}
    validate_complete_manifest(payload, protocol, complete)
    out.mkdir(parents=True, exist_ok=True)
    for path, obj in ((out / "episodes.json", payload), (out / "materialise_summary.json", complete)):
        if path.exists() and json.loads(path.read_text()) != obj:
            raise FileExistsError(f"immutable aggregate differs: {path}")
        atomic_json(path, obj)
    from course_tools.three_source_data import EpisodeManifest, fit_statistics, validate_statistics
    manifest = EpisodeManifest.load(out / "episodes.json")
    stats_path = out / "statistics.json"
    if stats_path.exists():
        validate_statistics(json.loads(stats_path.read_text()), manifest)
    else:
        atomic_json(stats_path, fit_statistics(manifest))
    print(json.dumps({"episodes": len(rows), "manifest": str(manifest.path), "statistics": str(stats_path)}))


if __name__ == "__main__":
    main()
