"""Collect all declared backlash scenes using the corrected physical model."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

from course_tools.collect_limit_backlash import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def run_one(job):
    from course_tools.collect_limit_backlash import main as collect
    root, thresholds, seed, split, level, tol, kmax = job
    cell = Path(root) / "data" / "Stack" / "Panda" / f"D4hi-L{level}" / f"seed-{seed}"
    cell.mkdir(parents=True, exist_ok=True)
    info = {"source": "backlash", "defect": "D4hi", "level": level,
            "split": split, "seed_env": seed, "started_at": time.time(), "pid": os.getpid()}
    atomic_json(cell / "progress.json", {**info, "status": "running"})
    try:
        with (cell / "collection.log").open("a") as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            collect(["--level", str(level), "--seed-env", str(seed), "--split", split,
                     "--threshold-report", str(thresholds), "--tol", str(tol),
                     "--k-max", str(kmax), "--output-root", str(Path(root) / "data")])
        records = json.loads((cell / "collection.json").read_text())
        for role in ("defective", "clean"):
            if sha256(records[role]["raw_path"]) != records[role]["raw_sha256"]:
                raise ValueError("collected file checksum mismatch")
        info.update(status="complete", defective=records["defective"], clean=records["clean"])
    except Exception as exc:
        info.update(status="error", error=repr(exc), traceback=traceback.format_exc())
    info["finished_at"] = time.time()
    atomic_json(cell / "job.json", info)
    atomic_json(cell / "progress.json", info)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "course_results/three-source-v2/collection_protocol.json")
    parser.add_argument("--thresholds", type=Path, default=ROOT / "course_results/three-source-v2/regime_thresholds.json")
    parser.add_argument("--output", type=Path, default=ROOT / "course_results/three-source-backlash-v4")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "collection.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = json.loads(args.protocol.read_text())
    specification = {"schema": "dl_course.limit_backlash_batch/1",
                     "source_protocol_sha256": sha256(args.protocol),
                     "thresholds_sha256": sha256(args.thresholds),
                     "splits": protocol["splits"], "levels": [1, 2],
                     "tol": protocol["ilc_tolerance_m"], "k_max": protocol["ilc_k_max"],
                     "code_sha256": {name: sha256(ROOT / "course_tools" / name) for name in (
                         "collect_limit_backlash.py", "limit_backlash_data.py", "backlash_physics.py")}}
    binding = args.output / "batch_protocol.json"
    if binding.exists() and json.loads(binding.read_text()) != specification:
        raise ValueError("batch code or protocol changed")
    atomic_json(binding, specification)
    jobs = [(str(args.output.resolve()), str(args.thresholds.resolve()), seed, split, level,
             specification["tol"], specification["k_max"])
            for split, seeds in protocol["splits"].items() for seed in seeds for level in (1, 2)]
    results, started = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(run_one, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            progress = {"planned_pairs": len(jobs),
                        "completed_pairs": sum(r["status"] == "complete" for r in results),
                        "errors": sum(r["status"] == "error" for r in results),
                        "workers": args.workers, "elapsed_seconds": time.time() - started,
                        "updated_at": time.time()}
            atomic_json(args.output / "collection_progress.json", progress)
            print(json.dumps(progress), flush=True)
    atomic_json(args.output / "collection_summary.json", {
        **progress, "status": "complete" if progress["errors"] == 0 else "completed_with_errors",
        "pairs": results})
    return int(progress["errors"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
