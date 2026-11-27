"""Resumable runtime preflight for the complete P0 cell matrix.

This command proves only that each runtime cell can build, reset, resolve its
reference and execute a few steps through the production OSC action path. It
does not run a 400-step tracker rollout and none of its artifacts is a
calibration result or an input to a Goldilocks decision.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from portal_trace.defects.params import defect as defect_spec
from portal_trace.env.versions import Backend, fingerprint
from portal_trace.lockfile import source_commit
from portal_trace.registry.tasks import CORE_TASKS, HELDOUT_TASKS

__all__ = [
    "P0PreflightCell",
    "enumerate_p0_preflight_cells",
    "runtime_probe",
    "run_preflight",
    "summarize_preflight",
    "main",
]

_SCHEMA_VERSION = 2
_SUMMARY_SCHEMA_VERSION = 2
_PRODUCER = "portal_trace.scripts.preflight_p0"
_PROBE_CONTRACT = "osc_short_step_plumbing_probe/per_level/not_calibration_result/v2"
_DEFAULT_PROBE_HORIZON = 2
_PREFLIGHT_SEED = 0
_UNUSED_TRAIN_SEED = -1

Probe = Callable[["P0PreflightCell", int], Mapping[str, Any] | None]


@dataclass(frozen=True)
class P0PreflightCell:
    """One formal defect-level-family-task runtime cell."""

    ordinal: int
    defect: str
    level: int
    family: str
    task: str

    @property
    def key(self) -> str:
        return f"{self.defect}/L{self.level}/{self.family}/{self.task}"

    @property
    def artifact_dirname(self) -> str:
        safe_task = self.task.replace("/", "_")
        return (
            f"{self.ordinal:04d}-{self.defect}-L{self.level}-"
            f"{self.family}-{safe_task}"
        )


def enumerate_p0_preflight_cells(
    *,
    families: Sequence[str] | None = None,
    tasks: Sequence[str] = (*CORE_TASKS, *HELDOUT_TASKS),
    defects: Sequence[str] | None = None,
) -> tuple[P0PreflightCell, ...]:
    """Enumerate the formal ordered 6 defects x 5 levels x 6 families x 8 tasks."""
    if families is None or defects is None:
        from portal_trace.configs.loader import load_bundle

        formal_subset = load_bundle().subsets["S36"]
        if families is None:
            families = formal_subset.families
        if defects is None:
            defects = formal_subset.defects
    cells: list[P0PreflightCell] = []
    for defect in defects:
        levels = defect_spec(defect).levels
        if not levels:
            raise ValueError(f"{defect}: no level exists for a runtime probe")
        for level in levels:
            for family in families:
                for task in tasks:
                    cells.append(
                        P0PreflightCell(
                            ordinal=len(cells),
                            defect=defect,
                            level=level.index,
                            family=family,
                            task=task,
                        )
                    )
    keys = [cell.key for cell in cells]
    if len(keys) != len(set(keys)):
        raise ValueError("P0 preflight matrix contains duplicate runtime cells")
    return tuple(cells)


def _worktree_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return bool(result.stdout.strip())


def _provenance() -> dict[str, Any]:
    return {
        "source_commit": source_commit(),
        "worktree_dirty": _worktree_dirty(),
        "environment": fingerprint(Backend.CPU_MUJOCO).to_json(),
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _encoded(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _write_json_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish one complete attempt atomically without replacing prior evidence."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite preflight artifact {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(_encoded(payload))
            fh.flush()
            os.fsync(fh.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite preflight artifact {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_replace(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically refresh the derived summary; immutable attempts remain the evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(_encoded(payload))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _attempt_paths(root: Path, cell: P0PreflightCell) -> list[Path]:
    return sorted((root / "cells" / cell.artifact_dirname).glob("attempt-*.json"))


def _artifact_matches(
    payload: Mapping[str, Any],
    cell: P0PreflightCell,
    probe_horizon: int,
    provenance: Mapping[str, Any],
) -> bool:
    return (
        payload.get("schema_version") == _SCHEMA_VERSION
        and payload.get("producer") == _PRODUCER
        and payload.get("artifact_kind") == "runtime_preflight_not_calibration"
        and payload.get("cell") == asdict(cell)
        and payload.get("probe_horizon") == probe_horizon
        and payload.get("source_commit") == provenance["source_commit"]
        and payload.get("worktree_dirty") == provenance["worktree_dirty"]
        and payload.get("environment") == provenance["environment"]
    )


def _matching_attempts(
    root: Path,
    cell: P0PreflightCell,
    probe_horizon: int,
    provenance: Mapping[str, Any],
) -> list[tuple[Path, dict[str, Any]]]:
    matches = []
    for path in _attempt_paths(root, cell):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if _artifact_matches(payload, cell, probe_horizon, provenance):
            matches.append((path, payload))
    return matches


def _publish_attempt(root: Path, cell: P0PreflightCell, payload: Mapping[str, Any]) -> Path:
    directory = root / "cells" / cell.artifact_dirname
    for attempt in range(1_000_000):
        path = directory / f"attempt-{attempt:06d}.json"
        try:
            _write_json_immutable(path, payload)
        except FileExistsError:
            continue
        return path
    raise RuntimeError(f"{cell.key}: exhausted preflight attempt identifiers")


def _probe_side(env: Any, reference: Any, tracker: Any, horizon: int) -> dict[str, Any]:
    from portal_trace.teacher import osc

    site_id = tracker.grip_site_id(env)
    base_pos, base_rot = tracker.base_frame(env)
    max_abs_raw = 0.0
    for step in range(horizon):
        position, rotation = tracker.tool_pose(env, site_id)
        delta = tracker.delta_action(
            reference.positions[step],
            reference.rotations[step],
            position,
            rotation,
            base_pos,
            base_rot,
        )
        raw = tracker.compose_action(delta, reference.gripper[step : step + 1])[0]
        if not np.all(np.isfinite(raw)):
            raise FloatingPointError(f"OSC produced a non-finite action at step {step}")
        max_abs_raw = max(max_abs_raw, float(np.max(np.abs(raw))))
        env.step(osc.env_action(env, tracker.clip_action(raw)))
        state = osc.physics_state(env.sim)
        if not np.all(np.isfinite(state)):
            raise FloatingPointError(f"MuJoCo state became non-finite at step {step}")
    return {
        "steps": horizon,
        "state_width": int(osc.physics_state(env.sim).size),
        "max_abs_raw_action": max_abs_raw,
        "controller_config_sha256": tracker.gains.sha256(),
    }


def runtime_probe(cell: P0PreflightCell, probe_horizon: int) -> Mapping[str, Any]:
    """Exercise real build/reset/reference/OSC plumbing without producing a rollout."""
    from portal_trace.defects.delta import Delta, default_joint_scale
    from portal_trace.defects.inject import build_twin_pair
    from portal_trace.registry.families import descriptor
    from portal_trace.registry.tasks import task as task_spec
    from portal_trace.teacher import osc, refs
    from portal_trace.types import SeedTriple

    n_dof = descriptor(cell.family).dof
    delta = Delta.from_level(
        cell.defect,
        cell.level,
        joint_scale=default_joint_scale(cell.defect, n_dof),
    )
    seeds = SeedTriple(env=_PREFLIGHT_SEED, train=_UNUSED_TRAIN_SEED, eval=_PREFLIGHT_SEED)
    pair = build_twin_pair(task_spec(cell.task), cell.family, delta, seeds)
    try:
        osc.reset_twin_pair(pair, _PREFLIGHT_SEED)
        reference = refs.resolve(refs.load_spec(cell.task), pair.defective.sim)
        if not 1 <= probe_horizon <= reference.horizon:
            raise ValueError(
                f"probe_horizon must be in [1, {reference.horizon}], got {probe_horizon}"
            )
        tracker = osc.OSCTracker()
        sides = {
            "defective": _probe_side(pair.defective, reference, tracker, probe_horizon),
            "clean": _probe_side(pair.clean, reference, tracker, probe_horizon),
        }
        return {
            "probe_contract": _PROBE_CONTRACT,
            "scientific_use": "forbidden",
            "seed_env": _PREFLIGHT_SEED,
            "reference_horizon": int(reference.horizon),
            "sides": sides,
        }
    finally:
        pair.defective.close()
        pair.clean.close()


def summarize_preflight(
    output_root: Path,
    *,
    cells: Sequence[P0PreflightCell] | None = None,
    probe_horizon: int = _DEFAULT_PROBE_HORIZON,
    provenance: Mapping[str, Any] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """List every expected cell and report ready only when every current cell passed."""
    matrix = tuple(enumerate_p0_preflight_cells() if cells is None else cells)
    run_provenance = dict(provenance or _provenance())
    rows = []
    for cell in matrix:
        matches = _matching_attempts(output_root, cell, probe_horizon, run_provenance)
        successful = [(path, item) for path, item in matches if item.get("status") == "passed"]
        chosen = successful[-1] if successful else (matches[-1] if matches else None)
        if chosen is None:
            status, artifact, error = "missing", None, None
        else:
            path, item = chosen
            status = str(item.get("status", "invalid"))
            artifact = str(path)
            error = item.get("error")
        rows.append(
            {
                "ordinal": cell.ordinal,
                "family": cell.family,
                "task": cell.task,
                "defect": cell.defect,
                "level": cell.level,
                "status": status,
                "artifact": artifact,
                "error": error,
            }
        )
    passed = sum(row["status"] == "passed" for row in rows)
    summary = {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "producer": _PRODUCER,
        "artifact_kind": "runtime_preflight_summary_not_calibration",
        **run_provenance,
        "probe_contract": _PROBE_CONTRACT,
        "probe_level_role": "formal_matrix_level_runtime_only_not_decision_evidence",
        "probe_horizon": probe_horizon,
        "expected_cells": len(matrix),
        "passed_cells": passed,
        "failed_cells": sum(row["status"] == "failed" for row in rows),
        "missing_cells": sum(row["status"] == "missing" for row in rows),
        "ready": passed == len(matrix),
        "cells": rows,
        "updated_at": _utc_now(),
    }
    if write:
        _write_json_replace(output_root / "summary.json", summary)
    return summary


def run_preflight(
    output_root: Path,
    *,
    probe: Probe = runtime_probe,
    cells: Sequence[P0PreflightCell] | None = None,
    selected_keys: frozenset[str] | None = None,
    probe_horizon: int = _DEFAULT_PROBE_HORIZON,
    provenance: Mapping[str, Any] | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Run selected cells, resume current successes, then summarize the whole matrix.

    ``workers`` controls concurrent runtime probes. Artifact publication stays in
    this process and remains ordered/immutable, so parallel probes cannot race
    on summary replacement or overwrite an attempt. The default of one worker
    preserves the original serial behavior for constrained environments.
    """
    if probe_horizon <= 0:
        raise ValueError(f"probe_horizon must be positive, got {probe_horizon}")
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    matrix = tuple(enumerate_p0_preflight_cells() if cells is None else cells)
    run_provenance = dict(provenance or _provenance())
    selected = matrix if selected_keys is None else tuple(
        cell for cell in matrix if cell.key in selected_keys
    )
    if selected_keys is not None:
        unknown = sorted(selected_keys - {cell.key for cell in matrix})
        if unknown:
            raise ValueError(f"selected preflight cells are not in the matrix: {unknown}")

    pending: list[P0PreflightCell] = []
    for cell in selected:
        matches = _matching_attempts(output_root, cell, probe_horizon, run_provenance)
        if any(item.get("status") == "passed" for _, item in matches):
            print(
                json.dumps(
                    {"event": "resumed", "cell": cell.key, "level": cell.level},
                    sort_keys=True,
                ),
                flush=True,
            )
            continue
        pending.append(cell)

    def execute(
        cell: P0PreflightCell,
    ) -> tuple[P0PreflightCell, str, dict[str, Any], dict[str, Any] | None, float, str]:
        started_at = _utc_now()
        clock = time.perf_counter()
        print(
            json.dumps(
                {"event": "started", "cell": cell.key, "level": cell.level},
                sort_keys=True,
            ),
            flush=True,
        )
        try:
            observations = dict(probe(cell, probe_horizon) or {})
            status = "passed"
            error = None
        except Exception as exc:  # each cell must leave durable failure evidence
            observations = {}
            status = "failed"
            error = {"type": type(exc).__name__, "message": str(exc)}
        return cell, started_at, observations, error, time.perf_counter() - clock, status

    if workers == 1:
        completed = (execute(cell) for cell in pending)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = [executor.submit(execute, cell) for cell in pending]
        completed = (future.result() for future in futures)
    try:
        for cell, started_at, observations, error, elapsed, status in completed:
            payload = {
                "schema_version": _SCHEMA_VERSION,
                "producer": _PRODUCER,
                "artifact_kind": "runtime_preflight_not_calibration",
                **run_provenance,
                "probe_contract": _PROBE_CONTRACT,
                "probe_level_role": "formal_matrix_level_runtime_only_not_decision_evidence",
                "probe_horizon": probe_horizon,
                "cell": asdict(cell),
                "status": status,
                "started_at": started_at,
                "finished_at": _utc_now(),
                "elapsed_seconds": round(elapsed, 6),
                "observations": observations,
                "error": error,
            }
            artifact = _publish_attempt(output_root, cell, payload)
            print(
                json.dumps(
                    {
                        "event": status,
                        "cell": cell.key,
                        "elapsed_seconds": payload["elapsed_seconds"],
                        "artifact": str(artifact),
                        "error": error,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        if workers > 1:
            executor.shutdown(wait=True)
    return summarize_preflight(
        output_root,
        cells=matrix,
        probe_horizon=probe_horizon,
        provenance=run_provenance,
    )


def _selected_keys(args: argparse.Namespace, cells: Sequence[P0PreflightCell]) -> frozenset[str]:
    filters_present = bool(args.family or args.task or args.defect or args.level)
    if args.full and filters_present:
        raise ValueError(
            "--full cannot be combined with --family/--task/--defect/--level filters"
        )
    if not args.full and not filters_present:
        raise ValueError("run requires --full or at least one cell filter")
    selected = {
        cell.key
        for cell in cells
        if (not args.family or cell.family in args.family)
        and (not args.task or cell.task in args.task)
        and (not args.defect or cell.defect in args.defect)
        and (not args.level or cell.level in args.level)
    }
    if not selected:
        raise ValueError("preflight filters selected no cells")
    return frozenset(selected)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/preflight/p0"))
    parser.add_argument("--probe-horizon", type=int, default=_DEFAULT_PROBE_HORIZON)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--full", action="store_true")
    run.add_argument("--family", action="append")
    run.add_argument("--task", action="append")
    run.add_argument("--defect", action="append")
    run.add_argument("--level", action="append", type=int)
    run.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent runtime probes; artifact publication remains ordered",
    )
    sub.add_parser("summary")
    args = parser.parse_args(argv)

    cells = enumerate_p0_preflight_cells()
    if args.command == "summary":
        summary = summarize_preflight(
            args.output, cells=cells, probe_horizon=args.probe_horizon
        )
    else:
        try:
            selected = _selected_keys(args, cells)
        except ValueError as exc:
            parser.error(str(exc))
        summary = run_preflight(
            args.output,
            cells=cells,
            selected_keys=selected,
            probe_horizon=args.probe_horizon,
            workers=args.workers,
        )
    print(
        json.dumps(
            {
                "expected_cells": summary["expected_cells"],
                "passed_cells": summary["passed_cells"],
                "failed_cells": summary["failed_cells"],
                "missing_cells": summary["missing_cells"],
                "ready": summary["ready"],
                "summary": str(args.output / "summary.json"),
            },
            sort_keys=True,
        )
    )
    return 0 if summary["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
