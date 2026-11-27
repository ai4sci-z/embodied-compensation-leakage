"""Collect and assemble P0's Goldilocks success-rate matrix.

``plan`` freezes the complete sampling scope before a formal run. ``collect``
runs the real uncompensated OSC tracker on both members of one twin cell, and
binds formal shards to that protocol by SHA-256. ``assemble`` combines immutable
JSON artifacts into the first P0 table. A partial assembly is a pipeline check
only; ``--formal`` requires the complete preregistered matrix, one explicit
protocol, clean single-commit inputs, and a clean checkout. This command reports
the measured matrix but does not choose or freeze defect decision levels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from portal_trace.configs.loader import ConfigBundle, load_bundle
from portal_trace.defects.params import defect as defect_spec
from portal_trace.env.versions import Backend, fingerprint
from portal_trace.eval.calibration_tables import (
    CLEAN_TWIN_CEILING_PCT,
    GOLDILOCKS_MIN_PASSING_DEFECTS,
    GOLDILOCKS_ROLLOUTS_PER_CELL,
    GOLDILOCKS_WINDOW_PCT,
    CalibrationTable,
    GoldilocksCell,
    build_goldilocks_table,
)
from portal_trace.lockfile import NO_CHECKOUT, source_commit
from portal_trace.registry.tasks import (
    CORE_TASKS,
    HELDOUT_TASKS,
    HORIZON,
    TASKS,
    TaskRole,
)
from portal_trace.registry.tasks import task as task_spec
from portal_trace.types import SeedTriple

__all__ = [
    "build_goldilocks_protocol",
    "write_goldilocks_protocol",
    "build_goldilocks_job_manifest",
    "write_goldilocks_job_manifest",
    "scan_goldilocks_jobs",
    "run_goldilocks_job",
    "assemble_goldilocks_manifest",
    "build_decision_level_result",
    "write_decision_level_result",
    "load_decision_level_result",
    "collect_goldilocks_artifact",
    "assemble_goldilocks",
    "load_goldilocks_artifact",
    "main",
]

_ARTIFACT_SCHEMAS = (1, 2)
_CURRENT_ARTIFACT_SCHEMA = 2
_PROTOCOL_SCHEMA = 2
_DECISION_RULE_SCHEMA = 2
_JOB_MANIFEST_SCHEMA = 1
_ROLLOUT_PART_SCHEMA = 2
_EVENT_SCHEMA = 1
_UNUSED_TRAIN_SEED = -1
_COLLECT_PRODUCER = "portal_trace.scripts.calibrate_p0 collect-goldilocks"
_ROLLOUT_PART_PRODUCER = f"{_COLLECT_PRODUCER} rollout-part"
_TRACKER_CONTRACT = "OSCTracker, feedforward=None (ILC k=0)"
_PAIRED_RESET_CONTRACT = "reset_twin_pair/clean-settled-common-frame-0/v1"
_DEFAULT_ROLLOUTS_PER_JOB = 10
_JOB_MANIFEST_PRODUCER = "portal_trace.scripts.calibrate_p0 plan-jobs"
_EVENT_PRODUCER = f"{_COLLECT_PRODUCER} event"
_PREFLIGHT_SUMMARY_SCHEMA = 2
_PREFLIGHT_PRODUCER = "portal_trace.scripts.preflight_p0"
_PREFLIGHT_ARTIFACT_KIND = "runtime_preflight_summary_not_calibration"
_DECISION_LEVEL_SCHEMA = "portal_trace.calibration.decision_levels/1"
_DECISION_LEVEL_PRODUCER = "portal_trace.scripts.calibrate_p0 decide-levels"


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


def _write_json_immutable(path: Path, payload: dict[str, Any]) -> None:
    """Publish one complete JSON file atomically, without replacing any path."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable P0 artifact {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(encoded)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            # A same-filesystem hard link is an atomic no-replace publication.
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite immutable P0 artifact {path}") from None
        published = True
    finally:
        temporary.unlink(missing_ok=True)
        if published:
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)


def _rollout_part_path(output: Path, demo_index: int) -> Path:
    return output.with_name(f"{output.name}.parts") / f"{demo_index:06d}.json"


def _event_dir(output: Path) -> Path:
    return output.with_name(f"{output.name}.events")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _validate_environment(path: Path, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: environment fingerprint must be a JSON object")
    required = {"backend", "versions", "python", "platform", "lite_physics"}
    if missing := sorted(required - set(value)):
        raise ValueError(f"{path}: environment fingerprint is missing fields {missing}")
    if value["backend"] != Backend.CPU_MUJOCO.value:
        raise ValueError(
            f"{path}: P0 environment backend {value['backend']!r}, "
            f"expected {Backend.CPU_MUJOCO.value!r}"
        )
    if not isinstance(value["versions"], dict):
        raise ValueError(f"{path}: environment versions must be a JSON object")
    if type(value["lite_physics"]) is not bool:
        raise ValueError(f"{path}: environment lite_physics must be a JSON boolean")
    for name in ("python", "platform"):
        if not isinstance(value[name], str) or not value[name]:
            raise ValueError(f"{path}: environment {name} must be a non-empty string")
    return value


def _validate_controller_hashes(path: Path, value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"defective", "clean"}:
        raise ValueError(f"{path}: controller_config_sha256 must contain defective and clean")
    hashes = {side: value[side] for side in ("defective", "clean")}
    if any(
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest.lower())
        for digest in hashes.values()
    ):
        raise ValueError(f"{path}: controller_config_sha256 values must be 64-character strings")
    if hashes["defective"] != hashes["clean"]:
        raise ValueError(f"{path}: defective and clean trackers use different controller configs")
    return hashes


def _provenance_signature(payload: dict[str, Any]) -> str:
    return json.dumps(
        {
            "environment": payload["environment"],
            "controller_config_sha256": payload["controller_config_sha256"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _event_scope(demo_index: int | None) -> str:
    return "job" if demo_index is None else f"rollout-{demo_index:06d}"


def _emit_progress(event: dict[str, Any]) -> None:
    try:
        print(json.dumps(event, sort_keys=True), flush=True)
    except OSError:
        # The immutable event remains authoritative if a detached stdout closes.
        pass


def _start_event(
    output: Path,
    *,
    identity: dict[str, Any],
    job_id: str | None,
    cell_ordinal: int | None,
    demo_index: int | None,
    seed_env: int | None,
) -> dict[str, Any]:
    start = _utc_now()
    base = {
        "schema_version": _EVENT_SCHEMA,
        "producer": _EVENT_PRODUCER,
        "job_id": job_id or "ad-hoc",
        "cell_ordinal": cell_ordinal,
        "task": identity["task"],
        "family": identity["family"],
        "defect": identity["defect"],
        "level": identity["level"],
        "demo_index": demo_index,
        "seed_env": seed_env,
        "start": start,
        "end": None,
        "elapsed_seconds": None,
        "result": "running",
    }
    scope = _event_scope(demo_index)
    for attempt in range(1_000_000):
        event = {**base, "event": "start", "attempt": attempt}
        path = _event_dir(output) / f"{scope}-a{attempt:06d}-start.json"
        try:
            _write_json_immutable(path, event)
        except FileExistsError:
            continue
        _emit_progress(event)
        return {
            "output": output,
            "scope": scope,
            "attempt": attempt,
            "start": start,
            "clock": time.perf_counter(),
            "base": base,
        }
    raise RuntimeError(f"{output}: exhausted event attempt identifiers for {scope}")


def _finish_event(
    handle: dict[str, Any],
    *,
    result: str,
    outcome: dict[str, Any] | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    if result not in {"complete", "failed"}:
        raise ValueError(f"unsupported P0 event result {result!r}")
    event = {
        **handle["base"],
        "event": result,
        "attempt": handle["attempt"],
        "end": _utc_now(),
        "elapsed_seconds": round(time.perf_counter() - handle["clock"], 6),
        "result": result,
    }
    if outcome is not None:
        event["outcome"] = outcome
    if error is not None:
        event["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    path = (
        _event_dir(handle["output"]) / f"{handle['scope']}-a{handle['attempt']:06d}-{result}.json"
    )
    _write_json_immutable(path, event)
    _emit_progress(event)
    return event


def _record_failure_event(
    output: Path,
    *,
    identity: dict[str, Any],
    job_id: str | None,
    cell_ordinal: int | None,
    error: BaseException,
) -> None:
    handle = _start_event(
        output,
        identity=identity,
        job_id=job_id,
        cell_ordinal=cell_ordinal,
        demo_index=None,
        seed_env=None,
    )
    _finish_event(handle, result="failed", error=error)


def _validate_outcome(path: Path, row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"{path}: outcome must be a JSON object")
    for name in ("defective_success", "clean_success"):
        if type(row.get(name)) is not bool:
            raise ValueError(f"{path}: outcome {name} must be a JSON boolean")
    for name in ("demo_index", "seed_env", "defective_t_success", "clean_t_success"):
        if type(row.get(name)) is not int:
            raise ValueError(f"{path}: outcome {name} must be a JSON integer")
    for side in ("defective", "clean"):
        success = row[f"{side}_success"]
        t_success = row[f"{side}_t_success"]
        if success and not 0 <= t_success < HORIZON:
            raise ValueError(
                f"{path}: successful {side} outcome has t_success={t_success}, "
                f"expected 0..{HORIZON - 1}"
            )
        if not success and t_success != -1:
            raise ValueError(
                f"{path}: unsuccessful {side} outcome has t_success={t_success}, expected -1"
            )
    return row


def _shard_identity(
    *,
    task: str,
    family: str,
    defect: str,
    level: int,
    start_index: int,
    n_rollouts: int,
    protocol_sha256: str | None,
    run_source_commit: str,
    run_worktree_dirty: bool | None,
    environment: dict[str, Any] | None = None,
    decision_rule_sha256: str | None = None,
    preflight_summary_sha256: str | None = None,
) -> dict[str, Any]:
    identity = {
        "producer": _COLLECT_PRODUCER,
        "source_commit": run_source_commit,
        "worktree_dirty": run_worktree_dirty,
        "task": task,
        "family": family,
        "defect": defect,
        "level": level,
        "tracker": _TRACKER_CONTRACT,
        "paired_reset_contract": _PAIRED_RESET_CONTRACT,
        "start_index": start_index,
        "n_rollouts": n_rollouts,
        "protocol_sha256": protocol_sha256,
        "decision_rule_sha256": decision_rule_sha256,
        "preflight_summary_sha256": preflight_summary_sha256,
    }
    if environment is not None:
        identity["environment"] = environment
    return identity


def _load_rollout_part(
    path: Path,
    *,
    identity: dict[str, Any],
    demo_index: int,
    seed_env: int,
) -> dict[str, Any]:
    part = json.loads(path.read_text(encoding="utf-8"))
    if part.get("schema_version") != _ROLLOUT_PART_SCHEMA:
        raise ValueError(
            f"{path}: rollout-part schema {part.get('schema_version')}, "
            f"expected {_ROLLOUT_PART_SCHEMA}"
        )
    if part.get("producer") != _ROLLOUT_PART_PRODUCER:
        raise ValueError(f"{path}: is not a P0 Goldilocks rollout-part")
    for name, expected in identity.items():
        if name == "producer":
            continue
        if part.get(name) != expected:
            raise ValueError(
                f"{path}: rollout-part {name}={part.get(name)!r} does not match "
                f"the requested shard value {expected!r}"
            )
    _validate_environment(path, part.get("environment"))
    _validate_controller_hashes(path, part.get("controller_config_sha256"))
    outcome = _validate_outcome(path, part.get("outcome"))
    if outcome["demo_index"] != demo_index or outcome["seed_env"] != seed_env:
        raise ValueError(
            f"{path}: rollout-part identifies demo/seed "
            f"{outcome['demo_index']}/{outcome['seed_env']}, expected {demo_index}/{seed_env}"
        )
    return part


def _validate_existing_shard(
    path: Path,
    *,
    identity: dict[str, Any],
    bundle: ConfigBundle,
) -> dict[str, Any]:
    artifact = load_goldilocks_artifact(path)
    if artifact.get("schema_version") != _CURRENT_ARTIFACT_SCHEMA:
        raise ValueError(
            f"{path}: existing shard uses schema {artifact.get('schema_version')}; "
            f"idempotent reuse requires schema {_CURRENT_ARTIFACT_SCHEMA}"
        )
    mismatches = {
        name: (artifact.get(name), expected)
        for name, expected in identity.items()
        if artifact.get(name) != expected
    }
    if mismatches:
        raise ValueError(
            f"{path}: existing immutable shard does not match this request: {mismatches}"
        )
    for row in artifact["outcomes"]:
        expected_seed = bundle.seeds.env_seed(
            task=identity["task"],
            family=identity["family"],
            demo_index=int(row["demo_index"]),
        )
        if row["seed_env"] != expected_seed:
            raise ValueError(
                f"{path}: existing demo {row['demo_index']} uses seed {row['seed_env']}, "
                f"expected {expected_seed}"
            )
    return artifact


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_exact_fields(
    path: Path, section: str, value: Any, expected: set[str]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: decision rule {section} must be a JSON object")
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(
            f"{path}: decision rule {section} fields mismatch; missing={missing}, extra={extra}"
        )
    return value


def _amendment_binding(
    bundle: ConfigBundle, index: int, *, require_before_unblinding: bool = True
) -> dict[str, Any]:
    matches = [entry for entry in bundle.amendments.entries if entry.index == index]
    if not matches:
        raise ValueError(f"decision rule cites unknown amendment index {index}")
    amendment = matches[0]
    if require_before_unblinding and not amendment.before_unblinding:
        raise ValueError(f"decision rule amendment {index} was not recorded before unblinding")
    if amendment.item_number != 2:
        raise ValueError(
            f"decision rule amendment {index} belongs to preregistration item "
            f"{amendment.item_number}, expected item 2"
        )
    return {
        "index": amendment.index,
        "sha256": _canonical_json_sha256(amendment.to_json()),
    }


def _decision_level_indices(bundle: ConfigBundle) -> list[int]:
    defects = bundle.subsets["S36"].defects
    levels_by_defect = [[level.index for level in defect_spec(code).levels] for code in defects]
    if not levels_by_defect or any(levels != levels_by_defect[0] for levels in levels_by_defect):
        raise ValueError("S36 defects do not share one ordered decision-level index set")
    return levels_by_defect[0]


def _validate_decision_rule(
    rule: Any, path: Path, bundle: ConfigBundle | None = None
) -> dict[str, Any]:
    resolved_bundle = bundle or load_bundle()
    top = _require_exact_fields(
        path,
        "document",
        rule,
        {
            "schema_version",
            "selection_unit",
            "ordered_core_tasks",
            "ordered_heldout_tasks",
            "aggregation_thresholds",
            "rollout_success",
            "tie_break",
            "defect_admission",
            "before_unblinding_amendment",
        },
    )
    if top["schema_version"] != _DECISION_RULE_SCHEMA:
        raise ValueError(
            f"{path}: decision-rule schema {top['schema_version']}, "
            f"expected {_DECISION_RULE_SCHEMA}"
        )
    if top["selection_unit"] != ["family", "defect"]:
        raise ValueError(f"{path}: selection_unit must be ['family', 'defect']")
    if top["ordered_core_tasks"] != list(CORE_TASKS):
        raise ValueError(f"{path}: ordered_core_tasks must exactly match the task registry")
    if top["ordered_heldout_tasks"] != list(HELDOUT_TASKS):
        raise ValueError(f"{path}: ordered_heldout_tasks must exactly match the task registry")

    aggregation = _require_exact_fields(
        path,
        "aggregation_thresholds",
        top["aggregation_thresholds"],
        {
            "cell_success_rate_pct",
            "minimum_clean_twin_rate_pct",
            "minimum_core_tasks_passing",
            "minimum_heldout_tasks_passing",
        },
    )
    if aggregation["cell_success_rate_pct"] != list(GOLDILOCKS_WINDOW_PCT):
        raise ValueError(
            f"{path}: cell_success_rate_pct must preserve {list(GOLDILOCKS_WINDOW_PCT)}"
        )
    if aggregation["minimum_clean_twin_rate_pct"] != CLEAN_TWIN_CEILING_PCT:
        raise ValueError(
            f"{path}: minimum_clean_twin_rate_pct must preserve {CLEAN_TWIN_CEILING_PCT:g}"
        )
    core_min = aggregation["minimum_core_tasks_passing"]
    heldout_min = aggregation["minimum_heldout_tasks_passing"]
    if type(core_min) is not int or not 0 <= core_min <= len(CORE_TASKS):
        raise ValueError(f"{path}: minimum_core_tasks_passing must be in 0..{len(CORE_TASKS)}")
    if type(heldout_min) is not int or not 0 <= heldout_min <= len(HELDOUT_TASKS):
        raise ValueError(
            f"{path}: minimum_heldout_tasks_passing must be in 0..{len(HELDOUT_TASKS)}"
        )
    if core_min + heldout_min == 0:
        raise ValueError(f"{path}: aggregation must require at least one passing task")

    rollout_success = _require_exact_fields(
        path,
        "rollout_success",
        top["rollout_success"],
        {
            "predicate",
            "horizon_steps",
            "terminal_hold_steps",
            "terminal_hold_seconds",
            "amendment",
        },
    )
    from portal_trace.success import SUCCESS_HOLD_SECONDS, SUCCESS_HOLD_STEPS

    expected_success = {
        "predicate": "environment_task_success",
        "horizon_steps": HORIZON,
        "terminal_hold_steps": SUCCESS_HOLD_STEPS,
        "terminal_hold_seconds": SUCCESS_HOLD_SECONDS,
    }
    for name, expected in expected_success.items():
        if rollout_success[name] != expected:
            raise ValueError(
                f"{path}: rollout_success.{name} must preserve {expected!r}"
            )
    success_amendment = _require_exact_fields(
        path, "rollout_success.amendment", rollout_success["amendment"], {"index", "sha256"}
    )
    if success_amendment != _amendment_binding(
        resolved_bundle,
        success_amendment["index"],
        require_before_unblinding=False,
    ):
        raise ValueError(f"{path}: rollout-success amendment index/hash does not match the log")

    tie_break = _require_exact_fields(path, "tie_break", top["tie_break"], {"ordered_levels"})
    expected_levels = _decision_level_indices(resolved_bundle)
    ordered_levels = tie_break["ordered_levels"]
    if (
        not isinstance(ordered_levels, list)
        or any(type(level) is not int for level in ordered_levels)
        or sorted(ordered_levels) != expected_levels
    ):
        raise ValueError(
            f"{path}: tie_break.ordered_levels must be a permutation of {expected_levels}"
        )

    admission = _require_exact_fields(
        path,
        "defect_admission",
        top["defect_admission"],
        {"minimum_families_with_selected_level", "minimum_admitted_defects"},
    )
    family_min = admission["minimum_families_with_selected_level"]
    n_families = len(resolved_bundle.subsets["S36"].families)
    if type(family_min) is not int or not 1 <= family_min <= n_families:
        raise ValueError(f"{path}: minimum_families_with_selected_level must be in 1..{n_families}")
    if admission["minimum_admitted_defects"] != GOLDILOCKS_MIN_PASSING_DEFECTS:
        raise ValueError(
            f"{path}: minimum_admitted_defects must preserve {GOLDILOCKS_MIN_PASSING_DEFECTS}"
        )

    amendment = _require_exact_fields(
        path,
        "before_unblinding_amendment",
        top["before_unblinding_amendment"],
        {"index", "sha256"},
    )
    if type(amendment["index"]) is not int or amendment["index"] < 1:
        raise ValueError(f"{path}: amendment index must be a positive integer")
    expected_binding = _amendment_binding(resolved_bundle, amendment["index"])
    if amendment != expected_binding:
        raise ValueError(f"{path}: before_unblinding amendment index/hash does not match the log")
    _canonical_json_sha256(top)
    return top


def _load_decision_rule(path: Path, bundle: ConfigBundle | None = None) -> dict[str, Any]:
    return _validate_decision_rule(json.loads(path.read_text(encoding="utf-8")), path, bundle)


def _default_results_root() -> Path:
    return Path(os.environ.get("PORTAL_RESULTS_ROOT", "results"))


def _ordered_formal_cells(bundle: ConfigBundle) -> list[dict[str, Any]]:
    subset = bundle.subsets["S36"]
    tasks = tuple(spec.task_id for spec in TASKS.values() if spec.role is not TaskRole.BASE)
    return [
        {"defect": code, "level": level.index, "family": family, "task": task}
        for code in subset.defects
        for level in defect_spec(code).levels
        for family in subset.families
        for task in tasks
    ]


def _load_preflight_summary(
    path: Path,
    *,
    expected_cells: Sequence[dict[str, Any]],
    expected_source_commit: str | None = None,
) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != _PREFLIGHT_SUMMARY_SCHEMA:
        raise ValueError(
            f"{path}: preflight summary schema {summary.get('schema_version')}; "
            f"formal P0 requires schema {_PREFLIGHT_SUMMARY_SCHEMA} with per-level coverage"
        )
    if summary.get("producer") != _PREFLIGHT_PRODUCER:
        raise ValueError(f"{path}: is not a P0 runtime preflight summary")
    if summary.get("artifact_kind") != _PREFLIGHT_ARTIFACT_KIND:
        raise ValueError(f"{path}: preflight summary has the wrong artifact kind")
    required = {
        "source_commit",
        "worktree_dirty",
        "environment",
        "expected_cells",
        "passed_cells",
        "failed_cells",
        "missing_cells",
        "ready",
        "cells",
    }
    if missing := sorted(required - set(summary)):
        raise ValueError(f"{path}: preflight summary is missing fields {missing}")
    if summary["source_commit"] == NO_CHECKOUT or summary["worktree_dirty"] is not False:
        raise ValueError(f"{path}: formal P0 preflight must come from a clean Git commit")
    if expected_source_commit is not None and summary["source_commit"] != expected_source_commit:
        raise ValueError(
            f"{path}: preflight source_commit {summary['source_commit']!r} does not "
            f"match {expected_source_commit!r}"
        )
    environment = _validate_environment(path, summary["environment"])
    rows = summary["cells"]
    if not isinstance(rows, list):
        raise ValueError(f"{path}: preflight cells must be a JSON array")
    n_expected = len(expected_cells)
    counts = {
        "expected_cells": n_expected,
        "passed_cells": n_expected,
        "failed_cells": 0,
        "missing_cells": 0,
    }
    if any(
        type(summary.get(name)) is not int or summary[name] != value
        for name, value in counts.items()
    ):
        raise ValueError(
            f"{path}: preflight counts must prove all {n_expected} ordered cells passed"
        )
    if summary["ready"] is not True:
        raise ValueError(f"{path}: preflight summary is not ready")
    if len(rows) != n_expected:
        raise ValueError(
            f"{path}: preflight has {len(rows)} cells; formal protocol requires "
            f"all {n_expected} defect-level-family-task cells"
        )
    normalized: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: preflight cell {ordinal} must be an object")
        if row.get("ordinal") != ordinal or row.get("status") != "passed":
            raise ValueError(
                f"{path}: preflight cell {ordinal} must have matching ordinal and passed status"
            )
        if type(row.get("level")) is not int:
            raise ValueError(f"{path}: preflight cell {ordinal} lacks explicit per-level evidence")
        normalized.append(
            {
                "defect": str(row.get("defect")),
                "level": row["level"],
                "family": str(row.get("family")),
                "task": str(row.get("task")),
            }
        )
    if normalized != list(expected_cells):
        raise ValueError(
            f"{path}: preflight cells do not equal the protocol's ordered 1440-cell matrix"
        )
    return {**summary, "environment": environment, "cells": rows}


def build_goldilocks_protocol(
    decision_rule_path: Path,
    preflight_summary_path: Path,
    bundle: ConfigBundle | None = None,
) -> dict[str, Any]:
    """The complete P0 sampling scope, fixed before any formal shard runs."""
    resolved_bundle = bundle or load_bundle()
    canonical_rule_path = Path(decision_rule_path).resolve()
    canonical_preflight_path = Path(preflight_summary_path).resolve()
    rule = _load_decision_rule(canonical_rule_path, resolved_bundle)
    cells = _ordered_formal_cells(resolved_bundle)
    run_source_commit = source_commit()
    preflight = _load_preflight_summary(
        canonical_preflight_path,
        expected_cells=cells,
        expected_source_commit=run_source_commit,
    )
    return {
        "schema_version": _PROTOCOL_SCHEMA,
        "producer": "portal_trace.scripts.calibrate_p0 plan-goldilocks",
        "source_commit": run_source_commit,
        "worktree_dirty": _worktree_dirty(),
        "start_index": 0,
        "n_rollouts_per_cell": GOLDILOCKS_ROLLOUTS_PER_CELL,
        "decision_rule": {
            "path": str(canonical_rule_path),
            "canonical_sha256": _canonical_json_sha256(rule),
            "before_unblinding_amendment": rule["before_unblinding_amendment"],
            "document": rule,
        },
        "preflight_summary": {
            "path": str(canonical_preflight_path),
            "sha256": _sha256(canonical_preflight_path),
            "environment": preflight["environment"],
        },
        "cells": cells,
    }


def write_goldilocks_protocol(
    output: Path,
    decision_rule_path: Path,
    preflight_summary_path: Path,
    bundle: ConfigBundle | None = None,
) -> dict[str, Any]:
    """Persist a clean-checkout protocol with an immutable preflight snapshot."""
    protocol = build_goldilocks_protocol(decision_rule_path, preflight_summary_path, bundle)
    if protocol["source_commit"] == NO_CHECKOUT or protocol["worktree_dirty"] is not False:
        raise ValueError(
            "formal P0 protocol creation requires a clean Git checkout; a dirty or "
            "non-Git protocol cannot establish the provenance of later shards"
        )
    canonical_output = Path(output).resolve()
    if canonical_output.exists():
        raise FileExistsError(f"refusing to overwrite immutable P0 artifact {canonical_output}")

    preflight = _load_preflight_summary(
        Path(preflight_summary_path).resolve(),
        expected_cells=protocol["cells"],
        expected_source_commit=str(protocol["source_commit"]),
    )
    snapshot_digest = _canonical_json_sha256(preflight)
    snapshot_path = canonical_output.with_name(
        f"{canonical_output.stem}.preflight-{snapshot_digest[:16]}.json"
    )
    if snapshot_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable P0 artifact {snapshot_path}")
    _write_json_immutable(snapshot_path, preflight)
    protocol["preflight_summary"] = {
        "path": str(snapshot_path),
        "sha256": _sha256(snapshot_path),
        "environment": preflight["environment"],
    }
    _write_json_immutable(canonical_output, protocol)
    return protocol


def _validate_protocol_cell(path: Path, ordinal: int, item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"{path}: P0 protocol cell {ordinal} must be a JSON object")
    expected = {"defect", "level", "family", "task"}
    missing = sorted(expected - set(item))
    extra = sorted(set(item) - expected)
    if missing or extra:
        raise ValueError(
            f"{path}: P0 protocol cell {ordinal} fields mismatch; "
            f"missing={missing}, extra={extra}"
        )
    for name in ("defect", "family", "task"):
        if type(item[name]) is not str or not item[name]:
            raise ValueError(
                f"{path}: P0 protocol cell {ordinal} {name} must be a non-empty string"
            )
    if type(item["level"]) is not int:
        raise ValueError(f"{path}: P0 protocol cell {ordinal} level must be a JSON integer")
    return {
        "defect": item["defect"],
        "level": item["level"],
        "family": item["family"],
        "task": item["task"],
    }


def _load_goldilocks_protocol(path: Path, bundle: ConfigBundle | None = None) -> dict[str, Any]:
    resolved_bundle = bundle or load_bundle()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError(f"{path}: P0 protocol must be a JSON object")
    if protocol.get("schema_version") == 1:
        raise ValueError(
            f"{path}: protocol schema 1 is a historical descriptive artifact; "
            "it cannot authorize formal collection without a preregistered decision rule"
        )
    if (
        type(protocol.get("schema_version")) is not int
        or protocol.get("schema_version") != _PROTOCOL_SCHEMA
    ):
        raise ValueError(
            f"{path}: protocol schema {protocol.get('schema_version')}, expected {_PROTOCOL_SCHEMA}"
        )
    expected_top_level = {
        "schema_version",
        "producer",
        "source_commit",
        "worktree_dirty",
        "start_index",
        "n_rollouts_per_cell",
        "decision_rule",
        "preflight_summary",
        "cells",
    }
    missing = sorted(expected_top_level - set(protocol))
    extra = sorted(set(protocol) - expected_top_level)
    if missing or extra:
        raise ValueError(
            f"{path}: P0 protocol fields mismatch; missing={missing}, extra={extra}"
        )
    if protocol["producer"] != "portal_trace.scripts.calibrate_p0 plan-goldilocks":
        raise ValueError(f"{path}: P0 protocol has the wrong producer")
    if type(protocol["source_commit"]) is not str or not protocol["source_commit"]:
        raise ValueError(f"{path}: P0 protocol source_commit must be a non-empty string")
    if type(protocol["worktree_dirty"]) is not bool:
        raise ValueError(f"{path}: P0 protocol worktree_dirty must be a JSON boolean")
    if type(protocol["start_index"]) is not int:
        raise ValueError(f"{path}: P0 protocol start_index must be a JSON integer")
    if type(protocol["n_rollouts_per_cell"]) is not int:
        raise ValueError(f"{path}: P0 protocol n_rollouts_per_cell must be a JSON integer")
    if not isinstance(protocol["cells"], list):
        raise ValueError(f"{path}: P0 protocol cells must be a JSON array")
    cells = [
        _validate_protocol_cell(path, ordinal, item)
        for ordinal, item in enumerate(protocol["cells"])
    ]
    keys = {(c["defect"], c["level"], c["family"], c["task"]) for c in cells}
    if len(keys) != len(cells):
        raise ValueError(f"{path}: P0 protocol cells must be unique")
    expected = _ordered_formal_cells(resolved_bundle)
    if cells != expected:
        raise ValueError(
            f"{path}: P0 protocol does not equal the ordered preregistered matrix; "
            f"expected {len(expected)} cells, got {len(cells)}"
        )
    start = protocol["start_index"]
    count = protocol["n_rollouts_per_cell"]
    if start != 0 or count != GOLDILOCKS_ROLLOUTS_PER_CELL:
        raise ValueError(
            f"{path}: formal P0 requires rollout indices 0.."
            f"{GOLDILOCKS_ROLLOUTS_PER_CELL - 1}, got start={start}, count={count}"
        )
    if protocol["source_commit"] == NO_CHECKOUT or protocol["worktree_dirty"] is not False:
        raise ValueError(f"{path}: formal P0 protocol must come from a clean Git commit")

    decision = _require_exact_fields(
        path,
        "protocol decision_rule",
        protocol["decision_rule"],
        {"path", "canonical_sha256", "before_unblinding_amendment", "document"},
    )
    decision_path = Path(str(decision["path"]))
    if not decision_path.is_absolute():
        raise ValueError(f"{path}: decision-rule path must be absolute")
    embedded_rule = _validate_decision_rule(decision["document"], decision_path, resolved_bundle)
    external_rule = _load_decision_rule(decision_path, resolved_bundle)
    expected_rule_sha = _canonical_json_sha256(embedded_rule)
    if (
        external_rule != embedded_rule
        or decision["canonical_sha256"] != expected_rule_sha
        or decision["before_unblinding_amendment"] != embedded_rule["before_unblinding_amendment"]
    ):
        raise ValueError(
            f"{path}: decision rule document, canonical SHA-256, amendment binding, "
            "and external file must agree"
        )

    preflight_binding = _require_exact_fields(
        path,
        "protocol preflight_summary",
        protocol["preflight_summary"],
        {"path", "sha256", "environment"},
    )
    preflight_path = Path(str(preflight_binding["path"]))
    if not preflight_path.is_absolute():
        raise ValueError(f"{path}: preflight-summary path must be absolute")
    if _sha256(preflight_path) != preflight_binding["sha256"]:
        raise ValueError(f"{path}: preflight summary SHA-256 no longer matches")
    preflight = _load_preflight_summary(
        preflight_path,
        expected_cells=cells,
        expected_source_commit=str(protocol["source_commit"]),
    )
    if preflight["environment"] != preflight_binding["environment"]:
        raise ValueError(f"{path}: preflight environment does not match its binding")
    return {
        **protocol,
        "start_index": start,
        "n_rollouts_per_cell": count,
        "decision_rule": {
            **decision,
            "document": embedded_rule,
        },
        "preflight_summary": preflight_binding,
        "cells": cells,
    }


def _job_id(cell_ordinal: int, start_index: int, stop_index: int) -> str:
    return f"p0-{cell_ordinal:06d}-r{start_index:02d}-{stop_index:02d}"


def _build_job_manifest_payload(
    *,
    protocol_path: Path,
    protocol: dict[str, Any],
    protocol_sha256: str,
    results_root: Path,
    rollouts_per_job: int,
) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    start = int(protocol["start_index"])
    stop = start + int(protocol["n_rollouts_per_cell"])
    shards_root = results_root / "calibration" / "p0" / protocol_sha256 / "shards"
    for cell_ordinal, cell in enumerate(protocol["cells"]):
        for shard_start in range(start, stop, rollouts_per_job):
            shard_stop = min(shard_start + rollouts_per_job, stop)
            job_id = _job_id(cell_ordinal, shard_start, shard_stop)
            jobs.append(
                {
                    "job_id": job_id,
                    "cell_ordinal": cell_ordinal,
                    **cell,
                    "start_index": shard_start,
                    "stop_index": shard_stop,
                    "n_rollouts": shard_stop - shard_start,
                    "protocol_sha256": protocol_sha256,
                    "decision_rule_sha256": protocol["decision_rule"]["canonical_sha256"],
                    "preflight_summary_sha256": protocol["preflight_summary"]["sha256"],
                    "output": str(shards_root / f"{job_id}.json"),
                }
            )
    return {
        "schema_version": _JOB_MANIFEST_SCHEMA,
        "producer": _JOB_MANIFEST_PRODUCER,
        "source_commit": protocol["source_commit"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "decision_rule_sha256": protocol["decision_rule"]["canonical_sha256"],
        "preflight_summary_sha256": protocol["preflight_summary"]["sha256"],
        "results_root": str(results_root),
        "rollouts_per_job": rollouts_per_job,
        "n_cells": len(protocol["cells"]),
        "n_jobs": len(jobs),
        "jobs": jobs,
    }


def build_goldilocks_job_manifest(
    protocol_path: Path,
    *,
    results_root: Path | None = None,
    rollouts_per_job: int = _DEFAULT_ROLLOUTS_PER_JOB,
    bundle: ConfigBundle | None = None,
) -> dict[str, Any]:
    """Build the deterministic, complete set of resumable formal P0 jobs."""
    if rollouts_per_job < 1:
        raise ValueError(f"rollouts_per_job must be positive, got {rollouts_per_job}")
    canonical_protocol = Path(protocol_path).resolve()
    protocol = _load_goldilocks_protocol(canonical_protocol, bundle)
    canonical_root = (results_root or _default_results_root()).resolve()
    return _build_job_manifest_payload(
        protocol_path=canonical_protocol,
        protocol=protocol,
        protocol_sha256=_sha256(canonical_protocol),
        results_root=canonical_root,
        rollouts_per_job=rollouts_per_job,
    )


def write_goldilocks_job_manifest(
    protocol_path: Path,
    output: Path,
    *,
    results_root: Path | None = None,
    rollouts_per_job: int = _DEFAULT_ROLLOUTS_PER_JOB,
    bundle: ConfigBundle | None = None,
) -> dict[str, Any]:
    manifest = build_goldilocks_job_manifest(
        protocol_path,
        results_root=results_root,
        rollouts_per_job=rollouts_per_job,
        bundle=bundle,
    )
    _write_json_immutable(output, manifest)
    return manifest


def _load_goldilocks_job_manifest(path: Path, bundle: ConfigBundle | None = None) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{path}: P0 job manifest must be a JSON object")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != _JOB_MANIFEST_SCHEMA
    ):
        raise ValueError(
            f"{path}: job-manifest schema {manifest.get('schema_version')}, "
            f"expected {_JOB_MANIFEST_SCHEMA}"
        )
    if manifest.get("producer") != _JOB_MANIFEST_PRODUCER:
        raise ValueError(f"{path}: is not a formal P0 job manifest")
    expected_top_level = {
        "schema_version",
        "producer",
        "source_commit",
        "protocol_path",
        "protocol_sha256",
        "decision_rule_sha256",
        "preflight_summary_sha256",
        "results_root",
        "rollouts_per_job",
        "n_cells",
        "n_jobs",
        "jobs",
    }
    missing = sorted(expected_top_level - set(manifest))
    extra = sorted(set(manifest) - expected_top_level)
    if missing or extra:
        raise ValueError(
            f"{path}: P0 job manifest fields mismatch; missing={missing}, extra={extra}"
        )
    if type(manifest["source_commit"]) is not str or not manifest["source_commit"]:
        raise ValueError(f"{path}: source_commit must be a non-empty string")
    for name in ("protocol_sha256", "decision_rule_sha256", "preflight_summary_sha256"):
        value = manifest[name]
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value.lower())
        ):
            raise ValueError(f"{path}: {name} must be a 64-character hexadecimal string")
    if type(manifest["protocol_path"]) is not str or not manifest["protocol_path"]:
        raise ValueError(f"{path}: protocol_path must be a non-empty string")
    if type(manifest["results_root"]) is not str or not manifest["results_root"]:
        raise ValueError(f"{path}: results_root must be a non-empty string")
    for name in ("protocol_path", "results_root"):
        if not Path(manifest[name]).is_absolute():
            raise ValueError(f"{path}: {name} must be absolute")
    for name in ("rollouts_per_job", "n_cells", "n_jobs"):
        if type(manifest[name]) is not int:
            raise ValueError(f"{path}: {name} must be a JSON integer")
    if manifest["rollouts_per_job"] < 1:
        raise ValueError(f"{path}: rollouts_per_job must be positive")
    if manifest["n_cells"] < 0 or manifest["n_jobs"] < 0:
        raise ValueError(f"{path}: n_cells and n_jobs must be non-negative")
    if not isinstance(manifest["jobs"], list):
        raise ValueError(f"{path}: jobs must be a JSON array")

    protocol_path = Path(manifest["protocol_path"])
    if _sha256(protocol_path) != manifest["protocol_sha256"]:
        raise ValueError(f"{path}: protocol SHA-256 no longer matches protocol_path")
    protocol = _load_goldilocks_protocol(protocol_path, bundle)
    expected = _build_job_manifest_payload(
        protocol_path=protocol_path,
        protocol=protocol,
        protocol_sha256=str(manifest["protocol_sha256"]),
        results_root=Path(str(manifest["results_root"])),
        rollouts_per_job=int(manifest["rollouts_per_job"]),
    )
    if manifest != expected:
        raise ValueError(
            f"{path}: job manifest does not equal the deterministic protocol expansion"
        )
    return manifest


def _job_identity(manifest: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    return _shard_identity(
        task=str(job["task"]),
        family=str(job["family"]),
        defect=str(job["defect"]),
        level=int(job["level"]),
        start_index=int(job["start_index"]),
        n_rollouts=int(job["n_rollouts"]),
        protocol_sha256=str(manifest["protocol_sha256"]),
        run_source_commit=str(manifest["source_commit"]),
        run_worktree_dirty=False,
        decision_rule_sha256=str(manifest["decision_rule_sha256"]),
        preflight_summary_sha256=str(manifest["preflight_summary_sha256"]),
    )


def _load_job_events(
    output: Path,
    identity: dict[str, Any],
    job: dict[str, Any],
    bundle: ConfigBundle,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in sorted(_event_dir(output).glob("*.json")):
        event = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(event, dict) or event.get("schema_version") != _EVENT_SCHEMA:
            raise ValueError(f"{path}: invalid P0 event schema")
        if event.get("producer") != _EVENT_PRODUCER:
            raise ValueError(f"{path}: is not a P0 progress event")
        expected = {
            "job_id": job["job_id"],
            "cell_ordinal": job["cell_ordinal"],
            "task": identity["task"],
            "family": identity["family"],
            "defect": identity["defect"],
            "level": identity["level"],
        }
        mismatches = {
            name: (event.get(name), value)
            for name, value in expected.items()
            if event.get(name) != value
        }
        if mismatches:
            raise ValueError(f"{path}: event identity mismatch {mismatches}")
        for name in ("attempt",):
            if type(event.get(name)) is not int or event[name] < 0:
                raise ValueError(f"{path}: event {name} must be a non-negative integer")
        demo_index = event.get("demo_index")
        seed_env = event.get("seed_env")
        if demo_index is not None:
            if type(demo_index) is not int or not (
                int(job["start_index"]) <= demo_index < int(job["stop_index"])
            ):
                raise ValueError(f"{path}: event demo_index is outside the job interval")
            expected_seed = bundle.seeds.env_seed(
                task=str(job["task"]),
                family=str(job["family"]),
                demo_index=demo_index,
            )
            if seed_env != expected_seed:
                raise ValueError(f"{path}: event seed does not match its rollout")
            if type(seed_env) is not int:
                raise ValueError(f"{path}: rollout event seed_env must be an integer")
        elif seed_env is not None:
            raise ValueError(f"{path}: job event seed_env must be null")
        try:
            datetime.fromisoformat(str(event["start"]))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{path}: event start is not an ISO timestamp") from exc
        kind = event.get("event")
        if kind == "start":
            if (
                event.get("end") is not None
                or event.get("elapsed_seconds") is not None
                or event.get("result") != "running"
            ):
                raise ValueError(f"{path}: malformed start event")
        elif kind in {"complete", "failed"}:
            try:
                datetime.fromisoformat(str(event["end"]))
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{path}: event end is not an ISO timestamp") from exc
            elapsed = event.get("elapsed_seconds")
            if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or elapsed < 0:
                raise ValueError(f"{path}: event elapsed_seconds must be non-negative")
            if event.get("result") != kind:
                raise ValueError(f"{path}: event result does not match event type")
            if kind == "complete":
                outcome = _validate_outcome(path, event.get("outcome"))
                if outcome["demo_index"] != demo_index or outcome["seed_env"] != seed_env:
                    raise ValueError(f"{path}: event outcome identity mismatch")
            else:
                error = event.get("error")
                if not isinstance(error, dict) or not all(
                    isinstance(error.get(name), str) for name in ("type", "message")
                ):
                    raise ValueError(f"{path}: failed event requires type and message")
        else:
            raise ValueError(f"{path}: unknown P0 event type {kind!r}")
        events.append(event)
    starts: dict[tuple[int | None, int], dict[str, Any]] = {}
    for event in events:
        key = (event["demo_index"], event["attempt"])
        if event["event"] == "start":
            if key in starts:
                raise ValueError(f"{output}: duplicate start event for {key}")
            starts[key] = event
    terminals: set[tuple[int | None, int]] = set()
    for event in events:
        if event["event"] == "start":
            continue
        key = (event["demo_index"], event["attempt"])
        if key not in starts:
            raise ValueError(f"{output}: terminal event has no matching start for {key}")
        if key in terminals:
            raise ValueError(f"{output}: duplicate terminal event for {key}")
        if event["start"] != starts[key]["start"]:
            raise ValueError(f"{output}: terminal/start timestamp mismatch for {key}")
        terminals.add(key)
    return events


def _latest_event_status(events: Sequence[dict[str, Any]]) -> str:
    if not events:
        return "pending"
    latest = max(
        events,
        key=lambda event: (
            str(event["start"]),
            int(event["attempt"]),
            0 if event["event"] == "start" else 1,
        ),
    )
    return "failed" if latest["result"] == "failed" else "pending"


def _load_job_parts(
    output: Path,
    identity: dict[str, Any],
    job: dict[str, Any],
    bundle: ConfigBundle,
    *,
    require_all: bool,
) -> dict[int, dict[str, Any]]:
    expected_indices = range(int(job["start_index"]), int(job["stop_index"]))
    expected_parts = {_rollout_part_path(output, index): index for index in expected_indices}
    parts_dir = output.with_name(f"{output.name}.parts")
    existing_parts = set(parts_dir.glob("*.json")) if parts_dir.exists() else set()
    unexpected = sorted(str(part) for part in existing_parts - set(expected_parts))
    if unexpected:
        raise ValueError(f"{output}: unexpected rollout parts: {unexpected[:5]}")
    if require_all and existing_parts != set(expected_parts):
        missing = sorted(str(part) for part in set(expected_parts) - existing_parts)
        raise ValueError(f"{output}: complete shard is missing rollout parts {missing[:5]}")

    loaded: dict[int, dict[str, Any]] = {}
    for part_path, demo_index in expected_parts.items():
        if not part_path.exists():
            continue
        seed_env = bundle.seeds.env_seed(
            task=str(job["task"]),
            family=str(job["family"]),
            demo_index=demo_index,
        )
        loaded[demo_index] = _load_rollout_part(
            part_path,
            identity=identity,
            demo_index=demo_index,
            seed_env=seed_env,
        )
    signatures = {_provenance_signature(part) for part in loaded.values()}
    if len(signatures) > 1:
        raise ValueError(f"{output}: rollout parts disagree on provenance")
    return loaded


def _scan_goldilocks_job(
    manifest: dict[str, Any], job: dict[str, Any], bundle: ConfigBundle
) -> dict[str, Any]:
    output = Path(str(job["output"]))
    identity = _job_identity(manifest, job)
    try:
        events = _load_job_events(output, identity, job, bundle)
        parts = _load_job_parts(
            output,
            identity,
            job,
            bundle,
            require_all=output.exists(),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "job_id": job["job_id"],
            "status": "corrupt",
            "output": str(output),
            "completed_rollouts": 0,
            "detail": str(exc),
        }
    if output.exists():
        try:
            artifact = _validate_existing_shard(output, identity=identity, bundle=bundle)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return {
                "job_id": job["job_id"],
                "status": "corrupt",
                "output": str(output),
                "completed_rollouts": 0,
                "detail": str(exc),
            }
        completed_event_indices = {
            event["demo_index"]
            for event in events
            if event["event"] == "complete" and event["demo_index"] is not None
        }
        artifact_indices = {row["demo_index"] for row in artifact["outcomes"]}
        if completed_event_indices != artifact_indices:
            return {
                "job_id": job["job_id"],
                "status": "corrupt",
                "output": str(output),
                "completed_rollouts": len(artifact["outcomes"]),
                "detail": "complete shard does not have one logged completion per rollout",
            }
        part_outcomes = [parts[index]["outcome"] for index in sorted(parts)]
        if part_outcomes != artifact["outcomes"]:
            return {
                "job_id": job["job_id"],
                "status": "corrupt",
                "output": str(output),
                "completed_rollouts": len(artifact["outcomes"]),
                "detail": "complete shard outcomes disagree with rollout parts",
            }
        if parts and _provenance_signature(artifact) != _provenance_signature(
            next(iter(parts.values()))
        ):
            return {
                "job_id": job["job_id"],
                "status": "corrupt",
                "output": str(output),
                "completed_rollouts": len(artifact["outcomes"]),
                "detail": "complete shard provenance disagrees with rollout parts",
            }
        return {
            "job_id": job["job_id"],
            "status": "complete",
            "output": str(output),
            "completed_rollouts": len(artifact["outcomes"]),
            "detail": None,
        }

    return {
        "job_id": job["job_id"],
        "status": _latest_event_status(events),
        "output": str(output),
        "completed_rollouts": len(parts),
        "detail": None,
    }


def scan_goldilocks_jobs(manifest_path: Path, bundle: ConfigBundle | None = None) -> dict[str, Any]:
    """Classify every planned job and return exact IDs needing another run."""
    resolved_bundle = bundle or load_bundle()
    manifest = _load_goldilocks_job_manifest(manifest_path, resolved_bundle)
    jobs = [_scan_goldilocks_job(manifest, job, resolved_bundle) for job in manifest["jobs"]]
    counts = {
        status: sum(job["status"] == status for job in jobs)
        for status in ("pending", "complete", "failed", "corrupt")
    }
    rerun_job_ids = [str(job["job_id"]) for job in jobs if job["status"] in {"pending", "failed"}]
    corrupt_job_ids = [str(job["job_id"]) for job in jobs if job["status"] == "corrupt"]
    failed_job_ids = [str(job["job_id"]) for job in jobs if job["status"] == "failed"]
    incomplete_job_ids = [str(job["job_id"]) for job in jobs if job["status"] != "complete"]
    return {
        "schema_version": 1,
        "producer": "portal_trace.scripts.calibrate_p0 status-jobs",
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": _sha256(Path(manifest_path)),
        "counts": counts,
        "rerun_job_ids": rerun_job_ids,
        "corrupt_job_ids": corrupt_job_ids,
        "failed_job_ids": failed_job_ids,
        "incomplete_job_ids": incomplete_job_ids,
        "jobs": jobs,
    }


def run_goldilocks_job(
    manifest_path: Path,
    job_id: str,
    *,
    bundle: ConfigBundle | None = None,
) -> Path:
    resolved_bundle = bundle or load_bundle()
    manifest = _load_goldilocks_job_manifest(manifest_path, resolved_bundle)
    matches = [job for job in manifest["jobs"] if job["job_id"] == job_id]
    if not matches:
        raise ValueError(f"{job_id!r} is not present in {manifest_path}")
    job = matches[0]
    output = Path(str(job["output"]))
    failures_before = len(list(_event_dir(output).glob("*-failed.json")))
    try:
        return collect_goldilocks_artifact(
            task=str(job["task"]),
            family=str(job["family"]),
            defect=str(job["defect"]),
            level=int(job["level"]),
            n_rollouts=int(job["n_rollouts"]),
            start_index=int(job["start_index"]),
            output=output,
            protocol_path=Path(str(manifest["protocol_path"])),
            bundle=resolved_bundle,
            job_id=job_id,
            cell_ordinal=int(job["cell_ordinal"]),
        )
    except BaseException as exc:
        failures_after = len(list(_event_dir(output).glob("*-failed.json")))
        if failures_after == failures_before:
            _record_failure_event(
                output,
                identity=_job_identity(manifest, job),
                job_id=job_id,
                cell_ordinal=int(job["cell_ordinal"]),
                error=exc,
            )
        raise


def _collect_goldilocks_artifact_impl(
    *,
    task: str,
    family: str,
    defect: str,
    level: int,
    n_rollouts: int,
    start_index: int,
    output: Path,
    protocol_path: Path | None = None,
    bundle: ConfigBundle | None = None,
    job_id: str | None = None,
    cell_ordinal: int | None = None,
) -> Path:
    """Run one P0 cell, checkpointing each paired rollout before final assembly."""
    if n_rollouts < 1:
        raise ValueError(f"n_rollouts must be positive, got {n_rollouts}")
    if start_index < 0:
        raise ValueError(f"start_index must be non-negative, got {start_index}")

    resolved_bundle = bundle or load_bundle()
    spec = task_spec(task)
    defect_spec(defect).level(level)
    eval_seed = resolved_bundle.seeds.eval_seed(task=task, family=family)
    protocol_sha256: str | None = None
    decision_rule_sha256: str | None = None
    preflight_summary_sha256: str | None = None
    preflight_environment: dict[str, Any] | None = None
    if protocol_path is not None:
        protocol = _load_goldilocks_protocol(protocol_path, resolved_bundle)
        key = (defect, level, family, task)
        protocol_keys = {
            (c["defect"], c["level"], c["family"], c["task"]) for c in protocol["cells"]
        }
        if key not in protocol_keys:
            raise ValueError(f"{key}: cell is outside the formal P0 protocol")
        stop_index = start_index + n_rollouts
        protocol_stop = protocol["start_index"] + protocol["n_rollouts_per_cell"]
        if start_index < protocol["start_index"] or stop_index > protocol_stop:
            raise ValueError(
                f"requested rollout range [{start_index}, {stop_index}) is outside "
                f"the protocol range [{protocol['start_index']}, {protocol_stop})"
            )
        if source_commit() != protocol["source_commit"] or _worktree_dirty() is not False:
            raise ValueError(
                "formal P0 collection must run at the protocol's source commit in a clean worktree"
            )
        protocol_sha256 = _sha256(protocol_path)
        decision_rule_sha256 = str(protocol["decision_rule"]["canonical_sha256"])
        preflight_summary_sha256 = str(protocol["preflight_summary"]["sha256"])
        preflight_environment = protocol["preflight_summary"]["environment"]

    run_source_commit = source_commit()
    run_worktree_dirty = _worktree_dirty()
    run_environment = fingerprint(Backend.CPU_MUJOCO).to_json()
    if preflight_environment is not None and run_environment != preflight_environment:
        raise ValueError(
            "formal P0 collection environment does not match the bound preflight summary"
        )
    identity = _shard_identity(
        task=task,
        family=family,
        defect=defect,
        level=level,
        start_index=start_index,
        n_rollouts=n_rollouts,
        protocol_sha256=protocol_sha256,
        run_source_commit=run_source_commit,
        run_worktree_dirty=run_worktree_dirty,
        environment=run_environment,
        decision_rule_sha256=decision_rule_sha256,
        preflight_summary_sha256=preflight_summary_sha256,
    )
    if output.exists():
        _validate_existing_shard(output, identity=identity, bundle=resolved_bundle)
        return output

    expected_indices = range(start_index, start_index + n_rollouts)
    expected_part_paths = {_rollout_part_path(output, i) for i in expected_indices}
    parts_dir = output.with_name(f"{output.name}.parts")
    if parts_dir.exists():
        unexpected = sorted(
            str(path) for path in parts_dir.glob("*.json") if path not in expected_part_paths
        )
        if unexpected:
            raise ValueError(
                f"{output}: recovery directory contains rollout parts outside the "
                f"requested interval: {unexpected[:5]}"
            )

    outcomes_by_index: dict[int, dict[str, Any]] = {}
    provenance_by_index: dict[int, dict[str, Any]] = {}
    for demo_index in expected_indices:
        seed_env = resolved_bundle.seeds.env_seed(task=task, family=family, demo_index=demo_index)
        part_path = _rollout_part_path(output, demo_index)
        if part_path.exists():
            part = _load_rollout_part(
                part_path,
                identity=identity,
                demo_index=demo_index,
                seed_env=seed_env,
            )
            outcomes_by_index[demo_index] = part["outcome"]
            provenance_by_index[demo_index] = {
                "environment": part["environment"],
                "controller_config_sha256": part["controller_config_sha256"],
            }
            completion_pattern = f"rollout-{demo_index:06d}-a*-complete.json"
            if not any(_event_dir(output).glob(completion_pattern)):
                recovered = _start_event(
                    output,
                    identity=identity,
                    job_id=job_id,
                    cell_ordinal=cell_ordinal,
                    demo_index=demo_index,
                    seed_env=seed_env,
                )
                _finish_event(
                    recovered,
                    result="complete",
                    outcome=part["outcome"],
                )

    if len({_provenance_signature(item) for item in provenance_by_index.values()}) > 1:
        raise ValueError(f"{output}: recovered rollout parts disagree on provenance")

    from portal_trace.defects.delta import Delta, default_joint_scale
    from portal_trace.defects.inject import build_twin_pair
    from portal_trace.registry.families import descriptor
    from portal_trace.teacher import osc, refs

    n_dof = descriptor(family).dof
    delta = Delta.from_level(defect, level, joint_scale=default_joint_scale(defect, n_dof))
    reference_spec = refs.load_spec(task)

    for demo_index in expected_indices:
        seed_env = resolved_bundle.seeds.env_seed(task=task, family=family, demo_index=demo_index)
        if demo_index in outcomes_by_index:
            continue
        event = _start_event(
            output,
            identity=identity,
            job_id=job_id,
            cell_ordinal=cell_ordinal,
            demo_index=demo_index,
            seed_env=seed_env,
        )
        try:
            seeds = SeedTriple(env=seed_env, train=_UNUSED_TRAIN_SEED, eval=eval_seed)
            pair = build_twin_pair(spec, family, delta, seeds)
            try:
                osc.reset_twin_pair(pair, seed_env)
                reference = refs.resolve(reference_spec, pair.defective.sim)
                defective = osc.OSCTracker().track(
                    pair.defective,
                    reference,
                    family=family,
                    seed_env=seed_env,
                    feedforward=None,
                )
                clean = osc.OSCTracker().track(
                    pair.clean,
                    reference,
                    family=family,
                    seed_env=seed_env,
                    feedforward=None,
                )
            finally:
                pair.defective.close()
                pair.clean.close()
            outcome = {
                "demo_index": demo_index,
                "seed_env": seed_env,
                "defective_success": defective.success,
                "defective_t_success": defective.t_success,
                "clean_success": clean.success,
                "clean_t_success": clean.t_success,
            }
            controller_hashes = _validate_controller_hashes(
                output,
                {
                    "defective": defective.controller_config_sha256,
                    "clean": clean.controller_config_sha256,
                },
            )
            provenance = {
                "environment": run_environment,
                "controller_config_sha256": controller_hashes,
            }
            existing_signatures = {
                _provenance_signature(item) for item in provenance_by_index.values()
            }
            if existing_signatures and _provenance_signature(provenance) not in existing_signatures:
                raise ValueError(
                    f"{output}: rollout {demo_index} provenance disagrees with recovered parts"
                )

            part_path = _rollout_part_path(output, demo_index)
            part = {
                "schema_version": _ROLLOUT_PART_SCHEMA,
                **{k: v for k, v in identity.items() if k != "producer"},
                "producer": _ROLLOUT_PART_PRODUCER,
                "controller_config_sha256": controller_hashes,
                "outcome": outcome,
            }
            try:
                _write_json_immutable(part_path, part)
                published = part
            except FileExistsError:
                published = _load_rollout_part(
                    part_path,
                    identity=identity,
                    demo_index=demo_index,
                    seed_env=seed_env,
                )
                if published != part:
                    raise ValueError(
                        f"{part_path}: concurrent rollout result disagrees with this run"
                    ) from None
            outcomes_by_index[demo_index] = published["outcome"]
            provenance_by_index[demo_index] = {
                "environment": published["environment"],
                "controller_config_sha256": published["controller_config_sha256"],
            }
            _finish_event(event, result="complete", outcome=outcome)
        except BaseException as exc:
            _finish_event(event, result="failed", error=exc)
            raise

    if protocol_path is not None and (
        source_commit() != run_source_commit or _worktree_dirty() is not False
    ):
        raise ValueError(
            "formal P0 worktree changed during collection; rollout parts remain "
            "available, but no formal shard was published"
        )

    outcomes = [outcomes_by_index[i] for i in expected_indices]
    provenance = [provenance_by_index[i] for i in expected_indices]
    if len({_provenance_signature(item) for item in provenance}) != 1:
        raise ValueError(f"{output}: rollout parts disagree on environment or controller")

    artifact = {
        "schema_version": _CURRENT_ARTIFACT_SCHEMA,
        **identity,
        "controller_config_sha256": provenance[0]["controller_config_sha256"],
        "outcomes": outcomes,
    }
    try:
        _write_json_immutable(output, artifact)
    except FileExistsError:
        existing = _validate_existing_shard(output, identity=identity, bundle=resolved_bundle)
        if existing != artifact:
            raise ValueError(
                f"{output}: concurrently published shard disagrees with this run"
            ) from None
    return output


def collect_goldilocks_artifact(
    *,
    task: str,
    family: str,
    defect: str,
    level: int,
    n_rollouts: int,
    start_index: int,
    output: Path,
    protocol_path: Path | None = None,
    bundle: ConfigBundle | None = None,
    job_id: str | None = None,
    cell_ordinal: int | None = None,
) -> Path:
    """Collect a resumable shard and persist a failure event for every exception."""
    failures_before = len(list(_event_dir(output).glob("*-failed.json")))
    try:
        return _collect_goldilocks_artifact_impl(
            task=task,
            family=family,
            defect=defect,
            level=level,
            n_rollouts=n_rollouts,
            start_index=start_index,
            output=output,
            protocol_path=protocol_path,
            bundle=bundle,
            job_id=job_id,
            cell_ordinal=cell_ordinal,
        )
    except BaseException as exc:
        failures_after = len(list(_event_dir(output).glob("*-failed.json")))
        if failures_after == failures_before:
            protocol_sha256 = None
            if protocol_path is not None and protocol_path.is_file():
                protocol_sha256 = _sha256(protocol_path)
            identity = _shard_identity(
                task=task,
                family=family,
                defect=defect,
                level=level,
                start_index=start_index,
                n_rollouts=n_rollouts,
                protocol_sha256=protocol_sha256,
                run_source_commit=source_commit(),
                run_worktree_dirty=_worktree_dirty(),
            )
            _record_failure_event(
                output,
                identity=identity,
                job_id=job_id,
                cell_ordinal=cell_ordinal,
                error=exc,
            )
        raise


def load_goldilocks_artifact(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") not in _ARTIFACT_SCHEMAS:
        raise ValueError(
            f"{path}: artifact schema {artifact.get('schema_version')}, "
            f"expected one of {_ARTIFACT_SCHEMAS}"
        )
    required = {
        "source_commit",
        "worktree_dirty",
        "task",
        "family",
        "defect",
        "level",
        "start_index",
        "n_rollouts",
        "outcomes",
    }
    if missing := sorted(required - set(artifact)):
        raise ValueError(f"{path}: artifact is missing fields {missing}")
    if artifact["schema_version"] == _CURRENT_ARTIFACT_SCHEMA:
        current_required = {
            "producer",
            "tracker",
            "paired_reset_contract",
            "protocol_sha256",
            "decision_rule_sha256",
            "preflight_summary_sha256",
            "environment",
            "controller_config_sha256",
        }
        if missing := sorted(current_required - set(artifact)):
            raise ValueError(f"{path}: schema-2 artifact is missing fields {missing}")
        if artifact["paired_reset_contract"] != _PAIRED_RESET_CONTRACT:
            raise ValueError(
                f"{path}: schema-2 paired-reset marker "
                f"{artifact['paired_reset_contract']!r}, expected {_PAIRED_RESET_CONTRACT!r}"
            )
        _validate_environment(path, artifact["environment"])
        _validate_controller_hashes(path, artifact["controller_config_sha256"])
        for name in ("decision_rule_sha256", "preflight_summary_sha256"):
            value = artifact[name]
            if value is not None and (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value.lower())
            ):
                raise ValueError(f"{path}: {name} must be null or a SHA-256 string")
    defect_spec(str(artifact["defect"])).level(int(artifact["level"]))
    start = int(artifact["start_index"])
    count = int(artifact["n_rollouts"])
    outcomes = artifact["outcomes"]
    if start < 0 or count < 1 or len(outcomes) != count:
        raise ValueError(f"{path}: invalid start_index, n_rollouts, or outcome count")
    expected_indices = list(range(start, start + count))
    actual_indices = [int(row["demo_index"]) for row in outcomes]
    if actual_indices != expected_indices:
        raise ValueError(
            f"{path}: outcome demo indices {actual_indices} do not match "
            f"the declared interval {expected_indices}"
        )
    for row in outcomes:
        _validate_outcome(path, row)
    return artifact


def _expected_formal_cells(bundle: ConfigBundle) -> set[tuple[str, int, str, str]]:
    return {
        (cell["defect"], cell["level"], cell["family"], cell["task"])
        for cell in _ordered_formal_cells(bundle)
    }


def _table_payload(table: CalibrationTable) -> dict[str, Any]:
    return {
        "name": table.name,
        "title": table.title,
        "proposal_ref": table.proposal_ref,
        "columns": list(table.columns),
        "units": dict(table.units),
        "rows": [dict(row) for row in table.rows],
        "notes": list(table.notes),
    }


def _validate_formal_provenance(
    protocol: dict[str, Any],
    protocol_sha256: str,
    loaded: Sequence[tuple[Path, dict[str, Any]]],
) -> None:
    _validate_formal_artifact_contract(loaded)
    commits = {str(artifact.get("source_commit")) for _, artifact in loaded}
    bad = [
        str(path)
        for path, artifact in loaded
        if artifact.get("source_commit") in (None, NO_CHECKOUT)
        or artifact.get("worktree_dirty") is not False
        or artifact.get("protocol_sha256") != protocol_sha256
        or artifact.get("decision_rule_sha256") != protocol["decision_rule"]["canonical_sha256"]
        or artifact.get("preflight_summary_sha256") != protocol["preflight_summary"]["sha256"]
    ]
    if commits != {str(protocol["source_commit"])} or bad:
        raise ValueError(
            "formal P0 artifacts must all come from the protocol's clean Git "
            "commit and carry its protocol, decision-rule and preflight SHA-256; "
            f"commits={sorted(commits)}, inadmissible={bad[:5]}"
        )
    signatures = {_provenance_signature(artifact) for _, artifact in loaded}
    if len(signatures) != 1:
        raise ValueError(
            "formal P0 artifacts disagree on environment fingerprint or controller_config_sha256"
        )


def _validate_formal_artifact_contract(
    loaded: Sequence[tuple[Path, dict[str, Any]]],
) -> None:
    legacy_or_unpaired = [
        str(path)
        for path, artifact in loaded
        if artifact.get("schema_version") != _CURRENT_ARTIFACT_SCHEMA
        or artifact.get("paired_reset_contract") != _PAIRED_RESET_CONTRACT
    ]
    if legacy_or_unpaired:
        raise ValueError(
            "formal P0 requires schema-2 artifacts carrying the paired-reset "
            f"contract marker; inadmissible={legacy_or_unpaired[:5]}"
        )


def assemble_goldilocks(
    inputs: Sequence[Path],
    *,
    report_path: Path,
    formal: bool = False,
    protocol_path: Path | None = None,
    manifest_path: Path | None = None,
    bundle: ConfigBundle | None = None,
) -> dict[str, Any]:
    """Combine cell shards, rejecting duplicate seeds and incomplete formal runs."""
    resolved_manifest_path: Path | None = None
    if formal and manifest_path is None:
        raise ValueError(
            "formal P0 assembly requires the completed job manifest; a protocol "
            "and manually selected shards cannot authorize a formal report"
        )
    if formal:
        resolved_bundle = bundle or load_bundle()
        assert manifest_path is not None
        resolved_manifest_path = Path(manifest_path).resolve()
        formal_manifest = _load_goldilocks_job_manifest(resolved_manifest_path, resolved_bundle)
        status = scan_goldilocks_jobs(resolved_manifest_path, resolved_bundle)
        if status["incomplete_job_ids"]:
            raise ValueError(
                "P0 job manifest is not complete; rerun_job_ids="
                f"{status['rerun_job_ids']}, corrupt_job_ids={status['corrupt_job_ids']}"
            )
        expected_inputs = [Path(str(job["output"])).resolve() for job in formal_manifest["jobs"]]
        actual_inputs = [Path(path).resolve() for path in inputs]
        if actual_inputs != expected_inputs:
            raise ValueError(
                "formal P0 inputs must exactly equal the job manifest's ordered "
                "outputs; manually selected, reordered, missing, or extra shards "
                "are not admissible"
            )
        manifest_protocol_path = Path(str(formal_manifest["protocol_path"])).resolve()
        if protocol_path is not None and Path(protocol_path).resolve() != manifest_protocol_path:
            raise ValueError("formal P0 protocol must be the protocol bound by the job manifest")
        protocol_path = manifest_protocol_path
    if not inputs:
        raise ValueError("at least one --input artifact is required")
    loaded = [(Path(path), load_goldilocks_artifact(Path(path))) for path in inputs]
    if formal:
        _validate_formal_artifact_contract(loaded)
    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    provenance: list[dict[str, str]] = []
    for path, artifact in loaded:
        key = (
            str(artifact["defect"]),
            int(artifact["level"]),
            str(artifact["family"]),
            str(artifact["task"]),
        )
        grouped[key].extend(artifact["outcomes"])
        provenance.append({"path": str(path), "sha256": _sha256(path)})

    cells: list[GoldilocksCell] = []
    for (code, level, family, task), outcomes in sorted(grouped.items()):
        demo_indices = [int(row["demo_index"]) for row in outcomes]
        if len(set(demo_indices)) != len(demo_indices):
            raise ValueError(
                f"{task}/{family}/{code}/L{level}: duplicate demo indices across artifacts"
            )
        cells.append(
            GoldilocksCell(
                defect=code,
                level=level,
                family=family,
                task=task,
                n_rollouts=len(outcomes),
                n_success=sum(bool(row["defective_success"]) for row in outcomes),
                clean_twin_n_rollouts=len(outcomes),
                clean_twin_n_success=sum(bool(row["clean_success"]) for row in outcomes),
            )
        )

    if formal:
        resolved_bundle = bundle or load_bundle()
        assert protocol_path is not None
        protocol = _load_goldilocks_protocol(protocol_path, resolved_bundle)
        protocol_sha256 = _sha256(protocol_path)
        expected = _expected_formal_cells(resolved_bundle)
        actual = set(grouped)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"formal P0 Goldilocks matrix is incomplete: missing {len(missing)} "
                f"cells and has {len(extra)} unexpected cells; first missing={missing[:5]}, "
                f"first unexpected={extra[:5]}"
            )
        required_indices = list(range(GOLDILOCKS_ROLLOUTS_PER_CELL))
        for (code, level, family, task), outcomes in sorted(grouped.items()):
            indices = sorted(int(row["demo_index"]) for row in outcomes)
            if indices != required_indices:
                raise ValueError(
                    f"formal {task}/{family}/{code}/L{level} uses demo indices "
                    f"{indices}; P0 requires {required_indices}"
                )
            for row in outcomes:
                expected_seed = resolved_bundle.seeds.env_seed(
                    task=task, family=family, demo_index=int(row["demo_index"])
                )
                if int(row["seed_env"]) != expected_seed:
                    raise ValueError(
                        f"formal {task}/{family}/{code}/L{level} demo "
                        f"{row['demo_index']} records seed {row['seed_env']}, "
                        f"seed table requires {expected_seed}"
                    )
        _validate_formal_provenance(protocol, protocol_sha256, loaded)
        if _worktree_dirty() is not False or source_commit() != protocol["source_commit"]:
            raise ValueError(
                "formal P0 assembly requires a clean worktree at the protocol's source commit"
            )

    table = build_goldilocks_table(cells)
    report = {
        "schema_version": 1,
        "producer": "portal_trace.scripts.calibrate_p0 assemble-goldilocks",
        "source_commit": source_commit(),
        "formal": formal,
        "protocol": (
            None
            if protocol_path is None
            else {
                "path": str(protocol_path),
                "sha256": _sha256(protocol_path),
                "decision_rule_sha256": (
                    protocol["decision_rule"]["canonical_sha256"] if formal else None
                ),
                "preflight_summary_sha256": (
                    protocol["preflight_summary"]["sha256"] if formal else None
                ),
            }
        ),
        "manifest": (
            None
            if resolved_manifest_path is None
            else {
                "path": str(resolved_manifest_path),
                "sha256": _sha256(resolved_manifest_path),
            }
        ),
        "n_cells": len(cells),
        "n_rollouts": sum(cell.n_rollouts for cell in cells),
        "artifacts": provenance,
        "table": _table_payload(table),
    }
    _write_json_immutable(report_path, report)
    return report


def assemble_goldilocks_manifest(
    manifest_path: Path,
    report_path: Path,
    *,
    bundle: ConfigBundle | None = None,
) -> dict[str, Any]:
    """Formally assemble exactly the completed shards named by one manifest."""
    resolved_bundle = bundle or load_bundle()
    manifest = _load_goldilocks_job_manifest(manifest_path, resolved_bundle)
    return assemble_goldilocks(
        [Path(str(job["output"])) for job in manifest["jobs"]],
        report_path=report_path,
        formal=True,
        protocol_path=Path(str(manifest["protocol_path"])),
        manifest_path=manifest_path,
        bundle=resolved_bundle,
    )


def _load_formal_report_for_decision(
    report_path: Path,
    protocol_path: Path,
    decision_rule_path: Path,
    bundle: ConfigBundle,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    canonical_report = Path(report_path).resolve()
    canonical_protocol = Path(protocol_path).resolve()
    canonical_rule = Path(decision_rule_path).resolve()
    protocol = _load_goldilocks_protocol(canonical_protocol, bundle)
    rule = _load_decision_rule(canonical_rule, bundle)
    protocol_rule = protocol["decision_rule"]
    if (
        Path(str(protocol_rule["path"])).resolve() != canonical_rule
        or protocol_rule["document"] != rule
        or protocol_rule["canonical_sha256"] != _canonical_json_sha256(rule)
    ):
        raise ValueError(
            "decision-rule input does not equal the external rule bound by the protocol"
        )

    report = json.loads(canonical_report.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"{canonical_report}: formal P0 report must be a JSON object")
    required = {
        "schema_version",
        "producer",
        "source_commit",
        "formal",
        "protocol",
        "manifest",
        "n_cells",
        "n_rollouts",
        "artifacts",
        "table",
    }
    missing = sorted(required - set(report))
    extra = sorted(set(report) - required)
    if missing or extra:
        raise ValueError(
            f"{canonical_report}: formal P0 report fields mismatch; "
            f"missing={missing}, extra={extra}"
        )
    if (
        report["schema_version"] != 1
        or report["producer"] != "portal_trace.scripts.calibrate_p0 assemble-goldilocks"
        or report["formal"] is not True
        or report["source_commit"] != protocol["source_commit"]
    ):
        raise ValueError(f"{canonical_report}: is not a formal P0 report from the protocol commit")

    report_protocol = report["protocol"]
    expected_protocol_binding = {
        "path": str(canonical_protocol),
        "sha256": _sha256(canonical_protocol),
        "decision_rule_sha256": protocol_rule["canonical_sha256"],
        "preflight_summary_sha256": protocol["preflight_summary"]["sha256"],
    }
    if (
        not isinstance(report_protocol, dict)
        or {
            **report_protocol,
            "path": str(Path(str(report_protocol.get("path"))).resolve()),
        }
        != expected_protocol_binding
    ):
        raise ValueError(f"{canonical_report}: formal report protocol binding does not match")

    manifest_binding = report["manifest"]
    if not isinstance(manifest_binding, dict) or set(manifest_binding) != {
        "path",
        "sha256",
    }:
        raise ValueError(f"{canonical_report}: formal report must bind exactly one job manifest")
    manifest_path = Path(str(manifest_binding["path"])).resolve()
    if _sha256(manifest_path) != manifest_binding["sha256"]:
        raise ValueError(f"{canonical_report}: bound job manifest SHA-256 is stale")
    manifest = _load_goldilocks_job_manifest(manifest_path, bundle)
    if (
        Path(str(manifest["protocol_path"])).resolve() != canonical_protocol
        or manifest["protocol_sha256"] != expected_protocol_binding["sha256"]
    ):
        raise ValueError(f"{canonical_report}: job manifest binds another protocol")

    artifact_bindings = report["artifacts"]
    if not isinstance(artifact_bindings, list):
        raise ValueError(f"{canonical_report}: artifacts must be a JSON array")
    expected_outputs = [Path(str(job["output"])).resolve() for job in manifest["jobs"]]
    if len(artifact_bindings) != len(expected_outputs):
        raise ValueError(f"{canonical_report}: artifact bindings do not cover every manifest job")
    for index, (binding, expected_path) in enumerate(
        zip(artifact_bindings, expected_outputs, strict=True)
    ):
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError(
                f"{canonical_report}: artifact binding {index} must contain path and sha256"
            )
        bound_path = Path(str(binding["path"])).resolve()
        if bound_path != expected_path or _sha256(bound_path) != binding["sha256"]:
            raise ValueError(
                f"{canonical_report}: artifact binding {index} is stale or not from the manifest"
            )

    table = report["table"]
    rows = table.get("rows") if isinstance(table, dict) else None
    if not isinstance(table, dict) or table.get("name") != "goldilocks_success_rate":
        raise ValueError(f"{canonical_report}: report has no Goldilocks table")
    if not isinstance(rows, list):
        raise ValueError(f"{canonical_report}: Goldilocks rows must be a JSON array")
    expected_cells = {
        (cell["defect"], cell["level"], cell["family"], cell["task"])
        for cell in _ordered_formal_cells(bundle)
    }
    if report["n_cells"] != len(expected_cells) or report["n_rollouts"] != len(
        expected_cells
    ) * int(protocol["n_rollouts_per_cell"]):
        raise ValueError(f"{canonical_report}: formal report cell or rollout count is wrong")

    aggregation = rule["aggregation_thresholds"]
    low, high = aggregation["cell_success_rate_pct"]
    clean_min = aggregation["minimum_clean_twin_rate_pct"]
    normalized: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{canonical_report}: Goldilocks row {index} is not an object")
        key = (row.get("defect"), row.get("level"), row.get("family"), row.get("task"))
        if key not in expected_cells or key in normalized:
            raise ValueError(
                f"{canonical_report}: Goldilocks row {index} is duplicate or unexpected: {key}"
            )
        count = row.get("n_rollouts")
        defective_rate = row.get("success_rate")
        clean_rate = row.get("clean_twin_rate")
        if type(count) is not int or count != protocol["n_rollouts_per_cell"]:
            raise ValueError(f"{canonical_report}: {key} has the wrong rollout count")
        if any(
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(rate)
            or not 0.0 <= rate <= 100.0
            for rate in (defective_rate, clean_rate)
        ):
            raise ValueError(f"{canonical_report}: {key} has an invalid success rate")
        for name, rate in (
            ("success_rate", defective_rate),
            ("clean_twin_rate", clean_rate),
        ):
            successes = rate * count / 100.0
            if not math.isclose(successes, round(successes), abs_tol=1e-9):
                raise ValueError(
                    f"{canonical_report}: {key} {name} is not attainable in {count} rollouts"
                )
        bites = low <= defective_rate <= high
        ceiling_holds = clean_rate >= clean_min
        if (
            row.get("bites") is not bites
            or row.get("ceiling_holds") is not ceiling_holds
            or row.get("passes") is not (bites and ceiling_holds)
        ):
            raise ValueError(
                f"{canonical_report}: {key} predicates disagree with the approved rule"
            )
        normalized[key] = row
    missing_cells = sorted(expected_cells - set(normalized))
    if missing_cells:
        raise ValueError(
            f"{canonical_report}: Goldilocks report is missing formal cells; "
            f"first missing={missing_cells[:5]}"
        )
    return report, protocol, {"rule": rule, "rows": normalized}


def build_decision_level_result(
    report_path: Path,
    protocol_path: Path,
    decision_rule_path: Path,
    bundle: ConfigBundle | None = None,
) -> dict[str, Any]:
    """Reduce a formal P0 table using only the externally approved rule."""
    resolved_bundle = bundle or load_bundle()
    canonical_report = Path(report_path).resolve()
    canonical_protocol = Path(protocol_path).resolve()
    report, protocol, inputs = _load_formal_report_for_decision(
        canonical_report,
        canonical_protocol,
        Path(decision_rule_path),
        resolved_bundle,
    )
    rule = inputs["rule"]
    rows = inputs["rows"]
    core_tasks = rule["ordered_core_tasks"]
    heldout_tasks = rule["ordered_heldout_tasks"]
    aggregation = rule["aggregation_thresholds"]
    tie_order = rule["tie_break"]["ordered_levels"]
    subset = resolved_bundle.subsets["S36"]

    decisions: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for ordinal, (family, code) in enumerate(subset.cells()):
        level_evidence = []
        eligible_by_level: dict[int, bool] = {}
        for level in _decision_level_indices(resolved_bundle):
            core_passing = [
                task for task in core_tasks if rows[(code, level, family, task)]["passes"]
            ]
            heldout_passing = [
                task for task in heldout_tasks if rows[(code, level, family, task)]["passes"]
            ]
            eligible = (
                len(core_passing) >= aggregation["minimum_core_tasks_passing"]
                and len(heldout_passing) >= aggregation["minimum_heldout_tasks_passing"]
            )
            eligible_by_level[level] = eligible
            level_evidence.append(
                {
                    "level": level,
                    "core_tasks_passing": core_passing,
                    "heldout_tasks_passing": heldout_passing,
                    "n_core_tasks_passing": len(core_passing),
                    "n_heldout_tasks_passing": len(heldout_passing),
                    "eligible": eligible,
                }
            )
        eligible_levels = [level for level in tie_order if eligible_by_level[level]]
        selected = eligible_levels[0] if eligible_levels else None
        decision = {
            "ordinal": ordinal,
            "family": family,
            "defect": code,
            "status": "selected" if selected is not None else "no_selection",
            "selected_level": selected,
            "eligible_levels_in_tie_break_order": eligible_levels,
            "level_evidence": level_evidence,
        }
        decisions.append(decision)
        if selected is not None:
            entries.append({"family": family, "defect": code, "level": selected})

    admission_rule = rule["defect_admission"]
    defect_rows = []
    for code in subset.defects:
        selected_families = [
            family
            for family in subset.families
            if any(
                row["family"] == family and row["defect"] == code and row["status"] == "selected"
                for row in decisions
            )
        ]
        admitted = len(selected_families) >= admission_rule["minimum_families_with_selected_level"]
        defect_rows.append(
            {
                "defect": code,
                "selected_families": selected_families,
                "n_families_with_selected_level": len(selected_families),
                "admitted": admitted,
            }
        )
    admitted_defects = [row["defect"] for row in defect_rows if row["admitted"]]
    admission_passed = len(admitted_defects) >= admission_rule["minimum_admitted_defects"]
    complete = len(entries) == len(tuple(subset.cells()))
    protocol_rule = protocol["decision_rule"]
    manifest_binding = report["manifest"]
    return {
        "schema": _DECISION_LEVEL_SCHEMA,
        "producer": _DECISION_LEVEL_PRODUCER,
        "status": "frozen" if admission_passed and complete else "blocked",
        "decision_unit": list(rule["selection_unit"]),
        "provenance": {
            "formal_p0_report": {
                "path": str(canonical_report),
                "sha256": _sha256(canonical_report),
            },
            "protocol": {
                "path": str(canonical_protocol),
                "sha256": _sha256(canonical_protocol),
            },
            "source_commit": protocol["source_commit"],
            "decision_rule_sha256": protocol_rule["canonical_sha256"],
            "before_unblinding_amendment": protocol_rule["before_unblinding_amendment"],
        },
        "source_manifest": {
            "path": str(Path(str(manifest_binding["path"])).resolve()),
            "sha256": manifest_binding["sha256"],
        },
        "entries": entries,
        "decisions": decisions,
        "admission": {
            "minimum_families_with_selected_level": admission_rule[
                "minimum_families_with_selected_level"
            ],
            "minimum_admitted_defects": admission_rule["minimum_admitted_defects"],
            "defects": defect_rows,
            "admitted_defects": admitted_defects,
            "passed": admission_passed,
            "complete_36_cell_selection": complete,
        },
    }


def write_decision_level_result(
    report_path: Path,
    protocol_path: Path,
    decision_rule_path: Path,
    output: Path,
    bundle: ConfigBundle | None = None,
) -> dict[str, Any]:
    result = build_decision_level_result(report_path, protocol_path, decision_rule_path, bundle)
    if (
        result["provenance"]["source_commit"] == NO_CHECKOUT
        or source_commit() != result["provenance"]["source_commit"]
        or _worktree_dirty() is not False
    ):
        raise ValueError(
            "decision-level publication requires a clean checkout at the formal "
            "P0 protocol's source commit"
        )
    _write_json_immutable(output, result)
    return result


def load_decision_level_result(path: Path, bundle: ConfigBundle | None = None) -> dict[str, Any]:
    artifact_path = Path(path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError(f"{artifact_path}: decision-level result must be an object")
    provenance = artifact.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{artifact_path}: decision-level result lacks provenance")
    report = provenance.get("formal_p0_report")
    protocol = provenance.get("protocol")
    if not isinstance(report, dict) or not isinstance(protocol, dict):
        raise ValueError(f"{artifact_path}: decision-level source bindings are malformed")
    protocol_document = _load_goldilocks_protocol(Path(str(protocol.get("path"))), bundle)
    decision_rule_path = Path(str(protocol_document["decision_rule"]["path"]))
    expected = build_decision_level_result(
        Path(str(report.get("path"))),
        Path(str(protocol.get("path"))),
        decision_rule_path,
        bundle,
    )
    if artifact != expected:
        raise ValueError(
            f"{artifact_path}: decision-level result is stale, tampered, or not the "
            "deterministic reduction of its bound sources"
        )
    return artifact


def _default_job_manifest_path(protocol_path: Path, results_root: Path | None = None) -> Path:
    return (
        (results_root or _default_results_root()).resolve()
        / "calibration"
        / "p0"
        / _sha256(protocol_path)
        / "job-manifest.json"
    )


def _default_job_report_path(manifest: dict[str, Any]) -> Path:
    return (
        Path(str(manifest["results_root"]))
        / "calibration"
        / "p0"
        / str(manifest["protocol_sha256"])
        / "goldilocks-report.json"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan-goldilocks")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--decision-rule", type=Path, required=True)
    plan.add_argument("--preflight-summary", type=Path, required=True)

    plan_jobs = sub.add_parser("plan-jobs")
    plan_jobs.add_argument("--protocol", type=Path, required=True)
    plan_jobs.add_argument("--output", type=Path, default=None)
    plan_jobs.add_argument("--results-root", type=Path)
    plan_jobs.add_argument("--rollouts-per-job", type=int, default=_DEFAULT_ROLLOUTS_PER_JOB)

    status_jobs = sub.add_parser("status-jobs")
    status_jobs.add_argument("--manifest", type=Path, required=True)

    run_job = sub.add_parser("run-job")
    run_job.add_argument("--manifest", type=Path, required=True)
    run_job.add_argument("--job-id", required=True)

    assemble_manifest = sub.add_parser("assemble-manifest")
    assemble_manifest.add_argument("--manifest", type=Path, required=True)
    assemble_manifest.add_argument("--report", type=Path, default=None)

    decide = sub.add_parser("decide-levels")
    decide.add_argument("--report", type=Path, required=True)
    decide.add_argument("--protocol", type=Path, required=True)
    decide.add_argument("--decision-rule", type=Path, required=True)
    decide.add_argument("--output", type=Path, required=True)

    collect = sub.add_parser("collect-goldilocks")
    collect.add_argument("--task", required=True)
    collect.add_argument("--family", required=True)
    collect.add_argument("--defect", required=True)
    collect.add_argument("--level", type=int, required=True)
    collect.add_argument("--n-rollouts", type=int, required=True)
    collect.add_argument("--start-index", type=int, default=0)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--protocol", type=Path)

    assemble = sub.add_parser("assemble-goldilocks")
    assemble.add_argument("--input", type=Path, action="append", required=True)
    assemble.add_argument("--report", type=Path, required=True)
    assemble.add_argument("--formal", action="store_true")
    assemble.add_argument("--protocol", type=Path)
    assemble.add_argument("--manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "plan-goldilocks":
        protocol = write_goldilocks_protocol(
            args.output,
            args.decision_rule,
            args.preflight_summary,
        )
        n_cells = len(protocol["cells"])
        n_rollouts = n_cells * int(protocol["n_rollouts_per_cell"])
        print(
            f"{args.output}: {n_cells} cells, {n_rollouts} twin rollouts, "
            f"source_commit={protocol['source_commit']}"
        )
        return 0
    if args.command == "plan-jobs":
        output = args.output or _default_job_manifest_path(args.protocol, args.results_root)
        manifest = write_goldilocks_job_manifest(
            args.protocol,
            output,
            results_root=args.results_root,
            rollouts_per_job=args.rollouts_per_job,
        )
        print(
            f"{output}: {manifest['n_jobs']} jobs for {manifest['n_cells']} cells, "
            f"protocol_sha256={manifest['protocol_sha256']}"
        )
        return 0
    if args.command == "status-jobs":
        status = scan_goldilocks_jobs(args.manifest)
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0
    if args.command == "run-job":
        path = run_goldilocks_job(args.manifest, args.job_id)
        print(path)
        return 0
    if args.command == "assemble-manifest":
        manifest = _load_goldilocks_job_manifest(args.manifest)
        report = assemble_goldilocks_manifest(
            args.manifest,
            args.report or _default_job_report_path(manifest),
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "decide-levels":
        result = write_decision_level_result(
            args.report,
            args.protocol,
            args.decision_rule,
            args.output,
        )
        print(
            f"{args.output}: status={result['status']}, "
            f"selected={len(result['entries'])}/36, "
            f"admitted_defects={len(result['admission']['admitted_defects'])}"
        )
        return 0
    if args.command == "collect-goldilocks":
        path = collect_goldilocks_artifact(
            task=args.task,
            family=args.family,
            defect=args.defect,
            level=args.level,
            n_rollouts=args.n_rollouts,
            start_index=args.start_index,
            output=args.output,
            protocol_path=args.protocol,
        )
        print(path)
        return 0

    report = assemble_goldilocks(
        args.input,
        report_path=args.report,
        formal=args.formal,
        protocol_path=args.protocol,
        manifest_path=args.manifest,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
