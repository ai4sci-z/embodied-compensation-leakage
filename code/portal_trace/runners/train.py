"""Training a single arm (proposal 10.4.0, 10.5, 11.3).

Three things happen before any gradient step, and all three can refuse.

`configs.validator` establishes that the preregistration is complete, that the
arm's design (M, D, S) is declared and matches its cells, that all three seed
layers are allocated, and that the lock-file measurements this stage needs are
present.

The unblinding guard of 11.3 establishes that the three zero-GPU gates and E12
have produced results. Those gates each decide whether a whole strand of the
argument stands, and none of them costs GPU time, so scheduling a training arm
in front of them is not allowed.

The pool is then resolved from the arm's preregistered composition table, which
is where the `delta = 0` membership flag turns into an actual file list. That
flag is carried into the model card rather than merely consulted, because 11.3
makes it change the reading of a null result.

The formal training loop reuses the registry-policy producer in
`runners.formal_training`; this module owns stage admission, pool/plan binding,
budget selection, and the hash-bound model card.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from portal_trace.cards import ModelCard
from portal_trace.configs.schema import ArmSpec, ConfigError
from portal_trace.registry.tasks import CORE_TASKS
from portal_trace.runners.common import (
    RunContext,
    dataset_paths,
    reject_legacy_dataset,
    require_module,
)
from portal_trace.runners.dataset_contract import (
    require_dataset_identity,
    require_delta_value_contract,
)

__all__ = [
    "LEVEL_MANIFEST_SCHEMA",
    "TRAINING_TASK_SET_SHA256",
    "FrozenLevelManifest",
    "PoolEntry",
    "ResolvedPool",
    "TrainingJob",
    "load_level_manifest",
    "source_cells",
    "resolve_pool",
    "train",
]


LEVEL_MANIFEST_SCHEMA = "portal_trace.calibration.decision_levels/1"
TRAINING_TASK_SET_SHA256 = hashlib.sha256(
    json.dumps(CORE_TASKS, separators=(",", ":")).encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class FrozenLevelManifest:
    """A frozen P0 decision-level table, never a level-selection procedure."""

    path: Path
    sha256: str
    levels: Mapping[tuple[str, str], int]
    provenance: Mapping[str, Any]

    def level_for(self, family: str, defect: str) -> int:
        try:
            return self.levels[(family, defect)]
        except KeyError as exc:
            raise ConfigError(
                f"{self.path}: frozen decision-level manifest has no entry for "
                f"({family}, {defect}); refusing to borrow the target or another "
                "cell's level"
            ) from exc


def load_level_manifest(path: Path, *, bundle: Any) -> FrozenLevelManifest:
    """Load the complete 36-cell P0 output used by non-E3 training.

    This validates and fingerprints a decision table; it deliberately contains
    no cross-task selection or tie-breaking rule. Until a separately produced
    frozen table exists, non-E3 training remains blocked.
    """
    path = Path(path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise ConfigError(
            f"frozen decision-level manifest {path} does not exist. Non-E3 training "
            "is blocked until P0 has produced the complete 36-cell table under an "
            "approved cross-task selection and tie-breaking rule"
        ) from exc
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{path}: decision-level manifest is not valid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise ConfigError(f"{path}: decision-level manifest must be a JSON object")
    if document.get("schema") != LEVEL_MANIFEST_SCHEMA:
        raise ConfigError(
            f"{path}: decision-level manifest schema must be {LEVEL_MANIFEST_SCHEMA!r}"
        )
    if document.get("status") != "frozen":
        raise ConfigError(
            f"{path}: decision-level manifest status must be 'frozen'; draft or "
            "unlabelled P0 output cannot authorize main training"
        )
    if document.get("decision_unit") != ["family", "defect"]:
        raise ConfigError(
            f"{path}: decision_unit must be ['family', 'defect']; one level per defect "
            "or per family is not the preregistered S36 statistical unit"
        )
    provenance = _validate_level_manifest_provenance(
        document.get("provenance"),
        manifest_path=path,
        bundle=bundle,
        source_manifest=document.get("source_manifest"),
    )
    rows = document.get("entries")
    if not isinstance(rows, list):
        raise ConfigError(f"{path}: entries must be a list")

    expected = set(bundle.subsets["S36"].cells())
    levels: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        where = f"{path}:entries[{index}]"
        if not isinstance(row, dict):
            raise ConfigError(f"{where} must be an object")
        family, defect, level = row.get("family"), row.get("defect"), row.get("level")
        cell = (family, defect)
        if not isinstance(family, str) or not isinstance(defect, str):
            raise ConfigError(f"{where}: family and defect must be strings")
        if isinstance(level, bool) or not isinstance(level, int) or level not in range(5):
            raise ConfigError(f"{where}: level must be one of 0, 1, 2, 3, 4")
        if cell in levels:
            raise ConfigError(f"{where}: duplicate decision-level entry for {cell}")
        levels[cell] = level

    missing = sorted(expected - set(levels))
    unexpected = sorted(set(levels) - expected)
    if missing or unexpected:
        raise ConfigError(
            f"{path}: frozen decision-level manifest must contain exactly the 36 S36 "
            f"cells; missing={missing}, unexpected={unexpected}"
        )
    canonical = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return FrozenLevelManifest(
        path=path.resolve(),
        sha256=hashlib.sha256(canonical).hexdigest(),
        levels=MappingProxyType(levels),
        provenance=MappingProxyType(provenance),
    )


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ConfigError(f"{where} must be a lowercase SHA-256 string")
    return value


def _bound_path(value: Any, *, manifest_path: Path, where: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where} must be a non-empty path string")
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise ConfigError(f"{where} does not exist: {path}")
    return path


def _load_formal_p0_protocol(path: Path, bundle: Any) -> dict[str, Any]:
    """Delegate the protocol's full decision-rule and amendment checks to P0."""
    from portal_trace.scripts.calibrate_p0 import _load_goldilocks_protocol

    return _load_goldilocks_protocol(path, bundle)


def _validate_level_manifest_provenance(
    value: Any, *, manifest_path: Path, bundle: Any, source_manifest: Any
) -> dict[str, Any]:
    """Verify the frozen table's complete formal-P0 provenance chain."""
    if not isinstance(value, dict):
        raise ConfigError(
            f"{manifest_path}: provenance must bind a formal P0 report and protocol"
        )
    expected = {
        "formal_p0_report",
        "protocol",
        "source_commit",
        "decision_rule_sha256",
        "before_unblinding_amendment",
    }
    if set(value) != expected:
        raise ConfigError(
            f"{manifest_path}: provenance fields mismatch; "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )

    def binding(name: str) -> tuple[Path, str]:
        item = value[name]
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ConfigError(
                f"{manifest_path}: provenance.{name} must contain only path and sha256"
            )
        bound_path = _bound_path(
            item["path"], manifest_path=manifest_path, where=f"provenance.{name}.path"
        )
        bound_sha = _require_sha256(
            item["sha256"], where=f"{manifest_path}: provenance.{name}.sha256"
        )
        actual_sha = _file_sha256(bound_path)
        if actual_sha != bound_sha:
            raise ConfigError(
                f"{manifest_path}: provenance.{name} SHA-256 is stale or tampered; "
                f"bound={bound_sha}, actual={actual_sha}"
            )
        return bound_path, bound_sha

    report_path, report_sha = binding("formal_p0_report")
    protocol_path, protocol_sha = binding("protocol")
    source_commit = value["source_commit"]
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(char not in "0123456789abcdef" for char in source_commit)
    ):
        raise ConfigError(
            f"{manifest_path}: provenance.source_commit must be a full lowercase Git SHA"
        )
    decision_sha = _require_sha256(
        value["decision_rule_sha256"],
        where=f"{manifest_path}: provenance.decision_rule_sha256",
    )
    amendment = value["before_unblinding_amendment"]
    if not isinstance(amendment, dict) or set(amendment) != {"index", "sha256"}:
        raise ConfigError(
            f"{manifest_path}: before_unblinding_amendment must contain index and sha256"
        )
    if isinstance(amendment["index"], bool) or not isinstance(amendment["index"], int):
        raise ConfigError(f"{manifest_path}: amendment index must be an integer")
    amendment_sha = _require_sha256(
        amendment["sha256"], where=f"{manifest_path}: amendment sha256"
    )

    protocol = _load_formal_p0_protocol(protocol_path, bundle)
    protocol_decision = protocol.get("decision_rule")
    if not isinstance(protocol_decision, dict):
        raise ConfigError(f"{protocol_path}: validated protocol has no decision_rule")
    if (
        protocol.get("schema_version") != 2
        or protocol.get("source_commit") != source_commit
        or protocol.get("worktree_dirty") is not False
        or protocol_decision.get("canonical_sha256") != decision_sha
        or protocol_decision.get("before_unblinding_amendment") != amendment
    ):
        raise ConfigError(
            f"{manifest_path}: protocol source commit, decision-rule SHA, or amendment "
            "binding disagrees with the frozen level manifest"
        )

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{report_path}: formal P0 report is not valid JSON") from exc
    report_protocol = report.get("protocol") if isinstance(report, dict) else None
    report_manifest = report.get("manifest") if isinstance(report, dict) else None
    if not isinstance(report_manifest, dict) or set(report_manifest) != {"path", "sha256"}:
        raise ConfigError(
            f"{manifest_path}: bound formal P0 report lacks an exact manifest path/SHA binding"
        )
    report_manifest_path = _bound_path(
        report_manifest["path"],
        manifest_path=report_path,
        where=f"{report_path}: manifest.path",
    )
    report_manifest_sha = _require_sha256(
        report_manifest["sha256"], where=f"{report_path}: manifest.sha256"
    )
    if _file_sha256(report_manifest_path) != report_manifest_sha:
        raise ConfigError(
            f"{manifest_path}: report manifest SHA-256 is stale or tampered; "
            f"bound={report_manifest_sha}, actual={_file_sha256(report_manifest_path)}"
        )
    source_manifest_path = report_manifest_path
    source_manifest_sha = report_manifest_sha
    if not isinstance(source_manifest, dict) or set(source_manifest) != {"path", "sha256"}:
        raise ConfigError(
            f"{manifest_path}: source_manifest must contain exactly path and sha256"
        )
    source_manifest_path = _bound_path(
        source_manifest["path"],
        manifest_path=manifest_path,
        where=f"{manifest_path}: source_manifest.path",
    )
    source_manifest_sha = _require_sha256(
        source_manifest["sha256"], where=f"{manifest_path}: source_manifest.sha256"
    )
    actual_source_manifest_sha = _file_sha256(source_manifest_path)
    if actual_source_manifest_sha != source_manifest_sha:
        raise ConfigError(
            f"{manifest_path}: source_manifest SHA-256 is stale or tampered; "
            f"bound={source_manifest_sha}, actual={actual_source_manifest_sha}"
        )
    if (
        source_manifest_path != report_manifest_path
        or source_manifest_sha != report_manifest_sha
    ):
        raise ConfigError(
            f"{manifest_path}: source_manifest disagrees with the bound report manifest"
        )
    if (
        not isinstance(report_protocol, dict)
        or report.get("schema_version") != 1
        or report.get("producer")
        != "portal_trace.scripts.calibrate_p0 assemble-goldilocks"
        or report.get("formal") is not True
        or report.get("source_commit") != source_commit
        or report_protocol.get("sha256") != protocol_sha
        or report_protocol.get("decision_rule_sha256") != decision_sha
    ):
        raise ConfigError(
            f"{manifest_path}: bound P0 report is not a formal assembly of the bound "
            "protocol, source commit, and decision rule"
        )
    return {
        "formal_p0_report_path": str(report_path),
        "formal_p0_report_sha256": report_sha,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "source_commit": source_commit,
        "decision_rule_sha256": decision_sha,
        "before_unblinding_amendment": {
            "index": amendment["index"],
            "sha256": amendment_sha,
        },
    }


@dataclass(frozen=True)
class PoolEntry:
    """One (embodiment, delta) contribution to a training pool."""

    task: str
    family: str
    defect: str | None      # None is the clean twin, i.e. delta = 0
    level: int
    level_owner: tuple[str, str]
    n_demos: int
    is_target: bool
    obs_file: Path

    def __post_init__(self) -> None:
        if self.task not in CORE_TASKS:
            raise ValueError(f"pool entry task {self.task!r} is not a core task")
        if isinstance(self.level, bool) or not isinstance(self.level, int) or self.level < 0:
            raise ValueError(f"pool entry level must be a non-negative integer, got {self.level!r}")
        if len(self.level_owner) != 2 or not all(
            isinstance(value, str) and value for value in self.level_owner
        ):
            raise ValueError(f"invalid pool-entry level owner {self.level_owner!r}")

    @property
    def is_clean_twin(self) -> bool:
        return self.defect is None


@dataclass(frozen=True)
class ResolvedPool:
    """The arm's pool, with the two flags 11.3 item 10 makes mandatory."""

    arm_id: str
    target_family: str
    target_defect: str
    target_level: int
    tasks: tuple[str, ...]
    task_set_sha256: str
    level_mode: str
    level_manifest_path: Path | None
    level_manifest_sha256: str | None
    level_provenance: Mapping[str, Any] | None
    entries: tuple[PoolEntry, ...]
    includes_delta_zero: bool
    target_in_pool: bool
    reading_consequence: str

    def __post_init__(self) -> None:
        if self.tasks != CORE_TASKS:
            raise ConfigError(
                f"arm {self.arm_id}: training tasks must be the ordered CORE_TASKS "
                f"{list(CORE_TASKS)}, got {list(self.tasks)}"
            )
        if self.task_set_sha256 != TRAINING_TASK_SET_SHA256:
            raise ConfigError(f"arm {self.arm_id}: training task-set hash is inconsistent")
        actual_tasks = {entry.task for entry in self.entries}
        if actual_tasks != set(self.tasks):
            raise ConfigError(
                f"arm {self.arm_id}: resolved entries cover tasks {sorted(actual_tasks)}, "
                f"expected {list(self.tasks)}"
            )
        # The declared flags and the resolved file list have to agree, or the
        # model card describes a pool that was not trained on.
        actual_zero = any(e.is_clean_twin for e in self.entries)
        if actual_zero != self.includes_delta_zero:
            raise ConfigError(
                f"arm {self.arm_id}: the pool table declares includes_delta_zero="
                f"{self.includes_delta_zero} but the resolved pool "
                f"{'contains' if actual_zero else 'does not contain'} a clean twin. "
                "11.3 item 10 makes that flag change the reading of a null result in "
                "E7 and E8, so a mismatch is not a bookkeeping detail"
            )
        actual_target = any(e.is_target for e in self.entries)
        if actual_target != self.target_in_pool:
            raise ConfigError(
                f"arm {self.arm_id}: the pool table declares target_in_pool="
                f"{self.target_in_pool}, the resolved pool disagrees"
            )

    @property
    def n_demos(self) -> int:
        return sum(e.n_demos for e in self.entries)

    @property
    def n_demos_per_task(self) -> dict[str, int]:
        return {
            task: sum(entry.n_demos for entry in self.entries if entry.task == task)
            for task in self.tasks
        }

    @property
    def alpha_demo(self) -> float:
        """Source share by demonstration count.

        Equal to the frame-level share bit for bit, because 9.6.1 fixes the
        horizon at 400 with early stopping off. The data card stores both and a
        difference between them is a bug, not a rounding artefact.
        """
        total = self.n_demos
        if total == 0:
            raise ConfigError(f"arm {self.arm_id}: empty pool")
        return sum(e.n_demos for e in self.entries if not e.is_target) / total


@dataclass
class TrainingJob:
    """One (arm, seed index) run."""

    arm: ArmSpec
    seed_index: int
    train_seed: int
    tasks: tuple[str, ...]
    task_set_sha256: str
    requested_level: int | None
    level_manifest_path: Path | None
    level_manifest_sha256: str | None
    pool: ResolvedPool
    output_dir: Path
    architecture: str
    model_card: ModelCard | None = None
    formal_plan_path: Path | None = None
    normalization_path: Path | None = None
    checkpoint_path: Path | None = None


def resolve_pool(
    ctx: RunContext,
    *,
    arm_id: str,
    target_family: str,
    target_defect: str,
    requested_level: int | None = None,
    level_manifest: FrozenLevelManifest | None = None,
) -> ResolvedPool:
    """Resolve E3 at one requested dose or other arms from a frozen S36 table."""
    arm = ctx.bundle.arm(arm_id)
    is_e3_dose = arm_id == "E3_dose"
    if is_e3_dose:
        if requested_level is None:
            raise ConfigError("E3_dose requires an explicit requested dose level")
        if level_manifest is not None:
            raise ConfigError("E3_dose uses --level, not a frozen main-training manifest")
        if isinstance(requested_level, bool) or requested_level not in range(5):
            raise ConfigError("E3_dose requested level must be one of 0, 1, 2, 3, 4")
    else:
        if requested_level is not None:
            raise ConfigError(
                f"arm {arm_id} cannot apply one --level to every pool cell; pass the "
                "complete frozen --level-manifest produced after P0 instead"
            )
        if level_manifest is None:
            raise ConfigError(
                f"arm {arm_id} requires a complete frozen per-(family, defect) level "
                "manifest; main training remains blocked until P0's cross-task "
                "selection and tie-breaking rule is approved and the table is frozen"
            )
    pool = arm.pool
    cells = ctx.bundle.cells(arm_id)
    if (target_family, target_defect) not in cells:
        raise ConfigError(
            f"arm {arm_id}: ({target_family}, {target_defect}) is not one of its "
            f"declared cells. 11.3 item 14 requires an arm that does not run all 36 "
            f"configurations to list the ones it does run; this arm's list is "
            f"{list(cells)}"
        )
    # An arm's subset identifies evaluation targets, not the universe from
    # which the standard 1+6 design-graph pool is drawn. In particular E3 has
    # four target cells in S6 but still uses the five other S36 families before
    # replacing its sixth cyclic source with the owning clean twin.
    source_universe = cells
    s36_cells = ctx.bundle.subsets["S36"].cells()
    if pool.k_sources == 6 and (target_family, target_defect) in s36_cells:
        source_universe = s36_cells
    source_cells = (
        ()
        if arm_id == "B7"
        else source_cells(
            source_universe,
            target_family=target_family,
            target_defect=target_defect,
            n_sources=pool.k_sources,
        )
    )
    entries: list[PoolEntry] = []

    def add(
        task: str,
        family: str,
        defect: str | None,
        *,
        level_owner: tuple[str, str],
        is_target: bool,
        n_demos: int,
    ) -> None:
        if is_e3_dose:
            assert requested_level is not None
            level = requested_level
        else:
            assert level_manifest is not None
            level = level_manifest.level_for(*level_owner)
        entries.append(
            PoolEntry(
                task=task,
                family=family,
                defect=defect,
                level=level,
                level_owner=level_owner,
                n_demos=n_demos,
                is_target=is_target,
                obs_file=dataset_paths(
                    ctx.dataset_root,
                    task=task,
                    family=family,
                    defect="clean" if defect is None else defect,
                    level=level,
                    clean_of=level_owner[1] if defect is None else None,
                )["obs"],
            )
        )
        reject_legacy_dataset(entries[-1].obs_file)

    for task in CORE_TASKS:
        if pool.target_in_pool:
            add(
                task,
                target_family,
                target_defect,
                level_owner=(target_family, target_defect),
                is_target=True,
                n_demos=pool.n_target_demos,
            )

        # B7 is one clean twin of the evaluated target embodiment, holding
        # 1,400 demonstrations per core task. It is not the defective target and
        # does not mirror B2's six source embodiments.
        if arm_id == "B7":
            add(
                task,
                target_family,
                None,
                level_owner=(target_family, target_defect),
                is_target=False,
                n_demos=pool.n_source_demos,
            )
        else:
            for index, (family, defect) in enumerate(source_cells):
                if arm_id == "E3_dose" and index == len(source_cells) - 1:
                    add(
                        task,
                        target_family,
                        None,
                        level_owner=(target_family, target_defect),
                        is_target=False,
                        n_demos=pool.n_source_demos,
                    )
                    continue
                add(
                    task,
                    family,
                    defect,
                    level_owner=(family, defect),
                    is_target=False,
                    n_demos=pool.n_source_demos,
                )
    target_level = (
        requested_level
        if is_e3_dose
        else level_manifest.level_for(target_family, target_defect)  # type: ignore[union-attr]
    )
    assert target_level is not None
    resolved = ResolvedPool(
        arm_id=arm_id,
        target_family=target_family,
        target_defect=target_defect,
        target_level=target_level,
        tasks=CORE_TASKS,
        task_set_sha256=TRAINING_TASK_SET_SHA256,
        level_mode="e3_requested_dose" if is_e3_dose else "frozen_per_cell_manifest",
        level_manifest_path=None if level_manifest is None else level_manifest.path,
        level_manifest_sha256=None if level_manifest is None else level_manifest.sha256,
        level_provenance=None if level_manifest is None else level_manifest.provenance,
        entries=tuple(entries),
        includes_delta_zero=pool.includes_delta_zero,
        target_in_pool=pool.target_in_pool,
        reading_consequence=pool.reading_consequence,
    )
    wrong_sizes = {
        task: count
        for task, count in resolved.n_demos_per_task.items()
        if count != pool.pool_size
    }
    if wrong_sizes:
        raise ConfigError(
            f"arm {arm_id}: resolved per-task demonstration counts {wrong_sizes} do "
            f"not match the declared pool size {pool.pool_size}"
        )
    if resolved.alpha_demo != pool.alpha_demo:
        raise ConfigError(
            f"arm {arm_id}: resolved alpha_demo={resolved.alpha_demo!r} does not "
            f"match the declared per-task alpha_demo={pool.alpha_demo!r}"
        )
    return resolved


def _require_pool_dataset_identities(pool: ResolvedPool) -> None:
    """Fail before training unless every weighted file proves its stored identity."""
    for entry in pool.entries:
        clean_of = entry.level_owner[1] if entry.is_clean_twin else None
        require_dataset_identity(
            entry.obs_file,
            task=entry.task,
            family=entry.family,
            defect="clean" if entry.is_clean_twin else entry.defect,
            level=entry.level,
            clean_of=clean_of,
            min_demos=entry.n_demos,
        )
        require_delta_value_contract(entry.obs_file)


def source_cells(
    cells: tuple[tuple[str, str], ...],
    *,
    target_family: str,
    target_defect: str,
    n_sources: int,
) -> tuple[tuple[str, str], ...]:
    """Resolve the six preregistered source edges for one target cell.

    Five sources carry the target defect on the other families. The sixth is the
    next defect in the subset's cyclic order on the target family. This is the
    concrete form of the design-graph description in ``subsets.yaml``; using all
    subset cells here would change a 1+6 pool into a 36-cell pool.
    """
    if n_sources == 0:
        return ()
    available = set(cells)
    target = (target_family, target_defect)
    if target not in available:
        raise ConfigError(f"target cell {target} is not in the arm's declared cells")
    families = tuple(dict.fromkeys(family for family, _ in cells))
    sources = [
        (family, target_defect)
        for family in families
        if family != target_family and (family, target_defect) in available
    ]
    defects = tuple(dict.fromkeys(defect for _, defect in cells))
    if target_defect in defects:
        start = defects.index(target_defect)
        for offset in range(1, len(defects) + 1):
            candidate = defects[(start + offset) % len(defects)]
            if (target_family, candidate) in available:
                sources.append((target_family, candidate))
                break
    if len(sources) != n_sources:
        raise ConfigError(
            f"target cell {target} has {len(sources)} resolvable cyclic sources, "
            f"but the pool declares k_sources={n_sources}; refusing to invent a "
            "source embodiment"
        )
    return tuple(sources)


# Kept for downstream callers written before the resolver became part of the
# public pool contract.
_source_cells = source_cells


def train(
    ctx: RunContext,
    *,
    seed_index: int,
    target_family: str,
    target_defect: str,
    requested_level: int | None = None,
    level_manifest_path: Path | None = None,
    formal_plan_path: Path | None = None,
    normalization_path: Path | None = None,
    epochs: int = 60,
    optimizer_steps: int | None = None,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    device: str = "cuda",
    dry_run: bool = False,
) -> TrainingJob:
    """Prepare and launch one training run.

    `ctx` has already been through the validator, which includes the unblinding
    guard for this stage; the guard is not re-run here, so there is one place
    where the order is enforced rather than two that can disagree.
    """
    req = ctx.request
    if req.arm_id is None:
        raise ConfigError("training needs --arm")
    arm = ctx.bundle.arm(req.arm_id)
    train_seed = ctx.bundle.seeds.train_seed(arm=arm.arm_id, seed_index=seed_index)
    level_manifest = (
        None
        if level_manifest_path is None
        else load_level_manifest(level_manifest_path, bundle=ctx.bundle)
    )
    pool = resolve_pool(
        ctx,
        arm_id=arm.arm_id,
        target_family=target_family,
        target_defect=target_defect,
        requested_level=requested_level,
        level_manifest=level_manifest,
    )
    if not dry_run:
        _require_pool_dataset_identities(pool)

    policies = require_module(
        "portal_trace.policies.architecture", needed_for="training configuration"
    )
    declared_architecture = arm.architecture or "resnet18_transformer4"
    architecture = policies.canonical_architecture(declared_architecture)
    level_identity = (
        f"dose-L{pool.target_level}"
        if pool.level_manifest_sha256 is None
        else f"levels-{pool.level_manifest_sha256[:16]}-target-L{pool.target_level}"
    )
    output_dir = (
        ctx.results_root
        / "train"
        / arm.arm_id
        / f"{target_family}-{target_defect}"
        / f"tasks-{pool.task_set_sha256[:16]}"
        / level_identity
        / f"seed{seed_index}"
    )

    job = TrainingJob(
        arm=arm,
        seed_index=seed_index,
        train_seed=train_seed,
        tasks=pool.tasks,
        task_set_sha256=pool.task_set_sha256,
        requested_level=requested_level,
        level_manifest_path=pool.level_manifest_path,
        level_manifest_sha256=pool.level_manifest_sha256,
        pool=pool,
        output_dir=output_dir,
        architecture=architecture,
        model_card=_model_card(ctx, arm=arm, pool=pool, seed_index=seed_index,
                              train_seed=train_seed, architecture=architecture,
                              declared_architecture=declared_architecture),
        formal_plan_path=(
            None if formal_plan_path is None else Path(formal_plan_path).resolve()
        ),
        normalization_path=(
            None if normalization_path is None else Path(normalization_path).resolve()
        ),
        checkpoint_path=output_dir / "checkpoint.pt",
    )
    if dry_run:
        return job

    if formal_plan_path is None or normalization_path is None:
        raise ConfigError(
            "formal training requires --formal-plan and --normalization; the current "
            "episode split and train-only statistics may not be inferred inside train"
        )
    _run_formal_policy_training(
        job,
        plan_path=Path(formal_plan_path),
        normalization_path=Path(normalization_path),
        epochs=epochs,
        optimizer_steps=optimizer_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )
    _write_model_card(job)
    return job


_FORMAL_ARM_NAMES = {
    "B2": "B2-naive-pooling",
    "B7": "B7-oracle",
    "B8": "B8-counterfactual-decompensation",
}


def _validate_plan_matches_pool(plan: Any, pool: ResolvedPool) -> None:
    """Bind the supplied current-schema episode plan to the resolved pool."""
    from portal_trace.runners.formal_data import FORMAL_DATA_PLAN_SCHEMA

    if plan.payload.get("schema") != FORMAL_DATA_PLAN_SCHEMA:
        raise ConfigError(
            f"formal train requires {FORMAL_DATA_PLAN_SCHEMA}, got "
            f"{plan.payload.get('schema')!r}"
        )
    expected = {str(entry.obs_file.resolve()): entry for entry in pool.entries}
    rows_by_path: dict[str, list[Mapping[str, Any]]] = {}
    for row in plan.payload["pool"]:
        rows_by_path.setdefault(str(Path(row["path"]).resolve()), []).append(row)
    if set(rows_by_path) != set(expected):
        raise ConfigError(
            "formal data plan paths do not exactly match the resolved training pool; "
            f"missing={sorted(set(expected) - set(rows_by_path))}, "
            f"unexpected={sorted(set(rows_by_path) - set(expected))}"
        )
    for path, entry in expected.items():
        rows = rows_by_path[path]
        demos = {str(row["demo"]) for row in rows}
        expected_demos = {f"demo_{index}" for index in range(entry.n_demos)}
        if demos != expected_demos:
            raise ConfigError(
                f"formal data plan episode membership for {path} does not equal the "
                f"resolved {entry.n_demos}-demo prefix"
            )
        if any(
            row.get("task") != entry.task
            or row.get("family") != entry.family
            or row.get("defect") != ("clean" if entry.is_clean_twin else entry.defect)
            or row.get("level") != entry.level
            or row.get("clean_of") != (
                entry.level_owner[1] if entry.is_clean_twin else None
            )
            for row in rows
        ):
            raise ConfigError(
                f"formal data plan identity for {path} disagrees with the resolved pool"
            )


def _run_formal_policy_training(
    job: TrainingJob,
    *,
    plan_path: Path,
    normalization_path: Path,
    epochs: int,
    optimizer_steps: int | None,
    batch_size: int,
    learning_rate: float,
    device: str,
) -> None:
    from portal_trace.policies.interface import PolicyConfig
    from portal_trace.policies.registry import build
    from portal_trace.runners.formal_checkpoint import _sha256
    from portal_trace.runners.formal_data import (
        load_formal_data_plan,
        load_formal_normalization,
    )
    from portal_trace.runners.formal_training import (
        FormalTrainingConfig,
        run_formal_release_training,
    )

    if job.arm.arm_id not in _FORMAL_ARM_NAMES:
        raise ConfigError(
            f"formal registry training currently admits only {sorted(_FORMAL_ARM_NAMES)}, "
            f"not {job.arm.arm_id!r}"
        )
    plan_path = plan_path.resolve()
    normalization_path = normalization_path.resolve()
    plan = load_formal_data_plan(plan_path)
    _validate_plan_matches_pool(plan, job.pool)
    normalization = load_formal_normalization(normalization_path, plan=plan)
    n_tasks = max(int(row["task_index"]) for row in plan.payload["pool"]) + 1
    n_embodiments = max(
        int(row["embodiment_index"]) for row in plan.payload["pool"]
    ) + 1
    policy_config = PolicyConfig(
        arm=_FORMAL_ARM_NAMES[job.arm.arm_id],
        n_tasks=n_tasks,
        n_pool_embodiments=n_embodiments,
    )
    policy = build(policy_config.arm, policy_config)
    parameter_report = policy.parameter_report()
    del policy
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ConfigError("formal training batch_size must be a positive integer")
    if optimizer_steps is None:
        optimizer_steps = (
            plan.payload["counts"]["train_windows"] + batch_size - 1
        ) // batch_size
    if optimizer_steps < 1:
        raise ConfigError("formal training optimizer_steps must be positive")
    if epochs < 60:
        raise ConfigError(
            "formal training epochs must be at least 60; the preregistered floor is "
            "part of the B2/B7/B8 comparison"
        )
    assert job.checkpoint_path is not None
    model_card_path = job.output_dir / "model_card.json"
    if model_card_path.exists():
        raise FileExistsError(f"refusing to overwrite model card {model_card_path}")
    config = FormalTrainingConfig(
        plan=plan,
        normalization=normalization,
        plan_path=plan_path,
        normalization_path=normalization_path,
        checkpoint_path=job.checkpoint_path,
        policy_config=policy_config,
        seed=job.train_seed,
        epochs=epochs,
        optimizer_steps=optimizer_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )
    result = run_formal_release_training(config)
    budget = job.model_card.values["architecture_and_budget"]  # type: ignore[union-attr]
    budget.update(
        {
            "parameter_count": parameter_report.total,
            "activated_parameter_count": parameter_report.active,
            "token_budget": result.optimizer_steps * batch_size,
            "training_steps": result.optimizer_steps,
            "epochs_per_config": result.epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": device,
        }
    )
    job.model_card.values["conditioning"].update(  # type: ignore[union-attr]
        {
            "history_window": policy_config.obs.k_hist,
            "embodiment_token": job.arm.arm_id not in {"B2", "B7", "B8"},
            "routing": "dense",
        }
    )
    job.model_card.values["endpoints"] = {  # type: ignore[union-attr]
        "status": "trained_pending_evaluation",
        "release_status": "formal",
        "label_transform": (
            "clip(actions - c_true, -1, 1)"
            if job.arm.arm_id == "B8"
            else "identity"
        ),
        "final_loss": result.final_loss,
        "heldout_action_mse": result.heldout_action_mse,
        "checkpoint": {
            "path": str(job.checkpoint_path.resolve()),
            "sha256": _sha256(job.checkpoint_path),
        },
        "formal_plan": {
            "path": str(plan_path),
            "sha256": _sha256(plan_path),
            "artifact_sha256": plan.plan_sha256,
        },
        "normalization": {
            "path": str(normalization_path),
            "sha256": _sha256(normalization_path),
            "artifact_sha256": normalization.artifact_sha256,
            "split": "train",
            "training_episode_ids": list(plan.train_episode_ids),
        },
    }


def _model_card(
    ctx: RunContext,
    *,
    arm: ArmSpec,
    pool: ResolvedPool,
    seed_index: int,
    train_seed: int,
    architecture: str,
    declared_architecture: str | None = None,
) -> ModelCard:
    """The 11.4 minimum field set for a model card.

    Missing any field bars the object from every preregistered endpoint, so the
    card is assembled before training rather than scraped from logs after.
    """
    if arm.design is None:
        raise ConfigError(f"arm {arm.arm_id}: no design; 11.4 makes (M, D, S) mandatory")
    pool_sources = [
        {
            "task": e.task,
            "family": e.family,
            "defect": e.defect if e.defect is not None else "clean(delta=0)",
            "level": e.level,
            "level_owner": list(e.level_owner),
            "n_demos": e.n_demos,
            "is_target": e.is_target,
        }
        for e in pool.entries
    ]
    values: dict[str, Any] = {
        "arm_identity": {
            "arm": arm.arm_id,
            "role": arm.role,
            "reference_arm": arm.reference_arm,
        },
        "subset_and_design": {
            "subset": arm.subset,
            "cells": [list(c) for c in ctx.bundle.cells(arm.arm_id)],
            "M": arm.design.n_families,
            "D": arm.design.n_defects,
            "S": arm.design.n_seeds,
        },
        "training_data_composition": {
            "tasks": list(pool.tasks),
            "task_set": {
                "schema": "portal_trace.training.core_tasks/1",
                "sha256": pool.task_set_sha256,
                "n_tasks": len(pool.tasks),
            },
            "n_demos_total": pool.n_demos,
            "n_demos_per_task": pool.n_demos_per_task,
            "pool_sources": pool_sources,
            "level_identity": {
                "mode": pool.level_mode,
                "requested_level": (
                    pool.target_level if pool.level_mode == "e3_requested_dose" else None
                ),
                "manifest_path": (
                    None if pool.level_manifest_path is None else str(pool.level_manifest_path)
                ),
                "manifest_sha256": pool.level_manifest_sha256,
                "provenance": (
                    None if pool.level_provenance is None else dict(pool.level_provenance)
                ),
            },
            "seed_env_scope": list(ctx.bundle.seeds.env_scope),
            "seed_eval_scope": list(ctx.bundle.seeds.eval_scope),
            "alpha_demo": pool.alpha_demo,
            "alpha_frame": pool.alpha_demo,
        },
        "pool_composition": {
            "pairs": [[e.family, e.defect] for e in pool.entries],
            "includes_delta_zero": pool.includes_delta_zero,
            "target_in_pool": pool.target_in_pool,
            "reading_consequence": pool.reading_consequence,
        },
        "design_graph_quantities": {
            "lambda_2": "pending_measurement",
            "R_eff": "pending_measurement",
            "Xi_G": "pending_measurement",
            "edge_weight_caliber": "pending_measurement",
        },
        "architecture_and_budget": {
            "architecture": architecture,
            "architecture_declaration": declared_architecture or architecture,
            "backend": ctx.backend.value,
            "parameter_count": 0,
            "activated_parameter_count": 0,
            "token_budget": 0,
            "training_steps": 0,
            "epochs_per_config": 60,
        },
        "isolation_variables": {
            "I": "pending_measurement",
            "routing_entropy": "pending_measurement",
            "paired_control": "pending_measurement",
        },
        "seed_train": {
            "seed_index": seed_index,
            "seed": train_seed,
        },
        "conditioning": {
            "embodiment_token": "pending_architecture",
            "latent_dim": 0,
            "history_window": 0,
            "routing": "pending_architecture",
        },
        "eval_embodiment_and_delta": {
            "embodiment": pool.target_family,
            "defect": pool.target_defect,
            "delta": pool.target_level,
            "level": pool.target_level,
        },
        "endpoints": {"status": "pending_training"},
        "discriminator_error_budget": {
            "w_ece": 0.0,
            "w_alpha_recovery": 0.0,
            "w_support_gap": 0.0,
            "domain_discriminator_auc": 0.5,
            "off_support_fraction": 0.0,
            "in_support_subset_reading": 0.0,
        },
        "phi_effective_rank": 150,
        "state_distribution": {"rho_off": "pending_evaluation"},
        "null_distribution": {
            "surrogate_defects": [],
            "effective_cardinality": 6,
            "quantiles": [],
        },
        "preregistration_pointer": {
            "file": "configs/preregistration/prereg.yaml",
            "version": ctx.bundle.preregistration.version,
            "git_tag": ctx.bundle.preregistration.git_tag,
            "verdict": "inconclusive",
            "amendments_after_unblinding": [
                a.summary for a in ctx.bundle.amendments.after_unblinding
            ],
        },
    }
    card = ModelCard(
        arm=arm.arm_id,
        values=values,
        notes=(
            "Prepared before training. Measurement-dependent fields are explicitly "
            "marked pending and must be replaced by bound evidence before any endpoint."
        ),
    )
    # This is a prepared card: measurements produced by training/evaluation are
    # intentionally absent at this stage.  ``_write_model_card`` runs the hard
    # completeness gate, so a pending card can be inspected by dry-run callers
    # but can never become a published artifact.
    return card


def _write_model_card(job: TrainingJob) -> None:
    if job.model_card is None:
        raise ConfigError("training job has no model card; refusing to write an ungoverned record")
    # Serialize before opening the destination. An incomplete card must not
    # leave behind a misleading empty file.
    document = job.model_card.to_json()
    canonical = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    document["artifact_sha256"] = hashlib.sha256(canonical).hexdigest()
    payload = json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True)
    path = job.output_dir / "model_card.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite model card {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
