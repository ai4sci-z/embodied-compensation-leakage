"""Shared plumbing for the pipeline stages.

Two things live here. `RunContext` bundles what every stage needs -- the loaded
configuration, the lock file, the results root, the backend -- so that the
stages take one argument instead of six and so that the preflight report is
carried alongside rather than recomputed.

`require_module` is how this package depends on the modules other parts of the
repository own. Those imports are deferred to call time rather than done at
module import, because `portal_trace.runners.cli` is the console entry point and
has to be importable in a checkout where, say, `teacher/ilc.py` is not written
yet. When the module really is needed and really is absent, the failure names
the module, the section that specifies it and what it was needed for -- which is
more useful than a bare ImportError, and is not a fallback: nothing is
substituted.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from portal_trace.configs.loader import ConfigBundle, load_bundle
from portal_trace.configs.validator import PreflightReport, RunRequest, validate
from portal_trace.env.versions import Backend
from portal_trace.lockfile import LockFile

__all__ = [
    "MissingComponent",
    "LegacyDatasetLayoutError",
    "require_module",
    "RunContext",
    "open_context",
    "dataset_paths",
    "reject_legacy_dataset",
    "require_level_dataset",
]


class MissingComponent(ImportError):
    """A module this stage delegates to has not been written in this checkout."""


class LegacyDatasetLayoutError(FileNotFoundError):
    """A level-less dataset exists where a level-aware stage expected one."""


# Modules this package delegates to, with the section that specifies each and
# what it owns. Listing them here rather than at the import sites keeps the
# dependency surface visible in one place.
_COMPONENTS: dict[str, tuple[str, str]] = {
    "portal_trace.defects.inject": (
        "9.2.0",
        "one XML plus one delta compiled into (e_delta, e_0); the wrapper "
        "operations the backends drive",
    ),
    "portal_trace.defects.fields": (
        "9.2, 11.2",
        "the closed-form compensation field psi_j and the evaluator for c_delta",
    ),
    "portal_trace.defects.surrogate": (
        "9.2.11",
        "the surrogate defect generator and the four-dimensional matching table",
    ),
    "portal_trace.registry.twins": (
        "9.3.3",
        "the 36 structurally aligned twin pairs and the two FGW matrices",
    ),
    "portal_trace.registry.families": (
        "9.3",
        "the ten clean morphology families and their descriptors",
    ),
    "portal_trace.registry.geometry": (
        "9.5.5",
        "the G0 / G1 / G2 geometry variants",
    ),
    "portal_trace.teacher.ilc": (
        "9.4",
        "the ILC demonstrator: every ground truth in the plan comes from here",
    ),
    "portal_trace.teacher.query": (
        "9.4",
        "query_clean(x, t[, z]), the single entry point for per-frame ground truth",
    ),
    "portal_trace.teacher.replay": (
        "10.3.1",
        "the bidirectional cross replay, zero-GPU gate 2",
    ),
    "portal_trace.teacher.project": (
        "9.6.4",
        "the feedforward's projection onto phi and the phase-split residual "
        "report, zero-GPU gate 1",
    ),
    "portal_trace.diagnostics.leakage": (
        "10.1",
        "the Lambda family, Phi and rho_dir in the L^2(rho) sense",
    ),
    "portal_trace.diagnostics.regimes": (
        "9.7.1",
        "the five-regime partition and the short-history window labels",
    ),
    "portal_trace.diagnostics.design_graph": (
        "11.4",
        "the design graph, R_eff, lambda_2 and Xi_G",
    ),
    "portal_trace.diagnostics.two_way_fit": (
        "10.4.12",
        "the two-way additive fit; the only producer of theta_hat and gamma_hat",
    ),
    "portal_trace.policies.architecture": (
        "10.4.0",
        "the ResNet-18 vision tower plus 4-layer Transformer decoder head",
    ),
    "portal_trace.eval.endpoints": (
        "10.7",
        "endpoint computation and the three-way verdict rendering",
    ),
}


def require_module(name: str, *, needed_for: str) -> ModuleType:
    """Import a component module or raise with what it is and who owns it."""
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        section, owns = _COMPONENTS.get(name, ("?", "?"))
        raise MissingComponent(
            f"{needed_for} needs {name}, which is not importable in this checkout.\n"
            f"  proposal section: {section}\n"
            f"  that module owns: {owns}\n"
            f"  import error    : {exc}\n"
            "Nothing is substituted for it: a stand-in would silently change what "
            "the produced numbers mean."
        ) from exc


@dataclass(frozen=True)
class RunContext:
    """Everything a stage needs, validated."""

    request: RunRequest
    report: PreflightReport
    bundle: ConfigBundle
    lock: LockFile
    dataset_root: Path
    results_root: Path

    @property
    def backend(self) -> Backend:
        return self.request.backend

    def threshold(self, name: str) -> float:
        """A preregistered threshold, read from the tagged file."""
        return self.bundle.preregistration.threshold(name)


def dataset_paths(
    dataset_root: Path,
    *,
    task: str,
    family: str,
    defect: str,
    level: int,
    clean_of: str | None = None,
) -> dict[str, Path]:
    """The 9.6.3 directory tree.

    `raw/` holds states and actions with no images; `obs/` holds the materialised
    observations. Both paths include ``L<level>`` because E3 stores all five dose
    rungs for the same task, family and defect. Keeping raw and obs apart is
    9.1.3's first consequence: rendering
    costs eight times what physics does, and the same states are reused by
    several arms, so rendering once per arm would repeat the most expensive step
    O(arms) times.
    """
    if isinstance(level, bool) or not isinstance(level, int) or level < 0:
        raise ValueError(f"dataset level must be a non-negative integer, got {level!r}")
    if defect == "clean" and clean_of is not None:
        # A clean twin retains graph-changing structure (notably D4hi's shadow
        # joint), so one family-wide clean file would be overwritten by the
        # next defect's collection and silently mix model topologies.
        defect = f"clean-{clean_of}"
    level_dir = f"L{level}"
    return {
        "raw": dataset_root / "raw" / task / family / defect / level_dir / "demo.hdf5",
        "obs": (
            dataset_root
            / "obs"
            / task
            / family
            / defect
            / level_dir
            / "obs_84_ag_eih.hdf5"
        ),
        "norm": dataset_root / "norm",
        "splits": dataset_root / "splits",
        "registry": dataset_root / "registry",
    }


def _legacy_dataset_path(path: Path) -> Path:
    """The pre-level-layout location corresponding to a raw or obs path."""
    if not path.parent.name.startswith("L"):
        raise ValueError(f"{path}: expected a parent directory named L<level>")
    return path.parent.parent / path.name


def reject_legacy_dataset(path: Path) -> None:
    """Refuse an old level-less file instead of silently treating it as this level."""
    legacy = _legacy_dataset_path(path)
    if not path.exists() and legacy.exists():
        raise LegacyDatasetLayoutError(
            f"legacy level-less dataset {legacy} cannot be read as {path}. The old path "
            "does not identify a dose level and may already have been overwritten by a "
            "different E3 rung. Re-collect or explicitly migrate it only after verifying "
            "the stored defect_level metadata"
        )


def require_level_dataset(path: Path, *, produced_by: str) -> Path:
    """Require a level-aware file, with a specific refusal for the old layout."""
    reject_legacy_dataset(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; {produced_by}")
    return path


def open_context(
    request: RunRequest,
    *,
    dataset_root: Path | str,
    results_root: Path | str,
    bundle: ConfigBundle | None = None,
    lock: LockFile | None = None,
) -> RunContext:
    """Validate the request and build the context. Raises if the run may not start."""
    b = bundle if bundle is not None else load_bundle()
    lk = lock if lock is not None else LockFile()
    report = validate(request, bundle=b, lock=lk)
    return RunContext(
        request=request,
        report=report,
        bundle=b,
        lock=lk,
        dataset_root=Path(dataset_root),
        results_root=Path(results_root),
    )


def summarise(value: Any) -> str:
    """Compact repr for log lines, so a stage's log stays one line per unit."""
    if isinstance(value, (list, tuple)):
        return f"[{len(value)} items]"
    return str(value)
