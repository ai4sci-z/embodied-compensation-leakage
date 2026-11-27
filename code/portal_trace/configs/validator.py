"""The pre-execution checker: nothing runs until the configuration is complete.

Before any stage of the pipeline starts, `validate` establishes six things and
refuses otherwise:

1. all 16 preregistration items are present (11.3), and the ones this
   experiment names are among them;
2. the arm's design (M, D, S) is declared and agrees with the cells it runs
   (11.3 item 14, 11.4's model card);
3. the three seed layers are allocated explicitly, with the environment and
   evaluation layers arm-independent and the training layer arm-scoped (11.3);
4. every lock-file measurement this stage needs is present, raising
   `MeasurementMissingError` naming the script that produces it (CONTRACT 3.1);
5. the requested backend is available (9.1.2) -- and if it is not, that is
   fatal, because mixing backends inside one endpoint is forbidden;
6. when the selected experiment is one of 10.6.1's fourteen ablations, that the
   arm is one of its levels, and -- for A8 -- which contact solver group the
   environments will be built with.

For a training stage it additionally consults the unblinding ledger, so a
training arm cannot produce a number before the three zero-GPU gates and E12
have produced theirs.

The checker collects problems and reports them together, except for a missing
measurement, which is re-raised as-is: `MeasurementMissingError`'s message
already names the producing script and the blocked work, and wrapping it would
bury that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from portal_trace.configs.loader import ConfigBundle, load_bundle
from portal_trace.configs.preregistration import (
    PreregistrationError,
    PreregistrationNotTagged,
)
from portal_trace.configs.schema import ConfigError
from portal_trace.env.versions import Backend, BackendUnavailable
from portal_trace.lockfile import (
    REQUIRED_MEASUREMENTS,
    LockFile,
    MeasurementMissingError,
)

__all__ = [
    "PreflightError",
    "RunRequest",
    "PreflightReport",
    "STAGES",
    "validate",
    "outstanding_measurements",
    # Re-exported so that a caller catching preflight failures has all three in
    # one place: an incomplete config, an absent measurement and an unavailable
    # backend are the three ways a run is refused.
    "MeasurementMissingError",
    "BackendUnavailable",
]

# The pipeline stages `runners/` exposes.
STAGES: tuple[str, ...] = (
    "collect", "materialise", "train", "evaluate", "diagnose",
)


class PreflightError(RuntimeError):
    """The run is not allowed to start.

    Deliberately not a warning. A run that starts with an incomplete
    preregistration produces numbers that 11.3 bars from the main narrative, and
    that cannot be fixed after the fact.
    """


@dataclass(frozen=True)
class RunRequest:
    """What is about to be run.

    The scope fields (`task`, `family`, ...) are the ones lock-file keys are
    parameterised by; the validator resolves each declared measurement's scope
    from them and refuses when one is missing rather than substituting a
    wildcard.
    """

    experiment_id: str
    stage: str
    backend: Backend
    arm_id: str | None = None
    task: str | None = None
    family: str | None = None
    defect: str | None = None
    level: str | None = None
    geom_variant: str = "G0"
    joint: str | None = None
    arch: str | None = None
    modality: str | None = None
    seed_index: int | None = None
    results_root: Path | None = None
    repo_root: Path | None = None
    require_prereg_tag: bool = True

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ConfigError(f"unknown stage {self.stage!r}; known: {STAGES}")

    def scope_values(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "family": self.family,
            "defect": self.defect,
            "level": self.level,
            "geom_variant": self.geom_variant,
            "joint": self.joint,
            "arch": self.arch,
            "modality": self.modality,
        }


@dataclass
class PreflightReport:
    """What the checker established. Returned only when nothing failed."""

    request: RunRequest
    experiment: str
    arm: str | None
    design: tuple[int, int, int] | None
    n_cells: int | None
    seeds_checked: tuple[str, ...]
    measurements_checked: tuple[str, ...]
    backend: Backend
    prereg_version: str
    prereg_tag_commit: str | None
    amendments_after_unblinding: int
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"preflight OK: {self.experiment}"
            + (f" / {self.arm}" if self.arm else "")
            + f" / {self.request.stage} on {self.backend.value}",
            f"  preregistration {self.prereg_version}"
            + (f" @ {self.prereg_tag_commit[:12]}" if self.prereg_tag_commit else " (untagged)"),
            f"  design {self.design}   cells {self.n_cells}",
            f"  seed layers {list(self.seeds_checked)}",
            f"  measurements checked: {len(self.measurements_checked)}",
        ]
        if self.amendments_after_unblinding:
            lines.append(
                f"  NOTE {self.amendments_after_unblinding} amendment(s) recorded "
                "after unblinding; they travel with every result from this run"
            )
        lines.extend(f"  NOTE {n}" for n in self.notes)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------


def _check_preregistration(
    bundle: ConfigBundle, request: RunRequest, problems: list[str]
) -> str | None:
    """Item completeness and the git tag. Returns the tagged commit, if any."""
    prereg = bundle.preregistration
    try:
        prereg.check_complete()
    except PreregistrationError as exc:
        problems.append(str(exc))
        return None

    exp = bundle.experiment(request.experiment_id)
    try:
        prereg.require_items(exp.prereg_items)
    except PreregistrationError as exc:
        problems.append(
            f"{exp.experiment_id} depends on preregistration items "
            f"{list(exp.prereg_items)}: {exc}"
        )

    if not request.require_prereg_tag:
        return None
    repo_root = request.repo_root or Path(__file__).resolve().parents[2]
    try:
        return prereg.require_tag(repo_root)
    except PreregistrationNotTagged as exc:
        # 11.3 ties the training-arm unblinding to the tag. Stages that produce
        # no training-arm number are not blocked by it, but the absence is
        # recorded either way.
        if request.stage == "train":
            problems.append(str(exc))
        return None
    except PreregistrationError as exc:
        problems.append(str(exc))
        return None


def _check_design(
    bundle: ConfigBundle, request: RunRequest, problems: list[str]
) -> tuple[tuple[int, int, int] | None, int | None]:
    if request.arm_id is None:
        return None, None
    arm = bundle.arm(request.arm_id)
    if arm.design is None:
        if arm.ablation is not None:
            # The generic message would send a reader looking for a design that
            # is deliberately absent. 10.6.1 budgets an ablation's runs on the
            # row, not on the arm, so the missing piece is upstream of this file.
            problems.append(
                f"arm {arm.arm_id} belongs to ablation {arm.ablation}, whose runs are "
                f"budgeted on the row: 10.6.1 gives {arm.ablation} "
                f"{bundle.ablation(arm.ablation).n_runs} runs over the six S6 cells at "
                "one seed. Before it can be trained, seeds.yaml has to allocate its "
                "training-seed indices and arms.yaml has to carry the matching design "
                "(M, D, S); 11.3 requires the three seed layers to be allocated "
                "explicitly, so the arm may not borrow the parent arm's index"
            )
            return None, None
        problems.append(
            f"arm {arm.arm_id}: no design (M, D, S) declared. 11.4's model card "
            "makes it mandatory on every summary row, because the row's MDE and its "
            "inconclusive rule are derived from it"
        )
        return None, None
    cells = bundle.cells(arm.arm_id)
    return arm.design.as_tuple(), len(cells)


def _check_seeds(
    bundle: ConfigBundle, request: RunRequest, problems: list[str]
) -> tuple[str, ...]:
    seeds = bundle.seeds
    checked: list[str] = []

    if request.task and request.family:
        seeds.env_seed(task=request.task, family=request.family, demo_index=0)
        checked.append("env")
        seeds.eval_seed(task=request.task, family=request.family)
        checked.append("eval")
    else:
        checked.extend(["env (scope not bound by this request)", "eval (idem)"])

    if request.arm_id is not None:
        allocated = seeds.train_seed_index_by_arm.get(request.arm_id)
        if allocated is None:
            problems.append(
                f"arm {request.arm_id}: seeds.yaml allocates no training seeds. 11.3 "
                "requires the three layers to be allocated explicitly in the config, "
                "never derived from one global seed"
            )
        elif request.seed_index is not None and request.seed_index not in allocated:
            problems.append(
                f"arm {request.arm_id}: seed index {request.seed_index} is not in the "
                f"allocated set {list(allocated)}"
            )
        else:
            checked.append("train")
    return tuple(checked)


def _resolve_scopes(
    requirement: Any, request: RunRequest, decl: Any
) -> list[dict[str, str]]:
    """Expand one requirement into the concrete lock-file scopes it needs."""
    available = {k: v for k, v in request.scope_values().items() if v is not None}
    available.update(requirement.fixed_scope)

    scopes: list[dict[str, str]] = [{}]
    for name in decl.scope_fields:
        if name in requirement.expand:
            scopes = [
                {**s, name: value} for s in scopes for value in requirement.expand[name]
            ]
        elif name in available:
            scopes = [{**s, name: str(available[name])} for s in scopes]
        else:
            raise PreflightError(
                f"measurement {decl.key_template!r} is scoped by {name!r}, and the "
                f"run request does not bind it. Pass --{name} or pin it in the "
                "config's `fixed_scope`; the validator does not substitute a "
                "wildcard, because a lock entry read under the wrong scope is a "
                "different measurement"
            )
    return scopes


def _check_measurements(
    bundle: ConfigBundle, request: RunRequest, lock: LockFile
) -> tuple[str, ...]:
    exp = bundle.experiment(request.experiment_id)
    checked: list[str] = []
    for requirement in exp.requires_measurements:
        if request.stage not in requirement.stages:
            continue
        decl = REQUIRED_MEASUREMENTS.get(requirement.key)
        if decl is None:
            raise PreflightError(
                f"{exp.experiment_id} requires measurement {requirement.key!r}, which "
                "is not declared in portal_trace.lockfile.REQUIRED_MEASUREMENTS. "
                "Every quantity the proposal marks as to-be-verified is declared "
                "there once and read only through LockFile.get"
            )
        for scope in _resolve_scopes(requirement, request, decl):
            # Propagates MeasurementMissingError unchanged: its message already
            # names the producing script and the work it blocks.
            lock.record(decl, **scope)
            checked.append(decl.key(**scope))
    return tuple(checked)


def _check_backend(request: RunRequest, problems: list[str]) -> None:
    from portal_trace.runners.backends import get_backend

    try:
        get_backend(request.backend).probe()
    except BackendUnavailable as exc:
        problems.append(str(exc))


def _check_unblinding(
    bundle: ConfigBundle, request: RunRequest, problems: list[str]
) -> None:
    if request.stage != "train" or request.arm_id is None:
        return
    from portal_trace.runners.unblinding import (
        UnblindingLedger,
        UnblindingOrderViolation,
    )

    if request.results_root is None:
        problems.append(
            "a training stage needs --results-root: the unblinding ledger of 11.3 "
            "lives next to the results, and without it the guard cannot tell whether "
            "the three zero-GPU gates and E12 have produced anything"
        )
        return
    ledger = UnblindingLedger.load(Path(request.results_root) / "unblinding.json")
    exp = bundle.experiment(request.experiment_id)
    is_primary = any(e.kind == "primary" for e in exp.endpoints)
    try:
        ledger.require_ready_for_training(
            arm_id=request.arm_id, is_primary_endpoint=is_primary
        )
    except UnblindingOrderViolation as exc:
        problems.append(str(exc))


def _check_ablation(
    bundle: ConfigBundle, request: RunRequest, problems: list[str], notes: list[str]
) -> None:
    """The checks that only apply when the selected experiment is a 10.6.1 row.

    An ablation reaches this function through the same `--experiment` selector
    as E1, which is the point: the fourteen rows are preregistered objects and
    are refused by the same preflight, rather than run through a side door that
    checks less.
    """
    from portal_trace.configs.schema import AblationSpec

    exp = bundle.experiment(request.experiment_id)
    if not isinstance(exp, AblationSpec):
        return
    if request.arm_id is not None:
        level_arms = exp.level_arms
        if request.arm_id not in level_arms:
            problems.append(
                f"{exp.experiment_id} is an ablation and its levels run "
                f"{sorted(set(level_arms))}; arm {request.arm_id} is not one of them. "
                "An ablation reading is a comparison between its own levels, so a run "
                "attributed to it has to be a level of it"
            )
            return
        level = next(x for x in exp.levels if x.arm == request.arm_id)
        if level.is_main:
            notes.append(
                f"{exp.experiment_id}.{level.level_id} is the main level: it is the "
                "setting the main experiments already ran, so it adds no runs and its "
                "numbers are read off the arm the paper already reports"
            )
    for code in exp.confounders_closed:
        notes.append(
            f"{exp.experiment_id} is the designated design for confounder {code} "
            "(10.6.2); a result from it is part of that confounder's discharge and "
            "not a sensitivity footnote"
        )
    if exp.experiment_id == "A8" and request.arm_id is not None:
        group = bundle.solver_group(request.arm_id)
        notes.append(
            f"contact solver group {group.group_id}: solref time constant x"
            f"{group.solref_timeconst_scale}, solimp gap to one x"
            f"{group.solimp_gap_scale}, applied to the compiled mjModel after every "
            "reset. 10.6.3 rule 3 makes the safety conclusion conditional on all "
            "three groups agreeing in sign"
        )


def _check_pool(bundle: ConfigBundle, request: RunRequest, notes: list[str]) -> None:
    if request.arm_id is None:
        return
    pool = bundle.arm(request.arm_id).pool
    # Not a failure -- both values are legal. It is annotated because 11.3 makes
    # the flag change how a null result is read, and the annotation has to reach
    # the model card.
    if pool.includes_delta_zero:
        notes.append(
            f"arm {request.arm_id}: the pool contains delta = 0, so route A's "
            "anchoring holds by construction; a leakage reading from this arm means "
            "the algorithm did not use an available anchor, not that information was "
            "missing (11.3 item 10)"
        )
    else:
        notes.append(
            f"arm {request.arm_id}: the pool has no delta = 0, so route A's anchoring "
            "does not hold by construction; a leakage reading is read as insufficient "
            "information (11.3 item 10)"
        )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def validate(
    request: RunRequest,
    *,
    bundle: ConfigBundle | None = None,
    lock: LockFile | None = None,
) -> PreflightReport:
    """Run every pre-execution check. Raise unless all of them pass."""
    b = bundle if bundle is not None else load_bundle()
    lk = lock if lock is not None else LockFile()
    problems: list[str] = []
    notes: list[str] = []

    tag_commit = _check_preregistration(b, request, problems)
    design, n_cells = _check_design(b, request, problems)
    seeds_checked = _check_seeds(b, request, problems)
    _check_backend(request, problems)
    _check_unblinding(b, request, problems)
    _check_ablation(b, request, problems, notes)
    _check_pool(b, request, notes)

    if problems:
        raise PreflightError(
            f"refusing to start {request.experiment_id}"
            + (f"/{request.arm_id}" if request.arm_id else "")
            + f" at stage {request.stage!r}:\n\n"
            + "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(problems))
        )

    # Last, and outside the collected list: MeasurementMissingError carries the
    # producing script in its own message and must reach the caller intact.
    measurements = _check_measurements(b, request, lk)

    return PreflightReport(
        request=request,
        experiment=request.experiment_id,
        arm=request.arm_id,
        design=design,
        n_cells=n_cells,
        seeds_checked=seeds_checked,
        measurements_checked=measurements,
        backend=request.backend,
        prereg_version=b.preregistration.version,
        prereg_tag_commit=tag_commit,
        amendments_after_unblinding=len(b.amendments.after_unblinding),
        notes=notes,
    )


def outstanding_measurements(
    experiment_id: str,
    stage: str,
    *,
    bundle: ConfigBundle | None = None,
    lock: LockFile | None = None,
) -> list[str]:
    """Report which declared measurements this stage would block on.

    A reporting helper for `portal-trace validate`; it does not read any value,
    so it cannot become a way of getting one without the lock file.
    """
    b = bundle if bundle is not None else load_bundle()
    lk = lock if lock is not None else LockFile()
    exp = b.experiment(experiment_id)
    absent = {d.key_template for d in lk.outstanding()}
    out = []
    for requirement in exp.requires_measurements:
        if stage not in requirement.stages:
            continue
        decl = REQUIRED_MEASUREMENTS[requirement.key]
        if decl.key_template in absent:
            out.append(f"{decl.key_template}  <- {decl.produced_by}")
    return out
