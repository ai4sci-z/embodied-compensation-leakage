"""The `portal-trace` command line.

One subcommand per pipeline stage, plus `validate` (run the pre-execution
checks and report), `gates` (inspect and record the unblinding ledger of 11.3),
`config` (print what the declarative configuration resolves to) and `teacher`
(produce the one demonstrator that is an artefact rather than a computation --
A13's frozen RL expert).

Every stage subcommand goes through `configs.validator` first. The checks are
not a `--check` flag that can be omitted: an incomplete preregistration or a
missing measurement produces numbers that 11.3 bars from the main narrative, and
that cannot be repaired after the run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

from portal_trace.ablations.teacher_family import TeacherFamilyUnavailable
from portal_trace.configs.loader import load_bundle
from portal_trace.configs.schema import ConfigError
from portal_trace.configs.validator import (
    PreflightError,
    RunRequest,
    outstanding_measurements,
    validate,
)
from portal_trace.env.versions import Backend, BackendUnavailable
from portal_trace.lockfile import LockFile, MeasurementMissingError
from portal_trace.runners.common import MissingComponent, open_context
from portal_trace.runners.unblinding import (
    GATE_EVALUATORS,
    Gate,
    UnblindingLedger,
    UnblindingOrderViolation,
)

__all__ = ["build_parser", "main"]


def _common(p: argparse.ArgumentParser) -> None:
    # One selector for both, because 10.6.1's rows are preregistered objects and
    # go through the same preflight as a main experiment rather than through a
    # side door that checks less.
    p.add_argument("--experiment", required=True,
                   help="P0, P1, B, E1..E12, or an ablation A1..A14 (10.6.1)")
    p.add_argument("--backend", default=Backend.CPU_MUJOCO.value,
                   choices=[b.value for b in Backend])
    p.add_argument("--dataset-root", type=Path, default=Path("data"))
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument("--lock", type=Path, default=None, help="path to registry/lock.json")
    p.add_argument("--repo-root", type=Path, default=None)
    p.add_argument(
        "--allow-untagged-prereg",
        action="store_true",
        help=(
            "skip the git-tag check. Permitted for stages that produce no "
            "training-arm number; a training stage refuses regardless (11.3)"
        ),
    )


def _cell(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", required=True)
    p.add_argument("--family", required=True)
    p.add_argument("--defect", required=True)
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--geom-variant", default="G0")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="portal-trace",
        description=(
            "PORTAL-Trace pipeline. Every stage validates its configuration "
            "against the preregistration file before it starts (proposal 11.3)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="run the pre-execution checks and report")
    _common(p)
    p.add_argument("--stage", default="collect",
                   choices=["collect", "materialise", "train", "evaluate", "diagnose"])
    p.add_argument("--arm", default=None)
    p.add_argument("--task", default=None)
    p.add_argument("--family", default=None)
    p.add_argument("--defect", default=None)
    p.add_argument("--level", default=None)
    p.add_argument("--geom-variant", default="G0")

    p = sub.add_parser("config", help="print the resolved configuration")
    p.add_argument("--arm", default=None)
    p.add_argument("--experiment", default=None)

    p = sub.add_parser("collect", help="collect demonstrations for one cell")
    _common(p)
    _cell(p)
    p.add_argument("--n-demos", type=int, required=True)
    p.add_argument(
        "--overrun-step", type=float, required=True,
        help="measured/assumed tau_step ratio (9.8.4); no default, it gates branch 0",
    )
    p.add_argument("--overrun-frame", type=float, required=True,
                   help="measured/assumed tau_frame ratio (9.8.4)")
    p.add_argument(
        "--source-validation-reservation",
        type=Path,
        default=None,
        help="pre-collection reservation whose exact seeds and runtime this run must use",
    )

    p = sub.add_parser(
        "collect-b7-clean",
        help="collect B7's fixed 1,400-demo clean twin for one core task",
    )
    _common(p)
    _cell(p)
    p.add_argument(
        "--overrun-step", type=float, required=True,
        help="measured/assumed tau_step ratio (9.8.4); no default, it gates branch 0",
    )
    p.add_argument(
        "--overrun-frame", type=float, required=True,
        help="measured/assumed tau_frame ratio (9.8.4)",
    )

    # Not a pipeline stage, and deliberately not modelled as one: it writes no
    # dataset and no result, and the five stages are the objects 11.3's preflight
    # table is written against. It is the one ablation level whose demonstrator has
    # to be *produced* before the demonstrations can be, so it validates as the
    # stage that consumes it (`collect`) and refuses on exactly the same grounds.
    # Every other 10.6.1 row is reached by pointing an existing stage at it --
    # `--experiment A8` picks up its solver group inside `collect` -- and this one
    # would be too if there were a stage whose job were to make a teacher.
    p = sub.add_parser(
        "teacher",
        help="train A13's frozen RL expert on one defective cell (10.6.1)",
    )
    _common(p)
    _cell(p)
    p.add_argument(
        "--seed-index", type=int, default=0,
        help=(
            "index into the arm's training-seed allocation. A13 is designed at S=1, "
            "so seeds.yaml allocates exactly one and 0 is the only value it permits"
        ),
    )
    p.add_argument(
        "--artifact-root", type=Path, default=Path("."),
        help=(
            "directory the level block's checkpoint path is taken relative to; the "
            "template names `artifacts/...` from the repository root"
        ),
    )

    p = sub.add_parser("materialise", help="render observations from stored states")
    _common(p)
    _cell(p)
    p.add_argument("--check-only", action="store_true",
                   help="verify the stored proprio width without re-rendering")

    p = sub.add_parser(
        "materialise-b7-clean",
        help="render B7's clean-only raw pool without requiring a defective twin",
    )
    _common(p)
    _cell(p)
    p.add_argument("--check-only", action="store_true",
                   help="verify the stored proprio width without re-rendering")

    p = sub.add_parser("train", help="train one arm at one seed index")
    _common(p)
    p.add_argument("--arm", required=True)
    p.add_argument("--seed-index", type=int, required=True)
    p.add_argument("--target-family", required=True)
    p.add_argument("--target-defect", required=True)
    level_group = p.add_mutually_exclusive_group()
    level_group.add_argument(
        "--level",
        type=int,
        help="E3_dose's explicit requested dose level; never applied to main pools",
    )
    level_group.add_argument(
        "--level-manifest",
        type=Path,
        help="complete frozen per-(family, defect) P0 decision-level manifest",
    )
    p.add_argument(
        "--formal-plan",
        type=Path,
        help="current formal data plan whose episode split is used verbatim",
    )
    p.add_argument(
        "--normalization",
        type=Path,
        help="train-only normalization artifact bound to --formal-plan",
    )
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument(
        "--optimizer-steps",
        type=int,
        default=None,
        help="updates per epoch; default is one complete pass over train windows",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true")

    # This is an engineering probe, not a pipeline stage. It consumes explicit
    # hash-bound formal artifacts but never writes a formal result or lock entry.
    p = sub.add_parser(
        "adapter-probe",
        help="run a bounded non-formal policy adapter probe on a formal data plan",
    )
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--normalization", type=Path, required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--n-tasks", type=int, default=6)
    p.add_argument("--n-pool-embodiments", type=int, default=7)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--n-heads", type=int, default=2)
    p.add_argument("--ffn-dim", type=int, default=64)
    p.add_argument("--no-augment", action="store_true")

    p = sub.add_parser(
        "train-source-posterior",
        help="train P(source|observation) and publish one held-out cell posterior",
    )
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument(
        "--validation-plan",
        type=Path,
        default=None,
        help=(
            "separate prospective plan used only for final source-posterior "
            "validation; required unless --allow-engineering-output is set"
        ),
    )
    p.add_argument(
        "--validation-reservation",
        type=Path,
        default=None,
        help="pre-collection seed reservation required for formal validation",
    )
    p.add_argument("--normalization", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--posterior", type=Path, required=True)
    p.add_argument("--source-embodiment-indices", type=int, nargs="+", required=True)
    p.add_argument("--target-embodiment-indices", type=int, nargs="+", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--family", required=True)
    p.add_argument("--defect", required=True)
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--optimizer-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-tasks", type=int, default=6)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument(
        "--fusion-mode",
        choices=("modality-mean", "camera-time"),
        default="modality-mean",
        help="source estimator fusion; camera-time retains camera and history order",
    )
    p.add_argument("--no-augment", action="store_true")
    p.add_argument(
        "--selection-interval",
        type=int,
        default=25,
        help="updates between worst-pair checks on reserved selection twin pairs",
    )
    p.add_argument("--allow-engineering-output", action="store_true")

    p = sub.add_parser(
        "materialise-surrogates",
        help="build matched surrogate fields for one held-out defective cell",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--evaluation-plan", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--family", required=True)
    p.add_argument("--defect", required=True)
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--allow-engineering-output", action="store_true")

    # Explicitly engineering-only: this exercises the real formal artifacts
    # without entering the preregistered training stage or publishing results.
    p = sub.add_parser(
        "engineering-formal-train",
        help="run the opt-in engineering checkpoint producer (never a paper run)",
    )
    p.add_argument("--allow-engineering-formal", action="store_true", required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--normalization", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument(
        "--optimizer-steps",
        "--steps",
        dest="steps",
        type=int,
        default=1,
        help="optimizer updates per epoch; --steps is retained as a short alias",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--window-sampling", choices=("shuffled", "sequential"), default="shuffled")
    p.add_argument("--sampling-seed", type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--loss-kind",
        choices=(
            "mse",
            "continuous-rms-mse+gripper-bce-v1",
            "continuous-standardized-mse+gripper-bce-v2",
        ),
        default="mse",
    )
    p.add_argument("--continuous-loss-weight", type=float, default=1.0)
    p.add_argument("--gripper-loss-weight", type=float, default=1.0)
    p.add_argument("--n-tasks", type=int, default=6)
    p.add_argument("--n-pool-embodiments", type=int, default=7)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--n-heads", type=int, default=2)
    p.add_argument("--ffn-dim", type=int, default=64)
    p.add_argument("--no-augment", action="store_true")

    p = sub.add_parser(
        "materialise-leakage",
        help="run an evaluated policy and B7 on one shared held-out frame set",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--evaluation-plan", type=Path, required=True)
    p.add_argument("--evaluated-checkpoint", type=Path, required=True)
    p.add_argument("--evaluated-plan", type=Path, required=True)
    p.add_argument("--evaluated-normalization", type=Path, required=True)
    p.add_argument("--evaluated-arm", required=True)
    p.add_argument("--oracle-checkpoint", type=Path, required=True)
    p.add_argument("--oracle-plan", type=Path, required=True)
    p.add_argument("--oracle-normalization", type=Path, required=True)
    p.add_argument("--oracle-arm", default="B7-oracle")
    p.add_argument("--task", required=True)
    p.add_argument("--family", required=True)
    p.add_argument("--defect", required=True)
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--evaluated-task-index", type=int, required=True)
    p.add_argument("--evaluated-embodiment-index", type=int, required=True)
    p.add_argument("--oracle-task-index", type=int, required=True)
    p.add_argument("--oracle-embodiment-index", type=int, required=True)
    p.add_argument("--smoothing-window", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--source-posterior", type=Path, default=None)
    p.add_argument("--surrogate-input", type=Path, required=True)
    p.add_argument("--allow-engineering-input", action="store_true")
    p.add_argument("--n-tasks", type=int, default=6)
    p.add_argument("--n-pool-embodiments", type=int, default=7)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--ffn-dim", type=int, default=2048)
    p.add_argument("--no-augment", action="store_true")

    p = sub.add_parser("evaluate", help="roll a checkpoint out on a defective target")
    _common(p)
    p.add_argument("--arm", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    _cell(p)

    p = sub.add_parser("diagnose", help="stream the leakage diagnostics over one cell")
    _common(p)
    _cell(p)
    p.add_argument(
        "--leakage-input",
        type=Path,
        required=True,
        help="hash-bound evaluated-policy/B7/teacher/nuisance input artifact",
    )
    p.add_argument(
        "--allow-engineering-input",
        action="store_true",
        help="admit an explicitly non-paper leakage pilot",
    )

    p = sub.add_parser("gates", help="inspect or record the unblinding ledger (11.3)")
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument("--record", default=None, choices=[g.value for g in Gate])
    p.add_argument("--passed", action="store_true")
    p.add_argument("--failed", action="store_true")
    p.add_argument("--value", type=float, default=None)
    p.add_argument("--artifact", default="")
    p.add_argument("--notes", default="")

    return parser


def _request(args: argparse.Namespace, stage: str) -> RunRequest:
    return RunRequest(
        experiment_id=args.experiment,
        stage=stage,
        backend=Backend(args.backend),
        arm_id=getattr(args, "arm", None),
        task=getattr(args, "task", None),
        family=getattr(args, "family", None),
        defect=getattr(args, "defect", None),
        level=None if getattr(args, "level", None) is None else str(args.level),
        geom_variant=getattr(args, "geom_variant", "G0"),
        seed_index=getattr(args, "seed_index", None),
        results_root=getattr(args, "results_root", None),
        repo_root=getattr(args, "repo_root", None),
        require_prereg_tag=not getattr(args, "allow_untagged_prereg", False),
    )


def _cmd_validate(args: argparse.Namespace) -> int:
    request = _request(args, args.stage)
    report = validate(
        request,
        bundle=load_bundle(),
        lock=LockFile(args.lock) if args.lock else LockFile(),
    )
    print(report)
    return 0


def _print_ablation(bundle, exp) -> None:
    """The four fields a 10.6.1 row has and a main experiment does not.

    Printed under the same `config --experiment` as everything else. The level
    ladder is the row's content, so a reader who cannot see it here would have
    to open the YAML to learn what the ablation varies.
    """
    from portal_trace.configs.schema import AblationSpec

    if not isinstance(exp, AblationSpec):
        return
    print(f"  dimension   {exp.dimension}")
    print(f"  runs        {exp.n_runs} (levels sum to the row; 10.6.1)")
    if exp.confounders_closed:
        print(f"  closes      {list(exp.confounders_closed)} (10.6.2)")
    for reference in exp.implemented_by:
        print(f"  implemented {reference}")
    for level in exp.levels:
        mark = "*" if level.is_main else " "
        arm = level.arm or "-"
        print(f"  level     {mark} {level.level_id:18s} arm={arm:18s} "
              f"+{level.incremental_runs} runs, {level.gpu_hours} GPU-h")
    if exp.experiment_id == "A8":
        for level in exp.levels:
            group = bundle.solver_group(level.arm) if level.arm else None
            if group is not None:
                print(f"  solver      {level.level_id:10s} solref x"
                      f"{group.solref_timeconst_scale}, solimp gap x"
                      f"{group.solimp_gap_scale}")


def _cmd_config(args: argparse.Namespace) -> int:
    bundle = load_bundle()
    if args.arm:
        arm = bundle.arm(args.arm)
        cells = bundle.cells(arm.arm_id)
        print(f"arm {arm.arm_id}  role={arm.role}  subset={arm.subset}")
        print(f"  design      {arm.design}")
        print(f"  cells       {len(cells)}: {list(cells)}")
        print(f"  runs        {arm.n_runs}   GPU-h {arm.gpu_hours}")
        print(f"  pool        sources={list(arm.pool.sources)}")
        print(f"              delta=0 in pool: {arm.pool.includes_delta_zero}")
        print(f"              target in pool : {arm.pool.target_in_pool}")
        print(f"              reading        : {arm.pool.reading_consequence}")
        print(f"  train seeds {list(bundle.seeds.train_seed_index_by_arm.get(arm.arm_id, ()))}")
        if arm.ablation is not None:
            print(f"  ablation    {arm.ablation} (10.6.1); its runs are budgeted there")
        if arm.solver_group is not None:
            group = bundle.solver_group(arm.arm_id)
            print(f"  solver grp  {group.group_id}: solref timeconst x"
                  f"{group.solref_timeconst_scale}, solimp gap x{group.solimp_gap_scale}")
        return 0
    if args.experiment:
        exp = bundle.experiment(args.experiment)
        print(f"{exp.experiment_id}  {exp.title}  [{exp.phase}]")
        print(f"  zero GPU    {exp.zero_gpu}   GPU-h {exp.gpu_hours}  CPU-h {exp.cpu_hours}")
        print(f"  arms        {list(exp.arms)}")
        print(f"  design      {exp.design}")
        print(f"  prereg items{list(exp.prereg_items)}")
        _print_ablation(bundle, exp)
        for e in exp.endpoints:
            rank = "" if e.holm_bonferroni_rank is None else f" HB#{e.holm_bonferroni_rank}"
            print(f"  endpoint    {e.name} [{e.kind}, {e.dimension}]{rank}")
        for stage in ("collect", "diagnose", "train"):
            missing = outstanding_measurements(exp.experiment_id, stage, bundle=bundle)
            for m in missing:
                print(f"  BLOCKED at {stage}: {m}")
        return 0
    print(f"arms        : {sorted(bundle.arms)}")
    print(f"experiments : {sorted(bundle.experiments)}")
    print(f"ablations   : {list(bundle.ablations)}")
    print(f"subsets     : {sorted(bundle.subsets)}")
    print(f"prereg      : {bundle.preregistration.version}, "
          f"{len(bundle.preregistration.items)} items, tag "
          f"{bundle.preregistration.git_tag}")
    return 0


def _cmd_collect(args: argparse.Namespace) -> int:
    from portal_trace.runners.collect import collect

    ctx = open_context(
        _request(args, "collect"),
        dataset_root=args.dataset_root,
        results_root=args.results_root,
        lock=LockFile(args.lock) if args.lock else None,
    )
    plan = collect(
        ctx,
        n_demos=args.n_demos,
        level=args.level,
        overrun_step=args.overrun_step,
        overrun_frame=args.overrun_frame,
        source_validation_reservation_path=args.source_validation_reservation,
    )
    print(f"collected {plan.n_demos} demonstrations -> {plan.output}")
    return 0


def _cmd_collect_b7_clean(args: argparse.Namespace) -> int:
    from portal_trace.runners.collect import collect_b7_clean

    ctx = open_context(
        _request(args, "collect"),
        dataset_root=args.dataset_root,
        results_root=args.results_root,
        lock=LockFile(args.lock) if args.lock else None,
    )
    plan = collect_b7_clean(
        ctx,
        level=args.level,
        overrun_step=args.overrun_step,
        overrun_frame=args.overrun_frame,
    )
    print(
        f"collected B7 clean twin {plan.n_demos} demonstrations -> "
        f"{plan.clean_output}"
    )
    return 0


def _cmd_teacher(args: argparse.Namespace) -> int:
    from portal_trace.ablations.teacher_family_rl import RLExpertPlan

    ctx = open_context(
        _request(args, "collect"),
        dataset_root=args.dataset_root,
        results_root=args.results_root,
        lock=LockFile(args.lock) if args.lock else None,
    )
    plan = RLExpertPlan.from_level(
        ctx.bundle,
        task=args.task,
        family=args.family,
        defect=args.defect,
        level=args.level,
        seed_index=args.seed_index,
        artifact_root=args.artifact_root,
    )
    artefact = plan.train()
    print(
        f"A13 rl_expert {plan.family}/{plan.delta.defect_code} L{plan.delta.level}: "
        f"{artefact.total_timesteps} timesteps on the {plan.reward_caliber.value} "
        f"reward -> {artefact.checkpoint}"
    )
    print(
        f"  seeds env {plan.seeds.env} train {plan.seeds.train} eval {plan.seeds.eval}"
        f"   observation {artefact.metadata['observation']['total_width']} wide "
        f"as {list(plan.observation_keys)}"
    )
    return 0


def _cmd_materialise(args: argparse.Namespace) -> int:
    from portal_trace.runners.materialise import materialise

    ctx = open_context(
        _request(args, "materialise"),
        dataset_root=args.dataset_root,
        results_root=args.results_root,
        lock=LockFile(args.lock) if args.lock else None,
    )
    spec = materialise(ctx, check_only=args.check_only)
    print(f"observations at {spec.output}")
    return 0


def _cmd_materialise_b7_clean(args: argparse.Namespace) -> int:
    from portal_trace.runners.materialise import materialise_b7_clean

    ctx = open_context(
        _request(args, "materialise"),
        dataset_root=args.dataset_root,
        results_root=args.results_root,
        lock=LockFile(args.lock) if args.lock else None,
    )
    spec = materialise_b7_clean(ctx, check_only=args.check_only)
    print(f"B7 clean observations at {spec.output}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from portal_trace.runners.train import train

    ctx = open_context(
        _request(args, "train"),
        dataset_root=args.dataset_root,
        results_root=args.results_root,
        lock=LockFile(args.lock) if args.lock else None,
    )
    job = train(
        ctx,
        seed_index=args.seed_index,
        target_family=args.target_family,
        target_defect=args.target_defect,
        requested_level=args.level,
        level_manifest_path=args.level_manifest,
        formal_plan_path=args.formal_plan,
        normalization_path=args.normalization,
        epochs=args.epochs,
        optimizer_steps=args.optimizer_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        dry_run=args.dry_run,
    )
    print(
        f"arm {job.arm.arm_id} target L{job.pool.target_level} seed "
        f"{job.seed_index} -> {job.output_dir}"
    )
    print(
        f"  pool {job.pool.n_demos} demos over {len(job.tasks)} core tasks, "
        f"alpha_demo {job.pool.alpha_demo:.3f}, delta=0 in pool: "
        f"{job.pool.includes_delta_zero}"
    )
    return 0


def _cmd_adapter_probe(args: argparse.Namespace) -> int:
    """Run the bounded adapter integration probe without formal preflight."""
    from portal_trace.policies.interface import PolicyConfig
    from portal_trace.runners.formal_adapter import (
        FORMAL_ADAPTER_PROBE_MARKER,
        FormalAdapterConfig,
        run_formal_adapter_probe,
    )
    from portal_trace.runners.formal_data import (
        load_formal_data_plan,
        load_formal_normalization,
    )

    try:
        plan = load_formal_data_plan(args.plan)
        normalization = load_formal_normalization(args.normalization, plan=plan)
        policy_config = PolicyConfig(
            arm=args.arm,
            n_tasks=args.n_tasks,
            n_pool_embodiments=args.n_pool_embodiments,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            ffn_dim=args.ffn_dim,
            augment=not args.no_augment,
        )
        result = run_formal_adapter_probe(
            FormalAdapterConfig(
                plan=plan,
                normalization=normalization,
                policy_config=policy_config,
                seed=args.seed,
                optimizer_steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                checkpoint_path=args.checkpoint,
            )
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise ConfigError(f"adapter probe refused: {exc}") from exc
    print(FORMAL_ADAPTER_PROBE_MARKER)
    print(
        f"arm {args.arm}: {result.optimizer_steps} updates, "
        f"train windows={result.train_windows}, held-out windows={result.heldout_windows}, "
        f"final loss={result.final_loss:.6g}"
    )
    if result.checkpoint_path is not None:
        print(f"probe checkpoint -> {result.checkpoint_path}")
    return 0


def _cmd_engineering_formal_train(args: argparse.Namespace) -> int:
    """Run the explicit engineering producer outside the formal train stage."""
    from portal_trace.policies.interface import PolicyConfig
    from portal_trace.runners.formal_data import (
        load_formal_data_plan,
        load_formal_normalization,
    )
    from portal_trace.runners.formal_training import (
        ENGINEERING_FORMAL_PRODUCER_MARKER,
        FormalTrainingConfig,
        run_formal_training,
    )

    if args.allow_engineering_formal is not True:
        # Keep this guard even though argparse marks the flag required: callers
        # and tests can invoke a command handler with a hand-built Namespace.
        raise ConfigError(
            "engineering formal training requires --allow-engineering-formal; "
            "the output is engineering-only and never a paper result"
        )
    try:
        plan = load_formal_data_plan(args.plan)
        normalization = load_formal_normalization(args.normalization, plan=plan)
        policy_config = PolicyConfig(
            arm=args.arm,
            n_tasks=args.n_tasks,
            n_pool_embodiments=args.n_pool_embodiments,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            ffn_dim=args.ffn_dim,
            augment=not args.no_augment,
        )
        result = run_formal_training(
            FormalTrainingConfig(
                plan=plan,
                normalization=normalization,
                plan_path=args.plan,
                normalization_path=args.normalization,
                checkpoint_path=args.checkpoint,
                policy_config=policy_config,
                seed=args.seed,
                epochs=args.epochs,
                optimizer_steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=args.device,
                loss_kind=args.loss_kind,
                continuous_loss_weight=args.continuous_loss_weight,
                gripper_loss_weight=args.gripper_loss_weight,
                window_sampling=getattr(args, "window_sampling", "shuffled"),
                sampling_seed=getattr(args, "sampling_seed", None),
            ),
            allow_engineering_formal=args.allow_engineering_formal,
        )
    except (OSError, ValueError, RuntimeError, PermissionError, KeyError, TypeError) as exc:
        raise ConfigError(f"engineering formal training refused: {exc}") from exc
    print(ENGINEERING_FORMAL_PRODUCER_MARKER)
    print(
        f"arm {args.arm}: epochs={result.epochs}, updates={result.optimizer_steps}, "
        f"train windows={result.train_windows}, held-out windows={result.heldout_windows}, "
        f"final loss={result.final_loss:.6g}"
    )
    print(f"engineering checkpoint -> {result.checkpoint_path}")
    return 0


def _cmd_materialise_leakage(args: argparse.Namespace) -> int:
    """Build the exact shared-frame artifact consumed by ``diagnose``."""
    from portal_trace.policies.interface import PolicyConfig
    from portal_trace.runners.leakage_materialize import (
        PolicyCheckpointSpec,
        materialize_leakage_input,
    )

    shared = PolicyConfig(
        arm=args.evaluated_arm,
        n_tasks=args.n_tasks,
        n_pool_embodiments=args.n_pool_embodiments,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        augment=not args.no_augment,
    )
    try:
        header = materialize_leakage_input(
            output=args.output,
            evaluation_plan_path=args.evaluation_plan,
            evaluated_spec=PolicyCheckpointSpec(
                checkpoint=args.evaluated_checkpoint,
                plan=args.evaluated_plan,
                normalization=args.evaluated_normalization,
                policy_config=shared,
                task_index=args.evaluated_task_index,
                embodiment_index=args.evaluated_embodiment_index,
            ),
            oracle_spec=PolicyCheckpointSpec(
                checkpoint=args.oracle_checkpoint,
                plan=args.oracle_plan,
                normalization=args.oracle_normalization,
                policy_config=shared.evolve(arm=args.oracle_arm),
                task_index=args.oracle_task_index,
                embodiment_index=args.oracle_embodiment_index,
            ),
            task=args.task,
            family=args.family,
            defect=args.defect,
            level=args.level,
            smoothing_window=args.smoothing_window,
            batch_size=args.batch_size,
            allow_engineering=args.allow_engineering_input,
            source_posterior_path=args.source_posterior,
            surrogate_input_path=args.surrogate_input,
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        raise ConfigError(f"leakage materialisation refused: {exc}") from exc
    print(
        f"{args.task}/{args.family}/{args.defect}/L{args.level}: "
        f"{header.n_frames} shared frames -> {args.output}"
    )
    print(
        f"  release={header.provenance.release_status}, "
        f"source_posterior={'yes' if header.has_source_posterior else 'no'}, "
        f"sha256={header.artifact_sha256}"
    )
    return 0


def _cmd_materialise_surrogates(args: argparse.Namespace) -> int:
    """Build the matched negative controls on the held-out rho_off frames."""
    from portal_trace.runners.surrogate_materialize import materialize_surrogate_input

    try:
        header = materialize_surrogate_input(
            output=args.output,
            evaluation_plan_path=args.evaluation_plan,
            task=args.task,
            family=args.family,
            defect=args.defect,
            level=args.level,
            seed=args.seed,
            allow_engineering=args.allow_engineering_output,
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        raise ConfigError(f"surrogate materialisation refused: {exc}") from exc
    print(
        f"{args.task}/{args.family}/{args.defect}/L{args.level}: "
        f"{header.n_frames} matched-surrogate frames -> {args.output}"
    )
    print(
        f"  admitted={list(header.provenance.admitted_surrogates)}, "
        f"struck={list(header.provenance.struck_surrogates)}, "
        f"release={header.provenance.release_status}, sha256={header.artifact_sha256}"
    )
    return 0


def _cmd_train_source_posterior(args: argparse.Namespace) -> int:
    from portal_trace.runners.source_discriminator import (
        SourceDiscriminatorConfig,
        train_source_discriminator,
    )

    try:
        result = train_source_discriminator(
            SourceDiscriminatorConfig(
                plan_path=args.plan,
                normalization_path=args.normalization,
                checkpoint_path=args.checkpoint,
                posterior_path=args.posterior,
                source_embodiment_indices=tuple(args.source_embodiment_indices),
                target_embodiment_indices=tuple(args.target_embodiment_indices),
                task=args.task,
                family=args.family,
                defect=args.defect,
                level=args.level,
                seed=args.seed,
                epochs=args.epochs,
                optimizer_steps=args.optimizer_steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=args.device,
                n_tasks=args.n_tasks,
                d_model=args.d_model,
                augment=not args.no_augment,
                selection_interval=args.selection_interval,
                fusion_mode=args.fusion_mode,
                validation_plan_path=args.validation_plan,
                validation_reservation_path=args.validation_reservation,
            ),
            allow_engineering=args.allow_engineering_output,
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        raise ConfigError(f"source discriminator refused: {exc}") from exc
    metrics = result.validation_metrics
    print(
        f"source discriminator: {result.optimizer_steps} updates, "
        f"best step={result.best_step}, "
        f"pooled AUC={metrics['auc']:.4f}, worst-pair AUC="
        f"{metrics['worst_pair_auc']:.4f}, worst-pair ECE={metrics['w_ece']:.4f}, "
        f"worst-pair alpha error={metrics['w_alpha_recovery']:.4f}, "
        f"support gap={metrics['w_support_gap']:.4f}, "
        f"information gain={metrics['information_gain_bits']:.4f} bits, "
        f"budget={metrics['error_budget_total']:.4f}/"
        f"{metrics['error_budget_limit']:.4f}"
    )
    print(
        f"  release={result.release_status}, checkpoint={result.checkpoint_path}, "
        f"posterior={result.posterior_path}"
    )
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from portal_trace.runners.evaluate import EvaluationTarget, evaluate

    ctx = open_context(
        _request(args, "evaluate"),
        dataset_root=args.dataset_root,
        results_root=args.results_root,
        lock=LockFile(args.lock) if args.lock else None,
    )
    result = evaluate(
        ctx,
        checkpoint=args.checkpoint,
        target=EvaluationTarget(
            family=args.family, defect=args.defect, level=args.level, task=args.task
        ),
    )
    print(
        f"{result.arm_id} on {result.target.family}/{result.target.defect}: "
        f"{result.success_rate:.1%} over {len(result.success)} shared initial states "
        f"(+- {result.absolute_rate_common_offset:.1f} pp common state-sampling "
        "offset, which does not enter any paired CI)"
    )
    return 0


def _cmd_diagnose(args: argparse.Namespace) -> int:
    from portal_trace.runners.diagnose import diagnose, diagnose_to_json

    request = _request(args, "diagnose")
    lock = LockFile(args.lock) if args.lock else LockFile()
    if args.allow_engineering_input:
        bundle = load_bundle()
        ctx = SimpleNamespace(
            request=request,
            bundle=bundle,
            lock=lock,
            dataset_root=Path(args.dataset_root),
            results_root=Path(args.results_root),
            threshold=bundle.preregistration.threshold,
        )
    else:
        ctx = open_context(
            request,
            dataset_root=args.dataset_root,
            results_root=args.results_root,
            lock=lock,
        )
    report = diagnose(
        ctx,
        task=args.task,
        family=args.family,
        defect=args.defect,
        level=args.level,
        leakage_input=args.leakage_input,
        allow_engineering=args.allow_engineering_input,
    )
    arm_slug = str(report.quantities["evaluated_arm"]).replace("/", "-")
    out = (
        Path(args.results_root)
        / "diagnose"
        / f"{args.task}-{args.family}-{args.defect}-L{args.level}-{arm_slug}.json"
    )
    diagnose_to_json(report, out)
    print(f"{report.cell}: {report.n_frames} frames -> {out}")
    return 0


def _cmd_gates(args: argparse.Namespace) -> int:
    ledger = UnblindingLedger.load(Path(args.results_root) / "unblinding.json")
    if args.record is None:
        print(ledger.summary())
        return 0
    gate = Gate(args.record)
    if gate in GATE_EVALUATORS:
        print(
            f"{gate.value} is one of 10.8's five stop-gates and its verdict is "
            f"computed, not typed: call {GATE_EVALUATORS[gate]} on the measurements "
            "and record the GateVerdict it returns. A pass standard that a person "
            "reads off a table and enters by hand is not a preregistered criterion",
            file=sys.stderr,
        )
        return 2
    if args.passed == args.failed:
        print(
            "recording a gate needs exactly one of --passed / --failed. The verdict "
            "is recorded but never gates a later stage: 11.3's order is about what "
            "has been looked at, not about what it said",
            file=sys.stderr,
        )
        return 2
    result = ledger.record(
        gate,
        passed=args.passed,
        value=args.value,
        artifact=args.artifact,
        notes=args.notes,
    )
    print(f"recorded {result.gate.value}: {'pass' if result.passed else 'FAIL'}")
    return 0


_COMMANDS = {
    "validate": _cmd_validate,
    "config": _cmd_config,
    "collect": _cmd_collect,
    "collect-b7-clean": _cmd_collect_b7_clean,
    "teacher": _cmd_teacher,
    "materialise": _cmd_materialise,
    "materialise-b7-clean": _cmd_materialise_b7_clean,
    "train": _cmd_train,
    "adapter-probe": _cmd_adapter_probe,
    "engineering-formal-train": _cmd_engineering_formal_train,
    "train-source-posterior": _cmd_train_source_posterior,
    "materialise-surrogates": _cmd_materialise_surrogates,
    "materialise-leakage": _cmd_materialise_leakage,
    "evaluate": _cmd_evaluate,
    "diagnose": _cmd_diagnose,
    "gates": _cmd_gates,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except (
        PreflightError,
        MeasurementMissingError,
        BackendUnavailable,
        UnblindingOrderViolation,
        ConfigError,
        MissingComponent,
        TeacherFamilyUnavailable,
    ) as exc:
        # Printed rather than raised: these are all "the run may not start"
        # conditions with a message written for a reader, and a traceback on top
        # of them only hides the message.
        print(f"\n{type(exc).__name__}: {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
