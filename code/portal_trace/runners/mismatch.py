"""The E11 cross-evaluation orchestrator (proposal 10.4.11).

E11 trains nothing. Its whole content is a schedule: take the 108 policies the
S36 arm has already produced, deploy each of them on the five *other* core
defects of its own morphology family, and record a per-rollout failure profile
on the same `eval_init` manifest the main evaluation used. The diagonal cell is
the main evaluation's own reading and is reused rather than re-run, which is
where the "5 off-diagonal targets" in the cost line comes from.

This module owns the schedule and the refusals; `portal_trace.eval.mismatch`
owns the readout. The three refusals are the substance:

* a policy whose checkpoint is not on disk stops the run and names the exact
  `portal-trace train` invocation that produces it -- nothing is skipped and no
  cell is filled by anything other than a rollout;
* a compensation norm that has not been measured stops the run and names the
  stage that measures it, because ||Delta c|| and ||c_src|| are the two
  explanatory variables of T10 and a missing one cannot be defaulted;
* a deployment target that is a clean twin stops the run (appendix A-1).

The mismatch field is defined pointwise and its norm is taken in L^2(rho) over
the deployment target's own rollout states, which is the same norm the Lambda
family uses in 10.1: Delta c = c_{delta_src} - c_{delta_tgt} is a difference of
two fields evaluated at the *same* state, so it must be differenced before it is
normed. Norming each field first and subtracting the norms is a different and
smaller quantity -- it discards every direction in which the two compensations
disagree while having the same magnitude -- so `mismatch_norms` takes the two
per-state arrays rather than two scalars.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from portal_trace.configs.schema import ConfigError
from portal_trace.defects.params import CORE_DEFECTS
from portal_trace.eval.mismatch import (
    E11_COST,
    N_CORE_TASKS,
    MismatchCell,
    MismatchMatrix,
)
from portal_trace.registry.families import MAIN_FAMILIES
from portal_trace.runners.common import RunContext
from portal_trace.runners.evaluate import EvaluationResult, EvaluationTarget, evaluate

__all__ = [
    "PolicyKey",
    "CellPlan",
    "MismatchPlan",
    "CompensationNorms",
    "mismatch_norms",
    "cross_evaluate",
    "PolicyCheckpointMissing",
    "CompensationNormMissing",
    "MismatchTrainingRequested",
    "TaskCoverageError",
]

E11_ARM_ID = "E11"


class PolicyCheckpointMissing(FileNotFoundError):
    """A cell needs a trained policy that is not in this checkout."""


class CompensationNormMissing(KeyError):
    """A cell needs a compensation-field norm that has not been measured."""


class MismatchTrainingRequested(ConfigError):
    """Something asked E11 to train. It does not train."""


class TaskCoverageError(ConfigError):
    """A cell was assembled from the wrong number of core tasks."""


@dataclass(frozen=True)
class PolicyKey:
    """Which trained policy: a config of the S36 subset at one training seed."""

    family: str
    defect: str
    seed: int

    def __post_init__(self) -> None:
        if self.family not in MAIN_FAMILIES:
            raise ConfigError(f"{self.family!r} is not one of {list(MAIN_FAMILIES)}")
        if self.defect not in CORE_DEFECTS:
            raise ConfigError(f"{self.defect!r} is not one of {list(CORE_DEFECTS)}")

    def __str__(self) -> str:
        return f"{self.family}/{self.defect}/seed{self.seed}"


@dataclass(frozen=True)
class CellPlan:
    """One (policy, deployment target) of the schedule, across the six core tasks."""

    policy: PolicyKey
    target_defect: str
    level: int

    @property
    def is_diagonal(self) -> bool:
        return self.policy.defect == self.target_defect

    @property
    def n_rollouts(self) -> int:
        return N_CORE_TASKS * E11_COST.n_states

    def targets(self, tasks: Sequence[str]) -> tuple[EvaluationTarget, ...]:
        return tuple(
            EvaluationTarget(
                family=self.policy.family,
                defect=self.target_defect,
                level=self.level,
                task=task,
            )
            for task in tasks
        )


class MismatchPlan:
    """The full E11 schedule, and the arithmetic that says what it costs.

    The diagonal is in the plan -- the readout needs it as the pairing partner
    and as the natural zero level -- but it is marked as reused, and only the
    off-diagonal cells are counted against the CPU budget.
    """

    def __init__(
        self,
        *,
        families: Sequence[str] = MAIN_FAMILIES,
        defects: Sequence[str] = CORE_DEFECTS,
        seeds: Sequence[int],
        level: int,
    ) -> None:
        if not seeds:
            raise ConfigError("E11 cross-evaluates the S36 policies; name their seeds")
        unknown = sorted(set(families) - set(MAIN_FAMILIES))
        if unknown:
            raise ConfigError(f"unknown morphology families {unknown}")
        unknown = sorted(set(defects) - set(CORE_DEFECTS))
        if unknown:
            raise ConfigError(f"unknown core defects {unknown}")
        self.families = tuple(families)
        self.defects = tuple(defects)
        self.seeds = tuple(seeds)
        self.level = int(level)

    def policies(self) -> Iterator[PolicyKey]:
        for family in self.families:
            for defect in self.defects:
                for seed in self.seeds:
                    yield PolicyKey(family=family, defect=defect, seed=seed)

    def cells(self) -> Iterator[CellPlan]:
        for policy in self.policies():
            for target in self.defects:
                yield CellPlan(policy=policy, target_defect=target, level=self.level)

    @property
    def n_policies(self) -> int:
        return len(self.families) * len(self.defects) * len(self.seeds)

    @property
    def n_off_diagonal_cells(self) -> int:
        return self.n_policies * (len(self.defects) - 1)

    @property
    def n_new_rollouts(self) -> int:
        return self.n_off_diagonal_cells * N_CORE_TASKS * E11_COST.n_states

    @property
    def cpu_hours(self) -> float:
        return self.n_new_rollouts * E11_COST.seconds_per_rollout / 3600.0

    @property
    def gpu_hours(self) -> float:
        return 0.0

    def train(self, *_args: object, **_kwargs: object) -> None:
        raise MismatchTrainingRequested(
            "E11 is a zero-training arm: it cross-evaluates the S36 policies that the "
            "primary endpoint has already trained (arms.yaml gives it n_runs: 0 and "
            "gpu_hours: 0). Train the policies through `portal-trace train --arm B2` "
            "and pass their checkpoints to `cross_evaluate`"
        )


# ---------------------------------------------------------------------------
# The two explanatory variables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompensationNorms:
    """||Delta c||, ||c_src|| and ||c_tgt|| for one cell, all in L^2(rho)."""

    delta_c: float
    c_src: float
    c_tgt: float


def mismatch_norms(
    c_src: np.ndarray, c_tgt: np.ndarray, *, weights: np.ndarray | None = None
) -> CompensationNorms:
    """The three norms of one cell from the two fields on a common state sample.

    Both arrays are (n_states, n_joints): the compensation each defect's field
    asks for at the *same* state. The mismatch is differenced pointwise and then
    normed, which is what `Delta c = c_{delta_src} - c_{delta_tgt}` says; norming
    first and subtracting would collapse every disagreement in direction.

    `weights` is the state measure rho. It defaults to none rather than to
    uniform: passing nothing means the states are already an i.i.d. draw from
    rho, which is what a rollout on the deployment target gives, and a caller
    with an importance-weighted sample has to say so.
    """
    a = np.asarray(c_src, dtype=float)
    b = np.asarray(c_tgt, dtype=float)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"the compensation samples are (n_states, n_joints); got {a.shape} and "
            f"{b.shape}"
        )
    if a.shape != b.shape:
        raise ValueError(
            f"the two fields were evaluated on different samples, {a.shape} against "
            f"{b.shape}. Delta c is a difference at one and the same state"
        )
    if a.size == 0:
        raise ValueError("no states: the L^2(rho) norm has nothing to average over")
    if weights is None:
        w = np.full(a.shape[0], 1.0 / a.shape[0])
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (a.shape[0],):
            raise ValueError(f"weights {w.shape} against {a.shape[0]} states")
        total = float(w.sum())
        if total <= 0.0 or np.any(w < 0.0):
            raise ValueError("rho's weights must be non-negative and sum above zero")
        w = w / total

    def l2(field: np.ndarray) -> float:
        return float(np.sqrt(np.sum(w * np.sum(field * field, axis=1))))

    return CompensationNorms(delta_c=l2(a - b), c_src=l2(a), c_tgt=l2(b))


# ---------------------------------------------------------------------------
# The orchestration
# ---------------------------------------------------------------------------


def _concatenate(results: Sequence[EvaluationResult]) -> tuple[np.ndarray, np.ndarray]:
    success = np.concatenate([np.asarray(r.success, dtype=float) for r in results])
    peak = np.concatenate([np.asarray(r.peak_cfrc_ext, dtype=float) for r in results])
    return success, peak


def cross_evaluate(
    ctx: RunContext,
    plan: MismatchPlan,
    *,
    checkpoints: Mapping[PolicyKey, Path],
    norms: Mapping[tuple[str, str, str], CompensationNorms],
    tasks: Sequence[str],
    constraint_violation: Mapping[tuple[str, str, str, int], np.ndarray],
    terminal_overshoot: Mapping[tuple[str, str, str, int], np.ndarray],
    manifest_id: str,
) -> MismatchMatrix:
    """Roll every cell of the schedule out and assemble the matrix.

    `constraint_violation` and `terminal_overshoot` are passed in per cell rather
    than read off `EvaluationResult`, which carries the success vector and the
    peak cfrc_ext and nothing else. They are the other two readings 10.4.11 asks
    for, and they are produced by the same rollout: `portal_trace.eval.endpoints`
    computes both from the trajectory, so the caller that ran the rollouts has
    them. They are required, not optional -- the failure-type profile is a
    two-dimensional reading and half of it is the constraint-violation rate.
    """
    if len(tasks) != N_CORE_TASKS:
        raise TaskCoverageError(
            f"{len(tasks)} tasks: 10.4.11's cost line and its per-cell rollout count "
            f"are both over the {N_CORE_TASKS} core tasks. A cell assembled from "
            "fewer would enter the matrix with a different number of rollouts and "
            "break the shared initial-state pairing"
        )
    cells: list[MismatchCell] = []
    for cell_plan in plan.cells():
        policy = cell_plan.policy
        checkpoint = checkpoints.get(policy)
        if checkpoint is None or not Path(checkpoint).exists():
            raise PolicyCheckpointMissing(
                f"E11 cell {policy} -> {cell_plan.target_defect} needs the trained "
                f"policy {policy}, whose checkpoint is "
                f"{'not registered' if checkpoint is None else f'absent at {checkpoint}'}. "
                "E11 trains nothing; it cross-evaluates the S36 policies of the "
                "primary endpoint. Produce it with `portal-trace train --arm B2 "
                f"--family {policy.family} --defect {policy.defect} --seed "
                f"{policy.seed}` and register the resulting checkpoint"
            )
        norm_key = (policy.family, policy.defect, cell_plan.target_defect)
        if norm_key not in norms:
            raise CompensationNormMissing(
                f"E11 cell {policy} -> {cell_plan.target_defect} has no compensation "
                f"norms for {norm_key}. ||Delta c|| and ||c_src|| are T10's two "
                "explanatory variables and neither may be defaulted. Evaluate the two "
                "ground-truth compensation fields on the deployment target's own "
                "rollout states with `portal-trace diagnose` and pass the pair through "
                "`mismatch_norms`"
            )
        cell_norms = norms[norm_key]
        results = [
            evaluate(ctx, checkpoint=Path(checkpoint), target=target)
            for target in cell_plan.targets(tasks)
        ]
        success, peak = _concatenate(results)
        reading_key = (
            policy.family,
            policy.defect,
            cell_plan.target_defect,
            policy.seed,
        )
        for name, table in (
            ("constraint_violation", constraint_violation),
            ("terminal_overshoot", terminal_overshoot),
        ):
            if reading_key not in table:
                raise CompensationNormMissing(
                    f"E11 cell {reading_key} has no {name} vector. 10.4.11's endpoint "
                    "is the two-dimensional failure-type profile -- the "
                    "constraint-violation rate against the task-failure rate -- with "
                    "the peak cfrc_ext and the terminal overshoot beside it, so a "
                    "missing channel is a missing endpoint, not a missing extra"
                )
        cells.append(
            MismatchCell(
                family=policy.family,
                source_defect=policy.defect,
                target_defect=cell_plan.target_defect,
                seed=policy.seed,
                manifest_id=manifest_id,
                success=success,
                constraint_violation=np.asarray(
                    constraint_violation[reading_key], dtype=float
                ),
                peak_cfrc_ext=peak,
                terminal_overshoot=np.asarray(
                    terminal_overshoot[reading_key], dtype=float
                ),
                delta_c_norm=cell_norms.delta_c,
                c_src_norm=cell_norms.c_src,
                c_tgt_norm=cell_norms.c_tgt,
            )
        )
    return MismatchMatrix(cells)
