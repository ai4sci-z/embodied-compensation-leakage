"""Typed schema for the declarative experiment configuration (proposal 11.2).

`configs/` is the directory that turns deliverables 3 and 4 into re-runnable
objects: P0/P1, E1-E12, B1-B13, the three-layer seed table (11.3), the E9
isolation ladder (10.4.9) and the preregistration file. This module holds the
types; `loader.py` reads the YAML into them and `validator.py` refuses to start
a run whose configuration is incomplete.

Three fields exist here only because the proposal says a missing one changes the
meaning of a result, so none of them is optional:

`PoolComposition.includes_delta_zero` -- 11.3 item 6 and the 10.4.0 pool table.
When the pool contains the clean twin, route A's anchoring holds by
construction, and a leakage reading then means "the algorithm did not use the
available anchor", not "there was not enough information". That sentence
inverts the reading of a null result in E7/E8, so the flag is required rather
than defaulted to False.

`ArmSpec.design` -- 11.3 and the 11.4 model card. Every summary-table row
carries (M, D, S) because the MDE and the inconclusive rule cannot be derived
without them.

`EndpointSpec.reference_arm` -- appendix A-4. Any share decomposition has to
compute Lambda^part against one and the same B7 Oracle, or the unexplained term
absorbs the whole reference-point drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "ConfigError",
    "SeedLayer",
    "Design",
    "SubsetSpec",
    "PoolComposition",
    "MeasurementRequirement",
    "EndpointSpec",
    "ArmSpec",
    "ExperimentSpec",
    "AblationLevel",
    "AblationSpec",
    "ABLATIONS_EXPECTED",
    "IsolationRung",
    "IsolationLadder",
    "SeedTable",
    "NAMED_SUBSETS_EXPECTED",
    "ISOLATION_RUNGS_EXPECTED",
]


class ConfigError(ValueError):
    """A configuration is malformed, incomplete, or contradicts the proposal."""


# 10.4.0: the five named subsets are referenced by every chapter and registered
# in the preregistration; a sixth name would silently widen a scope claim.
NAMED_SUBSETS_EXPECTED: tuple[str, ...] = ("S36", "S18", "S9", "S6", "S4fam")

# 10.4.9: four ladder rungs plus the decisive control. I3' is not one of the
# four -- it shares I3's isolation value and differs only in the split rule.
ISOLATION_RUNGS_EXPECTED: tuple[str, ...] = ("I0", "I1", "I2", "I3", "I3prime")

# 10.6.1: the ablation table has fourteen rows and its subtotal (114 runs, 570
# GPU-h) is quoted in the budget chapter. Three of the rows close a named
# confounder of 10.6.2, so a suite that silently lost one would leave that
# confounder open with nothing saying so.
ABLATIONS_EXPECTED: tuple[str, ...] = tuple(f"A{n}" for n in range(1, 15))


class SeedLayer(str, Enum):
    """The three independent layers of 11.3.

    ENV controls initial-state randomisation, object poses and textures; TRAIN
    controls initialisation, data order and augmentation; EVAL controls rollout
    initial states. They are allocated explicitly and never share one global
    seed.
    """

    ENV = "env"
    TRAIN = "train"
    EVAL = "eval"


@dataclass(frozen=True)
class Design:
    """(M, D, S): morphology families, defects, training seeds.

    11.3 requires this triple on every row of every summary table, because
    Var(D_bar) = sigma_m^2/M + sigma_d^2/D + sigma_e^2/(MD) +
    sigma_seed^2/(MDS) and neither the MDE nor the inconclusive rule can be
    derived without all three.
    """

    n_families: int
    n_defects: int
    n_seeds: int

    def __post_init__(self) -> None:
        for name in ("n_families", "n_defects", "n_seeds"):
            v = getattr(self, name)
            if not isinstance(v, int) or v <= 0:
                raise ConfigError(f"design.{name} must be a positive int, got {v!r}")

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.n_families, self.n_defects, self.n_seeds)

    @property
    def n_cells(self) -> int:
        return self.n_families * self.n_defects

    def __str__(self) -> str:
        return f"(M={self.n_families}, D={self.n_defects}, S={self.n_seeds})"


@dataclass(frozen=True)
class SubsetSpec:
    """One of the five named subsets of 10.4.0.

    `defects` is empty for a family-level subset (S^(4)), whose defect axis is
    set by the arm that uses it. `cells()` refuses to guess in that case: the
    three crowns and E4b share S^(4) so that their sigma_m is comparable, and
    inventing a defect axis here would make two arms look like they ran the same
    cells when they did not.
    """

    name: str
    families: tuple[str, ...]
    defects: tuple[str, ...]
    rationale: str
    family_level_only: bool = False

    def __post_init__(self) -> None:
        if not self.families:
            raise ConfigError(f"subset {self.name}: no families listed")
        if len(set(self.families)) != len(self.families):
            raise ConfigError(f"subset {self.name}: duplicate family")
        if self.family_level_only and self.defects:
            raise ConfigError(
                f"subset {self.name}: a family-level subset must not pin a defect axis"
            )
        if not self.family_level_only and not self.defects:
            raise ConfigError(f"subset {self.name}: no defects listed")

    def cells(self) -> tuple[tuple[str, str], ...]:
        if self.family_level_only:
            raise ConfigError(
                f"subset {self.name} is family-level; the arm supplies the defect axis. "
                "Call cells_with(defects=...) instead of guessing one here"
            )
        return tuple((f, d) for f in self.families for d in self.defects)

    def cells_with(self, defects: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
        return tuple((f, d) for f in self.families for d in defects)

    @property
    def size(self) -> int | None:
        return None if self.family_level_only else len(self.families) * len(self.defects)


@dataclass(frozen=True)
class PoolComposition:
    """The per-arm pool composition table of 10.4.0, preregistered by 11.3 item 6.

    Every field below is required. `includes_delta_zero` in particular: with the
    clean twin in the pool, route A's anchoring holds by construction and a
    leakage reading means the algorithm did not use an available anchor; without
    it, the same number means the information was not there. E7 and E8 report
    null results, so the two readings are not interchangeable.
    """

    arm: str
    sources: tuple[str, ...]
    includes_delta_zero: bool
    target_in_pool: bool
    reading_consequence: str
    n_target_demos: int
    n_source_demos: int
    k_sources: int

    def __post_init__(self) -> None:
        if not self.reading_consequence.strip():
            raise ConfigError(
                f"pool[{self.arm}]: reading_consequence is required; 11.3 makes the "
                "pool table part of the preregistration precisely because it decides "
                "how a leakage reading is to be read"
            )
        if self.k_sources < 0 or self.n_source_demos < 0 or self.n_target_demos < 0:
            raise ConfigError(f"pool[{self.arm}]: negative demonstration count")

    @property
    def pool_size(self) -> int:
        return self.n_target_demos + self.k_sources * self.n_source_demos

    @property
    def alpha_demo(self) -> float:
        """Nominal mixing weight (10.4.0).

        Equal to `alpha_frame` bit for bit under the fixed-horizon collection of
        9.6.1; the data card stores both and a non-zero difference is a bug.
        """
        total = self.pool_size
        if total == 0:
            raise ConfigError(f"pool[{self.arm}]: empty pool has no mixing weight")
        return (self.k_sources * self.n_source_demos) / total


@dataclass(frozen=True)
class MeasurementRequirement:
    """A lock-file entry this run needs before it may start.

    `key` is a `RequiredMeasurement.key_template`; the validator resolves its
    scope fields from the run request, from `fixed_scope`, and by expanding
    `expand`. Nothing here carries a value -- a value would be exactly the
    default that CONTRACT 3.1 forbids.
    """

    key: str
    stages: tuple[str, ...]
    fixed_scope: dict[str, str] = field(default_factory=dict)
    expand: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stages:
            raise ConfigError(f"measurement requirement {self.key!r}: no stages listed")
        overlap = set(self.fixed_scope) & set(self.expand)
        if overlap:
            raise ConfigError(
                f"measurement requirement {self.key!r}: {sorted(overlap)} both fixed "
                "and expanded"
            )


@dataclass(frozen=True)
class EndpointSpec:
    """One preregistered endpoint.

    `dimension` is checked against `diagnostics.units`: an endpoint declared as
    an amplitude may not be compared with one declared as an energy (14.5 E-1,
    appendix A-6).
    """

    name: str
    kind: str            # primary | secondary | descriptive | exploratory | design_diagnostic
    estimator: str
    statistic: str
    dimension: str       # amplitude | energy | probability | dimensionless | count
    verdict_rule: str
    reference_arm: str | None = None
    mde: float | None = None
    holm_bonferroni_rank: int | None = None
    notes: str = ""

    KINDS = ("primary", "secondary", "descriptive", "exploratory", "design_diagnostic")
    DIMENSIONS = ("amplitude", "energy", "probability", "dimensionless", "count", "seconds")

    def __post_init__(self) -> None:
        if self.kind not in self.KINDS:
            raise ConfigError(f"endpoint {self.name}: kind {self.kind!r} not in {self.KINDS}")
        if self.dimension not in self.DIMENSIONS:
            raise ConfigError(
                f"endpoint {self.name}: dimension {self.dimension!r} not in {self.DIMENSIONS}"
            )
        if not self.verdict_rule.strip():
            raise ConfigError(
                f"endpoint {self.name}: 11.3 allows exactly three verdicts and requires "
                "the condition for each to be preregistered; verdict_rule is empty"
            )
        # A-24: Q_div and Lambda_res contain no policy quantity and can be
        # computed before training, so they cannot be preregistered endpoints.
        if self.kind in ("primary", "secondary") and self.name in ("q_div", "lambda_res"):
            raise ConfigError(
                f"endpoint {self.name}: appendix A-24 -- it is built only from the "
                "injected ground truth and carries no policy quantity, so it is a "
                "design diagnostic, not a preregistered endpoint. The policy-side "
                "counterparts are lambda_emp_res and lambda_emp_con"
            )


@dataclass(frozen=True)
class ArmSpec:
    """One training or evaluation arm: B1-B13, or an experiment-local arm.

    `solver_group` names one of A8's three contact-parameter levels, and `None`
    means the model as robosuite compiles it. It is on the arm rather than on the
    experiment because the environment builder is reached with an arm and nothing
    else, and X10 is only discharged if the group travels all the way into the
    compiled `mjModel` rather than being a column in a results table.

    `ablation` names the 10.6.1 row an arm belongs to. It is what makes an arm
    with no design and no seed allocation legible: the runs are budgeted in that
    row, and the validator's refusal to train quotes it.

    `defect_axis` is how preregistration item 14 is satisfied for an arm whose
    cells are not exactly a named subset: the subset supplies the family list
    and the axis supplies the defect list, so the (family, defect) cells are
    enumerable from the config. `defect_axis_kind` is `experiment_local` when
    the axis is not made of core defects at all -- E7's sources are new delta
    vectors reserved in 9.6.5 -- and that case demands a written rationale,
    because an axis whose levels are not core defects cannot be checked against
    the defect catalogue.
    """

    arm_id: str
    role: str
    subset: str
    design: Design | None
    pool: PoolComposition
    n_runs: int
    gpu_hours: float
    defect_axis: tuple[str, ...] = ()
    defect_axis_kind: str = "core"
    defect_axis_rationale: str = ""
    reference_arm: str | None = None
    architecture: str = ""
    solver_group: str | None = None
    ablation: str | None = None
    notes: str = ""

    ROLES = (
        "pooled", "oracle", "single_embodiment", "counterfactual", "conditioned",
        "randomisation", "decoupling", "curation", "residual_head", "per_embodiment",
        "latent", "grouped", "decisive_control", "ablation",
    )
    AXIS_KINDS = ("core", "experiment_local")

    def __post_init__(self) -> None:
        if self.role not in self.ROLES:
            raise ConfigError(f"arm {self.arm_id}: role {self.role!r} not in {self.ROLES}")
        if self.pool.arm != self.arm_id:
            raise ConfigError(
                f"arm {self.arm_id}: pool table is labelled {self.pool.arm!r}"
            )
        if self.n_runs < 0:
            raise ConfigError(f"arm {self.arm_id}: negative run count")
        if self.defect_axis_kind not in self.AXIS_KINDS:
            raise ConfigError(
                f"arm {self.arm_id}: defect_axis_kind {self.defect_axis_kind!r} not in "
                f"{self.AXIS_KINDS}"
            )
        if self.defect_axis_kind == "experiment_local" and not self.defect_axis_rationale:
            raise ConfigError(
                f"arm {self.arm_id}: an experiment-local defect axis cannot be checked "
                "against the 9.2.10 catalogue, so preregistration item 14 needs it "
                "spelled out; defect_axis_rationale is empty"
            )
        if self.n_runs > 0 and self.design is None:
            raise ConfigError(
                f"arm {self.arm_id}: {self.n_runs} runs but no design. 11.4 requires "
                "(M, D, S) on every summary row, because the MDE and the inconclusive "
                "rule are derived from it"
            )
        if self.design is None and not self.notes.strip():
            raise ConfigError(
                f"arm {self.arm_id}: no design and no note saying where its runs are "
                "counted instead"
            )
        if self.solver_group is not None and self.ablation != "A8":
            raise ConfigError(
                f"arm {self.arm_id}: carries solver group {self.solver_group!r} but "
                f"belongs to ablation {self.ablation!r}. The three groups are A8's "
                "level set and 10.6.2 discharges X10 by pointing at that row; an arm "
                "that moved the contact parameters outside it would change a safety "
                "number with no preregistered ladder to read it on"
            )


@dataclass(frozen=True)
class ExperimentSpec:
    """P0, P1, E1-E12 or the baseline block.

    `zero_gpu` is load-bearing rather than descriptive: 11.3 fixes the
    unblinding order around it, and `runners.unblinding` reads this field to
    decide whether a stage may run before the gates.
    """

    experiment_id: str
    title: str
    phase: str            # calibration | gate | main | baseline
    zero_gpu: bool
    question: str
    independent_variables: tuple[str, ...]
    endpoints: tuple[EndpointSpec, ...]
    null_distribution: str
    primary_test: str
    bootstrap: str
    arms: tuple[str, ...]
    subset: str | None
    design: Design | None
    prereg_items: tuple[int, ...]
    requires_measurements: tuple[MeasurementRequirement, ...]
    zero_result_exit: str | None
    gpu_hours: float
    cpu_hours: float
    state_distribution: str = ""
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    PHASES = ("calibration", "gate", "main", "baseline", "ablation")

    def __post_init__(self) -> None:
        if self.phase not in self.PHASES:
            raise ConfigError(
                f"{self.experiment_id}: phase {self.phase!r} not in {self.PHASES}"
            )
        if self.zero_gpu and self.gpu_hours != 0.0:
            raise ConfigError(
                f"{self.experiment_id}: declared zero-GPU but budgets "
                f"{self.gpu_hours} GPU-h; 11.3 puts the zero-GPU gates before every "
                "training arm and that ordering is derived from this flag"
            )
        # A-5: the primary null must be the surrogate-defect distribution.
        # Permutation across states degenerates for constant compensation.
        if "permutation" in self.null_distribution and "within_family" not in (
            self.null_distribution
        ):
            raise ConfigError(
                f"{self.experiment_id}: appendix A-5 excludes a permutation null over "
                "states (a constant compensation field permutes to itself). The "
                "primary null is the surrogate-defect distribution; within-family "
                "restricted permutation is a different object and must say so"
            )

    def endpoint(self, name: str) -> EndpointSpec:
        for e in self.endpoints:
            if e.name == name:
                return e
        raise KeyError(
            f"{self.experiment_id} has no endpoint {name!r}; "
            f"declared: {[e.name for e in self.endpoints]}"
        )

    @property
    def primary_endpoints(self) -> tuple[EndpointSpec, ...]:
        return tuple(e for e in self.endpoints if e.kind == "primary")


@dataclass(frozen=True)
class AblationLevel:
    """One level of one row of the 10.6.1 ablation table.

    `incremental_runs` is the count of runs this level adds, not the count it is
    read on. The distinction is the whole arithmetic of the table: A1's main
    level is the pooled arm every other chapter already trained, so it adds
    nothing, and the row's 12 runs are its two non-main levels over the six S6
    cells at one seed. A main level with a non-zero increment would mean the
    main setting had been retrained for the ablation, and the ablation's
    comparison would then no longer be against the arm the paper reports.

    `parameters` is the level's own numbers, in the form the module named by the
    row's `implemented_by` reads them. They are here rather than in Python
    because 11.3 wants the level set preregistered: a level table that lives in
    a module can be edited after the numbers are seen and leaves no trace in the
    configuration the run validated against.
    """

    level_id: str
    label: str
    is_main: bool
    arm: str | None
    incremental_runs: int
    gpu_hours: float
    rationale: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ConfigError(f"level {self.level_id}: no label")
        if not self.rationale.strip():
            raise ConfigError(
                f"level {self.level_id}: no rationale. A level with no stated reason "
                "for being in the ladder cannot be distinguished after the fact from "
                "one chosen because of what it returned"
            )
        if self.incremental_runs < 0 or self.gpu_hours < 0.0:
            raise ConfigError(f"level {self.level_id}: negative run count or budget")
        if self.is_main and self.incremental_runs:
            raise ConfigError(
                f"level {self.level_id}: the main level of a row is the setting the "
                f"main experiments already ran, so it adds no runs; it declares "
                f"{self.incremental_runs}"
            )
        if self.incremental_runs and self.arm is None:
            raise ConfigError(
                f"level {self.level_id}: {self.incremental_runs} runs and no arm. "
                "11.4 makes the model card's arm field mandatory on every summary "
                "row, so a run with no arm has nowhere to be recorded"
            )
        if self.gpu_hours and not self.incremental_runs:
            raise ConfigError(
                f"level {self.level_id}: {self.gpu_hours} GPU-h against zero runs. A "
                "level that reuses another arm's runs reuses its budget too, and "
                "counting the hours twice would overstate the 9.8 total"
            )


@dataclass(frozen=True)
class AblationSpec(ExperimentSpec):
    """One row of the 10.6.1 ablation table, A1-A14.

    An `ExperimentSpec` subclass rather than a parallel type, so that everything
    the loader and the validator already enforce on an experiment -- the
    preregistration items exist, the arms are declared, the endpoints carry an
    estimator, a statistic, a dimension and a verdict rule, the zero-GPU flag
    agrees with the budget -- applies here word for word. 10.6.1's rows are
    preregistered objects: three of them close a confounder, and a sensitivity
    analysis whose levels were not fixed in advance closes nothing.

    Two fields have no counterpart in an experiment.

    `implemented_by` names the code objects the levels move, as `module:symbol`.
    It is what stops a row from becoming a description: A2's levels are the four
    entries of `policies.obs:K_HIST_LEVELS`, and if that tuple changed to three
    entries the row would be sweeping something that no longer exists.

    `confounders_closed` names the X-numbers of 10.6.2 this row is the
    designated design for. The confounder table is prose in the proposal and had
    no machine-readable counterpart; without one, "A8 closes X10" is a sentence
    that nothing checks, and X10 is precisely the confounder whose whole content
    is that a number can be moved by a knob.
    """

    dimension: str = ""
    levels: tuple[AblationLevel, ...] = ()
    implemented_by: tuple[str, ...] = ()
    confounders_closed: tuple[str, ...] = ()
    n_runs: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.phase != "ablation":
            raise ConfigError(
                f"{self.experiment_id}: an ablation's phase is 'ablation', not "
                f"{self.phase!r}. 11.3 orders the unblinding by phase and a row that "
                "claimed a main phase would be scheduled as a main experiment"
            )
        if not self.dimension.strip():
            raise ConfigError(
                f"{self.experiment_id}: no dimension. It is the ablation table's "
                "second column and names what is varied"
            )
        if len(self.levels) < 2:
            raise ConfigError(
                f"{self.experiment_id}: {len(self.levels)} level(s). A one-level "
                "ablation varies nothing"
            )
        ids = [level.level_id for level in self.levels]
        if len(set(ids)) != len(ids):
            raise ConfigError(f"{self.experiment_id}: duplicate level id in {ids}")
        mains = [level.level_id for level in self.levels if level.is_main]
        if len(mains) != 1:
            raise ConfigError(
                f"{self.experiment_id}: {len(mains)} levels marked main ({mains}). "
                "Exactly one level of every row is the setting the main experiments "
                "ran, and it is the reference the other levels are read against"
            )
        total_runs = sum(level.incremental_runs for level in self.levels)
        if total_runs != self.n_runs:
            raise ConfigError(
                f"{self.experiment_id}: the levels add {total_runs} runs and the row "
                f"declares {self.n_runs}. 10.6.1's run column is the sum of its "
                "levels' increments and the subtotal is quoted in the 9.8 budget"
            )
        total_gpu = sum(level.gpu_hours for level in self.levels)
        if abs(total_gpu - self.gpu_hours) > 1e-9:
            raise ConfigError(
                f"{self.experiment_id}: the levels add {total_gpu} GPU-h and the row "
                f"declares {self.gpu_hours}"
            )
        if not self.implemented_by:
            raise ConfigError(
                f"{self.experiment_id}: names no implementation. A row whose levels "
                "are not wired to a parameterisation is a description of an ablation, "
                "not one"
            )
        for code in self.confounders_closed:
            if not (code.startswith("X") and code[1:].isdigit()):
                raise ConfigError(
                    f"{self.experiment_id}: {code!r} is not an X-number of the 10.6.2 "
                    "confounder table"
                )
        for endpoint in self.endpoints:
            # 11.3 item 13 fixes the primary family at five members, all of them
            # in the main experiments. An ablation that promoted an endpoint
            # into it would change the Holm-Bonferroni denominator after the
            # ranks were registered.
            if endpoint.kind == "primary":
                raise ConfigError(
                    f"{self.experiment_id}.{endpoint.name}: an ablation is a "
                    "sensitivity analysis and carries no primary endpoint; the "
                    "primary family has five preregistered members"
                )
            # A-4, as `test_forbidden_paths` states it for the main experiments.
            if (
                self.arms
                and endpoint.dimension in ("amplitude", "probability")
                and endpoint.reference_arm != "B7"
            ):
                raise ConfigError(
                    f"{self.experiment_id}.{endpoint.name}: reference arm "
                    f"{endpoint.reference_arm!r}. Appendix A-4 makes the B7 Oracle "
                    "the only admissible reference point, and an ablation reading a "
                    "Lambda-family or success-rate quantity against a different one "
                    "could not be put beside the main table"
                )

    def level(self, level_id: str) -> AblationLevel:
        for level in self.levels:
            if level.level_id == level_id:
                return level
        raise KeyError(
            f"{self.experiment_id} has no level {level_id!r}; declared: "
            f"{[level.level_id for level in self.levels]}"
        )

    @property
    def main_level(self) -> AblationLevel:
        return next(level for level in self.levels if level.is_main)

    @property
    def level_arms(self) -> tuple[str, ...]:
        return tuple(level.arm for level in self.levels if level.arm is not None)


@dataclass(frozen=True)
class IsolationRung:
    """One rung of the E9 isolation ladder (10.4.9)."""

    rung_id: str
    implementation: str
    expected_isolation: float
    role: str
    split_rule: str
    is_decisive_control: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.expected_isolation <= 1.0):
            raise ConfigError(
                f"{self.rung_id}: I is a share and lies in [0, 1], got "
                f"{self.expected_isolation}"
            )


@dataclass(frozen=True)
class IsolationLadder:
    """I0-I3 plus the decisive control I3' (10.4.9, appendix A-25).

    Only one capacity caliber may be matched. Matching the total parameter count
    and the activated parameter count at once is arithmetically impossible -- a
    fully shared dense rung has activated equal to total, while K disjoint
    modules have one or the other off by a factor of K. The preregistered
    caliber is the activated count, because that is what decides fitting
    capacity; the total count is reported as a covariate and enters the main
    regression as log(total).
    """

    rungs: tuple[IsolationRung, ...]
    matched_quantity: str
    match_tolerance: float
    reported_covariate: str
    isolation_definition: str
    second_independent_variable: str

    def __post_init__(self) -> None:
        ids = tuple(r.rung_id for r in self.rungs)
        if ids != ISOLATION_RUNGS_EXPECTED:
            raise ConfigError(
                f"isolation ladder is {ids}, expected {ISOLATION_RUNGS_EXPECTED}"
            )
        if self.matched_quantity != "activated_parameters":
            raise ConfigError(
                "appendix A-25: exactly one caliber may be matched and it is the "
                f"activated parameter count, not {self.matched_quantity!r}"
            )
        controls = [r for r in self.rungs if r.is_decisive_control]
        if len(controls) != 1 or controls[0].rung_id != "I3prime":
            raise ConfigError(
                "I3' is the decisive control: K disjoint modules with the same "
                "parameter count and the same per-module data volume as I3, split by "
                "a random pooled subset rather than by embodiment identity. Without "
                "it the isolation slope cannot be separated from data fragmentation"
            )
        i3 = self.rung("I3")
        i3p = self.rung("I3prime")
        if i3.expected_isolation != i3p.expected_isolation:
            raise ConfigError(
                "I3' matches I3 in everything but the split rule, so its expected "
                "isolation is the same number"
            )

    def rung(self, rung_id: str) -> IsolationRung:
        for r in self.rungs:
            if r.rung_id == rung_id:
                return r
        raise KeyError(f"no isolation rung {rung_id!r}")


@dataclass(frozen=True)
class SeedTable:
    """The three-layer seed allocation of 11.3.

    Two properties are enforced by the shape of the interface rather than by a
    comment. First, `env_seed` has no `defect_id` parameter (9.1.6): the same
    (task, family, demo index) has bit-identical initial states across the six
    defects, which is what lets one clean-twin demonstration serve all six and
    what makes the paired design paired. Second, `env_seed` and `eval_seed`
    take no arm, while `train_seed` requires one -- that is the 11.3 rule that
    cross-arm comparisons align the environment and evaluation layers and vary
    only the training layer, so an arm difference carries no initial-state
    difference and the training-seed variance lands in the third layer of the
    decomposition.
    """

    env_base: int
    train_base: int
    eval_base: int
    train_seed_index_by_arm: dict[str, tuple[int, ...]]
    n_eval_init_states: int
    seed_pilot_values: tuple[int, ...]
    env_scope: tuple[str, ...]
    eval_scope: tuple[str, ...]
    train_scope: tuple[str, ...]
    notes: str = ""

    def __post_init__(self) -> None:
        if "defect_id" in self.env_scope or "defect" in self.env_scope:
            raise ConfigError(
                "9.1.6: seed_env must not contain defect_id, or the six defects of a "
                "family stop sharing initial states and the paired design collapses"
            )
        if "arm" in self.env_scope or "arm" in self.eval_scope:
            raise ConfigError(
                "11.3: the environment and evaluation layers are aligned across arms, "
                "so neither may be scoped by the arm"
            )
        if "arm" not in self.train_scope:
            raise ConfigError(
                "11.3: the training layer is the one that varies across arms, so it "
                "must be scoped by the arm"
            )
        for base in (self.env_base, self.train_base, self.eval_base):
            if not isinstance(base, int) or base <= 0:
                raise ConfigError(f"seed base must be a positive int, got {base!r}")
        if len({self.env_base, self.train_base, self.eval_base}) != 3:
            raise ConfigError(
                "11.3 forbids sharing one global seed across the three layers; the "
                "three bases are equal or overlapping"
            )

    # -- derivation ------------------------------------------------------
    #
    # A stable hash of the scope tuple, folded into the layer base. Python's
    # built-in hash is salted per process, so it cannot appear here: 9.1.6
    # requires the same (embodiment, task, seed) to reproduce a demonstration
    # byte for byte across processes and machines.

    @staticmethod
    def _fold(base: int, scope: tuple[str, ...], values: tuple[Any, ...]) -> int:
        import hashlib

        payload = "|".join(f"{k}={v}" for k, v in zip(scope, values, strict=True))
        digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
        return (base + int.from_bytes(digest, "big")) % (2**31 - 1)

    def env_seed(self, *, task: str, family: str, demo_index: int) -> int:
        return self._fold(self.env_base, self.env_scope, (task, family, demo_index))

    def eval_seed(self, *, task: str, family: str) -> int:
        return self._fold(self.eval_base, self.eval_scope, (task, family))

    def train_seed(self, *, arm: str, seed_index: int) -> int:
        if arm not in self.train_seed_index_by_arm:
            raise ConfigError(
                f"arm {arm!r} has no training seed allocation in seeds.yaml; 11.3 "
                "requires the three layers to be allocated explicitly in the config"
            )
        allowed = self.train_seed_index_by_arm[arm]
        if seed_index not in allowed:
            raise ConfigError(
                f"arm {arm!r} is allocated seed indices {allowed}, not {seed_index}"
            )
        return self._fold(self.train_base, self.train_scope, (arm, seed_index))

    def n_seeds(self, arm: str) -> int:
        if arm not in self.train_seed_index_by_arm:
            raise ConfigError(f"arm {arm!r} has no training seed allocation in seeds.yaml")
        return len(self.train_seed_index_by_arm[arm])

    def triple(
        self, *, task: str, family: str, demo_index: int, arm: str, seed_index: int
    ):
        """Assemble a `SeedTriple`, the object the teacher and the runners pass around."""
        from portal_trace.types import SeedTriple

        return SeedTriple(
            env=self.env_seed(task=task, family=family, demo_index=demo_index),
            train=self.train_seed(arm=arm, seed_index=seed_index),
            eval=self.eval_seed(task=task, family=family),
        )
