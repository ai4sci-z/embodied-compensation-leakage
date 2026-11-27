"""Read the YAML configuration into the typed schema, and validate it.

Two kinds of check live here. Per-object checks are in the dataclasses'
`__post_init__`; cross-object checks -- an arm's design against the subset it
names, a pool block against the arm that owns it, an experiment's arms against
the arm registry -- need the whole bundle and are done by `ConfigBundle`.

The loader has no `.get(key, default)` anywhere. A missing key raises with the
key's name and the section it belongs to, because a configuration that silently
acquires a default is exactly the failure mode CONTRACT 3.1 exists to stop, and
in this package the difference between `includes_delta_zero` absent and
`includes_delta_zero: false` is the difference between two readings of a null
result (11.3 item 10).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from portal_trace.configs.preregistration import (
    AmendmentLog,
    Preregistration,
    PreregistrationItem,
)
from portal_trace.configs.schema import (
    ABLATIONS_EXPECTED,
    NAMED_SUBSETS_EXPECTED,
    AblationLevel,
    AblationSpec,
    ArmSpec,
    ConfigError,
    Design,
    EndpointSpec,
    ExperimentSpec,
    IsolationLadder,
    IsolationRung,
    MeasurementRequirement,
    PoolComposition,
    SeedTable,
    SubsetSpec,
)

__all__ = ["ABLATIONS_FILENAME", "ConfigBundle", "config_root", "load_bundle"]

# 10.6.1's fourteen rows live beside the experiments because they use the same
# schema and are validated by the same code, and in their own file because they
# carry four fields an experiment does not have.
ABLATIONS_FILENAME = "ablations.yaml"


def config_root() -> Path:
    return Path(__file__).resolve().parent


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"configuration file {path} does not exist")
    with path.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return doc


def _need(blob: dict[str, Any], key: str, where: str) -> Any:
    """Fetch a required key. There is no default form of this function."""
    if key not in blob:
        raise ConfigError(
            f"{where}: required key {key!r} is missing. Present: {sorted(blob)}"
        )
    return blob[key]


def _tuple(value: Any, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise ConfigError(f"{where}: expected a list, got the string {value!r}")
    return tuple(str(v) for v in value)


def _design(blob: Any, where: str) -> Design | None:
    if blob is None:
        return None
    return Design(
        n_families=int(_need(blob, "n_families", where)),
        n_defects=int(_need(blob, "n_defects", where)),
        n_seeds=int(_need(blob, "n_seeds", where)),
    )


# ---------------------------------------------------------------------------
# section readers
# ---------------------------------------------------------------------------


def _load_seeds(path: Path) -> SeedTable:
    doc = _read_yaml(path)
    bases = _need(doc, "bases", str(path))
    scopes = _need(doc, "scopes", str(path))
    allocation = _need(doc, "train_seed_index_by_arm", str(path))
    return SeedTable(
        env_base=int(_need(bases, "env", f"{path}:bases")),
        train_base=int(_need(bases, "train", f"{path}:bases")),
        eval_base=int(_need(bases, "eval", f"{path}:bases")),
        train_seed_index_by_arm={
            str(arm): tuple(int(i) for i in idx) for arm, idx in allocation.items()
        },
        n_eval_init_states=int(_need(doc, "n_eval_init_states", str(path))),
        seed_pilot_values=tuple(int(v) for v in _need(doc, "seed_pilot_values", str(path))),
        env_scope=_tuple(_need(scopes, "env", f"{path}:scopes"), f"{path}:scopes.env"),
        eval_scope=_tuple(_need(scopes, "eval", f"{path}:scopes"), f"{path}:scopes.eval"),
        train_scope=_tuple(_need(scopes, "train", f"{path}:scopes"), f"{path}:scopes.train"),
        notes=str(doc.get("notes", "")),
    )


def _load_subsets(path: Path) -> tuple[dict[str, SubsetSpec], dict[str, PoolComposition],
                                       dict[str, Any]]:
    doc = _read_yaml(path)
    subsets: dict[str, SubsetSpec] = {}
    for name, blob in _need(doc, "subsets", str(path)).items():
        where = f"{path}:subsets.{name}"
        subsets[name] = SubsetSpec(
            name=name,
            families=_tuple(_need(blob, "families", where), where),
            defects=_tuple(blob.get("defects"), where),
            rationale=str(_need(blob, "rationale", where)),
            family_level_only=bool(blob.get("family_level_only", False)),
        )
    if tuple(subsets) != NAMED_SUBSETS_EXPECTED:
        raise ConfigError(
            f"{path}: the five named subsets of 10.4.0 are "
            f"{NAMED_SUBSETS_EXPECTED}, found {tuple(subsets)}. They are registered "
            "in the preregistration and referenced by name throughout, so the set "
            "cannot grow silently"
        )

    # The file's own header calls this line the defect axis of the whole study,
    # and nothing downstream reads it, so a disagreement with the registry would
    # sit here as documentation that contradicts the code it documents.
    from portal_trace.defects.params import CORE_DEFECTS

    declared = _tuple(_need(doc, "core_defects", str(path)), str(path))
    if declared != tuple(CORE_DEFECTS):
        raise ConfigError(
            f"{path}: core_defects is {list(declared)} but "
            f"defects.params.CORE_DEFECTS is {list(CORE_DEFECTS)}. 9.2.6's "
            "contingency rule decides which six those are and params.py records "
            "the decision; this file only restates it"
        )

    pools: dict[str, PoolComposition] = {}
    for arm, blob in _need(doc, "pools", str(path)).items():
        where = f"{path}:pools.{arm}"
        # Explicit membership test rather than .get(): 11.3 item 10 makes the
        # delta = 0 flag change the meaning of a null result, so an absent flag
        # must be an error and not False.
        if "includes_delta_zero" not in blob:
            raise ConfigError(
                f"{where}: `includes_delta_zero` is required. 11.3 item 10: with the "
                "clean twin in the pool, route A's anchoring holds by construction "
                "and a leakage reading means the algorithm did not use an available "
                "anchor; without it the same number means the information was not "
                "there. That inverts the reading of a null result in E7 and E8, so "
                "the flag has no default"
            )
        if "target_in_pool" not in blob:
            raise ConfigError(f"{where}: `target_in_pool` is required (11.3 item 10)")
        pools[arm] = PoolComposition(
            arm=arm,
            sources=_tuple(_need(blob, "sources", where), where),
            includes_delta_zero=bool(blob["includes_delta_zero"]),
            target_in_pool=bool(blob["target_in_pool"]),
            reading_consequence=str(_need(blob, "reading_consequence", where)),
            n_target_demos=int(_need(blob, "n_target_demos", where)),
            n_source_demos=int(_need(blob, "n_source_demos", where)),
            k_sources=int(_need(blob, "k_sources", where)),
        )
    return subsets, pools, doc


def _load_arms(path: Path, pools: dict[str, PoolComposition]) -> dict[str, ArmSpec]:
    doc = _read_yaml(path)
    arms: dict[str, ArmSpec] = {}
    for arm_id, blob in _need(doc, "arms", str(path)).items():
        where = f"{path}:arms.{arm_id}"
        if arm_id not in pools:
            raise ConfigError(
                f"{where}: no pool composition in subsets.yaml. 11.3 item 10 requires "
                "one per training arm"
            )
        arms[arm_id] = ArmSpec(
            arm_id=arm_id,
            role=str(_need(blob, "role", where)),
            subset=str(_need(blob, "subset", where)),
            design=_design(_need(blob, "design", where), f"{where}.design"),
            pool=pools[arm_id],
            n_runs=int(_need(blob, "n_runs", where)),
            gpu_hours=float(_need(blob, "gpu_hours", where)),
            defect_axis=_tuple(blob.get("defect_axis"), where),
            defect_axis_kind=str(blob.get("defect_axis_kind", "core")),
            defect_axis_rationale=str(blob.get("defect_axis_rationale", "")),
            reference_arm=blob.get("reference_arm"),
            architecture=str(blob.get("architecture", "")),
            solver_group=blob.get("solver_group"),
            ablation=blob.get("ablation"),
            notes=str(blob.get("notes", "")),
        )
    return arms


def _load_measurement_requirements(
    blob: Any, where: str
) -> tuple[MeasurementRequirement, ...]:
    out = []
    for entry in blob or ():
        out.append(
            MeasurementRequirement(
                key=str(_need(entry, "key", where)),
                stages=_tuple(_need(entry, "stages", where), where),
                fixed_scope={str(k): str(v) for k, v in (entry.get("fixed_scope") or {}).items()},
                expand={
                    str(k): tuple(str(x) for x in v)
                    for k, v in (entry.get("expand") or {}).items()
                },
            )
        )
    return tuple(out)


def _load_endpoints(blob: Any, where: str) -> tuple[EndpointSpec, ...]:
    out = []
    for entry in blob or ():
        name = str(_need(entry, "name", where))
        w = f"{where}.{name}"
        out.append(
            EndpointSpec(
                name=name,
                kind=str(_need(entry, "kind", w)),
                estimator=str(_need(entry, "estimator", w)),
                statistic=str(_need(entry, "statistic", w)),
                dimension=str(_need(entry, "dimension", w)),
                verdict_rule=str(_need(entry, "verdict_rule", w)),
                reference_arm=entry.get("reference_arm"),
                mde=None if entry.get("mde") is None else float(entry["mde"]),
                holm_bonferroni_rank=entry.get("holm_bonferroni_rank"),
                notes=str(entry.get("notes", "")),
            )
        )
    return tuple(out)


def _load_experiments(paths: list[Path]) -> dict[str, ExperimentSpec]:
    experiments: dict[str, ExperimentSpec] = {}
    for path in paths:
        doc = _read_yaml(path)
        for exp_id, blob in _need(doc, "experiments", str(path)).items():
            where = f"{path}:experiments.{exp_id}"
            if exp_id in experiments:
                raise ConfigError(f"{where}: experiment {exp_id} declared twice")
            experiments[exp_id] = ExperimentSpec(
                experiment_id=exp_id,
                title=str(_need(blob, "title", where)),
                phase=str(_need(blob, "phase", where)),
                zero_gpu=bool(_need(blob, "zero_gpu", where)),
                question=str(_need(blob, "question", where)),
                independent_variables=_tuple(
                    _need(blob, "independent_variables", where), where
                ),
                endpoints=_load_endpoints(_need(blob, "endpoints", where), where),
                null_distribution=str(_need(blob, "null_distribution", where)),
                primary_test=str(_need(blob, "primary_test", where)),
                bootstrap=str(_need(blob, "bootstrap", where)),
                arms=_tuple(_need(blob, "arms", where), where),
                subset=blob.get("subset"),
                design=_design(_need(blob, "design", where), f"{where}.design"),
                prereg_items=tuple(int(i) for i in _need(blob, "prereg_items", where)),
                requires_measurements=_load_measurement_requirements(
                    _need(blob, "requires_measurements", where), where
                ),
                zero_result_exit=blob.get("zero_result_exit"),
                gpu_hours=float(_need(blob, "gpu_hours", where)),
                cpu_hours=float(_need(blob, "cpu_hours", where)),
                state_distribution=str(blob.get("state_distribution", "")),
                notes=str(blob.get("notes", "")),
            )
    return experiments


def _module_file(module_name: str) -> Path | None:
    """Where a dotted module name would live, if it is inside this package.

    Used instead of importing it. `load_bundle` runs in every test session and
    in every CLI invocation, and resolving A4's implementation would pull torch
    and robomimic into a process that only wanted to read text -- 11.3's point
    about the configuration is that its consistency is checkable before any
    environment is built. The symbol inside the module is resolved by
    `ablations.resolve_symbol`, which the ablation tests call.
    """
    if not module_name.startswith("portal_trace."):
        return None
    parts = module_name.split(".")[1:]
    root = Path(__file__).resolve().parent.parent
    candidate = root.joinpath(*parts)
    if candidate.with_suffix(".py").exists():
        return candidate.with_suffix(".py")
    if (candidate / "__init__.py").exists():
        return candidate / "__init__.py"
    return None


def _load_levels(blob: Any, where: str) -> tuple[AblationLevel, ...]:
    out = []
    for entry in blob or ():
        level_id = str(_need(entry, "level_id", where))
        w = f"{where}.{level_id}"
        # Membership tests rather than .get(): `is_main` decides which level the
        # other levels are read against, and an absent flag defaulting to False
        # would leave a row with no reference at all.
        if "is_main" not in entry:
            raise ConfigError(
                f"{w}: `is_main` is required. Exactly one level of every 10.6.1 row "
                "is the setting the main experiments ran, and the row's whole reading "
                "is the other levels against it"
            )
        out.append(
            AblationLevel(
                level_id=level_id,
                label=str(_need(entry, "label", w)),
                is_main=bool(entry["is_main"]),
                arm=None if entry.get("arm") is None else str(entry["arm"]),
                incremental_runs=int(_need(entry, "incremental_runs", w)),
                gpu_hours=float(_need(entry, "gpu_hours", w)),
                rationale=str(_need(entry, "rationale", w)),
                parameters=dict(entry.get("parameters") or {}),
            )
        )
    return tuple(out)


def _load_ablations(path: Path) -> dict[str, AblationSpec]:
    """Read `experiments/ablations.yaml` with the strictness of an experiment.

    Every key `_load_experiments` requires is required here too, and the four
    ablation-only keys on top of them. The row set is checked against
    `ABLATIONS_EXPECTED` rather than merely counted, because 10.6.1's numbering
    is referenced from the confounder table and from the budget subtotal.
    """
    doc = _read_yaml(path)
    ablations: dict[str, AblationSpec] = {}
    for ablation_id, blob in _need(doc, "ablations", str(path)).items():
        where = f"{path}:ablations.{ablation_id}"
        implemented_by = _tuple(_need(blob, "implemented_by", where), where)
        for reference in implemented_by:
            module_name, sep, symbol = reference.partition(":")
            if not sep or not symbol:
                raise ConfigError(
                    f"{where}: implemented_by entry {reference!r} is not a "
                    "`module:symbol` reference"
                )
            if _module_file(module_name) is None:
                raise ConfigError(
                    f"{where}: implemented_by names {module_name!r}, which is not a "
                    "module of this package. A row has to point at the "
                    "parameterisation its levels move, or the level set is a "
                    "description rather than a sweep"
                )
        ablations[ablation_id] = AblationSpec(
            experiment_id=ablation_id,
            title=str(_need(blob, "title", where)),
            phase=str(_need(blob, "phase", where)),
            zero_gpu=bool(_need(blob, "zero_gpu", where)),
            question=str(_need(blob, "question", where)),
            independent_variables=_tuple(
                _need(blob, "independent_variables", where), where
            ),
            endpoints=_load_endpoints(_need(blob, "endpoints", where), where),
            null_distribution=str(_need(blob, "null_distribution", where)),
            primary_test=str(_need(blob, "primary_test", where)),
            bootstrap=str(_need(blob, "bootstrap", where)),
            arms=_tuple(_need(blob, "arms", where), where),
            subset=blob.get("subset"),
            design=_design(_need(blob, "design", where), f"{where}.design"),
            prereg_items=tuple(int(i) for i in _need(blob, "prereg_items", where)),
            requires_measurements=_load_measurement_requirements(
                _need(blob, "requires_measurements", where), where
            ),
            zero_result_exit=blob.get("zero_result_exit"),
            gpu_hours=float(_need(blob, "gpu_hours", where)),
            cpu_hours=float(_need(blob, "cpu_hours", where)),
            state_distribution=str(blob.get("state_distribution", "")),
            notes=str(blob.get("notes", "")),
            dimension=str(_need(blob, "dimension", where)),
            levels=_load_levels(_need(blob, "levels", where), f"{where}.levels"),
            implemented_by=implemented_by,
            confounders_closed=_tuple(
                _need(blob, "confounders_closed", where), where
            ),
            n_runs=int(_need(blob, "n_runs", where)),
        )
    if tuple(ablations) != ABLATIONS_EXPECTED:
        raise ConfigError(
            f"{path}: 10.6.1's ablation table is {ABLATIONS_EXPECTED}, found "
            f"{tuple(ablations)}. The numbering is cited by the confounder table of "
            "10.6.2 and by the 9.8 budget subtotal, so a row cannot be dropped or "
            "renumbered here alone"
        )
    return ablations


def _load_isolation(path: Path) -> IsolationLadder:
    doc = _read_yaml(path)
    rungs = []
    for entry in _need(doc, "rungs", str(path)):
        where = f"{path}:rungs"
        rungs.append(
            IsolationRung(
                rung_id=str(_need(entry, "rung_id", where)),
                implementation=str(_need(entry, "implementation", where)),
                expected_isolation=float(_need(entry, "expected_isolation", where)),
                role=str(_need(entry, "role", where)),
                split_rule=str(_need(entry, "split_rule", where)),
                is_decisive_control=bool(entry.get("is_decisive_control", False)),
            )
        )
    return IsolationLadder(
        rungs=tuple(rungs),
        matched_quantity=str(_need(doc, "matched_quantity", str(path))),
        match_tolerance=float(_need(doc, "match_tolerance", str(path))),
        reported_covariate=str(_need(doc, "reported_covariate", str(path))),
        isolation_definition=str(_need(doc, "isolation_definition", str(path))),
        second_independent_variable=str(
            _need(doc, "second_independent_variable", str(path))
        ),
    )


def _load_preregistration(path: Path) -> Preregistration:
    doc = _read_yaml(path)
    items = tuple(
        PreregistrationItem(
            number=int(_need(entry, "number", str(path))),
            title=str(_need(entry, "title", str(path))),
            text=str(_need(entry, "text", str(path))),
            proposal_ref=str(_need(entry, "proposal_ref", str(path))),
            blocks=_tuple(_need(entry, "blocks", str(path)), str(path)),
            source=str(entry.get("source", "")),
        )
        for entry in _need(doc, "items", str(path))
    )
    return Preregistration(
        version=str(_need(doc, "version", str(path))),
        frozen_at=str(_need(doc, "frozen_at", str(path))),
        git_tag=str(_need(doc, "git_tag", str(path))),
        items=items,
        hypotheses={str(k): str(v) for k, v in _need(doc, "hypotheses", str(path)).items()},
        exits={str(k): str(v) for k, v in _need(doc, "exits", str(path)).items()},
        retired_hypotheses={
            str(k): str(v) for k, v in (doc.get("retired_hypotheses") or {}).items()
        },
        primary_family=_tuple(_need(doc, "primary_family", str(path)), str(path)),
        holm_bonferroni_order=_tuple(
            _need(doc, "holm_bonferroni_order", str(path)), str(path)
        ),
        verdict_rules={
            str(k): str(v) for k, v in _need(doc, "verdict_rules", str(path)).items()
        },
        ci_thresholds={
            str(k): float(v) for k, v in _need(doc, "ci_thresholds", str(path)).items()
        },
        unblinding_order=_tuple(_need(doc, "unblinding_order", str(path)), str(path)),
        path=path,
    )


# ---------------------------------------------------------------------------
# the bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigBundle:
    """Everything in `configs/`, loaded and cross-checked."""

    root: Path
    seeds: SeedTable
    subsets: dict[str, SubsetSpec]
    pools: dict[str, PoolComposition]
    arms: dict[str, ArmSpec]
    experiments: dict[str, ExperimentSpec]
    ablations: dict[str, AblationSpec]
    isolation: IsolationLadder
    preregistration: Preregistration
    amendments: AmendmentLog
    subset_assignment: dict[str, tuple[str, ...]]

    def arm(self, arm_id: str) -> ArmSpec:
        try:
            return self.arms[arm_id]
        except KeyError:
            raise ConfigError(
                f"unknown arm {arm_id!r}; declared: {sorted(self.arms)}"
            ) from None

    def experiment(self, experiment_id: str) -> ExperimentSpec:
        """P0, P1, B, E1-E12 or one of the fourteen ablations.

        One namespace, so that `--experiment A8` reaches the validator through
        the same path as `--experiment E1` and is subject to the same checks.
        The two dictionaries are kept apart because an ablation carries four
        extra fields and `ablation()` returns them typed, but a caller that only
        wants "the thing being run" does not have to know which kind it is.
        """
        if experiment_id in self.experiments:
            return self.experiments[experiment_id]
        if experiment_id in self.ablations:
            return self.ablations[experiment_id]
        raise ConfigError(
            f"unknown experiment {experiment_id!r}; declared: "
            f"{sorted(self.experiments)} and ablations {sorted(self.ablations)}"
        )

    def ablation(self, ablation_id: str) -> AblationSpec:
        try:
            return self.ablations[ablation_id]
        except KeyError:
            raise ConfigError(
                f"unknown ablation {ablation_id!r}; 10.6.1 declares "
                f"{sorted(self.ablations)}"
            ) from None

    @property
    def all_experiments(self) -> dict[str, ExperimentSpec]:
        """Experiments and ablations together, for the checks that span both."""
        return {**self.experiments, **self.ablations}

    def solver_group(self, arm_id: str) -> Any:
        """The A8 contact-parameter group this arm's environments are built with.

        Returns a `SolverGroup` for every arm, including the ones that do not
        name a level: the default group is the identity pair and is still
        applied, so a solver-group arm and a main arm differ by the value of the
        scales and not by whether the write happened. That is the same
        discipline CONTRACT 3.3 imposes on the clean twin, and it is what makes
        "the ablation ran" checkable from the evaluation record.

        The parameters come from A8's level block, never from a table in the
        environment builder: 10.6.2 discharges X10 by pointing at A8's
        preregistered ladder, and a second copy of the numbers would let the
        thing that ran drift away from the thing that was registered.
        """
        from portal_trace.ablations.solver import SolverGroup

        named = self.arm(arm_id).solver_group
        ablation = self.ablation("A8")
        wanted = named if named is not None else next(
            level.level_id for level in ablation.levels if level.is_main
        )
        for level in ablation.levels:
            if level.level_id == wanted:
                return SolverGroup.from_parameters(
                    level.parameters, where=f"A8.{level.level_id}"
                )
        raise ConfigError(
            f"arm {arm_id}: solver group {named!r} is not one of A8's levels "
            f"{[level.level_id for level in ablation.levels]}. The contact "
            "parameters are only interpretable on that ladder, because the safety "
            "conclusion is read as an agreement across its three rungs"
        )

    def closes_confounder(self, code: str) -> tuple[str, ...]:
        """Which ablations are the designated design for confounder `code`.

        The 10.6.2 table is prose in the proposal, and this is the only
        machine-readable direction of it. X10 in particular -- the solver's
        freedom over every safety number -- is closed by A8 and by nothing else,
        so an empty answer here is a design gap and not a lookup miss.
        """
        return tuple(
            ablation_id
            for ablation_id, ablation in sorted(self.ablations.items())
            if code in ablation.confounders_closed
        )

    def cells(self, arm_id: str) -> tuple[tuple[str, str], ...]:
        """The (family, defect) cells the arm runs -- preregistration item 14."""
        arm = self.arm(arm_id)
        subset = self.subsets[arm.subset]
        if arm.defect_axis:
            return subset.cells_with(arm.defect_axis)
        return subset.cells()

    def cross_check(self) -> None:
        """Checks that need more than one section."""
        problems: list[str] = []

        for arm in self.arms.values():
            if arm.subset not in self.subsets:
                problems.append(
                    f"arm {arm.arm_id}: subset {arm.subset!r} is not one of the five "
                    f"named subsets {NAMED_SUBSETS_EXPECTED}"
                )
                continue
            subset = self.subsets[arm.subset]
            if subset.family_level_only and not arm.defect_axis:
                problems.append(
                    f"arm {arm.arm_id}: {subset.name} is a family-level subset, so the "
                    "arm has to state its own defect axis (preregistration item 14)"
                )
                continue
            if arm.defect_axis and arm.defect_axis_kind == "core" and not subset.family_level_only:
                extra = sorted(set(arm.defect_axis) - set(subset.defects))
                if extra:
                    problems.append(
                        f"arm {arm.arm_id}: defect axis {extra} is not inside "
                        f"{subset.name}'s defect list {list(subset.defects)}"
                    )
            if arm.design is not None:
                cells = self.cells(arm.arm_id)
                m = len({f for f, _ in cells})
                d = len({x for _, x in cells})
                if (m, d) != (arm.design.n_families, arm.design.n_defects):
                    problems.append(
                        f"arm {arm.arm_id}: declares design {arm.design} but its cells "
                        f"give M={m}, D={d}. 11.4 makes the row's MDE derive from "
                        "(M, D, S), so the two cannot disagree"
                    )
                allocated = self.seeds.train_seed_index_by_arm.get(arm.arm_id)
                if allocated is None:
                    problems.append(
                        f"arm {arm.arm_id}: no training-seed allocation in seeds.yaml; "
                        "11.3 requires the three layers to be allocated explicitly"
                    )
                elif len(allocated) != arm.design.n_seeds:
                    problems.append(
                        f"arm {arm.arm_id}: design says S={arm.design.n_seeds} but "
                        f"seeds.yaml allocates {len(allocated)} training seeds "
                        f"{list(allocated)}"
                    )
            if arm.reference_arm is not None and arm.reference_arm not in self.arms:
                problems.append(
                    f"arm {arm.arm_id}: reference arm {arm.reference_arm!r} is not declared"
                )

        for exp in self.all_experiments.values():
            for arm_id in exp.arms:
                if arm_id not in self.arms:
                    problems.append(
                        f"{exp.experiment_id}: arm {arm_id!r} is not in arms.yaml"
                    )
            for n in exp.prereg_items:
                try:
                    self.preregistration.item(n)
                except Exception as exc:  # noqa: BLE001 - collected, re-raised below
                    problems.append(f"{exp.experiment_id}: {exc}")
            for e in exp.endpoints:
                if e.kind in ("primary", "secondary") and e.reference_arm is None:
                    # A-4: the reference point discipline. Endpoints that are not
                    # differences against a policy (E5's T_comp, E12's variance
                    # ratio, P1's gates) legitimately have none, and say so by
                    # living in experiments with no arms.
                    if exp.arms:
                        problems.append(
                            f"{exp.experiment_id}.{e.name}: a policy-side endpoint in "
                            "an experiment with training arms must name its reference "
                            "arm. Appendix A-4 makes B7 the only admissible one, and "
                            "every share decomposition must use the same one"
                        )
                if e.reference_arm is not None and e.reference_arm not in self.arms:
                    problems.append(
                        f"{exp.experiment_id}.{e.name}: reference arm "
                        f"{e.reference_arm!r} is not declared"
                    )

        # The primary hypothesis family's members must exist as endpoints
        # somewhere, or their Holm-Bonferroni rank ranks nothing.
        declared = {e.name for exp in self.experiments.values() for e in exp.endpoints}
        missing = sorted(set(self.preregistration.primary_family) - declared)
        if missing:
            problems.append(
                f"primary hypothesis family members {missing} are preregistered but no "
                "experiment declares an endpoint by that name"
            )
        problems.extend(self._primary_family_agrees())
        # An ablation endpoint may not reuse a main endpoint's name. The Holm-
        # Bonferroni ranks and the MDE table of 10.7.8 are keyed by name, and a
        # second object under the same name would be read against a threshold
        # that was registered for the first one.
        for ablation in self.ablations.values():
            for endpoint in ablation.endpoints:
                if endpoint.name in declared:
                    problems.append(
                        f"{ablation.experiment_id}.{endpoint.name}: an experiment "
                        "already declares an endpoint by that name, and 10.7.8's MDE "
                        "table is keyed by name"
                    )

        problems.extend(self._ablation_cross_checks())
        problems.extend(self._check_defect_level_scopes())
        problems.extend(self._check_subset_families())
        problems.extend(self._check_subset_defects())

        if problems:
            raise ConfigError(
                "configuration cross-checks failed:\n  " + "\n  ".join(problems)
            )

    def _primary_family_agrees(self) -> list[str]:
        """The three places that name the primary family must name the same one.

        They are `prereg.yaml`, the `kind: primary` endpoints in the experiment
        configs, and `eval.stats.PRIMARY_HYPOTHESIS_FAMILY`. Each of them was
        self-consistent while the three disagreed with each other: the configs
        carried the safety endpoint as a Holm member and the E9 isolation slope
        as a secondary one, which is 10.7.6 backwards, and the statistics module
        had the right five under a different set of names, so nothing compared
        them. The disagreement is invisible at the point of use -- `holm_bonferroni`
        would have run over five plausible labels and controlled the family-wise
        rate of a family the preregistration never registered.
        """
        from portal_trace.eval.stats import PRIMARY_HYPOTHESIS_FAMILY

        problems: list[str] = []
        registered = self.preregistration.primary_family
        marked = {
            e.name
            for exp in self.experiments.values()
            for e in exp.endpoints
            if e.kind == "primary"
        }
        if marked != set(registered):
            problems.append(
                f"the endpoints marked kind: primary are {sorted(marked)} but the "
                f"preregistered family is {sorted(registered)}. 10.7.6 fixes the "
                "family at five members; an endpoint marked primary outside it would "
                "be reported as confirmatory without a Holm rank, and a member not "
                "marked primary would be dropped from the family that was registered"
            )
        if tuple(PRIMARY_HYPOTHESIS_FAMILY) != tuple(registered):
            problems.append(
                f"eval.stats.PRIMARY_HYPOTHESIS_FAMILY is {list(PRIMARY_HYPOTHESIS_FAMILY)} "
                f"but prereg.yaml registers {list(registered)}. `holm_bonferroni` "
                "checks the p values it is handed against the module tuple, so a "
                "divergence here means the test that runs is not the test that was "
                "preregistered"
            )
        order = self.preregistration.holm_bonferroni_order
        for exp in self.experiments.values():
            for endpoint in exp.endpoints:
                if endpoint.name not in order:
                    if endpoint.holm_bonferroni_rank is not None:
                        problems.append(
                            f"{exp.experiment_id}.{endpoint.name} carries Holm rank "
                            f"{endpoint.holm_bonferroni_rank} but is not a member of "
                            "the preregistered family"
                        )
                    continue
                expected = order.index(endpoint.name) + 1
                if endpoint.holm_bonferroni_rank != expected:
                    problems.append(
                        f"{exp.experiment_id}.{endpoint.name} carries Holm rank "
                        f"{endpoint.holm_bonferroni_rank}, but prereg.yaml puts it at "
                        f"position {expected} of the registered order"
                    )
        return problems

    def _check_subset_families(self) -> list[str]:
        """A subset's families are registry identifiers, not the 9.3.1 prose.

        The proposal's table calls them IIWA14, xArm7 and FR3; the registry
        resolves those against what robosuite 1.5.2 actually ships and ends up
        with `IIWA`, `XArm7` and -- FR3 being absent from the package -- with
        `Kinova3` promoted into its slot. A subset that carries the prose names
        declares a family axis nothing can run on, and because a subset is only
        read when a run is being resolved, it would surface as an unknown family
        at collection time rather than here.
        """
        from portal_trace.registry.families import MAIN_FAMILIES

        valid = set(MAIN_FAMILIES)
        problems: list[str] = []
        for name, subset in self.subsets.items():
            unknown = sorted(set(subset.families) - valid)
            if unknown:
                problems.append(
                    f"subset {name}: {unknown} are not main-experiment families. "
                    f"The six of 9.3.3 are {sorted(valid)}; the registry's "
                    "resolution_report() says which proposal row each one answers"
                )
        return problems

    def _check_subset_defects(self) -> list[str]:
        """The subsets' defect axis is the six of `defects.params.CORE_DEFECTS`.

        Which six is not a stable fact of the proposal. 9.2.6's contingency rule
        moved D6 out and D8 in after the calibration sweep priced D6's stiction
        offset at two orders of magnitude over its gate, and `params.py` records
        that; `subsets.yaml` was written before it and kept D6. Nothing compared
        the two, and the consequence is quiet rather than loud: S36 would still
        name six defects and still multiply out to n = 36, so every count in the
        power calculation stays right while a third of the runs collect a defect
        the twin registry has no cells for.

        Two invariants are checked. Each subset draws from the core six, and the
        union over subsets is the core six exactly -- the second is what catches
        a defect that was dropped everywhere rather than misspelled in one place.
        """
        from portal_trace.defects.params import CORE_DEFECTS
        from portal_trace.registry.twins import N_TWIN_CELLS

        core = set(CORE_DEFECTS)
        problems: list[str] = []
        seen: set[str] = set()
        for name, subset in self.subsets.items():
            seen.update(subset.defects)
            outside = sorted(set(subset.defects) - core)
            if outside:
                problems.append(
                    f"subset {name}: {outside} are not core defects. 9.2.6 fixes the "
                    f"six at {sorted(core)}, and the twin registry has cells for "
                    "those and no others"
                )
        if seen != core:
            problems.append(
                f"the subsets between them cover {sorted(seen)}, which is not the "
                f"core six {sorted(core)}. A core defect no subset carries is a "
                "column of the design that no arm will ever fill"
            )
        full = self.subsets.get("S36")
        if full is not None and len(full.families) * len(full.defects) != N_TWIN_CELLS:
            problems.append(
                f"subset S36 is {len(full.families)} families x {len(full.defects)} "
                f"defects = {len(full.families) * len(full.defects)}, and the twin "
                f"registry counts {N_TWIN_CELLS} cells. 10.4.0 names the subset after "
                "that product and 11.3's MDE is computed from it"
            )
        return problems

    def _check_defect_level_scopes(self) -> list[str]:
        """`defect.observable`'s level scope is the catalogue index, spelled one way.

        The lock key is `defect.observable.{defect}.{level}.{family}`, and the
        level that fills it is the 9.2.10 index the catalogue assigns -- "0"
        through "4" for a five-level ladder. A row that spells it "L1" asks for
        an entry the calibration sweep will never write, and because the lock
        starts empty that mismatch is invisible until P0 has run: the sweep
        would finish, write its 180 entries, and the ablation would still stop
        on a missing measurement. Checking the token against the catalogue is
        what makes the disagreement fail at load instead.
        """
        from portal_trace.defects.params import DEFECTS, defect

        valid = {str(level.index) for code in DEFECTS for level in defect(code).levels}
        problems: list[str] = []
        for spec in (*self.experiments.values(), *self.ablations.values()):
            for requirement in spec.requires_measurements:
                if not requirement.key.startswith("defect.observable."):
                    continue
                tokens = set(requirement.expand.get("level", ()))
                if "level" in requirement.fixed_scope:
                    tokens.add(requirement.fixed_scope["level"])
                unknown = sorted(tokens - valid)
                if unknown:
                    problems.append(
                        f"{spec.experiment_id}: {requirement.key} is scoped by the "
                        f"9.2.10 level index, and {unknown} are not level indices "
                        f"(the catalogue has {sorted(valid)}). A level spelled any "
                        "other way asks the lock file for an entry the calibration "
                        "sweep does not write"
                    )
        return problems

    def _ablation_cross_checks(self) -> list[str]:
        """The checks that need the ablation table and the arm registry at once.

        Kept apart from the loop above only for length: they are collected into
        the same report and are as fatal as the rest.
        """
        problems: list[str] = []
        for ablation in self.ablations.values():
            for level in ablation.levels:
                where = f"{ablation.experiment_id}.{level.level_id}"
                if level.arm is not None and level.arm not in self.arms:
                    problems.append(f"{where}: arm {level.arm!r} is not in arms.yaml")
                    continue
                if level.arm is not None and level.arm not in ablation.arms:
                    problems.append(
                        f"{where}: names arm {level.arm!r}, which is not in the row's "
                        "own arm list. The row's arms are what the validator resolves "
                        "a run against"
                    )
                try:
                    self._check_level_parameters(ablation.experiment_id, level, where)
                except (ConfigError, ValueError) as exc:
                    problems.append(str(exc))
        problems.extend(self._check_capacity_architectures())
        return problems

    def _check_capacity_architectures(self) -> list[str]:
        """A4's level set and its throughput requirement must name the same arches.

        `architecture` on an A4 level is a lock-file key, not the prose 11.4's
        model card carries -- `train.throughput.{arch}` is scoped by it, and the
        three levels are exactly the three architectures whose throughput has to
        have been measured before A4 can be trained. Tying the two together here
        is what stops a level from being added without its measurement, which is
        the shape a defaulted throughput would take.
        """
        ablation = self.ablations.get("A4")
        if ablation is None:
            return []
        levels = {str(level.parameters.get("architecture")) for level in ablation.levels}
        declared: set[str] = set()
        for requirement in ablation.requires_measurements:
            if requirement.key != "train.throughput.{arch}":
                continue
            declared |= set(requirement.expand.get("arch", ()))
            if "arch" in requirement.fixed_scope:
                declared.add(requirement.fixed_scope["arch"])
        if levels != declared:
            return [
                f"A4: the levels name architectures {sorted(levels)} and the row's "
                f"train.throughput requirement names {sorted(declared)}. The lock "
                "entry is scoped by that key, so a level outside the requirement "
                "would train under a throughput nobody measured"
            ]
        return []

    def _check_level_parameters(
        self, ablation_id: str, level: AblationLevel, where: str
    ) -> None:
        """Parse the level block with the module that owns it, and cross-check.

        A5's demonstration counts are checked against the arm's pool table,
        which is what `runners.train` actually reads, and A8's default group is
        checked to be the identity. A4's architecture strings are checked
        against the row's own throughput requirement by
        `_check_capacity_architectures`, because they are lock-file keys.

        All three imports are of the import-light halves of their modules --
        `capacity_nets`, which is the one that needs torch, is resolved by
        `ablations.resolve_symbol` at run time instead.
        """
        if ablation_id == "A4":
            from portal_trace.ablations.capacity import CapacitySpec

            CapacitySpec.from_parameters(level.parameters, where=where)
        elif ablation_id == "A5":
            from portal_trace.ablations.data_volume import (
                DataVolumeSpec,
                check_pool_matches_level,
            )

            spec = DataVolumeSpec.from_parameters(level.parameters, where=where)
            if level.arm is not None:
                check_pool_matches_level(self.arms[level.arm].pool, spec)
        elif ablation_id == "A8":
            from portal_trace.ablations.solver import SolverGroup

            group = SolverGroup.from_parameters(level.parameters, where=where)
            if level.is_main and not group.is_identity:
                raise ConfigError(
                    f"{where}: A8's main group is the model as robosuite compiles it, "
                    f"so its scales are 1.0; it declares "
                    f"({group.solref_timeconst_scale}, {group.solimp_gap_scale}). "
                    "Writing absolute values over the compiled ones would make the "
                    "default group a different environment from the one every other "
                    "chapter ran in"
                )


def load_bundle(root: Path | str | None = None) -> ConfigBundle:
    """Load and cross-check the whole configuration."""
    r = Path(root) if root is not None else config_root()
    seeds = _load_seeds(r / "seeds.yaml")
    subsets, pools, subsets_doc = _load_subsets(r / "subsets.yaml")
    arms = _load_arms(r / "arms.yaml", pools)
    exp_dir = r / "experiments"
    ablations_path = exp_dir / ABLATIONS_FILENAME
    exp_paths = [p for p in sorted(exp_dir.glob("*.yaml")) if p != ablations_path]
    if not exp_paths:
        raise ConfigError(f"{exp_dir} contains no experiment configuration")
    if not ablations_path.exists():
        raise ConfigError(
            f"{ablations_path} is missing. 10.6.1's fourteen rows are part of the "
            "design, not an optional extra: 10.6.2 discharges X10 by pointing at A8 "
            "and the 9.8 budget counts their runs"
        )
    experiments = _load_experiments(exp_paths)
    ablations = _load_ablations(ablations_path)
    isolation = _load_isolation(r / "isolation" / "ladder.yaml")
    prereg = _load_preregistration(r / "preregistration" / "prereg.yaml")
    amendments = AmendmentLog.load(r / "preregistration" / "amendments.yaml")
    bundle = ConfigBundle(
        root=r,
        seeds=seeds,
        subsets=subsets,
        pools=pools,
        arms=arms,
        experiments=experiments,
        ablations=ablations,
        isolation=isolation,
        preregistration=prereg,
        amendments=amendments,
        subset_assignment={
            str(k): tuple(str(x) for x in v)
            for k, v in (subsets_doc.get("subset_assignment") or {}).items()
        },
    )
    bundle.cross_check()
    return bundle
