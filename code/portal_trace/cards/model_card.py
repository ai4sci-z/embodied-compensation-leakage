"""The model card of proposal 11.4: one per training arm.

11.4's model-card table transcribed row for row. Two of its rows are the reason
the other modules in this package can be strict:

* `subset_and_design` carries (M, D, S). 11.4: "every row of the summary table
  must carry these three numbers, otherwise that row's MDE and its inconclusive
  rule cannot be derived". `eval.report` refuses to render a row without a
  `Design`, and this is where the arm's `Design` comes from.
* `preregistration_pointer` carries the preregistration file, its git tag and the
  verdict. It is checked against the three-way vocabulary, so a card cannot
  record a fourth kind of outcome.

`alpha_demo` and `alpha_frame` are cross-checked for bit equality. 11.4 and X14
say they must agree digit for digit under the fixed-horizon caliber and that a
non-zero difference is a bug, so the card is the place that catches it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from portal_trace.cards.schema import (
    Card,
    CardSchema,
    FieldKind,
    FieldProblem,
    FieldSpec,
)

if TYPE_CHECKING:
    from portal_trace.eval.stats import Design

__all__ = [
    "MODEL_CARD_SCHEMA",
    "ModelCard",
    "VERDICT_VOCABULARY",
    "MIN_EPOCHS_PER_CONFIG",
    "DISCRIMINATOR_ERROR_BUDGET",
    "DISCRIMINATOR_INFORMATION_GAIN_FLOOR_BITS",
    "DISCRIMINATOR_BUDGET_FIELDS",
    "OFFLINE_SHIFT_DIAGNOSTICS",
]

VERDICT_VOCABULARY: tuple[str, ...] = ("significant", "not_significant", "inconclusive")
"""11.3's three-way verdict. There is no fourth value and no free text."""

MIN_EPOCHS_PER_CONFIG = 60
"""9.8.1 / 11.4: at least 60 epochs per config, and this does not enter the
degradation ladder -- under-training inflates E||eps||^2, depresses Phi and
reads out a false low leakage, so the epoch count is part of the endpoint
definition."""

DISCRIMINATOR_ERROR_BUDGET = 0.05
"""Total absolute budget for the source discriminator (10.4.1, 11.4), set
against Lambda^part's MDE of 0.062-0.077."""

DISCRIMINATOR_INFORMATION_GAIN_FLOOR_BITS = 0.01
"""Minimum held-out log-score gain over the empirical-prior predictor.

This prevents an almost constant 0.5 posterior from satisfying the calibration
budget while erasing the covariance term that the source posterior measures.
"""

DISCRIMINATOR_BUDGET_FIELDS: tuple[str, ...] = (
    "w_ece",
    "w_alpha_recovery",
    "w_support_gap",
)

OFFLINE_SHIFT_DIAGNOSTICS: tuple[str, ...] = (
    "domain_discriminator_auc",
    "off_support_fraction",
    "in_support_subset_reading",
)


MODEL_CARD_SCHEMA = CardSchema(
    kind="model card",
    granularity="one per training arm",
    proposal_ref="11.4",
    fields=(
        FieldSpec(
            name="arm_identity",
            kind=FieldKind.MAPPING,
            description=(
                "B1-B13 or the experiment number, and whether the arm is the pooled arm, "
                "the Oracle arm or a control arm"
            ),
            proposal_ref="11.4",
            blocks="which reference point a Lambda reading used",
        ),
        FieldSpec(
            name="subset_and_design",
            kind=FieldKind.MAPPING,
            description=(
                "which (family, defect) cells this arm ran -- the full 36 or a named "
                "subset -- plus M, D and S"
            ),
            proposal_ref="11.4, 10.7.10",
            blocks=(
                "every summary row must carry (M, D, S); without them the row's MDE and "
                "its inconclusive rule cannot be derived, and a subset arm would borrow "
                "the (6,6,3) numbers it is not entitled to"
            ),
        ),
        FieldSpec(
            name="training_data_composition",
            kind=FieldKind.MAPPING,
            description=(
                "demonstrations per embodiment, N_tgt / N_src / K, the double record "
                "alpha_demo and alpha_frame, and the edge table of the design graph G"
            ),
            proposal_ref="11.4, X14",
            blocks=(
                "alpha_demo and alpha_frame must agree bit for bit under the fixed-length "
                "caliber; a non-zero difference is a bug, not a rounding artefact"
            ),
        ),
        FieldSpec(
            name="pool_composition",
            kind=FieldKind.MAPPING,
            description=(
                "which (embodiment, delta) pairs are in the pool, whether delta = 0 is "
                "among them, and whether the target embodiment's own data is in the pool"
            ),
            proposal_ref="11.4, 11.3",
            blocks=(
                "with a clean twin in the pool, route A's anchoring holds by "
                "construction, and a leakage reading then means 'the algorithm did not "
                "use an available anchor', not 'there was not enough information'. This "
                "changes what an E7/E8 null result means"
            ),
        ),
        FieldSpec(
            name="design_graph_quantities",
            kind=FieldKind.MAPPING,
            description=(
                "lambda_2(G), the pairwise R_eff table, Xi_G computed numerically per "
                "level, and which of the two edge-weight calibers (effective frame count "
                "/ heteroscedastic conductance) was used"
            ),
            proposal_ref="11.4, 10.4.8",
            blocks="the E8 slope's independent variable is not reproducible",
        ),
        FieldSpec(
            name="architecture_and_budget",
            kind=FieldKind.MAPPING,
            description=(
                "parameter count, token budget, training steps and epochs (at least 60 "
                "per config); on the isolation ladder the matched quantity is the "
                "ACTIVATED parameter count, so both the measured activated and the total "
                "parameter count are given, the latter entering the main regression as a "
                "log covariate"
            ),
            proposal_ref="11.4, 9.8.1, 10.4.9",
            blocks=(
                "under-training inflates E||eps||^2 and depresses Phi, reading out a "
                "false low leakage; the epoch floor is part of the endpoint definition"
            ),
        ),
        FieldSpec(
            name="isolation_variables",
            kind=FieldKind.MAPPING,
            description=(
                "parameter disjointness I (expected share of routing weight, no "
                "threshold) and HB-MoE routing entropy; an isolation level must be "
                "reported paired with the decisive control level I3'"
            ),
            proposal_ref="11.4, 10.4.9, A-25",
            blocks=(
                "without I3' an E9 effect cannot be told apart from data fragmentation"
            ),
        ),
        FieldSpec(
            name="seed_train",
            kind=FieldKind.MAPPING,
            description="the training layer of the three-layer seed; the list for this arm",
            proposal_ref="11.4, 11.3",
            blocks=(
                "cross-arm comparisons align the env and eval seeds and vary the training "
                "seed; per-seed results are reported alongside the mean, never the mean "
                "alone"
            ),
        ),
        FieldSpec(
            name="conditioning",
            kind=FieldKind.MAPPING,
            description=(
                "whether there is an embodiment token or one-hot; the latent dimension "
                "and history window of latent-extrinsics; routing type and routing entropy"
            ),
            proposal_ref="11.4, 10.4.9",
            blocks="E9's independent variable",
        ),
        FieldSpec(
            name="eval_embodiment_and_delta",
            kind=FieldKind.MAPPING,
            description=(
                "the evaluation embodiment and delta'; a mismatched arm must record the "
                "target defect. The main setting's target is the DEFECTIVE instance "
                "delta_tgt (the diagonal of the E11 cross matrix), never the twin by "
                "default; the diagonal and off-diagonal readings are reported separately"
            ),
            proposal_ref="11.4, 10.4.11",
            blocks=(
                "the off-diagonal is where delta_tgt not in D is tested; folding it into "
                "the diagonal destroys that test"
            ),
        ),
        FieldSpec(
            name="endpoints",
            kind=FieldKind.MAPPING,
            description=(
                "Lambda^part (primary) and Lambda (alongside), Lambda^ref, Lambda^orc "
                "with its three-term decomposition, Phi, Lambda^exc, Lambda^emp_res, "
                "Lambda^emp_con, Q_div, the success-rate paired difference, Leak_beh "
                "with its Fieller CI, peak cfrc_ext and the constraint violation rate"
            ),
            proposal_ref="11.4, 10.1",
            blocks="the arm has no readable result",
        ),
        FieldSpec(
            name="discriminator_error_budget",
            kind=FieldKind.MAPPING,
            description=(
                "w_ece, w_alpha_recovery and w_support_gap (total absolute budget 0.05, "
                "against Lambda^part's MDE of 0.062-0.077), plus the three rho^off "
                "covariate-shift diagnostics: domain discriminator AUC, off-support "
                "fraction, in-support subset reading"
            ),
            proposal_ref="11.4, 10.7.9",
            blocks=(
                "T1 uses Lambda - E_rho[w], so discriminator bias passes 1:1 into the "
                "statistic; a T1 verdict without these three numbers is uninterpretable. "
                "AUC > 0.90 or an in-support/full-set gap > 0.05 forces an annotation"
            ),
        ),
        FieldSpec(
            name="phi_effective_rank",
            kind=FieldKind.INTEGER,
            description=(
                "truncated-SVD rank of phi, expected 150-400 against 392 raw columns"
            ),
            proposal_ref="11.4, 10.7.9",
            blocks=(
                "'p is in 150-400' refers to the effective rank, not the column count; "
                "mixing the two makes the dimension accounting unreadable"
            ),
        ),
        FieldSpec(
            name="state_distribution",
            kind=FieldKind.MAPPING,
            description="rho^off for the primary endpoint; rho^on reported separately",
            proposal_ref="11.4, 6.2",
            blocks=(
                "evaluated on the policy's own occupancy, Lambda folds the consequences "
                "of covariate drift into the definition of leakage and stops being "
                "comparable across methods"
            ),
        ),
        FieldSpec(
            name="null_distribution",
            kind=FieldKind.MAPPING,
            description=(
                "the surrogate-defect list used, the per-cell effective cardinality |S|, "
                "the quantiles, and whether the random-rotation screen is also reported"
            ),
            proposal_ref="11.4, 10.4.1",
            blocks=(
                "per-cell |S| may be below 8 but must be at least 6, otherwise that "
                "cell's Lambda^exc is exploratory only"
            ),
        ),
        FieldSpec(
            name="preregistration_pointer",
            kind=FieldKind.MAPPING,
            description=(
                "the preregistration file and its git tag, and the verdict: significant / "
                "not_significant / inconclusive"
            ),
            proposal_ref="11.4, 11.3",
            blocks="the result cannot be tied back to a frozen hypothesis",
        ),
    ),
)


class ModelCard(Card):
    """One training-arm model card (11.4)."""

    SCHEMA = MODEL_CARD_SCHEMA

    def __init__(self, *, arm: str, values: Mapping[str, Any], notes: str = "") -> None:
        if not arm:
            raise ValueError("a model card is per training arm; the arm must be named")
        super().__init__(values, notes)
        self.arm = arm

    def __getitem__(self, key: str) -> Any:
        """Read the envelope or a legacy runner alias without weakening validation.

        ``runners.train`` used to expose a flat dictionary.  Keeping these
        read-only aliases lets orchestration code inspect a prepared job while
        serialization remains exclusively through :meth:`to_json`, which runs
        the 11.4 completeness gate first.  New consumers should use
        ``card.values`` or ``card.to_json()["fields"]``.
        """
        if key in {"card_kind", "granularity", "proposal_ref", "subject", "fields", "notes"}:
            document = self.to_json()
            if key in document:
                return document[key]
        aliases = {
            "arm_id": ("arm_identity", "arm"),
            "role": ("arm_identity", "role"),
            "subset": ("subset_and_design", "subset"),
            "design_MDS": ("subset_and_design",),
            "cells": ("subset_and_design", "cells"),
            "architecture": ("architecture_and_budget", "architecture"),
            "architecture_declaration": ("architecture_and_budget", "architecture_declaration"),
            "seed_train": ("seed_train", "seed"),
            "seed_index": ("seed_train", "seed_index"),
            "tasks": ("training_data_composition", "tasks"),
            "task_set": ("training_data_composition", "task_set"),
            "n_demos_total": ("training_data_composition", "n_demos_total"),
            "n_demos_per_task": ("training_data_composition", "n_demos_per_task"),
            "defect_level": ("eval_embodiment_and_delta", "level"),
            "level_identity": ("training_data_composition", "level_identity"),
            "pool_sources": ("training_data_composition", "pool_sources"),
            "includes_delta_zero": ("pool_composition", "includes_delta_zero"),
            "target_in_pool": ("pool_composition", "target_in_pool"),
            "reading_consequence": ("pool_composition", "reading_consequence"),
            "alpha_demo": ("training_data_composition", "alpha_demo"),
            "alpha_frame": ("training_data_composition", "alpha_frame"),
            "reference_arm": ("arm_identity", "reference_arm"),
            "backend": ("architecture_and_budget", "backend"),
            "prereg_version": ("preregistration_pointer", "version"),
            "prereg_tag": ("preregistration_pointer", "git_tag"),
            "amendments_after_unblinding": (
                "preregistration_pointer",
                "amendments_after_unblinding",
            ),
        }
        if key == "design_MDS":
            block = self.values["subset_and_design"]
            return [block["M"], block["D"], block["S"]]
        location = aliases.get(key)
        if location is not None:
            value: Any = self.values
            for part in location:
                value = value[part]
            return value
        raise KeyError(key)

    @property
    def subject(self) -> str:
        return self.arm

    @property
    def design(self) -> Design:
        """The arm's (M, D, S), which every summary row it feeds has to carry."""
        # Imported here, not at module scope: `eval.endpoints` gates itself on
        # cards, so a top-level import would close the loop cards -> eval -> cards.
        from portal_trace.eval.stats import Design

        self.require_complete()
        block = self.values["subset_and_design"]
        return Design(
            m_families=int(block["M"]),
            d_defects=int(block["D"]),
            s_seeds=int(block["S"]),
            label=f"{self.arm}:{block.get('subset', 'unnamed subset')}",
        )

    def _cross_field_problems(self) -> tuple[FieldProblem, ...]:
        problems: list[FieldProblem] = []
        problems.extend(self._check_pending_values())
        problems.extend(self._check_design())
        problems.extend(self._check_alpha_double_record())
        problems.extend(self._check_epoch_floor())
        problems.extend(self._check_discriminator_budget())
        problems.extend(self._check_verdict_vocabulary())
        return tuple(problems)

    def _check_pending_values(self) -> list[FieldProblem]:
        """A prepared runner record is not an admissible completed card.

        The training orchestrator constructs metadata before optimization, but
        endpoint cards must never serialize placeholders as if they were
        measurements.  Keep this recursive so a newly added nested field cannot
        silently bypass the gate.
        """
        def contains_pending(value: Any) -> bool:
            if isinstance(value, str):
                return value.startswith("pending_")
            if isinstance(value, Mapping):
                return any(contains_pending(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return any(contains_pending(item) for item in value)
            return False

        if any(contains_pending(value) for value in self.values.values()):
            return [
                FieldProblem(
                    "model_card",
                    "contains pending placeholders; replace them with measured, "
                    "provenance-bound values before serialization",
                    "11.4",
                )
            ]
        return []

    def _check_design(self) -> list[FieldProblem]:
        block = self.values.get("subset_and_design")
        if not isinstance(block, Mapping):
            return []
        missing = [k for k in ("M", "D", "S", "cells", "subset") if k not in block]
        if missing:
            return [
                FieldProblem(
                    "subset_and_design",
                    f"missing {missing}; 11.4 requires the cell list, the named subset "
                    "and all three of M, D, S on every row",
                    "11.4, 10.7.10",
                )
            ]
        n_cells = len(block["cells"])
        if n_cells != int(block["M"]) * int(block["D"]):
            return [
                FieldProblem(
                    "subset_and_design",
                    f"lists {n_cells} cells but M x D = {int(block['M'])} x "
                    f"{int(block['D'])} = {int(block['M']) * int(block['D'])}",
                    "11.4, 10.7.10",
                )
            ]
        return []

    def _check_alpha_double_record(self) -> list[FieldProblem]:
        block = self.values.get("training_data_composition")
        if not isinstance(block, Mapping):
            return []
        missing = [k for k in ("alpha_demo", "alpha_frame") if k not in block]
        if missing:
            return [
                FieldProblem(
                    "training_data_composition",
                    f"missing the double record {missing}; X14's check is that the two "
                    "agree bit for bit under the fixed-horizon caliber",
                    "11.4, X14",
                )
            ]
        a_demo, a_frame = float(block["alpha_demo"]), float(block["alpha_frame"])
        if a_demo != a_frame:
            return [
                FieldProblem(
                    "training_data_composition",
                    f"alpha_demo {a_demo!r} != alpha_frame {a_frame!r}. Under the fixed "
                    "horizon = 400 caliber these are equal digit for digit; a non-zero "
                    "difference is a bug in the mixing weights (X14), not a tolerance "
                    "question",
                    "11.4, X14",
                )
            ]
        return []

    def _check_epoch_floor(self) -> list[FieldProblem]:
        block = self.values.get("architecture_and_budget")
        if not isinstance(block, Mapping) or "epochs_per_config" not in block:
            return []
        epochs = int(block["epochs_per_config"])
        if epochs < MIN_EPOCHS_PER_CONFIG:
            return [
                FieldProblem(
                    "architecture_and_budget",
                    f"{epochs} epochs per config is below the floor of "
                    f"{MIN_EPOCHS_PER_CONFIG}; 9.8.1 puts the epoch count inside the "
                    "endpoint definition and outside the degradation ladder",
                    "11.4, 9.8.1",
                )
            ]
        return []

    def _check_discriminator_budget(self) -> list[FieldProblem]:
        block = self.values.get("discriminator_error_budget")
        if not isinstance(block, Mapping):
            return []
        missing = [
            k
            for k in (*DISCRIMINATOR_BUDGET_FIELDS, *OFFLINE_SHIFT_DIAGNOSTICS)
            if k not in block
        ]
        if missing:
            return [
                FieldProblem(
                    "discriminator_error_budget",
                    f"missing {missing}; 10.7.9 says a T1 verdict without these is "
                    "uninterpretable because discriminator bias enters the statistic 1:1",
                    "11.4, 10.7.9",
                )
            ]
        return []

    def _check_verdict_vocabulary(self) -> list[FieldProblem]:
        block = self.values.get("preregistration_pointer")
        if not isinstance(block, Mapping):
            return []
        missing = [k for k in ("file", "git_tag", "verdict") if k not in block]
        if missing:
            return [
                FieldProblem(
                    "preregistration_pointer",
                    f"missing {missing}",
                    "11.4, 11.3",
                )
            ]
        verdict = block["verdict"]
        if verdict not in VERDICT_VOCABULARY:
            return [
                FieldProblem(
                    "preregistration_pointer",
                    f"verdict {verdict!r} is not one of {list(VERDICT_VOCABULARY)}; "
                    "11.3 admits exactly three outcomes and inconclusive is one of them",
                    "11.4, 11.3",
                )
            ]
        return []
