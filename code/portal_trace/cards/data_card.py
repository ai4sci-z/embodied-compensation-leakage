"""The data card of proposal 11.4: one per (embodiment, task).

The field list below is 11.4's data-card table transcribed row for row, in its
order. Nothing has been added to it and nothing has been dropped: 11.4 calls it
a minimum set and ties omission to disqualification, so a shorter list would
weaken the gate and a longer one would let a reviewer wonder which entries are
really required.

Three rows carry cross-field checks that no per-field validator can express, and
they are in `_cross_field_problems`:

* the proprio layout has to equal its own component breakdown. 11.4 writes
  52 = 45 (32 native + 7 tau_g + 6 wrist F/T) + 7 DoF mask, but the native
  width measured here for a 7-DoF arm is 50, not 32 (contract section 6), which
  makes the real total 70. The card therefore records the components and checks
  the sum, instead of checking against either literal -- the identity is the
  invariant, the total is not.
* the ILC outcome counts must be given as three separate numbers and must add up
  to the demonstration count. 11.4 is explicit that the three outcomes are
  "listed separately, not merged into one exclusion rate".
* the clipping-saturation share above 5% forces an annotation (11.4, 10.7.9), so
  a card over the threshold without the annotation is incomplete.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from portal_trace.cards.schema import (
    Card,
    CardSchema,
    FieldKind,
    FieldProblem,
    FieldSpec,
    NotApplicable,
)
from portal_trace.types import EmbodimentId

__all__ = [
    "DATA_CARD_SCHEMA",
    "DataCard",
    "CLIP_SATURATION_ANNOTATION_THRESHOLD",
    "ILC_OUTCOMES",
    "TOL_KINDS",
    "DEFECT_IMPL_VARIANTS",
    "ILC_CONTACT_VARIANTS",
]

CLIP_SATURATION_ANNOTATION_THRESHOLD = 0.05
"""Above this share of clipped c_true frames the cell's Lambda must be annotated.

11.4 and 10.7.9: clipping flattens the large values of c_delta and systematically
depresses the denominator of Lambda.
"""

ILC_OUTCOMES: tuple[str, ...] = ("converged", "limit_cycle", "not_converged")
"""11.4's three-way ILC outcome, reported as separate counts."""

TOL_KINDS: tuple[str, ...] = ("radial_clearance", "marker_halfwidth")
"""9.5.4 / 11.4: the two semantics may not be listed against each other."""

DEFECT_IMPL_VARIANTS: tuple[str, ...] = (
    "D3:a",
    "D3:b",
    "D4:hi",
    "D4:lo",
    "D5:total_ctrl",
    "D5:task_only",
    "none",
)

ILC_CONTACT_VARIANTS: tuple[str, ...] = ("full", "normal_only", "freeze_at_contact")


DATA_CARD_SCHEMA = CardSchema(
    kind="data card",
    granularity="one per (embodiment, task)",
    proposal_ref="11.4",
    fields=(
        FieldSpec(
            name="embodiment_identity",
            kind=FieldKind.MAPPING,
            description=(
                "embodiment id, morphology family and compile hash, matching "
                "manifest.yaml one for one; includes the 'in the 36 pairs' column"
            ),
            proposal_ref="11.4",
            blocks="the cell cannot be matched to its twin or to registry/lock.json",
        ),
        FieldSpec(
            name="delta_components",
            kind=FieldKind.MAPPING,
            description=(
                "every component of delta with its unit, the five level labels and the "
                "admissible interval; the delta = 0 twin is listed explicitly"
            ),
            proposal_ref="11.4, 9.2.10",
            blocks="the dose axis of E1/E3 and cross-family comparability of a level",
        ),
        FieldSpec(
            name="defect_impl_variant",
            kind=FieldKind.ENUM,
            description="D3:a/D3:b, D4:hi/D4:lo, D5:total_ctrl/D5:task_only",
            proposal_ref="11.4",
            blocks=(
                "which injection caliber produced the cell; D5 must be identical across "
                "all six morphology families"
            ),
            allowed=DEFECT_IMPL_VARIANTS,
        ),
        FieldSpec(
            name="c_true_storage",
            kind=FieldKind.MAPPING,
            description=(
                "per-frame c_delta ground truth: storage format, evaluation signature "
                "query_clean(x, t[, z]), the reference policy pinned to the clean "
                "reference policy's action, and whether the sensitivity version is stored"
            ),
            proposal_ref="11.4, 10.1.4",
            blocks="every Lambda reading for this cell; c_true is the numerator's partner",
        ),
        FieldSpec(
            name="c_true_clip_saturation_share",
            kind=FieldKind.NUMBER,
            description="fraction of frames where c_true hit the action-range clip",
            proposal_ref="11.4, 10.7.9",
            blocks=(
                "above 5% clipping flattens the large values of c_delta and "
                "systematically depresses Lambda's denominator; the cell's Lambda must "
                "then be annotated"
            ),
        ),
        FieldSpec(
            name="clip_saturation_annotation",
            kind=FieldKind.TEXT,
            description=(
                "the annotation required when the clipping share exceeds 5%; "
                "NotApplicable below the threshold"
            ),
            proposal_ref="11.4, 10.7.9",
            blocks="an over-threshold Lambda would be read as if it were unclipped",
            may_be_not_applicable=True,
        ),
        FieldSpec(
            name="kappa_and_tol_kind",
            kind=FieldKind.MAPPING,
            description=(
                "kappa evaluated in the terminal fine-approach phase, with the phase "
                "definition, and tol_kind in {radial_clearance, marker_halfwidth}"
            ),
            proposal_ref="11.4, 9.5.4",
            blocks=(
                "the dimensionless dose statement; the two tol_kind semantics may not be "
                "listed against each other"
            ),
        ),
        FieldSpec(
            name="task_and_env_registration",
            kind=FieldKind.MAPPING,
            description=(
                "task and environment registration names; the tight-fit tasks register "
                "as NutAssemblySquareTight-v0 / NutAssemblyRoundTight-v0, which changed "
                "the arena geom and the success criterion"
            ),
            proposal_ref="11.4, 9.5.2",
            blocks=(
                "the card must state that results are NOT comparable with published "
                "robomimic / MimicGen numbers"
            ),
        ),
        FieldSpec(
            name="ilc_outcomes",
            kind=FieldKind.MAPPING,
            description=(
                "per-demonstration ILC iteration counts and the three-way outcome: "
                "converged / limit_cycle / not_converged, as three separate counts, plus "
                "whether the cell is in the balanced panel B; with each converged cell "
                "the last iterate's e_RMS drop rate, and with each limit cycle the "
                "distinct mode-hash count and the iterates at which the mode switched"
            ),
            proposal_ref="11.4, 10.4.3 X13, 9.4.7",
            blocks=(
                "X13 dose-dependent exclusion is informative missingness; merging the "
                "three outcomes into one exclusion rate hides it. The drop rate is what "
                "makes 9.4.7's 'converged means good enough, not fixed point' auditable, "
                "and the switch indices separate a cell that (C3) failed because it is "
                "alternating from one where (C1) stopped across a single late switch"
            ),
        ),
        FieldSpec(
            name="ilc_contact_variant",
            kind=FieldKind.ENUM,
            description="full / normal_only / freeze_at_contact",
            proposal_ref="11.4, 10.7.9",
            blocks=(
                "as visible to a reader and a reproduction script as "
                "defect_impl_variant is"
            ),
            allowed=ILC_CONTACT_VARIANTS,
        ),
        FieldSpec(
            name="ell_adv_and_projection_residual",
            kind=FieldKind.MAPPING,
            description=(
                "ILC lookahead ell_adv, and e_RMS and E_phi listed separately for the "
                "free-space and the contact phase"
            ),
            proposal_ref="11.4, 9.6.4",
            blocks=(
                "the G1 gate reads the phase-split residual share; the merged value is "
                "not reported"
            ),
        ),
        FieldSpec(
            name="regime_caliber",
            kind=FieldKind.MAPPING,
            description=(
                "the five regimes and their exclusivity priority, the short-history "
                "window length, the admission thresholds m_min = 0.02 AND n_min = 2000 "
                "raw frames, coordinate-descent joint calibration frozen into "
                "registry/lock.json, and the actual regime count after merging into r0"
            ),
            proposal_ref="11.4, 9.7.1",
            blocks="every per-regime quantity and the design graph",
        ),
        FieldSpec(
            name="regime_occupancy_and_edge_weights",
            kind=FieldKind.SEQUENCE,
            description=(
                "one row per (task, defect, regime): occupancy mass, effective frame "
                "count n/tau_corr as the edge weight, tau_corr itself, the connectivity "
                "verdict and its sensitivity to the N_0 threshold"
            ),
            proposal_ref="11.4, 10.7.9",
            blocks="R_eff and lambda_2 are not reproducible without per-cell tau_corr",
        ),
        FieldSpec(
            name="z_probe_balanced_accuracy",
            kind=FieldKind.MAPPING,
            description=(
                "per memory-type defect (D2/D4/D10) x 5 levels x K_hist in {1,2,5,10}, "
                "next to the majority-class baseline"
            ),
            proposal_ref="11.4, 10.7.9",
            blocks=(
                "c_delta is a function of (x, z) for memory-type defects; where z is not "
                "identifiable the leakage ceiling is set by representability, not by "
                "Lambda"
            ),
        ),
        FieldSpec(
            name="normalisation_caliber",
            kind=FieldKind.MAPPING,
            description=(
                "whether action normalisation is global or per embodiment, and which "
                "dataset the statistics come from"
            ),
            proposal_ref="11.4",
            blocks="Lambda is not comparable across cells normalised differently",
        ),
        FieldSpec(
            name="mode_hash_exclusion_rate",
            kind=FieldKind.MAPPING,
            description="per level, the fraction of trajectories excluded by the (A3) check",
            proposal_ref="11.4",
            blocks="the (A3) mode-sequence assumption has no measured violation rate",
        ),
        FieldSpec(
            name="fgw_readings",
            kind=FieldKind.MAPPING,
            description=(
                "fgw_ours and fgw_baseline; for the D4 high-fidelity caliber also "
                "shadow_joint_offset (tracking RMS and peak cfrc_ext against the "
                "shadow-joint-free embodiment)"
            ),
            proposal_ref="11.4, 10.7.9",
            blocks=(
                "X1's strength is measured on the baseline feature set; the safety "
                "endpoints inherit the shadow-joint model mismatch as absolute values"
            ),
        ),
        FieldSpec(
            name="reference_trajectory_id",
            kind=FieldKind.TEXT,
            description="the reference trajectory set shared with the Oracle arm",
            proposal_ref="11.4, 14.5 E-9",
            blocks="the Oracle matching condition cannot be checked",
        ),
        FieldSpec(
            name="demonstration_counts",
            kind=FieldKind.MAPPING,
            description=(
                "number of demonstrations, frames and effective sample size; fixed "
                "horizon = 400 with early stopping off; t_success recorded per "
                "demonstration (-1 when unsuccessful) but not determining length; "
                "effective sample size as n_traj * (T / tau_corr)"
            ),
            proposal_ref="11.4, 9.6.1",
            blocks="raw frame counts as weights overstate precision by an order of magnitude",
        ),
        FieldSpec(
            name="proprio_layout",
            kind=FieldKind.MAPPING,
            description=(
                "proprio width and shape, broken into native + tau_g + wrist F/T + DoF "
                "mask; any statement of the physical-channel count must carry the "
                "'excluding the mask' qualifier"
            ),
            proposal_ref="11.4, 9.6.1",
            blocks="every hdf5 shape assertion and the observation-space audit",
        ),
        FieldSpec(
            name="seed_env",
            kind=FieldKind.INTEGER,
            description="the environment layer of the three-layer seed; excludes defect_id",
            proposal_ref="11.4, 11.3",
            blocks="cross-arm alignment of the shared initial-state layer",
        ),
        FieldSpec(
            name="collection_failures",
            kind=FieldKind.MAPPING,
            description="collection failures and exclusions, with a reason classification",
            proposal_ref="11.4, 11.3",
            blocks=(
                "failed experiments enter the archive with their termination reason and "
                "the F-number they triggered"
            ),
        ),
    ),
)

_PROPRIO_COMPONENTS = ("native", "tau_g", "wrist_ft", "dof_mask")


class DataCard(Card):
    """One (embodiment, task) data card (11.4)."""

    SCHEMA = DATA_CARD_SCHEMA

    def __init__(
        self,
        *,
        embodiment: EmbodimentId,
        task: str,
        values: Mapping[str, Any],
        notes: str = "",
    ) -> None:
        if not task:
            raise ValueError("a data card is per (embodiment, task); task must be named")
        super().__init__(values, notes)
        self.embodiment = embodiment
        self.task = task

    @property
    def subject(self) -> str:
        return f"{self.embodiment}|{self.task}"

    def _cross_field_problems(self) -> tuple[FieldProblem, ...]:
        problems: list[FieldProblem] = []
        problems.extend(self._check_proprio_layout())
        problems.extend(self._check_ilc_outcomes())
        problems.extend(self._check_clip_annotation())
        return tuple(problems)

    def _check_proprio_layout(self) -> list[FieldProblem]:
        layout = self.values.get("proprio_layout")
        if not isinstance(layout, Mapping):
            return []
        missing = [c for c in (*_PROPRIO_COMPONENTS, "total") if c not in layout]
        if missing:
            return [
                FieldProblem(
                    "proprio_layout",
                    f"needs the component breakdown {list(_PROPRIO_COMPONENTS)} and "
                    f"'total'; missing {missing}. 11.4 writes the total as a sum, and "
                    "the native width is family dependent (measured 50 for a 7-DoF arm, "
                    "not the 32 assumed in 9.6.1), so the sum is the invariant",
                    "11.4",
                )
            ]
        total = sum(int(layout[c]) for c in _PROPRIO_COMPONENTS)
        if int(layout["total"]) != total:
            return [
                FieldProblem(
                    "proprio_layout",
                    f"total {layout['total']} != {' + '.join(_PROPRIO_COMPONENTS)} "
                    f"= {total}",
                    "11.4",
                )
            ]
        return []

    def _check_ilc_outcomes(self) -> list[FieldProblem]:
        outcomes = self.values.get("ilc_outcomes")
        if not isinstance(outcomes, Mapping):
            return []
        missing = [k for k in ILC_OUTCOMES if k not in outcomes]
        if missing:
            return [
                FieldProblem(
                    "ilc_outcomes",
                    f"needs all three outcome counts {list(ILC_OUTCOMES)} listed "
                    f"separately; missing {missing}. 11.4 forbids merging them into a "
                    "single exclusion rate because X13 makes the missingness informative",
                    "11.4",
                )
            ]
        if "in_balanced_panel" not in outcomes:
            return [
                FieldProblem(
                    "ilc_outcomes",
                    "needs 'in_balanced_panel': E3's dose narrative is carried by the "
                    "balanced panel B (11.3)",
                    "11.4",
                )
            ]
        counts = self.values.get("demonstration_counts")
        if isinstance(counts, Mapping) and "n_demonstrations" in counts:
            total = sum(int(outcomes[k]) for k in ILC_OUTCOMES)
            if total != int(counts["n_demonstrations"]):
                return [
                    FieldProblem(
                        "ilc_outcomes",
                        f"the three outcome counts sum to {total} but the cell holds "
                        f"{counts['n_demonstrations']} demonstrations",
                        "11.4",
                    )
                ]
        return []

    def _check_clip_annotation(self) -> list[FieldProblem]:
        share = self.values.get("c_true_clip_saturation_share")
        if not isinstance(share, (int, float)) or isinstance(share, bool):
            return []
        annotation = self.values.get("clip_saturation_annotation")
        over = share > CLIP_SATURATION_ANNOTATION_THRESHOLD
        if over and isinstance(annotation, NotApplicable):
            return [
                FieldProblem(
                    "clip_saturation_annotation",
                    f"clipping share {share:.3f} exceeds "
                    f"{CLIP_SATURATION_ANNOTATION_THRESHOLD:.2f}, so 11.4 requires the "
                    "annotation; it cannot be declared not applicable",
                    "11.4, 10.7.9",
                )
            ]
        return []
