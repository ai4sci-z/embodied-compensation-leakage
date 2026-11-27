"""E5 as one arm: both endpoints, one verdict, one exit ladder (proposal 10.4.6).

`transfer_ratio` produces endpoint A and `anchor_S` produces endpoint B. This
module is what makes them one experiment: it checks the institution axis against
the release that was actually downloaded, applies the two preregistered
predictions, renders the three-tier F5 exit when they are not met, and carries
the wording discipline into the output rather than into a comment.

Why E5 declares no training arm
-------------------------------
10.4.6's cost line is `0 complete runs + 9 GPU-h`. E5 trains no policy: endpoint
A is pure CPU (download, alignment and variance components, which 10.8.4 books
on its post-processing row), and the 9 GPU-h is the per-institution delta_hat
fit and the held-out check of endpoint B -- 13 small physical-basis regressions,
not a policy. So `arms: []` in the experiment config is the honest reading and
not an omission, and appendix A-4 agrees from the other side: with no training
arm there is no B7 Oracle to be a reference point, so every E5 endpoint carries
`reference_arm: null`.

What that leaves is the part that *was* missing, and it is endpoint B. The
experiment's `null_distribution` field names a held-out laboratory check, its
budget line pays 9 GPU-h for one, and 10.4.6 devotes a paragraph to it -- and no
endpoint declared it. `ENDPOINT_B_NAME` is that endpoint.

The identification problem, and why one reading will not do
-----------------------------------------------------------
DROID is human closed-loop teleoperation. An operator carrying no feedforward
compensation at all still commands differently on a laggy machine, because they
are reacting to the lag they can see; that alone raises sigma^2_lab(cmd), the
loop pulls achieved back, and T_comp > 1 comes out of an operator who
compensates nothing. The pre-latency window is what separates the two, and it is
the primary reading for that reason -- 200 ms is inside human visuomotor
latency, so a command there cannot be a reaction to the lag of the segment it
opens. `E5ArmReport.verdict` is therefore read off the window reading, and the
full-segment reading appears next to it as the contrast that makes the window
reading mean something.

Wording
-------
Observational association, always. (A2) excludes human teleoperation and these
are human teleoperation data; the causal chain is E1's and E7-E9's.
`E5ArmReport.causal_claim()` raises rather than returning a sentence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from portal_trace.eval.report import InconclusiveReason, Verdict
from portal_trace.lockfile import REQUIRED_MEASUREMENTS, LockFile
from portal_trace.real.anchor_S import (
    N_HELD_OUT_LABS,
    PREREGISTERED_VARIANCE_DROP,
    EndpointBResult,
)
from portal_trace.real.pair import N_DROID_LABS, DroidLabId
from portal_trace.real.transfer_ratio import (
    PREREGISTERED_T_COMP_MIN_LABS,
    PREREGISTERED_T_COMP_THRESHOLD,
    TCompReport,
)

__all__ = [
    "E5_COMPLETE_RUNS",
    "E5_GPU_HOURS",
    "E5_ENDPOINT_A_GPU_HOURS",
    "E5_ENDPOINT_B_GPU_HOURS",
    "ENDPOINT_A_NAMES",
    "ENDPOINT_B_NAME",
    "DROID_ROSTER",
    "droid_roster",
    "F5Tier",
    "E5Budget",
    "E5_BUDGET",
    "E5ArmReport",
    "e5_report",
    "DroidRosterMismatch",
    "CausalClaimRefused",
    "EndpointBMissing",
]

E5_COMPLETE_RUNS = 0
"""10.4.6: E5 trains no policy, so it contributes no complete run to 9.8.2."""

E5_ENDPOINT_A_GPU_HOURS = 0.0
"""Endpoint A is zero GPU. Download, alignment and the variance components are CPU."""

E5_ENDPOINT_B_GPU_HOURS = 9.0
"""10.4.6: the 13 per-institution delta_hat fits plus the held-out check."""

ENDPOINT_A_NAMES: tuple[str, ...] = (
    "t_comp_pre_latency_window",
    "t_comp_full_segment",
    "t_comp_reactive_feedforward_difference",
    "t_comp_regressed_out",
)
"""The four readings of 10.4.6, as the four endpoint names the config declares."""

ENDPOINT_B_NAME = "held_out_lab_variance_drop"
"""Route (S)'s anchoring on real data, which the experiment config was missing."""

DROID_ROSTER = REQUIRED_MEASUREMENTS["real.droid.institution_roster"]


class DroidRosterMismatch(ValueError):
    """The institutions in the analysis are not the release's institutions."""


class CausalClaimRefused(ValueError):
    """Something asked E5 for a causal statement. It does not make one."""


class EndpointBMissing(ValueError):
    """The arm's verdict was requested without endpoint B."""


def droid_roster(lock: LockFile) -> tuple[DroidLabId, ...]:
    """The institution ids of the pinned DROID release, from the lock file.

    Not `N_DROID_LABS` counted out: that constant is 10.4.6's *claim* about the
    dataset, and E5's preregistered prediction is a count against it ("> 1.3 on
    at least 8 of 13"). If the release that was actually downloaded exposes a
    different set of institutions, the denominator of that count changes and the
    prediction is being read against the wrong denominator. `LockFile.get` raises
    with the producing script when the roster has not been read off the data.
    """
    value = lock.get(DROID_ROSTER)
    if not isinstance(value, (list, tuple)) or not value:
        raise DroidRosterMismatch(
            f"the lock entry {DROID_ROSTER.key_template!r} holds {value!r}; it is the "
            "list of institution identifiers in the pinned DROID release"
        )
    ids = tuple(DroidLabId(str(x)) for x in value)
    if len(set(ids)) != len(ids):
        raise DroidRosterMismatch(f"the roster repeats an institution: {list(ids)}")
    if len(ids) != N_DROID_LABS:
        raise DroidRosterMismatch(
            f"the release exposes {len(ids)} institutions {list(ids)}, and 10.4.6's "
            f"institution axis is DROID's {N_DROID_LABS}. E5's preregistered "
            f"prediction is a count against that denominator -- pre-window T_comp > "
            f"{PREREGISTERED_T_COMP_THRESHOLD} on at least "
            f"{PREREGISTERED_T_COMP_MIN_LABS} of {N_DROID_LABS} -- so a different "
            "roster makes the count a different claim. Splicing the shortfall from "
            "another dataset is excluded by appendix A-27: across datasets the "
            "original control conventions are retained, so a cross-dataset "
            "sigma^2_lab(cmd) mixes in action-space convention differences"
        )
    return ids


class F5Tier(Enum):
    """10.4.6's three-tier zero-result exit, in the order the section writes them.

    The tiers are ordered by what they say about the data rather than by
    severity. Tier one is a statement about the measurement: T_comp near 1 while
    the between-institution component is itself tiny means there was nothing to
    take a ratio of. Tier two is a positive finding in disguise: T_comp below 1
    says the released action data have already been post-processed and
    normalised until the execution signature is gone, which *supports* the core
    mechanism and agrees with X7. Tier three is what the arm contributes when
    neither of the first two applies.
    """

    NOT_TRIGGERED = "the preregistered prediction was met"
    TIER_1_DATA_QUALITY = (
        "T_comp ~ 1 with a vanishing between-institution component: a data quality "
        "finding, to be redone stratified by DROID's own task categories"
    )
    TIER_2_SIGNATURE_ERASED = (
        "T_comp < 1: the released action data have been post-processed and normalised "
        "per laboratory until the execution signature is gone. This supports the core "
        "mechanism and agrees with X7"
    )
    TIER_3_METADATA_RECOMMENDATION = (
        "neither of the above: E5 becomes an appendix plus a community "
        "recommendation, the minimum calibration metadata a released dataset should "
        "report"
    )


@dataclass(frozen=True)
class E5Budget:
    """10.4.6's cost line as an arithmetic: 0 complete runs plus 9 GPU-h."""

    complete_runs: int = E5_COMPLETE_RUNS
    endpoint_a_gpu_hours: float = E5_ENDPOINT_A_GPU_HOURS
    endpoint_b_gpu_hours: float = E5_ENDPOINT_B_GPU_HOURS

    @property
    def gpu_hours(self) -> float:
        return self.endpoint_a_gpu_hours + self.endpoint_b_gpu_hours

    @property
    def training_arms(self) -> tuple[str, ...]:
        """Empty, and that is the honest declaration rather than an omission."""
        return ()

    def as_row(self) -> dict[str, Any]:
        return {
            "complete_runs": self.complete_runs,
            "gpu_hours": self.gpu_hours,
            "endpoint_a_gpu_hours": self.endpoint_a_gpu_hours,
            "endpoint_b_gpu_hours": self.endpoint_b_gpu_hours,
            "training_arms": list(self.training_arms),
            "cpu_note": (
                "download, alignment and the variance components are booked on "
                "10.8.4's post-processing row rather than on E5's own line"
            ),
        }


E5_BUDGET = E5Budget()
E5_GPU_HOURS = E5_BUDGET.gpu_hours


@dataclass(frozen=True)
class E5ArmReport:
    """Both endpoints of E5, the roster they were read on, and the verdict."""

    endpoint_a: TCompReport
    endpoint_b: EndpointBResult | None
    roster: tuple[DroidLabId, ...]
    analysed_labs: tuple[DroidLabId, ...]
    budget: E5Budget = E5_BUDGET

    def __post_init__(self) -> None:
        missing = sorted(set(self.roster) - set(self.analysed_labs))
        extra = sorted(set(self.analysed_labs) - set(self.roster))
        if extra:
            raise DroidRosterMismatch(
                f"institutions {extra} appear in the analysis but not in the pinned "
                f"release's roster {list(self.roster)}. The institution axis is "
                "defined inside DROID (appendix A-27)"
            )
        if missing:
            raise DroidRosterMismatch(
                f"institutions {missing} are in the release but were not analysed. "
                f"E5's prediction is a count over all {len(self.roster)} -- "
                f"'at least {PREREGISTERED_T_COMP_MIN_LABS} of {N_DROID_LABS}' -- so "
                "dropping an institution changes the denominator without changing the "
                "claim it is read against. If an institution has to be excluded, the "
                "exclusion and its reason belong in the preregistration, not in the "
                "analysis"
            )
        if len(self.endpoint_a.per_lab_pre_window) != len(self.roster):
            raise DroidRosterMismatch(
                f"the per-institution pre-window table has "
                f"{len(self.endpoint_a.per_lab_pre_window)} rows against a roster of "
                f"{len(self.roster)}"
            )

    # -- endpoint A ------------------------------------------------------

    @property
    def labs_above_threshold(self) -> int:
        return self.endpoint_a.labs_above_threshold

    @property
    def endpoint_a_prediction_met(self) -> bool:
        """> 1.3 on at least 8 of 13, and the pre-window minus full-segment > 0."""
        return self.endpoint_a.prediction_met

    @property
    def reactive_component_exists(self) -> bool:
        """The difference reading is positive: reactive is present, not the whole.

        10.4.6 predicts a positive difference and reads it as "a reactive
        component exists but is not all of it". A difference at or below zero
        does not mean the operators are purely feedforward -- it means the two
        readings did not separate, which is a statement about the decomposition
        and not about the operators.
        """
        return self.endpoint_a.difference > 0.0

    # -- endpoint B ------------------------------------------------------

    @property
    def endpoint_b_prediction_met(self) -> bool:
        if self.endpoint_b is None:
            raise EndpointBMissing(
                "endpoint B has not been run. It is route (S)'s anchoring on real "
                f"data -- fit delta_hat per institution, subtract psi_j(delta_hat), "
                f"and read the drop in the between-institution component on "
                f"{N_HELD_OUT_LABS} held-out institutions -- and 10.4.6 pays "
                f"{E5_ENDPOINT_B_GPU_HOURS} GPU-h for it. Produce it with "
                "`portal_trace.real.anchor_S.held_out_lab_test`"
            )
        return self.endpoint_b.median_relative_drop >= PREREGISTERED_VARIANCE_DROP

    @property
    def endpoint_b_uses_no_exclusion_restriction(self) -> bool:
        """True, and stated: this is what makes endpoint B worth its own paragraph.

        Route (S) identifies delta by widening the observation from (x, a) to
        (x, a, Delta x^ach), so it needs no (ER)-type exclusion restriction. That
        makes endpoint B the only identification route in the plan that runs on
        real hardware data without one -- and it is why the pessimism about the
        common mode is bounded: the common mode is unidentifiable only in the
        absence of an anchor, and real data carry an execution signature that can
        serve as one.
        """
        return True

    # -- the verdict and the exit ----------------------------------------

    @property
    def f5_tier(self) -> F5Tier:
        if self.endpoint_a_prediction_met:
            return F5Tier.NOT_TRIGGERED
        pre = self.endpoint_a.pre_window
        if pre.degenerate:
            # Tested before the ratio, in the order 10.4.6 lists the tiers: a
            # component sitting at the REML boundary is not a number the ratio
            # can be read off at all, so "T_comp < 1" would be reading a
            # denominator that is not an estimate.
            return F5Tier.TIER_1_DATA_QUALITY
        if pre.t_comp < 1.0:
            return F5Tier.TIER_2_SIGNATURE_ERASED
        return F5Tier.TIER_3_METADATA_RECOMMENDATION

    @property
    def verdict(self) -> str:
        if self.endpoint_a_prediction_met:
            return "supported"
        return f"F5: {self.f5_tier.name}"

    @property
    def verdict_kind(self) -> Verdict:
        """The same reading in 11.3's three-valued vocabulary.

        Tier one is `INCONCLUSIVE` rather than `NOT_SIGNIFICANT`, and it lands on
        a trigger that already exists: the between-institution component sat at
        the REML boundary, which is 11.3's zero-variance-component trigger. There
        was no ratio to read, so "no effect" is not what the data said.
        """
        if self.endpoint_a_prediction_met:
            return Verdict.SIGNIFICANT
        if self.f5_tier is F5Tier.TIER_1_DATA_QUALITY:
            return Verdict.INCONCLUSIVE
        return Verdict.NOT_SIGNIFICANT

    @property
    def inconclusive_reason(self) -> InconclusiveReason | None:
        if self.verdict_kind is not Verdict.INCONCLUSIVE:
            return None
        return InconclusiveReason.ZERO_VARIANCE_COMPONENT

    def causal_claim(self) -> str:
        raise CausalClaimRefused(
            "E5 states observational associations only. (A2) excludes human "
            "teleoperation and DROID is human teleoperation, so nothing here supports "
            "a causal statement; the causal chain is carried by E1 and by the three "
            "crowns E7-E9 (10.4.6). The sentence this arm may print is 'the commanded "
            "actions in this dataset carry a laboratory-specific component that the "
            "achieved poses do not', not 'the laboratories compensate'"
        )

    def as_row(self) -> dict[str, Any]:
        a = self.endpoint_a
        row: dict[str, Any] = {
            "experiment": "E5",
            "verdict": self.verdict,
            "verdict_kind": self.verdict_kind.value,
            "inconclusive_reason": (
                None if self.inconclusive_reason is None else self.inconclusive_reason.value
            ),
            "f5_tier": self.f5_tier.name,
            "f5_tier_text": self.f5_tier.value,
            "statement_scope": (
                "observational association only: (A2) excludes human teleoperation "
                "and DROID is human teleoperation (10.4.6)"
            ),
            "institution_axis": {
                "roster": [str(x) for x in self.roster],
                "n_institutions": len(self.roster),
                "defined_inside_droid": True,
                "why": (
                    "one nominal robot model shared by every institution, so the "
                    "morphology difference is zero by construction and no "
                    "per-institution execution signature can be explained by "
                    "morphology. Appendix A-27 excludes splicing an institution layer "
                    "across datasets"
                ),
            },
            "endpoint_a": {
                "t_comp_pre_latency_window": a.pre_window.t_comp,
                "t_comp_full_segment": a.whole_segment.t_comp,
                "t_comp_reactive_feedforward_difference": a.difference,
                "t_comp_regressed_out": a.reaction_regressed_out.t_comp,
                "labs_above_threshold": self.labs_above_threshold,
                "threshold": PREREGISTERED_T_COMP_THRESHOLD,
                "min_labs": PREREGISTERED_T_COMP_MIN_LABS,
                "reactive_component_exists": self.reactive_component_exists,
                "prediction_met": self.endpoint_a_prediction_met,
                "primary_reading": "t_comp_pre_latency_window",
                "why_primary": (
                    "human visuomotor latency is 200-300 ms, so a command inside the "
                    "first 200 ms of a motion segment cannot be a reaction to that "
                    "segment's own lag. The full-segment reading alone is excluded "
                    "(appendix A-26): a purely reactive operator also gives T_comp > 1"
                ),
            },
            "budget": self.budget.as_row(),
        }
        if self.endpoint_b is None:
            row["endpoint_b"] = None
        else:
            row["endpoint_b"] = {
                "name": ENDPOINT_B_NAME,
                "median_held_out_relative_drop": self.endpoint_b.median_relative_drop,
                "median_training_relative_drop": (
                    self.endpoint_b.median_training_relative_drop
                ),
                "threshold": PREREGISTERED_VARIANCE_DROP,
                "n_held_out_per_fold": N_HELD_OUT_LABS,
                "n_folds": len(self.endpoint_b.folds),
                "n_admissible_folds": len(self.endpoint_b.admissible_folds),
                "prediction_met": self.endpoint_b_prediction_met,
                "route": "S",
                "uses_exclusion_restriction": (
                    not self.endpoint_b_uses_no_exclusion_restriction
                ),
                "why_it_matters": (
                    "route (S) identifies delta by widening the observation to "
                    "(x, a, Delta x^ach), so it is the only identification route in "
                    "the plan that runs on real hardware data without an (ER)-type "
                    "exclusion restriction. It also bounds the pessimism about the "
                    "common mode: the common mode is unidentifiable only without an "
                    "anchor, and the execution signature is one"
                ),
            }
        return row


def e5_report(
    endpoint_a: TCompReport,
    *,
    lock: LockFile,
    endpoint_b: EndpointBResult | None = None,
    analysed_labs: Sequence[DroidLabId] | None = None,
) -> E5ArmReport:
    """Assemble the arm from its two endpoints, against the release's own roster."""
    roster = droid_roster(lock)
    labs = (
        tuple(DroidLabId(str(x)) for x in analysed_labs)
        if analysed_labs is not None
        else tuple(DroidLabId(str(x)) for x in endpoint_a.per_lab_pre_window)
    )
    return E5ArmReport(
        endpoint_a=endpoint_a,
        endpoint_b=endpoint_b,
        roster=roster,
        analysed_labs=labs,
    )


def _self_check() -> None:
    """10.4.6's cost line and its endpoint list, recomputed from their parts."""
    if E5_BUDGET.gpu_hours != 9.0:
        raise AssertionError(
            f"10.4.6's cost line is 0 complete runs + 9 GPU-h; the parts here give "
            f"{E5_BUDGET.gpu_hours}"
        )
    if E5_BUDGET.complete_runs != 0 or E5_BUDGET.training_arms:
        raise AssertionError("E5 trains nothing, so it declares no training arm")
    if len(ENDPOINT_A_NAMES) != 4:
        raise AssertionError(
            f"10.4.6's endpoint A has four readings; {len(ENDPOINT_A_NAMES)} are named"
        )
    if len(set(ENDPOINT_A_NAMES)) != len(ENDPOINT_A_NAMES):
        raise AssertionError("the endpoint A reading names repeat")
    if ENDPOINT_B_NAME in ENDPOINT_A_NAMES:
        raise AssertionError("endpoint B is not one of endpoint A's readings")
    if PREREGISTERED_T_COMP_MIN_LABS > N_DROID_LABS:
        raise AssertionError(
            f"the prediction asks for {PREREGISTERED_T_COMP_MIN_LABS} institutions out "
            f"of {N_DROID_LABS}"
        )
    if N_HELD_OUT_LABS >= N_DROID_LABS:
        raise AssertionError(
            f"holding out {N_HELD_OUT_LABS} of {N_DROID_LABS} institutions leaves no "
            "training side"
        )


_self_check()
