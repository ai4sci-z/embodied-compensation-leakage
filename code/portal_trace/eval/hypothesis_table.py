"""10.10's table of empirical propositions, as data rather than as prose.

The proposal's 10.10 is the one place where every claim this study makes is
lined up against the experiment that carries it, the endpoint that reads it, the
prediction registered in advance, and the numbered exit that fires if the
prediction fails. In the document it is a Markdown table, which means nothing
stops an endpoint from being renamed, an experiment from being re-scoped or an
exit from being re-worded while the table goes on describing the study as it was
planned rather than as it is built. That is exactly the drift the preregistration
machinery exists to prevent everywhere else.

So the table is transcribed here once, and every cross-reference in it is
resolved against the live configuration at import time:

* each carrying experiment has to exist in the bundle;
* each endpoint has to exist on one of that row's carrying experiments, under
  the name the row gives it;
* each exit has to be one of the nine registered in the preregistration;
* each row keyed by a T number has to be a live registered hypothesis -- and
  the retired T4 is refused specifically, because it is the number most likely
  to come back by copy-and-paste.

A row that fails any of these raises `HypothesisTableError` on import, which is
the whole point: a rename that silences a claim should stop the process, not
produce a report with a claim missing.

The second thing here is exit routing. Every experiment's `zero_result_exit`
opens with its F number, and 10.9 gives each F number a deliverable -- what the
study reports *instead* when that branch fires. `route_exit` maps the number to
that deliverable and to the experiments that can trigger it, so a null result
lands somewhere specified in advance rather than becoming a hole in the writeup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from portal_trace.configs.loader import ConfigBundle, load_bundle

__all__ = [
    "HypothesisTableError",
    "HypothesisRow",
    "ExitRoute",
    "HYPOTHESIS_TABLE",
    "RETIRED_HYPOTHESIS_NUMBERS",
    "THESIS_LEVEL_EXITS",
    "check_table_resolves",
    "route_exit",
    "exit_of_experiment",
    "rows_for_experiment",
]


class HypothesisTableError(RuntimeError):
    """A row of 10.10 no longer resolves against the configuration."""


# 10.10 retires T4 into T2 and T3, so the numbering jumps from T3 to T5. Named
# here as well as in the preregistration because this is the file a new row gets
# written in, and a row keyed T4 has to be refused at the point of writing.
RETIRED_HYPOTHESIS_NUMBERS: frozenset[str] = frozenset({"T4"})


@dataclass(frozen=True)
class HypothesisRow:
    """One row of 10.10.

    `hypothesis` is the T number when the row is keyed by one; the first eleven
    rows of the table are keyed by a prose proposition instead, and carry
    `None`. Both kinds are here because both are claims the study makes -- the
    prose rows are the ones that span several T numbers at once.
    """

    proposition: str
    hypothesis: str | None
    experiments: tuple[str, ...]
    endpoints: tuple[str, ...]
    prediction: str
    falsification: str
    exit: str | None

    def __post_init__(self) -> None:
        if self.hypothesis in RETIRED_HYPOTHESIS_NUMBERS:
            raise HypothesisTableError(
                f"row {self.proposition!r} is keyed by {self.hypothesis}, which "
                "10.10 retired into T2 and T3. Its content belongs to T9; a row "
                "filed under a retired number reads as a resurrected prediction"
            )
        if not self.experiments:
            raise HypothesisTableError(
                f"row {self.proposition!r} names no carrying experiment. An "
                "empirical proposition with nothing carrying it is not under test"
            )


# ---------------------------------------------------------------------------
# The table (10.10). Order is the proposal's.
# ---------------------------------------------------------------------------

HYPOTHESIS_TABLE: tuple[HypothesisRow, ...] = (
    HypothesisRow(
        proposition=(
            "whether the leakage of a BC-trained policy on a disconnected design "
            "graph reaches the ceiling the information-layer null space allows"
        ),
        hypothesis="T11",
        experiments=("E8",),
        endpoints=("connectivity_slope",),
        prediction=(
            "slope beta_hat >= 0.045 (MDE 0.042); rung B exceeds rung A by "
            ">= 0.10, reported alongside as the ceiling anchor"
        ),
        falsification=(
            "the empirical bridge T11 breaks -- a significant negative result for "
            "the whole route of guiding data design by identifiability arguments"
        ),
        exit="F9",
    ),
    HypothesisRow(
        proposition=(
            "whether the learner still cannot separate compensation on a "
            "single-embodiment pool at constant delta (proposition 1(b)'s scope, "
            "reproduced on the learner)"
        ),
        hypothesis=None,
        experiments=("E10", "E9"),
        endpoints=("delta_recovery_error", "isolation_slope"),
        prediction=(
            "both rungs' relative recovery error >= 0.80 and within 0.10 of each "
            "other; per-embodiment fine-tuning has the highest Lambda"
        ),
        falsification=(
            "compensation turns out to be identifiable under a single embodiment, "
            "so the hypothesis class is not closed and M has to be recharacterised"
        ),
        exit=None,
    ),
    HypothesisRow(
        proposition=(
            "whether the learner absorbs only the contrast component of the "
            "compensation and leaves the common mode"
        ),
        hypothesis="T9",
        experiments=("E4b", "E7"),
        endpoints=("lambda_emp_slope_pair", "success_rate_K_trend"),
        prediction=(
            "success rate rises monotonically in K; the lambda_emp_res slope CI "
            "falls within +/-0.033 AND the lambda_emp_con slope is <= -0.08"
        ),
        falsification=(
            "diversity is not the remedy (E4b side), or the common mode can be "
            "stripped after all (E7 side)"
        ),
        exit="F4",
    ),
    HypothesisRow(
        proposition=(
            "whether each branch of proposition 1(d)'s five-way alternative has a "
            "usable instance in this setting"
        ),
        hypothesis=None,
        experiments=("E4b", "E7", "E10", "E5"),
        endpoints=(
            "delta_recovery_error",          # (P)
            "t_comp_pre_latency_window",     # (S)
            "lambda_emp_slope_pair",         # (D)
        ),
        prediction=(
            "(P) recovery error <= 0.25 under the (ER) channel excision; (S) the "
            "held-out-laboratory variance component falls by >= 15%, without "
            "needing (ER); (I) is significant at N >= 20 target demonstrations"
        ),
        falsification=(
            "a branch with no instance has no carrier in this setting and must be "
            "labelled as such"
        ),
        exit=None,
    ),
    HypothesisRow(
        proposition=(
            "whether the measured deviation of the two-way additive model falls "
            "inside the bias bound, and whether the bound's two factors act as "
            "characterised"
        ),
        hypothesis=None,
        experiments=("E12", "E8"),
        endpoints=("theta_gamma_two_calibers", "xi_g"),
        prediction=(
            "both calibers reported; the twin-paired caliber has the smaller "
            "ratio, the task-scale first term having cancelled"
        ),
        falsification=(
            "the bound is loose or false, so the additive model's bias "
            "characterisation fails and only proposition 1(e') survives"
        ),
        exit=None,
    ),
    HypothesisRow(
        proposition=(
            "whether the two degraded forms -- regime stratification and the "
            "low-order component -- have carriers in this data"
        ),
        hypothesis=None,
        experiments=("A6", "A10"),
        endpoints=("a10_e_phi_by_basis", "a6_per_regime_connectivity"),
        prediction=(
            "at least 3 regimes connected; E_phi passes its three-tier rule"
        ),
        falsification=(
            "regimes are generally disconnected, stratified identification has no "
            "carrier and only the global additive form is left"
        ),
        exit=None,
    ),
    HypothesisRow(
        proposition=(
            "whether the two transfer directions differ in harm type, and whether "
            "the sign flips with the terminal directional margin"
        ),
        hypothesis="T8",
        experiments=("E11",),
        endpoints=("failure_type_profile",),
        prediction=(
            "D->C and C->D differ in failure TYPE; the sign flips with sign(A), "
            "predicted only at rungs where the angle is < 30 degrees"
        ),
        falsification=(
            "the geometric characterisation is wrong and degrades to a "
            "qualitative statement; or it is unmeasurable because the per-rung "
            "feasible set is empty"
        ),
        exit="F8",
    ),
    HypothesisRow(
        proposition=(
            "whether Lambda's slope in ln R_eff is of the same order as the GLS "
            "variance scale"
        ),
        hypothesis=None,
        experiments=("E12", "E8"),
        endpoints=("variance_ratio_vs_proposition3", "connectivity_slope"),
        prediction=(
            "the ratio falls in [0.7, 1.4]; slope beta_hat >= 0.045. The slope has "
            "no theoretical predicted value -- only its sign and monotonicity are "
            "preregistrable"
        ),
        falsification=(
            "E12 passing while E8 fails means the bridge broke, not that "
            "proposition 3 is wrong; both failing means the independence or "
            "homoscedasticity caliber does not apply"
        ),
        exit="F9",
    ),
    HypothesisRow(
        proposition=(
            "whether deployment harm grows only with the mismatch field and not "
            "with the absolute amplitude of the source compensation"
        ),
        hypothesis="T10",
        experiments=("E11",),
        endpoints=("failure_type_profile",),
        prediction=(
            "the coefficient on |Delta c| is significantly > 0; the coefficient on "
            "|c_delta_src| is not significant once the former is controlled for"
        ),
        falsification=(
            "the mismatch framework is incomplete and harm is partly set by the "
            "absolute amplitude of the source compensation"
        ),
        exit=None,
    ),
    HypothesisRow(
        proposition=(
            "whether parameter isolation really reconstructs single-embodiment "
            "degradation on the learner, rather than reconstructing data "
            "fragmentation"
        ),
        hypothesis="T6",
        experiments=("E9",),
        endpoints=("isolation_slope", "i3_vs_i3prime_paired_difference"),
        prediction=(
            "slope beta_hat >= 0.12 (MDE about 0.078 on the measured ladder "
            "0 / 0.28 / 0.42 / 1.00, 10.4.9); I3 - I0 >= 0.12; I3 - I3' >= 0.06"
        ),
        falsification=(
            "I3 about equal to I3' means what was measured is data fragmentation "
            "and bridge T6 is unsupported; an insignificant slope breaks the bridge"
        ),
        exit=None,
    ),
    HypothesisRow(
        proposition=(
            "whether the policy is fitting the pooled conditional mean (lemma 1's "
            "empirical counterpart)"
        ),
        hypothesis="T1",
        experiments=("E1", "E3"),
        endpoints=("lambda_part_excess",),
        prediction=(
            "the difference is > 0 after subtracting the reference-arm alignment "
            "term; masking under B9 shrinks it by >= 60%. The covariance term is "
            "reported on the pooled state marginal"
        ),
        falsification="the policy is not fitting the pooled conditional mean",
        exit="F1",
    ),
    HypothesisRow(
        proposition=(
            "T2: leakage falls when the pool anchors clean behaviour over the "
            "off-support state distribution"
        ),
        hypothesis="T2",
        experiments=("E3",),
        endpoints=("lambda_part_dose_slope",),
        prediction=(
            "flat at low rungs, monotone decreasing at high rungs (rho <= -0.7 on "
            "rho^off); the turning rung must be reported; absolute error rises "
            "monotonically (rho >= 0.8); surrogate fields and rotation are flat "
            "for both"
        ),
        falsification=(
            "leakage cannot be separated from a general performance loss"
        ),
        exit=None,
    ),
    HypothesisRow(
        proposition="T3 (exploratory): the observation modality changes inheritance",
        hypothesis="T3",
        experiments=("E4b",),
        endpoints=("t3_two_point_paired_difference",),
        prediction=(
            "the differing-defect arm beats the identical-defect arm. At (4,1,3) "
            "this endpoint has an MDE of 9.4 pp, hence exploratory; 9.5 pp is the "
            "minimum effect it would need to be confirmatory"
        ),
        falsification="no empirical meaning; the narrative reversal fails",
        exit=None,
    ),
    HypothesisRow(
        proposition=(
            "T5: per-task leakage rises with the occupancy of the regimes in which "
            "the defect is active"
        ),
        hypothesis="T5",
        experiments=("E1",),
        endpoints=("lambda_part_regime_occupancy_slope",),
        prediction=(
            "slope significantly > 0, Spearman rho >= 0.6 over 8 tasks x 6 "
            "defects. Read on rollouts that already exist, so zero training"
        ),
        falsification="the bias bound characterises something other than the leading term",
        exit=None,
    ),
    HypothesisRow(
        proposition="T7: single-frame and history policies leak differently",
        hypothesis="T7",
        experiments=("A2",),
        endpoints=("a2_lambda_by_window", "a2_z_probe_accuracy_by_window"),
        prediction=(
            "the regime distribution of Lambda^part differs significantly between "
            "K_hist = 1 and K_hist = 10 (chi-squared), and the z probe's accuracy "
            "differs with it"
        ),
        falsification=(
            "hidden state does not matter -- favourable for this proposal, since "
            "the regime machinery could then be simplified"
        ),
        exit=None,
    ),
    HypothesisRow(
        proposition="T12: morphology distance does not predict gradient cosine",
        hypothesis="T12",
        experiments=("E2prime",),
        endpoints=("gradient_cosine_two_sample",),
        prediction=(
            "the twin pairs and the cross-DoF pairs are indistinguishable in "
            "distribution. Zero new training"
        ),
        falsification=(
            "morphology clustering separates the twin pair after all, so the "
            "grouping signal sees a defect the geometric descriptor cannot"
        ),
        exit="F3",
    ),
)


# The two exits no single experiment can fire, named so that "no experiment
# routes here" reads as a fact about the exit rather than as a wiring omission.
# F2 is a thesis pivot triggered by a *combination* -- Lambda^part significantly
# positive while the success rate does not fall -- which is a joint reading of
# E1 and E3 rather than either one's null result, and 10.9.1 describes it that
# way. F7 is a budget overrun, which no measurement produces.
THESIS_LEVEL_EXITS: frozenset[str] = frozenset({"F2", "F7"})


@dataclass(frozen=True)
class ExitRoute:
    """Where a numbered null result is reported instead (10.9)."""

    number: str
    deliverable: str
    experiments: tuple[str, ...]

    def __str__(self) -> str:
        carried = ", ".join(self.experiments) if self.experiments else "no experiment"
        return f"{self.number} (from {carried}): {self.deliverable}"


# Any F followed by digits, not just the nine valid ones: an exit written as
# "F13" has to be refused, and a pattern that only recognises F1-F9 would read
# it as prose and route the null result nowhere.
_EXIT_PREFIX = re.compile(r"^\s*(F\d+)\b")


def exit_of_experiment(experiment_id: str, *, bundle: ConfigBundle | None = None) -> str | None:
    """The F number an experiment's `zero_result_exit` opens with, if any.

    Most experiments state their null result in prose, and that is not an
    oversight: 10.9's nine numbered exits are the branches that redirect the
    *thesis*, while a row like E4's "every form's cosine near the null level" is
    a local reading that changes no deliverable. Returning None for those is the
    honest answer.

    What is checked is the case that can go wrong silently: a row that does open
    with an F number has to open with a registered one, or the null result is
    routed to a deliverable that does not exist.
    """
    b = bundle if bundle is not None else load_bundle()
    text = getattr(b.experiment(experiment_id), "zero_result_exit", None)
    if not text or not text.strip():
        return None
    match = _EXIT_PREFIX.match(text)
    if match is None:
        return None
    number = match.group(1)
    if number not in b.preregistration.exits:
        raise HypothesisTableError(
            f"{experiment_id} routes its null result to {number}, which is not one "
            f"of the registered exits {sorted(b.preregistration.exits)}"
        )
    return number


def route_exit(number: str, *, bundle: ConfigBundle | None = None) -> ExitRoute:
    """Resolve an F number to its deliverable and the experiments that fire it.

    Two sources, unioned: the experiments whose `zero_result_exit` opens with
    this number, and the experiments that 10.10's table routes here. Both are
    needed -- 10.10 assigns T11's and T8's exits to experiments whose config row
    states the null in prose -- and disagreement between them is not possible to
    hide, since the union is what gets reported.
    """
    b = bundle if bundle is not None else load_bundle()
    if number not in b.preregistration.exits:
        raise HypothesisTableError(
            f"{number!r} is not a registered exit; the nine are "
            f"{sorted(b.preregistration.exits)}"
        )
    from_config = {
        exp_id
        for exp_id in b.all_experiments
        if exit_of_experiment(exp_id, bundle=b) == number
    }
    from_table = {
        exp_id
        for row in HYPOTHESIS_TABLE
        if row.exit == number
        for exp_id in row.experiments
    }
    return ExitRoute(
        number=number,
        deliverable=b.preregistration.exits[number],
        experiments=tuple(sorted(from_config | from_table)),
    )


def rows_for_experiment(experiment_id: str) -> tuple[HypothesisRow, ...]:
    """Every claim of 10.10 that this experiment carries."""
    return tuple(r for r in HYPOTHESIS_TABLE if experiment_id in r.experiments)


@lru_cache(maxsize=1)
def check_table_resolves() -> tuple[str, ...]:
    """Resolve every cross-reference in 10.10 against the live configuration.

    Returns the resolved `experiment.endpoint` keys so a caller can see what was
    checked; raises `HypothesisTableError` naming every row that fails, all of
    them at once, since a rename usually breaks several rows together.
    """
    bundle = load_bundle()
    prereg = bundle.preregistration
    known = set(bundle.all_experiments)
    problems: list[str] = []
    resolved: list[str] = []

    for row in HYPOTHESIS_TABLE:
        missing = [e for e in row.experiments if e not in known]
        if missing:
            problems.append(
                f"row {row.proposition[:60]!r} names experiment(s) "
                f"{missing}, which are not in the bundle"
            )
            continue

        if row.hypothesis is not None and row.hypothesis not in prereg.hypotheses:
            problems.append(
                f"row {row.proposition[:60]!r} is keyed by {row.hypothesis}, which "
                "is not a live registered hypothesis"
            )

        for endpoint in row.endpoints:
            carriers = [
                e for e in row.experiments
                if any(ep.name == endpoint for ep in bundle.experiment(e).endpoints)
            ]
            if not carriers:
                problems.append(
                    f"row {row.proposition[:60]!r} reads endpoint {endpoint!r}, "
                    f"which none of {list(row.experiments)} declares"
                )
            else:
                resolved.append(f"{carriers[0]}.{endpoint}")

        if row.exit is not None and row.exit not in prereg.exits:
            problems.append(
                f"row {row.proposition[:60]!r} exits to {row.exit}, which is not "
                "registered"
            )

    covered = {h for row in HYPOTHESIS_TABLE if (h := row.hypothesis) is not None}
    uncarried = sorted(
        set(prereg.hypotheses) - covered, key=lambda s: int(s[1:])
    )
    if uncarried:
        problems.append(
            f"registered hypotheses {uncarried} appear in no row of 10.10, so "
            "nothing in the table says which experiment tests them"
        )

    orphaned = sorted(
        number
        for number in prereg.exits
        if number not in THESIS_LEVEL_EXITS
        and not route_exit(number, bundle=bundle).experiments
    )
    if orphaned:
        problems.append(
            f"exits {orphaned} have no experiment routing to them and are not "
            f"declared thesis-level ({sorted(THESIS_LEVEL_EXITS)}). An exit "
            "nothing can fire is either mis-wired or should say so"
        )

    if problems:
        raise HypothesisTableError(
            "10.10's table no longer resolves against the configuration:\n\n"
            + "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(problems))
        )
    return tuple(resolved)
