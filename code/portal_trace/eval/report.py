"""Three-way verdict rendering (proposal 10.7.6, 11.3).

11.3: "each preregistered endpoint has exactly three verdicts: significant, not
significant, inconclusive. The third is a permitted formal conclusion and its
conditions are preregistered too. Rewriting inconclusive as 'the trend supports'
is not allowed."

That last sentence is the reason this module exists as code rather than as a
style guide. Three mechanisms carry it:

1. `Verdict.INCONCLUSIVE` has no soft rendering. `render_verdict` maps it to one
   frozen string that names the underpowering and refuses a direction, and it is
   the only public way to turn a verdict into prose.
2. `attach_narrative` scans caller-written text against a banned list. For an
   inconclusive row the list includes both the softening idioms ("a trend
   towards", "suggestive of", "approaching significance") and *any directional
   verb at all*, because 10.7.6 says an inconclusive endpoint is "reported
   explicitly as underpowered, with no directional statement of any kind". The
   check runs on English and on the Chinese equivalents, since the proposal is
   written in Chinese and a translated draft would otherwise slip through.
3. `ResultRow` refuses to exist without the columns a verdict is derived from --
   the design (M, D, S), the inference kind, the MDE with its caliber -- and a
   Lambda-family row additionally refuses to render without its full null
   distribution. 11.3: "every conclusion involving the Lambda family must give
   the point estimate, the CI, the complete plot of the null distribution used
   (not just a p value), and the MDE."

The inconclusive triggers are the preregistered ones, not a residual bucket:
a CI covering the practical-relevance threshold, a mixed model that failed to
converge or returned a zero variance component, and Leak_beh's denominator below
its floor (11.3), plus the crown 0.7-1.0 x MDE band of 10.7.5/10.7.6. A row that
matches none of the three classification rules is *also* inconclusive, with its
own reason, because the rules of 10.7.6 are not exhaustive and defaulting the
gaps to a decision is how a decision gets made by accident.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from portal_trace.defects.params import DEFECTS_REPORTED_WITH
from portal_trace.eval.endpoints import DenominatorRule, EndpointRole
from portal_trace.eval.stats import (
    ALPHA_PRIMARY,
    CROWN_UNDERPOWER_BAND,
    Design,
    InferenceKind,
    Mde,
    ModelDiagnostics,
)

__all__ = [
    "Verdict",
    "InconclusiveReason",
    "Decision",
    "ResultRow",
    "decide",
    "render_verdict",
    "render_row",
    "render_summary_table",
    "attach_narrative",
    "ForbiddenPhrasingError",
    "IncompleteRowError",
    "NullDistributionDigest",
]


class Verdict(Enum):
    """11.3's three verdicts. There is no fourth, and no ordering between them."""

    SIGNIFICANT = "significant"
    NOT_SIGNIFICANT = "not_significant"
    INCONCLUSIVE = "inconclusive"


class InconclusiveReason(Enum):
    """The preregistered triggers for the third verdict (10.7.6, 11.3)."""

    CI_SPANS_ZERO_AND_MDE = (
        "the 95% CI both contains 0 and extends beyond the MDE: underpowered"
    )
    CI_COVERS_PRACTICAL_THRESHOLD = (
        "the 95% CI covers the preregistered practical-relevance threshold"
    )
    MODEL_DID_NOT_CONVERGE = "the mixed model did not converge"
    ZERO_VARIANCE_COMPONENT = "the mixed model returned a zero variance component"
    LEAK_BEH_DENOMINATOR_BELOW_FLOOR = (
        "Leak_beh's denominator is below the preregistered floor (10.1.2)"
    )
    CROWN_UNDERPOWERED = (
        "a crown endpoint whose measured effect lands in 0.7-1.0 x MDE: underpowered, "
        "and it may not be pushed over the line with a one-sided test or a relaxed alpha"
    )
    NOT_CLASSIFIED_BY_PREREGISTERED_RULES = (
        "none of the three rules of 10.7.6 applies to this combination of CI, point "
        "estimate and adjusted p; the rules are not exhaustive and the gap is not a "
        "decision"
    )


class ForbiddenPhrasingError(ValueError):
    """A narrative would have softened or given a direction to an inconclusive row."""


class IncompleteRowError(ValueError):
    """A row is missing something 11.3 requires before it can be rendered."""


# --- what may not be written about an inconclusive row ---------------------

_SOFTENING_PATTERNS: tuple[str, ...] = (
    r"trend(s|ing)?\s+(toward|towards|to)",
    r"a\s+trend\b",
    r"trend\s+support",
    r"suggestive\s+of",
    r"suggests?\s+that",
    r"points?\s+(toward|towards)",
    r"lean(s|ing)?\s+(toward|towards)",
    r"approach(es|ing)\s+significance",
    r"(marginal|borderline|near(ly)?|almost|close\s+to)\s+significan",
    r"weak\s+evidence",
    r"some\s+evidence",
    r"consistent\s+with\s+(a|an|the)\s+(positive|negative|beneficial|harmful)",
    r"directionally",
    r"in\s+the\s+(expected|predicted|hypothesi[sz]ed)\s+direction",
    r"趋势",
    r"倾向",
    r"边缘显著",
    r"接近显著",
    r"方向.{0,4}一致",
    r"弱证据",
)

_DIRECTIONAL_PATTERNS: tuple[str, ...] = (
    r"\b(higher|lower|better|worse|greater|smaller|larger)\b",
    r"\b(increase[sd]?|decrease[sd]?|improve[sd]?|degrade[sd]?|reduce[sd]?)\b",
    r"\boutperform(s|ed)?\b",
    r"\b(benefit|advantage|penalty)\b",
    r"\b(more|less)\s+leak",
    r"(优于|劣于|上升|下降|提升|降低|更好|更差)",
)


def _compile(patterns: Sequence[str]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


_SOFTENING = _compile(_SOFTENING_PATTERNS)
_DIRECTIONAL = _compile(_DIRECTIONAL_PATTERNS)


# --- rows ------------------------------------------------------------------


@dataclass(frozen=True)
class NullDistributionDigest:
    """A summary of a null distribution, and a handle on the whole thing.

    11.3 asks for the complete distribution rather than a p value, so the array
    itself is carried; the digest exists so a text table has something to print
    without pretending the quantiles are the deliverable.
    """

    values: np.ndarray
    source: str  # "surrogate defects" (primary), "within-family permutation", ...

    def __post_init__(self) -> None:
        if self.values.ndim != 1 or self.values.size < 2:
            raise ValueError(
                f"a null distribution needs at least two draws, got shape {self.values.shape}"
            )
        if not self.source.strip():
            raise ValueError("say which null distribution this is; 11.3 fixes the primary "
                             "one as the surrogate-defect null")

    def quantiles(
        self, qs: Sequence[float] = (0.025, 0.25, 0.5, 0.75, 0.975)
    ) -> dict[float, float]:
        return {q: float(np.quantile(self.values, q)) for q in qs}

    def describe(self) -> str:
        q = self.quantiles()
        return (
            f"null[{self.source}] n={self.values.size} "
            f"min={self.values.min():.4g} q025={q[0.025]:.4g} med={q[0.5]:.4g} "
            f"q975={q[0.975]:.4g} max={self.values.max():.4g}"
        )


@dataclass(frozen=True)
class ResultRow:
    """One row of a summary table, with every column 11.3 and 11.4 require.

    `design` and `inference` have no defaults on purpose. 11.4: "every row of the
    summary table must carry these three numbers, otherwise that row's MDE and
    its inconclusive rule cannot be derived", and 10.7.2 ends with "each row must
    state which inference that row used" -- with a default, both would be
    satisfiable by forgetting.
    """

    label: str
    endpoint: str
    unit: str
    point: float
    ci: tuple[float, float]
    mde: Mde
    design: Design
    inference: InferenceKind
    inference_flag: str
    role: EndpointRole
    n_cells: int
    p_raw: float | None = None
    p_adjusted: float | None = None
    null_distribution: NullDistributionDigest | None = None
    # Set when the row is about one defect, which lets `render_summary_table`
    # enforce 9.2.8's pairing obligations. Optional because most rows are about
    # an arm or a whole family of cells and name no single defect; a row that
    # leaves it `None` is simply not subject to the check.
    defect_code: str | None = None
    is_lambda_family: bool = False
    is_crown: bool = False
    practical_relevance_threshold: float | None = None
    diagnostics: ModelDiagnostics | None = None
    leak_beh_denominator_rule: DenominatorRule | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        lo, hi = self.ci
        if lo > hi:
            raise ValueError(f"{self.label}: CI bounds are inverted, ({lo}, {hi})")
        if self.mde.value <= 0:
            raise ValueError(f"{self.label}: MDE must be positive, got {self.mde.value}")
        if self.mde.unit != self.unit:
            raise ValueError(
                f"{self.label}: the row is in {self.unit!r} but its MDE is in "
                f"{self.mde.unit!r}; 10.1 rule 1 requires each endpoint's MDE to carry "
                "that endpoint's own unit"
            )
        if self.is_lambda_family and self.null_distribution is None:
            raise IncompleteRowError(
                f"{self.label}: 11.3 requires every Lambda-family conclusion to carry "
                "the point estimate, the CI, the COMPLETE null distribution used (not "
                "just a p value) and the MDE. The null distribution is missing; the "
                "primary caliber is the surrogate-defect null, with the random rotation "
                "only as a zero-cost first screen"
            )

    @property
    def ci_within_mde(self) -> bool:
        lo, hi = self.ci
        m = self.mde.value
        return -m <= lo and hi <= m

    @property
    def ci_contains_zero(self) -> bool:
        lo, hi = self.ci
        return lo <= 0.0 <= hi

    @property
    def ci_exceeds_mde(self) -> bool:
        lo, hi = self.ci
        m = self.mde.value
        return lo < -m or hi > m

    @property
    def effect_over_mde(self) -> float:
        return abs(self.point) / self.mde.value


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    reasons: tuple[InconclusiveReason, ...]
    alpha: float

    def __post_init__(self) -> None:
        if self.verdict is Verdict.INCONCLUSIVE and not self.reasons:
            raise ValueError(
                "an inconclusive verdict must name its preregistered trigger; "
                "inconclusive is a conclusion, not an absence of one (11.3)"
            )
        if self.verdict is not Verdict.INCONCLUSIVE and self.reasons:
            raise ValueError(
                f"{self.verdict.name} carries inconclusive reasons {self.reasons}"
            )


def decide(row: ResultRow, *, alpha: float = ALPHA_PRIMARY) -> Decision:
    """The three-way rule of 10.7.6, with 11.3's forced triggers taking priority.

    Order matters. The forced triggers are conditions under which the number in
    the row is not trustworthy enough to classify at all, so they are checked
    before the classification rules; a zero variance component with a small p is
    still inconclusive.
    """
    forced: list[InconclusiveReason] = []

    if row.diagnostics is not None:
        if not row.diagnostics.converged:
            forced.append(InconclusiveReason.MODEL_DID_NOT_CONVERGE)
        if row.diagnostics.zero_variance_components:
            forced.append(InconclusiveReason.ZERO_VARIANCE_COMPONENT)

    if (
        row.leak_beh_denominator_rule is not None
        and row.leak_beh_denominator_rule is not DenominatorRule.PASSES
    ):
        forced.append(InconclusiveReason.LEAK_BEH_DENOMINATOR_BELOW_FLOOR)

    if row.practical_relevance_threshold is not None:
        lo, hi = row.ci
        thr = row.practical_relevance_threshold
        if lo <= thr <= hi or lo <= -thr <= hi:
            forced.append(InconclusiveReason.CI_COVERS_PRACTICAL_THRESHOLD)

    lo_band, hi_band = CROWN_UNDERPOWER_BAND
    if row.is_crown and lo_band <= row.effect_over_mde <= hi_band:
        forced.append(InconclusiveReason.CROWN_UNDERPOWERED)

    if forced:
        return Decision(Verdict.INCONCLUSIVE, tuple(forced), alpha)

    p = row.p_adjusted if row.p_adjusted is not None else row.p_raw
    if p is not None and p < alpha and abs(row.point) > row.mde.value:
        return Decision(Verdict.SIGNIFICANT, (), alpha)

    if row.ci_within_mde:
        return Decision(Verdict.NOT_SIGNIFICANT, (), alpha)

    if row.ci_contains_zero and row.ci_exceeds_mde:
        return Decision(
            Verdict.INCONCLUSIVE, (InconclusiveReason.CI_SPANS_ZERO_AND_MDE,), alpha
        )

    return Decision(
        Verdict.INCONCLUSIVE,
        (InconclusiveReason.NOT_CLASSIFIED_BY_PREREGISTERED_RULES,),
        alpha,
    )


# --- rendering -------------------------------------------------------------

_VERDICT_TEXT: dict[Verdict, str] = {
    Verdict.SIGNIFICANT: (
        "SIGNIFICANT: the adjusted p is below alpha and the point estimate exceeds this "
        "endpoint's MDE"
    ),
    Verdict.NOT_SIGNIFICANT: (
        "NOT SIGNIFICANT: the 95% CI lies entirely within +/- MDE. This is 'no "
        "detectable effect', which is not 'no effect'"
    ),
    Verdict.INCONCLUSIVE: (
        "INCONCLUSIVE: underpowered. No directional statement of any kind is admissible "
        "for this endpoint, and this verdict may not be restated as a trend"
    ),
}


def render_verdict(decision: Decision) -> str:
    """The only sanctioned way to turn a verdict into prose.

    The strings are frozen, and the inconclusive one carries its own prohibition,
    so a table printed from here cannot read as a soft positive.
    """
    text = _VERDICT_TEXT[decision.verdict]
    if decision.verdict is not Verdict.INCONCLUSIVE:
        return text
    reasons = "; ".join(r.value for r in decision.reasons)
    return f"{text}. Trigger: {reasons}"


def render_row(row: ResultRow, *, alpha: float = ALPHA_PRIMARY) -> str:
    """One rendered row, with every mandatory column of 11.3 and 11.4."""
    decision = decide(row, alpha=alpha)
    lo, hi = row.ci
    lines = [
        f"{row.label}  [{row.role.value}]",
        f"  endpoint     : {row.endpoint}",
        f"  point        : {row.point:.4g} {row.unit}",
        f"  95% CI       : [{lo:.4g}, {hi:.4g}] {row.unit}",
        f"  {row.mde.describe()}",
        f"  design       : {row.design} on n = {row.n_cells} cells "
        f"({row.design.label})",
        f"  inference    : {row.inference.value}",
        f"  inference note: {row.inference_flag}",
    ]
    if row.p_raw is not None:
        lines.append(f"  p (raw)      : {row.p_raw:.4g}")
    if row.p_adjusted is not None:
        lines.append(f"  p (adjusted) : {row.p_adjusted:.4g}")
    if row.null_distribution is not None:
        lines.append(f"  {row.null_distribution.describe()}")
        lines.append(
            "  null distribution: the full distribution is published with this row "
            "(11.3); the digest above is not a substitute"
        )
    if row.diagnostics is not None and row.diagnostics.degenerate:
        lines.append(
            "  model        : DEGENERATE -- "
            f"converged={row.diagnostics.converged}, zero variance components="
            f"{list(row.diagnostics.zero_variance_components)}"
        )
    if row.leak_beh_denominator_rule is not None:
        lines.append(f"  denominator rule: {row.leak_beh_denominator_rule.value}")
    for note in row.notes:
        lines.append(f"  note         : {note}")
    lines.append(f"  VERDICT      : {render_verdict(decision)}")
    return "\n".join(lines)


def render_summary_table(rows: Iterable[ResultRow], *, alpha: float = ALPHA_PRIMARY) -> str:
    rows = list(rows)
    if not rows:
        raise ValueError("no rows to render")
    _check_reporting_pairs(rows)
    return "\n\n".join(render_row(r, alpha=alpha) for r in rows)


def _check_reporting_pairs(rows: Sequence[ResultRow]) -> None:
    """9.2.8: a defect confounded with another may not appear on its own.

    D8 is the case that exists. Its compensation field is a constant
    multiplicative map, which is exactly the form a per-family affine action
    normalisation absorbs, so a D8 effect and a confounder-X7 effect are not
    separable from a D8 column alone. D3b's field moves with q and is immune to
    the same normalisation, so the two side by side say which of the two the
    number is. The obligation was a footnote while D8 was priority C; it binds
    now that D8 is one of the core six, and it binds here because the summary
    table is the last place a lone column could still be caught.
    """
    present = {r.defect_code for r in rows if r.defect_code is not None}
    for code, companions in DEFECTS_REPORTED_WITH.items():
        if code not in present:
            continue
        missing = [c for c in companions if c not in present]
        if missing:
            raise IncompleteRowError(
                f"the table reports {code} without {', '.join(missing)}. 9.2.8 "
                f"records {code} as strongly coupled with confounder X7, so its "
                "effect cannot be told apart from an action-normalisation effect "
                f"on its own; {', '.join(companions)} has to be in the same table"
            )


def attach_narrative(row: ResultRow, text: str, *, alpha: float = ALPHA_PRIMARY) -> str:
    """Return `text` unchanged, or refuse it.

    For an inconclusive row this rejects both the softening idioms and any
    directional wording, in English and in Chinese. 10.7.6 requires an
    inconclusive endpoint to be "reported explicitly as underpowered, with no
    directional statement", and 11.3 forbids rewriting it as a trend; a renderer
    that accepted "the Oracle arm was higher, though inconclusive" would satisfy
    neither.

    Rows that are significant or not significant pass through: the prohibition is
    on softening a *non-*conclusion, not on describing a conclusion.
    """
    decision = decide(row, alpha=alpha)
    if decision.verdict is not Verdict.INCONCLUSIVE:
        return text

    offences: list[str] = []
    for pattern in _SOFTENING:
        match = pattern.search(text)
        if match is not None:
            offences.append(f"softening phrase {match.group(0)!r}")
    for pattern in _DIRECTIONAL:
        match = pattern.search(text)
        if match is not None:
            offences.append(f"directional wording {match.group(0)!r}")

    if offences:
        raise ForbiddenPhrasingError(
            f"{row.label} is inconclusive ("
            + "; ".join(r.value for r in decision.reasons)
            + "). The proposed narrative contains: "
            + "; ".join(sorted(set(offences)))
            + ". 11.3 does not allow inconclusive to be rewritten as a trend, and "
            "10.7.6 forbids any directional statement on an underpowered endpoint. "
            "Report the point estimate, the CI, the MDE and the word 'inconclusive'."
        )
    return text
