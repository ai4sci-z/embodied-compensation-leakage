"""E4b's Q_div report: the removable share of the injected fields, per K rung.

Proposal 10.4.5, with 7.5's definition and appendix A-20's and A-24's two scope
rules. E4b sweeps K in {1, 2, 4, 6} sources, each carrying a *different* defect,
at a fixed total of 1 200 added demonstrations per morphology family, and adds
one control at K = 4 whose four sources carry *identical* defect parameters.
Q_div = 1 - ||gamma_bar_r||^2 / E_k ||gamma_{delta_k,r}||^2 is the energy share
of the compensation that source diversity makes identifiable at all, so the
sweep answers "is this set of bodies spread out enough" before anything is
trained.

Three rules decide what this module is allowed to produce.

**It is a design diagnostic and never a preregistered endpoint (A-24).** It
contains no policy quantity: gamma is built from the injected ground-truth
fields, so every value here is computable the moment the sources are chosen.
`EndpointSpec.__post_init__` already refuses to register it as primary or
secondary, and `QDivReport.computable_before_training` is True on every row so
that a table cannot present it as an outcome.

**Uniform mean, or nothing (A-20, 7.5 scope note 2).** Under the training
mixture weights the projection onto the unidentifiable subspace stops being
orthogonal, the identity stops being an energy decomposition and the word
"share" stops meaning anything. `CompensationEffects` enforces this on the mean
itself; what this module adds is that the *rung's* mixture weights are recorded
next to the value, because the weights are what the reader would otherwise
assume were used.

**An energy share is never compared with Lambda (7.5 scope note 3).** The two
differ by a square. `Q_div` is an `Energy` and the Lambda family is an
`Amplitude`, and `units` refuses the comparison at the type level; what remains
possible in a *report* is putting them in adjacent columns of the same table,
so the row carries `comparable_with` and the K = 4 rows carry Phi rather than
Lambda when a comparison is wanted.

Two rungs of the sweep behave differently from the rest and both are structural.

At K = 1 there is no mean over sources to subtract, and 7.5's decomposition
needs K >= 2; Q_div is not zero there, it is undefined, and a zero would read as
"this body's compensation is entirely common mode", which is a statement the
K = 1 rung cannot make. `QDivUndefinedAtSingleSourceError` says so instead.

The identical-defect control is the opposite: its Q_div is exactly zero and it
is zero *by construction*, which is the degenerate case 7.5 names -- the
information condition falls back to proposition 1(b). It is reported as an
anchor with that label rather than as a fifth data point, and the check that
matters for it is the one 10.4.5 spells out: it must carry the same morphology
diversity as the distinct-defect rungs, or "K identical defects" collapses into
one embodiment and the sweep confounds defect diversity with morphology
diversity.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from portal_trace.diagnostics.common_mode import CompensationEffects
from portal_trace.diagnostics.units import Energy
from portal_trace.types import Regime

__all__ = [
    "E4B_K_LADDER",
    "E4B_TOTAL_ADDED_DEMOS_PER_FAMILY",
    "E4B_CONTROL_K",
    "SourceSpec",
    "DiversityRung",
    "QDivRow",
    "QDivReport",
    "QDivUndefinedAtSingleSourceError",
    "UnmatchedDataVolumeError",
    "UnmatchedMorphologyDiversityError",
    "ControlNotIdenticalError",
    "RungNotDistinctError",
]

E4B_K_LADDER: tuple[int, ...] = (1, 2, 4, 6)
"""10.4.5's K rungs, each source carrying a different defect."""

E4B_TOTAL_ADDED_DEMOS_PER_FAMILY = 1_200
"""Fixed total added data per morphology family: larger K means fewer per source."""

E4B_CONTROL_K = 4
"""The all-identical-defect control sits at K = 4, at matched data volume."""


class QDivUndefinedAtSingleSourceError(ValueError):
    """Q_div was asked for at K = 1."""


class UnmatchedDataVolumeError(ValueError):
    """A rung does not spend 10.4.5's fixed total added data."""


class UnmatchedMorphologyDiversityError(ValueError):
    """The control's morphology diversity does not match the distinct-defect rung."""


class ControlNotIdenticalError(ValueError):
    """The all-identical-defect control carries sources that differ."""


class RungNotDistinctError(ValueError):
    """A distinct-defect rung carries two sources with the same defect parameters."""


@dataclass(frozen=True)
class SourceSpec:
    """One source body of a rung: its family, its defect and its share of the pool.

    `defect_parameters` is the injected delta vector, not a label. The control
    arm's definition is "identical defect parameters across four *different*
    families", so a label comparison would call two differently parameterised
    instances of the same defect class identical and would let the control pass
    a check it fails.
    """

    family: str
    defect: str
    defect_parameters: tuple[float, ...]
    n_demos: int

    def __post_init__(self) -> None:
        if self.n_demos <= 0:
            raise ValueError(f"{self.family}/{self.defect}: n_demos must be positive")
        if not self.defect_parameters:
            raise ValueError(
                f"{self.family}/{self.defect}: no injected defect parameters. Q_div is "
                "built from the injected ground-truth fields, so the parameters are "
                "what identifies the source"
            )
        if not all(math.isfinite(v) for v in self.defect_parameters):
            raise ValueError(f"{self.family}/{self.defect}: non-finite defect parameters")


@dataclass(frozen=True)
class DiversityRung:
    """One rung of the K sweep, with the gamma fields its Q_div is read from.

    `gamma` is keyed by the same source keys as `sources`, one regime-r mean per
    source (7.1). The regime is a field because Q_div is defined per regime and
    a Q_div averaged over regimes is not a Q_div of anything.
    """

    label: str
    k: int
    regime: Regime
    sources: tuple[SourceSpec, ...]
    gamma: Mapping[str, np.ndarray]
    identical_defect_control: bool = False

    def __post_init__(self) -> None:
        if self.k != len(self.sources):
            raise ValueError(
                f"rung {self.label}: K = {self.k} but {len(self.sources)} sources. "
                "14.5 E-2: K is the number of source bodies and nothing else"
            )
        keys = {s.family + "|" + s.defect for s in self.sources}
        if len(keys) != len(self.sources):
            raise ValueError(f"rung {self.label}: two sources share a (family, defect) key")
        if set(self.gamma) != keys:
            raise ValueError(
                f"rung {self.label}: gamma is keyed by {sorted(self.gamma)}, the "
                f"sources are {sorted(keys)}"
            )
        total = sum(s.n_demos for s in self.sources)
        if total != E4B_TOTAL_ADDED_DEMOS_PER_FAMILY:
            raise UnmatchedDataVolumeError(
                f"rung {self.label}: {total} added demonstrations per morphology "
                f"family, 10.4.5 fixes {E4B_TOTAL_ADDED_DEMOS_PER_FAMILY} on every "
                "rung. A K sweep at unequal volume measures data volume as well as "
                "diversity, and the control arm's whole point is matched volume"
            )
        parameters = [s.defect_parameters for s in self.sources]
        if self.identical_defect_control:
            if len(set(parameters)) != 1:
                raise ControlNotIdenticalError(
                    f"rung {self.label} is declared the all-identical-defect control "
                    f"but carries {len(set(parameters))} distinct defect parameter "
                    "vectors"
                )
        elif len(set(parameters)) != len(parameters):
            raise RungNotDistinctError(
                f"rung {self.label}: 10.4.5 gives every source of a K rung a "
                "*different* defect; two sources here carry the same parameters, which "
                "makes this rung a partial copy of the identical-defect control"
            )

    @property
    def source_keys(self) -> tuple[str, ...]:
        return tuple(sorted(s.family + "|" + s.defect for s in self.sources))

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted({s.family for s in self.sources}))

    @property
    def mixture_weights(self) -> dict[str, float]:
        """The training mixture, recorded but never used in the mean (A-20)."""
        total = sum(s.n_demos for s in self.sources)
        return {s.family + "|" + s.defect: s.n_demos / total for s in self.sources}

    def effects(self) -> CompensationEffects:
        if self.k < 2:
            raise QDivUndefinedAtSingleSourceError(
                f"rung {self.label} has K = 1. Q_div is 1 - ||gamma_bar||^2 / "
                "E_k||gamma_k||^2 over K >= 2 sources (7.5); with one source the mean "
                "is that source and the expression is 0/0, not zero. Reporting zero "
                "would read as 'this body's compensation is entirely common mode', "
                "which is a claim about a population of one. The K = 1 rung enters "
                "E4b's success-rate and Lambda^part columns and leaves the Q_div "
                "column empty"
            )
        return CompensationEffects(regime=self.regime, gamma=dict(self.gamma))

    def q_div(self) -> Energy:
        return self.effects().q_div


@dataclass(frozen=True)
class QDivRow:
    """One rung's line of the Q_div table."""

    label: str
    k: int
    regime: Regime
    families: tuple[str, ...]
    mixture_weights: dict[str, float]
    q_div: float | None
    mean_energy: float | None
    common_mode_energy: float | None
    identical_defect_control: bool
    undefined_because: str | None

    def as_row(self) -> dict[str, Any]:
        return {
            "rung": self.label,
            "K": self.k,
            "regime": self.regime.value,
            "n_families": len(self.families),
            "families": list(self.families),
            "mixture_weights": self.mixture_weights,
            "mean_convention": "uniform over sources (7.5 scope note 2, A-20)",
            "q_div": self.q_div,
            "mean_energy": self.mean_energy,
            "common_mode_energy": self.common_mode_energy,
            "undefined_because": self.undefined_because,
            "identical_defect_control": self.identical_defect_control,
            "comparable_with": "Phi = Lambda^2; never with Lambda (7.5 scope note 3)",
            "computable_before_training": True,
        }


class QDivReport:
    """E4b's Q_div column: every rung, with the control checked against 10.4.5.

    The rungs are held together rather than reported one at a time because the
    control is only interpretable next to the distinct-defect rung at the same
    K: on its own a Q_div of zero says "these bodies are identical", which is
    true of the control by construction and says nothing about the design.
    """

    def __init__(self, rungs: Sequence[DiversityRung]) -> None:
        if not rungs:
            raise ValueError("the Q_div report has no rungs")
        labels = [r.label for r in rungs]
        if len(set(labels)) != len(labels):
            raise ValueError(f"rung labels must be unique, got {labels}")
        regimes = {r.regime for r in rungs}
        if len(regimes) != 1:
            raise ValueError(
                f"the rungs mix regimes {sorted(x.value for x in regimes)}. Q_div is "
                "defined per regime (7.5); one table column is one regime"
            )
        self._rungs = tuple(rungs)
        self._controls = tuple(r for r in self._rungs if r.identical_defect_control)
        if len(self._controls) > 1:
            raise ValueError(
                f"10.4.5 has one identical-defect control; this report has "
                f"{[c.label for c in self._controls]}"
            )

    @property
    def rungs(self) -> tuple[DiversityRung, ...]:
        return self._rungs

    @property
    def control(self) -> DiversityRung | None:
        return self._controls[0] if self._controls else None

    def distinct_rung_at(self, k: int) -> DiversityRung:
        matches = [
            r for r in self._rungs if r.k == k and not r.identical_defect_control
        ]
        if len(matches) != 1:
            raise KeyError(
                f"expected one distinct-defect rung at K = {k}, found "
                f"{[m.label for m in matches]}"
            )
        return matches[0]

    def require_control_matches_morphology_diversity(self) -> None:
        """10.4.5's condition on the control arm, as a check rather than a note.

        Same number of morphology families as the distinct-defect rung at the
        same K. Without it the control's four sources can be four copies of one
        body, and the arm stops being "same morphology diversity, no defect
        diversity" and becomes "4x data from one body".
        """
        control = self.control
        if control is None:
            raise ValueError(
                "no identical-defect control in this report. 10.4.5 makes it the other "
                "side of T3: without it a rising K curve cannot be told apart from a "
                "rising data-diversity curve"
            )
        distinct = self.distinct_rung_at(control.k)
        if len(control.families) != len(distinct.families):
            raise UnmatchedMorphologyDiversityError(
                f"the control spans {len(control.families)} morphology families "
                f"{list(control.families)} against rung {distinct.label}'s "
                f"{len(distinct.families)} {list(distinct.families)}. 10.4.5: the "
                "control must match the distinct-defect arms' morphology diversity, or "
                "'K identical defects' with identical morphology is one embodiment and "
                "the experiment degenerates into 4x data from one body against 4 "
                "different defective bodies"
            )
        control_total = sum(s.n_demos for s in control.sources)
        distinct_total = sum(s.n_demos for s in distinct.sources)
        if control_total != distinct_total:
            raise UnmatchedDataVolumeError(
                f"the control spends {control_total} demonstrations per family against "
                f"rung {distinct.label}'s {distinct_total}"
            )

    def row(self, rung: DiversityRung) -> QDivRow:
        try:
            effects = rung.effects()
        except QDivUndefinedAtSingleSourceError as exc:
            return QDivRow(
                label=rung.label,
                k=rung.k,
                regime=rung.regime,
                families=rung.families,
                mixture_weights=rung.mixture_weights,
                q_div=None,
                mean_energy=None,
                common_mode_energy=None,
                identical_defect_control=rung.identical_defect_control,
                undefined_because=str(exc),
            )
        return QDivRow(
            label=rung.label,
            k=rung.k,
            regime=rung.regime,
            families=rung.families,
            mixture_weights=rung.mixture_weights,
            q_div=float(effects.q_div.value),
            mean_energy=float(effects.mean_energy.value),
            common_mode_energy=float(effects.common_mode_energy.value),
            identical_defect_control=rung.identical_defect_control,
            undefined_because=None,
        )

    def rows(self) -> list[QDivRow]:
        return [self.row(r) for r in self._rungs]

    def monotone_in_k(self) -> bool:
        """Whether Q_div rises with K over the rungs where it is defined.

        Not a test and not registered as one: 10.4.5 puts the trend test on the
        success rate. This is the design-side reading of the same axis, and it
        is here so that a Q_div column that fails to rise is visible next to a
        success-rate column that does.
        """
        defined = [
            (r.k, r.q_div)
            for r in self.rows()
            if r.q_div is not None and not r.identical_defect_control
        ]
        ordered = [v for _, v in sorted(defined)]
        if len(ordered) < 2:
            raise ValueError(
                "fewer than two rungs have a defined Q_div, so there is no trend to "
                "read. K = 1 has none by 7.5's definition"
            )
        return all(b >= a for a, b in zip(ordered, ordered[1:], strict=False))

    def control_is_degenerate(self, *, tolerance: float = 1e-9) -> bool:
        """The control's Q_div should be exactly zero, which 7.5 calls degenerate."""
        control = self.control
        if control is None:
            raise ValueError("this report has no identical-defect control")
        return abs(float(control.q_div().value)) <= tolerance

    def as_report(self) -> dict[str, Any]:
        self.require_control_matches_morphology_diversity()
        control = self.control
        return {
            "endpoint": "q_div",
            "kind": "design_diagnostic",
            "why_not_an_endpoint": (
                "no policy quantity and computable before training (appendix A-24)"
            ),
            "regime": self._rungs[0].regime.value,
            "mean_convention": "uniform (7.5 scope note 2, appendix A-20)",
            "total_added_demos_per_family": E4B_TOTAL_ADDED_DEMOS_PER_FAMILY,
            "rows": [r.as_row() for r in self.rows()],
            "monotone_in_k": self.monotone_in_k(),
            "control": {
                "rung": None if control is None else control.label,
                "q_div_is_zero_by_construction": self.control_is_degenerate(),
                "reading": (
                    "Q_div = 0 iff every source's compensation is identical, which is "
                    "the degenerate case of 7.5: the information condition falls back "
                    "to proposition 1(b)"
                ),
            },
        }


def uniform_split(total: int, k: int) -> int:
    """Demonstrations per source at a rung of K, at 10.4.5's fixed total.

    Refuses a total that does not divide: an uneven split would make the mixture
    weights unequal across sources, and while Q_div's mean stays uniform
    regardless, the *pool* would then carry a diversity the rung label does not
    describe.
    """
    if k <= 0:
        raise ValueError(f"K must be positive, got {k}")
    if total % k:
        raise UnmatchedDataVolumeError(
            f"{total} demonstrations do not split evenly over K = {k} sources. 10.4.5 "
            "fixes the total and divides it by K; an uneven split puts a data-volume "
            "difference between the sources of one rung"
        )
    return total // k
