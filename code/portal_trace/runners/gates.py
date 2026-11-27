"""The five stop-gates of proposal 10.8, computed from measured quantities.

10.8.1 tabulates three zero-GPU gates and 10.8.2 two more that ride on pilot
arms. Each has a `通过标准` column, and this module is that column in code:

* **G1**, 10.8.1: the ILC feedforward's projection residual share onto phi. Its
  pass standard is quoted as "three tiers: per cell <= 0.25, median <= 0.15, any
  cell > 0.5 judges that cell to have no c_delta; aggregate <= 0.35".
* **G2**, 10.8.1: cross replay. "Success rate drops by >= 15 pp **and** peak
  `cfrc_ext` rises by >= 5 N."
* **G3**, 10.8.1: T8's per-level three-way feasibility. "At least one (defect,
  level, task, geometry variant) quadruple simultaneously satisfies the P0 bite
  x the ceiling constraint x the intermediate amplitude band, with angle(v, u)
  or angle(v, n) < 30 degrees."
* **G4**, 10.8.2: the excess-leakage pilot. "Lambda^exc point estimate >= 0.10
  and above all 8 surrogate readings."
* **G5**, 10.8.2: the Oracle-collision pilot. "Neither B4 nor B6 presses
  Lambda^part down to B7 Oracle's level (difference > 0.05)."

A gate is a pass/fail condition on a measured quantity and is not a hypothesis
test: there is no null, no interval and no multiplicity correction here, and
none of the five appears in the primary hypothesis family of 10.7.6. What each
`evaluate_*` returns is a `GateVerdict`, which goes into the 11.3 ledger through
`record`, so the ordering guard and the verdict cannot come apart.

Three of the five standards are stated for a single reading while the gate is
run over a set of cells. Where the proposal quantifies the set itself it is
followed exactly -- G3 says "at least one quadruple" -- and where it does not,
the quantifier is chosen from what the gate decides and stated in the verdict's
notes:

* G1 aggregates by pooling. The aggregate is E_phi over the union of the cells,
  which is the same quantity as the per-cell reading rather than an average
  resembling it; that is why `ProjectionReport` carries the two energies and not
  only their ratio. A share of sums is not a mean of shares under any weight the
  cells supply, and choosing a weight would be choosing an answer.
* G2 and G4 ask existence questions -- is there compensation in the
  demonstrations, is there excess leakage -- and the exit each one triggers is
  the abandonment of a whole strand. So one cell clearing the effect sizes
  carries the gate, exactly as G3's own wording does for feasibility, and the
  share of cells that cleared it travels in the notes where a marginal one-of-
  many cannot hide.
* G5 is the opposite shape. It asks whether a collision happened, 10.8.2 calls
  B4 "the most dangerous collision", and a collision on one cell is a collision.
  Both arms must hold the margin on every pilot cell.

The three-tier rule of G1 is reported in full but only the aggregate decides the
gate. A cell reading above 0.5 is a statement about that cell -- 10.8.1's exit
redefines c on it as an (x, t)-indexed quantity -- while the gate decides
whether the strand survives, and letting one cell of thirty-six veto the strand
would be a stricter rule than the one preregistered.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from portal_trace.diagnostics.excess import (
    PREREGISTERED_SURROGATE_COUNT,
    ExcessLeakageReading,
)
from portal_trace.diagnostics.units import Amplitude
from portal_trace.eval.calibration_tables import CalibrationTable
from portal_trace.eval.endpoints import LAMBDA_EXC_PREREGISTERED_EFFECT
from portal_trace.policies.pools import SUBSET_DESIGN, SubsetName
from portal_trace.runners.unblinding import Gate, GateResult, UnblindingLedger
from portal_trace.teacher.project import (
    E_PHI_GATE,
    E_PHI_NONEXISTENT,
    E_PHI_PASS,
    E_PHI_STRONG,
    ProjectionReport,
)
from portal_trace.teacher.replay import CFRC_RISE_N, SUCCESS_DROP_PP, CrossReplayReport

__all__ = [
    "ORACLE_COLLISION_MARGIN",
    "S6_CELLS",
    "GateInputMissing",
    "GateVerdict",
    "PhiCellReading",
    "OracleCollisionReading",
    "evaluate_g1",
    "evaluate_g2",
    "evaluate_g3",
    "evaluate_g4",
    "evaluate_g5",
]

# 10.8.2: B4 and B6 pass by staying more than this far above B7 Oracle.
ORACLE_COLLISION_MARGIN = 0.05

# 10.8.2 runs the G4 pilot "on the 6 configurations of S_6"; 10.4.0 fixes that
# subset at two families x three defects.
S6_CELLS = SUBSET_DESIGN[SubsetName.S6][0] * SUBSET_DESIGN[SubsetName.S6][1]


class GateInputMissing(LookupError):
    """A gate was asked for a verdict without the measurements it decides on.

    Never a soft failure: an unevaluated gate and a failed gate lead to
    different chapters of 10.9, so returning "fail" here would forge an exit.
    """

    def __init__(self, gate: Gate, what: str, produced_by: str) -> None:
        self.gate = gate
        self.what = what
        self.produced_by = produced_by
        super().__init__(
            f"gate {gate.value!r} cannot be evaluated: {what}. It is produced by "
            f"{produced_by}. A gate with no inputs has no verdict, and 'fail' is "
            "not the same statement"
        )


@dataclass(frozen=True)
class GateVerdict:
    """One gate's pass/fail, the number it turned on, and the rows behind it."""

    gate: Gate
    passed: bool
    value: float | None
    criterion: str
    notes: tuple[str, ...]
    rows: tuple[Mapping[str, Any], ...]

    def render(self) -> str:
        head = f"{self.gate.value}: {'PASS' if self.passed else 'FAIL'}"
        if self.value is not None:
            head += f"  value={self.value:.6g}"
        return "\n".join([f"{head}", f"  criterion: {self.criterion}",
                          *(f"  {n}" for n in self.notes)])

    def record(self, ledger: UnblindingLedger, *, artifact: str) -> GateResult:
        """Write the verdict into the 11.3 ledger, in order or not at all."""
        return ledger.record(
            self.gate,
            passed=self.passed,
            value=self.value,
            artifact=artifact,
            notes=" | ".join((self.criterion, *self.notes)),
        )


# ---------------------------------------------------------------------------
# G1: the projection residual share (10.8.1, 9.4.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhiCellReading:
    """One cell's E_phi, kept as the two energies its ratio comes from."""

    cell: str
    residual_energy: float
    demeaned_energy: float
    n_frames: int

    def __post_init__(self) -> None:
        if self.demeaned_energy <= 0.0:
            raise ValueError(
                f"{self.cell}: the de-meaned feedforward energy is "
                f"{self.demeaned_energy}, so E_phi is undefined on this cell"
            )
        if self.residual_energy < 0.0:
            raise ValueError(f"{self.cell}: a squared residual cannot be negative")

    @classmethod
    def from_report(cls, report: ProjectionReport) -> PhiCellReading:
        return cls(
            cell=report.cell,
            residual_energy=report.residual_energy,
            demeaned_energy=report.demeaned_energy,
            n_frames=report.n_frames,
        )

    @property
    def e_phi(self) -> float:
        return self.residual_energy / self.demeaned_energy

    @property
    def tier(self) -> str:
        """9.4.4's three tiers, per cell."""
        if self.e_phi > E_PHI_NONEXISTENT:
            return "nonexistent"
        if self.e_phi <= E_PHI_STRONG:
            return "strong_pass"
        if self.e_phi <= E_PHI_PASS:
            return "pass"
        return "fail"


def evaluate_g1(readings: Sequence[PhiCellReading]) -> GateVerdict:
    """G1: does the object of study exist (10.8.1, 9.4.4)."""
    if not readings:
        raise GateInputMissing(
            Gate.PHI_PROJECTION_RESIDUAL,
            "no per-cell projection reports were supplied",
            "teacher.project.project_report on the ILC frames of each cell",
        )
    duplicates = sorted({r.cell for r in readings if
                         sum(1 for other in readings if other.cell == r.cell) > 1})
    if duplicates:
        raise ValueError(
            f"cells {duplicates} appear more than once; the aggregate pools frames "
            "and would count them twice"
        )
    shares = np.array([r.e_phi for r in readings], dtype=np.float64)
    pooled = float(
        sum(r.residual_energy for r in readings)
        / sum(r.demeaned_energy for r in readings)
    )
    median = float(np.median(shares))
    nonexistent = tuple(sorted(r.cell for r in readings if r.tier == "nonexistent"))
    above_cell_threshold = tuple(
        sorted(r.cell for r in readings if r.e_phi > E_PHI_PASS)
    )
    rows = [
        {
            "cell": r.cell,
            "e_phi": r.e_phi,
            "n_frames": r.n_frames,
            "residual_energy": r.residual_energy,
            "demeaned_energy": r.demeaned_energy,
            "tier": r.tier,
        }
        for r in readings
    ]
    notes = [
        f"{len(readings)} cells; pooled E_phi = {pooled:.4f}, median = {median:.4f}, "
        f"max = {shares.max():.4f}",
        f"median tier: {'strong pass' if median <= E_PHI_STRONG else 'not strong'} "
        f"(<= {E_PHI_STRONG})",
        f"cells above the per-cell {E_PHI_PASS} threshold: {list(above_cell_threshold)}",
        f"cells judged to carry no c_delta (> {E_PHI_NONEXISTENT}): {list(nonexistent)}",
        "the aggregate is E_phi over the pooled frames, not a mean of per-cell shares",
    ]
    if nonexistent:
        notes.append(
            "on those cells 10.8.1's exit applies to the cell and not to the gate: c is "
            "redefined as an (x, t)-indexed quantity there and the additive model is "
            "admitted only on the phi projection layer (F6)"
        )
    passed = pooled <= E_PHI_GATE
    if not passed:
        notes.append(
            "10.8.1's exit: re-measure with a causal ILC (ell_adv = 0); if it still "
            "fails, take F6"
        )
    return GateVerdict(
        gate=Gate.PHI_PROJECTION_RESIDUAL,
        passed=passed,
        value=pooled,
        criterion=(
            f"10.8.1 G1: aggregate E_phi <= {E_PHI_GATE}; tiers per cell <= {E_PHI_PASS}, "
            f"median <= {E_PHI_STRONG}, any cell > {E_PHI_NONEXISTENT} has no c_delta"
        ),
        notes=tuple(notes),
        rows=tuple(rows),
    )


# ---------------------------------------------------------------------------
# G2: cross replay (10.8.1, 10.3.1)
# ---------------------------------------------------------------------------


def evaluate_g2(reports: Sequence[CrossReplayReport]) -> GateVerdict:
    """G2: is there any compensation in the demonstrations (10.8.1, 10.3.1)."""
    if not reports:
        raise GateInputMissing(
            Gate.CROSS_REPLAY,
            "no cross-replay reports were supplied",
            "teacher.replay.cross_replay on each twin pair of the P1 subset",
        )
    rows = [
        {
            "family": r.family,
            "task": r.task_id,
            "defect": r.defect,
            "n_demos": r.n_demos,
            "success_drop_pp": r.success_drop_pp,
            "cfrc_rise_n": r.cfrc_rise_n,
            "reverse_success_drop_pp": r.reverse_success_drop_pp,
            "asymmetry_holds": r.asymmetry_holds,
            "passes": r.passes_g2,
        }
        for r in reports
    ]
    carrying = [row for row in rows if row["passes"]]
    drops = np.array([row["success_drop_pp"] for row in rows], dtype=np.float64)
    rises = np.array([row["cfrc_rise_n"] for row in rows], dtype=np.float64)
    notes = [
        f"{len(carrying)} of {len(rows)} replayed cells clear both effect sizes",
        f"median success drop {float(np.median(drops)):+.1f} pp, median peak cfrc_ext "
        f"rise {float(np.median(rises)):+.2f} N",
        f"harm asymmetry holds on {sum(1 for r in rows if r['asymmetry_holds'])} of "
        f"{len(rows)} cells (11.3 reports it alongside, it is not part of the standard)",
        "existence question: one cell clearing both effect sizes carries the gate, and "
        "the share above says how thin that evidence is",
    ]
    if not carrying:
        notes.append(
            "10.8.1's exit: no training run pays for itself, take F1 -- the "
            "demonstrations carry no compensation to inherit"
        )
    return GateVerdict(
        gate=Gate.CROSS_REPLAY,
        passed=bool(carrying),
        value=float(drops.max()) if len(drops) else None,
        criterion=(
            f"10.8.1 G2: success rate drops >= {SUCCESS_DROP_PP:g} pp and peak cfrc_ext "
            f"rises >= {CFRC_RISE_N:g} N on at least one cell"
        ),
        notes=tuple(notes),
        rows=tuple(rows),
    )


# ---------------------------------------------------------------------------
# G3: T8's three-way feasible set (10.8.1, 10.2.3)
# ---------------------------------------------------------------------------


def evaluate_g3(table: CalibrationTable) -> GateVerdict:
    """G3: is proposition 2's bet measurable at all (10.8.1).

    Reads P0's seventh table rather than recomputing the three conditions, so
    that the gate and the published table cannot disagree.
    """
    if table.name != "t8_three_way_feasibility":
        raise GateInputMissing(
            Gate.T8_FEASIBILITY,
            f"the table supplied is {table.name!r} and not the three-way feasibility "
            "table",
            "eval.calibration_tables.build_feasibility_table",
        )
    if not table.rows:
        raise GateInputMissing(
            Gate.T8_FEASIBILITY,
            "the feasibility table has no rows",
            "eval.calibration_tables.build_feasibility_table over the clearance sweep",
        )
    admitted = [
        row
        for row in table.rows
        if row["feasible"] and row["angle_admits_sign_prediction"]
    ]
    feasible_only = [row for row in table.rows if row["feasible"]]
    notes = [
        f"{len(admitted)} of {len(table.rows)} quadruples satisfy all three conditions "
        f"with an admitting angle; {len(feasible_only)} satisfy the three conditions "
        "before the angle is applied",
        "quadruples: (defect, level, task, geometry variant) at the T8 carrier family",
    ]
    if admitted:
        best = admitted[0]
        notes.append(
            f"first admitted quadruple: {best['defect']} L{best['level']} on "
            f"{best['task']}/{best['geom_variant']}"
        )
    else:
        notes.append(
            "10.8.1's exit F8: T8 is not measurable, and proposition 2 falls back to "
            "(a)'s type dichotomy plus the controlled sign flip as a qualitative "
            "observation"
        )
    return GateVerdict(
        gate=Gate.T8_FEASIBILITY,
        passed=bool(admitted),
        value=float(len(admitted)),
        criterion=(
            "10.8.1 G3: at least one (defect, level, task, geometry variant) satisfies "
            "the P0 bite, the ceiling constraint and the intermediate amplitude band, "
            "with angle(v, u) or angle(v, n) < 30 degrees"
        ),
        notes=tuple(notes),
        rows=tuple(admitted),
    )


# ---------------------------------------------------------------------------
# G4: the excess-leakage pilot (10.8.2)
# ---------------------------------------------------------------------------


def evaluate_g4(readings: Mapping[str, ExcessLeakageReading]) -> GateVerdict:
    """G4: is there excess leakage on the S_6 pilot (10.8.2).

    The standard names all eight surrogates. A cell that lost surrogates to
    9.2.11's matching tolerances cannot answer that clause literally, so the
    comparison is against every surrogate the cell admitted and the admitted
    count is in the row; a cell below the confirmatory floor of six carries no
    weight in the verdict at all, which is 11.3's rule for those readings rather
    than one invented here.
    """
    if not readings:
        raise GateInputMissing(
            Gate.EXCESS_LEAKAGE_PILOT,
            "no excess-leakage readings were supplied",
            "diagnostics.excess.excess_leakage on the W5 pilot of B1 / B2 / B7 / B8",
        )
    if len(readings) != S6_CELLS:
        raise GateInputMissing(
            Gate.EXCESS_LEAKAGE_PILOT,
            f"{len(readings)} configurations were supplied and 10.8.2 runs the pilot on "
            f"the {S6_CELLS} configurations of S_6",
            "diagnostics.excess.excess_leakage on the W5 pilot of B1 / B2 / B7 / B8",
        )
    rows = []
    for cell in sorted(readings):
        reading = readings[cell]
        point = float(reading.lambda_exc.value)
        null_max = float(reading.null.values.max())
        above_all = float(reading.lambda_part.value) > null_max
        rows.append(
            {
                "cell": cell,
                "lambda_exc": point,
                "lambda_part": float(reading.lambda_part.value),
                "null_max": null_max,
                "n_surrogates_admitted": reading.null.admitted,
                "rank": reading.rank,
                "quantile": reading.quantile,
                "status": reading.status,
                "meets_effect_size": point >= LAMBDA_EXC_PREREGISTERED_EFFECT,
                "above_every_surrogate": above_all,
                "carries_gate": (
                    reading.is_confirmatory
                    and point >= LAMBDA_EXC_PREREGISTERED_EFFECT
                    and above_all
                ),
            }
        )
    carrying = [row for row in rows if row["carries_gate"]]
    exploratory = [row["cell"] for row in rows if row["status"] == "exploratory"]
    points = np.array([row["lambda_exc"] for row in rows], dtype=np.float64)
    notes = [
        f"{len(carrying)} of {len(rows)} S_6 configurations meet the effect size and "
        "sit above every admitted surrogate",
        f"largest Lambda^exc point estimate {float(points.max()):+.4f}, median "
        f"{float(np.median(points)):+.4f}",
        f"the preregistered null has {PREREGISTERED_SURROGATE_COUNT} members; cells that "
        f"admitted fewer are compared against what they admitted, and exploratory cells "
        f"(below the confirmatory floor) do not carry the gate: {exploratory}",
        "existence question, so one confirmatory configuration carries the gate",
    ]
    if not carrying:
        notes.append(
            "10.8.2's exit F1, with G2 already passed: the narrative becomes "
            "'compensation exists but is not inherited', as a mechanism diagram"
        )
    return GateVerdict(
        gate=Gate.EXCESS_LEAKAGE_PILOT,
        passed=bool(carrying),
        value=float(points.max()),
        criterion=(
            f"10.8.2 G4: Lambda^exc point estimate >= "
            f"{LAMBDA_EXC_PREREGISTERED_EFFECT} and above all "
            f"{PREREGISTERED_SURROGATE_COUNT} surrogate readings"
        ),
        notes=tuple(notes),
        rows=tuple(rows),
    )


# ---------------------------------------------------------------------------
# G5: the Oracle-collision pilot (10.8.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleCollisionReading:
    """Lambda^part on one pilot cell for one arm and for B7 Oracle."""

    cell: str
    arm_id: str
    lambda_part: Amplitude
    oracle_lambda_part: Amplitude

    @property
    def margin(self) -> float:
        """How far the arm stays above Oracle. Negative means it went below."""
        return float(self.lambda_part.value) - float(self.oracle_lambda_part.value)

    @property
    def collides(self) -> bool:
        return self.margin <= ORACLE_COLLISION_MARGIN


def evaluate_g5(
    readings: Sequence[OracleCollisionReading],
    *,
    pilot_arms: Sequence[str] = ("B4", "B6"),
) -> GateVerdict:
    """G5: do B4 and B6 leave the positioning intact (10.8.2).

    Universally quantified over the pilot cells, unlike G2 and G4. The gate asks
    whether a collision happened and 10.8.2 calls B4 the most dangerous one, so
    a single cell where an arm reaches Oracle level is the collision the gate is
    watching for.
    """
    if not readings:
        raise GateInputMissing(
            Gate.ORACLE_COLLISION_PILOT,
            "no paired Lambda^part readings were supplied",
            "the W5 early run of B4 and B6 against B7 Oracle on the same cells",
        )
    covered = {r.arm_id for r in readings}
    missing = [arm for arm in pilot_arms if arm not in covered]
    if missing:
        raise GateInputMissing(
            Gate.ORACLE_COLLISION_PILOT,
            f"no readings for arm(s) {missing}; 10.8.2 runs the gate on "
            f"{list(pilot_arms)} together and one arm's verdict is not the gate's",
            "the W5 early run of B4 and B6 against B7 Oracle on the same cells",
        )
    unexpected = sorted(covered - set(pilot_arms))
    if unexpected:
        raise ValueError(
            f"readings carry arm(s) {unexpected}; 10.8.2's gate is about "
            f"{list(pilot_arms)} and another arm's collision is a different question"
        )
    rows = [
        {
            "cell": r.cell,
            "arm": r.arm_id,
            "lambda_part": float(r.lambda_part.value),
            "oracle_lambda_part": float(r.oracle_lambda_part.value),
            "margin": r.margin,
            "collides": r.collides,
        }
        for r in readings
    ]
    collisions = [row for row in rows if row["collides"]]
    margins = np.array([row["margin"] for row in rows], dtype=np.float64)
    notes = [
        f"{len(rows)} arm-cell readings over {len(covered)} arms; smallest margin "
        f"{float(margins.min()):+.4f}, median {float(np.median(margins)):+.4f}",
        f"collisions: {[(row['arm'], row['cell']) for row in collisions]}",
        "universally quantified: a collision on one cell is a collision",
    ]
    if collisions:
        notes.append(
            "10.8.2's exit F4: no algorithm-side positioning survives; the work is "
            "repositioned as paradigm + benchmark + diagnostics + harm asymmetry, with "
            "off-the-shelf decoupling recommended and A14 / E7 saying when it fails"
        )
    return GateVerdict(
        gate=Gate.ORACLE_COLLISION_PILOT,
        passed=not collisions,
        value=float(margins.min()),
        criterion=(
            f"10.8.2 G5: every pilot arm stays more than {ORACLE_COLLISION_MARGIN} above "
            "B7 Oracle's Lambda^part on every pilot cell"
        ),
        notes=tuple(notes),
        rows=tuple(rows),
    )
