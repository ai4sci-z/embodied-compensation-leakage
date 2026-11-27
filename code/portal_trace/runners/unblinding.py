"""The unblinding-order guard of proposal 11.3, and 10.8's five stop-gates.

The order is fixed and this module refuses to run out of it:

1. the three zero-GPU gates, in this order --
   the ILC feedforward's projection residual share onto phi (does the object of
   study exist), the cross replay (is there compensation in the demonstrations),
   and T8's three-way feasibility check (is T8 measurable at all);
2. E12, the direct fit of the two-way additive model, also zero-GPU, whose three
   mandatory outputs do not depend on any training arm;
3. 10.8.2's two pilot stop-gates, both scheduled in W5 -- G4's excess-leakage
   pilot and G5's early B4 / B6 -- which cost GPU time but decide whether the
   full arms behind them are worth starting;
4. the training arms, after the preregistration file has been tagged, primary
   endpoints before secondary ones.

Any of the three zero-GPU gates coming out negative overturns a whole strand of
the argument, and none of them costs GPU time, so 11.3 forbids scheduling them
after training. That is the entire reason this is code rather than a paragraph
in a protocol document: the ordering is only meaningful if a run that violates
it cannot happen.

Prerequisites are declared per gate rather than read off the enum's position.
The two W5 pilots are ordered *after* the zero-GPU gates and E12 and before the
full arms, but nothing downstream waits on them: 10.8.3 puts G5's verdict in
week 5 precisely so that seven weeks remain to act on it, and a guard that made
the primary endpoints wait for a pilot would invert that. Positional ordering
cannot express "after these, before nothing", so the relation is a map.

The ledger records that a stage has *produced a result*, not what the result
was. A negative gate is still a produced result -- 11.3 makes the gate outcomes
deliverables whichever way they come out -- and the exit clause, not this guard,
decides what happens next. `runners.gates` computes the five verdicts from
measured quantities and records them here; a verdict is not something a human
reads off a table and types in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

__all__ = [
    "Gate",
    "UnblindingOrderViolation",
    "GateResult",
    "UnblindingLedger",
    "ZERO_GPU_GATES",
    "PILOT_GATES",
    "PRE_TRAINING_ORDER",
    "GATE_PREREQUISITES",
    "GATE_PRODUCER",
    "GATE_EVALUATORS",
]


class UnblindingOrderViolation(RuntimeError):
    """A stage was requested before the stages 11.3 puts in front of it."""


class Gate(str, Enum):
    """The named stages of the unblinding order."""

    PHI_PROJECTION_RESIDUAL = "phi_projection_residual_share"
    CROSS_REPLAY = "cross_replay"
    T8_FEASIBILITY = "t8_three_way_feasibility"
    E12 = "E12"
    EXCESS_LEAKAGE_PILOT = "excess_leakage_pilot"
    ORACLE_COLLISION_PILOT = "oracle_collision_pilot"
    TRAINING_PRIMARY = "training_arms_primary"
    TRAINING_SECONDARY = "training_arms_secondary"


# The three zero-GPU gates, in the order 11.3 fixes.
ZERO_GPU_GATES: tuple[Gate, ...] = (
    Gate.PHI_PROJECTION_RESIDUAL,
    Gate.CROSS_REPLAY,
    Gate.T8_FEASIBILITY,
)

# Everything that must have produced a result before any training arm is
# unblinded.
PRE_TRAINING_ORDER: tuple[Gate, ...] = (*ZERO_GPU_GATES, Gate.E12)

# 10.8.2's second and third stop-gates. Unlike the first three they cost GPU
# time -- 360 and 420 GPU-h of pilot arms -- which is why they come after the
# free evidence rather than before it.
PILOT_GATES: tuple[Gate, ...] = (Gate.EXCESS_LEAKAGE_PILOT, Gate.ORACLE_COLLISION_PILOT)

# What must have produced a result before each stage may be unblinded.
GATE_PREREQUISITES: dict[Gate, tuple[Gate, ...]] = {
    Gate.PHI_PROJECTION_RESIDUAL: (),
    Gate.CROSS_REPLAY: (Gate.PHI_PROJECTION_RESIDUAL,),
    Gate.T8_FEASIBILITY: (Gate.PHI_PROJECTION_RESIDUAL, Gate.CROSS_REPLAY),
    Gate.E12: ZERO_GPU_GATES,
    Gate.EXCESS_LEAKAGE_PILOT: PRE_TRAINING_ORDER,
    Gate.ORACLE_COLLISION_PILOT: PRE_TRAINING_ORDER,
    Gate.TRAINING_PRIMARY: PRE_TRAINING_ORDER,
    Gate.TRAINING_SECONDARY: (*PRE_TRAINING_ORDER, Gate.TRAINING_PRIMARY),
}

# Which experiment produces which gate. P1 carries the three zero-GPU gates
# (10.3), E12 carries its own, and the two pilots are named by the arms 10.8.2
# schedules for them.
GATE_PRODUCER: dict[Gate, str] = {
    Gate.PHI_PROJECTION_RESIDUAL: "P1",
    Gate.CROSS_REPLAY: "P1",
    Gate.T8_FEASIBILITY: "P1",
    Gate.E12: "E12",
    Gate.EXCESS_LEAKAGE_PILOT: "the W5 pilot of B1 / B2 / B7 / B8 on S_6",
    Gate.ORACLE_COLLISION_PILOT: "the W5 early run of B4 and B6",
}


# The five stages of 10.8 that are pass/fail criteria on measured quantities,
# and the function that computes each one. The two training slots are absent
# because they are not criteria: they record that an arm has produced.
GATE_EVALUATORS: dict[Gate, str] = {
    Gate.PHI_PROJECTION_RESIDUAL: "portal_trace.runners.gates.evaluate_g1",
    Gate.CROSS_REPLAY: "portal_trace.runners.gates.evaluate_g2",
    Gate.T8_FEASIBILITY: "portal_trace.runners.gates.evaluate_g3",
    Gate.EXCESS_LEAKAGE_PILOT: "portal_trace.runners.gates.evaluate_g4",
    Gate.ORACLE_COLLISION_PILOT: "portal_trace.runners.gates.evaluate_g5",
}

_WHY_IT_COMES_FIRST: dict[Gate, str] = {
    **{
        g: (
            f"a zero-GPU gate produced by {GATE_PRODUCER[g]}. A negative reading here "
            "overturns a whole strand of the argument, and it costs no GPU time, so it "
            "may not be scheduled after training"
        )
        for g in ZERO_GPU_GATES
    },
    Gate.E12: (
        "also zero-GPU, and its three mandatory outputs do not depend on any training "
        "arm, so it is unblinded before them"
    ),
    **{
        g: (
            f"a 10.8.2 stop-gate produced by {GATE_PRODUCER[g]} in W5, before the full "
            "arms it decides on"
        )
        for g in PILOT_GATES
    },
    Gate.TRAINING_PRIMARY: "main endpoints are unblinded before secondary ones",
    Gate.TRAINING_SECONDARY: "nothing is unblinded after the secondary endpoints",
}

if set(_WHY_IT_COMES_FIRST) != set(Gate) or set(GATE_PREREQUISITES) != set(Gate):
    raise AssertionError(
        "every stage needs its prerequisites and its reason declared; a stage missing "
        "from either map would be silently unordered"
    )


@dataclass(frozen=True)
class GateResult:
    """One produced gate result.

    `passed` is recorded but never used to gate a later stage. The order is
    about what has been *looked at*, not about what it said; suppressing a
    downstream stage because a gate came out negative would be the exit clause's
    job, and conflating the two would make a negative gate invisible.
    """

    gate: Gate
    passed: bool
    recorded_at: str
    value: float | None
    artifact: str
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "gate": self.gate.value,
            "passed": self.passed,
            "recorded_at": self.recorded_at,
            "value": self.value,
            "artifact": self.artifact,
            "notes": self.notes,
        }

    @classmethod
    def from_json(cls, blob: dict[str, Any]) -> GateResult:
        return cls(
            gate=Gate(blob["gate"]),
            passed=bool(blob["passed"]),
            recorded_at=str(blob["recorded_at"]),
            value=blob.get("value"),
            artifact=str(blob.get("artifact", "")),
            notes=str(blob.get("notes", "")),
        )


@dataclass
class UnblindingLedger:
    """Which stages have produced results, in what order.

    Stored next to the results rather than in the config tree, because it is a
    record of what happened rather than a declaration of what should.
    """

    path: Path
    results: dict[Gate, GateResult]

    @classmethod
    def load(cls, path: Path | str) -> UnblindingLedger:
        p = Path(path)
        if not p.exists():
            return cls(path=p, results={})
        with p.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
        results = {}
        for blob in doc.get("results", []):
            r = GateResult.from_json(blob)
            results[r.gate] = r
        return cls(path=p, results=results)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ordered = [
            self.results[g].to_json()
            for g in Gate
            if g in self.results
        ]
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump({"results": ordered}, fh, indent=2)
            fh.write("\n")

    def record(
        self,
        gate: Gate,
        *,
        passed: bool,
        value: float | None,
        artifact: str,
        notes: str = "",
    ) -> GateResult:
        """Record a produced result, refusing to record out of order."""
        self.require_ready_for(gate)
        result = GateResult(
            gate=gate,
            passed=passed,
            recorded_at=datetime.now(UTC).isoformat(timespec="seconds"),
            value=value,
            artifact=artifact,
            notes=notes,
        )
        self.results[gate] = result
        self.save()
        return result

    def has(self, gate: Gate) -> bool:
        return gate in self.results

    def missing_before(self, gate: Gate) -> tuple[Gate, ...]:
        required: set[Gate] = set()
        pending = [gate]
        while pending:
            for g in GATE_PREREQUISITES[pending.pop()]:
                if g not in required:
                    required.add(g)
                    pending.append(g)
        return tuple(g for g in Gate if g in required and g not in self.results)

    def require_ready_for(self, gate: Gate) -> None:
        missing = self.missing_before(gate)
        if not missing:
            return
        lines = [
            f"cannot unblind {gate.value!r}: 11.3 fixes the unblinding order and "
            f"these come first and have produced nothing yet: "
            f"{[g.value for g in missing]}",
        ]
        lines += [f"  {g.value}: {_WHY_IT_COMES_FIRST[g]}" for g in missing]
        raise UnblindingOrderViolation("\n".join(lines))

    def require_ready_for_training(
        self, *, arm_id: str, is_primary_endpoint: bool
    ) -> None:
        """Guard a training arm.

        `is_primary_endpoint` decides which of the two training slots is being
        entered; 11.3 unblinds main endpoints before secondary ones, so a
        secondary arm additionally needs the primary slot to have produced.
        """
        gate = Gate.TRAINING_PRIMARY if is_primary_endpoint else Gate.TRAINING_SECONDARY
        try:
            self.require_ready_for(gate)
        except UnblindingOrderViolation as exc:
            raise UnblindingOrderViolation(
                f"arm {arm_id}: {exc}"
            ) from None

    def summary(self) -> str:
        rows = []
        for g in Gate:
            r = self.results.get(g)
            if r is None:
                rows.append(f"  {g.value:<32} not produced")
            else:
                verdict = "pass" if r.passed else "FAIL"
                value = "" if r.value is None else f" value={r.value:.6g}"
                rows.append(f"  {g.value:<32} {verdict}{value}  {r.recorded_at}")
        return "unblinding ledger (11.3):\n" + "\n".join(rows)
