"""Iterative learning control: the demonstrator that produces the compensation.

9.4.2 in full. A closed-loop tracker alone only ever produces *reactive*
correction -- the error has to appear before the controller answers it -- while
the compensation a human demonstrator carries is mostly feedforward. ILC turns
the first into the second by replaying the same reference over and over on the
same initial state and folding the tracking error back into an explicit
feedforward term:

    u^(k+1)(t) = Q[ u^(k)(t) + Gamma e^(k)(t + l_adv) ],
    e^(k)(t)   = x_ref(t) - x_ach^(k)(t)

`u` is in action units and `e` is in metres and radians, and the whole content
of `Gamma` is the conversion between them (9.4.2). Getting that backwards is the
single most expensive mistake available here, so the two are carried in
different variables with the units in their names and the conversion is written
out once, in `GAMMA`.

**Three-way outcome.** 9.4.2 refuses to collapse the outcome into one exclusion
rate, because the reasons differ and one of them is dose-correlated. A run ends
converged, in a limit cycle, or not converged, and the caller gets all three as
separate populations. The stopping rule alone is not enough to tell them apart:
the terminal insertion segment's feedforward-to-error map is discontinuous in
the feedforward -- a small change picks a different first-contact face, or a
different stick/slip branch -- so a sequence that oscillates between two contact
modes at similar RMS satisfies a plain "stopped improving" test while carrying a
mode-selection artefact rather than a compensation. Hence the monotonicity and
mode-stability conditions, both necessary.

**Dose axis.** 9.4.3 makes the iteration count the third dose axis, at
k in {0, 2, 5, 10, k_conv}. Every iterate's feedforward is therefore kept, not
just the last one, and `ILCResult.dose_feedforward` is the only place the ladder
is spelled out.

Nothing in this module reads the r4 regime label. r4 is defined by the wrist
force clearing a per-cell threshold that 9.7.1 calibrates *from* rollouts, and
the teacher's own iteration cannot depend on a measurement downstream of itself;
the contact phase here is the geometric mating test `TrackingResult.task_contact`
instead. The difference is documented at `PhaseSplit`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt

from portal_trace.defects.delta import Delta
from portal_trace.registry.tasks import CONTROL_FREQ, HORIZON, TaskSpec
from portal_trace.registry.tasks import task as task_spec
from portal_trace.teacher.osc import (
    ACTION_HIGH,
    OSCTracker,
    TrackingResult,
    reset_env,
)
from portal_trace.teacher.refs import ReferenceTrajectory
from portal_trace.types import Phase

__all__ = [
    "GAMMA_POS_UNITS_PER_M",
    "GAMMA_ORI_UNITS_PER_RAD",
    "GAMMA",
    "LEAD_STEPS",
    "CAUSAL_LEAD_STEPS",
    "Q_CUTOFF_HZ",
    "K_MAX",
    "C1_RELATIVE_DROP",
    "C1_TOL_FRACTION",
    "C2_WINDOW",
    "C3_WINDOW",
    "DOSE_LADDER",
    "CONTACT_FREE_RATIO_LIMIT",
    "NORMAL_ONLY_GAMMA_SCALE",
    "ContactVariant",
    "Outcome",
    "ILCConfig",
    "PhaseSplit",
    "IterationSnapshot",
    "ILCResult",
    "q_filter",
    "run_ilc",
    "feedforward_delta",
    "analytic_cosine",
]

# ---------------------------------------------------------------------------
# 9.4.2, the table of quantities. Every number here is stated numerically in the
# proposal, so each is a constant with its section beside it and none of them is
# a lock-file measurement. The one measured quantity the stopping rule needs is
# tol(tau), which arrives as an argument.
# ---------------------------------------------------------------------------

# Gamma is a unit conversion and nothing else. 9.6.1 scales one action unit to
# 50 mm of translation and 0.5 rad of rotation, so a dimensionless loop gain of
# g translates to g / 0.05 action units per metre and g / 0.5 per radian. The
# proposal picks 0.5 for translation and 0.3 for rotation -- the orientation
# channel's effective plant gain is higher, so it gets the smaller one -- giving
# per-iteration error factors of 0.5 and 0.7.
#
# Writing this out is not decoration. Treating u as if it were in metres and
# setting Gamma to a dimensionless 0.5 corrects 0.0005 action units per
# millimetre of error, an error factor of 0.975, and twenty iterations then
# remove 40% of the error and never reach a fixed point at all.
GAMMA_POS_UNITS_PER_M = 10.0
GAMMA_ORI_UNITS_PER_RAD = 0.6
GAMMA = np.array(
    [GAMMA_POS_UNITS_PER_M] * 3 + [GAMMA_ORI_UNITS_PER_RAD] * 3, dtype=np.float64
)

# The dimensionless loop gains the line above encodes, kept so that the
# arithmetic can be asserted rather than trusted.
DIMENSIONLESS_GAIN_POS = 0.5
DIMENSIONLESS_GAIN_ORI = 0.3

# 9.4.2: one control step of lead, 50 ms at 20 Hz. The causal variant of 9.4.3
# calliper (ii) uses zero and converges more slowly.
LEAD_STEPS = 1
CAUSAL_LEAD_STEPS = 0

# 9.4.2: without Q the high-frequency content grows exponentially with iteration
# and ILC diverges around the fifth to eighth. 5 Hz is roughly a quarter of the
# 20 Hz control rate, which is the usable bandwidth.
Q_CUTOFF_HZ = 5.0

# 9.4.2: eleven iterations suffice for the free-space segment's rotation
# channel; the budget doubles that to leave room for slow contact convergence.
K_MAX = 20

# 9.4.2, the (C1)/(C2)/(C3) table.
C1_RELATIVE_DROP = 0.02
C1_TOL_FRACTION = 0.2
C2_WINDOW = 5
C3_WINDOW = 3

# 9.4.3: the five levels of the ILC dose axis. k_conv is per-cell and is
# appended by `ILCResult.dose_ladder`. k = 0 is not optional -- it is the only
# anchor at which the defect is present and the compensation is zero.
DOSE_LADDER: tuple[int, ...] = (0, 2, 5, 10)

# 9.4.2's pre-registered phase-split verdict: if the contact segment's e_RMS
# median exceeds the free segment's by this factor, c_delta is only claimed on
# the free segment.
CONTACT_FREE_RATIO_LIMIT = 3.0

# 9.4.2's `normal_only` degradation: quarter gain on the terminal phase.
NORMAL_ONLY_GAMMA_SCALE = 0.25


class ContactVariant(str, Enum):
    """`ilc_contact_variant` (9.4.2), a registry field beside `defect_impl_variant`.

    `FULL` learns the terminal phase like any other. The other two are the
    pre-registered degradations, to be selected only when the phase-split verdict
    fires, and each costs something specific: `NORMAL_ONLY` never learns the
    tangential compensation, so c_delta is credible on the insertion normal
    alone; `FREEZE_AT_CONTACT` extrapolates the free segment's last value through
    the whole contact segment, so the contact-phase c_delta is not that segment's
    own compensation.
    """

    FULL = "full"
    NORMAL_ONLY = "normal_only"
    FREEZE_AT_CONTACT = "freeze_at_contact"


class Outcome(str, Enum):
    """The three ways an ILC run ends (9.4.2).

    The proposal's middle case is 极限环, a limit cycle: the run stopped
    improving but is still switching between contact modes or is no longer
    monotone. It is not a degradation of the trajectory and it is not the same
    thing as the `ilc_contact_variant` degradations, which are a deliberate
    choice made before the run rather than a verdict on it.
    """

    CONVERGED = "converged"
    LIMIT_CYCLE = "limit_cycle"
    NOT_CONVERGED = "not_converged"


@dataclass(frozen=True)
class ILCConfig:
    """Everything about the update that a caller is allowed to vary.

    The gains are not in here. Gamma is a unit conversion fixed by the action
    scale (9.6.1) and the loop gains of 9.4.2, and the tracker's own gains are
    frozen by 9.4.6; both would be tuning surfaces if they were arguments.
    """

    lead_steps: int = LEAD_STEPS
    q_cutoff_hz: float = Q_CUTOFF_HZ
    k_max: int = K_MAX
    contact_variant: ContactVariant = ContactVariant.FULL

    @classmethod
    def for_delta(cls, delta: Delta, **overrides: Any) -> ILCConfig:
        """The config for one defect, with 9.4.2's lead phase-aligned to D2.

        Everywhere except D2 the lead is 9.4.2's one control step. D2 is a pure
        action-side shift of n steps, so the error at t answers the feedforward
        at t - n, and folding e(t + 1) back into u(t) leaves n - 1 steps of
        uncorrected phase in the loop. Since n is the injected parameter rather
        than something estimated from the run, aligning the lead to it is
        compensating a known delay, not tuning: 9.4.6's freeze applies to the
        tracker's gains, which fix what the run converges *to*, while the lead
        only decides whether it gets there.

        Measured on Stack/Panda, e_RMS at the last iterate, lead 1 against lead
        n: n = 2 gives 0.00749 against 0.00484, n = 3 gives 0.0325 against
        0.0168. The alignment is a strict improvement at every dose and at every
        iterate, and it is why the choice is made here rather than left at 1.

        It does not rescue D2's upper ladder, and the code must not be read as
        claiming it does. Allowed to run past (C1), n = 2 turns up again at
        k = 8, n = 3 at k = 4, and n = 4 diverges outright to 0.133. Only n = 1
        reaches a fixed point, so 9.4.7's check is defined on that dose alone and
        D2's dose-correlated exclusion (9.4.2) is a property of the update law
        rather than of the lead.
        """
        if delta.defect_code == "D2":
            n = delta.params["n_control_steps"]
            if n != int(n):
                raise ValueError(
                    f"D2's dose is an integer number of control steps (9.2.2 item 1), "
                    f"got {n}; a fractional shift needs first-order hold, which is a "
                    "low-pass filter coupled to l and destroys 'pure delay is allpass'"
                )
            overrides.setdefault("lead_steps", max(int(n), 1))
        return cls(**overrides)

    def __post_init__(self) -> None:
        if self.lead_steps < 0:
            raise ValueError(
                f"lead_steps is a lead, not a lag: got {self.lead_steps}; 9.4.3's "
                "causal variant is 0 and the main calliper is 1"
            )
        if not 0.0 < self.q_cutoff_hz < CONTROL_FREQ / 2.0:
            raise ValueError(
                f"q_cutoff_hz must lie in (0, {CONTROL_FREQ / 2.0}), the Nyquist "
                f"band of {CONTROL_FREQ} Hz control; got {self.q_cutoff_hz}"
            )
        if self.k_max < C2_WINDOW:
            raise ValueError(
                f"k_max={self.k_max} is below the {C2_WINDOW}-iteration window "
                "(C2) needs, so no run could ever be declared converged"
            )


# ---------------------------------------------------------------------------
# The update
# ---------------------------------------------------------------------------


def q_filter(u: np.ndarray, *, cutoff_hz: float = Q_CUTOFF_HZ) -> np.ndarray:
    """The zero-phase low-pass Q of 9.4.2, applied down the time axis.

    Forward-backward first-order Butterworth, so the filter contributes no phase
    of its own: a phase shift here would compete with `lead_steps`, which is the
    one place the update is allowed to be non-causal.
    """
    u = np.asarray(u, dtype=np.float64)
    b, a = butter(1, cutoff_hz / (CONTROL_FREQ / 2.0), btype="low")
    return np.asarray(filtfilt(b, a, u, axis=0), dtype=np.float64)


def _lead(error: np.ndarray, steps: int) -> np.ndarray:
    """e(t + steps), holding the final value past the end of the horizon.

    Holding rather than wrapping: the reference finishes stationary, so the last
    error is the right thing to keep applying, whereas a wrap would fold the
    approach segment's error into the terminal segment's feedforward.
    """
    if steps == 0:
        return error
    out = np.empty_like(error)
    out[:-steps] = error[steps:]
    out[-steps:] = error[-1]
    return out


def _terminal_mask(reference: ReferenceTrajectory) -> np.ndarray:
    return reference.phase_mask(Phase.TERMINAL)


def _normal_projector(spec: TaskSpec) -> np.ndarray:
    """(6, 6) projector keeping the insertion normal and dropping the rest.

    9.4.2's `normal_only`: on the terminal phase only the component along the
    insertion direction is learned, and the rotation block is frozen outright.
    """
    if spec.insertion_axis is None:
        raise ValueError(
            f"{spec.task_id} declares no insertion_axis, so there is no normal to "
            f"project onto; {ContactVariant.NORMAL_ONLY.value} is not defined for it"
        )
    n = np.asarray(spec.insertion_axis, dtype=np.float64)
    n = n / np.linalg.norm(n)
    p = np.zeros((6, 6), dtype=np.float64)
    p[:3, :3] = np.outer(n, n)
    return p


def _apply_variant(
    increment: np.ndarray,
    *,
    variant: ContactVariant,
    reference: ReferenceTrajectory,
    spec: TaskSpec,
) -> np.ndarray:
    if variant is ContactVariant.FULL:
        return increment
    if variant is ContactVariant.NORMAL_ONLY:
        out = increment.copy()
        term = _terminal_mask(reference)
        out[term] = (increment[term] @ _normal_projector(spec).T) * NORMAL_ONLY_GAMMA_SCALE
        return out
    # FREEZE_AT_CONTACT is not an increment rule but a hold, applied to the
    # accumulated feedforward in `_freeze_after_contact`.
    return increment


def _freeze_after_contact(u: np.ndarray, contact: np.ndarray) -> np.ndarray:
    """Hold the feedforward at its last pre-contact value (9.4.2).

    The whole contact segment then carries an extrapolation of the free
    segment's final value rather than its own compensation, which is exactly the
    cost the proposal attaches to this variant.
    """
    if not contact.any():
        return u
    first = int(np.argmax(contact))
    if first == 0:
        raise ValueError(
            "the mating contact is already live at t = 0, so there is no "
            "pre-contact value to freeze at"
        )
    out = u.copy()
    out[first:] = u[first - 1]
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseSplit:
    """e_RMS on the free and contact segments, reported separately (9.4.2).

    9.4.2 forbids reporting the combined value, because kappa is evaluated at
    the terminal contact phase (14.5 E-6) and the two segments' credibility can
    differ by an order of magnitude: a contact residual much larger than the free
    one means Lambda's denominator is overstated exactly where it matters.

    The segments here are the geometric mating contact, not the r4 regime. r4
    additionally counts wrist load carried through the gripper without the part
    having reached its receptacle, and its threshold is a 9.7.1 calibration; when
    that calibration exists the same split can be recomputed on it and the two
    compared, which this dataclass deliberately does not pre-empt.
    """

    free_rms: float
    contact_rms: float
    free_frames: int
    contact_frames: int

    @property
    def ratio(self) -> float:
        """Contact over free. `inf` when the free segment is clean and contact is not."""
        if self.free_rms == 0.0:
            return math.inf if self.contact_rms > 0.0 else 1.0
        return self.contact_rms / self.free_rms

    @property
    def contact_credible(self) -> bool:
        """9.4.2's pre-registered verdict on whether c_delta holds in contact."""
        return self.contact_frames > 0 and self.ratio <= CONTACT_FREE_RATIO_LIMIT


@dataclass(frozen=True)
class IterationSnapshot:
    """One ILC iteration, kept in full because 9.4.3 doses on the iteration count."""

    k: int
    e_rms: float
    position_rms: float
    orientation_rms: float
    split: PhaseSplit
    mode_hash: int
    success: bool
    clipped_fraction: float
    peak_cfrc_ext: float
    feedforward: np.ndarray       # (T, 6) action units, the u that produced this

    @property
    def saturated(self) -> bool:
        return self.clipped_fraction > 0.0


def _rms(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(error * error, axis=1))))


def _split(result: TrackingResult) -> PhaseSplit:
    contact = result.task_contact
    free = ~contact
    return PhaseSplit(
        free_rms=_rms(result.error[free]) if free.any() else 0.0,
        contact_rms=_rms(result.error[contact]) if contact.any() else 0.0,
        free_frames=int(free.sum()),
        contact_frames=int(contact.sum()),
    )


def _snapshot(k: int, result: TrackingResult, u: np.ndarray) -> IterationSnapshot:
    return IterationSnapshot(
        k=k,
        e_rms=result.rms(),
        position_rms=result.position_rms(),
        orientation_rms=result.orientation_rms(),
        split=_split(result),
        mode_hash=result.mode_hash,
        success=result.success,
        clipped_fraction=float(result.clipped.mean()),
        peak_cfrc_ext=result.peak_cfrc_ext,
        feedforward=u.copy(),
    )


# ---------------------------------------------------------------------------
# The three conditions
# ---------------------------------------------------------------------------


def _c1_stop(history: Sequence[float], tol: float) -> bool:
    """(C1), sufficient: the run has stopped improving, or is already good enough."""
    if history[-1] <= C1_TOL_FRACTION * tol:
        return True
    if len(history) < 2:
        return False
    previous = history[-2]
    if previous == 0.0:
        return True
    return (previous - history[-1]) / previous < C1_RELATIVE_DROP


def _c2_monotone(history: Sequence[float]) -> bool:
    """(C2), necessary: the last five e_RMS are non-increasing."""
    if len(history) < C2_WINDOW:
        return False
    window = np.asarray(history[-C2_WINDOW:], dtype=np.float64)
    return bool(np.all(np.diff(window) <= 0.0))


def _c3_mode_stable(hashes: Sequence[int]) -> bool:
    """(C3), necessary: the last three mode sequence hashes agree.

    Reusing 9.6.4's hash rather than inventing a second contact descriptor is
    deliberate; the quantity is already stored per episode.
    """
    if len(hashes) < C3_WINDOW:
        return False
    return len(set(hashes[-C3_WINDOW:])) == 1


@dataclass
class ILCResult:
    """The outcome of one cell's ILC run, with every iterate kept."""

    task_id: str
    family: str
    seed_env: int
    outcome: Outcome
    k_conv: int | None
    tol: float
    config: ILCConfig
    snapshots: list[IterationSnapshot] = field(default_factory=list)
    # The rollout of the last iterate, in full. `IterationSnapshot` keeps only
    # summary statistics and the feedforward, which is the right trade at
    # k_max + 1 iterations, but 9.4.7 needs the frames the run actually visited:
    # c_delta is defined pointwise in the state (3.1), so the analytic field has
    # to be read on the states the defective side reached, not on the reference
    # it was aiming at. Keeping only the last one costs one rollout of memory
    # instead of thirty.
    final_tracking: TrackingResult | None = None

    def __post_init__(self) -> None:
        # None only for the degenerate result that ran nothing. Every other
        # combination is unreachable from `run_ilc` and would hand a caller an
        # `AttributeError` on `.states` several frames away from the cause.
        if bool(self.snapshots) != (self.final_tracking is not None):
            raise ValueError(
                f"{self.task_id}/{self.family}: {len(self.snapshots)} snapshots but "
                f"final_tracking is {'absent' if self.final_tracking is None else 'set'}. "
                "The last iterate's rollout is recorded with its snapshot or not at all"
            )

    @property
    def iterations(self) -> int:
        return len(self.snapshots) - 1

    @property
    def converged(self) -> bool:
        return self.outcome is Outcome.CONVERGED

    @property
    def trace(self) -> np.ndarray:
        """(k+1,) the e_RMS sequence, the thing 9.4.2 judges the run on."""
        return np.array([s.e_rms for s in self.snapshots], dtype=np.float64)

    @property
    def feedforward(self) -> np.ndarray:
        """u^(infinity): the last iterate's feedforward, in action units."""
        return self.snapshots[-1].feedforward

    @property
    def baseline_feedforward(self) -> np.ndarray:
        """u^(0). Zero by construction, and kept explicit because 9.4.2 and 9.4.7
        both write the compensation as a difference from it rather than as u alone."""
        return self.snapshots[0].feedforward

    @property
    def compensation(self) -> np.ndarray:
        """u^(infinity) - u^(0), the 9.4.2 feedforward compensation of this run.

        On a defective embodiment this still contains the tracker's own lag
        compensation, which is common to the twin pair and is what definition 3.1
        removes by differencing; `feedforward_delta` is that subtraction.
        """
        return self.feedforward - self.baseline_feedforward

    @property
    def dose_ladder(self) -> tuple[int, ...]:
        """9.4.3's five levels {0, 2, 5, 10, k_conv}, the only source in the code."""
        if self.k_conv is None:
            return DOSE_LADDER
        return (*DOSE_LADDER, self.k_conv)

    def dose_feedforward(self, k: int) -> np.ndarray:
        """The feedforward at dose level `k`.

        A genuinely converged run stays at its fixed point.  Consequently a
        fixed rung after ``k_conv`` uses the converged feedforward, as A9's
        preregistered level definition requires (for example, a cell converged
        at k=8 has identical k=10 and k_conv demonstrations).  A limit cycle or
        non-converged run is never clamped: it has no unobserved dose.
        """
        for snap in self.snapshots:
            if snap.k == k:
                return snap.feedforward
        if (
            self.converged
            and self.k_conv is not None
            and k in DOSE_LADDER
            and k > self.k_conv
        ):
            return self.snapshots[-1].feedforward
        raise ValueError(
            f"{self.task_id}/{self.family}/seed={self.seed_env} ran {self.iterations} "
            f"iterations and has no dose level k={k}; available: "
            f"{[s.k for s in self.snapshots]}"
        )

    def dose_source_iteration(self, k: int) -> int:
        """The actual iterate backing a requested fixed dose rung."""
        for snap in self.snapshots:
            if snap.k == k:
                return snap.k
        # Keep resolution and validation in one place.
        self.dose_feedforward(k)
        assert self.k_conv is not None
        return self.k_conv

    @property
    def final_split(self) -> PhaseSplit:
        return self.snapshots[-1].split

    @property
    def final_drop_rate(self) -> float | None:
        """How fast e_RMS was still falling on the last iterate, as a fraction.

        (C1)'s quality limb is sufficient, so a run can be marked converged well
        before the fixed point: on Stack, tol = 25 mm, the limb fires at k = 6
        while each iterate is still taking 29% off. `u_inf` therefore means "the
        feedforward where this cell was judged converged", and this number is
        what lets a reader see where it stopped. None for a run of one iterate,
        which has no previous value to compare against.
        """
        if len(self.snapshots) < 2:
            return None
        previous, last = self.snapshots[-2].e_rms, self.snapshots[-1].e_rms
        if previous == 0.0:
            return 0.0
        return (previous - last) / previous

    @property
    def mode_switches(self) -> tuple[int, ...]:
        """The iterates at which the contact mode sequence changed.

        Reported with every limit cycle because (C1) and (C3) interact: the run
        stops at the first iterate that is good enough, which may be the iterate
        where the mode has just switched, and then (C3)'s three-iterate window
        straddles the switch. A cell with one late switch and a cell that is
        genuinely alternating both fail (C3), and only this and the distinct
        count tell them apart.
        """
        hashes = [s.mode_hash for s in self.snapshots]
        return tuple(k for k in range(1, len(hashes)) if hashes[k] != hashes[k - 1])

    def summary(self) -> dict[str, Any]:
        s = self.snapshots[-1]
        return {
            "task": self.task_id,
            "family": self.family,
            "seed_env": self.seed_env,
            "outcome": self.outcome.value,
            "k_conv": self.k_conv,
            "iterations": self.iterations,
            "tol_m": self.tol,
            "e_rms_initial": self.snapshots[0].e_rms,
            "e_rms_final": s.e_rms,
            "position_rms_final_m": s.position_rms,
            "orientation_rms_final_rad": s.orientation_rms,
            "free_rms": s.split.free_rms,
            "contact_rms": s.split.contact_rms,
            "contact_free_ratio": s.split.ratio,
            "contact_credible": s.split.contact_credible,
            "success": s.success,
            "clipped_fraction": s.clipped_fraction,
            "contact_variant": self.config.contact_variant.value,
            "ell_adv": self.config.lead_steps,
            "final_drop_rate": self.final_drop_rate,
            "distinct_mode_hashes": len({sn.mode_hash for sn in self.snapshots}),
            "mode_switches": list(self.mode_switches),
        }


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def run_ilc(
    env: Any,
    reference: ReferenceTrajectory,
    *,
    family: str,
    seed_env: int,
    tol: float,
    config: ILCConfig | None = None,
    tracker: OSCTracker | None = None,
    noise: np.ndarray | None = None,
    reset: Callable[[], Any] | None = None,
) -> ILCResult:
    """Run 9.4.2's ILC on one (embodiment, task, seed) cell.

    `tol` is tol(tau), a lock-file measurement (`task.tol.{task}.{geom_variant}`)
    that the caller reads and passes in. It is an argument rather than a lookup
    so that this module holds no opinion about which geometry variant is in
    force, and so that no default for it can creep in here.

    `env` may be a bare robosuite environment or a `DefectWrapper`; only `step`
    and the tracker's read-only accessors are used. `reset` says how the
    environment returns to the initial state that `seed_env` selects, and
    defaults to the tracker's own `reset_env`; a wrapped environment whose reset
    carries defect state of its own has to supply it.

    Every iteration starts from the same initial state. That is the whole
    premise of ILC -- the error signal being folded back is only meaningful if
    the trial is repeated, not merely resembled -- and it is also what makes the
    per-iteration snapshots comparable as a dose ladder.
    """
    config = config or ILCConfig()
    tracker = tracker or OSCTracker()
    spec = task_spec(reference.task_id)
    if tol <= 0.0:
        raise ValueError(
            f"tol(tau) for {reference.task_id} must be positive, got {tol}; it is "
            "the denominator of the (C1) quality limb and of kappa"
        )
    if reference.horizon != HORIZON:
        raise ValueError(
            f"reference has {reference.horizon} steps, 9.6.1 fixes the horizon at "
            f"{HORIZON}"
        )
    do_reset = reset if reset is not None else (lambda: reset_env(env, seed_env))

    u = np.zeros((reference.horizon, 6), dtype=np.float64)
    snapshots: list[IterationSnapshot] = []
    rms_history: list[float] = []
    hash_history: list[int] = []
    outcome = Outcome.NOT_CONVERGED
    k_conv: int | None = None

    final: TrackingResult | None = None
    for k in range(config.k_max + 1):
        do_reset()
        result = tracker.track(
            env, reference, family=family, seed_env=seed_env,
            feedforward=u, noise=noise,
        )
        final = result
        snapshots.append(_snapshot(k, result, u))
        rms_history.append(result.rms())
        hash_history.append(result.mode_hash)

        # (C1) is only put to the reader once (C2) and (C3) have enough
        # iterations to be answerable. Testing it earlier would force a verdict
        # on evidence that does not exist yet, and since both are necessary the
        # verdict would always be the pessimistic one.
        if len(rms_history) >= max(C2_WINDOW, C3_WINDOW) and _c1_stop(rms_history, tol):
            monotone = _c2_monotone(rms_history)
            stable = _c3_mode_stable(hash_history)
            outcome = Outcome.CONVERGED if monotone and stable else Outcome.LIMIT_CYCLE
            k_conv = k if outcome is Outcome.CONVERGED else None
            break

        if k == config.k_max:
            break

        increment = _lead(result.error, config.lead_steps) * GAMMA
        increment = _apply_variant(
            increment, variant=config.contact_variant, reference=reference, spec=spec
        )
        u = q_filter(u + increment, cutoff_hz=config.q_cutoff_hz)
        if config.contact_variant is ContactVariant.FREEZE_AT_CONTACT:
            u = _freeze_after_contact(u, result.task_contact)
        # The feedforward shares the action budget with the tracker's own delta,
        # so a feedforward beyond the action range could never be applied in
        # full; clipping it here keeps the recorded u equal to the u that acted.
        u = np.clip(u, -ACTION_HIGH, ACTION_HIGH)

    return ILCResult(
        task_id=reference.task_id,
        family=family,
        seed_env=seed_env,
        outcome=outcome,
        k_conv=k_conv,
        tol=tol,
        config=config,
        snapshots=snapshots,
        final_tracking=final,
    )


# ---------------------------------------------------------------------------
# 9.4.7: the numerical-analytic consistency check
# ---------------------------------------------------------------------------


def feedforward_delta(defective: ILCResult, clean: ILCResult) -> np.ndarray:
    """c_delta on the reference trajectory: the twin pair's feedforward difference.

    Definition 3.1 makes the compensation a *difference between twins*, and the
    subtraction is what removes the part of the converged feedforward that
    compensates the tracker's own lag rather than the defect. Both sides must
    have run the same reference from the same `seed_env`, which is checked here
    because getting it wrong produces a plausible-looking field that is really
    two different initial conditions.
    """
    if defective.task_id != clean.task_id:
        raise ValueError(
            f"twin pair spans two tasks, {defective.task_id!r} and {clean.task_id!r}"
        )
    if defective.seed_env != clean.seed_env:
        raise ValueError(
            f"twin pair has different env seeds, {defective.seed_env} and "
            f"{clean.seed_env}; 9.1.6 derives seed_env without the defect id "
            "precisely so that these agree"
        )
    return defective.compensation - clean.compensation


def analytic_cosine(
    compensation: np.ndarray,
    analytic: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> float:
    """9.4.7's cosine between the ILC fixed point and the analytic compensation.

    9.4.7 asks for > 0.8 and is explicit that this is a unit test for
    implementation correctness and linearisation error, not independent evidence:
    the ILC fixed point along a reference *is* the inverse-model difference on
    that reference, and the analytic form is its first-order expansion, so the
    two are not independent paths to the same answer.

    `analytic` comes from `portal_trace.defects.fields`, whose fields evaluate to
    the full action width; only the six task-space channels are compared, since
    the analytic basis carries no gripper column.
    """
    a = np.asarray(compensation, dtype=np.float64)[:, :6]
    b = np.asarray(analytic, dtype=np.float64)[:, :6]
    if a.shape != b.shape:
        raise ValueError(
            f"compensation is {a.shape} and the analytic field is {b.shape}; 9.4.7 "
            "compares them frame by frame"
        )
    if mask is not None:
        a, b = a[mask], b[mask]
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        raise ValueError(
            "one side of the 9.4.7 cosine is identically zero, so the angle "
            "between them is undefined; a zero ILC compensation means the run "
            "learned nothing and a zero analytic field means the defect has no "
            "first-order form"
        )
    return float(np.sum(a * b) / (na * nb))
