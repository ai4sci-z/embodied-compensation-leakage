"""The defect catalogue: layers, writable fields, and the five-level tables.

This module is the single source for two tables the proposal declares
authoritative: the defect-to-written-field matrix of 9.2.0 and the level table
of 9.2.10. Everything downstream -- the injectors, the registry, the FGW
invariance argument, the dose axes -- derives from `DEFECTS` rather than
restating it.

Two consequences of encoding it as data rather than prose:

`jnt_ref` is not a member of `WritableField`, so no injector can express a write
to it. 9.2.0 forbids it because `mjModel.qpos0` is a compile-time constant whose
runtime modification needs `mj_setConst`, and that breaks bit-exact replay. D1
and D10 route their zero-point offsets through the wrapper's sensing twin
instead.

D6 ships on the `qfrc_applied` variant, not the `dof_frictionloss` one. 9.2.6
made that conditional on whether the baseline's FGW node features read friction
or damping; they do -- the baseline (arXiv:2602.18025 appendix E) takes URMA's
joint descriptor verbatim, and URMA's d_j lists damping and a friction
coefficient among its entries. Writing `dof_frictionloss` would therefore make
`fgw_baseline` non-zero on the D6 twin pair and take X1 with it. With the
fallback active, all six core defects leave `mjModel` untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "InjectionLayer",
    "WritableField",
    "HarmClass",
    "Priority",
    "DefectLevel",
    "DefectSpec",
    "DEFECTS",
    "CORE_DEFECTS",
    "DEFECTS_REPORTED_WITH",
    "DESCRIPTOR_VISIBLE_FIELDS",
    "core_defect_codes",
    "defect",
    "DEG",
]

# The proposal states D1/D4 levels in degrees and everything downstream works in
# radians. Converting at the boundary keeps the table readable against 9.2.10.
DEG = math.pi / 180.0


class InjectionLayer(str, Enum):
    """Where a write lands, which fixes its reset and concurrency semantics (9.2.0)."""

    MJMODEL = "mjModel"      # compiled constant table; survives mj_step, dies on hard_reset
    MJDATA = "mjData"        # user input buffer; mj_step does NOT zero it
    WRAPPER = "wrapper"      # Python-side state; reset semantics are ours to define


class WritableField(str, Enum):
    """The closed set of fields an injector may write.

    Closed on purpose. `jnt_ref` is absent (9.2.0), and adding it would be a
    visible change to this enum rather than a line buried in an injector.
    """

    DOF_FRICTIONLOSS = "dof_frictionloss"
    QFRC_APPLIED = "qfrc_applied"
    ACTUATOR_GAINPRM = "actuator_gainprm"
    ACTUATOR_BIASPRM = "actuator_biasprm"
    GAINTYPE_BIASTYPE = "gaintype/biastype"
    JNT_RANGE = "jnt_range"
    EQ_SOLREF_SOLIMP = "eq_solref/eq_solimp"
    CTRL_REWRITE = "ctrl"
    WRAPPER_STATE = "wrapper_state"
    SENSING_TWIN = "sensing_twin"

    @property
    def layer(self) -> InjectionLayer:
        if self in _MJMODEL_FIELDS:
            return InjectionLayer.MJMODEL
        if self in _MJDATA_FIELDS:
            return InjectionLayer.MJDATA
        return InjectionLayer.WRAPPER


_MJMODEL_FIELDS = frozenset({
    WritableField.DOF_FRICTIONLOSS,
    WritableField.ACTUATOR_GAINPRM,
    WritableField.ACTUATOR_BIASPRM,
    WritableField.GAINTYPE_BIASTYPE,
    WritableField.JNT_RANGE,
    WritableField.EQ_SOLREF_SOLIMP,
})
_MJDATA_FIELDS = frozenset({WritableField.QFRC_APPLIED})


class HarmClass(str, Enum):
    """Proposition 2(c) classification (8.4).

    `UNCLASSIFIED` is not a gap in the taxonomy -- D1 and D6 genuinely have no
    prior sign, because their compensation direction moves with the components
    of delta. Membership is decided per level from the measured angle, and only
    levels with angle < 30 degrees enter a sign prediction.
    """

    UNDERTRAVEL = "undertravel"          # cl_+ ~ 0, cl_- >> 0 => A > 0, danger is D->C
    OBSTACLE_DRIFT = "obstacle_drift"    # cl_+z >> 0, cl_-z ~ 0 => A < 0, danger is C->D
    UNCLASSIFIED = "unclassified"


class Priority(str, Enum):
    A = "A"
    B = "B"
    C = "C"


@dataclass(frozen=True)
class DefectLevel:
    """One dose level.

    `params` holds the injection parameter, which the proposal fixes. The
    observable is a different matter: 9.2.10 requires both, and notes that the
    map from injection parameter to physical effect runs through J, Lambda_eff
    and k_p, so one level means six different doses across six families. The
    per-family observable therefore comes out of the lock file
    (`defect.observable.*`), and `nominal_observable` here is only the
    single-family figure printed in 9.2.10, kept for cross-checking.
    """

    index: int
    params: dict[str, float]
    nominal_observable: float | None
    observable_unit: str
    is_origin_anchor: bool = False

    def __post_init__(self) -> None:
        if not self.params:
            raise ValueError("a level must carry at least one injection parameter")


@dataclass(frozen=True)
class DefectSpec:
    """One row of the 9.2.0 matrix joined with its row in the 9.2.10 table."""

    code: str
    name: str
    priority: Priority
    primary_fields: tuple[WritableField, ...]
    fallback_fields: tuple[WritableField, ...]
    levels: tuple[DefectLevel, ...]
    changes_kinematic_graph: bool
    in_36_pairs: bool
    analytic_field: bool
    needs_memory_state: bool
    harm_class: HarmClass
    # True when the observable column of 9.2.10 *is* the injection parameter, so
    # there is nothing per family to measure. 9.2.10's premise -- that one level
    # means six different doses because the parameter reaches the end effector
    # through J, Lambda_eff and k_p -- holds for every defect whose parameter has
    # a physical unit, and fails for D8 alone, whose parameter is a dimensionless
    # ratio of commanded to delivered torque. Declared rather than derived from
    # the unit, because a future dimensionless parameter need not have this
    # property and should have to say so.
    observable_is_parameter: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.observable_is_parameter:
            for lv in self.levels:
                declared = lv.nominal_observable
                if declared is None or len(lv.params) != 1:
                    raise ValueError(
                        f"{self.code} L{lv.index}: observable_is_parameter needs one "
                        "injection parameter and a declared observable to equate"
                    )
                (value,) = lv.params.values()
                if declared != value:
                    raise ValueError(
                        f"{self.code} L{lv.index}: observable {declared} is not the "
                        f"injection parameter {value}, so it is a measurement after all"
                    )
        overlap = set(self.primary_fields) & set(self.fallback_fields)
        if overlap:
            raise ValueError(
                f"{self.code}: {sorted(f.value for f in overlap)} listed as both "
                "primary and fallback"
            )
        if not self.primary_fields:
            raise ValueError(f"{self.code}: a defect that writes nothing cannot be injected")

    @property
    def layers(self) -> frozenset[InjectionLayer]:
        return frozenset(f.layer for f in self.primary_fields)

    @property
    def touches_mjmodel(self) -> bool:
        return InjectionLayer.MJMODEL in self.layers

    def level(self, index: int) -> DefectLevel:
        for lv in self.levels:
            if lv.index == index:
                return lv
        available = sorted(lv.index for lv in self.levels)
        raise KeyError(f"{self.code} has no level {index}; available: {available}")

    def observable_scope(self, *, family: str, level: int) -> dict[str, Any]:
        """Scope for the `defect.observable.*` lock entry."""
        return {"defect": self.code, "level": str(level), "family": family}


def _levels(
    key: str,
    values: tuple[float, ...],
    *,
    observables: tuple[float | None, ...],
    unit: str,
    scale: float = 1.0,
    anchor_first: bool = False,
) -> tuple[DefectLevel, ...]:
    if len(values) != len(observables):
        raise ValueError(f"{key}: {len(values)} values but {len(observables)} observables")
    return tuple(
        DefectLevel(
            index=i,
            params={key: v * scale},
            nominal_observable=o,
            observable_unit=unit,
            is_origin_anchor=anchor_first and i == 0,
        )
        for i, (v, o) in enumerate(zip(values, observables, strict=True))
    )


# ---------------------------------------------------------------------------
# The catalogue. Level values are transcribed from the 9.2.10 table; nominal
# observables are the figures printed in the same row, with None where the
# proposal marks 待核实 and the value has to come out of the lock file.
# ---------------------------------------------------------------------------

_D1 = DefectSpec(
    code="D1",
    name="joint calibration / encoder zero offset",
    priority=Priority.A,
    primary_fields=(WritableField.SENSING_TWIN,),
    fallback_fields=(),
    # 9.2.0: routed through the wrapper's sensing twin precisely so that
    # jnt_ref stays untouched.
    levels=_levels(
        "delta_q",
        (0.05, 0.15, 0.4, 0.7, 1.0),
        observables=(0.44e-3, 1.3e-3, 3.5e-3, 6.1e-3, 8.7e-3),
        unit="m",
        scale=DEG,
    ),
    changes_kinematic_graph=False,
    in_36_pairs=True,
    analytic_field=True,
    needs_memory_state=False,
    harm_class=HarmClass.UNCLASSIFIED,
    notes="c = J(q) dq, constant in the joint frame; sign decides the class per level",
)

_D2 = DefectSpec(
    code="D2",
    name="control delay",
    priority=Priority.A,
    primary_fields=(WritableField.CTRL_REWRITE, WritableField.WRAPPER_STATE),
    fallback_fields=(),
    # n = 0 coincides with the clean twin bit for bit and is the dose-curve
    # origin, so D2 contributes four non-zero cells rather than five (9.2.10).
    levels=_levels(
        "n_control_steps",
        (0.0, 1.0, 2.0, 3.0, 4.0),
        observables=(0.0, 50e-3, 100e-3, 150e-3, 200e-3),
        unit="s",
        anchor_first=True,
    ),
    changes_kinematic_graph=False,
    in_36_pairs=True,
    analytic_field=True,
    needs_memory_state=True,
    harm_class=HarmClass.UNDERTRAVEL,
    notes=(
        "action-side delay at the control step, never torque-side (appendix A-23): "
        "the closed form c = l * adot only holds for the action-side caliber"
    ),
)

_D3A = DefectSpec(
    code="D3a",
    name="finite transmission stiffness (series elastic joint)",
    priority=Priority.A,
    primary_fields=(WritableField.JNT_RANGE, WritableField.EQ_SOLREF_SOLIMP),
    fallback_fields=(),
    levels=_levels(
        "k_transmission",
        (3200.0, 1600.0, 800.0, 400.0, 200.0),
        observables=(None,) * 5,
        unit="m",
    ),
    changes_kinematic_graph=True,
    in_36_pairs=False,
    analytic_field=True,
    needs_memory_state=False,
    harm_class=HarmClass.OBSTACLE_DRIFT,
    notes=(
        "adds a co-located joint, so it changes the graph and stays out of the 36 "
        "pairs; dose arms of E1/E3 only. Not orthogonal to D2 (omega_n ~ 4.5 Hz "
        "sits inside the control bandwidth), which is why T8 uses D3b instead"
    ),
)

_D3B = DefectSpec(
    code="D3b",
    name="gravity droop (purely static)",
    priority=Priority.A,
    primary_fields=(WritableField.CTRL_REWRITE,),
    fallback_fields=(),
    levels=_levels(
        "sigma_g",
        (0.05, 0.10, 0.20, 0.40, 0.60),
        observables=(0.8e-3, 1.7e-3, 3.4e-3, 6.8e-3, 10e-3),
        unit="m",
    ),
    changes_kinematic_graph=False,
    in_36_pairs=True,
    analytic_field=True,
    needs_memory_state=False,
    harm_class=HarmClass.OBSTACLE_DRIFT,
    notes=(
        "perturbs the gravity compensation coefficient only, introducing no "
        "dynamic term; this is the caliber T8 needs so that the sign flip is not "
        "confounded with D2 (appendix A-16)"
    ),
)

_D4_LEVELS = _levels(
    "backlash",
    (0.05, 0.1, 0.3, 1.0, 3.0),
    observables=(0.22e-3, 0.44e-3, 1.3e-3, 4.4e-3, 13e-3),
    unit="m",
    scale=DEG,
)

_D4HI = DefectSpec(
    code="D4hi",
    name="gear backlash (high fidelity, shadow joint)",
    priority=Priority.A,
    primary_fields=(WritableField.JNT_RANGE, WritableField.EQ_SOLREF_SOLIMP),
    fallback_fields=(),
    levels=_D4_LEVELS,
    changes_kinematic_graph=True,
    in_36_pairs=True,
    analytic_field=True,
    needs_memory_state=True,
    harm_class=HarmClass.UNDERTRAVEL,
    notes=(
        "in the 36 pairs under the relaxed assertion of 9.3.4: the clean twin "
        "carries the same shadow joint with range=-1e-5 1e-5, so the graphs match "
        "up to one node whose position and axis agree bit for bit. Needed for T8 "
        "and the safety endpoint because only this caliber produces the real "
        "re-engagement impact"
    ),
)

_D4LO = DefectSpec(
    code="D4lo",
    name="gear backlash (play operator)",
    priority=Priority.A,
    primary_fields=(WritableField.CTRL_REWRITE, WritableField.WRAPPER_STATE),
    fallback_fields=(),
    levels=_D4_LEVELS,
    changes_kinematic_graph=False,
    in_36_pairs=False,
    analytic_field=True,
    needs_memory_state=True,
    harm_class=HarmClass.UNDERTRAVEL,
    notes=(
        "parallel data for the same cell. Its injection layer is homomorphic to "
        "the other five core defects, so this is the caliber that supports the "
        "strict 'FGW is exactly zero' argument. It has no transmission-side "
        "dynamics, hence not usable for T8 or the safety endpoint"
    ),
)

_D5 = DefectSpec(
    code="D5",
    name="actuator dead zone",
    priority=Priority.A,
    primary_fields=(WritableField.CTRL_REWRITE,),
    fallback_fields=(WritableField.ACTUATOR_BIASPRM, WritableField.GAINTYPE_BIASTYPE),
    levels=_levels(
        "w_over_tau_max",
        (0.005, 0.01, 0.03, 0.06, 0.10),
        observables=(None,) * 5,
        unit="m",
    ),
    changes_kinematic_graph=False,
    in_36_pairs=True,
    analytic_field=True,
    needs_memory_state=False,
    harm_class=HarmClass.UNDERTRAVEL,
    notes="one caliber across all six families (defect_impl_variant records which)",
)

_D6 = DefectSpec(
    code="D6",
    name="friction asymmetry",
    priority=Priority.A,
    # The 9.2.6 fallback, now the shipping caliber: both the symmetric base and
    # the directional part go through qfrc_applied in one write, leaving
    # dof_frictionloss at zero. See the module docstring.
    primary_fields=(WritableField.QFRC_APPLIED,),
    fallback_fields=(WritableField.DOF_FRICTIONLOSS,),
    levels=_levels(
        "delta_f",
        (0.05, 0.1, 0.2, 0.4, 0.8),
        observables=(None,) * 5,
        unit="m",
    ),
    changes_kinematic_graph=False,
    # Out, by 9.2.6's own decision rule. The measurement it asks for came back at
    # 0.64 to 3.52 N m across the six families against a 0.025 N m gate. That is
    # not a near miss and no choice of f_bar rescues it: the offset the
    # contingency introduces *is* the breakaway bound it gave up, so it is of
    # order f_bar and the gate is of order the smallest dose. Nor can D6 keep the
    # constraint caliber -- the baseline's FGW node features read a friction
    # coefficient (verified in registry.fgw), and with f_bar in dof_frictionloss
    # the twin pair measures 6.98 apart, further than Panda from IIWA at 2.55.
    # D8 takes the sixth slot, per the candidate list of 9.2.5.
    in_36_pairs=False,
    analytic_field=True,
    needs_memory_state=False,
    harm_class=HarmClass.UNCLASSIFIED,
    notes=(
        "left the 36 pairs on the preregistered stiction rule of 9.2.6. The "
        "qfrc_applied caliber gives up the pre-emptive stiction of the constraint "
        "implementation, and defect.d6_stiction_offset measures the cost at 0.64 "
        "to 3.52 N m, far above half the smallest level. Still runnable, still "
        "carries its own dose ladder; it is out of the primary 36 and out of the "
        "n = 36 tables, not out of the catalogue"
    ),
)

_D7 = DefectSpec(
    code="D7",
    name="structural compliance vibration",
    priority=Priority.C,
    primary_fields=(WritableField.JNT_RANGE,),
    fallback_fields=(),
    levels=(
        DefectLevel(index=0, params={"link_compliance": 1.0},
                    nominal_observable=None, observable_unit="m"),
    ),
    changes_kinematic_graph=True,
    in_36_pairs=False,
    analytic_field=False,
    needs_memory_state=False,
    harm_class=HarmClass.UNCLASSIFIED,
    notes=(
        "numerical ground truth only, so it cannot enter the surrogate matching "
        "for Lambda^exc. Kept because it deliberately breaks (A4) and carries the "
        "zero-GPU open-loop replay (appendix A-21 forbids dropping it)"
    ),
)

_D8 = DefectSpec(
    code="D8",
    # Promoted from class C to fill the slot D6 vacated, which is the first
    # option on 9.2.5's replacement list. Two things make it the cheap
    # substitution and one makes it the awkward one. Its compensation field is
    # multiplicative and closed form, so it enters the surrogate matching for
    # Lambda^exc without further work; and its observable *is* its injection
    # parameter, a dimensionless gain, so the five-level ladder needs no
    # measurement at all -- unlike D6, whose observable column was empty and
    # blocked on the reference trajectories. Against that, 9.2.8 records that it
    # is strongly coupled with the confounder X7, so its result may never be
    # read on its own: it is reported alongside D3b, and `DEFECTS_REPORTED_WITH`
    # is what carries that obligation into the report layer.
    name="transmission gain error",
    priority=Priority.A,
    primary_fields=(WritableField.ACTUATOR_GAINPRM,),
    fallback_fields=(WritableField.GAINTYPE_BIASTYPE,),
    levels=_levels(
        "gain",
        (0.95, 0.90, 0.80, 0.70, 0.55),
        observables=(0.95, 0.90, 0.80, 0.70, 0.55),
        unit="dimensionless",
    ),
    changes_kinematic_graph=False,
    in_36_pairs=True,
    analytic_field=True,
    needs_memory_state=False,
    harm_class=HarmClass.UNCLASSIFIED,
    observable_is_parameter=True,
    notes=(
        "the only defect class for which gain asymmetry actually holds (appendix "
        "A-12); the motor shortcut expands to gainprm='1 0 0', so the constant "
        "levels need no callback. Sixth core defect since D6 left on the 9.2.6 "
        "stiction rule; must be reported with D3b (9.2.8, X7 coupling)"
    ),
)

_D9 = DefectSpec(
    code="D9",
    name="cogging torque",
    priority=Priority.C,
    primary_fields=(WritableField.QFRC_APPLIED,),
    fallback_fields=(),
    levels=(
        DefectLevel(index=0, params={"amplitude": 1.0, "n_poles": 1.0, "phase": 0.0},
                    nominal_observable=None, observable_unit="N m"),
    ),
    changes_kinematic_graph=False,
    in_36_pairs=False,
    analytic_field=False,
    needs_memory_state=False,
    harm_class=HarmClass.UNCLASSIFIED,
    notes="-A sin(N q + phi) superposed on qfrc_applied; zero-GPU replay carrier",
)

_D10 = DefectSpec(
    code="D10",
    name="time-varying thermal drift",
    priority=Priority.C,
    primary_fields=(WritableField.SENSING_TWIN, WritableField.WRAPPER_STATE),
    fallback_fields=(),
    # 3 time constants x 2 asymptotes, not a five-level ladder (9.2.10).
    levels=tuple(
        DefectLevel(
            index=i,
            params={"tau_thermal": tau, "delta_q_inf": _D1.level(li).params["delta_q"]},
            nominal_observable=None,
            observable_unit="m",
        )
        for i, (tau, li) in enumerate(
            [(60.0, 1), (60.0, 2), (300.0, 1), (300.0, 2), (1800.0, 1), (1800.0, 2)]
        )
    ),
    changes_kinematic_graph=False,
    in_36_pairs=False,
    analytic_field=True,
    needs_memory_state=True,
    harm_class=HarmClass.UNCLASSIFIED,
    notes=(
        "reuses the D1 path plus a drift accumulator; the asymptotes are D1 levels "
        "L2 and L3. Deliberately violates (A4) -- that is its purpose, so it must "
        "not be dropped for budget (appendix A-21)"
    ),
)

DEFECTS: dict[str, DefectSpec] = {
    d.code: d
    for d in (_D1, _D2, _D3A, _D3B, _D4HI, _D4LO, _D5, _D6, _D7, _D8, _D9, _D10)
}

# 9.3.3: the 36 structurally aligned twin pairs are 6 families x 6 core defects.
# D4 contributes one cell, whose primary data comes from the high-fidelity
# caliber; D4lo supplies parallel data for the same cell.
#
# D8 sits in the sixth slot rather than D6. 9.2.6's contingency rule decided it,
# not a preference: with the baseline's FGW features reading a friction
# coefficient, D6's symmetric component has to move to qfrc_applied, and the
# stiction offset that move costs came back at 0.64 to 3.52 N m against a
# 0.025 N m gate. 9.2.5's replacement list puts D8 first. The count is still six
# and every n = 36 downstream of it is unchanged.
CORE_DEFECTS: tuple[str, ...] = ("D1", "D2", "D3b", "D4hi", "D5", "D8")

# Defects whose result may not be read alone. 9.2.8 records D8 as strongly
# coupled with the confounder X7, so a gain-degradation effect and a
# gripper-noise effect cannot be told apart from D8's column by itself; the pair
# is reported together. This was a footnote while D8 was class C and is a
# reporting obligation now that it is one of the six.
DEFECTS_REPORTED_WITH: dict[str, tuple[str, ...]] = {"D8": ("D3b",)}

# mjModel fields that a morphology node feature reads, in either feature set of
# 9.3.4. Writing one of these makes a twin pair distinguishable and breaks X1,
# which is the invariant `_self_check` enforces on the core six.
#
# Kept here as a declaration rather than imported from `registry.fgw`: this
# module is the bottom of the dependency graph and must not pull in POT and
# networkx to state a six-element set. `tests` cross-checks it against the real
# `FEATURE_FIELDS`, so a feature set that grows a column cannot drift away from
# it silently.
#
# `actuator_gainprm` is deliberately absent, which is what lets D8 be a core
# defect at all. Both feature sets take the torque limit from
# `actuator_ctrlrange`, and D8 leaves that alone. The gain is read into
# `GraphNode.gain` but is not a feature column in either set.
DESCRIPTOR_VISIBLE_FIELDS: frozenset[WritableField] = frozenset(
    {
        WritableField.JNT_RANGE,
        WritableField.DOF_FRICTIONLOSS,
    }
)


def core_defect_codes() -> tuple[str, ...]:
    return CORE_DEFECTS


def defect(code: str) -> DefectSpec:
    try:
        return DEFECTS[code]
    except KeyError:
        raise KeyError(
            f"unknown defect {code!r}; the catalogue is {sorted(DEFECTS)}"
        ) from None


def _self_check() -> None:
    """Invariants the rest of the package relies on.

    Runs at import because these are cheap and a violation means the catalogue
    contradicts the proposal, which is worth failing loudly rather than at some
    later assertion.
    """
    if len(CORE_DEFECTS) != 6:
        raise AssertionError("9.3.3 fixes the core defect count at six")
    for code in CORE_DEFECTS:
        spec = defect(code)
        if not spec.in_36_pairs:
            raise AssertionError(f"{code} is a core defect but not marked in_36_pairs")
        read = set(spec.primary_fields) & DESCRIPTOR_VISIBLE_FIELDS
        if read and code != "D4hi":
            raise AssertionError(
                f"{code} writes {sorted(f.value for f in read)}, which a morphology "
                "node feature reads, so its twin pair is distinguishable and "
                "fgw_baseline stops being zero. Only D4hi may, and only because "
                "its shadow joint is on both sides of the pair"
            )
    for spec in DEFECTS.values():
        if spec.needs_memory_state and not spec.analytic_field and spec.code not in {"D7", "D9"}:
            raise AssertionError(f"{spec.code}: memory state without an analytic field")


_self_check()
