"""A8: the `solref` / `solimp` sensitivity ablation, which closes confounder X10.

10.6.2 states X10 as follows: MuJoCo contacts are soft constraints, so
"environment stiffness" is a solver parameter and not a physical constant, and
every number a safety endpoint produces therefore inherits a knob. The design
that removes it is this ablation: the safety conclusion must point the same way
under three parameter groups, and the force-calibration curve is measured once
per group.

What the three groups move, and what they deliberately do not
-------------------------------------------------------------

The groups are *relative*. Robosuite 1.5.2 does not compile a single contact
stiffness: on a Panda / NutAssemblySquare model the geoms carry two distinct
`solref` rows, (0.02, 1.0) on the arm and the table and (0.01, 0.5) on the
gripper pads and the manipulated objects, and two distinct `solimp` rows.
Writing one absolute triple over all of them would make the "default" group a
different environment from the one every other chapter ran in, and A8 would
then be comparing three novel worlds rather than perturbing the one under test.
So a group is a pair of scale factors applied to the compiled values, and the
default group is all-ones -- a bit-exact no-op that still runs the same
arithmetic, which is the same discipline CONTRACT 3.3 imposes on the clean twin.

`solref[0]` is the contact time constant: smaller is stiffer. It is scaled
directly, with a floor at two simulation steps, below which MuJoCo's implicit
reference model is not integrable at the 9.1.1 timestep of 2 ms and the run
would be measuring divergence rather than stiffness. `solimp`'s `dmin` and
`dmax` are impedances in (0, 1), and they are moved by scaling their *gap to
one*: d' = 1 - s (1 - d). That parameterisation is monotone in the right
direction (s < 1 is harder), and it cannot leave the open interval for any
positive s, so there is no clipping step anywhere -- a clipped impedance would
silently make two groups equal at the geoms that clipped.

Three fields are left alone, each for a reason:

* `solref[1]`, the damping ratio. Moving it off the compiled value introduces
  contact bounce, which is a claim about restitution rather than about
  stiffness, and X10 is stated in terms of stiffness.
* `solimp[2:5]`, the width and the two shape parameters. They set where the
  impedance transitions, not how hard the contact is at penetration.
* `eq_solref` / `eq_solimp`. Those belong to the equality constraints, and
  9.3.4's D4hi backlash band is realised as one (`defects.inject` writes
  `shadow_joint.BACKLASH_SOLREF` into it after every reset). Scaling them here
  would make A8 a defect-amplitude ablation on D4hi and nothing else.

When the write happens
----------------------

`hard_reset` is True in `defects.inject._make_env` and that branch rebuilds
`sim` from MJCF, so any `mjModel` write made once in a constructor disappears
mid-session. `defects.inject.DefectWrapper` solves this by re-applying its own
model-level writes inside `reset`; `build_with_solver_group` does the same by
rebinding `reset`, so the solver group is re-applied after the wrapper's writes
on every episode. The order matters: the wrapper writes `eq_*` for D4hi and this
module writes `geom_*`, and the two do not overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from portal_trace.configs.schema import ConfigError

__all__ = [
    "MIN_TIMECONST_STEPS",
    "SolverGroup",
    "apply_solver_group",
    "build_with_solver_group",
]

# MuJoCo's soft-constraint reference model is integrated at the simulation
# timestep; a time constant below two steps is not resolvable and the contact
# becomes an impulse whose magnitude depends on the step boundary. 9.1.1 pins
# the timestep at 2 ms, so this floor is 4 ms.
MIN_TIMECONST_STEPS = 2


@dataclass(frozen=True)
class SolverGroup:
    """One of A8's three groups, as the preregistered level block states it.

    Both fields are multiplicative on the compiled model. `default` is the
    identity and is still applied, so the three arms differ by the value of the
    scale and not by whether the code path ran.
    """

    group_id: str
    solref_timeconst_scale: float
    solimp_gap_scale: float
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.group_id.strip():
            raise ConfigError("a solver group needs an id; it names the arm that runs it")
        for name in ("solref_timeconst_scale", "solimp_gap_scale"):
            value = getattr(self, name)
            if value <= 0.0:
                raise ConfigError(
                    f"solver group {self.group_id}: {name} is {value!r}. A scale at or "
                    "below zero is not a softer contact, it is a different constraint "
                    "model: the impedance would leave (0, 1) and the reference time "
                    "constant would change sign into MuJoCo's direct form"
                )

    @classmethod
    def from_parameters(cls, parameters: dict[str, Any], *, where: str) -> SolverGroup:
        missing = sorted(
            {"group_id", "solref_timeconst_scale", "solimp_gap_scale"} - set(parameters)
        )
        if missing:
            raise ConfigError(f"{where}: A8 level is missing {missing}")
        return cls(
            group_id=str(parameters["group_id"]),
            solref_timeconst_scale=float(parameters["solref_timeconst_scale"]),
            solimp_gap_scale=float(parameters["solimp_gap_scale"]),
            rationale=str(parameters.get("rationale", "")),
        )

    @property
    def is_identity(self) -> bool:
        return (
            self.solref_timeconst_scale == 1.0 and self.solimp_gap_scale == 1.0
        )

    def solref(self, compiled: np.ndarray) -> np.ndarray:
        """The group's `solref`, given the model's compiled rows (N, 2)."""
        out = np.array(compiled, dtype=np.float64, copy=True)
        out[:, 0] = out[:, 0] * self.solref_timeconst_scale
        return out

    def solimp(self, compiled: np.ndarray) -> np.ndarray:
        """The group's `solimp`, given the model's compiled rows (N, 5)."""
        out = np.array(compiled, dtype=np.float64, copy=True)
        out[:, 0:2] = 1.0 - self.solimp_gap_scale * (1.0 - out[:, 0:2])
        return out


def _check_timeconst(solref: np.ndarray, group: SolverGroup, timestep: float) -> None:
    floor = MIN_TIMECONST_STEPS * timestep
    # Direct solref (a negative first entry) is MuJoCo's stiffness/damping form
    # rather than a time constant, and the floor does not apply to it. No
    # robosuite model in the registry compiles one, so this is a refusal rather
    # than a branch: a model that did would need its own preregistered scaling.
    if (solref[:, 0] < 0.0).any():
        raise ConfigError(
            f"solver group {group.group_id}: the model carries direct-form solref "
            "rows (negative first entry). A8's scales are defined on the "
            "time-constant form; scaling a stiffness by the same factor would move "
            "the contact in the opposite direction"
        )
    too_fast = solref[:, 0] < floor
    if too_fast.any():
        raise ConfigError(
            f"solver group {group.group_id}: scaling by "
            f"{group.solref_timeconst_scale} takes {int(too_fast.sum())} geom "
            f"contacts to a time constant below {floor:.4f} s, which is "
            f"{MIN_TIMECONST_STEPS} simulation steps at the 9.1.1 timestep of "
            f"{timestep:.4f} s. Below that the reference model is not integrable "
            "and the safety endpoint would be reading divergence, not stiffness"
        )


def apply_solver_group(model: Any, group: SolverGroup) -> dict[str, int]:
    """Write the group's contact parameters into a compiled `mjModel`.

    `model` is a raw `mujoco.MjModel`. Returns how many rows each field
    received, which the evaluation record carries so that "the ablation ran"
    is a countable statement rather than an assumption.
    """
    geom_solref = group.solref(np.asarray(model.geom_solref, dtype=np.float64))
    _check_timeconst(geom_solref, group, float(model.opt.timestep))
    model.geom_solref[:, :] = geom_solref
    model.geom_solimp[:, :] = group.solimp(
        np.asarray(model.geom_solimp, dtype=np.float64)
    )
    written = {"geom": int(model.ngeom)}
    # Explicit contact pairs override the per-geom values when present, so a
    # model that has them and does not get them written would run the default
    # group under another name. No registry task compiles any today; the branch
    # is on the model's own count, not on a task name.
    if int(model.npair):
        pair_solref = group.solref(np.asarray(model.pair_solref, dtype=np.float64))
        _check_timeconst(pair_solref, group, float(model.opt.timestep))
        model.pair_solref[:, :] = pair_solref
        model.pair_solimp[:, :] = group.solimp(
            np.asarray(model.pair_solimp, dtype=np.float64)
        )
        written["pair"] = int(model.npair)
    return written


def build_with_solver_group(
    task_spec: Any, family: str, delta: Any, seeds: Any, group: SolverGroup
) -> Any:
    """A `DefectWrapper` whose contacts carry the group, on every reset.

    Built through `defects.inject.build_single` rather than through a second
    construction path, so the A8 arms and the main arms compile the same MJCF
    with the same robosuite keyword arguments and differ only in the two scale
    factors written afterwards.
    """
    from portal_trace.defects.inject import build_single

    wrapper = build_single(task_spec, family, delta, seeds)
    original_reset = wrapper.reset

    def reset(episode: int | None = None) -> Any:
        # After the wrapper's own model-level writes: robosuite's hard reset
        # rebuilds the model, and the wrapper re-applies `eq_*` inside `reset`.
        out = original_reset(episode)
        apply_solver_group(wrapper.env.sim.model._model, group)
        return out

    # Both handles, because `DefectWrapper.__init__` rebinds `env.reset` to its
    # own bound method precisely so that a caller reaching past the wrapper
    # cannot skip the re-application; that binding still points at the wrapper's
    # original method and would skip this one.
    wrapper.reset = reset  # type: ignore[method-assign]
    wrapper.env.reset = reset
    apply_solver_group(wrapper.env.sim.model._model, group)
    wrapper.solver_group = group  # type: ignore[attr-defined]
    return wrapper
