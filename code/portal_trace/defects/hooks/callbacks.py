"""The `mjcb_act_gain` / `mjcb_act_bias` alternate caliber for D5 and D8.

9.1.2 fixes the primary caliber for both defects as a rewrite of `sim.data.ctrl`
inside `_pre_action`: same 500 Hz, same position ahead of MuJoCo's actuators,
but local state, no GIL, and a pure function under `vmap` if the physics ever
moves to MJX. This module is the other caliber -- `gaintype`/`biastype = "user"`
with a Python-side callback -- and 9.2.5 / 9.2.8 keep it for the one case the
primary cannot express: a force law that has to run *inside* MuJoCo's actuator
pipeline, e.g. coupled to activation dynamics `act`.

It is not a fallback. Nothing in this package installs it automatically, and
`inject` never reaches for it when something fails; a caller has to name it.
CONTRACT 3.6 is the reason: a caliber that engages on its own would put two
different physical models under one defect label with no trace in the data.

**Cost, stated rather than implied.** `mjcb_act_gain` and `mjcb_act_bias` are
process-global C function pointers. A Python callback therefore (i) takes the
GIL once per USER actuator per `mj_step`, which at 500 Hz x 7 arm actuators is
3500 acquisitions per simulated second, and (ii) cannot be per-environment:
installing a second calibration in the same process would silently reinterpret
the first environment's actuators. Both are handled by refusing rather than by
coping -- the installer raises on a second install, and 9.1.2 says that when
this caliber is actually needed the callback is written in C.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

__all__ = [
    "UserActuatorCaliber",
    "CalibrationConflict",
    "install_user_actuator_caliber",
    "uninstall_user_actuator_caliber",
    "installed_caliber",
]


class CalibrationConflict(RuntimeError):
    """A second USER-actuator calibration was requested in one process.

    Fatal on purpose: `mjcb_act_gain` is a single global pointer, so the second
    install would take over the first environment's actuators as well.
    """


@dataclass(frozen=True)
class UserActuatorCaliber:
    """The per-actuator gain and dead-zone table the callbacks read.

    `gain` implements D8 (`scalar_force = gain_term * ctrl + bias_term`, so a
    constant transmission error is a constant gain term). `dead_zone` implements
    D5's standard dead zone through the bias term: with gain g,

        g * u - g * sign(u) * min(|u|, w) = g * sign(u) * max(|u| - w, 0)

    which is 9.2.5's form and not the pass-through variant that section rejects.
    """

    actuator_ids: tuple[int, ...]
    gain: tuple[float, ...]
    dead_zone: tuple[float, ...]
    model_signature: tuple[int, int, int]
    _table: dict[int, tuple[float, float]] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        n = len(self.actuator_ids)
        if len(self.gain) != n or len(self.dead_zone) != n:
            raise ValueError(
                f"{n} actuators but {len(self.gain)} gains and "
                f"{len(self.dead_zone)} dead zones"
            )
        for w in self.dead_zone:
            if w < 0.0:
                raise ValueError(f"a dead zone of {w} N m is not a dead zone")
        object.__setattr__(
            self,
            "_table",
            {
                i: (g, w)
                for i, g, w in zip(self.actuator_ids, self.gain, self.dead_zone, strict=True)
            },
        )

    def lookup(self, actuator_id: int) -> tuple[float, float]:
        try:
            return self._table[actuator_id]
        except KeyError:
            raise CalibrationConflict(
                f"the USER actuator callback fired for actuator {actuator_id}, which "
                "this calibration does not cover. MuJoCo only calls it for "
                "gaintype/biastype=user, so another model in this process has USER "
                "actuators and is now reading our table. Run the two models in "
                "separate processes, or move both onto the primary ctrl-rewrite "
                "caliber (9.1.2)."
            ) from None


_INSTALLED: UserActuatorCaliber | None = None


def installed_caliber() -> UserActuatorCaliber | None:
    return _INSTALLED


def _model_signature(model: Any) -> tuple[int, int, int]:
    return (int(model.nq), int(model.nv), int(model.nu))


def _gain_callback(model: Any, data: Any, actuator_id: int) -> float:
    caliber = _INSTALLED
    if caliber is None:
        raise CalibrationConflict(
            "mjcb_act_gain fired with no calibration installed; the callback was "
            "uninstalled while a model still declared gaintype=user"
        )
    gain, _ = caliber.lookup(int(actuator_id))
    return gain


def _bias_callback(model: Any, data: Any, actuator_id: int) -> float:
    caliber = _INSTALLED
    if caliber is None:
        raise CalibrationConflict(
            "mjcb_act_bias fired with no calibration installed; the callback was "
            "uninstalled while a model still declared biastype=user"
        )
    gain, width = caliber.lookup(int(actuator_id))
    if width == 0.0:
        return 0.0
    u = float(data.ctrl[actuator_id])
    return -gain * np.sign(u) * min(abs(u), width)


def install_user_actuator_caliber(
    model: Any,
    *,
    actuator_ids: tuple[int, ...],
    gain: tuple[float, ...],
    dead_zone: tuple[float, ...],
) -> UserActuatorCaliber:
    """Switch the named actuators to USER gain/bias and install the callbacks.

    `model` is a raw `mujoco.MjModel`. The switch is a write to
    `actuator_gaintype`/`actuator_biastype`, i.e. the `gaintype/biastype` column
    of the 9.2.0 matrix, and like every mjModel write it has to be re-applied
    after a `hard_reset`.
    """
    global _INSTALLED
    if _INSTALLED is not None:
        raise CalibrationConflict(
            "a USER-actuator calibration is already installed in this process. "
            "mjcb_act_gain / mjcb_act_bias are global function pointers, so the "
            "second install would take over the first model's actuators. "
            "Uninstall first, or keep the two environments in separate processes."
        )
    caliber = UserActuatorCaliber(
        actuator_ids=tuple(int(i) for i in actuator_ids),
        gain=tuple(float(g) for g in gain),
        dead_zone=tuple(float(w) for w in dead_zone),
        model_signature=_model_signature(model),
    )
    apply_user_types(model, caliber)
    _INSTALLED = caliber
    mujoco.set_mjcb_act_gain(_gain_callback)
    mujoco.set_mjcb_act_bias(_bias_callback)
    return caliber


def apply_user_types(model: Any, caliber: UserActuatorCaliber) -> None:
    """Re-declare the USER gain/bias types on a freshly compiled model."""
    if _model_signature(model) != caliber.model_signature:
        raise CalibrationConflict(
            f"this calibration was built against a model with "
            f"(nq, nv, nu) = {caliber.model_signature}, and is being applied to "
            f"{_model_signature(model)}. Actuator ids do not survive a change of "
            "model, so re-derive the calibration by name."
        )
    for i in caliber.actuator_ids:
        model.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_USER
        model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_USER


def uninstall_user_actuator_caliber(model: Any | None = None) -> None:
    """Drop the callbacks and, if a model is given, restore FIXED/NONE types."""
    global _INSTALLED
    caliber = _INSTALLED
    mujoco.set_mjcb_act_gain(None)
    mujoco.set_mjcb_act_bias(None)
    _INSTALLED = None
    if model is not None and caliber is not None:
        for i in caliber.actuator_ids:
            model.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
            model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_NONE
            # The motor shortcut expands to gainprm="1 0 0" (measured); restoring
            # the type without restoring the parameter would leave a silent zero.
            model.actuator_gainprm[i, 0] = 1.0
            model.actuator_biasprm[i, :3] = 0.0
