"""Injection hooks that reach outside the `_pre_action` rewrite.

Two of them, and both are exceptional in the same way: they touch something the
wrapper layer cannot reach.

`shadow_joint` rewrites the MJCF, because D4hi, D3a and D7 add a joint node and
no runtime write can do that. It is registered as a robosuite xml processor so
that the node survives `hard_reset`.

`callbacks` installs `mjcb_act_gain` / `mjcb_act_bias`, the alternate caliber for
D5 and D8. Process-global, GIL-bound, and never engaged automatically -- see its
module docstring and 9.1.2.
"""

from portal_trace.defects.hooks.callbacks import (
    CalibrationConflict,
    UserActuatorCaliber,
    apply_user_types,
    install_user_actuator_caliber,
    installed_caliber,
    uninstall_user_actuator_caliber,
)
from portal_trace.defects.hooks.shadow_joint import (
    COMPLIANCE_DAMPING_RATIO,
    ShadowJointSpec,
    ShadowKind,
    assert_driven_armature,
    plan_for,
    resolve_inertia,
    rewrite_xml,
    shadow_joint_names,
)

__all__ = [
    "CalibrationConflict",
    "UserActuatorCaliber",
    "apply_user_types",
    "install_user_actuator_caliber",
    "installed_caliber",
    "uninstall_user_actuator_caliber",
    "COMPLIANCE_DAMPING_RATIO",
    "ShadowJointSpec",
    "ShadowKind",
    "assert_driven_armature",
    "plan_for",
    "resolve_inertia",
    "rewrite_xml",
    "shadow_joint_names",
]
