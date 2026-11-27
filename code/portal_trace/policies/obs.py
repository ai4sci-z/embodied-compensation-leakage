"""The policy input: two 84x84 cameras and the 70-dimensional proprioception.

Implements the observation row of 9.6.1 and the hdf5 shapes of 9.6.3, with one
override that the whole package depends on.

**The stored proprioception vector is 70 wide, not the 52 of 9.6.1.** That
section assumes `robot0_proprio-state` is 32 dimensional on a 7-DoF arm and marks
the number as unverified; it also says in the same row that if the native width
differs then 52 changes with it. It was measured at 50 on robosuite 1.5.2 (see
`PROPRIO_GROUPS` for the per-sensor arithmetic), so:

    50 native + 7 tau_g + 6 wrist force/torque = 63 physical channels
    63 physical channels + 7 DoF mask         = 70 stored

The discipline of 9.6.1 carries over verbatim with the new numbers: any table
quoting **63** must say "physical channels, mask excluded", and the hdf5 dataset
`obs/robot0_proprio-state` has shape `(N, 70)`.

The concatenation order in `PROPRIO_GROUPS` is measured, not assumed. robosuite
registers `joint_pos` first, ahead of `joint_pos_cos`/`joint_pos_sin`, which is
not the order 9.6.1 lists the sensors in; the layout below is what
`robosuite.make("Lift", robots="Panda")` actually returns.

Joint-indexed channels are zero-padded to 7 across the 5/6/7-DoF families and the
`dof_mask` group says which entries are real. Without the mask a padded zero and
a genuinely zero joint velocity are the same vector (9.6.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from portal_trace.registry.tasks import CONTROL_FREQ

__all__ = [
    "NATIVE_PROPRIO_DIM",
    "TAU_G_DIM",
    "WRIST_FT_DIM",
    "DOF_MASK_DIM",
    "PHYSICAL_PROPRIO_DIM",
    "PROPRIO_DIM",
    "MAX_DOF",
    "ACTION_DIM",
    "ARM_ACTION_DIM",
    "TRANSLATION_DIMS",
    "GRIPPER_DIM_INDEX",
    "CHUNK",
    "IMAGE_SIZE",
    "CAMERAS",
    "K_HIST_LEVELS",
    "K_HIST_MAIN",
    "NormKind",
    "ProprioGroup",
    "PROPRIO_GROUPS",
    "proprio_group",
    "control_step_joint_acceleration",
    "PROPRIO_ACCELERATION_LEGACY",
    "PROPRIO_ACCELERATION_CAUSAL",
    "assemble_proprio",
    "ObsSpec",
]

# --- proprioception -------------------------------------------------------

# Measured on robosuite 1.5.2, Panda, 7 DoF: joint_pos 7 + joint_pos_cos 7 +
# joint_pos_sin 7 + joint_vel 7 + joint_acc 7 + eef_pos 3 + eef_quat 4 +
# eef_quat_site 4 + gripper_qpos 2 + gripper_qvel 2 = 50.
NATIVE_PROPRIO_DIM = 50
TAU_G_DIM = 7        # gravity torque channels, 9.7.3 needs them in the policy input
WRIST_FT_DIM = 6     # wrist force/torque, the r4 contact predicate reads it
DOF_MASK_DIM = 7     # 1 on a real joint, 0 on a zero-padded one

# 50 + 7 + 6 = 63. Quote this number only as "physical channels, mask excluded".
PHYSICAL_PROPRIO_DIM = NATIVE_PROPRIO_DIM + TAU_G_DIM + WRIST_FT_DIM
# 63 + 7 = 70. This is the hdf5 width and the policy input width.
PROPRIO_DIM = PHYSICAL_PROPRIO_DIM + DOF_MASK_DIM

MAX_DOF = 7          # families cover 5/6/7 DoF; joint channels pad up to this

# --- action space (9.6.1) -------------------------------------------------

ARM_ACTION_DIM = 6   # OSC_POSE incremental pose
ACTION_DIM = 7       # plus one gripper dimension, all of it in [-1, 1]
GRIPPER_DIM_INDEX = 6
# 1 action unit = 50 mm of translation / 0.5 rad of rotation (9.6.1). B11's
# residual is confined to these three dimensions.
TRANSLATION_DIMS = (0, 1, 2)

# Action chunk length, 10.4.0. Deliberately a module constant and not a config
# field: no experiment in 10.4-10.6 varies it, and a knob here would let two arms
# be compared across different chunk lengths.
CHUNK = 8

# --- cameras (9.6.1) -----------------------------------------------------

IMAGE_SIZE = 84
CAMERAS: tuple[str, ...] = ("agentview_image", "robot0_eye_in_hand_image")

# History window, A2's ablation axis; 5 frames = 0.25 s at 20 Hz is the main
# caliber. K_hist = 1 is the single-frame arm of T7.
K_HIST_LEVELS: tuple[int, ...] = (1, 2, 5, 10)
K_HIST_MAIN = 5


class NormKind(str, Enum):
    """How a proprioception group is normalised under the 9.6.2 global caliber.

    The statistics themselves are computed over the training split by the data
    pipeline, not here; this enum fixes *which* transform each group gets, which
    is a property of the layout and therefore belongs with the layout.
    """

    STANDARDISE = "standardise"      # subtract mean, divide by max(sigma, 1e-3)
    ZERO_MEAN = "zero_mean"          # cos/sin are already bounded, mean only
    UNIT_NORM = "unit_norm"          # quaternions: renormalise, never re-centre
    NONE = "none"                    # the DoF mask is already 0/1


@dataclass(frozen=True)
class ProprioGroup:
    """One contiguous slice of the 70-dimensional vector."""

    name: str
    start: int
    width: int
    norm: NormKind
    joint_indexed: bool  # True if the slice is one entry per joint and zero-padded

    @property
    def stop(self) -> int:
        return self.start + self.width

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)


def _groups() -> tuple[ProprioGroup, ...]:
    # 9.6.2 lists the normalisation groups against the 32-dim layout it assumed,
    # so it says nothing about raw joint_pos or joint_acc. Both are standardised:
    # the cos/sin argument for mean-only centring is that those channels are
    # bounded by construction, which raw angles and accelerations are not.
    layout: tuple[tuple[str, int, NormKind, bool], ...] = (
        ("joint_pos", MAX_DOF, NormKind.STANDARDISE, True),
        ("joint_pos_cos", MAX_DOF, NormKind.ZERO_MEAN, True),
        ("joint_pos_sin", MAX_DOF, NormKind.ZERO_MEAN, True),
        ("joint_vel", MAX_DOF, NormKind.STANDARDISE, True),
        ("joint_acc", MAX_DOF, NormKind.STANDARDISE, True),
        ("eef_pos", 3, NormKind.STANDARDISE, False),
        ("eef_quat", 4, NormKind.UNIT_NORM, False),
        ("eef_quat_site", 4, NormKind.UNIT_NORM, False),
        ("gripper_qpos", 2, NormKind.STANDARDISE, False),
        ("gripper_qvel", 2, NormKind.STANDARDISE, False),
        ("tau_g", TAU_G_DIM, NormKind.STANDARDISE, True),
        ("wrist_ft", WRIST_FT_DIM, NormKind.STANDARDISE, False),
        ("dof_mask", DOF_MASK_DIM, NormKind.NONE, True),
    )
    out: list[ProprioGroup] = []
    start = 0
    for name, width, norm, joint_indexed in layout:
        out.append(ProprioGroup(name, start, width, norm, joint_indexed))
        start += width
    if start != PROPRIO_DIM:
        raise AssertionError(f"proprio layout sums to {start}, expected {PROPRIO_DIM}")
    return tuple(out)


PROPRIO_GROUPS: tuple[ProprioGroup, ...] = _groups()

_BY_NAME: dict[str, ProprioGroup] = {g.name: g for g in PROPRIO_GROUPS}


def proprio_group(name: str) -> ProprioGroup:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(
            f"no proprioception group {name!r}; the 70-dim layout has "
            f"{sorted(_BY_NAME)}"
        ) from None


_NATIVE_OBS_KEYS: tuple[tuple[str, str], ...] = (
    ("joint_pos", "robot0_joint_pos"),
    ("joint_pos_cos", "robot0_joint_pos_cos"),
    ("joint_pos_sin", "robot0_joint_pos_sin"),
    ("joint_vel", "robot0_joint_vel"),
    ("joint_acc", "robot0_joint_acc"),
    ("eef_pos", "robot0_eef_pos"),
    ("eef_quat", "robot0_eef_quat"),
    ("eef_quat_site", "robot0_eef_quat_site"),
    ("gripper_qpos", "robot0_gripper_qpos"),
    ("gripper_qvel", "robot0_gripper_qvel"),
)


PROPRIO_ACCELERATION_LEGACY = "simulator_qacc_v0"
PROPRIO_ACCELERATION_CAUSAL = "causal_velocity_difference_20hz_v1"


def control_step_joint_acceleration(
    velocity: np.ndarray,
    previous_velocity: np.ndarray | None = None,
) -> np.ndarray:
    """Causal control-step velocity difference in rad/s², with zero at reset.

    Inputs may be one joint vector or matching batches of joint vectors. Each
    row uses only its current and preceding 20 Hz control-boundary velocities;
    this is not MuJoCo's instantaneous ``qacc``. Materialisation computes the
    differences over the complete episode before history windows are sliced.
    """
    current = np.asarray(velocity, dtype=np.float64)
    if current.ndim < 1 or not 1 <= current.shape[-1] <= MAX_DOF:
        raise ValueError(f"joint velocity must end in 1..{MAX_DOF} joints")
    if not np.isfinite(current).all():
        raise ValueError("joint velocity contains non-finite values")
    if previous_velocity is None:
        return np.zeros_like(current)
    previous = np.asarray(previous_velocity, dtype=np.float64)
    if previous.shape != current.shape or not np.isfinite(previous).all():
        raise ValueError("previous joint velocity must be finite and match current shape")
    return (current - previous) * CONTROL_FREQ


def assemble_proprio(
    observation: dict[str, np.ndarray],
    *,
    tau_g: np.ndarray,
    wrist_ft: np.ndarray,
) -> np.ndarray:
    """Build the stored 70-vector from robosuite observations and extra signals.

    All inputs may carry arbitrary matching leading dimensions. Joint-indexed
    fields are padded to seven on their last axis and the final mask records the
    physical DoF. The native ``robot0_proprio-state`` key is intentionally not
    consumed: robosuite builds it at the robot's native width, so using it would
    make the concatenation order and cross-family padding implicit.
    """
    missing = [key for _, key in _NATIVE_OBS_KEYS if key not in observation]
    if missing:
        raise KeyError(f"cannot assemble proprioception; missing observation keys {missing}")

    tau = np.asarray(tau_g)
    wrist = np.asarray(wrist_ft)
    if tau.ndim < 1 or not 1 <= tau.shape[-1] <= MAX_DOF:
        raise ValueError(
            f"tau_g must end in 1..{MAX_DOF} arm joints, got {tau.shape}"
        )
    n_dof = int(tau.shape[-1])
    leading = tau.shape[:-1]
    if wrist.shape != (*leading, WRIST_FT_DIM):
        raise ValueError(
            f"wrist_ft must have shape {(*leading, WRIST_FT_DIM)}, got {wrist.shape}"
        )

    def fixed(name: str, key: str) -> np.ndarray:
        arr = np.asarray(observation[key])
        group = proprio_group(name)
        expected = n_dof if group.joint_indexed else group.width
        if arr.shape != (*leading, expected):
            raise ValueError(
                f"{key} must have shape {(*leading, expected)}, got {arr.shape}"
            )
        if not group.joint_indexed:
            return arr
        out = np.zeros((*leading, MAX_DOF), dtype=np.result_type(arr, np.float32))
        out[..., :n_dof] = arr
        return out

    parts = [fixed(name, key) for name, key in _NATIVE_OBS_KEYS]
    tau_padded = np.zeros((*leading, MAX_DOF), dtype=np.result_type(tau, np.float32))
    tau_padded[..., :n_dof] = tau
    mask = np.zeros((*leading, MAX_DOF), dtype=np.float32)
    mask[..., :n_dof] = 1.0
    parts.extend((tau_padded, wrist, mask))
    result = np.concatenate(parts, axis=-1).astype(np.float32, copy=False)
    if result.shape != (*leading, PROPRIO_DIM):
        raise AssertionError(
            f"assembled proprioception has shape {result.shape}, expected "
            f"{(*leading, PROPRIO_DIM)}"
        )
    return result


@dataclass(frozen=True)
class ObsSpec:
    """What the policy is allowed to look at.

    `use_proprio=False` is not a masking flag on an existing input: the encoder
    then has no proprioception pathway at all. B9 needs the intervention to be
    structural, because a zeroed input still lets the first layer's bias carry a
    per-embodiment offset (10.5, B9). `k_hist` keeps its value in that case, so
    the image history is untouched and B9 remains a single intervention.
    """

    cameras: tuple[str, ...] = CAMERAS
    image_size: int = IMAGE_SIZE
    k_hist: int = K_HIST_MAIN
    use_proprio: bool = True
    use_task_token: bool = True

    def __post_init__(self) -> None:
        if not self.cameras:
            raise ValueError(
                "10.4.0 fixes two RGB streams; an arm with no camera is not in the "
                "design (the proprio-only level of A1 drops images at the data "
                "layer, and still declares both camera names)"
            )
        if self.image_size != IMAGE_SIZE:
            raise ValueError(
                f"9.6.1 fixes {IMAGE_SIZE}x{IMAGE_SIZE} rendering; changing it changes "
                "the dataset, not the policy"
            )
        if self.k_hist not in K_HIST_LEVELS:
            raise ValueError(
                f"k_hist={self.k_hist} is not one of the A2 levels {K_HIST_LEVELS}"
            )

    @property
    def proprio_dim(self) -> int:
        return PROPRIO_DIM if self.use_proprio else 0

    @property
    def n_cameras(self) -> int:
        return len(self.cameras)
