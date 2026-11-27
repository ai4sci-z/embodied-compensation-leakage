"""Robosuite observation/action representation boundary for deployment.

The simulator exposes one-frame dictionaries.  Policies consume a causal
five-frame ``Observation`` with an assembled 70-channel proprioceptive vector;
this adapter owns only that representation conversion.  Gravity torque and
wrist force/torque are required from an explicit provider because neither can
be reconstructed safely from the observable dictionary alone.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from portal_trace.policies.interface import Observation
from portal_trace.policies.obs import (
    ACTION_DIM,
    CAMERAS,
    CHUNK,
    IMAGE_SIZE,
    K_HIST_MAIN,
    PROPRIO_ACCELERATION_CAUSAL,
    PROPRIO_ACCELERATION_LEGACY,
    control_step_joint_acceleration,
)
from portal_trace.teacher.osc import env_action

__all__ = ["RobosuiteAuxProvider", "RobosuiteObservationAdapter"]


@runtime_checkable
class RobosuiteAuxProvider(Protocol):
    """Provide the non-observable proprioception channels per raw frame."""

    def __call__(self, raw_observation: Mapping[str, Any], *, t: int) -> tuple[Any, Any]: ...


class RobosuiteObservationAdapter:
    """Build policy inputs from a causal history of robosuite observations."""

    def __init__(
        self,
        *,
        env: Any,
        aux_provider: RobosuiteAuxProvider,
        task_index: int,
        embodiment_index: int,
        k_hist: int = K_HIST_MAIN,
        acceleration_semantics: str = PROPRIO_ACCELERATION_LEGACY,
    ) -> None:
        if not isinstance(aux_provider, RobosuiteAuxProvider):
            raise TypeError("aux_provider must provide tau_g and wrist_ft per frame")
        if k_hist < 1:
            raise ValueError("k_hist must be positive")
        if acceleration_semantics not in {
            PROPRIO_ACCELERATION_LEGACY,
            PROPRIO_ACCELERATION_CAUSAL,
        }:
            raise ValueError(f"unsupported acceleration semantics: {acceleration_semantics!r}")
        for name, value in (("task_index", task_index), ("embodiment_index", embodiment_index)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        self.env = env
        self.aux_provider = aux_provider
        self.task_index = task_index
        self.embodiment_index = embodiment_index
        self.k_hist = int(k_hist)
        self.acceleration_semantics = acceleration_semantics
        self._previous_velocity: np.ndarray | None = None
        self._frames: deque[Mapping[str, Any]] = deque(maxlen=self.k_hist)
        self._tau: deque[np.ndarray] = deque(maxlen=self.k_hist)
        self._wrist: deque[np.ndarray] = deque(maxlen=self.k_hist)
        self._last_t: int | None = None

    def reset(self) -> None:
        self._frames.clear()
        self._tau.clear()
        self._wrist.clear()
        self._last_t = None
        self._previous_velocity = None

    def to_observation(self, raw_observation: Mapping[str, Any], *, t: int) -> Observation:
        """Convert one raw frame, left-padding history by repeating frame zero."""
        if not isinstance(t, int) or isinstance(t, bool) or t < 0:
            raise ValueError("observation time index must be a non-negative integer")
        if t == 0:
            self.reset()
        elif self._last_t is None or t != self._last_t + 1:
            raise ValueError("raw observation time indices must start at zero and increase by one")
        if not isinstance(raw_observation, Mapping):
            raise TypeError("robosuite raw observation must be a mapping")
        self._validate_images(raw_observation)
        if self.acceleration_semantics == PROPRIO_ACCELERATION_CAUSAL:
            velocity = np.asarray(raw_observation["robot0_joint_vel"])
            # Copy only the mapping we change; keep the caller's observation
            # intact and retain the preceding velocity across history eviction.
            frame = dict(raw_observation)
            frame["robot0_joint_acc"] = control_step_joint_acceleration(
                velocity, self._previous_velocity
            )
            self._previous_velocity = velocity.copy()
            raw_observation = frame
        tau, wrist = self.aux_provider(raw_observation, t=t)
        tau_arr = np.asarray(tau)
        wrist_arr = np.asarray(wrist)
        self._frames.append(raw_observation)
        self._tau.append(tau_arr)
        self._wrist.append(wrist_arr)
        self._last_t = t

        frames = list(self._frames)
        taus = list(self._tau)
        wrists = list(self._wrist)
        if len(frames) < self.k_hist:
            n_pad = self.k_hist - len(frames)
            frames = [frames[0]] * n_pad + frames
            taus = [taus[0]] * n_pad + taus
            wrists = [wrists[0]] * n_pad + wrists

        # robomimic's EnvRobosuite.get_observation flips MuJoCo RGB frames on
        # the height axis before materialisation. Deployment must apply the
        # same transform or the policy sees vertically inverted images relative
        # to every training window.
        images = {
            camera: torch.from_numpy(
                np.stack([np.asarray(frame[camera])[::-1] for frame in frames], axis=0)
                .transpose(0, 3, 1, 2)
                .copy()
            ).to(torch.float32)
            / 255.0
            for camera in CAMERAS
        }
        images = {camera: value.unsqueeze(0) for camera, value in images.items()}
        native = {
            key: np.stack([np.asarray(frame[key]) for frame in frames], axis=0)
            for key in (
                "robot0_joint_pos",
                "robot0_joint_pos_cos",
                "robot0_joint_pos_sin",
                "robot0_joint_vel",
                "robot0_joint_acc",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_eef_quat_site",
                "robot0_gripper_qpos",
                "robot0_gripper_qvel",
            )
        }
        from portal_trace.policies.obs import assemble_proprio

        proprio = torch.from_numpy(
            assemble_proprio(native, tau_g=np.stack(taus), wrist_ft=np.stack(wrists))
        )
        return Observation(
            images=images,
            proprio=proprio.unsqueeze(0),
            task_index=torch.tensor([self.task_index], dtype=torch.long),
            embodiment_index=torch.tensor([self.embodiment_index], dtype=torch.long),
        )

    @staticmethod
    def _validate_images(raw: Mapping[str, Any]) -> None:
        for camera in CAMERAS:
            image = np.asarray(raw.get(camera))
            if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or image.dtype != np.uint8:
                raise ValueError(
                    f"{camera} must be HWC uint8 ({IMAGE_SIZE}, {IMAGE_SIZE}, 3), got "
                    f"{image.shape} {image.dtype}"
                )

    def to_action(self, action_chunk: np.ndarray, *, t: int) -> np.ndarray:
        """Map one predicted canonical chunk to the environment action space.

        ``env_action`` is the repository's sole environment-dimension mapping;
        no metres/radians conversion is duplicated here.  The first chunk frame
        is clipped with ``OSCTracker.clip_action`` semantics before mapping.
        """
        del t
        chunk = np.asarray(action_chunk, dtype=np.float64)
        if chunk.shape[-2:] != (CHUNK, ACTION_DIM) or chunk.ndim != 3:
            raise ValueError(f"action chunk must have shape (B, {CHUNK}, {ACTION_DIM})")
        if not np.isfinite(chunk).all():
            raise ValueError("action chunk contains non-finite values")
        canonical = np.clip(chunk[:, 0, :], -1.0, 1.0)
        return np.asarray(env_action(self.env, canonical), dtype=np.float64)
