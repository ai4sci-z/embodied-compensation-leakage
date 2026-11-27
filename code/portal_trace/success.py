"""One fixed success definition shared by teacher and policy rollouts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from portal_trace.registry.tasks import CONTROL_FREQ, HORIZON

__all__ = [
    "SUCCESS_HOLD_SECONDS",
    "SUCCESS_HOLD_STEPS",
    "terminal_hold_start",
]

# A task must remain solved through the final second of the fixed 20 s rollout.
# This rejects momentary tolerance crossings without changing episode length.
SUCCESS_HOLD_SECONDS = 1.0
SUCCESS_HOLD_STEPS = int(round(SUCCESS_HOLD_SECONDS * CONTROL_FREQ))

if SUCCESS_HOLD_STEPS != 20 or SUCCESS_HOLD_STEPS > HORIZON:
    raise AssertionError("the preregistered terminal hold must be 20 of 400 frames")


def terminal_hold_start(trace: Sequence[bool] | np.ndarray) -> int:
    """Return the start of the terminal success run, or -1 if it is too short."""
    values = np.asarray(trace, dtype=bool)
    if values.ndim != 1:
        raise ValueError(f"success trace must be one-dimensional, got {values.shape}")
    if len(values) < SUCCESS_HOLD_STEPS:
        raise ValueError(
            f"success trace has {len(values)} frames; terminal hold needs "
            f"{SUCCESS_HOLD_STEPS}"
        )
    if not bool(values[-SUCCESS_HOLD_STEPS:].all()):
        return -1
    false_frames = np.flatnonzero(~values)
    return int(false_frames[-1] + 1) if false_frames.size else 0
