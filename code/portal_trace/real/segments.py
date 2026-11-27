"""Motion-segment extraction and the 200 ms pre-latency window (proposal 10.4.6).

A motion segment starts at a velocity zero crossing. The pre-latency window is
the first 200 ms of that segment, and it is the primary reading of T_comp,
because human visuomotor latency is roughly 200-300 ms and a command issued
inside that window cannot contain a reaction to the lag of the segment it opens.
That is the whole identification argument of the E5 arm: 10.4.6 concedes that an
operator with no feedforward compensation at all will still command differently
on a laggy machine, purely by reacting to the lag they can see, and the
pre-window reading is what separates the feedforward component from that.

Because the argument rests on the window sitting exactly at the front of the
segment, both the zero-crossing rule and the window length come from the frozen
preregistration (`E5Preregistration`) and are never arguments with defaults.
Two knobs would otherwise be available: a velocity threshold that decides which
crossings count, and a minimum segment length that decides which segments
survive. Both change which frames enter the numerator and the denominator of
T_comp.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from portal_trace.real.pair import DroidLabId, E5Preregistration, PairedPoseSeries

__all__ = [
    "MotionSegment",
    "SegmentSet",
    "segment_by_zero_crossing",
]


@dataclass(frozen=True)
class MotionSegment:
    """One motion segment: [start, stop) in frames of its parent series.

    `pre_window_stop` is the end of the 200 ms pre-latency window, clipped to the
    segment. A segment shorter than the window keeps `pre_window_stop == stop`
    and is marked `pre_window_truncated`, so the truncation shows up in the
    reported frame counts rather than being silently padded or dropped.
    """

    lab: DroidLabId
    task_category: str
    episode_id: str
    start: int
    stop: int
    pre_window_stop: int
    axis: int
    direction: int  # +1 or -1, the sign of the velocity over the segment
    pre_window_truncated: bool

    def __post_init__(self) -> None:
        if not self.start < self.stop:
            raise ValueError(f"empty segment [{self.start}, {self.stop})")
        if not self.start < self.pre_window_stop <= self.stop:
            raise ValueError(
                f"pre-window [{self.start}, {self.pre_window_stop}) is not inside "
                f"segment [{self.start}, {self.stop})"
            )
        if self.direction not in (-1, 1):
            raise ValueError(f"direction must be +/-1, got {self.direction}")

    @property
    def n_frames(self) -> int:
        return self.stop - self.start

    @property
    def n_pre_window_frames(self) -> int:
        return self.pre_window_stop - self.start

    def slice_whole(self) -> slice:
        return slice(self.start, self.stop)

    def slice_pre_window(self) -> slice:
        return slice(self.start, self.pre_window_stop)


@dataclass(frozen=True)
class SegmentSet:
    """The segments of one paired series, with the series they index into."""

    series: PairedPoseSeries
    segments: tuple[MotionSegment, ...]
    prereg: E5Preregistration

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def n_truncated(self) -> int:
        return sum(1 for s in self.segments if s.pre_window_truncated)

    def stack(self, *, pre_window_only: bool) -> tuple[np.ndarray, np.ndarray]:
        """Concatenate the cmd and ach frames of every segment.

        Returns (cmd, ach), each (n_frames, D). `pre_window_only` selects the
        first 200 ms of each segment, which is the primary reading of 10.4.6;
        False gives the whole-segment reading.
        """
        if not self.segments:
            return (
                np.empty((0, self.series.cmd.shape[1])),
                np.empty((0, self.series.ach.shape[1])),
            )
        slices = [
            s.slice_pre_window() if pre_window_only else s.slice_whole()
            for s in self.segments
        ]
        cmd = np.concatenate([self.series.cmd[sl] for sl in slices], axis=0)
        ach = np.concatenate([self.series.ach[sl] for sl in slices], axis=0)
        return cmd, ach


def _sign_with_deadband(v: np.ndarray, threshold: float) -> np.ndarray:
    """Signum with a dead band, so sensor noise around zero is not a crossing.

    Without the dead band a series resting near zero produces a crossing every
    few frames, and each of those would open a 200 ms window; the pre-window
    reading would then be dominated by frames that are not the front of any real
    motion.
    """
    out = np.zeros(v.shape, dtype=np.int8)
    out[v > threshold] = 1
    out[v < -threshold] = -1
    return out


def segment_by_zero_crossing(
    series: PairedPoseSeries,
    *,
    axis: int,
) -> SegmentSet:
    """Split one axis of the achieved motion at velocity zero crossings (10.4.6).

    Segmentation runs on the **achieved** rate, not the commanded one. The
    commanded series is the quantity under test; deriving the segment boundaries
    from it would let a lab's commanding style change which frames the statistic
    sees, and the between-lab variance of the commands is exactly what T_comp
    measures.

    The rule, all three parts of it frozen in the preregistration:

    1. a frame is moving when |v| exceeds `zero_crossing_velocity_threshold`;
    2. a segment runs from a sign change (or from the start of motion) to the
       next sign change or to the end of motion;
    3. segments shorter than `zero_crossing_min_segment_s` are discarded.
    """
    prereg = series.prereg
    if not 0 <= axis < series.ach_rate.shape[1]:
        raise IndexError(
            f"axis {axis} is outside the {series.ach_rate.shape[1]} pose components"
        )

    v = series.ach_rate[:, axis]
    sign = _sign_with_deadband(v, prereg.zero_crossing_velocity_threshold)
    min_frames = max(int(np.ceil(prereg.zero_crossing_min_segment_s * prereg.sample_rate_hz)), 1)
    window_frames = prereg.pre_latency_window_frames

    segments: list[MotionSegment] = []
    start: int | None = None
    current = 0
    for i, s in enumerate(sign):
        if s == 0:
            if start is not None:
                _emit(segments, series, start, i, current, min_frames, window_frames, axis)
                start = None
                current = 0
            continue
        if start is None:
            start, current = i, int(s)
        elif s != current:
            _emit(segments, series, start, i, current, min_frames, window_frames, axis)
            start, current = i, int(s)
    if start is not None:
        _emit(segments, series, start, len(sign), current, min_frames, window_frames, axis)

    return SegmentSet(series=series, segments=tuple(segments), prereg=prereg)


def _emit(
    out: list[MotionSegment],
    series: PairedPoseSeries,
    start: int,
    stop: int,
    direction: int,
    min_frames: int,
    window_frames: int,
    axis: int,
) -> None:
    if stop - start < min_frames:
        return
    pre_stop = min(start + window_frames, stop)
    out.append(
        MotionSegment(
            lab=series.lab,
            task_category=series.task_category,
            episode_id=series.episode_id,
            start=start,
            stop=stop,
            pre_window_stop=pre_stop,
            axis=axis,
            direction=direction,
            pre_window_truncated=pre_stop < start + window_frames,
        )
    )
