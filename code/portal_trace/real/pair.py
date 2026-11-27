"""E5: cmd/ach pose pairing and time alignment for DROID (proposal 10.4.6).

E5 is the real-data observation arm. Its natural experiment is that DROID's 13
institutions share one nominal robot model, so morphology differences are zero
by construction and a per-lab difference in the execution signature cannot be
explained by morphology. That only works if the pairing is exactly the one
10.4.6 pins down:

    action_dict/cartesian_position   (commanded target pose)
    observation/cartesian_position   (achieved pose)

both **poses**. 10.4.6 explicitly refuses commanded velocity against a
differentiated achieved velocity: differentiation noise lands only in the
denominator and systematically depresses T_comp.

Why the filter and the difference format are not arguments with defaults
------------------------------------------------------------------------
T_comp is a ratio of between-lab variance components. Every one of the filter
cutoff, the filter order, the difference stencil, the zero-crossing rule and the
pre-latency window length moves both variances, and not by the same factor. So
each of them is a researcher degree of freedom that could be tuned until
T_comp > 1.3 on 8 of 13 labs, which is the preregistered prediction. 10.7.8 item
16 puts all of them in the frozen preregistration for that reason.

The way that is expressed here: `E5Preregistration.load(path)` reads them from
the preregistration file and every field is required. There is no keyword
argument with a default anywhere on this path, no `cfg.get(key, fallback)`, and
no shipped default configuration -- if the preregistration file is absent the
loader raises and names the 10.7.8 item that produces it. The same discipline
`portal_trace.lockfile` applies to measured constants, applied to frozen
analysis choices.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import signal

__all__ = [
    "CMD_FIELD",
    "ACH_FIELD",
    "DroidLabId",
    "DifferenceFormat",
    "FilterSpec",
    "E5Preregistration",
    "PreregistrationMissingError",
    "PairedPoseSeries",
    "pair_cmd_ach",
]

CMD_FIELD = "action_dict/cartesian_position"
"""The commanded target pose field (10.4.6). Not commanded velocity."""

ACH_FIELD = "observation/cartesian_position"
"""The achieved pose field (10.4.6). Not a differentiated achieved velocity."""

N_DROID_LABS = 13
"""10.4.6: the institution axis is defined *inside* DROID, 13 institutions.

OXE's Franka subset is not admissible as an institution axis: across datasets the
original control conventions (absolute/relative, position/velocity) are retained,
so a cross-dataset sigma^2_lab(cmd) mixes in action-space convention differences.
"""


class PreregistrationMissingError(FileNotFoundError):
    """The frozen E5 preregistration is not where the caller said it was."""

    def __init__(self, path: Path, missing_keys: tuple[str, ...] = ()) -> None:
        self.path = path
        self.missing_keys = missing_keys
        super().__init__(str(self))

    def __str__(self) -> str:
        lines = [
            f"the E5 preregistration is not readable at {self.path}",
            "  what it is : the frozen filter, difference format, zero-crossing rule and",
            "               pre-latency window length for the DROID arm",
            "  produced by: preregistration item 16 of 10.7.8, frozen with a git tag at",
            "               the end of W2 (11.3)",
            "  blocks     : every T_comp reading. All four of these move T_comp, so a",
            "               default would be a researcher degree of freedom inside the",
            "               statistic (10.4.6)",
        ]
        if self.missing_keys:
            lines.append(f"  missing    : {list(self.missing_keys)}")
        return "\n".join(lines)


class DroidLabId(str):
    """An institution identifier scoped to DROID.

    A distinct type because 10.4.6 forbids treating an OXE dataset as an
    institution; a bare string would let one be passed where a DROID lab is
    expected and the resulting sigma^2_lab would silently include action-space
    convention differences.
    """

    __slots__ = ()


class DifferenceFormat(Enum):
    """How a pose series is differenced. Frozen by preregistration.

    The three formats do not agree on the amount of high-frequency content they
    pass, so the choice is part of T_comp's definition rather than an
    implementation detail.
    """

    NONE = "none"
    BACKWARD = "backward"                  # x[t] - x[t-1]
    CENTRAL = "central"                    # (x[t+1] - x[t-1]) / 2
    SAVITZKY_GOLAY = "savitzky_golay"      # least-squares derivative over a window


@dataclass(frozen=True)
class FilterSpec:
    """A zero-phase Butterworth low pass, as frozen in the preregistration.

    Zero phase (filtfilt) rather than a causal filter: a causal filter shifts the
    command series in time, and the pre-latency window of 10.4.6 is defined by
    the time from a velocity zero crossing, so a phase shift there would move the
    window relative to the segment it is supposed to sit at the front of.
    """

    kind: str
    order: int
    cutoff_hz: float
    sample_rate_hz: float

    def __post_init__(self) -> None:
        if self.kind != "butterworth_zero_phase":
            raise ValueError(
                f"filter kind {self.kind!r} is not the preregistered one "
                "('butterworth_zero_phase'); changing the filter changes T_comp"
            )
        if self.order < 1:
            raise ValueError(f"filter order must be >= 1, got {self.order}")
        if not 0 < self.cutoff_hz < self.sample_rate_hz / 2:
            raise ValueError(
                f"cutoff {self.cutoff_hz} Hz must lie strictly inside "
                f"(0, {self.sample_rate_hz / 2}) Hz for a {self.sample_rate_hz} Hz series"
            )

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Filter along axis 0. Columns are pose components and stay independent."""
        sos = signal.butter(
            self.order, self.cutoff_hz, btype="low", fs=self.sample_rate_hz, output="sos"
        )
        return signal.sosfiltfilt(sos, x, axis=0)


@dataclass(frozen=True)
class E5Preregistration:
    """Item 16 of the 10.7.8 registration list, loaded rather than defaulted.

    `content_hash` is over the file as read, so a result can record which frozen
    version it used and a later edit is visible instead of silent.
    """

    filter_spec: FilterSpec
    difference_format: DifferenceFormat
    savgol_window: int | None
    savgol_polyorder: int | None
    zero_crossing_velocity_threshold: float
    zero_crossing_min_segment_s: float
    pre_latency_window_s: float
    reaction_regression_lag_s: float
    sample_rate_hz: float
    cmd_field: str
    ach_field: str
    source_path: Path
    content_hash: str
    git_tag: str

    REQUIRED_KEYS: tuple[str, ...] = (
        "filter",
        "difference_format",
        "zero_crossing",
        "pre_latency_window_s",
        "reaction_regression_lag_s",
        "sample_rate_hz",
        "cmd_field",
        "ach_field",
        "git_tag",
    )

    def __post_init__(self) -> None:
        if self.cmd_field != CMD_FIELD or self.ach_field != ACH_FIELD:
            raise ValueError(
                f"10.4.6 pins the pairing to {CMD_FIELD!r} against {ACH_FIELD!r}, both "
                f"poses; the preregistration names {self.cmd_field!r} against "
                f"{self.ach_field!r}. Pairing commanded velocity against a "
                "differentiated achieved velocity puts differentiation noise in the "
                "denominator and depresses T_comp"
            )
        if self.pre_latency_window_s <= 0:
            raise ValueError("the pre-latency window must be positive")
        if self.difference_format is DifferenceFormat.SAVITZKY_GOLAY and (
            self.savgol_window is None or self.savgol_polyorder is None
        ):
            raise ValueError(
                "difference_format savitzky_golay needs savgol_window and "
                "savgol_polyorder in the preregistration"
            )

    @property
    def pre_latency_window_frames(self) -> int:
        """The 200 ms window in frames, rounded up.

        10.4.6: human visuomotor latency is roughly 200-300 ms, so commands
        inside the window cannot contain a reaction to that segment's own lag.
        Rounding up keeps the window at least as long as preregistered; rounding
        down would quietly shorten it at low sample rates.
        """
        return int(np.ceil(self.pre_latency_window_s * self.sample_rate_hz))

    @property
    def reaction_regression_lags(self) -> int:
        """L, the number of achieved lags covering 300 ms (10.4.6, fourth reading)."""
        return int(np.ceil(self.reaction_regression_lag_s * self.sample_rate_hz))

    @classmethod
    def load(cls, path: Path | str) -> E5Preregistration:
        """Read the frozen configuration. Every field is required.

        There is no default configuration in this package and no per-key
        fallback: a missing key raises with the list of what is missing.
        """
        path = Path(path)
        if not path.exists():
            raise PreregistrationMissingError(path)
        raw = path.read_bytes()
        doc: dict[str, Any] = yaml.safe_load(raw.decode("utf-8"))
        if not isinstance(doc, dict):
            raise PreregistrationMissingError(path, ("<file is not a mapping>",))
        missing = tuple(k for k in cls.REQUIRED_KEYS if k not in doc)
        if missing:
            raise PreregistrationMissingError(path, missing)

        sample_rate = float(doc["sample_rate_hz"])
        f = doc["filter"]
        for key in ("kind", "order", "cutoff_hz"):
            if key not in f:
                raise PreregistrationMissingError(path, (f"filter.{key}",))
        zc = doc["zero_crossing"]
        for key in ("velocity_threshold", "min_segment_s"):
            if key not in zc:
                raise PreregistrationMissingError(path, (f"zero_crossing.{key}",))

        fmt = DifferenceFormat(doc["difference_format"])
        savgol = doc.get("savitzky_golay")
        if fmt is DifferenceFormat.SAVITZKY_GOLAY and savgol is None:
            raise PreregistrationMissingError(path, ("savitzky_golay",))

        return cls(
            filter_spec=FilterSpec(
                kind=str(f["kind"]),
                order=int(f["order"]),
                cutoff_hz=float(f["cutoff_hz"]),
                sample_rate_hz=sample_rate,
            ),
            difference_format=fmt,
            savgol_window=int(savgol["window"]) if savgol else None,
            savgol_polyorder=int(savgol["polyorder"]) if savgol else None,
            zero_crossing_velocity_threshold=float(zc["velocity_threshold"]),
            zero_crossing_min_segment_s=float(zc["min_segment_s"]),
            pre_latency_window_s=float(doc["pre_latency_window_s"]),
            reaction_regression_lag_s=float(doc["reaction_regression_lag_s"]),
            sample_rate_hz=sample_rate,
            cmd_field=str(doc["cmd_field"]),
            ach_field=str(doc["ach_field"]),
            source_path=path,
            content_hash=hashlib.sha256(raw).hexdigest(),
            git_tag=str(doc["git_tag"]),
        )

    def difference(self, x: np.ndarray) -> np.ndarray:
        """Apply the frozen difference format along axis 0, keeping the length."""
        if self.difference_format is DifferenceFormat.NONE:
            return x
        if self.difference_format is DifferenceFormat.BACKWARD:
            out = np.empty_like(x)
            out[0] = 0.0
            out[1:] = np.diff(x, axis=0)
            return out * self.sample_rate_hz
        if self.difference_format is DifferenceFormat.CENTRAL:
            out = np.zeros_like(x)
            out[1:-1] = (x[2:] - x[:-2]) / 2.0
            out[0] = x[1] - x[0]
            out[-1] = x[-1] - x[-2]
            return out * self.sample_rate_hz
        window = int(self.savgol_window or 0)
        poly = int(self.savgol_polyorder or 0)
        if x.shape[0] < window:
            raise ValueError(
                f"a {x.shape[0]}-frame series is shorter than the preregistered "
                f"Savitzky-Golay window of {window} frames"
            )
        return signal.savgol_filter(
            x, window_length=window, polyorder=poly, deriv=1,
            delta=1.0 / self.sample_rate_hz, axis=0,
        )


@dataclass(frozen=True)
class PairedPoseSeries:
    """One trajectory's cmd/ach pose pair, filtered, differenced and time aligned.

    `cmd` and `ach` are (T, 6) Cartesian poses (position and orientation), on a
    common time base. `residual` is cmd - ach, which 10.4.6 excludes as an
    endpoint but keeps as a diagnostic: it measures the plant's failure to
    execute, i.e. the defect delta *before* compensation, not the compensation.
    """

    lab: DroidLabId
    task_category: str
    episode_id: str
    time_s: np.ndarray
    cmd: np.ndarray
    ach: np.ndarray
    cmd_rate: np.ndarray
    ach_rate: np.ndarray
    prereg: E5Preregistration

    def __post_init__(self) -> None:
        t = self.time_s.shape[0]
        for name in ("cmd", "ach", "cmd_rate", "ach_rate"):
            arr = getattr(self, name)
            if arr.shape[0] != t:
                raise ValueError(f"{name} has {arr.shape[0]} frames, time has {t}")
            if arr.ndim != 2:
                raise ValueError(f"{name} must be (T, D), got shape {arr.shape}")

    @property
    def n_frames(self) -> int:
        return self.time_s.shape[0]

    @property
    def residual(self) -> np.ndarray:
        """cmd - ach. Diagnostic only: 10.4.6 excludes its variance ratio.

        "How much of the hardware defect is explained by which machine" is true
        of hardware parameters almost by definition, so the residual variance
        ratio measures delta, not c.
        """
        return self.cmd - self.ach


def _align_on_common_time(
    *,
    cmd_time: np.ndarray,
    cmd: np.ndarray,
    ach_time: np.ndarray,
    ach: np.ndarray,
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample both series onto one uniform grid over their overlap.

    Commanded and achieved poses are logged by different producers and their
    timestamps do not coincide. Interpolating both onto a common uniform grid --
    rather than interpolating one onto the other's stamps -- keeps the two sides
    symmetric: interpolating only the achieved side would smooth only the
    denominator of T_comp, which is the same asymmetry 10.4.6 rules out when it
    forbids differentiating the achieved series.
    """
    t0 = max(float(cmd_time[0]), float(ach_time[0]))
    t1 = min(float(cmd_time[-1]), float(ach_time[-1]))
    if not t1 > t0:
        raise ValueError(
            f"commanded [{cmd_time[0]}, {cmd_time[-1]}] and achieved "
            f"[{ach_time[0]}, {ach_time[-1]}] do not overlap"
        )
    n = int(np.floor((t1 - t0) * sample_rate_hz)) + 1
    grid = t0 + np.arange(n) / sample_rate_hz
    cmd_i = np.column_stack([np.interp(grid, cmd_time, cmd[:, j]) for j in range(cmd.shape[1])])
    ach_i = np.column_stack([np.interp(grid, ach_time, ach[:, j]) for j in range(ach.shape[1])])
    return grid, cmd_i, ach_i


def pair_cmd_ach(
    *,
    lab: DroidLabId,
    task_category: str,
    episode_id: str,
    cmd_time_s: np.ndarray,
    cmd_pose: np.ndarray,
    ach_time_s: np.ndarray,
    ach_pose: np.ndarray,
    prereg: E5Preregistration,
) -> PairedPoseSeries:
    """Pair, align, filter and difference one episode (10.4.6).

    Order matters and is fixed: align first (so both series live on the same
    grid), then filter (so the zero-phase filter sees a uniform sample rate), then
    difference (so the difference stencil is applied to filtered poses, which is
    what the preregistration describes). Filtering after differencing would give
    a different frequency response and therefore a different T_comp.

    Both sides are poses. The rate series returned here exist for the
    zero-crossing segmentation of `segments.py`, not as the paired quantity.
    """
    cmd_pose = np.asarray(cmd_pose, dtype=float)
    ach_pose = np.asarray(ach_pose, dtype=float)
    if cmd_pose.ndim != 2 or ach_pose.ndim != 2:
        raise ValueError(
            f"both sides must be (T, D) pose arrays; got {cmd_pose.shape} and {ach_pose.shape}"
        )
    if cmd_pose.shape[1] != ach_pose.shape[1]:
        raise ValueError(
            f"commanded pose has {cmd_pose.shape[1]} components, achieved has "
            f"{ach_pose.shape[1]}; 10.4.6 pairs pose against pose"
        )

    grid, cmd_i, ach_i = _align_on_common_time(
        cmd_time=np.asarray(cmd_time_s, dtype=float),
        cmd=cmd_pose,
        ach_time=np.asarray(ach_time_s, dtype=float),
        ach=ach_pose,
        sample_rate_hz=prereg.sample_rate_hz,
    )
    cmd_f = prereg.filter_spec.apply(cmd_i)
    ach_f = prereg.filter_spec.apply(ach_i)
    return PairedPoseSeries(
        lab=lab,
        task_category=task_category,
        episode_id=episode_id,
        time_s=grid,
        cmd=cmd_f,
        ach=ach_f,
        cmd_rate=prereg.difference(cmd_f),
        ach_rate=prereg.difference(ach_f),
        prereg=prereg,
    )
