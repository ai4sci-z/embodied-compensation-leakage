"""The eight surrogate defects and their four-dimensional matching (9.2.11).

The surrogate null distribution is the reference on the denominator side of the
excess leakage `lambda_exc = lambda_delta - E_{tilde delta}[lambda_tilde]`. The
question it answers is: what does Lambda read when a direction field that is
*indistinguishable from the true compensation in learnability, but is not
compensation*, is substituted for it? So each `c_tilde` has to align with the
matched true field along four measured dimensions, while being unambiguously not
the feedforward a teacher needs to overcome the defect.

The four dimensions and their preregistered tolerances are 9.2.11's table,
transcribed into `MATCH_TOLERANCES`. Two details of the metrics carry weight:

*Acting joints are matched as a concentration profile, not as an identity.* The
per-joint energy shares are sorted before they are compared, deliberately -- if
joint identity had to agree, then "move the offset from joint 3 to joint 5",
which is the cleanest surrogate in the set, would be excluded by construction.
Equal concentration is what makes the two fields equally hard to learn, which is
what confounder X9 needs controlled.

*Roughness is measured against arc length, not time.* `c_delta` is a function of
state, so walking the same path quickly and slowly has to give one spectrum. The
end-effector path is resampled onto 256 uniform arc-length bins per trajectory,
the DC bin is dropped (amplitude already covers it, and keeping it would make a
constant field look similar to everything), and spectra are compared with a
1-Wasserstein distance because frequency is ordered and KL gives infinity to
non-overlapping support.

Admission is all-or-nothing per cell: four out of four inside tolerance, or the
surrogate is struck from that cell and the null distribution's effective cardinality
drops. It may fall below 8 but not below `MIN_EFFECTIVE_SURROGATES` = 6; under
that, the cell's `lambda_exc` is exploratory only, and `SurrogateSet` refuses to
be read as confirmatory rather than leaving that to a convention.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from portal_trace.defects.delta import (
    DEFAULT_TARGET_JOINT_NUMBER,
    SURROGATE_SWAP_JOINT_NUMBER,
)
from portal_trace.defects.fields import apply_pinv
from portal_trace.types import REGIME_ORDINAL, Regime

__all__ = [
    "MatchTolerances",
    "MATCH_TOLERANCES",
    "MIN_EFFECTIVE_SURROGATES",
    "N_ARCLENGTH_BINS",
    "PHASE_SHIFT_FRACTION",
    "SurrogateContext",
    "MatchReport",
    "SurrogateResult",
    "SurrogateSet",
    "SURROGATES",
    "calibrate",
    "build_surrogate_set",
    "random_rotation_null",
    "energy",
    "regime_shares",
    "joint_energy_profile",
    "arclength_spectrum",
    "wasserstein1",
    "tolerance_table",
    "construction_hash",
]

# 9.2.11 step 3.
N_ARCLENGTH_BINS = 256
# tilde delta_5 shifts by 12% of the trajectory's total arc length.
PHASE_SHIFT_FRACTION = 0.12
# 9.2.11 admission rule, repeated in 14.5 E-11 and appendix D.
MIN_EFFECTIVE_SURROGATES = 6
MAX_CALIBRATION_ROUNDS = 10
JOINT_ROTATION_CANDIDATES = 512
CONTINUOUS_ACTION_DIMS = 6


@dataclass(frozen=True)
class MatchTolerances:
    """9.2.11's preregistered tolerance table.

    The proposal states the amplitude and regime tolerances twice. Section 9.2.11
    -- the detailed specification, and the one `surrogates.json` is written
    against -- gives +-10% on energy and 0.05 per regime; the summary at the E1
    design table quotes +-5% on energy. The detailed section wins here, and the
    discrepancy is a write-back.
    """

    energy_log_ratio: float = math.log(1.10)
    regime_l1: float = 0.10
    regime_per_regime: float = 0.05
    frequency_log_ratio: float = math.log(1.25)
    frequency_w1: float = 0.10
    joint_profile_l1: float = 0.10


MATCH_TOLERANCES = MatchTolerances()


@dataclass(frozen=True)
class SurrogateContext:
    """One (embodiment, defect, task) cell's frames, evaluated on rho^off.

    14.5 E-8 requires the surrogate matching, the discriminator's importance
    weights and the covariate-shift diagnostics to share one batch of rho^off
    frames; this object is that batch.
    """

    c_true: np.ndarray          # (N, A) the matched compensation field, action units
    episode_index: np.ndarray   # (N,)
    regime: np.ndarray          # (N,) uint8, Regime ordinal
    jacobian: np.ndarray        # (N, 6, J)
    qvel: np.ndarray            # (N, J)
    ee_pos: np.ndarray          # (N, 3)
    ee_vel: np.ndarray          # (N, 3)
    output_max: np.ndarray      # (6,) action unit in m and rad
    a_nom: np.ndarray | None = None   # (N, A) clean teacher action, for tilde delta_6
    phi: np.ndarray | None = None     # (N, P) state feature basis, for tilde delta_7

    def __post_init__(self) -> None:
        n = self.c_true.shape[0]
        for name in ("episode_index", "regime", "jacobian", "qvel", "ee_pos", "ee_vel"):
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(f"{name} has {len(arr)} frames, c_true has {n}")
        if self.jacobian.shape[1] != 6:
            raise ValueError(f"jacobian is {self.jacobian.shape}, expected (N, 6, J)")

    @property
    def n_frames(self) -> int:
        return self.c_true.shape[0]

    @property
    def action_dim(self) -> int:
        return self.c_true.shape[1]

    @property
    def n_joints(self) -> int:
        return self.jacobian.shape[2]

    def episodes(self) -> list[np.ndarray]:
        """Frame indices per episode, in first-appearance order."""
        _, first = np.unique(self.episode_index, return_index=True)
        return [
            np.flatnonzero(self.episode_index == self.episode_index[i])
            for i in np.sort(first)
        ]

    def require(self, name: str) -> np.ndarray:
        value = getattr(self, name)
        if value is None:
            raise ValueError(
                f"this surrogate needs {name}, which the cell did not supply. "
                "9.2.11 admits a surrogate only when all four dimensions can be "
                "measured; a missing input is a struck surrogate, not a defaulted one."
            )
        return value


# ---------------------------------------------------------------------------
# The four metrics
# ---------------------------------------------------------------------------


def energy(c: np.ndarray) -> float:
    """E = E_rho ||c||^2, in squared action units."""
    return float(np.mean(np.sum(np.asarray(c, dtype=np.float64) ** 2, axis=1)))


def regime_shares(c: np.ndarray, regime: np.ndarray) -> np.ndarray:
    """Per-regime energy share sh_r, in `Regime` ordinal order (9.7.1)."""
    power = np.sum(np.asarray(c, dtype=np.float64) ** 2, axis=1)
    total = float(power.sum())
    shares = np.zeros(len(REGIME_ORDINAL), dtype=np.float64)
    if total == 0.0:
        return shares
    for r in Regime:
        shares[REGIME_ORDINAL[r]] = float(power[regime == REGIME_ORDINAL[r]].sum()) / total
    return shares


def joint_energy_profile(c: np.ndarray, ctx: SurrogateContext) -> np.ndarray:
    """Descending-sorted joint energy shares p_j = E ||J^+ c||_j^2 / sum.

    The action-unit field is converted back to metres and radians before the
    pseudo-inverse: the two action units differ by a factor of ten, so a ratio
    taken in action units would not be the ratio 9.2.11 defines.
    """
    c = np.asarray(c, dtype=np.float64)
    task = c[:, :6] * ctx.output_max
    joint = apply_pinv(ctx.jacobian, task)
    power = np.mean(joint**2, axis=0)
    total = float(power.sum())
    if total == 0.0:
        return np.zeros_like(power)
    return np.sort(power / total)[::-1]


def _resample_arclength(values: np.ndarray, s: np.ndarray, n_bins: int) -> np.ndarray:
    grid = np.linspace(0.0, float(s[-1]), n_bins)
    return np.stack([np.interp(grid, s, values[:, k]) for k in range(values.shape[1])], axis=1)


def arclength_spectrum(
    c: np.ndarray, ctx: SurrogateContext, *, n_bins: int = N_ARCLENGTH_BINS
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """9.2.11 step 1-3: (nu, P_hat, nu_bar, nu_max), arc-length weighted over episodes.

    Returns the spectrum on the grid of the arc-length-weighted mean total path
    length, so that two fields measured on the same cell share one nu axis and
    `wasserstein1` compares like with like.
    """
    c = np.asarray(c, dtype=np.float64)
    spectra: list[np.ndarray] = []
    lengths: list[float] = []
    for rows in ctx.episodes():
        pos = ctx.ee_pos[rows]
        steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(steps)])
        total = float(s[-1])
        if total <= 0.0:
            continue
        resampled = _resample_arclength(c[rows], s, n_bins)
        spectrum = np.sum(np.abs(np.fft.rfft(resampled, axis=0)) ** 2, axis=1)
        # Drop k = 0: the DC component is the amplitude dimension, and keeping it
        # would make every field look spectrally like a constant one.
        spectrum = spectrum[1 : n_bins // 2 + 1]
        spectra.append(spectrum)
        lengths.append(total)

    if not spectra:
        raise ValueError(
            "no episode in this cell has positive end-effector path length; the "
            "arc-length spectrum of 9.2.11 is undefined on a stationary cell"
        )
    weights = np.asarray(lengths) / float(np.sum(lengths))
    power = np.einsum("e,ek->k", weights, np.stack(spectra))
    total_power = float(power.sum())
    p_hat = power / total_power if total_power > 0.0 else power
    mean_length = float(np.average(lengths, weights=weights))
    nu = np.arange(1, p_hat.size + 1, dtype=np.float64) / mean_length
    nu_bar = float(np.sum(nu * p_hat))
    nu_max = float(n_bins / (2.0 * mean_length))
    return nu, p_hat, nu_bar, nu_max


def wasserstein1(p: np.ndarray, q: np.ndarray, nu: np.ndarray) -> float:
    """W_1 between two normalised spectra on a shared, ordered nu grid."""
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return float(np.trapz(np.abs(cdf_p - cdf_q), nu))


# ---------------------------------------------------------------------------
# Match reporting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchReport:
    """Measured deviation on each of the four dimensions, plus the verdict."""

    energy_log_ratio: float
    regime_l1: float
    regime_max: float
    frequency_log_ratio: float
    frequency_w1_normalised: float
    joint_profile_l1: float
    tolerances: MatchTolerances = MATCH_TOLERANCES

    @property
    def per_dimension(self) -> dict[str, float]:
        """Deviation on each dimension as a fraction of its tolerance.

        Dimensionless so that the four are comparable in one table; a value at or
        below 1.0 is inside tolerance, and a dimension whose spec has two
        conditions reports the binding one.
        """
        t = self.tolerances
        return {
            "amplitude": abs(self.energy_log_ratio) / t.energy_log_ratio,
            "regime_support": max(
                self.regime_l1 / t.regime_l1, self.regime_max / t.regime_per_regime
            ),
            "spatial_frequency": max(
                abs(self.frequency_log_ratio) / t.frequency_log_ratio,
                self.frequency_w1_normalised / t.frequency_w1,
            ),
            "acting_joints": self.joint_profile_l1 / t.joint_profile_l1,
        }

    @property
    def amplitude_ok(self) -> bool:
        return abs(self.energy_log_ratio) <= self.tolerances.energy_log_ratio

    @property
    def regime_ok(self) -> bool:
        return (
            self.regime_l1 <= self.tolerances.regime_l1
            and self.regime_max <= self.tolerances.regime_per_regime
        )

    @property
    def frequency_ok(self) -> bool:
        return (
            abs(self.frequency_log_ratio) <= self.tolerances.frequency_log_ratio
            and self.frequency_w1_normalised <= self.tolerances.frequency_w1
        )

    @property
    def joints_ok(self) -> bool:
        return self.joint_profile_l1 <= self.tolerances.joint_profile_l1

    @property
    def admitted(self) -> bool:
        return self.amplitude_ok and self.regime_ok and self.frequency_ok and self.joints_ok

    def to_json(self) -> dict[str, Any]:
        return {
            "energy_log_ratio": self.energy_log_ratio,
            "regime_l1": self.regime_l1,
            "regime_max": self.regime_max,
            "frequency_log_ratio": self.frequency_log_ratio,
            "frequency_w1_normalised": self.frequency_w1_normalised,
            "joint_profile_l1": self.joint_profile_l1,
            "admitted": self.admitted,
        }


def measure(c_tilde: np.ndarray, ctx: SurrogateContext) -> MatchReport:
    """The four measured deviations between `c_tilde` and the cell's true field."""
    e_true = energy(ctx.c_true)
    e_tilde = energy(c_tilde)
    if e_true <= 0.0:
        raise ValueError(
            "the matched compensation field has zero energy on this cell; there is "
            "nothing for a surrogate to match and Lambda's denominator is zero"
        )
    log_ratio = math.log(e_tilde / e_true) if e_tilde > 0.0 else -math.inf

    sh_true = regime_shares(ctx.c_true, ctx.regime)
    sh_tilde = regime_shares(c_tilde, ctx.regime)

    nu, p_true, nu_bar_true, nu_max = arclength_spectrum(ctx.c_true, ctx)
    _, p_tilde, nu_bar_tilde, _ = arclength_spectrum(c_tilde, ctx)
    freq_log = (
        math.log(nu_bar_tilde / nu_bar_true)
        if nu_bar_true > 0.0 and nu_bar_tilde > 0.0
        else -math.inf
    )

    return MatchReport(
        energy_log_ratio=log_ratio,
        regime_l1=float(np.abs(sh_true - sh_tilde).sum()),
        regime_max=float(np.abs(sh_true - sh_tilde).max()),
        frequency_log_ratio=freq_log,
        frequency_w1_normalised=wasserstein1(p_true, p_tilde, nu) / nu_max,
        joint_profile_l1=float(
            np.abs(joint_energy_profile(ctx.c_true, ctx) - joint_energy_profile(c_tilde, ctx)).sum()
        ),
    )


# ---------------------------------------------------------------------------
# The eight closed forms
# ---------------------------------------------------------------------------


def _task_to_action(task: np.ndarray, ctx: SurrogateContext) -> np.ndarray:
    out = np.zeros((task.shape[0], ctx.action_dim), dtype=np.float64)
    out[:, :6] = task / ctx.output_max
    return out


def _joint_column(ctx: SurrogateContext, joint_number: int) -> np.ndarray:
    if not 1 <= joint_number <= ctx.n_joints:
        raise ValueError(
            f"joint{joint_number} does not exist on this {ctx.n_joints} DoF arm; "
            "9.2.11's default 3 -> 5 swap needs at least five joints"
        )
    return ctx.jacobian[:, :, joint_number - 1]


def _rotate_about(vectors: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation of (N, 3) rows about (N, 3) unit axes."""
    norm = np.linalg.norm(axis, axis=1, keepdims=True)
    unit = np.divide(axis, norm, out=np.zeros_like(axis), where=norm > 0.0)
    cos, sin = math.cos(angle), math.sin(angle)
    cross = np.cross(unit, vectors)
    dot = np.sum(unit * vectors, axis=1, keepdims=True)
    return vectors * cos + cross * sin + unit * dot * (1.0 - cos)


def surrogate_1(ctx: SurrogateContext, rng: np.random.Generator) -> np.ndarray:
    """Same concentration profile, different acting joint (matches D1)."""
    return _task_to_action(_joint_column(ctx, SURROGATE_SWAP_JOINT_NUMBER), ctx)


def surrogate_2(ctx: SurrogateContext, rng: np.random.Generator) -> np.ndarray:
    """Rotate the field 90 degrees about the instantaneous direction of motion."""
    c = ctx.c_true.copy()
    task = c[:, :6] * ctx.output_max
    rotated = np.empty_like(task)
    rotated[:, :3] = _rotate_about(task[:, :3], ctx.ee_vel, 0.5 * math.pi)
    rotated[:, 3:] = _rotate_about(task[:, 3:], ctx.ee_vel, 0.5 * math.pi)
    return _task_to_action(rotated, ctx)


def surrogate_3(ctx: SurrogateContext, rng: np.random.Generator) -> np.ndarray:
    """Anti-compensation: the sign flipped and nothing else."""
    return -ctx.c_true.copy()


def surrogate_4(ctx: SurrogateContext, rng: np.random.Generator) -> np.ndarray:
    """Sign from joint j', direction from joint j: breaks the memory/direction pairing."""
    sign = np.sign(ctx.qvel[:, SURROGATE_SWAP_JOINT_NUMBER - 1])[:, None]
    return _task_to_action(sign * _joint_column(ctx, DEFAULT_TARGET_JOINT_NUMBER), ctx)


def surrogate_5(ctx: SurrogateContext, rng: np.random.Generator) -> np.ndarray:
    """The true field shifted 12% of the arc length, periodically continued.

    Spectrum and regime shares survive almost intact; the alignment with the
    trajectory phase does not.
    """
    out = np.empty_like(ctx.c_true)
    for rows in ctx.episodes():
        pos = ctx.ee_pos[rows]
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])
        total = float(s[-1])
        if total <= 0.0:
            out[rows] = ctx.c_true[rows]
            continue
        shifted = np.mod(s - PHASE_SHIFT_FRACTION * total, total)
        source = np.searchsorted(s, shifted).clip(0, len(rows) - 1)
        out[rows] = ctx.c_true[rows][source]
    return out


def surrogate_6(ctx: SurrogateContext, rng: np.random.Generator) -> np.ndarray:
    """Second arc-length derivative of the nominal action, low-pass shaped.

    Breaks the order in velocity -- "act early" versus "feed acceleration
    forward" -- which is the thing D2's closed form actually asserts. Its raw
    spectrum is high by construction, so the taper below is part of the closed
    form rather than a repair: 9.2.11 states the shaping in the same line.
    """
    a_nom = ctx.require("a_nom")
    raw = np.zeros_like(ctx.c_true)
    for rows in ctx.episodes():
        pos = ctx.ee_pos[rows]
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])
        total = float(s[-1])
        if total <= 0.0 or len(rows) < 3:
            continue
        grid = np.linspace(0.0, total, N_ARCLENGTH_BINS)
        resampled = _resample_arclength(a_nom[rows], s, N_ARCLENGTH_BINS)
        second = np.gradient(np.gradient(resampled, grid, axis=0), grid, axis=0)
        back = np.stack(
            [np.interp(s, grid, second[:, k]) for k in range(second.shape[1])], axis=1
        )
        raw[rows] = back
    scale = np.linalg.norm(raw)
    if scale > 0.0:
        raw = raw / scale
    return _lowpass_to_match(raw, ctx)


def _lowpass_to_match(raw: np.ndarray, ctx: SurrogateContext) -> np.ndarray:
    """Bisect a first-order arc-length taper until nu_bar lands inside tolerance."""
    _, _, target, _ = arclength_spectrum(ctx.c_true, ctx)
    lo, hi = 1.0e-3, 1.0e4
    best = raw
    for _ in range(40):
        cutoff = math.sqrt(lo * hi)
        shaped = _apply_taper(raw, ctx, cutoff)
        _, _, nu_bar, _ = arclength_spectrum(shaped, ctx)
        best = shaped
        if nu_bar <= 0.0 or target <= 0.0:
            break
        gap = math.log(nu_bar / target)
        if abs(gap) <= MATCH_TOLERANCES.frequency_log_ratio:
            return shaped
        if gap > 0.0:
            hi = cutoff
        else:
            lo = cutoff
    return best


def _apply_taper(raw: np.ndarray, ctx: SurrogateContext, cutoff: float) -> np.ndarray:
    out = np.zeros_like(raw)
    for rows in ctx.episodes():
        pos = ctx.ee_pos[rows]
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])
        total = float(s[-1])
        if total <= 0.0 or len(rows) < 3:
            out[rows] = raw[rows]
            continue
        grid = np.linspace(0.0, total, N_ARCLENGTH_BINS)
        resampled = _resample_arclength(raw[rows], s, N_ARCLENGTH_BINS)
        spectrum = np.fft.rfft(resampled, axis=0)
        nu = np.arange(spectrum.shape[0], dtype=np.float64) / total
        spectrum *= np.exp(-nu / cutoff)[:, None]
        smoothed = np.fft.irfft(spectrum, n=N_ARCLENGTH_BINS, axis=0)
        out[rows] = np.stack(
            [np.interp(s, grid, smoothed[:, k]) for k in range(smoothed.shape[1])], axis=1
        )
    return out


def surrogate_7(ctx: SurrogateContext, rng: np.random.Generator) -> np.ndarray:
    """A Gaussian field in the column space of phi, projected off c_delta.

    Orthogonality to the true field holds by construction, so this is the
    surrogate that breaks every directional agreement at once.
    """
    phi = ctx.require("phi")
    xi = np.zeros_like(ctx.c_true)
    xi[:, :CONTINUOUS_ACTION_DIMS] = phi @ rng.standard_normal(
        (phi.shape[1], CONTINUOUS_ACTION_DIMS)
    )
    c = ctx.c_true
    denominator = float(np.sum(c * c))
    if denominator > 0.0:
        xi = xi - (float(np.sum(xi * c)) / denominator) * c
    return _lowpass_to_match(xi, ctx)


def surrogate_8(ctx: SurrogateContext, rng: np.random.Generator) -> np.ndarray:
    """A configuration-independent constant direction (matches D3b, D8).

    D3b's compensation moves with q and D8's does not; this one does not either,
    which is what makes it the negative control for configuration dependence.
    """
    u = np.zeros(ctx.action_dim, dtype=np.float64)
    u[:CONTINUOUS_ACTION_DIMS] = rng.standard_normal(CONTINUOUS_ACTION_DIMS)
    u = u / np.linalg.norm(u)
    return np.tile(u, (ctx.n_frames, 1))


SurrogateBuilder = Callable[[SurrogateContext, np.random.Generator], np.ndarray]

SURROGATES: dict[str, SurrogateBuilder] = {
    "s1_joint_swap": surrogate_1,
    "s2_normal_plane_rotation": surrogate_2,
    "s3_anti_compensation": surrogate_3,
    "s4_mismatched_memory": surrogate_4,
    "s5_arclength_phase_shift": surrogate_5,
    "s6_acceleration_feedforward": surrogate_6,
    "s7_orthogonal_random_field": surrogate_7,
    "s8_constant_direction": surrogate_8,
}

if len(SURROGATES) != 8:
    raise AssertionError("9.2.11 fixes the surrogate family at eight closed forms")


# ---------------------------------------------------------------------------
# Calibration and admission
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurrogateResult:
    """One surrogate on one cell after calibration."""

    name: str
    field: np.ndarray
    report: MatchReport
    beta: float
    rounds: int
    joint_rotation_trials: int = 0

    @property
    def admitted(self) -> bool:
        return self.report.admitted


def calibrate(
    name: str,
    raw: np.ndarray,
    ctx: SurrogateContext,
    *,
    max_rounds: int = MAX_CALIBRATION_ROUNDS,
) -> SurrogateResult:
    """9.2.11's calibration loop: amplitude in closed form, regime shares by iteration.

    beta = sqrt(E / E_tilde) matches amplitude exactly in one shot. If the regime
    shares are then out of tolerance a diagonal per-regime rescale is applied and
    the amplitude is re-solved, up to ten rounds; the two steps do not commute,
    which is why the loop exists rather than a single solve.
    """
    target_energy = energy(ctx.c_true)
    target_shares = regime_shares(ctx.c_true, ctx.regime)
    c = np.array(raw, dtype=np.float64, copy=True)
    total_beta = 1.0
    report = None
    rounds = 0

    for round_index in range(1, max_rounds + 1):
        rounds = round_index
        e = energy(c)
        if e <= 0.0:
            raise ValueError(
                f"surrogate {name} came out identically zero on this cell; its "
                "closed form has no support here and it is struck rather than scaled"
            )
        beta = math.sqrt(target_energy / e)
        c *= beta
        total_beta *= beta
        report = measure(c, ctx)
        if report.regime_ok:
            break
        shares = regime_shares(c, ctx.regime)
        weights = np.ones_like(shares)
        active = shares > 0.0
        weights[active] = np.sqrt(target_shares[active] / shares[active])
        c = c * weights[ctx.regime][:, None]

    assert report is not None
    return SurrogateResult(name=name, field=c, report=report, beta=total_beta, rounds=rounds)


def _orthogonal_matrix(rng: np.random.Generator, dimension: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def _joint_rotation_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{name}\0joint-profile".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _match_joint_profile_by_rotation(
    result: SurrogateResult,
    ctx: SurrogateContext,
    *,
    seed: int,
    n_candidates: int = JOINT_ROTATION_CANDIDATES,
) -> SurrogateResult:
    """Match joint concentration without changing energy or temporal spectrum.

    A fixed block-orthogonal map is applied in normalized task-action space.
    Orthogonality preserves every frame norm, so amplitude, regime shares, and
    the total arc-length spectrum remain unchanged. Separate translation and
    rotation blocks avoid mixing their physical units.
    """
    if result.report.joints_ok or result.name == "s3_anti_compensation":
        return result
    if ctx.action_dim < CONTINUOUS_ACTION_DIMS:
        raise ValueError("joint-profile rotation requires six continuous action dims")
    if isinstance(n_candidates, bool) or not isinstance(n_candidates, int) or n_candidates < 1:
        raise ValueError("joint-profile rotation candidate count must be positive")

    rng = np.random.default_rng(_joint_rotation_seed(seed, result.name))
    best = result.field
    best_report = result.report
    best_score = result.report.joint_profile_l1
    trials = 0
    for _ in range(n_candidates):
        trials += 1
        candidate = result.field.copy()
        candidate[:, :3] = result.field[:, :3] @ _orthogonal_matrix(rng, 3).T
        candidate[:, 3:6] = result.field[:, 3:6] @ _orthogonal_matrix(rng, 3).T
        report = measure(candidate, ctx)
        if report.joint_profile_l1 < best_score:
            best = candidate
            best_report = report
            best_score = report.joint_profile_l1
        if best_report.joints_ok:
            break
    return SurrogateResult(
        name=result.name,
        field=best,
        report=best_report,
        beta=result.beta,
        rounds=result.rounds,
        joint_rotation_trials=trials,
    )


@dataclass
class SurrogateSet:
    """The null distribution for one cell, with its effective cardinality rule."""

    cell: str
    results: dict[str, SurrogateResult] = field(default_factory=dict)

    @property
    def admitted(self) -> tuple[str, ...]:
        return tuple(name for name, r in sorted(self.results.items()) if r.admitted)

    @property
    def struck(self) -> tuple[str, ...]:
        return tuple(name for name, r in sorted(self.results.items()) if not r.admitted)

    @property
    def effective_count(self) -> int:
        return len(self.admitted)

    @property
    def is_exploratory_only(self) -> bool:
        """9.2.11: below six admitted surrogates, lambda_exc is exploratory."""
        return self.effective_count < MIN_EFFECTIVE_SURROGATES

    def require_confirmatory(self) -> None:
        if self.is_exploratory_only:
            raise ValueError(
                f"cell {self.cell} admitted {self.effective_count} surrogates "
                f"({', '.join(self.admitted) or 'none'}); 9.2.11 allows the effective "
                f"cardinality to fall below eight but not below "
                f"{MIN_EFFECTIVE_SURROGATES}. Its lambda_exc may be reported as "
                f"exploratory and may not enter a confirmatory endpoint. Struck: "
                f"{', '.join(self.struck)}"
            )

    def null_fields(self) -> dict[str, np.ndarray]:
        return {name: self.results[name].field for name in self.admitted}

    def to_json(self) -> dict[str, Any]:
        return {
            "cell": self.cell,
            "effective_count": self.effective_count,
            "exploratory_only": self.is_exploratory_only,
            "construction_hash": construction_hash(),
            "surrogates": {
                name: {
                    "beta": r.beta,
                    "rounds": r.rounds,
                    "joint_rotation_trials": r.joint_rotation_trials,
                    **r.report.to_json(),
                }
                for name, r in sorted(self.results.items())
            },
        }


def build_surrogate_set(ctx: SurrogateContext, *, cell: str, seed: int) -> SurrogateSet:
    """Generate, calibrate and admit all eight surrogates on one cell.

    `seed` makes the two stochastic constructions (the random field of
    tilde delta_7 and the constant direction of tilde delta_8) reproducible; the
    other six are deterministic functions of the cell.
    """
    rng = np.random.default_rng(seed)
    out = SurrogateSet(cell=cell)
    for name, builder in SURROGATES.items():
        calibrated = calibrate(name, builder(ctx, rng), ctx)
        out.results[name] = _match_joint_profile_by_rotation(
            calibrated, ctx, seed=seed
        )
    return out


def random_rotation_null(
    c: np.ndarray, rng: np.random.Generator, *, n_repeats: int = 1
) -> list[np.ndarray]:
    """The zero-cost first screen: one Haar-random orthogonal map applied at every state.

    It preserves the norm and the state alignment and breaks only the direction.
    8.x is explicit that it is a screen and not the primary criterion: when
    `c_delta` is concentrated in a low-dimensional part of the action space --
    backlash on one joint, droop mostly in z -- a random rotation lands it where
    the policy error is nearly zero and the null distribution comes out
    optimistic.
    """
    c = np.asarray(c, dtype=np.float64)
    dim = c.shape[1]
    out = []
    for _ in range(n_repeats):
        q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
        q = q * np.sign(np.diag(r))
        out.append(c @ q.T)
    return out


def tolerance_table(sets: dict[str, SurrogateSet]) -> dict[str, dict[str, dict[str, float]]]:
    """The 8 x 4 measured-deviation table 9.2.11 requires in the results section.

    Median and maximum over the cells, per surrogate and per dimension. The
    proposal expects tilde delta_6 and tilde delta_8 to be struck most often --
    the first is naturally high frequency, the second naturally flat across
    regimes -- and this is the table that says whether that held.
    """
    dimensions = ("amplitude", "regime_support", "spatial_frequency", "acting_joints")
    table: dict[str, dict[str, dict[str, float]]] = {}
    for name in SURROGATES:
        per_dim: dict[str, dict[str, float]] = {}
        for dim in dimensions:
            values = [
                s.results[name].report.per_dimension[dim]
                for s in sets.values()
                if name in s.results
            ]
            per_dim[dim] = {
                "median": float(np.median(values)) if values else math.nan,
                "max": float(np.max(values)) if values else math.nan,
                "n_cells": len(values),
            }
        table[name] = per_dim
    return table


def construction_hash() -> str:
    """SHA-256 of this module's source (9.2.11).

    Frozen alongside `registry/lock.json` and entered in the preregistration:
    any change to a closed form, a tolerance or the calibration loop is a new
    preregistration, and this is what makes that checkable rather than asserted.
    """
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
