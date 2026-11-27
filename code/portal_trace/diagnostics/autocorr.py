"""Residual autocorrelation estimates for effective-sample-size weights.

``tau_corr`` is a property of the evaluated-policy residual inside one
task/defect/regime cell.  This module deliberately does not accept arbitrary
frames or actions: callers must provide episode-blocked residuals so contacts
or episode resets cannot create a spurious long-lag correlation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["TauCorrEstimate", "estimate_tau_corr"]


@dataclass(frozen=True)
class TauCorrEstimate:
    """A scalar IACT estimate plus the immutable choices behind it."""

    tau_corr: float
    n_episodes: int
    n_frames: int
    window_lag: int
    positive_lags: int
    method: str = "episode_mean_multivariate_acf_initial_positive_sequence_v1"


def estimate_tau_corr(
    residual: np.ndarray,
    *,
    max_lag: int | None = None,
    min_episodes: int = 8,
    min_frames: int = 400,
) -> TauCorrEstimate:
    """Estimate IACT from ``(episode, time[, channel])`` residual blocks.

    Each episode is demeaned independently.  For a multi-channel residual the
    lag covariance is the channel-averaged inner product at that lag; this is a
    multivariate ACF, not an ACF of the per-frame Euclidean norm.  Lag
    covariances are pooled by the number of within-episode pairs.  The sum uses
    Geyer's initial-positive-sequence rule: adjacent lag pairs are added until
    the first non-positive pair, then the sum is truncated.  This prevents a
    noisy far-tail ACF from inflating effective sample counts and never crosses
    an episode reset.
    """
    values = np.asarray(residual, dtype=np.float64)
    if values.ndim == 2:
        values = values[..., None]
    if values.ndim != 3:
        raise ValueError("residual must have shape (episodes, time[, channels])")
    n_episodes, horizon, channels = values.shape
    if n_episodes < min_episodes:
        raise ValueError(
            f"tau_corr needs at least {min_episodes} complete episodes, got {n_episodes}"
        )
    if horizon < 2:
        raise ValueError("tau_corr needs at least two frames per episode")
    if min_frames < 1 or n_episodes * horizon < min_frames:
        raise ValueError(
            f"tau_corr needs at least {min_frames} frames, got {n_episodes * horizon}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("tau_corr residual contains NaN or infinity")
    if channels < 1:
        raise ValueError("tau_corr residual has no channels")

    lag_limit = horizon - 1 if max_lag is None else int(max_lag)
    if lag_limit < 1 or lag_limit >= horizon:
        raise ValueError(f"max_lag must be in 1..{horizon - 1}, got {max_lag!r}")

    centered = values - values.mean(axis=1, keepdims=True)
    variance = np.sum(centered * centered, axis=(1, 2)) / (horizon * channels)
    if np.any(variance <= 0.0):
        bad = np.flatnonzero(variance <= 0.0).tolist()
        raise ValueError(f"tau_corr residual has zero variance in episode(s) {bad}")

    # Pair-count-weighted mean autocovariance across complete episode blocks.
    acf = np.empty(lag_limit + 1, dtype=np.float64)
    acf[0] = 1.0
    for lag in range(1, lag_limit + 1):
        products = centered[:, :-lag, :] * centered[:, lag:, :]
        cov = np.sum(products, axis=(1, 2)) / ((horizon - lag) * channels)
        weights = np.full(n_episodes, horizon - lag, dtype=np.float64)
        acf[lag] = float(np.sum(weights * (cov / variance)) / np.sum(weights))

    tau = 1.0
    positive_lags = 0
    for lag in range(1, lag_limit + 1, 2):
        pair = acf[lag]
        if lag + 1 <= lag_limit:
            pair += acf[lag + 1]
        if pair <= 0.0:
            break
        tau += 2.0 * max(pair, 0.0)
        positive_lags = lag + (1 if lag + 1 <= lag_limit else 0)

    return TauCorrEstimate(
        tau_corr=float(max(tau, 1.0)),
        n_episodes=n_episodes,
        n_frames=n_episodes * horizon,
        window_lag=lag_limit,
        positive_lags=positive_lags,
    )
