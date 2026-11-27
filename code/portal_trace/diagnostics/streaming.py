"""Streaming sufficient statistics for every diagnostic in this package.

Proposal 11.3 fixes the data volume at roughly 45 M frames and 11.2 requires the
diagnostics to be computed from sufficient statistics over `FrameTable` chunks
rather than from a materialised frame matrix. That is not a performance
preference: at 45 M frames a single float64 column is 360 MB, and the Lambda
family needs a dozen of them at seven action components each.

Everything downstream (`leakage`, `excess`, `common_mode`, `two_way_fit`,
`regimes`) is written against the accumulators here, so no module has to know
how the frames were partitioned into chunks.

Numerical discipline
--------------------
Two rules, both of which matter at 10^7-10^8 terms:

* Within a chunk, numpy's pairwise summation is used (error grows like
  log n rather than n) and the cross-products go through BLAS.
* Across chunks, every running total is Neumaier-compensated. A naive running
  total of 45 M positive terms loses roughly log2(4.5e7) ~ 26 bits of the 53
  available; compensation puts that back for the price of one extra array.

Variances are never obtained as E[x^2] - E[x]^2. `WelfordAccumulator` carries
the second central moment directly and merges chunks with the Chan et al.
parallel formula, and `StreamingLeastSquares` keeps an upper-triangular R
factor instead of a normal-equation Gram, so the regression never squares the
condition number of the interference basis.

Passes over the data
--------------------
One pass fills `L2GramAccumulator`, `StreamingLeastSquares`,
`WelfordAccumulator` and `RegimeOccupancyAccumulator` simultaneously. A second
pass is needed only where a quantity is defined against a statistic of the
first pass -- the shrinkage field needs the pooled mean action a_bar (6.4
layer 2), and the twin-paired caliber of `two_way_fit` needs the clean-arm cell
means. Nothing in this package needs three.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg.lapack import dgeqrf

from portal_trace.diagnostics.units import Amplitude, EffectiveSampleCount, Energy
from portal_trace.types import REGIME_BY_ORDINAL, REGIME_ORDINAL, Cell, FrameTable, Regime

__all__ = [
    "NeumaierSum",
    "WelfordAccumulator",
    "L2GramAccumulator",
    "StreamingLeastSquares",
    "LeastSquaresSolution",
    "RankDeficientDesign",
    "RegimeOccupancyAccumulator",
    "CellRegimeMoments",
    "CellRegimeKey",
    "iter_regime_masks",
    "smoothing_field",
    "effective_sample_count",
]

# Chunks wider than this make the transient stack in `L2GramAccumulator` larger
# than a few hundred MB (m fields x n frames x d_a components x 8 bytes). It is
# a guidance number reported in the exception, not a silent clamp.
_CHUNK_ROWS_ADVISORY = 200_000


class RankDeficientDesign(ValueError):
    """A least-squares design matrix is numerically rank deficient.

    Raised rather than falling back to a pseudo-inverse: proposal 11.3 makes
    Lambda^part the primary endpoint, and a pseudo-inverse solution would
    silently report the minimum-norm coefficient of a collinear basis as if it
    were the partial regression coefficient.
    """


def _as_2d(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be (n,) or (n, d), got shape {arr.shape}")
    return arr


class NeumaierSum:
    """Compensated running total, scalar or elementwise over a fixed shape.

    `add` takes one already-reduced term (typically the pairwise sum of a whole
    chunk), so the compensation cost is paid once per chunk rather than once
    per frame.
    """

    __slots__ = ("_total", "_compensation", "_shape")

    def __init__(self, shape: tuple[int, ...] = ()) -> None:
        self._shape = shape
        self._total = np.zeros(shape, dtype=np.float64)
        self._compensation = np.zeros(shape, dtype=np.float64)

    def add(self, term: np.ndarray | float) -> None:
        t = np.asarray(term, dtype=np.float64)
        if t.shape != self._shape:
            raise ValueError(f"expected a term of shape {self._shape}, got {t.shape}")
        total = self._total + t
        # Neumaier: which operand is larger decides where the lost low bits are.
        big_total = np.abs(self._total) >= np.abs(t)
        self._compensation += np.where(
            big_total, (self._total - total) + t, (t - total) + self._total
        )
        self._total = total

    @property
    def value(self) -> np.ndarray:
        return self._total + self._compensation

    def merge(self, other: NeumaierSum) -> None:
        if other._shape != self._shape:
            raise ValueError(f"cannot merge shape {other._shape} into {self._shape}")
        self.add(other.value)

    def __repr__(self) -> str:
        return f"NeumaierSum(shape={self._shape}, value={self.value!r})"


class WelfordAccumulator:
    """Per-column mean and second central moment, merged chunk by chunk.

    Uses the Chan/Golub/LeVeque parallel form rather than the frame-by-frame
    recurrence, so a chunk is reduced with vectorised numpy and only the merge
    is scalar work. Weights are frame weights (all-ones unless the caller is
    reweighting by occupancy mass).
    """

    __slots__ = ("_width", "_count", "_mean", "_m2")

    def __init__(self, width: int) -> None:
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        self._width = width
        self._count = 0.0
        self._mean = np.zeros(width, dtype=np.float64)
        self._m2 = np.zeros(width, dtype=np.float64)

    @property
    def width(self) -> int:
        return self._width

    @property
    def count(self) -> float:
        return self._count

    def update(self, x: np.ndarray, weights: np.ndarray | None = None) -> None:
        arr = _as_2d(x, "x")
        if arr.shape[1] != self._width:
            raise ValueError(f"expected {self._width} columns, got {arr.shape[1]}")
        if arr.shape[0] == 0:
            return
        if weights is None:
            n = float(arr.shape[0])
            mean = arr.mean(axis=0)
            centred = arr - mean
            m2 = np.einsum("nd,nd->d", centred, centred)
        else:
            w = np.asarray(weights, dtype=np.float64)
            if w.shape != (arr.shape[0],):
                raise ValueError(f"weights must be ({arr.shape[0]},), got {w.shape}")
            if np.any(w < 0):
                raise ValueError("frame weights must be non-negative")
            n = float(w.sum())
            if n == 0.0:
                return
            mean = (w[:, None] * arr).sum(axis=0) / n
            centred = arr - mean
            m2 = np.einsum("n,nd,nd->d", w, centred, centred)
        self._merge_moments(n, mean, m2)

    def merge(self, other: WelfordAccumulator) -> None:
        if other._width != self._width:
            raise ValueError(f"cannot merge width {other._width} into {self._width}")
        if other._count == 0.0:
            return
        self._merge_moments(other._count, other._mean, other._m2)

    def _merge_moments(self, n_b: float, mean_b: np.ndarray, m2_b: np.ndarray) -> None:
        n_a = self._count
        if n_a == 0.0:
            self._count = n_b
            self._mean = np.array(mean_b, dtype=np.float64, copy=True)
            self._m2 = np.array(m2_b, dtype=np.float64, copy=True)
            return
        total = n_a + n_b
        delta = mean_b - self._mean
        self._mean = self._mean + delta * (n_b / total)
        self._m2 = self._m2 + m2_b + np.square(delta) * (n_a * n_b / total)
        self._count = total

    @property
    def mean(self) -> np.ndarray:
        if self._count == 0.0:
            raise ValueError("no frames accumulated; a mean over nothing is not a number")
        return self._mean.copy()

    @property
    def variance(self) -> np.ndarray:
        """Unbiased per-column variance."""
        if self._count < 2.0:
            raise ValueError(
                f"variance needs at least two effective frames, have {self._count}"
            )
        return self._m2 / (self._count - 1.0)

    @property
    def total_variance(self) -> float:
        """sigma^2 in the isotropic sense of proposition 3: mean over components."""
        return float(self.variance.mean())

    def __repr__(self) -> str:
        return f"WelfordAccumulator(width={self._width}, count={self._count:g})"


class L2GramAccumulator:
    """Gram matrix of named vector fields in L^2(rho), partitioned by regime.

    The entry (i, j) is E_rho<f_i(x), f_j(x)> -- the expectation is taken over
    frames first and the ratio second, which is the only reading under which
    Phi = rho_dir^2 holds (proposal 6.3). Every scalar the Lambda family needs
    is one entry of this matrix, so one pass over the frames serves
    Lambda^ref, Lambda^orc, Phi, rho_dir, the surrogate readings of
    `excess`, and the common-mode readings of `common_mode`.

    Regimes are kept apart because 10.4.7 weights numerator and denominator by
    occupancy mass separately; a pooled Gram cannot be reweighted afterwards.
    """

    def __init__(self, names: Sequence[str]) -> None:
        names = tuple(names)
        if len(set(names)) != len(names):
            raise ValueError(f"field names must be unique, got {names}")
        if not names:
            raise ValueError("at least one field is required")
        self._names = names
        self._index = {name: i for i, name in enumerate(names)}
        m = len(names)
        self._sums: dict[Regime, NeumaierSum] = {r: NeumaierSum((m, m)) for r in Regime}
        self._counts: dict[Regime, int] = dict.fromkeys(Regime, 0)
        self._width: int | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def action_width(self) -> int:
        if self._width is None:
            raise ValueError("no frames accumulated yet")
        return self._width

    def update(self, fields: Mapping[str, np.ndarray], regime: np.ndarray) -> None:
        """Accumulate one chunk. `fields` must carry exactly the declared names."""
        missing = set(self._names) - set(fields)
        extra = set(fields) - set(self._names)
        if missing or extra:
            raise ValueError(
                f"fields must be exactly {sorted(self._names)}; "
                f"missing={sorted(missing)} unexpected={sorted(extra)}"
            )
        first = _as_2d(fields[self._names[0]], self._names[0])
        n, d = first.shape
        if self._width is None:
            self._width = d
        elif self._width != d:
            raise ValueError(f"action width changed from {self._width} to {d} mid-stream")
        if n == 0:
            return
        if n > _CHUNK_ROWS_ADVISORY:
            raise ValueError(
                f"chunk has {n} rows; the transient stack would be "
                f"{len(self._names) * n * d * 8 / 1e6:.0f} MB. Split the FrameTable into "
                f"chunks of at most {_CHUNK_ROWS_ADVISORY} rows"
            )
        reg = np.asarray(regime)
        if reg.shape != (n,):
            raise ValueError(f"regime must be ({n},), got {reg.shape}")

        stack = np.empty((len(self._names), n, d), dtype=np.float64)
        for i, name in enumerate(self._names):
            arr = _as_2d(fields[name], name)
            if arr.shape != (n, d):
                raise ValueError(
                    f"field {name!r} has shape {arr.shape}, expected {(n, d)}"
                )
            stack[i] = arr

        for ordinal, mask in iter_regime_masks(reg):
            block = stack[:, mask, :].reshape(len(self._names), -1)
            self._sums[REGIME_BY_ORDINAL[ordinal]].add(block @ block.T)
            self._counts[REGIME_BY_ORDINAL[ordinal]] += int(mask.sum())

    def merge(self, other: L2GramAccumulator) -> None:
        if other._names != self._names:
            raise ValueError("cannot merge accumulators with different field names")
        for r in Regime:
            self._sums[r].merge(other._sums[r])
            self._counts[r] += other._counts[r]

    def count(self, regime: Regime | None = None) -> int:
        if regime is None:
            return sum(self._counts.values())
        return self._counts[regime]

    def occupancy(self) -> dict[Regime, float]:
        """m_r: the share of frames in each regime (9.7.4)."""
        total = self.count()
        if total == 0:
            raise ValueError("no frames accumulated; occupancy is undefined")
        return {r: self._counts[r] / total for r in Regime}

    def gram(self, regime: Regime | None = None) -> np.ndarray:
        """E_rho<f_i, f_j>, over all frames or conditional on one regime."""
        n = self.count(regime)
        if n == 0:
            raise ValueError(
                f"no frames in {'the whole stream' if regime is None else regime.value}; "
                "the conditional expectation is undefined and must not be reported as zero"
            )
        if regime is None:
            total = NeumaierSum((len(self._names), len(self._names)))
            for r in Regime:
                total.merge(self._sums[r])
            return total.value / n
        return self._sums[regime].value / n

    def cross(self, a: str, b: str, regime: Regime | None = None) -> Energy:
        """E_rho<f_a, f_b>. Second order in the fields, hence an `Energy`."""
        g = self.gram(regime)
        return Energy(float(g[self._index[a], self._index[b]]))

    def energy(self, a: str, regime: Regime | None = None) -> Energy:
        """E_rho||f_a||^2."""
        return self.cross(a, a, regime)

    def occupancy_weighted_ratio(
        self, numerator: tuple[str, str], denominator: tuple[str, str]
    ) -> Amplitude:
        """sum_r m_r E_{rho|r}<..> / sum_r m_r E_{rho|r}<..> (proposal 10.4.7).

        The occupancy weighting is applied to numerator and denominator
        separately and the ratio taken last; an occupancy-weighted average of
        per-regime ratios is a different number and is not the slope.
        """
        m = self.occupancy()
        num = 0.0
        den = 0.0
        for r in Regime:
            if self._counts[r] == 0:
                continue
            num += m[r] * float(self.cross(*numerator, regime=r).value)
            den += m[r] * float(self.cross(*denominator, regime=r).value)
        if den == 0.0:
            raise ValueError(
                f"the occupancy-weighted denominator E||{denominator[0]}||^2 is zero; "
                "a leakage slope against a field of zero energy has no value"
            )
        return Amplitude(num / den)


@dataclass(frozen=True)
class LeastSquaresSolution:
    """Result of a weighted least-squares fit, with what a reviewer needs.

    `residual_sum_of_squares` is read off the R factor rather than recomputed
    from residuals, so it costs nothing and does not need a second pass.
    """

    names: tuple[str, ...]
    coefficients: np.ndarray
    residual_sum_of_squares: float
    total_sum_of_squares: float
    n_rows: int
    condition_number: float

    def coefficient(self, name: str) -> float:
        return float(self.coefficients[self.names.index(name)])

    @property
    def degrees_of_freedom(self) -> int:
        return self.n_rows - len(self.names)

    @property
    def residual_variance(self) -> float:
        dof = self.degrees_of_freedom
        if dof <= 0:
            raise ValueError(
                f"{self.n_rows} rows and {len(self.names)} coefficients leave no "
                "residual degrees of freedom"
            )
        return self.residual_sum_of_squares / dof

    @property
    def explained_share(self) -> float:
        if self.total_sum_of_squares == 0.0:
            raise ValueError("the target is identically zero; an explained share is undefined")
        return 1.0 - self.residual_sum_of_squares / self.total_sum_of_squares


class StreamingLeastSquares:
    """Weighted least squares over a stream, via a running R factor.

    Each chunk is reduced by stacking it under the current upper-triangular
    factor and re-triangularising ("tall-skinny QR"). Memory is (p+1)^2 and the
    condition number of the design enters the error once, not twice as it would
    through normal equations. That matters here: the interference basis of 6.4
    layer 2 deliberately contains fields that are correlated with c_delta, and
    Lambda^part is the primary endpoint (11.3).

    The target column is carried inside the same factor, so the residual sum of
    squares comes out of the last diagonal entry.

    The chunk is written straight into a reused Fortran-ordered buffer that
    already holds R in its first rows, and `dgeqrf` triangularises that buffer
    in place. Building the stack with `concatenate` and `vstack` costs three
    copies of the chunk per call, and `numpy.linalg.qr` or `scipy.linalg.qr`
    then return `triu` of the whole block -- another array the height of the
    chunk, for a factor that occupies its top p+1 rows. Going to LAPACK
    directly is 3.9x faster at the chunk sizes Lambda^part runs at and returns
    the same R to 1e-15 relative, being the same routine on the same rows;
    `reports/verification/bench_streaming.py` is the measurement.
    """

    def __init__(self, feature_names: Sequence[str]) -> None:
        names = tuple(feature_names)
        if len(set(names)) != len(names):
            raise ValueError(f"feature names must be unique, got {names}")
        if not names:
            raise ValueError("at least one feature is required")
        self._names = names
        self._r: np.ndarray | None = None
        self._n_rows = 0
        self._tss = NeumaierSum(())
        self._buffer: np.ndarray | None = None

    def _stack(self, n_rows: int) -> np.ndarray:
        """A scratch block with R already in place, and room for `n_rows` below.

        Sized to the exact call so that the slice handed to LAPACK is Fortran
        contiguous and can be overwritten rather than copied. Chunking is
        regular, so the buffer is allocated once and reused; a short final chunk
        reallocates once and is still correct.

        R has fewer than p+1 rows while fewer than p+1 rows have been seen in
        total, which is legal -- `solve` is what refuses an underdetermined
        design -- so the offset is read off R rather than assumed.
        """
        m = len(self._names) + 1
        top = 0 if self._r is None else self._r.shape[0]
        want = top + n_rows
        if self._buffer is None or self._buffer.shape[0] != want:
            self._buffer = np.empty((want, m), dtype=np.float64, order="F")
        if self._r is not None:
            self._buffer[:top] = self._r
        return self._buffer

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def n_rows(self) -> int:
        return self._n_rows

    def update(
        self,
        design: np.ndarray,
        target: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        x = _as_2d(design, "design")
        if x.shape[1] != len(self._names):
            raise ValueError(f"expected {len(self._names)} columns, got {x.shape[1]}")
        y = np.asarray(target, dtype=np.float64).reshape(-1)
        if y.shape[0] != x.shape[0]:
            raise ValueError(f"design has {x.shape[0]} rows, target has {y.shape[0]}")
        if x.shape[0] == 0:
            return
        n, p = x.shape
        w = None
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
            if w.shape[0] != n:
                raise ValueError(f"weights must have {n} entries, got {w.shape[0]}")
            if np.any(w < 0):
                raise ValueError("least-squares weights must be non-negative")

        block = self._stack(n)
        top = block.shape[0] - n
        block[top:, :p] = x
        block[top:, p] = y
        if w is not None:
            block[top:] *= np.sqrt(w)[:, None]
            self._tss.add(float(np.dot(w, np.square(y))))
        else:
            self._tss.add(float(np.dot(y, y)))
        # LAPACK destroys the buffer, so R is copied out before the next call
        # writes over it.
        factored, _, _, info = dgeqrf(block, overwrite_a=True)
        if info < 0:
            raise ValueError(f"dgeqrf rejected argument {-info}")
        # `triu` both drops the reflectors LAPACK leaves below the diagonal and
        # lifts R out of the scratch buffer, which the next chunk overwrites.
        self._r = np.triu(factored[: min(block.shape[0], p + 1)])
        self._n_rows += n

    def update_fields(
        self,
        fields: Mapping[str, np.ndarray],
        target: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        """Same, for vector-valued fields: frames x action components are rows.

        Minimising sum over frames of ||eps - sum_i beta_i nu_i||^2 is exactly
        the L^2(rho) regression of 6.4 layer 2, so flattening the action axis
        into the row axis is the definition, not an approximation.
        """
        y = _as_2d(target, "target")
        n, d = y.shape
        columns = []
        for name in self._names:
            arr = _as_2d(fields[name], name)
            if arr.shape != (n, d):
                raise ValueError(f"field {name!r} has shape {arr.shape}, expected {(n, d)}")
            columns.append(arr.reshape(-1))
        design = np.stack(columns, axis=1)
        w = None
        if weights is not None:
            w = np.repeat(np.asarray(weights, dtype=np.float64).reshape(-1), d)
        self.update(design, y.reshape(-1), w)

    def solve(self, *, condition_limit: float = 1e8) -> LeastSquaresSolution:
        if self._r is None:
            raise ValueError("no rows accumulated")
        p = len(self._names)
        if self._n_rows <= p:
            raise ValueError(
                f"{self._n_rows} rows cannot determine {p} coefficients"
            )
        r1 = self._r[:p, :p]
        z = self._r[:p, p]
        rss = float(np.sum(np.square(self._r[p:, p])))
        singular = np.linalg.svd(r1, compute_uv=False)
        if singular[-1] <= 0.0:
            raise RankDeficientDesign(
                f"design {self._names} is exactly singular; two basis fields are collinear"
            )
        cond = float(singular[0] / singular[-1])
        if cond > condition_limit:
            raise RankDeficientDesign(
                f"design {self._names} has condition number {cond:.3g} > {condition_limit:.3g}. "
                "The interference basis of 6.4 layer 2 is mandatory, so drop nothing: "
                "report which pair is collinear and rescale or re-derive it"
            )
        coefficients = np.linalg.solve(r1, z)
        return LeastSquaresSolution(
            names=self._names,
            coefficients=coefficients,
            residual_sum_of_squares=rss,
            total_sum_of_squares=float(self._tss.value),
            n_rows=self._n_rows,
            condition_number=cond,
        )


@dataclass(frozen=True, order=True)
class CellRegimeKey:
    """A (cell, regime) bucket -- the unit of the two-way model's cell means."""

    cell: Cell
    regime: Regime


class CellRegimeMoments:
    """Per (cell, regime) action mean and within-cell variance.

    Produces eta_{tau delta, r} for `two_way_fit` and sigma^2_{tau delta} for
    the heteroscedastic conductance caliber of proposition 3.
    """

    def __init__(self, action_width: int) -> None:
        self._width = action_width
        self._buckets: dict[CellRegimeKey, WelfordAccumulator] = {}

    def update_table(self, table: FrameTable, values: np.ndarray | None = None) -> None:
        """Accumulate one `FrameTable`; `values` defaults to the observed actions."""
        arr = table.actions_defective if values is None else np.asarray(values)
        arr = _as_2d(arr, "values")
        if arr.shape[0] != table.n_frames:
            raise ValueError(f"values has {arr.shape[0]} rows, table has {table.n_frames}")
        for ordinal, mask in iter_regime_masks(table.regime):
            key = CellRegimeKey(table.cell, REGIME_BY_ORDINAL[ordinal])
            acc = self._buckets.get(key)
            if acc is None:
                acc = WelfordAccumulator(self._width)
                self._buckets[key] = acc
            acc.update(arr[mask])

    def keys(self) -> list[CellRegimeKey]:
        return sorted(self._buckets)

    def moments(self, key: CellRegimeKey) -> WelfordAccumulator:
        acc = self._buckets.get(key)
        if acc is None:
            raise KeyError(f"{key} was never observed; an unobserved cell has no mean")
        return acc

    def n_frames(self, key: CellRegimeKey) -> int:
        return int(self.moments(key).count)


class RegimeOccupancyAccumulator:
    """Per (cell, regime) frame counts -- the 9.7.4 occupancy mass table."""

    def __init__(self) -> None:
        self._counts: dict[Cell, np.ndarray] = {}

    def update(self, cell: Cell, regime: np.ndarray) -> None:
        reg = np.asarray(regime)
        counts = self._counts.setdefault(cell, np.zeros(len(Regime), dtype=np.int64))
        binned = np.bincount(reg.astype(np.int64), minlength=len(Regime))
        if binned.shape[0] != len(Regime):
            raise ValueError(f"regime ordinals out of range: max {reg.max()}")
        counts += binned

    def update_table(self, table: FrameTable) -> None:
        self.update(table.cell, table.regime)

    def cells(self) -> list[Cell]:
        return sorted(self._counts)

    def frames(self, cell: Cell, regime: Regime) -> int:
        counts = self._counts.get(cell)
        if counts is None:
            raise KeyError(f"cell {cell} was never observed")
        return int(counts[REGIME_ORDINAL[regime]])

    def total_frames(self, cell: Cell) -> int:
        counts = self._counts.get(cell)
        if counts is None:
            raise KeyError(f"cell {cell} was never observed")
        return int(counts.sum())

    def occupancy(self, cell: Cell) -> dict[Regime, float]:
        total = self.total_frames(cell)
        if total == 0:
            raise ValueError(f"cell {cell} has no frames; occupancy mass is undefined")
        return {r: self.frames(cell, r) / total for r in Regime}

    def as_rows(self) -> list[dict[str, Any]]:
        """The long-table form of 9.7.4, minus the columns other modules own."""
        rows: list[dict[str, Any]] = []
        for cell in self.cells():
            occ = self.occupancy(cell)
            for r in Regime:
                rows.append(
                    {
                        "task": cell.task,
                        "family": cell.family,
                        "defect": cell.defect_code,
                        "regime": r.value,
                        "m": occ[r],
                        "n_frames": self.frames(cell, r),
                    }
                )
        return rows


def iter_regime_masks(regime: np.ndarray) -> Iterable[tuple[int, np.ndarray]]:
    """Yield (ordinal, boolean mask) for the regimes present in this chunk."""
    reg = np.asarray(regime)
    if reg.ndim != 1:
        raise ValueError(f"regime must be one-dimensional, got shape {reg.shape}")
    present = np.unique(reg)
    if present.size and (present.min() < 0 or present.max() >= len(Regime)):
        raise ValueError(f"regime ordinals must lie in [0, {len(Regime)}), got {present}")
    for ordinal in present:
        yield int(ordinal), reg == ordinal


def smoothing_field(
    values: np.ndarray, episode_index: np.ndarray, window: int
) -> np.ndarray:
    """(S_h f - f)(x): the time-smoothing interference field of 6.4 layer 2.

    S_h is a centred moving average of odd width `window` applied within an
    episode. Crossing an episode boundary would smooth two unrelated
    trajectories together and put a spurious step exactly where 9.6.1's
    fixed-length collection concatenates them.
    """
    if window < 3 or window % 2 == 0:
        raise ValueError(f"window must be an odd integer >= 3, got {window}")
    arr = _as_2d(values, "values")
    episodes = np.asarray(episode_index)
    if episodes.shape != (arr.shape[0],):
        raise ValueError(f"episode_index must be ({arr.shape[0]},), got {episodes.shape}")
    out = np.empty_like(arr)
    half = window // 2
    boundaries = np.flatnonzero(np.diff(episodes)) + 1
    for start, stop in zip(
        np.concatenate([[0], boundaries]),
        np.concatenate([boundaries, [arr.shape[0]]]),
        strict=True,
    ):
        segment = arr[start:stop]
        n = segment.shape[0]
        padded = np.concatenate(
            [
                np.repeat(segment[:1], min(half, n), axis=0),
                segment,
                np.repeat(segment[-1:], min(half, n), axis=0),
            ],
            axis=0,
        )
        cumulative = np.concatenate(
            [np.zeros((1, arr.shape[1])), np.cumsum(padded, axis=0)], axis=0
        )
        pad = min(half, n)
        lower = np.clip(np.arange(n) + pad - half, 0, padded.shape[0])
        upper = np.clip(np.arange(n) + pad + half + 1, 0, padded.shape[0])
        out[start:stop] = (cumulative[upper] - cumulative[lower]) / (upper - lower)[:, None]
    return out - arr


def effective_sample_count(
    *, n_traj: int, horizon: int, tau_corr: float, sigma2: float | None = None
) -> EffectiveSampleCount:
    """n_traj * (T / tau_corr), optionally divided by sigma^2 (11.4, 7.9).

    Present so that callers never build an `EffectiveSampleCount` by hand and
    the two calibers of proposition 3 are constructed the same way.
    """
    if sigma2 is None:
        return EffectiveSampleCount.from_trajectories(
            n_traj=n_traj, horizon=horizon, tau_corr=tau_corr
        )
    return EffectiveSampleCount.from_trajectories_heteroscedastic(
        n_traj=n_traj, horizon=horizon, tau_corr=tau_corr, sigma2=sigma2
    )
