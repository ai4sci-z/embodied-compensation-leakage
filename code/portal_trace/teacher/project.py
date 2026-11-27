"""Projection of the ILC feedforward onto the state feature basis phi (9.4.4, 9.4.5).

9.4.4 states the structural problem this module measures. The ILC update is
non-causal: it reads the error `ell_adv` steps ahead, so the converged feedforward
`u_inf(t)` depends on the whole shape of the reference after `t`, and the
reference is per task. The same state visited in two tasks can therefore carry
two different feedforward values, and if that happens badly enough then
`c_delta(x)` as a function of the state does not exist -- not approximately, at
all. Both the pointwise definition of 5.1 and the operator form of (A4) lose
their carrier.

The main caliber is the projection: regress `u_inf` on a state feature basis
`phi(x)` and report the residual share

    E_phi = E||u_inf - W phi||^2 / E||u_inf - mean(u_inf)||^2,

whose denominator is de-meaned because a constant feedforward would otherwise
push the ratio down for free. This is the earliest of the three zero-GPU gates
(14.5 E-17): it decides whether the object of study exists, which comes before
asking whether leakage happens, and it costs no GPU. `python -m
portal_trace.teacher.project` runs it standalone for that reason.

**phi deliberately excludes t, the trajectory index and the task id.** Putting
any of them in would drive the residual to near zero and the test would be
vacuous -- what it asks is precisely whether the feedforward can be written as a
function of state.

**The rank is measured, not assumed.** The four groups come to 482 raw columns on
a 7-DoF arm with three injected joints (9.4.4's table says 392, written before
9.2.6 put D8 into the core six; D8's compensation direction is the commanded
torque and needs eighteen columns of its own, which the regime tensor then
carries into group 3), but the regime-by-basis products are
strongly collinear, so `Phi` is preprocessed by a truncated SVD retaining 99.9%
of the energy and `p` means the surviving rank. That rank is the lock-file
measurement `phi.effective_rank`; this module produces it and never reads a
default for it.

9.4.5 adds the hold-out protocol, because a variance decomposition by task id is
not identifiable: each task has its own reference, tasks visit nearly disjoint
states, so task id and state are collinear and the answer would be whatever the
covariate specification made it. `loto_r2` runs the six-fold leave-one-task-out
instead, with lambda chosen by an inner grouped split that never sees the held
out task, and `task_id_accuracy` turns the collinearity itself into a number.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.model_selection import GroupKFold

from portal_trace.defects.fields import FieldContext, apply_pinv_transpose
from portal_trace.lockfile import REQUIRED_MEASUREMENTS, LockFile
from portal_trace.types import REGIME_ORDINAL, Phase, Regime

__all__ = [
    "SVD_ENERGY",
    "QUADRATIC_ENERGY",
    "LAMBDA_GRID",
    "N_FOLDS",
    "E_PHI_PASS",
    "E_PHI_STRONG",
    "E_PHI_NONEXISTENT",
    "E_PHI_GATE",
    "LOTO_R2_PASS",
    "LOTO_MAX_NEGATIVE_FOLDS",
    "CONTACT_RESIDUAL_RATIO",
    "NOMINAL_RAW_COLUMNS",
    "PhiInputs",
    "PhiMatrix",
    "PhiBasis",
    "RidgeFit",
    "StratifiedShare",
    "ProjectionReport",
    "LotoFold",
    "LotoReport",
    "build_phi",
    "truncate",
    "fit_ridge",
    "residual_energies",
    "residual_share",
    "project_report",
    "loto_r2",
    "task_id_accuracy",
]

# 9.4.4, stated numerically in the proposal, so constants rather than
# lock-file measurements (contract 3.1).
SVD_ENERGY = 0.999
QUADRATIC_ENERGY = 0.80
LAMBDA_GRID: np.ndarray = np.logspace(-6.0, 2.0, 33)
N_FOLDS = 5

# 9.4.4's three-tier verdict and the G1 aggregate threshold.
E_PHI_PASS = 0.25
E_PHI_STRONG = 0.15
E_PHI_NONEXISTENT = 0.50
E_PHI_GATE = 0.35

# 9.4.5 hold-out thresholds.
LOTO_R2_PASS = 0.50
LOTO_MAX_NEGATIVE_FOLDS = 1

# 10.3.3: the free-space and contact segments are reported as two tables and
# never pooled. Past this ratio between their residual medians, the ground truth
# of c_delta is declared trustworthy on the free-space segment only.
CONTACT_RESIDUAL_RATIO = 3.0

# The 9.4.4 column table, for a 7-DoF arm with the default three injected
# joints. Reached only when the group-2 variance truncation keeps 34 of its 56
# quadratic columns, which is data dependent; `PhiMatrix` reports what it
# actually built and compares against this.
#
# 482 and not the table's 392: group 1 grew from 66 to 84 when D8 replaced D6
# among the core six (9.2.6) and needed a block of its own, and group 3 is group
# 1 tensored with four regimes, so it grew from 264 to 336. 84 + 61 + 336 + 1.
NOMINAL_RAW_COLUMNS = 482

# 9.4.4 drops one regime block from the tensor product. The five regimes are an
# exclusive partition, so sum_r 1_r psi == psi: keeping the un-tensored group 1
# alongside all five blocks makes the design matrix exactly rank deficient. r0
# is the block that goes, being the one that is hardest to interpret.
_TENSOR_REGIMES: tuple[Regime, ...] = (
    Regime.R4_CONTACT,
    Regime.R1_REVERSAL,
    Regime.R2_QUASISTATIC,
    Regime.R3_FREESPACE,
)


@dataclass(frozen=True)
class PhiInputs:
    """Per-frame material for one cell, and the model the frames were recorded on.

    Three columns cannot be recovered from `qpos`/`qvel` and are therefore
    required inputs rather than something this module reconstructs:
    `a_dot_nominal` is the closed-loop derivative of 5.1, which only
    `defects.adot.ClosedLoopDerivative` can produce; `joint_torque` is the
    commanded torque whose sign gates D5's basis element; and `regime` is the
    9.7.1 label, which depends on thresholds calibrated elsewhere. Guessing any
    of them would invent the support of a basis function.
    """

    states: np.ndarray            # (N, nq + nv)
    u_inf: np.ndarray             # (N, 6) converged ILC feedforward, action units
    regime: np.ndarray            # (N,) uint8 Regime ordinal
    task: np.ndarray              # (N,) task id per frame, the GroupKFold group
    phase: np.ndarray             # (N,) uint8 index into refs.PHASE_ORDER
    a_dot_nominal: np.ndarray     # (N, 6) 5.1's a_dot on the clean twin
    joint_torque: np.ndarray      # (N, n_joints) commanded joint torque
    contact: np.ndarray           # (N,) bool, the 10.3.3 free-space / contact split
    injected_joints: tuple[int, ...]
    model: Any
    data: Any
    ctx: FieldContext

    def __post_init__(self) -> None:
        n = len(self.states)
        columns = ("u_inf", "regime", "task", "phase", "a_dot_nominal", "joint_torque", "contact")
        for name in columns:
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(f"{name} has {len(arr)} rows, states have {n}")
        if self.u_inf.shape[1] != 6:
            raise ValueError(f"u_inf is {self.u_inf.shape[1]} wide, the feedforward is 6")
        n_joints = self.ctx.n_joints
        if self.joint_torque.shape[1] != n_joints:
            raise ValueError(
                f"joint_torque is {self.joint_torque.shape[1]} wide, this arm has "
                f"{n_joints} joints"
            )
        for j in self.injected_joints:
            if not 0 <= j < n_joints:
                raise ValueError(f"injected joint {j} is outside 0..{n_joints - 1}")


# ---------------------------------------------------------------------------
# The four groups
# ---------------------------------------------------------------------------


def _kinematics(inputs: PhiInputs) -> dict[str, np.ndarray]:
    """Per-frame Jacobians, gravity torque and tool pose.

    One MuJoCo pass over the frames, because none of these is a batched object.
    Gravity torque is read off `qfrc_bias` with the velocities zeroed, which
    leaves the Coriolis and centrifugal terms out -- 9.4.4's group 1 wants
    `J^-T tau_g(q)`, a function of configuration alone.
    """
    model, data, ctx = inputs.model, inputs.data, inputs.ctx
    states = inputs.states
    nq = int(model.nq)
    qpos_idx = list(ctx.qpos_index)
    qvel_idx = list(ctx.qvel_index)
    n, nj = len(states), len(qvel_idx)

    jac = np.empty((n, 6, nj), dtype=np.float64)
    tau_g = np.empty((n, nj), dtype=np.float64)
    eef_pos = np.empty((n, 3), dtype=np.float64)
    eef_quat = np.empty((n, 4), dtype=np.float64)
    qpos = np.empty((n, len(qpos_idx)), dtype=np.float64)
    qvel = np.empty((n, nj), dtype=np.float64)
    jacp = np.zeros((3, int(model.nv)))
    jacr = np.zeros((3, int(model.nv)))

    for i, state in enumerate(states):
        data.qpos[:] = state[:nq]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        jacp[:] = 0.0
        jacr[:] = 0.0
        mujoco.mj_jacSite(model, data, jacp, jacr, ctx.site_id)
        jac[i, :3, :] = jacp[:, qvel_idx]
        jac[i, 3:, :] = jacr[:, qvel_idx]
        tau_g[i] = data.qfrc_bias[qvel_idx]
        eef_pos[i] = data.site_xpos[ctx.site_id]
        eef_quat[i] = Rotation.from_matrix(
            np.asarray(data.site_xmat[ctx.site_id]).reshape(3, 3)
        ).as_quat()
        qpos[i] = state[qpos_idx]
        qvel[i] = state[nq:][qvel_idx]

    return {
        "jac": jac,
        "tau_g": tau_g,
        "eef_pos": eef_pos,
        "eef_quat": eef_quat,
        "qpos": qpos,
        "qvel": qvel,
        "twist": np.einsum("nij,nj->ni", jac, qvel),
    }


def _group1(inputs: PhiInputs, kin: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """The physical basis psi_j of 14.5 E-4: 84 columns at 7 DoF, 3 injected joints.

    Every block is the task-space image of one defect's analytic compensation
    form, so a defect whose field lies in the span of its own block projects
    exactly and the residual measures the part 9.4.4 is asking about rather than
    a basis that was never rich enough.

    84 and not the 66 of the table: the table predates 9.2.6's replacement of D6
    by D8 among the core six, and D8's block is the eighteen columns at the end.
    """
    jac = kin["jac"]
    blocks: list[np.ndarray] = []
    names: list[str] = []

    for j in inputs.injected_joints:                     # D1 / D10: J(q) e_j
        blocks.append(jac[:, :, j])
        names += [f"psi.Je{j}.{k}" for k in range(6)]

    blocks.append(inputs.a_dot_nominal)                  # D2: a_dot_nom
    names += [f"psi.adot.{k}" for k in range(6)]

    blocks.append(apply_pinv_transpose(jac, kin["tau_g"]))  # D3: J^-T tau_g
    names += [f"psi.Jinv_taug.{k}" for k in range(6)]

    sign_qvel = np.sign(kin["qvel"])
    for j in inputs.injected_joints:                     # D4 / D6
        blocks.append(jac[:, :, j] * sign_qvel[:, j, None])
        names += [f"psi.sgn_qd{j}_Je{j}.{k}" for k in range(6)]

    sign_u = np.sign(inputs.joint_torque)
    for j in inputs.injected_joints:                     # D5
        blocks.append(jac[:, :, j] * sign_u[:, j, None])
        names += [f"psi.sgn_u{j}_Je{j}.{k}" for k in range(6)]

    # D8: c = (1/g_j - 1) u_j, so the direction is the commanded torque itself
    # rather than its sign. 14.5 E-4's table has no such block because D8 was
    # priority C when it was written; 9.2.6 put D8 into the core six, and without
    # its own block the D8 cells would report a projection residual measuring the
    # basis rather than (A4). The sign block above does not cover it: sign(u) and
    # u are collinear only where |u| is constant, which is nowhere in a reaching
    # motion.
    for j in inputs.injected_joints:                     # D8
        blocks.append(jac[:, :, j] * inputs.joint_torque[:, j, None])
        names += [f"psi.u{j}_Je{j}.{k}" for k in range(6)]

    return np.concatenate(blocks, axis=1), names


def _group2(inputs: PhiInputs, kin: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Proprioceptive polynomial: 27 linear columns plus a truncated quadratic block.

    9.4.4 quotes the quadratic part as "56 columns, kept to the leading 80% of
    the variance, about 34". The 80% is a rule, the 34 is an expectation, so the
    truncation is applied and the surviving count reported rather than assumed.
    """
    q, qd = kin["qpos"], kin["qvel"]
    linear = np.concatenate([q, qd, kin["eef_pos"], kin["eef_quat"], kin["twist"]], axis=1)
    names = (
        [f"poly.q{i}" for i in range(q.shape[1])]
        + [f"poly.qd{i}" for i in range(qd.shape[1])]
        + [f"poly.eef_pos.{i}" for i in range(3)]
        + [f"poly.eef_quat.{i}" for i in range(4)]
        + [f"poly.twist.{i}" for i in range(6)]
    )

    quad_blocks: list[np.ndarray] = []
    quad_names: list[str] = []
    for source, tag in ((qd, "qd"), (kin["twist"], "twist")):
        iu, ju = np.triu_indices(source.shape[1])
        quad_blocks.append(source[:, iu] * source[:, ju])
        quad_names += [f"poly.{tag}{a}{b}" for a, b in zip(iu, ju, strict=True)]
    quad_blocks.append(q * q)
    quad_names += [f"poly.qq{i}" for i in range(q.shape[1])]

    quad = np.concatenate(quad_blocks, axis=1)
    keep = _variance_truncation(quad, QUADRATIC_ENERGY)
    return (
        np.concatenate([linear, quad[:, keep]], axis=1),
        names + [quad_names[i] for i in keep],
    )


def _variance_truncation(block: np.ndarray, energy: float) -> np.ndarray:
    """Column indices holding the leading `energy` share of the block's variance.

    Ordered by variance and cut at the cumulative share, then returned in the
    original column order so that a basis built twice on the same frames has the
    same column layout and the two `W` matrices are comparable entry by entry.
    """
    var = block.var(axis=0)
    total = float(var.sum())
    if total <= 0.0:
        raise ValueError(
            "the quadratic block has zero variance on these frames; a basis cannot "
            "be truncated by an energy share that does not exist"
        )
    order = np.argsort(var)[::-1]
    cumulative = np.cumsum(var[order]) / total
    n_keep = int(np.searchsorted(cumulative, energy) + 1)
    return np.sort(order[:n_keep])


def _group3(
    group1: np.ndarray, names1: list[str], regime: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Regime one-hot tensored with group 1, r0 dropped: 336 columns (4 x 84)."""
    blocks, names = [], []
    for r in _TENSOR_REGIMES:
        indicator = (regime == REGIME_ORDINAL[r]).astype(np.float64)[:, None]
        blocks.append(group1 * indicator)
        names += [f"{r.value}x{n}" for n in names1]
    return np.concatenate(blocks, axis=1), names


@dataclass(frozen=True)
class PhiMatrix:
    """The raw design matrix, with its group layout kept for the report."""

    raw: np.ndarray
    names: tuple[str, ...]
    group_slices: dict[str, slice]

    @property
    def n_columns(self) -> int:
        return self.raw.shape[1]

    @property
    def matches_nominal(self) -> bool:
        """Whether the build reproduced 9.4.4's table, D8 block included."""
        return self.n_columns == NOMINAL_RAW_COLUMNS


def build_phi(
    inputs: PhiInputs,
    *,
    kinematics: Mapping[str, np.ndarray] | None = None,
) -> PhiMatrix:
    """Assemble the four groups of 9.4.4 into one raw design matrix."""
    kin = _kinematics(inputs) if kinematics is None else dict(kinematics)
    required = {"jac", "tau_g", "eef_pos", "eef_quat", "qpos", "qvel", "twist"}
    if set(kin) != required or any(len(kin[name]) != len(inputs.states) for name in required):
        raise ValueError("precomputed kinematics do not match the Phi input frames")
    g1, n1 = _group1(inputs, kin)
    g2, n2 = _group2(inputs, kin)
    g3, n3 = _group3(g1, n1, inputs.regime)
    const = np.ones((len(inputs.states), 1), dtype=np.float64)

    raw = np.concatenate([g1, g2, g3, const], axis=1)
    widths = [g1.shape[1], g2.shape[1], g3.shape[1], 1]
    edges = np.cumsum([0, *widths])
    return PhiMatrix(
        raw=raw,
        names=tuple(n1 + n2 + n3 + ["const"]),
        group_slices={
            "physical": slice(edges[0], edges[1]),
            "proprio_poly": slice(edges[1], edges[2]),
            "regime_x_physical": slice(edges[2], edges[3]),
            "const": slice(edges[3], edges[4]),
        },
    )


# ---------------------------------------------------------------------------
# Truncation and the ridge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhiBasis:
    """A standardised, SVD-truncated basis, plus the transform that produced it.

    `effective_rank` is the lock-file measurement `phi.effective_rank`. It is
    produced here and written by `scripts/measure_constants.py`; nothing reads a
    default for it, so a run on an un-measured cell reports its own rank rather
    than inheriting one.
    """

    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray        # (p_raw, rank) right singular vectors kept
    singular_values: np.ndarray   # (rank,)
    energy_kept: float
    n_raw_columns: int

    @property
    def effective_rank(self) -> int:
        return int(self.components.shape[1])

    def transform(self, raw: np.ndarray) -> np.ndarray:
        return ((raw - self.mean) / self.scale) @ self.components


def truncate(raw: np.ndarray, *, energy: float = SVD_ENERGY) -> PhiBasis:
    """Standardise the columns, then keep the leading `energy` share of the spectrum.

    Standardisation comes first because the groups are in wildly different units
    -- a Jacobian column in metres per radian against a squared joint velocity --
    and an unscaled SVD would report the rank of whichever group happened to be
    largest.
    """
    raw = np.asarray(raw, dtype=np.float64)
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)
    # Constant columns carry no information for the regression but must survive
    # standardisation without dividing by zero; the SVD then drops them, which is
    # the correct outcome and is visible in the rank.
    scale = np.where(scale > 0.0, scale, 1.0)
    z = (raw - mean) / scale

    _, s, vt = np.linalg.svd(z, full_matrices=False)
    energy_curve = np.cumsum(s**2) / float(np.sum(s**2))
    rank = int(np.searchsorted(energy_curve, energy) + 1)
    return PhiBasis(
        mean=mean,
        scale=scale,
        components=vt[:rank].T,
        singular_values=s[:rank],
        energy_kept=float(energy_curve[rank - 1]),
        n_raw_columns=raw.shape[1],
    )


@dataclass(frozen=True)
class RidgeFit:
    """Ridge coefficients in the truncated basis, with the lambda that was chosen."""

    basis: PhiBasis
    weights: np.ndarray           # (rank, 6)
    intercept: np.ndarray         # (6,)
    lam: float
    cv_curve: dict[float, float] = field(default_factory=dict)

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return self.basis.transform(raw) @ self.weights + self.intercept


def _ridge_solve(z: np.ndarray, u: np.ndarray, lam: float) -> np.ndarray:
    """Ridge on centred inputs, solved through the normal equations in rank space.

    The basis is already orthogonalised by `truncate`, so `z.T @ z` is diagonal up
    to the sample covariance and the solve is small: rank is a few hundred, not
    the 45 M frames.
    """
    gram = z.T @ z
    gram.flat[:: gram.shape[0] + 1] += lam
    return np.linalg.solve(gram, z.T @ u)


def fit_ridge(
    raw: np.ndarray,
    u: np.ndarray,
    groups: np.ndarray,
    *,
    basis: PhiBasis | None = None,
    grid: np.ndarray = LAMBDA_GRID,
    n_folds: int = N_FOLDS,
) -> RidgeFit:
    """Fit W with lambda chosen by task-grouped cross validation (9.4.4).

    The folds are grouped by task and not drawn at random. Neighbouring frames of
    one task are strongly autocorrelated, so a random split leaks the training
    fold into the validation fold, lambda comes out too small and the residual is
    systematically understated -- which would flatter exactly the quantity this
    module exists to report.
    """
    basis = basis if basis is not None else truncate(raw)
    z = basis.transform(raw)
    u = np.asarray(u, dtype=np.float64)

    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError(
            f"cross validation is grouped by task and these frames span "
            f"{len(unique)} task(s); lambda cannot be selected without at least two"
        )
    splitter = GroupKFold(n_splits=min(n_folds, len(unique)))
    scores = dict.fromkeys((float(g) for g in grid), 0.0)
    for train, test in splitter.split(z, u, groups=groups):
        z_mu, u_mu = z[train].mean(axis=0), u[train].mean(axis=0)
        zc, uc = z[train] - z_mu, u[train] - u_mu
        for lam in grid:
            w = _ridge_solve(zc, uc, float(lam))
            residual = u[test] - ((z[test] - z_mu) @ w + u_mu)
            scores[float(lam)] += float(np.sum(residual * residual))

    lam = min(scores, key=lambda k: scores[k])
    z_mu, u_mu = z.mean(axis=0), u.mean(axis=0)
    weights = _ridge_solve(z - z_mu, u - u_mu, lam)
    return RidgeFit(
        basis=basis,
        weights=weights,
        intercept=u_mu - z_mu @ weights,
        lam=lam,
        cv_curve=scores,
    )


def residual_energies(u: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """The two sums E_phi is a ratio of, kept separately.

    G1's aggregate is E_phi over the union of the cells, and a share of sums is
    not a weighted mean of shares by any weight the cells supply. Carrying both
    sums lets the aggregate be the same quantity as the per-cell reading rather
    than an average that resembles it.
    """
    u = np.asarray(u, dtype=np.float64)
    numerator = float(np.sum((u - predicted) ** 2))
    denominator = float(np.sum((u - u.mean(axis=0)) ** 2))
    if denominator <= 0.0:
        raise ValueError(
            "the feedforward is constant across these frames, so the de-meaned "
            "denominator of E_phi is zero and the share is undefined"
        )
    return numerator, denominator


def residual_share(u: np.ndarray, predicted: np.ndarray) -> float:
    """E_phi, with the de-meaned denominator of 9.4.4.

    Against the total energy instead, a feedforward that is mostly a constant
    offset would score well for a reason that has nothing to do with phi.
    """
    numerator, denominator = residual_energies(u, predicted)
    return numerator / denominator


# ---------------------------------------------------------------------------
# The stratified report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StratifiedShare:
    """E_phi restricted to one stratum, with the frame count behind it."""

    label: str
    n_frames: int
    e_phi: float


def _stratify(
    u: np.ndarray, predicted: np.ndarray, key: np.ndarray, labels: dict[Any, str]
) -> tuple[StratifiedShare, ...]:
    out = []
    for value, name in labels.items():
        mask = key == value
        n = int(mask.sum())
        if n == 0:
            continue
        out.append(StratifiedShare(name, n, residual_share(u[mask], predicted[mask])))
    return tuple(out)


@dataclass(frozen=True)
class ProjectionReport:
    """The 9.4.4 gate: one E_phi, its strata, and the measured rank."""

    cell: str
    e_phi: float
    residual_energy: float
    demeaned_energy: float
    n_frames: int
    basis: PhiBasis
    lam: float
    by_contact: tuple[StratifiedShare, ...]
    by_regime: tuple[StratifiedShare, ...]
    by_phase: tuple[StratifiedShare, ...]
    by_task: tuple[StratifiedShare, ...]
    raw_columns: int
    matches_nominal_columns: bool

    @property
    def verdict(self) -> str:
        """9.4.4's three tiers, in the proposal's own words."""
        if self.e_phi > E_PHI_NONEXISTENT:
            return "nonexistent"
        if self.e_phi <= E_PHI_STRONG:
            return "strong_pass"
        if self.e_phi <= E_PHI_PASS:
            return "pass"
        return "fail"

    @property
    def contact_free_ratio(self) -> float:
        """Contact-segment E_phi over free-space E_phi (10.3.3).

        Reported rather than pooled: 10.3.3 forbids a combined number for the two
        segments, because the contact segment is where the reference is
        unreachable by construction and a pooled figure hides that.
        """
        shares = {row.label: row.e_phi for row in self.by_contact}
        if set(shares) != {"free_space", "contact"}:
            raise ValueError(
                "the free-space / contact split needs frames on both sides; these "
                f"frames only cover {sorted(shares)}"
            )
        return shares["contact"] / shares["free_space"]

    @property
    def contact_segment_dominates(self) -> bool:
        """10.3.3: past 3x, c_delta is declared trustworthy only in free space."""
        return self.contact_free_ratio > CONTACT_RESIDUAL_RATIO

    def render(self) -> str:
        lines = [
            f"9.4.4 E_phi projection report -- {self.cell}",
            f"  frames            : {self.n_frames}",
            f"  raw columns       : {self.raw_columns}"
            + ("" if self.matches_nominal_columns
               else f" (9.4.4 tabulates {NOMINAL_RAW_COLUMNS} at 7 DoF / 3 injected joints)"),
            f"  effective rank    : {self.basis.effective_rank} "
            f"at {100 * self.basis.energy_kept:.3f}% energy  [phi.effective_rank]",
            f"  lambda            : {self.lam:.3e}",
            f"  E_phi             : {self.e_phi:.4f}  -> {self.verdict}"
            f"  (pass <= {E_PHI_PASS}, strong <= {E_PHI_STRONG}, "
            f"nonexistent > {E_PHI_NONEXISTENT})",
        ]
        for title, rows in (
            ("by segment", self.by_contact),
            ("by regime", self.by_regime),
            ("by phase", self.by_phase),
            ("by task", self.by_task),
        ):
            lines.append(f"  {title}:")
            for row in rows:
                lines.append(f"    {row.label:<14} n={row.n_frames:<7} E_phi={row.e_phi:.4f}")
        return "\n".join(lines)


def project_report(inputs: PhiInputs, *, cell: str) -> ProjectionReport:
    """Build phi, fit the ridge and report E_phi with the 9.4.4 strata."""
    phi = build_phi(inputs)
    fit = fit_ridge(phi.raw, inputs.u_inf, inputs.task)
    predicted = fit.predict(phi.raw)
    numerator, denominator = residual_energies(inputs.u_inf, predicted)
    return ProjectionReport(
        cell=cell,
        e_phi=numerator / denominator,
        residual_energy=numerator,
        demeaned_energy=denominator,
        n_frames=len(inputs.states),
        basis=fit.basis,
        lam=fit.lam,
        by_contact=_stratify(
            inputs.u_inf, predicted, inputs.contact.astype(bool),
            {False: "free_space", True: "contact"},
        ),
        by_regime=_stratify(
            inputs.u_inf, predicted, inputs.regime,
            {REGIME_ORDINAL[r]: r.value for r in Regime},
        ),
        by_phase=_stratify(
            inputs.u_inf, predicted, inputs.phase,
            {i: p.value for i, p in enumerate(Phase)},
        ),
        by_task=_stratify(
            inputs.u_inf, predicted, inputs.task,
            {t: str(t) for t in np.unique(inputs.task)},
        ),
        raw_columns=phi.n_columns,
        matches_nominal_columns=phi.matches_nominal,
    )


# ---------------------------------------------------------------------------
# 9.4.5: the leave-one-task-out R^2 that tests (A4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LotoFold:
    held_out: str
    n_frames: int
    r2: float
    lam: float


@dataclass(frozen=True)
class LotoReport:
    """Six folds of leave-one-task-out, the only identifiable test of (A4)."""

    folds: tuple[LotoFold, ...]
    task_id_accuracy: float

    @property
    def median_r2(self) -> float:
        return float(np.median([f.r2 for f in self.folds]))

    @property
    def n_negative(self) -> int:
        return int(sum(1 for f in self.folds if f.r2 < 0.0))

    @property
    def passes(self) -> bool:
        return self.median_r2 >= LOTO_R2_PASS and self.n_negative <= LOTO_MAX_NEGATIVE_FOLDS

    def render(self) -> str:
        lines = [
            "9.4.5 leave-one-task-out R^2 for (A4)",
            f"  task id recoverable from phi with accuracy {self.task_id_accuracy:.3f} "
            "-- the collinearity that makes a variance decomposition unidentifiable",
        ]
        for f in self.folds:
            lines.append(f"    hold out {f.held_out:<22} n={f.n_frames:<7} "
                         f"R^2={f.r2:+.4f}  lambda={f.lam:.2e}")
        span = [f.r2 for f in self.folds]
        lines.append(
            f"  median R^2 = {self.median_r2:+.4f}, range "
            f"[{min(span):+.4f}, {max(span):+.4f}], {self.n_negative} negative fold(s) "
            f"-> {'PASS' if self.passes else 'FAIL'} "
            f"(median >= {LOTO_R2_PASS}, at most {LOTO_MAX_NEGATIVE_FOLDS} negative)"
        )
        return "\n".join(lines)


def loto_r2(raw: np.ndarray, u: np.ndarray, task: np.ndarray) -> LotoReport:
    """Leave one task out, refitting lambda inside the training tasks each fold.

    The baseline in the denominator is the training tasks' mean, not the held-out
    task's own mean (9.4.5 step 2). Using the held-out mean would smuggle that
    task's information into the baseline and lift every R^2.
    """
    raw = np.asarray(raw, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    tasks = np.unique(task)
    if len(tasks) < 3:
        raise ValueError(
            f"leave-one-task-out needs at least three tasks to have an inner "
            f"grouped split; got {len(tasks)}"
        )

    folds = []
    for held in tasks:
        test = task == held
        train = ~test
        # The basis is fitted on the training tasks only: standardisation and the
        # SVD are part of the model, and deriving them from all frames would let
        # the held-out task shape the very axes it is then scored on.
        basis = truncate(raw[train])
        fit = fit_ridge(raw[train], u[train], task[train], basis=basis)
        residual = u[test] - fit.predict(raw[test])
        baseline = u[test] - u[train].mean(axis=0)
        folds.append(
            LotoFold(
                held_out=str(held),
                n_frames=int(test.sum()),
                r2=float(1.0 - np.sum(residual**2) / np.sum(baseline**2)),
                lam=fit.lam,
            )
        )
    return LotoReport(folds=tuple(folds), task_id_accuracy=task_id_accuracy(raw, task))


def task_id_accuracy(raw: np.ndarray, task: np.ndarray) -> float:
    """Multiclass accuracy of predicting the task id from phi, on held-out frames.

    9.4.5 asks for this number at zero cost: if phi recovers the task id almost
    perfectly then task and state are collinear, and the unidentifiability of a
    variance decomposition by task id stops being an argument and becomes a
    measurement. The classifier is a ridge on one-hot targets, scored by argmax,
    so it uses the same basis and the same regulariser family as the projection
    it is diagnosing.
    """
    tasks = np.unique(task)
    onehot = (task[:, None] == tasks[None, :]).astype(np.float64)
    # A frame-level split would put neighbouring frames of one episode on both
    # sides; the split is by parity of a frame's position within its task block,
    # which keeps both halves covering every task while separating the frames.
    order = np.concatenate([np.flatnonzero(task == t) for t in tasks])
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    train = rank % 2 == 0
    test = ~train

    basis = truncate(raw[train])
    z_train, z_test = basis.transform(raw[train]), basis.transform(raw[test])
    mu_z, mu_y = z_train.mean(axis=0), onehot[train].mean(axis=0)
    w = _ridge_solve(z_train - mu_z, onehot[train] - mu_y, 1.0)
    scores = (z_test - mu_z) @ w + mu_y
    return float(np.mean(tasks[np.argmax(scores, axis=1)] == task[test]))


# ---------------------------------------------------------------------------
# Standalone entry point (14.5 E-17: this gate runs before any training)
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", default=None, help="path to registry/lock.json")
    args = parser.parse_args()

    lock = LockFile(args.lock)
    decl = REQUIRED_MEASUREMENTS["phi.effective_rank"]
    if lock.has(decl):
        print(f"phi.effective_rank in the lock file: {lock.get(decl)}")
    else:
        print(
            "phi.effective_rank has not been measured yet. It is produced by "
            f"{decl.produced_by}, which builds a PhiInputs per cell from the ILC "
            "frames and calls project_report; this module supplies that call and "
            "holds no default for the rank."
        )


if __name__ == "__main__":
    _main()
