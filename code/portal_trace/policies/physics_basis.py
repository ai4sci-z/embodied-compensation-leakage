"""Layout of the physical basis `psi_j` and the state feature basis `phi` (9.4.4).

The columns are not computed here. `psi_j` is `J(q) e_j`, `J^{-T} tau_g(q)` and
their sign-modulated relatives, all of which need the Jacobian of the live MuJoCo
model, so the teacher evaluates them and they arrive on the batch. What this module
owns is the *layout*: how many columns there are, which defect each group serves,
and the checks that a `psi_basis` tensor handed to a head is the basis and not
something else with the right rank.

Two numbers from 9.4.4 that a head has to size itself against:

* `psi` has 66 raw columns on a 7-DoF arm with three injected joints, and the
  arithmetic is `18 + 6 + 6 + 18 + 18`, one 6-dimensional vector field per column.
* `phi` has 392 raw columns, but the dimension a low-rank head actually sees is the
  **effective rank after the truncated SVD**, which 9.4.4 marks unverified and
  expects between 150 and 400. That number is a measurement, so nothing here has a
  default for it: a head takes `phi_dim` from the caller, who reads it off the
  measured basis it is about to feed in.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from portal_trace.policies.obs import ARM_ACTION_DIM

__all__ = [
    "PSI_GROUPS",
    "PsiGroup",
    "DEFAULT_INJECTED_JOINTS",
    "psi_column_count",
    "psi_group_slices",
    "PHI_RAW_COLUMNS",
    "PHI_RANK_RANGE",
    "validate_psi_basis",
]

# 9.2.0's default is three injected joints per defect.
DEFAULT_INJECTED_JOINTS = 3


@dataclass(frozen=True)
class PsiGroup:
    """One group of physical-basis columns.

    `per_joint` says whether the group has one 6-dimensional field per injected
    joint or a single one; that is the whole difference between the 18-column and
    the 6-column groups of 9.4.4.
    """

    name: str
    defects: tuple[str, ...]
    expression: str
    per_joint: bool

    def n_columns(self, n_injected_joints: int) -> int:
        return n_injected_joints if self.per_joint else 1


PSI_GROUPS: tuple[PsiGroup, ...] = (
    PsiGroup("calibration_offset", ("D1", "D10"), "J(q) e_j", True),
    PsiGroup("control_delay", ("D2",), "d/dt a_nom(x)", False),
    PsiGroup("gravity_sag", ("D3", "D3b"), "J(q)^{-T} tau_g(q)", False),
    PsiGroup("backlash_friction", ("D4", "D4hi", "D6"), "sign(qdot_j) J(q) e_j", True),
    PsiGroup("deadband", ("D5",), "sign(u_j) J(q) e_j", True),
)

# 9.4.4's own column count for phi, before the truncated SVD:
# 66 psi + 61 proprioception polynomial + 264 regime-indicator tensor product + 1.
PHI_RAW_COLUMNS = 392
# The expected range of the post-SVD effective rank. A range, not a value: 9.4.4
# marks it unverified and W1 measures it per cell.
PHI_RANK_RANGE = (150, 400)


def psi_column_count(n_injected_joints: int = DEFAULT_INJECTED_JOINTS) -> int:
    """Number of `psi_j` columns, 66 at the 9.4.4 default of three joints."""
    if n_injected_joints < 1:
        raise ValueError("a defect is injected on at least one joint")
    return sum(g.n_columns(n_injected_joints) for g in PSI_GROUPS)


def psi_group_slices(
    n_injected_joints: int = DEFAULT_INJECTED_JOINTS,
) -> dict[str, slice]:
    """Column ranges per group, so a coefficient vector can be read per defect.

    E10 reports `||delta_hat - delta|| / ||delta||`, which only means something if
    the coefficient a head learned can be matched to the defect whose basis it
    multiplies.
    """
    out: dict[str, slice] = {}
    start = 0
    for group in PSI_GROUPS:
        width = group.n_columns(n_injected_joints)
        out[group.name] = slice(start, start + width)
        start += width
    return out


def validate_psi_basis(
    psi: torch.Tensor, *, chunk: int, n_injected_joints: int = DEFAULT_INJECTED_JOINTS
) -> torch.Tensor:
    """Check a `psi_basis` batch: (B, CHUNK, n_psi, 6)."""
    expected = psi_column_count(n_injected_joints)
    per_group = " + ".join(
        str(g.n_columns(n_injected_joints) * ARM_ACTION_DIM) for g in PSI_GROUPS
    )
    if psi.ndim != 4:
        raise ValueError(
            f"psi_basis must be (B, CHUNK, n_psi, {ARM_ACTION_DIM}), got "
            f"{tuple(psi.shape)}"
        )
    if psi.shape[1] != chunk:
        raise ValueError(
            f"psi_basis covers {psi.shape[1]} chunk steps, the head predicts {chunk}"
        )
    if psi.shape[2] != expected:
        raise ValueError(
            f"psi_basis has {psi.shape[2]} columns; 9.4.4 gives {expected} for "
            f"{n_injected_joints} injected joints "
            f"({per_group}"
            f" = {expected * ARM_ACTION_DIM} scalar columns over "
            f"{ARM_ACTION_DIM}-dimensional fields)"
        )
    if psi.shape[3] != ARM_ACTION_DIM:
        raise ValueError(
            f"psi_basis fields are {psi.shape[3]}-dimensional; the physical basis "
            f"lives in the {ARM_ACTION_DIM}-dimensional task-space block of the action "
            "and carries no gripper column"
        )
    return psi
