"""PORTAL-Trace: compensation leakage in cross-embodiment imitation learning.

The package layout follows proposal 11.2 verbatim -- `defects/`, `registry/`,
`teacher/`, `diagnostics/`, `configs/`, `eval/`, `real/`, `cards/`, `env/` are
the delivered artefacts and keep their names. Four directories are additions
that 11.2 does not cover because it lists deliverables rather than the machinery
needed to run them: `policies/` (the architecture of 10.4.0 and the thirteen
baselines of 10.5), `runners/` (turns the declarative configs into runs),
`scripts/` (the calibration entry points the proposal names by filename), and
the top-level `tests/`.

Three mechanisms cut across everything and are worth knowing before reading any
module:

1. `portal_trace.lockfile` -- every 待核实 constant is declared once and can only
   be read out of the lock file. Nothing in this package contains a default for
   one of them.
2. `portal_trace.diagnostics.units` -- amplitude and energy are different types,
   and the design graph will not accept a raw frame count as an edge weight.
3. `portal_trace.defects` -- the clean twin is `apply(delta)` evaluated at
   delta = 0, sharing every line of code with the defective path. There is no
   `if defect is None` shortcut anywhere.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Proposal 9.1.1: these are pinned rather than bounded, because bit-exact
# reproduction is a stated requirement and MuJoCo makes no cross-version
# floating point promise. env.versions checks the running interpreter against
# them and refuses rather than warning.
PINNED_VERSIONS: dict[str, str] = {
    "mujoco": "3.3.2",
    "robosuite": "1.5.2",
    "robomimic": "0.5.0",
}

__all__ = ["__version__", "PINNED_VERSIONS"]
