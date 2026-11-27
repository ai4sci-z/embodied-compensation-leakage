"""The A1-A14 ablation suite of proposal 10.6.1.

The ablation table is not a list of "things we could also try". Four of its rows
close a named confounder of 10.6.2 -- A3 closes X7, A8 closes X10, A11 closes X4
-- and two of them (A9, A14) are the same runs a main experiment already pays
for, read along a second axis. That is why the suite is configured with the same
schema as `experiments/main.yaml` rather than as a looser side file: an ablation
whose levels are not preregistered cannot close anything, because the level set
could be chosen after seeing the numbers.

Ten of the fourteen rows sweep a parameter that already exists somewhere in the
package, and their configuration entry points at that parameter by name:

===  =========================================================================
row  the parameterisation the levels move
===  =========================================================================
A1   `policies.obs:ObsSpec` -- `cameras` and `use_proprio`
A2   `policies.obs:K_HIST_LEVELS` -- the four history lengths, which
     `diagnostics.regimes:K_HIST_LADDER` reads for the label side
A7   `defects.surrogate:build_surrogate_set` -- the surrogate cardinality
A9   `teacher.ilc:DOSE_LADDER` -- the ILC iteration count, E3's third dose axis
A10  `policies.physics_basis:PSI_GROUPS` -- the physical basis of phi
A11  `configs.schema:SeedTable.seed_pilot_values` -- 3 / 5 / 8 seeds
A12  `diagnostics.leakage:LeakageAccumulator` -- the rho and c calibers
A14  `policies.losses:CLUBEstimator` / `MINEEstimator` and B4's lambda ladder
===  =========================================================================

The remaining four rows had no parameterisation at all, and this package is
where it now lives:

* `capacity` (A4) -- three visual trunks and two head depths, wired to the
  `architecture` string that `runners.train` carries into the model card. Split
  into `capacity` (what a level says) and `capacity_nets` (what it builds), so
  that the loader can check the level set without a torch import.
* `data_volume` (A5) -- the per-(config, task) demonstration budget, wired to
  the `n_target_demos` / `n_source_demos` columns of the arm's pool table.
* `solver` (A8) -- the `solref` / `solimp` groups, applied to the compiled
  `mjModel` after every reset, which is what closes X10.
* `teacher_family` (A13) -- the analytic compensation script and the contract the
  frozen RL expert is read under, beside `teacher.ilc`; `teacher_family_rl` is
  the training run that produces the expert. Split for `capacity`'s reason: the
  loader checks A13's level set and must not import stable-baselines3, gymnasium
  and torch to do it.

Two more rows had a level set that was declarable but not applicable, so they
have a small module each: `normalisation` (A3) and `granularity` (A6).

Nothing here supplies a value that the lock file owns, and nothing here has a
default level. `resolve_symbol` is the only entry point the configuration
loader uses, and it raises rather than returning `None`: a configuration naming
an implementation that is not there has to fail at load time, not at the point
where a run would otherwise have produced a number.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["AblationImplementationMissing", "resolve_symbol", "split_reference"]


class AblationImplementationMissing(ImportError):
    """An ablation names an implementation that is not importable.

    A separate type because the configuration loader has to distinguish it from
    an ordinary import failure inside the implementation itself: the first is a
    configuration error the reader can fix in the YAML, the second is a bug.
    """


def split_reference(reference: str) -> tuple[str, str]:
    """Split `module:symbol` into its two halves, refusing anything else."""
    module, sep, symbol = reference.partition(":")
    if not sep or not module.strip() or not symbol.strip():
        raise AblationImplementationMissing(
            f"{reference!r} is not a `module:symbol` reference. An ablation names "
            "the code object its levels move, so that a level set cannot drift "
            "away from the parameterisation it claims to sweep"
        )
    return module, symbol


def resolve_symbol(reference: str) -> Any:
    """Import `module:symbol` and return it.

    Used by the tests and by `portal-trace config --ablation`, not by
    `load_bundle`: resolving A4 would import torch and robomimic, and the
    configuration is text whose consistency 11.3 wants checkable before any
    heavy dependency is touched. The loader checks that the module's file is on
    disk; this checks that the symbol in it exists.
    """
    module_name, symbol = split_reference(reference)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise AblationImplementationMissing(
            f"{reference}: cannot import {module_name!r} ({exc})"
        ) from exc
    try:
        return getattr(module, symbol)
    except AttributeError:
        raise AblationImplementationMissing(
            f"{reference}: {module_name!r} has no attribute {symbol!r}. Present: "
            f"{sorted(n for n in vars(module) if not n.startswith('_'))}"
        ) from None
