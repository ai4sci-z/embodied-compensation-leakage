"""Declarative configuration for P0/P1, E1-E12 and B1-B13 (proposal 11.2).

What is in here:

- `seeds.yaml`      the three-layer seed allocation of 11.3.
- `subsets.yaml`    the five named subsets of 10.4.0 and the per-arm pool
                    composition, including the mandatory delta = 0 flag.
- `arms.yaml`       the thirteen baselines, the E9 ladder rungs and the
                    experiment-local arms, each with its design (M, D, S).
- `isolation/`      the E9 isolation ladder I0-I3 and the decisive control I3'.
- `preregistration/` the 16-item preregistration file, its git tag and the
                    append-only amendment log.
- `experiments/`    P0, P1, E1-E12 and the baseline block.

`loader.load_bundle()` reads all of it and cross-checks it; `validator.validate`
refuses to start a run whose configuration is incomplete. Nothing in this
package supplies a default for a preregistered or measured constant.
"""

from __future__ import annotations

from portal_trace.configs.loader import ConfigBundle, config_root, load_bundle
from portal_trace.configs.preregistration import (
    Amendment,
    AmendmentLog,
    Preregistration,
    PreregistrationIncomplete,
    PreregistrationNotTagged,
)
from portal_trace.configs.schema import (
    ArmSpec,
    ConfigError,
    Design,
    EndpointSpec,
    ExperimentSpec,
    IsolationLadder,
    IsolationRung,
    PoolComposition,
    SeedLayer,
    SeedTable,
    SubsetSpec,
)
from portal_trace.configs.validator import (
    PreflightError,
    PreflightReport,
    RunRequest,
    validate,
)

__all__ = [
    "PreflightError",
    "PreflightReport",
    "RunRequest",
    "validate",
    "ConfigBundle",
    "config_root",
    "load_bundle",
    "ConfigError",
    "Design",
    "SeedLayer",
    "SeedTable",
    "SubsetSpec",
    "PoolComposition",
    "EndpointSpec",
    "ArmSpec",
    "ExperimentSpec",
    "IsolationRung",
    "IsolationLadder",
    "Preregistration",
    "PreregistrationIncomplete",
    "PreregistrationNotTagged",
    "Amendment",
    "AmendmentLog",
]
