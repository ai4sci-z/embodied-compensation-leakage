"""One name per arm, so that `runners/` and `eval/` never import an arm module.

Everything in 10.4-10.6 that is a trainable policy has a name here: the main
architecture, the three E4 decomposition forms, the five E9 isolation levels, and
B1-B13. `build` takes the name and a `PolicyConfig` and returns a `Policy`; the arms
that have variants take them as keyword options, and an unknown option raises rather
than being ignored, because a silently dropped `variant=` would train the wrong arm
under the right name.

The names are the strings that appear in run directories and in the results tables, so
they are stable: `E9-I3prime` is the decisive control of 10.4.9 and must be greppable
next to `E9-I3`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from portal_trace.policies.baselines import (
    B1TargetOnlyPolicy,
    B2NaivePoolingPolicy,
    B3MorphologyGroupedPolicy,
    B4DisentanglementPolicy,
    B5DomainRandomisationPolicy,
    B6EmbodimentTokenPolicy,
    B7OraclePolicy,
    B8CounterfactualPolicy,
    B9ProprioMaskedPolicy,
    B10CuratedPoolingPolicy,
    B11ResidualCalibratorPolicy,
    B12LatentActionPolicy,
    B13LatentExtrinsicsPolicy,
)
from portal_trace.policies.decomposition import DecompositionForm, E4DecompositionPolicy
from portal_trace.policies.interface import Policy, PolicyConfig
from portal_trace.policies.isolation import IsolationLadder, IsolationLevel
from portal_trace.policies.portal_policy import PortalTracePolicy

__all__ = ["ARMS", "arm_names", "build"]

_Builder = Callable[[PolicyConfig, dict[str, Any]], Policy]


def _plain(cls: type[Policy]) -> _Builder:
    def build(cfg: PolicyConfig, options: dict[str, Any]) -> Policy:
        return cls(cfg, **options)

    return build


def _e4(form: DecompositionForm) -> _Builder:
    def build(cfg: PolicyConfig, options: dict[str, Any]) -> Policy:
        return E4DecompositionPolicy(cfg, form=form, **options)

    return build


def _e9(level: IsolationLevel) -> _Builder:
    def build(cfg: PolicyConfig, options: dict[str, Any]) -> Policy:
        if options:
            raise TypeError(
                f"the isolation ladder's {level.value} level takes no options; the "
                "levels differ only in routing key and scope, which is the property "
                "10.4.9 depends on"
            )
        return IsolationLadder(cfg).build(level)

    return build


ARMS: dict[str, _Builder] = {
    "portal-10.4.0": _plain(PortalTracePolicy),
    "E4-low-rank": _e4(DecompositionForm.LOW_RANK),
    "E4-physical-basis": _e4(DecompositionForm.PHYSICAL_BASIS),
    "E4-unconstrained-mlp": _e4(DecompositionForm.UNCONSTRAINED_MLP),
    **{f"E9-{level.value}": _e9(level) for level in IsolationLevel},
    "B1-target-only": _plain(B1TargetOnlyPolicy),
    "B2-naive-pooling": _plain(B2NaivePoolingPolicy),
    "B3-morphology-grouped": _plain(B3MorphologyGroupedPolicy),
    "B4-disentanglement": _plain(B4DisentanglementPolicy),
    "B5-domain-randomisation": _plain(B5DomainRandomisationPolicy),
    "B6-embodiment-token": _plain(B6EmbodimentTokenPolicy),
    "B7-oracle": _plain(B7OraclePolicy),
    "B8-counterfactual-decompensation": _plain(B8CounterfactualPolicy),
    "B9-proprio-masked": _plain(B9ProprioMaskedPolicy),
    "B10-curated-pooling": _plain(B10CuratedPoolingPolicy),
    "B11-residual-calibrator": _plain(B11ResidualCalibratorPolicy),
    "B12-latent-action-decoders": _plain(B12LatentActionPolicy),
    "B13-latent-extrinsics": _plain(B13LatentExtrinsicsPolicy),
}


def arm_names() -> tuple[str, ...]:
    return tuple(ARMS)


def build(name: str, cfg: PolicyConfig | None = None, **options: Any) -> Policy:
    """Build the arm called `name`.

    `cfg` defaults to `PolicyConfig(arm=name)`, which is the 10.4.0 architecture with
    every default at the value that section fixes; an arm that needs a different config
    -- B9 removes the proprioception pathway -- rewrites it itself, so a caller never has
    to know which.
    """
    try:
        builder = ARMS[name]
    except KeyError:
        raise KeyError(f"no arm {name!r}; the registry holds {sorted(ARMS)}") from None
    return builder(cfg if cfg is not None else PolicyConfig(arm=name), options)
