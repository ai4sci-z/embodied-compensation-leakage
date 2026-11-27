"""Policy architectures for PORTAL-Trace: 10.4.0, the E4 heads, E9's ladder, B1-B13.

`registry.py` is the entry point a runner uses; this module re-exports the small
surface that `runners/` and `eval/` are written against.
"""

from __future__ import annotations

from portal_trace.policies.interface import (
    Batch,
    Observation,
    ParameterBlock,
    ParameterReport,
    Policy,
    PolicyConfig,
    PolicyOutput,
    RoutingKey,
)
from portal_trace.policies.obs import (
    ACTION_DIM,
    CHUNK,
    PHYSICAL_PROPRIO_DIM,
    PROPRIO_DIM,
    ObsSpec,
)
from portal_trace.policies.portal_policy import PortalTracePolicy
from portal_trace.policies.registry import ARMS, arm_names, build
from portal_trace.policies.schedule import EpisodeLength, OptimSpec

__all__ = [
    "ACTION_DIM",
    "ARMS",
    "CHUNK",
    "PHYSICAL_PROPRIO_DIM",
    "PROPRIO_DIM",
    "Batch",
    "EpisodeLength",
    "Observation",
    "ObsSpec",
    "OptimSpec",
    "ParameterBlock",
    "ParameterReport",
    "Policy",
    "PolicyConfig",
    "PolicyOutput",
    "PortalTracePolicy",
    "RoutingKey",
    "arm_names",
    "build",
]
