"""Evaluation rollouts (proposal 10.7, 11.3, appendix A-1, A-29).

Two design decisions are enforced here rather than described.

The initial-state list is shared across arms. 11.3 aligns the environment and
evaluation seed layers across arms and varies only the training layer, so the
paired difference carries no initial-state difference. Appendix A-29 draws the
consequence: rollout-layer fluctuation is then perfectly correlated across arms,
so a bootstrap that resamples rollouts per arm independently treats a correlated
component as independent and inflates the paired difference's variance. The
resampling unit is therefore the initial-state *index*, shared by every arm, and
the layer is named the shared-initial-state layer.

The deployment target is always a defective embodiment. Appendix A-1 excludes
defining harm on transfer to a delta = 0 target: no real robot can be declared
defect-free, so such a definition can never be instantiated on hardware. The
clean twin appears only as B7 Oracle's reference point.

Absolute success rates carry a systematic error the paired CIs do not: with 50
fixed initial states, state-sampling error at p = 0.5 is about 7 pp and is a
common offset across all arms. `EvaluationResult` reports it separately so the
number cannot be quoted without it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from portal_trace.configs.schema import ConfigError
from portal_trace.policies.interface import Observation, PolicyConfig
from portal_trace.runners.backends import RolloutResult, RolloutSpec
from portal_trace.runners.common import RunContext
from portal_trace.runners.formal_checkpoint import load_deployment_checkpoint

__all__ = [
    "EvaluationUnavailable",
    "EvaluationTarget",
    "EvaluationResult",
    "DeploymentObservationAdapter",
    "EvaluationBackend",
    "shared_initial_states",
    "paired_bootstrap",
    "evaluate",
    "run_engineering_evaluation_fixture",
]


class EvaluationUnavailable(RuntimeError):
    """The custom policy deployment adapter is not available yet."""


@runtime_checkable
class DeploymentObservationAdapter(Protocol):
    """Translate backend observations and policy chunks at the deployment edge.

    Backends expose simulator-native observations, while :class:`Policy` takes
    the typed five-frame ``Observation`` and returns an eight-step action chunk.
    The adapter owns that representation boundary; it must not alter values or
    silently select a different history/chunk caliber.
    """

    def to_observation(self, raw_observation: Any, *, t: int) -> Observation: ...

    def to_action(self, action_chunk: np.ndarray, *, t: int) -> np.ndarray: ...


@runtime_checkable
class EvaluationBackend(Protocol):
    """Minimal injected backend surface used by the engineering fixture."""

    backend: Any

    def probe(self) -> None: ...

    def rollout(self, spec: RolloutSpec, wrapper: Any, policy: Any) -> RolloutResult: ...


@dataclass(frozen=True)
class EvaluationTarget:
    """Where a policy is deployed.

    `defect` is not optional. Appendix A-1: harm is defined on the mismatch
    field between two real embodiments, and no endpoint in the plan is defined
    on a delta = 0 deployment target.
    """

    family: str
    defect: str
    level: int
    task: str

    def __post_init__(self) -> None:
        if not self.defect or self.defect in ("clean", "none", "delta0"):
            raise ConfigError(
                "the deployment target must carry a defect. Appendix A-1 excludes "
                "building harm on transfer to a clean embodiment: assumption (A7) "
                "says no real robot can be declared defect-free, so an endpoint "
                "defined on a delta = 0 target can never be instantiated on "
                "hardware. The clean twin is B7 Oracle's reference point, not a "
                "deployment target"
            )


@dataclass(frozen=True)
class EvaluationResult:
    """One arm's rollouts on one target."""

    arm_id: str
    target: EvaluationTarget
    success: np.ndarray               # (n_init,) bool, indexed by shared init state
    peak_cfrc_ext: np.ndarray         # (n_init,) N, the 10.7 safety endpoint
    init_state_indices: tuple[int, ...]
    eval_seed: int

    def __post_init__(self) -> None:
        if len(self.success) != len(self.init_state_indices):
            raise ValueError(
                "the success vector is indexed by the shared initial-state list; "
                f"got {len(self.success)} rollouts for "
                f"{len(self.init_state_indices)} states"
            )

    @property
    def success_rate(self) -> float:
        return float(np.mean(self.success))

    @property
    def absolute_rate_common_offset(self) -> float:
        """The state-sampling error on an absolute rate, in percentage points.

        Common to every arm, so it cancels in a paired difference and does not
        cancel in an absolute statement. Appendix A-29 requires it to be carried
        alongside any "arm X reaches Y%" sentence.
        """
        n = len(self.success)
        return 100.0 * math.sqrt(0.25 / n)


def shared_initial_states(ctx: RunContext, *, task: str, family: str) -> tuple[int, ...]:
    """The evaluation initial-state list, identical for every arm.

    Derived from the evaluation seed layer, which 11.3 scopes by (task, family)
    and explicitly not by arm. That is what makes the difference between two
    arms free of an initial-state difference.
    """
    seed = ctx.bundle.seeds.eval_seed(task=task, family=family)
    n = ctx.bundle.seeds.n_eval_init_states
    rng = np.random.default_rng(seed)
    return tuple(int(x) for x in rng.integers(0, 2**31 - 1, size=n))


def paired_bootstrap(
    a: EvaluationResult,
    b: EvaluationResult,
    *,
    n_resamples: int,
    rng: np.random.Generator,
) -> tuple[float, tuple[float, float]]:
    """Paired difference a - b with a bootstrap CI over shared state indices.

    Appendix A-29: the two arms are resampled with one and the same index draw.
    Resampling each arm independently would treat a perfectly correlated
    component as independent, overstate the paired difference's variance, and
    put the CI out of step with the MDE table.
    """
    if a.init_state_indices != b.init_state_indices:
        raise ConfigError(
            f"arms {a.arm_id} and {b.arm_id} were evaluated on different initial-state "
            "lists, so their difference is not paired. 11.3 aligns the evaluation seed "
            "layer across arms precisely so that this cannot happen"
        )
    n = len(a.success)
    point = a.success_rate - b.success_rate
    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)     # one index draw, used by both arms
        draws[i] = np.mean(a.success[idx]) - np.mean(b.success[idx])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, (float(lo), float(hi))


def evaluate(
    ctx: RunContext,
    *,
    checkpoint: Path,
    target: EvaluationTarget,
) -> EvaluationResult:
    """Roll one checkpoint out on one defective target.

    The custom PORTAL-Trace policy/checkpoint contract is not implemented yet.
    Keeping this boundary explicit is important: the old implementation called
    stock robosuite geometry and checkpoint helpers that do not exist in the
    current codebase, which could never produce a valid evaluation.  Once the
    custom adapter is complete, this function is the single place where it is
    allowed to construct a deployment policy and hand it to the backend.
    """
    req = ctx.request
    if req.arm_id is None:
        raise ConfigError("evaluation needs --arm, so the result can be attributed")
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint {checkpoint} does not exist")

    raise EvaluationUnavailable(
        "evaluation is intentionally blocked: the custom PORTAL-Trace checkpoint "
        "loader and deployment policy adapter are not implemented. A stock "
        "robomimic checkpoint cannot satisfy the 5-frame/8-step/chunked policy "
        "contract, so no rollout or endpoint is produced"
    )


def run_engineering_evaluation_fixture(
    ctx: RunContext,
    *,
    checkpoint: Path,
    plan_path: Path,
    normalization_path: Path,
    policy_config: PolicyConfig,
    target: EvaluationTarget,
    spec: RolloutSpec,
    backend: EvaluationBackend,
    wrapper: Any,
    observation_adapter: DeploymentObservationAdapter,
    eval_seed: int,
) -> EvaluationResult:
    """Exercise deployment admission and a caller-supplied backend in a fixture.

    This is intentionally separate from :func:`evaluate`: it never becomes a
    formal result path and it does not write artifacts.  The backend and
    observation adapter are injected so tests can verify the policy boundary
    without pretending that a CPU MuJoCo rollout is available.  A real backend
    still has to pass ``probe`` and provide all required endpoint fields; any
    unavailable backend or incomplete ``RolloutResult`` fails closed.
    """
    if not isinstance(eval_seed, int) or isinstance(eval_seed, bool):
        raise TypeError("engineering evaluation fixture eval_seed must be an integer")
    if spec.task_id != target.task or spec.family != target.family:
        raise ConfigError(
            "engineering evaluation fixture RolloutSpec does not match its target task/family"
        )
    if not isinstance(observation_adapter, DeploymentObservationAdapter):
        raise TypeError("observation_adapter does not implement DeploymentObservationAdapter")
    if not isinstance(backend, EvaluationBackend):
        raise TypeError("backend does not implement EvaluationBackend")

    req = ctx.request
    if req.arm_id is None:
        raise ConfigError("evaluation needs --arm, so the result can be attributed")
    loaded = load_deployment_checkpoint(
        checkpoint,
        plan_path=plan_path,
        normalization_path=normalization_path,
        policy_config=policy_config,
        expected_arm_id=req.arm_id,
    )

    def policy(raw_observation: Any, t: int) -> np.ndarray:
        observation = observation_adapter.to_observation(raw_observation, t=t)
        action_chunk = loaded.act(observation)
        action = observation_adapter.to_action(action_chunk, t=t)
        action = np.asarray(action)
        expected = (observation.batch_size, policy_config.action_dim)
        if action.shape != expected:
            raise ValueError(
                f"deployment observation adapter returned {action.shape}, expected {expected}"
            )
        if not np.isfinite(action).all():
            raise ValueError("deployment observation adapter returned non-finite actions")
        return action

    backend.probe()
    result = backend.rollout(spec, wrapper, policy)
    if not isinstance(result, RolloutResult):
        raise TypeError("evaluation backend returned a non-RolloutResult")
    if result.backend is not backend.backend:
        raise ConfigError(
            "evaluation backend returned a RolloutResult from a different backend; "
            "cross-backend endpoint assembly is forbidden"
        )
    if result.success.shape != (spec.n_env,) or result.t_success.shape != (spec.n_env,):
        raise ValueError("evaluation backend returned malformed success vectors")
    peak_raw = result.observations.get("peak_cfrc_ext")
    if peak_raw is None:
        raise EvaluationUnavailable(
            "evaluation backend did not provide peak_cfrc_ext; the fixture cannot "
            "stand in for the required rollout safety endpoint"
        )
    peak = np.asarray(peak_raw, dtype=float)
    if peak.ndim == 2 and peak.shape[0] == spec.n_env:
        peak = np.max(peak, axis=1)
    if peak.shape != (spec.n_env,) or not np.isfinite(peak).all():
        raise ValueError("evaluation backend returned malformed peak_cfrc_ext")
    return EvaluationResult(
        arm_id=req.arm_id,
        target=target,
        success=np.asarray(result.success, dtype=bool),
        peak_cfrc_ext=peak,
        init_state_indices=tuple(spec.env_seeds),
        eval_seed=eval_seed,
    )
