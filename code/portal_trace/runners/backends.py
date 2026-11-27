"""The three physics backends of 9.1.2 and 9.8.4, and degradation branch 0.

CPU MuJoCo is the primary caliber. MJX is degradation branch 0, triggered when
the overrun factor exceeds 4 and taken *before* any experiment arm is cut and
*before* the demonstration collection stage. Isaac Lab is the third-party
cross-check and carries no primary endpoint.

None of the three is a fallback for another. 9.1.2's caliber discipline is that
the numbers entering one endpoint definition must all come from one backend --
MJX and CPU MuJoCo have different floating point paths and bit-exact
reproduction (9.1.6) holds only inside one backend. So a requested backend that
cannot be probed raises `BackendUnavailable`; it never quietly becomes a
different backend, because that would corrupt a result in a way nothing
downstream could see.

What is and is not portable (9.1.2). Only one thing genuinely pins load to the
CPU: the *fallback* caliber for D5 and D8 needs `gaintype`/`biastype = USER`,
which MJX does not have, and the matching `mjcb_act_gain` / `mjcb_act_bias`
callbacks are process-global and not part of `mjModel`. The primary caliber does
not use them -- it rewrites `sim.data.ctrl` in `_pre_action` -- so the
pure-physics load is vmappable. The six pieces of wrapper state (action buffer,
hysteresis state machine, drift accumulator, ctrl rewrite, sensing twin,
`qfrc_applied` write) become per-environment carry state under `vmap`, handled
exactly like a policy's recurrent state.

This module owns the driving loop and the carry-state layout. It does not own
the defect injection itself: `defects.inject` supplies the wrapper operations
through the `WrapperOps` / `JaxWrapperOps` protocols below, so that the same
injection logic drives both backends rather than being written twice.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

import numpy as np

from portal_trace.env.versions import (
    Backend,
    BackendUnavailable,
    EnvironmentFingerprint,
    check_pinned_versions,
    fingerprint,
)
from portal_trace.registry.tasks import CONTROL_FREQ, HORIZON, SIMULATION_TIMESTEP
from portal_trace.success import terminal_hold_start

__all__ = [
    "WrapperCarry",
    "WrapperOps",
    "JaxWrapperOps",
    "RolloutSpec",
    "RolloutResult",
    "PhysicsBackend",
    "CpuMujocoBackend",
    "MjxBackend",
    "IsaacLabBackend",
    "get_backend",
    "BackendMixingError",
    "degradation_branch_zero",
    "BranchZeroOrderingError",
]


class BackendMixingError(RuntimeError):
    """Two backends contributed to one endpoint.

    9.1.2: "whatever backend it runs on, the numbers entering an endpoint
    definition must come from the same backend". Cross-backend comparison is
    permitted for exactly one thing, a direction-cosine consistency check, and
    that is a separate quantity with its own name.
    """


class BranchZeroOrderingError(RuntimeError):
    """Degradation branch 0 was reached in the wrong order.

    9.8.4 puts the MJX port before any arm is cut and requires it to land as one
    block after S1 and before S2. Porting after collection throws away the
    329 CPU-h already spent, because 9.1.2 forbids mixing the two backends'
    demonstrations.
    """


# ---------------------------------------------------------------------------
# per-environment carry state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WrapperCarry:
    """The wrapper-layer state that travels with each environment (9.1.2).

    Three of the six wrapper pieces hold state across control steps and are
    therefore carry: the action buffer (D2's control-step action-side delay),
    the hysteresis state machine (D4lo's play operator) and the drift
    accumulator (D10's thermal drift). The other three -- the ctrl rewrite, the
    sensing twin, the `qfrc_applied` write -- are pure functions of the current
    step and need no carry.

    Laid out with a leading environment axis in both backends so that the CPU
    loop and the `vmap`ed MJX version index it identically; a layout difference
    here is the kind of thing that would make the bit-exactness regression of
    9.8.4 fail for a reason unrelated to physics.
    """

    action_buffer: np.ndarray       # (n_env, n_delay_steps, action_dim)
    engagement: np.ndarray          # (n_env, n_dof) backlash engagement state
    drift: np.ndarray               # (n_env, n_dof) thermal drift accumulator
    step_index: np.ndarray          # (n_env,) control steps elapsed since reset

    def __post_init__(self) -> None:
        n = self.action_buffer.shape[0]
        for name in ("engagement", "drift", "step_index"):
            arr = getattr(self, name)
            if arr.shape[0] != n:
                raise ValueError(
                    f"WrapperCarry.{name} has leading dimension {arr.shape[0]}, "
                    f"expected the environment count {n}"
                )

    @property
    def n_env(self) -> int:
        return int(self.action_buffer.shape[0])

    @classmethod
    def zeros(
        cls, *, n_env: int, n_delay_steps: int, action_dim: int, n_dof: int
    ) -> WrapperCarry:
        return cls(
            action_buffer=np.zeros((n_env, n_delay_steps, action_dim), dtype=np.float64),
            engagement=np.zeros((n_env, n_dof), dtype=np.float64),
            drift=np.zeros((n_env, n_dof), dtype=np.float64),
            step_index=np.zeros((n_env,), dtype=np.int64),
        )


@runtime_checkable
class WrapperOps(Protocol):
    """What `defects.inject` supplies to the CPU backend.

    `pre_action` runs at the control step, before the controller sees the
    action; `pre_substep` runs at the 500 Hz physics hook, between `step1` and
    `step2`, which is where `qfrc_applied` has to be written.
    """

    n_delay_steps: int

    def init_carry(self, n_env: int) -> WrapperCarry: ...

    def pre_action(
        self, carry: WrapperCarry, action: np.ndarray, observation: dict[str, np.ndarray]
    ) -> tuple[WrapperCarry, np.ndarray]: ...

    def pre_substep(self, carry: WrapperCarry, sim: Any) -> WrapperCarry: ...


@runtime_checkable
class JaxWrapperOps(Protocol):
    """What `defects.inject` supplies to the MJX backend.

    The same six pieces of logic rewritten as pure functions of (carry, input),
    which is the whole content of the 80-120 person-hour port estimate in 9.8.4.
    The signature differs from `WrapperOps` on purpose: pretending one interface
    serves both would hide the fact that the port has to be regression-tested
    piece by piece against the CPU version.
    """

    n_delay_steps: int

    def init_carry(self, n_env: int) -> Any: ...

    def pre_action(self, carry: Any, action: Any, data: Any) -> tuple[Any, Any]: ...

    def pre_substep(self, carry: Any, data: Any) -> tuple[Any, Any]: ...


# ---------------------------------------------------------------------------
# rollout description
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RolloutSpec:
    """One batch of rollouts on one (embodiment, delta) configuration.

    The batch axis is initial states at fixed delta. 9.1.3 makes the CPU
    parallel unit a process, one process per (embodiment, delta), so that the
    `mjcb_*` fallback caliber cannot cross-talk if it is ever switched on; on
    MJX the same axis is the `vmap` axis.
    """

    task_id: str
    env_name: str
    env_kwargs: dict[str, Any]
    family: str
    env_seeds: tuple[int, ...]
    horizon: int = HORIZON
    control_freq: int = CONTROL_FREQ
    simulation_timestep: float = SIMULATION_TIMESTEP
    render: bool = False
    lite_physics: bool = True

    def __post_init__(self) -> None:
        if self.horizon != HORIZON:
            raise ValueError(
                f"horizon is fixed at {HORIZON} with early stopping off (9.6.1); "
                f"got {self.horizon}. Variable-length episodes make the frame-level "
                "mixing weights incomparable across cells and break alpha_frame == "
                "alpha_demo by construction"
            )
        if "horizon" in self.env_kwargs and self.env_kwargs["horizon"] != self.horizon:
            raise ValueError(
                f"env_kwargs horizon {self.env_kwargs['horizon']} disagrees with the "
                f"rollout horizon {self.horizon}"
            )
        if not self.env_seeds:
            raise ValueError("a rollout batch needs at least one environment seed")

    @property
    def n_env(self) -> int:
        return len(self.env_seeds)

    @property
    def robosuite_env_kwargs(self) -> dict[str, Any]:
        """Task kwargs excluding values owned explicitly by the rollout contract."""
        kwargs = dict(self.env_kwargs)
        kwargs.pop("horizon", None)
        return kwargs

    @property
    def substeps_per_control_step(self) -> int:
        """25 at 0.002 s and 20 Hz -- the 500 Hz injection hook is real."""
        n = (1.0 / self.control_freq) / self.simulation_timestep
        if abs(n - round(n)) > 1e-9:
            raise ValueError(
                f"control period {1 / self.control_freq} s is not an integer multiple "
                f"of the simulation timestep {self.simulation_timestep} s; the "
                "500 Hz injection hook would land between substeps"
            )
        return int(round(n))


@dataclass(frozen=True)
class RolloutResult:
    """States, actions and observations for one batch, plus its provenance."""

    states: np.ndarray                      # (n_env, T, nq + nv)
    actions: np.ndarray                     # (n_env, T, action_dim)
    observations: dict[str, np.ndarray]     # each (n_env, T, ...)
    success: np.ndarray                     # (n_env,) bool
    t_success: np.ndarray                   # (n_env,) terminal-run start, else -1
    carry_trace: np.ndarray                 # (n_env, T, S) the wrapper memory state z
    backend: Backend
    environment: EnvironmentFingerprint

    def __post_init__(self) -> None:
        n, t = self.states.shape[0], self.states.shape[1]
        if self.actions.shape[:2] != (n, t):
            raise ValueError(
                f"actions has shape {self.actions.shape[:2]}, states has ({n}, {t})"
            )
        if self.environment.backend is not self.backend:
            raise BackendMixingError(
                f"result claims backend {self.backend} but its fingerprint records "
                f"{self.environment.backend}"
            )


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------


class PhysicsBackend(ABC):
    """Common surface. `probe` is the only way to find out whether a backend is
    usable, and it raises rather than returning False so that a caller cannot
    accidentally treat "unavailable" as "try the other one"."""

    backend: Backend

    @abstractmethod
    def probe(self) -> None:
        """Raise `BackendUnavailable` unless this backend can run here."""

    @abstractmethod
    def rollout(self, spec: RolloutSpec, wrapper: Any, policy: Any) -> RolloutResult:
        """Run one batch. `policy(observation, t) -> (n_env, action_dim)`."""

    def fingerprint(self, *, lite_physics: bool = True) -> EnvironmentFingerprint:
        return fingerprint(self.backend, lite_physics=lite_physics)


class CpuMujocoBackend(PhysicsBackend):
    """The primary caliber (9.1.2).

    Off-screen rendering and the `mjcb_*` fallback caliber both need it, and
    9.8's budget is stated on this backend alone.
    """

    backend = Backend.CPU_MUJOCO

    def probe(self) -> None:
        try:
            check_pinned_versions(required=("mujoco", "robosuite"))
        except Exception as exc:
            raise BackendUnavailable(
                f"CPU MuJoCo is the primary caliber and is not usable here: {exc}"
            ) from exc

    def rollout(self, spec: RolloutSpec, wrapper: WrapperOps, policy: Any) -> RolloutResult:
        self.probe()
        import robosuite

        if not isinstance(wrapper, WrapperOps):
            raise TypeError(
                "the CPU backend needs the numpy-side wrapper operations from "
                "portal_trace.defects.inject; got "
                f"{type(wrapper).__name__}, which does not implement WrapperOps"
            )

        env = robosuite.make(
            spec.env_name,
            robots=spec.family,
            has_renderer=False,
            has_offscreen_renderer=spec.render,
            use_camera_obs=spec.render,
            control_freq=spec.control_freq,
            horizon=spec.horizon,
            ignore_done=True,
            **spec.robosuite_env_kwargs,
        )
        try:
            return self._run(env, spec, wrapper, policy)
        finally:
            env.close()

    def _run(
        self, env: Any, spec: RolloutSpec, wrapper: WrapperOps, policy: Any
    ) -> RolloutResult:
        # robosuite's hard_reset default is True, so every model-level write has
        # to be re-applied after each reset; the wrapper's init_carry is called
        # per episode for the same reason.
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        traces: list[np.ndarray] = []
        successes: list[bool] = []
        t_successes: list[int] = []
        observations: dict[str, list[np.ndarray]] = {}

        for env_index, seed in enumerate(spec.env_seeds):
            np.random.seed(seed)
            obs = env.reset()
            carry = wrapper.init_carry(1)
            ep_states = np.empty((spec.horizon, env.sim.get_state().flatten().size))
            ep_actions = np.empty((spec.horizon, env.action_dim))
            ep_trace = np.empty((spec.horizon, _carry_width(carry)))
            ep_peak_cfrc_ext = np.empty(spec.horizon, dtype=np.float64)
            ep_obs: dict[str, list[np.ndarray]] = {}
            ep_success_trace = np.empty(spec.horizon, dtype=bool)

            for t in range(spec.horizon):
                ep_states[t] = env.sim.get_state().flatten()
                raw_action = np.asarray(
                    policy(obs, t), dtype=np.float64
                ).reshape(1, env.action_dim)
                carry, action = wrapper.pre_action(carry, raw_action, obs)
                ep_actions[t] = action[0]
                ep_trace[t] = _flatten_carry(carry)
                obs, _reward, _done, _info = env.step(action[0])
                # Keep the safety observable on the same frame axis as the
                # native observations. Evaluation reduces this trace to one
                # peak per episode; dropping it here would make a rollout
                # impossible to audit against endpoint 10.7.
                ep_peak_cfrc_ext[t] = float(
                    np.max(np.linalg.norm(env.sim.data.cfrc_ext, axis=1))
                )
                for key, value in obs.items():
                    ep_obs.setdefault(key, []).append(np.asarray(value))
                # Early stopping remains off. Success requires the task predicate
                # to hold through the fixed terminal window, so a transient
                # tolerance crossing does not become a successful rollout.
                ep_success_trace[t] = bool(env._check_success())

            t_success = terminal_hold_start(ep_success_trace)
            states.append(ep_states)
            actions.append(ep_actions)
            traces.append(ep_trace)
            successes.append(t_success >= 0)
            t_successes.append(t_success)
            for key, frames in ep_obs.items():
                observations.setdefault(key, []).append(np.stack(frames))
            observations.setdefault("peak_cfrc_ext", []).append(ep_peak_cfrc_ext)
            del env_index

        return RolloutResult(
            states=np.stack(states),
            actions=np.stack(actions),
            observations={k: np.stack(v) for k, v in observations.items()},
            success=np.array(successes, dtype=bool),
            t_success=np.array(t_successes, dtype=np.int64),
            carry_trace=np.stack(traces),
            backend=self.backend,
            environment=self.fingerprint(lite_physics=spec.lite_physics),
        )


def _carry_width(carry: WrapperCarry) -> int:
    return (
        carry.action_buffer.shape[1] * carry.action_buffer.shape[2]
        + carry.engagement.shape[1]
        + carry.drift.shape[1]
        + 1
    )


def _flatten_carry(carry: WrapperCarry) -> np.ndarray:
    """The `defect_state` column of 9.6.3: the wrapper memory state z, per frame."""
    return np.concatenate(
        [
            carry.action_buffer[0].reshape(-1),
            carry.engagement[0],
            carry.drift[0],
            carry.step_index[:1].astype(np.float64),
        ]
    )


class MjxBackend(PhysicsBackend):
    """Degradation branch 0 (9.1.2, 9.8.4).

    The pure-physics load -- ILC iterations during collection, cross replay, the
    P0 level sweep, the physics half of evaluation rollouts -- is a batch of
    identical models differing only in a per-environment constant array, so it
    `vmap`s over the initial-state axis and `scan`s over time. The wrapper's
    three stateful pieces ride along as carry, exactly as a policy's recurrent
    state would.

    Not covered, and stated here rather than discovered later: image
    materialisation, the rendering half of evaluation, the teleoperation pilot,
    and the `mjcb_*` fallback caliber. Those stay on CPU MuJoCo, and 9.1.2
    forbids feeding both into one endpoint, so a port means recollecting every
    demonstration.
    """

    backend = Backend.MJX

    def probe(self) -> None:
        missing = [
            name for name in ("jax", "mujoco.mjx") if not _importable(name)
        ]
        if missing:
            raise BackendUnavailable(
                "MJX is degradation branch 0 of 9.8.4 and is not installed here: "
                f"{missing} cannot be imported.\n"
                "Install jax with the accelerator plugin for the target device and "
                "mujoco>=3.2.3 (frictionloss support in MJX arrives there), then "
                "re-run. Do not fall back to CPU MuJoCo for part of a quantity: "
                "9.1.2 forbids mixing backends inside one endpoint, and the two "
                "floating point paths differ, so a mixed quantity is wrong in a way "
                "nothing downstream can detect."
            )
        check_pinned_versions(required=("mujoco",))

    def rollout(self, spec: RolloutSpec, wrapper: JaxWrapperOps, policy: Any) -> RolloutResult:
        self.probe()
        if spec.render:
            raise BackendUnavailable(
                "MJX cannot render: 9.1.2 keeps off-screen rendering on the CPU "
                "stack. Materialise images with the CPU backend from the stored "
                "states instead of rendering during the rollout"
            )
        import jax
        import jax.numpy as jnp
        from mujoco import mjx

        mj_model = self._compile_model(spec)
        mjx_model = mjx.put_model(mj_model)
        n_substeps = spec.substeps_per_control_step

        def one_control_step(state, t):
            data, carry, key = state
            obs = _mjx_observation(data)
            raw_action = policy(obs, t)
            carry, action = wrapper.pre_action(carry, raw_action, data)
            data = data.replace(ctrl=action)

            def one_substep(inner, _):
                d, c = inner
                # The injection hook lands between step1 and step2, which is
                # where qfrc_applied has to be written (9.1.5). mjx.step is the
                # fused pair, so the write happens immediately before it.
                c, d = wrapper.pre_substep(c, d)
                return (mjx.step(mjx_model, d), c), None

            (data, carry), _ = jax.lax.scan(
                one_substep, (data, carry), None, length=n_substeps
            )
            recorded = jnp.concatenate([data.qpos, data.qvel])
            return (data, carry, key), (recorded, action)

        def rollout_one(seed):
            key = jax.random.PRNGKey(seed)
            data = mjx.make_data(mjx_model)
            carry = wrapper.init_carry(1)
            (_, final_carry, _), (states, actions) = jax.lax.scan(
                one_control_step,
                (data, carry, key),
                jnp.arange(spec.horizon),
            )
            return states, actions, final_carry

        batched = jax.vmap(rollout_one)
        states, actions, final_carry = batched(jnp.asarray(spec.env_seeds))
        states_np = np.asarray(states)
        actions_np = np.asarray(actions)
        return RolloutResult(
            states=states_np,
            actions=actions_np,
            observations={},
            success=np.zeros(spec.n_env, dtype=bool),
            t_success=np.full(spec.n_env, -1, dtype=np.int64),
            carry_trace=np.asarray(jax.tree_util.tree_leaves(final_carry)[0]).reshape(
                spec.n_env, 1, -1
            ),
            backend=self.backend,
            environment=self.fingerprint(lite_physics=spec.lite_physics),
        )

    @staticmethod
    def _compile_model(spec: RolloutSpec) -> Any:
        """Build the MuJoCo model MJX will batch over.

        robosuite builds the MJCF, so the model still comes from the CPU stack;
        what moves to MJX is the stepping, not the model construction.
        """
        import robosuite

        env = robosuite.make(
            spec.env_name,
            robots=spec.family,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=spec.control_freq,
            horizon=spec.horizon,
            ignore_done=True,
            **spec.robosuite_env_kwargs,
        )
        try:
            return env.sim.model._model
        finally:
            env.close()


def _mjx_observation(data: Any) -> Any:
    """The observation the policy sees on the MJX path.

    Deliberately the raw physics state rather than robosuite's observation dict:
    robosuite's observable machinery is Python-side and not traceable, so the
    port has to reconstruct whatever the policy needs from qpos/qvel. Which
    channels those are is part of the per-piece regression 9.8.4 requires.
    """
    import jax.numpy as jnp

    return jnp.concatenate([data.qpos, data.qvel])


class IsaacLabBackend(PhysicsBackend):
    """The third-party cross-check (9.1.4). It carries no primary endpoint.

    It does exactly two things: reproduce D2's delay across simulators (the
    delay must be applied on the *action* side, in the wrapper's action buffer,
    not through `DelayedPDActuatorCfg`, which is actuator-side and re-samples an
    integer shift per environment at every reset -- a different physical
    quantity), and check the external validity of the ILC teacher by asking
    whether an RL expert's spontaneous compensation direction agrees with the
    ILC one.
    """

    backend = Backend.ISAAC_LAB

    def probe(self) -> None:
        if not _importable("isaaclab"):
            raise BackendUnavailable(
                "Isaac Lab is the third-party cross-check of 9.1.4 and is not "
                "installed in this environment: `import isaaclab` fails.\n"
                "It needs an NVIDIA GPU with the Omniverse runtime; the package is "
                "not installable on this machine. Install Isaac Lab on the cluster "
                "image and re-run the cross-check there.\n"
                "This raises rather than degrading to CPU MuJoCo on purpose. The "
                "cross-check's whole content is that a second simulator agrees; "
                "running it on the primary backend would return agreement by "
                "construction, and 9.1.2 forbids mixing backends inside one "
                "endpoint in any case.\n"
                "Two implementation details when it does run: the delay must be "
                "applied on the action side in the wrapper's action buffer (Isaac "
                "Lab's DelayedPDActuatorCfg is actuator-side and re-samples the "
                "shift per environment at every reset, so it is not the same "
                "quantity), and both sides must use a 50 ms control step or the "
                "'integer number of control steps' property fails on one of them."
            )

    def rollout(self, spec: RolloutSpec, wrapper: Any, policy: Any) -> RolloutResult:
        self.probe()
        # Unreachable while isaaclab is absent; kept so that the interface is
        # complete rather than a stub with a different signature.
        raise BackendUnavailable(
            "Isaac Lab is importable but no rollout adapter is wired up in this "
            "checkout. 9.1.4 limits it to the D2 delay cross-check and the RL "
            "teacher's external validity check, both of which report a direction "
            "cosine and neither of which produces a primary endpoint; wire the "
            "adapter to those two only."
        )


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


_BACKENDS: dict[Backend, type[PhysicsBackend]] = {
    Backend.CPU_MUJOCO: CpuMujocoBackend,
    Backend.MJX: MjxBackend,
    Backend.ISAAC_LAB: IsaacLabBackend,
}


def get_backend(backend: Backend | str) -> PhysicsBackend:
    """Instantiate the requested backend. Never substitutes another one."""
    b = Backend(backend) if not isinstance(backend, Backend) else backend
    return _BACKENDS[b]()


# ---------------------------------------------------------------------------
# degradation branch 0
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchZeroDecision:
    """Whether 9.8.4's engineering branch is triggered, and what it implies."""

    overrun_step: float
    overrun_frame: float
    triggered: bool
    reason: str

    def __str__(self) -> str:
        return (
            f"branch 0 {'TRIGGERED' if self.triggered else 'not triggered'} "
            f"(varsigma_step={self.overrun_step:.3g}, "
            f"varsigma_frame={self.overrun_frame:.3g}): {self.reason}"
        )


def degradation_branch_zero(
    *,
    overrun_step: float,
    overrun_frame: float,
    threshold: float,
    stage: str,
) -> BranchZeroDecision:
    """Evaluate 9.8.4's branch 0 and enforce its position in the ladder.

    `threshold` has no default: it is the preregistered overrun factor
    (`overrun_factor_branch_zero`) and comes out of the preregistration file,
    not out of this signature.

    Two orderings are enforced. Branch 0 comes before any arm is cut, because it
    scales with the batch axis while cutting arms scales with the number of
    experiments -- the former loses no statistical unit and changes no caliber
    annotation, the latter narrows the claimed scope at every step. And it has
    to land before demonstration collection: 9.1.2 forbids mixing the two
    backends' data, so a port after collection means recollecting all 112,800
    trajectories.
    """
    if threshold <= 0:
        raise ValueError("the branch 0 overrun threshold must be positive")
    triggered = max(overrun_step, overrun_frame) > threshold
    if triggered and stage in ("collect", "materialise", "train", "evaluate"):
        raise BranchZeroOrderingError(
            f"the overrun factor exceeds {threshold} (step {overrun_step:.3g}, frame "
            f"{overrun_frame:.3g}), so 9.8.4's degradation branch 0 -- porting the "
            f"pure-physics load to MJX -- is triggered, but the pipeline is already "
            f"at stage {stage!r}.\n"
            "Branch 0 must land as one block after S1 and before S2. Porting later "
            "invalidates the demonstrations already collected, because 9.1.2 forbids "
            "mixing MJX and CPU MuJoCo data inside one endpoint and the port requires "
            "recollecting all of them.\n"
            "Either complete the port and recollect, or record in the plan that the "
            "port was declined as an engineering-budget choice (80-120 person-hours "
            "on the critical path) and cut arms in the preregistered order instead. "
            "The declination has to be written as a budget choice, not as a technical "
            "impossibility: MJX has no functional gap blocking the primary caliber."
        )
    reason = (
        "port the pure-physics load to MJX before cutting any arm; it scales with "
        "the batch axis and loses no statistical unit"
        if triggered
        else "overrun is within the threshold; the CPU caliber stands"
    )
    return BranchZeroDecision(
        overrun_step=overrun_step,
        overrun_frame=overrun_frame,
        triggered=triggered,
        reason=reason,
    )


def with_render(spec: RolloutSpec, *, render: bool) -> RolloutSpec:
    """9.1.3: rendering is decoupled from collection, so this is explicit."""
    return replace(spec, render=render)
