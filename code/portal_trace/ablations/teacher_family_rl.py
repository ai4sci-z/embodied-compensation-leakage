"""A13's RL level, producer side: SAC on the defective embodiment, then frozen.

`ablations.teacher_family` says what the level *is* and reads the artefact;
this module trains it. The split is the one `capacity` / `capacity_nets` already
makes, and for the same reason: the configuration loader checks A13's level set
and must not pull torch, gymnasium and stable-baselines3 in to do it.

Why an agent at all, and why this narrow
----------------------------------------

Appendix A-18 is emphatic that a policy retrained on the twin is *not* a source
of compensation ground truth -- c_delta is the difference of two policies at the
same state, and retraining gives a different policy whose difference carries the
seed, the local optimum and a shifted state distribution. So this trainer
produces exactly one thing: a demonstrator whose compensation *direction* can be
compared with the ILC teacher's (10.6.1). It produces no endpoint of the primary
or secondary family, and nothing here computes a leakage number.

The three properties 10.6.1 names -- frozen weights, the mean action, a fixed
seed -- are what make the artefact admissible under (A2) at all, and each is
realised structurally rather than by convention:

* frozen: `FrozenMeanActionPolicy` is built by copying the trained actor's
  layers into a module with `requires_grad_(False)` on every parameter and
  `eval()` set, and `RLExpertTeacher` re-checks both on load.
* mean action: the frozen module holds the trunk and the mean head and *nothing
  else*. SAC's log-std head and its squashed Gaussian are left behind, so there
  is no sampling path to take by accident and no `deterministic=` flag to get
  wrong. The action is `tanh(mu(phi(o)))`, unscaled to the action range exactly
  as `stable_baselines3` unscales it, which the trainer checks against
  `model.predict(deterministic=True)` before it writes the file.
* fixed seed: the agent's randomness (exploration, the warm-up actions, the
  initialisation) comes from `SeedTriple.train` through SAC's own `seed=`, and
  the environment's from `SeedTriple.env` through `defects.inject.reseed_env`.
  Two seed layers, both 11.3's, neither invented here.

The environment
---------------

`DefectiveEmbodimentEnv` is a gymnasium view of `defects.inject.DefectWrapper`
rather than of a robosuite env. That is the whole point of the level: 10.6.1 asks
what an agent arrives at on the *broken* robot, so the 500 Hz injection hook, the
model-level re-application after every reset and the sensing twin all have to be
in the loop the agent closes. robosuite 1.5.2 ships a `GymWrapper`, and it is not
usable here -- it imports `gym`, not `gymnasium`, and it wraps the raw env, which
would step past the injector.

Three decisions the proposal does not make, made here and recorded in the
checkpoint so that a reader never has to infer them:

reward
    robosuite's task reward is sparse unless `reward_shaping=True`, and sparse
    means identically zero on all 400 frames of every episode until a success
    lands in the buffer -- with which SAC's critic has no gradient to descend and
    the run measures nothing. The level therefore ships on the shaped caliber.
    This cannot bias A13 towards agreement: the shaping term is a function of the
    gripper-to-object geometry and is bit-identical on the clean twin, so it
    carries no information about the defect and cannot tell the agent which way to
    compensate. `RewardCaliber` is read off the constructed env rather than passed
    beside it, so the recorded caliber cannot disagree with the rewards.

horizon and termination
    400 control steps, early stopping off, which is 9.6.1's collection horizon
    and is already what `defects.inject._make_env` builds. robosuite's own `done`
    is `timestep >= horizon` and nothing else, so it is reported as gymnasium
    truncation and `terminated` is always False. Ending an episode on success
    instead would give the agent a shorter horizon than the demonstrations have
    and would make the value function's bootstrap disagree with the frames A13
    compares on.

what a gymnasium seed means
    `reset(seed=k)` selects episode k of the (task, family) environment-seed
    stream that `DefectWrapper.reset` already implements as
    `default_rng([seeds.env, episode])`, and an unseeded reset advances to the
    next episode. It is not a second source of randomness -- every stochastic
    choice in a robosuite 1.5.2 episode flows through `env.rng`, which
    `reseed_env` rebinds -- and the generator gymnasium allocates for
    `self.np_random` is deliberately never read.

    One consequence is worth stating because it is not what a reader would guess:
    `stable_baselines3` seeds an environment exactly once, from the model's own
    seed, so a training run walks the stream from episode `seeds.train` rather
    than from episode 0. The run is still reproducible from the two recorded
    seeds, and the origin and the episode count are recorded as well. It is also
    not a breach of 11.3's rule that the environment layer is aligned across
    arms: the initial states of an RL *training* rollout are not a measured
    quantity and are compared with nothing. The demonstrations A13 reads its
    cosine off are collected afterwards by `runners.collect`, from the
    environment layer's own per-demonstration allocation.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import FlattenExtractor
from torch import nn

from portal_trace.ablations.teacher_family import (
    ObservationLayout,
    TeacherFamily,
    TeacherFamilySpec,
    TeacherFamilyUnavailable,
)
from portal_trace.configs.loader import ConfigBundle
from portal_trace.configs.schema import ConfigError
from portal_trace.defects.delta import Delta, default_joint_scale
from portal_trace.defects.inject import DefectWrapper, build_single
from portal_trace.env.versions import (
    Backend,
    EnvironmentFingerprint,
    check_pinned_versions,
    fingerprint,
)
from portal_trace.lockfile import source_commit
from portal_trace.registry.families import descriptor
from portal_trace.registry.tasks import CONTROL_FREQ, HORIZON, TaskSpec
from portal_trace.registry.tasks import task as registry_task
from portal_trace.types import SeedTriple

__all__ = [
    "ABLATION_ROW",
    "ARM",
    "LEVEL_ID",
    "OBSERVATION_KEYS",
    "PRODUCED_BY",
    "RewardCaliber",
    "SACSettings",
    "DefectiveEmbodimentEnv",
    "FrozenMeanActionPolicy",
    "RLExpertArtefact",
    "RLExpertPlan",
    "build_defective_env",
    "train_rl_expert",
]

# The row, its third level and the arm that runs it. Named rather than passed in:
# this module produces one preregistered object and reading the plan off any other
# row would mean the artefact's cells came from somewhere 10.6.1 does not describe.
ABLATION_ROW = "A13"
LEVEL_ID = "rl_expert"
ARM = "A13_rl_expert"

# The command that produces the artefact, quoted verbatim in the checkpoint's
# `produced_by` and in `RLExpertTeacher`'s refusal. One string, because a reader
# who is told the file is missing has to be told the same thing the file itself
# claims about its origin.
PRODUCED_BY = "portal-trace teacher --experiment A13"

# The observation the expert sees, in this order. `DefectWrapper` refuses a body
# with more than one arm (9.3.1's families are single-arm), so robosuite's naming
# prefix is `robot0_` and these two literals are the only keys this can be.
#
# The proprioceptive block leads, matching `runners.collect.PROPRIO_LAYOUT` and
# the 9.6.3 storage order rather than robosuite's own `GymWrapper`, which puts
# `object-state` first. Either order works and neither is inherited: the reason
# the layout is recorded at all is that the choice is arbitrary and the consumer
# has to make the same one.
#
# This is the env's native 50-column proprioceptive vector, not the 70-column
# one the hdf5 stores. The extra twenty columns (tau_g, wrist force/torque, the
# DoF mask) are assembled by `runners.materialise` for the BC arms; putting the
# RL expert on them would train a policy on an input its own environment does not
# emit, and the agent has to be able to run in the loop it was trained in.
OBSERVATION_KEYS: tuple[str, ...] = ("robot0_proprio-state", "object-state")


class RewardCaliber(str, Enum):
    """Which reward the expert optimised. Recorded, never assumed.

    `SPARSE_TASK` is robosuite's default for every registry task but `Wipe`: 1.0
    on the success predicate and 0.0 otherwise. `DENSE_SHAPED` is
    `reward_shaping=True`, which adds the reaching and grasping terms. A13 ships
    on the shaped one for the reason the module docstring gives, but both are
    nameable, because "the agent that was trained" is not defined without saying
    which of the two it maximised.
    """

    SPARSE_TASK = "sparse_task"
    DENSE_SHAPED = "dense_shaped"

    @property
    def reward_shaping(self) -> bool:
        return self is RewardCaliber.DENSE_SHAPED

    @classmethod
    def of_env(cls, env: Any) -> RewardCaliber:
        """Read the caliber off a constructed robosuite env.

        Read rather than accepted as an argument beside the env: the caliber is a
        property of the object that will emit the rewards, and a version that was
        passed alongside could name one caliber while the agent trained on the
        other. The recorded value is then wrong in the one way no later check
        could catch.
        """
        shaping = getattr(env, "reward_shaping", None)
        if shaping is None:
            raise TeacherFamilyUnavailable(
                f"{type(env).__name__} exposes no `reward_shaping` attribute, so which "
                "reward A13's expert would be maximising cannot be established. Every "
                "robosuite 1.5.2 manipulation environment carries one; a task that "
                "does not has to declare its reward caliber before an RL expert on it "
                "means anything"
            )
        return cls.DENSE_SHAPED if bool(shaping) else cls.SPARSE_TASK


@dataclass(frozen=True)
class SACSettings:
    """The SAC hyperparameters, written out rather than inherited from SB3.

    10.6.1 fixes the expert's *interface* (frozen, mean action, fixed seed) and
    says nothing about its optimiser, so these are choices. They are named here,
    and recorded in the checkpoint, instead of being left to
    `stable_baselines3`'s defaults: a library default is a value that moves when
    the library is upgraded, and an artefact whose hyperparameters moved with an
    upgrade is not the artefact the configuration described.

    All of them except `learning_starts` happen to coincide with SB3 2.9's
    defaults, which is not an accident worth hiding -- they are the settings the
    published SAC results use and there is no reason from the proposal to move
    them. `learning_starts` is the one deliberate departure: SB3's 100 steps is a
    quarter of one 400-step episode, so the critic's first update would see a
    buffer holding a single strongly autocorrelated fragment. 10 000 steps is 25
    episodes.

    `total_timesteps` has no default. It is the budget, the level block states
    it, and a training length that could be omitted would make two checkpoints
    with the same name different objects.
    """

    total_timesteps: int
    learning_starts: int = 10_000
    learning_rate: float = 3.0e-4
    buffer_size: int = 1_000_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto"
    net_arch: tuple[int, ...] = (256, 256)

    def __post_init__(self) -> None:
        if self.total_timesteps <= self.learning_starts:
            raise ValueError(
                f"total_timesteps={self.total_timesteps} against "
                f"learning_starts={self.learning_starts}: the run would collect "
                "experience and never take a gradient step, and the checkpoint would "
                "hold the initialisation under the name of a trained expert"
            )
        for name in ("learning_rate", "tau", "gamma"):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"SAC {name} is {value!r}, which is outside (0, 1]")
        for name in ("batch_size", "buffer_size", "train_freq", "gradient_steps"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"SAC {name} is {getattr(self, name)!r}")
        if not self.net_arch or any(width <= 0 for width in self.net_arch):
            raise ValueError(f"net_arch {self.net_arch} has an empty or non-positive layer")

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any], *, where: str) -> SACSettings:
        """Read the level block. `total_timesteps` is required, the rest optional.

        Optional in the sense that the dataclass above states them, not in the
        sense that they have no value: a level that omits `batch_size` runs the
        256 written here and records 256, which is checkable against the
        configuration afterwards.
        """
        if "total_timesteps" not in parameters:
            raise ValueError(
                f"{where}: A13's RL level declares no `total_timesteps`. The training "
                "length is the budget the row has to fit and cannot be defaulted"
            )
        integers = (
            "total_timesteps", "learning_starts", "buffer_size", "batch_size",
            "train_freq", "gradient_steps",
        )
        reals = ("learning_rate", "tau", "gamma")
        kwargs: dict[str, Any] = {
            name: int(parameters[name]) for name in integers if name in parameters
        }
        kwargs.update(
            {name: float(parameters[name]) for name in reals if name in parameters}
        )
        if "ent_coef" in parameters:
            kwargs["ent_coef"] = str(parameters["ent_coef"])
        if "net_arch" in parameters:
            kwargs["net_arch"] = tuple(int(width) for width in parameters["net_arch"])
        return cls(**kwargs)

    def to_json(self) -> dict[str, Any]:
        blob = asdict(self)
        blob["net_arch"] = list(self.net_arch)
        return blob


class DefectiveEmbodimentEnv(gym.Env):
    """A gymnasium env over one `DefectWrapper`: the broken robot, nothing else.

    Wraps the *wrapper*, never `wrapper.env`. `DefectWrapper.__init__` rebinds
    `env._pre_action`, `env.reset` and the arm controller's `update` precisely so
    that reaching past it is not a way to skip an injection step, and it captures
    `self._env_reset` at construction; `ablations.solver` documents what that
    costs anyone who rebinds `reset` afterwards. This class needs none of that --
    it calls the public `wrapper.reset` and `wrapper.step` and rebinds nothing --
    so the agent closes its loop through the 500 Hz hook by construction.
    """

    # No rendering. 9.1.3's first consequence is that pixels cost eight times
    # what physics does, and the expert is a state-based policy: a render mode
    # would be an offer this env has no reason to make good on.
    metadata: dict[str, Any] = {"render_modes": []}
    render_mode: str | None = None

    def __init__(
        self,
        wrapper: DefectWrapper,
        *,
        observation_keys: Sequence[str] = OBSERVATION_KEYS,
    ) -> None:
        self.wrapper = wrapper
        self.reward_caliber = RewardCaliber.of_env(wrapper.env)

        low, high = wrapper.env.action_spec
        self.action_space = spaces.Box(
            low=np.asarray(low, dtype=np.float32),
            high=np.asarray(high, dtype=np.float32),
            dtype=np.float32,
        )

        # Episode 0 is reset here to measure the layout on the real body, and the
        # episode counter is left at 0 so that the first `reset()` repeats it
        # rather than skipping it. `DefectWrapper.__init__` has already reset once
        # or twice while compiling its shadow joints, so this is not the episode's
        # first compile.
        self.observation_layout = ObservationLayout.measure(
            wrapper.reset(0), observation_keys
        )
        # Unbounded rather than the observed range: robosuite's proprioceptive
        # block contains joint velocities and accelerations whose excursions are
        # not known before the run, and a Box that clipped them would silently
        # censor exactly the frames a defect makes extreme.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_layout.total_width,),
            dtype=np.float32,
        )
        self._episode = 0
        self._steps = 0
        # Where in the (task, family) episode stream this env's first episode sat,
        # and how many episodes it has run since. Recorded in the checkpoint: the
        # stream's origin is chosen by whoever seeds the first reset, and without
        # it an auditor could not say which initial states the expert ever saw.
        self.episode_origin: int | None = None
        self.n_episodes = 0

    # -- gymnasium surface ------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start an episode. `seed` is the episode index, not a new random stream.

        `stable_baselines3` seeds an env once, from the model's seed, so during
        training this is the call that decides where in the stream the run starts;
        the module docstring says why walking the stream from there rather than
        from zero costs nothing that 11.3 owns.

        The seeded and unseeded cases differ in one further respect, and it is
        9.2.9's: D10's drift accumulator survives an ordinary reset on purpose,
        because the whole content of that defect is that it accumulates across the
        episodes of a session. A seeded reset, on the other hand, names an episode
        and has to reproduce it, and for D10 the session clock is part of what
        that episode was -- so it starts a new session. During training the two
        fall out correctly without a special case: the first reset carries the
        agent's seed and the rest do not.
        """
        if seed is not None:
            if int(seed) < 0:
                raise ValueError(
                    f"reset(seed={seed}) is negative. A13's environment reads the seed "
                    "as the episode index of the (task, family) stream 9.1.6 "
                    "allocates, and `default_rng([env_seed, episode])` has no negative "
                    "episodes"
                )
            self._episode = int(seed)
            self.wrapper.new_session()
        # Sets `self.np_random` and records `np_random_seed`, which the gymnasium
        # API promises a caller can read back. That generator is never used: the
        # episode is decided by `reseed_env`, and a second stream drawn from here
        # would be a source of variation with no seed layer in 11.3 to own it.
        super().reset(seed=seed, options=options)

        episode = self._episode
        observations = self.wrapper.reset(episode)
        if self.episode_origin is None:
            self.episode_origin = episode
        self.n_episodes += 1
        self._episode = episode + 1
        self._steps = 0
        return self.observation_layout.flatten(observations), {"episode_index": episode}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observations, reward, done, _ = self.wrapper.step(
            np.asarray(action, dtype=np.float64)
        )
        self._steps += 1
        # robosuite computes `done` as `timestep >= horizon and not ignore_done`
        # and never from the success predicate, so the flag is a time limit and
        # belongs on gymnasium's truncated channel. Reporting it as `terminated`
        # would tell SAC the value beyond the horizon is zero, and the 400th
        # frame of a held terminal pose is worth as much as the 399th.
        return (
            self.observation_layout.flatten(observations),
            float(reward),
            False,
            bool(done),
            {
                "episode_index": self._episode - 1,
                "steps": self._steps,
                # `is_success` is the key stable-baselines3 aggregates into a
                # success rate. It is logging, not a reward and not a terminal
                # condition: 9.6.1 runs the horizon out either way.
                "is_success": bool(self.wrapper.env._check_success()),
            },
        )

    def close(self) -> None:
        self.wrapper.close()


def build_defective_env(
    *,
    task_spec: TaskSpec,
    family: str,
    delta: Delta,
    seeds: SeedTriple,
    reward_caliber: RewardCaliber,
    observation_keys: Sequence[str] = OBSERVATION_KEYS,
) -> DefectiveEmbodimentEnv:
    """The defective embodiment as a gymnasium env, built through `build_single`.

    Through `defects.inject.build_single` rather than a second construction path,
    for the reason `ablations.solver.build_with_solver_group` gives: the ablation
    arm and the main arms then compile the same MJCF with the same robosuite
    keyword arguments, and differ only in the one flag this level moves.
    """
    wrapper = build_single(
        task_spec, family, delta, seeds, reward_shaping=reward_caliber.reward_shaping
    )
    env = DefectiveEmbodimentEnv(wrapper, observation_keys=observation_keys)
    if env.reward_caliber is not reward_caliber:
        raise TeacherFamilyUnavailable(
            f"the level asked for the {reward_caliber.value} reward and the built "
            f"environment reports {env.reward_caliber.value}. robosuite ignored the "
            "flag, which would leave the agent optimising something other than what "
            "the checkpoint records"
        )
    return env


class FrozenMeanActionPolicy(nn.Module):
    """The trained actor's mean, with the sampling machinery left behind.

    Holds a copy of SAC's `latent_pi` trunk and its `mu` head and nothing else.
    Keeping the `stable_baselines3` actor itself would have been shorter and is
    the wrong artefact twice over. It would carry `log_std` and a squashed
    Gaussian, so "never a sample" would rest on passing `deterministic=True` at
    every call site rather than on there being no other option; and it would make
    the checkpoint unreadable without stable-baselines3, where what it needs is
    torch and `ablations.teacher_family`'s layout record.

    The action is `tanh(mu(o))` mapped onto the environment's action range by the
    same affine map `stable_baselines3` uses, so this module answers exactly what
    `model.predict(o, deterministic=True)` answered before the file was written --
    which `train_rl_expert` checks rather than asserts.
    """

    def __init__(
        self,
        actor: Any,
        *,
        action_low: np.ndarray,
        action_high: np.ndarray,
        observation_layout: ObservationLayout,
    ) -> None:
        super().__init__()
        extractor = getattr(actor, "features_extractor", None)
        if not isinstance(extractor, FlattenExtractor):
            raise TeacherFamilyUnavailable(
                f"the trained actor's feature extractor is {type(extractor).__name__}, "
                "not a FlattenExtractor. This module reproduces SAC's deterministic "
                "action by applying the trunk to the observation directly, which is "
                "exact only while the extractor is the identity on an already flat "
                "vector. An extractor with parameters of its own has to be copied in "
                "here before the frozen policy means the same thing as the trained one"
            )
        self.observation_layout = observation_layout
        self.trunk = copy.deepcopy(actor.latent_pi)
        self.mu = copy.deepcopy(actor.mu)
        self.register_buffer(
            "action_low", torch.as_tensor(np.asarray(action_low), dtype=torch.float32)
        )
        self.register_buffer(
            "action_high", torch.as_tensor(np.asarray(action_high), dtype=torch.float32)
        )
        self.to(next(actor.parameters()).device)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """(N, obs) -> (N, action), the squashed mean in environment units."""
        squashed = torch.tanh(self.mu(self.trunk(observations)))
        return self.action_low + 0.5 * (squashed + 1.0) * (
            self.action_high - self.action_low
        )

    def mean_action(self, observations: Any) -> np.ndarray:
        """The interface `RLExpertTeacher.feedforward` looks for.

        Accepts one frame or a batch and answers in kind, because the two callers
        want different things: A13's cosine is computed over all of the ILC
        teacher's visited frames at once, and a rollout asks about one.
        """
        x = np.asarray(observations, dtype=np.float32)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.observation_layout.total_width:
            raise ValueError(
                f"the expert was trained on {self.observation_layout.total_width} "
                f"columns laid out as {list(self.observation_layout.keys)} and was "
                f"handed {x.shape}. Use `ObservationLayout.flatten` on the observation "
                "dict: a vector of the right width in another order is a permuted "
                "input and the direction cosine would not show it"
            )
        with torch.no_grad():
            inputs = torch.from_numpy(np.ascontiguousarray(x)).to(self.action_low.device)
            actions = self(inputs)
        out = actions.cpu().numpy().astype(np.float64)
        return out[0] if single else out


@dataclass(frozen=True)
class RLExpertArtefact:
    """What one training run produced, and where."""

    checkpoint: Path
    metadata: dict[str, Any]

    @property
    def total_timesteps(self) -> int:
        return int(self.metadata["sac"]["total_timesteps"])


@dataclass(frozen=True)
class RLExpertPlan:
    """One cell's training run, resolved from the preregistered level block.

    The level block is the authority for every quantity a reader would have to
    know to reproduce the artefact -- the checkpoint's name, the reward caliber,
    the observation keys, the training length -- and this class is the only place
    that reads it. A trainer invocation assembled by hand could disagree with the
    row on any of them and the checkpoint would still record what it was actually
    given, so the disagreement would be discoverable only by comparing two files.
    Resolving the whole plan from the row instead means the command line carries
    the cell and nothing else.
    """

    task_spec: TaskSpec
    family: str
    delta: Delta
    seeds: SeedTriple
    checkpoint: Path
    settings: SACSettings
    reward_caliber: RewardCaliber
    observation_keys: tuple[str, ...]

    @classmethod
    def from_level(
        cls,
        bundle: ConfigBundle,
        *,
        task: str,
        family: str,
        defect: str,
        level: int,
        seed_index: int,
        artifact_root: Path,
    ) -> RLExpertPlan:
        where = f"{ABLATION_ROW}.{LEVEL_ID}"
        block = bundle.ablation(ABLATION_ROW).level(LEVEL_ID)
        spec = TeacherFamilySpec.from_parameters(block.parameters, where=where)
        if spec.family is not TeacherFamily.RL_EXPERT:
            raise ConfigError(
                f"{where} declares family {spec.family.value!r}. This trainer produces "
                "the RL level's artefact and cannot produce the other two: they are "
                "computed from the ILC teacher and from `defects.fields`"
            )

        cells = bundle.cells(ARM)
        if (family, defect) not in cells:
            raise ConfigError(
                f"{ARM} does not run the cell ({family}, {defect}); its cells are "
                f"{[f'{f}/{d}' for f, d in cells]}. An expert trained outside the row's "
                "cells is an artefact no preregistered level reads, and the 12 runs "
                f"{ABLATION_ROW} is budgeted for are the ones inside it"
            )

        declared_horizon = block.parameters.get("horizon")
        if declared_horizon is None or int(declared_horizon) != HORIZON:
            raise ConfigError(
                f"{where} declares horizon {declared_horizon!r} and 9.6.1 fixes the "
                f"collection horizon at {HORIZON}. The expert's episodes and the "
                "demonstrations it will be compared on have to be the same length, or "
                "its value function bootstraps at a boundary the comparison never sees"
            )
        for key in ("reward_caliber", "observation_keys"):
            if key not in block.parameters:
                raise ConfigError(
                    f"{where} declares no {key!r}. Both are choices 10.6.1 leaves open "
                    "and both change what the trained agent is, so the level states "
                    "them rather than inheriting whatever this module's defaults happen "
                    "to be at the time of the run"
                )
        observation_keys = tuple(str(k) for k in block.parameters["observation_keys"])
        reward_caliber = RewardCaliber(str(block.parameters["reward_caliber"]))

        # Match the preregistered acting-joint pattern of the row's own cells:
        # A13 holds the defect fixed and moves only the teacher, so a joint scope
        # invented here would be a second difference from its B2 reference.
        n_joints = descriptor(family).dof
        delta = Delta.from_level(
            defect,
            level,
            joint_scale=default_joint_scale(defect, n_joints),
        )

        return cls(
            task_spec=registry_task(task),
            family=family,
            delta=delta,
            # `demo_index=0` is not a demonstration index here -- the run collects
            # none. It is the environment layer's third scope coordinate, and what
            # it selects is the *base* of the episode stream: index 0 makes the
            # expert draw its initial states from the same generator family as the
            # cell's demonstrations rather than from a stream keyed to nothing.
            # Which episodes of it the run visits is a separate question, and the
            # checkpoint's `episode_stream` block records the answer.
            seeds=bundle.seeds.triple(
                task=task, family=family, demo_index=0, arm=ARM, seed_index=seed_index
            ),
            checkpoint=cls._resolve_checkpoint(
                spec.checkpoint,
                artifact_root=artifact_root,
                task=task,
                family=family,
                defect=defect,
                level=level,
                where=where,
            ),
            settings=SACSettings.from_parameters(block.parameters, where=where),
            reward_caliber=reward_caliber,
            observation_keys=observation_keys,
        )

    @staticmethod
    def _resolve_checkpoint(
        template: Path | None,
        *,
        artifact_root: Path,
        task: str,
        family: str,
        defect: str,
        level: int,
        where: str,
    ) -> Path:
        """Fill the level block's path template. The cell has to be in the name.

        `TeacherFamilySpec` has already refused a level with no checkpoint at all,
        so `template` cannot be None here. What it cannot check is that the name
        distinguishes the cells: one path shared by the row's six cells would let
        the second training run refuse to overwrite the first and read as "the
        artefact is already there".
        """
        if template is None:   # pragma: no cover - TeacherFamilySpec refuses first
            raise ConfigError(f"{where} states no checkpoint path")
        fields = {"task": task, "family": family, "defect": defect, "level": level}
        try:
            filled = str(template).format(**fields)
        except (KeyError, IndexError) as exc:
            raise ConfigError(
                f"{where}: checkpoint template {str(template)!r} names the field {exc} "
                f"and the cell supplies {sorted(fields)}"
            ) from None
        for name in ("family", "defect", "level"):
            if f"{{{name}}}" not in str(template):
                raise ConfigError(
                    f"{where}: checkpoint template {str(template)!r} does not vary with "
                    f"{{{name}}}, so the row's cells would share one file. A frozen "
                    "artefact is never overwritten in place, so the second cell's run "
                    "would refuse and the first cell's expert would be read under the "
                    "second cell's name"
                )
        return artifact_root / filled

    def train(self) -> RLExpertArtefact:
        """Run it. Everything this passes on came from the level block."""
        return train_rl_expert(
            task_spec=self.task_spec,
            family=self.family,
            delta=self.delta,
            seeds=self.seeds,
            checkpoint=self.checkpoint,
            settings=self.settings,
            reward_caliber=self.reward_caliber,
            observation_keys=self.observation_keys,
        )


def train_rl_expert(
    *,
    task_spec: TaskSpec,
    family: str,
    delta: Delta,
    seeds: SeedTriple,
    checkpoint: Path,
    settings: SACSettings,
    reward_caliber: RewardCaliber,
    observation_keys: Sequence[str] = OBSERVATION_KEYS,
) -> RLExpertArtefact:
    """Train SAC on the defective embodiment, freeze it, and write the checkpoint.

    Returns the artefact record. The file is `{"policy": module, "metadata":
    dict}`, which is what `RLExpertTeacher._load` reads: it takes `policy` by
    name and refuses a payload that carries only metrics, so the two keys are the
    interface rather than a convention.
    """
    check_pinned_versions(required=("mujoco", "robosuite"))
    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        raise TeacherFamilyUnavailable(
            f"{checkpoint} already exists. A13's expert is a frozen artefact that a "
            "cosine has been or will be read against, so it is not overwritten in "
            "place: a rerun that silently replaced it would leave two different "
            "agents behind one recorded path. Move or delete the old file if the "
            "intent is to retrain"
        )

    env = build_defective_env(
        task_spec=task_spec,
        family=family,
        delta=delta,
        seeds=seeds,
        reward_caliber=reward_caliber,
        observation_keys=observation_keys,
    )
    model = SAC(
        "MlpPolicy",
        env,
        seed=seeds.train,
        learning_rate=settings.learning_rate,
        buffer_size=settings.buffer_size,
        learning_starts=settings.learning_starts,
        batch_size=settings.batch_size,
        tau=settings.tau,
        gamma=settings.gamma,
        train_freq=settings.train_freq,
        gradient_steps=settings.gradient_steps,
        ent_coef=settings.ent_coef,
        policy_kwargs={"net_arch": list(settings.net_arch)},
        verbose=0,
    )
    started = time.perf_counter()
    model.learn(total_timesteps=settings.total_timesteps, log_interval=None)
    wall_clock = time.perf_counter() - started
    # Read before the verification rollout below, which resets the env again: the
    # recorded stream is the one the agent learned from, not the one that includes
    # the episode spent checking the frozen copy.
    episodes = (env.episode_origin, env.n_episodes)

    policy = FrozenMeanActionPolicy(
        model.policy.actor,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        observation_layout=env.observation_layout,
    )
    _check_frozen_reproduces_the_trained_actor(model, policy, env)

    metadata = _metadata(
        task_spec=task_spec,
        family=family,
        delta=delta,
        seeds=seeds,
        env=env,
        model=model,
        policy=policy,
        settings=settings,
        wall_clock=wall_clock,
        episodes=episodes,
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": policy, "metadata": metadata}, checkpoint)
    env.close()
    return RLExpertArtefact(checkpoint=checkpoint, metadata=metadata)


# The frozen module recomputes SAC's deterministic action from copied layers, so
# the two can only agree to float32 rounding on the affine unscaling, not bit for
# bit. A disagreement above this is a wiring error -- the wrong head copied, the
# extractor skipped when it was not the identity -- and not a tolerance to widen.
_MEAN_ACTION_TOLERANCE = 1.0e-6


def _check_frozen_reproduces_the_trained_actor(
    model: SAC, policy: FrozenMeanActionPolicy, env: DefectiveEmbodimentEnv
) -> None:
    """Refuse to write a checkpoint whose mean action is not the trained one.

    Checked on observations from the trained agent's own rollout rather than on
    noise, because the states A13 evaluates on are states a policy actually
    reaches, and a discrepancy that only appears where the trunk saturates would
    hide from a uniform sample.
    """
    observations, _ = env.reset(seed=0)
    batch = [observations]
    for _ in range(3):
        action, _ = model.predict(observations, deterministic=True)
        observations, _, _, truncated, _ = env.step(action)
        batch.append(observations)
        if truncated:
            break
    stacked = np.asarray(batch, dtype=np.float32)
    expected, _ = model.predict(stacked, deterministic=True)
    produced = policy.mean_action(stacked)
    deviation = float(np.abs(np.asarray(expected, dtype=np.float64) - produced).max())
    if deviation > _MEAN_ACTION_TOLERANCE:
        raise TeacherFamilyUnavailable(
            f"the frozen mean-action module disagrees with the trained actor's "
            f"deterministic action by {deviation:.3e} over {stacked.shape[0]} frames. "
            "The checkpoint is not written: A13 would report the direction of "
            "whatever this module computes, under the name of the agent that was "
            "trained"
        )


def _metadata(
    *,
    task_spec: TaskSpec,
    family: str,
    delta: Delta,
    seeds: SeedTriple,
    env: DefectiveEmbodimentEnv,
    model: SAC,
    policy: FrozenMeanActionPolicy,
    settings: SACSettings,
    wall_clock: float,
    episodes: tuple[int | None, int],
) -> dict[str, Any]:
    """Everything needed to say which agent this is, in the package's own form.

    `produced_by` / `produced_at` / `source_commit` are `lockfile
    .MeasurementRecord`'s three provenance fields, and the environment block is
    `env.versions.fingerprint`, so an auditor reads a teacher checkpoint the same
    way as a lock entry or a model card. The two versions that matter here and are
    not in `PINNED_VERSIONS` -- stable-baselines3 and gymnasium -- go into the
    fingerprint's `extra`, which exists for exactly this.
    """
    import gymnasium
    import stable_baselines3

    stamp: EnvironmentFingerprint = replace(
        # `defects.inject` builds on CPU MuJoCo and there is no second path
        # through it, so recording anything else here would be a claim the
        # environment cannot support. 9.1.2 forbids mixing backends inside one
        # quantity, which is why the backend is written down rather than inferred.
        fingerprint(Backend.CPU_MUJOCO),
        extra={
            "stable_baselines3": stable_baselines3.__version__,
            "gymnasium": gymnasium.__version__,
            "torch": torch.__version__,
            "torch_device": str(model.device),
        },
    )
    return {
        "role": (
            "A13 rl_expert teacher (10.6.1): frozen weights, mean action, fixed seed. "
            "Reports a compensation direction cosine against the ILC teacher and no "
            "endpoint of the primary or secondary family (appendix A-18)"
        ),
        "teacher_family": TeacherFamily.RL_EXPERT.value,
        "task": task_spec.task_id,
        "family": family,
        "defect_code": delta.defect_code,
        "defect_level": delta.level,
        "delta": delta.to_json(),
        "seeds": {"env": seeds.env, "train": seeds.train, "eval": seeds.eval},
        "observation": env.observation_layout.to_json(),
        "action": {
            "dim": int(env.action_space.shape[0]),
            "low": [float(v) for v in env.action_space.low],
            "high": [float(v) for v in env.action_space.high],
            "rule": (
                "tanh-squashed Gaussian mean, unscaled to the action range; never a "
                "sample (10.6.1)"
            ),
        },
        "reward_caliber": env.reward_caliber.value,
        "horizon": HORIZON,
        "control_freq": CONTROL_FREQ,
        # Which initial states the agent actually met: episodes
        # [origin, origin + n) of the (task, family) stream 9.1.6 allocates.
        "episode_stream": {
            "origin": episodes[0],
            "n_episodes": episodes[1],
            "rule": "default_rng([seeds.env, episode]) per episode (defects.inject)",
        },
        "algorithm": "sac",
        "sac": settings.to_json(),
        "n_frozen_parameters": int(sum(p.numel() for p in policy.parameters())),
        "wall_clock_seconds": round(wall_clock, 3),
        "environment": stamp.to_json(),
        "produced_by": PRODUCED_BY,
        "produced_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_commit": source_commit(),
    }
