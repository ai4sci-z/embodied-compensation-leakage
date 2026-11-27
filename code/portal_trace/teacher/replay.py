"""Cross replay between a defective body and its twin, and byte-exact replay (10.3.1, 9.1.6).

**Cross replay** is the zero-GPU gate G2. 10.3.1 replays a recorded action
sequence open loop on the *other* member of a twin pair, in both directions, and
reads three endpoints off each run: success, peak `cfrc_ext`, and which kind of
failure occurred. The pre-registered prediction is asymmetric, and the asymmetry
is the point (11.3 calls it the first evidence of harm asymmetry):

  defective -> clean   a sequence carrying compensation overshoots on a body that
                       does not need it. Success drops by at least 15 pp and peak
                       `cfrc_ext` rises by at least 5 N.
  clean -> defective   the same sequence under-drives a body that needs the
                       compensation. Task failure rises; constraint violation
                       does not.

A null result here -- under 15 pp and no rise in `cfrc_ext` -- means the
demonstrations carry no detectable compensation and no training run is worth
starting (10.3.1, exit F1). Nothing in this module reads a GPU.

**Byte-exact replay** is the 9.1.6 requirement underneath all of it. A cell is
addressed by `manifest.yaml` plus `seed_env`, and re-running that address has to
reproduce the trajectory bit for bit, not to a tolerance. `ReplayCheck` compares
with `array_equal` for exactly that reason: a near miss is a failure, because it
means something outside the recorded address is entering the rollout.

Open loop is meant literally. `cross_replay` writes the recorded actions into the
other body without recomputing anything from the state it reaches -- the whole
question is what those actions do somewhere else, and a tracker in the loop would
quietly correct the very deviation being measured.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import yaml

from portal_trace.defects.inject import TwinPair
from portal_trace.registry.tasks import HORIZON
from portal_trace.registry.tasks import task as task_spec
from portal_trace.teacher.osc import (
    OSCTracker,
    env_action,
    fold_mode_codes,
    make_env,
    mode_code,
    physics_state,
    reset_env,
    reset_twin_pair,
    wrist_ft,
)
from portal_trace.teacher.refs import load_spec, resolve

__all__ = [
    "SUCCESS_DROP_PP",
    "CFRC_RISE_N",
    "FailureKind",
    "OpenLoopRun",
    "ReplayDirection",
    "CrossReplayReport",
    "ReplayCheck",
    "open_loop",
    "cross_replay",
    "replay_from_manifest",
    "verify_byte_exact",
]

# 10.3.1's pre-registered effect sizes, stated numerically in the proposal.
SUCCESS_DROP_PP = 15.0
CFRC_RISE_N = 5.0


class FailureKind(str, Enum):
    """10.3.1's failure profile. The two directions are predicted to differ here.

    `CONSTRAINT_VIOLATION` outranks `TASK_FAILURE` because a run that both
    collides and misses is a collision: the endpoint 10.1.2 cares about is
    whether the body did something unsafe, and demoting that to "also failed the
    task" would erase the asymmetry the gate is looking for.
    """

    NONE = "none"
    TASK_FAILURE = "task_failure"
    CONSTRAINT_VIOLATION = "constraint_violation"


class ReplayDirection(str, Enum):
    DEFECT_TO_CLEAN = "defect_to_clean"
    CLEAN_TO_DEFECT = "clean_to_defect"


@dataclass(frozen=True)
class OpenLoopRun:
    """One action sequence executed on one body, with nothing in the loop."""

    actions: np.ndarray           # (T, A) exactly what was replayed
    states: np.ndarray            # (T, nq + nv)
    peak_cfrc_ext: float
    max_wrist_force: float
    success: bool
    mode_hash: int

    @property
    def horizon(self) -> int:
        return len(self.states)

    def failure_kind(self, force_limit: float) -> FailureKind:
        """Classify against a wrist-force limit that the caller has to supply.

        There is no default. 10.1.2's constraint-violation endpoint is defined by
        a force limit, and picking one here would be inventing the threshold that
        decides which of the two predicted failure profiles a run lands in.
        """
        if self.max_wrist_force > force_limit:
            return FailureKind.CONSTRAINT_VIOLATION
        return FailureKind.NONE if self.success else FailureKind.TASK_FAILURE


def open_loop(
    env: Any,
    actions: np.ndarray,
    *,
    seed_env: int,
    task_id: str,
    reset: Callable[[], np.ndarray] | None = None,
) -> OpenLoopRun:
    """Replay `actions` on `env` from the `seed_env` reset, reading nothing back.

    The reset goes through `teacher.osc.reset_env`, so the scene settles exactly
    as it did during collection. Skipping that would put the replay on a body
    whose objects are still falling, and every deviation would be attributed to
    the defect.
    """
    actions = np.atleast_2d(np.asarray(actions, dtype=np.float64))
    if len(actions) != HORIZON:
        raise ValueError(f"replaying {len(actions)} actions, 9.6.1 fixes the horizon at {HORIZON}")

    expected = task_spec(task_id).env_name
    if type(env).__name__ != expected:
        raise ValueError(
            f"replaying a {task_id} sequence on a {type(env).__name__} env; the "
            f"address names {expected}"
        )

    reset_env(env, seed_env) if reset is None else reset()
    nstate = int(env.sim.model.nq) + int(env.sim.model.nv)
    states = np.empty((len(actions), nstate), dtype=np.float64)
    mode_codes = np.empty(len(actions), dtype=np.uint64)
    peak_cfrc, peak_force = 0.0, 0.0
    success = False

    for t, action in enumerate(actions):
        states[t] = physics_state(env.sim)
        env.step(env_action(env, action))
        peak_cfrc = max(peak_cfrc, float(np.max(np.linalg.norm(env.sim.data.cfrc_ext, axis=1))))
        peak_force = max(peak_force, float(np.linalg.norm(wrist_ft(env)[:3])))
        mode_codes[t] = mode_code(env)
        success = success or bool(env._check_success())

    return OpenLoopRun(
        actions=actions,
        states=states,
        peak_cfrc_ext=peak_cfrc,
        max_wrist_force=peak_force,
        success=success,
        mode_hash=fold_mode_codes(mode_codes),
    )


@dataclass(frozen=True)
class CrossReplayReport:
    """Both directions of 10.3.1, with the pre-registered prediction evaluated."""

    task_id: str
    family: str
    defect: str
    n_demos: int
    force_limit: float
    native_defect: tuple[OpenLoopRun, ...]
    native_clean: tuple[OpenLoopRun, ...]
    defect_on_clean: tuple[OpenLoopRun, ...]
    clean_on_defect: tuple[OpenLoopRun, ...]

    @staticmethod
    def _rate(runs: tuple[OpenLoopRun, ...]) -> float:
        return 100.0 * float(np.mean([r.success for r in runs]))

    @staticmethod
    def _peak(runs: tuple[OpenLoopRun, ...]) -> float:
        return float(np.mean([r.peak_cfrc_ext for r in runs]))

    def _profile(self, runs: tuple[OpenLoopRun, ...]) -> dict[str, float]:
        kinds = [r.failure_kind(self.force_limit) for r in runs]
        return {k.value: 100.0 * kinds.count(k) / len(kinds) for k in FailureKind}

    @property
    def success_drop_pp(self) -> float:
        """Success lost when the defective body's actions run on the clean twin."""
        return self._rate(self.native_defect) - self._rate(self.defect_on_clean)

    @property
    def cfrc_rise_n(self) -> float:
        return self._peak(self.defect_on_clean) - self._peak(self.native_clean)

    @property
    def reverse_success_drop_pp(self) -> float:
        return self._rate(self.native_clean) - self._rate(self.clean_on_defect)

    @property
    def passes_g2(self) -> bool:
        """10.3.1: below both effect sizes the demonstrations carry no compensation."""
        return self.success_drop_pp >= SUCCESS_DROP_PP and self.cfrc_rise_n >= CFRC_RISE_N

    @property
    def asymmetry_holds(self) -> bool:
        """11.3's harm asymmetry: constraint violations appear in one direction only."""
        forward = self._profile(self.defect_on_clean)[FailureKind.CONSTRAINT_VIOLATION.value]
        reverse = self._profile(self.clean_on_defect)[FailureKind.CONSTRAINT_VIOLATION.value]
        baseline = self._profile(self.native_clean)[FailureKind.CONSTRAINT_VIOLATION.value]
        return forward > baseline and reverse <= baseline

    def render(self) -> str:
        rows = (
            ("native defective", self.native_defect),
            ("native clean", self.native_clean),
            ("defect -> clean", self.defect_on_clean),
            ("clean -> defect", self.clean_on_defect),
        )
        lines = [
            f"10.3.1 cross replay -- {self.family}/{self.task_id}/{self.defect}, "
            f"{self.n_demos} demos, force limit {self.force_limit:g} N",
            f"  {'arm':<18} {'success%':>9} {'peak cfrc':>10}  failure profile",
        ]
        for name, runs in rows:
            profile = self._profile(runs)
            detail = ", ".join(
                f"{k}={v:.0f}%" for k, v in profile.items() if v > 0.0
            )
            lines.append(
                f"  {name:<18} {self._rate(runs):>8.1f}% {self._peak(runs):>10.2f}  {detail}"
            )
        lines += [
            f"  defect -> clean: success drop {self.success_drop_pp:+.1f} pp "
            f"(predicted >= {SUCCESS_DROP_PP:.0f}), peak cfrc rise "
            f"{self.cfrc_rise_n:+.2f} N (predicted >= {CFRC_RISE_N:.0f})",
            f"  clean -> defect: success drop {self.reverse_success_drop_pp:+.1f} pp, "
            "predicted as task failure without a rise in constraint violation",
            f"  G2 {'PASS' if self.passes_g2 else 'FAIL -- exit F1, no training run pays'}"
            f"; harm asymmetry {'holds' if self.asymmetry_holds else 'not observed'}",
        ]
        return "\n".join(lines)


def cross_replay(
    pair: TwinPair,
    *,
    seeds: tuple[int, ...],
    force_limit: float,
    feedforward_defective: dict[int, np.ndarray],
    feedforward_clean: dict[int, np.ndarray],
) -> CrossReplayReport:
    """Run 10.3.1 in both directions over `seeds`.

    Four rollouts per seed: each body driving itself closed loop, which produces
    the action sequences and the native baselines, and each body replaying the
    other's sequence open loop. The native arms are not optional -- a success
    drop is a difference against the body's own performance, and 10.3.1's 15 pp
    is stated against that baseline rather than against 100%.

    The per-seed feedforwards are the converged ILC columns for that demo, keyed
    by `seed_env`. There is no default: 9.6.7 item 3 makes the clean twin's
    feedforward non-zero, so an absent entry is a missing measurement.
    """
    tracker = OSCTracker()
    spec = load_spec(pair.task)
    native_d, native_c, d_on_c, c_on_d = [], [], [], []

    for seed in seeds:
        for label, source in (("defective", feedforward_defective), ("clean", feedforward_clean)):
            if seed not in source:
                raise ValueError(
                    f"no converged feedforward for seed {seed} on the {label} side; "
                    "cross replay compares demonstrations, not untrained rollouts"
                )

        def reset_pair(pair=pair, seed=seed) -> np.ndarray:
            return reset_twin_pair(pair, seed)

        reset_pair()
        ref = resolve(spec, pair.defective.env.sim)
        run_d = tracker.track(
            pair.defective, ref, family=pair.family, seed_env=seed,
            feedforward=feedforward_defective[seed],
        )
        reset_pair()
        run_c = tracker.track(
            pair.clean, ref, family=pair.family, seed_env=seed,
            feedforward=feedforward_clean[seed],
        )

        native_d.append(_as_open_loop(run_d))
        native_c.append(_as_open_loop(run_c))
        d_on_c.append(
            open_loop(
                pair.clean.env,
                run_d.actions,
                seed_env=seed,
                task_id=pair.task,
                reset=reset_pair,
            )
        )
        c_on_d.append(
            open_loop(
                pair.defective.env,
                run_c.actions,
                seed_env=seed,
                task_id=pair.task,
                reset=reset_pair,
            )
        )

    return CrossReplayReport(
        task_id=pair.task,
        family=pair.family,
        defect=str(pair.delta),
        n_demos=len(seeds),
        force_limit=force_limit,
        native_defect=tuple(native_d),
        native_clean=tuple(native_c),
        defect_on_clean=tuple(d_on_c),
        clean_on_defect=tuple(c_on_d),
    )


def _as_open_loop(result: Any) -> OpenLoopRun:
    """View a closed-loop `TrackingResult` as an `OpenLoopRun` for the same table."""
    return OpenLoopRun(
        actions=result.actions,
        states=result.states,
        peak_cfrc_ext=result.peak_cfrc_ext,
        max_wrist_force=float(np.max(np.linalg.norm(result.wrist_ft[:, :3], axis=1))),
        success=result.success,
        mode_hash=result.mode_hash,
    )


# ---------------------------------------------------------------------------
# 9.1.6: byte-exact replay from manifest.yaml + seed_env
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayCheck:
    """Whether re-running an address reproduced the recording bit for bit."""

    address: str
    states_equal: bool
    actions_equal: bool
    mode_hash_equal: bool
    max_state_diff: float
    max_action_diff: float
    state_sha256: tuple[str, str]

    @property
    def ok(self) -> bool:
        return self.states_equal and self.actions_equal and self.mode_hash_equal

    def render(self) -> str:
        return (
            f"9.1.6 byte-exact replay -- {self.address}: "
            f"{'bit-exact' if self.ok else 'NOT bit-exact'}\n"
            f"  states  : equal={self.states_equal}  max|d|={self.max_state_diff:g}\n"
            f"  actions : equal={self.actions_equal}  max|d|={self.max_action_diff:g}\n"
            f"  sha256  : {self.state_sha256[0][:16]} vs {self.state_sha256[1][:16]}"
        )


def _sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def replay_from_manifest(manifest_path: str, seed_env: int) -> Any:
    """Rebuild a cell from its manifest and re-run one demonstration.

    The manifest is the address 9.1.6 fixes: task, family, and the controller
    config hash the rollout was recorded under. The hash is checked rather than
    trusted -- a replay under different gains would run, produce a plausible
    trajectory, and silently not be the recorded one, which is the failure mode
    9.4.6 froze the gains to prevent.
    """
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)
    for key in ("task", "family", "controller_config_sha256"):
        if key not in manifest:
            raise ValueError(f"{manifest_path} has no {key!r}; it is not a replay address")

    tracker = OSCTracker()
    if tracker.gains.sha256() != manifest["controller_config_sha256"]:
        raise ValueError(
            f"{manifest_path} was recorded under controller config "
            f"{manifest['controller_config_sha256'][:16]} and the frozen gains hash to "
            f"{tracker.gains.sha256()[:16]}. 9.4.6 forbids replaying across a gain "
            "change; re-record or check out the commit that produced it"
        )
    env = make_env(manifest["task"], family=manifest["family"], seed_env=seed_env)
    try:
        return tracker.track(
            env, resolve(load_spec(manifest["task"]), env.sim),
            family=manifest["family"], seed_env=seed_env,
        )
    finally:
        env.close()


def verify_byte_exact(manifest_path: str, seed_env: int) -> ReplayCheck:
    """Run the same address twice and compare with `array_equal`, not a tolerance.

    9.1.6 is a bit-exactness requirement, and 11.2 puts the strictest version of
    it on `teacher/` because every ground truth downstream is derived from these
    trajectories. A difference of 1e-15 is still a failure: it means some state
    the address does not name is reaching the rollout.
    """
    first = replay_from_manifest(manifest_path, seed_env)
    second = replay_from_manifest(manifest_path, seed_env)
    return ReplayCheck(
        address=f"{manifest_path}#seed_env={seed_env}",
        states_equal=bool(np.array_equal(first.states, second.states)),
        actions_equal=bool(np.array_equal(first.actions, second.actions)),
        mode_hash_equal=first.mode_hash == second.mode_hash,
        max_state_diff=float(np.max(np.abs(first.states - second.states))),
        max_action_diff=float(np.max(np.abs(first.actions - second.actions))),
        state_sha256=(_sha256(first.states), _sha256(second.states)),
    )
