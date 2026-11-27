"""A13: the teacher-family ablation of 10.6.1 -- ILC / analytic script / RL expert.

The row is an external-validity check and its question is narrow, so the narrow
form is what is implemented here. 10.6.1 words it as: does the compensation
*direction* an RL agent arrives at on its own agree with the ILC teacher's
(cosine), asking about direction only -- not about magnitude, and not about
whether the leakage the two induce is the same. Appendix A-18 is the reason for
that narrowness: c_delta is defined as the difference between two policies at
the same state, so a policy retrained on the twin is a different policy and its
difference from the defective one is not the compensation field. An RL expert
with a random seed is not the deterministic operator assumption (A2) asks for
either, which is why the RL level carries a frozen checkpoint, the mean action
rather than a sample, and a fixed seed, and why nothing in this module produces
a preregistered endpoint of the primary or secondary family.

The three families
------------------

`ilc` is the main setting and is not re-implemented: it is `teacher.ilc.run_ilc`
and its converged feedforward.

`analytic_script` is the closed-form compensation of `defects.fields`, replayed
as a feedforward rather than solved for. It is the second teacher family in the
proposal's sense -- a hand-written compensation script -- and it is the same
object 9.4.7's cosine floor already gates the ILC against, which is what makes
the comparison interpretable: `preregistration.threshold(
"analytic_vs_ilc_cosine_floor")` is 0.80, and the A13 cosine is read against
that same floor rather than against a new one invented for the ablation.
`fields.NoAnalyticField` propagates: the defects with a numerical ground truth
only (9.2.7) have no analytic teacher, and substituting a twin-difference
estimate would put an object of a different kind into the comparison.

`rl_expert` is a policy trained with reinforcement learning on the defective
embodiment, then frozen. The trainer is `ablations.teacher_family_rl`: nothing
in the 9.1 stack fits a value function -- robomimic 0.5.0 is a behaviour-cloning
library -- so the agent is stable-baselines3 SAC driven through a gymnasium view
of `defects.inject.DefectWrapper`, i.e. of the broken robot itself, because an
agent trained on a plain robosuite env would answer a question about the clean
body.

What stays here is the *contract*, and it is unchanged by the trainer's
existence: `RLExpertTeacher` loads the artefact and refuses, by name, when there
is not one. It does not fall back to the ILC teacher, to a scripted policy or to
a random one, and it does not quietly train one either. The level's whole content
is that the compensation direction was arrived at *independently*, so the run
that produced it is an event with its own record -- seeds, reward caliber,
observation layout and package versions travel in the checkpoint -- rather than
a side effect of asking for a cosine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "TeacherFamily",
    "TeacherFamilyUnavailable",
    "TeacherFamilySpec",
    "ObservationLayout",
    "AnalyticScriptTeacher",
    "RLExpertTeacher",
    "direction_cosine",
]


class TeacherFamily(str, Enum):
    """The three families of the A13 row. The value is what the config writes."""

    ILC = "ilc"
    ANALYTIC_SCRIPT = "analytic_script"
    RL_EXPERT = "rl_expert"


class TeacherFamilyUnavailable(RuntimeError):
    """A teacher family cannot be instantiated, and says exactly what is absent.

    Raised instead of returning a substitute demonstrator. A13's endpoint is an
    agreement cosine between two independently obtained compensation
    directions; a substitute would make the agreement true by construction, and
    a reader could not tell the two cases apart from the number.
    """


@dataclass(frozen=True)
class TeacherFamilySpec:
    """One A13 level, as the preregistered level block states it."""

    family: TeacherFamily
    demonstrator: str
    checkpoint: Path | None = None
    deterministic: bool = True

    def __post_init__(self) -> None:
        if self.family is TeacherFamily.RL_EXPERT:
            if not self.deterministic:
                raise ValueError(
                    "A13's RL level is specified as frozen weights, the mean action "
                    "and a fixed seed (10.6.1). A sampling policy is not the "
                    "deterministic operator assumption (A2) requires, and the "
                    "per-frame ground truth would stop being reproducible"
                )
            if self.checkpoint is None:
                raise ValueError(
                    "A13's RL level needs the path of the frozen checkpoint. The "
                    "trainer is `ablations.teacher_family_rl`, but the level block is "
                    "a statement about the artefact and not about the run: it has to "
                    "exist under the name the preregistered configuration fixes before "
                    "the ablation reads a cosine off it"
                )
        elif self.checkpoint is not None:
            raise ValueError(
                f"{self.family.value}: a checkpoint is the RL level's parameter; this "
                "family is computed, not loaded"
            )

    @classmethod
    def from_parameters(cls, parameters: dict[str, Any], *, where: str) -> TeacherFamilySpec:
        missing = sorted({"family", "demonstrator"} - set(parameters))
        if missing:
            raise ValueError(f"{where}: A13 level is missing {missing}")
        checkpoint = parameters.get("checkpoint")
        return cls(
            family=TeacherFamily(str(parameters["family"])),
            demonstrator=str(parameters["demonstrator"]),
            checkpoint=None if checkpoint is None else Path(str(checkpoint)),
            deterministic=bool(parameters.get("deterministic", True)),
        )


class AnalyticScriptTeacher:
    """The closed-form compensation of `defects.fields`, used as a feedforward.

    Built from a `DefectWrapper` so that the joint indices and the OSC gains are
    the injector's own, not a second transcription of them -- the same rule
    `FieldContext.from_wrapper` exists to enforce for the diagnostics.
    """

    family = TeacherFamily.ANALYTIC_SCRIPT

    def __init__(self, wrapper: Any) -> None:
        from portal_trace.defects.fields import FieldContext, field_for

        self.wrapper = wrapper
        self.context = FieldContext.from_wrapper(wrapper)
        # `NoAnalyticField` is not caught: a defect with a numerical ground
        # truth only has no analytic teacher, and 9.2.7 makes that a property of
        # the defect rather than a gap to be filled in.
        self.field = field_for(wrapper.delta, self.context)

    def feedforward(self, states: np.ndarray) -> np.ndarray:
        """(N, nq + nv) states in, (N, action_dim) feedforward out.

        The sign convention is the field's: `defects.fields` returns the
        compensation a teacher would have to add, so the script's feedforward is
        the field itself and no negation happens on the way.
        """
        model = self.wrapper.env.sim.model._model
        data = self.wrapper.env.sim.data._data
        return np.asarray(self.field.evaluate(states, model, data), dtype=np.float64)


@dataclass(frozen=True)
class ObservationLayout:
    """The key order and widths the RL expert's input vector was built from.

    A file on disk separates the trainer from `RLExpertTeacher`, and the one thing
    the weights cannot carry implicitly is which observation the first layer
    expects. robosuite hands out an `OrderedDict` of about twenty entries plus two
    concatenated modality blocks, so a flattened vector only means something
    together with the order it was flattened in: two orders of the same keys have
    the same width and the same dtype, and feeding one to a policy trained on the
    other is a silent permutation of its input. The cosine A13 reports would then
    be a property of that permutation rather than of the agent, and nothing in
    the number would show it. So the layout is measured on the body the expert
    trained on, travels in the checkpoint, and is the only implementation of the
    concatenation that either side uses.
    """

    keys: tuple[str, ...]
    widths: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.keys:
            raise ValueError("an observation layout with no keys is not a layout")
        if len(self.keys) != len(self.widths):
            raise ValueError(
                f"{len(self.keys)} keys against {len(self.widths)} widths: the layout "
                "has to pin one width per key or it cannot be checked"
            )
        if len(set(self.keys)) != len(self.keys):
            raise ValueError(
                f"duplicate key in {self.keys}; the concatenation would carry the same "
                "block twice and the width would still look plausible"
            )
        for key, width in zip(self.keys, self.widths, strict=True):
            if width <= 0:
                raise ValueError(f"observation key {key!r} has width {width}")

    @property
    def total_width(self) -> int:
        return sum(self.widths)

    @classmethod
    def measure(
        cls, observations: Mapping[str, Any], keys: Sequence[str]
    ) -> ObservationLayout:
        """Read the widths off a real observation dict of the target body.

        Measured rather than tabulated for the reason CONTRACT section 6 records
        for the proprioceptive width: `object-state` is as wide as the task's
        object set and the proprioceptive block as wide as the arm, so a table
        here would be a second transcription of something robosuite already
        computes, and the two would part company on the first family that is not
        a 7-DoF Panda.
        """
        missing = [key for key in keys if key not in observations]
        if missing:
            raise KeyError(
                f"the environment emits no {missing} to build the A13 expert's "
                f"observation from; it offers {sorted(observations)}. The layout is "
                "not narrowed to what happens to be present: a policy trained on a "
                "shorter vector is a different policy, and A13 compares directions "
                "produced by the one the checkpoint records"
            )
        return cls(
            keys=tuple(keys),
            widths=tuple(int(np.asarray(observations[key]).size) for key in keys),
        )

    def flatten(self, observations: Mapping[str, Any]) -> np.ndarray:
        """One observation dict -> one float32 row, in the recorded order.

        float32 and not the env's float64 because this vector's only destination
        is an `nn.Linear`, which casts anyway: allocating a million-step replay
        buffer at float64 would double it to throw the low bits away at the first
        layer. The cast belongs to the layout rather than to either side of the
        file, so that the frozen expert is asked its opinion on exactly the bits
        the trainer showed it.
        """
        blocks = [
            np.asarray(observations[key], dtype=np.float32).reshape(-1)
            for key in self.keys
        ]
        widths = tuple(int(b.size) for b in blocks)
        if widths != self.widths:
            raise ValueError(
                f"this body emits widths {widths} for {list(self.keys)} and the "
                f"checkpoint records {self.widths}. That is a different embodiment or "
                "a different task, not a reshape: the policy's input layer is sized "
                "for the recorded one"
            )
        return np.concatenate(blocks)

    def to_json(self) -> dict[str, Any]:
        return {
            "keys": list(self.keys),
            "widths": list(self.widths),
            "total_width": self.total_width,
            "dtype": "float32",
        }

    @classmethod
    def from_json(cls, blob: Mapping[str, Any]) -> ObservationLayout:
        return cls(
            keys=tuple(str(k) for k in blob["keys"]),
            widths=tuple(int(w) for w in blob["widths"]),
        )


class RLExpertTeacher:
    """A frozen RL policy on the defective embodiment; mean action, fixed seed.

    The class exists so that the level has a real contract rather than a
    configuration stanza: the checkpoint has to be on disk, the policy has to be
    in evaluation mode with every parameter frozen, and the action has to be the
    distribution's mean. Each of the three is checked, and a failure names which
    one failed.

    It is the reading half of the level and does not train. That is a separation
    of responsibility rather than a missing feature: `teacher_family_rl` trains,
    and it does so as a run with a recorded seed, reward caliber and observation
    layout. Producing the expert here, on demand, would put an unrecorded
    training run inside the evaluation of a cosine and leave A13's headline number
    depending on when it was first asked for.
    """

    family = TeacherFamily.RL_EXPERT

    def __init__(self, spec: TeacherFamilySpec) -> None:
        if spec.family is not TeacherFamily.RL_EXPERT:
            raise ValueError(f"{spec.family.value} is not the RL expert level")
        if spec.checkpoint is None or not spec.checkpoint.exists():
            raise TeacherFamilyUnavailable(
                f"A13's RL expert level needs a frozen checkpoint at "
                f"{spec.checkpoint}, and there is none. It is an artefact of its own "
                "training run and is produced by\n"
                "    portal-trace teacher --experiment A13 --task <task> "
                "--family <family> --defect <defect> --level <level>\n"
                "which trains stable-baselines3 SAC on the defective embodiment "
                "(`ablations.teacher_family_rl.train_rl_expert`), freezes the weights "
                "and writes the mean-action module together with its seeds, its "
                "reward caliber and its observation layout. Nothing here substitutes "
                "the ILC teacher, a scripted policy or a random policy for it, and "
                "nothing trains one on the way past: the whole content of the level is "
                "that the direction was arrived at independently, and a cosine read "
                "off a stand-in would look exactly like a cosine read off an agent "
                "(10.6.1, appendix A-18)"
            )
        self.spec = spec
        self.policy = self._load(spec.checkpoint)

    @staticmethod
    def _load(checkpoint: Path) -> Any:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        policy = payload.get("policy") if isinstance(payload, dict) else payload
        if policy is None:
            raise TeacherFamilyUnavailable(
                f"{checkpoint} carries no `policy` entry; A13 needs the frozen module, "
                "not a metrics dump"
            )
        policy.eval()
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        return policy

    def feedforward(self, observations: Any) -> np.ndarray:
        """The mean action, never a sample.

        10.6.1 spells the level as "frozen weights + mean action + fixed seed".
        A policy that exposes only a sampling interface is refused rather than
        sampled once: a single draw would make the cosine a function of the
        draw, and the ablation is about the agent's direction, not one of its
        realisations.
        """
        import torch

        for name in ("mean_action", "deterministic_action"):
            if hasattr(self.policy, name):
                with torch.no_grad():
                    return np.asarray(
                        getattr(self.policy, name)(observations), dtype=np.float64
                    )
        raise TeacherFamilyUnavailable(
            f"{type(self.policy).__name__} exposes neither `mean_action` nor "
            "`deterministic_action`. 10.6.1 fixes the RL level at the mean action; "
            "drawing one sample instead would make the reported cosine a property of "
            "that draw"
        )


def direction_cosine(reference: np.ndarray, other: np.ndarray) -> float:
    """cos(reference, other) over the stacked frames, magnitude divided out.

    The A13 endpoint asks about direction only, so the two feedforwards are
    compared as one vector each in the flattened (frames x action dimensions)
    space rather than frame by frame: a per-frame mean would weight the frames
    where both fields are near zero as heavily as the ones carrying the
    compensation, and 9.4.7's cosine -- the floor this number is read against --
    is defined the same way.
    """
    a = np.asarray(reference, dtype=np.float64).reshape(-1)
    b = np.asarray(other, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(
            f"the two teachers produced {a.shape} and {b.shape} feedforward samples; "
            "A13 compares them on the same frames"
        )
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        raise ValueError(
            "one of the two teachers produced an identically zero feedforward, so the "
            "angle between them is undefined. That is a finding about the teacher, "
            "not a cosine of zero"
        )
    return float(a @ b / denominator)
