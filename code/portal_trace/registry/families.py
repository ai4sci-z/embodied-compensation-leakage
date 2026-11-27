"""The ten clean morphology families of 9.3.1, resolved against robosuite 1.5.2.

9.3.1 names ten families and sources four of them from `mujoco_menagerie`
(`franka_fr3`, `ufactory_lite6`, `trs_so_arm100`) or `robosuite_models`
(`ViperX-300`). Neither package is installed, and CONTRACT 3.6 forbids
substituting a plausible asset for a missing one, so this registry is built from
what `robosuite.models.robots.robot_model.REGISTERED_ROBOTS` actually contains.
`available_robosuite_robots()` reports that set, and `resolution_report()` says
family by family what was used and what the proposal asked for.

Three consequences of that resolution, all of them decisions rather than
accidents:

FR3 is unavailable, so its slot among the six main-experiment families goes to
`Kinova3` -- row 7 of the 9.3.1 table, robosuite-native, and the only remaining
7-DoF arm in the package with a distinct kinematic chain.

`PandaOmron`, `PandaDexLH` and `PandaDexRH` are *not* usable as extra families
even though they instantiate: their `RobotModel` bodies are the Panda arm, and
what differs (an Omron base, a dexterous hand) is attached later. Registering one
as a family would put a pair with FGW distance exactly zero into the population
that defines the within-cluster threshold of 9.3.4, which is the one number the
D4hi relaxed assertion is measured against.

The last three slots are filled by robots the package can instantiate but which
9.3.5 and 9.6.1 bar from every experiment arm (a legged mobile manipulator, a
bimanual arm, a mobile bimanual arm). They carry `ScopeExclusion` and are refused
by `main_families()`; they exist so that the registry spans the morphologies
robosuite 1.5.2 actually ships rather than silently shrinking to seven.

Every structural descriptor is read out of the *compiled* model, never out of the
raw MJCF. `robosuite.models.robots.robot_model.RobotModel.__init__` fills in
`frictionloss`, `damping` and `armature` with `force=False`, so a family that
declares them keeps its own values and one that does not gets robosuite's: Panda
ends up at `dof_frictionloss = 0.1` on every arm joint, UR5e at 0.01, XArm7 at
1.0. Reading the XML would give 0 for Panda and take `fgw_baseline` with it.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from enum import Enum
from functools import cache, lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from robosuite.models.robots.robot_model import RobotModel

__all__ = [
    "AssetSource",
    "FamilyRole",
    "ScopeExclusion",
    "NodeType",
    "GraphNode",
    "StructuralDescriptor",
    "MorphologyFamily",
    "FAMILIES",
    "MAIN_FAMILIES",
    "family",
    "main_families",
    "descriptor",
    "descriptor_from_model",
    "actuator_for_joint",
    "available_robosuite_robots",
    "resolution_report",
    "canonical_mjcf",
    "mjcf_sha256",
    "ROBOSUITE_ASSET_TOKEN",
]

# Absolute asset paths are baked into the XML by
# `robosuite.models.base.MujocoXML.resolve_asset_dependency`, so the raw string
# hashes differently on every machine. Rewriting the install prefix to a token
# makes `mjcf_sha256` a function of the robosuite version and nothing else,
# which is what 11.3 needs the manifest hash to be.
ROBOSUITE_ASSET_TOKEN = "<robosuite-assets>"


class AssetSource(str, Enum):
    """Where a family's MJCF comes from (9.3.6 `asset_source`).

    Only one member, because only one of the three sources 9.3.1 lists is
    installed. `mujoco_menagerie` and `robosuite_models` get members when they
    are, and the missing-source case raises rather than falling back.
    """

    ROBOSUITE = "robosuite"


class FamilyRole(str, Enum):
    """Whether a family reaches the primary endpoint arms."""

    MAIN = "main"            # one of the six that form the 36 twin pairs (9.3.3)
    RESERVE = "reserve"      # in scope, kept out of the 36 to keep n = 6 x 6
    OUT_OF_SCOPE = "out_of_scope"


class ScopeExclusion(str, Enum):
    """Why an instantiable robot is barred from every experiment arm."""

    DEXTEROUS_HAND = "dexterous_hand"                # 9.3.5
    MOBILE_MANIPULATOR = "mobile_manipulator"        # 9.3.5
    ACTION_SPACE_NOT_SEVEN = "action_space_not_seven"  # 9.6.1 fixes d_a = 7


class NodeType(str, Enum):
    """Node kinds of the baseline's morphology graph (arXiv:2602.18025).

    The baseline builds its graph over torso / joints / feet. A manipulator has
    no feet; its end effector is the node at which the chain terminates and the
    task is exerted, so it takes that role. Named `EEF` rather than `FOOT`
    because calling a gripper mount a foot in the code would be worse than the
    one-line explanation.
    """

    BASE = "base"
    JOINT = "joint"
    EEF = "eef"


@dataclass(frozen=True)
class GraphNode:
    """One node of a morphology graph, with everything either feature set reads.

    Every field is read from a compiled `mjModel`; none of them has a default.
    `torque_limit` comes from `actuator_ctrlrange` and not from
    `actuator_forcerange`: robosuite drives the arm with `motor` actuators whose
    `forcerange` is left at (0, 0), which MuJoCo reads as unlimited, while
    `ctrllimited="true" ctrlrange="-80 80"` in the MJCF is the limit that
    actually binds. `gain` is carried so that reading `ctrlrange` as a torque is
    checkable rather than assumed -- it is 1.0 for a `motor` shortcut and 0 for
    a passive joint, which has no actuator and hence a torque limit of zero.
    """

    name: str
    node_type: NodeType
    parent: str | None
    anchor: tuple[float, float, float]      # world anchor at qpos0, base frame origin
    axis: tuple[float, float, float]        # world joint axis at qpos0
    jnt_range: tuple[float, float]
    jnt_limited: bool
    torque_limit: tuple[float, float]
    gain: float
    armature: float
    damping: float
    frictionloss: float
    body_mass: float
    body_inertia: tuple[float, float, float]
    nominal_qpos: float

    @property
    def is_joint(self) -> bool:
        return self.node_type is NodeType.JOINT


@dataclass(frozen=True)
class StructuralDescriptor:
    """The kinematic graph plus the per-node physical descriptor of one model.

    This is the object 9.3.1 calls the family's structural descriptor and 9.3.4
    turns into (graph, node feature tensor). It is built from a compiled model,
    so it is equally well defined for a clean family and for the shadow-joint
    variant of D4hi -- which is what lets `twins` compare the two without a
    special case.
    """

    name: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[tuple[int, int], ...]
    joint_names: tuple[str, ...]
    dof: int
    chain_length_m: float
    mjcf_sha256: str

    def __post_init__(self) -> None:
        n = len(self.nodes)
        for i, j in self.edges:
            if not (0 <= i < n and 0 <= j < n):
                raise ValueError(f"{self.name}: edge ({i}, {j}) is out of range for {n} nodes")
            if i == j:
                raise ValueError(f"{self.name}: self loop at node {i}")
        if len(self.edges) != n - 1:
            raise ValueError(
                f"{self.name}: a kinematic tree on {n} nodes has {n - 1} edges, "
                f"found {len(self.edges)}"
            )

    @property
    def node_types(self) -> tuple[NodeType, ...]:
        return tuple(node.node_type for node in self.nodes)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def index_of(self, node_name: str) -> int:
        for i, node in enumerate(self.nodes):
            if node.name == node_name:
                return i
        raise KeyError(f"{self.name}: no node named {node_name!r}")

    def sorted_edges(self) -> tuple[tuple[str, str], ...]:
        """Edges as ordered name pairs, for the step (1) comparison of 9.3.4."""
        return tuple(
            sorted(
                (self.nodes[i].name, self.nodes[j].name)
                if self.nodes[i].name <= self.nodes[j].name
                else (self.nodes[j].name, self.nodes[i].name)
                for i, j in self.edges
            )
        )


@dataclass(frozen=True)
class MorphologyFamily:
    """One row of the 9.3.1 table, bound to a robot robosuite 1.5.2 can build."""

    family: str
    robosuite_robot: str
    asset_source: AssetSource
    role: FamilyRole
    proposal_row: str
    excluded_by: ScopeExclusion | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if (self.role is FamilyRole.OUT_OF_SCOPE) != (self.excluded_by is not None):
            raise ValueError(
                f"{self.family}: an out-of-scope family carries the exclusion that "
                "puts it out of scope, and an in-scope one carries none"
            )

    @property
    def in_36_pairs(self) -> bool:
        return self.role is FamilyRole.MAIN

    def robot_model(self) -> RobotModel:
        """A fresh robosuite `RobotModel` for this family."""
        from robosuite.models.robots.robot_model import REGISTERED_ROBOTS

        try:
            cls = REGISTERED_ROBOTS[self.robosuite_robot]
        except KeyError:
            raise KeyError(
                f"{self.family}: robosuite has no robot {self.robosuite_robot!r}; "
                f"installed robots are {sorted(REGISTERED_ROBOTS)}"
            ) from None
        return cls()

    def mjcf_path(self) -> Path:
        """The base MJCF shipped with robosuite, before robosuite's own edits.

        Kept for provenance only. The descriptor is not read from this file:
        `RobotModel.__init__` writes `frictionloss`, `damping` and `armature`
        into the tree afterwards, so the file on disk and the model that runs
        disagree on exactly the three fields the baseline feature set reads.
        """
        return Path(self.robot_model().file)

    def canonical_xml(self) -> str:
        return canonical_mjcf(self.robot_model().get_xml())

    def mjcf_sha256(self) -> str:
        return mjcf_sha256(self.robot_model().get_xml())

    def compiled_model(self) -> mujoco.MjModel:
        """The compiled family model: arm joints only, no gripper, no base.

        `RobotModel.get_model` compiles the same string `get_xml` returns, so
        the hash in the manifest is the hash of what was compiled.
        """
        return self.robot_model().get_model(mode="mujoco")

    def descriptor(self) -> StructuralDescriptor:
        return descriptor(self.family)

    def require_in_36_pairs(self) -> None:
        if self.in_36_pairs:
            return
        reason = (
            f"9.3.5 excludes it ({self.excluded_by.value})"
            if self.excluded_by is not None
            else "it is a reserve family and 11.3 fixes n = 36 = 6 x 6"
        )
        raise ValueError(f"{self.family} is not one of the six main families: {reason}")


# ---------------------------------------------------------------------------
# MJCF canonicalisation and hashing
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _robosuite_assets_root() -> str:
    import robosuite

    return os.path.abspath(os.path.join(os.path.dirname(robosuite.__file__), "models", "assets"))


def canonical_mjcf(xml: str) -> str:
    """Rewrite install-specific asset paths so the hash is machine independent."""
    return xml.replace(_robosuite_assets_root(), ROBOSUITE_ASSET_TOKEN)


def mjcf_sha256(xml: str) -> str:
    return hashlib.sha256(canonical_mjcf(xml).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Descriptor extraction
# ---------------------------------------------------------------------------


def actuator_for_joint(model: mujoco.MjModel, joint_id: int) -> int | None:
    """The single actuator transmitting to a joint, or `None` for a passive one.

    A passive joint is not a missing measurement: D4hi's shadow joint (9.2.4)
    carries no actuator by construction, and the torque an absent actuator can
    apply is exactly zero. Two actuators on one joint is a modelling error and
    raises, because then the node's torque limit is genuinely ambiguous.
    """
    hits = [
        a
        for a in range(model.nu)
        if int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT)
        and int(model.actuator_trnid[a, 0]) == joint_id
    ]
    if len(hits) > 1:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        raise ValueError(
            f"joint {name!r} is driven by {len(hits)} actuators; the torque limit of "
            "a node is only defined when at most one actuator transmits to it"
        )
    return hits[0] if hits else None


def _torque_limit(model: mujoco.MjModel, actuator_id: int) -> tuple[tuple[float, float], float]:
    """Torque limit and gain of a motor actuator.

    `actuator_forcerange` is (0, 0) on every robosuite arm actuator, which
    MuJoCo reads as unlimited, so the binding limit is `ctrlrange` -- valid as a
    torque only because the `motor` shortcut compiles to an affine gain of 1
    with no bias. Both facts are checked here rather than assumed, because a
    family whose MJCF used `position` or `general` actuators would otherwise get
    a control setpoint recorded as if it were a torque.
    """
    if int(model.actuator_gaintype[actuator_id]) != int(mujoco.mjtGain.mjGAIN_FIXED):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        raise ValueError(f"actuator {name!r} is not a fixed-gain motor; ctrlrange is not a torque")
    if int(model.actuator_biastype[actuator_id]) != int(mujoco.mjtBias.mjBIAS_NONE):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        raise ValueError(f"actuator {name!r} has a bias term; ctrlrange is not a torque")
    if not bool(model.actuator_ctrllimited[actuator_id]):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        raise ValueError(
            f"actuator {name!r} has no ctrlrange, so this family declares no torque "
            "limit; 9.3.4 lists the torque limit as a node feature and there is no "
            "value to substitute"
        )
    lo, hi = (float(v) for v in model.actuator_ctrlrange[actuator_id])
    return (lo, hi), float(model.actuator_gainprm[actuator_id, 0])


def _nearest_ancestor_joint(
    model: mujoco.MjModel, body_id: int, joints_of_body: dict[int, list[int]]
) -> int | None:
    """The last joint of the closest ancestor body that carries one.

    "Last" rather than "the" because a body may hold several joints, which
    MuJoCo composes in declaration order: D4hi puts its shadow joint in the same
    body as the driven joint (9.2.4), so the chain through that body is
    joint4 -> joint4_lash and everything below hangs off the second one.
    """
    b = int(model.body_parentid[body_id])
    while b > 0:
        if b in joints_of_body:
            return joints_of_body[b][-1]
        b = int(model.body_parentid[b])
    return None


def _chain_length(model: mujoco.MjModel, from_body: int, to_body: int) -> float:
    """Fully extended length of the body chain, in metres.

    The sum of the successive frame offsets from `from_body` to `to_body`. For a
    serial revolute chain this is the reach at full extension, hence an upper
    bound on the manufacturer figure of the 9.3.1 reach column. It is *not* that
    figure: the datasheet reach is measured from the shoulder to the wrist flange
    and is not derivable from the installed assets, and it is not one of the
    `REQUIRED_MEASUREMENTS`, so this module reports what the model contains and
    leaves the datasheet column empty rather than guessing it.
    """
    total = 0.0
    b = to_body
    while b != from_body:
        total += float(np.linalg.norm(model.body_pos[b]))
        parent = int(model.body_parentid[b])
        if parent == b:
            raise ValueError("chain walk reached the world without meeting the base body")
        b = parent
    return total


def descriptor_from_model(
    model: mujoco.MjModel,
    *,
    name: str,
    joint_names: tuple[str, ...],
    base_body: str,
    eef_bodies: tuple[str, ...],
    mjcf_sha256_hex: str,
) -> StructuralDescriptor:
    """Build the 9.3.4 morphology graph from a compiled model.

    Node set: the base body, one node per named joint, one node per end
    effector. Edge set: the kinematic tree restricted to those nodes, i.e. each
    node is joined to its nearest ancestor that is also a node. That is the
    graph the baseline builds (torso / joints / feet, arXiv:2602.18025), and
    keeping the same construction for both feature sets is what makes the two
    FGW numbers comparable.

    `joint_names` fixes the node order and must be given in MuJoCo declaration
    order within each body: several joints on one body compose in that order, so
    reordering them would describe a different kinematic chain.

    Poses are evaluated at `qpos0` with `mj_forward`, so `xanchor` and `xaxis`
    are the joint's world anchor and axis in the model's own reference
    configuration -- a deterministic function of the MJCF, not of any rollout.
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body)
    if base_id < 0:
        raise KeyError(f"{name}: no body named {base_body!r}")

    joint_ids: list[int] = []
    for jname in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise KeyError(f"{name}: no joint named {jname!r} in the compiled model")
        joint_ids.append(jid)
    joints_of_body: dict[int, list[int]] = {}
    for jid in joint_ids:
        joints_of_body.setdefault(int(model.jnt_bodyid[jid]), []).append(jid)
    for body_id, ids in joints_of_body.items():
        if ids != sorted(ids):
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            raise ValueError(
                f"{name}: the joints of body {bname!r} are listed out of declaration "
                "order, which would describe a different composition of transforms"
            )

    base_pos = np.asarray(data.xpos[base_id], dtype=np.float64)

    nodes: list[GraphNode] = [
        GraphNode(
            name=base_body,
            node_type=NodeType.BASE,
            parent=None,
            anchor=(0.0, 0.0, 0.0),
            axis=(0.0, 0.0, 0.0),
            jnt_range=(0.0, 0.0),
            jnt_limited=False,
            torque_limit=(0.0, 0.0),
            gain=0.0,
            armature=0.0,
            damping=0.0,
            frictionloss=0.0,
            body_mass=float(model.body_mass[base_id]),
            body_inertia=tuple(float(v) for v in model.body_inertia[base_id]),
            nominal_qpos=0.0,
        )
    ]
    node_index: dict[str, int] = {base_body: 0}
    edges: list[tuple[int, int]] = []

    for jid in joint_ids:
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        body_id = int(model.jnt_bodyid[jid])
        dof_adr = int(model.jnt_dofadr[jid])
        qpos_adr = int(model.jnt_qposadr[jid])
        act = actuator_for_joint(model, jid)
        torque, gain = ((0.0, 0.0), 0.0) if act is None else _torque_limit(model, act)
        siblings = joints_of_body[body_id]
        position = siblings.index(jid)
        parent_joint = (
            siblings[position - 1]
            if position > 0
            else _nearest_ancestor_joint(model, body_id, joints_of_body)
        )
        parent_name = (
            base_body
            if parent_joint is None
            else mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, parent_joint)
        )
        nodes.append(
            GraphNode(
                name=jname,
                node_type=NodeType.JOINT,
                parent=parent_name,
                anchor=tuple(float(v) for v in (np.asarray(data.xanchor[jid]) - base_pos)),
                axis=tuple(float(v) for v in data.xaxis[jid]),
                jnt_range=tuple(float(v) for v in model.jnt_range[jid]),
                jnt_limited=bool(model.jnt_limited[jid]),
                torque_limit=torque,
                gain=gain,
                armature=float(model.dof_armature[dof_adr]),
                damping=float(model.dof_damping[dof_adr]),
                frictionloss=float(model.dof_frictionloss[dof_adr]),
                body_mass=float(model.body_mass[body_id]),
                body_inertia=tuple(float(v) for v in model.body_inertia[body_id]),
                nominal_qpos=float(model.qpos0[qpos_adr]),
            )
        )
        node_index[jname] = len(nodes) - 1
        edges.append((node_index[parent_name], node_index[jname]))

    for eef in eef_bodies:
        eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, eef)
        if eef_id < 0:
            raise KeyError(f"{name}: no end-effector body named {eef!r}")
        parent_joint = _nearest_ancestor_joint(model, eef_id, joints_of_body)
        parent_name = (
            base_body
            if parent_joint is None
            else mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, parent_joint)
        )
        nodes.append(
            GraphNode(
                name=eef,
                node_type=NodeType.EEF,
                parent=parent_name,
                anchor=tuple(float(v) for v in (np.asarray(data.xpos[eef_id]) - base_pos)),
                axis=(0.0, 0.0, 0.0),
                jnt_range=(0.0, 0.0),
                jnt_limited=False,
                torque_limit=(0.0, 0.0),
                gain=0.0,
                armature=0.0,
                damping=0.0,
                frictionloss=0.0,
                body_mass=float(model.body_mass[eef_id]),
                body_inertia=tuple(float(v) for v in model.body_inertia[eef_id]),
                nominal_qpos=0.0,
            )
        )
        node_index[eef] = len(nodes) - 1
        edges.append((node_index[parent_name], node_index[eef]))

    first_eef = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, eef_bodies[0])
    return StructuralDescriptor(
        name=name,
        nodes=tuple(nodes),
        edges=tuple(edges),
        joint_names=tuple(joint_names),
        dof=len(joint_ids),
        chain_length_m=_chain_length(model, base_id, first_eef),
        mjcf_sha256=mjcf_sha256_hex,
    )


@cache
def descriptor(family_name: str) -> StructuralDescriptor:
    """The structural descriptor of a clean family. Cached; compiling is not free."""
    fam = family(family_name)
    robot = fam.robot_model()
    model = robot.get_model(mode="mujoco")
    return descriptor_from_model(
        model,
        name=family_name,
        joint_names=tuple(robot.joints),
        base_body=robot.root_body,
        eef_bodies=tuple(robot.eef_name[arm] for arm in robot.arms),
        mjcf_sha256_hex=mjcf_sha256(robot.get_xml()),
    )


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

_FAMILIES: tuple[MorphologyFamily, ...] = (
    MorphologyFamily(
        family="Panda",
        robosuite_robot="Panda",
        asset_source=AssetSource.ROBOSUITE,
        role=FamilyRole.MAIN,
        proposal_row="Panda, 7 DoF, robosuite native",
    ),
    MorphologyFamily(
        family="Sawyer",
        robosuite_robot="Sawyer",
        asset_source=AssetSource.ROBOSUITE,
        role=FamilyRole.MAIN,
        proposal_row="Sawyer, 7 DoF, robosuite native",
        notes=(
            "the only main family whose body tree branches: the head, the screen and "
            "the ITB sensors hang off the same links as the arm. They carry no joint, "
            "so they are not nodes, and the morphology graph is still the arm chain"
        ),
    ),
    MorphologyFamily(
        family="IIWA",
        robosuite_robot="IIWA",
        asset_source=AssetSource.ROBOSUITE,
        role=FamilyRole.MAIN,
        proposal_row="IIWA14, 7 DoF, robosuite native",
    ),
    MorphologyFamily(
        family="Kinova3",
        robosuite_robot="Kinova3",
        asset_source=AssetSource.ROBOSUITE,
        role=FamilyRole.MAIN,
        proposal_row="Kinova Gen3, 7 DoF, robosuite native -- promoted into FR3's slot",
        notes=(
            "9.3.1 puts Kinova Gen3 outside the main six and fills the sixth slot with "
            "FR3 from mujoco_menagerie. That package is not installed, and CONTRACT 3.6 "
            "forbids substituting a different Franka; Kinova3 is the next row of the "
            "same table and is native, so it takes the slot and FR3 becomes a W1 item"
        ),
    ),
    MorphologyFamily(
        family="XArm7",
        robosuite_robot="XArm7",
        asset_source=AssetSource.ROBOSUITE,
        role=FamilyRole.MAIN,
        proposal_row="xArm7, 7 DoF -- native here, not from robosuite_models",
        notes=(
            "9.3.1 sources xArm7 from robosuite_models. robosuite 1.5.2 ships it "
            "natively, so the fallback that row anticipated is not needed. Its MJCF "
            "declares frictionloss=1.0, ten times Panda's robosuite default"
        ),
    ),
    MorphologyFamily(
        family="UR5e",
        robosuite_robot="UR5e",
        asset_source=AssetSource.ROBOSUITE,
        role=FamilyRole.MAIN,
        proposal_row="UR5e, 6 DoF, robosuite native",
        notes="the only 6-DoF arm among the main six, so it carries the DoF contrast alone",
    ),
    MorphologyFamily(
        family="Jaco",
        robosuite_robot="Jaco",
        asset_source=AssetSource.ROBOSUITE,
        role=FamilyRole.RESERVE,
        proposal_row="not in the 9.3.1 table; the seventh native fixed-base arm",
        notes=(
            "in scope and instantiable, held out of the 36 pairs because 11.3 fixes "
            "n = 6 x 6. It is the stand-in for a main family that fails P0 admission"
        ),
    ),
    MorphologyFamily(
        family="SpotWithArm",
        robosuite_robot="SpotWithArm",
        asset_source=AssetSource.ROBOSUITE,
        role=FamilyRole.OUT_OF_SCOPE,
        proposal_row="not in the 9.3.1 table; fills a slot left empty by trs_so_arm100",
        excluded_by=ScopeExclusion.MOBILE_MANIPULATOR,
        notes=(
            "a 6-DoF arm on a legged base. Registered because it is the only other "
            "distinct arm chain robosuite 1.5.2 can build, and refused everywhere "
            "9.3.5 refuses mobile manipulators"
        ),
    ),
    MorphologyFamily(
        family="Baxter",
        robosuite_robot="Baxter",
        asset_source=AssetSource.ROBOSUITE,
        role=FamilyRole.OUT_OF_SCOPE,
        proposal_row="not in the 9.3.1 table; fills a slot left empty by ViperX-300",
        excluded_by=ScopeExclusion.ACTION_SPACE_NOT_SEVEN,
        notes=(
            "2 x 7 DoF, action dimension 14. 9.6.1 fixes d_a = 7, so it cannot share "
            "the policy interface with the other families"
        ),
    ),
    MorphologyFamily(
        family="Tiago",
        robosuite_robot="Tiago",
        asset_source=AssetSource.ROBOSUITE,
        role=FamilyRole.OUT_OF_SCOPE,
        proposal_row="not in the 9.3.1 table; fills a slot left empty by ufactory_lite6",
        excluded_by=ScopeExclusion.MOBILE_MANIPULATOR,
        notes=(
            "mobile bimanual, and the only registered family with a prismatic joint "
            "(the torso lift). Kept for that: it is the one model here on which the "
            "descriptor's joint-type handling is exercised by something other than a hinge"
        ),
    ),
)

FAMILIES: dict[str, MorphologyFamily] = {f.family: f for f in _FAMILIES}
MAIN_FAMILIES: tuple[str, ...] = tuple(f.family for f in _FAMILIES if f.in_36_pairs)


def family(name: str) -> MorphologyFamily:
    try:
        return FAMILIES[name]
    except KeyError:
        raise KeyError(f"unknown family {name!r}; registered: {sorted(FAMILIES)}") from None


def main_families() -> tuple[MorphologyFamily, ...]:
    return tuple(FAMILIES[n] for n in MAIN_FAMILIES)


def available_robosuite_robots() -> tuple[str, ...]:
    """Every robot the installed robosuite can build, whether we use it or not."""
    from robosuite.models.robots.robot_model import REGISTERED_ROBOTS

    return tuple(sorted(REGISTERED_ROBOTS))


def resolution_report() -> str:
    """Family by family, what 9.3.1 asked for and what the package supplied."""
    lines = [
        f"robosuite robots registered in this install: {len(available_robosuite_robots())}",
        "  " + ", ".join(available_robosuite_robots()),
        "",
        f"{'family':14s} {'robot':14s} {'role':13s} {'dof':>4s} {'nodes':>6s} "
        f"{'chain_m':>8s}  proposal row",
    ]
    for fam in _FAMILIES:
        d = fam.descriptor()
        lines.append(
            f"{fam.family:14s} {fam.robosuite_robot:14s} {fam.role.value:13s} "
            f"{d.dof:4d} {d.n_nodes:6d} {d.chain_length_m:8.4f}  {fam.proposal_row}"
        )
    return "\n".join(lines)


if len(_FAMILIES) != 10:
    raise AssertionError(f"9.3.1 fixes the family count at ten; found {len(_FAMILIES)}")
if len(MAIN_FAMILIES) != 6:
    raise AssertionError(
        f"9.3.3 needs exactly six main families for the 36 pairs; found {len(MAIN_FAMILIES)}"
    )


def main() -> int:
    """Instantiate every registered family and print what the package supplied.

    Building the descriptor compiles the model, so this doubles as the check
    that all ten rows of 9.3.1 resolve to something the installed robosuite can
    actually produce.
    """
    print(resolution_report())
    print()
    for name in MAIN_FAMILIES:
        d = descriptor(name)
        print(f"{name:14s} sha256={d.mjcf_sha256}  joints={len(d.joint_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
