"""The 36 structurally aligned twin pairs and the 9.3.4 invariance assertion.

11.3 fixes n = 36 = 6 morphology families x 6 core defects. 9.3.3 says why the
pairs are structurally aligned by construction: four of the six core defects
write only the wrapper and `mjData`, so the compiled `mjModel` on the defective
side of a pair is the same table as on the clean side. The morphology graph is a
function of that table, so the two graphs are equal element by element and the
FGW distance is zero -- under *any* feature set, which is the form of the claim
X1 needs.

D8 is the fifth, and it is the one that makes the claim testable. It writes
`actuator_gainprm` (9.2.7), so its two sides are genuinely two compilations and
their `GraphNode.gain` differs. The distance is still zero because no node
feature in either set of 9.3.4 reads that field -- both take the torque limit
from `actuator_ctrlrange`, which D8 leaves alone. `twin_descriptors` builds both
sides rather than returning one object twice, so the D8 rows of the table are a
measurement while the four `mjData` rows are, unavoidably, a tautology. D8 is in
the core six at all because 9.2.6's stiction rule took D6 out: moving D6's
symmetric component to `qfrc_applied` -- forced, because the baseline's node
features read a friction coefficient -- costs a breakaway offset that measured
0.64 to 3.52 N m against a 0.025 N m gate.

D4hi is the sixth and the registered exception (9.2.4, 9.3.3, 14.5 E-3). Its
high-fidelity
caliber adds a shadow joint co-located with the driven joint, so it does change
the graph, and the clean twin therefore carries the same shadow joint with
`range="-1e-5 1e-5"`. The graphs then agree node for node with bit-identical
anchors and axes, and differ in exactly one entry of one node's feature row --
`jnt_range` on the shadow node. 9.3.4 replaces the exact assertion with
`d_FGW(twin) < (1/10) min over real morphology pairs`, and requires the number
to be printed rather than described as small. `report()` prints it, next to the
threshold and next to the strictly larger number the *unrelaxed* comparison
gives (defective side with the shadow joint against a clean family without one),
which is what `graph_exception: shadow_joint` in the registry refers to.

Both feature sets are asserted every time. `fgw_ours` and `fgw_baseline` are
different propositions -- "our descriptor cannot separate the pair" and "the
published descriptor cannot separate the pair" -- and 9.3.4 forbids reporting
only one.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from functools import cache
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from portal_trace.defects.delta import Delta
from portal_trace.defects.params import CORE_DEFECTS, DefectSpec, WritableField, defect
from portal_trace.lockfile import LockFile
from portal_trace.registry.families import (
    MAIN_FAMILIES,
    StructuralDescriptor,
    actuator_for_joint,
    descriptor,
    descriptor_from_model,
    family,
    mjcf_sha256,
)
from portal_trace.registry.fgw import (
    FeatureSet,
    alpha_for,
    build_morphology_graph,
    feature_fields,
    fgw_distance,
    node_features,
)
from portal_trace.types import EmbodimentId

if TYPE_CHECKING:  # pragma: no cover - typing only
    from robosuite.models.robots.robot_model import RobotModel

__all__ = [
    "GraphException",
    "TwinPair",
    "AlignmentResult",
    "PairReport",
    "TWIN_PAIRS",
    "TWIN_CELLS",
    "N_TWIN_CELLS",
    "N_TWIN_PAIRS",
    "SHADOW_JOINT_ORDINAL",
    "CLEAN_SHADOW_HALF_RANGE",
    "FGW_ASSERT_TOL",
    "twin_descriptors",
    "assert_structurally_aligned",
    "within_cluster_threshold",
    "fgw_matrix",
    "report",
    "main",
]

# 9.3.4 step (3): the FGW value on a pair whose feature tensors are bit-identical
# is zero up to the solver's own arithmetic, not up to a modelling tolerance.
FGW_ASSERT_TOL = 1e-12

# 9.2.4 puts the shadow joint in link4, co-located with joint4, and its example
# is written for that joint. Zero based here, so 3 is the fourth arm joint. The
# same ordinal on every family, so that the D4 cell means the same thing across
# the six -- which joint it is physically differs, and the resulting end-effector
# dose is `defect.observable.D4hi.*` in the lock file, per family.
SHADOW_JOINT_ORDINAL = 3

# 9.3.3 / 14.5 E-3: the clean twin carries the same shadow joint pinned shut.
CLEAN_SHADOW_HALF_RANGE = 1e-5

# 9.2.4's soft equality constraint. `solimp[2]` is the low-impedance dead-zone
# width and equals the half range, so it is filled per level rather than fixed.
_SHADOW_SOLREF = "0.005 1"
_SHADOW_SOLIMP = "0.2 0.95 {width:.17g} 0.9 6"


class GraphException(str, Enum):
    """9.3.6 field `graph_exception`."""

    NONE = "none"
    SHADOW_JOINT = "shadow_joint"


@dataclass(frozen=True)
class TwinPair:
    """One dose level of one cell: a family, a core defect, and a level index.

    Not one of the 36. The 36 of 9.3.3 and 11.3 are the (family, defect) cells,
    which are the statistical unit that n = 36 counts; each of them carries the
    five-level ladder, so there are 180 twin pairs to align and only 36 rows in
    any table whose n is 36. `TWIN_CELLS` holds the former, `TWIN_PAIRS` the
    latter, and mixing them inflates the sample size fivefold.
    """

    family: str
    defect_code: str
    level: int

    def __post_init__(self) -> None:
        family(self.family).require_in_36_pairs()
        if self.defect_code not in CORE_DEFECTS:
            raise ValueError(
                f"{self.defect_code} is not one of the six core defects {CORE_DEFECTS}; "
                "only those form the 36 pairs (9.3.3)"
            )
        defect(self.defect_code).level(self.level)

    @property
    def spec(self) -> DefectSpec:
        return defect(self.defect_code)

    @property
    def clean(self) -> EmbodimentId:
        return EmbodimentId(self.family)

    @property
    def defective(self) -> EmbodimentId:
        return EmbodimentId(self.family, self.defect_code, self.level)

    @property
    def graph_exception(self) -> GraphException:
        return (
            GraphException.SHADOW_JOINT
            if self.spec.changes_kinematic_graph
            else GraphException.NONE
        )

    @property
    def is_relaxed(self) -> bool:
        """True when 9.3.4's relaxed assertion applies instead of the exact one."""
        return self.graph_exception is GraphException.SHADOW_JOINT

    def __str__(self) -> str:
        return f"{self.family}|{self.defect_code}|L{self.level}"


TWIN_PAIRS: tuple[TwinPair, ...] = tuple(
    TwinPair(fam, code, lv.index)
    for fam in MAIN_FAMILIES
    for code in CORE_DEFECTS
    for lv in defect(code).levels
)

# The 36 cells of 11.3, each of which carries the five-level ladder above.
TWIN_CELLS: tuple[tuple[str, str], ...] = tuple(
    (fam, code) for fam in MAIN_FAMILIES for code in CORE_DEFECTS
)

N_TWIN_CELLS = len(TWIN_CELLS)
N_TWIN_PAIRS = len(TWIN_PAIRS)

if N_TWIN_CELLS != 36:
    raise AssertionError(
        f"9.3.3 fixes the cell count at 36 = 6 families x 6 core defects, got "
        f"{N_TWIN_CELLS}. n = 36 in 11.3 counts these, so a mismatch here silently "
        "changes every MDE in the paper"
    )


# ---------------------------------------------------------------------------
# The shadow joint (9.2.4)
# ---------------------------------------------------------------------------


def _shadow_joint_name(driven: str) -> str:
    return f"{driven}_lash"


def _apply_shadow_joint(
    robot: RobotModel, model: mujoco.MjModel, driven: str, half_range: float
) -> str:
    """Insert 9.2.4's backlash joint next to `driven`, in the same body.

    `pos` and `axis` are written from the compiled model with full float64
    precision rather than copied as attribute text: a family whose MJCF leaves
    them to a `<default>` class would otherwise get a shadow joint that is not
    quite co-located, and the whole relaxed assertion is that the two sides
    agree bit for bit on anchors and axes.

    The equality constraint is 9.2.4's, not a limit: the documented reason is
    that it is always active, so a reader of the compensation-field ground truth
    can query the engagement state every frame. `solimp[2]` is the dead-zone
    width and is the half range, so it moves with the level.
    """
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, driven)
    if joint_id < 0:
        raise KeyError(f"no joint named {driven!r} to attach a shadow joint to")
    if float(model.dof_armature[int(model.jnt_dofadr[joint_id])]) <= 0.0:
        raise ValueError(
            f"{driven}: two co-located joints with no added inertia make the "
            "joint-space inertia matrix singular (9.2.4 trap 3), and the driven "
            "joint's armature is zero. Fix the family MJCF before injecting D4hi"
        )

    target = None
    parents = {child: parent for parent in robot.root.iter() for child in parent}
    for element in robot.root.iter("joint"):
        if element.get("name") == driven:
            target = element
            break
    if target is None:
        raise KeyError(f"joint {driven!r} is in the compiled model but not in the XML tree")

    body = parents[target]
    pos = " ".join(f"{v:.17g}" for v in model.jnt_pos[joint_id])
    axis = " ".join(f"{v:.17g}" for v in model.jnt_axis[joint_id])
    name = _shadow_joint_name(driven)
    lash = ET.Element(
        "joint",
        {
            "name": name,
            "type": target.get("type", "hinge"),
            "pos": pos,
            "axis": axis,
            "limited": "true",
            "range": f"{-half_range:.17g} {half_range:.17g}",
            # 9.2.4's `backlash` default class, written out rather than declared
            # as a class so that the shadow joint carries no attribute that a
            # family-level default could reach.
            "damping": "0",
            "armature": "0",
            "frictionloss": "0",
        },
    )
    body.insert(list(body).index(target) + 1, lash)

    equality = robot.create_default_element("equality")
    ET.SubElement(
        equality,
        "joint",
        {
            "joint1": name,
            "solref": _SHADOW_SOLREF,
            "solimp": _SHADOW_SOLIMP.format(width=half_range),
        },
    )
    return name


def _shadow_descriptor(
    family_name: str, *, half_range: float, label: str
) -> StructuralDescriptor:
    fam = family(family_name)
    robot = fam.robot_model()
    base_model = robot.get_model(mode="mujoco")
    joints = list(robot.joints)
    if not 0 <= SHADOW_JOINT_ORDINAL < len(joints):
        raise ValueError(
            f"{family_name} has {len(joints)} arm joints, so 9.2.4's joint4 "
            f"(ordinal {SHADOW_JOINT_ORDINAL}) does not exist"
        )
    driven = joints[SHADOW_JOINT_ORDINAL]
    name = _apply_shadow_joint(robot, base_model, driven, half_range)
    joints.insert(SHADOW_JOINT_ORDINAL + 1, name)
    model = robot.get_model(mode="mujoco")
    return descriptor_from_model(
        model,
        name=label,
        joint_names=tuple(joints),
        base_body=robot.root_body,
        eef_bodies=tuple(robot.eef_name[arm] for arm in robot.arms),
        mjcf_sha256_hex=mjcf_sha256(robot.get_xml()),
    )


@cache
def _cached_shadow_descriptor(
    family_name: str, half_range: float, label: str
) -> StructuralDescriptor:
    return _shadow_descriptor(family_name, half_range=half_range, label=label)


def shadow_half_range(pair: TwinPair) -> float:
    """b/2 for the pair's level, in radians (9.2.4: `range="-b/2 b/2"`)."""
    params = pair.spec.level(pair.level).params
    if "backlash" not in params:
        raise ValueError(f"{pair.defect_code} level {pair.level} carries no `backlash`")
    return 0.5 * float(params["backlash"])


# ---------------------------------------------------------------------------
# The two sides of a pair
# ---------------------------------------------------------------------------


@cache
def _gain_descriptor(
    family_name: str, defect_code: str, level: int, clean: bool
) -> StructuralDescriptor:
    """A descriptor of the family with D8's gain written into the compiled model.

    The parameter comes from `Delta.joint_values`, the same arithmetic
    `inject._apply_model_level` runs, and the clean side is `Delta.zero()` of the
    same level rather than a separate branch (CONTRACT 3.3). The write is
    repeated here instead of being reached for through `inject` because a
    `DefectWrapper` needs a whole robosuite environment -- a task, an arena, a
    gripper -- and a descriptor is a property of the arm alone.
    """
    fam = family(family_name)
    robot = fam.robot_model()
    model = robot.get_model(mode="mujoco")
    joints = tuple(robot.joints)

    delta = Delta.from_level(defect_code, level, joint_scale=(1.0,) * len(joints))
    if clean:
        delta = delta.zero()
    gains = delta.joint_values("gain")

    for name, g in zip(joints, gains, strict=True):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = actuator_for_joint(model, joint_id)
        if actuator_id is None:
            raise ValueError(
                f"{family_name}: arm joint {name!r} has no actuator, so D8's gain "
                "has nowhere to land. 9.2.7 scales a transmission that exists"
            )
        model.actuator_gainprm[actuator_id, 0] = float(g)

    side = "clean@g=1" if clean else f"{defect_code}/L{level}"
    return descriptor_from_model(
        model,
        name=f"{family_name}/{side}",
        joint_names=joints,
        base_body=robot.root_body,
        eef_bodies=tuple(robot.eef_name[arm] for arm in robot.arms),
        mjcf_sha256_hex=mjcf_sha256(robot.get_xml()),
    )


def twin_descriptors(pair: TwinPair) -> tuple[StructuralDescriptor, StructuralDescriptor]:
    """The clean and defective descriptors of a pair, on one code path.

    CONTRACT 3.3: the clean twin is the injection path evaluated at delta = 0,
    never a separate branch. That is what this does at the model level -- both
    sides go through the same builder, and the only thing that differs is the
    parameter. For the four defects that write no `mjModel` field the parameter
    is absent on both sides and the two descriptors come out of the same
    compilation; for D4hi it is the shadow joint's half range, 1e-5 on the clean
    side and b/2 on the defective one; for D8 it is the transmission gain, 1 on
    the clean side and g on the defective one.

    D8 is the case worth stating. It does write the compiled model -- 9.2.7 puts
    it in `actuator_gainprm` -- so the two sides are genuinely two compilations
    and `GraphNode.gain` differs between them. What makes it a core defect is
    that no node feature in either set of 9.3.4 reads that field: both take the
    torque limit from `actuator_ctrlrange`, which D8 leaves alone. Building both
    sides here is what turns that from an argument into a measurement, and it is
    why the D8 rows of the 9.3.4 table are not vacuous the way the `mjData` rows
    are.
    """
    if pair.is_relaxed:
        return (
            _cached_shadow_descriptor(
                pair.family, CLEAN_SHADOW_HALF_RANGE, f"{pair.family}/clean+lash"
            ),
            _cached_shadow_descriptor(
                pair.family,
                shadow_half_range(pair),
                f"{pair.family}/{pair.defect_code}/L{pair.level}",
            ),
        )

    if WritableField.ACTUATOR_GAINPRM in pair.spec.primary_fields:
        return (
            _gain_descriptor(pair.family, pair.defect_code, pair.level, True),
            _gain_descriptor(pair.family, pair.defect_code, pair.level, False),
        )

    clean = descriptor(pair.family)
    # Deliberately the same object: the defect writes nothing the compiled model
    # can see, and building a second descriptor from a second compilation would
    # test the compiler's determinism rather than the invariance claim.
    return clean, clean


# ---------------------------------------------------------------------------
# The assertion of 9.3.4
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlignmentResult:
    """What the 9.3.4 script produces for one (pair, feature set)."""

    pair: TwinPair
    featset: FeatureSet
    relaxed: bool
    edges_equal: bool
    node_types_equal: bool
    features_equal: bool
    n_feature_mismatches: int
    mismatching_nodes: tuple[str, ...]
    fgw: float
    alpha: float

    def to_json(self) -> dict[str, Any]:
        return {
            "pair": str(self.pair),
            "featset": self.featset.value,
            "relaxed": self.relaxed,
            "edges_equal": self.edges_equal,
            "node_types_equal": self.node_types_equal,
            "node_feature_equal": self.features_equal,
            "n_feature_mismatches": self.n_feature_mismatches,
            "mismatching_nodes": list(self.mismatching_nodes),
            f"fgw_{self.featset.value}": self.fgw,
            "alpha": self.alpha,
        }


def assert_structurally_aligned(
    pair: TwinPair,
    *,
    tol: float = FGW_ASSERT_TOL,
    lock: LockFile | None = None,
) -> dict[FeatureSet, AlignmentResult]:
    """9.3.4, both feature sets, on one twin pair.

    Steps, in the order the proposal writes them:

    1. the edge sets agree as ordered name pairs, and so do the node types;
    2. the node feature tensors are equal *element by element* -- `==`, not
       `allclose`. The twin's XML is the same XML, so any non-zero difference
       means the injector wrote somewhere it should not have, and a tolerance
       would let exactly that class of bug through;
    3. the FGW value is below `tol`.

    D4hi takes the relaxed form of 9.3.4: step (1) allows the two sides to share
    one extra joint node whose anchor and axis agree bit for bit, step (2) is
    allowed to fail on that node alone, and step (3) is replaced by the
    comparison against `within_cluster_threshold`, which the caller makes -- the
    number is returned rather than swallowed, because 9.3.4 requires it to be
    reported.
    """
    lock = lock if lock is not None else LockFile()
    clean, defective = twin_descriptors(pair)
    out: dict[FeatureSet, AlignmentResult] = {}

    for featset in FeatureSet:
        fields = feature_fields(featset, lock)
        if clean.sorted_edges() != defective.sorted_edges():
            raise AssertionError(
                f"{pair} [{featset.value}]: edge sets differ.\n"
                f"  clean    : {clean.sorted_edges()}\n"
                f"  defective: {defective.sorted_edges()}"
            )
        if clean.node_types != defective.node_types:
            raise AssertionError(f"{pair} [{featset.value}]: node type sequences differ")

        xc = node_features(clean, fields)
        xd = node_features(defective, fields)
        if xc.shape != xd.shape:
            raise AssertionError(
                f"{pair} [{featset.value}]: feature tensors have shapes "
                f"{xc.shape} and {xd.shape}"
            )
        equal = bool(np.array_equal(xc, xd))
        rows = np.flatnonzero(np.any(xc != xd, axis=1))
        mismatching = tuple(clean.nodes[int(i)].name for i in rows)

        if not equal and not pair.is_relaxed:
            raise AssertionError(
                f"{pair} [{featset.value}]: node features differ on {list(mismatching)}. "
                "The two sides compile the same XML, so this is the injector writing "
                "to a field it does not own -- 9.3.4 asserts `==` for exactly this "
                "reason and a tolerance would hide it"
            )
        if not equal:
            _check_relaxed_mismatch(pair, featset, clean, defective, rows)

        ga = build_morphology_graph(clean, featset, lock=lock)
        gb = build_morphology_graph(defective, featset, lock=lock)
        alpha = alpha_for(featset, lock)
        value = fgw_distance(ga, gb, alpha=alpha)

        if not pair.is_relaxed and value >= tol:
            raise AssertionError(
                f"{pair} [{featset.value}]: FGW is {value:.6e}, not below {tol:.1e}, "
                "although the feature tensors are equal element by element. The "
                "structure term or the solver has changed"
            )

        out[featset] = AlignmentResult(
            pair=pair,
            featset=featset,
            relaxed=pair.is_relaxed,
            edges_equal=True,
            node_types_equal=True,
            features_equal=equal,
            n_feature_mismatches=int(len(rows)),
            mismatching_nodes=mismatching,
            fgw=value,
            alpha=alpha,
        )
    return out


def _check_relaxed_mismatch(
    pair: TwinPair,
    featset: FeatureSet,
    clean: StructuralDescriptor,
    defective: StructuralDescriptor,
    rows: np.ndarray,
) -> None:
    """The relaxed assertion's own conditions (9.3.4, 9.3.3).

    Exactly one node may differ, it has to be the shadow joint, and its anchor
    and axis have to be identical on both sides -- the difference is allowed to
    be the free-play range and nothing else.
    """
    if len(rows) != 1:
        raise AssertionError(
            f"{pair} [{featset.value}]: the relaxed assertion allows one differing "
            f"node, found {len(rows)}: "
            f"{[clean.nodes[int(i)].name for i in rows]}"
        )
    i = int(rows[0])
    node_c, node_d = clean.nodes[i], defective.nodes[i]
    expected = _shadow_joint_name(clean.joint_names[SHADOW_JOINT_ORDINAL])
    if node_c.name != expected:
        raise AssertionError(
            f"{pair} [{featset.value}]: the differing node is {node_c.name!r}, not the "
            f"shadow joint {expected!r}"
        )
    if node_c.anchor != node_d.anchor or node_c.axis != node_d.axis:
        raise AssertionError(
            f"{pair} [{featset.value}]: the shadow joint's anchor or axis differs "
            "between the two sides; 9.3.4 requires them bit for bit identical"
        )
    changed = {
        name
        for name, (attr_c, attr_d) in {
            "jnt_range": (node_c.jnt_range, node_d.jnt_range),
            "armature": (node_c.armature, node_d.armature),
            "damping": (node_c.damping, node_d.damping),
            "frictionloss": (node_c.frictionloss, node_d.frictionloss),
            "torque_limit": (node_c.torque_limit, node_d.torque_limit),
            "body_mass": (node_c.body_mass, node_d.body_mass),
            "body_inertia": (node_c.body_inertia, node_d.body_inertia),
        }.items()
        if attr_c != attr_d
    }
    if changed != {"jnt_range"}:
        raise AssertionError(
            f"{pair} [{featset.value}]: the shadow joint differs in {sorted(changed)}; "
            "9.2.4 changes its range and nothing else"
        )


# ---------------------------------------------------------------------------
# The within-cluster threshold and the summary matrix
# ---------------------------------------------------------------------------


@cache
def _family_distances(featset: FeatureSet) -> tuple[tuple[str, str, float], ...]:
    lock = LockFile()
    alpha = alpha_for(featset, lock)
    graphs = {f: build_morphology_graph(descriptor(f), featset, lock=lock) for f in MAIN_FAMILIES}
    return tuple(
        (a, b, fgw_distance(graphs[a], graphs[b], alpha=alpha))
        for a, b in combinations(MAIN_FAMILIES, 2)
    )


def within_cluster_threshold(featset: FeatureSet) -> tuple[float, tuple[str, str], float]:
    """9.3.4's relaxed bound: one tenth of the closest real morphology pair.

    "Real morphology pair" is a pair of distinct clean families, which is the
    population the baseline's hierarchical clustering actually sees. The six main
    families are used rather than all ten, because the out-of-scope three never
    enter a cluster and including them could only lower the bar.

    Returns (threshold, closest pair, that pair's distance).
    """
    pairs = _family_distances(featset)
    a, b, d = min(pairs, key=lambda t: t[2])
    return 0.1 * d, (a, b), d


@dataclass(frozen=True)
class PairReport:
    """One cell of the FGW matrix that 11.2 asks `registry/twins/` to hold."""

    family: str
    defect_code: str
    level: int
    graph_exception: GraphException
    node_feature_equal: bool
    fgw_ours: float
    fgw_baseline: float

    def to_json(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "defect_code": self.defect_code,
            "level": self.level,
            "graph_exception": self.graph_exception.value,
            "node_feature_equal": self.node_feature_equal,
            "fgw_ours": self.fgw_ours,
            "fgw_baseline": self.fgw_baseline,
        }


def fgw_matrix(*, lock: LockFile | None = None) -> tuple[PairReport, ...]:
    """Run the assertion over every pair and every level, and collect the numbers."""
    lock = lock if lock is not None else LockFile()
    out = []
    for pair in TWIN_PAIRS:
        res = assert_structurally_aligned(pair, lock=lock)
        out.append(
            PairReport(
                family=pair.family,
                defect_code=pair.defect_code,
                level=pair.level,
                graph_exception=pair.graph_exception,
                node_feature_equal=res[FeatureSet.OURS].features_equal
                and res[FeatureSet.BASELINE].features_equal,
                fgw_ours=res[FeatureSet.OURS].fgw,
                fgw_baseline=res[FeatureSet.BASELINE].fgw,
            )
        )
    return tuple(out)


def shadow_versus_bare(family_name: str, level: int, featset: FeatureSet) -> float:
    """The unrelaxed D4hi number: shadow-jointed defect against a bare clean family.

    This is what `graph_exception: shadow_joint` costs if the clean twin does
    *not* get the same shadow joint. 9.3.3 requires it to, and this number is
    the reason: the node counts differ by one and the distance stops being a
    feature difference on a single node.
    """
    lock = LockFile()
    pair = TwinPair(family_name, "D4hi", level)
    _, defective = twin_descriptors(pair)
    bare = build_morphology_graph(descriptor(family_name), featset, lock=lock)
    with_lash = build_morphology_graph(defective, featset, lock=lock)
    return fgw_distance(bare, with_lash, alpha=alpha_for(featset, lock))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report(*, lock: LockFile | None = None) -> tuple[str, dict[str, Any]]:
    """The text 9.3.4 asks to be printed, and the json 11.2 asks to be stored."""
    lock = lock if lock is not None else LockFile()
    rows = fgw_matrix(lock=lock)
    lines = [
        f"{len(TWIN_CELLS)} twin cells x levels = {len(rows)} (family, defect, level) pairs",
        "",
        f"{'family':10s} {'defect':7s} {'lvl':>3s} {'exception':13s} "
        f"{'feat_eq':>7s} {'fgw_ours':>14s} {'fgw_baseline':>14s}",
    ]
    for r in rows:
        lines.append(
            f"{r.family:10s} {r.defect_code:7s} {r.level:3d} {r.graph_exception.value:13s} "
            f"{str(r.node_feature_equal):>7s} {r.fgw_ours:14.6e} {r.fgw_baseline:14.6e}"
        )

    thresholds: dict[str, Any] = {}
    lines.append("")
    for featset in FeatureSet:
        thr, (a, b), closest = within_cluster_threshold(featset)
        worst = max(r.fgw_ours if featset is FeatureSet.OURS else r.fgw_baseline for r in rows)
        thresholds[featset.value] = {
            "closest_real_pair": [a, b],
            "closest_real_distance": closest,
            "within_cluster_threshold": thr,
            "worst_twin_distance": worst,
            "passes": worst < thr,
        }
        lines.append(
            f"[{featset.value:8s}] closest real morphology pair {a}-{b} = {closest:.6f}; "
            f"threshold = {thr:.6f}; worst twin = {worst:.6e}; "
            f"{'PASS' if worst < thr else 'FAIL'}"
        )

    lines.append("")
    lines.append("D4hi, relaxed assertion (both sides carry the shadow joint):")
    d4hi: dict[str, Any] = {}
    for level in range(len(defect("D4hi").levels)):
        entry: dict[str, Any] = {}
        for featset in FeatureSet:
            key = f"fgw_{featset.value}"
            vals = [
                getattr(r, key)
                for r in rows
                if r.defect_code == "D4hi" and r.level == level
            ]
            entry[key] = {"min": min(vals), "max": max(vals)}
        d4hi[f"L{level}"] = entry
        half = shadow_half_range(TwinPair(MAIN_FAMILIES[0], "D4hi", level))
        lines.append(
            f"  L{level}  b/2 = {half:.6e} rad"
            f"   ours [{entry['fgw_ours']['min']:.6e}, {entry['fgw_ours']['max']:.6e}]"
            f"   baseline [{entry['fgw_baseline']['min']:.6e}, {entry['fgw_baseline']['max']:.6e}]"
        )

    lines.append("")
    lines.append(
        "D4hi, shadow-joint variant reported separately (defective side with the "
        "shadow joint against a clean family without one; 9.3.3 forbids this "
        "configuration, the number says what it would cost):"
    )
    bare: dict[str, Any] = {}
    top_level = len(defect("D4hi").levels) - 1
    for fam in MAIN_FAMILIES:
        bare[fam] = {
            fs.value: shadow_versus_bare(fam, top_level, fs) for fs in FeatureSet
        }
        lines.append(
            f"  {fam:10s} L{top_level}  ours {bare[fam]['ours']:.6f}   "
            f"baseline {bare[fam]['baseline']:.6f}"
        )

    doc = {
        "n_cells": len(TWIN_CELLS),
        "n_pairs": len(rows),
        "shadow_joint_ordinal": SHADOW_JOINT_ORDINAL,
        "clean_shadow_half_range_rad": CLEAN_SHADOW_HALF_RANGE,
        "assert_tol": FGW_ASSERT_TOL,
        "pairs": [r.to_json() for r in rows],
        "thresholds": thresholds,
        "d4hi_relaxed": d4hi,
        "d4hi_shadow_vs_bare": bare,
        "real_morphology_pairs": {
            fs.value: [
                {"a": a, "b": b, "fgw": d} for a, b, d in _family_distances(fs)
            ]
            for fs in FeatureSet
        },
    }
    return "\n".join(lines), doc


def default_output_path() -> Path:
    return Path(__file__).resolve().parent / "twins" / "fgw_matrix.json"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=None, help="write the json report here")
    args = p.parse_args(argv)

    text, doc = report()
    print(text)
    out = args.out if args.out is not None else default_output_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n[write] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
