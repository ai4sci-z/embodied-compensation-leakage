"""Fused Gromov-Wasserstein distance between morphology graphs (9.3.4).

X1 rests on the claim that the published embodiment-grouping baseline
(arXiv:2602.18025) cannot separate a structurally aligned twin pair. 9.3.4 turns
that claim into an assertion that runs in CI, and requires it under *two* node
feature sets, both reported:

`FEATSET_OURS`
    joint axis, joint position, limits, torque limit, inertia including
    armature. Our own caliber. It contains no friction and no damping term, so
    the invariance argument under it is ours to make and proves nothing about
    anybody else's method.

`FEATSET_BASELINE`
    transcribed field by field from the baseline. Its appendix E takes URMA's
    local joint descriptor verbatim (arXiv:2409.06366 appendix A), and URMA's
    d_j lists joint damping and a friction coefficient among its entries. That
    single fact is why D6 ships on the `qfrc_applied` caliber (CONTRACT 3.5): a
    D6 that wrote `dof_frictionloss` would move a baseline node feature and take
    `fgw_baseline = 0` -- hence X1 -- with it. It is also, at one remove, why D6
    is no longer one of the core six: `qfrc_applied` was the only caliber X1
    allowed, and the stiction it gives up exceeded 9.2.6's gate by two orders of
    magnitude. What this list does *not* contain is the actuator gain, which is
    what lets D8 take the vacated slot while still writing `mjModel`.

The baseline field list is a `LiteratureFact`, not a literal. It is read through
`LockFile.fact(LITERATURE_FACTS["fgw.featset_baseline"])` and a missing entry
raises with the citation to transcribe from; `python -m portal_trace.registry.fgw
--record-facts` is the transcription step. Same for the solver settings
(`fgw.hyperparameters_baseline`), which carry the baseline's alpha -- the one
number that decides how structure and features trade off, and therefore the one
number a later implementation swap could change silently.

Every node feature is read out of the compiled model by
`registry.families.descriptor_from_model`. Nothing is defaulted: Panda's
`dof_frictionloss` is 0.1 N m on every arm joint because robosuite's
`RobotModel.__init__` puts it there, not 0 as the MJCF would suggest, and a
feature extractor that assumed the MJCF value would report `fgw_baseline = 0`
for the wrong reason.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cache
from typing import Any

import networkx as nx
import numpy as np
import ot

from portal_trace.lockfile import LITERATURE_FACTS, LiteratureFact, LockFile
from portal_trace.registry.families import (
    MAIN_FAMILIES,
    GraphNode,
    NodeType,
    StructuralDescriptor,
    descriptor,
)

__all__ = [
    "FeatureSet",
    "MorphologyGraph",
    "ALPHA_OURS",
    "FEATSET_OURS_FIELDS",
    "FEATURE_FIELDS",
    "feature_fields",
    "build_morphology_graph",
    "fgw_distance",
    "node_features",
    "column_scales",
    "baseline_hyperparameters",
    "record_facts",
    "main",
]

# 9.3.4 calls this ALPHA_LOCKED and does not give it a value: the FGW number on a
# twin pair is zero for every alpha, and reporting it exists only to pin the
# structure/feature trade-off to an implementation. Equal weight is the neutral
# choice, and it is a constant here rather than a config key because a caliber
# that can be set per run is not locked. The baseline's own alpha is a different
# quantity and comes from the lock file.
ALPHA_OURS = 0.5

# Conditional-gradient settings for `ot.gromov.fused_gromov_wasserstein2`. Tight
# enough that a genuine zero comes back as an exact zero rather than as solver
# noise, which matters because 9.3.4 asserts against 1e-12.
_MAX_ITER = 10_000
_TOL = 1e-12


class FeatureSet(str, Enum):
    OURS = "ours"
    BASELINE = "baseline"


# ---------------------------------------------------------------------------
# Node feature fields
# ---------------------------------------------------------------------------

# Each field maps a node to a fixed-width block of the feature row. Non-joint
# nodes (the base, the end effector) have no joint attributes, and their blocks
# are zero; the node-type one-hot in front of every row is what keeps a zeroed
# joint block from being confused with a real joint whose attributes happen to
# vanish.
FEATURE_FIELDS: dict[str, tuple[int, Callable[[GraphNode], Sequence[float]]]] = {
    "jnt_pos": (3, lambda n: n.anchor),
    "jnt_axis": (3, lambda n: n.axis),
    "jnt_range": (2, lambda n: n.jnt_range),
    "torque_limit": (2, lambda n: n.torque_limit),
    "dof_armature": (1, lambda n: (n.armature,)),
    "dof_damping": (1, lambda n: (n.damping,)),
    "dof_frictionloss": (1, lambda n: (n.frictionloss,)),
    "body_mass": (1, lambda n: (n.body_mass,)),
    "body_inertia": (3, lambda n: n.body_inertia),
    "qpos0": (1, lambda n: (n.nominal_qpos,)),
}

# 9.3.4 table, row `FEATSET_OURS`. The velocity limit that row also lists has no
# field in MuJoCo 3.3.2 -- MJCF has `range` on a joint and nothing for velocity,
# and robosuite 1.5.2 keeps its velocity limits inside controller configs rather
# than in the model -- so it is absent here and reported as a write-back rather
# than filled with a controller-side number that is not a property of the
# morphology.
FEATSET_OURS_FIELDS: tuple[str, ...] = (
    "jnt_pos",
    "jnt_axis",
    "jnt_range",
    "torque_limit",
    "body_mass",
    "body_inertia",
    "dof_armature",
)

# The transcription of arXiv:2602.18025 appendix E, read against both sources.
# Appendix E.1: "Each node carries a feature vector constructed from the local
# descriptors that URMA uses as encoder inputs [...] only the local descriptors
# {d_j} and {d_f} are used as node attributes". So the baseline's node features
# are URMA's d_j, and URMA (arXiv:2409.06366) lists d_j as: relative 3D position
# in the nominal configuration, rotation axis, number of direct child joints,
# joint nominal position, torque limit, velocity limit, damping, rotor inertia,
# stiffness, friction coefficient, min and max control range; plus a general
# robot part kp, kd, sigma_a, m, L, W, H.
#
# Damping and the friction coefficient are in that list, which is what forces
# D6 onto the qfrc_applied caliber (9.2.6). The baseline's own text never says
# "friction" or "damping" -- it delegates the field list to URMA by reference --
# so the transcription has to go through both papers, and reading only the
# baseline would conclude the opposite.
#
# Four of URMA's entries have no column here, none of them load-bearing:
#   velocity limit   MuJoCo 3.3.2 has no such field. MJCF gives a joint `range`
#                    and nothing for velocity, and robosuite 1.5.2 keeps its
#                    velocity limits in controller configs, which are not a
#                    property of the morphology.
#   stiffness        `jnt_stiffness` is identically zero on every robosuite arm
#                    joint, so the column is constant and changes no distance.
#                    The one defect that would move it, D7, already declares
#                    `changes_kinematic_graph` and is outside the invariance
#                    claim.
#   child count      structural, and the FGW structure term already carries it
#                    as the shortest-path matrix over the same edges.
#   kp, kd, sigma_a, m, L, W, H
#                    constant within a robot, so they cancel exactly on a twin
#                    pair, which is the only comparison 9.3.4 asserts on.
_BASELINE_FIELDS_TRANSCRIPTION: tuple[str, ...] = (
    "jnt_pos",
    "jnt_axis",
    "jnt_range",
    "torque_limit",
    "dof_damping",
    "dof_frictionloss",
    "dof_armature",
    "qpos0",
)

# Omitted entries of URMA's d_j, recorded alongside the transcription so that
# the field list is auditable as a *reading* and not just as a list.
_BASELINE_FIELDS_OMITTED: dict[str, str] = {
    "velocity_limit": "no MuJoCo 3.3.2 field; robosuite keeps it in controller configs",
    "stiffness": "jnt_stiffness is identically zero on every robosuite arm joint",
    "n_child_joints": "structural; carried by the FGW shortest-path term over the same edges",
    "general_robot_part": "kp, kd, sigma_a, m, L, W, H are constant within a robot",
}

_NODE_TYPE_ORDER: tuple[NodeType, ...] = (NodeType.BASE, NodeType.JOINT, NodeType.EEF)


def feature_fields(featset: FeatureSet, lock: LockFile | None = None) -> tuple[str, ...]:
    """The field list of a feature set.

    `FeatureSet.OURS` is ours to choose, so it is a constant. `FeatureSet.BASELINE`
    is somebody else's published choice, so it is a lock-file fact and raises
    when it has not been transcribed.
    """
    if featset is FeatureSet.OURS:
        return FEATSET_OURS_FIELDS
    fact: LiteratureFact = LITERATURE_FACTS["fgw.featset_baseline"]
    value = (lock if lock is not None else LockFile()).fact(fact)
    fields = tuple(value)
    unknown = [f for f in fields if f not in FEATURE_FIELDS]
    if unknown:
        raise ValueError(
            f"{fact.key}: transcribed fields {unknown} have no extractor in "
            f"FEATURE_FIELDS ({sorted(FEATURE_FIELDS)}). Add an exact extractor "
            "before using the baseline feature set; a field that is read as zero "
            "would make the baseline look blinder than it is"
        )
    return fields


def baseline_hyperparameters(lock: LockFile | None = None) -> dict[str, Any]:
    """POT settings the baseline uses, from `fgw.hyperparameters_baseline`."""
    fact = LITERATURE_FACTS["fgw.hyperparameters_baseline"]
    return dict((lock if lock is not None else LockFile()).fact(fact))


def alpha_for(featset: FeatureSet, lock: LockFile | None = None) -> float:
    if featset is FeatureSet.OURS:
        return ALPHA_OURS
    params = baseline_hyperparameters(lock)
    if "alpha" not in params:
        raise ValueError(
            "fgw.hyperparameters_baseline carries no 'alpha'; the baseline's "
            "structure/feature weight is what the FGW number is reported to lock"
        )
    return float(params["alpha"])


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------


def node_features(desc: StructuralDescriptor, fields: Sequence[str]) -> np.ndarray:
    """The (n_nodes, d) raw node feature tensor, in SI units.

    Columns are the node-type one-hot followed by the requested fields in the
    given order. No standardisation happens here; `column_scales` applies it,
    and it is a diagonal linear map, so an exactly equal pair of tensors stays
    exactly equal.
    """
    rows = []
    for node in desc.nodes:
        row: list[float] = [1.0 if node.node_type is t else 0.0 for t in _NODE_TYPE_ORDER]
        for name in fields:
            width, extract = FEATURE_FIELDS[name]
            block = tuple(float(v) for v in extract(node))
            if len(block) != width:
                raise ValueError(f"field {name!r} produced {len(block)} values, expected {width}")
            row.extend(block)
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


@cache
def column_scales(fields: tuple[str, ...]) -> tuple[float, ...]:
    """Per-column scale, frozen from the six main families.

    Raw SI columns are not comparable: a torque limit is order 10^2 N m and a
    link inertia order 10^-2 kg m^2, so an unscaled Euclidean feature cost is a
    torque comparison with a rounding error attached. Dividing each column by
    its standard deviation across the six clean main descriptors puts them on
    one footing. Two properties make this safe for 9.3.4: the map is diagonal
    and data independent once the six clean families are fixed, so a bit-exact
    equality is preserved bit for bit, and it is the same scale on both sides of
    every pair, so the relaxed D4hi ratio is a ratio of like quantities.

    Zero-variance columns keep scale 1: a column that is constant across the
    registry contributes nothing to any distance, and dividing it by zero would
    turn that into a NaN.
    """
    stacked = np.concatenate([node_features(descriptor(f), fields) for f in MAIN_FAMILIES], axis=0)
    scales = stacked.std(axis=0)
    scales[scales <= 0.0] = 1.0
    return tuple(float(s) for s in scales)


@dataclass(frozen=True)
class MorphologyGraph:
    """A morphology graph ready for POT: structure, features, node masses."""

    name: str
    structure: np.ndarray       # (n, n) shortest-path distances
    features: np.ndarray        # (n, d) scaled node features
    weights: np.ndarray         # (n,) node masses, uniform
    node_types: tuple[NodeType, ...]
    node_names: tuple[str, ...]
    fields: tuple[str, ...]

    @property
    def n_nodes(self) -> int:
        return len(self.node_names)


def build_morphology_graph(
    desc: StructuralDescriptor, featset: FeatureSet, *, lock: LockFile | None = None
) -> MorphologyGraph:
    """The (graph, node feature tensor) pair of 9.3.4 for one model.

    The structure matrix is the shortest-path (hop) distance on the kinematic
    tree, which is what a graph-only method has to use and what the baseline
    uses. Node masses are uniform: the baseline has no per-node weighting, and
    inventing one here would change which coupling POT finds.
    """
    fields = feature_fields(featset, lock)
    raw = node_features(desc, fields)
    scaled = raw / np.asarray(column_scales(fields), dtype=np.float64)

    graph = nx.Graph()
    graph.add_nodes_from(range(desc.n_nodes))
    graph.add_edges_from(desc.edges)
    n = desc.n_nodes
    structure = np.zeros((n, n), dtype=np.float64)
    for i, lengths in nx.all_pairs_shortest_path_length(graph):
        for j, d in lengths.items():
            structure[i, j] = float(d)
    if not np.isfinite(structure).all():
        raise ValueError(f"{desc.name}: the morphology graph is disconnected")

    return MorphologyGraph(
        name=desc.name,
        structure=structure,
        features=scaled,
        weights=np.full(n, 1.0 / n, dtype=np.float64),
        node_types=desc.node_types,
        node_names=tuple(node.name for node in desc.nodes),
        fields=fields,
    )


def _feature_cost(a: MorphologyGraph, b: MorphologyGraph) -> np.ndarray:
    if a.features.shape[1] != b.features.shape[1]:
        raise ValueError(
            f"{a.name} has {a.features.shape[1]} feature columns and {b.name} has "
            f"{b.features.shape[1]}; the two were built under different feature sets"
        )
    # Squared Euclidean, to match `loss_fun="square_loss"` on the structure term:
    # a fused distance that mixes a squared structure loss with a linear feature
    # loss is not homogeneous, and alpha would then mean something different at
    # every scale.
    return ot.dist(a.features, b.features, metric="sqeuclidean")


def fgw_distance(a: MorphologyGraph, b: MorphologyGraph, *, alpha: float) -> float:
    """FGW_2 between two morphology graphs, via POT.

    Returns the objective value (1 - alpha) <M, T> + alpha * GW(T), minimised
    over couplings. Zero exactly when the two graphs are isomorphic with equal
    features, which is the case 9.3.4 asserts on every twin pair.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    value = ot.gromov.fused_gromov_wasserstein2(
        _feature_cost(a, b),
        a.structure,
        b.structure,
        a.weights,
        b.weights,
        loss_fun="square_loss",
        symmetric=True,
        alpha=alpha,
        max_iter=_MAX_ITER,
        tol_rel=_TOL,
        tol_abs=_TOL,
    )
    value = float(value)
    if value < 0.0:
        # The conditional-gradient solver can undershoot zero by a few ulp on a
        # pair whose true distance is zero. Anything below -1e-12 is not that.
        if value < -1e-12:
            raise AssertionError(
                f"FGW({a.name}, {b.name}) came back as {value!r}; a squared "
                "transport cost cannot be negative, so the solver has failed"
            )
        value = 0.0
    return value


def fgw_between_families(
    family_a: str, family_b: str, featset: FeatureSet, *, lock: LockFile | None = None
) -> float:
    lock = lock if lock is not None else LockFile()
    ga = build_morphology_graph(descriptor(family_a), featset, lock=lock)
    gb = build_morphology_graph(descriptor(family_b), featset, lock=lock)
    return fgw_distance(ga, gb, alpha=alpha_for(featset, lock))


# ---------------------------------------------------------------------------
# Transcription of the literature facts
# ---------------------------------------------------------------------------


def record_facts(
    *,
    alpha: float,
    loss_fun: str,
    armijo: bool,
    verified: bool,
    epsilon: float | None = None,
    lock: LockFile | None = None,
) -> LockFile:
    """Write the two `fgw.*` literature facts into the lock file.

    W1 item 5 (9.8.5) is to read the baseline's graph constructor field by
    field. This is where the reading lands. `alpha` has no default on the
    command line on purpose: it is a number from appendix E table 8, and a
    default would be this package inventing the baseline's hyperparameter.

    `epsilon` is table 8's regularisation. It is recorded rather than used,
    because a non-zero value means the baseline runs the *entropic* solver, and
    an entropic FGW does not return zero on an isomorphic pair with equal
    features -- it returns something of order epsilon. 9.3.4 asserts
    `fgw_baseline = 0` to within 1e-12 on every twin pair, and that assertion is
    unreachable under the baseline's own setting. The exact solver is used here
    and the deviation travels with the fact.
    """
    lock = lock if lock is not None else LockFile()
    lock.put_fact(LITERATURE_FACTS["fgw.featset_baseline"], list(_BASELINE_FIELDS_TRANSCRIPTION))
    lock.put_fact(
        LITERATURE_FACTS["fgw.hyperparameters_baseline"],
        {
            "alpha": float(alpha),
            "loss_fun": loss_fun,
            "armijo": bool(armijo),
            "solver": "ot.gromov.fused_gromov_wasserstein2",
            "pot_version": ot.__version__,
            "verified_against_source": bool(verified),
            "epsilon_reported": epsilon,
            "structure_ground_metric": "shortest path distance",
            "feature_ground_metric_reported": "euclidean",
            "feature_ground_metric_used": "sqeuclidean",
            "node_weights": "uniform",
            "descriptors_standardised": True,
            "omitted_urma_fields": dict(_BASELINE_FIELDS_OMITTED),
            "deviations": [
                "table 8 gives epsilon = 1e-3, i.e. the entropic solver; the exact "
                "conditional-gradient solver is used so that an isomorphic pair "
                "returns an exact zero, which 9.3.4's 1e-12 assertion requires",
                "table 8 gives a Euclidean feature ground metric; the squared "
                "Euclidean is used so the fused objective stays homogeneous with "
                "the square_loss structure term and alpha means one thing at every "
                "scale",
            ],
        },
    )
    return lock


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--record-facts",
        action="store_true",
        help="transcribe the baseline's feature set and solver settings into the lock file",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="the baseline's FGW alpha, from arXiv:2602.18025 appendix E table 8",
    )
    p.add_argument("--loss-fun", default="square_loss")
    p.add_argument("--armijo", action="store_true")
    p.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="table 8's FGW regularisation, recorded as a deviation rather than used",
    )
    p.add_argument(
        "--verified",
        action="store_true",
        help="set only when the values were read off the source, not inferred",
    )
    args = p.parse_args(argv)

    if not args.record_facts:
        p.error("nothing to do; pass --record-facts")
    if args.alpha is None:
        p.error("--alpha is required and has no default (W1 item 5, 9.8.5)")

    lock = record_facts(
        alpha=args.alpha,
        loss_fun=args.loss_fun,
        armijo=args.armijo,
        verified=args.verified,
        epsilon=args.epsilon,
    )
    lock.save()
    print(f"[lock] wrote fgw.featset_baseline and fgw.hyperparameters_baseline to {lock.path}")
    print(f"       baseline fields: {list(_BASELINE_FIELDS_TRANSCRIPTION)}")
    if not args.verified:
        print("       verified_against_source=False -- W1 item 5 is still open")
    return 0


def _self_check() -> None:
    leak = {"dof_frictionloss", "dof_damping"} & set(FEATSET_OURS_FIELDS)
    if leak:
        raise AssertionError(
            f"FEATSET_OURS contains {sorted(leak)}; 9.3.4 defines it as the feature "
            "set without friction or damping, and the whole point of reporting two "
            "numbers is that the second one may contain them and ours may not"
        )
    if not {"dof_frictionloss", "dof_damping"} <= set(_BASELINE_FIELDS_TRANSCRIPTION):
        raise AssertionError(
            "the baseline transcription no longer contains damping and frictionloss; "
            "if that is a corrected reading of appendix E then CONTRACT 3.5 has to be "
            "revisited, and with it the core roster -- D6 is on the qfrc_applied "
            "caliber only because of this reading, and it left the core six only "
            "because of what that caliber costs in stiction (9.2.6)"
        )


_self_check()


if __name__ == "__main__":
    raise SystemExit(main())
