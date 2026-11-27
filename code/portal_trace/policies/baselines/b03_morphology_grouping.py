"""B3: "group the update by morphological similarity".

**Proposal item.** Baseline B3 of 10.5, reimplemented from `Cross-Embodiment Offline
RL for Heterogeneous Robot Datasets` (ICLR 2026, arXiv 2602.18025) as described in
2.1, 9.3 and 10.5.7. Fused Gromov-Wasserstein distance on the morphology graph,
hierarchical clustering into fixed groups, a globally updated shared component and a
group-wise updated actor, in two gradient-surgery variants (PCGrad, GradNorm).

**What it controls for.** The strongest published "negative transfer is a gradient
conflict between morphologies" method, and therefore the arm that decides whether the
defect axis needs anything new at all. On the 36 structurally aligned twin pairs the
control is constructive rather than empirical: the grouping signal is static and
purely geometric, so a twin pair has FGW distance exactly zero, the clustering returns
a single group, and the group-wise update rule becomes *literally* naive pooling. The
wording discipline of 9.3 applies to anything said about it -- "it is undefined on the
defect axis", never "morphology clustering does not work".

That equivalence is a property of this code, not only of the prose: with one group the
module holds exactly the parameters of `PortalTracePolicy` in exactly the same order,
and `forward` takes the same path B2 does. `assert_pooling_equivalent` is the check.

**Provenance.** Reimplemented from the paper; the authors publish no code. Two
components are taken from installed packages rather than written here: the FGW solver
is `ot.gromov.fused_gromov_wasserstein2` from POT, and the clustering is
`sklearn.cluster.AgglomerativeClustering` on the precomputed distance matrix. 10.5.7
requires the baseline's own discipline -- the full node-feature set, and a correct
implementation on a manipulator, since the method was only ever run on legged
platforms -- so the node features are declared explicitly and both feature sets of
9.3.4 are supported: `NODE_FEATURES` is this project's, `BASELINE_NODE_FEATURES` is
URMA's joint descriptor as the paper's appendix E adopts it, which reads damping and
friction and is what makes the D6 caliber choice of `defects/params.py` load bearing.

Building a `MorphologyGraph` from an MJCF belongs to the registry, not here; this
module takes the graphs as data so that the same distance matrix can be reported for
both feature sets (9.3.4 requires the numerical matrix and the realised partition to
be published).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
import ot
import torch
from scipy.sparse.csgraph import shortest_path
from sklearn.cluster import AgglomerativeClustering
from torch import nn

from portal_trace.policies.decoder import ActionChunkDecoder
from portal_trace.policies.interface import (
    Batch,
    ParameterBlock,
    PolicyConfig,
    PolicyOutput,
    RoutingKey,
    dense_blocks,
)
from portal_trace.policies.losses import chunk_bc_loss
from portal_trace.policies.pools import PoolSpec, SubsetName, pool
from portal_trace.policies.portal_policy import PortalTracePolicy

__all__ = [
    "POOL",
    "SUBSET",
    "SEEDS",
    "NODE_FEATURES",
    "BASELINE_NODE_FEATURES",
    "MorphologyGraph",
    "FGW_ALPHA",
    "INTRA_CLUSTER_THRESHOLD",
    "fgw_distance",
    "fgw_distance_matrix",
    "cluster_embodiments",
    "GradientSurgery",
    "PCGrad",
    "GradNorm",
    "B3MorphologyGroupedPolicy",
    "assert_pooling_equivalent",
]

POOL: PoolSpec = pool("B2-like")
# The subset B3's runs are actually run on, which is E2's -- S36 -- because 10.5 gives
# B3 zero runs of its own and counts the two variants' training inside E2'. The S6 in
# 9.8.3 is the *porting* cost row, a different quantity, and reading it as the design
# subset is what left this line disagreeing with `arms.yaml`.
SUBSET = SubsetName.S36
# Per-run seed count, which is not `seeds.yaml`'s allocation: that file gives B3 an empty
# list because 10.5 gives it zero runs, while this is what one run would cost if launched.
SEEDS = 1

# 10.5.7: the full node-feature set, and correctly on a manipulator. Joint axis is
# three entries; `armature` is inside the inertia term and is the field 9.2's D4
# discussion requires to be non-zero on both sides of a twin pair.
NODE_FEATURES: tuple[str, ...] = (
    "axis_x",
    "axis_y",
    "axis_z",
    "range_lo",
    "range_hi",
    "torque_limit",
    "velocity_limit",
    "inertia_with_armature",
)

# The baseline's own feature set, URMA's joint descriptor as adopted in the paper's
# appendix E. It reads damping and a friction coefficient, which is why D6 ships on the
# `qfrc_applied` caliber: writing `dof_frictionloss` would make the twin-pair distance
# non-zero under *this* set and take the whole X1 argument with it (9.2.6, 9.3.4).
BASELINE_NODE_FEATURES: tuple[str, ...] = (
    *NODE_FEATURES,
    "damping",
    "friction_coefficient",
)

# POT's convention: alpha weights the structure term, (1 - alpha) the feature term.
# Even weighting, which is the paper's setting and the only choice that keeps the
# twin-pair zero from being an artefact of one term dominating.
FGW_ALPHA = 0.5

# Below this the pair is inside one cluster. Set at the numerical noise floor of the
# solver rather than at a modelling choice: 9.3.4 asserts the twin-pair distance is
# zero, and every non-zero distance in the pool is orders of magnitude larger.
INTRA_CLUSTER_THRESHOLD = 1e-9


@dataclass(frozen=True)
class MorphologyGraph:
    """One robot as the paper represents it: a graph with per-node features.

    `adjacency` is the unweighted kinematic-tree adjacency; the structure matrix the
    FGW solver consumes is its shortest-path metric, which is the standard choice for
    FGW on graphs and is invariant to node ordering only up to the coupling FGW
    solves for -- so no canonical ordering is assumed anywhere.
    """

    name: str
    node_features: np.ndarray  # (n_nodes, F), one row per node
    adjacency: np.ndarray      # (n_nodes, n_nodes), symmetric, zero diagonal
    feature_names: tuple[str, ...] = NODE_FEATURES

    def __post_init__(self) -> None:
        f = np.asarray(self.node_features, dtype=np.float64)
        a = np.asarray(self.adjacency, dtype=np.float64)
        if f.ndim != 2:
            raise ValueError(f"{self.name}: node_features must be (n_nodes, F)")
        if f.shape[1] != len(self.feature_names):
            raise ValueError(
                f"{self.name}: {f.shape[1]} feature columns for "
                f"{len(self.feature_names)} declared names"
            )
        if a.shape != (f.shape[0], f.shape[0]):
            raise ValueError(
                f"{self.name}: adjacency {a.shape} does not match {f.shape[0]} nodes"
            )
        if not np.allclose(a, a.T):
            raise ValueError(f"{self.name}: the kinematic tree is undirected")

    @property
    def n_nodes(self) -> int:
        return int(self.node_features.shape[0])

    def structure_matrix(self) -> np.ndarray:
        d = shortest_path(np.asarray(self.adjacency, dtype=np.float64), directed=False)
        if not np.isfinite(d).all():
            raise ValueError(
                f"{self.name}: the graph is disconnected, so its shortest-path metric "
                "is infinite; a kinematic tree is connected by construction"
            )
        return d


def _standardise(graphs: Sequence[MorphologyGraph]) -> list[np.ndarray]:
    """Per-column standardisation over the pooled node set.

    Necessary and not cosmetic: torque limits are ~10^2 N m and axis entries are in
    [-1, 1], so an unscaled feature cost is a torque-limit cost. Statistics are pooled
    across graphs so the transform is one map applied to all of them -- a per-graph
    transform would make identical graphs land at different features.
    """
    names = {g.feature_names for g in graphs}
    if len(names) != 1:
        raise ValueError(
            f"all graphs must use one feature set; got {len(names)} different ones. "
            "9.3.4 requires the FGW matrix to be reported per feature set, computed "
            "separately"
        )
    stacked = np.concatenate([np.asarray(g.node_features, dtype=np.float64) for g in graphs])
    mean = stacked.mean(axis=0)
    scale = stacked.std(axis=0)
    # A constant column carries no information; dividing by its zero standard
    # deviation would turn the whole matrix into NaN.
    scale = np.where(scale > 0.0, scale, 1.0)
    return [(np.asarray(g.node_features, dtype=np.float64) - mean) / scale for g in graphs]


def fgw_distance(
    a: MorphologyGraph,
    b: MorphologyGraph,
    *,
    alpha: float = FGW_ALPHA,
    features: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """Fused Gromov-Wasserstein distance between two morphology graphs.

    Uniform node weights, squared-loss structure term, squared-euclidean feature cost;
    solved by POT's conditional-gradient implementation.
    """
    if features is None:
        fa, fb = _standardise((a, b))
    else:
        fa, fb = features
    m = ot.dist(fa, fb, metric="sqeuclidean")
    ca, cb = a.structure_matrix(), b.structure_matrix()
    p = ot.unif(a.n_nodes)
    q = ot.unif(b.n_nodes)
    value = ot.gromov.fused_gromov_wasserstein2(
        m, ca, cb, p, q, loss_fun="square_loss", alpha=alpha
    )
    # The solver can return a tiny negative value on identical graphs; the distance is
    # non-negative by definition and a twin pair must read exactly zero.
    return max(float(value), 0.0)


def fgw_distance_matrix(
    graphs: Sequence[MorphologyGraph], *, alpha: float = FGW_ALPHA
) -> np.ndarray:
    """The numerical matrix 9.3.4 requires to be published alongside the partition."""
    if len(graphs) < 2:
        raise ValueError("a distance matrix needs at least two graphs")
    standardised = _standardise(graphs)
    n = len(graphs)
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = fgw_distance(
                graphs[i],
                graphs[j],
                alpha=alpha,
                features=(standardised[i], standardised[j]),
            )
            out[i, j] = out[j, i] = d
    return out


def cluster_embodiments(
    distances: np.ndarray, *, threshold: float = INTRA_CLUSTER_THRESHOLD
) -> tuple[int, ...]:
    """Average-linkage hierarchical clustering on the precomputed FGW matrix.

    Returns the group index per embodiment. On a pool of structurally aligned twins
    every distance is zero, so this returns all zeros -- one group -- which is the
    constructive fact 2.1 rests on.
    """
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("expected a square distance matrix")
    if not np.allclose(distances, distances.T):
        raise ValueError("FGW distances are symmetric; this matrix is not")
    if float(distances.max()) <= threshold:
        return (0,) * distances.shape[0]
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = model.fit_predict(distances)
    return tuple(int(v) for v in labels)


# --- gradient surgery -----------------------------------------------------


class GradientSurgery(str, Enum):
    """The two variants 10.5 attaches to B3."""

    PCGRAD = "pcgrad"
    GRADNORM = "gradnorm"


class PCGrad:
    """Projecting conflicting gradients (Yu et al. 2020), reimplemented.

    For each ordered pair of groups, the gradient of group `i` is projected onto the
    normal plane of group `j`'s gradient whenever their inner product is negative, and
    the projected gradients are summed. Written against a parameter list rather than an
    optimiser so that it applies to the shared trunk only: the per-group heads receive
    exactly one group's gradient by construction and have nothing to conflict with.
    """

    def __init__(self, parameters: Sequence[nn.Parameter]) -> None:
        self.parameters = [p for p in parameters if p.requires_grad]
        if not self.parameters:
            raise ValueError("PCGrad needs at least one trainable shared parameter")

    def _flat_grad(self, loss: torch.Tensor, retain: bool) -> torch.Tensor:
        grads = torch.autograd.grad(
            loss, self.parameters, retain_graph=retain, allow_unused=True
        )
        return torch.cat(
            [
                torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
                for p, g in zip(self.parameters, grads, strict=True)
            ]
        )

    def backward(self, losses: Sequence[torch.Tensor]) -> torch.Tensor:
        """Accumulate the surgered gradient into `.grad` and return the plain sum.

        The returned scalar is the unmodified sum of the group losses, for logging: it
        is not the quantity whose gradient was written, and calling `.backward()` on it
        afterwards would undo the surgery.
        """
        if not losses:
            raise ValueError("no group losses; a grouped update needs at least one")
        flat = [self._flat_grad(loss, retain=True) for loss in losses]
        projected = []
        for i, g_i in enumerate(flat):
            g = g_i.clone()
            for j, g_j in enumerate(flat):
                if i == j:
                    continue
                dot = torch.dot(g, g_j)
                if dot < 0:
                    g = g - dot / g_j.dot(g_j).clamp_min(1e-12) * g_j
            projected.append(g)
        total = torch.stack(projected).sum(dim=0)
        offset = 0
        for p in self.parameters:
            n = p.numel()
            slice_ = total[offset : offset + n].view_as(p)
            p.grad = slice_.clone() if p.grad is None else p.grad + slice_
            offset += n
        return torch.stack([loss.detach() for loss in losses]).sum()


class GradNorm(nn.Module):
    """GradNorm loss balancing (Chen et al. 2018), reimplemented.

    Learns one weight per group so that each group's gradient norm at a chosen shared
    layer tracks the average norm, rescaled by how fast that group's loss is falling.
    The weights are parameters of this module and are trained by their own objective,
    so a runner steps them with a separate optimiser -- summing the two objectives
    would let the balancing term change the trunk.
    """

    def __init__(self, n_groups: int, *, alpha: float = 1.5) -> None:
        super().__init__()
        if n_groups < 1:
            raise ValueError("GradNorm needs at least one group")
        self.alpha = alpha
        self.weights = nn.Parameter(torch.ones(n_groups))
        self.register_buffer("initial_losses", torch.zeros(n_groups))
        self.register_buffer("has_initial", torch.zeros((), dtype=torch.bool))

    def weighted_loss(self, losses: Sequence[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(list(losses))
        if not bool(self.has_initial):
            self.initial_losses.copy_(stacked.detach())
            self.has_initial.fill_(True)
        return (self.weights * stacked).sum()

    def balance_loss(
        self, losses: Sequence[torch.Tensor], shared_parameter: nn.Parameter
    ) -> torch.Tensor:
        """The objective that trains the weights.

        `shared_parameter` is the last shared layer's weight, per the paper: taking the
        norm over the whole trunk makes the term dominated by the largest tensor
        instead of by the layer where the groups actually interfere.
        """
        stacked = torch.stack(list(losses))
        norms = torch.stack(
            [
                torch.autograd.grad(
                    self.weights[i] * stacked[i],
                    shared_parameter,
                    retain_graph=True,
                    create_graph=True,
                )[0].norm()
                for i in range(stacked.shape[0])
            ]
        )
        with torch.no_grad():
            ratio = stacked.detach() / self.initial_losses.clamp_min(1e-12)
            relative = ratio / ratio.mean().clamp_min(1e-12)
            target = norms.mean().detach() * relative.pow(self.alpha)
        return (norms - target).abs().sum()


# --- the arm --------------------------------------------------------------


class B3MorphologyGroupedPolicy(PortalTracePolicy):
    """Shared trunk, one action head per morphology group.

    The paper updates the critic globally and the actor per group. Behaviour cloning
    has no critic, so the split is carried by the two halves that do exist: the
    observation encoder is shared and receives every group's gradient, and the 4-layer
    chunk decoder -- the actor -- is replicated per group and receives only its own.

    With one group there is no replica at all: `group_decoders` is empty, `self.decoder`
    serves group 0, and both the parameter set and the forward path are B2's.
    """

    arm = "B3-morphology-grouped"

    def __init__(
        self,
        cfg: PolicyConfig,
        *,
        groups: Sequence[int] = (0,) * 7,
        surgery: GradientSurgery = GradientSurgery.PCGRAD,
    ) -> None:
        super().__init__(cfg)
        groups = tuple(int(g) for g in groups)
        if len(groups) != cfg.n_pool_embodiments:
            raise ValueError(
                f"{len(groups)} group labels for {cfg.n_pool_embodiments} pool "
                "embodiments; the partition covers the pool"
            )
        n_groups = len(set(groups))
        if sorted(set(groups)) != list(range(n_groups)):
            raise ValueError(f"group labels must be 0..G-1, got {sorted(set(groups))}")
        self.n_groups = n_groups
        self.surgery = surgery
        self.register_buffer(
            "group_of_embodiment", torch.tensor(groups, dtype=torch.long), persistent=True
        )
        # Group 0 uses the inherited decoder, so the single-group case is bit-for-bit
        # the pooled arm; groups 1.. get their own replicas.
        self.group_decoders = nn.ModuleList(
            ActionChunkDecoder(
                cfg.d_model,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                ffn_dim=cfg.ffn_dim,
                dropout=cfg.dropout,
                action_dim=cfg.action_dim,
            )
            for _ in range(n_groups - 1)
        )

    def _decoder_for(self, group: int) -> ActionChunkDecoder:
        return self.decoder if group == 0 else self.group_decoders[group - 1]

    @property
    def last_shared_parameter(self) -> nn.Parameter:
        """The tensor GradNorm measures its per-group gradient norms on."""
        return self.encoder.norm.weight

    def shared_parameters(self) -> list[nn.Parameter]:
        """The parameters PCGrad operates on: everything every group touches."""
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def group_losses(self, batch: Batch) -> dict[int, torch.Tensor]:
        """One behaviour-cloning loss per group present in the batch.

        This is what the two variants consume. A group absent from the batch is absent
        from the mapping rather than contributing a zero: a zero loss would enter
        GradNorm's ratio as a converged group.
        """
        if self.n_groups == 1:
            out = self.forward(batch)
            if out.loss is None:
                raise ValueError("group losses need a batch that carries actions")
            return {0: out.loss}
        obs = batch.obs
        memory = self.encode(obs)
        target = self.action_target(batch)
        groups = self.group_of_embodiment[obs.routing_index(RoutingKey.EMBODIMENT)]
        losses: dict[int, torch.Tensor] = {}
        for group in groups.unique().tolist():
            mask = groups == group
            prediction, _ = self._decoder_for(group)(memory[mask])
            losses[group] = chunk_bc_loss(
                prediction,
                target[mask],
                pad_mask=None if batch.action_pad_mask is None else batch.action_pad_mask[mask],
                sample_weight=(
                    None if batch.sample_weight is None else batch.sample_weight[mask]
                ),
            )
        return losses

    def forward(self, batch: Batch) -> PolicyOutput:
        if self.n_groups == 1:
            return super().forward(batch)
        obs = batch.obs
        memory = self.encode(obs)
        groups = self.group_of_embodiment[obs.routing_index(RoutingKey.EMBODIMENT)]
        prediction = torch.zeros(
            obs.batch_size, self.cfg.chunk, self.cfg.action_dim, device=obs.device
        )
        for group in groups.unique().tolist():
            mask = groups == group
            chunk, _ = self._decoder_for(group)(memory[mask])
            prediction[mask] = chunk
        if batch.actions is None:
            return PolicyOutput(action_chunk=prediction)
        # The reported loss is the plain pooled one. The two variants do not change
        # this scalar, they change how its per-group parts are combined into gradients,
        # and that combination is the trainer's job (`group_losses`).
        loss = chunk_bc_loss(
            prediction,
            self.action_target(batch),
            pad_mask=batch.action_pad_mask,
            sample_weight=batch.sample_weight,
        )
        return PolicyOutput(
            action_chunk=prediction, loss=loss, aux_losses={"bc": loss.detach()}
        )

    def parameter_blocks(self) -> tuple[ParameterBlock, ...]:
        k = self.cfg.n_pool_embodiments
        if self.n_groups == 1:
            return dense_blocks(self, k)
        blocks = list(dense_blocks(self, k, exclude=("decoder", "group_decoders")))
        groups = self.group_of_embodiment.tolist()
        for group in range(self.n_groups):
            module = self._decoder_for(group)
            n = sum(p.numel() for p in module.parameters())
            blocks.append(
                ParameterBlock(
                    name=f"actor[group={group}]",
                    total_params=n,
                    # One head runs per sample; billing group 0 keeps the activated
                    # count right and the heads are all the same size.
                    active_params=n if group == 0 else 0,
                    route_shares=tuple(1.0 if g == group else 0.0 for g in groups),
                )
            )
        return tuple(blocks)


def assert_pooling_equivalent(policy: B3MorphologyGroupedPolicy) -> None:
    """The code-level half of 2.1's constructive argument.

    Raises unless a single-group B3 is parameter-for-parameter the pooled arm. This is
    what makes "FGW = 0 implies the update rule is literally naive pooling" a checked
    statement rather than a claim about an implementation the reader cannot see.
    """
    if policy.n_groups != 1:
        raise ValueError(
            f"this arm has {policy.n_groups} groups; the equivalence only holds when "
            "the FGW clustering returns one group, which is the twin-pair case"
        )
    if len(policy.group_decoders):
        raise AssertionError("a single-group arm must hold no decoder replica")
    reference = PortalTracePolicy(policy.cfg)
    ours = [(n, tuple(p.shape)) for n, p in policy.named_parameters()]
    theirs = [(n, tuple(p.shape)) for n, p in reference.named_parameters()]
    if ours != theirs:
        raise AssertionError(
            "single-group B3 does not have the pooled arm's parameter set: "
            f"{len(ours)} vs {len(theirs)} tensors"
        )
