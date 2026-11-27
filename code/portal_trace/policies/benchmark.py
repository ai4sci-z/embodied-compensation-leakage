"""Build every arm, run a forward and a backward pass, report capacity and step time.

Run as `.venv/bin/python -m portal_trace.policies.benchmark`.

What this is for: 10.4.9 requires the activated and total parameter counts of every
isolation level to be published, 10.5.5 requires B7 to match the pooled arm's training
budget, and 10.8 costs the whole plan in GPU-hours per run. All three are properties of
the built model rather than of the proposal, so they are measured here and the table is
what a results section quotes.

The batch is small (4 samples) because this machine has no GPU, but nothing about it is
a stand-in: two 84x84 RGB streams with the real history window, the real 70-dimensional
proprioception layout, the real 8-step chunk, the real 66-column physical basis. The
tensors are random, which is the one thing they may be -- a timing and a parameter count
do not depend on the values, and no result in the proposal is computed here.

Timings are wall-clock seconds per forward-plus-backward step on CPU, median of a few
repeats after one warm-up step. They are useful as ratios between arms, not as absolute
cluster figures.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import statistics
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import torch

from portal_trace.policies.baselines import BASELINES
from portal_trace.policies.baselines.b03_morphology_grouping import (
    B3MorphologyGroupedPolicy,
    GradientSurgery,
    GradNorm,
    MorphologyGraph,
    PCGrad,
    assert_pooling_equivalent,
    cluster_embodiments,
    fgw_distance,
    fgw_distance_matrix,
)
from portal_trace.policies.baselines.b04_disentanglement import (
    B4DisentanglementPolicy,
    B4Settings,
    B4Variant,
)
from portal_trace.policies.baselines.b05_domain_randomisation import DEFECT_RANDOMISATION
from portal_trace.policies.baselines.b06_embodiment_token import B6EmbodimentTokenPolicy
from portal_trace.policies.baselines.b07_oracle import POOL as B7_POOL
from portal_trace.policies.baselines.b07_oracle import check_matched
from portal_trace.policies.baselines.b08_counterfactual_decompensation import (
    clip_saturation_fraction,
    decompensated_actions,
)
from portal_trace.policies.baselines.b10_curated_pooling import (
    B10CuratedPoolingPolicy,
    CurationLevel,
    CurationRule,
    fit_frame_threshold,
)
from portal_trace.policies.baselines.b11_residual_calibrator import (
    TARGET_DEMO_LEVELS,
    B11ResidualCalibratorPolicy,
)
from portal_trace.policies.baselines.b12_latent_action_decoders import B12LatentActionPolicy
from portal_trace.policies.conditioning import EmbodimentTokenKind
from portal_trace.policies.decomposition import DecompositionForm, E4DecompositionPolicy
from portal_trace.policies.himoe_adapter import (
    ASMoE,
    ASMoESettings,
    HBMoE,
    HBMoESettings,
    himoe_layer_plan,
)
from portal_trace.policies.interface import Batch, Observation, Policy, PolicyConfig
from portal_trace.policies.isolation import (
    IsolationLadder,
    IsolationLevel,
    isolation_index,
    measure_route_shares,
    with_measured_shares,
)
from portal_trace.policies.obs import ACTION_DIM, CHUNK, PROPRIO_DIM
from portal_trace.policies.physics_basis import psi_column_count
from portal_trace.policies.pools import POOLS, check_alpha_identity
from portal_trace.policies.registry import build
from portal_trace.policies.schedule import OptimSpec

__all__ = ["make_batch", "ArmResult", "benchmark_arm", "main"]

# Small enough to run on a laptop CPU, large enough that every code path that depends on
# more than one embodiment in a batch is exercised: with 4 samples and 7 pool
# embodiments the hard routers see several distinct keys per step.
BATCH = 4
# The measured effective rank of phi is a per-cell measurement (9.4.4 marks it
# unverified, expected 150-400). 256 is used here only to size a tensor for the timing
# run; the low-rank head takes the width from its caller and has no default.
PHI_DIM_FOR_TIMING = 256


def make_batch(
    cfg: PolicyConfig, *, batch: int = BATCH, seed: int = 0, with_labels: bool = True
) -> Batch:
    """One realistic batch for the given config."""
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    k = cfg.obs.k_hist
    size = cfg.obs.image_size
    images = {
        cam: torch.rand(batch, k, 3, size, size, generator=generator)
        for cam in cfg.obs.cameras
    }
    obs = Observation(
        images=images,
        proprio=randn(batch, k, PROPRIO_DIM) if cfg.obs.use_proprio else None,
        task_index=torch.randint(0, cfg.n_tasks, (batch,), generator=generator)
        if cfg.obs.use_task_token
        else None,
        embodiment_index=torch.randint(
            0, cfg.n_pool_embodiments, (batch,), generator=generator
        ),
        # E9's I3' routes on a random partition of the same pool, with the same module
        # count; drawing it independently of the embodiment is exactly the difference.
        partition_index=torch.randint(
            0, cfg.n_pool_embodiments, (batch,), generator=generator
        ),
    )
    if not with_labels:
        return Batch(obs=obs)
    return Batch(
        obs=obs,
        actions=randn(batch, CHUNK, ACTION_DIM).clamp(-1.0, 1.0),
        action_pad_mask=torch.ones(batch, CHUNK, dtype=torch.bool),
        c_true=0.1 * randn(batch, CHUNK, ACTION_DIM),
        phi_features=randn(batch, CHUNK, PHI_DIM_FOR_TIMING),
        psi_basis=randn(batch, CHUNK, psi_column_count(), 6),
        extras={"action_history": randn(batch, k, ACTION_DIM).clamp(-1.0, 1.0)},
    )


@dataclass(frozen=True)
class ArmResult:
    """One row of the table."""

    arm: str
    total: int
    trainable: int
    frozen: int
    active: int
    n_blocks: int
    isolation_index: float | None
    step_seconds: float
    loss: float
    grad_coverage: float

    def as_row(self) -> str:
        index = "-" if self.isolation_index is None else f"{self.isolation_index:.3f}"
        return (
            f"{self.arm:<34} {self.total / 1e6:>9.3f} {self.trainable / 1e6:>10.3f} "
            f"{self.frozen / 1e6:>8.3f} {self.active / 1e6:>9.3f} {self.n_blocks:>7} "
            f"{index:>7} {self.step_seconds * 1e3:>10.1f} {self.loss:>9.4f} "
            f"{self.grad_coverage:>6.2f}"
        )


_HEADER = (
    f"{'arm':<34} {'total/M':>9} {'train/M':>10} {'frozen/M':>8} {'active/M':>9} "
    f"{'blocks':>7} {'I':>7} {'step/ms':>10} {'loss':>9} {'dLdw':>6}"
)


def benchmark_arm(
    policy: Policy, batch: Batch, *, repeats: int = 3, label: str | None = None
) -> ArmResult:
    """Forward, backward, time it, and read the parameter report."""
    policy.train()
    report = policy.parameter_report()
    try:
        index: float | None = isolation_index(report.blocks)
    except ValueError:
        # I1's soft router has no share table until it has been run on per-embodiment
        # batches; `main` measures it separately rather than leaving a hole in the table.
        index = None

    timings: list[float] = []
    loss_value = 0.0
    for step in range(repeats + 1):
        policy.zero_grad(set_to_none=True)
        start = time.perf_counter()
        out = policy(batch)
        if out.loss is None:
            raise AssertionError(f"{policy.arm} returned no loss on a labelled batch")
        out.loss.backward()
        elapsed = time.perf_counter() - start
        if step:  # the first pass pays lazy allocation and is not timed
            timings.append(elapsed)
        loss_value = float(out.loss.detach())
    grads = sum(
        1 for p in policy.parameters() if p.requires_grad and p.grad is not None
    )
    expected = sum(1 for p in policy.parameters() if p.requires_grad)
    if not grads:
        raise AssertionError(
            f"{policy.arm}: no trainable tensor received a gradient, so the loss is "
            "disconnected from the parameters"
        )
    # `dLdw` is the fraction of trainable tensors the step reached, and for a
    # hard-routed arm it is *supposed* to be below one: with `BATCH` samples over K
    # modules, only the routed replicas are on the path. The column is therefore a
    # direct reading of the routing, and 1.00 on I2/I3 would mean the router leaked.
    return ArmResult(
        arm=label or policy.arm,
        total=report.total,
        trainable=report.trainable,
        frozen=report.frozen,
        active=report.active,
        n_blocks=len(report.blocks),
        isolation_index=index,
        step_seconds=statistics.median(timings),
        loss=loss_value,
        grad_coverage=grads / expected,
    )


def _arms(cfg: PolicyConfig) -> Iterator[tuple[str, Callable[[], Policy]]]:
    """Every arm, plus the variants that would otherwise never be built.

    The registry's default for a variant arm is one point of its sweep; the extra
    entries here cover the rest, because "instantiate every baseline" means every
    variant 10.5 lists, not one representative per code.
    """
    yield "portal-10.4.0", lambda: build("portal-10.4.0", cfg)
    for form in DecompositionForm:
        yield (
            f"E4-{form.value}",
            lambda form=form: E4DecompositionPolicy(
                cfg, form=form, phi_dim=PHI_DIM_FOR_TIMING
            ),
        )
    for level in IsolationLevel:
        yield f"E9-{level.value}", lambda level=level: IsolationLadder(cfg).build(level)
    for code, spec in BASELINES.items():
        yield f"{code} {spec.claim}", lambda spec=spec: spec.build(cfg)
    for variant in B4Variant:
        yield (
            f"B4 variant {variant.value}",
            lambda variant=variant: B4DisentanglementPolicy(
                cfg, settings=B4Settings(variant=variant)
            ),
        )
    yield (
        "B6 one-hot form",
        lambda: B6EmbodimentTokenPolicy(cfg, kind=EmbodimentTokenKind.ONE_HOT),
    )
    for level in CurationLevel:
        yield (
            f"B10 level {level.value}",
            lambda level=level: B10CuratedPoolingPolicy(
                cfg,
                rule=CurationRule(
                    level=level,
                    # Plausible per-source readings: the target at zero and the sources
                    # spread over the range B10 curates on. Real values come from a
                    # previous run's diagnostic.
                    source_lambda=(0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8),
                    frame_threshold=1.0,
                ),
            ),
        )
    for n in TARGET_DEMO_LEVELS:
        yield (
            f"B11 N={n}",
            lambda n=n: B11ResidualCalibratorPolicy(cfg, n_target_demos=n),
        )
    yield (
        "B3 grouped (3 groups, PCGrad)",
        lambda: B3MorphologyGroupedPolicy(
            cfg,
            groups=(0, 0, 1, 1, 2, 2, 2),
            surgery=GradientSurgery.PCGRAD,
        ),
    )


def _twin_graphs() -> tuple[MorphologyGraph, MorphologyGraph, MorphologyGraph]:
    """A structurally aligned twin pair and one graph that differs.

    The features are a 7-joint chain with plausible manipulator numbers; what matters for
    9.3.4's argument is that the twin's are bit-identical, which is the case here because
    the injection layers of the six core defects leave `mjModel` alone.
    """
    n = 7
    adjacency = np.zeros((n, n))
    for i in range(n - 1):
        adjacency[i, i + 1] = adjacency[i + 1, i] = 1.0
    features = np.stack(
        [
            np.array([0.0, 0.0, 1.0, -2.9, 2.9, 87.0, 2.2, 5.0]),
            np.array([1.0, 0.0, 0.0, -1.8, 1.8, 87.0, 2.2, 2.5]),
            np.array([0.0, 0.0, 1.0, -2.9, 2.9, 87.0, 2.2, 1.667]),
            np.array([1.0, 0.0, 0.0, -3.1, 0.0, 87.0, 2.2, 1.25]),
            np.array([0.0, 0.0, 1.0, -2.9, 2.9, 12.0, 2.6, 1.0]),
            np.array([1.0, 0.0, 0.0, -0.02, 3.8, 12.0, 2.6, 0.833]),
            np.array([0.0, 0.0, 1.0, -2.9, 2.9, 12.0, 2.6, 0.714]),
        ]
    )
    clean = MorphologyGraph("panda-clean", features, adjacency)
    defective = MorphologyGraph("panda-D6", features.copy(), adjacency.copy())
    other = features.copy()
    other[:, 5] *= 0.5   # a genuinely different family: half the torque limits
    return clean, defective, MorphologyGraph("other-family", other, adjacency)


def verify(cfg: PolicyConfig) -> None:
    """The checks that are not timings: the ones that would make an arm the wrong arm."""
    print("Vendored HiMoE-VLA action module (third_party/himoe)")
    plan = himoe_layer_plan(cfg.n_layers)
    as_settings = ASMoESettings.matched_to_dense(d_model=cfg.d_model, ffn_dim=cfg.ffn_dim)
    hb_settings = HBMoESettings.matched_to_dense(d_model=cfg.d_model, ffn_dim=cfg.ffn_dim)
    print(f"  import ok: ASMoE={ASMoE.__module__}, HBMoE={HBMoE.__module__}")
    print(f"  layer plan for {cfg.n_layers} layers: {[k.value for k in plan]}")
    print(
        f"  expert widths matched to the dense block: AS={as_settings.moe_intermediate_size}, "
        f"HB={hb_settings.moe_intermediate_size}"
    )

    print("B3: FGW distance and the clustering it feeds")
    clean, defective, other = _twin_graphs()
    d_twin = fgw_distance(clean, defective)
    matrix = fgw_distance_matrix((clean, defective, other))
    labels = cluster_embodiments(matrix)
    print(f"  twin pair distance: {d_twin:.3e} (9.3.4 requires exactly zero)")
    print(f"  distance to a different family: {matrix[0, 2]:.4f}")
    print(f"  clustering of (clean, twin, other): {labels}")
    print(f"  clustering of a pure twin pool: {cluster_embodiments(matrix[:2, :2])}")
    single = B3MorphologyGroupedPolicy(cfg, groups=(0,) * cfg.n_pool_embodiments)
    assert_pooling_equivalent(single)
    print("  single-group B3 is parameter-for-parameter the pooled arm: ok")
    del single

    print("B3: the two gradient-surgery variants")
    grouped = B3MorphologyGroupedPolicy(cfg, groups=(0, 0, 1, 1, 2, 2, 2))
    batch = make_batch(cfg, seed=7)
    losses = grouped.group_losses(batch)
    surgery = PCGrad(grouped.shared_parameters())
    reported = surgery.backward([losses[k] for k in sorted(losses)])
    touched = sum(1 for p in grouped.shared_parameters() if p.grad is not None)
    print(f"  PCGrad over {len(losses)} groups, summed loss {float(reported):.4f}, "
          f"{touched} shared tensors updated")
    gradnorm = GradNorm(len(losses))
    grouped.zero_grad(set_to_none=True)
    fresh = grouped.group_losses(batch)
    ordered = [fresh[k] for k in sorted(fresh)]
    weighted = gradnorm.weighted_loss(ordered)
    balance = gradnorm.balance_loss(ordered, grouped.last_shared_parameter)
    print(
        f"  GradNorm weighted loss {weighted.item():.4f}, balance term "
        f"{balance.item():.4f}, weights {gradnorm.weights.detach().tolist()}"
    )
    del grouped

    print("B5: the defect-parameter randomisation distribution")
    drawn = DEFECT_RANDOMISATION.sample(torch.Generator().manual_seed(0))
    delay = drawn["D2"]["n_control_steps"]
    print(f"  sampled {len(drawn)} defect codes, D2 delay = {delay:.2f} control steps")

    print("B6: the inference-time token permutation")
    token_arm = B6EmbodimentTokenPolicy(cfg)
    token_arm.permute_tokens(torch.tensor([1, 0, 2, 3, 4, 5, 6]))
    print(f"  permutation installed: {token_arm.token.permutation.tolist()}")
    del token_arm

    print("B7: the matched items of 10.5.5 that are checkable in code")
    check_matched(
        cfg.evolve(arm="B7-oracle"),
        cfg.evolve(arm="B2-naive-pooling"),
        oracle_optim=OptimSpec(),
        pooled_optim=OptimSpec(),
    )
    print(f"  architecture and budget match; pool holds {B7_POOL.total_demos} demonstrations")

    print("B8: the label transform and its clipping cost")
    print(
        f"  relabelled action range "
        f"[{float(decompensated_actions(batch).min()):.3f}, "
        f"{float(decompensated_actions(batch).max()):.3f}], "
        f"clip saturation {clip_saturation_fraction(batch):.4f}"
    )

    print("B10: the frame-level cut, fitted over the pool")
    magnitudes = [
        make_batch(cfg, seed=s).require_c_true().flatten(1).norm(dim=1) for s in range(8)
    ]
    print(f"  80th percentile of ||c_delta||: {fit_frame_threshold(magnitudes):.4f}")

    print("B12: the DDPM sampling path used at rollout")
    with contextlib.redirect_stdout(io.StringIO()):
        latent_arm = B12LatentActionPolicy(cfg)
    chunk = latent_arm.act(make_batch(cfg, with_labels=False).obs)
    print(f"  act() over {latent_arm.n_inference_steps} denoising steps returned {chunk.shape}")
    del latent_arm

    print("Pools: alpha_frame equals alpha_demo bit for bit (X14)")
    for name in ("B1", "B2", "B7", "B8"):
        spec = POOLS[name]
        check_alpha_identity(spec)
        print(
            f"  {name}: {spec.total_demos} demonstrations, alpha = {spec.alpha_demo:.6f}"
        )
    print()


def _measure_i1_index(cfg: PolicyConfig) -> float:
    """The realised isolation index of E9's I1, from measured routing shares.

    One batch per pool embodiment, because `s_{k,p}` is an expectation under that
    embodiment's own state distribution (10.4.9).
    """
    policy = IsolationLadder(cfg).build(IsolationLevel.I1_SOFT_ROUTED)
    batches = {}
    for k in range(cfg.n_pool_embodiments):
        batch = make_batch(cfg, seed=100 + k)
        batch.obs.embodiment_index = torch.full_like(batch.obs.embodiment_index, k)
        batches[k] = batch
    shares = measure_route_shares(policy, batches)
    blocks = with_measured_shares(policy.parameter_blocks(), shares)
    return isolation_index(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    torch.manual_seed(0)
    cfg = PolicyConfig(arm="benchmark")

    verify(cfg)

    print(_HEADER)
    print("-" * len(_HEADER))
    for label, factory in _arms(cfg):
        # robomimic's ConditionalUnet1D prints its parameter count from `__init__`; the
        # count is in the table already and the print lands in the middle of a row.
        with contextlib.redirect_stdout(io.StringIO()):
            policy = factory()
        # B9 has no proprioception pathway, so its batch must not carry one; every arm
        # is fed the batch its own config declares.
        batch = make_batch(policy.cfg, batch=args.batch)
        result = benchmark_arm(policy, batch, repeats=args.repeats, label=label)
        print(result.as_row())
        del policy

    print()
    print("E9 isolation ladder (10.4.9): activated and total parameters per level")
    ladder = IsolationLadder(cfg)
    rows = ladder.report()
    for row in rows:
        print(f"  {row.as_row()}")
    # The number below is the index of an *untrained* HB-MoE router on random inputs.
    # The router gates on the hidden states and never sees the embodiment index, so with
    # noise for input the share table is the same for every embodiment and the index is
    # near zero by construction. 10.4.9's 0.28 is the expectation for a trained router on
    # real per-embodiment data; this line verifies the measurement path, not the value.
    print(f"  I1 isolation index, untrained router on random inputs: {_measure_i1_index(cfg):.3f}")
    print()
    print("Inference path: one action chunk from `act`, no labels on the batch")
    policy = build("portal-10.4.0", cfg)
    chunk = policy.act(make_batch(cfg, with_labels=False).obs)
    print(f"  act() returned {chunk.shape}, finite: {bool(chunk.__abs__().max() < 1e9)}")


if __name__ == "__main__":
    main()
