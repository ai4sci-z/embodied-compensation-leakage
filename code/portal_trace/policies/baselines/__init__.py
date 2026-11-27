"""The thirteen baselines of 10.5, one module each, plus the table that indexes them.

10.5's own table is reproduced here as data, because three of its columns are load
bearing and are quoted in other places:

* the **claim** each arm stands for, which is how the results section reads it;
* what it is a **control** for, which is why it is in the design at all;
* the **run and GPU-hour** budget, which 10.8 sums and which contains two rows that are
  easy to double count -- B11's 18 GPU-h are head training and its run count is zero, and
  B3, B12's runs are counted inside E2' and E9 respectively.

`Provenance` records where each implementation came from. The contract is to reuse
established open source where it exists and reimplement from the paper where it does not,
so the distinction has to be visible per arm rather than described once in a README.

Total across the table: 438 runs, 2 190 GPU-h at 5.0 h per run, plus B11's 18 GPU-h of
head training counted in 10.8.4 instead. `check_totals` asserts the first two.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from portal_trace.policies.baselines import (
    b01_target_only,
    b02_naive_pooling,
    b03_morphology_grouping,
    b04_disentanglement,
    b05_domain_randomisation,
    b06_embodiment_token,
    b07_oracle,
    b08_counterfactual_decompensation,
    b09_proprio_masked,
    b10_curated_pooling,
    b11_residual_calibrator,
    b12_latent_action_decoders,
    b13_latent_extrinsics,
)
from portal_trace.policies.baselines.b01_target_only import B1TargetOnlyPolicy
from portal_trace.policies.baselines.b02_naive_pooling import B2NaivePoolingPolicy
from portal_trace.policies.baselines.b03_morphology_grouping import B3MorphologyGroupedPolicy
from portal_trace.policies.baselines.b04_disentanglement import (
    B4DisentanglementPolicy,
    B4Settings,
    B4Variant,
)
from portal_trace.policies.baselines.b05_domain_randomisation import (
    B5DomainRandomisationPolicy,
)
from portal_trace.policies.baselines.b06_embodiment_token import B6EmbodimentTokenPolicy
from portal_trace.policies.baselines.b07_oracle import B7OraclePolicy
from portal_trace.policies.baselines.b08_counterfactual_decompensation import (
    B8CounterfactualPolicy,
)
from portal_trace.policies.baselines.b09_proprio_masked import B9ProprioMaskedPolicy
from portal_trace.policies.baselines.b10_curated_pooling import (
    B10CuratedPoolingPolicy,
    CurationLevel,
    CurationRule,
)
from portal_trace.policies.baselines.b11_residual_calibrator import (
    B11ResidualCalibratorPolicy,
)
from portal_trace.policies.baselines.b12_latent_action_decoders import B12LatentActionPolicy
from portal_trace.policies.baselines.b13_latent_extrinsics import B13LatentExtrinsicsPolicy
from portal_trace.policies.interface import Policy, PolicyConfig
from portal_trace.policies.pools import PoolSpec, SubsetName

__all__ = [
    "Provenance",
    "BaselineSpec",
    "BASELINES",
    "baseline",
    "check_totals",
    "TOTAL_RUNS",
    "TOTAL_GPU_HOURS",
    "B1TargetOnlyPolicy",
    "B2NaivePoolingPolicy",
    "B3MorphologyGroupedPolicy",
    "B4DisentanglementPolicy",
    "B4Settings",
    "B4Variant",
    "B5DomainRandomisationPolicy",
    "B6EmbodimentTokenPolicy",
    "B7OraclePolicy",
    "B8CounterfactualPolicy",
    "B9ProprioMaskedPolicy",
    "B10CuratedPoolingPolicy",
    "B11ResidualCalibratorPolicy",
    "B12LatentActionPolicy",
    "B13LatentExtrinsicsPolicy",
    "CurationLevel",
    "CurationRule",
]

TOTAL_RUNS = 438
TOTAL_GPU_HOURS = 2190


class Provenance(str, Enum):
    """Where an arm's implementation came from."""

    INSTALLED_PACKAGE = "installed package"     # robomimic / diffusers / POT / sklearn
    FROM_PAPER = "reimplemented from paper"     # no public code exists
    DIRECT = "implemented directly"             # a technique, not a codebase
    VENDORED = "vendored third-party source"    # copied verbatim into third_party/


@dataclass(frozen=True)
class BaselineSpec:
    """One row of 10.5's baseline table."""

    code: str
    claim: str
    mechanism: str
    controls_for: str
    provenance: Provenance
    source: str
    subset: SubsetName
    seeds: int
    runs: int
    gpu_hours: int
    pool: PoolSpec
    build: Callable[[PolicyConfig], Policy]

    def __post_init__(self) -> None:
        if self.runs < 0 or self.gpu_hours < 0:
            raise ValueError(f"{self.code}: budgets are non-negative")
        if self.seeds < 1:
            raise ValueError(f"{self.code}: an arm runs at least one seed")


BASELINES: dict[str, BaselineSpec] = {
    spec.code: spec
    for spec in (
        BaselineSpec(
            code="B1",
            claim="don't use other people's data",
            mechanism="behaviour cloning on the target embodiment's own 200 demonstrations",
            controls_for=(
                "the lower anchor of the leakage axis: no other embodiment in the pool, "
                "so Lambda goes to zero by construction and the price is data volume"
            ),
            provenance=Provenance.INSTALLED_PACKAGE,
            source="robomimic 0.5.0 through the shared 10.4.0 architecture",
            subset=b01_target_only.SUBSET,
            seeds=b01_target_only.SEEDS,
            runs=18,
            gpu_hours=90,
            pool=b01_target_only.POOL,
            build=B1TargetOnlyPolicy,
        ),
        BaselineSpec(
            code="B2",
            claim="more data is better",
            mechanism="naive pooling, no embodiment conditioning, alpha = 0.857",
            controls_for=(
                "nothing -- it is the evaluated arm of the primary endpoint, and its pool "
                "has no delta = 0 cell, so a leakage measured on it reads as "
                "'insufficient information' rather than 'no leakage'"
            ),
            provenance=Provenance.INSTALLED_PACKAGE,
            source="robomimic 0.5.0 through the shared 10.4.0 architecture",
            subset=b02_naive_pooling.SUBSET,
            seeds=b02_naive_pooling.SEEDS,
            runs=108,
            gpu_hours=540,
            pool=b02_naive_pooling.POOL,
            build=B2NaivePoolingPolicy,
        ),
        BaselineSpec(
            code="B3",
            claim="group the update by morphological similarity",
            mechanism=(
                "FGW distance on the morphology graph with the full node feature set, "
                "hierarchical clustering, global shared component and group-wise actor "
                "updates, in a PCGrad and a GradNorm variant"
            ),
            controls_for=(
                "the published gradient-conflict account of negative transfer; on the "
                "structurally aligned twin pairs FGW is exactly zero, so the arm reduces "
                "to naive pooling as an algebraic fact"
            ),
            provenance=Provenance.FROM_PAPER,
            source=(
                "arXiv 2602.18025 (ICLR 2026), no public code; FGW solved by POT, "
                "clustering by scikit-learn"
            ),
            subset=b03_morphology_grouping.SUBSET,
            seeds=b03_morphology_grouping.SEEDS,
            # 10.5: the two variants' training runs are counted inside E2'.
            runs=0,
            gpu_hours=0,
            pool=b03_morphology_grouping.POOL,
            build=B3MorphologyGroupedPolicy,
        ),
        BaselineSpec(
            code="B4",
            claim="task-embodiment disentanglement is already solved",
            mechanism=(
                "four variants: ported CLUB + InfoNCE objective, DisDP with the private "
                "branch as the embodiment branch, embodiment discriminator with gradient "
                "reversal, learned style token in the head's FiLM layer"
            ),
            controls_for=(
                "the most dangerous collision; expected to lower the contrastive "
                "component and to leave the common mode untouched, which E7 then tests"
            ),
            provenance=Provenance.FROM_PAPER,
            source=(
                "Disentangled Cross-Embodiment Video Editing (objective ported from the "
                "pixel layer to the action layer) and DisDP (RLC 2025), whose repository "
                "is empty; estimators in policies/losses.py"
            ),
            subset=b04_disentanglement.SUBSET,
            seeds=b04_disentanglement.SEEDS,
            runs=48,
            gpu_hours=240,
            pool=b04_disentanglement.POOL,
            build=lambda cfg: B4DisentanglementPolicy(
                cfg, settings=B4Settings(variant=B4Variant.PORTED_MI)
            ),
        ),
        BaselineSpec(
            code="B5",
            claim="just randomise it away",
            mechanism="training-time domain randomisation over the defect parameters",
            controls_for=(
                "the sim2real reflex; it randomises the environment and not the "
                "demonstrator's body, so it cannot remove a compensation already inside "
                "the action label"
            ),
            provenance=Provenance.DIRECT,
            source="ranges read off the 9.2.10 level tables in defects/params.py",
            subset=b05_domain_randomisation.SUBSET,
            seeds=b05_domain_randomisation.SEEDS,
            runs=6,
            gpu_hours=30,
            pool=b05_domain_randomisation.POOL,
            build=B5DomainRandomisationPolicy,
        ),
        BaselineSpec(
            code="B6",
            claim="one token per embodiment is enough",
            mechanism="one-hot and learned embedding, with an inference-time token permutation",
            controls_for=(
                "existing practice (RT-X, Octo, OpenVLA): a per-embodiment bias absorbs "
                "the contrastive component and cannot touch the common mode"
            ),
            provenance=Provenance.DIRECT,
            source="conditioning.EmbodimentToken, entering at the decoder's FiLM layers",
            subset=b06_embodiment_token.SUBSET,
            seeds=b06_embodiment_token.SEEDS,
            runs=36,
            gpu_hours=180,
            pool=b06_embodiment_token.POOL,
            build=B6EmbodimentTokenPolicy,
        ),
        BaselineSpec(
            code="B7",
            claim="Oracle",
            mechanism=(
                "the defect-free twin of this embodiment, same reference trajectories, "
                "1 400 demonstrations, same teacher operator, same training budget"
            ),
            controls_for=(
                "the primary endpoint's reference point; it cancels the data-volume "
                "shrinkage difference and leaves a known positive bias along c_bar for "
                "layer 2 to absorb"
            ),
            provenance=Provenance.INSTALLED_PACKAGE,
            source="robomimic 0.5.0 through the shared 10.4.0 architecture",
            subset=b07_oracle.SUBSET,
            seeds=b07_oracle.SEEDS,
            runs=108,
            gpu_hours=540,
            pool=b07_oracle.POOL,
            build=B7OraclePolicy,
        ),
        BaselineSpec(
            code="B8",
            claim="counterfactual decompensation",
            mechanism="the same defective demonstrations with the label replaced by a - c_true",
            controls_for=(
                "X3 (teacher quality) and X8 (covariate shift) at once, by elimination "
                "rather than by hardening; impossible without simulator ground truth"
            ),
            provenance=Provenance.DIRECT,
            source="label transform on the shared architecture; c_true from the hdf5 dataset",
            subset=b08_counterfactual_decompensation.SUBSET,
            seeds=b08_counterfactual_decompensation.SEEDS,
            runs=54,
            gpu_hours=270,
            pool=b08_counterfactual_decompensation.POOL,
            build=B8CounterfactualPolicy,
        ),
        BaselineSpec(
            code="B9",
            claim="the leakage rides on the embodiment cues",
            mechanism="no proprioception pathway; RGB and the task instruction only",
            controls_for=(
                "which input channel carries the leakage; the interventional half of "
                "Lambda -> E_rho[w], preregistered to shrink the statistic by >= 60%"
            ),
            provenance=Provenance.DIRECT,
            source="ObsSpec(use_proprio=False) on the shared encoder",
            subset=b09_proprio_masked.SUBSET,
            seeds=b09_proprio_masked.SEEDS,
            runs=18,
            gpu_hours=90,
            pool=b09_proprio_masked.POOL,
            build=B9ProprioMaskedPolicy,
        ),
        BaselineSpec(
            code="B10",
            claim="data curation",
            mechanism=(
                "Lambda-guided source weighting in three levels: down-weight by "
                "per-source Lambda^part, drop the top 20% of frames by ||c_delta||, drop "
                "the whole high-Lambda source"
            ),
            controls_for=(
                "the only data-side baseline, and the practitioner's first move; it turns "
                "the diagnostic into a curation tool and may well not lose"
            ),
            provenance=Provenance.DIRECT,
            source="sample weighting through Batch.sample_weight",
            subset=b10_curated_pooling.SUBSET,
            seeds=b10_curated_pooling.SEEDS,
            runs=18,
            gpu_hours=90,
            pool=b10_curated_pooling.POOL,
            build=lambda cfg: B10CuratedPoolingPolicy(
                cfg,
                rule=CurationRule(
                    level=CurationLevel.SOURCE_DOWNWEIGHT,
                    source_lambda=(0.0,) * cfg.n_pool_embodiments,
                ),
            ),
        ),
        BaselineSpec(
            code="B11",
            claim="just add a residual head at deployment",
            mechanism=(
                "frozen trunk, implicit fault regime from recent joint dynamics and "
                "execution history, one shared residual calibrator, residual on the "
                "translational arm dimensions only"
            ),
            controls_for=(
                "route (I) of Proposition 1(d), expected to work from N >= 20 target "
                "demonstrations; also the published support for (A5)(ii)"
            ),
            provenance=Provenance.FROM_PAPER,
            source="J-PARC, arXiv 2606.10501, no public code",
            subset=b11_residual_calibrator.SUBSET,
            seeds=b11_residual_calibrator.SEEDS,
            # 10.5 lists zero runs and 18 GPU-h; the hours belong to 10.8.4's head
            # training table and must not be added into the baseline subtotal twice.
            runs=0,
            gpu_hours=0,
            pool=b11_residual_calibrator.POOL,
            build=B11ResidualCalibratorPolicy,
        ),
        BaselineSpec(
            code="B12",
            claim="one decoder per embodiment",
            mechanism="embodiment-agnostic latent action with per-embodiment decoders",
            controls_for=(
                "Corollary 1's empirical strike instance, and the second implementation "
                "of E9's I3 rung: per-instance parameter isolation, literally"
            ),
            provenance=Provenance.FROM_PAPER,
            source=(
                "Latent Action Diffusion, arXiv 2506.14608; denoising UNet from "
                "robomimic, noise schedule from diffusers"
            ),
            subset=b12_latent_action_decoders.SUBSET,
            seeds=b12_latent_action_decoders.SEEDS,
            # Counted inside E9's I3 rung (12 of E9's 60 runs).
            runs=0,
            gpu_hours=0,
            pool=b12_latent_action_decoders.POOL,
            build=B12LatentActionPolicy,
        ),
        BaselineSpec(
            code="B13",
            claim="learn the embodiment representation from the data",
            mechanism=(
                "K_hist frames of proprioception to a low-dimensional continuous latent "
                "that conditions the action head, with no embodiment label"
            ),
            controls_for=(
                "the missing middle rung between B6 and B4, and the 'soft conditioning' "
                "control of E9's isolation axis; it fixes the contrastive component and "
                "not the common mode"
            ),
            provenance=Provenance.FROM_PAPER,
            source=(
                "RMA-style latent extrinsics and the regime-inference half of J-PARC; a "
                "family of methods with no canonical implementation"
            ),
            subset=b13_latent_extrinsics.SUBSET,
            seeds=b13_latent_extrinsics.SEEDS,
            runs=24,
            gpu_hours=120,
            pool=b13_latent_extrinsics.POOL,
            build=B13LatentExtrinsicsPolicy,
        ),
    )
}


def baseline(code: str) -> BaselineSpec:
    try:
        return BASELINES[code]
    except KeyError:
        raise KeyError(
            f"no baseline {code!r}; 10.5 defines B1-B13, present here as "
            f"{sorted(BASELINES)}"
        ) from None


def check_totals() -> None:
    """Assert the run and GPU-hour subtotals of 10.5 against the table above."""
    runs = sum(s.runs for s in BASELINES.values())
    hours = sum(s.gpu_hours for s in BASELINES.values())
    if runs != TOTAL_RUNS:
        raise AssertionError(
            f"baseline runs sum to {runs}, 10.5's subtotal is {TOTAL_RUNS}"
        )
    if hours != TOTAL_GPU_HOURS:
        raise AssertionError(
            f"baseline GPU-hours sum to {hours}, 10.5's subtotal is {TOTAL_GPU_HOURS}"
        )


check_totals()
