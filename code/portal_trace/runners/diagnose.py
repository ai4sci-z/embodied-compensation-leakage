"""Diagnostics over collected frames (proposal 10.1, 11.4, appendix A-8, A-14).

Frames are streamed as `FrameTable` chunks rather than materialised at once:
the plan's 45 M frames do not fit in memory, and the diagnostics are all
accumulations of sufficient statistics, so a chunked pass computes the same
numbers as a whole-dataset pass would.

Two calibers are imposed here because they cannot be recovered afterwards. The
Lambda family is computed in the L^2(rho) sense -- expectation first, ratio
second -- because appendix A-14's identity Phi = rho_dir^2 holds only in that
sense, and a per-state cosine averaged afterwards satisfies no inequality
against it in either direction. And every design-graph edge weight is an
`EffectiveSampleCount` built from a measured tau_corr, because appendix A-8
shows raw frame counts overstate the information an edge carries by about an
order of magnitude.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from portal_trace.configs.schema import ConfigError
from portal_trace.diagnostics.excess import ExcessLeakageReading, excess_leakage
from portal_trace.diagnostics.leakage import (
    LeakageAccumulator,
    LeakageReport,
    PosteriorCovarianceAccumulator,
)
from portal_trace.diagnostics.units import EffectiveSampleCount
from portal_trace.lockfile import REQUIRED_MEASUREMENTS
from portal_trace.registry.tasks import HORIZON
from portal_trace.runners.common import RunContext, dataset_paths, require_level_dataset
from portal_trace.runners.dataset_contract import (
    require_analysis_trace_contract,
    require_current_defect_state_contract,
    require_dataset_identity,
)
from portal_trace.runners.leakage_input import (
    load_leakage_input_header,
    stream_leakage_input,
)
from portal_trace.types import Cell, FrameTable

__all__ = [
    "DiagnosticsUnavailable",
    "stream_frames",
    "edge_weight",
    "diagnose",
    "CHUNK_EPISODES",
]


class DiagnosticsUnavailable(RuntimeError):
    """The stored collection is not yet a complete leakage-analysis input."""

# One chunk is 16 episodes: 6400 frames, which keeps the largest intermediate
# (the phi design matrix, 392 raw columns) in the low tens of megabytes.
CHUNK_EPISODES = 16


def stream_frames(
    ctx: RunContext,
    *,
    task: str,
    family: str,
    defect: str,
    level: int,
    chunk_episodes: int = CHUNK_EPISODES,
) -> Iterator[FrameTable]:
    """Yield `FrameTable` chunks for one cell.

    `c_true` is produced during collection by the real twin query and is stored
    as a replay companion.  Diagnostics consume that audited field here rather
    than trying to reconstruct a `TeacherArm` from an HDF5 path (the query API
    requires a live twin and a reference trajectory).
    """
    import h5py

    paths = dataset_paths(
        ctx.dataset_root, task=task, family=family, defect=defect, level=level
    )
    raw = require_level_dataset(
        paths["raw"], produced_by="collection has not run for this cell and dose level"
    )
    require_dataset_identity(
        raw, task=task, family=family, defect=defect, level=level
    )
    require_current_defect_state_contract(raw, requested_defect=defect)

    cell = Cell(task=task, family=family, defect_code=defect)
    with h5py.File(raw, "r") as fh:
        # New collections advertise the analysis trace contract.  Legacy test
        # or pilot files may still be streamed for c_true-only inspection, but
        # a file that advertises the contract must satisfy it atomically.
        if "analysis_trace_schema_version" in fh.attrs:
            require_analysis_trace_contract(raw)
        demo_keys = sorted(
            fh["data"].keys(), key=lambda k: int(k.split("_")[1])
        )
        for start in range(0, len(demo_keys), chunk_episodes):
            batch = demo_keys[start : start + chunk_episodes]
            episode_index, time_index, regime = [], [], []
            defective, clean = [], []
            extras: dict[str, list[np.ndarray]] = {
                name: []
                for name in (
                    "u_inf", "u_inf_clean", "phase", "contact", "eef_pos_cmd",
                    "eef_pos_ach", "tracking_error", "joint_torque",
                )
            }
            replay_extras: dict[str, list[np.ndarray]] = {
                name: [] for name in ("states", "tau_g", "actuation_state", "defect_state")
            }
            provenance: dict[str, Any] = {}
            analysis_presence: bool | None = None
            replay_presence: bool | None = None
            for offset, key in enumerate(batch):
                g = fh["data"][key]
                actions = np.asarray(g["actions"], dtype=np.float64)
                n = actions.shape[0]
                if "c_true" not in g:
                    raise ConfigError(
                        f"{raw}:{key} has no c_true replay companion; collection must "
                        "finish the twin query before diagnostics can run"
                    )
                c_true = np.asarray(g["c_true"], dtype=np.float64)
                if c_true.shape != actions.shape:
                    raise ConfigError(
                        f"{raw}:{key} stores c_true with shape {c_true.shape}, expected "
                        f"{actions.shape}; refusing to infer a clean action"
                    )
                episode_index.append(np.full(n, start + offset, dtype=np.int64))
                time_index.append(np.arange(n, dtype=np.int64))
                regime.append(np.asarray(g["regime"], dtype=np.uint8))
                defective.append(actions)
                clean.append(actions - c_true)
                replay_present = {name for name in replay_extras if name in g}
                if replay_present and replay_present != set(replay_extras):
                    missing = sorted(set(replay_extras) - replay_present)
                    raise ConfigError(
                        f"{raw}: replay trace is partial in {key}: missing {missing}"
                    )
                has_replay = bool(replay_present)
                if replay_presence is None:
                    replay_presence = has_replay
                elif replay_presence != has_replay:
                    raise ConfigError(
                        f"{raw}: replay trace presence differs between demos in chunk {start}"
                    )
                if replay_present:
                    for name in replay_extras:
                        values = np.asarray(g[name], dtype=np.float64)
                        if values.ndim < 1 or values.shape[0] != n:
                            raise ConfigError(
                                f"{raw}:{key}/{name} has shape {values.shape}; "
                                f"expected first dimension {n}"
                            )
                        replay_extras[name].append(values)
                if not provenance:
                    for name in (
                        "model_file", "controller_config_sha256", "assets_commit",
                        "defect_impl_variant", "mode_hash", "environment",
                    ):
                        if name in g.attrs:
                            value = g.attrs[name]
                            provenance[name] = (
                                value.decode("utf-8") if isinstance(value, bytes) else value
                            )
                for name in extras:
                    if name in g:
                        extras[name].append(np.asarray(g[name], dtype=np.float64))
            present = {name for name, values in extras.items() if values}
            has_analysis = bool(present)
            if analysis_presence is None:
                analysis_presence = has_analysis
            elif analysis_presence != has_analysis:
                raise ConfigError(
                    f"{raw}: analysis trace presence differs between demos in chunk {start}"
                )
            if present and present != set(extras):
                missing = sorted(set(extras) - present)
                raise ConfigError(
                    f"{raw}: analysis trace is partial in chunk {start}: missing {missing}"
                )
            merged_extras = (
                {name: np.concatenate(values) for name, values in extras.items()}
                if present
                else {}
            )
            if any(replay_extras.values()):
                merged_extras.update(
                    {name: np.concatenate(values) for name, values in replay_extras.items()}
                )
            yield FrameTable(
                cell=cell,
                episode_index=np.concatenate(episode_index),
                time_index=np.concatenate(time_index),
                regime=np.concatenate(regime),
                actions_defective=np.concatenate(defective),
                actions_clean=np.concatenate(clean),
                extras=merged_extras,
                provenance=provenance,
            )


def edge_weight(
    ctx: RunContext, *, n_traj: int, task: str, defect: str, regime: str
) -> EffectiveSampleCount:
    """The design-graph edge weight for one cell and regime.

    tau_corr is read from the lock file and has no default. It is measured per
    cell and per regime because a reversal segment's autocorrelation time is far
    shorter than a free-space segment's, and using one number for both would
    reweight the graph.
    """
    decl = REQUIRED_MEASUREMENTS["stats.tau_corr.{task}.{defect}.{regime}"]
    tau_corr = float(ctx.lock.get(decl, task=task, defect=defect, regime=regime))
    return EffectiveSampleCount.from_trajectories(
        n_traj=n_traj, horizon=HORIZON, tau_corr=tau_corr
    )


@dataclass(frozen=True)
class DiagnosticReport:
    """What one diagnostic pass produced, with its caliber annotations."""

    cell: Cell
    level: int
    n_frames: int
    n_episodes: int
    quantities: dict[str, Any]
    clip_annotation: bool
    notes: tuple[str, ...] = ()


def _excess_summary(
    reading: ExcessLeakageReading,
    *,
    observed_quantity: str,
    contrast_quantity: str,
    role: str,
) -> dict[str, Any]:
    """Serialise one internally consistent real-versus-surrogate contrast."""
    surrogate_rows = [
        {
            "surrogate": row["surrogate"],
            "admitted": row["admitted"],
            "value": row["lambda_part"],
        }
        for row in reading.null.as_rows()
    ]
    cardinality_status = (
        "preregistered_minimum_met"
        if reading.is_confirmatory
        else "below_preregistered_minimum"
    )
    return {
        "observed_quantity": observed_quantity,
        "surrogate_reading_quantity": observed_quantity,
        "observed_value": float(reading.lambda_part.value),
        "surrogate_mean": float(reading.null.mean.value),
        "contrast_quantity": contrast_quantity,
        contrast_quantity: float(reading.lambda_exc.value),
        "role": role,
        "surrogate_cardinality_status": cardinality_status,
        "surrogate_cardinality_minimum_met": reading.is_confirmatory,
        "n_surrogates_effective": reading.null.admitted,
        "rejected_surrogates": list(reading.null.rejected),
        "negative_control_rank": reading.rank,
        "negative_control_rank_fraction": reading.negative_control_rank_fraction,
        "rank_resolution": reading.rank_resolution,
        "rank_interpretation": "descriptive_only_not_a_p_value",
        "surrogates": surrogate_rows,
    }


def _leakage_quantities(report: LeakageReport) -> dict[str, Any]:
    """Convert the typed report without losing units or diagnostic labels."""
    partial_excess = excess_leakage(
        report.lambda_part,
        report.surrogate_partial,
        rejected_surrogates=report.excluded_surrogates,
    )
    # This adjacent diagnostic uses naive values on both sides of the contrast.
    # It is never substituted for the preregistered partial-regression endpoint.
    naive_excess = excess_leakage(
        report.lambda_orc,
        report.surrogate_naive,
        rejected_surrogates=report.excluded_surrogates,
    )
    partial_summary = _excess_summary(
        partial_excess,
        observed_quantity="lambda_part",
        contrast_quantity="lambda_exc",
        role="primary_preregistered_endpoint",
    )
    naive_summary = _excess_summary(
        naive_excess,
        observed_quantity="lambda_naive_orc_reference",
        contrast_quantity="lambda_naive_negative_control_contrast",
        role="descriptive_companion_only",
    )
    quantities: dict[str, Any] = {
        "table": report.as_table(),
        "lambda_part": float(report.lambda_part.value),
        "lambda_exc": partial_summary["lambda_exc"],
        "lambda_exc_surrogate_mean": partial_summary["surrogate_mean"],
        "lambda_exc_effective_surrogates": partial_summary[
            "n_surrogates_effective"
        ],
        "lambda_exc_surrogate_cardinality_status": partial_summary[
            "surrogate_cardinality_status"
        ],
        "lambda_exc_surrogate_cardinality_minimum_met": partial_summary[
            "surrogate_cardinality_minimum_met"
        ],
        "lambda_exc_inference_status": "not_evaluated",
        "excess_readings": {
            "partial": partial_summary,
            "naive_companion": naive_summary,
        },
        "lambda_orc": float(report.lambda_orc.value),
        "lambda_ref": float(report.lambda_ref.value),
        "rho_dir_orc": float(report.rho_dir_orc.value),
        "rho_dir_ref": float(report.rho_dir_ref.value),
        "phi_leakage_share": float(report.phi_leakage_share.value),
        "energy_c_delta": float(report.energy_c_delta.value),
        "energy_eps_orc": float(report.energy_eps_orc.value),
        "energy_u_orc": (
            None if report.energy_u_orc is None else float(report.energy_u_orc.value)
        ),
        "oracle_residual_energy_ratio": report.oracle_residual_energy_ratio,
        "overall_nuisance_adjustment": float(
            report.overall_nuisance_adjustment.value
        ),
        "nested_shrinkage_effect": float(report.nested_shrinkage_effect.value),
        "lambda_without_shrinkage": float(report.lambda_without_shrinkage.value),
        "target_residual_energy_share": report.target_residual_energy_share,
        "target_vif": report.target_vif,
        "target_residual_share_floor": report.target_residual_share_floor,
        "condition_number": report.condition_number,
        "surrogate_partial": {
            name: float(value.value) for name, value in report.surrogate_partial.items()
        },
        "surrogate_naive": {
            name: float(value.value) for name, value in report.surrogate_naive.items()
        },
        "surrogate_role": "descriptive_negative_controls_not_exchangeable_null",
        "nuisance_coefficients": dict(report.nuisance_coefficients),
        "per_regime_lambda_orc": {
            regime.value: float(value.value)
            for regime, value in report.per_regime_lambda_orc.items()
        },
        "occupancy": {
            regime.value: value for regime, value in report.occupancy.items()
        },
        "excluded_surrogates": list(report.excluded_surrogates),
    }
    if report.three_term is not None:
        quantities["three_term"] = {
            "distribution": report.three_term.distribution,
            "terms": report.three_term.as_rows(),
            "identity_residual": report.three_term.identity_residual,
        }
    else:
        quantities["three_term"] = None
    return quantities


def diagnose(
    ctx: RunContext,
    *,
    task: str,
    family: str,
    defect: str,
    level: int,
    leakage_input: Path,
    allow_engineering: bool = False,
) -> DiagnosticReport:
    """Run the leakage diagnostics over one cell, streaming.

    The raw collection is deliberately not accepted here.  A complete,
    hash-bound leakage input must carry both policy actions, both teacher
    actions, every nuisance column and the descriptive surrogate fields on the
    same frame rows.  The loader rechecks the action identities and artifact
    SHA before any sufficient statistic is accumulated.
    """
    req = ctx.request
    if req.experiment_id is None:
        raise ConfigError("diagnostics run under an experiment, so --experiment is required")
    source = Path(leakage_input)
    if not source.is_file():
        raise FileNotFoundError(f"leakage input {source} does not exist")
    header = load_leakage_input_header(source)
    provenance = header.provenance
    expected = (task, family, defect, level)
    observed = (provenance.task, provenance.family, provenance.defect, provenance.level)
    if observed != expected:
        raise ConfigError(
            f"leakage input identifies {observed}, but diagnose requested {expected}"
        )
    if provenance.release_status != "formal" and not allow_engineering:
        raise DiagnosticsUnavailable(
            "formal diagnose refuses release_status=engineering_only; pass a checkpoint "
            "pair produced by the formal training path. Engineering callers may opt in "
            "only for an explicitly non-paper pilot"
        )
    if provenance.release_status == "formal" and not header.has_source_posterior:
        raise DiagnosticsUnavailable(
            "formal diagnose requires the source posterior used by the registered "
            "three-term decomposition"
        )
    if req.arm_id is not None and req.arm_id != provenance.evaluated_arm:
        raise ConfigError(
            f"diagnose request arm {req.arm_id!r} disagrees with leakage input arm "
            f"{provenance.evaluated_arm!r}"
        )

    accumulator = LeakageAccumulator(
        header.surrogate_names,
        with_reference_arm=True,
        distribution="rho_off",
        excluded_surrogates=provenance.excluded_surrogates,
    )
    posterior = (
        PosteriorCovarianceAccumulator("rho_off")
        if header.has_source_posterior
        else None
    )
    n_episodes = 0
    clipped_frames = 0
    observed_frames = 0
    for episode, chunk in stream_leakage_input(source):
        accumulator.update(chunk)
        if posterior is not None:
            if episode.posterior_w is None:
                raise RuntimeError("validated source posterior disappeared while streaming")
            posterior.update(episode.posterior_w, episode.c_true)
        n_episodes += 1
        observed_frames += episode.n_frames
        clipped_frames += int(np.count_nonzero(episode.c_true_clipped))
    if observed_frames != header.n_frames:
        raise RuntimeError(
            "leakage input changed between header validation and streaming; refusing "
            "to report a mixed artifact"
        )
    leakage = accumulator.finalise(posterior_off=posterior)
    clip_share = clipped_frames / observed_frames
    clip_threshold = ctx.threshold("c_true_clip_annotation_share")
    notes = [
        f"leakage_input_sha256={header.artifact_sha256}",
        f"evaluated_checkpoint_sha256={provenance.evaluated_checkpoint_sha256}",
        f"oracle_checkpoint_sha256={provenance.oracle_checkpoint_sha256}",
        f"evaluation_plan_sha256={provenance.evaluation_plan_sha256}",
        f"source_state_manifest_sha256={provenance.source_state_manifest_sha256}",
        f"source_posterior_artifact_sha256={provenance.source_posterior_artifact_sha256}",
        f"surrogate_artifact_sha256={provenance.surrogate_artifact_sha256}",
        "surrogates are descriptive negative controls, not an exchangeable permutation null",
    ]
    if leakage.three_term is None:
        notes.append(
            "three-term posterior decomposition not reported: this artifact carries the "
            "same-distribution policy/Oracle fields but no source-posterior accumulator"
        )
    if provenance.release_status == "engineering_only":
        notes.append("engineering-only diagnostic; forbidden as a paper result")
    return DiagnosticReport(
        cell=Cell(task=task, family=family, defect_code=defect),
        level=level,
        n_frames=observed_frames,
        n_episodes=n_episodes,
        quantities=_leakage_quantities(leakage)
        | {
            "c_true_clip_saturation_share": clip_share,
            "release_status": provenance.release_status,
            "evaluated_arm": provenance.evaluated_arm,
            "oracle_arm": provenance.oracle_arm,
            "provenance": {
                "leakage_input_sha256": header.artifact_sha256,
                "evaluated_checkpoint_sha256": provenance.evaluated_checkpoint_sha256,
                "oracle_checkpoint_sha256": provenance.oracle_checkpoint_sha256,
                "evaluated_task_index": provenance.evaluated_task_index,
                "evaluated_embodiment_index": provenance.evaluated_embodiment_index,
                "oracle_task_index": provenance.oracle_task_index,
                "oracle_embodiment_index": provenance.oracle_embodiment_index,
                "evaluation_plan_sha256": provenance.evaluation_plan_sha256,
                "source_state_manifest_sha256": provenance.source_state_manifest_sha256,
                "source_posterior_artifact_sha256": (
                    provenance.source_posterior_artifact_sha256
                ),
                "surrogate_artifact_sha256": provenance.surrogate_artifact_sha256,
            },
        },
        clip_annotation=clip_share > clip_threshold,
        notes=tuple(notes),
    )


def diagnose_to_json(report: DiagnosticReport, path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "cell": str(report.cell),
                "level": report.level,
                "n_frames": report.n_frames,
                "n_episodes": report.n_episodes,
                "quantities": report.quantities,
                "clip_annotation": report.clip_annotation,
                "notes": list(report.notes),
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
        fh.write("\n")
