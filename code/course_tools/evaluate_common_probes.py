#!/usr/bin/env python3
"""Evaluate each policy once on fixed observations, then project source fields.

Lambda=<error,c>/<c,c>; Phi=<error,c>^2/(<error,error><c,c>).
The error always subtracts the anchor's saved normal-teacher action. Source
directions are not assumed orthogonal and their Phi values are not summed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np

from course_tools.evaluate_three_source import (
    atomic_json, field_metrics, hash_json, sha256, verify_contrast,
)
from course_tools.build_compensation_probes import field_geometry

SCHEMA = "dl_course.common_probe_evaluation/1"
CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
COMPONENTS = {
    "translation_command_m": (slice(0, 3), .05, 2.5e-15),
    "rotation_command_rad": (slice(3, 6), .5, 2.5e-13),
}


def inspect_probe(path, report, protocol):
    """Read actual pixels, histories and fields; validate their common anchor."""
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema") != "dl_course.compensation_probes/2":
            raise ValueError(f"unsupported common-probe schema: {path}")
        if handle.attrs["protocol_sha256"] != protocol["protocol_sha256"]:
            raise ValueError("probe data and source protocol differ")
        metadata = json.loads(handle.attrs["anchor_metadata_json"])
        if metadata["split"] != "test":
            raise ValueError("model evaluation requires held-out test probes")
        n = len(handle["actions_clean"])
        k = int(handle.attrs["history_frames"])
        history = handle["history_indices"][:]
        expected = np.maximum(np.arange(n)[:, None] - np.arange(k - 1, -1, -1), 0)
        if not np.array_equal(history, expected):
            raise ValueError("probe history is not the saved causal left-padded history")
        arrays = {"history": history, "clean": handle["actions_clean"][:],
                  "proprio": handle["obs/robot0_proprio-state"][:],
                  "phase": handle["reference_trace/phase"][:],
                  "control_step": handle["control_step"][:], "images": {}, "fields": {}}
        if arrays["clean"].shape != (n, 7) or arrays["proprio"].shape != (n, 70):
            raise ValueError("invalid normal action/proprioception shape")
        if not np.isfinite(arrays["clean"]).all() or not np.isfinite(arrays["proprio"]).all():
            raise ValueError("nonfinite common inputs")
        if not np.array_equal(arrays["control_step"], np.arange(n)):
            raise ValueError("control steps are not the full ordered episode")
        for camera in CAMERAS:
            image = handle[f"obs/{camera}"][:]
            if image.dtype != np.uint8 or image.shape != (n, 84, 84, 3):
                raise ValueError(f"invalid camera {camera}")
            arrays["images"][camera] = image
        labels = sorted(handle["fields"])
        expected_labels = sorted(f"{s}-L{metadata['level']}" for s in ("delay", "backlash", "droop"))
        if labels != expected_labels:
            raise ValueError("the probe does not contain all three source fields")
        differences = {}
        for label in labels:
            group = handle[f"fields/{label}"]
            field = {key: group[key][:] for key in (
                "c_source", "actions_source", "actions_normal_source", "saturated")}
            if any(value.shape != (n, 7) or not np.isfinite(value).all()
                   for key, value in field.items() if key != "saturated"):
                raise ValueError("invalid source action fields")
            if not np.allclose(field["actions_source"] - field["actions_normal_source"],
                               field["c_source"], atol=1e-10, rtol=0):
                raise ValueError("source field is not its recorded teacher contrast")
            if not np.allclose(field["c_source"][:, 6], 0, atol=1e-10, rtol=0):
                raise ValueError("unexpected gripper compensation")
            field["definition"] = json.loads(group.attrs["definition_json"])
            arrays["fields"][label] = field
            gap = field["actions_normal_source"] - arrays["clean"]
            differences[label] = {"rms": float(np.sqrt(np.mean(gap[:, :6] ** 2))),
                                  "max_abs": float(np.abs(gap[:, :6]).max())}
        geometry = field_geometry({name: values["c_source"] for name, values in arrays["fields"].items()})
        for component in ("translation", "rotation"):
            old = report["field_geometry"][component]
            new = geometry[component]
            if old["rank"] != new["rank"] or not np.allclose(old["normalized_gram"], new["normalized_gram"]):
                raise ValueError("stored source Gram/rank differs from actual fields")
        info = {"anchor": metadata, "frames": n, "history": k,
                "fields": labels, "field_geometry": geometry,
                "normal_source_minus_fixed_anchor_reference": differences,
                "input_range": {"proprio_min": float(arrays["proprio"].min()),
                                "proprio_max": float(arrays["proprio"].max())}}
        return info, arrays


def load_probes(directories):
    probes, unique = [], set()
    for directory in directories:
        directory = directory.resolve()
        protocol = json.loads((directory / "protocol.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        if protocol["split"] != "test" or summary.get("status") != "complete":
            raise ValueError("a completed held-out probe collection is required")
        if summary["anchors"] != summary["fully_defined_anchors"]:
            raise ValueError("some requested source fields are undefined")
        for reported in summary["reports"]:
            report_path = directory / Path(reported).name
            report = json.loads(report_path.read_text())
            path = directory / Path(report["path"]).name
            data_sha = sha256(path)
            if data_sha != report["sha256"] or report["protocol_sha256"] != protocol["protocol_sha256"]:
                raise ValueError("probe source checksums differ")
            info, _ = inspect_probe(path, report, protocol)
            identity = (info["anchor"]["seed_env"], info["anchor"]["level"], info["anchor"]["source"])
            if identity in unique:
                raise ValueError("duplicate common anchor")
            unique.add(identity)
            probes.append({"path": path, "sha256": data_sha, "report": report,
                           "protocol": protocol, "info": info, "stem": path.stem})
        if len(summary["reports"]) != summary["anchors"]:
            raise ValueError("probe summary does not list all anchors")
    return sorted(probes, key=lambda p: (p["info"]["anchor"]["level"], p["info"]["anchor"]["seed_env"]))


def observation_batch(arrays, start, stop):
    import torch
    from portal_trace.policies.interface import Observation

    ids = arrays["history"][start:stop]
    images = {name: torch.from_numpy(values[ids].transpose(0, 1, 4, 2, 3).copy()).float() / 255
              for name, values in arrays["images"].items()}
    return Observation(images=images,
        proprio=torch.from_numpy(arrays["proprio"][ids].astype(np.float32, copy=True)),
        task_index=torch.zeros(stop - start, dtype=torch.long),
        embodiment_index=None, partition_index=None)


def prediction_cache(policy, checkpoint_path, probe, arrays, output, batch_size):
    """One full-episode model inference reused across sources and contrasts."""
    checkpoint_sha = policy.metadata["checkpoint_sha256"]
    directory = output / "prediction-cache" / checkpoint_sha
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (probe["stem"] + ".hdf5")
    report_path = path.with_suffix(".json")
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        binding = {"checkpoint_sha256": checkpoint_sha, "probe_sha256": probe["sha256"],
                   "history": policy.history, "prediction": "all eight canonical actions; source-independent forward",
                   "evaluator_sha256": sha256(__file__),
                   "loader_sha256": sha256(ROOT / "course_tools/train_three_source_pool.py"),
                   "normalizer_sha256": sha256(ROOT / "course_tools/three_source_data.py")}
        if path.exists() or report_path.exists():
            if not path.exists() or not report_path.exists():
                raise ValueError("incomplete prediction cache")
            record = json.loads(report_path.read_text())
            if record["binding"] != binding or sha256(path) != record["sha256"]:
                raise ValueError("prediction cache binding/checksum changed")
            with h5py.File(path, "r") as handle:
                return handle["action_chunks_unclipped"][:], record
        if policy.history != arrays["history"].shape[1]:
            raise ValueError("checkpoint and common probe history lengths differ")
        n = len(arrays["clean"])
        chunks = []
        started = time.monotonic()
        for start in range(0, n, batch_size):
            obs = observation_batch(arrays, start, min(n, start + batch_size))
            chunks.append(policy.predict(obs, clip=False))
        prediction = np.concatenate(chunks)
        if prediction.shape != (n, 8, 7) or not np.isfinite(prediction).all():
            raise ValueError("model returned invalid canonical action chunks")
        temporary = path.with_suffix(f".{os.getpid()}.partial.hdf5")
        with h5py.File(temporary, "x") as handle:
            handle.attrs["binding_json"] = json.dumps(binding, sort_keys=True)
            handle.create_dataset("action_chunks_unclipped", data=prediction, compression="gzip")
        os.link(temporary, path)
        temporary.unlink()
        record = {"schema": SCHEMA, "binding": binding, "path": str(path), "sha256": sha256(path),
                  "frames": n, "checkpoint": str(checkpoint_path.resolve()),
                  "forward_frames": n, "seconds": time.monotonic() - started}
        atomic_json(report_path, record)
        return prediction, record


def project_fields(error, fields, masks, *, intervention=False):
    result = {}
    for component, (sl, scale, floor) in COMPONENTS.items():
        result[component] = {}
        for source, values in fields.items():
            c = values["c_source"][:, sl] * scale
            result[component][source] = {}
            for label, selected in masks.items():
                row = field_metrics(error[selected, sl] * scale, c[selected], energy_floor=floor)
                if intervention:
                    row["intervention_projection_coefficient"] = row.pop("lambda")
                    row["intervention_direction_energy_share"] = row.pop("phi")
                result[component][source][label] = row
    return result


def aggregate_scene_readings(reports, key, metric):
    buckets = defaultdict(list)
    for report in reports:
        for component, sources in report[key].items():
            for source, strata in sources.items():
                buckets[(report["level"], component, source)].append(strata["all"])
    rows = []
    for (level, component, source), values in sorted(buckets.items()):
        defined = [value[metric] for value in values if value[metric] is not None]
        rows.append({"level": level, "component": component, "source": source,
                     "metric": metric, "defined_scenes": len(defined), "scenes": len(values),
                     "scene_mean": float(np.mean(defined)) if len(defined) == len(values) else None,
                     "defined_scene_values": defined})
    return rows


def evaluate(args, probes):
    import torch
    from course_tools.train_three_source_pool import load_three_source_checkpoint

    torch.set_num_threads(args.threads)
    primary = load_three_source_checkpoint(args.checkpoint, device=args.device)
    contrast = load_three_source_checkpoint(args.contrast_checkpoint, device=args.device) if args.contrast_checkpoint else None
    if contrast:
        verify_contrast(primary, contrast)
        shared_primary = {k: v for k, v in primary.metadata["run_identity"].items() if k != "config"}
        shared_contrast = {k: v for k, v in contrast.metadata["run_identity"].items() if k != "config"}
        if shared_primary != shared_contrast:
            raise ValueError("label contrast changed data, normalization or training implementation")
    for probe in probes:
        if probe["protocol"]["manifest_sha256"] != primary.metadata["run_identity"]["manifest_sha256"]:
            raise ValueError("probe and model have different bound dataset manifests")
    protocol = {"schema": SCHEMA, "checkpoint_sha256": primary.metadata["checkpoint_sha256"],
                "checkpoint_step": primary.metadata["step"], "config": primary.metadata["config"],
                "contrast_sha256": contrast.metadata["checkpoint_sha256"] if contrast else None,
                "probe_hashes": {p["stem"]: p["sha256"] for p in probes},
                "runner_sha256": sha256(__file__), "device": args.device,
                "error_reference": "fixed saved anchor actions_clean",
                "lambda": "sum(error*c)/sum(c*c)",
                "phi": "sum(error*c)^2/(sum(error^2)*sum(c^2))",
                "direction_interpretation": "separate directional readings; nonorthogonal projections are not added",
                "statistical_unit": "paired scene; frames are not independent experimental replicates"}
    protocol_sha = hash_json(protocol)
    result_dir = args.output / "evaluations" / protocol_sha
    result_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(result_dir / "protocol.json", protocol)
    reports = []
    for probe in probes:
        report_path = result_dir / (probe["stem"] + ".json")
        data_path = result_dir / (probe["stem"] + ".hdf5")
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report["protocol_sha256"] != protocol_sha or sha256(data_path) != report["sha256"]:
                raise ValueError("existing evaluation differs")
            reports.append(report)
            continue
        info, arrays = inspect_probe(probe["path"], probe["report"], probe["protocol"])
        chunks, pred_record = prediction_cache(primary, args.checkpoint, probe, arrays, args.output, args.batch_size)
        prediction = chunks[:, 0]
        error = prediction - arrays["clean"]
        clipped_error = np.clip(prediction, -1, 1) - arrays["clean"]
        masks = {"all": np.ones(len(error), bool), **{f"phase_{int(k)}": arrays["phase"] == k for k in np.unique(arrays["phase"])}}
        report = {"schema": SCHEMA, "protocol_sha256": protocol_sha,
                  "seed_env": info["anchor"]["seed_env"], "level": info["anchor"]["level"],
                  "probe_sha256": probe["sha256"], "probe": str(probe["path"]),
                  "frames": len(error), "prediction_cache": pred_record,
                  "field_geometry": info["field_geometry"],
                  "normal_source_minus_fixed_anchor_reference": info["normal_source_minus_fixed_anchor_reference"],
                  "unclipped": project_fields(error, arrays["fields"], masks),
                  "clipped": project_fields(clipped_error, arrays["fields"], masks),
                  "saturation_fraction": float((np.abs(prediction[:, :6]) >= 1).mean())}
        alternate = None
        if contrast:
            alt_chunks, contrast_record = prediction_cache(contrast, args.contrast_checkpoint, probe, arrays, args.output, args.batch_size)
            alternate = alt_chunks[:, 0]
            report["contrast_prediction_cache"] = contrast_record
            report["intervention_label_mode"] = contrast.metadata["config"]["label_mode"]
            report["intervention_matched_training_verified"] = True
            report["intervention_unclipped"] = project_fields(prediction - alternate, arrays["fields"], masks, intervention=True)
            report["intervention_clipped"] = project_fields(np.clip(prediction, -1, 1) - np.clip(alternate, -1, 1),
                                                          arrays["fields"], masks, intervention=True)
        temporary = data_path.with_suffix(f".{os.getpid()}.partial.hdf5")
        with h5py.File(temporary, "x") as handle:
            handle.attrs["protocol_sha256"] = protocol_sha
            handle.attrs["probe_sha256"] = probe["sha256"]
            handle.attrs["reference"] = "fixed anchor actions_clean"
            for name, value in {"prediction_unclipped": prediction, "prediction_clipped": np.clip(prediction, -1, 1),
                                "anchor_actions_clean": arrays["clean"], "error_unclipped": error,
                                "error_clipped": clipped_error, "history_indices": arrays["history"],
                                "control_step": arrays["control_step"], "phase": arrays["phase"]}.items():
                handle.create_dataset(name, data=value, compression="gzip")
            for source, values in arrays["fields"].items():
                group = handle.create_group("fields/" + source)
                group.attrs["definition_json"] = json.dumps(values["definition"], sort_keys=True)
                for name, value in values.items():
                    if name != "definition":
                        group.create_dataset(name, data=value, compression="gzip")
            if alternate is not None:
                for name, value in {"contrast_prediction_unclipped": alternate,
                                    "contrast_prediction_clipped": np.clip(alternate, -1, 1),
                                    "intervention_delta_unclipped": prediction - alternate,
                                    "intervention_delta_clipped": np.clip(prediction, -1, 1) - np.clip(alternate, -1, 1)}.items():
                    handle.create_dataset(name, data=value, compression="gzip")
        os.link(temporary, data_path)
        temporary.unlink()
        report.update(path=str(data_path), sha256=sha256(data_path))
        atomic_json(report_path, report)
        reports.append(report)
        atomic_json(result_dir / "progress.json", {"status": "running", "anchors": len(reports), "planned": len(probes)})
    summary = {"schema": SCHEMA, "status": "complete", "protocol_sha256": protocol_sha,
               "anchors": len(reports), "unique_scenes": len({r["seed_env"] for r in reports}),
               "scene_statistics": [row for variant in ("unclipped", "clipped") for metric in ("lambda", "phi")
                                    for row in [{**x, "variant": variant} for x in aggregate_scene_readings(reports, variant, metric)]],
               "reports": [str(result_dir / (probe["stem"] + ".json")) for probe in probes]}
    if contrast:
        summary["intervention_statistics"] = [row for variant in ("intervention_unclipped", "intervention_clipped")
            for metric in ("intervention_projection_coefficient", "intervention_direction_energy_share")
            for row in [{**x, "variant": variant} for x in aggregate_scene_readings(reports, variant, metric)]]
    atomic_json(result_dir / "summary.json", summary)
    atomic_json(result_dir / "progress.json", summary)
    return result_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--contrast-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.batch_size < 1:
        parser.error("thread and batch counts must be positive")
    if args.audit_only and (args.checkpoint or args.contrast_checkpoint):
        parser.error("input audit does not load checkpoints")
    if not args.audit_only and args.checkpoint is None:
        parser.error("model evaluation requires a real checkpoint")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    probes = load_probes(args.probe_dir)
    if args.audit_only:
        result = {"schema": SCHEMA, "status": "inputs_verified", "model_inference": False,
                  "anchors": len(probes), "frames": sum(p["info"]["frames"] for p in probes),
                  "runner_sha256": sha256(__file__),
                  "probes": [{"path": str(p["path"]), "sha256": p["sha256"], **p["info"]} for p in probes]}
        atomic_json(args.output / "input-audit.json", result)
        print(json.dumps({"status": "inputs_verified", "anchors": len(probes), "model_inference": False}))
    else:
        destination = evaluate(args, probes)
        print(json.dumps({"status": "complete", "output": str(destination), "anchors": len(probes)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
