"""Persistent compatibility gates for collected PORTAL-Trace datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "DATASET_IDENTITY_SCHEMA_VERSION",
    "PAIRED_TWIN_SCHEMA_VERSION",
    "PAIRED_TWIN_SEMANTICS",
    "DEFECT_STATE_SCHEMA_VERSION",
    "DEFECT_STATE_SEMANTICS",
    "ANALYSIS_TRACE_SCHEMA_VERSION",
    "ANALYSIS_TRACE_FIELDS",
    "SURROGATE_CONTEXT_SCHEMA_VERSION",
    "SURROGATE_CONTEXT_FIELDS",
    "DatasetIdentityError",
    "DefectStateSemanticsError",
    "stamp_dataset_identity",
    "require_dataset_identity",
    "require_delta_value_contract",
    "stamp_defect_state_contract",
    "require_current_defect_state_contract",
    "stamp_analysis_trace_contract",
    "require_analysis_trace_contract",
    "require_surrogate_context_contract",
    "stamp_paired_twin_contract",
    "require_paired_twin_contract",
]


# Version 1 denotes the legacy layout whose D4hi engagement bits came from
# wrapper reset state; historical files are generally unversioned. Version 2
# samples D4hi engagement from the live shadow equality's efc_pos at each
# recorded pre-step boundary.
DEFECT_STATE_SCHEMA_VERSION = 2
DEFECT_STATE_SEMANTICS = "d4hi_shadow_equality_efc_pos_pre_step_v1"
DATASET_IDENTITY_SCHEMA_VERSION = 2
PAIRED_TWIN_SCHEMA_VERSION = 2
PAIRED_TWIN_SEMANTICS = "same_defective_state_clean_query_actions_v1"
ANALYSIS_TRACE_SCHEMA_VERSION = 2
SURROGATE_CONTEXT_SCHEMA_VERSION = 1
SURROGATE_CONTEXT_FIELDS = (
    "a_dot_nominal",
    "qpos_index",
    "qvel_index",
    "kp",
    "output_max",
    "injected_joints",
)
ANALYSIS_TRACE_FIELDS = (
    "c_true_clipped",
    "u_inf",
    "u_inf_clean",
    "phase",
    "contact",
    "eef_pos_cmd",
    "eef_pos_ach",
    "tracking_error",
    "joint_torque",
)

# The widths which are fixed by the analysis contract.  ``joint_torque`` is
# deliberately left dynamic: the registered UR5e has six arm joints while the
# other main families have seven.  The frame axis is checked against the raw
# episode anchor (``states``, ``actions`` or ``c_true``), not against a caller's
# requested horizon, so this validator cannot turn a truncation into a valid
# trace by assumption.
_ANALYSIS_TRACE_WIDTHS = {
    "c_true_clipped": None,
    "u_inf": 6,
    "u_inf_clean": 6,
    "phase": None,
    "contact": None,
    "eef_pos_cmd": 3,
    "eef_pos_ach": 3,
    "tracking_error": 6,
    "joint_torque": None,
}


class DatasetIdentityError(ValueError):
    """A dataset's stored cell identity disagrees with the requested cell."""


class DefectStateSemanticsError(ValueError):
    """A D4hi dataset predates the current defect-state semantics."""


def require_surrogate_context_contract(path: Path) -> None:
    """Require the replay material needed to build matched surrogates.

    The nominal-action derivative has one fewer row than the control trace by
    construction.  Downstream leakage analysis therefore uses the first
    ``T - 1`` frames of every admitted defective episode.
    """
    import h5py
    import numpy as np

    path = Path(path)
    with h5py.File(path, "r") as root:
        try:
            version = int(root.attrs.get("surrogate_context_schema_version"))
        except (TypeError, ValueError):
            raise DatasetIdentityError(
                f"{path}: missing surrogate_context_schema_version; re-collect the "
                "defective dataset with the current collector"
            ) from None
        if version != SURROGATE_CONTEXT_SCHEMA_VERSION:
            raise DatasetIdentityError(
                f"{path}: surrogate_context_schema_version={version}, expected "
                f"{SURROGATE_CONTEXT_SCHEMA_VERSION}"
            )
        if "data" not in root or not root["data"]:
            raise DatasetIdentityError(f"{path}: surrogate-context dataset has no episodes")
        for demo_name in root["data"]:
            demo = root["data"][demo_name]
            location = f"{path}:data/{demo_name}"
            if "surrogate_context" not in demo:
                raise DatasetIdentityError(f"{location}: surrogate_context is missing")
            context = demo["surrogate_context"]
            try:
                context_version = int(context.attrs.get("schema_version"))
                site_id = int(context.attrs.get("site_id"))
                action_dim = int(context.attrs.get("action_dim"))
            except (TypeError, ValueError):
                raise DatasetIdentityError(
                    f"{location}: surrogate_context attributes are incomplete"
                ) from None
            if context_version != SURROGATE_CONTEXT_SCHEMA_VERSION:
                raise DatasetIdentityError(
                    f"{location}: surrogate context schema is {context_version}, "
                    f"expected {SURROGATE_CONTEXT_SCHEMA_VERSION}"
                )
            if set(context) != set(SURROGATE_CONTEXT_FIELDS):
                raise DatasetIdentityError(
                    f"{location}: surrogate context fields are incomplete or unexpected"
                )
            if "states" not in demo or "actions" not in demo or "c_true" not in demo:
                raise DatasetIdentityError(
                    f"{location}: surrogate replay requires states, actions and c_true"
                )
            horizon = len(demo["actions"])
            if horizon < 2 or len(demo["states"]) != horizon:
                raise DatasetIdentityError(
                    f"{location}: surrogate replay needs at least two aligned states"
                )
            if action_dim < 6 or demo["actions"].shape != (horizon, action_dim):
                raise DatasetIdentityError(
                    f"{location}: action_dim disagrees with the stored actions"
                )
            if demo["c_true"].shape != (horizon, action_dim):
                raise DatasetIdentityError(
                    f"{location}: c_true disagrees with the stored action shape"
                )
            a_dot = np.asarray(context["a_dot_nominal"], dtype=np.float64)
            qpos_index = np.asarray(context["qpos_index"], dtype=np.int64)
            qvel_index = np.asarray(context["qvel_index"], dtype=np.int64)
            kp = np.asarray(context["kp"], dtype=np.float64)
            output_max = np.asarray(context["output_max"], dtype=np.float64)
            injected = np.asarray(context["injected_joints"], dtype=np.int64)
            if a_dot.shape != (horizon - 1, 6) or not np.isfinite(a_dot).all():
                raise DatasetIdentityError(
                    f"{location}: a_dot_nominal must have shape ({horizon - 1}, 6)"
                )
            if (
                qpos_index.ndim != 1
                or qvel_index.ndim != 1
                or len(qpos_index) != len(qvel_index)
                or len(qvel_index) < 1
                or len(set(qpos_index.tolist())) != len(qpos_index)
                or len(set(qvel_index.tolist())) != len(qvel_index)
                or np.any(qpos_index < 0)
                or np.any(qvel_index < 0)
            ):
                raise DatasetIdentityError(
                    f"{location}: qpos/qvel indices must be unique aligned vectors"
                )
            if (
                kp.shape != (6,)
                or output_max.shape != (6,)
                or not np.isfinite(kp).all()
                or not np.isfinite(output_max).all()
                or np.any(output_max <= 0.0)
            ):
                raise DatasetIdentityError(
                    f"{location}: kp/output_max must be finite six-vectors with "
                    "positive output scales"
                )
            if (
                injected.ndim != 1
                or len(injected) < 1
                or len(set(injected.tolist())) != len(injected)
                or np.any(injected < 0)
                or np.any(injected >= len(qvel_index))
            ):
                raise DatasetIdentityError(
                    f"{location}: injected_joints is outside the arm joint vector"
                )
            if site_id < 0:
                raise DatasetIdentityError(f"{location}: site_id must be non-negative")
            model_file = demo.attrs.get("model_file")
            if isinstance(model_file, bytes):
                model_file = model_file.decode("utf-8", errors="strict")
            if not isinstance(model_file, str) or "<mujoco" not in model_file:
                raise DatasetIdentityError(
                    f"{location}: model_file is unavailable for MuJoCo replay"
                )


_PAIRED_ROOT_TEXT_FIELDS = (
    "paired_twin_semantics",
    "paired_file_role",
    "paired_binding_id",
    "paired_episode_ids_json",
    "paired_transaction_marker",
    "collection_source_commit",
    "collection_runtime_json",
)
_PAIRED_EPISODE_TEXT_FIELDS = (
    "pair_id",
    "pair_role",
    "paired_binding_id",
    "initial_physics_sha256",
    "initial_actuation_sha256",
    "initial_frame_sha256",
    "reference_spec_sha256",
    "reference_trajectory_sha256",
    "controller_identity_defective_sha256",
    "controller_identity_clean_sha256",
    "teacher_identity_defective_sha256",
    "teacher_identity_clean_sha256",
    "collection_source_commit",
    "collection_runtime_json",
)
_SHA256_EPISODE_FIELDS = tuple(
    name for name in _PAIRED_EPISODE_TEXT_FIELDS if name.endswith("_sha256")
)


def stamp_paired_twin_contract(
    attrs: Any,
    *,
    role: str,
    binding_id: str,
    pair_ids: tuple[str, ...],
    transaction_marker: str,
    source_commit: str,
    runtime_json: str,
) -> None:
    """Stamp one raw file with its immutable paired-state collection identity."""
    if role not in {"defective", "clean"}:
        raise ValueError(f"paired file role must be defective or clean, got {role!r}")
    if len(binding_id) != 64 or any(c not in "0123456789abcdef" for c in binding_id):
        raise ValueError("paired binding id must be a lower-case SHA-256")
    if not pair_ids or len(set(pair_ids)) != len(pair_ids):
        raise ValueError("paired file must contain unique non-empty pair ids")
    attrs["paired_twin_schema_version"] = PAIRED_TWIN_SCHEMA_VERSION
    attrs["paired_twin_semantics"] = PAIRED_TWIN_SEMANTICS
    attrs["paired_file_role"] = role
    attrs["paired_binding_id"] = binding_id
    attrs["paired_episode_ids_json"] = json.dumps(pair_ids, separators=(",", ":"))
    attrs["paired_transaction_marker"] = transaction_marker
    attrs["collection_source_commit"] = source_commit
    attrs["collection_runtime_json"] = runtime_json


def require_paired_twin_contract(path: Path, *, verify_pair_marker: bool = False) -> None:
    """Fail closed unless every episode proves its same-state twin provenance.

    Historical pilot files are intentionally not inferred into this schema. A
    missing version, action companion, identity, or raw-pair marker is a hard
    re-collection error rather than an invitation to reconstruct provenance.
    """
    import hashlib

    import h5py
    import numpy as np

    path = Path(path)
    with h5py.File(path, "r") as fh:
        try:
            version = int(fh.attrs.get("paired_twin_schema_version"))
        except (TypeError, ValueError):
            raise DatasetIdentityError(
                f"{path}: missing paired_twin_schema_version; historical pilot data "
                "is not automatically admissible, re-collect with the current collector"
            ) from None
        if version != PAIRED_TWIN_SCHEMA_VERSION:
            raise DatasetIdentityError(
                f"{path}: paired_twin_schema_version={version}, expected "
                f"{PAIRED_TWIN_SCHEMA_VERSION}; re-collect this dataset"
            )
        root = {
            name: _required_contract_text(fh.attrs, name, location=str(path))
            for name in _PAIRED_ROOT_TEXT_FIELDS
        }
        if root["paired_twin_semantics"] != PAIRED_TWIN_SEMANTICS:
            raise DatasetIdentityError(f"{path}: incompatible paired twin semantics")
        if root["paired_file_role"] not in {"defective", "clean"}:
            raise DatasetIdentityError(f"{path}: invalid paired_file_role")
        _require_sha256(root["paired_binding_id"], path, "paired_binding_id")
        try:
            expected_pair_ids = json.loads(root["paired_episode_ids_json"])
            runtime = json.loads(root["collection_runtime_json"])
        except json.JSONDecodeError as exc:
            raise DatasetIdentityError(f"{path}: invalid paired provenance JSON") from exc
        if not isinstance(expected_pair_ids, list) or not expected_pair_ids:
            raise DatasetIdentityError(f"{path}: paired_episode_ids_json must be non-empty")
        if not isinstance(runtime, dict) or not runtime:
            raise DatasetIdentityError(f"{path}: collection runtime must be a JSON object")
        if "data" not in fh or not fh["data"]:
            raise DatasetIdentityError(f"{path}: paired dataset has no demonstrations")
        actual_pair_ids: list[str] = []
        names = sorted(fh["data"], key=lambda name: int(name.split("_")[-1]))
        for name in names:
            demo = fh["data"][name]
            location = f"{path}:data/{name}"
            try:
                demo_version = int(demo.attrs.get("paired_twin_schema_version"))
            except (TypeError, ValueError):
                raise DatasetIdentityError(
                    f"{location}: missing paired_twin_schema_version"
                ) from None
            if demo_version != PAIRED_TWIN_SCHEMA_VERSION:
                raise DatasetIdentityError(
                    f"{location}: paired_twin_schema_version={demo_version}, expected "
                    f"{PAIRED_TWIN_SCHEMA_VERSION}"
                )
            fields = {
                field: _required_contract_text(demo.attrs, field, location=location)
                for field in _PAIRED_EPISODE_TEXT_FIELDS
            }
            if fields["pair_role"] != root["paired_file_role"]:
                raise DatasetIdentityError(f"{location}: pair role disagrees with file role")
            if fields["paired_binding_id"] != root["paired_binding_id"]:
                raise DatasetIdentityError(f"{location}: paired binding id disagrees with root")
            if fields["collection_source_commit"] != root["collection_source_commit"]:
                raise DatasetIdentityError(f"{location}: source commit disagrees with root")
            if fields["collection_runtime_json"] != root["collection_runtime_json"]:
                raise DatasetIdentityError(f"{location}: runtime fingerprint disagrees with root")
            _require_sha256(fields["pair_id"], path, f"data/{name}/pair_id")
            for field in _SHA256_EPISODE_FIELDS:
                _require_sha256(fields[field], path, f"data/{name}/{field}")
            required = ("actions", "actions_defective", "actions_clean", "c_true")
            missing = [field for field in required if field not in demo]
            if missing:
                raise DatasetIdentityError(
                    f"{location}: missing paired action datasets {missing}; re-collect"
                )
            actions = np.asarray(demo["actions"], dtype=np.float32)
            defective = np.asarray(demo["actions_defective"], dtype=np.float32)
            clean = np.asarray(demo["actions_clean"], dtype=np.float32)
            truth = np.asarray(demo["c_true"], dtype=np.float32)
            if (
                actions.ndim != 2
                or actions.shape != defective.shape
                or actions.shape != clean.shape
            ):
                raise DatasetIdentityError(f"{location}: paired actions have incompatible shapes")
            if truth.shape != actions.shape:
                raise DatasetIdentityError(f"{location}: c_true shape disagrees with actions")
            if not all(np.isfinite(values).all() for values in (actions, defective, clean, truth)):
                raise DatasetIdentityError(f"{location}: paired actions contain NaN or infinity")
            if not np.array_equal(actions, defective):
                raise DatasetIdentityError(f"{location}: actions is not actions_defective")
            if not np.array_equal(defective - clean, truth):
                raise DatasetIdentityError(
                    f"{location}: actions_defective - actions_clean does not equal c_true"
                )
            actual_pair_ids.append(fields["pair_id"])
        if actual_pair_ids != expected_pair_ids:
            raise DatasetIdentityError(
                f"{path}: root pair-id sequence disagrees with demonstration order"
            )
        marker_path = Path(root["paired_transaction_marker"])
        binding_id = root["paired_binding_id"]

    if not verify_pair_marker:
        return
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetIdentityError(
            f"{path}: paired transaction marker {marker_path} is absent or invalid"
        ) from exc
    if marker.get("schema_version") != PAIRED_TWIN_SCHEMA_VERSION:
        raise DatasetIdentityError(f"{marker_path}: incompatible pair marker schema")
    if marker.get("paired_binding_id") != binding_id:
        raise DatasetIdentityError(f"{marker_path}: marker binding id disagrees with dataset")
    raw = marker.get("raw")
    if not isinstance(raw, dict) or set(raw) != {"defective", "clean"}:
        raise DatasetIdentityError(f"{marker_path}: marker must bind both raw files")
    for role in ("defective", "clean"):
        record = raw[role]
        raw_path = Path(record.get("path", "")) if isinstance(record, dict) else Path()
        expected_sha = record.get("sha256") if isinstance(record, dict) else None
        _require_sha256(expected_sha, marker_path, f"raw.{role}.sha256")
        if not raw_path.is_file() or _file_sha256(raw_path) != expected_sha:
            raise DatasetIdentityError(
                f"{marker_path}: {role} raw file is missing or its SHA-256 changed"
            )
    binding_payload = {
        "paired_binding_id": binding_id,
        "pair_ids": marker.get("pair_ids"),
        "defective_sha256": raw["defective"]["sha256"],
        "clean_sha256": raw["clean"]["sha256"],
    }
    encoded = json.dumps(binding_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if marker.get("raw_pair_sha256") != hashlib.sha256(encoded).hexdigest():
        raise DatasetIdentityError(f"{marker_path}: raw pair binding hash is invalid")


def _required_contract_text(attrs: Any, name: str, *, location: str) -> str:
    value = attrs.get(name)
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DatasetIdentityError(f"{location}: {name} is not UTF-8") from exc
    if not isinstance(value, str) or not value:
        raise DatasetIdentityError(f"{location}: {name} must be non-empty text")
    return value


def _require_sha256(value: Any, path: Path, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise DatasetIdentityError(f"{path}: {name} must be a lower-case SHA-256")


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stamp_dataset_identity(
    attrs: Any,
    *,
    task: str,
    family: str,
    defect: str,
    level: int,
    clean_of: str | None = None,
) -> None:
    """Persist the complete cell identity on an HDF5 attribute manager."""
    attrs["dataset_identity_schema_version"] = DATASET_IDENTITY_SCHEMA_VERSION
    attrs["dataset_task"] = task
    attrs["dataset_family"] = family
    attrs["dataset_defect"] = defect
    attrs["dataset_level"] = level
    attrs["dataset_clean_of"] = clean_of or ""


def require_dataset_identity(
    path: Path,
    *,
    task: str,
    family: str,
    defect: str,
    level: int,
    clean_of: str | None = None,
    min_demos: int | None = None,
) -> int:
    """Verify a raw or materialised file against its requested cell identity."""
    import h5py

    expected = {
        "dataset_identity_schema_version": DATASET_IDENTITY_SCHEMA_VERSION,
        "dataset_task": task,
        "dataset_family": family,
        "dataset_defect": defect,
        "dataset_level": level,
        "dataset_clean_of": clean_of or "",
    }
    with h5py.File(path, "r") as fh:
        if "data" not in fh:
            raise DatasetIdentityError(f"{path}: dataset has no data group")
        demos = list(fh["data"].values())
        if not demos:
            raise DatasetIdentityError(f"{path}: dataset contains no demonstrations")
        locations = [(str(path), fh.attrs)]
        locations.extend(
            (f"{path}:data/demo_{index}", demo.attrs)
            for index, demo in enumerate(demos)
        )
        for location, attrs in locations:
            for name, value in expected.items():
                actual = attrs.get(name)
                if isinstance(value, str) and actual is not None:
                    actual = _text(actual)
                if actual != value:
                    raise DatasetIdentityError(
                        f"{location}: {name}={actual!r}, expected {value!r}; refusing "
                        "a dataset whose stored cell identity disagrees with its path"
                    )
        count = len(demos)
    if min_demos is not None and count < min_demos:
        raise DatasetIdentityError(
            f"{path}: requested {min_demos} demonstrations but the file contains "
            f"only {count}; refusing to inflate its pool weight"
        )
    return count


def require_delta_value_contract(path: Path) -> None:
    """Require the numeric delta vector to agree with its named JSON parameters."""
    import h5py
    import numpy as np

    with h5py.File(path, "r") as fh:
        if "data" not in fh or not fh["data"]:
            raise DatasetIdentityError(f"{path}: delta contract has no demonstrations")
        for name, demo in fh["data"].items():
            location = f"{path}:data/{name}"
            if "delta_true" not in demo:
                raise DatasetIdentityError(f"{location}: missing delta_true")
            fields_raw = demo.attrs.get("delta_fields")
            delta_raw = demo.attrs.get("delta_json")
            try:
                fields = json.loads(_text(fields_raw))
                delta = json.loads(_text(delta_raw))
                params = delta["params"]
                expected = np.asarray([params[field] for field in fields], dtype=np.float64)
                actual = np.asarray(demo["delta_true"], dtype=np.float64).reshape(-1)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise DatasetIdentityError(
                    f"{location}: invalid delta_fields/delta_json contract"
                ) from exc
            if actual.shape != expected.shape or not np.array_equal(
                actual, expected, equal_nan=False
            ):
                raise DatasetIdentityError(
                    f"{location}: delta_true={actual.tolist()} disagrees with "
                    f"delta_json params selected by delta_fields={fields!r}: "
                    f"expected {expected.tolist()}"
                )


def stamp_defect_state_contract(attrs: Any, *, owner_defect: str | None = None) -> None:
    """Persist the defect-state interpretation on an HDF5 attribute manager."""
    attrs["defect_state_schema_version"] = DEFECT_STATE_SCHEMA_VERSION
    attrs["defect_state_semantics"] = DEFECT_STATE_SEMANTICS
    if owner_defect is not None:
        attrs["defect_state_owner_defect"] = owner_defect


def stamp_analysis_trace_contract(attrs: Any) -> None:
    """Persist the version of the per-frame analysis trace written at collection."""
    attrs["analysis_trace_schema_version"] = ANALYSIS_TRACE_SCHEMA_VERSION


def require_analysis_trace_contract(path: Path) -> None:
    """Require the collection-time columns needed by physical diagnostics.

    These columns cannot be reconstructed from a materialised image dataset:
    ``u_inf`` and the torque command are tied to the exact ILC/control-step
    boundary, while phase/contact and tracking error are labels of that same
    boundary.  This is a necessary collection contract, not a complete
    ``PhiInputs`` record: ``a_dot_nominal`` still has to be derived with a live
    clean twin, its reference trajectory and the stored actuation state. Refuse
    legacy files before any analysis consumes them.
    """
    import h5py
    import numpy as np

    with h5py.File(path, "r") as fh:
        version = fh.attrs.get("analysis_trace_schema_version")
        try:
            parsed = int(version)
        except (TypeError, ValueError):
            raise DatasetIdentityError(
                f"{path}: missing analysis_trace_schema_version; re-collect with "
                "the current collector"
            ) from None
        if parsed != ANALYSIS_TRACE_SCHEMA_VERSION:
            raise DatasetIdentityError(
                f"{path}: analysis_trace_schema_version={parsed}, expected "
                f"{ANALYSIS_TRACE_SCHEMA_VERSION}; re-collect this dataset"
            )
        if "data" not in fh or not fh["data"]:
            raise DatasetIdentityError(f"{path}: analysis trace has no demonstrations")
        for name, demo in fh["data"].items():
            try:
                demo_version = int(demo.attrs.get("analysis_trace_schema_version"))
            except (TypeError, ValueError):
                raise DatasetIdentityError(
                    f"{path}: data/{name} has no analysis trace schema version; "
                    "re-collect this dataset"
                ) from None
            if demo_version != ANALYSIS_TRACE_SCHEMA_VERSION:
                raise DatasetIdentityError(
                    f"{path}: data/{name} analysis_trace_schema_version="
                    f"{demo_version}, expected {ANALYSIS_TRACE_SCHEMA_VERSION}"
                )
            missing = [field for field in ANALYSIS_TRACE_FIELDS if field not in demo]
            if missing:
                raise DatasetIdentityError(
                    f"{path}: data/{name} is missing analysis trace fields {missing}; "
                    "re-collect before computing phi or physical observables"
                )
            # Validate the per-frame shape and numeric payload here, before a
            # downstream producer can silently concatenate misaligned columns.
            # Raw files always carry c_true; the states/actions fallback keeps
            # this check useful for small materialisation fixtures that omit the
            # policy-facing arrays.
            anchor = next(
                (
                    demo[field]
                    for field in ("states", "actions", "c_true", "u_inf")
                    if field in demo
                ),
                None,
            )
            if anchor is None or len(anchor.shape) < 1:
                raise DatasetIdentityError(
                    f"{path}: data/{name} has no one-dimensional frame anchor "
                    "(states, actions, c_true or u_inf)"
                )
            n_frames = int(anchor.shape[0])
            if n_frames < 1:
                raise DatasetIdentityError(
                    f"{path}: data/{name} has an empty analysis trace"
                )
            for field, width in _ANALYSIS_TRACE_WIDTHS.items():
                dataset = demo[field]
                shape = tuple(int(size) for size in dataset.shape)
                if width is None:
                    expected_rank = (
                        1 if field in {"c_true_clipped", "phase", "contact"} else 2
                    )
                    if len(shape) != expected_rank:
                        raise DatasetIdentityError(
                            f"{path}: data/{name}/{field} has shape {shape}; expected "
                            f"rank {expected_rank} with {n_frames} frames"
                        )
                    if field == "joint_torque" and shape[1] < 1:
                        raise DatasetIdentityError(
                            f"{path}: data/{name}/joint_torque has no joint channels"
                        )
                elif shape != (n_frames, width):
                    raise DatasetIdentityError(
                        f"{path}: data/{name}/{field} has shape {shape}; expected "
                        f"({n_frames}, {width})"
                    )
                if shape[0] != n_frames:
                    raise DatasetIdentityError(
                        f"{path}: data/{name}/{field} has {shape[0]} frames; expected "
                        f"{n_frames}"
                    )
                values = np.asarray(dataset)
                try:
                    finite = bool(np.all(np.isfinite(values)))
                except TypeError:
                    raise DatasetIdentityError(
                        f"{path}: data/{name}/{field} must contain numeric values"
                    ) from None
                if not finite:
                    raise DatasetIdentityError(
                        f"{path}: data/{name}/{field} contains NaN or infinity"
                    )
            contact = np.asarray(demo["contact"])
            if not np.isin(contact, (0, 1)).all():
                raise DatasetIdentityError(
                    f"{path}: data/{name}/contact must contain only 0/1 values"
                )
            clipped = np.asarray(demo["c_true_clipped"])
            if clipped.dtype != np.bool_:
                raise DatasetIdentityError(
                    f"{path}: data/{name}/c_true_clipped must be a boolean vector"
                )
            phase = np.asarray(demo["phase"])
            if not np.isin(phase, (0, 1, 2, 3)).all():
                raise DatasetIdentityError(
                    f"{path}: data/{name}/phase must contain PHASE_ORDER ordinals 0..3"
                )


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _is_d4hi_file(fh: Any, path: Path, requested_defect: str | None) -> bool:
    if requested_defect == "D4hi":
        return True
    if _text(fh.attrs.get("defect_state_owner_defect", "")) == "D4hi":
        return True
    if "clean-D4hi" in path.parts:
        return True
    if "data" not in fh:
        return False
    return any(
        _text(group.attrs.get("defect_id", "")) == "D4hi"
        for group in fh["data"].values()
    )


def _reject(path: Path, detail: str) -> None:
    raise DefectStateSemanticsError(
        f"{path}: {detail}. Re-collect this D4hi cell with the current collector; "
        "legacy or unversioned files may contain wrapper-reset engagement instead "
        "of engagement derived from the live shadow equality efc_pos"
    )


def require_current_defect_state_contract(
    path: Path, *, requested_defect: str | None = None
) -> None:
    """Require current D4hi engagement semantics without rejecting other defects."""
    if requested_defect is not None and requested_defect != "D4hi":
        return

    import h5py

    with h5py.File(path, "r") as fh:
        if not _is_d4hi_file(fh, path, requested_defect):
            return

        version = fh.attrs.get("defect_state_schema_version")
        semantics = fh.attrs.get("defect_state_semantics")
        try:
            current_version = int(version)
        except (TypeError, ValueError):
            _reject(path, "missing or invalid defect_state_schema_version")
        if current_version != DEFECT_STATE_SCHEMA_VERSION:
            _reject(
                path,
                f"defect_state_schema_version is {current_version}, expected "
                f"{DEFECT_STATE_SCHEMA_VERSION}",
            )
        if semantics is None or _text(semantics) != DEFECT_STATE_SEMANTICS:
            _reject(path, "missing or incompatible defect_state_semantics")

        for name, group in fh["data"].items():
            demo_version = group.attrs.get("defect_state_schema_version")
            demo_semantics = group.attrs.get("defect_state_semantics")
            try:
                parsed_demo_version = int(demo_version)
            except (TypeError, ValueError):
                _reject(path, f"data/{name} has no valid defect-state version")
            if (
                parsed_demo_version != DEFECT_STATE_SCHEMA_VERSION
                or demo_semantics is None
                or _text(demo_semantics) != DEFECT_STATE_SEMANTICS
            ):
                _reject(path, f"data/{name} has incompatible defect-state semantics")
