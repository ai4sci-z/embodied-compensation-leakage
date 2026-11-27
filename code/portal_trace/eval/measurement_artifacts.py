"""Immutable evidence contracts for defect observables and P0 kappa.

These artifacts deliberately stop before measurement.  They define what a
future producer has to prove, keep the defect-specific dose observable separate
from the terminal physical compensation norm, and make every method choice that
is still subject to amendment explicit in the evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from portal_trace.defects.params import CORE_DEFECTS
from portal_trace.defects.params import defect as defect_spec
from portal_trace.lockfile import NO_CHECKOUT
from portal_trace.registry.families import MAIN_FAMILIES

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "DEFECT_OBSERVABLE_KIND",
    "KAPPA_KIND",
    "MEASUREMENT_ARTIFACT_PRODUCER",
    "canonical_json_sha256",
    "seal_measurement_artifact",
    "validate_measurement_artifact",
    "validate_measurement_artifact_set",
    "write_measurement_artifact",
    "load_measurement_artifact",
]

ARTIFACT_SCHEMA_VERSION = 1
MEASUREMENT_ARTIFACT_PRODUCER = "portal_trace.eval.measurement_artifacts"
DEFECT_OBSERVABLE_KIND = "defect_observable_measurement"
KAPPA_KIND = "kappa_c_norm_m_measurement"

_SHA256_HEX_LENGTH = 64
_COMMON_FIELDS = {
    "schema_version",
    "producer",
    "artifact_kind",
    "source_commit",
    "worktree_dirty",
    "produced_at",
    "environment",
    "measurement_protocol",
    "scope",
    "raw_evidence",
    "method",
    "measurement",
}
_KAPPA_FIELDS = {
    "c_true_provenance",
    "tolerance_provenance",
    "controller_config_sha256",
    "reference_provenance",
    "paired_reset_contract",
}


def canonical_json_sha256(payload: Any) -> str:
    """SHA-256 over the repository's canonical JSON representation."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_object(source: Path, section: str, value: Any, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{source}: {section} must be a JSON object")
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{source}: {section} fields mismatch; missing={missing}, extra={extra}")
    return value


def _require_string(source: Path, section: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: {section} must be a non-empty string")
    return value


def _require_sha256(source: Path, section: str, value: Any) -> str:
    digest = _require_string(source, section, value)
    if (
        digest != digest.lower()
        or len(digest) != _SHA256_HEX_LENGTH
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError(f"{source}: {section} must be a 64-character SHA-256")
    return digest


def _require_source_commit(source: Path, value: Any) -> str:
    commit = _require_string(source, "source_commit", value)
    if (
        commit == NO_CHECKOUT
        or len(commit) != 40
        or commit != commit.lower()
        or any(char not in "0123456789abcdef" for char in commit)
    ):
        raise ValueError(f"{source}: formal measurement needs a full lowercase Git source commit")
    return commit


def _require_number(
    source: Path, section: str, value: Any, *, minimum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source}: {section} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{source}: {section} must be a finite number")
    if minimum is not None and number < minimum:
        raise ValueError(f"{source}: {section} must be at least {minimum}, got {number}")
    return number


def _validate_timestamp(source: Path, value: Any) -> None:
    timestamp = _require_string(source, "produced_at", value)
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{source}: produced_at must be an ISO-8601 timestamp") from None
    if parsed.tzinfo is None:
        raise ValueError(f"{source}: produced_at must include a timezone")


def _validate_environment(source: Path, value: Any) -> None:
    allowed = {"backend", "versions", "python", "platform", "lite_physics", "extra"}
    required = allowed - {"extra"}
    if not isinstance(value, dict):
        raise ValueError(f"{source}: environment must be a JSON object")
    missing = sorted(required - set(value))
    extra = sorted(set(value) - allowed)
    if missing or extra:
        raise ValueError(f"{source}: environment fields mismatch; missing={missing}, extra={extra}")
    for name in ("backend", "python", "platform"):
        _require_string(source, f"environment.{name}", value[name])
    if not isinstance(value["versions"], dict) or not value["versions"]:
        raise ValueError(f"{source}: environment.versions must be a non-empty JSON object")
    if type(value["lite_physics"]) is not bool:
        raise ValueError(f"{source}: environment.lite_physics must be a JSON boolean")
    if "extra" in value and not isinstance(value["extra"], dict):
        raise ValueError(f"{source}: environment.extra must be a JSON object")


def _validate_file_reference(source: Path, section: str, value: Any) -> dict[str, Any]:
    ref = _require_object(source, section, value, {"path", "sha256"})
    path = Path(_require_string(source, f"{section}.path", ref["path"]))
    if not path.is_absolute():
        raise ValueError(f"{source}: {section}.path must be absolute")
    _require_sha256(source, f"{section}.sha256", ref["sha256"])
    return ref


def _validate_raw_evidence(source: Path, value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{source}: raw_evidence must be a non-empty JSON array")
    paths: list[str] = []
    for index, item in enumerate(value):
        section = f"raw_evidence[{index}]"
        ref = _require_object(source, section, item, {"path", "sha256", "role"})
        _validate_file_reference(source, section, {"path": ref["path"], "sha256": ref["sha256"]})
        _require_string(source, f"{section}.role", ref["role"])
        paths.append(ref["path"])
    if len(paths) != len(set(paths)):
        raise ValueError(f"{source}: raw_evidence contains duplicate paths")


def _validate_named_rule(source: Path, section: str, value: Any) -> None:
    rule = _require_object(source, section, value, {"name", "parameters"})
    _require_string(source, f"{section}.name", rule["name"])
    if not isinstance(rule["parameters"], dict):
        raise ValueError(f"{source}: {section}.parameters must be a JSON object")
    canonical_json_sha256(rule["parameters"])


def _validate_conditional_methods(source: Path, defect: str, method: dict[str, Any]) -> None:
    d4hi = method["d4hi_support"]
    if defect == "D4hi":
        if not isinstance(d4hi, dict) or d4hi.get("status") != "applicable":
            raise ValueError(f"{source}: D4hi support status must be applicable")
        d4hi = _require_object(
            source,
            "method.d4hi_support",
            d4hi,
            {"status", "denominator", "zero_support"},
        )
        _require_string(source, "method.d4hi_support.denominator", d4hi["denominator"])
        _require_string(source, "method.d4hi_support.zero_support", d4hi["zero_support"])
    else:
        d4hi = _require_object(source, "method.d4hi_support", d4hi, {"status"})
        if d4hi["status"] != "not_applicable":
            raise ValueError(f"{source}: non-D4hi artifact must mark D4hi support not_applicable")

    d5 = method["d5_torque"]
    if defect == "D5":
        if not isinstance(d5, dict) or d5.get("status") != "applicable":
            raise ValueError(f"{source}: D5 torque caliber must be total_ctrl or task_only")
        d5 = _require_object(source, "method.d5_torque", d5, {"status", "caliber"})
        if d5["status"] != "applicable" or d5["caliber"] not in {"total_ctrl", "task_only"}:
            raise ValueError(f"{source}: D5 torque caliber must be total_ctrl or task_only")
    else:
        d5 = _require_object(source, "method.d5_torque", d5, {"status"})
        if d5["status"] != "not_applicable":
            raise ValueError(f"{source}: non-D5 artifact must mark D5 torque not_applicable")

    d8 = method["d8_severity"]
    if defect == "D8":
        if not isinstance(d8, dict) or d8.get("status") != "applicable":
            raise ValueError(f"{source}: D8 severity coordinate is not an approved explicit form")
        d8 = _require_object(
            source,
            "method.d8_severity",
            d8,
            {"status", "coordinate", "increases_with_severity"},
        )
        expected_direction = {
            "gain": False,
            "one_minus_gain": True,
            "inverse_gain_minus_one": True,
        }
        coordinate = d8["coordinate"]
        if d8["status"] != "applicable" or coordinate not in expected_direction:
            raise ValueError(f"{source}: D8 severity coordinate is not an approved explicit form")
        if type(d8["increases_with_severity"]) is not bool:
            raise ValueError(f"{source}: D8 increases_with_severity must be a JSON boolean")
        if d8["increases_with_severity"] is not expected_direction[coordinate]:
            raise ValueError(f"{source}: D8 severity direction contradicts coordinate {coordinate}")
    else:
        d8 = _require_object(source, "method.d8_severity", d8, {"status"})
        if d8["status"] != "not_applicable":
            raise ValueError(f"{source}: non-D8 artifact must mark D8 severity not_applicable")


def _validate_method(source: Path, defect: str, value: Any, *, kappa: bool) -> None:
    method = _require_object(
        source,
        "method",
        value,
        {"aggregation", "frame_selection", "d4hi_support", "d5_torque", "d8_severity"},
    )
    _validate_named_rule(source, "method.aggregation", method["aggregation"])
    _validate_named_rule(source, "method.frame_selection", method["frame_selection"])
    if kappa:
        selection = method["frame_selection"]
        if selection["name"] != "terminal_fine_approach":
            raise ValueError(f"{source}: kappa frame selection must be terminal_fine_approach")
        params = _require_object(
            source,
            "method.frame_selection.parameters",
            selection["parameters"],
            {"distance_lt_m", "speed_lt_m_s"},
        )
        distance = _require_number(source, "distance_lt_m", params["distance_lt_m"], minimum=0.0)
        speed = _require_number(source, "speed_lt_m_s", params["speed_lt_m_s"], minimum=0.0)
        if distance != 0.015 or speed != 0.05:
            raise ValueError(f"{source}: kappa terminal thresholds must be 0.015 m and 0.05 m/s")
    _validate_conditional_methods(source, defect, method)


def _validate_observable(source: Path, artifact: dict[str, Any]) -> None:
    scope = _require_object(source, "scope", artifact["scope"], {"defect", "level", "family"})
    defect = _require_string(source, "scope.defect", scope["defect"])
    if defect not in CORE_DEFECTS:
        raise ValueError(f"{source}: scope.defect must be one of {list(CORE_DEFECTS)}")
    spec = defect_spec(defect)
    level = scope["level"]
    if type(level) is not int or level not in {item.index for item in spec.levels}:
        raise ValueError(f"{source}: scope.level is not declared for {defect}")
    family = _require_string(source, "scope.family", scope["family"])
    if family not in MAIN_FAMILIES:
        raise ValueError(f"{source}: scope.family must be a registered main family")
    measurement = _require_object(
        source, "measurement", artifact["measurement"], {"name", "value", "unit"}
    )
    if measurement["name"] != "defect_observable":
        raise ValueError(f"{source}: observable measurement name must be defect_observable")
    value = _require_number(source, "measurement.value", measurement["value"], minimum=0.0)
    expected_unit = spec.levels[level].observable_unit
    if measurement["unit"] != expected_unit:
        raise ValueError(f"{source}: {defect} observable must use unit {expected_unit!r}")
    _validate_method(source, defect, artifact["method"], kappa=False)
    if defect == "D8":
        gain = float(spec.levels[level].params["gain"])
        coordinate = artifact["method"]["d8_severity"]["coordinate"]
        expected_value = {
            "gain": gain,
            "one_minus_gain": 1.0 - gain,
            "inverse_gain_minus_one": 1.0 / gain - 1.0,
        }[coordinate]
        if not math.isclose(value, expected_value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"{source}: D8 observable value does not match its declared "
                f"severity coordinate {coordinate!r}"
            )


def _validate_controller_hashes(source: Path, value: Any) -> None:
    hashes = _require_object(source, "controller_config_sha256", value, {"defective", "clean"})
    defective = _require_sha256(source, "controller_config_sha256.defective", hashes["defective"])
    clean = _require_sha256(source, "controller_config_sha256.clean", hashes["clean"])
    if defective != clean:
        raise ValueError(f"{source}: defective and clean controller hashes must match")


def _validate_kappa(source: Path, artifact: dict[str, Any]) -> None:
    scope = _require_object(
        source,
        "scope",
        artifact["scope"],
        {"task", "geom_variant", "defect", "level", "family"},
    )
    for name in ("task", "geom_variant", "defect", "family"):
        _require_string(source, f"scope.{name}", scope[name])
    if scope["defect"] not in CORE_DEFECTS:
        raise ValueError(f"{source}: scope.defect must be one of {list(CORE_DEFECTS)}")
    spec = defect_spec(scope["defect"])
    if type(scope["level"]) is not int or scope["level"] not in {
        item.index for item in spec.levels
    }:
        raise ValueError(f"{source}: scope.level is not declared for {scope['defect']}")
    if scope["family"] not in MAIN_FAMILIES:
        raise ValueError(f"{source}: scope.family must be a registered main family")

    measurement = _require_object(
        source,
        "measurement",
        artifact["measurement"],
        {"name", "value", "unit", "n_frames_in_phase"},
    )
    if measurement["name"] != "c_norm_m" or measurement["unit"] != "m":
        raise ValueError(f"{source}: kappa numerator must be named c_norm_m and use unit m")
    _require_number(source, "measurement.value", measurement["value"], minimum=0.0)
    if type(measurement["n_frames_in_phase"]) is not int or measurement["n_frames_in_phase"] <= 0:
        raise ValueError(f"{source}: kappa n_frames_in_phase must be a positive integer")

    c_true = _require_object(
        source,
        "c_true_provenance",
        artifact["c_true_provenance"],
        {"path", "sha256", "source_commit", "environment_sha256"},
    )
    _validate_file_reference(
        source, "c_true_provenance", {"path": c_true["path"], "sha256": c_true["sha256"]}
    )
    environment_sha256 = canonical_json_sha256(artifact["environment"])
    if c_true["source_commit"] != artifact["source_commit"]:
        raise ValueError(f"{source}: c_true source commit does not match the measurement")
    if c_true["environment_sha256"] != environment_sha256:
        raise ValueError(f"{source}: c_true environment does not match the measurement")
    raw_refs = {row["path"]: row["sha256"] for row in artifact["raw_evidence"]}
    if raw_refs.get(c_true["path"]) != c_true["sha256"]:
        raise ValueError(f"{source}: c_true path and SHA-256 must match one raw_evidence entry")

    tolerance = _require_object(
        source,
        "tolerance_provenance",
        artifact["tolerance_provenance"],
        {
            "key",
            "value_m",
            "unit",
            "tol_kind",
            "artifact",
            "source_commit",
            "environment_sha256",
        },
    )
    _require_string(source, "tolerance_provenance.key", tolerance["key"])
    _require_number(source, "tolerance_provenance.value_m", tolerance["value_m"], minimum=0.0)
    if tolerance["value_m"] <= 0.0 or tolerance["unit"] != "m":
        raise ValueError(f"{source}: kappa tolerance must be positive and use unit m")
    _require_string(source, "tolerance_provenance.tol_kind", tolerance["tol_kind"])
    _validate_file_reference(source, "tolerance_provenance.artifact", tolerance["artifact"])
    if tolerance["source_commit"] != artifact["source_commit"]:
        raise ValueError(f"{source}: tolerance source commit does not match the measurement")
    if tolerance["environment_sha256"] != environment_sha256:
        raise ValueError(f"{source}: tolerance environment does not match the measurement")
    _validate_controller_hashes(source, artifact["controller_config_sha256"])
    _validate_file_reference(source, "reference_provenance", artifact["reference_provenance"])
    _require_string(source, "paired_reset_contract", artifact["paired_reset_contract"])
    _validate_method(source, scope["defect"], artifact["method"], kappa=True)


def _validate_body(source: Path, artifact: dict[str, Any]) -> None:
    kind = artifact.get("artifact_kind")
    expected = _COMMON_FIELDS | (_KAPPA_FIELDS if kind == KAPPA_KIND else set())
    _require_object(source, "artifact", artifact, expected)
    if artifact["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"{source}: measurement schema {artifact['schema_version']}, "
            f"expected {ARTIFACT_SCHEMA_VERSION}"
        )
    if artifact["producer"] != MEASUREMENT_ARTIFACT_PRODUCER:
        raise ValueError(f"{source}: unexpected measurement artifact producer")
    if kind not in {DEFECT_OBSERVABLE_KIND, KAPPA_KIND}:
        raise ValueError(f"{source}: unknown measurement artifact kind {kind!r}")
    _require_source_commit(source, artifact["source_commit"])
    if artifact["worktree_dirty"] is not False:
        raise ValueError(f"{source}: formal measurement requires worktree_dirty=false")
    _validate_timestamp(source, artifact["produced_at"])
    _validate_environment(source, artifact["environment"])
    _validate_file_reference(source, "measurement_protocol", artifact["measurement_protocol"])
    _validate_raw_evidence(source, artifact["raw_evidence"])
    if kind == DEFECT_OBSERVABLE_KIND:
        _validate_observable(source, artifact)
    else:
        _validate_kappa(source, artifact)
    canonical_json_sha256(artifact)


def seal_measurement_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and add a hash over every field except the hash itself."""
    source = Path("<in-memory measurement artifact>")
    artifact = json.loads(json.dumps(dict(payload), allow_nan=False))
    if "artifact_sha256" in artifact:
        raise ValueError(f"{source}: seal input already contains artifact_sha256")
    _validate_body(source, artifact)
    artifact["artifact_sha256"] = canonical_json_sha256(artifact)
    return artifact


def validate_measurement_artifact(
    payload: Mapping[str, Any], *, source: Path | str = "<measurement artifact>"
) -> dict[str, Any]:
    """Validate schema and the embedded tamper-evident canonical hash."""
    path = Path(source)
    artifact = json.loads(json.dumps(dict(payload), allow_nan=False))
    expected_hash = _require_sha256(path, "artifact_sha256", artifact.get("artifact_sha256"))
    body = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    _validate_body(path, body)
    actual_hash = canonical_json_sha256(body)
    if actual_hash != expected_hash:
        raise ValueError(f"{path}: artifact SHA-256 no longer matches its canonical payload")
    return artifact


def validate_measurement_artifact_set(
    payloads: Sequence[Mapping[str, Any]], *, source: Path | str = "<measurement set>"
) -> tuple[dict[str, Any], ...]:
    """Reject duplicate scopes or artifacts assembled across provenance generations."""
    path = Path(source)
    if not payloads:
        raise ValueError(f"{path}: measurement set is empty")
    artifacts = tuple(
        validate_measurement_artifact(item, source=f"{path}[{index}]")
        for index, item in enumerate(payloads)
    )
    kinds = {item["artifact_kind"] for item in artifacts}
    if len(kinds) != 1:
        raise ValueError(f"{path}: measurement set mixes artifact kinds {sorted(kinds)}")
    generations = {
        (
            item["source_commit"],
            canonical_json_sha256(item["environment"]),
            canonical_json_sha256(item["measurement_protocol"]),
        )
        for item in artifacts
    }
    if len(generations) != 1:
        raise ValueError(
            f"{path}: measurement set mixes commit, environment, or protocol provenance"
        )
    identities = [canonical_json_sha256(item["scope"]) for item in artifacts]
    if len(identities) != len(set(identities)):
        raise ValueError(f"{path}: measurement set contains duplicate scopes")
    return artifacts


def _verify_reference(source: Path, section: str, value: Mapping[str, Any]) -> None:
    path = Path(value["path"])
    if not path.is_file():
        raise ValueError(f"{source}: {section} does not exist: {path}")
    if _file_sha256(path) != value["sha256"]:
        raise ValueError(f"{source}: {section} SHA-256 no longer matches: {path}")


def _verify_references(source: Path, artifact: Mapping[str, Any]) -> None:
    _verify_reference(source, "measurement_protocol", artifact["measurement_protocol"])
    for index, ref in enumerate(artifact["raw_evidence"]):
        _verify_reference(source, f"raw_evidence[{index}]", ref)
    if artifact["artifact_kind"] == KAPPA_KIND:
        _verify_reference(source, "c_true_provenance", artifact["c_true_provenance"])
        _verify_reference(
            source,
            "tolerance_provenance.artifact",
            artifact["tolerance_provenance"]["artifact"],
        )
        _verify_reference(source, "reference_provenance", artifact["reference_provenance"])


def write_measurement_artifact(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate, seal and atomically publish one JSON artifact without replacement."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable measurement artifact {path}")
    artifact = (
        validate_measurement_artifact(payload, source=path)
        if "artifact_sha256" in payload
        else seal_measurement_artifact(payload)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    fd, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(encoded)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite immutable measurement artifact {path}"
            ) from None
        published = True
    finally:
        temporary.unlink(missing_ok=True)
        if published:
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    return artifact


def load_measurement_artifact(path: Path, *, verify_references: bool = False) -> dict[str, Any]:
    """Load one artifact and optionally re-hash all referenced evidence files."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot read measurement artifact: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: measurement artifact must be a JSON object")
    artifact = validate_measurement_artifact(payload, source=path)
    if verify_references:
        _verify_references(path, artifact)
    return artifact
