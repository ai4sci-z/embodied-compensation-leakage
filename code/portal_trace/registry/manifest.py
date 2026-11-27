"""`registry/manifest.yaml`: embodiment ID -> (family, delta, compile hash).

11.2 asks for a manifest that pins what was actually compiled. The hash is
SHA-256 over the *canonical* MJCF of `registry.families`, that is, the XML
robosuite hands to the compiler with the machine-specific asset prefix replaced
by a token. Without that substitution the digest encodes the checkout path and
nothing reproduces on a second machine.

The manifest makes 9.3.3's bit-identical claim auditable instead of asserted.
Five of the six core defects never touch `mjModel`, so their rows carry the same
digest as their clean twin, and `identical_to_clean` says so. D4hi is the one
exception: it edits `jnt_range`, its MJCF differs, and the manifest shows a
different digest per level. If a defect that is supposed to live in `mjData` ever
changes its digest, this file is where it shows up.

Reproducibility. The document is emitted with sorted keys, no timestamp and no
absolute path, and `--check` recomputes it and diffs rather than rewriting, so
CI can fail on a drift instead of silently committing one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from portal_trace.defects.params import CORE_DEFECTS, defect
from portal_trace.env.versions import installed_versions
from portal_trace.registry.embodiments import (
    IMPL_VARIANTS,
    DeltaComponent,
    delta_components,
    embodiment_ids,
)
from portal_trace.registry.families import (
    MAIN_FAMILIES,
    StructuralDescriptor,
    descriptor,
    family,
)
from portal_trace.registry.twins import TwinPair, twin_descriptors
from portal_trace.types import EmbodimentId

__all__ = [
    "ManifestRow",
    "build_manifest",
    "to_document",
    "default_output_path",
    "main",
]


@dataclass(frozen=True)
class ManifestRow:
    """One embodiment: what it is, what delta it carries, what got compiled."""

    embodiment: EmbodimentId
    family: str
    robosuite_robot: str
    defect_impl_variant: str | None
    delta: tuple[DeltaComponent, ...]
    compile_sha256: str
    clean_sha256: str
    dof: int
    n_joints: int

    @property
    def identical_to_clean(self) -> bool:
        return self.compile_sha256 == self.clean_sha256

    def to_yaml_obj(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "robosuite_robot": self.robosuite_robot,
            "defect_id": self.embodiment.defect_code,
            "level": self.embodiment.level,
            "defect_impl_variant": self.defect_impl_variant,
            "delta": {c.name: {"value": c.value, "unit": c.unit} for c in self.delta},
            "compile_sha256": self.compile_sha256,
            "identical_to_clean": self.identical_to_clean,
            "dof": self.dof,
            "n_joints": self.n_joints,
        }


def _compiled_descriptor(eid: EmbodimentId) -> StructuralDescriptor:
    """The descriptor of what this embodiment actually compiles.

    Two of the core six reach the compiled model: D4hi rewrites the MJCF to add
    its shadow joint, and D8 writes `actuator_gainprm` after the compile. Both go
    through `registry.twins`, the same builder the 9.3.4 assertion runs on, so
    the manifest cannot describe a model that the alignment check never saw. The
    other four are `mjData` or wrapper writes, so the model is the family's own
    and the digests coincide by construction rather than by luck.

    Note what `compile_sha256` is and is not. It hashes the MJCF, so D8's is the
    family's own -- truthfully, because D8 changes no XML. What distinguishes a
    D8 row is its `delta` block, which carries the gain with its unit.
    """
    if eid.defect_code is None:
        return descriptor(eid.family)
    if defect(eid.defect_code).touches_mjmodel:
        assert eid.level is not None
        return twin_descriptors(TwinPair(eid.family, eid.defect_code, eid.level))[1]
    return descriptor(eid.family)


def build_manifest() -> tuple[ManifestRow, ...]:
    rows: list[ManifestRow] = []
    for family_name in MAIN_FAMILIES:
        fam = family(family_name)
        clean = descriptor(family_name)
        for eid in embodiment_ids(family_name):
            spec = None if eid.defect_code is None else defect(eid.defect_code)
            desc = _compiled_descriptor(eid)
            rows.append(
                ManifestRow(
                    embodiment=eid,
                    family=family_name,
                    robosuite_robot=fam.robosuite_robot,
                    defect_impl_variant=None if spec is None else IMPL_VARIANTS[spec.code],
                    delta=delta_components(spec, eid.level),
                    compile_sha256=desc.mjcf_sha256,
                    clean_sha256=clean.mjcf_sha256,
                    dof=desc.dof,
                    n_joints=len(desc.joint_names),
                )
            )
    return tuple(rows)


def _self_check(rows: tuple[ManifestRow, ...]) -> None:
    """9.3.3: only D4hi may change the compiled MJCF."""
    for row in rows:
        code = row.embodiment.defect_code
        expect_identical = code != "D4hi"
        if row.identical_to_clean != expect_identical:
            verb = "changed" if expect_identical else "did not change"
            raise AssertionError(
                f"{row.embodiment}: {verb} the compiled MJCF; 9.3.3 requires the "
                "opposite, and the invariance argument for this pair is void"
            )
    d4 = {r.compile_sha256 for r in rows if r.embodiment.defect_code == "D4hi"}
    n_expected = len(MAIN_FAMILIES) * len(defect("D4hi").levels)
    if len(d4) != n_expected:
        raise AssertionError(
            f"D4hi should yield {n_expected} distinct compiled models "
            f"(one per family and level); got {len(d4)}"
        )


def to_document(rows: tuple[ManifestRow, ...]) -> dict[str, Any]:
    """A dict whose YAML rendering is byte-stable across machines and runs."""
    entries = {str(r.embodiment): r.to_yaml_obj() for r in rows}
    # Digest of the entries alone, so that adding a version field to the envelope
    # does not invalidate an audit of the models themselves.
    payload = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "portal_trace.registry.manifest/1",
        "proposal_ref": "9.3.3, 11.2",
        "hash_input": (
            "canonical MJCF from RobotModel.get_xml(); the robosuite asset prefix is "
            "replaced by a token so the digest is path independent"
        ),
        "versions": installed_versions(),
        "core_defects": list(CORE_DEFECTS),
        "n_entries": len(rows),
        "entries_sha256": payload,
        "entries": entries,
    }


def default_output_path() -> Path:
    return Path(__file__).resolve().parent / "manifest.yaml"


def _render(doc: dict[str, Any]) -> str:
    return yaml.safe_dump(doc, sort_keys=True, default_flow_style=False, allow_unicode=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--check",
        action="store_true",
        help="recompute and diff against the committed file instead of writing",
    )
    args = p.parse_args(argv)

    rows = build_manifest()
    _self_check(rows)
    doc = to_document(rows)
    text = _render(doc)
    out = args.out if args.out is not None else default_output_path()

    if args.check:
        if not out.exists():
            print(f"[check] {out} does not exist", file=sys.stderr)
            return 1
        if out.read_text(encoding="utf-8") != text:
            print(f"[check] {out} is stale", file=sys.stderr)
            return 1
        print(f"[check] {out} reproduces: entries_sha256={doc['entries_sha256'][:16]}")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    n_changed = sum(1 for r in rows if not r.identical_to_clean)
    print(
        f"[write] {out}: {len(rows)} entries, {n_changed} with a modified MJCF, "
        f"entries_sha256={doc['entries_sha256'][:16]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
