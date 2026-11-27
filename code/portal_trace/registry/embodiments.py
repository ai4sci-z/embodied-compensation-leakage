"""The resolved embodiment registry: (family, defect, level) -> `EmbodimentId`.

11.2 lists `registry/embodiments.json` as a deliverable and 9.6.3 fixes its
extent: six morphology families x {delta = 0, the six core defects} x five
levels. The six are `CORE_DEFECTS`, which is D1, D2, D3b, D4hi, D5 and D8 --
9.2.6's stiction rule took D6 out and 9.2.5's replacement list put D8 in, and
reading the extent off the defect codes rather than off the constant would
silently drop a cell. 9.3.6 fixes the field groups. This module produces both.

Two of those field groups are filled here and the rest are not, on purpose.

Filled: identity (family, defect, the per-component delta with its unit, the
implementation variant), asset (source, MJCF hash, DoF, chain length), the
name-resolved index block, the injection block, and the graph-invariance block
whose two FGW numbers come from `registry.twins`.

Not filled: everything 9.3.6 calls calibration -- kappa, cl_pm, the equivalent
delay, the deadzone duty cycle, tau_step, the ILC diagnostics. Those are
measurements, they are declared in `portal_trace.lockfile`, and P0 has not run.
Each entry therefore carries `pending_measurements`, the list of lock keys it is
waiting on, instead of a null that a later reader could mistake for a zero.

The index block stores *names*, not numbers. 9.2.6 trap (iii) is explicit about
it: a raw `qpos` offset recorded against the arm-only family model is wrong the
moment a gripper and a nut are merged into the same `mjModel`. `resolve_indices`
turns the names into offsets against whatever model is actually running, and the
numbers written into the artefact are labelled as belonging to the family base
model so that nobody indexes a live simulation with them.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco

from portal_trace.defects.params import (
    CORE_DEFECTS,
    DEFECTS,
    DESCRIPTOR_VISIBLE_FIELDS,
    DefectSpec,
    defect,
)
from portal_trace.env.versions import installed_versions
from portal_trace.lockfile import REQUIRED_MEASUREMENTS, LockFile
from portal_trace.registry.families import (
    FAMILIES,
    MAIN_FAMILIES,
    MorphologyFamily,
    descriptor,
    family,
)
from portal_trace.registry.twins import fgw_matrix
from portal_trace.types import EmbodimentId

__all__ = [
    "DeltaComponent",
    "EmbodimentEntry",
    "PARAMETER_UNITS",
    "IMPL_VARIANTS",
    "delta_components",
    "embodiment_ids",
    "build_registry",
    "resolve_indices",
    "default_output_path",
    "main",
]

# Units of the injection parameters of 9.2.10. The level table states the values;
# the unit is what makes a value a physical quantity, and 9.3.6 requires `delta`
# to be recorded per component with one.
PARAMETER_UNITS: dict[str, str] = {
    "delta_q": "rad",
    "n_control_steps": "control_steps",
    "k_transmission": "N m / rad",
    "sigma_g": "dimensionless",
    "backlash": "rad",
    "w_over_tau_max": "dimensionless",
    "delta_f": "N m",
    "gain": "dimensionless",
    "link_compliance": "dimensionless",
    "tau_thermal": "s",
    "delta_q_inf": "rad",
    "amplitude": "N m",
    "n_poles": "count",
    "phase": "rad",
}

# 9.3.6 field `defect_impl_variant`. Two entries carry a decision rather than a
# name. D6 ships the 9.2.6 contingency as its primary caliber, which is what put
# it on `qfrc_applied` and, once the stiction that cost was measured, out of the
# core six. D8 took the vacated slot on `actuator_gainprm`, an mjModel write that
# a core defect would normally be barred from -- it is allowed because no node
# feature of 9.3.4 reads that field, which `registry.twins` measures rather than
# assumes by building both sides of the pair.
IMPL_VARIANTS: dict[str, str] = {
    "D1": "D1:sensing_twin",
    "D2": "D2:action_side",
    "D3a": "D3:a",
    "D3b": "D3:b",
    "D4hi": "D4:hi",
    "D4lo": "D4:lo",
    "D5": "D5:total_ctrl",
    "D6": "D6:qfrc_applied",
    "D7": "D7:link_compliance",
    "D8": "D8:gainprm",
    "D9": "D9:qfrc_applied",
    "D10": "D10:sensing_twin_drift",
}

# Lock declarations that an embodiment entry needs before it is complete (9.3.6
# calibration group). Listed by key template; the scope is the entry itself.
_PENDING_TEMPLATES: tuple[str, ...] = (
    "mjcf.armature.{family}.{joint}",
    "obs.proprio_native_dim.{family}",
    "defect.observable.{defect}.{level}.{family}",
    "sim.tau_step.{family}.{task}",
    "sim.tau_frame.{family}.{task}",
)


@dataclass(frozen=True)
class DeltaComponent:
    """One component of delta, with the unit that makes it a physical quantity."""

    name: str
    value: float
    unit: str

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "unit": self.unit}


@dataclass(frozen=True)
class EmbodimentEntry:
    """One row of `registry/embodiments.json` (9.3.6)."""

    embodiment: EmbodimentId
    family: str
    asset_source: str
    robosuite_robot: str
    mjcf_sha256: str
    dof: int
    chain_length_m: float
    in_36_pairs: bool
    defect_impl_variant: str | None
    delta: tuple[DeltaComponent, ...]
    injection_layers: tuple[str, ...]
    written_fields: tuple[str, ...]
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    index_in_family_model: dict[str, dict[str, int]]
    graph_exception: str
    node_feature_equal: bool | None
    fgw_ours: float | None
    fgw_baseline: float | None
    pending_measurements: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "embodiment_id": str(self.embodiment),
            "identity": {
                "family": self.family,
                "defect_id": self.embodiment.defect_code,
                "level": self.embodiment.level,
                "defect_impl_variant": self.defect_impl_variant,
                "delta": [c.to_json() for c in self.delta],
            },
            "asset": {
                "asset_source": self.asset_source,
                "robosuite_robot": self.robosuite_robot,
                "mjcf_sha256": self.mjcf_sha256,
                "dof": self.dof,
                "chain_length_m": self.chain_length_m,
                "reach_m": None,
                "reach_source": (
                    "manufacturer datasheet; 9.3.1 requires it per family and it is "
                    "not derivable from the installed assets, so it is left empty "
                    "rather than guessed from chain_length_m"
                ),
                "in_36_pairs": self.in_36_pairs,
            },
            "index": {
                "resolve_by": "joint and actuator name, never a stored offset (9.2.6 iii)",
                "joint_names": list(self.joint_names),
                "actuator_names": list(self.actuator_names),
                "index_in_family_model": self.index_in_family_model,
            },
            "injection": {
                "injection_layers": list(self.injection_layers),
                "written_fields": list(self.written_fields),
            },
            "graph_invariance": {
                "graph_exception": self.graph_exception,
                "node_feature_equal": self.node_feature_equal,
                "fgw_to_clean_twin": {
                    "ours": self.fgw_ours,
                    "baseline": self.fgw_baseline,
                },
            },
            "pending_measurements": list(self.pending_measurements),
        }


def embodiment_ids(family_name: str) -> tuple[EmbodimentId, ...]:
    """delta = 0 plus the five levels of each of the six core defects (9.6.3)."""
    out = [EmbodimentId(family_name)]
    for code in CORE_DEFECTS:
        out.extend(
            EmbodimentId(family_name, code, level.index) for level in defect(code).levels
        )
    return tuple(out)


def delta_components(spec: DefectSpec | None, level: int | None) -> tuple[DeltaComponent, ...]:
    """The delta vector. The clean twin's is empty, which is delta = 0.

    CONTRACT 3.3 makes that literal rather than symbolic: the clean twin is the
    same injection path with no component set, not a separate object.
    """
    if spec is None or level is None:
        return ()
    params = spec.level(level).params
    missing = [k for k in params if k not in PARAMETER_UNITS]
    if missing:
        raise KeyError(
            f"{spec.code} level {level} has parameters {missing} with no unit in "
            "PARAMETER_UNITS; 9.3.6 records delta per component with a unit"
        )
    return tuple(
        DeltaComponent(name=k, value=float(v), unit=PARAMETER_UNITS[k])
        for k, v in sorted(params.items())
    )


def _family_indices(fam: MorphologyFamily) -> tuple[dict[str, dict[str, int]], tuple[str, ...]]:
    """Name-keyed offsets in the family's own arm-only model.

    Recorded for provenance. They are *not* the offsets a running environment
    uses: merging a gripper and the task objects shifts everything, which is
    exactly the trap 9.2.6 (iii) warns about.
    """
    robot = fam.robot_model()
    model = robot.get_model(mode="mujoco")
    index: dict[str, dict[str, int]] = {}
    for name in robot.joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        index[name] = {
            "qpos_index": int(model.jnt_qposadr[jid]),
            "qvel_index": int(model.jnt_dofadr[jid]),
            "dof_index": int(model.jnt_dofadr[jid]),
        }
    actuators = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"actuator{a}"
        for a in range(model.nu)
    )
    for a, name in enumerate(actuators):
        index[name] = {"actuator_index": a}
    return index, actuators


def resolve_indices(
    model: mujoco.MjModel, entry: EmbodimentEntry
) -> dict[str, dict[str, int]]:
    """Resolve the entry's joint and actuator names against a live model.

    This is the only supported way to get an offset for an embodiment. It raises
    on an unknown name rather than returning -1, because a -1 index silently
    reads the last element of every MuJoCo array.
    """
    out: dict[str, dict[str, int]] = {}
    for name in entry.joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"{entry.embodiment}: this model has no joint {name!r}")
        out[name] = {
            "qpos_index": int(model.jnt_qposadr[jid]),
            "qvel_index": int(model.jnt_dofadr[jid]),
            "dof_index": int(model.jnt_dofadr[jid]),
        }
    for name in entry.actuator_names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise KeyError(f"{entry.embodiment}: this model has no actuator {name!r}")
        out[name] = {"actuator_index": aid}
    return out


def _pending(embodiment: EmbodimentId) -> tuple[str, ...]:
    """Lock keys this entry needs, restricted to the ones still absent."""
    lock = LockFile()
    out: list[str] = []
    desc = descriptor(embodiment.family)
    for template in _PENDING_TEMPLATES:
        decl = REQUIRED_MEASUREMENTS[template]
        fields = set(decl.scope_fields)
        if "defect" in fields and embodiment.defect_code is None:
            continue
        if (
            decl.key_template.startswith("defect.observable.")
            and embodiment.defect_code is not None
            and defect(embodiment.defect_code).observable_is_parameter
        ):
            # 9.2.10's per-family observable exists because the injection
            # parameter reaches the end effector through a family-dependent map.
            # D8's does not: the parameter is the ratio of delivered to commanded
            # torque and is the observable on every family. Listing it as pending
            # would put thirty keys into W1-8 that no measurement can fill,
            # because there is nothing to measure.
            continue
        scopes: list[dict[str, str]] = [{}]
        if "family" in fields:
            scopes = [{**s, "family": embodiment.family} for s in scopes]
        if "defect" in fields:
            scopes = [{**s, "defect": str(embodiment.defect_code)} for s in scopes]
        if "level" in fields:
            scopes = [{**s, "level": str(embodiment.level)} for s in scopes]
        if "joint" in fields:
            scopes = [{**s, "joint": j} for s in scopes for j in desc.joint_names]
        if "task" in fields:
            # Scoped by task as well; the registry entry is task independent, so
            # the whole family of keys is reported rather than one instance.
            out.append(decl.key_template)
            continue
        for scope in scopes:
            if not lock.has(decl, **scope):
                out.append(decl.key(**scope))
    return tuple(sorted(set(out)))


def build_registry(*, include_fgw: bool = True) -> tuple[EmbodimentEntry, ...]:
    """Every embodiment of the six main families, with its 9.3.6 fields."""
    fgw: dict[tuple[str, str, int], tuple[float, float, bool, str]] = {}
    if include_fgw:
        for row in fgw_matrix():
            fgw[(row.family, row.defect_code, row.level)] = (
                row.fgw_ours,
                row.fgw_baseline,
                row.node_feature_equal,
                row.graph_exception.value,
            )

    entries: list[EmbodimentEntry] = []
    for family_name in MAIN_FAMILIES:
        fam = family(family_name)
        desc = descriptor(family_name)
        index, actuators = _family_indices(fam)
        for eid in embodiment_ids(family_name):
            spec = None if eid.defect_code is None else defect(eid.defect_code)
            key = (family_name, eid.defect_code or "", eid.level if eid.level is not None else -1)
            numbers = fgw.get(key)
            entries.append(
                EmbodimentEntry(
                    embodiment=eid,
                    family=family_name,
                    asset_source=fam.asset_source.value,
                    robosuite_robot=fam.robosuite_robot,
                    mjcf_sha256=desc.mjcf_sha256,
                    dof=desc.dof,
                    chain_length_m=desc.chain_length_m,
                    in_36_pairs=fam.in_36_pairs and spec is not None,
                    defect_impl_variant=None if spec is None else IMPL_VARIANTS[spec.code],
                    delta=delta_components(spec, eid.level),
                    injection_layers=(
                        ()
                        if spec is None
                        else tuple(sorted(layer.value for layer in spec.layers))
                    ),
                    written_fields=(
                        ()
                        if spec is None
                        else tuple(sorted(f.value for f in spec.primary_fields))
                    ),
                    joint_names=desc.joint_names,
                    actuator_names=actuators,
                    index_in_family_model=index,
                    graph_exception=(
                        "none" if numbers is None else numbers[3]
                    ),
                    node_feature_equal=None if numbers is None else numbers[2],
                    fgw_ours=0.0 if spec is None else (None if numbers is None else numbers[0]),
                    fgw_baseline=0.0 if spec is None else (None if numbers is None else numbers[1]),
                    pending_measurements=_pending(eid),
                )
            )
    return tuple(entries)


def _self_check(entries: tuple[EmbodimentEntry, ...]) -> None:
    per_family = 1 + sum(len(defect(c).levels) for c in CORE_DEFECTS)
    expected = len(MAIN_FAMILIES) * per_family
    if len(entries) != expected:
        raise AssertionError(
            f"9.6.3 expects {len(MAIN_FAMILIES)} families x {per_family} embodiments "
            f"= {expected}; built {len(entries)}"
        )
    for e in entries:
        if e.embodiment.is_clean:
            continue
        spec = defect(str(e.embodiment.defect_code))
        # Not "writes mjModel": D8 does, and is a core defect anyway. What
        # breaks X1 is writing a field a morphology node feature reads, which
        # is a strictly smaller set. `registry.twins` proves the difference
        # matters by building both sides of the D8 pair and measuring zero.
        seen = set(spec.primary_fields) & DESCRIPTOR_VISIBLE_FIELDS
        if seen and spec.code != "D4hi":
            raise AssertionError(
                f"{e.embodiment}: writes {sorted(f.value for f in seen)}, which a "
                "morphology node feature reads, although it is a core defect other "
                "than D4hi; CONTRACT 3.5 has been violated upstream"
            )


def default_output_path() -> Path:
    return Path(__file__).resolve().parent / "embodiments.json"


def to_document(entries: tuple[EmbodimentEntry, ...]) -> dict[str, Any]:
    return {
        "schema": "portal_trace.registry.embodiments/1",
        "proposal_ref": "9.3.6, 9.6.3, 11.2",
        "versions": installed_versions(),
        "families": {
            name: {
                "robosuite_robot": fam.robosuite_robot,
                "role": fam.role.value,
                "excluded_by": None if fam.excluded_by is None else fam.excluded_by.value,
                "proposal_row": fam.proposal_row,
                "dof": descriptor(name).dof,
                "mjcf_sha256": descriptor(name).mjcf_sha256,
            }
            for name, fam in FAMILIES.items()
        },
        "core_defects": list(CORE_DEFECTS),
        "all_defects": sorted(DEFECTS),
        "n_embodiments": len(entries),
        "embodiments": [e.to_json() for e in entries],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--no-fgw",
        action="store_true",
        help="skip the FGW columns; the artefact is then incomplete and says so",
    )
    args = p.parse_args(argv)

    entries = build_registry(include_fgw=not args.no_fgw)
    _self_check(entries)
    out = args.out if args.out is not None else default_output_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(to_document(entries), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    n_pending = sum(len(e.pending_measurements) for e in entries)
    print(f"[write] {out}: {len(entries)} embodiments, {n_pending} outstanding lock keys")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
