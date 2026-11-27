"""Measured constants and their provenance.

The proposal marks a number of constants 待核实 (to be verified) and requires
(9.8.5, 14.5 E-clauses) that every one of them come out of a calibration script
before anything downstream may run. This module is how that requirement is
enforced: a quantity is declared once as a `RequiredMeasurement`, and the only
way to obtain its value is to look it up in the lock file. There is no default,
no fallback and no "reasonable guess" branch anywhere in this package -- a
missing measurement raises, and the exception says which script produces it and
which part of the experiment it blocks.

Literature facts (quantities copied from a published paper rather than measured
here) go through `LiteratureFact` instead. They are also stored in the lock
file, but they carry a citation rather than a producing script, and they are
never confused with measurements.
"""

from __future__ import annotations

import json
import string
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "MeasurementMissingError",
    "MeasurementProvenanceError",
    "RequiredMeasurement",
    "LiteratureFact",
    "MeasurementRecord",
    "LockFile",
    "REQUIRED_MEASUREMENTS",
    "LITERATURE_FACTS",
    "default_lock_path",
    "source_commit",
]

SCHEMA_VERSION = 1

# The string that goes in a record's `source_commit` when the tree is not a git
# checkout. Named rather than written twice, because it is the one value a reader
# has to be able to recognise as "provenance absent" instead of mistaking for a
# revision.
NO_CHECKOUT = "not-a-git-checkout"


def source_commit() -> str:
    """The commit that produced an artefact, for its provenance field.

    Lives here because this module owns provenance: `MeasurementRecord`,
    `registry/lock.json`, the model cards and the A13 teacher checkpoint all
    carry the same field, and two implementations of "which revision was this"
    would eventually disagree about the answer. A tree with no git metadata
    returns `NO_CHECKOUT` rather than raising -- the absence is recorded in the
    artefact, where a reader sees it, instead of blocking a run that is otherwise
    complete.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return NO_CHECKOUT
    return out.stdout.strip()


class MeasurementMissingError(KeyError):
    """A declared quantity has not been measured yet.

    Raised instead of returning a default. The message names the producing
    script and the blocked work so the reader does not have to grep the
    proposal to find out what to do next.
    """

    def __init__(self, decl: RequiredMeasurement, key: str, lock_path: Path) -> None:
        self.declaration = decl
        self.key = key
        self.lock_path = lock_path
        super().__init__(str(self))

    def __str__(self) -> str:  # noqa: D105 - message is the whole point
        d = self.declaration
        lines = [
            f"measurement {self.key!r} is not in {self.lock_path}",
            f"  what it is : {d.description} [{d.unit}]",
            f"  proposal   : {d.proposal_ref}",
            f"  produced by: {d.produced_by}",
            f"  blocks     : {d.blocks}",
        ]
        if d.w1_item is not None:
            lines.append(f"  W1 item    : {d.w1_item}")
        return "\n".join(lines)


class MeasurementProvenanceError(ValueError):
    """A lock file entry exists but does not carry acceptable provenance.

    This is what stops a value from being pasted in by hand: an entry has to
    name the script that produced it, and that script has to be the one the
    declaration expects.
    """


def _format_fields(template: str) -> tuple[str, ...]:
    return tuple(
        name for _, name, _, _ in string.Formatter().parse(template) if name is not None
    )


@dataclass(frozen=True)
class RequiredMeasurement:
    """A quantity that must be measured before it can be used.

    `key_template` may contain named fields (e.g. ``"{family}"``); call `key()`
    with those fields bound to get the concrete lock file key. Declarations with
    no fields are scalars.
    """

    key_template: str
    unit: str
    description: str
    proposal_ref: str
    produced_by: str
    blocks: str
    w1_item: str | None = None
    valid_range: tuple[float, float] | None = None

    @property
    def scope_fields(self) -> tuple[str, ...]:
        return _format_fields(self.key_template)

    def key(self, **scope: Any) -> str:
        expected = set(self.scope_fields)
        got = set(scope)
        if expected != got:
            missing = sorted(expected - got)
            extra = sorted(got - expected)
            raise TypeError(
                f"{self.key_template!r} is scoped by {sorted(expected)}; "
                f"missing={missing} unexpected={extra}"
            )
        return self.key_template.format(**scope)


@dataclass(frozen=True)
class LiteratureFact:
    """A value copied from a publication rather than measured here.

    Kept separate from `RequiredMeasurement` because the two have different
    failure modes: a measurement can be produced by running something, a
    literature fact can only be checked against the source.
    """

    key: str
    description: str
    citation: str
    proposal_ref: str
    blocks: str


@dataclass(frozen=True)
class MeasurementRecord:
    """One entry in the lock file, with the provenance that makes it usable."""

    value: Any
    unit: str
    produced_by: str
    produced_at: str
    source_commit: str
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        out = {
            "value": self.value,
            "unit": self.unit,
            "produced_by": self.produced_by,
            "produced_at": self.produced_at,
            "source_commit": self.source_commit,
        }
        if self.notes:
            out["notes"] = self.notes
        if self.extra:
            out["extra"] = self.extra
        return out

    @classmethod
    def from_json(cls, blob: dict[str, Any], key: str) -> MeasurementRecord:
        required = ("value", "unit", "produced_by", "produced_at", "source_commit")
        missing = [f for f in required if f not in blob]
        if missing:
            raise MeasurementProvenanceError(
                f"lock entry {key!r} is missing provenance fields {missing}; "
                "entries written by hand are rejected on purpose"
            )
        return cls(
            value=blob["value"],
            unit=blob["unit"],
            produced_by=blob["produced_by"],
            produced_at=blob["produced_at"],
            source_commit=blob["source_commit"],
            notes=blob.get("notes", ""),
            extra=blob.get("extra", {}),
        )


def default_lock_path() -> Path:
    return Path(__file__).resolve().parent / "registry" / "lock.json"


class LockFile:
    """Read/write access to ``registry/lock.json``.

    Reads go through `get`, which takes the declaration rather than a bare
    string so that the unit and the producing script can be checked at the point
    of use.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else default_lock_path()
        self._doc: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "schema_version": SCHEMA_VERSION,
                "measurements": {},
                "literature": {},
                "versions": {},
                "backend": None,
            }
        with self.path.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
        version = doc.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"{self.path} has schema_version {version!r}, this build expects "
                f"{SCHEMA_VERSION}; migrate the file rather than loosening the check"
            )
        doc.setdefault("measurements", {})
        doc.setdefault("literature", {})
        doc.setdefault("versions", {})
        doc.setdefault("backend", None)
        return doc

    # -- reading ---------------------------------------------------------

    def has(self, decl: RequiredMeasurement, **scope: Any) -> bool:
        return decl.key(**scope) in self._doc["measurements"]

    def record(self, decl: RequiredMeasurement, **scope: Any) -> MeasurementRecord:
        key = decl.key(**scope)
        blob = self._doc["measurements"].get(key)
        if blob is None:
            raise MeasurementMissingError(decl, key, self.path)
        rec = MeasurementRecord.from_json(blob, key)
        if rec.unit != decl.unit:
            raise MeasurementProvenanceError(
                f"lock entry {key!r} is in {rec.unit!r} but {decl.key_template!r} "
                f"is declared in {decl.unit!r}"
            )
        if rec.produced_by != decl.produced_by:
            raise MeasurementProvenanceError(
                f"lock entry {key!r} claims to come from {rec.produced_by!r} but "
                f"the declaration names {decl.produced_by!r}; a value produced by "
                "something else is not the same measurement"
            )
        return rec

    def get(self, decl: RequiredMeasurement, **scope: Any) -> Any:
        rec = self.record(decl, **scope)
        if decl.valid_range is not None and isinstance(rec.value, (int, float)):
            lo, hi = decl.valid_range
            if not (lo <= rec.value <= hi):
                raise MeasurementProvenanceError(
                    f"lock entry {decl.key(**scope)!r} has value {rec.value} "
                    f"outside the declared valid range [{lo}, {hi}]"
                )
        return rec.value

    def fact(self, decl: LiteratureFact) -> Any:
        blob = self._doc["literature"].get(decl.key)
        if blob is None:
            raise MeasurementMissingError(
                RequiredMeasurement(
                    key_template=decl.key,
                    unit="(literature)",
                    description=decl.description,
                    proposal_ref=decl.proposal_ref,
                    produced_by=f"transcribe from {decl.citation}",
                    blocks=decl.blocks,
                ),
                decl.key,
                self.path,
            )
        if blob.get("citation") != decl.citation:
            raise MeasurementProvenanceError(
                f"literature entry {decl.key!r} cites {blob.get('citation')!r}, "
                f"declaration cites {decl.citation!r}"
            )
        return blob["value"]

    # -- writing ---------------------------------------------------------

    def put(
        self,
        decl: RequiredMeasurement,
        value: Any,
        *,
        source_commit: str,
        notes: str = "",
        extra: dict[str, Any] | None = None,
        **scope: Any,
    ) -> None:
        key = decl.key(**scope)
        rec = MeasurementRecord(
            value=value,
            unit=decl.unit,
            produced_by=decl.produced_by,
            produced_at=datetime.now(UTC).isoformat(timespec="seconds"),
            source_commit=source_commit,
            notes=notes,
            extra=extra or {},
        )
        self._doc["measurements"][key] = rec.to_json()

    def put_fact(self, decl: LiteratureFact, value: Any) -> None:
        self._doc["literature"][decl.key] = {
            "value": value,
            "citation": decl.citation,
            "description": decl.description,
            "proposal_ref": decl.proposal_ref,
        }

    def set_versions(self, versions: dict[str, str]) -> None:
        self._doc["versions"] = dict(versions)

    def set_backend(self, backend: str) -> None:
        self._doc["backend"] = backend

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(self._doc, fh, indent=2, ensure_ascii=False, sort_keys=True)
            fh.write("\n")

    # -- reporting -------------------------------------------------------

    def outstanding(self) -> list[RequiredMeasurement]:
        """Declarations with no entry at all.

        Scoped declarations are reported when none of their instances are
        present; partially filled ones are the caller's problem to enumerate,
        since only the caller knows which scopes it needs.
        """
        out = []
        for decl in REQUIRED_MEASUREMENTS.values():
            if decl.scope_fields:
                prefix = decl.key_template.split("{", 1)[0]
                if not any(k.startswith(prefix) for k in self._doc["measurements"]):
                    out.append(decl)
            elif decl.key_template not in self._doc["measurements"]:
                out.append(decl)
        return out


def _decl(m: RequiredMeasurement) -> RequiredMeasurement:
    return m


# ---------------------------------------------------------------------------
# The declarations. Proposal 9.8.5 lists ten W1 items; appendix B lists the rest
# of the 待核实 quantities. Every one of them lives here, and nowhere else in
# the package is any of these values written as a literal.
# ---------------------------------------------------------------------------

REQUIRED_MEASUREMENTS: dict[str, RequiredMeasurement] = {
    m.key_template: m
    for m in [
        # -- simulator throughput and timing (9.1.3, 9.8.4) -----------------
        _decl(RequiredMeasurement(
            key_template="sim.tau_step.{family}.{task}",
            unit="s",
            description="wall-clock seconds per env.step at 20 Hz control, render off",
            proposal_ref="9.1.3, 9.8.4",
            produced_by="scripts/measure_constants.py --what tau_step",
            blocks="every CPU-hour line in the budget (9.8)",
            w1_item="W1-1",
        )),
        _decl(RequiredMeasurement(
            key_template="sim.tau_frame.{family}.{task}",
            unit="s",
            description="wall-clock seconds per rendered frame, two 84x84 cameras",
            proposal_ref="9.1.3, 9.8.4",
            produced_by="scripts/measure_constants.py --what tau_frame",
            blocks="the image materialisation line in the budget (9.8)",
            w1_item="W1-1",
        )),
        _decl(RequiredMeasurement(
            key_template="train.throughput.{arch}",
            unit="gradient_steps_per_second",
            description="training throughput used to convert epochs into GPU-hours",
            proposal_ref="9.8.1",
            produced_by="scripts/measure_constants.py --what train_throughput",
            blocks="the 5.0 GPU-h/run figure and the whole GPU budget",
            w1_item="W1-1",
        )),
        _decl(RequiredMeasurement(
            key_template="storage.gzip_ratio.{modality}",
            unit="dimensionless",
            description="observed gzip compression ratio for the given modality",
            proposal_ref="9.8.3",
            produced_by="scripts/measure_constants.py --what gzip_ratio",
            blocks="the 758 GB storage total",
            w1_item="W1-2",
        )),
        # -- task tolerances (9.5.4) ---------------------------------------
        _decl(RequiredMeasurement(
            key_template="task.tol.{task}.{geom_variant}",
            unit="m",
            description="fit tolerance tol(tau), the denominator of kappa",
            proposal_ref="9.5.4",
            produced_by="scripts/compute_task_tol.py",
            blocks="kappa, and therefore every dimensionless dose statement",
            w1_item="W1-3",
        )),
        _decl(RequiredMeasurement(
            key_template="task.tol_kind.{task}.{geom_variant}",
            unit="enum",
            description="radial_clearance or marker_halfwidth; the two are not comparable",
            proposal_ref="9.5.4",
            produced_by="scripts/compute_task_tol.py",
            blocks="kappa reporting discipline (14.5 E-6)",
            w1_item="W1-3",
        )),
        # -- geometry (8.2, 9.5.6) -----------------------------------------
        _decl(RequiredMeasurement(
            key_template="geom.clearance.{task}.{geom_variant}.{direction}",
            unit="m",
            description="cl_+ / cl_- directional clearance at the terminal pose",
            proposal_ref="8.1, 9.5.6",
            produced_by="scripts/measure_constants.py --what clearance",
            blocks="proposition 2(b) intermediate band, T8 feasibility",
            w1_item="W1-4",
        )),
        # -- robot model facts (9.2.4, 9.3) --------------------------------
        _decl(RequiredMeasurement(
            key_template="mjcf.armature.{family}.{joint}",
            unit="kg m^2",
            description="armature on each arm joint as shipped in the family MJCF",
            proposal_ref="9.2.4",
            produced_by="scripts/measure_constants.py --what armature",
            blocks="D4 shadow joint (singular inertia) and the FGW inertia feature",
            w1_item="W1-5",
        )),
        _decl(RequiredMeasurement(
            key_template="obs.proprio_native_dim.{family}",
            unit="count",
            description="width of robosuite's robot0_proprio-state for this family",
            proposal_ref="9.6.1",
            produced_by="scripts/measure_constants.py --what proprio_dim",
            blocks="the 52-dim proprio layout and every hdf5 shape",
            w1_item="W1-6",
        )),
        _decl(RequiredMeasurement(
            key_template="mjcf.agentview_pose.{task}",
            unit="pose7",
            description="agentview camera pose actually used for the task",
            proposal_ref="9.6.2",
            produced_by="scripts/measure_constants.py --what camera_pose",
            blocks="image materialisation reproducibility",
            w1_item="W1-7",
        )),
        _decl(RequiredMeasurement(
            key_template="sim.hard_reset_default",
            unit="bool",
            description="robosuite hard_reset default for the pinned version",
            proposal_ref="9.1.4",
            produced_by="scripts/measure_constants.py --what hard_reset",
            blocks="whether model-level injection has to survive an XML recompile",
            w1_item="W1-7",
        )),
        # -- defect observables (9.2.10) -----------------------------------
        _decl(RequiredMeasurement(
            key_template="defect.observable.{defect}.{level}.{family}",
            unit="defect_specific",
            description=(
                "the observable-quantity column of the defect level table; the "
                "injection parameter alone means six different doses across six families"
            ),
            proposal_ref="9.2.10",
            produced_by="scripts/measure_constants.py --what defect_observable",
            blocks="dose axes in E1/E3, and cross-family comparability of a level",
            w1_item="W1-8",
        )),
        _decl(RequiredMeasurement(
            key_template="defect.d6_stiction_offset.{family}.{joint}",
            unit="N m",
            description=(
                "torque threshold at which |qdot| first exceeds 1e-3 rad/s, "
                "constraint-based minus force-based friction; the known cost of the "
                "qfrc_applied fallback for D6"
            ),
            proposal_ref="9.2.6",
            produced_by="scripts/measure_constants.py --what stiction",
            blocks=(
                "whether D6 stays in the 36 twin pairs -- measured, and it does "
                "not; D8 holds the sixth slot and this entry is the evidence"
            ),
            w1_item="W1-8",
        )),
        _decl(RequiredMeasurement(
            key_template="defect.d7_clean_compliance",
            unit="dimensionless",
            description=(
                "delta = 0 value of D7's link_compliance. The rigid limit cannot be "
                "taken arbitrarily far the way D3a's can: MuJoCo integrates joint "
                "stiffness explicitly, so the clean twin's bending mode is capped by "
                "the 2 ms step. Measured as the divergence boundary of a compliance "
                "sweep, divided by the safety factor. Not keyed by family or task: "
                "the fragment sets stiffness per unit compiled inertia, so "
                "omega = omega_ref / sqrt(ratio) holds whatever the arm"
            ),
            proposal_ref="9.2.7, 9.2.10",
            produced_by="scripts/measure_constants.py --what d7_stability",
            blocks="D7's clean twin, hence every D7 cell and the (A4) counterexample",
            w1_item="W1-8",
            # A ratio at or above 1 puts the rigid twin's mode at or below the
            # defective one, which inverts the defect.
            valid_range=(1e-6, 1.0),
        )),
        # -- regimes (9.7.1) ------------------------------------------------
        _decl(RequiredMeasurement(
            key_template="regime.threshold.{name}.{task}.{family}",
            unit="threshold_specific",
            description="coordinate-descent calibrated regime threshold, frozen after P1",
            proposal_ref="9.7.1",
            produced_by="scripts/calibrate_regimes.py",
            blocks="every per-regime quantity, the design graph, proposition 1(e')",
            w1_item="W1-9",
        )),
        _decl(RequiredMeasurement(
            key_template="stats.tau_corr.{task}.{defect}.{regime}",
            unit="frames",
            description="autocorrelation time of the within-cell residual",
            proposal_ref="11.4",
            produced_by="scripts/measure_constants.py --what tau_corr",
            blocks="effective sample size, hence every edge weight, R_eff, lambda_2, Xi_G",
            w1_item="W1-9",
        )),
        # -- the DROID release E5 runs on (10.4.6, appendix A-27) -----------
        _decl(RequiredMeasurement(
            key_template="real.droid.institution_roster",
            unit="lab_ids",
            description=(
                "the institution identifiers actually present in the pinned DROID "
                "release, read off the downloaded metadata. 10.4.6 says 13, and E5's "
                "preregistered prediction is a count over that denominator -- "
                "'pre-window T_comp > 1.3 on at least 8 of 13'. A release exposing a "
                "different set turns that count into a different claim, so the roster "
                "is verified against the data rather than taken from the proposal"
            ),
            proposal_ref="10.4.6, 9.6.6, appendix A-27",
            produced_by="scripts/prepare_droid.py --what institution_roster",
            blocks=(
                "E5 endpoint A's 8-of-13 count and endpoint B's held-out institution "
                "folds, which are keyed on the institution id"
            ),
        )),
        # -- projection basis (9.6.4) --------------------------------------
        _decl(RequiredMeasurement(
            key_template="phi.effective_rank",
            unit="count",
            description="truncated-SVD rank of phi at 99.9% energy (392 raw columns)",
            proposal_ref="9.6.4",
            produced_by="scripts/measure_constants.py --what phi_rank",
            blocks="the E_phi gate and the (A4) hold-out R^2",
            w1_item="W1-10",
        )),
    ]
}


LITERATURE_FACTS: dict[str, LiteratureFact] = {
    f.key: f
    for f in [
        LiteratureFact(
            key="fgw.featset_baseline",
            description=(
                "node attributes the baseline embodiment-grouping method reads. Its "
                "appendix E takes URMA's local joint/foot descriptors verbatim, and "
                "URMA's d_j contains damping and a friction coefficient -- which is "
                "what forces D6 onto the qfrc_applied variant (proposal 9.2.6)"
            ),
            citation="arXiv:2602.18025 appendix E; URMA arXiv:2409.06366 appendix A",
            proposal_ref="9.3.4",
            blocks="X1, and the choice of D6 injection variant",
        ),
        LiteratureFact(
            key="fgw.hyperparameters_baseline",
            description="POT settings the baseline uses for the FGW distance",
            citation="arXiv:2602.18025 appendix E table 8",
            proposal_ref="9.3.4",
            blocks="reproducing the baseline's grouping",
        ),
    ]
}
