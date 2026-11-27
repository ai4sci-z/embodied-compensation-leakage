"""The preregistration file, its git tag hook, and the append-only amendment log.

Proposal 11.3 requires a preregistration document to be committed before any
training runs, to live in `configs/preregistration/` under a git tag, and to
consist of **16 items**; a missing item means the endpoints that item covers may
not enter the main narrative. That count is enforced here rather than checked by
eye, and the check is by item number so that a renumbered or duplicated item
fails too.

The amendment log is append-only in the operational sense the proposal asks
for: every amendment records whether it happened before or after unblinding, and
`AmendmentLog.append` refuses to write if the entries already on disk differ
from the ones it was loaded with. An amendment made after unblinding is not
forbidden -- it is required to be visible.

The git tag hook is not decoration. 11.3 says the training arms are unblinded
*after* the preregistration file is tagged, and `runners.unblinding` asks this
module whether that has happened.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "PreregistrationError",
    "PreregistrationIncomplete",
    "PreregistrationNotTagged",
    "AmendmentLogTampered",
    "PreregistrationItem",
    "Preregistration",
    "Amendment",
    "AmendmentLog",
    "REQUIRED_ITEM_COUNT",
    "git_commit",
    "git_tag_exists",
    "git_tag_commit",
]

# 11.3, last bullet of the preregistration clause.
REQUIRED_ITEM_COUNT = 16


class PreregistrationError(RuntimeError):
    """Base class for preregistration problems."""


class PreregistrationIncomplete(PreregistrationError):
    """An item of the 16 is missing, empty, or duplicated.

    Fatal on purpose: 11.3 says the endpoints related to a missing item may not
    enter the main narrative, and the only way to make that stick is to refuse
    to produce those endpoints at all.
    """


class PreregistrationNotTagged(PreregistrationError):
    """The preregistration file has not been tagged in git.

    11.3 puts the training-arm unblinding after the tag. A run that would
    produce a training-arm number before the tag exists cannot be made
    compliant afterwards, so it does not start.
    """


class AmendmentLogTampered(PreregistrationError):
    """An existing amendment entry changed. The log is append-only."""


# ---------------------------------------------------------------------------
# git hooks
# ---------------------------------------------------------------------------


def _git(args: list[str], repo_root: Path) -> str:
    """Run a read-only git command, returning stdout stripped.

    Raises `PreregistrationError` when git is unavailable or the command fails,
    rather than returning a sentinel: an unverifiable tag is the same situation
    as an absent one for 11.3's purposes.
    """
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise PreregistrationError(
            "git is not on PATH, so the preregistration tag cannot be verified. "
            "11.3 ties the training-arm unblinding to that tag; install git or run "
            "on the cluster image, do not disable the check"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise PreregistrationError(
            f"git {' '.join(args)} failed in {repo_root}: {exc.stderr.strip()}"
        ) from exc
    return out.stdout.strip()


def git_commit(repo_root: Path) -> str:
    """The commit that would be recorded next to a result (11.3 version locking)."""
    return _git(["rev-parse", "HEAD"], repo_root)


def git_tag_exists(tag: str, repo_root: Path) -> bool:
    return tag in _git(["tag", "--list", tag], repo_root).splitlines()


def git_tag_commit(tag: str, repo_root: Path) -> str:
    if not git_tag_exists(tag, repo_root):
        raise PreregistrationNotTagged(
            f"git tag {tag!r} does not exist in {repo_root}.\n"
            "11.3: the preregistration file goes into configs/preregistration/ and is "
            "tagged; training arms are unblinded only after that. Create the tag on "
            "the commit that froze the file:\n"
            f"    git tag -a {tag} -m 'preregistration frozen'"
        )
    return _git(["rev-list", "-n", "1", tag], repo_root)


# ---------------------------------------------------------------------------
# the document
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreregistrationItem:
    """One of the 16 items.

    `blocks` names the endpoints that may not enter the main narrative if this
    item is missing -- the sentence 11.3 attaches to the count.
    """

    number: int
    title: str
    text: str
    proposal_ref: str
    blocks: tuple[str, ...]
    source: str = ""

    def __post_init__(self) -> None:
        if not (1 <= self.number <= REQUIRED_ITEM_COUNT):
            raise PreregistrationIncomplete(
                f"item number {self.number} is outside 1..{REQUIRED_ITEM_COUNT}"
            )
        for name in ("title", "text", "proposal_ref"):
            if not getattr(self, name).strip():
                raise PreregistrationIncomplete(
                    f"preregistration item {self.number}: {name} is empty"
                )
        if not self.blocks:
            raise PreregistrationIncomplete(
                f"preregistration item {self.number}: no endpoint listed under "
                "`blocks`. 11.3 makes a missing item block its related endpoints, "
                "so an item that blocks nothing has no enforceable content"
            )


@dataclass(frozen=True)
class Preregistration:
    """The whole document: 16 items, the hypothesis family, the exits, the tag."""

    version: str
    frozen_at: str
    git_tag: str
    items: tuple[PreregistrationItem, ...]
    hypotheses: dict[str, str]
    exits: dict[str, str]
    primary_family: tuple[str, ...]
    holm_bonferroni_order: tuple[str, ...]
    verdict_rules: dict[str, str]
    ci_thresholds: dict[str, float]
    unblinding_order: tuple[str, ...]
    # Numbers 10.10 retired. Kept, not deleted: a reader who finds T3 followed by
    # T5 has to be able to see why, and a retired number must never be reused.
    retired_hypotheses: dict[str, str] = field(default_factory=dict)
    path: Path | None = None

    def __post_init__(self) -> None:
        self.check_complete()
        reused = sorted(set(self.hypotheses) & set(self.retired_hypotheses))
        if reused:
            raise PreregistrationIncomplete(
                f"{', '.join(reused)} is registered as a live hypothesis and as a "
                "retired one. A retired number cannot be reused: a verdict filed "
                "against it would be read as a verdict on the retired prediction"
            )
        if len(self.primary_family) != 5:
            raise PreregistrationIncomplete(
                "11.3 fixes the primary hypothesis family at 5 members with their "
                f"Holm-Bonferroni ranks; found {len(self.primary_family)}"
            )
        if sorted(self.holm_bonferroni_order) != sorted(self.primary_family):
            raise PreregistrationIncomplete(
                "the Holm-Bonferroni order and the primary family list different "
                "members; both are written down so that neither can drift"
            )
        for verdict in ("significant", "not_significant", "inconclusive"):
            if verdict not in self.verdict_rules:
                raise PreregistrationIncomplete(
                    f"11.3 allows exactly three verdicts and requires the condition "
                    f"for each to be preregistered; {verdict!r} has none"
                )

    def check_complete(self) -> None:
        """Raise unless all 16 items are present exactly once."""
        numbers = [i.number for i in self.items]
        duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
        missing = sorted(set(range(1, REQUIRED_ITEM_COUNT + 1)) - set(numbers))
        if duplicates or missing or len(self.items) != REQUIRED_ITEM_COUNT:
            lines = [
                f"the preregistration file must carry {REQUIRED_ITEM_COUNT} items "
                f"(11.3); found {len(self.items)}"
            ]
            if missing:
                lines.append(f"  missing item numbers: {missing}")
                for n in missing:
                    lines.append(f"    item {n}: {ITEM_TITLES[n]}")
            if duplicates:
                lines.append(f"  duplicated item numbers: {duplicates}")
            lines.append(
                "  a missing item means the endpoints it covers may not enter the "
                "main narrative, so the run does not start"
            )
            raise PreregistrationIncomplete("\n".join(lines))

    def item(self, number: int) -> PreregistrationItem:
        for i in self.items:
            if i.number == number:
                return i
        raise PreregistrationIncomplete(f"preregistration item {number} is missing")

    def require_items(self, numbers: tuple[int, ...]) -> None:
        for n in numbers:
            self.item(n)

    def threshold(self, name: str) -> float:
        """Read a preregistered threshold. No default, by CONTRACT 3.1."""
        try:
            return self.ci_thresholds[name]
        except KeyError:
            raise PreregistrationIncomplete(
                f"threshold {name!r} is not in the preregistration file "
                f"({self.path}). It is a preregistered bound, so there is no default "
                "to fall back to: add it to ci_thresholds and amend the log"
            ) from None

    def require_tag(self, repo_root: Path) -> str:
        """Return the commit the preregistration tag points at, or raise."""
        return git_tag_commit(self.git_tag, repo_root)


# Titles are duplicated here so that `check_complete` can name what is missing
# without having the file that is missing it. Kept in the module rather than the
# YAML for the same reason.
ITEM_TITLES: dict[int, str] = {
    1: "hypotheses under test, by T number",
    2: "endpoint definitions and their estimators",
    3: "criteria: thresholds, tests, significance level, multiple-comparison handling",
    4: "exits, by F number",
    5: "the surrogate defect set S, its construction hash and the four-dimensional "
       "matching tolerance table",
    6: "the primary endpoint is Lambda^part after nuisance-basis partial regression",
    7: "T9 uses lambda_emp_res and lambda_emp_con, with the registered alternative",
    8: "T1 uses the three-term decomposition of Lambda^orc",
    9: "the regime caliber: E_r, N_0 = 200 and the sensitivity sweep",
    10: "the per-arm pool composition, including delta = 0 membership",
    11: "the T8 three-way feasibility check, per geometry variant",
    12: "E3's dose conclusion on the balanced panel",
    13: "the main decision endpoint is the success-rate paired difference",
    14: "the subset list and the design (M, D, S) of every arm",
    15: "the conditions under which an endpoint is declared inconclusive",
    16: "T3's exploratory status",
}


# ---------------------------------------------------------------------------
# amendments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Amendment:
    """One appended change to the preregistration.

    `before_unblinding` is required and is the whole point of the log: 11.3 says
    subsequent changes are recorded by appending and must state whether the
    change happened before or after the data were unblinded.
    """

    index: int
    recorded_at: str
    item_number: int | None
    summary: str
    rationale: str
    before_unblinding: bool
    git_commit: str
    author: str
    unblinded_stages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.summary.strip() or not self.rationale.strip():
            raise PreregistrationError(
                f"amendment {self.index}: summary and rationale are both required"
            )
        if self.item_number is not None and not (1 <= self.item_number <= REQUIRED_ITEM_COUNT):
            raise PreregistrationError(
                f"amendment {self.index}: item_number {self.item_number} is outside "
                f"1..{REQUIRED_ITEM_COUNT}"
            )
        if not self.before_unblinding and not self.unblinded_stages:
            raise PreregistrationError(
                f"amendment {self.index}: an amendment made after unblinding must name "
                "which stages had already been unblinded. That list is what lets a "
                "reader judge the amendment; 'after unblinding' alone does not"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "recorded_at": self.recorded_at,
            "item_number": self.item_number,
            "summary": self.summary,
            "rationale": self.rationale,
            "before_unblinding": self.before_unblinding,
            "git_commit": self.git_commit,
            "author": self.author,
            "unblinded_stages": list(self.unblinded_stages),
        }

    @classmethod
    def from_json(cls, blob: dict[str, Any]) -> Amendment:
        required = (
            "index", "recorded_at", "item_number", "summary", "rationale",
            "before_unblinding", "git_commit", "author",
        )
        missing = [k for k in required if k not in blob]
        if missing:
            raise PreregistrationError(
                f"amendment entry is missing {missing}; in particular "
                "`before_unblinding` has no default -- an amendment whose timing "
                "relative to unblinding is unknown is not a recorded amendment"
            )
        return cls(
            index=int(blob["index"]),
            recorded_at=str(blob["recorded_at"]),
            item_number=blob["item_number"],
            summary=str(blob["summary"]),
            rationale=str(blob["rationale"]),
            before_unblinding=bool(blob["before_unblinding"]),
            git_commit=str(blob["git_commit"]),
            author=str(blob["author"]),
            unblinded_stages=tuple(blob.get("unblinded_stages") or ()),
        )


@dataclass
class AmendmentLog:
    """Append-only log of amendments to the preregistration file.

    "Append-only" is enforced two ways. `append` re-reads the file and refuses
    to write if the entries on disk differ from the ones this object was loaded
    with, and it rejects an index that is not the next one. Neither stops a
    determined editor with a text editor -- nothing in a repository can -- but
    both stop the ordinary failure mode, which is a script that rewrites the log
    while thinking it is adding to it.
    """

    path: Path
    entries: tuple[Amendment, ...] = ()

    HEADER = (
        "Append-only amendment log for the preregistration file (proposal 11.3). "
        "Every entry states whether the amendment was made before or after the "
        "data were unblinded; entries made after unblinding name the stages that "
        "had already been unblinded."
    )

    @classmethod
    def load(cls, path: Path | str) -> AmendmentLog:
        import yaml

        p = Path(path)
        if not p.exists():
            return cls(path=p, entries=())
        with p.open("r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        raw = doc.get("amendments") or []
        entries = tuple(Amendment.from_json(b) for b in raw)
        expected = tuple(range(1, len(entries) + 1))
        if tuple(e.index for e in entries) != expected:
            raise AmendmentLogTampered(
                f"{p}: amendment indices are {[e.index for e in entries]}, expected "
                f"{list(expected)}. A gap or a reordering means the log was rewritten"
            )
        return cls(path=p, entries=entries)

    def append(
        self,
        *,
        summary: str,
        rationale: str,
        before_unblinding: bool,
        git_commit_sha: str,
        author: str,
        item_number: int | None = None,
        unblinded_stages: tuple[str, ...] = (),
    ) -> Amendment:
        on_disk = AmendmentLog.load(self.path)
        if on_disk.entries != self.entries:
            raise AmendmentLogTampered(
                f"{self.path} changed since it was loaded; refusing to append on top "
                "of an unknown history. Reload the log and re-apply the amendment"
            )
        amendment = Amendment(
            index=len(self.entries) + 1,
            recorded_at=datetime.now(UTC).isoformat(timespec="seconds"),
            item_number=item_number,
            summary=summary,
            rationale=rationale,
            before_unblinding=before_unblinding,
            git_commit=git_commit_sha,
            author=author,
            unblinded_stages=unblinded_stages,
        )
        self.entries = (*self.entries, amendment)
        self._write()
        return amendment

    def _write(self) -> None:
        import yaml

        self.path.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "header": self.HEADER,
            "amendments": [a.to_json() for a in self.entries],
        }
        with self.path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(doc, fh, allow_unicode=True, sort_keys=False)

    @property
    def after_unblinding(self) -> tuple[Amendment, ...]:
        return tuple(a for a in self.entries if not a.before_unblinding)
