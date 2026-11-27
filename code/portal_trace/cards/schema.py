"""Card schemas and the completeness gate of proposal 11.4.

11.4 opens with the rule that makes cards more than documentation: the listed
fields are a *minimum* set, and "if any one of them is missing, that object may
not be used in the computation of any preregistered endpoint". This module is
where that rule is mechanical rather than aspirational.

Three pieces:

* `FieldSpec` -- one minimum field, with the section that requires it and the
  sentence 10.7.9 gives for what becomes uninterpretable without it. That
  sentence is the exception message, so a missing field explains itself.
* `CardSchema` -- an ordered set of `FieldSpec`, with `validate` (report every
  problem at once) and `require_complete` (raise).
* `Card` -- the base of `DataCard` and `ModelCard`. `to_json` runs the schema
  first, so an incomplete card cannot be serialised, and `require_admissible`
  is what `eval.endpoints` calls before it will accept the object.

`MISSING` exists because `None` is a legal value for some fields (a clean twin
has no `defect_impl_variant`). A field that is genuinely inapplicable is set to
`NotApplicable(reason=...)`, which is recorded; a field nobody has filled in is
`MISSING` and blocks the object. The distinction is the whole point: 11.4 is
defeated by writing `None` everywhere and calling the card complete.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "MISSING",
    "NotApplicable",
    "FieldKind",
    "FieldSpec",
    "CardSchema",
    "Card",
    "CardIncompleteError",
    "CardValidationError",
    "FieldProblem",
]


class _Missing:
    """Sentinel for "nobody has filled this in"."""

    _instance: _Missing | None = None

    def __new__(cls) -> _Missing:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING = _Missing()


@dataclass(frozen=True)
class NotApplicable:
    """An explicit, reasoned declaration that a minimum field does not apply.

    Accepted only where the `FieldSpec` says it may be. The reason is stored and
    printed with the card, so "not applicable" is a claim on the record rather
    than a way of emptying a field.
    """

    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("NotApplicable needs a reason; an empty one is a blank field")


class FieldKind(Enum):
    TEXT = "text"
    NUMBER = "number"
    INTEGER = "integer"
    BOOL = "bool"
    ENUM = "enum"
    SEQUENCE = "sequence"
    MAPPING = "mapping"


@dataclass(frozen=True)
class FieldSpec:
    """One minimum field of 11.4.

    `blocks` is the consequence of leaving it out, taken from 11.4 or 10.7.9.
    It goes into the exception message because the point of the field list is
    that each entry has a named failure, not that the list is long.
    """

    name: str
    kind: FieldKind
    description: str
    proposal_ref: str
    blocks: str
    allowed: tuple[Any, ...] | None = None
    may_be_not_applicable: bool = False
    min_length: int = 1

    def check(self, value: Any) -> str | None:
        """Return a problem description, or None if the value is acceptable."""
        if value is MISSING:
            return f"missing (blocks: {self.blocks})"
        if isinstance(value, NotApplicable):
            if not self.may_be_not_applicable:
                return (
                    "declared not applicable, but 11.4 lists it as a minimum field for "
                    f"every object of this kind (blocks: {self.blocks})"
                )
            return None
        if value is None:
            return (
                "is None; use NotApplicable(reason=...) if it genuinely does not apply, "
                "so the claim is on the record"
            )
        if self.kind is FieldKind.TEXT:
            if not isinstance(value, str) or not value.strip():
                return f"must be a non-empty string, got {value!r}"
        elif self.kind is FieldKind.NUMBER:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return f"must be a number, got {type(value).__name__}"
        elif self.kind is FieldKind.INTEGER:
            if isinstance(value, bool) or not isinstance(value, int):
                return f"must be an integer, got {type(value).__name__}"
        elif self.kind is FieldKind.BOOL:
            if not isinstance(value, bool):
                return f"must be a bool, got {type(value).__name__}"
        elif self.kind is FieldKind.ENUM:
            if self.allowed is None:
                raise AssertionError(f"{self.name}: ENUM field declared without `allowed`")
            if value not in self.allowed:
                return f"must be one of {list(self.allowed)}, got {value!r}"
        elif self.kind is FieldKind.SEQUENCE:
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                return f"must be a sequence, got {type(value).__name__}"
            if len(value) < self.min_length:
                return f"needs at least {self.min_length} entries, got {len(value)}"
        elif self.kind is FieldKind.MAPPING:
            if not isinstance(value, Mapping):
                return f"must be a mapping, got {type(value).__name__}"
            if len(value) < self.min_length:
                return f"needs at least {self.min_length} entries, got {len(value)}"
        return None


@dataclass(frozen=True)
class FieldProblem:
    field_name: str
    problem: str
    proposal_ref: str

    def __str__(self) -> str:
        return f"  {self.field_name} [{self.proposal_ref}]: {self.problem}"


class CardValidationError(ValueError):
    """A card carries a field the schema does not know about, or a malformed value."""


class CardIncompleteError(ValueError):
    """A card is missing at least one 11.4 minimum field.

    Raised by `Card.to_json` and by `Card.require_admissible`, which is what
    disqualifies the object from every preregistered endpoint.
    """

    def __init__(self, card_kind: str, subject: str, problems: Sequence[FieldProblem],
                 *, endpoint: str | None = None) -> None:
        self.card_kind = card_kind
        self.subject = subject
        self.problems = tuple(problems)
        self.endpoint = endpoint
        super().__init__(str(self))

    def __str__(self) -> str:
        head = f"{self.card_kind} for {self.subject} is missing 11.4 minimum fields"
        if self.endpoint is not None:
            head += (
                f"; 11.4 therefore disqualifies it from endpoint {self.endpoint!r} "
                "(a card missing any minimum field may not be used in the computation "
                "of any preregistered endpoint)"
            )
        return "\n".join([head + ":", *[str(p) for p in self.problems]])


@dataclass(frozen=True)
class CardSchema:
    """The minimum field set for one kind of card."""

    kind: str
    granularity: str
    proposal_ref: str
    fields: tuple[FieldSpec, ...]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for spec in self.fields:
            if spec.name in seen:
                raise AssertionError(f"{self.kind}: duplicate field {spec.name!r}")
            seen.add(spec.name)

    def __iter__(self) -> Iterator[FieldSpec]:
        return iter(self.fields)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.fields)

    def spec(self, name: str) -> FieldSpec:
        for s in self.fields:
            if s.name == name:
                return s
        raise KeyError(name)

    def validate(self, payload: Mapping[str, Any]) -> tuple[FieldProblem, ...]:
        """Every problem at once, so a card is fixed in one pass, not twenty."""
        unknown = sorted(set(payload) - set(self.names))
        if unknown:
            raise CardValidationError(
                f"{self.kind}: fields {unknown} are not in the 11.4 minimum set. Extra "
                "material belongs in `notes`; silently accepting unknown keys would let "
                "a typo in a minimum field name look like a filled-in card."
            )
        problems = []
        for spec in self.fields:
            problem = spec.check(payload.get(spec.name, MISSING))
            if problem is not None:
                problems.append(FieldProblem(spec.name, problem, spec.proposal_ref))
        return tuple(problems)


class Card:
    """Base for the two 11.4 cards.

    Subclasses supply `SCHEMA` and `subject`. Everything that enforces 11.4 is
    here so the two cards cannot drift apart.
    """

    SCHEMA: CardSchema

    def __init__(self, values: Mapping[str, Any], notes: str = "") -> None:
        self.values: dict[str, Any] = dict(values)
        self.notes = notes

    @property
    def subject(self) -> str:
        raise NotImplementedError

    def problems(self) -> tuple[FieldProblem, ...]:
        return self.SCHEMA.validate(self.values) + self._cross_field_problems()

    def _cross_field_problems(self) -> tuple[FieldProblem, ...]:
        """Checks that span fields; several 11.4 rows are consistency requirements.

        `alpha_demo` and `alpha_frame` agreeing bit for bit, or the proprio width
        equalling its own component breakdown, cannot be expressed per field.
        Subclasses that have none return the empty tuple.
        """
        return ()

    @property
    def is_complete(self) -> bool:
        return not self.problems()

    def require_complete(self) -> None:
        problems = self.problems()
        if problems:
            raise CardIncompleteError(self.SCHEMA.kind, self.subject, problems)

    def require_admissible(self, endpoint: str) -> None:
        """The 11.4 gate an endpoint calls before it will read this object."""
        problems = self.problems()
        if problems:
            raise CardIncompleteError(
                self.SCHEMA.kind, self.subject, problems, endpoint=endpoint
            )

    def to_json(self) -> dict[str, Any]:
        """Serialise -- after the completeness check, never before."""
        self.require_complete()
        out: dict[str, Any] = {
            "card_kind": self.SCHEMA.kind,
            "granularity": self.SCHEMA.granularity,
            "proposal_ref": self.SCHEMA.proposal_ref,
            "subject": self.subject,
            "fields": {},
        }
        for spec in self.SCHEMA:
            value = self.values[spec.name]
            if isinstance(value, NotApplicable):
                out["fields"][spec.name] = {
                    "not_applicable": True,
                    "reason": value.reason,
                }
            else:
                out["fields"][spec.name] = value
        if self.notes:
            out["notes"] = self.notes
        return out

    def dumps(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_json(), indent=indent, ensure_ascii=False, sort_keys=True)

    def missing_report(self) -> str:
        problems = self.problems()
        if not problems:
            return f"{self.SCHEMA.kind} for {self.subject}: complete"
        return "\n".join(
            [f"{self.SCHEMA.kind} for {self.subject}: {len(problems)} problem(s)"]
            + [str(p) for p in problems]
        )


def require_all_admissible(cards: Iterable[Card], *, endpoint: str) -> None:
    """Gate a whole bundle at once; the first incomplete card stops the endpoint."""
    for card in cards:
        card.require_admissible(endpoint)
