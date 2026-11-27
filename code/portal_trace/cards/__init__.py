"""Data cards and model cards, and the 11.4 admissibility gate.

Proposal 11.4 sets the granularity ("one object that can be cited on its own":
a data card per (embodiment, task), a model card per training arm) and the rule
that gives the cards teeth -- an object missing any minimum field may not be
used in the computation of any preregistered endpoint.

`require_admissible` is that rule as a function call. `eval.endpoints` builds an
`EndpointProvenance` out of cards and cannot be constructed from incomplete
ones, so the gate sits on the path into the endpoint rather than in a checklist
next to it.
"""

from __future__ import annotations

from portal_trace.cards.data_card import (
    CLIP_SATURATION_ANNOTATION_THRESHOLD,
    DATA_CARD_SCHEMA,
    ILC_OUTCOMES,
    DataCard,
)
from portal_trace.cards.model_card import (
    DISCRIMINATOR_ERROR_BUDGET,
    DISCRIMINATOR_INFORMATION_GAIN_FLOOR_BITS,
    MIN_EPOCHS_PER_CONFIG,
    MODEL_CARD_SCHEMA,
    VERDICT_VOCABULARY,
    ModelCard,
)
from portal_trace.cards.schema import (
    MISSING,
    Card,
    CardIncompleteError,
    CardSchema,
    CardValidationError,
    FieldKind,
    FieldProblem,
    FieldSpec,
    NotApplicable,
    require_all_admissible,
)

__all__ = [
    "MISSING",
    "NotApplicable",
    "Card",
    "CardSchema",
    "CardIncompleteError",
    "CardValidationError",
    "FieldKind",
    "FieldProblem",
    "FieldSpec",
    "require_all_admissible",
    "DATA_CARD_SCHEMA",
    "DataCard",
    "CLIP_SATURATION_ANNOTATION_THRESHOLD",
    "ILC_OUTCOMES",
    "MODEL_CARD_SCHEMA",
    "ModelCard",
    "VERDICT_VOCABULARY",
    "MIN_EPOCHS_PER_CONFIG",
    "DISCRIMINATOR_ERROR_BUDGET",
    "DISCRIMINATOR_INFORMATION_GAIN_FLOOR_BITS",
]
