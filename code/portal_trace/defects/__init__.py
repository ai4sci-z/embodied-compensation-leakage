"""Defect injection: the analytic representation of delta and how it is written.

`params` holds the catalogue (the 9.2.0 field matrix and the 9.2.10 level
table); everything else in this package derives from it. The clean twin is not a
separate class -- it is `apply(delta)` evaluated at delta = 0, running the same
code path, so that a bug which perturbs floating point during injection shows up
on the clean side too instead of hiding behind an early return.
"""

from portal_trace.defects.params import (
    CORE_DEFECTS,
    DEFECTS,
    DEG,
    DefectLevel,
    DefectSpec,
    HarmClass,
    InjectionLayer,
    Priority,
    WritableField,
    core_defect_codes,
    defect,
)

__all__ = [
    "DEFECTS",
    "DEG",
    "CORE_DEFECTS",
    "DefectLevel",
    "DefectSpec",
    "HarmClass",
    "InjectionLayer",
    "Priority",
    "WritableField",
    "core_defect_codes",
    "defect",
]
