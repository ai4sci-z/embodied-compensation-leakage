"""Pipeline entry points: collect, materialise, train, evaluate, diagnose.

The stages are deliberately separate processes rather than one script with
flags. 9.1.3 decouples rendering from collection because rendering costs about
eight times what physics does and the same states feed several arms, and 11.3's
unblinding order applies to stages rather than to functions, so each has to be
schedulable on its own.

Submodules are not imported here. `portal_trace.runners.cli` is the console
entry point declared in `pyproject.toml`, and importing the stages eagerly would
pull in robosuite and h5py for `portal-trace config`, which needs neither.
"""

from __future__ import annotations

__all__ = [
    "backends",
    "cli",
    "collect",
    "common",
    "diagnose",
    "evaluate",
    "formal_adapter",
    "formal_checkpoint",
    "formal_data",
    "formal_training",
    "materialise",
    "robosuite_adapter",
    "source_validation",
    "train",
    "unblinding",
]
