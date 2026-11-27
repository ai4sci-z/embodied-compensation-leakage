"""The ILC demonstrator: the sole source of every ground truth (11.2).

Six pieces, in the order the data flows through them:

`refs`     the embodiment-independent task-space reference trajectory X_tau (9.4.2)
`osc`      the deterministic OSC tracker whose gains are frozen once (9.4.2, 9.4.6)
`ilc`      the iterative learning controller that produces the feedforward (9.4.2)
`query`    the off-branch query `query_clean`, the only entry point for c_true (9.6.7)
`project`  the projection of the feedforward onto phi, and the (A4) report (9.4.4-9.4.5)
`replay`   bidirectional cross replay and bit-exact replay from a manifest (10.3.1)

Two of these are zero-GPU gates that unblind before any training runs (14.5 E-17):
`project` decides whether the object of study exists at all, and `replay` decides
whether the demonstrations contain compensation. Both are importable and runnable
without a GPU and without the training stack.

`refs`, `osc` and `ilc` stand alone: they build and track the reference and need
no defect model, which is what lets the tracker gains be tuned on clean twins
before any injection exists. `query`, `project` and `replay` sit on the twin
interface of `portal_trace.defects` -- `Delta`, `build_twin_pair`,
`DefectWrapper.defect_state`, and `fields.FieldContext` -- because their subject
is the difference between two bodies.
"""

from __future__ import annotations

__all__ = ["refs", "osc", "ilc", "query", "project", "replay"]
