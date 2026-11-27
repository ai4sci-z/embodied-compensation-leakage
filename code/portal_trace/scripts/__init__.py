"""Calibration entry points.

These produce the 待核实 constants of chapter 9 and write them into
`registry/lock.json`. Nothing else in the package computes them, and nothing
reads them except through `portal_trace.lockfile`.
"""
