"""
Additional metrics: stubs and wrappers.
"""

from __future__ import annotations

from typing import Dict


def aggregate_metrics(metrics: Dict[str, float]) -> float:
    """Simple aggregation of metrics into a single score."""
    keys = [k for k in ("accuracy", "f1", "convergence") if k in metrics]
    if not keys:
        return 0.0
    return sum(metrics[k] for k in keys) / len(keys)