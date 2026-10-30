"""Byte/token traffic accounting (hard-constraints A4, B3, D4)."""

from __future__ import annotations

from typing import Any


def bytes_per_token(
    weight_or_factor_bytes: float,
    chart_reread_bytes: float = 0.0,
    tail_reread_bytes: float = 0.0,
    kv_bytes: float = 0.0,
) -> dict[str, Any]:
    total = weight_or_factor_bytes + chart_reread_bytes + tail_reread_bytes + kv_bytes
    return {
        "weight_or_factor_bytes": weight_or_factor_bytes,
        "chart_reread_bytes": chart_reread_bytes,
        "tail_reread_bytes": tail_reread_bytes,
        "kv_bytes": kv_bytes,
        "bytes_per_token": total,
    }


def ratio_vs_baseline(candidate_bpt: float, baseline_bpt: float) -> float:
    return candidate_bpt / max(baseline_bpt, 1e-9)


def residency_ok(
    working_set_mib: float,
    usable_mib: float,
) -> bool:
    return working_set_mib <= usable_mib
