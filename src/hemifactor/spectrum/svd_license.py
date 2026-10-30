"""Whitened spectrum tests vs distortion floor D (Phase 2)."""

from __future__ import annotations

from typing import Any

import numpy as np


def discarded_energy(singular_values: np.ndarray, k: int) -> float:
    """Sum of σ_i^2 for i > k (0-based truncation keep-k)."""
    s = np.asarray(singular_values, dtype=np.float64).ravel()
    if k >= s.size:
        return 0.0
    return float(np.sum(s[k:] ** 2))


def license_rank(
    singular_values: np.ndarray,
    d_floor: float,
    candidate_ks: list[int],
) -> dict[str, Any]:
    """Return smallest k in candidates with discarded energy <= d_floor, else None."""
    rows = []
    chosen = None
    for k in sorted(candidate_ks):
        e = discarded_energy(singular_values, k)
        ok = e <= d_floor
        rows.append({"k": k, "discarded_energy": e, "ok": ok})
        if ok and chosen is None:
            chosen = k
    return {
        "d_floor": d_floor,
        "chosen_k": chosen,
        "licensed": chosen is not None,
        "curve": rows,
    }
