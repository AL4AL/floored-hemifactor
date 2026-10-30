"""Whitened singular spectra for floored hemifactor licensing."""

from __future__ import annotations

from typing import Any

import numpy as np


def sym_sqrt_pinv(cov: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Σ^{1/2} via eigenclip (SPD assumed)."""
    cov = np.asarray(cov, dtype=np.float64)
    w, q = np.linalg.eigh(cov)
    w = np.clip(w, 0.0, None)
    scale = np.sqrt(w)
    # floor tiny eigenvalues for numerical stability when used as whitener
    scale = np.where(w > eps, scale, 0.0)
    return (q * scale) @ q.T


def diag_sqrt(diag_var: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(diag_var, dtype=np.float64).ravel()
    return np.sqrt(np.maximum(v, eps))


def whitened_singular_values(
    W: np.ndarray,
    cov: np.ndarray | None = None,
    diag_var: np.ndarray | None = None,
) -> np.ndarray:
    """Singular values of W Σ^{1/2}.

    Prefer full ``cov`` when available; else diagonal activation variance.
    """
    W = np.asarray(W, dtype=np.float64)
    if cov is not None:
        s_half = sym_sqrt_pinv(cov)
        M = W @ s_half
    elif diag_var is not None:
        M = W * diag_sqrt(diag_var)[None, :]
    else:
        M = W
    # economy SVD; only singular values needed
    s = np.linalg.svd(M, full_matrices=False, compute_uv=False)
    return np.asarray(s, dtype=np.float64)


def spectrum_report(
    singular_values: np.ndarray,
    d_floor: float,
    candidate_ks: list[int],
    nbytes_baseline: int,
    out_dim: int,
    in_dim: int,
    factor_bits: float = 4.0,
) -> dict[str, Any]:
    from hemifactor.spectrum.svd_license import license_rank

    lic = license_rank(singular_values, d_floor, candidate_ks)
    s = np.asarray(singular_values, dtype=np.float64)
    total_e = float(np.sum(s**2))
    pr = float((np.sum(s) ** 2) / (np.sum(s**2) + 1e-30)) if s.size else 0.0
    # rough PR on energy: (tr)^2/Fro for Gram of whitened map ~ (sum σ)^2 / sum σ^2 is not
    # participation of Σ; report spectral effective rank:
    eff_rank = float((np.sum(s**2) ** 2) / (np.sum(s**4) + 1e-30)) if s.size else 0.0

    def bytes_at_k(k: int) -> int:
        # AB with A (out,k), B (k,in)
        return int(np.ceil(k * (out_dim + in_dim) * factor_bits / 8.0))

    curve = []
    for row in lic["curve"]:
        k = row["k"]
        b = bytes_at_k(k)
        curve.append(
            {
                **row,
                "factor_nbytes_est": b,
                "byte_ratio_vs_baseline": b / max(nbytes_baseline, 1),
            }
        )
    chosen = lic["chosen_k"]
    decision = "stop"
    if chosen is not None:
        ratio = bytes_at_k(chosen) / max(nbytes_baseline, 1)
        if ratio <= 0.55:
            decision = "go_hemifactor"
        else:
            decision = "borderline_traffic"
    elif eff_rank > 0.85 * min(out_dim, in_dim):
        decision = "branch_b_monarch"
    else:
        decision = "branch_b_or_stop"

    return {
        "d_floor": d_floor,
        "total_energy": total_e,
        "effective_rank_energy": eff_rank,
        "singular_l1_over_l2_sq": pr,
        "licensed": lic["licensed"],
        "chosen_k": chosen,
        "decision": decision,
        "curve": curve,
        "factor_bits": factor_bits,
        "nbytes_baseline": nbytes_baseline,
    }
