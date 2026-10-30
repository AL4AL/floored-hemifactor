"""Branch-B Monarch / block-diagonal×permutation factors (least-squares on μ)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MonarchFactors:
    """Depth-2 block-diagonal factors with a fixed permutation (research probe)."""

    R_blocks: list[np.ndarray]
    L_blocks: list[np.ndarray]
    perm: np.ndarray
    in_dim: int
    out_dim: int
    block: int

    def matvec(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        pieces = []
        offset = 0
        for b in self.R_blocks:
            d = b.shape[1]
            pieces.append(b @ x[offset : offset + d])
            offset += d
        mid = np.concatenate(pieces, axis=0)
        mid_p = mid[self.perm]
        outs = []
        offset = 0
        for b in self.L_blocks:
            d = b.shape[1]
            outs.append(b @ mid_p[offset : offset + d])
            offset += d
        return np.concatenate(outs, axis=0)

    def matvec_batch(self, x: np.ndarray) -> np.ndarray:
        return np.stack([self.matvec(row) for row in np.asarray(x, dtype=np.float64)], axis=0)

    def nbytes_at_bits(self, bits: float) -> int:
        n = sum(b.size for b in self.R_blocks) + sum(b.size for b in self.L_blocks)
        return int(np.ceil(n * bits / 8.0))


def _fit_block_size(in_dim: int, out_dim: int, block: int) -> int:
    if in_dim % block == 0 and out_dim % block == 0:
        return block
    for b in range(min(block, in_dim, out_dim), 0, -1):
        if in_dim % b == 0 and out_dim % b == 0:
            return b
    return 1


def fit_monarch(
    W: np.ndarray,
    x: np.ndarray,
    block: int = 64,
    rng: np.random.Generator | None = None,
) -> MonarchFactors:
    """Fit block-diagonal R, L on batches by local SVD + LS."""
    rng = rng or np.random.default_rng(0)
    W = np.asarray(W, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    out_dim, in_dim = W.shape
    block = _fit_block_size(in_dim, out_dim, block)
    n_in_blocks = in_dim // block
    n_out_blocks = out_dim // block
    y = x @ W.T

    R_blocks: list[np.ndarray] = []
    mid_cols: list[np.ndarray] = []
    for i in range(n_in_blocks):
        xb = x[:, i * block : (i + 1) * block]
        Wc = W[:, i * block : (i + 1) * block]
        _u, _s, vt = np.linalg.svd(Wc, full_matrices=False)
        R = np.zeros((block, block), dtype=np.float64)
        k = min(block, vt.shape[0])
        R[:k, :] = vt[:k, :]
        R_blocks.append(R)
        mid_cols.append(xb @ R.T)

    mid = np.concatenate(mid_cols, axis=1)
    perm = rng.permutation(in_dim)
    mid_p = mid[:, perm]

    L_blocks: list[np.ndarray] = []
    for j in range(n_out_blocks):
        yb = y[:, j * block : (j + 1) * block]
        src = j % n_in_blocks
        mb = mid_p[:, src * block : (src + 1) * block]
        beta, *_ = np.linalg.lstsq(mb, yb, rcond=None)  # (block, block): yb ≈ mb @ beta
        L_blocks.append(beta.T)

    return MonarchFactors(
        R_blocks=R_blocks,
        L_blocks=L_blocks,
        perm=perm,
        in_dim=in_dim,
        out_dim=out_dim,
        block=block,
    )
