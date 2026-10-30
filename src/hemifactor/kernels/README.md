# Kernels

Phase 4 lives here: fused factor GEMV that **never materializes dense W in GDDR**.

Until then, use NumPy / PyTorch reference paths in `factorize/` and `eval/`.

Hard constraints: B1, B2, D1, D4 in `docs/hard-constraints.md`.
