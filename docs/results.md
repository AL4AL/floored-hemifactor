# Results

First full pipeline run: **2026-10-30** on local `gemma4:12b` GGUF (Ollama digest `4eb23ef187e2`).

| Gate | Outcome |
|---|---|
| Spectrum aggregate | **branch-B** (no candidate `k` licensed under measured `D`) |
| Down-proj prototype | **fail** quality (`activation_mse` ≫ `D`); byte_ratio ≈ **0.38** at `k=1920`, 4-bit factors |
| Monarch probe (held-out) | Also above `D` on layer 0 with block=64 |
| Success predicate (theory §7) | **False** |

See `artifacts/phase*_status.json` and `artifacts/rank_license.md`.

**Caveats (binding):** Phase 1 μ is an MLP surrogate (Gaussian residual → SiLU(gate)·up), not hooked full-model residuals. Distortion control uses Q4_0 round-trip (not Q4_K pack). Phase 4 timing is CPU NumPy fused path only.

Do not claim ~2× tok/s or full-model success: e2e NLL and GDDR-resident device timing are not cleared. Prefer this **negative spectrum license** over silently changing the vow to smaller quants.
