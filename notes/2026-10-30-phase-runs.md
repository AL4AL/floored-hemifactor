# Lab note — first full phase run (2026-10-30)

## Setup
- GGUF: Ollama `gemma4:12b` blob `sha256-1278394…606` (~6.88 GiB), digest `4eb23ef187e2`
- Chassis: RTX 3050 Laptop 6GB (GA107)
- Activation μ: Gaussian residual → SiLU(gate)·up surrogate (not hooked full-model residuals)
- D control codec: Q6-dequant → **Q4_0** round-trip (gguf Python cannot pack Q4_K)

## Outcomes
| Phase | Result |
|---|---|
| 0 | Freeze OK |
| 1 | D + participation written for layers 0,12,24,36,47 |
| 2 | Aggregate **branch-B** — no candidate k licensed under measured D |
| 3 | Down prototype **fail** quality vs D (bytes ≤0.55× OK at k=1920); monarch held-out MSE also > D |
| 4 | NumPy fused microbench only — no CUDA tok/s claim |
| 5 | Success predicate **false** (honest) |

## Interpretation
Prefer recording this **negative spectrum license** over silently switching the vow to Q2.
Initial monarch probe that matched eval exactly was train/eval leakage; after a disjoint
fit set, held-out MSE remains above `D` on layer 0 (block=64).

## Non-claims
- No ~2× tok/s
- No exact-GGUF-in-6GB
- No chat-vibe pass
