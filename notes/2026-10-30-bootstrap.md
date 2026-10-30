# Lab note — 2026-10-30

## Origin

Conversation on fitting local `gemma4:12b` entirely on RTX 3050 6GB:

1. VRAM shortfall (~6.88 GiB weights vs 6 GB) is hard for exact GGUF residency.
2. Janus (3050 + Intel column-split) preserves codes but is not a clear win vs Ollama CPU offload on this HX CPU.
3. Open problem reframed as **operator compression on activation law μ** with distortion floor `D` from Q4.
4. Literature check: ASVD / SVD-LLM / QuIP# / AQLM / LACE — **no** demonstrated half-weight Q4-parity result on 7–12B.

## Repo created

`/home/sajjad/PP/floored-hemifactor` with theory + plan + hardware + related-work.

## Next physical action

Freeze exact GGUF path/hash; start Phase 1 calibration harness under `src/`.
