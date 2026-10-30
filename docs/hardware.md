# Hardware under test

Captured from the development laptop during the problem framing (2026-10-30). Re-measure before publishing numbers.

**Do not** treat PCIe or Intel aperture as GDDR, or brochure clocks as sustained 60 W reality — see [hard-constraints.md](hard-constraints.md) §B.

## System

| Field | Value |
|---|---|
| Machine | Lenovo LOQ 15IAX9 (`83GS`) |
| Hostname | `sajjad-LOQ-15IAX9` |
| CPU | Intel Core i5-12450HX (8 cores / 12 threads, up to 4.4 GHz) |
| RAM | ~62 GiB usable (DMI reports 64 GiB class; ~2 GiB stolen aperture) |
| OS | Ubuntu 24.04.x · kernel as installed |
| Firmware | NECN38WW |

## Discrete GPU — NVIDIA

| Field | Value |
|---|---|
| Name | GeForce RTX 3050 6GB Laptop GPU |
| Device | `0x25EC` · GN20-P0-R / GA107 Ampere |
| VRAM | **6144 MiB** total |
| Free (idle, display on Intel) | **~5792 MiB** |
| CUDA capability | 8.6 |
| CUDA cores | 2560 (20 SMs) |
| Max SM clock (reported) | 2100 MHz |
| Memory clock (reported) | 7001 MHz |
| Memory bus | 96-bit GDDR6 |
| Peak bandwidth (calc) | **~168 GB/s** |
| PCIe | Gen4 · link often **x8** (~10–13 GB/s real pinned host reads) |
| Power limit | **60 W** default · **95 W** max (firmware permitting) |
| FP32 peak (order) | ~8–11 TFLOPS brochure; sustained lower at 60 W |

### Roofline notes (batch-1 decode)

- Weight fetch of a 6.88 GiB file @ 168 GB/s ≈ **41–44 ms/token** (~23 tok/s) if fully resident.
- ~24 GFLOP/token ≪ memory time → **memory-bound**; spare FLOPs for structured factors.
- Host overflow of ~1.5 GiB @ ~12 GB/s ≈ **+120–160 ms** → hybrid offload lands ~**5–7 tok/s**.

## Integrated GPU — Intel

| Field | Value |
|---|---|
| Name | UHD Graphics (Alder Lake-S GT0.5), device `0x468b` |
| EUs | 16 (this SKU) |
| Max GT clock | ~1300 MHz (idle often lower) |
| FP32 order | ~0.33 TFLOPS at peak clock |
| Memory | GTT into system DRAM (not private 64 GB VRAM) |

Relevant only to the deferred **Janus** dual-GPU path. Primary hemifactor path aims at **NVIDIA GDDR residency**.

## Target model (local)

| Field | Value |
|---|---|
| Ollama name | `gemma4:12b` |
| Digest (list) | `4eb23ef187e2` |
| Size on disk (ollama list) | 7.6 GB |
| Measured GGUF bytes (research pass) | **7,381,382,048 B (~6.88 GiB)** |
| Parameters | 11.9B (`general.parameter_count` ≈ 1.1907e10) |
| Quant | Q4_K_M family mix (Q4_K + Q6_K tensors; emb Q6-heavy) |
| Architecture | `gemma4` · 48 blocks · emb 3840 · FFN 15360 |
| Attention | SWA 1024 · pattern 5 local + 1 global · head quirks per GGUF |
| Context field | 262144 (not a residency plan) |
| Capabilities | completion, vision, audio, tools, thinking |

**Weight budget on-die with SWA KV ≈ 320 MiB:** ~5472 MiB → need **≲ 0.78×** of current file for residency; **0.50×** for the speed doubling target.

## Software context (non-binding)

| Field | Value |
|---|---|
| Ollama | 0.34.4 (service present; not required for math experiments) |
| Driver | NVIDIA 595.x / CUDA 13.x reported by `nvidia-smi` |

Experiments may use any stack that can read the GGUF and run CUDA kernels on this GPU.
