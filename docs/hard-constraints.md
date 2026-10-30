# Hard constraints (where LLMs usually lie to themselves)

This document is **binding** for the project. Before proposing a “clever” fix, check it against every item below. Most failed ideas in the brainstorming phase violated one of these.

Agents and humans both fail here. Treat this as a pre-flight checklist for every design note, PR, and result claim.

---

## A. Information and storage

### A1. Entropy does not yield to optimism
Already-quantized Q4_K/Q6_K weights are high-entropy. **Lossless** recompression of the GGUF typically recovers **~1–3%**, not the **≳20%** needed to drop ~6.88 GiB into ~5.5 GiB usable GDDR **and** leave KV/scratch.

**Forbidden claim:** “a dedicated driver / ANS / LZ / holography stores the exact codes in 6 GB.”

### A2. Spare FLOPs ≠ spare bits
Batch-1 decode on this 3050 is memory-bound: ~24 GFLOP/token vs ~40+ ms of weight traffic. Extra ALUs can evaluate a **short program**. They cannot conjure missing weight information that the program’s description does not contain.

**Forbidden claim:** “idle CUDA cores regenerate the missing gigabyte.”

### A3. Uniform approximation on all of ℝⁿ is closed at half size
Half the entries of a full-rank `m×n` map are not a function of the other half for arbitrary inputs. Any 2× story **must** restrict the domain (activation law `μ`) or change the algebra (structured factors), and must **measure** that restriction.

### A4. “Half the parameters” ≠ “half the bytes”
Embeddings may stay Q6; norms stay FP32; charts are re-read every layer; KV and CUDA context exist. Byte/token accounting must include **rereads, KV, scratch**, not only `numel(W)/2`.

---

## B. Memory hierarchy and buses

### B1. PCIe is not GDDR
This chassis: GDDR ~**168 GB/s**, PCIe Gen4 x8 real pinned reads ~**10–13 GB/s** (~12× slower). Any byte that misses GDDR and arrives over PCIe each token pays that tax.

**Forbidden claim:** “CUDA Unified Memory / BAR mapping = full GPU speed.”

### B2. Dense decode’s working set is essentially the model
For a dense GEMV, essentially every weight tile participates every token. Demand-paging / driver compress-on-cold that assumes a small hot set **reverts to streaming the whole file**.

### B3. Fitting weights ≠ fitting a runnable model
Budget must cover simultaneously:

- weight (or factor) working set  
- SWA KV (~320 MiB at window 1024 for this arch’s local layers)  
- global-layer KV (grows with context; 262K context field is **not** a plan)  
- CUDA context, scratch, fragmentation  
- display / other processes if they touch the NVIDIA device  

**Rule:** quote **usable** MiB after KV watermark, not brochure 6144.

### B4. Intel “64 GB video memory” is not private VRAM
iGPU aperture maps **system DRAM**. Bandwidth is DRAM-class (~40–70 GB/s if dual-channel), shared with the CPU, and this SKU has only **16 EUs** (~0.33 TFLOPS). It is not a second 6 GB GDDR twin.

### B5. Brochure clocks ≠ sustained clocks
Power limit here is **60 W** default (**95 W** max if firmware allows). Token-time claims must log power/clocks under load.

---

## C. Quality and evaluation

### C1. Operator match ≠ bitwise GGUF identity
Preserving every Q4/Q6 code is a **different** goal from preserving `Wx` on `μ` within distortion `D`. Do not mix them in claims.

### C2. Chat vibes ≠ pass bar
Primary bars: activation MSE vs `D`, held-out NLL/PPL vs the **Q6→Q4 control**, then task probes. “Sounds fine in one chat” is not a milestone exit.

### C3. Calibration leakage
Fit factors on `μ` estimated from data the model (or the compressor) has already adapted to without a **held-out** corpus → optimistic `D` and fake passes. Held-out text is mandatory.

### C4. Layer-local MSE ≠ end-to-end fidelity
Errors enter the residual stream and accumulate (see LACE-SVD motivation). Local pass + global fail is expected; end-to-end NLL is required before celebrating.

### C5. KV papers ≠ weight papers
ASVD/Palu “50%” often means **KV cache**, not half the weight file. Cite the object compressed.

---

## D. Performance semantics

### D1. Memory-bound decode: bytes moved ≈ latency
On this GPU, ~2× fewer weight bytes/token ≈ ~2× tok/s **only if** the working set is resident and the kernel does not reintroduce large intermediates. FLOP reduction alone is secondary.

### D2. Prefill ≠ decode
Long-prompt prefill can be compute-bound and hide PCIe; interactive decode is batch-1 and is not. Never sell prefill speedups as chatbot speedups without saying so.

### D3. “Faster than Ollama” is not automatic
Ollama hybrid uses a strong **CPU** for overflow. A weak iGPU split can be **slower**. Any dual-device claim needs a head-to-head on **this** chassis.

### D4. Factorization can add traffic
Shared charts and Monarch intermediates are re-read or materialized. Count them in the roofline or you will “compress” on paper and stall on the bus.

---

## E. Methodological failure modes (LLM favorites)

| Temptation | Why it fails here |
|---|---|
| “Just use Q2/IQ2” as the discovery | Changes the vow; may fit, but is not hemifactor and usually fails Q4-parity on 7–12B |
| “Driver-level compression” | Same entropy wall; working set ≈ model |
| “Predictive layer prefetch” | Consumer finishes in ≪1 ms; PCIe layer load ~10 ms — overlap is tiny at batch-1 |
| “Expand Q4→FP16 on iGPU then send” | **Increases** bytes on the slow link |
| “Small procedural prior + residual” | Residual stays ~full file if prior is tiny |
| “Activation sparsity gates exact GEMV” | Dense layers; exact product needs the columns |
| Cite SVD-LLM 50% as success | Their PPL at 50% is catastrophic vs baseline |
| Cite QuIP# 2-bit as Q4 quality on 7B | Tables show a clear gap; closer only at much larger scale |
| Ignore tied embedding byte share | Leaves layer budget harsher (~2.1 bpw) |
| Claim success before spectrum license | Flat spectrum → low-rank 2× is dead; must switch branch or record negative result |

---

## F. One-line creed

**Do not store what you cannot afford; do not claim what you have not measured; do not confuse a slower wire with a larger chip.**

See also: [theory.md](theory.md) §1 and §7, [plan.md](plan.md) guiding rules, [hardware.md](hardware.md) rooflines.
