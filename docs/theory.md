# Theory: Floored Hemifactor

Operational mistakes (entropy myths, bus myths, vibe eval) are catalogued in **[hard-constraints.md](hard-constraints.md)**. This file states the mathematics; that file forbids lying about it.

## 1. Two questions (only one is open)

### Closed — bitwise residency

Let the on-disk quantized artifact be a byte string `B` with `|B| ≈ 6.88 GiB` (measured GGUF for `gemma4:12b`: mixed Q4_K / Q6_K, ≈ 4.96 bits/parameter over ≈ 11.91B params).

Usable GDDR on the test GPU (panel on iGPU) is on the order of **5792 MiB**, of which sliding-window KV alone can claim **≈ 320 MiB**, leaving roughly **≈ 5472 MiB** for weights.

A lossless codec that reconstructs `B` exactly must therefore erase **≳ 1.5 GiB** of already-quantized codes. Those codes sit near ~5 bits of entropy per weight; generic compressors recover only a few percent. Idle ALUs are not a source of new weight information. **Storing the same integers in 6 GB is closed.**

### Open — operator on the activation law

Decode never needs the matrix `W` as an object in `R^{m×n}`. It needs the map

```text
x ↦ Wx    for x ∼ μ
```

where `μ` is the law of residual-stream (or module-input) activations this network actually produces under language.

The shipped Q4/Q6 file already accepts an activation mean-square error. Call it

```text
D(W, μ) = E_{x∼μ} ‖ (W − W_dequant) x ‖²
```

(with the higher-rate tensors in the same file, or a controlled Q6→Q4 probe, as the yardstick).

**Any code `C` whose evaluation reads about `|C|` bytes and satisfies**

```text
E_μ ‖ (W − Ŵ_C) x ‖²  ≤  D(W, μ)
```

**matches “Q4 quality on language activations” by the standard the file already paid** — even if `C` is not the GGUF integers.

Existence of such a `C` with `|C| ≈ |B|/2` is **not** settled by the counting argument that closes bitwise residency. That argument only forbids reconstructing the integers.

---

## 2. Roofline (why bytes are performance)

On this RTX 3050 Laptop:

| Quantity | Approx. value |
|---|---|
| GDDR peak | **~168 GB/s** (96-bit · ~14 Gbps class) |
| Weight-only token if full file fit | `6.88e9 / 168e9 ≈ 41–44 ms` → **~23 tok/s** |
| Same + full SWA KV read (320 MiB) | **~22 tok/s** |
| Half weight bytes + KV | **~42 tok/s** class |
| Overflow of ~1.5 GiB over PCIe Gen4 x8 (~10–13 GB/s) | **+~120–160 ms** → today’s wall **~5–7 tok/s** |

Batch-1 decode is **memory-bound**. Arithmetic for one token is ~`2 × 11.9e9 ≈ 24 GFLOP`; at a few TFLOPS that is a few milliseconds against ~44 ms of fetches. Spare FLOPs exist for structured factors; they do **not** invent bits.

**Corollary:** cutting bytes moved per token by ~2× is the performance prize on this chassis once the working set is resident. Residency alone needs only ~**0.78×** the file; **0.5×** is the speed target.

---

## 3. Definition — floored hemifactor

Let `Σ = E_μ[xxᵀ] = Q Λ Qᵀ`. Whitened singular values of a map `W` are the singular values of `W Σ^{1/2}` (equivalently, square roots of eigenvalues of the small Gram in the input dimension `d`, here `d = 3840` for residual-fed maps — never form the wide×wide matrix).

**Definition.** A **floored hemifactor** of `W ∈ R^{m×n}` is a rank-`k` factorization `AB` plus an optional tail `T` such that

```text
E_μ ‖ (W − AB − T) x ‖²  ≤  D(W, μ)
```

and `k` is chosen as small as the byte budget allows.

### Sharing structure (Gemma-4–aware)

For maps that eat the residual (`Q`, `K`, `V`, gate, up):

- Share an input chart `C ∈ R^{d×k}` across the **40 local (sliding-window) layers** and across those roles where calibration supports it.
- Keep a private tall factor in `R^{m×k}` per map.

For maps that eat a wide activation (down, output projection):

- Private two-sided factors; for a `15360×3840` tensor, half parameters via `k(m+n) = ½ mn` suggests **`k ≈ 1536`**.

For charted residual-fed maps, half private parameters via shared `C` suggests **`k ≈ 1920`**.

**Tail (only if truncated energy still exceeds `D`):** a thin joint residual across the 40 local layers, e.g.

```text
T_ℓ x = Σ_{a=1}^{s} c_{ℓ,a} (u_a v_aᵀ) x
```

with small `s` (≤ 4) and low rank per atom (e.g. ≤ 32), `(u_a, v_a)` shared, `c_{ℓ,a}` per layer. Global layers (every 6th in the 5+1 SWA pattern) get their own charts; no rich shared tail assumed a priori.

**Evaluation:** two (or few) GEMVs + tail. The dense `W` is **never** written back to GDDR.

### Traffic tax

A shared chart re-read every layer costs extra bytes/token. At ~4.5 bits, a `d×k` chart can be a few MB × 48 layers ≈ **~5%** of today’s fetch if careless. Private factors must therefore land nearer **~0.45×** for a true half including chart/tail rereads — or the experiment honestly reports **~0.55×** if the spectrum only clears `D` at higher `k`.

---

## 4. Why a factor of two *can* exist

After whitening, activation MSE of truncated SVD equals discarded energy `Σ_{i>k} σ_i²`.

Q4/Q6 already spends some `D > 0` of that currency. Singular components quieter than `D` are stored and shipped today, yet they do not buy quality the file is willing to claim. Uniform low-bit quant (plain Q2) still spends bits on directions `μ` cannot see and starves large `σ_i`. The hemifactor spends roughly today’s 4–6 bits **only on kept components**.

**Where this dies:** if the whitened spectrum is flat nearly to full rank, and layer residuals are mutually incoherent, low-rank is the wrong algebra. Fallback branch: depth-2 **Monarch / butterfly** (block-diagonal × permutation) fit by least squares on pairs `(x, Wx)`, `x ∼ μ`, using the same FLOP slack. If that also misses `D`, this model does not give a clean 2× — and the measured rate–distortion curve is still a contribution.

**Where the mass is:** MLP local gate/up/down dominate parameters (~60%); global MLP and attention follow; tied embedding is smaller in count but Q6-heavy in bytes. Prototype priority: **down projections**.

---

## 5. Bits the discovery must hit (this file)

| Spread | Requirement |
|---|---|
| Half the whole file evenly | **~2.48 bit / original param** |
| Leave tied embedding as Q6; compress the rest | **~2.10 bit / layer param** (as bytes moved/token) |
| Mild factor on output GEMV | Relaxes layers toward ~**2.3** |

Kept components stay near 4–6 bits; discarded ones are **absent**, not stored as 2-bit integers.

Residency with KV: total weight+factor footprint must sit under ~**5472 MiB**. Half-file + 320 MiB KV leaves substantial scratch inside 5792 MiB free.

---

## 6. Relation to Janus (historical)

**Ridge-Split Janus** (column-split GEMV across RTX 3050 + Intel UHD) preserves every Q4/Q6 code and keeps all multiplies on GPUs. It does not cut GDDR traffic of the NVIDIA shard below what fits, and the 16-EU iGPU is a weak overflow engine vs this CPU.

Janus answered the bitwise question correctly and stopped. It never set allowed distortion equal to `D(W, μ)` nor asked how much of `W` is mutual information with `μ`. Preserving every super-block made the hemifactor illegal. **Primary path: hemifactor. Janus is a fallback residency story, not the theory.**

---

## 7. Success predicate (theory-level)

Before declaring success, re-read hard-constraints **A–E**. Declare success on a role (then model) only if **all** hold:

1. **Distortion:** activation MSE (or calibrated proxy) ≤ `D` on held-out sequences not used to fit factors.
2. **End-to-end:** perplexity / NLL move ≤ the Q6→Q4 control measured on the same held-out text.
3. **Traffic:** measured bytes/token (factors + chart reread + tail + KV) ≤ **0.55×** of the baseline file fetch (stretch **0.50×**).
4. **Residency:** working set fits in 6 GB GDDR with SWA KV at window 1024 and a stated global-KV watermark.
5. **Speed:** tok/s approaches `168e9 / bytes_per_token`, not PCIe-bound overflow.

Failure of (1)–(2) at the `k` required for (3) is a **negative scientific result**, not a soft land into “just use Q2.”
