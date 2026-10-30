# Related work

Survey date: 2026-10-30. Question asked: *has anyone already achieved ~½ weight bytes / traffic with quality matching strong Q4 on ~7–12B models?*

**Answer: no published result matches that win condition.** The *direction* (activation-aware / whitened low-rank) is active. The *result* (clean 2× at Q4 parity) is open.

LLM-typical citation errors (calling KV “50%” a weight win; calling SVD-LLM@50% a quality success; calling QuIP# 2-bit “same as Q4” on 7B) are called out in [hard-constraints.md](hard-constraints.md) §C5 and §E.

---

## Activation-aware / whitened SVD (closest cousins)

### ASVD — Activation-aware SVD
- Paper: [arXiv:2312.05821](https://arxiv.org/abs/2312.05821)
- Idea: scale weights by activation channel stats, then SVD; layer-wise rank probing.
- Claim: **~10–20%** model compression without serious loss; **~50% KV** reduction via low-rank KV projections.
- Gap vs us: mild weight cut only; “50%” is not full-weight half-size at Q4 parity.

### SVD-LLM — Truncation-aware whitening + updates
- Paper: [arXiv:2403.07378](https://arxiv.org/abs/2403.07378) (ICLR 2025)
- Idea: Cholesky whitening so singular values map cleanly to compression loss; sequential low-rank updates.
- LLaMA-7B WikiText-2 (reported): baseline **5.68** → **7.94** at 20% compression → **23.97** at **50%**.
- Gap vs us: best-in-class among early SVD-LLM methods, but **half-size destroys LM quality**. Proves the spectrum often does *not* freely give 2×.

### LACE-SVD / ResSVD
- LACE-SVD: loss-aware rank allocation + cumulative residual-stream error correction ([html](https://arxiv.org/html/2607.03057)).
- ResSVD: residual-compensated SVD, selective layers ([arXiv:2505.20112](https://arxiv.org/abs/2505.20112)).
- Gap vs us: better allocation and error accounting (aligned with our `D` / residual-stream thinking); still not a demonstrated Q4-parity 2× weight cut.

### Palu
- [arXiv:2407.21118](https://arxiv.org/abs/2407.21118) — low-rank **KV cache** path, up to ~50% KV with strong accuracy.
- Gap vs us: wrong object for the 6.88 GiB **weight** wall on a 6 GB card.

---

## Extreme quantization (~2 bpw ≈ “half of Q4”)

### QuIP#
- [arXiv:2402.04396](https://arxiv.org/abs/2402.04396)
- Llama-2-7B WikiText-2: FP16 **5.12** · QuIP# 4-bit **5.19** · QuIP# 2-bit **6.19** (with fine-tune).
- 70B 2-bit gets closer to 4-bit; **7B-class does not**.

### AQLM
- [arXiv:2401.06118](https://arxiv.org/abs/2401.06118)
- Same story: 2-bit is usable but **not Q4-parity** on small models; competitive at higher scales / bitwidths.

**Takeaway:** “Half the bits” via better codebooks is the industrial neighbor of our bit target (~2.1–2.5 bpw average), but it is **uniform-ish rate allocation**, not floored singular components. Quality tax remains real at 7–12B.

---

## Hybrid / sidecar

### SVSK
- [github.com/Dookoo2/SVSK](https://github.com/Dookoo2/SVSK)
- Keep Q4-class budget (~4.4–4.6 bpw) + tile-local low-rank sidecar to recover harmful quant residual.
- Gap vs us: **opposite** objective (quality at Q4 size), not half size.

---

## Structured matrices (Branch B)

### Monarch / Monarch Mixer / BLR on GPUs
- Monarch (Dao et al.): block-diagonal × permutation factors.
- Recent resource-constrained GPU work: single-token decode can benefit from fewer weight bytes; multi-token / naive kernels can **lose** to dense due to intermediates and layout ([arXiv:2512.20861](https://arxiv.org/abs/2512.20861)).
- Role here: **fallback** if whitened spectra are flat — not assumed free lunch.

---

## What we keep from the literature

| Idea | Keep? |
|---|---|
| Activation / whitening before SVD | **Yes** — core of hemifactor |
| Map singular values to activation loss | **Yes** — our `D` floor |
| Residual-stream / cumulative error awareness | **Yes** — LACE-like corrections later |
| Aggressive 50% uniform SVD | **No** — empirically hostile to PPL |
| Plain 2-bit as “done” | **No** — quality gap on our size class |
| KV-only 50% | Useful later; **not** Phase 1–3 |

---

## Positioning statement

> Floored hemifactor asks whether the **error budget Q4 already paid** licenses discarding whitened singular mass until weight traffic halves — scored on `μ`, with optional shared tails across Gemma-4 local layers, and a Monarch branch if the spectrum is flat. Prior SVD-LLM work shows mild compression works and naive 50% fails; prior 2-bit work shows half-bits without manifold flooring still hurts 7–12B. **The specific win condition remains open.**
