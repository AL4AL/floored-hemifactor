# Floored Hemifactor

Post-training compression research: represent LLM layer maps so that **bytes moved per decode token** fall by ~2× on memory-bound GPUs, while staying inside the activation error already paid by strong Q4 quantization.

> **Agents / collaborators:** read [AGENTS.md](AGENTS.md) and **[docs/hard-constraints.md](docs/hard-constraints.md)** first. Those constraints are where LLMs (and overconfident designs) usually fail.

## Goal (this machine)

Fit and run **`gemma4:12b`** (~11.9B, Q4_K/Q6_K mix ≈ 6.88 GiB) **fully resident** on an **RTX 3050 Laptop 6GB**, then push toward **~2× tokens/s** vs a mythically resident Q4 file on the **same** GDDR bus.

| Constraint | Status |
|---|---|
| Exact GGUF codes in 6 GB GDDR | **Closed** (entropy) — see hard-constraints A1 |
| Same *operator on activation law μ* within Q4’s distortion `D` | **Open** |
| Match Q4 quality by held-out NLL ≤ Q6→Q4 control | Target |
| Half weight traffic ⇒ ~2× decode on this bus | Target (only if resident) |

## Docs

| Doc | Contents |
|---|---|
| [docs/hard-constraints.md](docs/hard-constraints.md) | **Binding** physics + methodology; LLM mistake catalog |
| [docs/theory.md](docs/theory.md) | Formal floored hemifactor, success predicate |
| [docs/plan.md](docs/plan.md) | Phases 0–5, exits, milestones |
| [docs/hardware.md](docs/hardware.md) | LOQ 15IAX9 / 3050 rooflines |
| [docs/related-work.md](docs/related-work.md) | ASVD, SVD-LLM, QuIP#, … — gap analysis |

## Repo layout

```text
configs/default.yaml     # experiment knobs + model freeze fields
src/hemifactor/          # calibrate, spectrum, factorize, eval, gguf_io, kernels
scripts/                 # preflight, freeze_model, phase runners
artifacts/               # measurements (gitignored bulk; keep .gitkeep)
notes/                   # dated lab notes
templates/               # result / spectrum-license templates
docs/                    # theory and constraints
```

## Setup

```bash
cd /home/sajjad/PP/floored-hemifactor
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export PYTHONPATH=src
bash scripts/preflight.sh
# After locating GGUF:
python scripts/freeze_model.py --gguf /path/to/model.gguf
```

## Non-goals (for now)

- Shipping inside Ollama as a product
- Lossless packing of exact GGUF into 6 GB
- Dual-GPU Janus as the primary path
- Claiming success before spectrum + held-out bars clear

## Hardware under test

Lenovo LOQ 15IAX9 · RTX 3050 6GB Laptop (GA107) · Intel UHD · i5-12450HX · ~62 GiB RAM · Ubuntu

## Status

Scaffold + documentation ready for Phase 0 freeze and Phase 1 calibration. No compression result claimed.
