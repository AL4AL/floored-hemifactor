# Lab note — hard constraints + full scaffold

- Added binding `docs/hard-constraints.md` (entropy, buses, eval, LLM mistake table).
- Wired references into README, theory, plan, hardware, related-work, AGENTS.md.
- Scaffolded `src/hemifactor` (config, constraints lint, calibrate Welford, spectrum license, factorize stub, eval metrics, gguf inspect).
- Scripts: `preflight.sh`, `freeze_model.py`, `phase1_calibrate_stub.py`.
- Config: `configs/default.yaml`. Templates for spectrum license + phase results.

Next: locate GGUF path for `gemma4:12b`, freeze hash, set held-out corpus, implement real Phase 1 hooks.
