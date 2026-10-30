# Agent / collaborator contract

Read **before** writing code, changing theory, or claiming results.

1. **[docs/hard-constraints.md](docs/hard-constraints.md) is binding.** If a proposal violates it, stop and revise.
2. Primary path is **floored hemifactor** (operator on `μ` within distortion `D`). Janus / driver lossless packing are deferred or closed.
3. **Never** mark Phase success on chat vibes alone.
4. **Never** claim ~2× tok/s until traffic bytes are measured and the working set is GDDR-resident.
5. Prefer recording a **negative spectrum result** over silently switching to Q2 and calling it the discovery.
6. Update `artifacts/` with JSON metrics; update `notes/` with dated lab notes.
7. Do not commit secrets, GGUF weights, or huge covariance dumps if policy forbids — keep paths in config.

## Quick commands

```bash
cd /home/sajjad/PP/floored-hemifactor
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export PYTHONPATH=src
bash scripts/preflight.sh
python -m hemifactor.cli lint-claim "we losslessly fit exact GGUF into 6GB"
```

## Doc map

| Doc | Use |
|---|---|
| docs/hard-constraints.md | Pre-flight physics / methodology |
| docs/theory.md | Formal object |
| docs/plan.md | Phases and exit criteria |
| docs/hardware.md | This chassis |
| docs/related-work.md | What is NOT already solved |
