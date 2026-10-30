#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "== Floored Hemifactor preflight =="
echo "Hard constraints: docs/hard-constraints.md"
echo

python -m hemifactor.cli preflight || true
echo
echo "== nvidia-smi (if present) =="
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,memory.free,power.limit --format=csv
else
  echo "nvidia-smi not found"
fi
echo
echo "== Reminder =="
echo "Do not claim 2x until spectrum + held-out bars clear."
echo "Do not treat PCIe/UVM as GDDR."
echo "Do not use chat vibes as a pass."
