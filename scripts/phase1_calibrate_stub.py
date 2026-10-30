#!/usr/bin/env python3
"""Phase 1 entrypoint stub — wire real forward hooks after model freeze."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hemifactor.config import artifacts_dir, load_config


def main() -> int:
    cfg = load_config()
    out = artifacts_dir(cfg) / "phase1_status.json"
    payload = {
        "phase": 1,
        "status": "not_started",
        "blocked_by": [],
        "hard_constraints": "docs/hard-constraints.md",
        "next": [
            "Freeze GGUF via scripts/freeze_model.py",
            "Set calibration.heldout_corpus_path in configs/default.yaml",
            "Implement activation hook runner (calibrate package)",
        ],
    }
    if not cfg.get("model", {}).get("gguf_path"):
        payload["blocked_by"].append("model.gguf_path")
    if not cfg.get("calibration", {}).get("heldout_corpus_path"):
        payload["blocked_by"].append("calibration.heldout_corpus_path")
    if payload["blocked_by"]:
        payload["status"] = "blocked"
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 2 if payload["blocked_by"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
