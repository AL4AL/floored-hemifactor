#!/usr/bin/env python3
"""CLI wrapper: python scripts/run_phases.py all|0|1|2|3|4|5"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hemifactor.cli import main

if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    raise SystemExit(main(["run", phase, *sys.argv[2:]]))
