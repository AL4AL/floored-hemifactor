from __future__ import annotations

import argparse
import json
import sys

from hemifactor import __version__
from hemifactor.config import load_config
from hemifactor.constraints import lint_success_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hemifactor", description="Floored hemifactor tools")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cfg = sub.add_parser("show-config", help="Print resolved config JSON")
    p_cfg.add_argument("--config", default=None)

    p_lint = sub.add_parser("lint-claim", help="Lint a success claim against hard constraints")
    p_lint.add_argument("text")

    p_check = sub.add_parser("preflight", help="Print hard-constraint reminder + config gaps")
    p_check.add_argument("--config", default=None)

    p_run = sub.add_parser("run", help="Run a phase (0–5) or all")
    p_run.add_argument("phase", choices=["0", "1", "2", "3", "4", "5", "all"])
    p_run.add_argument("--config", default=None)

    args = parser.parse_args(argv)

    if args.cmd == "show-config":
        cfg = load_config(args.config)
        print(json.dumps(cfg, indent=2, default=str))
        return 0

    if args.cmd == "lint-claim":
        hits = lint_success_text(args.text)
        if hits:
            print("FAIL — forbidden phrases:", file=sys.stderr)
            for h in hits:
                print(f"  - {h}", file=sys.stderr)
            return 1
        print("OK — no forbidden phrases matched")
        return 0

    if args.cmd == "preflight":
        cfg = load_config(args.config)
        print("Read docs/hard-constraints.md before proposing designs.")
        gaps = []
        model = cfg.get("model", {})
        if not model.get("gguf_path"):
            gaps.append("model.gguf_path is null — run scripts/freeze_model.py")
        if not cfg.get("calibration", {}).get("heldout_corpus_path"):
            gaps.append("calibration.heldout_corpus_path is null")
        else:
            from pathlib import Path

            hp = Path(cfg["calibration"]["heldout_corpus_path"])
            if not hp.is_absolute():
                hp = Path(cfg["_root"]) / hp
            if not hp.is_file():
                gaps.append(f"held-out corpus missing: {hp}")
        if gaps:
            print("Config gaps:")
            for g in gaps:
                print(f"  - {g}")
            return 2
        print("Preflight OK — model path and held-out corpus set")
        return 0

    if args.cmd == "run":
        from hemifactor import pipeline

        cfg = load_config(args.config)
        # resolve relative held-out path
        hp = cfg.get("calibration", {}).get("heldout_corpus_path")
        if hp and not str(hp).startswith("/"):
            from pathlib import Path

            cfg["calibration"]["heldout_corpus_path"] = str(Path(cfg["_root"]) / hp)
        dispatch = {
            "0": pipeline.phase0_freeze,
            "1": pipeline.phase1_calibrate,
            "2": pipeline.phase2_spectrum,
            "3": pipeline.phase3_down_prototype,
            "4": pipeline.phase4_fuse_time,
            "5": pipeline.phase5_assemble,
            "all": pipeline.run_all,
        }
        result = dispatch[args.phase](cfg)
        print(json.dumps(result if args.phase != "all" else {k: v.get("status", v) for k, v in result.items()}, indent=2, default=str))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
