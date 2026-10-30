#!/usr/bin/env python3
"""Phase 0: locate gemma4:12b GGUF blobs, hash them, write freeze report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def find_ollama_blobs() -> list[Path]:
    homes = [
        Path.home() / ".ollama" / "models",
        Path("/usr/share/ollama/.ollama/models"),
    ]
    # Ollama service user may store elsewhere
    extra = os.environ.get("OLLAMA_MODELS")
    if extra:
        homes.insert(0, Path(extra))
    found: list[Path] = []
    for root in homes:
        manifests = root / "manifests"
        blobs = root / "blobs"
        if not manifests.is_dir():
            continue
        for p in manifests.rglob("*"):
            if p.is_file() and "gemma" in str(p).lower():
                found.append(p)
        if blobs.is_dir():
            found.append(blobs)
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, help="Explicit GGUF path if known")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "model_freeze.json",
    )
    args = ap.parse_args()

    report: dict = {
        "hard_constraints": "docs/hard-constraints.md",
        "note": "Exact codes in 6GB remains CLOSED; freeze is identity only.",
        "ollama_hints": [str(p) for p in find_ollama_blobs()],
    }

    if args.gguf:
        p = args.gguf
        report["gguf_path"] = str(p.resolve())
        report["size_bytes"] = p.stat().st_size
        report["sha256"] = sha256_file(p)
        try:
            from hemifactor.gguf_io.inspect import summarize_gguf

            report["summary"] = summarize_gguf(p)
        except Exception as e:
            report["summary_error"] = repr(e)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.out}")
    if not args.gguf:
        print("No --gguf given. Inspect ollama_hints and re-run with --gguf PATH.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
