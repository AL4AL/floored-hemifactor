"""GGUF inspection helpers (Phase 0–1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def summarize_gguf(path: str | Path) -> dict[str, Any]:
    """Return basic metadata. Requires ``gguf`` package."""
    from gguf import GGUFReader

    p = Path(path)
    reader = GGUFReader(p)
    fields: dict[str, Any] = {}
    for key, field in reader.fields.items():
        # Keep only small scalar-ish metadata for freeze reports
        try:
            val = field.contents()
        except Exception:
            continue
        if isinstance(val, (int, float, bool, str)):
            fields[key] = val
        elif isinstance(val, (list, tuple)) and len(val) <= 64:
            fields[key] = list(val)

    tensors = []
    total = 0
    for t in reader.tensors:
        nbytes = int(t.n_bytes)
        total += nbytes
        tensors.append(
            {
                "name": t.name,
                "shape": list(t.shape),
                "nbytes": nbytes,
                "n_elements": int(t.n_elements),
            }
        )

    return {
        "path": str(p.resolve()),
        "size_bytes": p.stat().st_size,
        "tensor_nbytes_sum": total,
        "n_tensors": len(tensors),
        "fields": fields,
        "tensors": tensors,
    }
