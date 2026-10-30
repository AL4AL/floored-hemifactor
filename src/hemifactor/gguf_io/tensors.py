"""Load and dequantize individual GGUF tensors without materializing the whole file."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import numpy as np

ROLE_SUFFIXES = {
    "down": "ffn_down.weight",
    "gate": "ffn_gate.weight",
    "up": "ffn_up.weight",
    "q": "attn_q.weight",
    "k": "attn_k.weight",
    "v": "attn_v.weight",
    "o": "attn_output.weight",
}


@dataclass(frozen=True)
class TensorRef:
    name: str
    shape: tuple[int, ...]
    nbytes: int
    tensor_type: int
    layer: int | None
    role: str | None


def _parse_layer_role(name: str) -> tuple[int | None, str | None]:
    if not name.startswith("blk."):
        return None, None
    parts = name.split(".")
    try:
        layer = int(parts[1])
    except (IndexError, ValueError):
        return None, None
    for role, suffix in ROLE_SUFFIXES.items():
        if name.endswith(suffix):
            return layer, role
    return layer, None


@lru_cache(maxsize=4)
def _reader(path: str):
    from gguf import GGUFReader

    return GGUFReader(path)


def list_tensors(path: str | Path) -> list[TensorRef]:
    reader = _reader(str(Path(path).resolve()))
    out: list[TensorRef] = []
    for t in reader.tensors:
        shape = tuple(int(x) for x in t.shape)
        layer, role = _parse_layer_role(t.name)
        out.append(
            TensorRef(
                name=t.name,
                shape=shape,
                nbytes=int(t.n_bytes),
                tensor_type=int(t.tensor_type),
                layer=layer,
                role=role,
            )
        )
    return out


def find_tensor(path: str | Path, name: str):
    reader = _reader(str(Path(path).resolve()))
    for t in reader.tensors:
        if t.name == name:
            return t
    raise KeyError(name)


def dequantize_named(path: str | Path, name: str, dtype=np.float32) -> np.ndarray:
    from gguf.quants import dequantize

    t = find_tensor(path, name)
    arr = dequantize(t.data, t.tensor_type)
    return np.asarray(arr, dtype=dtype)


def iter_role(path: str | Path, role: str) -> Iterator[TensorRef]:
    suffix = ROLE_SUFFIXES[role]
    for ref in list_tensors(path):
        if ref.name.endswith(suffix):
            yield ref


def tensor_inventory(path: str | Path) -> dict[str, Any]:
    refs = list_tensors(path)
    by_role: dict[str, list[dict[str, Any]]] = {}
    total = 0
    for r in refs:
        total += r.nbytes
        if r.role:
            by_role.setdefault(r.role, []).append(
                {"name": r.name, "layer": r.layer, "shape": list(r.shape), "nbytes": r.nbytes}
            )
    return {
        "path": str(Path(path).resolve()),
        "n_tensors": len(refs),
        "tensor_nbytes_sum": total,
        "roles": {k: {"count": len(v), "nbytes": sum(x["nbytes"] for x in v)} for k, v in by_role.items()},
        "by_role": by_role,
    }
