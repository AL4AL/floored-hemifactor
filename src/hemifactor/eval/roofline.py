"""GDDR / PCIe roofline helpers (docs/hardware.md, theory §2)."""

from __future__ import annotations

from typing import Any


def tok_s_from_bytes(bytes_per_token: float, bandwidth_gbs: float) -> float:
    """Memory-bound decode estimate: bandwidth / bytes_per_token."""
    if bytes_per_token <= 0:
        return float("inf")
    return (bandwidth_gbs * 1e9) / bytes_per_token


def compare_buses(
    bytes_per_token: float,
    gddr_gbs: float = 168.0,
    pcie_gbs: float = 12.0,
) -> dict[str, Any]:
    return {
        "bytes_per_token": bytes_per_token,
        "tok_s_gddr": tok_s_from_bytes(bytes_per_token, gddr_gbs),
        "tok_s_pcie": tok_s_from_bytes(bytes_per_token, pcie_gbs),
        "gddr_gbs": gddr_gbs,
        "pcie_gbs": pcie_gbs,
        "note": "Batch-1 decode is memory-bound here; FLOP cuts alone are secondary.",
    }
