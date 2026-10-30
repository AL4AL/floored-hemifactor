"""Machine-checkable reminders of hard constraints.

Import and call ``assert_claim`` before writing success language into artifacts.
"""

from __future__ import annotations

FORBIDDEN_SUBSTRINGS = (
    "lossless fit into 6gb",
    "lossless fit into 6 gb",
    "losslessly fit",
    "exact gguf into 6",
    "exact codes in 6gb",
    "exact codes in 6 gb",
    "unified memory is as fast as gddr",
    "pcie equals gddr",
    "pcie as fast as gddr",
    "spare flops create bits",
    "spare flops",
    "idle alus regenerate",
    "chat vibes",
    "bitwise identical and half size",
    "same quality as q4 at 2-bit on 7b",
)

CLOSED_CLAIMS = {
    "exact_gguf_in_6gb_lossless": (
        "A1: Exact GGUF codes cannot be losslessly stored in 6 GB usable GDDR "
        "with KV/scratch on this chassis."
    ),
    "spare_flops_make_bits": (
        "A2: Spare decode FLOPs do not create missing weight information."
    ),
    "pcie_as_gddr": (
        "B1: PCIe host reads are ~10–13 GB/s here; GDDR is ~168 GB/s."
    ),
    "igpu_private_vram": (
        "B4: Intel aperture is system DRAM, not a second GDDR pool."
    ),
}


def assert_claim(claim_id: str, allowed: bool, detail: str = "") -> None:
    """Raise if a closed claim is asserted as solved."""
    if claim_id in CLOSED_CLAIMS and allowed:
        raise AssertionError(f"Closed claim marked solved: {CLOSED_CLAIMS[claim_id]} {detail}")


def lint_success_text(text: str) -> list[str]:
    """Return list of forbidden phrases found in a draft success blurb."""
    lowered = text.lower()
    return [s for s in FORBIDDEN_SUBSTRINGS if s in lowered]
