"""Evaluation helpers — never use chat vibes as a pass (hard-constraints C2)."""

from __future__ import annotations

import math
from typing import Iterable


def nll_from_token_logprobs(logprobs: Iterable[float]) -> float:
    vals = list(logprobs)
    if not vals:
        raise ValueError("empty logprobs")
    return -sum(vals) / len(vals)


def ppl_from_nll(nll: float) -> float:
    return math.exp(nll)


def within_control(delta: float, control_delta: float, eps: float = 1e-9) -> bool:
    """Pass if measured degradation is no worse than Q6→Q4 control."""
    return delta <= control_delta + eps
