# Rank license (Phase 2)

Aggregate decision: **branch-B**

Per-layer down-projection (whitened by FF diag-var from Phase 1 surrogate μ):

| layer | licensed | chosen_k | decision | eff_rank |
|---|---|---|---|---|
| 0 | False | None | branch_b_or_stop | 59.1 |
| 12 | False | None | branch_b_or_stop | 1104.9 |
| 24 | False | None | branch_b_or_stop | 1343.0 |
| 36 | False | None | branch_b_or_stop | 2288.3 |
| 47 | False | None | branch_b_or_stop | 2091.0 |

Activation method caveat: see `phase1_status.json`.
Negative / branch-B outcomes are first-class results (hard-constraints E).
