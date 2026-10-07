# Choices13k external validation

Authoritative production files:
- `run_choices13k_external.py`
- `rfdt_power.py`

This is the external validation of the HAB22-frozen RFDT-P specification.

The upstream archive documented in `DATA_SOURCES.md` is filtered to `Feedback=False`, `Amb=False`, `Corr=0`, yielding 1,766 unique description-only problems. The runner reconstructs the exact displayed payoff distributions from `c13k_problems.json`, uses independent/product hypothetical sampling, and assigns ten deterministic problem folds with seed 20260911.

The folds are generated from a seeded permutation of the 1,766 selected source rows followed by round-robin assignment. Fold sizes are recorded in `choices13k_fold_counts.csv`; no human outcome rates or participant-level data are committed.

Saved final configuration is in `analysis_spec.json`.
