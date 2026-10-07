# HAB22 production analysis

Authoritative production files:
- `run_hab22_final_frozen.py`
- `rfdt_power.py`
- `read_rdata_min.py`

The runner reconstructs the 15-context HAB22 analysis from the upstream Plonsky archive and the Phase-3b recovery/design archive documented in `DATA_SOURCES.md`. It verifies the stored fold choice rates before fitting.

Two architectures must remain distinct:

1. **Participant-mixture benchmark.** RFDT-P is fitted separately to each eligible participant on training task folds. For a held-out task, predictions are averaged over fitted training participants from that task's context, excluding the held-out participant fold.
2. **Pooled restriction analysis.** One common parameter vector is fitted to pooled training responses in each crossed task-fold x participant-fold cell, identically for full and restricted RFDT.

Their absolute MSEs are therefore not interchangeable.

Task folds are the upstream stored `crossValidation_id` values. Five participant folds are recreated within each context using the embedded R-compatible `set.seed(123)` algorithm. Participant-level fold assignments, parameters and predictions are deliberately not redistributed. `hab22_participant_fold_counts.csv` records aggregate counts by context/fold.

Saved final configuration is in `analysis_spec.json`.
