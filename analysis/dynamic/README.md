# Corrected Guo dynamic analysis

This directory contains the production code used for the joint choice and response-time analyses in the manuscript.

The final analysis uses:
- signed-power utility;
- the exact finite-`N` first-passage process;
- single-click observations;
- 14 joint choice-by-RT cells for the all-RT target;
- candidate-level conditioning for the separately refitted 0.3--10 s target;
- normalized finite-set empirical Bayes;
- full RFDT, a separately refitted uncoupled RFDT control, and an analytic nonlinear DDM.

The production source was restored verbatim from `RFDT_Guo_corrected_analysis_code_v1.zip`. Recorded hashes are in `SOURCE_HASHES.md`; run

```bash
python analysis/dynamic/verify_frozen_sources.py
```

from the repository root to check them.

The archived unit tests can be run with:

```bash
PYTHONPATH=analysis/dynamic python -m unittest discover -s analysis/dynamic/tests -v
```

`run_rfdt_guo_correction.py` is the command-line entry point. The frozen numerical settings are in `config_reference.json`.

Earlier linear-utility recovery code and earlier human-analysis pipelines are not part of the reported analysis. The numerical checks that did and did not meet their declared tolerances are recorded in `provenance/NUMERICAL_STATUS.md`.
