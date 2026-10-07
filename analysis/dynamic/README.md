# Corrected Guo dynamic analysis

This directory documents the final production analysis used for the joint choice and response-time results.

The reported analysis uses:
- signed-power utility;
- exact finite-N first-passage simulation;
- single-click observations;
- 14 joint choice-by-RT cells for the all-RT target;
- candidate-level conditioning for the separately refitted 0.3--10 s target;
- normalized finite-set empirical Bayes;
- full RFDT, separately refitted uncoupled RFDT, and an analytic nonlinear DDM.

Earlier linear-utility recovery code and earlier human-analysis pipelines are not part of this analysis.

`run_rfdt_guo_correction.py` is the production command-line entry point. The source hashes of its imported modules are recorded in `SOURCE_HASHES.md`.
