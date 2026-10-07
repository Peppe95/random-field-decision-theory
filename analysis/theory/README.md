# Theory checks and illustrations

This directory contains numerical checks and illustrative calculations for the analytical results in the manuscript.

These scripts do **not** read human data or fitted empirical parameters.

Run:

```bash
python analysis/theory/make_theory_predictions.py
```

The script writes numerical diagnostics under `analysis/theory/theory_checks/` and regenerates the illustrative theory figures in this directory.

`theory_validation.json` preserves an earlier independent set of numerical validation results used during manuscript development. These numerical checks support the algebraic derivations; they are not substitutes for the proofs in the paper.
