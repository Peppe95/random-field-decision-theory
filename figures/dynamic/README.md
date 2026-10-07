# Dynamic manuscript figures

These scripts rebuild the main dynamic summary figures from compact, frozen summaries. They do **not** refit RFDT or the DDM and they do not run the finite-population simulator.

Run from the repository root:

```bash
python figures/dynamic/make_figures_dynamic.py
```

Inputs:

- `results/dynamic/figure5_primary_comparisons.csv` — primary-sample held-out comparisons and participant-bootstrap intervals.
- `results/dynamic/figure6_primary_rt_calibration.csv` — observed and mean predicted RT-category frequencies for the independently evaluated all-RT fits.
- `results/dynamic/figure6_primary_score_regions.csv` — decomposition of the RFDT-minus-DDM score difference by observed RT region.

Outputs:

- `figures/dynamic/Figure5_dynamic.pdf` and `.png`
- `figures/dynamic/Figure6_dynamic.pdf` and `.png`

The compact values come from the frozen corrected-analysis summaries and the read-only prediction audit. Numerical uncertainty from simulation and candidate resolution is not represented by the participant-bootstrap intervals; see `provenance/NUMERICAL_STATUS.md`.
