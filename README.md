# Random Field Decision Theory

Reproducibility materials for **Random Field Decision Theory: Deliberation through persistent and adaptive representations**, by Giuseppe M. Ferro and Didier Sornette.

The repository is organized around the analyses reported in the manuscript rather than around the history of model development.

- `analysis/theory/` — numerical checks and illustrations for the analytical results
- `analysis/static/` — frozen HAB22 and Choices13k analyses
- `analysis/dynamic/` — corrected Guo choice and response-time analysis
- `results/` — compact tables used to verify manuscript results
- `figures/` — figure-generation code using the compact saved results
- `data/` — instructions for obtaining the original datasets
- `provenance/` — checksums and numerical-status records

The source datasets were collected by other research teams and are **not redistributed here**. Download them from the original sources listed in [DATA_SOURCES.md](DATA_SOURCES.md).

## Quick start

Create an environment and install the lightweight dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Rebuild the manuscript summary figures without refitting the models:

```bash
python figures/static/make_figure4_static.py
python figures/dynamic/make_figures_dynamic.py
```

Run the analytical illustration/check script:

```bash
python analysis/theory/make_theory_predictions.py
```

Full empirical refits require the upstream datasets and, for the dynamic analysis, the complete frozen production package described below. See [REPRODUCE.md](REPRODUCE.md).

## Static analysis

HAB22 is the development benchmark. Choices13k is the external validation of the HAB22-frozen signed-power RFDT-P specification.

The participant-mixture HAB22 benchmark and the pooled restriction analysis are different fitting architectures and are kept separate. The production implementation, frozen settings, fold accounting and source hashes are under `analysis/static/`.

## Dynamic analysis

The final Guo analysis uses signed-power utility, the exact finite-population first-passage process, single-click observations, and normalized empirical-Bayes fitting. Earlier linear-utility recovery code and earlier human-analysis pipelines are not part of the final analysis.

The command-line runner and frozen configuration are under `analysis/dynamic/`. The imported `rfdt_correction` production modules must be copied **verbatim** from the archived production source before the first public release; this is tracked in [issue #2](https://github.com/Peppe95/random-field-decision-theory/issues/2). They should not be reconstructed from the manuscript.

The large manuscript prediction archive is kept outside Git:

`rfdt_guo_precision_for_manuscript_14177659.zip`

SHA-256:

`737500db81b9da24a46d05eea4c5864a3dd4ac494470ceb38b6a74cab168dc73`

It will be deposited as a separate archival artifact and linked from the first tagged release.

The numerical comparisons are intentionally reported together with their failed as well as passed accuracy diagnostics; see [provenance/NUMERICAL_STATUS.md](provenance/NUMERICAL_STATUS.md).

## Reproducibility boundary

The repository separates three tasks:

1. **Rebuild figures and manuscript tables** from compact saved summaries.
2. **Verify analytical calculations** with lightweight numerical checks.
3. **Refit the empirical models**, which requires the original third-party data and substantially more computation.

A successful figure rebuild is not a refit, and a successful consistency check is not evidence that the dynamic model-family comparison has numerically converged.

## Archival release

The repository is currently being prepared for its first tagged archival release. The release checklist is in [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md). The software release and the large dynamic numerical artifact will receive persistent archival records before submission/publication materials are finalized.

## License

Original RFDT software in this repository is released under the MIT License. Third-party datasets, code and benchmark materials retain their original licenses and access conditions; see [DATA_SOURCES.md](DATA_SOURCES.md).
