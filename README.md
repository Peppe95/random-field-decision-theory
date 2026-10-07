# Random Field Decision Theory

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23213829.svg)](https://doi.org/10.5281/zenodo.23213829)

Reproducibility materials for **Random Field Decision Theory: Deliberation through persistent and adaptive representations**, by Giuseppe M. Ferro and Didier Sornette.

The repository is organized around the analyses reported in the manuscript.

- `analysis/theory/` — numerical checks and illustrations for the analytical results
- `analysis/static/` — frozen HAB22 and Choices13k analyses
- `analysis/dynamic/` — corrected Guo choice and response-time analysis
- `results/` — compact tables used to verify manuscript results
- `figures/` — figure-generation code using compact saved results
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

Check the frozen dynamic source and run its unit tests:

```bash
python analysis/dynamic/verify_frozen_sources.py
PYTHONPATH=analysis/dynamic python -m unittest discover -s analysis/dynamic/tests -v
```

Rebuild the manuscript summary figures without refitting the models:

```bash
python figures/static/make_figure4_static.py
python figures/dynamic/make_figures_dynamic.py
```

Run the analytical checks:

```bash
python analysis/theory/make_theory_predictions.py
```

Full empirical refits require the upstream datasets and substantially more computation. See [REPRODUCE.md](REPRODUCE.md).

## Static analysis

HAB22 is the development benchmark. Choices13k is the external validation of the HAB22-frozen signed-power RFDT-P specification.

The participant-mixture HAB22 benchmark and the pooled restriction analysis are different fitting architectures and are kept separate. The production implementation, frozen settings, fold accounting and source hashes are under `analysis/static/`.

## Dynamic analysis

The final Guo analysis uses signed-power utility, the exact finite-population first-passage process, single-click observations, and normalized empirical-Bayes fitting.

The frozen production package is included under `analysis/dynamic/` and checked against the hashes recorded during the completed analysis. Earlier linear-utility recovery code and earlier human-analysis pipelines are not part of the reported analysis.

The large manuscript prediction archive is kept outside Git:

`rfdt_guo_precision_for_manuscript_14177659.zip`

SHA-256:

`737500db81b9da24a46d05eea4c5864a3dd4ac494470ceb38b6a74cab168dc73`

Archived numerical artifact: https://doi.org/10.5281/zenodo.23213576

The numerical comparisons are reported together with their failed as well as passed accuracy diagnostics; see [provenance/NUMERICAL_STATUS.md](provenance/NUMERICAL_STATUS.md).

## Reproducibility boundary

The repository separates three tasks:

1. **Rebuild figures and manuscript tables** from compact saved summaries.
2. **Verify analytical and implementation checks** without refitting the human data.
3. **Refit the empirical models**, which requires the original third-party data and substantially more computation.

A successful figure rebuild is not a refit, and a successful consistency check is not evidence that the dynamic model-family comparison has numerically converged.

## Archival release

Version 1.0.0 of the software is archived at https://doi.org/10.5281/zenodo.23213829. The large dynamic numerical artifact is archived separately at https://doi.org/10.5281/zenodo.23213576. The remaining release steps are listed in [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).

## License

Original RFDT software in this repository is released under the MIT License. Third-party datasets, code and benchmark materials retain their original licenses and access conditions; see [DATA_SOURCES.md](DATA_SOURCES.md).
