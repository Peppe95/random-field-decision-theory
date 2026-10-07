# Random Field Decision Theory

Reproducibility materials for **Random Field Decision Theory: Deliberation through persistent and adaptive representations**, by Giuseppe M. Ferro and Didier Sornette.

The repository mirrors the empirical structure of the paper:

- `analysis/static/` — HAB22 and Choices13k
- `analysis/dynamic/` — Guo choice and response-time analysis
- `results/` — compact tables used to check the reported results
- `figures/` — figure-generation code and small inputs
- `data/` — instructions for obtaining the original datasets
- `provenance/` — checksums and numerical records

The source datasets were collected by other research teams and are **not copied into this repository**. Download them from the original sources listed in [DATA_SOURCES.md](DATA_SOURCES.md).

## Dynamic analysis

The final Guo analysis uses signed-power utility, the exact finite-population first-passage process, single-click observations, and normalized empirical-Bayes fitting. Earlier linear-utility recovery code and earlier human-analysis pipelines are not part of the final analysis.

The large manuscript prediction archive is kept outside Git:

`rfdt_guo_precision_for_manuscript_14177659.zip`

SHA-256:

`737500db81b9da24a46d05eea4c5864a3dd4ac494470ceb38b6a74cab168dc73`

It should be deposited on Zenodo with the final release.

## Static analysis

The static model is frozen. The exact final HAB22 and Choices13k production export will be added under `analysis/static/`; the directory already states the expected layout and data policy.

## Reproduction

See [REPRODUCE.md](REPRODUCE.md). Rebuilding figures from saved summaries is deliberately separated from the expensive fitting and simulation stages.

## License

The MIT license applies to original RFDT software in this repository. Third-party data and code retain their original terms.
