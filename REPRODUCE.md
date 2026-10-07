# Reproduction

The repository separates lightweight manuscript reproduction from full model fitting.

## 1. Install the lightweight environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. Analytical checks and illustrations

```bash
python analysis/theory/make_theory_predictions.py
```

This script reads no human observations or fitted empirical parameters. It writes its numerical checks to `analysis/theory/theory_checks/` and regenerates its illustrative theory figures.

The saved reference diagnostics are in `analysis/theory/theory_validation.json`.

## 3. Rebuild manuscript figures from compact results

No model fitting or finite-population simulation is required for these commands:

```bash
python figures/static/make_figure4_static.py
python figures/dynamic/make_figures_dynamic.py
```

The static script reads `results/static/`. The dynamic script reads only compact verified summaries under `results/dynamic/`.

## 4. Static empirical analyses

The exact frozen production runners are:

```bash
python analysis/static/hab22/run_hab22_final_frozen.py --help
python analysis/static/choices13k/run_choices13k_external.py --help
```

Full refits require the upstream archives listed in `DATA_SOURCES.md`; raw third-party data are not stored here.

See `analysis/static/README.md` and `analysis/static/config_reference.json` for the frozen parameter bounds, normalization, fold construction, objectives, optimizer settings and seeds.

## 5. Dynamic empirical analysis

The corrected command-line entry point is:

```bash
python analysis/dynamic/run_rfdt_guo_correction.py --help
```

Its stages are `prepare`, `validate`, `atlas`, `fit`, `precision`, `convergence`, and `package`.

**Pre-release source requirement.** The runner imports the frozen `rfdt_correction` package used for the completed fits. Those modules and `tests/test_core.py` must be copied verbatim from `RFDT_Guo_corrected_analysis_code_v1.zip` and verified against `analysis/dynamic/SOURCE_HASHES.md` before a full dynamic refit is advertised as reproducible. This is tracked in GitHub issue #2.

The production study contains:
- the base fit;
- an independent training replica;
- enlarged candidate sets;
- independent fixed-weight evaluation of the original fits using four new batches of 4,000 paths.

The exact settings are recorded in `analysis/dynamic/config_reference.json`. The numerical status, including criteria that were not attained, is recorded in `provenance/NUMERICAL_STATUS.md`.

## 6. Large dynamic numerical archive

The manuscript prediction archive is not stored in Git:

`rfdt_guo_precision_for_manuscript_14177659.zip`

Recorded SHA-256:

`737500db81b9da24a46d05eea4c5864a3dd4ac494470ceb38b6a74cab168dc73`

The first archival release will link to its persistent record.

## 7. Data

Download source data from the locations in `DATA_SOURCES.md`. Keep local copies under `data/raw/`; that path is ignored by Git.

The repository does not relicense or redistribute the upstream participant datasets.
