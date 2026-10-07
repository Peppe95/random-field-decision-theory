# Reproduction

The repository separates figure/table reproduction from full model fitting.

## Figures and tables

Use the compact result files under `results/` and scripts under `figures/`. This route does not rerun fitting or the finite-population simulator.

## Static analysis

The exact frozen production runners are:

```bash
python analysis/static/hab22/run_hab22_final_frozen.py --help
python analysis/static/choices13k/run_choices13k_external.py --help
```

Full refits require the upstream archives listed in `DATA_SOURCES.md`; raw third-party data are not stored here.

To rebuild the static manuscript figure from compact saved summaries only:

```bash
python figures/static/make_figure4_static.py
```

See `analysis/static/README.md` and `analysis/static/config_reference.json` for the frozen parameter bounds, normalization, fold construction, objectives, optimizer settings and seeds.

## Dynamic analysis

The corrected production entry point is:

```bash
python analysis/dynamic/run_rfdt_guo_correction.py --help
```

The main stages are `prepare`, `validate`, `atlas`, `fit`, `precision`, and `convergence`.

The production study contains:
- the base fit;
- an independent training replica;
- enlarged candidate sets;
- independent fixed-weight evaluation of the original fits using four new batches of 4,000 paths.

The full dynamic calculation is expensive. The exact settings are recorded in `analysis/dynamic/config_reference.json`.

## Large numerical archive

The manuscript prediction archive is not stored in Git:

`rfdt_guo_precision_for_manuscript_14177659.zip`

Recorded SHA-256:

`737500db81b9da24a46d05eea4c5864a3dd4ac494470ceb38b6a74cab168dc73`

The release will link to its Zenodo record.

## Data

Download source data from the locations in `DATA_SOURCES.md`. Keep local copies under `data/raw/`; that path is ignored by Git.
