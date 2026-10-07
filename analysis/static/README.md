# Frozen static analyses

This directory contains the exact production implementation used for the static RFDT results.

## Status

- **HAB22 is the development benchmark.** The signed-power RFDT-P form was finalized there.
- **Choices13k is the external validation.** Its functional form, bounds and normalization were frozen before human Choices13k outcomes were inspected.
- No code in this module changes the frozen model.

Frozen RFDT-P:
`u_rho(x) = sign(x) * |x|^rho`,
with `alpha in [0,1]`, `beta in [0.05,4]`, `kappa in [0,0.995]`,
`lambda in [0.1,8]`, and `rho in [0.05,1.5]`.

For every candidate `rho`, comparative evidence uses the stimulus-only scale

`median_j sqrt(E[D_j(rho)^2]) = 1`.

## Layout

- `hab22/` — exact final runner, imported RFDT module, RData reader, saved analysis specification and aggregate fold accounting.
- `choices13k/` — exact external-validation runner, its imported RFDT module, saved analysis specification and fold accounting.
- `../../results/static/` — compact result tables and Figure 4 inputs only; no participant-level third-party data.
- `../../figures/static/` — Figure 4 generation from those compact inputs.

The two `rfdt_power.py` files are retained separately on purpose. The Choices13k version contains the even-sample median handling required by the 1,766-problem design.

## Dependencies

```bash
python3 -m pip install -r analysis/static/requirements.txt
```

## Full refits

Full refits require locally downloaded upstream data listed in `DATA_SOURCES.md`.

HAB22 production entry point:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python analysis/static/hab22/run_hab22_final_frozen.py \
  --plonsky /path/to/Plonsky_RFDT.zip \
  --recovery /path/to/rfdt_phase3b_static_recovery_output.zip \
  --out hab22_rfdt_final_frozen_output \
  --jobs 12 --starts 24 --recovery-reps 18 --recovery-starts 24 \
  --individual --individual-starts 4
```

This is the final package's recommended command. The saved HAB22 result archive does not retain the historical command line or per-start logs, so the exact executed `--individual-starts` flag cannot be independently certified from the output alone. The production code and frozen saved predictions/results are the authoritative record.

Choices13k production entry point:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python analysis/static/choices13k/run_choices13k_external.py \
  --choices /path/to/choices13k-main.zip \
  --out choices13k_rfdt_external_output \
  --jobs 10 --starts 32
```

## Objectives and optimizer

### HAB22

- Training objective: binary-response/binomial negative log likelihood.
- Optimizer: L-BFGS-B with analytic gradients.
- `ftol=1e-11`, `gtol=2e-7`, `maxiter=1000`, `maxls=80`.
- Two fixed starts plus scrambled Halton starts; seed `20260910`.
- Task folds: upstream stored `crossValidation_id` (10 folds).
- Participant folds: exact Plonsky recreation within context using the embedded R-compatible `set.seed(123)` implementation.
- Participant-mixture benchmark and pooled restrictions are **different fitting architectures** and must not be mixed.

Participant-level fold identifiers, predictions and parameter estimates are not committed. The runner reconstructs the folds from the upstream archive. `hab22_participant_fold_counts.csv` records aggregate fold counts only.

### Choices13k

- Frozen subset: `Feedback=False`, `Amb=False`, `Corr=0` (1,766 problems).
- Training objective: unweighted problem-level MSE on aggregate `bRate`.
- Optimizer: L-BFGS-B with analytic gradients.
- `ftol=1e-13`, `gtol=1e-8`, `maxiter=1500`, `maxls=100`.
- Two fixed starts plus 32 scrambled Halton starts per model/fold.
- Fold seed: `20260911`; fold/model fit seed is `20260911 + fold`.
- Ten deterministic problem folds formed by seeded permutation then round-robin assignment.

## Reproducing manuscript tables and Figure 4

No fitting is needed:

```bash
python figures/static/make_figure4_static.py
```

The manuscript scores are in `results/static/table_B1_hab22_selected_scores.csv` and `results/static/table_B2_static_restrictions.csv`.
