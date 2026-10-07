# Concrete decisions implementing the supplied corrective specification

This is a post-result corrective analysis. It is not called preregistered,
confirmatory or outcome-blind in the historical sense. New candidate banks are
stimulus-only, but the decision to perform this correction followed inspection
of historical human results. These decisions are locked by `prepare`, before
any corrected human fitting or comparison.

## Scope and observation target

The supplied NEXT_DYNAMIC_ANALYSIS_SPEC.md is retained verbatim. This runner
implements the three-model core, both single-click observation targets, both
catch-rule samples, numerical prerequisites, independent evaluation, an
independent training/refitting replicate, and a predetermined larger-bank check.
The optional legacy-all-trial refit is deliberately NOT included in this first
corrected pipeline. It is optional in the supplied specification and is not
needed to define the primary process-specific target. No historical five-bin
atlas is presented as if it contained the new tail information.

`change_mind == 1` is operationally **single click**, not a proved psychological
absence of revision. The processed `rt` field is used under that restriction;
we do not reassert the discredited claim that this column always means first
click. The raw reconciliation reported by the main chat is not re-run here.

The finest table has 14 cells: safe then risky, with seven ordered time bins
within each choice. Cutoffs are 0.3, the four inherited internal cutoffs, 10.
Equality at 10 belongs to the final within-window cell; equality at internal
cutoffs belongs to the cell to their right. All finite single-click tails are
retained. Nonfinite or negative single-click RTs trigger an error rather than an
unannounced exclusion.

The secondary window model fits **candidate-level conditional distributions**
q_theta / Z_theta and averages those conditional models using their fitted
weights. It is not obtained by conditioning the unconditional posterior mixture.
No click-count generation model is introduced. No-training participants get the
fitted population distribution; canonical participant bookkeeping is preserved.

## Same microscopic model, changed numerical operations

Four simulator functions are extracted verbatim from the validated
run_rfdt_phase2b_recovery_v3.py. The original file and function hashes are
included. Initial spins and persistent/adaptive fields, r=1, m_-i, adaptive-field
heat-bath probabilities, NA=round(alpha*N), event selection and exact
between-jump first passage are not rewritten.

The caller extends the computational horizon from 90 to **5760 internal units**
(64 times the former cutoff). Paths are followed until first passage. A cell
with any unresolved trajectory is saved as a failed diagnostic and is NOT
normalized, censored into a human category or quietly retried until it succeeds.
Counts exceeding 90 and maximum realized internal times are saved. Zero
observed unresolved paths is not claimed as a proof of zero population tail
probability. The long human RT category remains RT > 10, unrelated to this
computational guard.

RFDT probabilities use one specified total Dirichlet mass **0.5**, not 0.5 per
cell. Its base is uniform over structurally possible fine cells. Since |m|<=1,
RFDT RT is at least t0 + tau_s*bound; impossible early cells have zero base mass.
Choice marginals and conditional RT probabilities are always derived from the
single 14-cell table, including after posterior averaging. This is numerical
regularization, NOT an estimated contamination mechanism. Analytic DDM tables
receive no Monte Carlo pseudo-observations. Fixed-weight smoothing sensitivities
use total mass 0.05 and 5 with the same raw counts. These are not refitted
alternative models and are labelled as such.

## Empirical Bayes

We optimize the normalized finite-bank marginal likelihood, not the historical
posterior-moment iteration. Optimization uses unit coordinates, mu in [0,1],
sd in [0.055,0.48], matching the previous sd bounds and the range of the
historical mean update. The sd coordinates optimized are log(sd).

Four starts are used: centred sd .28; centred sd .48; one training-posterior
moment start; centred sd .11. The moment is ONLY a start, never the update rule.
L-BFGS-B uses an analytic gradient including finite-bank normalization. The
accepted objective traces, each start's outcome, hyperparameter boundaries and
projected gradients are retained. A predeclared second solve tightens ftol if
the projected gradient is not adequate. A same-objective SLSQP fallback is
predeclared if L-BFGS-B still fails the diagnostic; it does not change the prior
family, objective, parameter supports, or data. Any candidate whose best result fails
the projected-gradient threshold 1e-4 on the mean-per-canonical-participant
objective causes a stop before that model's held-out prediction.

The global prior is the original discrete quadrature measure. Global weights
are proportional to this prior times exp(profile training marginal log
likelihood). These are **empirical-Bayes discrete model/quadrature weights**,
not a full continuous hierarchical posterior or integrated Bayesian evidence.
All numerically nonzero global weights enter predictions, without a top-48 or
cumulative-mass truncation. Conditional nuisance weights are saved in float64
for every global/fold/participant, including zero-training participants.

## Fixed models and bank resolution

- RFDT: shared alpha, kappa, N_eff; participant rho, beta, bound, tau_s, t0.
- RFDT_K0: same, kappa fixed at zero; all remaining parameters refitted.
- DDM: participant rho, signed vcoeff, vmax, a, z, t0;
  v = vmax*tanh(vcoeff*DeltaV/2), within-trial diffusion coefficient fixed at 1.

Physical supports and original bank seeds are in config.json. They match the
last heterogeneous audit, including the signed DDM coefficient. No equal-
complexity claim is made. RFDT rho is [.05,1.5]; DDM rho is [.05,3].

Base resolution: RFDT 96 alpha/kappa points crossed with N={80,120,180};
192 nuisance candidates; DDM 4096 candidates. The control projects the same
alpha/N measure onto kappa=0. Exactly identical fixed-count processes are
collapsed and their prior masses added; this yields 222 control global rows
at base resolution. This is not selection using outcomes.

Predetermined larger bank: RFDT 128 global points crossed with the same N and
256 nuisance candidates; DDM 8192 candidates. Existing Sobol prefixes are
retained. Existing RFDT cell streams can be reused; appended cells are new.
This is a numerical resolution comparison, not an alternative mechanism.

## Numerical replication and stopping diagnostics

Initial RFDT training count: **2000 paths per candidate/problem cell**, four
 times the historical 500. Two independent batch namespaces, train_a/train_b,
are available. Both are refitted to distinguish training-atlas noise from
prediction-evaluation noise. The analytic DDM does not change across those
simulation batches and its identical fits may be reused.

Independent evaluation uses four batches of 4000 paths for all nuisance cells
of all numerically nonzero global candidates in the reference fits, unioned
over samples and targets. If inadequate, the fixed schedule is 8000, then
16000 paths/batch, in new independent namespaces. Raw counts are pooled BEFORE
applying the single total pseudo-count. The saved weights are not refitted in
this stage. Delete-one-batch jackknife with a t interval (df=batches-1) provides
an approximate MC uncertainty estimate, separately from participant uncertainty.
The proposed target is a 95% MC half-width <= .001 nat/trial and <= 10% of the
participant-bootstrap half-width for **each** planned comparison and target.
Fixed-weight smoothing shifts must also be <= .001 nat/trial. Few-batch error
estimates are approximate; a pass is a practical resolution statement, not a
proof of negligible bias.

For the independent training replica and larger-bank check, the declared
absolute shift diagnostic is <= min(.005 nat/trial, 25% of the reference
participant-bootstrap half-width). This additional practical threshold is an
implementation decision, not a number specified by the main chat. All paired
directions are reported; these checks do not pick whichever model wins. A failed
check is returned for review rather than launching a new model family.

The bootstrap resamples canonical participant score SUMS and eligible-trial
COUNTS, then divides total resampled sums by total resampled counts. It does not
replace the trial-average target with an average of individual means. Zero-row
participants remain in bookkeeping. These intervals condition on the fixed
problem design and numerical model specification.

## What is not established just by running the code

A software test does not validate the entire historical mechanism recovery,
prove parameter identification, establish that every RT outlier was generated
by first passage, make post-result analyses confirmatory, or ensure a particular
model comparison. No new scientific conclusion is generated by this package's
validation tests. Production human fitting is performed only on the user's
machine after validation.
