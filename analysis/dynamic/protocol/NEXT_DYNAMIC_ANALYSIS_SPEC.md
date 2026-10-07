# Bounded dynamic correction: handoff to the implementation chat

**Status:** proposed protocol written after inspection of the historical results and discovery of implementation defects. It is not preregistered confirmation, and it must not be retrospectively labelled outcome-blind. Freeze the corrective protocol and its numerical stopping criteria before inspecting corrected held-out comparisons.

## Objective

Correct the probability and estimation operations, isolate a defensible single-click process comparison, determine numerical stability, and test the necessity of interaction with one control. Do not restart static benchmarking or add new valuation/decision mechanisms.

## Preserve

Keep all original results immutable, under a legacy directory. Preserve the signed-power utility construction (transform raw rewards before normalization), product evidence, exact finite-N simulator, random initial spins/fields, fixed r=1, and initial physical parameter supports. Keep the 63 Guo problems, seven modular problem folds, primary literal-catch sample and count-matching sensitivity sample. Do not select trials by which model predicts them better.

The original model comparisons are historical working-score comparisons. Do not describe them as a verified likelihood for the complete choice/revision/timing observation process.

## A. Numerical prerequisites (both RFDT and DDM)

### One primitive joint distribution

Construct a single nonnegative joint table over choice and a declared exhaustive RT partition. Normalize it once. Derive every choice marginal by summing this table, and every conditional RT probability by dividing by the corresponding derived choice marginal.

Required invariants at candidate level AND after posterior averaging:

- total joint mass = 1 (including only genuine declared observation categories);
- P(C=c) = sum_b P(C=c,B=b);
- sum_b P(B=b | C=c) = 1 whenever P(C=c)>0;
- probabilities are finite/nonnegative;
- the saved score equals the log of the saved observed-event probability;
- all model comparisons use identical rows and bin definitions.

Do not maintain independent `PSEUDO_CHOICE` and `PSEUDO_JOINT` conventions. If a numerical Dirichlet smoothing rule is retained, use a specified total pseudo-count mass and one base distribution on the finest event partition; coarser pseudo-counts are sums of the finer ones. Document the sensitivity to simulation sample size and smoothing. An explicit measurement-contamination mixture is a separate modeling assumption; do not silently conflate it with Monte Carlo smoothing.

Computational truncation at internal time 90 is not an observed human deadline. Record unresolved simulations and extend the horizon or otherwise bound their contribution. Do not invent timeout probability by smoothing a non-observed timeout category. The microscopic dynamics must remain unchanged.

A cheap legacy diagnostic can derive coherent `logc` by log-summing the old `logj` arrays, retaining the old 11th category only for faithful numerical comparison. It cannot reconstruct the new window-tail cells or solve the observation-selection problem.

### Correct finite-grid empirical Bayes

The existing Gaussian-shaped prior is normalized on the finite candidate bank. Optimize its actual marginal log likelihood, including derivatives of this normalization. Replace the untruncated Gaussian posterior-moment heuristic. Use the same correction for all model families.

`numerical_repair_core.py` supplies tested gradients, coherent marginalization, and a bounded multistart optimizer. It is a building block, not a complete production patch. Verify gradients independently; record optimizer/KKT diagnostics and objective traces. Never use test scores to choose starts, tolerances, candidate supports or hyperparameters.

Global weights based on plug-in optimized hyperparameters are empirical-Bayes quadrature/model weights, not a full continuous hierarchical posterior. Use the accurate terminology.

### DDM numerical evaluator

Use a tolerance-controlled short-time/long-time representation or an independently validated solver. The supplied 200-term implementation was mostly accurate on the regenerated heterogeneous bank, but maximum bin error was ~3.06e-4. Check the entire candidate domain used in the new run, not just one central parameter point. Verify complement, label symmetry, nonnegative bins, total hitting probability, and convergence with tolerance.

## B. Corrected process-specific observation analysis

### Recommended target: single-click records, with RT tails explicitly coarsened

Use `change_mind == 1` as an operational single-click indicator, not as a psychological no-revision diagnosis. Restrict the process likelihood to those rows. Do not feed a multiple-click final choice into a first-passage likelihood under an untested identification of final with first choice.

The primary canonical sample contains 41,022 such rows from 897 contributing participants; the sensitivity sample contains 42,522 from 934. Keep the complete canonical participant/fold bookkeeping. A participant with no training process rows receives the population predictive distribution; do not exclude them because of the number or content of held-out observations.

For each problem j retain the original four simulated internal bin edges e1j,...,e4j. All lie inside [0.3,10] in this bundle. Define seven exhaustive RT categories:

1. RT < 0.3;
2. 0.3 <= RT < e1j;
3. e1j <= RT < e2j;
4. e2j <= RT < e3j;
5. e3j <= RT < e4j;
6. e4j <= RT <= 10;
7. RT > 10.

Together with two choices this gives a **14-cell observation space**. The short and long tails preserve the known information without modelling the exact magnitude of a 40-second or 200-second response. This does not prove outliers are generated by the decision process. It is a deliberately explicit coarsened observation model, preferable to silently discarding known tail information. Report tail contributions separately.

The original five-bin histograms cannot recover these 14 cells: their endpoint bins conflate within-window and outside-window mass. New RFDT bin counts/trajectories or sufficient retained raw simulation times are required. For DDM the additional probabilities are analytic CDF differences.

No fitted click-count mechanism is introduced. Claims are restricted to prediction in the operationally defined single-click subset; do not generalize these likelihoods to the entire raw click/revision process.

### Predefined secondary check: conditional within-window likelihood

The same 14-cell primitive table can support a within-window sensitivity with no extra RFDT simulations. For each fixed parameter candidate, define

q^W_theta(c,b) = q_theta(c,b) / Z_theta,
Z_theta = sum_{c,b in the five within-window bins} q_theta(c,b).

Fit the conditional selected-window model on the 39,446 primary / 40,876 sensitivity in-window single-click rows, using these normalized probabilities consistently in both training and prediction. Specify explicitly that this is a conditional-on-selection working model. Do not confuse a mixture of candidate-level conditional models with conditioning an unconditional posterior-predictive mixture: the latter instead reweights by each candidate's Z_theta. State the adopted sampling interpretation and compute the chosen convention coherently. Do not silently switch conventions between training and test.

This secondary check assesses whether the comparative conclusion depends on modeling the observed tails. It is not used to select whichever result favors RFDT. Always report both directions and their estimands.

### Legacy all-trial working-score analysis

A coherent-margin re-fit under the original single-click-RT/otherwise-choice rule may be reported in the appendix to isolate the numerical repair from the change in process target. Its click-count/selection assumptions remain unmodelled. It is not a replacement for the process-specific check above. If computational cost is a concern, perform this only for the two strongest historical models, using the user's existing local atlases after the coherence fix.

## C. Fixed model set

Core comparison:

- RFDT with participant-level rho, beta, bound, tau_s and t0; population alpha, kappa, N_eff.
- The existing nonlinear DDM with participant-level rho and the recorded signed value coefficient, vmax, boundary, start and t0 support.

Do not describe these as identical complexity. They match the availability of utility heterogeneity, but have different supports and finite-bank resolutions. Preserve the stronger DDM specification and disclose that its signed coefficient allows negative value-to-drift slopes.

One mechanistic control:

- RFDT kappa=0 with the same utility heterogeneity, initial fields, adaptation, noise, bounds and observation protocol. Refit all remaining parameters on each training fold. Do not substitute kappa=0 into a full-model posterior.

This tests the predictive contribution of coupling. Do not add alpha-endpoint, collapsing-bound, new probability-weighting, or other models unless a specific implementation failure prevents the predeclared comparisons. If kappa=0 performs similarly, report that and restrict mechanistic claims.

## D. Numerical precision and reproducibility

First run a small software test with no scientific interpretation. Then:

1. Save raw event counts or reusable simulated first-passage draws, not only float32 logs. Save per-candidate accepted/tail/timeout counts and full candidate coordinates.
2. Save all fold-specific EB hyperparameters, scores, diagnostics, global weights, and participant nuisance posterior weights. Full posterior predictions are not a substitute.
3. Use independent simulation batches to re-evaluate the same corrected frozen predictive mixtures. Start with a several-fold increase over 500 paths for relevant candidate/problem cells; choose adequacy by measured stability, not a fixed sample number alone.
4. Suggested reporting target: simulation-induced uncertainty in the paired mean held-out score difference below ~0.001 nat/trial, and appreciably smaller than its participant-sampling uncertainty. This is a proposed numerical tolerance, not a discovered property of the current run. If not attained, increase precision or qualify the resolution of the comparison.
5. Re-evaluation with fixed weights does not test training-atlas noise. Include an independent training-atlas/refitting replicate or a principled controlled refinement sufficient to assess whether fitted mixtures and rankings are stable. Keep test outcomes out of adaptation decisions.
6. Assess finite-candidate resolution separately from simulation count. A modest predetermined bank expansion/replication within the existing physical ranges is a numerical check, not a new mechanism. Do not truncate the stronger DDM bank merely to give RFDT equal candidate counts.
7. For variable numbers of eligible trials per participant, estimate the trial-average difference as total score difference divided by total included trials. In participant-cluster bootstrap, resample participant score **sums and counts**, then take their ratio. Do not silently substitute an equally weighted mean of individual averages; that changes the estimand.

Do not infer parameter identification from predictive stability or from correlations of point-estimation errors across truths. Earlier recovery studies should be labelled by their actual utility, observation and estimator specifications.

## E. Outputs to return (compact bundle)

- frozen corrective protocol and configuration, source/self/input hashes and software versions;
- source and unit tests;
- source count/probability normalization checks;
- full fold-by-model fit diagnostics, including boundary diagnostics in fitting coordinates;
- saved posterior weights (may be a separate local archive if large) and all small fit objects;
- trial-level held-out joint probabilities, derived choice and conditional probabilities, observed events and eligibility metadata;
- paired score tables and participant-cluster intervals for core and kappa=0 comparisons;
- tail/within-window decomposition, sample-flow table, simulation/grid-convergence diagnostics;
- explicit distinction between historical, mechanically corrected diagnostic, and newly fitted results.

## F. Figures and manuscript after numerical results are settled

Replace the dynamic main figure only after the corrected outputs are frozen. Show the process-specific full RFDT/DDM comparison and kappa control; derive the choice/RT decomposition from one coherent probability table; provide calibrated normalized RT probabilities with tails distinguished. Put the long historical competitor-strengthening sequence and numerical convergence checks in the appendix rather than suggesting all scores use an identical estimand.

Keep the static figure and static numerical results unchanged. Amend timing terminology, uncertainty language and the theory's conditional diffusion discussion. Then write the Discussion. A loss or absence of a coupling advantage is a possible scientific outcome, not a reason for additional post-hoc model modifications.
