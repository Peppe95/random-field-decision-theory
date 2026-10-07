# Numerical status

The dynamic analysis reports four distinct numerical calculations:
- original fits from 2,000-path training tables;
- an independent training simulation and refit at the same bank size;
- enlarged candidate sets refitted within the same parameter ranges;
- independent fixed-weight evaluation of the original fits using four new 4,000-path batches.

The recorded accuracy criteria were:
- Monte Carlo 95% half-width no larger than `min(0.001, 0.1 * participant-CI half-width)`;
- pseudocount sensitivity no larger than 0.001 nats/trial;
- training-replica and candidate-set shifts no larger than `min(0.005, 0.25 * participant-CI half-width)`.

Only the two sensitivity-sample window comparisons met all fixed-weight criteria. Primary-window comparisons passed pseudocount sensitivity but narrowly exceeded Monte Carlo limits. All four all-RT comparisons exceeded both Monte Carlo and pseudocount criteria. One independent-training comparison and all eight candidate-enlargement comparisons exceeded their score-shift thresholds.

The directional findings that remained stable across the completed checks are:
- nonlinear DDM predicts the full single-click response set better than RFDT;
- full RFDT predicts better than the separately refitted uncoupled RFDT control;
- in the separately refitted 0.3--10 s window, RFDT has a timing advantage and a choice disadvantage relative to DDM, while the overall ranking depends on numerical version.

No accuracy threshold was relaxed after observing the results.
