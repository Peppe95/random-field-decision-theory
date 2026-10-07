# First release checklist

## Code and reproduction

- [x] Static production runners and frozen configuration are committed.
- [x] Static source hashes and compact result summaries are committed.
- [x] Frozen dynamic production modules and unit tests are committed from the archived source.
- [x] Dynamic production source hashes are checked automatically.
- [x] The archived dynamic unit tests pass locally.
- [x] Theory checks and illustrative theory-generation code are separated from empirical fitting.
- [x] Static and dynamic manuscript figures can be rebuilt from compact saved summaries.
- [x] Third-party participant data are excluded from Git and documented in `DATA_SOURCES.md`.
- [x] Dynamic numerical limitations are retained in `provenance/NUMERICAL_STATUS.md`.
- [x] The large dynamic archive has a recorded SHA-256.
- [x] Repository CI passes with the restored dynamic source.

## GitHub metadata

These settings are changed from the repository's **About** panel:

- [ ] Add a short repository description.
- [ ] Add topics such as `decision-theory`, `risky-choice`, `cognitive-modeling`, `response-time`, `stochastic-processes`, and `mathematical-psychology`.
- [x] GitHub recognizes the root license as MIT.

## Archival release

- [x] Deposit the large dynamic numerical archive separately and record its persistent identifier: https://doi.org/10.5281/zenodo.23213576
- [x] Create GitHub tag/release `v1.0.0`.
- [x] Archive that tagged software release and record its DOI in `CITATION.cff` and `README.md`: https://doi.org/10.5281/zenodo.23213829
- [x] Add `version` and `date-released` to `CITATION.cff` at release time.
- [ ] Add the paper as `preferred-citation` once the article has a stable citation/DOI.
