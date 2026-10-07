# First release checklist

Do not create the first tagged release until the items under **Blocker** are complete.

## Blocker

- [ ] Copy the frozen dynamic `rfdt_correction/` modules and `tests/test_core.py` verbatim from `RFDT_Guo_corrected_analysis_code_v1.zip`.
- [ ] Verify every copied dynamic source file against `analysis/dynamic/SOURCE_HASHES.md`.
- [ ] Run the dynamic validation/smoke tests in a clean environment and record the result.

This is tracked in GitHub issue #2.

## Repository checks

- [x] Static production runners and frozen configuration are committed.
- [x] Static source hashes and compact result summaries are committed.
- [x] Theory checks and illustrative theory-generation code are separated from empirical fitting.
- [x] Static and dynamic manuscript figures can be rebuilt from compact saved summaries.
- [x] Third-party participant data are excluded from Git and documented in `DATA_SOURCES.md`.
- [x] Dynamic numerical limitations are retained in `provenance/NUMERICAL_STATUS.md`.
- [x] The large dynamic archive has a recorded SHA-256.
- [ ] Confirm `python -m pip install -r requirements.txt` in a clean environment.
- [ ] Confirm the lightweight theory/static/dynamic figure commands in `REPRODUCE.md`.

## GitHub metadata

These settings are changed from the repository's **About** panel rather than from tracked files:

- [ ] Add a short repository description.
- [ ] Add topics such as `decision-theory`, `risky-choice`, `cognitive-modeling`, `response-time`, `stochastic-processes`, and `mathematical-psychology`.
- [ ] Confirm GitHub recognizes the root license as MIT after the canonical license file is merged.

## Archival release

- [ ] Deposit the large dynamic numerical archive separately and record its persistent identifier.
- [ ] Create GitHub tag/release `v1.0.0` only after the blocker above is closed.
- [ ] Archive that tagged software release and record its DOI in `CITATION.cff` and `README.md`.
- [ ] Add `version` and `date-released` to `CITATION.cff` at release time.
- [ ] Add the paper as `preferred-citation` once the article has a stable citation/DOI.
