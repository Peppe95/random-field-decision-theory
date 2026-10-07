# Production source archive

The corrected Guo production source was recovered from:

`RFDT_Guo_corrected_analysis_code_v1.zip`

The `rfdt_correction` package, command-line runner, unit tests, protocol files, and inherited simulator source are now included under `analysis/dynamic/`.

The source files used for the completed fits match the SHA-256 values recorded in `SOURCE_HASHES.md`. The check is automated by:

```bash
python analysis/dynamic/verify_frozen_sources.py
```

The large generated prediction archive is intentionally kept out of Git:

`rfdt_guo_precision_for_manuscript_14177659.zip`

Its recorded SHA-256 is:

`737500db81b9da24a46d05eea4c5864a3dd4ac494470ceb38b6a74cab168dc73`

That archive is intended for a separate persistent deposit rather than Git history.
