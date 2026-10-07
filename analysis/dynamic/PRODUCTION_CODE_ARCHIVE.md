# Production source archive

The corrected Guo production source was recovered from:

`RFDT_Guo_corrected_analysis_code_v1.zip`

SHA-256 of the recovered source archive:

`b7b428e4f84673e0a5d4f17ee3d556583dd8a292ae6d049aaabd63a29a22b21c`

The `rfdt_correction` package, command-line runner, unit tests, protocol files, and inherited simulator source are included under `analysis/dynamic/`.

The source files used for the completed fits match the SHA-256 values recorded in `SOURCE_HASHES.md`. Check them with:

```bash
python analysis/dynamic/verify_frozen_sources.py
```

The large generated prediction archive is intentionally kept out of Git:

`rfdt_guo_precision_for_manuscript_14177659.zip`

Its recorded SHA-256 is:

`737500db81b9da24a46d05eea4c5864a3dd4ac494470ceb38b6a74cab168dc73`

That archive belongs in a separate persistent deposit rather than in Git history.
