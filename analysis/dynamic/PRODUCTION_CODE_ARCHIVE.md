# Production module archive

The completed Guo fits recorded hashes for the `rfdt_correction` package listed in `SOURCE_HASHES.md`.

The source archive used for those fits is:

`RFDT_Guo_corrected_analysis_code_v1.zip`

The command-line runner and frozen configuration are already tracked here. The remaining imported modules should be copied verbatim from that archive before the first public release. They should not be reconstructed from the manuscript or rewritten for style.

Expected module paths:

```
analysis/dynamic/rfdt_correction/__init__.py
analysis/dynamic/rfdt_correction/core.py
analysis/dynamic/rfdt_correction/ddm.py
analysis/dynamic/rfdt_correction/engine.py
analysis/dynamic/rfdt_correction/io.py
analysis/dynamic/rfdt_correction/pipeline.py
analysis/dynamic/rfdt_correction/simulation.py
analysis/dynamic/rfdt_correction/validation.py
analysis/dynamic/tests/test_core.py
```

After copying, verify every SHA-256 value against `SOURCE_HASHES.md`.
