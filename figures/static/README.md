# Static figures

`make_figure4_static.py` rebuilds the manuscript static benchmark figure from the compact CSVs under `results/static/`. It does not fit RFDT or read raw third-party data.

Run from the repository root:

```bash
python figures/static/make_figure4_static.py
```

Output:
- `figures/static/Figure4_static.pdf`
- `figures/static/Figure4_static.png`

Panel A uses the HAB22 participant-mixture benchmark.
Panel B uses the separate pooled HAB22 restrictions.
Panel C uses the frozen Choices13k external validation.
