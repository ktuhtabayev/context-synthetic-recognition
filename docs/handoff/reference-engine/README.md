# Reference engine (test oracle)

Small, dependency-free Python (openpyxl only for loading) written while building the Excel experiment. It reproduces every
value of the full workbook and, in template mode, the author's HAG and meta-algorithm template workbooks.

- `csmodel/core.py` — loading, scale unification, operators ρ / ρ_I / ρ_J, neighbours, permitted k, formula (5), (1)–(4), (6)
- `csmodel/hag.py` — HAG Steps 1–5 with the switches `centres = 'running' | 'final'`, `step4_passes = 2 | 1`
- `csmodel/meta.py` — meta-algorithm Steps 1–5 (B1/B2, refusal = 0)
- `csmodel/pipeline.py` — `fit(X, y, types, **params)`, `represent(model, x)` (new object, class-free), `predict`
- `csmodel/evaluate.py` — confusion counts, scores, AUC (scores rounded to 10 decimals), margins, k-NN baseline
- `golden_values.json` — outputs for Heart-Disease (10, 13, 2): `ref` (default settings), `loo`, `base`, `variants`, `tpl`

Usage (from this folder): `python -c "from csmodel import core, pipeline; X, y, t = core.load_dataset('<workbook>.xlsx'); m = pipeline.fit(X, y, t); print(m['tuplam'])"`

It is an executable specification for tests — the project should be a new, well-architected implementation.
