# Analysis precision

How precise format inference is, with and without digit-bound analysis, on
the MMA-Sim designs (`evaluation/common/mma/compile.py`'s `DESIGNS`).  Each design
is specialized per call site as the C++ compiler does, analyzed both ways, and
run on 5000 inputs.  Every value each assignment takes is abstracted, as a
set, to the least format A(p′, exp′, b′) containing it.  That format is
compared with the inferred A(p, exp, b): the tightness of each parameter is
the fraction of it observed values reach (p′/p, 2^exp/2^exp′, b′/b).  The plan
and its record: `docs/todos/analysis-precision.md`.

## Run

From the repository root, one command (about 8 minutes):

```sh
.venv/bin/python evaluation/analysis_precision/run.py
```

It replaces `evaluation/analysis_precision/results/` with:

- `<design>.json` per design: each site's inferred and observed parameters,
  under format inference alone and with digit bounds, over all inputs and
  over the random inputs alone;
- `anf/<design>.json`: the same sites with every scalar subexpression named
  (analysis only, for site counts);
- `precision-tops.tex`: sites bounded only by ℝ*, per architecture, without
  and with digit bounds;
- `precision-tightness.tex`: each parameter's mean tightness, per
  architecture;
- `precision-tops-vendor.tex`, `precision-tightness-vendor.tex`: the same,
  per vendor;
- `precision-slack.{pdf,png}`: each bounded site's slack with digit bounds;
- `precision-paper.json`: every table and number the evaluation cites
  (tightness as fractions in [0, 1]);
- `precision-summary.txt`: the same, as text.

Inputs are `mixed`: the random source of `compile.py -r`, with every fifth
input adversarial (every operand at one extreme of its format at once).  It
exits 1 if any observed value lies outside its inferred format, or any
design fails.  Options: `run.py -h` (designs by substring, `-n`, `-s`,
`--inputs random|adversarial|mixed`, `-j`, `-o`).  `report.py RESULTS`
regenerates the tables and summary alone.

## Tests

```sh
.venv/bin/python -m pytest -n 8 -q evaluation/analysis_precision/tests
```
