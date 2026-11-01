# Analysis precision

How precise format inference is, with and without digit-bound analysis, on
the MMA-Sim designs (`examples/mmasim/compile.py`'s `DESIGNS`).  Each design
is specialized per call site as the C++ compiler does, analyzed both ways, and
run on random inputs.  Every value each assignment takes is abstracted, as a
set, to the least format containing it.  That format is compared with the
static bound in three dimensions: significant digits, lowest digit, highest
digit.  The plan and its record: `docs/todos/analysis-precision.md`.

## Run

From the repository root (about 2 minutes with `-j 8`):

```sh
.venv/bin/python evaluation/analysis_precision/run.py -o evaluation/analysis_precision/results -j 8
.venv/bin/python evaluation/analysis_precision/report.py evaluation/analysis_precision/results
```

`run.py` writes `<design>.json` per design and exits 1 if any observed value
lies outside its static bound.  `report.py` writes:

- `precision-tops.tex`: sites bounded only by ℝ* per architecture, without
  and with digit bounds;
- `precision-coverage.tex`: per architecture, how much of each bound was
  observed (precision p′/p, digit span, magnitude max|v|/b);
- `precision-slack.{pdf,png}`: each bounded site's slack with digit bounds.

It also prints summary numbers.  Given several result directories (other
seeds, `--inputs adversarial`), it combines their observations per site.
Options: `run.py -h`, `report.py -h`.

## Tests

```sh
.venv/bin/python -m pytest -n 8 -q evaluation/analysis_precision/tests
```
