# Sensitivity to the solver backend

How much of compilation the digit-bound solver takes, and how z3 compares
with its own optimizer and with cvc5 as its backend, on the
MMA-Sim designs (`evaluation/common/mma/compile.py`'s `DESIGNS`).  Each design
is compiled to C++ as `compilation/compile_times.py` times it (the C++
alone, without `compile_design`'s entry signature, which would run
digit-bound inference a second time), once per backend and repeat.

The interface and the base every backend shares are
`fpy2/analysis/digit_bound/solver.py`; the z3 backend is `z3_solver.py`
beside it, and the cvc5 backend `cvc5_solver.py` here.  `backends.py` names
them:

- `z3`: bisection over satisfiability checks, the default;
- `z3-opt`: z3's `Optimize`, one search per query, all its objectives
  together;
- `cvc5`: bisection.

A query is one `maximize` call: a format's three bounds are one query of
three objectives.

## Run

From `evaluation/`, as part of the whole evaluation (`uv run run.py
results`, `--disable sensitivity` to skip it) or alone (about 25 minutes at `-j 4`):

```sh
uv run sensitivity/run.py
uv run sensitivity/run.py --smoke   # the first 2 designs, once each
```

Its defaults (backends, repeats, time limits) are `config.py`'s
`SENSITIVITY_*`; `run.py -h` lists the options.  Every compilation runs in
a fresh process, killed past `-t` seconds, and each backend call in it may
take `-c` seconds (default 60, over the solvers' own 10: a large CDNA3
design has calls near 10 s, and one that crosses its limit answers loosely
and sends the analysis down another path).  `-j` runs that many at once
(default 1, the least noisy times); `-j 4` trades some noise for a quarter of
the wall time.

It replaces `evaluation/results/sensitivity/` (or `-o DIR`; a directory
holding files a run did not write is refused) with:

- `<design>.json`: each backend's runs (wall time, solver statistics, C++
  hash, or the error or timeout), and the median run by compile time, with
  its queries (`trace`: each one's component size and time).  A backend
  compiled a design only where every repeat did;
- `sensitivity-run.json`: the run's settings;
- `sensitivity-table.tex`: one row per backend, over the designs every
  backend compiled: its method, compile and solver time, the solver's
  share, and solver time per query;
- `sensitivity-macros.tex`: the numbers the text cites, as macros: each
  backend's solver time and share (`\zthreebintime`, `\zthreebinshare`,
  `\zthreeopttime`, `\zthreeoptshare`, `\cvctime`, `\cvcshare`), and the
  others' solver time over z3's (`\solverslowdown`);
- `sensitivity-scatter.{pdf,png}`: each query's time against the
  constraints in its component (all a backend sees), a panel per backend,
  log-log;
- `sensitivity-paper.json` and `sensitivity-summary.txt`: every number the
  evaluation cites, with the designs whose C++ differs from z3's and those
  a backend failed.

Solver time is the time inside the backend: stating constraints and
answering queries.  Queries the store answers from its cache, or settles
because a variable is free, never reach a backend.  The run exits 1 if a
backend fails a design z3 compiles; differing C++ is reported, not fatal.
`report.py RESULTS` redraws the report alone (`run.py results --plot` in
`evaluation/`).

Any test suite runs under another backend with the pytest plugin
`solver_plugin.py`, from `evaluation/`:

```sh
PYTHONPATH=sensitivity uv run pytest -p solver_plugin --digit-bound-solver cvc5 -n 8 -q \
  ../tests/unit/analysis/test_digit_bound.py
```

## Tests

From `evaluation/`, apart from `evaluation/analysis/tests` (both name their
modules `run` and `report`):

```sh
uv run pytest -n 8 -q sensitivity/tests
```
