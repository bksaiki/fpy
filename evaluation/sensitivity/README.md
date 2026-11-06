# Sensitivity to the solver backend

How much of compilation the digit-bound solver takes, and how z3 compares
with its own optimizer, cvc5 (SMT) and HiGHS (MILP) as its backend, on the
MMA-Sim designs (`evaluation/common/mma/compile.py`'s `DESIGNS`).  Each design is compiled to C++
with `compile_design`, which always runs digit-bound inference, once per
backend and repeat.  The plan and its record: `docs/todos/solver-backends.md`.

The backends are in `fpy2/analysis/digit_bound/`: `solver.py` (the
interface, and the base every backend shares), `z3_solver.py`,
`cvc5_solver.py` and `highs_solver.py`.  cvc5 and HiGHS are only in the
evaluation's environment (`../pyproject.toml`); FPy itself depends on
neither.

## Run

As part of the whole evaluation, `uv run run.py results` from `evaluation/`
(`--disable sensitivity` to skip it), or alone, also from `evaluation/`
(about 25 minutes):

```sh
uv run sensitivity/run.py
```

Its defaults (backends, repeats, time limits) are `config.py`'s
`SENSITIVITY_*`.  It replaces `evaluation/results/sensitivity/` (or `-o
DIR`; a directory holding files a run did not write is refused) with:

- `<design>.json` per design: each backend's runs (wall time, solver
  statistics summed over every store, C++ hash, or the error or timeout),
  and the median run by compile time, whose time, counts and queries are
  reported together.  Each query (`trace`) is the constraints in the
  component it was asked of, its backend calls, its time, and how many
  objectives it asked.  A backend compiled a design only where every
  repeat did; otherwise its outcome names how the rest failed (`error`,
  `timeout`, or both), and how many compiled, and the design is left out
  of the totals;
- `sensitivity-run.json`: the run's settings, and whether each backend
  bisects (`bisects`, read from the backend), which the table's method
  column shows;

and `report.py`'s table and figure beside them:

- `sensitivity-totals.tex`: one row per backend, over the designs every backend
  compiled: how it finds a maximum (binary search or optimize), compile
  and solver time, the solver's share, and solver time per query.
  Constraints, queries and backend calls are left out: the summary prints
  them;
- `sensitivity-scatter.{pdf,png}`: each query's time against the
  constraints in the component it was asked of, over every design's median
  run, a panel per backend in one row, on shared log-log axes.  The component is all a
  backend sees, so it is the size that matters to the solver, not the
  design's total;
- `sensitivity-paper.json`: every number the evaluation cites: each
  backend's totals, per query (time, backend calls) and against z3, and
  the designs whose C++ or counts differ;
- `sensitivity-summary.txt`: the same, as text, also printed.

The two JSON files are the run's data; the table and the figures are all the
report writes.  It prints a summary, with the designs whose C++ differs from
z3's, and those whose counts differ between one backend's repeats: there a
call hit its limit in some repeats and not others, so the repeats did not do
the same work, and `-c` is worth raising.

The backends: `z3` (bisection, the default), `z3-opt` (z3's `Optimize`, one
search per query, all its objectives together), `cvc5` (bisection) and
`highs` (one MIP solve per objective).  A query is one `maximize` call: a
format's three bounds are one query of three objectives.

Every compilation runs in a fresh process, killed past `-t` seconds, and
each backend call in it may take `-c` seconds (default 60).  The solvers'
own limit, 10 s, is lifted for the evaluation: a large CDNA3 design has calls
near it, and a call that crosses it answers loosely and sends the analysis
down another path, so the backends would no longer do the same work.  Solver
time is the time inside the backend: stating constraints (`assume`) and
answering queries (`maximize`).  Queries the store answers from its cache, or
settles because a variable is free, never reach a backend.  The table's
totals cover only designs every backend compiled.

It exits 1 if a backend fails a design z3 compiles.  Different C++ is
reported, not fatal: a backend that gives up on a query (`degraded`) loosens
a bound, which can change the code.  HiGHS is not a sound backend in general
(`highs_solver.py`'s docstring), so differing C++ under it is worth reading.

Options: `run.py -h` (designs by substring, `-b` backends, `-r` repeats,
`-j` parallel compilations, `-t` and `-c` time limits, `-o`).  `-j 1` gives the least
noisy times; the default, 4, trades some noise for a quarter of the wall
time.  `report.py RESULTS` redraws the table, the figure and the summary
alone (`run.py results --plot` in `evaluation/`).

Any test suite runs under another backend with `FPY_DIGIT_BOUND_SOLVER`,
from `evaluation/`, whose environment has every backend:

```sh
FPY_DIGIT_BOUND_SOLVER=cvc5 uv run pytest -n 8 -q ../tests/unit/analysis/test_digit_bound.py
```

## Tests

From `evaluation/`:

```sh
uv run pytest -n 8 -q sensitivity/tests
```

Run apart from `evaluation/analysis/tests`: both name their modules `run`
and `report`.
