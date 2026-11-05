# Solver backends

How much of compilation the digit-bound solver takes, and how z3 compares
with cvc5 (SMT) and HiGHS (MILP) as its backend, on the MMA-Sim designs
(`examples/mmasim/compile.py`'s `DESIGNS`).  Each design is compiled to C++
with `compile_design`, which always runs digit-bound inference, once per
backend and repeat.  The plan and its record: `docs/todos/solver-backends.md`.

The backends are in `fpy2/analysis/digit_bound/`: `solver.py` (z3, and the
base the others share), `cvc5_solver.py` and `highs_solver.py`.  The two
others are the optional `solvers` extra:

```sh
uv pip install -e '.[solvers]'
```

## Run

From the repository root:

```sh
.venv/bin/python evaluation/solver_backends/run.py
```

It replaces `evaluation/solver_backends/results/` with:

- `<design>.json` per design: each backend's runs (wall time, solver
  statistics summed over every store, C++ hash, or the error or timeout),
  and their medians;
- `backends-run.json`: the run's settings;
- `backends-totals.tex`: one row per backend, over the designs every backend
  compiled: compile and solver time, the solver's share, backend calls,
  and calls and solver time per query.  Constraints and queries are left
  out: they are the same under every backend, and the summary prints them;
- `backends-scatter.{pdf,png}`: each design's solver time against its number of
  constraints, one point per design and backend.  Designs every backend
  answers in under 10 ms of solver time (the two FMA chains) are left out:
  their time is noise, and they only stretch the axes;
- `backends-scatter-all.{pdf,png}`: the same, with every design.

Both figures are log-log; `backends-scatter-linear.{pdf,png}` and
`backends-scatter-all-linear.{pdf,png}` are the same on linear axes.

The two JSON files are the run's data; the table and the figures are all the
report writes.  It prints a summary, with the designs whose C++ differs from
z3's.

The backends: `z3` (bisection, the default), `z3-opt` (z3's `Optimize`, one
search per query), `cvc5` (bisection) and `highs` (one MIP solve per query).

Every compilation runs in a fresh process, killed past `-t` seconds.  Solver
time is the time inside the backend: stating constraints (`assume`) and
answering queries (`maximize`).  Queries the store answers from its cache, or
settles because a variable is free, never reach a backend.  The table's
totals cover only designs every backend compiled.

It exits 1 if a backend fails a design z3 compiles.  Different C++ is
reported, not fatal: a backend that gives up on a query (`degraded`) loosens
a bound, which can change the code.  HiGHS is not a sound backend in general
(`highs_solver.py`'s docstring), so differing C++ under it is worth reading.

Options: `run.py -h` (designs by substring, `-b` backends, `-r` repeats,
`-j` parallel compilations, `-t` timeout, `-o`).  `-j 1` gives the least
noisy times; the default, 4, trades some noise for a quarter of the wall
time.  `report.py RESULTS` regenerates the tables and summary alone.

Any test suite runs under another backend with `FPY_DIGIT_BOUND_SOLVER`:

```sh
FPY_DIGIT_BOUND_SOLVER=cvc5 .venv/bin/python -m pytest -n 8 -q tests/unit/analysis/test_digit_bound.py
```

## Tests

```sh
.venv/bin/python -m pytest -n 8 -q evaluation/solver_backends/tests
```

Run apart from `evaluation/analysis_precision/tests`: both name their modules
`run` and `report`.
