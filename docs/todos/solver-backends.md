# Digit-bound solver backends

Measure how much time the digit-bound store's solver
(`fpy2/analysis/digit_bound/`) takes, and compare z3 with cvc5 (SMT) and HiGHS
through `highspy` (MILP), by compiling the MMA-Sim designs
(`examples/mmasim/compile.py`'s `DESIGNS`) under each.  This is a research
comparison: z3 stays the default.

## 1. Starting point

- **The backend is already separate.**  `Solver` (`solver.py`) is a Protocol
  with two methods, `assume(constraint)` and
  `maximize(objective, cutoff, assuming)`.  `Z3Solver` is the only
  implementation.
- **One creation site.**  `DigitBoundInfer.analyze` (`infer.py:2741`) is the
  only place a store is created, as `DigitBoundStore()`, which defaulted to
  `Z3Solver()`.  Callees continue their caller's store through
  `DigitBoundParams`.  A hook on that default sees every solver a
  compilation makes.
- **Full compilation always uses digit bounds.**  `compile_design` builds the
  C++ with `CppCompiler`, which runs format inference with
  `use_digit_bounds=True` (`backend/cpp/compiler.py:628`).  Compiling a
  design is therefore the end-to-end workload.
- **Some queries never reach the solver.**  `DigitBoundStore` caches answers
  and settles terms naming a free variable itself.  Only queries reaching
  the solver are timed as solver time.

## 2. Instrumentation

Always on.  Integer increments and `time.perf_counter()` calls, which cost
little next to a solver check.  The largest design makes about 48,000
`assume` calls and 2,400 queries, two clock reads each: milliseconds
against an 18 s compile, under 0.1%.  A parallel run of every design with
and without the counters differed by less than its own noise.

A `SolverStats` dataclass in `solver.py`, and a `stats` attribute on every
backend, updated where the work happens:

| Counter | What it is |
|---|---|
| `solver_time` | seconds in the backend: inside `assume`, which encodes into live solvers, and inside `maximize`, the queries |
| `constraints` | constraints assumed |
| `queries` | `maximize` calls, each one query of one or more objectives |
| `calls` | backend calls (a satisfiability check, or a HiGHS solve) |
| `degraded` | queries with an answer loosened because the backend could not decide: a timeout or unknown, or (HiGHS) a box too wide to solve |

Every solver also keeps a `trace`: a `QueryRecord` per query, of the
constraints in the component it was asked of, its backend calls, its
time, and how many objectives it asked.  It is always kept, a tuple per query.

These are what the evaluation reads.  A first version also split the time
(`assume`, `maximize`, building solvers) and broke the counts down (calls by
result, constraints by relation, guarded constraints, variables, cutoff
queries); nothing read them, so they were dropped.

`assume` is timed too: z3 and cvc5 encode a constraint into a component's
live solvers as it arrives, so leaving it out would hide part of their work.

**Collecting.**  `fpy2/analysis/digit_bound/backends.py`.  A
`contextvars.ContextVar` holds a solver factory, read by
`DigitBoundStore()` when it is not handed a solver.  A helper,

```python
with solver_backend('cvc5') as solvers:
    compile_design(build)
total = sum((s.stats for s in solvers), SolverStats())
```

sets the factory for the duration and keeps every solver it made.  No public
signature changes.  An environment variable, `FPY_DIGIT_BOUND_SOLVER`, sets
the default, so the existing tests run under any backend unchanged.

## 3. Backends

### 3.1 Shared base

Move what does not depend on z3 from `Z3Solver` into a base class,
`_ComponentSolver`:

- components (variables grouped by shared constraints) and their merging;
- the ceiling past which the answer is `inf`;
- bisection;
- statistics.

A backend supplies four hooks:

- make a solver for a component (`_new`);
- encode a constraint into it (`_encode`);
- check it under a set of literals and an extra bound on the objective
  (`_check`);
- read the objective's value in the model (`_value`).

Bisection over those hooks is the default way to find a maximum
(`_maximum`), one objective of a query at a time (`_maxima`).  A backend that
optimizes natively, as HiGHS does, overrides `_maximum`; z3 with
`bisect=False` overrides `_maxima`, to ask every objective in one check
(§3.5).

Each backend is a module of its own: `z3_solver.py`, `cvc5_solver.py`,
`highs_solver.py`, over `solver.py`'s interface and base.

A registry, `make_solver('z3' | 'cvc5' | 'highs', **opts)`.  The cvc5 and
HiGHS backends import their dependency lazily.  The dependencies were first
an optional extra of FPy's, `solvers`; they are now only in the
evaluation's environment (`evaluation/pyproject.toml`), with its other
dependencies.

Moving `Z3Solver` onto the base must leave its answers unchanged, so the
C++ for every design is identical before and after.  It is, for all 62
designs (60 compile; `amd.cdna1.bf16` and `amd.cdna1.f16` fail storage
selection before the change as after).

### 3.2 cvc5

A near copy of the z3 encoding.

- The base API, in logic `QF_LIA`.  `cvc5.pythonic`, which mirrors z3's
  Python API, was the first choice, but it rejects a one-argument `And`,
  which a one-literal guard is.  The base API wants at least two arguments
  for every n-ary kind too, so a one-argument `ADD`, `OR` or `AND` is its
  argument.  `mkInteger` takes a C `int`, so constants go in as strings.
- Incremental mode on, guard literals passed as assumptions
  (`checkSatAssuming`), and the timeout set with `tlimit-per`.
- cvc5 does not optimize, so it uses bisection only, as z3 does by default.
- An unknown answers `inf`, as for z3.

### 3.3 HiGHS: `highspy`

One `highspy.Highs` model per component, kept across queries.  Output off,
one thread, `mip_rel_gap = 0` (the default `1e-4` lets a large objective
stop short of its optimum), `time_limit` set, presolve on.

**Encoding.**

- **Integer variables:** one column per store variable, integral, with
  bounds `[-B, B]`.
- **0/1 columns:** one per guard literal `g`, `b_g`.
- **`LE` and `EQ`:** one row each, unchanged.
- **Guards:** a guarded constraint `AND(g) -> lhs <= rhs` is the row
  `lhs - rhs <= M * (|g| - sum(b_g))`.
- **`LE_MAX`:** `lhs <= max(r_1..r_n)` becomes a 0/1 selector column `d_i`
  per term, the row `sum(d_i) >= 1`, and the rows
  `lhs - r_i <= M * (1 - d_i)`.  A guard relaxes `sum(d_i) >= 1` by
  `|g| - sum(b_g)`.
- **M:** chosen per row as the largest value its left side can take over the
  column bounds.  A tight per-row M is numerically safer than one global M.
- **B:** the next power of two past `4 * (span + 1) * scale`, from the
  *component's own* constants and coefficients, and past twice the ceiling
  of the objective asked.  The store-wide `_span` the bisecting backends use
  does not work here: one MMA-Sim component bounds an integer by `2^50`,
  which put every component's box past the integers a double holds exactly.
- **Rows are added lazily.**  A constraint is kept with its component's
  model and becomes rows at the next query.  If the box has to grow, the
  model is rebuilt from its constraints with the new box, since the rows'
  M depend on it.  A box past `2^40` is not solved: the answer is `inf`,
  degraded.

**Incremental use.**

- `assume` adds a constraint to the component's live model, as z3's solvers
  learn incrementally; it becomes rows at the next query.
- Assumed literals are the bounds of `b_g`, set to `[1, 1]` for the query and
  reset after.  Nothing is added as a row.
- Merging two components states the smaller one's constraints into the
  larger one's model, as for z3.
- HiGHS keeps the model between solves, but each MIP solve searches from
  scratch.

**Queries.**

- **Maximum:** set the objective's costs, maximize, one solve per
  objective of a query.  A MIP has one objective, and HiGHS's
  multi-objective mode is lexicographic, which is not each objective's own
  maximum (§3.5).
- HiGHS discards its status and solution when a bound changes, so a solve's
  result is read before the assumed literals are released.

**Results.**

| Model status | Answer |
|---|---|
| optimal | `floor(mip_dual_bound + tol)` |
| time limit | the same, counted as degraded: the dual bound is still an upper bound, so a timeout loses precision rather than answering `inf` |
| infeasible | `-inf` |
| unbounded, unbounded-or-infeasible, anything else | `inf` |

The answer comes from the dual bound, not the objective value.  The value is
attained, so it bounds the maximum from below, not above.

An answer past the ceiling reads as `inf`, as the bisecting backends' upward
probe stops at it.

The first plan also answered `inf` where a solution put a variable at
`±B`.  That fires on ordinary optima: in `g >= l - 10`, maximizing `l - g`,
nothing bounds `g` from below except the box, so HiGHS often parks it there.
It loosened four unit tests' answers to `inf`, and was dropped.

**Soundness risks.**  This backend is for comparison, not a sound default.

- **Bounds.**  The bounds `[-B, B]` remove solutions.  If `B` is too small, a
  maximum can come back below the true one, or a feasible system can come
  back infeasible and answer `-inf`.  Where every coefficient is a unit, as
  nearly all the store's are, each disjunct is a system of difference
  constraints, every face of which has a point within the total of its
  constants of the origin, and `B` is four times past that.  A coefficient
  beyond a unit can need more than `B`'s one factor of `scale`.  §4's C++
  comparison shows whether this changed any bound.
- **Floating point.**  HiGHS works in floating point.  `tol` and the
  integrality tolerance decide the rounding.

### 3.4 Later

- z3 with `Optimize` is in, as `z3-opt` (`Z3Solver(bisect=False)`).  Other
  z3 configurations (`SolverFor('QF_LIA')`, components off) are one registry
  entry each.
- No `cvc5-opt`: cvc5 has no optimizer (1.4.2: no maximize in its API, no
  optimization option), and a search of our own around it would be a third
  strategy rather than cvc5's.  Decided against.
- OR-Tools CP-SAT: exact integer arithmetic, and guards and `LE_MAX` without
  big-M (`OnlyEnforceIf`, `AddBoolOr`).

### 3.5 Queries of several objectives

`maximize` takes a list of objectives under one set of assumed literals, and
returns each one's maximum.  One call is one query.  The maxima are
independent: each may be attained by a different solution.  A format needs
exactly that: its magnitude is the largest msb, its grid the smallest lsb,
and its precision the largest `msb - lsb + 1`, which no one solution need
reach together.

- **The store** asks a format's three bounds as one query
  (`DigitBoundStore.maximums`, from `DigitBoundInfer.bounds`).  A term
  already answered, or naming a free variable, is settled without the
  solver; the rest go together.  A lone `maximum` is a query of one.
- **z3 with `Optimize`** answers a query in one check, with
  `opt.priority` set to `box`: each objective's own optimum.  The default,
  `lex`, optimizes each objective given the ones before it, which is not a
  bound on the later ones.  `pareto` returns one front point per check; only
  the whole front's box would bound them.
- **Bisecting backends and HiGHS** answer the objectives one at a time, as
  before.  A query is still one record: its calls are every check or solve
  it took.

The cutoff went with the change.  `maximize(objective, cutoff)` let a
backend settle "can `t` reach `k`?" in one check, for `DigitBoundStore.reaches`;
its one caller in `infer.py` was removed in #337, so the store's `reaches`,
its cache, and the cutoff path were dropped.  Bisection's own probe
(`_reaches`) stays.

## 4. Correctness

- **Store tests under every backend.**  The store-level tests in
  `tests/unit/analysis/test_digit_bound.py` take a backend parameter.  A test
  skips a backend that is not installed: cvc5 and HiGHS run from
  `evaluation/`'s environment.
- **Soundness oracle.**  `tests/unit/analysis/test_digit_bound_oracle.py` runs
  under each backend through `FPY_DIGIT_BOUND_SOLVER`.
- **Same C++.**  The evaluation compares each design's generated C++ across
  backends.  Different code means the backends inferred different bounds
  somewhere.  That is reported, not fatal: it may be a degraded answer
  rather than a bug.

## 5. Evaluation: `evaluation/sensitivity/`

One of `evaluation/run.py`'s evaluations, `sensitivity`, laid out as
`evaluation/analysis/`: `run.py`, `report.py` and a README, its defaults
`config.py`'s `SENSITIVITY_*`, and its results `evaluation/results/sensitivity/`,
replaced by each run.  It began apart, as `evaluation/solver_backends/`, a new
folder per run; it moved when the evaluations were unified (#348).

### 5.1 Running

For each design and each backend, compile with `compile_design` under
`solver_backend(name)` and record:

- total compile time;
- §2's counters, summed over every solver the compilation made;
- a hash of the generated C++.

Each compilation runs in a fresh process (`run.py --child`, through
`subprocess` so the limit can kill it), on one thread, under a wall-clock
limit.  A run that hits the limit is recorded as a timeout, not dropped.
Each backend call is limited to `-c` seconds (default 60, against the
solvers' own 10): at 10 s, the CDNA3 FP8 designs had calls cross the limit
in some repeats and not others, and a degraded answer sends the analysis
down another path (`amd.cdna3.fp8.bf8` under HiGHS stated 61k constraints
in one repeat, 49k in the others).
Each is repeated `N` times (default 3), and the median run by compile time
is reported, its time and its counts together; the summary names designs
whose counts differ between repeats.  `-j`
compilations run at once (default 4).

### 5.2 Outputs

Besides each design's record (`<design>.json`) and the run's settings
(`sensitivity-run.json`), the report writes the table and the figure beside
them, and prints a summary:

- `sensitivity-totals.tex`: one row per backend, over the designs every backend
  compiled: its solver method, compile and solver time, the solver's share,
  and solver time per query.  Constraints and queries are left out: they
  are the same under every backend, and the summary prints them.
- `sensitivity-scatter.{pdf,png}`: each query's time against the constraints in
  its component, over every design's median run, a panel per backend,
  log-log.  It was a point per design (total solver time against total
  constraints), but the designs span only 400 to 49k constraints, most of
  them 10k-30k, and a backend never sees a design's total, only the
  component a query names.  A linear-axes copy was dropped: the log-log
  one is the one the report uses.
- `sensitivity-paper.json` and `sensitivity-summary.txt`: every number the
  evaluation cites, as `analysis` keeps `precision-paper.json`; the JSON is
  the evaluation's mark of completion for `evaluation/run.py`.

## 6. Status

All six steps are done.  The backends are in `fpy2/analysis/digit_bound/`
(`solver.py`, `backends.py`, `z3_solver.py`, `cvc5_solver.py`,
`highs_solver.py`), their
tests in `tests/unit/analysis/test_digit_bound_backends.py`, and the
evaluation in `evaluation/sensitivity/`.  The digit-bound suites
(`test_digit_bound.py`, `test_digit_bound_oracle.py`) pass under every
backend through `FPY_DIGIT_BOUND_SOLVER`.

**Open: HiGHS is wrong on wide boxes (§3.3, "Floating point").**  Found in
review, not yet fixed; the HiGHS numbers are not to be cited until it is.
On random stores of 8 variables with `LE_MAX` rows and guards, whose
constants put the box near `2^29`, HiGHS against z3:

- **Loose.**  The incumbent breaks `LE_MAX` rows by 2 to 13 once rounded:
  a selector a hair below 1 releases `lhs - r_i <= M * (1 - d_i)` by
  `M * eps`.  Answers came back loose by up to 2e7.
- **Below the maximum: unsound.**  One query came back `kOptimal` at
  30000032 with an exactly feasible incumbent, where z3 attains 30000035.
  The dual bound itself is wrong, so checking the incumbent after the solve
  cannot catch it.
- **Tolerances alone do not fix it.**  `mip_feasibility_tolerance` and
  `primal_feasibility_tolerance` at `1e-9` keep one hand-built `max` exact
  to a box of `2^31` (default: `kSolveError` from `2^30`), `1e-10` to
  `2^35`, but random stores still differed at `1e-9`.

Next: tally the disagreements by box size, under the default tolerances and
under `1e-9` for primal, dual and MIP feasibility, to find the box below
which HiGHS agrees with z3; then tighten the tolerances and lower
`_MAX_BOX` to it (past it the answer is `inf`, degraded and counted), and
add the fuzz as a test.  Check what boxes the MMA-Sim components need: the
review estimated `2^22` to `2^25` for the large CDNA3 ones.

## 7. Order

1. The counters and the factory hook (§2), and the overhead check.
2. A z3-only run on MMA-Sim: how much of compile time the solver takes.
3. The shared base and the registry (§3.1), with the C++ unchanged.
4. cvc5 (§3.2) and the tests under it (§4).
5. HiGHS (§3.3), the tests under it, and the counts of degraded and
   infeasible answers.
6. The evaluation and its report (§5).
