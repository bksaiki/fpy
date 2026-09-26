# Digit-bound inference: guard the `logb` floor

Implementation plan.  The design is settled; what follows is the phase
breakdown, one phase per commit.  Background is in
[digit-bound-inference.md](digit-bound-inference.md#the-floor-under-logb-is-stated-too-widely).

## Working policy

- **Pause after each phase for review.** Do not begin the next phase until the
  current one has been looked at.
- **Do not commit.** The author of the change leaves the working tree dirty;
  commits are made by the repository owner.
- **Run only the tests relevant to the phase.** The full unit suite runs once,
  at the end, after the last phase.
- **Comments stay succinct**, and notes about *process* -- what was tried, what
  a phase decided, why an ordering was chosen -- belong in this document, not in
  source comments.

## Context

`value_of(Logb)` in `fpy2/analysis/digit_bound/infer.py` returns `msb(x)` as
the value of `logb(x)`, and states `msb(x) >= lo`, `lo` the least exponent of
`x`'s format.  That floor is what anchors a position like `logb(x) - k`
absolutely: without it the rounding's grid has no least digit, and its format
comes back `RealFormat`.  It is stated on every path, though it holds only
where `x != 0`.

### It costs ten designs

At `15ad8c3f`, `examples/mmasim/compile.py` compiles 50/62.  Two of the
refusals, `amd.cdna1.*`, need 280 bits of exact sum and are out of scope.  The
other ten are every FP16-accumulator design at `e_zero = -133`
(`nv.hopper.f16.f16.wgmma`, `nv.hopper.*.f16`, `nv.blackwell.*.f16.tcgen05`).
T-FDPA writes

```python
e_c = e_zero if c == 0 else exponent(c, emin_c)    # exponent: max(logb(x), emin)
```

and `_merge_arms` ties the merge back to `c` with the disjunct `msb(c) - 1`.
The floor puts that disjunct above `e_zero`, `_usable` rejects it, and `c`
loses its anchor: the fused sum needs 179 bits.  A design compiles exactly when
`e_zero >= expmin_c - 1`.

Measured at `15ad8c3f`, and on a copy of it with the floor deleted:

| | HEAD | no floor |
|---|---|---|
| `compile.py` (C++) | 50/62 | 60/62 |
| `compile_triton.py` | 50/62 | 60/62 |
| `tests/unit/analysis` | pass | 30 fail |
| `tests.infra.backend.cpp --mode run` | 130/136 bit-compared | identical |

Deleting the floor fixes the ten, regresses none of the fifty or of the C++
corpus (the same 138 functions compile, with the same output), and uncovers no
second blocker.  The 30 failures are `TestSelfAnchoredRounding` (12),
`TestRoundCarry` (9), `TestBoundsThroughAMax` (4), `TestAnchorsAcrossACall`
(2), `TestAZeroGuardDischargesABranch` (2) and `TestAZeroGuardNoPathNames`
(1).  In every one, the rounding loses its absolute anchor.

### It is also unsound

```python
@fp.fpy(ctx=fp.REAL)
def f(x, y):                        # x: FP16, y: FP32
    e = max(fp.logb(x), -1000)
    with fp.MPFixedContext(e - 10, fp.RM.RTZ):
        return fp.round(y)
```

The round's bound is `MPBFloatFormat(pmax=24, emin=-10, ...)`, which does not
contain `f(0, 2 ** -149) = 2 ** -149`.  At `x = 0`, `e` is `-1000`, but the
floor says `e >= -24`.

## The floor is a precondition

`logb(0)` is `-inf`, so `x != 0` may be assumed exactly where `-inf` would make
the program undefined:

- **A rounding at a non-finite position is undefined.**  The interpreter
  raises (`expected an integer argument for nmin=Float('-inf')`), and this is
  the precondition the current comment on `value_of(Logb)` already names.
- **A `max` with a finite operand is not.**  `exponent(0, -14)` is `-14`.
  T-FDPA's `e_c` is an `IfExpr`, whose arms a backend may evaluate on every
  input (`ValueClassInfer` does not refine them), so `exponent(c, -14)` does
  run at `c = 0` and is well defined there.

So the floor holds where a rounding's position is affine in `logb(x)`, and on
the arm of a zero test that excludes `x = 0`.  Nowhere else.

## Guard literals already exist

`DigitBoundStore` takes a guard on every constraint and a set of literals on
every query (`maximum(term, assuming)`).  The finiteness facts use them:
`_lit(d)` is "`d` is finite", `_assumed(e)` collects the literals that hold at
`e`, and a callee inherits its caller's through `arg_lit` and `outer`.  The
path-sensitive store the background document calls for is this machinery with
a second kind of literal.

## Design

### The principle

`msb(x)` is both an upper bound on `x`'s exponent and, in the value channel,
the exact `logb(x)`.  A zero has no exponent: its `logb` is `-inf`, which the
integer store stands in for with an arbitrarily low value.  An execution is a
model of the store only if every lower bound on an `msb` is conditional on its
value being non-zero.  Everything below follows from that:

1. the floor is stated under `nz(x)`;
2. a query assumes `nz(x)` where `x != 0` holds on every defined execution
   reaching it;
3. a merge joins its edges' bounds, each computed under that edge's facts;
4. `_merge_arms`' vacuous disjunct, which reads a zero's `msb` as `-inf`, is
   exactly true rather than in tension with the floor, so the checks layered
   on to catch that tension (`_usable`, `_check_vacuous`) go.

### A non-zero literal per `logb` term

`value_of(Logb)` states `msb(x) >= lo` under a literal `nz(x)`, one per `msb`
term, instead of unconditionally.  The map from term to literal is shared with
callee instances, as the store is.  A guarded constraint only narrows where it
is assumed, so the floor is sound wherever it is not.

### Assumed at a strict rounding

`_rounding(e)` already returns a `Round`/`Cast`'s position as a term.  Where
that term names `msb(x)` with a non-zero coefficient, the position is affine in
`logb(x)`, hence non-finite at `x = 0`, hence undefined there.  So the
rounding's queries assume `nz(x)`.

`max`, `min` and `IfExpr` mint fresh terms, so a position built through
`exponent(...)` never qualifies.  A callee's parameter carries the caller's
term, so `round_at(x, logb(x) - 12)` does.  This covers the 27 tests in the
first four classes above.

### Projected at a merge

`_merge_arms` also states `m >= min(f_then, f_else)`, each `f` being that
arm's least value under that arm's facts.  The `else` arm of a test in which
every path zeroes a single term `t` (a singleton in `_zero_paths`) assumes
`nz(t)`.  Both bounds are constants, so nothing about `msb(c)` leaks onto the
zero path.  This covers the three `TestAZeroGuard*` tests, among them
`test_the_sentinel_stands_when_another_value_is_rounded`, which is the
`65535.999999940395` case.

### What goes away

With no unconditional floor, `_floor(t)` of a `logb`'s term is `-inf`, so
`_usable` always holds, and `_check_vacuous` and `_vacuous_used` have nothing
to check.  They go, once a measurement confirms it (Phase 4).  `_floor` stays,
gaining an `assuming` parameter, since the merge projection reads it.

### Not a literal-encoded store

The alternative considered was to encode zero-ness and branch edges as
literals joined by clauses: a merge as guarded per-edge equalities, and the
vacuous disjunct as "the `then` edge implies `msb(c) <= m + 1`".  z3 would then
do the zero reasoning that `_vacuous`, `_zero_paths`, `_forced_zero` and
`_universal_zeros` (about 190 lines) now do by syntax.  Rejected:

- **It is unsound over element summaries unless managed.**  A list's terms
  describe an arbitrary element, and `sum` takes its greatest `msb` and least
  `lsb` from possibly different elements.  A branch literal is one boolean per
  model, so it would tie every element term to one branch, and understate a
  sum whose elements took different branches.  The universal-literal and
  instance machinery exists to keep such couplings out.  The `nz` literals
  here are assumed only where they are facts, so they add none.
- **Its gains are reachable here.**  The relation between a merge and its edge
  is the guarded equality `nz(c) -> m = iff`, the pattern `_untaken` already
  uses for finiteness.  The `And`/`Not` holes are gaps in reading conditions.
- **It asks z3 harder questions**, where the performance work so far came
  from asking fewer.

The consolidation worth doing instead is one condition reader shared by
`ValueClassInfer._implied` and `_zero_paths`, the audit's "same concept in two
places".  That is independent of this plan.

## Phases

### Phase 1 - A differential check for the compiled designs

`compile.py` compiles and never runs.  Nothing checks the C++ of any mmasim
design against the interpreter, and `compile_triton.py -r` needs a GPU.

- Move `_fmt`, `_length`, `_hard_cases`, `_sample`, `_vector` and `_row` from
  `compile_triton.py` to `compile.py`.  The former already imports from the
  latter.
- Add `-r DRAWS` and `-s SEED` to `compile.py`.  For each design that compiles,
  emit a `main` that calls the entry on each draw and prints each result with
  `%a`, taking parameter types from `CppCompiler.signature(func,
  module=mod)`.  Build it with `c++`, then compare bit for bit with the
  interpreter: the same value and sign, or both NaN.  As in
  `compile_triton.py`, every fourth draw is heavy in hard cases, which covers
  `c = 0`, zero products and the least subnormal.
- The driver is local to `compile.py`.  An example does not import the private
  helpers in `tests/infra/backend/cpp.py`.

**Why first.** It is the net under the analysis change, and a baseline on the
fifty separates a harness bug from a miscompile.

**Tests.**

```sh
cd examples/mmasim
python compile.py -j 8 -r 256      # 50/62 compile, every one agrees
python compile_triton.py -j 8      # still 50/62 after the move
```

**Done.**  Where it diverged from the plan:

- The driver reads its draws from stdin and deduces each parameter's type
  from the entry's function pointer.  So it needs no `CppCompiler.signature`
  pass, and its build time does not grow with `DRAWS`.  `_HARD`,
  `_HARD_EVERY` and a shared `_same` moved too.
- **47 of the 50 agree on every draw, and 3 do not build.**
  `nv.volta.f16.f16`, `nv.turing.f16.f16` and `nv.hopper.f16.f16.mma` emit
  two identical definitions of one `exponent__...` specialization.  All
  three operands are FP16 there, so `exponent(a, -14)`, `exponent(b, -14)` and
  `exponent(c, -14)` share a mangled name.  This is a C++ backend bug that
  predates this plan; `compile.py` never invoked a compiler, so nothing saw it.
  `compile_triton.py -j 8` and `compile.py -j 8` are unchanged at 50/62.

### Phase 2 - Regression tests

In `tests/unit/analysis/test_format_infer.py`, both marked
`xfail(strict=True)` until Phase 3:

- The probe above, asserting `bound.representable_in(f(0, 2 ** -149))`.
- A T-FDPA shape in miniature, where `e_zero` is below `expmin_c - 1`: two
  terms whose exponents are merged as
  `-26 if v == 0 else max(logb(v), -14)`, both rounded at the merge less a
  fixed `F`, then summed.  It asserts the sum's precision against the value
  the no-floor copy gives (about `F`, not `pmax=68`).

**Why separate.** It pins both defects before the change, so Phase 3's diff is
the fix alone.

**Tests.**

```sh
python -m pytest tests/unit/analysis/test_format_infer.py -n 8
```

### Phase 3 - Guard the floor

In `infer.py`:

- The `nz` literal map, passed to `_DigitBoundInferInstance` alongside
  `arg_lit`.
- `value_of(Logb)` states the floor under `nz(x)`.
- `_assumed(e)` adds the strict literals of a `Round`/`Cast`, read off
  `_rounding(e)`'s position term.
- `_merge_arms` states the projected arm floors, with the `else` arm's facts
  taken from `_zero_paths` singletons.

**Why one phase.** Guarding the floor without the strict rule fails 27 tests,
and without the projection it fails 3.  The three pieces together are the
smallest change that keeps the suite green.

**Tests.** Remove Phase 2's `xfail` markers.

```sh
python -m pytest tests/unit/analysis -n 8
cd examples/mmasim
python compile.py -j 8 -r 256      # 60/62 compile, every one agrees
python compile_triton.py -j 8      # 60/62
```

Also time `compile.py -j 8` against HEAD's 38 s: more distinct assumption sets
mean more z3 calls.

### Phase 4 - Drop `_usable` and `_check_vacuous`

First confirm, with a temporary assert, that `_usable` never returns `False` on
`tests/unit/analysis` and the corpus.  Then delete `_usable`,
`_check_vacuous`, `_vacuous_used` and the `_check_vacuous()` call in
`analyze`, and rewrite the comment on `value_of(Logb)`.  If the assert fires,
keep them and record the case here.

**Why separate.** It is a deletion justified by a measurement, and reviewing it
apart from the fix keeps each diff about one thing.

**Tests.** The same as Phase 3.

### Phase 5 - Docs

In `digit-bound-inference.md`, replace "The floor under `logb` is stated too
widely" and "It now costs ten designs" with a short account of the guarded
floor.  The claim that the root "is not localisable in this domain" goes.
Only that page changes.

**Tests.** None.

### After the last phase

```sh
python -m pytest tests/unit -n 8
mypy fpy2 && ruff check fpy2
python -m tests.infra
python -m tests.infra.backend.cpp --mode run
cd examples/mmasim && pytest tests && python compile.py -j 8 -r 256 && python compile_triton.py -j 8
```

## Open items

### Does anything downstream of a strict rounding need the floor?

The strict rule assumes `nz(x)` at the rounding alone, so a later use of its
result has no absolute anchor where today it has one.  `RescaleFixed` runs
before both backends, and turns a strict position into a `2 ** k` scale around
a round at a concrete position.  So on the compile path the rule never fires,
and the scale-out product loses its floor either way.  Such a program could
start refusing.  Extending the assumption to every query the rounding
dominates costs about 20 lines: carry the set of established facts through the
walk, and restore it after `if` arms and loop bodies.

*Provisional:* defer.  With no floor at all, `tests.infra.backend.cpp --mode
run` is unchanged (see Context), so nothing in the corpus needs the floor
downstream.  Reopen if a program that compiles today refuses after Phase 3.

### Where does the duplicate-specialization fix go?

Phase 1 found that three designs do not build.  The cause is in
`fpy2/transform/specialize.py`: a spec's key is `(fdef, inst)`, and from the
second fixpoint round on `fdef` is the previous round's spec, not the source
function.  The mangled name digests only `inst`.  So two identical specs of
`exponent` reach one name.  The fix is to key specs on their source function,
carrying a map from each spec back to it across rounds.  A scratchpad trial
brought `compile.py -r 16` from 47 to 50 agreeing, with nothing else changing;
the unit suite was not run.

*Provisional:* its own change, before Phase 3, so all fifty designs are under
the differential check when the analysis changes.  It is a backend fix,
unrelated to digit bounds.

### Should `nz` on a list-element summary be universal?

A universal literal is replayed onto per-index instances such as slices and
ranges, so it is more precise.  But it claims something of every element,
while a strict rounding of `xs[i]` proves only that `xs[i]` is non-zero.  A
non-universal literal is sound and dropped on replay.

*Provisional:* non-universal, since the corpus compiles with no floor at all.
Reopen if a shape needs a per-element anchor.

### Validating the Triton kernels on a GPU

The two backends share the analysis, so Phase 1's check covers what this
change touches.  What remains is Triton's code generation for the new
storage: an `int32` term and a `double` sum, reduced across a tile's lanes.

*Provisional:* the CPU check is the gate.  `compile_triton.py -r` on a GPU is a
follow-up for the repository owner.

### When does `triton-mmasim-matmul.md`'s status change?

It will still say 50/62, with FP16 output at `e_zero = -133` out of scope.
But that page reports what has been *run*.

*Provisional:* unchanged by this plan.  Update it when the GPU run happens.
