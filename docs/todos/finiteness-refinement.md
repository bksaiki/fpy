# Finiteness refinement: one fact, two spellings

Implementation plan.  The diagnosis is settled and measured; what follows is the
phase breakdown, one phase per commit.

This page began as a reading of the code `digit-bound` emits for `fused_sum` --
a float where an integer belongs, a scale inside the loop that belongs outside,
and an assertion that cannot fail.  The diagnosis it first recorded, that
*nothing in the pipeline can conclude that a value is finite*, is wrong: the
pipeline concludes it whenever the guard arrives as a **fold**, and loses it
when the same guard arrives as a **materialised mask**.  Nothing below needs
`digit-bound`; every number on this page was measured on `todo-finiteness` at
`3fb7939c`.

## Working policy

- **Pause after each phase for review.**  Do not begin the next phase until the
  current one has been looked at.
- **Do not commit.**  Leave the working tree dirty; commits are the repository
  owner's.
- **Run only the tests relevant to the phase.**  The full suites run once, after
  the last phase.
- **Comments stay succinct**, and process notes -- what was tried, what a phase
  decided, why an ordering was chosen -- belong here, not in source comments.

## Context

The program, in the shape `digit-bound`'s `rescale_fixed` leaves it (the
`max(..., emin)` clamp is inserted there, not written by hand -- see
[Not a bug](#not-a-bug)):

```python
@fp.fpy(ctx=fp.REAL)
def fused_sum(xs: list[fp.Real]) -> fp.Real:
    if all([fp.isfinite(x) for x in xs]):
        e = max([max(fp.logb(x), fp.FP16.emin) for x in xs])
        with fp.MPFixedContext(e - 12, rm=fp.RM.RTZ, enable_neg_zero=False):
            ts = [fp.round(x) for x in xs]
        return sum(ts)
    else:
        with fp.FP32:
            return sum(xs)
```

lowered with `comp_to_loop`, `rescale_fixed`, `simplify`, `hoist_invariant`,
`hoist_scale`, at `list[Real[FP16]]` of 32.  Three things are wrong with the
emitted code:

1. **The scale is not hoisted.**  `t14 *` runs once per element where the shape
   wants `t14 * sum(ts)`.  `HoistScale.refusals` says *"the factor may be an
   infinity or a NaN"*.
2. **`e` is a `float`.**  It is a `logb`, so an integer belongs there.
3. **The `isfinite` assertion cannot fail**, the branch being guarded by
   `all([fp.isfinite(x) for x in xs])`.

### What actually decides them

Two knobs, measured over the program above.  `fuse` is `fpy2.strategies.fuse`
(`ReduceFusion`) run before `comp_to_loop`; `mono` is `monomorphize` to
`list[Real[FP16]]` of 32.  `e storage` is `choose_storage(fmt, cls)`, which is
what picks the C++ type; `(t * x)` is the operand of symptom 3's assertion,
which the emitter drops exactly when `_is_finite` holds of it.

| fuse | mono | hoist sites | blocking refusal | `e` class | `e` storage | `(t * x)` |
|---|---|---|---|---|---|---|
| no  | no  | 0 | the factor may be an infinity or a NaN | `NAN\|POS_INF\|ZERO\|FINITE` | (symbolic) | `TOP` |
| no  | yes | 0 | the factor may be an infinity or a NaN | `NAN\|POS_INF\|ZERO\|FINITE` | `F32` | `TOP` |
| yes | no  | **1** | -- | `ZERO\|FINITE` | (symbolic) | `ZERO\|FINITE` |
| yes | yes | 0 | the loop may not write every element of `ts` | `ZERO\|FINITE` | **`S8`** | `ZERO\|FINITE` |

**That table is the state before Phases 2 and 3.**  After them every cell
reads like row 3 -- one site -- and the monomorphized ones give `S8` and a
finite product, so the three symptoms are closed on the schedule that produced
them.  Row 3 is the existing unit-test schedule (`tests/unit/transform/test_hoist_scale.py`,
via `fused_sum` in `test_hoist_invariant.py`), which is why none of this was
caught.  Row 2 is the schedule that produced the listing on `digit-bound`.  The
target is for row 2 to read like rows 3 and 4 together: 1 site, `S8`, finite
product.

## One fact, two spellings

Both gaps are an analysis recognising one spelling of a fact the pipeline
produces in two.

### `all(mask)` is not read as a universal

`ValueClassAnalysis._implied_elements` / `_implied_universal`
(`fpy2/analysis/value_class.py`) read a *fold*:

```python
acc = True                  # what `fuse` leaves
for x in xs:
    acc = acc and fp.isfinite(x)
```

and refine the element class of `xs`'s region to `ZERO|FINITE`.  Without `fuse`,
`comp_to_loop` leaves a **materialised mask** instead:

```python
t7 = fp.empty(32)           # what `comp_to_loop` leaves
for t8 in range(32):
    t7[t8] = fp.isfinite(xs[t8])
if all(t7): ...
```

whose condition is an `AllOf` over a list, which `_implied_elements` does not
match at all, so `xs[i]` stays `TOP`.  Everything downstream follows from that
one class: `logb(x)` `TOP`, the clamp `TOP`, `e` `TOP`, `2 ** _k` possibly
infinite -- hence the refusal, hence `F32` for `e`, hence the assertion.  The
look-through wanted is the one `digit_bound`'s `_zero_paths` already does for
`all(...)` on the other branch.

**Settled: the analysis learns the mask, rather than schedules being required to
run `fuse` first.**  `fuse` is a performance rewrite, and making a
correctness-relevant fact depend on having run it hands every consumer of value
classes a scheduling precondition nobody states -- and the mask is what a
hand-written program produces anyway.

### `range(K)` is not read as covering

`HoistScale._covers` (`fpy2/transform/hoist_scale.py:211`) proves the loop fills
the list only for the spelling `for i in range(len(v))`, so monomorphizing --
which turns `len(xs)` into `32` -- loses it.  That is row 4's refusal, and it
sits *after* the finiteness check in `_why_not`, which is why rows 1 and 2 never
reach it.

### What is **not** the cause

The first draft of this page blamed format inference: `exact_logb` sets
`has_neg_inf` unconditionally, `exact_select` keeps the join's specials, and
`FormatInfer._implied` has no `IsFinite` case.  All three are true and none of
them matters here, because `storage_infer.without_absent` already intersects the
format's specials with the *value class* before storage is chosen -- `e`'s
format carries `enable_nan=True, enable_inf=True` in row 4 too, and it still
gets `S8`.  Measured, for the same reason:

- `AbstractFormat(has_neg_inf=True, has_pos_inf=False).format()` materialises to
  `MPBFixedFormat(enable_inf=True)` and round-trips back with **both** infinities.
  One-sided infinity is not representable in a concrete fixed format (see the
  note at `fpy2/analysis/format_infer/format.py:510`), so no `Min`/`Max` clamp
  can pay for itself through the format alone.
- `ValueClass`'s `Max` rule *already* clamps the specials the ordered way:
  `max(TOP, FINITE)` is `NAN|POS_INF|ZERO|FINITE`, the `-inf` gone.

**Settled: out of scope here.**  What that work would still buy is a tighter
*format* -- `FormatInfer._implied` gaining `IsFinite` / `IsNan` / `IsInf` cases,
and `exact_select` deriving the specials the ordered way (`max` is `-inf` only if
every operand may be, `+inf` if any may be) rather than from the join.  Both are
small and sound, and neither has a consumer today.  Revisit when something reads
a format's specials with no class to hand -- `round_is_identity` and the
double-round rules are the likely first.

## Phases

### Phase 1 -- Pin the grid.  **Done.**

`tests/unit/transform/test_fused_sum_schedule.py`, importing `fused_sum`
from `.test_hoist_invariant` and adding the clamped sibling above (the shape
`digit-bound` lowers).  A `_sched(f, *, fuse, mono)` helper, and assertions for
every cell of the table: `HoistScale.sites` / `refusals`, and for the `mono`
rows `ValueClassAnalysis.by_def[e]`, `choose_storage(fmt, cls)` and the class of
`(t * x)`.

Separate because it is the regression net: it pins behaviour no test covers
today, and Phases 2 and 3 are each a deliberate edit to it.  Everything it
asserts is the measured table -- the test passes as written, before any fix.

The net stops at the analyses.  `fused_sum` does not compile on this branch --
storage selection refuses the return value (`no storage format contains
MPBFloatFormat(pmax=89, emin=62, ...)`), which only digit-bound inference
narrows -- so the golden-listing test that would pin symptoms 2 and 3 in
*emitted* code is a follow-up on `digit-bound`, once these phases land there.

As written: a `TestGrid` parametrized over both programs and all four cells,
pinning `HoistScale.sites` and the full refusal list, and a `TestFiniteness`
holding the two monomorphized rows -- `e`'s class and `choose_storage`, and the
class of the rescaled round's operand, which is what the emitter's
`std::isfinite` assertion tests.  The clamped and plain programs refuse
identically, so the clamp appears only in `TestFiniteness`: it is what rules
`logb(0)`'s `-inf` out of `e`, and so what stands between `F32` and `S8` even
once the guard is read.  18 tests, all passing before any fix.

```
uv run pytest tests/unit/transform/test_fused_sum_schedule.py
```

### Phase 2 -- A literal trip count covers a list of known size.  **Done.**

`fpy2/analysis/array_size.py` gains a `trip_count(iterable, def_use, sizes)`
returning the loop's iteration count as an `ArraySize` -- the size of `v` for
`range(len(v))`, the integer for `range(32)`, `None` otherwise.
`HoistScale._covers` becomes `is_size_eq(trip_count(...), <list size>)`.

Separate from Phase 3 because it is a size question, not a class question, and
because Phase 3 needs the same helper.  Flips row 4 of the Phase 1 grid to 1
site; rows 1 and 2 are unchanged, their refusal being raised earlier.

As written: `trip_count` takes the iterable and an `ArraySizeAnalysis`, which
already carries the `DefineUseAnalysis` it needs to resolve the stop
expression.  A negative literal stop answers `0` -- it runs zero times, and so
covers an empty list -- rather than a negative size nothing is equal to.
`array_size._size_eq` is now public as `size_eq`, since `_covers` compares two
*sizes* where it used to compare two whole bounds, which pointlessly required
the element bounds to agree as well.  Tests: `TestTripCount` in
`tests/unit/analysis/test_array_size.py` for both spellings and the three tops,
and `test_a_literal_trip_count_covers_the_list` in `test_hoist_scale.py` for the
monomorphized lowered form, where the site resolves to `sum(ts)`.

```
uv run pytest tests/unit/transform/test_hoist_scale.py \
  tests/unit/analysis/test_array_size.py \
  tests/unit/transform/test_fused_sum_schedule.py
```

### Phase 3 -- Value classes through a materialised mask.  **Done.**

`ValueClassAnalysis._implied_elements` gains an `AllOf` / `AnyOf` case: resolve
the scanned list's region, find the `ForStmt` that filled it, require the write
to be at the loop index and to cover the mask, then read the written predicate
with `_implied` and map any refinement of the body's element read back onto the
iterated list's region.  The existing `_one_list` / `_scanned` guards carry over
unchanged -- a store since the exit, or a region holding two lists, still says
nothing.

Coverage is proved, not assumed: the case uses Phase 2's `trip_count` and
requires the mask's size to equal the scanned list's.  Value-class analysis
already describes only executions in which every operation has a result, and
reading an unwritten `Empty` slot has none -- so the mask could be taken as
fully written by assumption.  Phase 2 builds the check anyway, and leaning on a
soundness assumption to skip it would make it carry more than it was written
for.

Separate because it is the deeper fact and the one that stands alone: it closes
symptoms 2 and 3 by itself, and with Phase 2 closes symptom 1.  Flips rows 1 and
2 of the grid to match rows 3 and 4.

As written, with three divergences from the plan:

- **`_scanned` could not be reused.**  It holds the stamp of the list a loop
  *iterates*, and a mask loop iterates a `range` -- the list being scanned is
  read inside the body.  So `_entered` records the `_clock` each loop began at
  instead, and the scanned region's stamp must not exceed it: no store to it
  during the scan or between the scan and the guard.  That is one condition
  where the fold has two, and it subsumes both.
- **Coverage cost value-class analysis a new dependency.**  `trip_count` needs
  an `ArraySizeAnalysis`, which `ValueClassInfer` did not build.  It is
  computed lazily, on the first mask a program guards on, so nothing else pays
  for it.
- **The element has to be bound.**  The refinement travels through the
  definition the predicate reads, so `x = xs[i]; m[i] = isfinite(x)` works and
  `m[i] = isfinite(xs[i])` says nothing.  Every lowering binds it; a
  hand-written program need not, and `test_an_unbound_element_says_nothing`
  records that.

Tests: `TestAMaterialisedGuard` in `tests/unit/analysis/test_value_class.py`,
which mirrors `TestAGuardOverAWholeList` -- the same facts, lowered only as far
as `CompToLoop` -- plus the prefix scan, the store between, and the unbound
read.

A review pass afterwards found four unsound cases in the first cut of this,
all now guarded and each with a test: a guard *inside* the scan reading a
half-filled mask (`_scan_clocks` is dropped before the body, exactly as
`_scanned` is); a store through a second name for the mask; a mask handed to a
callee; and a second write to the mask earlier in the same round, which the
last-definition check alone did not see.  The first cut checked the *scanned*
list thoroughly and the mask not at all -- the asymmetry to remember is that
the fold's accumulator is a scalar, so it cannot be aliased, and a mask is a
list, so it can.  The same pass found a pre-existing miscompile in
`HoistScale._indexed_by_target`, which compared the index's *name* to the
loop's target rather than its definition; Phase 2's literal trip count made it
newly reachable, so it is fixed here and `index_rebound_in_the_body` joins
`UNSOUND`.

The grid collapsed further than expected: with both fixes in, all four cells
hoist, so `fuse` no longer changes the answer either.  One thing the pins make
explicit that the plan did not: **the clamp is load-bearing for symptom 2 on its
own.**  `logb(0)` is `-inf` whatever the guard proves about the elements, so the
unclamped program keeps `NEG_INF` in `e` and stays on `F32` even fused; only
`max(logb(x), emin)` reaches `S8`.  Since that clamp is inserted by
`digit-bound`'s `rescale_fixed` and not by this branch's, symptom 2 is only
fully observable there.

```
uv run pytest tests/unit/analysis/test_value_class.py \
  tests/unit/transform/test_fused_sum_schedule.py
```

### After the last phase

```
uv run pytest -n 8 tests/unit
uv run python -m tests.infra
uv run python -m tests.infra.backend.cpp --mode run
make lint
```

## Weirdness found by reading the output

Not scheduled, and none of them is a wrong answer -- they are emitted code
nobody would write.  All were read off `digit-bound`'s listing.

- [ ] **(C6) The overflow assert and the storage disagree on width.**  The assert
      checks the `int64_t` range; the value is then narrowed to `int16_t` by an
      implicit conversion with no check and no cast.  Digit-bound inference is
      what proves 13 bits suffice, so the narrowing is sound -- but the assertion
      guards a bound the program does not rely on and says nothing about the one
      it does.  Worth checking whether `-Wconversion` fires.
- [ ] **A `NaN` test on a literal.**  `max(logb(x), -14)` emits
      `std::isnan(_tmp3)` where `_tmp3` is the constant `-14`.
- [ ] **A double cast.**  `static_cast<float>(static_cast<float>(-14))`.
- [ ] **A dead multiply.**  `std::pow(2.0, _k) * 1` in both scale expressions.
- [ ] **A signed-zero tie-break on exponents.**  The `max` fold emits
      `(_tmp4 == _tmp6 && std::signbit(_tmp4))` for clamped `logb`s -- integers,
      never a signed zero.
- [ ] **Reciprocal scales computed twice.**  `t = 2 ** -_k` and `t14 = 2 ** _k`
      are each a guarded `ldexp`; one is the other's reciprocal.
- [ ] **The `isfinite` mask is materialised.**  All 32 elements are computed into
      a `std::array<bool, 32>` before `std::all_of`, so a non-finite first
      element still costs the whole pass.  A short-circuit loop would do.  Note
      this is the *emitted* mask; Phase 3 is about the mask in the AST, and does
      not change what is emitted.

## Not a bug

- **The `max(logb(x), emin)` clamp is not in the source.**  `digit-bound`'s
  `rescale_fixed` inserts it.  It is semantics-preserving here -- checked against
  the untransformed function on all-subnormal, mixed and all-normal FP16 inputs,
  identical results -- because a clamp to `emin` only moves the rounding
  position, and `emin - P` still sits below the finest representable digit when
  `P >= p - 1` (here `12 >= 10`).  **That inequality is an unstated
  precondition**; worth asserting wherever the clamp is introduced.
- **Three separate reads of `xs[i]` in three loops.**  The first cannot fuse with
  the others because `all(...)` must finish before the branch, and the second
  cannot fuse with the third because `e` must finish before `_k`.
- **The two remaining refusals.**  `HoistScale.refusals` reports *"no scaled list
  write fills the reduction"* twice, for the other two reductions.  They are
  separate, they persist in every row of the grid including the working one, and
  they have not been looked at.

## Open items

The four questions this plan opened were all settled before Phase 1 -- the
analysis learns the mask, coverage is proved via `trip_count`, the C++
end-to-end regression is a follow-up on `digit-bound`, and the
format-inference work stays out of scope -- and each is recorded where it
applies, above.  Review opened one more.

### Does `_one_list` have to separate the rows of a nested list?

It does not today, and that is unsound:

```python
r0 = xss[0]
r1 = xss[1]                     # one region, one allocation site
ok = all([fp.isfinite(x) for x in r0])
if ok:
    return max(r1)              # inferred `ZERO|FINITE`; returns `+inf`
```

`_one_list` counts allocation sites, and two rows of one 2-D list share
theirs, so a fact proved about one row lands on every other.  **The fold
spelling is wrong in exactly the same way**, so this is pre-existing in
`_implied_universal` rather than introduced by `_implied_mask`, which inherits
it by using the same predicate.

At stake: the cheap fix is to make `_one_list` refuse a region reached through
another list's element slot, which costs nothing this repo's programs rely on
but is a guess at the right condition; the real fix is for `AliasAnalysis` to
give rows distinct regions, which is larger and touches every consumer.  Doing
neither leaves a known-unsound refinement in two places.

Provisional: left as it stands, because it is pre-existing and because
tightening `_one_list` by hand risks silently costing the refinement this page
exists for.  It should be reopened before anything ships that trusts element
classes on a nested list -- nothing here does, every program on this page
scanning a flat one.
