# Value classes for list elements

A list carries a class for its *elements*, so a guard over the list reaches the
reads inside it.  Built once and abandoned: the design keyed a property of the
runtime *object* by *definition*, and an FPy list is a reference.  What follows
is what the attempt established and the order to rebuild it in.

The motivating program, which the attempt compiled to `int8_t max_e` and `main`
compiles to `float`:

```python
if all([fp.isfinite(x) for x in xs]):
    max_e = max([max(fp.logb(x), fp.FP32.emin) for x in xs])
```

`logb`'s format is an integer grid over `[-149, 127]` that also admits a NaN and
both infinities, because `logb(0)` is `-inf` and `logb(inf)` is `+inf`.  No
integer rung holds those, so storage falls to `float`.  The guard rules out the
NaN and the `+inf`, the clamp rules out the `-inf` — and neither reaches the
element read without this.

## What the attempt established

- **Forward.** A list definition carries its elements' class, joined from the
  stores that build it: `empty(...)` is bottom, a store joins in, a literal
  joins its elements, a copy inherits.  Phis merge it and the loop fixpoint
  iterates on it, with *absent* reading as the top.  Consumed by an element
  read, a `for` target over a list, and `AMin` / `AMax`.
- **Backward.** `_implied_elements` matches the lowered reduction loop and reads
  it as a universal: `all(...)` refines the taken arm, `any(...)` the untaken
  one.  Sound because FPy has no `break`.  The match is strict — a literal seed,
  a step that is exactly `acc <op> b` naming that phi, and a refinement the loop
  target itself carries.
- **A guard**, `_shared_and_mutated`, dropping any list whose copy group holds a
  store or which reaches a call, an aggregate, or a call's result -- a
  hand-rolled approximation of `Alias`, and not the shape to rebuild.

## Why it was abandoned

Two cases reported a finite class for a value the interpreter makes a NaN:

```python
ys = xs; ys[0] = fp.nan(); fp.logb(xs[0])   # a store through an alias
z = poison(xs);            fp.logb(xs[0])   # a store inside a callee
```

Each was guarded syntactically, which is the thing
`reaching_defs.same_object_defs` warns against:

> Stating it once keeps allocation tracking, alias analysis and the C++
> backend's storage coalescing from drifting apart -- they had each written it
> out separately.

`Alias` is the real answer for the first case — it reports the region a store
writes — but `written_regions` is intraprocedural, so the second does not
appear; that needs escape summaries.  `Alias` also runs *after* this analysis,
and `unfold_special` uses `ValueClassInfer` standalone.

Both were found by inspection, not by a failing test, which is the part that
should not be trusted.

## The plan

Six phases, each about one commit, ordered so the two that make the feature
*safe* land before the two that make it *pay*, and the feature itself last.
Phases 1-4 are worth merging on their own.

One test to apply to each: does the rule follow from the *semantics of the
operation*, or from the shape of one program?  Phases 1-5 pass it outright.
Phase 6's rule passes and its implementation is where the judgement is.

**1. A differential check for value classes.**  Every defect in the first
attempt was found by reading code, not by a test: the suites stayed green
throughout, and the storage narrowing that did merge turns out to be inert on
all 127 corpus functions.  Extend the `--mode run` machinery so a program's
*claims* are checked against its run -- for each expression the analysis gives a
class, assert no observed value falls outside it.  It costs nothing to build
before the feature, checks the existing scalar classes meanwhile, and is the one
thing that would have caught both unsound cases.

A cheaper down payment, worth doing first either way: one guarded-`logb`
program in `tests/infra/examples/`, which pulls the *merged* storage narrowing
into the bit-exact differential it currently sits outside of.

**2. `Alias` available to `ValueClassInfer`.**  It needs only def-use and escape
summaries, so there is no cycle with this analysis -- it is merely ordered after
it today.  Plumb it as an optional input and decide what a standalone caller
gets without it: `unfold_special` constructs `ValueClassInfer` directly, and the
answer should be that element tracking is simply off.  Escape summaries are what
cover a callee storing through a list it was handed, which `written_regions`
alone does not.

**3. One storage-narrowing site.**  (Its payoff is thin and worth knowing:
`logb` is nearly the only operation whose format is integer-valued, range-bounded
*and* special-admitting, so it is nearly the only beneficiary -- `floor`,
`trunc` and `nearbyint` under `REAL` are unbounded and refused before narrowing
applies.  The mechanism is right; the population is small.)  Storage is chosen at four independent
places -- `StorageInfer._aggregate`, `return_storage`,
`CppEmitter._storage_for_expr` and `CppStorage.of_expr` -- and only the first two
consult a value class.  Narrowing a list's elements without the other two
produced an `int8_t` array reduced by a `float` fold.  Make one place own it and
the rest read it, as `SpecAnalyses.ret_ty` already does one level up.  No new
precision, so the output should not move.

**4. An exact store into a narrower slot.**  `_require_no_narrowing` refuses on
storage types alone.  Where the value's *format* fits the slot the conversion is
exact and should emit the cast: `fmt(max(logb(x), emin)) <= int8_t` holds, so the
store wants `static_cast<int8_t>(std::max(...))` while the `max` itself stays
`float -> float -> float`, since `fmt(logb(x)) <= float`.  Useful on its own --
the refusal is currently over-strict -- and without it a narrowed element type
only produces a compile error.

**5. Element classes, forward, keyed by region.**  A list carries its elements'
class, joined from the stores that build it, keyed by the alias region from
phase 2 rather than by definition.  Consumed by an element read, a `for` target
over a list, and `AMin`/`AMax`.  Phase 1 is what says it is right.

**6. The reduction refinement.**  The rule follows from the language:
`all(p(x) for x in xs)` holding means every element satisfies `p`, and FPy has
no `break`, so the loop covers the list.  The question is how to know it.

The first attempt matched the *lowered* accumulator loop -- a literal seed, a
step that is exactly ``acc <op> b`` with ``b`` a name bound to the predicate, a
particular phi.  **A failed match returns nothing**, so the risk is not a wrong
answer but a silent one: writing the loop by hand with the predicate inlined
(``acc = acc and fp.isfinite(x)``, no ``b``) stops matching, and so would a
change in `CompToLoop` or `Simplify`, with no signal either way.

Three routes, in increasing cost:

- Keep the match, and make its failure loud: tests that pin each shape it is
  meant to accept, so a lowering change is a test failure rather than a
  regression nobody sees.  Defensible as long as the coupling is written down.
- Recognize the universal where ``any`` / ``all`` is still a first-class form
  and carry the fact through lowering.  Needs a carrier the lowering preserves,
  since value classes are keyed by expression identity.
- Infer the loop summary properly, which subsumes both and is a project.

Pick on evidence: how many shapes the match has to accept, and whether the
lowering is stable enough that pinning them is honest.  Phase 5 stands on its
own; phase 6 is what the motivating program additionally needs, and neither
should land before 1.
