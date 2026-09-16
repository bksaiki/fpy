# Value classes for list elements

A list carries a class for its *elements*, so a guard over the list reaches the
reads inside it.  Built once and abandoned: the design keyed a property of the
runtime *object* by *definition*, and an FPy list is a reference.  What follows
is what the attempt established and the order to rebuild it in.

## The acceptance test

This program, which the first attempt compiled to `int8_t max_e` and `main`
still compiles to `float`:

```python
if all([fp.isfinite(x) for x in xs]):
    max_e = max([max(fp.logb(x), fp.FP32.emin) for x in xs])
```

`logb`'s format is an integer grid over `[-149, 127]` that also admits a NaN and
both infinities, because `logb(0)` is `-inf` and `logb(inf)` is `+inf`.  No
integer rung holds those, so storage falls to `float`.  The guard rules out the
NaN and the `+inf`, the clamp rules out the `-inf` — and neither reaches the
element read without this.

Done means `max_e` is an integer type here, soundly: with the two witnesses
below still reporting the top class, and phase 1's differential check green.

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

**Phases 1 and 2 are done; 3 and 4 did not survive being attempted.**  Both
were written from one observation during the failed attempt, and neither holds
up on its own (see their entries).  The storage question they were meant to
answer is real but underdetermined until phase 5 produces a concrete case, so
the arc is 2 -> 5 -> 6, and whatever coherence phase 5 needs gets written then
against something that actually fails.

That is a statement about *shape*, not about worth.  The corpus is example
programs and the runtime is dominated by the FP16 ladder, so neither says
anything about whether this matters -- the acceptance test is the program
below, and a compiler that emits a ``float`` for a value it can prove is a
small integer is the thing being fixed.

One test to apply to each: does the rule follow from the *semantics of the
operation*, or from the shape of one program?  Phases 1-5 pass it outright.
Phase 6's rule passes and its implementation is where the judgement is.

**1. A differential check for value classes.**  **Done.**  Every defect in the
first attempt was found by reading code, not by a test: the suites stayed green
throughout, and the storage narrowing that did merge was inert on every corpus
function.  Extend the `--mode run` machinery so a program's
*claims* are checked against its run -- for each expression the analysis gives a
class, assert no observed value falls outside it.  It costs nothing to build
before the feature, checks the existing scalar classes meanwhile, and is the one
thing that would have caught both unsound cases.

A cheaper down payment, worth doing first either way: one guarded-`logb`
program in `tests/infra/examples/`, which pulls the *merged* storage narrowing
into the bit-exact differential it currently sits outside of.

**2. `Alias` available to `ValueClassInfer`** (prerequisite).  **Done.**  It is
an optional argument, computed here when absent: `Alias` costs less than this
analysis does, so there is no reason to let a caller silently lose a fact.

Escape summaries turn out not to be needed.  Without them `Alias` marks *every*
list handed to a call as escaping, which is the conservative reading and
exactly what is wanted -- so `ValueClassAnalysis.element_region` is
"the region, unless it escapes", and the callee case falls out.  The aliasing
case falls out too, and is why the region is the key rather than something to
refuse: ``ys = xs`` is one object under two names, so a store through either
lands on the region both resolve to.

**3. One storage-narrowing site.**  **Tried and withdrawn.**  Narrowing
`_storage_for_expr` by the expression's class is wrong: an expression's storage
is the type its *operands* are cast to, and the operands of a selection are not
bounded by its result.  ``max(logb(x), -126)`` is finite, but ``logb(x)`` is not
-- ``max`` must stay ``float -> float -> float`` and the narrowing belongs at
the *boundary*, not at the operation.  Attempting it broke the double-clamp
witness with a refused ``float`` to ``int8_t`` cast, and moved one corpus
function.

**4. An exact store into a narrower slot.**  **No witness.**
`_require_no_narrowing` is reached 48 times over the corpus and fits every time,
and a list whose declared element format is narrower than the stored value's
storage compiles today without reaching it -- the emitted store is an implicit
conversion.  Making the check class-aware changes no corpus output.  Whatever
phase 5 needs here should be written from the case phase 5 actually produces,
not from this one.

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
