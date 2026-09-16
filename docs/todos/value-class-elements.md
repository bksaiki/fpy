# Value classes for list elements

A list carries a class for its *elements*, so a guard over the list reaches the
reads inside it.  Built on the `value-class-elements` branch and held back from
merge: the design keys a property of the runtime *object* by *definition*, and
an FPy list is a reference.

The motivating program, which the branch compiles to `int8_t max_e` and `main`
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

## What the branch has

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
  store or which reaches a call, an aggregate, or a call's result.
- Regression tests for both unsound cases below.

## Why it is not merged

Two cases reported a finite class for a value the interpreter makes a NaN:

```python
ys = xs; ys[0] = fp.nan(); fp.logb(xs[0])   # a store through an alias
z = poison(xs);            fp.logb(xs[0])   # a store inside a callee
```

Both are guarded now, but the guard is a syntactic approximation of `Alias`.
That is the thing `reaching_defs.same_object_defs` warns against:

> Stating it once keeps allocation tracking, alias analysis and the C++
> backend's storage coalescing from drifting apart -- they had each written it
> out separately.

`Alias` is the real answer for the first case — it reports the region a store
writes — but `written_regions` is intraprocedural, so the second does not
appear; that needs escape summaries.  `Alias` also runs *after* this analysis,
and `unfold_special` uses `ValueClassInfer` standalone.

Both were found by inspection, not by a failing test, which is the part that
should not be trusted.

## What converging needs

1. **Key element facts by alias region, not definition.**  `AliasAnalysis`
   already supplies `region_of`, `written_regions` and `referrers`.  Needs
   `Alias` available to this analysis — it depends only on def-use and escape
   summaries, so there is no cycle, but the standalone callers need an answer
   for when it is absent.
2. **Escape summaries for the callee case**, or a documented refusal to track
   any list that reaches a call.
3. **One storage-narrowing site.**  Storage is chosen at four independent
   places: `StorageInfer._aggregate` (definitions), `return_storage` (the
   return), `CppEmitter._storage_for_expr`, and `CppStorage.of_expr`.  Only the
   first two consult a value class, and narrowing a list's elements without the
   other two produced an `int8_t` array reduced by a `float` fold.  Landing the
   element class *usefully* means all four agreeing — the same divergence the
   merged `SpecAnalyses.ret_ty` already fixed once, one level up.
4. **A store into an integer slot from a float.**  Even coherent, this needs
   `_require_no_narrowing` to accept an exact conversion: `fmt(max(logb(x),
   emin)) <= int8_t` holds, and the store wants
   `static_cast<int8_t>(std::max(...))` while the `max` itself stays
   `float -> float -> float`, since `fmt(logb(x)) <= float`.
5. **Property tests against the interpreter.**  `TestTransferFunctionsAreSound`
   sweeps every scalar table against real `Float` arithmetic, and is the reason
   the sign split is trustworthy.  The element rules have nothing equivalent.

Do 1-3 before 4: a narrowing that only half the backend agrees with produces
worse code than none.
