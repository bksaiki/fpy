# Value classes for list elements

A list carries a class for its *elements*, so a guard over the list reaches the
reads inside it.  **Done.**  Phases 3 and 4 did not survive being attempted;
what they were reaching for landed as phase 7, once phases 5 and 6 produced the
case they had no witness for.

## What exists

- **A class per region.**  `_elt` maps an alias `Region` -- the set of run-time
  locations a place may hold -- to what every element there is, joined from the
  stores that build the list and merged at a branch.  Keyed by location rather
  than by name: `ys = xs; ys[0] = fp.nan()` is one list under two names, and the
  first attempt, keyed by definition, reported a class a run contradicted.
  Read by an element read, a `for` target, and `AMin`/`AMax`.
- **The universal a guard implies.**  `_implied_universal` reads a reduction
  loop.  `acc` at the exit is `seed and X_1 and ... and X_n`, so `acc` true
  forces every `X` -- whatever the seed, with an empty list vacuous -- and the
  loop covers the list, FPy having no `break`.  Dually for `any`, an `Or` that
  speaks when it is false.  The accumulator must be an operand of its own new
  value: that is what makes the fold monotone, and `acc = p(x)` would otherwise
  speak for the last element as if for every one -- and where `Hoistable` has
  moved the fold into a guarded assignment, what it guards on must be what the
  loop carried in, since `ok = x > 0; if ok: ok = p(x)` rebuilds it each round.
  Nothing in the match names what `ReduceFusion` mints, so an
  inlined predicate and a hand-written fold reach it too.
- **Freshness.**  The fact is about the contents at the loop's *exit*.
  `_touched` stamps every region a store lands on, and a fact whose stamp has
  moved is dropped -- covering a store inside the scan itself, one between the
  scan and the guard, and one in a branch nested inside the guarded arm.  The
  last is why the stamp exists rather than dropping the mask on a store: an arm
  *restores* its mask, so leaving the inner branch would bring back a fact that
  branch's own store had invalidated.
- **A differential check.**  `tests/infra/analysis/value_class.py` runs each
  corpus program and compares every observed value against the class claimed.
  It drives both the written form and the one the C++ backend analyzes: a
  comprehension is an expression and `all` a single node, so the written form
  has neither an element store nor a reduction loop, and checking it alone left
  every one of these phases uncovered.
- **Element storage.**  A list stores at what its elements can be rather than
  at their format, so the guarded `fp.logb` list is a byte array folded on the
  integer path.  Three things had to line up, and all three are the general
  rule rather than a special case:
  - `by_elt` joins every class ever stored into a region, monotone where `_elt`
    is flow-sensitive: a buffer holds what a list *ever* held.
  - `_require_no_narrowing` asks whether the **value** fits the slot, not
    whether the expression's storage does.  An expression's storage is what its
    *operands* are cast to -- `max(logb(x), -126)` computes at `float` because
    `logb` does -- and the store spells the conversion it then needs.
  - `_storage_for_expr` defers to the declaration for a `Var` or a `ListRef`.
    There were two storage oracles, one per-expression and format-driven and
    one per-definition and class-aware; they differ exactly where a class
    narrowed a definition, and the declaration is the one that was emitted.

## What is left

**Signed zero.**  `_emit_amin_amax` still emits the `signbit` tie for a float
reduction, because `ValueClass.ZERO` does not track the sign and `logb` yields
only `+0`.  Splitting it the way `INF` was split into `POS_INF` / `NEG_INF`
would drop the term.

**`_emit_min_max` takes its type from the active context**, working around the
two oracles disagreeing (`library_core.max_e` in its docstring).  With one
oracle that workaround looks unnecessary -- taking the type from the operands
passes the unit suite -- but it was not measured against the differentials and
is not this phase's business.
