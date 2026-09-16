# Value classes for list elements

A list carries a class for its *elements*, so a guard over the list reaches the
reads inside it.  **Phases 1, 2, 5 and 6 are done**; 3 and 4 did not survive
being attempted, and what they were reaching for is phase 7 below, which now
has the witness they lacked.

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

## What is left

**7. A list's element storage.**  `t9` in `sandbox3.py` holds
`max(fp.logb(x), FP32_EMIN)` for a guarded `x` -- a finite integer in
`[-126, 127]` -- and is still a `std::array<float, 32>`.  `of_bound` narrows by
a class at a scalar only, deliberately: storage is chosen at several sites and
only that one consults a class, so narrowing a list's elements there alone
would disagree with the stores into it and the reduction over it.  Doing it
means the element class reaching `StorageInfer` per *definition* rather than
only per read, and every site agreeing on the answer.

Phases 3 and 4 were this without a case that failed.  Phase 6 produces one, and
the rule to apply is the same: does it follow from the semantics of the
operation, or from the shape of one program?

**Signed zero.**  `_emit_amin_amax` still emits the `signbit` tie for `max(t9)`,
because `ValueClass.ZERO` does not track the sign and `logb` yields only `+0`.
Splitting it the way `INF` was split into `POS_INF`/`NEG_INF` would drop the
term.
