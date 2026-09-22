# Loop splitting: a masked tail

Implementation plan.  The design is settled; what follows is the phase
breakdown, one phase per commit.

## Working policy

- **Pause after each phase for review.**  Do not begin the next phase until
  the current one has been looked at.
- **Do not commit.**  The working tree is left dirty; commits are made by the
  repository owner.
- **Run only the tests relevant to the phase.**  The full unit suite runs once,
  at the end, after the last phase.
- **Comments stay succinct**, and notes about *process* — what was tried, what
  a phase decided, why an ordering was chosen — belong in this document, not in
  source comments.

## Context

`fpy2/transform/split_loop.py` turns `for i in range(n)` into an outer loop
over chunks and an inner loop over a chunk's elements.  It offers two ways to
handle a length that is not a multiple of the factor, and neither suits a
target that masks.

For `n = 10`, factor `4`:

| Strategy | What it emits | Why it does not suit a SIMD target |
|---|---|---|
| `PEEL` | chunks over `[0, 8)`, then a **second loop** over `[8, 10)` holding a *clone* of the body | two copies of the body to emit, and the residual is not a tile |
| `STRICT` | chunks over `[0, 10)`, guarded by `assert fmod(n, f) == 0` | refuses the length outright — here the assert fails |

Verbatim, the tail each produces today:

```python
# PEEL — a duplicated residual body
for j8 in range(t5, t4, 1):
    i = t[j8]
    out[i] = (xs[i] * 2)

# STRICT — a runtime divisibility demand
assert fp.fmod(t4, t3) == 0
```

A SIMD target wants neither.  Its idiom is a full-width tile every iteration
with the overflow lanes disabled — `range(0, cdiv(n, B))` plus a mask — so the
body is emitted once and the tail costs a predicate rather than a second loop.

## The shape `MASK` emits

Every chunk is a full factor wide; the tail is a guard.

```python
for b in range(0, n, f):             # already runs ceil(n / f) times
    for j in range(b, b + f, 1):     # always a full f wide
        if j < n:                    # the mask
            i = t[j]
            <body>
```

The guard is load-bearing rather than cosmetic: without it the last chunk
reads `t[j]` past the end, which FPy leaves undefined and the interpreter
raises on.  With it the rewrite is semantics-preserving, so the usual
differential check applies.

Checked by hand against the unsplit program for `n = 0 … 9` at `f = 4` — every
remainder, including the empty and exactly-divisible cases — and all ten agree.

**Why a guard and not a clamped inner bound.**  `range(b, min(b + f, n))` would
avoid the predicate, but it makes the inner trip count vary by chunk.  A tile
has to be a compile-time constant width, so a varying bound is the one thing
the shape cannot have.  The guard keeps the width fixed and moves the tail into
a value the target already knows how to handle.

## Where it goes in the existing code

`_build_mask` sits beside `_build_strict` and `_build_peel`, selected by the
same `match` in `_visit_for`.  Three smaller points:

- `_chunk_loop` grows an optional mask bound.  When given, it wraps the inner
  body — *including* the `i = t[j]` element read, which is the access the guard
  exists to prevent — in an `If1Stmt`.
- `_refuses` is unchanged.  `MASK` is correct for any length, so like `PEEL` it
  refuses nothing and every loop stays a site.
- The static path drops the guard when it cannot fire.  Where the length and
  factor are both known and the length divides, `MASK` emits exactly what
  `STRICT` does, with no predicate and no assert.

**No padded bound, and so no remainder.**  The first draft rounded the length
up to a multiple of the factor and chunked *that*, which needed a remainder to
compute.  It is unnecessary: `range(0, n, f)` already yields `ceil(n / f)`
chunks, so the bound is the length itself and the last chunk simply over-runs
it -- which the guard was already there to handle.  Checked for every `n < 25`
at `f` in 1..8: the chunk count matches `ceil(n / f)`, and the guarded inner
loops visit `[0, n)` exactly once, in order.

That is what makes `MASK` lowerable by a target with no integer remainder, so
it is a correctness property of this design rather than a tidy-up.

**The `f >= 1` assert is still load-bearing.**  `range(0, n, f)` with a
non-positive `f` is empty, silently skipping every iteration, which is why
`_dynamic_prelude` rejects it loudly.

## `rem` is the caller's choice: a `use_fmod` flag

FPy spells a remainder two ways and they are not interchangeable.  `Fmod`
truncates, `Mod` (`%`) floors, and they part on a negative dividend
(`-5 fmod 4 = -1`, `-5 % 4 = 3`).  Here both are numerically fine — `n` is a
length and `f - rem(n, f)` lands in `[1, f]`, so nothing is ever negative — but
which one a *backend* can emit differs, and that is not this transform's
business to know.

So the transform takes `use_fmod: bool = True` and the caller picks.  The
default preserves today's behaviour, since `_build_peel` and `_build_strict`
already emit `Fmod`.

The flag belongs to the whole transform, not to `MASK`: all three strategies
emit a remainder, so all three honour it.

**What this deliberately does not do.**  It does not make the output compile on
the cpp backend, because *neither* spelling does — `Mod` has no entry in the
cpp op table and `Fmod` has no integer signature, so a dynamic-length split
does not compile today under either choice.  That is a pre-existing gap,
recorded as *No integer remainder compiles* in
[backend-cpp.md](backend-cpp.md), and closing it is that backend's work rather
than a reason to hold this one.  The flag defers the policy; it does not
pretend to resolve it.

## Phases

### Phase 1 — the `MASK` strategy

**Done.**  `use_fmod` ended up transform-wide rather than `MASK`-only, as
planned; `_lister` deliberately does not take it, since it changes which node
spells a remainder and never whether a loop is a site.


`fpy2/transform/split_loop.py`: add `SplitLoopStrategy.MASK`, `_build_mask`,
the optional mask bound on `_chunk_loop`, and the `match` arm in `_visit_for`.

Also the `use_fmod` flag, threaded through `SplitLoop.apply`,
`apply_with_edits`, `sites`, `refusals` and the `_lister`, and honoured by
`_build_peel` and `_build_strict` as well as `_build_mask` — one helper
returning `Fmod(...)` or `Mod(...)` is enough.

A separate phase because it is the whole behavioural change, and because the
differential check below is what establishes the rewrite is sound — nothing
should be bundled ahead of it.

Tests, new in `tests/unit/transform/test_split_loop.py`:

- the emitted shape — one loop nest, no residual, no divisibility assert;
- `use_fmod` selects the node, for every strategy, and defaults to `Fmod` so
  existing output is unchanged;
- a differential over `n = 0 … 9` at `f = 4`, against the unsplit program;
- the exactly-divisible static case emits no guard;
- the empty case (`n = 0`) emits nothing to run.

```
.venv/bin/python -m pytest tests/unit/transform/test_split_loop.py -q
```

### Phase 2 — fold `MASK` into the shared contract

**Done.**  `_BOTH` became `_ALL`.  The four `MASK` cases added to
`TestRemainder` are not copies of the `PEEL` ones: `test_mask_mutation_keeps_its_order`
pins a *different* property, since `PEEL` has to carry a mutation across the
residual boundary and `MASK` has no boundary to cross.


The suite parameterizes 27 tests over `_BOTH = (STRICT, PEEL)`.  Extend that to
`MASK` wherever the property is strategy-independent — cursor forwarding,
`sites`/`refusals` agreement, nested loops, the `with fp.INTEGER` wrapping of
inserted arithmetic — and update the `fpy2.strategies.split` wrapper docstring.

Separate from Phase 1 so that the new strategy is reviewed on its own output
before it is held to the shared properties, and so a failure in the shared
suite is unambiguous about which change caused it.

```
.venv/bin/python -m pytest tests/unit/transform/test_split_loop.py tests/unit/strategies -q
```

### Phase 3 — `use_fmod` on `ForUnroll`

**Done.**  One bug worth recording: `ForUnroll.apply` took the flag and
dropped it, forwarding to `apply_with_edits` without it, so `use_fmod=False`
silently kept emitting `fmod`.  Caught by the test asserting `%` appears, not
by the equivalence test -- which passed either way, because both spellings
compute the same value.  The shape assertion is what has teeth here.


`fpy2/transform/for_unroll.py` is the only other transform that synthesizes a
remainder: a module-level `_fmod` helper with two call sites, both inside
`_ForUnroll` methods — the `STRICT` divisibility assert and the `PEEL` prefix
bound.  Make it a method honouring the flag, and thread `use_fmod` through
`ForUnroll.apply`, `apply_with_edits`, `sites`, `refusals`, `_lister` and the
`fpy2.strategies.unroll_for` wrapper.  The same surgery as Phase 1, on a
smaller surface.

**Why this is not optional, and not scope creep.**  Today the codebase is
*consistent*: every synthesized remainder is an `Fmod`.  Phase 1 is what
introduces the possibility of a program that mixes the two — a pipeline
running `split(use_fmod=False)` alongside `unroll_for`, which hardcodes
`Fmod`, emits both spellings and can be lowered by neither backend.  That is
strictly worse than either choice alone, so Phase 3 finishes what Phase 1
starts rather than extending it.

The route these two share is real, not hypothetical: `Specialize` →
`unroll_for` → `single_exit` is how a loop-carried `return` is removed, and
`split` runs over the same programs.

Tests, in `tests/unit/transform/test_for_unroll.py`:

- `use_fmod` selects the node for both strategies, defaulting to `Fmod`;
- the spelling does not change the answer, over the same lengths the existing
  equivalence tests use.

```
.venv/bin/python -m pytest tests/unit/transform/test_for_unroll.py tests/unit/strategies -q
```

### After the last phase

```
.venv/bin/python -m pytest tests/unit -q -n 8
.venv/bin/mypy fpy2 && .venv/bin/ruff check fpy2
```

## Fitness for the Triton backend

Checked against the roadmap on `triton-compiler`, not assumed.

Its *How a loop lowers* requires idiom 2 for the general case: "the trip count
is a *runtime* value and the body is tile-shaped, so one compiled kernel serves
every `K`.  This is what item 2's `split` produces."  A compile-time trip count
is the fallback it explicitly argues against -- one kernel per input length.

So the path that matters is the **dynamic** one, and the first draft failed it:
that path computed a padded bound with a remainder, and the Triton op table has
neither `Fmod` nor `Mod` -- only `Div`, with integer `Div` deliberately omitted
because FPy truncates where Triton's `//` floors.

Dropping the padded bound fixes it.  The dynamic path now emits:

```
for i in range(0, t4, t3):     # runtime trip count
    for j in range(i, t5, 1):  # constant width
        if j < t4:             # the mask
```

which needs only `len`, `+` and `<`, all of which that table has.  Measured
across the three paths:

| | remainder | `len()` | guard |
|---|---|---|---|
| static, divisible (8 / 4) | 0 | 0 | 0 |
| static, indivisible (10 / 4) | 0 | 0 | 1 |
| dynamic length | 0 | 1 | 1 |

`use_fmod` therefore does not reach `MASK` at all.  It remains necessary for
`PEEL` and `STRICT`, which do synthesize a remainder.

## Open items

Two of the three questions this plan opened are settled; the reasoning is
recorded so a phase does not relitigate them.

**The guard is an `If1Stmt`, not a clamped iterable.**  `range(b, min(b + f,
n))` would avoid the predicate, but it gives a variable-width tail chunk —
which is what `PEEL` already provides.  A `MASK` that clamps is `PEEL` with the
residual folded in, and loses the one property it exists for.  The constant
width is the feature, not a preference.

**The ceiling is spelled with `fmod`**, for consistency with `_build_peel`; see
*Where it goes*.

### Where does `split` run relative to the Triton normal form?

The one genuinely open question, and it is a *pipeline order* decision rather
than anything in this transform.

A masked body is a guarded `IndexedAssign`.  `SimplifyIf` refuses to hoist a
list write — correctly, since hoisting it would make the out-of-range store
unconditional — so a masked loop does not reduce to an if-expression.  A
consumer that normalizes *after* splitting would therefore reject its own
output.

The answer is to split after normalizing, not to carve out an exemption.  The
normal form's job is to get the *source* program to expression form; splitting
introduces loop structure for the emitter to consume, which is a lowering step.
Run it second and the normal form never sees the mask.  It is also the better
order independently: splitting first would split loops inside callees that are
about to be inlined away.

This binds the consumer, not this transform, so nothing in the phases below
depends on it.  Reopen if a consumer appears that must re-normalize after
lowering, which would be the case for encoding "this guard is a mask" in the
AST rather than in the pipeline order.
