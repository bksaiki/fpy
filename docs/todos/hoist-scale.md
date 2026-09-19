# Hoist-scale: pulling an invariant factor out of a reduction

Implementation plan.  The design is settled; what follows is the phase
breakdown, one phase per commit.  This is PR 2 of
[algebraic-rewrites.md](algebraic-rewrites.md); PR 1 landed as `hoist_invariant`
in #307, whose plan doc was pruned on merge — as this one will be.

## Working policy

- **Pause after each phase for review.**  Do not begin the next phase until the
  current one has been looked at.
- **Do not commit.**  The working tree is left dirty; commits are made by the
  repository owner.
- **Run only the tests relevant to the phase.**  The full unit suite runs once,
  at the end, after the last phase.
- **Comments stay succinct**, and notes about *process* — what was tried, what a
  phase decided, why an ordering was chosen — belong in this document, not in
  source comments.

## Context

After `fuse; comp_to_loop; rescale_fixed; simplify; hoist_invariant`, the
motivating schedule leaves each element scaled on its way into the result list:

```python
        e = max(t7)
        ts = fp.empty(len(xs))
        _k = ((e - 12) + 1)
        t = (2 ** -_k)
        t14 = (2 ** _k)
        for t10 in range(len(xs)):
            x = xs[t10]
            _t = (t * x)
            with fp.MPFixedContext(-1, rm=fp.RM.RTZ, enable_neg_zero=False):
                _t13 = fp.round(_t)
            t12 = (t14 * _t13)          # the factor, per element
            ts[t10] = t12
        return sum(ts)
```

`t14` is loop-invariant, so `sum(t14 * _t13ᵢ)` is `t14 * sum(_t13ᵢ)` — and the
`_t13` are *integers*, since `MPFixedContext(-1)` is position zero.  Pulling the
factor out turns the reduction into an integer accumulation scaled once at the
end, which is what the backend wants.

Nothing does this.  `fpy2.rewrite` is syntactic and unverified; `ConstFold` is
partial evaluation, so a symbolic `2 ** _k` is inert.

## The five conditions, and where each one comes from

The rewrite is `sum(c * eᵢ) -> c * sum(eᵢ)`.  It is wrong without all five.

| # | condition | discharged by |
|---|---|---|
| 1 | the reduction's scope rounds exactly | `RewriteUtils.is_exact` |
| 2 | every name `c` reads is bound outside the loop | `hoist_invariant`, #307 |
| 3 | `c` is not NaN or an infinity | `value_class`, after Phase 2 |
| 4 | `c` is not negative | a syntactic check in the pass |
| 5 | the loop writes every element of the list | `ArraySize`, after Phase 3 |

Each has a counterexample.  For 1, under `fp.FP32` the partial sums round, so
`xs = [1e20, 1.0, -1e20]`, `c = 3` goes from `0` to `6.01226e12`.  For 3,
`c = inf` and `xs = [1.0, -1.0, 1.0]` goes from `NaN` to `+inf` — a cancellation
becomes a survivor.  For 4, `c < 0` and `xs = []` goes from `+0.0` to `-0.0`.
For 5, an unwritten element holds whatever `fp.empty` left there, and the
rewrite scales that too:

```
before:  sum(c*eᵢ for written) + sum(unwritten)
after:   c * (sum(eᵢ) + sum(unwritten))
```

Condition 1 is a **per-site** question: `fused_sum` holds one reduction under
the `ctx=fp.REAL` annotation and another under `fp.FP32`, and the pass must
take the first and refuse the second.  `scope_ctx` answering `None` — a
symbolic scope — declines rather than falling through.

## What the spike settled

Three routes were on the table for conditions 3 and 4.  Measured, not argued:

**Format bounds are dead by construction.**  Under `ctx=fp.REAL` every
expression's format is `RealFormat`, which carries no bounds, so `exact_exp2`
never fires — and the reduction being under an exact scope is condition 1.  The
context that makes the rewrite legal is the one that makes format bounds
vacuous.  No amount of work fixes this.

**Finiteness is a small sharpening of an acknowledged imprecision.**
`value_class` already has a rule for a positive literal base, and its docstring
names what it gives up: *"The literal is not inspected, so all three stand."*
For base 2 the two infinities are not alike — `2 ** +inf` is an infinity,
`2 ** -inf` is zero:

| | today | base literal inspected |
|---|---|---|
| `2 ** _k` | `POS_INF\|ZERO\|FINITE` | **`ZERO\|FINITE`** |
| `2 ** -_k` | `POS_INF\|ZERO\|FINITE` | `POS_INF\|FINITE` |

`2 ** _k` — the reduction's factor — comes out finite, which is condition 3.
`2 ** -_k` stays possibly-infinite, correctly; that is the in-loop scaling
factor and this rewrite does not touch it.

**A dead end worth recording.**  `2 ** ±inf` raises `NotImplementedError` under
`REAL` while being well-defined under `FP64` (`inf`, `0`, `nan`).  That is the
generic dispatch fallback every operation in `fpy2/ops.py` carries — an
implementation gap, not a rule — so no soundness argument may rest on it.  The
sharpening needs none: mapping `NEG_INF -> ZERO` is right if the operation is
ever implemented, and an over-approximation if it stays a hole, since an
execution with no result contributes no class.

**Sign stays syntactic.**  `ZERO|FINITE` still does not say *non-negative*:
`FINITE` is sign-blind by design.  Splitting it would touch `_exact_add`,
`_exact_sub`, `_exact_mul`, `_exact_select`, `_negate`, `_magnitude`,
`representable_classes` and the branch-refinement machinery — its own PR, and
not one this needs.  The pass instead requires the factor to be a power with a
positive literal base, which is what `rescale_fixed` emits.  Sound: a positive
base to any finite power is positive, `±inf` gives `+inf` or `+0` (never
`-0.0`), and NaN is already excluded by condition 3.  Incomplete, not unsound —
it declines a correct factor of another shape, and the lattice split remains
available later.

**Array sizes already carry symbols; one producer drops them.**
`ArraySize: TypeAlias = 'int | NamedId | None'`, and an unannotated list
parameter gets a fresh symbol registered in a union-find, so `xs` is
`ListSize(size=NamedId('n'))` with no annotation anywhere.  But the `Empty`
rule asks `_const_int`, which answers only with concrete integers, so
`ts = fp.empty(len(xs))` loses `n`:

| | `xs` | `ts` | `is_size_eq` |
|---|---|---|---|
| today | `NamedId('n')` | `[None, None, None]` | `False` |
| `Empty` accepts a symbol | `NamedId('n')` | `[NamedId('n')×3]` | **`True`** |

`_const_int` cannot simply widen: its other callers do integer arithmetic on
the result, and returning a `NamedId` dies in `_visit_unaryop` on `max(0, n)`.
The upgrade is a sibling query that may answer with a symbol, used by the
allocation rule.

## What the pass matches

- The reduction is `sum(name)` over a list the loop built.
- The element write is `ts[i] = t`, and the product is reached through
  `DefineUseAnalysis.defining_expr` — `rescale_fixed` binds the scaled value to
  a name first, so the write does not read `ts[i] = c * e` syntactically.
- The write is a **direct child** of the loop body, so that a matching trip
  count really does mean every element is written.
- Statements may sit between the loop and the reduction as long as they do not
  touch `ts`, and `ts` has no use other than the reduction (`live_vars`).
- `where` aims at reductions.  A reduction the pass refuses is not a site and
  takes no index; naming it with a cursor says which condition failed.

`max` / `min` are out of scope: a monotone selection needs its own proof around
NaN, signed zero and a non-positive factor, and the roadmap already records it
as a follow-up.

## Phases

Six.  The two analysis upgrades ship inside this PR but get commits of their
own, because each can move output elsewhere and needs its own suites.

### Phase 1 — regression net — **Done.**

Pin what the tree does today, before anything moves.

- New `tests/unit/transform/test_hoist_scale.py`: on the scheduled `fused_sum`,
  the element write is a product whose factor is `2 ** _k`; `value_class` puts
  that factor at `POS_INF|ZERO|FINITE`; `ArraySize` gives `ts` no size and
  `is_size_eq(xs, ts)` is `False`.
- Helpers matching `tests/unit/transform/test_hoist_invariant.py`: a `_text` formatter, a
  `_scheduled()` returning the before/after pair, and an `_agrees_by_value`
  differential check.

Separate because these are the assertions Phases 2 and 3 flip, and a flip is
only reviewable if it is not buried in the change that causes it.

```bash
python3 -m pytest tests/unit/transform/test_hoist_scale.py -q
```

9 passed.  One thing the phase found, and it changes Phase 4:

- **#307 moved the factor behind a name.**  `hoist_invariant` lifts `2 ** _k`
  above the loop, so the element write reads `t12 = (t14 * _t13)` and the
  factor arrives as `Var(t14)`, not as a syntactic `Pow`.  The sign check
  therefore needs `defining_expr` *too* — one indirection to reach the product
  from `ts[i] = t12`, and a second to reach the power from `t14`.  The plan
  described only the first.  `value_class` is unaffected: a `Var` takes its
  definition's class, so the Phase 2 sharpening still reaches it.

The `_scheduled()` helper returns one program rather than a before/after pair,
since there is no "after" until Phase 4; `_scale_factor`, `_text`,
`_agrees_by_value` and the `fused_sum` fixture are imported from
`test_hoist_invariant.py` rather than copied.

### Phase 2 — `value_class` inspects the base literal — **Done.**

- `_POW_POS_BASE` splits by the base: greater than one, equal to one, less than
  one.  `_visit_binaryop`'s `Pow` case picks the table from the literal.
- Flip the Phase 1 class assertion.
- Tests in `tests/unit/analysis/test_value_class.py` for all three bases at
  `NAN`, `POS_INF`, `NEG_INF`, `ZERO`, `FINITE`.

**Not done until `tests/unit`, `tests.infra` and the cpp corpus are green.**
Tighter classes let consumers *drop* guards — `round_elim` and the cpp backend
both do — so this can change emitted code rather than merely accept more.  Any
diff there is explained in this document before the phase closes.

```bash
python3 -m pytest tests/unit/analysis/test_value_class.py \
    tests/unit/transform/test_hoist_scale.py -q
```

108 passed.  Full suites green and unchanged: unit 4779, `tests.infra` exit 0,
cpp corpus exit 0 at 130/136 bit-compared — the same coverage as before.  So
the open item below resolves: no consumer's output moved.

`_POW_POS_BASE` became three tables — `_POW_BIG_BASE`, `_POW_SMALL_BASE`,
`_POW_ONE_BASE` — selected by `_pow_table(base)`.  Two things the phase found:

- **`1 ** nan` is `1.0`, not NaN.**  The first cut carried `NAN -> NAN` into the
  base-one table by analogy with the other two.  The existing sweep against the
  interpreter caught it: IEEE 754 has `pow(1, y) = 1` for every `y`.  The table
  is now `dict.fromkeys(_ATOMS, _FINITE)`.
- **The sweep does the verifying.**  `test_pow_at_a_concrete_context` is
  parametrized over the three bases and checks each against the interpreter at
  `NAN`, `POS_INF` and `NEG_INF` — the rows `REAL` cannot reach, and exactly
  the rows the base literal tells apart (`2 ** -inf` is `+0` where
  `0.5 ** -inf` is `+inf`).

### Phase 3 — `array_size` carries a symbolic dimension

- A `_size_of` sibling to `_const_int` in `fpy2/analysis/array_size.py`, which
  may answer with a `NamedId` from the union-find; the `Empty` rule uses it.
  `_const_int` is left alone, since its other callers do arithmetic.
- Flip the Phase 1 size assertion.
- Tests in `tests/unit/analysis/test_array_size.py`: `fp.empty(len(xs))`
  unifies with `xs` unannotated, with `xs` annotated, and across two parameters
  of the same symbolic length.

Separate from Phase 2 for the same reason and with the same bar: the cpp
backend's storage selection reads sizes.

```bash
python3 -m pytest tests/unit/analysis/test_array_size.py \
    tests/unit/transform/test_hoist_scale.py -q
```

### Phase 4 — the transform

- New `fpy2/transform/hoist_scale.py`: `_HoistScale(SiteRewriter)` and
  `HoistScale` with `apply`, `apply_with_edits`, `sites`, `refusals`; exported
  from `fpy2/transform/__init__.py`.
- The five conditions, each with its own refusal message.
- Shape tests per refusal, and **differential tests** in the house style —
  values straddling cancellation, overflow, `inf`, `NaN`, the empty list, and a
  reduction under `fp.FP32` that must be declined.

```bash
python3 -m pytest tests/unit/transform/test_hoist_scale.py -q
```

### Phase 5 — the scheduling primitive

- New `fpy2/strategies/scale_hoist.py` exporting `hoist_scale(func,
  where=None)`, in `fpy2/strategies/__init__.py`'s imports and `__all__`, and
  registered in `fpy2/strategies/sites.py` for both `sites` and `refusals`.
- `.. autofunction:: fpy2.strategies.hoist_scale` in
  `docs/source/strategies.rst`, alphabetical after `hoist_invariant`.
- New `tests/unit/strategies/test_hoist_scale.py`, and rows in
  `tests/unit/strategies/test_where_contract.py` — acting, nested and refusing.
  `test_every_aimable_strategy_is_covered` fails without them.

```bash
python3 -m pytest tests/unit/transform/test_hoist_scale.py \
    tests/unit/strategies/test_hoist_scale.py \
    tests/unit/strategies/test_where_contract.py -q
```

### Phase 6 — end to end

- The full schedule reaches the target form of
  [algebraic-rewrites.md](algebraic-rewrites.md): `ts[t10] = _t13` in the loop,
  `return (t14 * sum(ts))` outside, values unchanged.
- The `fp.FP32` branch of the same function is declined, asserted rather than
  assumed.
- Mark PR 2 **Done.** in the roadmap and record whether the `max` / `min`
  follow-up still looks worth doing.

```bash
python3 -m pytest tests/unit/transform/test_hoist_scale.py -q
```

### After the last phase

```bash
make lint
python3 -m pytest tests/unit -n 8
python3 -m tests.infra
python3 -m tests.infra.fpcore
python3 -m tests.infra.backend.cpp --mode run
```

## Open items

### Does the `value_class` sharpening move any consumer's output?

Unknown until Phase 2 measures it.  Tighter classes are strictly better
information, but consumers use them to drop guards, so "more precise" is not
the same as "no diff".  If the cpp corpus changes, the question becomes whether
each change is an improvement or a latent bug the old imprecision was masking.

**Resolved in Phase 2: no.**  Unit, infra and the cpp corpus are all green and
the corpus reports the same 130/136 bit-compared as before.  The sharpening
only splits rows that were already joined, so no consumer sees a class it did
not see before — it sees a narrower one, and none of them narrowed enough to
change a decision.

### Should the write be allowed inside an `if` in the loop body?

The plan requires the element write to be a direct child of the body, so a
matching trip count means every element is written.  A body that writes
`ts[i]` in both arms of an `if` also writes every element, and refusing it is
conservative.

**Provisional call:** direct child only.  The motivating schedule emits exactly
that, and admitting branches means proving both arms write the same index,
which is a second analysis for no demonstrated gain.  Reopen if a real schedule
produces the branched shape.

### Should the accumulator form be matched as well?

`reduce_fusion` turns `sum(ts)` into a loop carrying an accumulator.  Run
before this pass, it would leave a shape this rewrite does not recognise —
`acc = acc + c * e` rather than `sum(ts)`.  The same algebra applies.

**Provisional call:** `sum(name)` only, and document that `fuse` over the
*reduction* must run after this pass rather than before.  Matching both shapes
doubles the matcher for a schedule nobody has asked for.  Reopen if the cpp
path wants the fused form earlier.
