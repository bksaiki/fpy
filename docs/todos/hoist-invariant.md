# Hoist-invariant: loop-invariant code motion

Implementation plan.  The design is settled; what follows is the phase
breakdown, one phase per commit.  This is PR 1 of
[algebraic-rewrites.md](algebraic-rewrites.md); `HoistScale` is PR 2 and gets
its own plan once its analysis spike lands.

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

`rescale_fixed` emits the shifted context's scale factor inside the loop it
rewrites.  On the motivating schedule
(`comp_to_loop; rescale_fixed; simplify` over `fused_sum`):

```python
for t12 in range(len(xs)):
    x = xs[t12]
    _k = ((e - 12) + 1)                 # invariant: recomputed every iteration
    _t = ((2 ** -_k) * x)
    with fp.MPFixedContext(-1, rm=fp.RM.RTZ, enable_neg_zero=False):
        _t15 = fp.round(_t)
    ts[t12] = ((2 ** _k) * _t15)
```

`_k` depends only on `e`, which is bound before the loop.  Nothing in the tree
moves it: the only mention of loop invariance anywhere is a comment in
`fpy2/analysis/format_infer/analysis.py:2702`.  `Simplify` (const-fold,
copy-prop, DCE, unnest-context) cannot — the expression is not constant, and
copy propagation does not relocate statements.

This is a wart in *every* `rescale_fixed` output, not just this one.  It is also
what blocks `HoistScale`: that pass's condition "every free variable of the
factor is defined outside the loop" is false for `2 ** _k` while `_k` is bound
inside the body.

## Design

### What is hoistable

A statement `x = e` at the **top level of a loop body** moves above the loop
when all of:

| condition | how it is checked |
|---|---|
| simple assignment, `NamedId` target | `isinstance(stmt, Assign)`, target not a `TupleBinding` |
| `e` is pure | `Purity.analyze_expr(e, def_use)` |
| every free variable of `e` is bound outside the loop | `DefineUseAnalysis`: each use's def site is not within the loop body, and is not a loop-carried `PhiDef` |
| `x` is not rebound in the body, and is not a loop-carried phi | `def_use` — a second def of `x` in the body, or a `PhiDef` with `is_loop`, refuses |
| the destination has the same active rounding context | `ContextUse`: the scope of `e` equals the scope the loop statement sits in |

The context check is the one that is easy to get wrong.  Motion is sound under
*any* context — it relocates an expression rather than re-associating one — but
only to a destination that rounds the same way.  A body statement inside a
`with` nested in the loop must not land outside that `with`.  Restricting to the
top level of the body makes this a single equality rather than a walk, and costs
nothing on the motivating example, where `_k` is a direct child of the `for`.

### Decisions taken

- **Both loop forms.**  `ForStmt` and `WhileStmt` are candidate sites.  The
  predicate is the same for both once "defined outside the loop" is read through
  `def_use` phis rather than through the target binding — a `while` has no
  target, and its loop-carried names are exactly the `PhiDef`s with `is_loop`.
  Deferring `while` would have meant writing that phi reasoning twice.
- **Zero-trip loops are hoisted out of anyway.**  Motion makes the expression
  evaluate even when the body never runs, which per the soundness assumption
  `value_class` documents can turn a clean run into an abort.  FPy has undefined
  behaviour by design and the interpreter's runtime checks are not a contract a
  transform must preserve, so no guard is emitted.  Phase 1 and Phase 3 pin the
  empty-list case so the behaviour is visible rather than merely permitted.
- **It joins `Simplify`.**  Like `UnnestContext` in #306, this is cleanup every
  lowering-heavy pipeline wants, and leaving it opt-in would mean every
  `rescale_fixed` output keeps the wart until a user knows to ask.  It lands
  behind an `enable_hoist_invariant` flag, on by default, in the same fixpoint.

### `Simplify`'s termination measure has to grow

`fpy2/transform/simplify.py` documents its termination argument as a
lexicographic measure on **(statement count, nested `ContextStmt` ancestor
pairs)**.  Const-fold, copy-prop and DCE strictly decrease the first;
`UnnestContext` holds the first fixed and strictly decreases the second.

`HoistInvariant` fits neither: it holds the statement count fixed and does not
touch `with` nesting.  The measure becomes

**(statement count, Σ over statements of enclosing-loop depth, nested `with`
ancestor pairs)**

which `HoistInvariant` strictly decreases in the second component while holding
the first, and which the other three leave alone or decrease — removing a
statement cannot raise anyone's loop depth, and unnesting a `with` moves
statements between `with` bodies, never out of a loop.  The module docstring is
part of the change, not an afterthought.

### What is out of scope for this PR

- **Nested positions.**  Only direct children of the loop body.  A statement
  inside a `with` or `if` within the body is left alone even when invariant.
- **Expression-level motion.**  `(2 ** -_k)` is an invariant *subexpression* of
  `_t`, not a statement.  `to_anf` already names such subexpressions, so a
  schedule that wants this runs it first.  This pass moves statements.

### Shape of the pass

Follows `SplitLoop`: a `_HoistInvariant(SiteRewriter)` instance plus a
`HoistInvariant` class with `apply` / `apply_with_edits`, `check_where` on the
`where`, `SyntaxCheck.check` on the result.  Sites are loops, counted in visit
order, outermost first — so `where` aims at a loop, and the pass hoists every
qualifying statement out of it.  A loop with no qualifying statement is not a
candidate and takes no index.  `apply_with_status` is added alongside, since
`Simplify`'s fixpoint needs the `changed` flag.

## Phases

### Phase 1 — regression net — **Done.**

Pin today's behaviour before anything moves.

- New `tests/unit/transform/test_hoist_invariant.py` with current-output
  assertions only: on the `fused_sum` pipeline, `_k`'s assignment is inside the
  `for` body; on synthetic `for` and `while` loops, an invariant assignment
  stays put; `simplify` leaves all of them alone.
- Helpers matching `tests/unit/transform/test_unnest_context.py`: a `_text`
  formatter, a body-statement counter, and an `_agrees` differential check whose
  value sweep includes the empty list.

Separate because it is the only phase whose assertions are *supposed* to flip in
Phase 3 and Phase 5 — keeping it alone makes those flips a reviewable diff
rather than noise inside the phase that causes them.

```bash
python3 -m pytest tests/unit/transform/test_hoist_invariant.py -q
```

7 passed.  Two things the phase found:

- **The schedule baseline needs value equality, not `repr`.**  The house
  `_agrees` compares `repr`, which includes the result's attached context tag.
  `rescale_fixed` moves where the rounding happens, so on a one-element list the
  source's `sum` returns the rounded element still tagged `MPFixedContext` while
  the rescaled schedule's final unscaling multiply re-tags it `RealContext` —
  same `exp`, same `c`, same flags.  A second helper, `_agrees_by_value`, covers
  that case; the synthetic loops keep the strict `repr` comparison.
- **`fused_sum` has no value on the empty list**, since `max([])` does not —
  the schedule baseline sweeps `_VALUES[1:]`.  The synthetic `for` and `while`
  functions do include it, which is where the zero-trip decision gets exercised.

Not fixed, deliberately: `ruff` reports `C419` on the `all([...])` inside the
`@fp.fpy` body.  That comprehension is FPy source, not Python — it is what
`comp_to_loop` lowers — and `tests/` is outside `make ruff`'s target, which
checks `fpy2` only.  The tree already carries 37 of these.

### Phase 2 — the invariance query

- New `fpy2/transform/hoist_invariant.py` with the private helper only: given a
  `ForStmt` or `WhileStmt` and a `DefineUseAnalysis`, return the body statements
  meeting the table above.  No rewriting yet.
- Tests for the query in isolation, for both loop forms: invariant,
  depends-on-loop-target, depends-on-loop-carried-phi, rebound-in-body, impure,
  tuple target, inside a nested `with`.

Separate because the predicate is where the soundness lives and the rewrite is
mechanical.  Reviewing them together buries the conditions under AST surgery.

```bash
python3 -m pytest tests/unit/transform/test_hoist_invariant.py -q
```

### Phase 3 — the transform

- `_HoistInvariant(SiteRewriter)` and `HoistInvariant` in the same module, with
  `apply`, `apply_with_status` and `apply_with_edits`; export from
  `fpy2/transform/__init__.py`.
- Flip the Phase 1 assertions that this pass owns: `_k` now sits above the loop.
- Shape tests for each refusal, and **differential tests** in the house style —
  interpret before and after across inputs chosen to expose a wrong hoist,
  including an empty list (zero-trip loop) and a single-element list.

```bash
python3 -m pytest tests/unit/transform/test_hoist_invariant.py -q
```

### Phase 4 — the scheduling primitive

- New `fpy2/strategies/invariant_hoist.py` exporting `hoist_invariant(func,
  where=None)`, added to `fpy2/strategies/__init__.py`'s imports and `__all__`.
- Docstring with the `Examples` block the other strategies carry, and the
  `Raises` section naming `TransformDeclined` / `TransformReferenceError`.
- `.. autofunction:: fpy2.strategies.hoist_invariant` in
  `docs/source/strategies.rst`, alphabetical between `fuse` and `inline`.
- New `tests/unit/strategies/test_hoist_invariant.py` — the per-strategy file
  the sibling primitives each have: `where` as an index, as a cursor, out of
  range, and naming a refused loop.

Separate because the `where` contract is its own surface — cursor forwarding and
the two failure kinds — and because nothing before this phase is user-visible.

```bash
python3 -m pytest tests/unit/transform/test_hoist_invariant.py \
    tests/unit/strategies/test_hoist_invariant.py -q
```

### Phase 5 — into `Simplify`

- `enable_hoist_invariant: bool = True` through `Simplify.apply` and
  `apply_with_status`, and through the `fpy2.strategies.simplify` wrapper.
- Rewrite the termination paragraph in `fpy2/transform/simplify.py`'s module
  docstring for the three-component measure.
- Flip the Phase 1 assertion that `simplify` leaves invariant statements alone.
- **Repair the fallout.**  Every existing golden that runs `simplify` over a
  loop with an invariant body statement changes.  The phase is not done until
  `tests/unit/transform` and `tests/unit/strategies` are green.

Last, and separate, because it is the only phase with blast radius outside its
own test file — bundling it with Phase 3 would mix "the pass is correct" with
"the pipeline's goldens moved".

```bash
python3 -m pytest tests/unit/transform tests/unit/strategies -q
```

### Phase 6 — end to end

- Integration test: the full `fused_sum` schedule, asserting `_k` above the loop
  and the interpreted results unchanged across the `_VALUES`-style sweep.
- A second case with `to_anf` in the schedule, which names `2 ** -_k` and so
  lets this pass hoist it as well — two of the three hoists in the target form
  of [algebraic-rewrites.md](algebraic-rewrites.md) are this pass's, and the
  `to_anf` dependency is worth pinning rather than leaving to PR 2 to discover.
- Mark PR 1 **Done.** in [algebraic-rewrites.md](algebraic-rewrites.md) and
  record whether `HoistScale`'s "free variables defined outside the loop"
  condition now holds on the example — that is the handoff to PR 2.

```bash
python3 -m pytest tests/unit/transform/test_hoist_invariant.py -q
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

### What resolves `HoistScale`'s finiteness and sign condition?

Not this PR's problem, but it gates PR 2 and is the one question the discussion
did not settle.

**What is needed.**  To pull `c` out of `sum`, `c` must be finite — an infinite
`c` turns a cancellation into a survivor, `sum([c*1, c*-1, c*1])` going from
`NaN` to `+inf` — and non-negative, since a negative `c` signs a zero the
original never signed, `sum([])` going from `+0.0` to `-0.0`.

**What is available.**  Measured, `value_class` puts `2 ** _k` at
`POS_INF|ZERO|FINITE`.  That rules out `NaN` and `-inf`, but `POS_INF` survives
and `FINITE` covers either sign, so neither half of the condition discharges.
Three routes, and the spike is to try each on `fused_sum`:

- **`exact_exp2` hook.**  `format_infer` already exports `exact_exp2`.  Teach the
  pass a narrow syntactic rule: `2 ** k` is non-negative whenever it has a value
  at all, and finite whenever `k` is.  Cheapest and lands exactly on the shape
  `rescale_fixed` emits — and that is also the objection, since it generalizes to
  nothing else.
- **`format_infer` bounds.**  `AbstractFormat` carries `pos_bound` / `neg_bound`,
  so a `neg_bound` at or above zero would say non-negative and a finite bound
  would say finite.  Reuses machinery the tree already trusts.  Unverified: the
  `value_class` docstring notes a format structurally cannot say *non-zero*, and
  whether it can say *non-negative* has not been checked.
- **Context-constructor argument domains.**  The root fact is that
  `MPFixedContext(nmin=e - 12)` requires an integer `nmin`, so `e` is finite and
  the residual `POS_INF` is unreachable — it corresponds to the all-zero input,
  where `fp.logb(0) = -inf` and the source program raises.  Teaching
  `value_class` that a context constructor constrains its arguments is the
  general fix.  But `rescale_fixed` has already rewritten the context to the
  literal `MPFixedContext(-1)` by the time `HoistScale` runs, so this only helps
  if the fact is captured before rescaling — it is an ordering change as much as
  an analysis one.

**Provisional call:** none.  Pick during the spike, on evidence.  What must not
happen is `HoistScale` shipping against an analysis that cannot discharge its
side conditions, since the pass would then decline on its own motivating example.
