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
| `x` is bound exactly once in the body | `def_use.name_to_defs`, counting defs sited inside the body |
| `x` is not read from outside the body | `def_use.uses` — see below; **added in Phase 2** |

Motion is sound under *any* rounding context — it relocates an expression
rather than re-associating one — but only to a destination that rounds the same
way.  Restricting to direct children of the body settles that outright rather
than merely cheaply: neither `ForStmt` nor `WhileStmt` is a `ContextScopeSite`
(only `FuncDef` and `ContextStmt` are), so a direct child of the body is
*already* in the scope the loop statement sits in, which is the scope it would
move to.  No context query is needed at all — Phase 2 dropped the planned
`ContextUse` check.  A statement under a `with` in the body is excluded by the
direct-child rule, which is the case that check existed for.

**The last row is new.**  The plan's zero-trip decision covered the hoisted
expression being *evaluated* where it previously was not.  It missed the mirror
case: if `x` is read after the loop and the loop runs zero times, the reader saw
whatever reached the loop, and after hoisting it sees the invariant value
instead.

```python
c = 0.0
for x in xs:
    c = n + 1          # hoisting this changes `return acc + c` on `xs = []`
    acc = acc + c * x
return acc + c
```

That is a wrong answer, not a permitted extra evaluation, so it is refused
rather than waved through under the undefined-behaviour argument.

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
- **It stays out of `Simplify`.**  Briefly planned to join it, by analogy with
  `UnnestContext` in #306, and reversed after Phase 4: the passes `Simplify`
  runs shrink or reformat a program, and this one *relocates computation*.
  That is an optimization, not a simplification, and nothing downstream depends
  on it having happened.  The tell was the termination measure — `Simplify`
  argues termination from **(statement count, nested `with` ancestor pairs)**,
  and this pass decreases neither, so admitting it would have meant inventing a
  third component for a pass that did not belong.  A schedule that wants the
  motion asks for it.

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
candidate and takes no index.  No `apply_with_status`: in this tree that method
exists for `Simplify`'s fixpoint, which this pass does not join, and `sites`
already answers whether anything is left to hoist.

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
Phase 3 — keeping it alone makes that flip a reviewable diff rather than noise
inside the phase that causes it.

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

### Phase 2 — the invariance query — **Done.**

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

18 passed; `ruff` and `mypy` clean on the new module.  What the phase decided:

- **The context check is unnecessary**, for the reason now recorded in the
  design above.  One fewer analysis to thread through.
- **A condition was missing**: the target must not be read from outside the
  body.  Also recorded above.
- **The query is one round.**  Where an invariant statement reads another, the
  second stays behind until the first has moved — `b = a * 2` after
  `a = n + 1` yielded `['a']`, not `['a', 'b']`.  **Superseded in Phase 3**:
  the query now takes body statements in order, each hoisted one counting as
  invariant for those after it, so a chain comes out in a single pass and no
  fixpoint is needed inside the transform.

Each refusal was checked to fire for the reason it claims rather than
vacuously, by instrumenting the predicate and printing which condition rejected
each body statement.

### Phase 3 — the transform — **Done.**

- `_HoistInvariant(SiteRewriter)` and `HoistInvariant` in the same module, with
  `apply` and `apply_with_edits`; export from
  `fpy2/transform/__init__.py`.
- Flip the Phase 1 assertions that this pass owns: `_k` now sits above the loop.
- Shape tests for each refusal, and **differential tests** in the house style —
  interpret before and after across inputs chosen to expose a wrong hoist,
  including an empty list (zero-trip loop) and a single-element list.

```bash
python3 -m pytest tests/unit/transform/test_hoist_invariant.py -q
```

38 passed; `tests/unit/transform` green at 1184; `ruff` and `mypy` clean across
the package.  What the phase found:

- **`to_anf` is not needed, and should not have been in the plan.**  It got
  there because the target form was written with `2 ** -_k` bound to a name,
  which only a statement-level pass could then move.  But `hoist_scale` does
  not need its factor pre-named: its condition is that the factor's free
  variables are bound outside the loop, and hoisting `_k` alone establishes
  that.  Phase 6's ANF case is dropped.  The chain rule that ANF was showing
  off is real and stays, tested directly on `a = n + 1; b = a * 2`.
- **`where=None` must not turn a refusal into a decline.**  `_selects(block,
  pos, -1)` answers `True` whenever `where is None`, since that means "every
  site", so the refusal branch has to test `self._target is not None` first.
  Without the guard the pass raised `TransformDeclined` on the motivating
  schedule, whose first two loops have nothing to hoist.  See the open item
  below: `SimplifyIf` has the same bare call.
- **Phase 1's flip annotations were wrong** on three tests.  Those assertions
  pin the *source* program and stay true; it is `TestTheTransform` that asserts
  the after-side.  Corrected in place.  The `TestSimplify` pair became
  permanent when the `Simplify` integration was dropped: they now assert that
  `simplify` leaves an invariant binding alone, by design.
- **The impure fixture cannot be interpreted** — the interpreter refuses to call
  a foreign Python function — so it is excluded from the differential sweeps
  and checked by shape only.

### Phase 4 — the scheduling primitive — **Done.**

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

46 passed there, and the full unit suite at 4752; `ruff` and `mypy` clean.  Two
things the phase found:

- **The `where` contract has a coverage guard**, and it is worth knowing about:
  `test_every_aimable_strategy_is_covered` in
  `tests/unit/strategies/test_where_contract.py` fails when a strategy joins
  `_SITES` without a row in its table.  So registering in
  `fpy2/strategies/sites.py` also means adding acting, nested and refusing
  rows there — `_two_invariant_for`, `_nested_invariant_for`, and the existing
  `_two_for`, which this pass refuses outright.
- **Rebuilding the loop body broke cursors, and the nested row caught it.**  The
  first cut replaced the body with a freshly built `StmtBlock`.  A block this
  pass synthesized is in no entry of `_paths`, so `_selects_at` answers `False`
  for everything inside it — and a cursor aimed at an outer loop therefore
  stopped reaching a loop nested in it, though the contract says a cursor takes
  every candidate at or beneath it.  The fix is to leave the body to the normal
  walk and mark the hoisted statements, letting `_visit_block`'s `_dropped`
  path leave them out and record the removal.  Shorter, and the block keeps its
  path.

### Phase 5 — end to end — **Done.**

- Integration test: the full `fused_sum` schedule, asserting `_k` above the loop
  and the interpreted results unchanged across the `_VALUES`-style sweep.
- Assert the handoff shape directly: `_k` bound above the loop, so that
  `2 ** _k`'s only free variable is outside it — which is exactly
  `hoist_scale`'s precondition in [algebraic-rewrites.md](algebraic-rewrites.md).
- Mark PR 1 **Done.** in [algebraic-rewrites.md](algebraic-rewrites.md) and
  record whether `HoistScale`'s "free variables defined outside the loop"
  condition now holds on the example — that is the handoff to PR 2.

```bash
python3 -m pytest tests/unit/transform/test_hoist_invariant.py -q
```

41 passed.  The handoff assertion is the one that matters: it resolves each of
the factor's free names to its reaching definition and checks none is sited at
or inside the loop — `False` before the hoist, `True` after.  PR 2 inherits
that rather than re-deriving it.

`_scale_factor` has to follow a name to find the product: the write reads
`ts[i] = t`, not `ts[i] = c * e`, because `rescale_fixed` binds the scaled value
first.  `DefineUseAnalysis.defining_expr` exists for exactly that, and
`HoistScale` will need it for the same reason — its matcher cannot key on the
`IndexedAssign`'s expression alone.

### After the last phase

```bash
make lint
python3 -m pytest tests/unit -n 8
python3 -m tests.infra
python3 -m tests.infra.fpcore
python3 -m tests.infra.backend.cpp --mode run
```

## Open items

### Does `SimplifyIf` decline when it should not?

Found while fixing the same shape in this pass, not introduced by it.
`_SimplifyIfInstance._claims` (`fpy2/transform/simplify_if.py:189`) tests
`self._selects(block, pos, -1)` without first checking that a cursor was given,
and `_selects` answers `True` for `where=None`.  So:

```python
@fp.fpy(ctx=fp.REAL)
def f(x: fp.Real) -> fp.Real:
    if x > 0:
        return x
    else:
        return -x
```

`SimplifyIf.apply(f.ast)` raises `TransformDeclined: a `return` escapes the
branch and has no expression form`, where `docs/source/strategies.rst` says
"``where=None`` rewrites every site the strategy can and skips the rest".

**Provisional call:** leave it.  It is a different pass, the fix is one
conjunct, but it needs its own regression test and may move goldens — neither
belongs in this PR.  Worth its own small change; reopen if a schedule here
starts composing `simplify_if`.

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
