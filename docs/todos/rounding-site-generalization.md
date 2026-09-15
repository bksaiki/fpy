# Generalizing the rounding rewrites beyond the `with`-block shape

## Goal

Five strategies — `unfold_special`, `unfold_neg_zero`, `unfold_overflow`,
`float_to_fixed`, `rescale_fixed` — apply only to a literal block of the form

```python
with C:
    y = fp.round(x)
```

They should apply wherever **the active rounding context is `C`** (by
context-use analysis) and **a `fp.round(...)` sits in a hoistable position**.
A `with` left holding no rounding is dead-code-eliminated, not deleted by the
rewrite itself.

## Where we are

All five are `BlockRewriter` subclasses whose `_candidate` delegates verbatim
to `rounding_block` (`fpy2/transform/utils.py:270`), a purely syntactic
predicate with five conjuncts:

1. the statement is a `ContextStmt` — the active context is never consulted;
2. `with C:` and not `with C as c:`;
3. every body statement is an unannotated `Assign` to a `NamedId`, or a `ReturnStmt`;
4. that statement's *whole* expression is `Round(arg=Var)` (or `Cast(arg=Var)`
   where `_casts`) — the `Var` is a convenience, not a soundness constraint;
   see "Binding the operand" below;
5. *all* of the body — one non-round statement disqualifies the block.

Measured site counts at `8afbdafa`:

```
shape                             special  neg_zero  overflow  f2f  rescale
with C: y = fp.round(x)               1        1        1       1      1
with C: return fp.round(x)            1        1        1       1      1
@fpy(ctx=C) ... y = fp.round(x)       0        0        0       0      0
@fpy(ctx=C) ... return fp.round(x)    0        0        0       0      0
with C: y = fp.round(x); z = y + 1    0        0        0       0      0
with C: y = fp.round(x + 1)           0        0        0       0      0   *
with C as c: y = fp.round(x)          0        0        0       0      0
with C: y: fp.Real = fp.round(x)      0        0        0       0      0
```

A returned round *is* rewritten today, but only inside the block form.
`*` — this row is a site under the target design; see "Binding the operand".

Two consequences:

- **No diagnostic.** `refusals()` returns `[]` for every zero row above. A
  non-candidate is neither a site nor a refusal, so the strategy is silently
  inapplicable.
- **A workaround already exists.** `fpy2/backend/cpp/unfold_round.py:331`
  carries `_isolate`, whose whole job is to re-synthesize the `with` shape that
  `Specialize` folded into the function annotation, and `_isolatable`
  (line 292) is an open-coded copy of conjuncts 3–4.

The machinery to do this right is already in the same file the five ignore:
`RoundingScopes` (`utils.py:317`) answers "what context is active here", and
`RoundingRewriter` (`utils.py:636`) is expression-sited and seals
non-hoistable positions via `PreambleScoped`. `insert_round` and `split_round`
are built on it.

## Target shape

A site is a `Round` (or `Cast`) expression, not a statement. For

```python
with C:
    y = fp.round(x)
```

the rewrite emits its ladder into the statement slot *inside* the block and
leaves `y = <temp>` behind. The emitted ladder re-scopes to `REAL` and sets its
own rounding context, so nesting is semantics-neutral; the outer `with C:` then
has an empty use set and `DeadCodeEliminate._dead_contexts` drops it. Verified:
`simplify` removes both `with fp.FP16: y = 1.0` and a `with C:` holding only a
`with fp.REAL:` ladder.

## Binding the operand

The `arg=Var` conjunct is not about soundness. An operand of a round is
evaluated under the *same* active context as the round, so binding it to a name
in that scope is an identity — the rounding it picks up was already in the
source. Measured over 81 values under a 4-bit `MPSFloatContext`:

```python
with C: y = fp.round(x + 1)           # nested
with C: t = x + 1; y = fp.round(t)    # bound in scope    -> 0 mismatches
with REAL: t = x + 1                  # bound under REAL
with C: y = fp.round(t)               #                   -> 15 mismatches
```

`RoundingRewriter._emit` (`utils.py:688`) already performs the first bind for
`insert_round` and `split_round`: "Each operand that survives the visit as a
non-`Var` is bound under the *original* scope first."

The REAL-bound variant is why the bind is **mandatory** rather than merely
permitted. Every one of the five references the operand from inside a
`with fp.REAL:` wrapper — `rescale_fixed._rescale_round` puts it in
`Mul(shift.up(), e.arg)` inside one, and `unfold_special._unfold` calls `arg()`
several times inside one (the zero test, each special's test and sign choice,
and the surviving round). Inlining a non-`Var` operand there would evaluate it exactly. Binding it
outside the ladder, under the active context, is what makes the ladder's
repeated references safe — which is the reason `rounding_block` wanted a `Var`
to begin with.

So the gate does not refuse a non-`Var` operand; it binds it, and the five
`_unfold` / `_rescale_round` bodies keep their `Var`-only assumption untouched.

## Working policy

Pause after each phase for review. Do not commit. Run only the tests named in
the phase; the full suite (`python -m pytest tests/unit -q -n 8`,
`python -m tests.infra`, `python -m tests.infra.backend.cpp`) runs once at the
end.

---

## Phase 1 — the shared gate, and `unfold_overflow` as proof

**`fpy2/transform/utils.py`**

- Make `RoundingScopes.format_info` lazy (a cached property). The five need
  `ctx_use` only; `FormatInfer` is dead weight for them.
- Add `RoundingScopes.scope_site(e)` returning the `ContextScopeSite` that
  introduced `e`'s scope, and `scope_ctx_expr(e)` returning the introducing
  `ContextStmt`'s context expression, or `None` for a function-annotation
  scope. Phases 3 and 4 need it to rebuild a context syntactically.
- Add `ScopedRoundingRewriter(RoundingRewriter)`:
  - `_casts: bool` class attribute, as on the five today.
  - `_candidate(e)`: `e` is a `Round` (or `Cast` when `_casts`) and
    `scope_ctx(e)` is a concrete `Context` that is not `REAL`. A symbolic or
    exact scope is not a candidate, so it consumes no index — same rule
    `insert_round` uses for an already-rounding scope.
  - `_arg_name(e, out) -> NamedId`: `e.arg.name` where the operand is already
    a `Var`, else a fresh temp bound by an `Assign` appended to `out` — a bare
    statement in the current block, so it lands under the active context and
    *not* inside the ladder's `with fp.REAL:`. See "Binding the operand".
  - classify the *original* `e.arg` node, not the temp: `ValueClassAnalysis`
    is keyed by expression identity and a fresh temp has no entry, so the
    passes that ask (`unfold_special`, `unfold_overflow`, `float_to_fixed`)
    must classify before binding. A miss degrades to `_TOP`, which emits every
    branch — correct, just not minimal.
  - `_emit(e, out)` overridden: the subclass appends its ladder to `out` and
    returns the replacement expression, instead of `RoundingRewriter`'s
    lift-into-a-block-and-`_wrap`.
- Add the **direct-target** path: where the enclosing statement is
  `Assign(target=NamedId, type=None)` whose expression *is* the site, the
  emitter writes the ladder's result into that target and the statement is
  dropped rather than rebuilt as `y = _t`. This needs a "statement dropped"
  signal out of `_visit_assign` into `SiteRewriter._visit_block`. It is what
  keeps the emitted output — and the five docstring examples — identical to
  today's modulo the vestigial `with`. If it turns out to tangle the site
  bookkeeping, fall back to emitting a temp and letting copy-propagation
  collapse it, and accept the churn in Phase 6.

**`fpy2/transform/unfold_overflow.py`** — migrate first: it builds its emitted
context fresh from the `fpy2` alias (`_unbounded_expr`, line 391) and reads
`stmt.ctx` only for its value, so it needs nothing from Phase 1's
`scope_ctx_expr`.

- `_UnfoldOverflowInstance(ScopedRoundingRewriter)`; `_verify` takes an `Expr`
  and asks `self.scopes.scope_ctx(e)` instead of
  `self.eval_info.by_expr.get(stmt.ctx)`.
- `_rewrite`'s loop over `stmt.body.stmts` goes away; `_unfold` is called once
  per site. Keep `eval_info` on the constructor for `Gensym` names, so the
  `apply` / `apply_with_edits` signatures do not change.

**Tests**

```
python -m pytest tests/unit/transform/test_unfold_overflow.py \
                 tests/unit/strategies/test_unfold_overflow.py \
                 tests/unit/strategies/test_where_contract.py -q
```

Expect churn in: cursor kind (`StmtCursor` → `ExprCursor`) wherever a test
aims `unfold_overflow`; site counts for programs that were previously
invisible; `refusals()` rows that were previously empty. The `is_equiv`
negative checks ("must not touch") should hold unchanged — if one flips, the
gate is too permissive and that is the signal to look at.

---

## Phase 2 — `float_to_fixed`

Same migration, same reason it is easy: `_ctx_call`
(`float_to_fixed.py:302`) constructs its per-value context from the alias and
never reads `stmt.ctx`.

```
python -m pytest tests/unit/transform/test_float_to_fixed.py \
                 tests/unit/strategies/test_float_to_fixed.py \
                 tests/unit/strategies/test_where_contract.py -q
```

---

## Phase 3 — `unfold_special` and `unfold_neg_zero`

One commit: both rebuild the *shed* context by editing the source constructor
call — `_ctx_expr` at `unfold_special.py:282` and `unfold_neg_zero.py:158`.

- Feed `_ctx_expr` from `scopes.scope_ctx_expr(e)` instead of `stmt.ctx`.
- Both already fall back to `ForeignVal(src.dropped)` for a non-`Call`, which
  is exactly what a function-annotation scope (no context expression) needs.
  No new failure mode, but assert it: a `@fpy(ctx=C)` program should now
  rewrite and emit a `ForeignVal` context.

```
python -m pytest tests/unit/transform/test_unfold_special.py \
                 tests/unit/transform/test_unfold_neg_zero.py \
                 tests/unit/strategies/test_unfold_special.py \
                 tests/unit/strategies/test_unfold_neg_zero.py \
                 tests/unit/strategies/test_where_contract.py -q
```

---

## Phase 4 — `rescale_fixed`

The only genuine blocker. `_symbolic_shift` (`rescale_fixed.py:351`) takes
`stmt.ctx` as a `Call` and rewrites the constructor's position argument — that
is how a run-time-known scale is supported. With no syntactic context there is
no call to shift.

- Drive it from `scopes.scope_ctx_expr(e)`. A `with`-introduced scope still
  yields the `Call`. A function-annotation scope yields `None`, and must
  decline with the same message a non-`Call` context gets today, rather than
  crash. **This is a deliberate coverage gap, not an oversight**: a run-time
  scale written as a function annotation cannot be shifted symbolically.
- `_rescale_round` builds `Mul(shift.up(), e.arg)` *inside* a `with fp.REAL:`
  block. Phase 1's `_arg_name` must have bound the operand before this runs —
  this is the pass where inlining a non-`Var` operand is observably wrong.
- `_rewrite` currently wraps `[up, round_, down]` for *every* round in the
  block in one `ContextStmt(stmt.target, shift.ctx())`. Per-site, emit that
  wrapper around the one round's three statements, with an `UnderscoreId`
  target — `rounding_block` already guaranteed the target was underscore, so
  the single-round shape is unchanged.

```
python -m pytest tests/unit/transform/test_rescale_fixed.py \
                 tests/unit/strategies/test_rescale_fixed.py \
                 tests/unit/transform/test_dead_code.py \
                 tests/unit/strategies/test_where_contract.py -q
```

---

## Phase 5 — delete the cpp workaround

With the gate generalized, `_isolate` has nothing to do.

- Remove `_isolatable`, `_Isolate`, `_isolate` from
  `fpy2/backend/cpp/unfold_round.py` (~60 lines) and the now-unused imports
  (`ContextStmt`, `UnderscoreId`, `ForeignVal`, `StmtBlock`, `Stmt`,
  `resolve_stmt`) — all six are used only by the deleted code.
- `_unfold_roundings` calls the four passes directly. The `todo` list is still
  needed for the early return; `_isolate`'s "only the sites are wrapped, which
  is what makes running the ladder over the whole program safe" argument is
  now carried by each pass's own `_verify`, so check that the ladder over a
  whole program still declines the roundings the emitter already spells.

```
python -m pytest tests/unit/backend/cpp/ -q
```

`test_lowered_roundtrip.py` is the bit-exactness pin and is the one that
matters here.

---

## Phase 6 — docs and the site contract

- Replace the "Only blocks whose body is entirely ``x = fp.round(v)`` … are
  rewritten" paragraph in all five strategy docstrings
  (`fpy2/strategies/{special_unfold,neg_zero_unfold,overflow_unfold,float_lower,fixed_rescale}.py`)
  with the active-context / hoistable-position statement, and re-generate each
  worked example against the real output.
- `fpy2/strategies/sites.py`: the module docstring and `sites()` both say the
  rounding rewrites are aimed with a `StmtCursor`. They are expression-sited now.
- `fpy2/strategies/__init__.py` module docstring: same.
- `docs/todos/native-lowering-roadmap.md` §2 states the recipe as
  `monomorphize → unfold_special → unfold_overflow → float_to_fixed →
  rescale_fixed → simplify` and never mentions `_isolate`, so the sequence is
  still accurate. Add one line recording that the passes now find their sites
  by active context, which is why the backend no longer synthesizes blocks.
- Note in `docs/todos/rounding-operator-basis.md` (or here) that `RoundAt` is
  still matched by none of the five — `fp.round_at` is a real operator the cpp
  emitter already refuses, and generalizing the gate does not change that.

```
python -m pytest tests/unit/strategies/ -q
```

---

## Risks

- **Emitted-output churn.** Every structural test that counts `ContextStmt`s
  sees one more (the vestigial `with`) until DCE runs. `test_rescale_fixed.py`
  has 14 such assertions and is the worst case. The `is_equiv` negative checks
  are unaffected.
- **Termination is not a new risk.** The blocks these passes emit are already
  in the block form and are already candidates today, so convergence already
  rests on `_verify` declining the shed / unbounded / position-zero context,
  not on the block wrapper.
- **Operand binding widens the sites more than the context change does.**
  Accepting a non-`Var` operand means `fp.round(<any expression>)` is a site,
  which is a larger jump in coverage than dropping the `with` requirement.
  The `is_equiv` negative checks across the five transform test files are the
  net that catches an operand bound into the wrong scope; a flip there means
  `_arg_name` emitted inside the ladder instead of before it.
- **Wider blast radius per pass.** A program that previously had zero sites now
  has some. That is the point, but it means `apply(where=None)` over an
  existing pipeline can rewrite more than it used to — the cpp ladder in
  particular. Phase 5's whole-program check is where that shows up.
