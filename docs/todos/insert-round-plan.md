# Implement `insert_round`

## Context

FPy's scheduling language can *remove* a rounding that provably does nothing
(`elim_round`, backed by `RoundElim`) but cannot *add* one. That is gap 1 of
`docs/todos/rounding-axes.md`: the double-rounding paper's §6.2 identity reads
in two directions, and FPy implements only the left-to-right
(canonicalization) direction.

The right-to-left direction — *finitization* — is what maps a real-valued
specification onto the operations an environment actually provides. Without
it, a program hoisted into `with fp.REAL:` can never be lowered back onto a
concrete format, so the scheduling language can only make programs more
abstract, never more implementable.

`insert_round` is the cheapest of the four roadmap operators: the verification
predicate already exists and is already load-bearing in `RoundElim`, so nothing
in the number tower changes.

### Verification, confirmed experimentally

`prod3` monomorphized to FP32 arguments under an FP64 context, then
`elim_round`:

```python
def prod3(x, y, z):
    with fp.REAL:
        _t = (x * y)
    return (_t * z)
```

Format inference already gives the `Mul` inside a REAL block its *exact*
format, since REAL does not round. `by_expr[mul]` is
`MPBFloatFormat(pmax=48, emin=-251, ...)` — the FP32⊗FP32 product. It is a
`Format`, not an `AbstractFormat`, so it must be lifted with
`AbstractFormat.from_format` before `round_is_identity` will accept it (the
same lift `RoundElim._unrounded_format` does). Then:

```
A(48, -298, 1.157920754338239e+77)
  round_is_identity(.., FP64) = True     -> rewrite
  round_is_identity(.., FP32) = False    -> decline
  round_is_identity(.., FP16) = False    -> decline
```

So `insert_round(hoisted, fp.FP64)` inverts `elim_round` here, and declines the
formats where inserting would change the result.

## Design

`insert_round(func, ctx, where=None)` — `ctx` is a **required** positional, as
in `monomorphize(func, ctx, args)`. Inferring it is not worth it: reading the
enclosing scope only reproduces `elim_round`'s own choice, and synthesizing one
from the inferred format (via `AbstractFormat.format()`) yields formats no
hardware has, defeating the purpose of finitization. Choosing from a supplied
`available=[...]` list is the roadmap's separate `finitize`.

A `BlockRewriter` subclass, matching `rescale_fixed` / `unfold_neg_zero`:

- **`_candidate`** — purely structural, and it must be: a hand-written
  `with fp.REAL:` parses to `Attribute(fp.REAL)` while `elim_round` emits
  `ForeignVal(RealContext())`, so REAL cannot be recognized syntactically.
  Match a `ContextStmt` with an `UnderscoreId` target whose body is entirely
  `Assign(NamedId, type=None)` statements with roundable-op right-hand sides
  (`Add`/`Sub`/`Mul`/`Abs`/`Neg`/`Round`/`Cast` — the same set
  `RoundElim._is_eliminable` accepts, so the two operators mirror each other).
  Add an `exact_block(stmt) -> list[Assign] | None` helper to
  `fpy2/transform/utils.py` beside `rounding_block`, plus an
  `is_exact_block(stmt)` for `sites` to share, keeping `sites` and `_candidate`
  in lockstep.

- **`_verify`** — resolve `stmt.ctx` through `eval_info.by_expr` and decline
  unless it is `REAL`; decline a body with more than one assignment, since
  rounding the first changes the second's operand and that needs sequential
  bound propagation; then look up `format_info.by_expr[assign.expr]`, lift via
  `AbstractFormat.from_format` when it is an `AbstractableFormat`, and require
  `round_is_identity(lifted, ctx)`. Also require
  `specials_contained_in`, which containment does not imply — a bound carrying
  `-0.0` reaching a format without one is a wrong answer, not an imprecise one.
  Decline `ctx.is_stochastic()` for now (open question in the roadmap).

- **`_rewrite`** — swap the context expression, keeping the body:
  `[ContextStmt(stmt.target, ForeignVal(ctx, stmt.loc), stmt.body, stmt.loc)]`.
  One statement in, one out, so `inserted == 1` and cursors forward as a
  `StmtCursor`; `exprs_preserved=True`.

Multi-statement REAL blocks are candidates that decline, not non-candidates —
matching the documented convention that "listing is syntactic: a site that
appears here may still be declined." It also keeps `where` numbering stable
when a block gains a statement.

## Phases

Each phase is one commit. **Pause after each for review. Do not commit — the
user commits.** Run only the tests named in that phase; the full suite runs at
the end.

### Phase 1 — the transform

- Copy this plan to `docs/todos/insert-round-plan.md`.
- `fpy2/transform/round_insert.py` — module docstring with the identity and a
  before/after block, `_RoundInsertInstance(BlockRewriter)`, and the
  `RoundInsert` façade with `sites` / `apply` / `apply_with_edits`
  (`apply_with_edits` takes `func, ctx, *, where=None, eval_info=None,
  format_info=None` and follows the five-step recipe: type-check,
  `check_where`, default the analyses, run, `check_site('a candidate exact
  block')`, return `EditLog`).
- `exact_block` / `is_exact_block` in `fpy2/transform/utils.py`.
- Export `RoundInsert` from `fpy2/transform/__init__.py`.
- Tests: `tests/unit/transform/test_round_insert.py` — the FP64 rewrite, the
  FP32/FP16 declines, a non-REAL block declining, a multi-statement block
  declining, `where` index and cursor aiming, `TransformReferenceError` for a
  bad `where`, idempotence via `is_equiv`, and input non-mutation.

Run: `pytest -n 8 tests/unit/transform/test_round_insert.py`

### Phase 2 — the strategy wrapper

- `fpy2/strategies/round_insert.py` — `insert_round(func, ctx, where=None)`
  with the full numpydoc docstring (Parameters incl. the standard `where`
  blurb, Returns, Raises `TransformDeclined` + `TransformReferenceError`,
  Examples showing real `func.format()` output). Body: the
  `Expected a 'Function'` type check, then
  `func.with_edits(RoundInsert.apply_with_edits(func.ast, ctx, where=func.rebase(where)))`.
- Register in `fpy2/strategies/__init__.py` (import + `__all__`),
  `_SITES` in `fpy2/strategies/sites.py`, and
  `docs/source/strategies.rst` (`.. autofunction::`).
- Tests: `tests/unit/strategies/test_insert_round.py` — wrapper behavior,
  `where` pass-through, type rejection, composition with `simplify`; and add
  `insert_round` to the enumerations in `tests/unit/strategies/test_sites.py`
  (the import list, the rounding-strategy loop, and the index/cursor
  correspondence parametrization).

Run: `pytest -n 8 tests/unit/strategies/test_insert_round.py tests/unit/strategies/test_sites.py`

### Phase 3 — round-trip against `elim_round`, and the doc

- Add the headline property test: `insert_round(elim_round(pinned), fp.FP64)`
  recovers the original program's structure and agrees with it numerically on
  sample inputs. Include a case where the two are *not* inverses — an
  unbounded scope that `RoundElim`'s strictly-tighter guard refuses to hoist —
  to pin the asymmetry the roadmap flags, and to see whether the composition
  cycles.
- Update `docs/todos/rounding-axes.md`: mark gap 1 done, move the widening mode
  to the head of the remaining work, and record what the round-trip test
  settled about the termination open question.

Run: `pytest -n 8 tests/unit/strategies/ tests/unit/transform/test_round_insert.py`

## Verification

Per phase, only the tests listed above. At the end, the full gate:

```
pytest -n 8 tests/unit
python -m tests.infra
python -m tests.infra.backend.cpp
```

`tests.infra.backend.cpp` matters here specifically: the cpp emitter's storage
selection reads containment results, and this operator moves ops out of REAL
and into concrete formats, which is exactly what that dispatch consumes.

## Out of scope

- **Widening mode** (operand sites, §8's promotion) — same predicate, different
  site kind; the next commit after this lands.
- **`finitize(func, available=[...])`** — the §6.3 search; a recipe over this
  operator, not part of it.
- **Sequential verification** of multi-statement REAL blocks.
- `split_round` / `merge_round` and the rounding-mode plumbing they need.
