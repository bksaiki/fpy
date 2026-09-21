# Single-exit normalization: one trailing return per function

Implementation plan.  The design is settled; what follows is the phase
breakdown, one phase per commit.

## Working policy

- **Pause after each phase for review.**  Do not begin the next phase until the
  current one has been looked at.
- **Do not commit.**  The working tree is left dirty; the repository owner
  commits.
- **Run only the tests relevant to the phase.**  The full unit suite runs once,
  at the end.
- **Comments stay succinct**, and process notes -- what was tried, why an
  ordering was chosen -- belong here, not in source.

## Context

Nothing in `fpy2/transform/` touches `ReturnStmt` except to refuse it:

| Pass | Behaviour |
|---|---|
| `SimplifyIf` | declines a `return` in a branch -- no expression form for it |
| `FuncInline._refuses` | declines a callee without exactly one trailing return |

So a function with an early return can be neither normalized nor inlined, and a
branch calling one declines by both routes.  That blocks item 1 of
`backend-triton.md`: its pipeline inlines everything, which
is what leaves `SimplifyIf` no call to refuse.

Measured on `examples/mmasim`, the corpus the Triton work exists for -- 7 of 20
module-level functions have more than one return.  `fpy2/libraries` is clean at
0 of 74, so this is specific to the modelling style: a cascade of early exits
for NaN, infinity and zero.

Where those returns sit decides the work:

| Function | returns |
|---|---|
| `amd.exponent0` | 1 in an `if`, 1 trailing |
| `amd.overflow_inf` | 1 in an `if`, 1 trailing |
| `utils.sum_special_values` | 1 in an `if`, **1 in a loop**, 1 trailing |
| `amd.sum_special_values` | 1 in an `if`, **1 in a loop**, 1 trailing |
| `utils.dpa_special_values` | 1 in an `if`, **2 in a loop**, 1 trailing |
| `nv.dpa_special_values` | 1 in an `if`, **2 in a loop**, 1 trailing |

Four of six need a return out of a loop, so that is the main case, not an edge.

## Design

**The shape.**  A returned value becomes an assignment to one result name, and
everything after the return point is guarded:

    if c:                     if c:
        return a       ⇝          r = a
    ...                       else:
    return b                      ...
                                  r = b
                              return r

**A return inside a loop is refused, not rewritten.**  FPy has neither `break`
nor `continue`, so it would take a flag suppressing the rest of the body and
every later iteration -- the loop still running to completion.  Measured
instead of assumed: every loop return in `examples/mmasim` is inside a `for`,
none in a `while`, and a `for` over a list has a provable trip count after
`Specialize`, so all of them unroll away.  `unroll_for` with `times` at least
the trip count leaves zero loops; partial unrolling does not help, since the
loop survives with a larger body.

The Triton pipeline unrolls static-length lists into registers regardless, so
the unroll is happening anyway and this pass never sees the shape.  A `while`
whose condition is data-dependent has no such route -- see [Open
items](#open-items).

**Why this pass rather than relaxing the consumers.**  `SimplifyIf` cannot
express `return` as a value, and `FuncInline` cannot splice an early exit into
a caller.  Both refusals are correct; the fix is to remove the shape.

## Phases

### Phase 1 -- returns under `if`, loops refused

**Done.**  `fpy2/transform/single_exit.py`, exported as `SingleExit`;
17 tests in `tests/unit/transform/test_single_exit.py`.

Two things the corpus found that the plan did not anticipate:

- **`ContextStmt` is a third block-bearing statement.**  `nv.fdpa_round` ends
  `with rho: return fp.round(s)`.  A `with` whose body always returns sinks in
  place -- the assignment stays inside, since its right-hand side rounds under
  that context.  One whose body *conditionally* returns is refused: moving the
  continuation inside would change its rounding context.
- **A postcondition, because the first attempt failed silently.**  Before it
  existed, `fdpa_round` came back with two returns and no complaint.  `apply`
  now counts the returns it left and refuses if more than one survives, so an
  unhandled shape declines rather than passing through.

**Composition report** (the [open item](#does-normalization-create-shapes-simplifyif-then-refuses) this phase owed):

| function | after `SingleExit`, `SimplifyIf` says |
|---|---|
| `nv.fdpa_round` | accepted |
| `amd.overflow_inf` | accepted |
| `amd.exponent0` | declined -- *call*, not assert |

The feared assert collision did not appear.  The one refusal is the known call
rule, and `inline` -> `single_exit` -> `simplify_if` runs clean on
`exponent0` with the interpreter agreeing on every sampled input.


`fpy2/transform/single_exit.py`: a `SingleExit` pass handling returns at any
`if` depth, refusing one inside a `ForStmt` or `WhileStmt` with a reason.
Exported from `fpy2.transform`.

Separate from the wrapper so the rewrite is reviewed on its own: it is where
the semantics are, and where the report on `SimplifyIf` composition comes from.

Tests: `tests/unit/transform/test_single_exit.py` -- structural (one trailing
return survives) and semantic (interpreter agreement) over nested `if` shapes,
the two mmasim functions needing no unroll, the loop refusal, and the four
loop functions accepted *after* `unroll_for`.

    pytest tests/unit/transform/test_single_exit.py

### Phase 2 -- strategy wrapper

**Done, and not as planned.**  `fpy2/strategies/exit_single.py`, exported as
`fpy2.strategies.single_exit`, listed in `docs/source/strategies.rst`;
9 tests in `tests/unit/strategies/test_single_exit.py`.

**No `where` / `sites` / `refusals`, and no `SiteRewriter`.**  The plan called
for them by analogy with `simplify_if`, which was wrong: a function has one
exit structure, not one per return, so normalizing "one return" is not a
thing a caller could ask for.  `to_hoistable` is the right model and says so
itself -- *"Takes no `where`: normal form is not a per-site decision."*  The
wrapper is therefore a plain `Function -> Function`, and nothing is registered
in the `_SITES` / `_REFUSALS` tables.  `test_wrapper_contract.py` still covers
it, since that table derives from `__all__`.

Cursors do not forward, as with `to_hoistable`: the function body is rebuilt.


`fpy2/strategies/`, re-exported as `fpy2.strategies.single_exit`, with
`where` / `sites` / `refusals` via `SiteRewriter` and an `EditLog`, matching
`simplify_if`.  Registered in the `_SITES` / `_REFUSALS` tables, listed in
`docs/source/strategies.rst`.

Separate because the contract tests in `tests/unit/strategies/` sweep every
registered strategy, so this phase is where those must pass.

Tests: `tests/unit/strategies/test_single_exit.py` plus the shared contract
suites.

    pytest tests/unit/strategies tests/unit/transform/test_single_exit.py

### After the last phase

    make lint
    python -m pytest tests/unit -q -n 8

Then confirm the point of the exercise: every mmasim function normalizes, and
`inline` no longer refuses a call to one.

## Open items

### Does normalization create shapes `SimplifyIf` then refuses?

Much smaller now that loops are refused: the guarded-code-after-a-loop case,
where `utils.sum_special_values` has an `assert` that `SimplifyIf` declines,
does not arise.  What remains is whether guarding an `if` body puts an
`assert`, a list write or an `fp.cast` under a branch that `SimplifyIf` then
refuses.

The Triton compiler's flag to disable asserts covers the first, on two
conditions: it must be a *transform* removing `AssertStmt` before `SimplifyIf`
-- an emitter that merely declines to emit them does not help, since
`SimplifyIf` still sees them, and nothing in `fpy2/transform/` removes them
today -- and it narrows the contract, since the emitted kernel then matches the
interpreter only on inputs where no assert would have fired.

Phase 1 should report what `SimplifyIf` makes of each normalized mmasim
function, so this is measured rather than predicted.

### How far does the noexcept flag reach?

`AssertStmt` is one of several aborts `SimplifyIf` refuses.  The others are
`fp.cast` (asserts its result is exact) and any operation under an `ASSERT`
overflow context.  Both abort for the same reason an assert does.

If noexcept covers them too, `SimplifyIf`'s abort set shrinks to `return`,
effects, list writes and loops -- a much smaller refusal surface for the Triton
pipeline, and one this plan's guards are unlikely to collide with.  If it
covers only `AssertStmt`, a guarded `fp.cast` still refuses.

No provisional call; it is the Triton roadmap's decision, not this pass's.
Worth settling before Phase 1 reports, since it decides what that report means.

### When should loop returns be handled at all?

Refusing them rests on every one being unrollable.  That holds for the corpus
today -- all `for`, all over lists with provable lengths -- but a `while` with a
data-dependent condition has no unroll, and a `for` whose length is not proven
does not either.

Reopen when such a shape appears.  The rewrite is known (a `done` flag guarding
the body, conjoined into a `while` condition so the loop still terminates); it
is the cost that argues against building it now -- one more delicate rewrite
for no current caller.

### Should the pass run before or after `SimplifyIf` in the Triton pipeline?

Before, since `SimplifyIf` refuses the shape this removes.  But this pass
*creates* `if`/`else` where there was a bare early return, which `SimplifyIf`
then has to handle -- so the two may need to alternate rather than run once
each.  Settle when Phase 1 reports on the item above.

### Where does this work live?

This document is on `triton-compiler` so it can link to
`backend-triton.md`, but the pass is a general FPy transform
with no Triton in it and should be branched off `main`, as `SimplifyIf` was.
The document may want to move with it.
