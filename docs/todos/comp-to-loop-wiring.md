# TODO: wire `fix(Hoistable ; CompToLoop)` into the cpp pipeline

Four fixes, then the wiring, then totality. Each phase is one commit.

Phase 1 has landed; 1b came out of reviewing it.

## Why

`CompToLoop` lowers a comprehension into an `fp.empty` allocation plus a `for`
loop, and it is not in the cpp pipeline — the emitter keeps its own lowering
instead. The audit in [backend-independence.md](backend-independence.md) measured
what wiring it in costs today and found the blockers are three small defects plus
one missing fold, not the capability gap that
*Considered: lowering comprehensions in the pipeline* in
[backend-cpp.md](backend-cpp.md) records.

What the wiring buys immediately is what `comp_to_loop.py`'s own docstring asks
for: a comprehension body gets a statement slot, so `RoundElim` and `RoundInsert`
can enter it. Phase 6 makes the pass total; phase 7 decides whether the pipeline
uses that, since a total pass and the emitter's fused flatten cannot both fire.
Phases 1–5 stand on their own whichever way phase 7 goes.

**The fixpoint.** Neither pass is a fixpoint alone and each supplies what the
other lacks: `Hoistable` seals a comprehension's element because it has no
statement slot, and `CompToLoop` creates that slot by making the loop;
`CompToLoop` declines a comprehension in a ternary arm or a `while` condition
because *it* has no slot, and `Hoistable` creates that one. Iterating clears all
three of `CompToLoop`'s *no statement-level position* refusals — measured, a
ternary arm and a `while` condition take one iteration each, a nested
comprehension takes two — and leaves only the dependent-clause refusal.

## Working policy

- **Pause for review after each phase.** Do not commit; Brett commits.
- **Run only the tests a phase names.** The full suite runs once, at the end.
- Every measurement asserts its own tree (see
  [Reproducing the measurements](#reproducing-the-measurements)); the editable
  install resolves `import fpy2` to the working tree regardless of cwd.

## Baseline

The corpus is the 219 functions of `tests/infra/examples` plus the `core`, `eft`,
`vector` and `matrix` libraries, compiled at FP64 through `CppCompiler.compile`.

| | today |
|---|---|
| corpus | 201 compiled / 18 refused |
| `tests/unit/backend/cpp` | 606 passed |
| the fixpoint, wired | −5 programs (all `matrix`), 14 emit differently |

## Phase 1 — `RoundElim` skips a `with`'s context expression

**Independent of the rest**; it is here because it is the one defect the audit
turned up, and it is small.

*What.* `ContextUse._visit_context` deliberately does not visit `stmt.ctx` —
**E-Context** evaluates a context expression under `REAL`, so nothing inside it
is a use of the enclosing context. `RoundElim`'s transform visitor does descend
into it, finds an `Add`, and calls `find_scope_from_use`, which raises. Give
`_RoundElimInstance` a `_visit_context` that rewrites the body and leaves
`stmt.ctx` alone.

```python
@fp.fpy
def example_static_context2():
    ES = 2
    NB = 8
    with fp.IEEEContext((ES + 2), (NB + 2)):
        return fp.round(1)
```

```
KeyError: no context scope found for use site Add(args=(Var(SourceId('ES')), Integer(2)))
```

*Why not catch the `KeyError`.* `_resolved_ctx`'s docstring commits to failing
loudly, on the reasoning that a node without a scope means a node built outside
scope analysis. That reasoning is right and the contract should stay; this node
is simply not one of those — it is a node `ContextUse` deliberately skips, so the
fix is to not ask.

*Tests.* `tests/unit/transform/test_round_elim.py`,
`tests/unit/strategies/test_elim_round.py`, and
**`tests/unit/backend/cpp/test_bind_profile.py`** — `EXPECTED_COMPILED` pins the
corpus count this phase moves, so the acceptance criterion *is* a test. Add the
program above as a regression.

*Acceptance.* The corpus goes 201 → **202**, and `optimize=True` and
`optimize=False` compile the same set.

**Landed** as `09e6027`, with `EXPECTED_COMPILED` 201 → 202 following it. Phase
1b then removed the override again — see below.

## Phase 1b — `ContextUse` records the context expression's `REAL` scope

*Why.* Phase 1 fixed the one consumer that crashed; it did not fix the gap.
`ContextUse` records no use site inside a `with`'s context expression, and the
convention that nobody may look there is not enforced — three walkers descend
anyway:

| | descends into `stmt.ctx`? |
|---|---|
| `ContextUse`, `FormatInfer`, the cpp emitter | no |
| `ValueClassInfer`, `ArraySizeInfer`, `DefineUse` (default visitor) | **yes** |

`DefineUse` *must* — `ES` and `NB` are genuinely read there. The other two ask
`use_to_scope.get(e)`, get `None`, and take the conservative branch of a question
**E-Context** answers plainly. Measured on `fp.IEEEContext(ES + 2, NB + 2)`:
`ValueClassInfer` said `TOP` for `ES + 2`, which under `REAL` is `ZERO|FINITE`;
`array_size._is_exact` is `scope is not None and scope.ctx == REAL`, so it says
`False` where the arithmetic is exact and a size that would cancel is lost.
Neither is unsound. Both are wrong.

*What.* Visit `stmt.ctx` under a `ContextScope(stmt, REAL)`, built *after* the
body's scope so a `with` whose context is itself `REAL` — where the two compare
equal — merges into it rather than clearing it.

*Keep that scope out of `scopes`.* `emitter.py` and `dead_code.py` both build
`{scope.site: scope}`, and two scopes sharing a `ContextStmt` would silently
collapse one. It goes in `uses`, hence `use_to_scope`, and nowhere else — so
`find_scope_from_use` answers what the semantics give while scope *iteration* is
untouched.

*Tests.* `tests/unit/analysis/test_context_use.py`,
`tests/unit/analysis/test_value_class.py`,
`tests/unit/analysis/test_array_size.py`, `tests/unit/backend/cpp/`.

*Acceptance.* `use_to_scope` answers `REAL` for every use inside a context
expression; the corpus stays at 202; `scopes` is unchanged.

*Phase 1's guard comes out with it.* Revisited once 1b was in, and measured:
dropping `RoundElim._visit_context` leaves the corpus at 202 with **zero** emit
differences and all 35 round-elim tests passing. `_resolved_ctx` now answers
`REAL` and `_is_eliminable` declines on its own, so the override prevented
nothing reachable. Three reasons it went rather than staying as belt-and-braces:

- the invariant "nothing under `REAL` is eliminable" belongs in `_is_eliminable`,
  stated once, not restated as a special case in one visitor;
- the analogy to `Hoistable._visit_context` / `ANF._visit_context` does not hold —
  those two *actively hoist* into the context expression and need somewhere to
  put the result; `RoundElim` never hoists there;
- it was the wrong shape for a tripwire anyway. It silently *skipped*, where
  `_emit_inline` **raises**. A guard that hides an invariant breaking is worse
  than none.

The regression test stays: it asserts the outcome (`is_equiv` on the context
expression plus interpreter agreement), not the mechanism, so it passes either
way and now pins 1b's guarantee.

*One coupling to keep in view.* `_is_eliminable`'s check is `ctx is REAL` —
identity against the singleton. It holds only while 1b records that same object;
a fresh `RealContext()` there would fall through and make the hoisting hazard
reachable with nothing watching for it.

## Phase 2 — `CompToLoop` fills the assignment target

*What.* `_lower` mints `acc`, returns `Var(acc)`, and the caller emits
`ys = acc`. Two names hold one list, `_shares_storage` sees a second *place*, and
`UnboxMode.STRICT` refuses:

```
strict unboxing failed for `zeros`: these lists must keep their shared handle
  `acc` (depth 1): shared
  `acc6` (depth 0): shared
```

Thread the destination through — `_lower(e, out, target=None)` — and use it as
the accumulator when the site being lowered is the whole right-hand side of an
`Assign`. `acc` then survives only where the comprehension is in a `return` or an
argument.

*Why not fix the analysis instead.* The general fix is
`AliasAnalysis.consumed_defs` discounting a name whose only readers are the
write-throughs that fill it and the one construction that takes it — which is the
same blocker as *Aggregate naming in ANF* in
[backend-independence.md](backend-independence.md). That is worth doing and it is
not this. Not minting the name is strictly smaller and removes the question.

*Tests.* `tests/unit/transform/test_comp_to_loop.py`,
`tests/unit/strategies/test_comp_to_loop.py`. Pin that `ys = [f(x) for x in xs]`
lowers with no intermediate name.

*Acceptance.* Wiring the fixpoint (locally, not committed) costs **0** corpus
programs, down from 5.

## Phase 3 — `CompToLoop` skips the iterable temp where it is pure

*What.* `_lower` binds every iterable to a temp so it is evaluated once. That is
right in general and wrong for a `range`: it moves the `range` out of the
iterated position, where the emitter fuses it into a counted loop, into a value
position, where it must be materialized.

```c++
// before                                  // after
for (int8_t x = 0; x < 5; ++x)             std::array<uint8_t, 5> _tmp1{};
    _tmp1[_tmp2++] = x + 1;                std::iota(_tmp1.begin(), _tmp1.end(), 0);
                                           for (int8_t i = 0; i < t7; ++i) { ... }
```

Skip the temp where the iterable is pure and read once — a `Range1`/`2`/`3` over
atoms always is. The temp exists to give "evaluate each iterable exactly once"
and to keep the loop bound out of reach of a body that rebinds the source name;
neither applies to an expression with no effects and no name to rebind.

*Quality only.* No program's outcome changes either way. It is here because the
fixpoint would otherwise regress every `for x in range(...)` comprehension in the
corpus.

*Tests.* `tests/unit/transform/test_comp_to_loop.py`. Pin that
`[x + 1 for x in range(5)]` lowers to a loop over `range(5)` directly, with no
binding for the range.

*Acceptance.* `test_list_comp1` emits the counted loop again — no `std::iota`,
no materialized range.

## Phase 4 — `_const_int` folds arithmetic over known lengths

*What.* `ArraySizeInfer._const_int` tries partial-eval, then matches `Len`,
`Dim` and `Size` through a name. It has no arithmetic case, so
`fp.empty(len(xs) * len(ys))` has no static size and a multi-clause comprehension
loses its `std::array`. Because it is the return type, the signature changes with
it:

```cpp
// [x + y for x in xs for y in ys], xs and ys proven 3 and 4
std::array<double, 12> prod(const std::array<double, 3>&, const std::array<double, 4>&);
std::vector<double>    prod(const std::array<double, 3>&, const std::array<double, 4>&);  // lowered
```

Add `Mul` / `Add` / `Sub` cases over operands `_const_int` already answers,
gated on the node being exact.

*The exactness gate is already satisfied here.* `CompToLoop` wraps the size
computation in `integer_ctx`, so the arithmetic is under a context that rounds
integers exactly. That is the narrow slice of
[array-size-integer-exactness.md](array-size-integer-exactness.md) this needs —
not `_affine`'s harder question of proving a *symbolic* index exact under an
inherited context. Do not widen the gate beyond what the enclosing scope states.

*Why this phase cannot be skipped.* **The corpus cannot see this regression.**
Its multi-clause comprehensions iterate literal `range`s, which `_const_int`
folds whole through partial eval. The witness needs proven-length *parameters*,
so this phase brings its own.

*Tests.* `tests/unit/analysis/test_array_size.py`. Add the `prod` shape above
with parameter lengths, asserting the size is derived; add a negative case where
the arithmetic is not exact and the size stays unknown.

*Acceptance.* `prod` keeps `std::array<double, 12>` under the fixpoint.

## Phase 5 — wire the fixpoint

*What.* In `CppCompiler.specialize()`, replace the single `Hoistable.apply` with
the fixpoint:

```python
for _ in range(LIMIT):
    fd = Hoistable.apply(fd)
    if not CompToLoop.sites(fd):
        break
    fd = CompToLoop.apply(fd)
```

Unconditionally, not behind `optimize` — it is a normalization, and `ANF` already
requires the form the first half of it establishes. It stays where `Hoistable` is
today, after `Specialize` and before `RoundElim`, so `RoundElim`'s suppressed
hoists still find their slots.

*The iteration bound is real, not defensive.* Each round either lowers a
comprehension or reaches a fixpoint, and the comprehension count strictly
decreases, so it terminates; the cap is there to turn a bug into an error rather
than a hang.

*Tests.* `tests/unit/backend/cpp/`, plus the profile tests that pin counts this
moves — `test_unbox_profile.py`, `tests/unit/transform/test_anf_profile.py`,
`tests/unit/transform/test_hoistable_profile.py`. Expect shape assertions to
need updating; a *count* moving the wrong way is a finding, not a test to fix.

*Acceptance.*

- corpus stays at 202, with no program lost;
- `CompToLoop.refusals` over the corpus names only dependent clauses, which the
  emitter still handles;
- the parameter-length witness set from phase 4 proves the same lengths as
  before the wiring, and no function's `fn_fmt` moves to `REAL_FORMAT`.

The last one is the standing gate from
*Precision is the acceptance test for an unfold* in
[backend-independence.md](backend-independence.md), and it is the one the corpus
alone does not discharge.

## Phase 6 — make `CompToLoop` capable of the dependent clause

**Capability only. The cpp pipeline lowers exactly what it lowered after phase 5;
what it *applies* is phase 7.** Splitting it this way keeps the backend out of
this commit, and it is the split the pass already supports: `sites` and
`refusals` distinguish "cannot lower" from "did not lower", and `apply` takes a
`where`.

*What.* `_verify` declines exactly one thing after phase 5: a clause whose
iterable mentions an earlier clause's target, where the length is a sum rather
than a product and `fp.empty` has nowhere to get it.
[derived-semantics](../source/dev/derived-semantics.rst) gives the rewrite —
build the rows with the single-clause form, sum their lengths, allocate, flatten.
With it `_verify` returns `None` everywhere and the pass is total.

*Write the index arithmetic under `integer_ctx`, not `fp.REAL`.* derived-semantics
spells the counters `with fp.REAL:` because it is specifying *meaning*; a
transform that copies that spelling produces a program with no storage —

```
storage selection failed for `ragged_lowered`: cannot pick storage for `t4`
```

— because a `REAL` accumulator in a loop of unknown trip count widens to
`REAL_FORMAT`. Under `integer_ctx` the same counter is an `int64_t` and the whole
rewrite compiles, `UnboxMode.STRICT` included. `CompToLoop`, `for_unroll` and
`split_loop` already take this precaution for the same reason.

*Tests.* `tests/unit/transform/test_comp_to_loop.py`,
`tests/unit/strategies/test_comp_to_loop.py`. Backend tests should not move.

*Acceptance.* `CompToLoop.refusals` is empty on every corpus function, and the
corpus still compiles 202 — unchanged, because the pipeline is not applying the
new case yet.

## Phase 7 — decide whether the pipeline applies it

The two outcomes are mutually exclusive, and this is the whole of the choice:
**a total `CompToLoop` leaves nothing for the emitter to fuse.** Applying the
dependent rewrite means `_emit_list_comp_at` never runs, so keeping it would
preserve dead code rather than the fast path.

*Apply it.* Every comprehension lowers, so `_visit_list_comp`,
`_emit_list_comp_at`, `_open_comp_loop`, the `_emit_at` case, two `storage_infer`
match arms and one `_ALLOC_EXPRS` entry — about 90 lines — come out. Only
statements introduce identifiers, one less case for storage selection, and
comprehension-freedom becomes a language property rather than one backend's
invariant; 22 files outside this backend still handle `ListComp`.

*Decline it.* The fixpoint skips dependent sites by policy and the emitter keeps
handling them. `ANF.refusals` then still names those comprehensions, which is
what it does today and is already tolerated — §1 records its refusals as
"entirely comprehensions, with zero in the three positions the emitter cannot
slot".

*What it costs to apply, measured.* The rows are real.
`[x for xs in xss for x in xs]` today, against the same program hand-written in
the lowered form:

```c++
// emitter                                  // lowered
std::vector<double> _tmp1(0);               std::vector<std::vector<double>> _tmp1(0);
for (const auto& xs : xss)                  for (const auto& xs : xss) { /* copy each row */ }
    for (double x : xs)                     for (const auto& t3 : t1) t2 += t3.size();
        _tmp1.push_back(x);                 std::vector<double> z(t2);
                                            for (const auto& t3 : t1)
                                                for (double t5 : t3) z[t4++] = t5;
```

One pass and one allocation become a full copy of every row plus two more passes.
**Recovering the fused form after lowering is a language change, not a pass.**
The emitter's flatten needs a *growable* list, and derived-semantics is explicit
that no rule changes a list's length, so there is no `append` to rewrite into —
the same reason `_emit_scale_by_pow2` and `_for_header` stay
([backend-independence.md](backend-independence.md)). So this is a real trade
between ~90 lines and every ragged flatten, not a cleanup.

*If it is applied*, delete behind a tripwire rather than by removal: a `ListComp`
reaching the emitter should raise a `CppEmitError` naming a backend bug, the way
`_emit_inline` does. Unreachable *and* impossible.

*Tests.* `tests/unit/backend/cpp/`, `tests/unit/analysis/test_storage_infer.py`.

## Not in scope

- **Demoting `ANF`.** It is the last step of the unfold plan, not this one;
  doing it now churns the same shape assertions twice.
- **The other unfolds** — `zip`, `enumerate`, slice, the chained comparison,
  `sum` / `any` / `all` / `amin` / `amax`, `min` / `max`.
- **`consumed_defs` for a name that is genuinely shared**, which phase 2 sidesteps
  rather than answers.

## Reproducing the measurements

Every number above comes from compiling the corpus with the pass under test
monkeypatched. Three ways this has gone wrong before, so:

- **Assert the tree.** `import fpy2` resolves to the editable install, so a
  script outside the worktree still measures the worktree:
  `assert '<worktree>' in fpy2.__file__`.
- **Use the harness entry point**, `compile(f, ctx=fp.FP64, arg_types=...)` with
  `arg_types` instantiated the way `tests/infra/backend/cpp.py` does. Bare
  `compile(f)` gives a different, misleading answer.
- **Name the baseline by commit, not `HEAD`** — phases land between probes.

The corpus is `tests.infra.examples.all_tests()` plus the four library modules
less `tests.infra.backend.cpp._library_ignore`. `_bind_operand` fires 113 times
over it, which is the count §7 of
[backend-independence.md](backend-independence.md) records, so it is the same
corpus that file measures on.
