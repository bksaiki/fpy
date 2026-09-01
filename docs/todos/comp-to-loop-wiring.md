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

Only where the element cannot read the target: the loops overwrite it before the
element runs. An *iterable* may — it is bound to a temp first, so it still sees
the list the name held.

*Tests.* `tests/unit/transform/test_comp_to_loop.py`,
`tests/unit/strategies/test_comp_to_loop.py`. Pin that `ys = [f(x) for x in xs]`
lowers with no intermediate name, that a `return` still mints one, that an
element reading the target keeps its accumulator, and that a comprehension in an
*iterable* does not claim the outer target.

*Acceptance.* Wiring the fixpoint (locally, not committed) costs **0** corpus
programs, down from 5.

**Landed, and the acceptance criterion is not met: still 5.** The rewrite does
what it says — measured, no `acc` survives an assignment — but none of the five
`matrix` failures is an assignment-target case, and they split two ways.

## Phase 2b — `_emit_empty` allows a partially-dimensioned allocation

*What.* `_emit_empty` refuses unless `_list_depth(result_ty) == len(dims)`, so
`fp.empty(n)` for a `list[list[T]]` is rejected — even though FPy's semantics
allow it, since cells start `UNINIT` and a later `xs[i] = <list>` fills them.
Every lowered comprehension producing a nested list allocates exactly that way,
so this blocks `zeros`, `ones`, `identity`, `set_row` and `set_column` alike.

*Pre-existing, and not caused by phase 2.* Verified with a hand-written program
and no `CompToLoop` anywhere:

```python
out = fp.empty(len(A))          # out : list[list[fp.Real]]
for i in range(len(A)):
    out[i] = A[i][:]
```

```
empty(...) shape mismatch: result type `CppList(elt=CppList(...))` has depth 2,
but 1 dimensions were given
```

`set_row` and `set_column` reached it only because phase 2 stopped them failing
earlier, at strict unboxing. What phase 2 changed is which error they report.

*Tests.* `tests/unit/backend/cpp/test_emit_list.py`, and the witness above.

## Phase 2c — a comprehension in element position fills its slot

*What.* `zeros` is `[[fp.round(0) for _ in range(cols)] for _ in range(rows)]`.
The outer comprehension is in a `return`, so it keeps its `acc` — correctly,
there is no name to fill. The inner one is the outer's *element*, so lowering
gives

```python
acc6 = fp.empty(len(t5))
for i7 in range(len(t5)):
    acc6[i7] = fp.round(0)
acc[i] = acc6
```

and `acc6` is a second name on the list `acc[i]` holds. Phase 2 cannot reach it:
an element position has no assignment target.

*Two ways to fix it.*

- **Fill the slot.** Generalize phase 2's target from a *name* to a *place*:
  `acc[i] = fp.empty(len(t5))`, then `acc[i][j] = <elt>`. Local to `CompToLoop`
  and exactly the same idea; it needs 2b, since `acc` itself is a
  partially-dimensioned allocation.
- **Discount it in the analysis.** `AliasAnalysis.consumed_defs` already
  discounts a name read exactly once by the construction that takes its value.
  It refuses here because one of its three guards is *the use must not
  re-execute*, and `acc[i] = acc6` is inside a loop. The guard is right for a def
  that does not repeat and over-conservative for one that repeats in lockstep
  with its use — each iteration allocates a fresh `acc6` and transfers it once.
  This is the general fix, and it is the same one *Aggregate naming in ANF* in
  [backend-independence.md](backend-independence.md) waits on.

Prefer the first to unblock, the second to close the class.

## Phase 3a — a discarded target binds nothing

*What.* `_lower`'s single-clause branch always emits
`Assign(copy_target(e.targets[0]), ..., ListRef(src, i))`. Where the target is
`_` that is `_ = t[i]`, which the emitter refuses:

```
compilation failed for `zeros`: unsupported assignment target UnderscoreId()
```

Skip the binding. A discarded target cannot be read by the element, and a
subscript has no effect to preserve — `ZipElim` and `EnumerateElim` already take
this position for their own discarded slots.

*Acceptance.* `zeros` compiles under the fixpoint when its arguments are typed
the way the source annotates them (`int`). **Met.**

## Phase 3 — `CompToLoop` skips the iterable temp where it is pure

**Not quality-only, and not sufficient on its own.** The plan had this as a
performance item; it is the direct cause of the last three corpus failures, and
fixing it needs a companion change in the emitter.

*What.* `_lower` binds every iterable to a temp so it is evaluated once. That is
right in general and wrong for a `range`: it moves the `range` out of the
iterated position, where the emitter fuses it, into a value position, where it
must be materialized.

```c++
// before                                  // after
for (int8_t x = 0; x < 5; ++x)             std::array<uint8_t, 5> _tmp1{};
    _tmp1[_tmp2++] = x + 1;                std::iota(_tmp1.begin(), _tmp1.end(), 0);
                                           for (int8_t i = 0; i < t7; ++i) { ... }
```

Skip the temp where the iterable is pure and read once — a `Range1`/`2`/`3` over
atoms always is. The temp gives "evaluate each iterable exactly once" and keeps
the loop bound out of reach of a body that rebinds the source name; neither
applies to an expression with no effects and no name to rebind.

*Why it is not sufficient on its own.* Removing the temp leaves the lowering
asking for `len(range(rows))`, which failed the same way — measured, all three of
`len(range(n))`, `range(n)[0]` and `[x for x in range(n)]` refused for a real
`n`. The length is the blocker, not the temp; `fp.empty` is FPy's only allocator
and needs one up front, where `_open_list_build` never does. Phase 3b supplies
it.

*And an inlined iterable must not be indexed.* Leaving the `range` inline is
only half of it: `_lower`'s single-clause branch reads `<iterable>[i]`, which
materialises the very list the inlining avoids. Where the iterable is inlined
*and* the target is named, the loop iterates it directly and carries a write
index, as the multi-clause branch does.

Only where the target is named. A discarded target reads no element, so the
indexed form materialises nothing — and it is the better shape, because its
index is the `range` loop's own variable, which `format_infer` bounds. A carried
counter widens instead, without bound where the trip count is not static:
switching `zeros` to it cost storage selection for `acc` and lost five corpus
programs before the condition was narrowed.

*Tests.* `tests/unit/transform/test_comp_to_loop.py`: a `range` iterable is left
inline, a non-`range` one still binds.

## Phase 3b — `len` of a range does not materialize the range

*What.* `len(range(stop))` is a count, and a count needs no integer *element*
type. Give `Len` a fused case for a `Range1` argument, taken before the operand
visit that would build the range — the same shape `Fst`/`Snd` already use. It is
the sort of fusion [backend-independence.md](backend-independence.md) says stays
in the emitter: FPy has no counted loop that is not a list, so there is nothing
to rewrite into.

*And the conversion is the language's, not the emitter's.* Every argument of
`range` must be an integer — the interpreter enforces it — so converting a real
bound is exact for every value the language admits, and `_range_bound`'s
`_maybe_cast` refusal of `double` → `int64_t` was over-strict. It now casts
explicitly, the category `_explicit_cast` already documents as "casts the
language requires". A non-integral bound is stuck in the interpreter, which
leaves the backend owing nothing — the same reasoning as an out-of-range
subscript.

> Settled while doing this: `derived-semantics.rst` defines `range` by a
> counting loop, which is total over the reals — run as an FPy program it gives
> `len(range(2.5)) == 3`. The interpreter's integer requirement is the intended
> rule, so the *page* is what is loose here, not `_eval_range`. Worth a separate
> fix; nothing in this plan depends on it.

`Range3` keeps the materialising path: its count divides by the step and the
step's sign picks the comparison, so a symbolic one needs a branch, and its
integer bounds make materialising available anyway.

*Tests.* `tests/unit/backend/cpp/test_emit_for.py`: a length builds no range,
and a real bound converts rather than refusing.

*Acceptance.* The fixpoint costs **0** corpus programs — 202 either way, no
program lost or gained, 19 emit differently. Phase 2's original criterion,
reached here. **Met.**

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

Add `Mul` / `Add` / `Sub` cases over operands `_const_int` already answers.

*The gate is not `_is_exact`.* That asks whether the operation rounds **at
all**, which takes a `REAL` scope — and `CompToLoop` wraps its size arithmetic
in `integer_ctx`, which is not `REAL`. An integer result needs less: the active
context has only to *represent* it. `_holds_int` asks exactly that
(`representable_under`), which carries a length through `integer_ctx` and still
declines under a symbolic context, where nothing is known.

That is the narrow slice of
[array-size-integer-exactness.md](array-size-integer-exactness.md) this needs —
not `_affine`'s harder question of proving a *symbolic* index exact under an
inherited context.

*Why this phase cannot be skipped.* **The corpus cannot see this regression.**
Its multi-clause comprehensions iterate literal `range`s, which `_const_int`
folds whole through partial eval. The witness needs proven-length *parameters*,
so this phase brings its own.

*Tests.* `tests/unit/analysis/test_array_size.py`. Add the `prod` shape above
with parameter lengths, asserting the size is derived; add a negative case where
the arithmetic is not exact and the size stays unknown.

*Acceptance.* `prod` keeps `std::array<double, 12>` under the fixpoint. **Met**
— and the corpus holds at 202 with the fixpoint wired, 19 emitting differently.

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

*Acceptance.* **Met.** The corpus stays at 202 with nothing lost or gained, 19
emit differently, and the `prod` witness keeps `std::array<double, 12>` — the
standing gate from *Precision is the acceptance test for an unfold* in
[backend-independence.md](backend-independence.md), which the corpus alone does
not discharge. `CompToLoop.refusals` names only dependent clauses, which the
emitter still handles until phase 6.

## Phase 5b — a filled list still moves out

*Why.* Wiring the fixpoint made `xs = [f(y) for y in ys]; return (xs, 1.0)`
**fail** under the default `UnboxMode.STRICT`, where it compiled before. The
corpus could not see it: 202 either way, because it holds no
comprehension-assigned-then-consumed shape.

*What the def graph said.* For `xs = fp.empty(n); xs[i] = ...; return (xs, 1.0)`:

```
#3 AssignDef  site=Assign          same_object=()      uses=[]
#4 PhiDef     site=phi             same_object=(3, 7)  uses=[IndexedAssign, Var]
#7 AssignDef  site=IndexedAssign   same_object=(4,)    uses=[]
```

Two independent refusals in `_note_consumed`, and the fix for both is that the
unit is the **object**, not one definition of it:

- the reaching definition is a *phi*, which the guard refused outright ("a phi
  has several definitions"). This one's operands are all `same_object` with it —
  one list threaded through its own fills, not a merge of two lists. The guard
  is now "no phi *outside* the class", which still refuses a value leaving
  through an unrelated merge.
- the phi has *two* uses, the loop's write-through and the read after it. An
  `xs[i] = v` is a use of the name but not a second *place*: it makes a
  definition `same_object` with what it wrote through. `_moves_out` discounts
  exactly those.

Recording the whole class matters downstream: `referrers_after_moves` drops a
name only when *every* definition of it is consumed, so recording the phi alone
would have changed nothing.

*Tests.* `tests/unit/backend/cpp/test_unbox.py` — a loop-filled list moves out,
and a loop-filled list *read again* keeps its handle. The negative one carries
the weight: an over-eager discount is a use-after-move, not a missed
optimization.

*Acceptance.* The program above compiles unboxed and moved under `STRICT`; the
corpus boxing profile is unmoved. **Met.**

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

*The switch is `dependent=False`, threaded through `sites` / `refusals` /
`apply`.* `_verify` returning `None` makes a comprehension a *site*, and
`apply` with no `where` takes every site — so without a flag the pipeline would
start applying this the moment it became possible, which is phase 7's decision
and not this one's.

*Every name the pass mints takes `temp_id`.* The dependent lowering needs
several temporaries of its own, and adding a third vocabulary of hard-coded
prefixes beside `acc` / `n` / `i` / `j` made the inconsistency plain: `temp_id`
was documented as naming "the iterable binding" while four other names ignored
it. All of them now go through `Gensym.refresh(temp_id)`, so a caller owns the
whole namespace the rewrite introduces — which is also what lets a test pin a
shape by passing a distinctive prefix instead of matching on gensym numbering.

The first clause's iterable is bound here exactly as `_lower` binds one; the
rest of the iterables and the element go into the nested comprehension and are
bound when *it* is lowered. `_descend` follows the same split, or the listing
would count a site the rewrite does not take.

*Tests.* `tests/unit/transform/test_comp_to_loop.py` — `_ragged` and `_ragged3`
lower to nothing under `dependent=True` and agree with the interpreter, the flag
is off by default, and the iterable binding takes `temp_id`.

*Acceptance.* `CompToLoop.refusals(..., dependent=True)` is empty on every
witness, three ragged shapes agree with the interpreter, and the corpus still
compiles 202 — unchanged, because the pipeline is not applying the new case.
**Met.**

## Phase 6b — a slot store checks the value it is actually emitting

*Found while measuring phase 6, and not caused by it.* The fully-lowered
`ragged` refused with

```
unsupported: this value is `std::vector<uint8_t>` where `std::vector<double>`
is needed, and C++ has no conversion between them.
```

*What.* `_visit_indexed_assign` checked the right-hand side's own storage
against the slot and *then* called the thing that made the check moot:

```python
src = self._storage_or_none(stmt.expr)
self._require_bridgeable(src, level, stmt.expr)    # refuses
self._require_no_narrowing(src, level, stmt.expr)
rhs = self._emit_at(stmt.expr, level, ctx)         # builds at `level`
```

For `out[i] = fp.empty(n)` the allocation's own bound is the lattice bottom, so
its own storage is the ladder's first rung — `std::vector<uint8_t>` — while the
slot is `std::vector<double>`. `_emit_at`'s `Empty` arm builds at the target and
its comment says so; the guard never let it.

*Pre-existing, with no `CompToLoop` anywhere:*

```python
out = fp.empty(len(xss))
for i in range(len(xss)):
    row = xss[i]
    out[i] = fp.empty(len(row))     # refused
    for j in range(len(row)):
        out[i][j] = row[j]
```

The same program storing `0` compiles — its stores are `uint8_t`, so the
bottom-typed allocation happens to agree. That coincidence is also why `zeros`
compiled and `ragged` did not.

*The fix goes in `_emit_at`, not at the call site.* A first version gave the
emitter a `_builds_at` predicate listing the forms `_emit_at` constructs — which
is a *second copy* of its `match`, and would go stale the moment an arm was
added. Instead `_emit_at` takes `cannot_convert`, and applies the two checks on
its own converting fall-through. One dispatch decides, so a constructing arm is
exempt by construction. `_visit_indexed_assign` is the only caller: the other
three `_require_bridgeable` sites guard a value arriving from elsewhere — a
range-for's element, a `std::get<i>` field, a callee's result — and construct
nothing.

*Tests.* `tests/unit/backend/cpp/test_emit_indexed_assign.py`: the witness
above, compiling as `std::vector<std::vector<double>>` with no `uint8_t`.

**The refusing half has no coverage, and did not before either.** Disabling
`cannot_convert` outright leaves all 612 backend tests passing, and three
attempts at a witness that reaches it all compiled instead: the alias replay
widens a slot's element to take the store, so the value fits by the time the
guard runs. The same story as `_convert_storage`, which
[backend-cpp.md](backend-cpp.md) measures as reached 37 times with `src == want`
every time. Recorded rather than papered over with a test that passes for the
wrong reason.

**For phase 7.** With this in, the fully-lowered `ragged` compiles, so "apply
it" is unblocked apart from the rows materialisation.

## Phase 7 — decide whether the pipeline applies it

**Declined. No code change: `dependent` stays `False` in the pipeline and the
emitter keeps lowering a dependent clause list itself.**

The two outcomes were always mutually exclusive — a total `CompToLoop` leaves
nothing for the emitter to fuse — so this is the whole of the choice, and the
measurement settles it.

| | |
|---|---|
| corpus comprehensions | 25 |
| ...with a dependent clause | **1** (`test_list_comp5`) |
| applying it | **−1 program**, 0 other emissions change |

The one program is
`[x for xs in [[1, 2], [3, 4]] for x in xs]`. The emitter compiles it to a fused
flatten with no intermediate at all:

```c++
std::array<uint8_t, 4> _tmp1{};  size_t _tmp2 = 0;
for (const std::array<uint8_t, 2>& xs : _tmp3)
    for (uint8_t x : xs)
        _tmp1[_tmp2++] = x;
```

Lowered, it does not compile: the flatten's accumulator widens to
`REAL_FORMAT` under the loop fixpoint —

```
cannot store an unconstrained real value in any storage format
```

— which is *Precision is the acceptance test for an unfold* in
[backend-independence.md](backend-independence.md) failing on the pass's own
output. So applying it costs a program, changes nothing else, and does not even
reach the ~90-line deletion it was for, since a lowering that loses a program
cannot be the one the emitter defers to.

**Beyond the measurement, the rule points the same way.** *What legitimately
stays in the emitter is what FPy has no syntax for.* The fused flatten needs a
**growable** list, and derived-semantics is explicit that no rule changes a
list's length — there is no `append` to rewrite into. That is the same reason
`_emit_scale_by_pow2` and `_for_header` stay.

*What phase 6 bought anyway.* `CompToLoop` is now *capable* of the dependent
case, which is what a consumer other than this backend needs — the scheduling
language can ask for a comprehension-free program and get one. The cpp pipeline
simply declines to, which is what the `dependent` flag is for.

*What would flip this.* Two things, in order:

1. `FormatInfer` giving a loop-carried integer accumulator a bound without
   needing the trip count. Under `integer_ctx` the value is an unbounded
   integer, which has a storage (`int64_t`) — the widen-mode fallback reaches
   `REAL_FORMAT` before that is considered. This looks like a gap rather than a
   necessity, and closing it would make the lowered form compile.
2. Then the rows materialisation is the remaining cost, and only a growable
   list or a deforestation pass removes it.

Until (1), there is nothing to weigh: the option that would delete emitter code
also deletes a working program.

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
