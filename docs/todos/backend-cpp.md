# C++ backend — design notes & open TODOs

The cpp backend (`fpy2/backend/cpp/`) compiles FPy to C++ end-to-end across
scalar arithmetic, control flow, lists, tuples, in-place mutation, the `<cmath>`
family, and rounding-context boundaries. Unit coverage lives at
`tests/unit/backend/cpp/`; the differential harness is
`tests/infra/backend/cpp.py`.

Module layout:

- `compiler.py` — public `CppCompiler`, pipeline orchestration.
- `emitter.py` — AST walker that produces C++ source.
- `ops.py` — per-op tables of supported C++ signatures.
- `storage.py` — the C++ format ladder, `CppStorageDomain`, and the
  format-to-`CppType` translation.
- `variables.py` — names, declaration sites, and `binds_by_reference`, keyed by
  the classes `StorageInfer` found.
- `unbox.py` — which lists may drop the `std::shared_ptr` handle.  The
  region-level facts it used to gather are now
  `AliasAnalysis.written_regions` / `.slot_replaced` / `.returned_levels`
  and `analysis.region_sizes`.
- `types.py` — `CppScalar` / `CppList` / `CppTuple` and source formatting.
- `target.py`, `utils.py` — target description and header preamble.  There is no
  runtime: `CPP_HELPERS` is empty, and every spelling the emitter produces is
  `std::`.

Read the design before changing anything; [Open issues](#open-issues) is at
the bottom.

Everything C++-specific lives here. Analyses this backend depends on but does
not own have their own documents: `round-elim.md`, `array-size-symbolic.md`,
`array-size-integer-exactness.md`.

The emitter's input is in **hoistable form**: `fpy2.transform.Hoistable` runs in
`specialize()`, so every expression sits where a statement may be inserted above
it, and the three positions a statement must never escape — a `while` condition,
a ternary arm, a short-circuited operand — hold atoms.  Operands are *not*
flattened; the emitter mints its own name where it reads one twice
(`_bind_operand`), and `_emit_inline` refuses if a statement would ever escape
its guard.

The correctness criterion: *if the compiler succeeds, the emitted C++ must
compile and must behave as the FPy interpreter does wherever FPy's semantics are
defined.* A refusal is always acceptable — the compiler may be limited — so a
shape it cannot handle should raise, never miscompile.

The qualifier is load-bearing. FPy has undefined behavior, and the interpreter's
checks are not the contract: it raises on an out-of-range subscript and on a
mismatched-length `zip`, but neither is promised, so the backend owes nothing
there. Nothing currently violates the criterion as stated.

**Storage *contains* a format; it does not equal it.** `choose_storage` picks the
smallest ladder type holding a format, which is right for storing a value and
wrong as a *rounding*. Three separate miscompiles came from conflating the two,
so the rule is worth stating once: a `static_cast` into a context's storage is
that context's `round` only when the whole *context* is one the op table
dispatches on (`target.is_native_ctx`). Comparing formats is not enough, because
a format carries neither the overflow rule nor the random bits — a saturating
`IEEEContext(8, 32)` and a `-128..127` context under `ASSERT` are format-equal to
`FP32` and `int8_t` respectively, and neither behaves like the cast. Anything
else must be lowered explicitly or refused; see `native-lowering-roadmap.md`.

**An expression's storage can disagree with its declaration's.**
`_storage_for_expr` answers from `format_info.by_expr`, while a variable is
*declared* with what `StorageInfer` chose for its coalesced class — and the two
are not the same type. Anything deciding whether a cast is needed has to read the
one the operand is actually emitted as, or the cast silently goes missing.
`_emit_min_max` takes its target from the active context for this reason, oddly
for an operation that does no rounding: `library_core.max_e` puts an `int16_t`
accumulator beside an `int64_t` `logb`, and an operand-derived target leaves
`std::max` with two types to deduce from.

## Design

### Storage vs. rounding

The core insight is that **storage type and rounding format are separate**:

- **Rounding format** (`FormatInfer`): the smallest format that bounds an
  expression's value at runtime. Per-expression.
- **Storage format**: the C++ type that *holds* the value. Per-definition. Must
  contain the rounding format of everything assigned into that variable.

We pick the smallest valid storage from a fixed ladder: `int{8,16,32,64}_t` /
`uint*` / `float` / `double`. Unbounded integer formats fall back to `int64_t`;
non-abstractable / `REAL_FORMAT` results are rejected with an error naming the
offending expression.

Two things complicate the per-expression story, both documented separately: a
list also has a *representation* (handle or value — `unbox.py`), and where
several expressions reach one place they must agree on one C++ type
(`emitter._emit_at`).

### SSA rebinds → fresh C++ variables

The emitter is free to give every SSA def its own C++ variable. The one
constraint is that defs denoting the same runtime object share storage —
`reaching_defs.same_object_defs` is that rule, and `StorageInfer` unions
over it. Argument and free-variable defs anchor a class to the bare source name;
other classes for the same name take `_1`, `_2`, … suffixes.

Per-class declaration shape:

- **`declare_at_assign`** — the lowest-index writer is the declaration site, so
  the type folds into the assign (`double t = (a + b);`).
- **`hoists_before`** — a class with writers in disjoint `if`/`else` branches
  where the variable did not exist before the `if` (the merge phi is
  `is_intro=True`). No single writer dominates, so the emitter hoists
  `T name{};` just before the responsible `IfStmt`.

### Operation type matching

C++ has no ad-hoc polymorphism for primitive numeric ops, so `ops.py` enumerates
supported signatures per FPy op, parameterized by *argument C++ types* and
*output context*. At an op site the emitter matches the active rounding context
(`ContextUseAnalysis`) against the signature's `out_ctx`, and each operand's
storage against `in_ty`. On a miss it falls back to the all-active-context
signature and casts every operand in.

**Every** conversion is an explicit `static_cast` — no reliance on implicit
promotion, even for lossless widenings. Two paths:

- `_maybe_cast` — *implicit* casts (op-dispatch fallback, comparison
  cast-to-supremum). **Rejects** a lossy conversion, telling the user to wrap the
  operand in `fp.round(...)`. So FP64 → FP32 fails to compile until the rounding
  is written in the source.
- `_explicit_cast` — *user-explicit* casts (`Round`, subscript `size_t`). Emitted
  unconditionally; the user already accepted the conversion.

### Context boundaries

The active context at every `FuncDef` / `ContextStmt` comes from
`ContextUseAnalysis`.

**Validation is gated on use.** A scope is validated only when some primitive op
actually dispatches under it. Scopes with no uses are skipped entirely — no
validation, no `fesetround` — so a program with no rounding-context use needs no
supported function-level context, and `with UnsupportedCtx:` compiles as long as
nothing inside dispatches under it.

When a scope is used:

- The context must be concrete; a symbolic context variable is rejected at its
  introduction site.
- **Float contexts** need an `fesetround`-supported mode (RNE / RTZ / RTP / RTN).
  `_current_rm` tracks the mode the live `fenv` is guaranteed to hold, and a
  matching mode makes a `with` a C++-level no-op. It is seeded from the
  function-level scope, which is a **precondition on the caller**: an emitted
  kernel is entered with the RM its top-level annotation names, or with
  `FE_TONEAREST` when that annotation names no FP mode (absent, `REAL`, or
  integer). Either way no entry `fesetround` is emitted. `None` = unknown — a
  context nothing resolved, or an RM `fesetround` cannot express — is the one
  case where a nested concrete `with` must set the mode unconditionally.
- **Integer contexts** must use RTZ — that is what C++ integer truncation does.

`Round` / `Cast`: both bypass the op table, so both carry its discipline
themselves (`_require_cast_is_round`) — see the storage-versus-format rule at the
top.

- `Round(arg)` under a **native** context is `static_cast<target>(arg)`, whose
  rounding mode comes from the surrounding `fesetround` boundary. A **literal**
  argument is folded at compile time instead (`_fold_rounded_literal`) — C++ has
  no exact-real literal, so this is the only way an inexact constant is
  representable at all, and it also gets the mode the program asked for rather
  than whatever `fesetround` last left behind.
- `Round(arg)` under a **fixed-point** context goes to `_emit_integral_round`,
  which either lowers it faithfully or refuses; it never falls through to a bare
  cast. Float storage rounds by libm (`trunc`/`floor`/`ceil`/`round`/`nearbyint`),
  integer storage by the cast itself. Either way the bound is asserted, on the
  *rounded* value — `100.7` is in bounds under `RTZ` even though `100.7 > 100`.
  An overflow *rule* other than `ASSERT` is refused: `SATURATE`/`WRAP`/`OVERFLOW`
  are behavior this lowering does not implement, and `unfold_overflow` is what
  turns them into program text.
- `Cast(arg)` — the node `fp.cast` and `fp.round_exact` both parse to — is the
  same cast plus a runtime assertion that it was lossless. Under a native context
  a storage round-trip *is* that claim, NaN-aware for FP operands. Under a
  fixed-point context it is not, since storage is wider than the format, so
  `_assert_fixed_exact` adds the specials, representability and bound tests; the
  same-type short-circuit no longer skips them.

### Pipeline

```
Module
  → Specialize                 # one FuncDef per (callee, ctx, arg formats)
  → fix(Hoistable ; CompToLoop)  # a statement slot wherever one is needed,
                                 # and every comprehension lowered into the
                                 # loop that is that slot
  → RoundElim                  # (optimize only)
  → Simplify                   # (optimize only) fold, copy-propagate and
                               # delete the debris the lowerings above leave
  → DefineUse
  → ContextUse                 # resolves with-block contexts
  → ArraySizeInfer             # FormatInfer needs it for bounded iteration
  → FormatInfer                # rounding format per def/expr
  → ValueClassInfer            # what a value cannot be (NaN, zero, negative)
  → StorageInfer               # one storage format per runtime object
  → VariableAlloc              # names and declaration sites for its classes
  → Alias / Escape             # what may refer to what; what a callee retains
  → Unbox                      # handle or value, per alias region
  → emit C++
```

`Specialize` means a callee's formats follow its call site — so a callee called with a wider
list is automatically instantiated wider, which is the workaround
*A narrower value meeting a wider place* relies on.

### Translation-unit preamble

`CppCompiler.compile` returns a function definition only, so single-function
tests can use exact-string equality. Callers wanting a full translation unit
pull `headers()`, `helpers()`, or `prelude()`. `helpers()` is **empty** —
everything is emitted in standard-library spellings at the use site
(`std::shared_ptr<std::vector<T>>`, `std::array`), including the IEEE
`minimum`/`maximum` that used to be an `fpy::min`/`max` template. It is kept as a
method so callers need not care whether the backend currently emits any support
code. Headers track exactly what the emitted code uses.

## Open issues

### An operand emitting statements must not escape its guard

Closed twice over: `Hoistable` empties the positions where it could happen — a
`while` condition, a ternary arm, an `and` / `or` tail — and `_emit_inline`
refuses if one ever reaches the emitter anyway, comparing the writer's line count
around the emission rather than guessing from the syntax. `test_statement_form.py`
runs the three witnesses with both passes monkeypatched to the identity, since
`Hoistable` alone would let every witness pass without witnessing anything.

An `if` / `if1` condition is deliberately *not* gated: it runs once, just before
the branch, so its statements belong in the enclosing block — which is why
`_emit_guarded_block` takes its condition already emitted.

### `Simplify` evaporates a static witness

`Simplify` runs under the default `optimize=True`, so a test program whose
result is fully determined compiles to `return <constant>;`. An emitter witness
has to take a parameter, or read its own result more than once, or it pins
nothing. See §9 in [backend-independence.md](backend-independence.md).

### `ANF` is not in this pipeline

`fpy2.transform.ANF` flattens every operand to a name.  It ran last in
`specialize()` and was removed: measured over the corpus it lost no program,
and it cost 258 emitted lines and 448 temporary mentions — `determinant_3x3`
alone went from thirty `tN` assignments to three expressions.

It also cost the `2 ** n * x` fusion. `_emit_scale_by_pow2` matches a
*syntactic* `Pow` as an operand of a `Mul`, and naming the power first stopped
it firing; with the pass gone, `std::ldexp(x, n)` is emitted again rather than a
separate power and product.

What replaced it is nothing: the emitter mints a name where it needs one, which
is 50 sites over the corpus (`test_bind_profile.py` pins them), against the
1236 mentions the pass was producing. Two routes were considered and rejected
while it was still in place, and both stay rejected:

- **Resolving the power through its name** in the emitter
  (`defining_expr(scale)`) is *unsound*: `StorageInfer` gives two definitions of
  one source name a single C++ variable, so re-emitting the exponent at the
  product reads whatever a later branch put there. `k = n; p = 2 ** k; if c: k =
  m; return x * p` computed `x * 2**m`. Pinned by
  `test_the_exponent_is_not_re_read_at_the_product`.
- **Teaching `ANF` not to name a `Pow`** would let one backend's peephole
  restrict a backend-independent pass, and only for the shape that happened to
  have a test — `_fold_rounded_literal`, `_concrete_int_of` and
  `_range_counter_scalar` match syntax the same way.

The pass remains available as `fpy2.strategies.to_anf`.

### Unchecked subscripts are not a bug

An out-of-range subscript is undefined in FPy. `_visit_list_ref`,
`_visit_list_slice` and `_visit_indexed_assign` emit raw `xs[i]`, and that is the
intended lowering. The interpreter raises on `xs[10]` over a shorter list, but
that is an artifact of how it is written, not a promise the language makes.

Recorded because the shape invites the opposite conclusion: the emitted code
looks like it disagrees with the interpreter, so it reads as violating the
criterion above. It does not — that applies where FPy's semantics are *defined*.
Do not add a checked-subscript helper on this reasoning.

Were bounds checking ever wanted, it would be a debug-build feature like the
rounding assertions, and it would want the array-size work first: the checks
worth keeping are the ones `array-size-symbolic.md`'s size equalities cannot
discharge statically.

### One question answered in four places

"Can this value inhabit that place?" is decided four times, with different rules:

| path | sites | scalar | list element type | boxing |
|---|---|---|---|---|
`_emit_at` / `_convert_storage` | 10 | cast, never refuses | rebuild if unboxed, refuse if boxed | unboxed→boxed free; reverse refused |
`_maybe_cast` | 23 | **refuses if lossy** | — | — |
`_adapt_result` | callee results | — | refuse on mismatch | refuse if boxed; unboxed→boxed via `make_shared` |
`_adapt_arg` | call arguments | — | refuse on mismatch | **boxed→unboxed via deref**; reverse refused |

The scalar rules only *look* contradictory: `_emit_at` returns early when `want`
is a scalar, so `_convert_storage` sees scalars only as tuple fields, where the
target is the join and never narrower than a contributor. Measured — zero lossy
narrowings through it across the corpus. But that reasoning is written nowhere
and has to be rederived from two early-returns in different functions. Every site
is defensible alone, none states its precondition, so the set can only be audited
by reconstructing it.

The fix: one predicate per *place kind* — return, argument, slot store, container
field, operand rebind, callee result — in one module, called from each site
instead of reimplemented. Deferred: instrumenting every `raise` and running the
corpus plus 400 generated programs fired only 8 of ~75 sites, so this buys
maintainability rather than correctness, at ~33 call sites in the most delicate
part of the emitter. Worth doing when something next changes representation
handling.

Those 8 also say where the compilable set is actually bounded: *unconstrained
real in a finite C++ type* accounts for 233 of 543 refusals, and the next two are
downstream of the same storage question. That is the only lever that would move
the *number* of compilable programs, and it is out of scope above.

### Recovering from an unsupported rounding instead of refusing

*Done, behind `CppCompiler(unfold=UnfoldMode....)`* — `ROUNDINGS` for
the second row alone, `DOUBLE_ROUND` for both. Every refusal in this area
named the operator that fixes it; the flag runs them instead. Detection is
`fpy2/backend/cpp/unfold_round.py`, which asks `is_native_ctx` on the
specialized AST — before the analyses the emitter needs, and before the format
inference the rewrite exists to make succeed. Two rows:

| unsupported | recovery |
|---|---|
| *arithmetic* under a non-native context | `SplitRound` through a native intermediate: compute wide, re-round to the target |
| the rounding that leaves, and any other | `UnfoldSpecial → UnfoldOverflow → FloatToFixed → RescaleFixed` |

The first row is a deliberate double rounding, safe for a specific reason:
`SplitRound` is gated on the correct-double-rounding rules, so it applies only
where the two roundings compose to what the single one gave, and declines
otherwise. That is the difference between this and computing wide and
truncating.

**What it buys is coverage, not a smaller backend.** `_emit_integral_round`,
`_emit_integral_value` and `_bound_test` all still fire after the sequence — the
libm call and the bound assertion are the emitter's work either way. The gain is
that programs which refused now compile, bit-exactly:
`test_lowered_roundtrip` drives all fourteen targets through the flag, and
`TestArithRoundtrip` covers `sqrt`, `/` and `+` under three of them.

It never costs a diagnosis: where the rewrite leaves a program that still
fails, it fails further along than the original would have, so
`compile_module` asks the unrewritten one and reports that.

**No round-to-odd level**, though it is the mode `derive_intermediate` returns
and the one Figure 8 covers for arbitrary reals — so it is accepted exactly
where every native candidate is refused (`exp` anywhere, `div` / `sqrt` under a
directed or saturating target). It gives `MPFloatContext(pmax=13, rm=RTO)`, and
no native mode is RTO, so the split moves the site rather than removing it;
splitting *that* one needs RTO over RTO, which widens by a bit each time, or
exactness, which those three operations have not got. It becomes reachable if an
unbounded RTO operation becomes emittable — the bit-reinterpreting soft-float
direction in [native-lowering-roadmap.md](native-lowering-roadmap.md).

What is left:

- **What `DOUBLE_ROUND` costs** is unmeasured: `FloatToFixed` states a
  value-class branch per site, so an `FP16` add is 47 emitted lines and a
  two-operation polynomial 135. The corpus does not exercise it, the flag being
  off there.
- **A user-written fixed-point target never benefits.** The row is reachable
  only from inside the flow, where `FloatToFixed` produces a bounded context at
  a known position over an operand `UnfoldSpecial` has classified. A
  hand-written one fails for two reasons outside this work: `UnfoldOverflow`
  states no rule for `WRAP` or for an unbounded format, and `RescaleFixed`'s
  shift lands under a context the storage ladder has no entry for — the same gap
  that makes `MPFixedContext(-1)` refuse with no site at all.
- **Arithmetic needs its operands in the target format.** The per-operation
  rules hold for operands the target represents, so `x + y` under `FP16` wants
  `FP16` arguments; a program holding `FP32` values and rounding to `FP16` per
  operation is the natural shape and reaches no rule. What it needs is a rule
  quantified over the operand format.
- **A transcendental has no rule at all.** `exp` keeps its refusal under a
  nearest target; the only remaining route is exactness, which needs a
  correctly-rounded implementation to compare against.

### A narrower value meeting a wider place

Where several values reach one place — a `return`, a ternary arm, a list's
elements, a tuple's field — that place admits one C++ type, while `format_infer`
bounds each expression by *its own* values. Reconciling the two is a **storage**
question, so it lives in `emitter._emit_at`: it builds a *constructor* at the
place's type and converts anything else (`_convert_storage`).

> Do not push this into `format_infer`. An earlier version did, by overwriting
> each contributor's `by_expr` entry with the join. Sound, strictly less precise,
> and it makes a backend-independent analysis answer a C++ question: `[1.5, 2.5]`
> and `[3.0]` both came out bounded by `{3/2, 5/2, 3}`, so `round_elim`,
> `const_fold`, error analysis and the FPCore backend all paid for a decision
> only this backend cares about.

**What cannot be converted.** A *shared* list cannot be rebuilt: converting
allocates, a new allocation is a different object, and sharing exists precisely
so FPy's aliasing survives. `_convert_storage` refuses and `_refuse_unsharing`
writes the message. Unboxed → boxed is fine and is done — a value has no aliases
to lose. A callee's result is the same refusal from the other side: its
representation is fixed by the callee's own body.

The refusal has to be ours rather than the C++ compiler's, because reference-bound
names are spelled `const auto&` — *nothing in the emitted text states the element
type*, so only C++ would object, and only sometimes.
`test_a_shared_narrower_list_is_refused` pins seven shapes.

Four `[32_64]` matrix instantiations refuse today:
`_gen_return_param_or_literal`, `_gen_ternary_param`, `_gen_list_into_tuple`,
`_gen_comprehension_of_rows`. The discriminator is not fresh-vs-shared — a fresh
narrower local compiles, because storage assignment unifies it with the return
type. It is whether the narrower value's storage is fixed by something outside
the function: a parameter's ABI, a callee's return, or a reference binding.

**The workaround is real and pinned.** Widening at the *call site* is enough:
`Specialize` then instantiates the callee at the wider argument format
(`test_widening_the_call_site_is_a_real_workaround`).

Three things considered and not done:

- **Raise the definition instead of converting at the place** (`place_floors`,
  landed in `f05ea99`, removed; `cpp-old` is the last state with it). Measured
  cost of removing it: the corpus failure set was unchanged and the matrix went
  29 → 27 instantiations, both losses being "return an FP32 list parameter or an
  FP64 literal list". A pure capability, fixing no defect — against which it has
  to skip exactly the values with no storage of their own to raise, and getting
  that set wrong is a *silent* miscompile for the `const auto&` reason above. It
  got that wrong four times. Raising a *parameter* also changes the ABI, which is
  fine for an entry point and wrong for a function compiled code calls.
- **Closing the callee-result case properly** is harder than propagating a floor
  across the call edge: a callee's return format is a *function* of its parameter
  formats, so a caller can only ask for parameter formats that *produce* a wider
  return. That is inverting the callee's body, and for many callees no answer
  exists — one returning `[1.5]` cannot be widened by any argument.
- **A language answer.** A returned list's element format is arguably part of
  what the function *is*; a signature claiming `-> list[fp.Real]` at FP64 while
  returning an FP32 parameter could be rejected by `TypeInfer` rather than by
  storage selection.

### What stays boxed

A list is a plain `std::vector<T>` wherever `fpy2.analysis.alias` proves nothing
can observe the difference, and keeps the `std::shared_ptr<std::vector<T>>` handle otherwise. **All
166 of the corpus's signature list levels come out unboxed**, so both shapes below
are ones the corpus does not contain — which is why they are written down.
`test_unbox_profile.py` pins the count *and* the corpus size: an empty result
only means something while the corpus is as large as when it was measured.

**A list a local name and a container both hold.** *Settled.* A name read
exactly once, by the construction that takes its value, hands the value over
rather than holding it alongside: `AliasAnalysis.consumed_defs`, discounted by
`referrers_after_moves`. Both disjuncts of the boxing decision take the
discount, and `_shares_storage` needs it on both halves of
`slots = referrers - len(by_name)` or `slots` absorbs the name back.

The unit is the *object*, not one definition. A list a loop fills reaches its
consumer through a phi over the allocation and the stores, all one runtime
object per `same_object_defs`, so the whole class is recorded — a write-through
is a use of the name but not a second *read*, and `referrers_after_moves` drops
a name only when every definition of it is consumed.

The discount asserts a *transfer*, so `_emit_deduced` emits `std::move` under
the same condition. C++ will not: implicit move covers `return xs;`, not `xs`
inside the returned expression. Whether it matters depends on the
representation — for `std::vector` it is O(1) against an O(n) copy; for
`std::array<T, K>` it is element-wise and the gain is only the lost allocation.

Soundness needs the construction to provably run once for that value, and a
sibling-statement check alone does not give that. Three guards: the use must not
re-execute (a `while` condition is an expression, so it repeats *within* a
block), its definition must be a sibling statement, and the value must not leave
through a phi *outside* its own object — a phi is not a use site, so sole-use
says nothing about the read after a merge that carries the value elsewhere.
`std::array` hides all three, since moving from one leaves it readable, so every
runtime test uses an unsized list whose `vector` move empties the source.

**A projection whose slot is replaced.** *Deliberate.* `row = xss[i]` binds a
reference, which is what lets `for a, b in zip(...)` over nested lists unbox at
all. A C++ reference follows the *slot* while FPy keeps referring to the list that
was in it, so any `xss[i] = <list>` anywhere in the function rules it out.
Function-wide by choice: nothing else in the analysis is flow-sensitive.

**A materialized `range` is copied into the name bound to it.** *Open.*
`std::vector<int64_t> t = _tmp1;` — seven instances, all
`[... for ... in range(len(xs))]` after `ZipElim`. `CompToLoop._inlinable` keeps
a `range` out of a name so it stays fused, but only where its operands are
atoms, and `len(xs)` is not one. It is pure, though, so the gate is stricter
than the property it is testing for; widening it to pure operands fuses these.

**Not planned:** interprocedural precision beyond retention — a caller-driven
representation choice with the callee specialized per argument representation. It
needs one body per representation vector and nothing has measured a gain that
justifies it. If ever wanted, the representation must join the specialization key
rather than be patched on after, since storage is decided per spec.

### Aliasing: what still refuses

`format_infer` consumes `Alias` and replays a region's inserts against every bound
written through it, so a store through one name widens all of them. Two shapes
still refuse:

- **A callee's list parameter.** Widening it moves the signature out from under
  the call site. The call-site workaround above does not apply, because here the
  argument is the *narrower* one.
- **A list aliased through a tuple**, in the backend. `_gen_list_into_tuple` gets
  a sound bound, but the emitter computes a stale element type for the tuple
  field and wants a `float` somewhere.

Two imprecisions, neither unsound: flow-sensitivity comes from an *exclusion*
(the two defs `reaching_defs` already refreshes are skipped) rather than by
construction — the alternative is for `IndexedAssign` to refresh a def for every
may-alias, which would benefit every analysis. And `Alias` runs without escape
summaries, so a callee that provably does not retain its argument still forces a
widening at the caller.

### Whether integer narrowing earns its keep

The backend narrows a `RealType` value to an integer type when its bound says
every value is a small integer — `acc = 0.0` becomes `uint8_t`. This was the root
cause of several wrong answers, since an integer holds neither a signed zero nor
a NaN; it no longer is, because the narrowing is gated on the bound genuinely
excluding those, in the analysis rather than here.

What remains is a question of size, not correctness. Measured: 21 of the corpus's
112 list element types are value-narrowed reals, plus scalars like `uint8_t acc`
inside FP64 functions. Dropping it would trade smaller objects for emitted code
that says `double` wherever FPy says real. *Integer*-typed FPy values keep integer
storage regardless; `range(...)` needs an integer list and must stay one — that is
what an early measured attempt broke.

### Narrowing inside `std::accumulate`, so `Sum` can fuse

`ReduceFusion` fuses `any` / `all` over a comprehension into one loop, skipping
the intermediate vector. `Sum` / `AMin` / `AMax` pay the same allocation cost but
are not fused, because the fused shape needs an implicit narrowing inside
`std::accumulate` that `_maybe_cast` rejects at an ordinary assignment. Today
`sum([x * x for x in xs])` still materializes:

```cpp
std::vector<double> _tmp1 = std::vector<double>(0);
for (double x : xs) { _tmp1.push_back((x * x)); }
return std::accumulate(_tmp1.begin(), _tmp1.end(), static_cast<double>(0));
```

The emitter side is the blocker; see `fpy2/transform/reduce_fusion.py`'s module
docstring for the transform side.

### A slot store's refusal is untested

`_emit_at(..., cannot_convert=True)` refuses a value whose own storage cannot
reach the slot's. Nothing exercises the refusal: disabling it leaves all 612
backend tests passing, and three attempts at a witness compiled instead, because
`format_infer`'s alias replay widens a slot's element to take the store before
the check runs. The same shape as `_convert_storage`, reached 37 times with
`src == want` every time.

Not evidence the check is wrong — evidence the storages agree by the time
anything asks. A witness needs a slot whose element storage is fixed by
something outside the store: a parameter's ABI, or a callee's return.

### `fpy2/backend/cpp/README.md`

A short package README pointing at this file and listing the public surface
(`CppCompiler.compile` / `headers` / `helpers` / `prelude`, exception types).

## Comprehension lowering: in the pipeline

`_to_statement_form` runs `Hoistable` and `CompToLoop` to a fixpoint, so every
comprehension is an `fp.empty` allocation plus a `for` loop before the emitter
sees one, and `_visit_list_comp` is a tripwire that names a backend bug. Why the
lowering is total, and what it cost, is §8 of
[backend-independence.md](backend-independence.md).

## Out of scope

- Linking an external multi-precision library.
- Emitting exact arithmetic where format inference reports `REAL_FORMAT`; those
  programs are rejected with an error naming the symbolic expression.
- `with FP64 as ctx:` — binding the active context to a name.

