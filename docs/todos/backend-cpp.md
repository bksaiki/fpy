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

The emitter's input is in **statement form**: `fpy2.transform.ANF` runs last in
`specialize()`, so every operand is a name, a literal or a nullary constant, and
the emitter never invents a place for one.  Only aggregates are left nested.
ANF does not create the statement slots it needs — `fpy2.transform.Hoistable`
does, earlier in `specialize()` — and ANF raises where a sealed position holds
something it would itself have to name.

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
                                 # and every comprehension but the ragged one
                                 # lowered into the loop that is that slot
  → RoundElim                  # (optimize only)
  → ANF                        # statement form; every operand is an atom
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

### Settled: an operand emitting statements escapes its guard

**Was a defect; now impossible by construction.** `_emit_guarded_block` and
`_visit_if_expr` interpolate an expression visitor into an f-string, so anything
it wrote through `writer.add_line` landed *before* the construct being guarded.
Three shapes, each verified by compiling and running the output:

- **a `while` condition evaluated once**, so the loop tested a stale name —
  `while max([y, 0.0]) > 0.0` did not terminate where the interpreter returned;
- **a ternary arm's assertion on the untaken path** —
  `0.0 if x > 1e30 else fp.cast(x)` aborted where the interpreter returned;
- **an `and` / `or` tail past the short circuit** — same assertion, same abort.

All three are in territory where FPy's semantics *are* defined, unlike the
subscript case below.

Closed two ways. `fpy2.transform.Hoistable` lowers each position before codegen,
so none reaches the emitter — and `_emit_inline` refuses if one ever does,
comparing the writer's line count around the emission rather than approximating
from the syntax. Unreachable *and* impossible: the lowering fixes real programs,
the tripwire survives a future change to the pass.
`tests/unit/backend/cpp/test_statement_form.py` runs all three witnesses and
checks the refusal with both passes monkeypatched to the identity — both,
because patching out `ANF` alone leaves `Hoistable` to empty the positions and
every witness would pass without witnessing anything.

An `if` / `if1` condition is deliberately not gated: it runs once, just before
the branch, so its statements belong in the enclosing block. That is why
`_emit_guarded_block` now takes its condition already emitted.

### Statement form defeats the `2 ** n * x` fusion

`_emit_scale_by_pow2` matches a *syntactic* `Pow` as an operand of a `Mul`, and
`ANF` names the power first, so the fusion no longer fires:

```c++
float t14 = std::ldexp(1, static_cast<int>(t13));   // was: std::ldexp(x, t13)
float _t8 = (t14 * x);
```

**Not an accuracy regression.** The peephole only fires where `_pow2_is_exact`
and `_result_fits_ctx` both hold — that is, where the power *and* the product
are exact — so the two forms agree bit-for-bit. `test_lowered_roundtrip.py`
confirms it across fourteen formats. The cost is one multiply and one extra
`float`, and the scale still runs in `float` with no widening.

Two fixes were tried and rejected:

- **Resolving the power through its name** in the emitter
  (`defining_expr(scale)`) is *unsound*: `StorageInfer` gives two definitions of
  one source name a single C++ variable, so re-emitting the exponent at the
  product reads whatever a later branch put there. `k = n; p = 2 ** k; if c: k =
  m; return x * p` computed `x * 2**m`. Pinned by
  `test_the_exponent_is_not_re_read_at_the_product`.
- **Teaching `ANF` not to name a `Pow`** works, but lets one backend's peephole
  restrict a backend-independent pass — and only the shape that happened to have
  a test, since `_fold_rounded_literal`, `_concrete_int_of` and
  `_range_counter_scalar` match syntax the same way.

The real fix is to strength-reduce before codegen, which needs a language-level
scale operation: FPy has no `ldexp`/`scalb`, so there is nothing for an FPy-level
transform to rewrite *into*. See
[backend-independence.md](backend-independence.md) §6.

### Unchecked subscripts are not a bug

**Settled: an out-of-range subscript is undefined in FPy.** `_visit_list_ref`,
`_visit_list_slice`, and `_visit_indexed_assign` emit raw `xs[i]`, and that is the
intended lowering. The interpreter happens to raise on `xs[10]` over a shorter
list, but that is an artifact of how it is written, not a promise the language
makes — so a backend is free to do anything, and a raw subscript is the normal
C/C++ idiom.

Recorded because the shape invites the opposite conclusion: it looks like the
emitted code disagrees with the interpreter, so it reads as the criterion above
being violated. It is not. Do not add a checked-subscript helper on this
reasoning; the criterion applies where FPy's semantics are *defined*, and here
they are not.

(Should bounds checking ever be wanted, it would be a debug-build feature like the
rounding assertions, and it would want the array-size work first — the checks
worth keeping are the ones `array-size-symbolic.md`'s size equalities cannot
discharge statically.)

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

*Open.* Every refusal in this area already names the operator that fixes it —
`_require_cast_is_round` says "lower it first with `monomorphize ->
unfold_overflow -> float_to_fixed -> rescale_fixed`", and
`_emit_integral_round` names `rescale_fixed` and `unfold_overflow` by hand. The
proposal is to run them rather than print them, as a ladder keyed to what is
unsupported:

| unsupported | recovery |
|---|---|
| a rounding under a non-native *float* context | `unfold_special → unfold_overflow → float_to_fixed → rescale_fixed → simplify` |
| a rounding under a fixed-point context the backend cannot lower | `unfold_overflow → rescale_fixed → simplify` — the two the refusals name |
| *arithmetic* under either | `split_round` first: compute at an intermediate the op table has, re-round to the target, then the residual rounding is one of the rows above |

The third row is a deliberate double rounding, and it is safe for a specific
reason: `split_round` is gated on the correct-double-rounding table, so it
applies only where the two roundings compose to what the single one gave, and
declines otherwise. That is the difference between this and simply computing
wide and truncating.

**What it buys is coverage, not a smaller backend.** Measured on the
`test_lowered_roundtrip` programs: after the full sequence
`_emit_integral_round`, `_emit_integral_value` and `_bound_test` all still fire
— the libm call and the bound assertion are the emitter's work either way. The
gain is that programs which today refuse would compile.

What it needs:

- **Detection before analysis.** The refusals fire during emission, after every
  analysis has run, which is too late to rewrite. The condition has to be found
  on the specialized AST first — `st.sites` plus a cursor per site, since
  `simplify` does not take cursors and a whole-program run would rewrite
  roundings that did not need it.
- **A place in the order.** After `RoundElim` and before `ANF`, for the reason
  in [backend-independence.md](backend-independence.md) §1: naming materializes,
  so a pass that folds or deletes has to precede one that names.
- **An opt-out.** This turns the compiler from a checker into a rewriter, and
  `float_to_fixed` states a value-class branch per site — on a program with many
  roundings that is a large amount of emitted code, so far unmeasured. A flag
  alongside `unsafe_cast_int` and `UnboxMode` keeps the refusal available for
  callers who would rather see it.

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

The discount asserts a *transfer*, so `_emit_deduced` emits `std::move` under
the same condition. C++ will not: implicit move covers `return xs;`, not `xs`
inside the returned expression. Whether it matters depends on the
representation — for `std::vector` it is O(1) against an O(n) copy; for
`std::array<T, K>` it is element-wise and the gain is only the lost allocation.

Soundness needs the construction to provably run once for that value, and a
sibling-statement check alone does not give that. Three guards: the use must not
re-execute (a comprehension body and a `while` condition are expressions, so
they repeat *within* a block), its definition must be a sibling statement, and
the value must not leave its branch through a phi — a phi is not a use site, so
sole-use says nothing about the read after the merge. `std::array` hides all
three, since moving from one leaves it readable, so every runtime test uses an
unsized list whose `vector` move empties the source.

**A projection whose slot is replaced.** *Deliberate.* `row = xss[i]` binds a
reference, which is what lets `for a, b in zip(...)` over nested lists unbox at
all. A C++ reference follows the *slot* while FPy keeps referring to the list that
was in it, so any `xss[i] = <list>` anywhere in the function rules it out.
Function-wide by choice: nothing else in the analysis is flow-sensitive.

**An aggregate bound from an emitter temp is copied.** *Open.*
`std::vector<std::vector<double>> result = _tmp1;` — a comprehension
materializes into a temp, then the real name copies it. Two instances in the
corpus today. Free while every list was a handle, O(n) since they became values,
and `binds_by_reference` does not reach it: the FPy `Assign`'s expression is a
`ListComp`, not a `Var`, so the temp is invisible to the binding rule. Either
build the comprehension into the target, or move from a temp that is dead by
construction.

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

### `fpy2/backend/cpp/README.md`

A short package README pointing at this file and listing the public surface
(`CppCompiler.compile` / `headers` / `helpers` / `prelude`, exception types).

## Comprehension lowering: in the pipeline, except the ragged case

`CppCompiler.specialize()` runs `fpy2.transform.Hoistable` and
`fpy2.transform.CompToLoop` to a **fixpoint** (`_to_statement_form`), so every
comprehension but one shape is an `fp.empty` allocation plus a `for` loop before
the emitter sees it. Neither pass is a fixpoint alone: `Hoistable` seals a
comprehension's element for want of a statement slot and `CompToLoop` makes the
loop that *is* that slot; `CompToLoop` declines a comprehension in a ternary arm
or a `while` condition for want of one, and `Hoistable` creates it. It
terminates because `CompToLoop` reports an edit only where it lowered a
comprehension and neither pass builds one.

Wiring it cost nothing: 202 corpus programs either way, 19 emitting differently.
Getting there needed four fixes, in
[comp-to-loop-wiring.md](comp-to-loop-wiring.md) — the lowering fills its
assignment target rather than copying an `acc` into it, `_emit_empty` allows an
allocation with fewer dimensions than the type's depth, a `range` iterable stays
inline so it is not forced into a value position, and `_const_int` folds
arithmetic over lengths it knows.

**The dependent clause list stays with the emitter.** Where a clause's iterable
reads an earlier clause's target the length is a sum rather than a product, and
`fp.empty` has nowhere to get it. `_emit_list_comp_at` needs no length up front —
`_open_list_build` picks `std::array` filled through a running index where the
length was proven, `std::vector` with `push_back` otherwise — so the ragged
flatten compiles here and nowhere else:

```cpp
// [x for xs in xss for x in xs]
std::array<uint8_t, 4> _tmp1{};  size_t _tmp2 = 0;
for (const std::array<uint8_t, 2>& xs : _tmp3)
    for (uint8_t x : xs)
        _tmp1[_tmp2++] = x;
```

`CompToLoop` *can* lower it — `apply(..., dependent=True)` builds the rows, sums
their lengths and flattens, which is what derived-semantics prescribes — and this
pipeline declines to ask. Measured, applying it loses `test_list_comp5`, the
corpus's one dependent comprehension, and changes no other emission: the
flatten's accumulator widens to `REAL_FORMAT` under the loop fixpoint. The
reasoning behind the refusal outlives that defect, though. A one-pass flatten
needs a **growable** list, and derived-semantics is explicit that no rule changes
a list's length — so there is no `append` to rewrite into, and this is a fuse for
a form FPy cannot state, like `_emit_scale_by_pow2` and `_for_header`.

Deleting the emitter's support therefore stays out of reach: it is ~90 lines
(`_visit_list_comp`, `_emit_list_comp_at`, `_open_comp_loop`, the `_emit_at`
case, two `storage_infer` match arms, one `_ALLOC_EXPRS` entry), and the only
route to it is a lowering that does not lose a program. See *Phase 7* in
[comp-to-loop-wiring.md](comp-to-loop-wiring.md) for what would flip that.

## Out of scope

- Linking an external multi-precision library.
- Emitting exact arithmetic where format inference reports `REAL_FORMAT`; those
  programs are rejected with an error naming the symbolic expression.
- `with FP64 as ctx:` — binding the active context to a name.

