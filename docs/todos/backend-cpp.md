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
- `storage.py` — storage-type ladder, format-containment helpers.
- `storage_infer.py` — per-SSA-def storage assignment via union-find.
- `unbox.py` — which lists may drop the `std::shared_ptr` handle.
- `types.py` — `CppScalar` / `CppList` / `CppTuple` and source formatting.
- `target.py`, `utils.py` — target description and header preamble.  There is no
  runtime: `CPP_HELPERS` is empty, and every spelling the emitter produces is
  `std::`.

Read the design before changing anything; [Open issues](#open-issues) is at
the bottom.

Everything C++-specific lives here. Analyses this backend depends on but does
not own have their own documents: `round-elim.md`, `array-size-symbolic.md`,
`array-size-integer-exactness.md`.

The correctness criterion: *if the compiler succeeds, the emitted C++ must
compile and must behave as the FPy interpreter does wherever FPy's semantics are
defined.* A refusal is always acceptable — the compiler may be limited — so a
shape it cannot handle should raise, never miscompile.

The qualifier is load-bearing. FPy has undefined behavior, and the interpreter's
checks are not the contract: it raises on an out-of-range subscript and on a
mismatched-length `zip`, but neither is promised, so the backend owes nothing
there. One shape does violate the criterion — an operand whose emission is not
pure escapes the construct guarding it; see [Open issues](#open-issues).

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
`reaching_defs.same_object_defs` is that rule, and `storage_infer.py` unions
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
  → DefineUse
  → ContextUse                 # resolves with-block contexts
  → ArraySizeInfer             # FormatInfer needs it for bounded iteration
  → FormatInfer                # rounding format per def/expr
  → StorageInfer               # storage per SSA def
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

### An operand emitting statements escapes its guard

**A defect, not a limitation — the only one in this list.** `_emit_guarded_block`
and `_visit_if_expr` interpolate an expression visitor into an f-string, so
Python evaluates the visitor first and whatever it wrote through
`writer.add_line` lands *before* the construct being guarded. Any operand whose
emission is not pure — `_bind_operand`, a rounding assertion, a reduction's loop
— therefore leaves the scope it belongs to. Three shapes, each verified by
compiling and running the output:

**A `while` condition is evaluated once.** The loop then tests a stale name.

```python
y = x
while max([y, 0.0]) > 0.0:
    y = y - 1.0
return y
```
```c++
double y = x;
auto&& _tmp1 = std::array<double, 2>{{y, static_cast<double>(0)}};
double _tmp2 = _tmp1[0];
for (size_t _tmp3 = 1; _tmp3 < _tmp1.size(); ++_tmp3) { /* … max … */ }
while ((_tmp2 > static_cast<double>(0))) { y = (y - static_cast<double>(1)); }
```

`f(3.0)` returns `+0.0` in the interpreter; the binary does not terminate.

**A ternary arm's assertion runs on the untaken path.** With
`y = 0.0 if x > 1e30 else fp.cast(x)` under `FP32`, the cast's
losslessness assertion is hoisted out of the conditional:

```c++
float _tmp1 = static_cast<float>(x);
assert(x == _tmp1 || (std::isnan(x) && std::isnan(_tmp1)));
float y = ((x > 1e+30) ? static_cast<float>(0) : _tmp1);
```

`f(1e300)` returns `+0.0` in the interpreter; the binary aborts.

**An `and` / `or` tail is evaluated past the short circuit.**
`x > 1e30 or fp.cast(x) > 0.0` hoists the same assertion ahead of the `||`.
`f(1e300)` is `True` in the interpreter; the binary aborts.

All three are in territory where FPy's semantics *are* defined — loop
termination, a branch not taken, a short circuit — so unlike the subscript case
below, the criterion at the top of this document is violated.

**The fix is a refusal, and the predicate already exists.** `_is_pure_cond` is
exactly "does this expression emit statements", but it is wired only to
`_flattenable`'s `else if` decision. Gating `while` conditions, `IfExpr` arms and
boolean tails on it turns three miscompiles into refusals. Worth sizing first:
the refusal rejects programs that compile today, so measure how many corpus
functions it newly turns away.

Lowering these positions would also make the shapes unreachable — see
[backend-independence.md](backend-independence.md) §1, where they are class 3 —
but unreachable is not impossible: the emitter would still hold the shape,
guarded only by an upstream pass having run.

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

**A list a local name and a container both hold.** *Open.* `xs = [n, n]; return
(xs, 1.0)` keeps its handle — the name and the tuple's field are two places, even
though the name is dead after the return. Inline as `return ([n, n], 1.0)` it
unboxes. Closing it needs liveness, not just a sharing verdict, *and* the emitter
must learn to **move** into the container: `std::make_tuple(xs, 1)` copies a value
where it merely bumps a refcount for a handle. The two have to land together or
the change makes things slower.

**A projection whose slot is replaced.** *Deliberate.* `row = xss[i]` binds a
reference, which is what lets `for a, b in zip(...)` over nested lists unbox at
all. A C++ reference follows the *slot* while FPy keeps referring to the list that
was in it, so any `xss[i] = <list>` anywhere in the function rules it out.
Function-wide by choice: nothing else in the analysis is flow-sensitive.

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

## Considered: lowering comprehensions in the pipeline

`CompToLoop` (`fpy2/transform/comp_to_loop.py`) rewrites a comprehension into an
`fp.empty` allocation plus a `for` loop. **Not wired into this pipeline, and the
emitter keeps its own support**, measured both ways.

The emitter already performs the same lowering, and better: `_emit_list_comp_at`
opens a list build, emits one `for` per clause, and appends, so it never needs a
length up front — `_open_list_build` picks `std::array` filled through a running
index where the length was proven, `std::vector` with `push_back` otherwise.
That is why every clause shape compiles here, including the ragged flatten the
FPy-level lowering leaves alone.

```cpp
// [x * 2 for x in xs], xs proven length 3
std::array<double, 3> _tmp1{};  size_t _tmp2 = 0;
for (double x : xs) { _tmp1[_tmp2++] = (x * 2); }
```

Lowering first is a regression today: a multi-clause comprehension loses its
`std::array` for a `std::vector`, because `ArraySizeInfer._const_int` folds
`len(xs)` but not arithmetic over it, so `fp.empty(len(xs) * len(ys))` has no
static size — see
[array-size-integer-exactness.md](array-size-integer-exactness.md). The
single-clause form also leaves a dead `auto&& _tmp1 = xs.size();`.

**Deleting the emitter's support** — so that only statements introduce
identifiers, one less case for storage selection — buys ~90 lines
(`_visit_list_comp`, `_emit_list_comp_at`, `_open_comp_loop`, the `_emit_at`
case, two `storage_infer` match arms, one `_ALLOC_EXPRS` entry) and costs two
program classes that compile today: a dependent clause list, and a comprehension
in an `IfExpr` branch. It would also leave the invariant local — 22 files outside
this backend still handle `ListComp`. The invariant is a language property, so
the route to it is making `CompToLoop` total rather than lowering per-backend;
what that needs is the last gap in
[rounding-axes.md](rounding-axes.md).

Were it ever wired in, the slot is after `ReduceFusion` — which pattern-matches a
syntactic `ListComp` — and before `Specialize`, inside `optimize=True`.

## Out of scope

- Linking an external multi-precision library.
- Emitting exact arithmetic where format inference reports `REAL_FORMAT`; those
  programs are rejected with an error naming the symbolic expression.
- `with FP64 as ctx:` — binding the active context to a name.

