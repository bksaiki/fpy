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
- `unbox.py` — which lists may drop the `fpy::list` handle.
- `types.py` — `CppScalar` / `CppList` / `CppTuple` and source formatting.
- `target.py`, `utils.py` — target description, header / helper preamble.

Read the design before changing anything; [Open TODOs](#open-todos) is at the
bottom. Narrower questions have their own documents:

- `unboxing-gaps.md` — what stays boxed, and why.
- `cpp-narrower-variable-at-a-join.md` — two element types meeting at one place.
- `format-infer-aliasing.md` — an unsound bound the backend inherits.
- **`reals-in-integer-storage.md`** — the largest open correctness problem: a
  real narrowed into an integer holds neither `-0.0` nor NaN.
- `cpp-literal-tokens-and-sum.md` — three smaller disagreements between the
  emitted C++ and the interpreter.

The correctness criterion those last three are measured against: *if the
compiler succeeds, the emitted C++ must compile and must behave as the FPy
interpreter does.* A refusal is always acceptable — the compiler may be limited —
so a shape it cannot handle should raise, never miscompile.

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
(`storage_infer.place_floors` and `emitter._emit_at`).

### SSA rebinds → fresh C++ variables

The emitter is free to give every SSA def its own C++ variable. The one
constraint is that defs joined by a coalescing edge share storage:

- **Phi edges** — a phi merge means both incoming defs write one variable.
- **In-place mutation edges** — `xs[i] = e` mutates in place per the interpreter
  (`interpret/byte.py:_visit_indexed_assign`), so the SSA-fresh def at the
  `IndexedAssign` is unioned with its `prev`. Same name, no rename.

`storage_infer.py` computes the partition with `Unionfind[Definition]`. Argument
and free-variable defs anchor a class to the bare source name; other classes for
the same name take `_1`, `_2`, … suffixes.

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
  `_current_rm` tracks the mode the live `fenv` is guaranteed to hold, seeded at
  entry from the function-level scope (`None` = unknown, so a nested concrete
  `with` must emit `fesetround` unconditionally). A matching mode makes the
  `with` a C++-level no-op.
- **Integer contexts** must use RTZ — that is what C++ integer truncation does.

`Round` / `Cast`:

- `Round(arg)` lowers to `static_cast<target>(arg)`, whose rounding mode comes
  from the surrounding `fesetround` boundary. A **literal** argument is folded at
  compile time instead (`_fold_rounded_literal`) — C++ has no exact-real literal,
  so this is the only way an inexact constant is representable at all, and it
  also gets the mode the program asked for rather than whatever `fesetround`
  last left behind.
- `Cast(arg)` — the node `fp.cast` and `fp.round_exact` both parse to — is the
  same cast plus a runtime assertion that it was lossless, NaN-aware for FP
  operands. Same-type short-circuits to the identity.

### Pipeline

```
Module
  → Specialize                 # one FuncDef per (callee, ctx, arg formats)
  → DefineUse
  → ContextUse                 # resolves with-block contexts
  → ArraySizeInfer             # FormatInfer needs it for bounded iteration
  → FormatInfer                # rounding format per def/expr
  → StorageInfer  ×2           # storage per SSA def; see place_floors
  → Alias / Escape             # what may refer to what; what a callee retains
  → Unbox                      # handle or value, per alias region
  → emit C++
```

`StorageInfer` runs twice: `place_floors` needs to know which names the emitter
binds by reference, and that is a property of the storage analysis. `Specialize`
means a callee's formats follow its call site — so a callee called with a wider
list is automatically instantiated wider, which is the workaround
`cpp-narrower-variable-at-a-join.md` relies on.

### Translation-unit preamble

`CppCompiler.compile` returns a function definition only, so single-function
tests can use exact-string equality. Callers wanting a full translation unit
pull `headers()`, `helpers()`, or `prelude()`. `helpers()` carries the runtime:
`fpy::list<T>` (a `shared_ptr<vector<T>>`), `fpy::make_list`, and the nary
`fpy::min` / `fpy::max`. Headers track exactly what the emitted code uses.

## Open TODOs

### Bounds-checked list operations

`_visit_list_ref`, `_visit_list_slice`, and `_visit_indexed_assign` emit raw
`xs[i]` with no range check — `xs[10]` on a shorter list is undefined behaviour,
while the interpreter raises. Likely shape: a checked subscript helper in
`CPP_HELPERS`, called from each subscript site.

### RAII fenv guard

A `return` inside an active `fesetround` scope leaves the restore unreachable, so
the caller's rounding mode leaks:

```cpp
double leak(double x) {
    const auto _tmp1 = std::fegetround();
    std::fesetround(FE_TOWARDZERO);
    return (x + static_cast<double>(1));
    std::fesetround(_tmp1);          // dead
}
```

Fix with a guard in the helper preamble whose destructor runs on every exit path:

```cpp
struct __cpp_FenvGuard {
    int prev;
    explicit __cpp_FenvGuard(int rm) : prev(std::fegetround()) { std::fesetround(rm); }
    ~__cpp_FenvGuard() { std::fesetround(prev); }
};
```

`_visit_function` / `_visit_context` declare a guard instead of the manual
save / set / restore.

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

## Out of scope

- Linking an external multi-precision library.
- Emitting exact arithmetic where format inference reports `REAL_FORMAT`; those
  programs are rejected with an error naming the symbolic expression.
- `with FP64 as ctx:` — binding the active context to a name.

## Done since this document was written

Kept as a record so they are not re-proposed: classification ops
(`isfinite`/`isinf`/`isnan`/`isnormal`/`signbit`) and nary `Min`/`Max`; the
execute-and-bit-compare harness (`tests/infra/backend/cpp.py --mode run`, plus a
generated format matrix); and whole-call-graph optimization — `Specialize` runs
before the pipeline, so `ZipElim` and friends reach callees, not just the entry.
