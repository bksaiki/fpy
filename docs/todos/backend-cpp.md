# C++ backend — design notes & open TODOs

The cpp backend (`fpy2/backend/cpp/`) compiles FPy to C++ end-to-end
across scalar arithmetic, control flow, lists, tuples, in-place
mutation, the `<cmath>` family, and rounding-context boundaries.  Unit
coverage lives at `tests/unit/backend/cpp/`.

Module layout:

- `compiler.py` — public `CppCompiler`, pipeline orchestration.
- `emitter.py` — AST walker that produces C++ source.
- `ops.py` — per-op tables of supported C++ signatures.
- `storage.py` — storage-type ladder, format-containment helpers.
- `storage_infer.py` — per-SSA-def storage assignment via union-find.
- `types.py` — `CppScalar` / `CppList` / `CppTuple` and source-string
  formatting.
- `utils.py` — header / helper preamble.

The remaining work is at the bottom under [Open TODOs](#open-todos).
The sections in between are the design pieces that any of the open
TODOs build on — read the design before making changes.

## Strategy

The backend was built up in **vertical slices**: each commit got a
small subset of the language compiling end-to-end before broadening
coverage.  The unit suite stays green at every commit.

## Design

### Storage vs. rounding

The core insight is that **storage type and rounding format are
separate**:

- **Rounding format** (from `FormatInfer`): the smallest format that
  bounds the value of an expression at runtime.  Per-expression.
- **Storage format**: the C++ type used to *hold* the value in a
  variable.  Per-definition.  Must contain the rounding format of
  every expression assigned into that variable (across phi merges
  and SSA rebinds).

A storage choice is **valid** iff for every expression assigned into
the variable, the rounding format is contained in (representable by)
the storage format.  We pick the smallest valid storage type from a
fixed ladder: `int{8,16,32,64}_t` / `uint*` / `float` / `double`.
Unbounded integer formats (`MPFixedFormat` with `expmin >= 0`) fall
back to `int64_t`; non-abstractable / `REAL_FORMAT` results are
rejected with an error pointing at the offending expression.

### SSA rebinds → fresh C++ variables

The cpp emitter is free to give every SSA def its own C++ variable.
The *one* constraint is that defs joined by either of two coalescing
edges must share storage:

- **Phi edges** — a phi merge means both incoming defs write to the
  same C++ variable.
- **In-place mutation edges** — `xs[i] = e` is in-place per the FPy
  interpreter (`interpret/byte.py:_visit_indexed_assign`), so the
  SSA-fresh def at the `IndexedAssign` site is unioned with its
  `prev`.  Same C++ name, no widening, no rename.

`storage_infer.py` computes the partition with `Unionfind[Definition]`.
Function-argument and free-variable defs anchor a class to the bare
source name; other classes for the same source name pick up `_1`,
`_2`, … suffixes.

Per-class declaration shape:

- **Single-writer (`declare_at_assign`)** — the lowest-index writer
  is its declaration site; the emitter folds the type into the
  assign (`double t = (a + b);`, `for (int64_t i = 0; …)`,
  `double y = x;` followed by reassignments inside an `if1` body or
  loop).
- **Multi-writer hoisted (`hoists_before`)** — only required when a
  class has writers in disjoint branches of an `if/else` and the
  variable did not exist before the `if` (the merge phi has
  `is_intro=True`).  In that case the emitter hoists `T name{};`
  *just before* the responsible `IfStmt` (anchoring to the outermost
  responsible if when nested).

### Operation type matching

C++ doesn't have ad-hoc polymorphism for primitive numeric ops:
each operator is only defined on a fixed set of operand-type
combinations.  `ops.py` enumerates the supported signatures per
FPy op type.  Each `UnaryCppOp` / `BinaryCppOp` / `TernaryCppOp` is
parameterized by:

- *argument C++ types* (`CppScalar`) — the concrete scalar types
  the generated C++ feeds the operator.  ``int8_t + int8_t`` is
  one signature, ``float + float`` is another.
- *output context* (a full `Context` — format + rounding mode).
  Its C++ type comes from `choose_storage(out_ctx.format())`; the
  rounding-mode half is enforced separately by the `fesetround`
  boundary at `with` blocks.

At an op site the emitter consults:

- The **active rounding context** at the expression
  (`ContextUseAnalysis.find_scope_from_use(e).ctx`) — must equal
  the signature's `out_ctx`.
- Each operand's **C++ storage type** (from `StorageAnalysis`) —
  must equal the signature's `in_ty`.

Direct-match preferred; on miss the emitter falls back to the
all-active-context signature and casts every operand into the active
context's storage type.  **Every** type conversion goes through an
explicit `static_cast` — no reliance on C++ implicit promotion, even
for "lossless" widenings.

The cast helper splits into two paths:

- `_maybe_cast` — used for *implicit* casts (op-dispatch fallback,
  comparison cast-to-supremum).  Rejects the compilation when the
  conversion is lossy (`not scalar_fits_in(arg_ty, target_ty)`).
  The error tells the user to wrap the operand in `fp.round(...)`
  or pick a context that holds the value.
- `_explicit_cast` — used for *user-explicit* casts (`Round` /
  `RoundExact` lowerings, `xs[static_cast<size_t>(i)]` subscripts).
  Emits the `static_cast` unconditionally; the user has already
  said they accept the conversion.

So a lossy implicit cast like FP64 → FP32 (or int64 → FP64) fails
to compile until the program wraps the wider operand in
`fp.round(...)` — making the rounding step part of the source.

The default table covers `Add`/`Sub`/`Mul`/`Div`, `Neg`, `Abs`, all
of `<cmath>` (transcendental + algebraic + FP rounding helpers),
`Pow`/`Hypot`/`Atan2`/etc., and `Fma` — across FP32 / FP64 with each
of the four `fesetround`-supported rounding modes (RNE / RTZ / RTP /
RTN), plus the integer ladder (`SINT8…64`, `UINT8…64`, `INTEGER`)
where applicable.

### Context boundaries

The active rounding context at every `FuncDef` / `ContextStmt` site
comes from `ContextUseAnalysis`.

**Validation is gated on use.**  A scope is only validated when
some primitive op (`NullaryOp` / `UnaryOp` / `BinaryOp` /
`TernaryOp` / `Call`) actually dispatches under it — i.e., when
`ctx_use.uses[scope]` is non-empty.  Scopes with no uses (e.g.,
a function-level scope where every op lives inside a nested `with`,
or a `with` block that holds an exotic context but only does
context-free work like list indexing) are skipped entirely: no
validation, no `fesetround`.  This means programs without a
rounding-context use don't need a supported function-level context,
and `with UnsupportedCtx:` blocks compile freely as long as nothing
inside them dispatches under that scope.

When a scope *is* used, validation runs:

- The context must be a concrete :class:`Context` — symbolic
  context variables (`ContextUse` falls back to a fresh `NamedId`
  when partial-eval can't pin one) are rejected at the
  introduction site.
- **Float contexts** must use a rounding mode supported by
  `fesetround` (RNE / RTZ / RTP / RTN).  `_visit_function` /
  `_visit_context` save / set / restore `fenv` only when the active
  mode actually *changes*.  The active mode is tracked on
  `_current_rm: RM | None`:
  - At function entry it's seeded from the function-level scope:
    a concrete FP context's RM (the FPy contract says the caller
    delivers it), or `None` for a symbolic / integer / unsupported
    outer scope.
  - `None` means "unknown" — any nested concrete-FP `with` block
    must emit `fesetround` unconditionally to recover certainty,
    matching the user's "safest option" rule.
  - When `_current_rm` is concrete and matches the target, the
    `with` block is a no-op at the C++ level (no fenv noise for
    plain `with FP64:` under an FP64 function).
- **Integer contexts** must use RTZ — that matches C++'s integer
  truncation, and other modes would require per-operation
  emulation.  No runtime support emitted.

`Round`, `RoundExact`, `Cast`:

- `Round(arg)` lowers to `static_cast<target>(arg)` — the cast's
  rounding mode comes from the surrounding `fesetround` boundary,
  not the cast itself.
- `RoundExact(arg)` adds a runtime assertion that the cast was
  lossless: cast → bind to a temp →
  `assert(arg == tmp || (std::isnan(arg) && std::isnan(tmp)))`
  for FP operands (NaN-aware) or `assert(arg == tmp)` for purely
  integer pairs.
- `Cast(arg)` is the identity — analysis-only annotation, no
  generated code.  Same-type `Round` / `RoundExact` short-circuit
  to the identity.

### Pipeline

```
FuncDef
  → Monomorphize.apply_by_arg(arg_types)   # ground type vars
  → DefineUse.analyze
  → ContextUse.analyze                     # resolves with-block ctxs
  → ArraySizeInfer.analyze
  → FormatInfer.analyze                     # rounding format per def/expr
  → StorageInfer.infer                      # storage class per SSA def
  → emit C++
```

`ArraySizeInfer` is needed because `FormatInfer` consults it for
bounded-iteration mode (loops with statically-known length).
`ContextUse` builds a `site → ContextScope` lookup that the emitter
also consults for `with` boundaries and per-op active contexts.

### Translation-unit preamble

`CppCompiler.compile` returns a function definition only so
single-function tests can use exact-string equality.  Callers that
want a complete translation unit pull `headers()`, `helpers()`, or
`prelude()` (the two combined) explicitly.  Header coverage tracks
exactly what the emitted code uses (`<cassert>` for assertions,
`<cfenv>` for `fesetround`, `<cmath>` for transcendentals,
`<cstdint>` for fixed-width ints, `<numeric>` for `accumulate`,
`<vector>` and `<tuple>`).

Helpers is currently empty — cpp doesn't yet need custom runtime
support — but the slot exists for future additions
(see [TODOs](#open-todos)).

## Open TODOs

These items are queued for a future pass.  The design above defines
the constraints they need to fit; pick whichever is most relevant
to your changes.

### Bounds-checked list operations

`_visit_list_ref`, `_visit_list_slice`, and `_visit_indexed_assign`
currently emit raw `xs[i]` / iterator arithmetic with no out-of-
range check.  The FPy interpreter is strict (raises on out-of-range
indices); cpp should match.  Likely shape: a small bounds-checked
subscript helper added to `CPP_HELPERS`, called from each subscript
site.  See the TODO comments at `emitter.py:_visit_list_ref` and
`_visit_list_slice` for current behavior.

### RAII fenv guard

When a function-level `fesetround` is active and the body executes
`return X;`, the trailing `fesetround(prev)` is dead code — the
caller's rounding mode is leaked.  Best fix is an RAII guard
emitted as part of the helper preamble:

```cpp
struct __cpp_FenvGuard {
    int prev;
    explicit __cpp_FenvGuard(int new_rm) : prev(std::fegetround()) {
        std::fesetround(new_rm);
    }
    ~__cpp_FenvGuard() { std::fesetround(prev); }
};
```

`_visit_function` and `_visit_context` would then declare a guard at
the start of the affected scope; the destructor runs on every exit
path (including `return`).  Drop the manual save / set / restore
emission once the guard lands.

### Classification ops + nary `Min` / `Max`

`IsFinite` / `IsInf` / `IsNan` / `IsNormal` / `Signbit` return
`bool` — they don't fit the existing `(in_fmt, out_ctx)` shape (the
output isn't a numeric context).  Either add a `bool`-output slot to
the op-table classes, or special-case these in `_visit_unaryop`
alongside `Len` / `Sum` / `Enumerate`.

`Min` / `Max` are nary in FPy.  These reduce naturally to pairwise
`std::fmin` / `std::fmax` in `_visit_naryop`.

### Round-trip tests against `cc -std=c++17`

Mirror `tests/infra/backend/cpp.py`: compile each test program with
the cpp backend, run it through `cc -std=c++17`, link, and exec —
asserting the runtime result matches the FPy interpreter's.  Catches
header-omission bugs and silent dispatch errors that pure-string
unit tests miss.

### `fpy2/backend/cpp/README.md`

A short README in the package directory pointing at this file and
listing the public surface (`CppCompiler.compile` / `headers` /
`helpers` / `prelude`, exception types).

## Out of scope (for the first pass)

- Linking against an external multi-precision library (mpfr, etc.).
- Generating exact arithmetic where the format-inference fallback
  reports `REAL_FORMAT` — those programs are rejected with a clear
  error pointing at the symbolic expression.  (Future work: emit a
  multi-precision fallback rather than rejecting.)
- Binding the active rounding context to a name in `with`
  (`with FP64 as ctx:` …) — the emitter currently rejects.
