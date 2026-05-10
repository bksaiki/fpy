# C++ backend - Plan

The strategy is **vertical slices**: get a minimal program compiling
end-to-end first, then expand the language coverage one node-kind at a
time.  Each slice is a commit point — the suite stays green throughout.

### Storage vs. rounding

The core insight is that **storage type and rounding format are
separate**:

- **Rounding format** (from `FormatInfer`): the smallest format that
  bounds the value of an expression at runtime.  Per-expression.
- **Storage format**: the C++ type used to *hold* the value in a
  variable.  Per-definition.  Must contain the rounding format of every
  expression assigned into that variable (across phi merges and SSA
  rebinds).

A storage choice is **valid** iff for every expression assigned into the
variable, the rounding format is contained in (representable by) the
storage format.  We pick the smallest valid storage type from a fixed
ladder: `int{8,16,32,64}_t` / `uint*` / `float` / `double` / a
multi-precision fallback.

### SSA rebinds → fresh C++ variables

`IndexedAssign` (and any mutation that produces a fresh SSA def) becomes
a *new C++ variable* whose storage may differ from the predecessor's.
A loop phi whose lhs and rhs have different formats yields a
**widened** C++ variable initialised before the loop.

### Operation type matching

C++ doesn't have ad-hoc polymorphism for the primitive numeric ops:
each operator is only defined on a fixed set of operand-type
combinations.  For `+` (modulo type promotion) those are roughly:

```
+ : uint{8,16,32,64}_t -> uint{8,16,32,64}_t -> uint{...}_t
+ : int{8,16,32,64}_t  -> int{8,16,32,64}_t  -> int{...}_t
+ : float              -> float              -> float
+ : double             -> double             -> double
```

For an FPy expression `a + b` evaluated under context `C`, with
`storage(a) = F_a` and `storage(b) = F_b`, emission is valid iff
`F_a` and `F_b` are both subsets of `C`'s storage type.  When they
aren't (e.g., mixed int + float, or narrower-than-context operands),
the emitter must either insert explicit casts at the operator's
boundary or reject as a type-shape error pointing at the offending
expression.  Each FPy op needs a table of supported C++ signatures so
the dispatch is explicit rather than relying on C++'s implicit
conversion rules — which can silently drop precision or change
rounding behaviour.

This is the cpp2 analogue of `cpp/types.py` + `ScalarOpTable`.

### Pipeline

```
FuncDef
  → Monomorphize.apply_by_arg(arg_types)   # ground type vars
  → DefineUse.analyze
  → TypeInfer.check
  → ContextUse.analyze
  → ArraySizeInfer.analyze
  → FormatInfer.analyze                     # rounding format per def/expr
  → choose storage formats per def          # NEW: storage analysis
  → emit C++
```

`ArraySizeInfer` is needed because `FormatInfer` consults it for
bounded-iteration mode (loops with statically-known length).

### Phase plan (each phase is a commit)

Progress markers: ✅ done, ⏳ in progress, ☐ todo.

#### Phase 0 — Skeleton ✅
- `fpy2/backend/cpp2/__init__.py` exporting `Cpp2Compiler`,
  `Cpp2CompileError`.
- `fpy2/backend/cpp2/compiler.py` with the public API class and a
  `compile(func, …) -> str` stub that raises NotImplementedError.
- A minimal test in `tests/unit/backend/cpp2/test_compile.py` that
  asserts the import works and the stub raises.

#### Phase 1 — Pipeline + storage selection ✅
- Wire all pre-analyses through `Cpp2Compiler.compile`.
- Define a `StorageType` ladder (mirroring `cpp/types.py:CppScalar` but
  driven by format inference rather than concrete contexts).
- Implement `_choose_storage(def_bound: FormatBound) -> StorageType`:
  smallest scalar type containing the inferred bound.  Default fallback
  is `double` for unknown / non-abstractable formats.
- Aggregate across SSA defs that share a name (loop phi widening).
- Test: storage selection on a few hand-built `FormatBound` instances.

#### Phase 2 — Vertical slice: scalar arithmetic ✅
Goal: compile this end-to-end.

```python
@fp.fpy
def f(x: fp.Real, y: fp.Real) -> fp.Real:
    with fp.FP64:
        return x + y
```

→
```cpp
double f(double x, double y) {
    return x + y;
}
```

- Emit function signature with monomorphized arg types.
- Emit `Assign` statements as `auto NAME = EXPR;`.
- Emit `Return` as `return EXPR;`.
- Emit `+`, `-`, `*`, `/`, `Neg`, `Abs` as native C++ operators / `std::fabs`.
- Test: compile + verify output is byte-identical to the expected
  string.

#### Phase 3 — Booleans, comparisons, control flow ✅
- 3a ✅ Hoist local declarations to the top of the function; assigns
  become reassignments (foundation for phi handling).
- 3b ✅ Bool literals and comparisons.
- 3c ✅ `if` / `if1` with phi assignment.
- 3d ✅ `while` with phi declaration before the loop.
- 3e ✅ `for` over `range(...)` (plain integer counter; iterables
  other than `range` and tuple-binding targets land with list
  support in Phase 4).

#### Phase 4 — Lists & tuples ✅
- 4a ✅ List literals (`[a, b, c]`), indexing (`xs[i]`), `len(xs)`.
- 4b ✅ List comprehension (`[expr for x in iter ...]`) — temp +
  range-based for; nested clauses supported.
- 4c ✅ For-over-list (`for x in xs:` — non-`range` iterables in
  `_visit_for`).
- 4d ✅ List slicing (`xs[a:b]`, with `[:b]`/`[a:]`/`[:]` defaults).
  Strict bounds checking against the interpreter is still TODO.
- 4e ✅ `IndexedAssign` (`xs[i] = e`) — in-place mutation, matching
  the FPy interpreter (`interpret/byte.py:_visit_indexed_assign`).
  C++ already supports in-place vector-element mutation, so the
  pipeline does **not** run `FuncUpdate`.  `IndexedAssign` is
  emitted directly as `xs[idx_chain] = e;`.  `StorageInfer` adds
  an in-place coalescing edge: any `AssignDef` whose site is an
  `IndexedAssign` is unioned with its `prev`, ensuring the post-
  mutation def shares a storage class with the pre-mutation def
  — same C++ name, no widening, no rename.
- 4f ✅ Tuples: `TupleExpr` (`std::make_tuple(...)`) and
  `TupleBinding` destructuring (`std::get<i>` extraction in
  `Assign`, `ForStmt`, and `ListComp`, with underscore skips and
  nested bindings).
- 4g ✅ Remaining built-ins: `sum` (`std::accumulate`),
  `enumerate` (vector of `tuple<I,T>` populated by an indexed
  for-loop), `zip` (variadic, vector of `tuple<T1,…,Tn>`).

#### Phase 5 — Rounding & contexts ✅
- 5a ✅ **Operation type tables.**  `fpy2/backend/cpp2/ops.py`
  enumerates supported C++ signatures per FPy op type.  Each
  signature is parameterized by *argument formats* and an
  *active rounding context* — inputs only carry value-range info
  (a `Format`); the operation rounds its mathematical result
  under the active context, so the output slot carries a full
  `Context` (format + RM).  Dispatch consults
  `ContextUseAnalysis.find_scope_from_use(e)` to get the active
  context and `format_info.by_expr` to get operand formats; a
  signature matches when its `out_ctx == active_ctx` and each
  operand's format ⊆ the corresponding `in_fmt`.  When no sig
  matches directly, the all-active-context sig is used and both
  operands are explicit-cast to that context's storage.  **Every**
  conversion goes through `static_cast` — no reliance on C++
  implicit promotion.  Same explicit-cast policy applies to
  comparisons (`_visit_compare` casts to scalar supremum) and
  vector subscripting (`xs[static_cast<size_t>(i)]`).  Coverage:
  `Add`, `Sub`, `Mul`, `Div`, `Neg`, `Abs` across native FP / int
  contexts (FP32 / FP64 × four RMs, SINT8/16/32/64, UINT8/16/32/64,
  INTEGER).  Algebraic / transcendental ops add their entries
  here as 5d lands.
- 5b ✅ Context boundaries.  Active rounding context is taken from
  :class:`ContextUseAnalysis` at every ``FuncDef`` / ``ContextStmt``
  site.  Programs whose contexts can't be statically resolved are
  rejected (``ConstFold`` runs first to resolve attribute
  references).  Float contexts must use a rounding mode supported
  by ``fesetround`` (RNE / RTZ / RTP / RTN) — the emitter saves /
  sets / restores ``fenv`` only when the active mode actually
  changes, so plain ``with FP64:`` blocks emit no fenv noise.
  Integer contexts must use RTZ (matches C++ truncation) and emit
  no runtime support.  **Known limitation**: when a function-level
  fesetround is active, a ``return`` inside the body skips the
  trailing ``fesetround(prev)``, leaking the mode to the caller.
  Best fix is an RAII guard, deferred to Phase 6 helpers.
- 5c ✅ `Round`, `RoundExact`, `Cast` expressions.  `Round` is a
  plain `static_cast<target>(arg)` — the cast's rounding mode
  comes from Phase 5b's `fesetround` boundary, not the cast
  itself.  `RoundExact` adds a runtime assertion that the cast
  was lossless: cast-bind-to-temp, then
  `assert(arg == tmp || (std::isnan(arg) && std::isnan(tmp)))`
  for FP operands (NaN-aware) or `assert(arg == tmp)` for purely
  integer operand pairs.  `Cast` is the identity — analysis-only
  annotation, no generated code.  Same-type round / round-exact
  short-circuit to the identity.
- 5d ✅ Algebraic / transcendental ops: dispatch each FPy op to
  its `<cmath>` counterpart through the op table from 5a.
  Coverage: roots (`Sqrt`, `Cbrt`); FP rounding (`Ceil`, `Floor`,
  `Trunc`, `RoundInt`, `NearbyInt`); trig + inverse + hyperbolic
  + inverse-hyperbolic (`Sin/Cos/Tan/Asin/Acos/Atan/Sinh/Cosh/Tanh/
  Asinh/Acosh/Atanh`); exp / log family (`Exp`, `Exp2`, `Expm1`,
  `Log`, `Log2`, `Log10`, `Log1p`); special (`Erf`, `Erfc`,
  `Lgamma`, `Tgamma`); `Logb`.  Binary: `Pow`, `Fmod`, `Remainder`,
  `Copysign`, `Fdim`, `Hypot`, `Atan2`.  Ternary: `Fma`.  Each is
  registered for FP32 and FP64 across all four `fesetround`-
  supported rounding modes.  Classification ops (`IsFinite`,
  `IsNan`, `Signbit`, …) and nary `Min` / `Max` aren't in this
  pass — they need a bool / nary slot in the table.

#### Phase 6 — Polish ⏳
- 6a ✅ Header / helper code emission.  `fpy2/backend/cpp2/utils.py`
  defines `CPP_HEADERS` (every `#include` the emitted code uses —
  `<cassert>`, `<cfenv>`, `<cmath>`, `<cstddef>`, `<cstdint>`,
  `<numeric>`, `<vector>`, `<tuple>`) and `CPP_HELPERS` (currently
  empty; reserved for future RAII fenv guard / bounds-checked
  subscript helpers).  `Cpp2Compiler` exposes `headers()`,
  `helpers()`, and a `prelude()` convenience.  `compile()` itself
  still returns the function definition only so exact-string tests
  in the rest of the suite remain stable.
- 6b ☐ Round-trip tests against a reference compiler
  (`cc -std=c++17`) — same pattern as
  `tests/infra/backend/cpp.py`.
- 6c ☐ Compare output to the existing `cpp/` backend on a corpus
  of programs.
- 6d ☐ Document the new backend's contract in a brief README
  under `fpy2/backend/cpp2/`.

### Out of scope (for the first pass)
- Linking against an external multi-precision library (mpfr, etc.).
- Generating exact arithmetic where the format-inference fallback
  reports `REAL_FORMAT` — those programs are rejected with a clear
  error pointing at the symbolic expression.  (Future work: emit a
  multi-precision fall-back rather than rejecting.)
