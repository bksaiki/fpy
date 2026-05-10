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

#### Phase 5 — Rounding & contexts ☐
- 5a ✅ **Operation type tables.**  `fpy2/backend/cpp2/ops.py`
  enumerates the supported C++ signatures per FPy op type.  The
  emitter dispatches every `UnaryOp` / `BinaryOp` through the
  table: direct match if operand and result storage types already
  agree; cast-to-result fallback otherwise, with `static_cast`
  inserted *only* when the implicit C++ widening would actually
  drop precision (`scalar_fits_in` predicate).  Lossless cases
  like `U8 → F64` stay implicit so `(y - 1)` reads as expected.
  Coverage: `Add`, `Sub`, `Mul`, `Div`, `Neg`, `Abs`.  Algebraic
  / transcendental ops add their entries here as 5d lands.
- 5b ☐ `with FP32: …` blocks: emit explicit casts at the rounding
  boundary, set rounding mode (`fesetround`) if needed, restore on
  block exit.  Composes with 5a — the op-table check uses the
  innermost context's storage type.
- 5c ☐ `Round`, `RoundExact`, `Cast` expressions.
- 5d ☐ Algebraic / transcendental ops: dispatch each FPy op to its
  `<cmath>` counterpart through the op table from 5a.

#### Phase 6 — Polish ☐
- Header / helper code emission (mirror `cpp/utils.py:CPP_HELPERS`).
- Round-trip tests against a reference compiler (`cc -std=c++17`) — same
  pattern as `tests/infra/backend/cpp.py`.
- Compare output to the existing `cpp/` backend on a corpus of programs.
- Document the new backend's contract in a brief README under
  `fpy2/backend/cpp2/`.

### Out of scope (for the first pass)
- Linking against an external multi-precision library (mpfr, etc.).
- Generating exact arithmetic where the format-inference fallback
  reports `REAL_FORMAT` — those programs are rejected with a clear
  error pointing at the symbolic expression.  (Future work: emit a
  multi-precision fall-back rather than rejecting.)
