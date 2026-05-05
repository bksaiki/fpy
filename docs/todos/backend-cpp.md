# C++ backend

I'd like to replace the current C++ backend with a new implementation.
In particular, I'd like to take advantage of the new format inference algorithm
that we just developed to infer the "minimal" number format that can be used to represent the data.

Here are a few new insights that might be helpful in the implementation:
 - storage and rounding are separate, albeit related, concerns: while FPy exposes how to _round_ operations,
   in C++ we also need to be able to _store_ data in a specific format. As long as the inferred format
   is contained in some data format for each expression, and each operation can be rounded
   accordingly with the correctly-typed inputs, then compilation is valid
 - mutation in FPy can be compiled as a new variable in C++. This is especially helpful
   for loops where the loop variable may be in a wider format.
 - use type checking, context-use, and format inference to determine the format of each variable and expression, and use that information
   to generate the appropriate C++ code with the correct types and rounding operations.

Make a plan for the implementation of the new C++ backend, write the todo list in this directory, and begin working on it.
Adhere to the following steps:
- when working on the new backend, you can put it under `fpy2/backend/cpp2` to avoid conflicts with the current implementation
- feel free to edit this Markdown file to add more details to the plan as you go along.
- make commits after each significant step in the implementation, and write clear commit messages to document your progress.
- please use testing to validate the correctness of your implementation at each step

## Plan

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

#### Phase 0 — Skeleton
- `fpy2/backend/cpp2/__init__.py` exporting `Cpp2Compiler`,
  `Cpp2CompileError`.
- `fpy2/backend/cpp2/compiler.py` with the public API class and a
  `compile(func, …) -> str` stub that raises NotImplementedError.
- A minimal test in `tests/unit/backend/cpp2/test_compile.py` that
  asserts the import works and the stub raises.

#### Phase 1 — Pipeline + storage selection
- Wire all pre-analyses through `Cpp2Compiler.compile`.
- Define a `StorageType` ladder (mirroring `cpp/types.py:CppScalar` but
  driven by format inference rather than concrete contexts).
- Implement `_choose_storage(def_bound: FormatBound) -> StorageType`:
  smallest scalar type containing the inferred bound.  Default fallback
  is `double` for unknown / non-abstractable formats.
- Aggregate across SSA defs that share a name (loop phi widening).
- Test: storage selection on a few hand-built `FormatBound` instances.

#### Phase 2 — Vertical slice: scalar arithmetic
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

#### Phase 3 — Booleans, comparisons, control flow
- `if` / `if1` with phi assignment.
- `while` with phi declaration before the loop.
- `for` over `range`/list (plain integer counter for known sizes; iterator
  pattern otherwise).
- Boolean ops, comparisons.
- Test: compile a mix of branching/looping programs.

#### Phase 4 — Lists & tuples
- `std::vector<T>` for lists, `std::tuple<T...>` for tuples.
- List literal, list comprehension, indexing, slicing (with strict
  bounds — match the interpreter), `IndexedAssign` as fresh C++ var.
- Built-ins: `len`, `range`, `enumerate`, `zip`, `sum`.
- Test: round-trip list-shaped programs.

#### Phase 5 — Rounding & contexts
- `with FP32: …` blocks: emit explicit casts at the boundary, set
  rounding mode if needed.
- `Round`, `RoundExact`, `Cast` operations.
- Algebraic / transcendental ops: dispatch to `<cmath>`.

#### Phase 6 — Polish
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
