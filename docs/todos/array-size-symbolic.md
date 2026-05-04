# Array-size analysis: symbolic sizes via union-find

## Context

The current `array_size` analysis tracks list sizes as a flat lattice
`ArraySize: TypeAlias = int | None`.  This is precise for
compile-time-known constants but throws away **all** information about
runtime sizes that are statically constrained to be equal:

```python
def f(xs: list[fp.Real]) -> list[fp.Real]:
    ys = xs                       # len(ys) == len(xs), but we record None
    return ys

def g(xs: list[fp.Real]) -> list[fp.Real]:
    ys = [0.0 for _ in xs]        # len(ys) == len(xs)
    for i, x in enumerate(xs):
        ys[i] = x                 # in-bounds because lengths match
    return ys
```

Several downstream consumers would benefit from knowing
`len(ys) == len(xs)` symbolically:

- `mpfx/format_infer.py:Sum` could fall back to a runtime-bounded loop
  rather than refusing to expand when the size isn't a static `int`.
- A future bounds-check elimination pass could discharge `ys[i]` given
  `i < len(xs)` and `len(ys) == len(xs)`.
- A future allocation analysis could match a `[…]`-comprehension's
  allocation size to the iterable.

## Goals

1. Extend the lattice so equal-but-unknown sizes are tracked.
2. Preserve the existing API for consumers that only care about
   concrete `int` sizes.
3. Don't regress fixpoint convergence (loops in the inferred function
   must still terminate the analysis).
4. Stay scoped — pick the operations whose symbolic propagation
   actually buys precision; don't gold-plate.

## Non-goals

- Reasoning about size *expressions* (e.g., `len(xs) + 1`).  Only
  equivalence classes of "same size."
- Cross-function symbolic propagation.  Each analysis run is
  self-contained; symbols are fresh per-function.
- Recognising user-written `assert len(xs) == len(ys)` constraints in
  the first cut (it falls out for free if/when assertions are part of
  the analysis input, but isn't required day-1).
- Alias analysis.  The fresh-def model for `IndexedAssign` already
  handles size propagation across mutations of the *same* name; aliased
  variables remain a separate, pre-existing limitation.

## Design

### Lattice extension

```python
@dataclass(frozen=True)
class SymbolicSize:
    """Identifier for an equivalence class of unknown sizes."""
    id: int

ArraySize: TypeAlias = int | SymbolicSize | None
```

- `int n` — known concrete size (current behavior, unchanged).
- `SymbolicSize(id)` — unknown size, but tracked by an id that's
  shared with other sizes proven equal via the union-find.
- `None` — top (no information; equivalent to "we gave up").

### Union-find: `_SizeUF`

```python
class _SizeUF:
    """Tracks equivalence classes of symbolic sizes within one analysis run."""

    def fresh(self, key: object | None = None) -> SymbolicSize:
        """Mint a new symbol.  When *key* is non-None, repeated calls
        with the same key return the same symbol (used to anchor symbols
        to AST node identity for fixpoint convergence)."""

    def merge(self, a: SymbolicSize, b: SymbolicSize) -> None:
        """Merge two classes.  Monotone — only collapses, never splits."""

    def equiv(self, a: SymbolicSize, b: SymbolicSize) -> bool:
        """True iff *a* and *b* are in the same equivalence class."""

    def pin(self, s: SymbolicSize, n: int) -> None:
        """Record that the class equals concrete *n*.  Future
        :meth:`resolve` calls return *n*."""

    def resolve(self, s: SymbolicSize) -> int | None:
        """Return the class's pinned concrete size, or ``None``."""
```

The UF lives on the visitor and is exposed in `ArraySizeAnalysis`:

```python
@dataclass
class ArraySizeAnalysis:
    by_expr: dict[Expr, ArraySizeBound]
    by_def: dict[Definition, ArraySizeBound]
    ret_size: ArraySizeBound | None
    def_use: DefineUseAnalysis
    size_uf: _SizeUF              # NEW
```

Consumers can ask `info.size_uf.equiv(a, b)` to test equivalence.  A
helper `is_size_eq(b1: ArraySizeBound, b2: ArraySizeBound, uf) -> bool`
should wrap the common case of checking two `ListSize.size` fields.

### Mint sites — where new symbols are created

Symbols **must be keyed on AST node identity** so that revisiting a
node during the loop fixpoint returns the *same* symbol.  Without this,
every iteration mints a fresh symbol, the phi never stabilizes, and the
analysis diverges.  Concrete keys per site:

| Site                         | Key                | Notes                               |
|------------------------------|--------------------|-------------------------------------|
| `Argument: list[T]`          | the `Argument` node | One symbol per arg.                |
| Free var `list[T]`           | `(FuncDef, name)`   |                                    |
| `range(n)`                   | `Range1` node       | If `n` is static, use `int`.       |
| `range(start, stop)`         | `Range2` node       | Same idea.                         |
| `range(start, stop, step)`   | `Range3` node       | Same.                              |
| `Empty(…dims)`               | `Empty` node        | One symbol per dimension.          |
| `xs[a:b]`                    | `ListSlice` node    | If `a, b` static, pin to `b - a`.  |
| `xs + ys` (concat)           | concat-expr node    | If both static, pin to sum.        |

For operations that **preserve** length, propagate the existing symbol
without minting:

- `[expr for x in xs]` → same symbol as `xs.size`.
- `enumerate(xs)`      → same symbol.
- `zip(xs)` (1-arg)    → same symbol.
- `zip(xs, ys, …)`     → all-equal: see "Operation rules" below.
- `xs[i] = e`          → same symbol (length is unchanged by element
  mutation; this is a free win from the existing fresh-def
  IndexedAssign treatment).
- `reverse(xs)`        → same symbol.
- `ys = xs` (rebind)   → same symbol.

### Operation rules

Where the rule isn't "fresh symbol" or "propagate symbol," the visitor
needs to compute relationships explicitly.  Highlights:

- **`zip(xs1, …, xsN)`**: result size = min of input sizes.  If any
  input is concrete, the result size is at most that int; if all
  inputs are *equal* (UF-equiv) symbols, the result keeps that symbol.
  Otherwise: fresh symbol with no constraints.
- **List concatenation `xs + ys`**: result size = `len(xs) + len(ys)`.
  Concrete + concrete → int.  Anything else → fresh symbol (we don't
  track sums, see Non-goals).
- **`Empty(d1, …, dN)`**: each dimension's symbol is keyed on the
  corresponding `Expr` argument (or pinned to the int value if static).

### Join rule extension

```python
def _join_size(uf: _SizeUF, a: ArraySize, b: ArraySize) -> ArraySize:
    if a == b:                     # int == int, or same symbol
        return a
    match a, b:
        case None, _ | _, None:
            return None
        case int(), int():         # different concrete values
            return None
        case SymbolicSize(), SymbolicSize() if uf.equiv(a, b):
            return a
        case SymbolicSize() as s, int(n) | int(n), SymbolicSize() as s:
            return n if uf.resolve(s) == n else None
        case _:
            return None
```

### Phi handling in loops

When a phi merges `lhs` (pre-loop) and `rhs` (post-body) symbols, and
they're distinct, **the right action is `merge(lhs, rhs)`** — the loop
body is repeatedly assigning to the same name, so by the loop-back
edge those two symbols denote the same runtime size.  After the merge,
they're UF-equivalent and the join collapses.

**Convergence argument:** UF merges are monotone (classes only
collapse).  The set of symbols a single function ever creates is
bounded (one per AST mint site, plus one per loop's lhs/rhs phi pair).
So the lattice has finite height per program, and the fixpoint
terminates.

### Public API impact

- `ArraySize` becomes a wider union; consumers that wrote
  `isinstance(b.size, int)` keep working (they'll now correctly classify
  `SymbolicSize` as "not a known concrete size").
- Consumers that wrote `b.size == n` keep working (different type ⇒
  unequal).
- Consumers that wrote `b.size is None` may want to broaden to "no
  *concrete* size known" — depends on semantics.  Add a helper
  `concrete_size(b: ArraySize) -> int | None`.
- New helper `is_size_eq(b1: ArraySizeBound, b2: ArraySizeBound, uf) ->
  bool` for the recursive structural-equality-of-sizes check.

## Implementation plan

### Phase 1 — UF + lattice extension, no symbolic propagation

- Add `SymbolicSize`, `_SizeUF`.
- Wire `size_uf` through `_ArraySizeInferInstance` and
  `ArraySizeAnalysis`.
- Mint a fresh symbol for each `Argument: list[T]` (replacing today's
  `None`).  Keep all other mint sites at `None` for now.
- Update `_join_size` to handle the new variant.
- Verify nothing breaks: existing tests must still pass, just with
  argument lists having a `SymbolicSize` instead of `None`.

### Phase 2 — propagate through size-preserving operations

- `Var` (rebind), `[expr for x in xs]`, `enumerate`, `IndexedAssign`'s
  fresh def, `reverse`, slices that resolve to `b - a`.
- Add tests confirming `ys = xs ⇒ size_uf.equiv(ys.size, xs.size)`.

### Phase 3 — phi merging in loops

- In `_iterate_to_fixpoint`'s phi-update step, when both edges are
  `SymbolicSize` and unequal, call `uf.merge(lhs, rhs)` instead of
  going to `None`.
- Tests confirming a loop that re-binds `xs = xs[1:] + [0.0]`-style
  preserves the symbol.

### Phase 4 — opt-in concrete pinning

- When a symbol gets joined with a concrete int, optionally `pin` the
  class.  Skip on first pass — implement only if a consumer needs it.

### Phase 5 — operation rules for size-changing ops

- `zip(*xss)` with all-equal symbols.
- Concatenation: pin to sum when all concrete.
- Skip rules whose programs in the codebase don't actually exercise.

### Phase 6 — consumer integration

- Update `mpfx/format_infer.py:Sum` to consult `size_uf` and emit a
  runtime-bounded loop when the size is symbolic.
- (Future) bounds-check elimination consumer.

## Risks / open questions

1. **Symbol keying details**: `Empty(m, n)` mints two symbols, one per
   dimension argument.  What if the *same* `Expr` is reused as
   different dimensions of nested `Empty` calls?  Probably fine — each
   `Empty` is its own AST node, so the key tuple includes the position.
   Worth a focused test.

2. **Cross-function symbol identity**: each `analyze` call gets a
   fresh UF.  If a downstream pass invokes `analyze` on multiple
   functions and tries to compare their symbols, equality fails.  This
   is the right behavior (a function's args are unrelated to another
   function's args), but consumers should be aware.  Document on
   `_SizeUF`.

3. **Mutability of UF inside fixpoint**: the loop fixpoint runs the
   body multiple times; each pass might call `uf.merge`.  Convergence
   needs the merge operation to be monotone (it is — it only
   collapses).  But the *visible bound* of a phi doesn't change just
   because merge happened — the UF has more equivalences but the
   `SymbolicSize` value stored in `by_def` is still the same id.  So
   the fixpoint's `prev != current` check might miss a UF change.
   Need to either (a) include UF state in the convergence check, or (b)
   prove that any UF change forces some `by_def` change downstream.
   (b) seems likely but needs verification.

4. **Type alias vs class**: `SymbolicSize` could be a `NewType(int)`
   instead of a dataclass.  Dataclass is friendlier for `match`
   patterns; `NewType` is leaner.  Pick during implementation.

5. **`hash` and `==` semantics for `SymbolicSize`**: must be by `id`,
   not by representative.  That way two `ArraySize` values are `==` iff
   they're the same symbol, leaving UF consultation as the only way to
   ask the deeper "are these equivalent?" question.  Avoids surprising
   hash-equal behavior changing as classes merge.

## Out of scope

- Reasoning about arbitrary integer expressions (`len(xs) + 1`,
  `2 * len(xs)`, etc.).  Useful but a much bigger design (linear
  arithmetic, Presburger, etc.).  This proposal stops at equivalence.
- Whole-program / inter-procedural symbolic sizes.
- User assertions (`assert len(xs) == len(ys)`) — natural extension
  once assertions are part of the analysis pipeline.
