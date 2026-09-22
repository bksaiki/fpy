# The Triton emitter

Implementation plan.  The design is settled; what follows is the phase
breakdown, one phase per commit.

## Working policy

- **Pause after each phase for review.**  Do not begin the next phase until
  the current one has been looked at.
- **Do not commit.**  The working tree is left dirty; commits are made by the
  repository owner.
- **Run only the tests relevant to the phase.**  The full unit suite runs once,
  at the end, after the last phase.
- **Comments stay succinct**, and notes about *process* — what was tried, what
  a phase decided, why an ordering was chosen — belong in this document, not in
  source comments.

## Context

Items 1 and 2 are done at the FPy level, and their output is the emitter's
input.  Nothing here has to infer the kernel's shape: the pipeline already
produces it, and `exploration/triton/kernels.py` already says what it should
become.  Both sides are written down.

Running the batched dot product through `normalize` then `tile_loops` gives

```python
for i in range(0, t9, t8):
    with fp.INTEGER:
        t10 = (i + t8)
    for j in range(i, t10, 1):
        if j < t9:
            r = t[j]
            acc = fp.round(0)
            for k in range(K):
                with fp.FP32:
                    acc = (acc + (xss[r][k] * yss[r][k]))
            out[r] = acc
```

against `kernels.dot_exact`:

```python
row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
mask = row < n_rows
acc = tl.zeros((BLOCK,), dtype=tl.float32)
for k in tl.static_range(K):
    off = row * K + k
    x = tl.load(xs_ptr + off, mask=mask, other=0.0)
    y = tl.load(ys_ptr + off, mask=mask, other=0.0)
    acc = acc + x.to(tl.float32) * y.to(tl.float32)
tl.store(out_ptr + row, acc, mask=mask)
```

| pipeline output | `dot_exact` | phase |
|---|---|---|
| `acc + (x * y)`, `fp.round(0)` | `acc + x.to(tl.float32) * y.to(tl.float32)` | 1 |
| `xss[r][k]`, `out[r] = acc` | `tl.load` / `tl.store` | 3 |
| `for i in range(0, n, B)` | `tl.program_id(0) * BLOCK` | 4 |
| `for j in range(i, i + B)` | `tl.arange(0, BLOCK)` | 4 |
| `if j < n` | `mask = row < n_rows` | 4 |
| `for k in range(K)` | `tl.static_range(K)` | 2 |

## What the input cannot contain

The emitter is not a port of the cpp emitter's 4291 lines, because most of
what those lines handle is impossible here:

- **No calls** — the normal form inlines them, and a survivor is an error.
- **One exit** — no early-return reconciliation.
- **No `IfStmt`** — `SimplifyIf` made them `IfExpr`, which is `tl.where`.  The
  one exception is the mask guard `tile_loops` emits, which is a *shape*, not a
  branch.
- **No list storage** — a proven-length list unrolls into registers and an
  unproven one is refused, so nothing to port for `std::vector` or
  `std::shared_ptr`, and no aliasing or unboxing discipline.
- **No rounding-mode changes** — Triton has no per-instruction modifier, so a
  non-RNE context reaches codegen through `unfold_round` or not at all.
  Nothing to port for `fesetround`.

What *does* port is the spelling machinery: `_IndentedWriter`, visitor
dispatch, `_dispatch`, and the cast discipline — `_maybe_cast` rejecting lossy
implicit conversions, `_explicit_cast` for user casts.  That discipline matters
more here than there: the fp16 trap is a silent narrowing of exactly the kind
it exists to catch, and it is measured at 2000/2000 differing.

## What already exists to drive it

- `target.make_op_table()` → `ScalarOpTable`, with `TritonOp` carrying `name`,
  `in_tys`, `out_ctx` and a `TritonOpStyle` of `CALL` / `INFIX` / `PREFIX`.
  Every omission is a refusal, so a missing entry is the answer, not a gap.
- `storage.choose_storage(bound, cls)` → `TritonType`, and
  `choose_storage_scalar(bound)` for the op tables.
- `types.TritonScalar` / `TritonTuple` for spelling a type.
- `FormatInfer`, `ValueClassInfer`, `StorageInfer` from the shared pipeline.

## Phases

### Phase 1 — scalar expressions

**Done.**  One thing the plan had wrong: it said this table carries no
operation whose wider form rounds to itself, so no widening-under-`REAL` phase
was needed.  That is false, and the running example is the counterexample --
an fp16 product under `REAL` has no signature at all until the widening phase
finds one at the width holding its exact result.  Without it `x * y` was
refused; with it the emitter produces
`(x.to(tl.float32) * y.to(tl.float32))`, which is `dot_exact`'s spelling
character for character.

`fpy2/backend/triton/emitter.py`: `_IndentedWriter`, the visitor skeleton,
`_dispatch` over `ScalarOpTable`, and the cast discipline.  Emits an expression
and a straight-line body; no loops, no memory, no kernel wrapper.

First because everything else composes it, and because the fp16 cast trap is
the one numerical hazard already measured — it belongs under test before any
structure is built on top.

Tests: each `TritonOpStyle` spells correctly; an op absent from the table is
refused rather than guessed; `x * y` on two fp16 operands emits the casts
rather than `(x * y).to(...)` — the `dot_trap` shape is an assertion, not a
comment.

```
.venv/bin/python -m pytest tests/unit/backend/triton/test_emitter.py -q
```

### Phase 2 — statements and sequential loops

**Done.**  Two things the plan did not anticipate.

`fp.round` is a *cast*, not an op the table dispatches, so Phase 1's cast
discipline was half-built: `_explicit_cast` existed but nothing reached it.
A `Round` now emits nothing where `rounds_exactly` says the round changes
nothing, a `.to(...)` where the context's round *is* the hardware conversion
(`is_native_ctx`), and a refusal otherwise.

`trip_count` declines `range(K)` for a *foreign* constant -- `K` arrives as
`Var(SourceId('K'))` resolved from the closure, and `Specialize` monomorphizes
contexts and types, not foreign values.  `ConstFold` resolves it, and then the
count is 8.  So the refusal is the pipeline's to fix rather than the
emitter's, and a test pins that ordering.

Assignment, `with` (a context change is a storage change, not a statement),
`IfExpr` → `tl.where`, and a refused `for` → `tl.static_range` where the trip
count is proven, refused where it is not.

Separate because it closes the running example's inner loop, which is the one
`why_not_tileable` declines — the first end-to-end shape that is checkable
against a hand-written kernel.

Tests: the `for k in range(K)` body emits `dot_exact`'s loop; an unproven trip
count is refused.

### Phase 3b — port the dispatch onto `Visitor`

**Done.**  Behaviour-preserving: all 91 triton tests passed unchanged across
the port, which is what a port should look like.  35 `_visit_` methods against
`Visitor`'s 35 abstract ones, and a test asserts the class has no abstract
methods left -- so a node kind added to the AST breaks this backend's build
rather than reaching a catch-all.

Phase 3 dispatched with a hand-rolled `match` over node types.  Replace it
with `Visitor`, as `CppEmitter` does.

Two reasons, and the weaker one is not the one it looks like.  Dispatch walks
the *MRO* -- `UnaryOp` maps to `_visit_unaryop` and there is no `Round` entry
-- so the standard visitor does not separate `Round` from arithmetic by
itself; that isinstance check just moves inside one method instead of riding
on the order of `match` arms.  What it does buy is **exhaustiveness**: 35
abstract methods mean a node this backend cannot spell has a *named* refusal,
and a new AST node breaks the build rather than falling into a `case _`.

### Phase 3 — memory

**Done.**  Three things worth recording.

A numeric literal cannot take `.to(...)` directly: `2.to(tl.float32)` lexes as
`2.` followed by `to`, which is a different program.  `_explicit_cast`
parenthesizes anything that is not a name or a call, so `dot_exact`'s bare
`x.to(tl.float32)` is unchanged.

Comparisons and the boolean connectives needed spellings of their own.
Neither is in the op table, correctly -- they round nothing -- but a chain
joins with `&` rather than `and`, because Python's keyword short-circuits and
returns an operand rather than a tile.

`other=0.0` on a masked load is *safe* rather than meaningful: a masked-off
lane's value feeds only that lane, and the store that would commit it carries
the same mask.

`tl.load` / `tl.store` at the loop boundary, with the mask from the enclosing
guard and `other=` from the operation's identity.  Subscript chains
(`xss[r][k]`) become flat offsets.

Separate because the offset arithmetic is where a silent indexing bug would
hide, and it wants its own differential rather than being bundled with the
kernel wrapper.

### Phase 4 — the kernel wrapper

**Done.**  `emit_kernel` produces the `@triton.jit` function: a list argument
becomes a pointer, the tile width a `tl.constexpr`, and a trailing `return` is
dropped.

**`enable_fp_fusion` is not in the source.**  Triton takes it at the *launch*,
not the definition, so the emitter derives it and hands it over in
`KernelSource` rather than emitting it.  Checked against the hardware audit
both ways: the FP16-in program, whose products are exact, derives `True` --
and the audit measured it unchanged by fusion, 0 of 2000 either way.  The
all-FP32 program derives `False`, and the audit measured it differing on 590
of 2000.  The predicate agrees with the card.

Left as cosmetic differences from the hand-written kernel: `acc = 0` where
`dot_exact` writes `tl.zeros((BLOCK,), dtype=tl.float32)` -- Triton
broadcasts, so it is the same tile -- and the proven length inlined as `4`
where the hand-written one takes `n_rows` at runtime.

**Earlier in the phase; the body.**  Four things the kernel ABI decided,
each found by emitting the running example rather than by design:

- **The output is a parameter, not a value.**  A kernel writes through a
  pointer the launcher owns and returns nothing, so `out = [... for ...]` --
  an allocation -- has no spelling.  The FPy program that maps to a kernel
  takes `out` as an argument.  That is the ABI shaping the program, and it is
  the same reasoning *Not recommended* uses for the batch: state it in the
  program rather than invent a convention.
- **A proven length emits as its constant.**  A kernel argument is a bare
  pointer and carries no length, so the only length available is the one
  `Specialize` proved.  The kernel is therefore specific to the shape it was
  compiled for -- which it already was, since a proven length is what lets
  any of this be emitted.  `dot_exact` instead takes `n_rows` at runtime; that
  is the same question `BLOCK` raised and has the same answer if wanted, a
  free variable.
- **A materialized `range` is arithmetic, not memory.**  `SplitLoop` binds
  `t = range(n)` and indexes it; a range holds nothing, so `t[j]` is
  `start + j * step` and neither the binding nor the subscript is an access.
- **`drop_asserts` is opt-in.**  A kernel cannot raise, so an `assert` has no
  spelling either way -- the flag picks which answer.  Dropping one is a
  *semantic* change, so the caller asks for it; a launcher wanting the check
  runs it host-side.

**Prerequisites done.**  `tile_loops` takes `int | str` -- a name makes the
width a free variable, which is what becomes the `tl.constexpr` -- and returns
a `TileResult` carrying which loops it tiled.

One correction along the way: the first version returned the `ForStmt` nodes
themselves, and they go stale.  Each `SplitLoop.apply` rebuilds the AST, so a
node captured after one split is not in the function after the next.  It
records *positions* instead and resolves them against the final function: a
split at `i` leaves the outer loop at `i` and shifts only what follows, and
the scan never returns below `i`, so an index stays valid where a node does
not.

`tl.program_id`, `tl.arange`, the `tl.constexpr` parameters, and
`enable_fp_fusion` emitted per kernel from `scalar_fits_in` rather than pinned.
This is the phase that turns the outer chunk loop into a launch dimension.

Tests: the emitted kernel for the batched dot product is accepted by
`triton.jit`, and its structure matches `kernels.dot_exact`.

### After the last phase

```
.venv/bin/python -m pytest tests/unit -q -n 8
.venv/bin/mypy fpy2 && .venv/bin/ruff check fpy2
```

Running the emitted kernel is item 4, and needs the GPU.

## Open items

All three questions this plan opened are settled; the reasoning is recorded so
a phase does not relitigate them.

### The launch dimension: `tile_loops` says which loop, the emitter does not guess

The emitter has to decide that one loop is *not* a loop.  The first draft had
it recognize the outermost split loop by shape -- a `for` whose body is an
`INTEGER` context binding plus a guarded inner `for` -- which is brittle
pattern-matching on another pass's output and misidentifies the moment
`tile_loops` changes.

`tile_loops` already knows exactly which loops it split, so it returns them.
The emitter then chooses from a list rather than inferring from syntax.

That also fixes a rule the first draft got wrong.  "The outermost split loop is
the grid" is not true in general, because both idioms occur: the fused softmax
makes the row dimension the grid and it vanishes from the kernel, while the
matmul takes its block indices from the grid but keeps the `K` loop *as a loop
over tiles*.  With the list in hand the emitter can state the rule -- outermost
tiled loop is the grid, any other stays a loop over tiles -- and refuse a
function with two candidate grid dimensions rather than picking one.

Cost: `tile_loops` grows a return value, which touches its tests.  Do it before
Phase 1 depends on the current signature.

### `BLOCK` is a free variable, which is what makes it a `tl.constexpr`

Checked against the tutorials rather than assumed.  The convention is uniform:
`BLOCK` is a `tl.constexpr` *parameter*, never baked into the kernel source.
Only the chooser differs -- the softmax computes it host-side from the data
(`triton.next_power_of_2(n_cols)`), the matmul leaves it to `triton.autotune`.

`split` already takes `factor: int | str`, where a string names a free variable
of the function.  A free variable is exactly a `tl.constexpr` parameter, so
`tile_loops` should take the name and the emitter should spell it as one.

The first draft treated this as a trade -- a literal keeps the interpreter
differential, a symbol defers to the autotuner -- and that was wrong.  There is
no trade: a symbolic factor is still checkable, because the interpreter just
takes a value for it.  Checked for `n` in {0, 1, 5, 8, 9} against `BLOCK` in
{1, 4, 8}, all agreeing.  Varying the width is a *better* differential than one
literal, since it exercises the mask at every remainder.

One consequence for Phase 4: `split` emits `assert BLOCK >= 1`, and against a
`tl.constexpr` that is a compile-time claim.  Either fold it or drop it under
the assert-disabling flag; do not emit a runtime assert into a kernel.

### `tl.static_range` refuses an unproven count, and the fix is five lines

Phase 2 refuses, and the refusal message should name why, because the limit is
`trip_count`'s modeling rather than the programs'.

`trip_count` models only `Range1`, so it answers `None` for the running
example's `zip(xs, ys)` -- but `ArraySizeInfer` already knows that iterable's
length is 8.  The information is there; the query does not ask for it.

The fix, when it bites, is a *separate* query -- `static_trip_count(iterable,
sizes)` falling back to `by_expr`'s `ListSize.size` -- not a widening of
`trip_count`.  Its two callers, `hoist_scale` and `value_class`, both use it as
`size_eq(trip_count(...), size)` to ask "does this loop cover exactly that
list", where top-means-unequal is the safe answer.  Widening it would silently
change both.
