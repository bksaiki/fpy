# Triton: the mmasim matmuls

## Status

`examples/mmasim/compile_triton.py` compiles each design as an `m x k` by
`n x k` matmul and, with `-r DRAWS`, compares every output bit for bit with
the interpreter.  **50/62 compile**; the 14 of the first 16 that compiled
agreed, and the rest have not been run on a GPU.  `amd.cdna1.*` is out of scope
(280-525 bits of exact sum, which C++ refuses too), and so, for now, is FP16
output at `e_zero = -133` (Hopper wgmma, Blackwell tcgen05): the zero sentinel
stretches the exact sum to 179 bits.

The shape the kernels take:

- **A local list is a register tile.** A list of static length `N` is a
  `[BLOCK, P]` tile, `P` the largest power of two in `N`, plus the `N - P`
  elements past it as per-row values.  A loop over a list's elements runs
  once across the tile's lanes and once per tail element.
- **Reductions run across the lanes.** `max` / `min` / `any` / `all` always
  do; `sum` does under `REAL`, where every partial sum is exact in its
  storage, and otherwise folds left to right.  An aligned slice of a tile is
  a row of it reshaped.
- **Sizes and the grid.** `m`, `n` and `k` are kernel arguments.  `j` is
  `program_id(0)` in blocks, `i` is `program_id(1)`, and the loop over blocks
  of `k` runs at runtime.  `launch` without a block tunes block and warps.

Rules the backend keeps:

- **The emitter flattens `if` statements**, after the analyses, which read
  their guards.  Path sensitivity is read at `if` statements only, never an
  `IfExpr`: a backend may evaluate both of its arms.
- **The Triton contract**, stated on `_Emitter.mask`: a value an inactive row
  computes may be garbage, provided it is masked off before it is observed.
- **A directed rounding is a cast**, not unfolded: libdevice's conversions
  into `f32`, then truncation onto a narrower format with `f32`'s exponents.
  Unfolded, the analysis loses the format, and the store check refuses.

## Performance

TITAN V, `examples/mmasim/bench/speed.py`, best over block sizes (1024 x
1024, `k = 256`; mxfp8 (now `mx.e5m2`) and nvfp4 at their own `k`), the
first 16 designs:

| kernel | GFLOP/s |
|---|---|
| volta, turing, ada, hopper | 124-132 |
| ampere x2, nvfp4, cdna3.f16 | 110-116 |
| mxfp8, cdna3.bf16 | 88-99 |
| cdna3.bf8 | 59 |
| cdna2 x2, fp64 | 185-199 |
| an FP32 matmul in FPy, compiled here | 196 |
| cuBLAS FP32 / FP16 tensor cores | 8,600 / 27,500 |

First launch is under 2 s for every design.  A `-r 64` run is about 70 s,
most of it FPy's own compile at about 5 s a design.

## What is left

- **FPy's compile time.** Size inference still runs in both tiling and the
  emitter.  Profile it.
- **An FP32 matmul at 196 GFLOP/s** is 44x under cuBLAS, and bound by memory:
  each output reads its two rows with no reuse.  Two-dimensional output tiles,
  or program ids grouped as the matmul tutorial groups them, would reuse
  them; grouping also lifts the 65535 limit on `grid.y`.
- **cdna3.bf8's even and odd gathers** run one element at a time.  A lane
  loop needs `CompToLoop` to loop over the count, which breaks digit bound's
  gather pairing, keyed on a loop whose target is the index.
- **A uniform condition.** The special-value arm runs on every row even
  where no row in the program is special.  A real `if` on a condition the
  whole program shares would skip it.
- **Copies in the emitted kernel**, most from the if-flattening (`__tN = r`
  before an arm, `r72 = t71`).  Readability only; Triton emits no IR for
  `a = b`.
- **Not blocking a design:** iterating a pointer-backed list (`for t in
  xss[r]`); a `zip` over slice expressions, which `ZipElim` leaves since it
  takes names; a row of a slice, refused.

## Open items

### Which loop is the row axis, and who picks?

Trying different tilings is part of the point of Triton, so the axes should
be a knob.  Today `vectorize._rows` picks by rule: the innermost tileable
loop of runtime count, else the outermost tileable one; the grid's second
axis and the lanes follow.
**Provisional:** a `TritonCompiler` option naming loops as the scheduling
language does (`scheduling-language.md`).

### When is a tail too long?

A list of `2P - 1` elements is a tile of `P` and a tail of `P - 1`, which is
per-element code again for that part of the list.  Padding to the next power
of two trades that for idle lanes, measured 1.4-3x slower at `L + 1`.
**Provisional:** no bound; every list in the designs has a tail of at most
one.

### Shape-matching, both ways

`TileResult.tiled` exists so the emitter need not recognize a tile by shape,
and `_emit_tile` then destructures that output by shape.  One should give.

### Unbounded `INTEGER` is `int64`

Storage holds an unbounded integer in `int64`, which wraps past it where FPy
does not.  It breaks the compile-or-refuse contract only for values past
`2 ** 63`.
**Provisional:** leave it; refusing every `INTEGER` would refuse every index.

### An element guard for a list allocated in a loop

Digit bounds' "every element is finite" is one literal for every
iteration's list, so a value stored in an earlier iteration could be bounded
by a later one's check.  It behaves as a definition guard does inside the
loop, and `_one_list` blocks it after the loop.  No probe gave a wrong bound.

### Should `_Unhoistable` refuse rounding a special it cannot hold?

It checks poles and bounded-format overflow, but not a `round` of an infinity
or NaN under a context without one, which raises: `SimplifyIf` turns `if
abs(x) < 65536: ... round(x * 1024) ... else: y = x` at `x = inf` from `inf`
into a raise.  Only the opt-in `fpy2.strategies.if_simplify` runs it.
**Provisional:** a refusal keyed on `enable_inf` / `enable_nan`.

### Is `_is_exact` narrower than it needs to be?

`_affine` descends only through arithmetic under `REAL`, so index arithmetic
under `fp.INTEGER` does not decompose, and a slice bounded by it is unsized.
**Provisional:** leave it; precision, not capability.
