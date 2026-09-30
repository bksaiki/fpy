# Triton backend

`fpy2/backend/triton/` compiles FPy to Triton kernels.  Its test bed is
`examples/mmasim/compile_triton.py`, which compiles each MMA-Sim design as an
`m x k` by `n x k` matmul and, with `-r DRAWS`, compares every output bit for
bit with the interpreter.

## Status

**60/62 designs compile, and all 60 agree** (`compile_triton.py -j 8 -r 8`).
Refused: `amd.cdna1.*`, 280-525 bits of exact sum, which C++ refuses too.
The FP16-output designs at `e_zero = -133` (Hopper wgmma, Blackwell
tcgen05) compile since #325 (`format_infer` and digit bound fixes).  `examples/mmasim/serve/` runs an LLM through them.

Speed on the TITAN V (`bench/speed.py --best`, 1024 x 1024, `k` = 256, each
design's fastest tile; measured 2026-09-27):

| family | GFLOP/s |
|---|---|
| AMD CDNA2 (bf16, bf16_1k, f16) | 1,210-1,290 |
| NV Hopper / Blackwell / RTX Blackwell FP8 and FP4 chains | 260-420 |
| NV Volta, Turing, Ada, Hopper f16 / bf16 / tf32 | 200-375 |
| NV Blackwell block-scaled, one instruction (`k` = 32 / 64) | 180-290 |
| AMD CDNA3 | 140-310 |
| FPy's own FP32 / FP64 FMA dot product, compiled alike | 2,009 / 1,517 |
| `torch.matmul` FP16 / FP32 | 28,650 / 8,620 |

Bit-accurate models of newer hardware on an older GPU are expected to run
far below its FP16 peak.  The FMA rows are the harness's own ceiling.

## The shape of a kernel

- **A local list is a register tile.** A list of static length `N` is a
  `[BLOCK, P]` tile, `P` the largest power of two in `N`, plus the `N - P`
  elements past it as per-row values.  A loop over a list's elements runs
  once across the tile's lanes and once per tail element.
- **Reductions run across the lanes.** `max` / `min` / `any` / `all` always
  do; `sum` does under `REAL`, where every partial sum is exact in its
  storage, and otherwise folds left to right.  An aligned slice of a tile is
  a row of it reshaped.
- **Sizes and the grid.** `m`, `n` and `k` are kernel arguments.  `j` is
  `program_id(0)` in blocks, `i` is `program_id(1)` in blocks of `block_m`,
  and the loop over blocks of `k` runs at runtime.  `launch` without a block
  tunes block and warps.

Rules the backend keeps:

- **The emitter flattens `if` statements**, after the analyses, which read
  their guards.  Path sensitivity is read at `if` statements only, never an
  `IfExpr`: a backend may evaluate both of its arms.
- **The Triton contract**, stated on `_Emitter.mask`: a value an inactive row
  computes may be garbage, provided it is masked off before it is observed.
- **A directed rounding is a cast**, not unfolded: libdevice's conversions
  into `f32`, then truncation onto a narrower format with `f32`'s exponents.
  Unfolded, the analysis loses the format, and the store check refuses.

## Done: computed indices and in-kernel chains (2026-09-28)

The `triton` branch.  The backend compiles or refuses; computed indices broke
that, and blocked chaining block-scaled designs over `k` in one kernel.

- **An index is an integer where it is used** (`_index`): kept as is in
  integer storage, else cast to `int32`/`int64` where its format is proven
  integral (`index_scalar`), else refused.  A static `range` states its true
  interval, an empty one the empty set, so index products are rarely floats.
- **Index arithmetic under `INTEGER` is exact to `array_size`**
  (`_is_int_op`): integer operands under a context holding every integer.
- **Shapes and types a runtime loop needs:** a `for` header resets its
  target's shape, a literal list indexed by a lane broadcasts its elements,
  an op's result is cast into its storage, casts are `tl.cast` (a Python
  number has no `.to`), a runtime loop enters its carried values and tails
  in their class's storage, and `split_names` gives each class of
  definitions its own name.
- **A static loop is unrolled only where its body indexes a list in
  registers by its target** (`_indexes_registers`); nothing else gained
  from unrolling.  The NVFP4 chain's Triton compile went from 802 s to
  1.5 s.
- **Format inference stops a loop at a fixed point** of its phis and its
  store record (`_step`); the chains' FPy compile went from 6-20 s, linear
  in `k`, to ~4 s.
- **Acceptance:** all eight block-scaled designs chained at `k` = 1024,
  2048, 3072 are bit-identical to a launch per instruction.  At `m = 1`
  (`n = 2048`) fused is 22-64x faster (NVFP4 at 3072: 0.17 against 5.96
  ms); at `m = 2048` 10-15% slower for NVFP4 and MXFP4, 2-8% faster for MX.

## What is left

- **`serve` runs a block-scaled design chained in the kernel** (done,
  `mmasim-serving.md`, Record): fused at every `m` for MX and `fp8-block`,
  at most 32 rows for NVFP4 and MXFP4.
- **A slice along an outer dimension** was offset without its stride on
  `main`, a silent miscompile; fixed on the `mmasim` branch, which also
  compiles a slice of a slice.

- **FPy's compile time:** size inference runs in both tiling and the
  emitter.
- **The grid's second axis** is `cdiv(m, block_m)` programs, capped at
  65,535 by CUDA; grouping program ids as the matmul tutorial does would
  lift it.
- **cdna3.bf8's even and odd gathers** run one element at a time.  A lane
  loop needs `CompToLoop` to loop over the count, which breaks digit bound's
  gather pairing, keyed on a loop whose target is the index.
- **A uniform condition.** The special-value arm runs on every row even
  where no row in the program is special; a real `if` on a condition the
  whole program shares would skip it.
- **Not blocking a design:** iterating a pointer-backed list (`for t in
  xss[r]`); a `zip` over slice expressions, which `ZipElim` leaves since it
  takes names; a row of a slice, refused.

## Open items

### Which loop is the row axis, and who picks?

`vectorize._rows` picks by rule: the innermost tileable loop of runtime
count, else the outermost tileable one; the grid's second axis and the lanes
follow.  Trying tilings is part of the point of Triton, so it should be a
knob.  **Provisional:** a `TritonCompiler` option naming loops, as the
scheduling language does (`scheduling-language.md`).

### When is a tail too long?

A list of `2P - 1` elements is a tile of `P` and a tail of `P - 1`.  Padding
to the next power of two trades that for idle lanes, measured 1.4-3x slower
at `L + 1`.  **Provisional:** no bound; every list in the designs has a tail
of at most one.

### Shape-matching, both ways

`TileResult.tiled` exists so the emitter need not recognize a tile by shape,
and `_emit_tile` then destructures that output by shape.  One should give.

### Unbounded `INTEGER` is `int64`

It wraps past `2 ** 63`, where FPy does not, breaking the compile-or-refuse
contract only there.  **Provisional:** leave it; refusing every `INTEGER`
would refuse every index.

### An element guard for a list allocated in a loop

Digit bounds' "every element is finite" is one literal for every
iteration's list, so a value stored in an earlier iteration could be bounded
by a later one's check.  It behaves as a definition guard does inside the
loop, and `_one_list` blocks it after the loop; no probe gave a wrong bound.

### Should `_Unhoistable` refuse rounding a special it cannot hold?

It checks poles and bounded-format overflow, but not a `round` of an infinity
or NaN under a context without one, which raises: `SimplifyIf` can turn a
guarded `round(x * 1024)` at `x = inf` into a raise.  Only the opt-in
`fpy2.strategies.if_simplify` runs it.  **Provisional:** a refusal keyed on
`enable_inf` / `enable_nan`.

### A runtime `k` for a chain?

With the scales' length symbolic, `i = t * g` over a runtime `range` is
unbounded and possibly `-0`, and no storage holds it
(`StorageSelectionError`).  **Provisional:** out of scope; `serve` compiles
per `k`.  Reopen if a client needs one kernel for every `k`.

### `max` / `min` of an `fp16` value and a Python float?

`tl.maximum` promotes a tensor to a Python float's `fp32`, unlike
arithmetic, so the result is `fp32` where its storage says `fp16`.  Typing
the literal is consistent but made 55 designs 3-20% slower: their
exponents, stored `fp16` for `-inf` and NaN, then run in `fp16` with
conversions.  **Provisional:** left untyped; the speed belongs to storage
selection for registers.  Reopen when a runtime loop writes such a result
into a tile.

### Integer indices by storage inference, or by conversion at the use?

Conversion at the use (`_index`) is local and always correct, at one
integer conversion per use; storage inference would need two storages for a
value used both as an index and in float arithmetic.  **Settled:** at the
use; `int32` where the bound fits, else `int64`.  Reopen if conversions show
in the kernels' speed, or a large-tensor launch disagrees.

### Should formats track that a value cannot be zero?

A constant like `16` is possibly `+0`, which makes a signable factor's
product possibly `-0`.  A zero-free flag touches every arithmetic rule.
**Settled:** no.  Reopen if `-0` forces float or refused storage in real
programs.

### One integer-valued analysis for `FormatInfer` and `array_size`?

`array_size._is_int_valued` copies a small rule set `FormatInfer` already
encodes.  **Settled:** the copy.  Reopen if the two drift.
