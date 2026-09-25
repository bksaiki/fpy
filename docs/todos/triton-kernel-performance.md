# Triton: kernel performance

Implementation plan.  The design is settled; what follows is the phase
breakdown, one phase per commit.

## Working policy

- **Pause after each phase for review.** Do not begin the next phase until the
  current one has been looked at.
- **Do not commit.** The author of the change leaves the working tree dirty;
  commits are made by the repository owner.
- **Run only the tests relevant to the phase.** The full unit suite runs once,
  at the end, after the last phase.
- **Comments stay succinct**, and notes about *process* -- what was tried, what
  a phase decided, why an ordering was chosen -- belong in this document, not in
  source comments.
- **Every change is a rule about an operation, with its proof.**  A lowering
  states what it computes, when that equals the program's value, and where the
  preconditions come from (an analysis, not a guess); the docstring carries the
  argument.  No storage or spelling exception keyed on how one design happens
  to use a name.  Each phase below has a *Review* note: the principled form and
  what a reviewer checks.

## Context

The mmasim kernels run at 59-199 GFLOP/s on a TITAN V
(`examples/mmasim/bench/speed.py`, 1024 x 1024, `k = 256`); an FP32 matmul
written in FPy runs at 194.  Every change below was first made by hand in the
emitted source and checked bit for bit against the unchanged kernel's output;
the scripts are in the session scratchpad and are not part of the plan.

**Where the time goes** (`ncu`, `nv.volta.f16.f32`):

| | as emitted | f32 truncation |
|---|---|---|
| SM throughput / L1 busy | 82% / 81% | 69% / 96% |
| DRAM throughput | 0.3% | 0.4% |
| top stall | math pipe (f64) | load/store throttle |
| sectors per global load | 16.5 | 16.5 |

As emitted, the kernel is bound by the f64 in `fused_sum`'s truncation; with
that gone, by the number of load instructions.  DRAM is idle: the operands hit
in L1 (91%).

**Measured, Volta, each on top of the last, all bit-identical:**

| change | GFLOP/s |
|---|---|
| as emitted | 132 |
| truncation in f32, power-of-two scale from bits | 157 |
| each operand loaded once per iteration | 221 |
| special-value arm skipped when no row takes it (+ unmasked `A` load) | 245 |
| `max(logb(x), c)` as a bitfield | 367 |
| integer accumulation of the aligned sum | 373 |

Each mostly unblocks the next: f32 truncation alone is +19%, with single loads
+68%.

**Measured elsewhere:**

| change | before | after |
|---|---|---|
| FP32 matmul, 2-D output tile 64 x 64, same summation order | 194 | 2,019 |
| Volta, hand-written, 2-D tile 4 x 64 against one row per program | 212 | 309 |
| Volta, hand-written, 16-wide `k` loads split in registers, one row | 212 | 267 |
| the same, with the 4 x 64 tile | 309 | 305 |
| `cdna3.bf8`, even/odd gathers as `reshape` + `split` | 59 | 76 |

cuBLAS FP32 is 8,041; a bit-exact kernel is capped near half of FMA peak,
since FPy's `acc + a * b` rounds the product.

**Tried and rejected** (no gain, or a loss, on the 367-373 kernel):
eliding the NaN guard on `e_max`'s reduction; `tl.range` with `num_stages` 2-3
or `loop_unroll_factor` 2-4 (all within 1%); integer storage for exponents
(-1%); truncation as a mantissa shift (370 -> 241) or a bit mask (370 -> 263) --
a power-of-two multiply and `trunc` are two float instructions.

## The changes

### Truncation in f32 (`fused_sum`, via `RescaleFixed`)

`RescaleFixed` splits a runtime-grid round into a scale-in under `REAL`,
`_t = 2 ** n * x`, an integral round of `_t`, and a scale-out.  `_t`'s bound
admits values `x`'s storage cannot hold -- truly, and all below one half -- so
the emitter held it in `f64` and scaled with `libdevice.ldexp`.  A round that
sends anything below one half to zero never reads those, so the round and its
scale-in are one operation, lowered together in `x`'s storage.

### Each operand loaded once

Per iteration the Volta kernel loads `A` and `B` three times each: for the
products, again in the special-value arm (`dpa_special_values` recomputes
`a * b`), and again for the exponents.  The loads differ only in their masks --
the arm's and the guard's -- so Triton does not merge them.  An element is
loaded once, under the row mask, and reused: the Triton contract already lets
an inactive lane hold garbage until it is masked.

### Skip an arm no row takes

The flattened special-value arm runs on every row every iteration.  Where a
branch guards a flattened arm, `if tl.max(guard & row_mask) != 0:` around it
skips the work when no active row needs it.  A garbage row can only turn it on,
which costs time, not bits.

### `max(logb(x), c)` as a bitfield

For finite `x` held in IEEE storage `S` and `c >= emin_S`,
`max(logb(x), c) == max(field(x) - bias_S, c)`: a subnormal or zero reads
`emin_S - 1 < c` and clamps.  The emitter spells the general `logb` -- a
four-deep `where` chain with a subnormal rescale -- and then `tl.maximum` with
NaN propagation.  Infinities and NaN keep their answer through one `where` on
`isfinite`, dropped where value classes prove `x` finite (every mmasim use
sits under the special-value guard).  `exponent0`'s
`where(isfinite(x), max(logb(x), emin), -1)` is the same peephole.

### Gathers as `reshape` + `split`

`cdna3.bf8` splits a 16-wide tile into its even and odd elements, emitted as a
static loop of masked-sum extracts: O(L^2).  A stride-2 gather of a tile is
`tl.split(tl.reshape(t, (BLOCK, n // 2, 2)))`.

### 2-D output tiles

Each program computes one row of `A` against `BLOCK` columns, so everything
that depends only on `B` -- its loads, exponents and finiteness -- is recomputed
for every row of `A`, and every `B` element is loaded `m` times.  Tiling the
grid's second axis by `BM` makes row values `[BM, BLOCK]` and lane tiles
`[BM, BLOCK, L]`; what depends only on `A` is computed at `[BM, 1, L]`, only on
`B` at `[1, BLOCK, L]`.  `vectorize._grid` already finds the loop; it becomes a
tile axis rather than `program_id(1)`.

### Integer accumulation

Every term of `fused_sum`'s exact sum is `q * 2 ** g` on one grid, `q` an
integer of at most 25 bits, so the sum is an `int32` sum and one scaled
conversion.  It needs the terms to have no `-0`, which an integer has no
spelling of: `fused_sum` now rounds with `enable_neg_zero=False`, matching
MMA-Sim (#321).

## Phases

### Phase 1 -- A regression net of hard test cases

**Done.** `compile_triton._hard_cases` lists each format's hard test cases (filtered
by `representable_in`: E8M0 has no zero, FP4 no specials), and every
`_HARD_EVERY`-th `-r` draw puts one in an element with probability `_HARD`.
The GPU test went to `examples/mmasim/tests/test_triton.py`, not
`tests/unit`: no unit test imports the example, and the test needs its
harness.  With the special-value select deleted from the Volta kernel, plain
draws agree 128/128 and hard test cases 34/128.  Tracker unchanged: 14/16, all
agree.

- **What:** `examples/mmasim/compile_triton.py`'s `-r` draws only in-range
  values; it never samples a special, an FP32 subnormal, a zero of either sign,
  or values near the overflow boundary.  Add hard test cases to the draws (`_sample`), on
  by default for a share of the draws, and a GPU test in
  `tests/unit/backend/triton/test_launch.py` running two designs (Volta, bf8)
  on a batch of hard test cases against the interpreter.
- **Why first:** every later phase rewrites special-value, zero and exponent
  handling; the current net would not see a mistake there.
- **Tests:** `FPY_REQUIRE_GPU=1 .venv/bin/python -m pytest
  tests/unit/backend/triton/test_launch.py -q -n auto`;
  `cd examples/mmasim && ../../.venv/bin/python compile_triton.py -r 64`.

### Phase 2 -- Truncation in f32

**Done.** `_Emitter._fuse_rounds` finds a `round(_t)` whose argument is a
scale-in `_t = 2 ** n * x` under `REAL` that nothing else reads, and
`_emit_fused_round` lowers the pair as one operation in `x`'s float storage
`S`; the scale-in is not materialized.  Preconditions, each from an analysis:
the round is integral under RTZ, RNE or RNA (floor and ceil send an
underflowed `-tiny` to `-0`, not `-1`); its bound is below `2 ** bias_S`, so
`2 ** n * x` cannot overflow; `n` is integral with `|n| <= 2 * (bias_S - 1)`;
and what the scale-in reads reaches the round unchanged (reaching
definitions), since it is re-read there.  Then every argument of magnitude
one half or more -- the only ones the round distinguishes -- is `x` shifted,
exact in `S`.  The shift is two multiplies by powers of two built from bits
(`_scale_by_halves`): one factor is not normal at `n = 149`, which a row of
zeros and a subnormal `c` reaches.  Both factors scale the same way, so the
product between them lies between `x` and the result and is exact wherever
the result is.

A first version kept the scale-in and overrode its storage by how it was used;
it measured the same and was replaced, since the proof is about the round, not
the name.  `libdevice.ldexp` in `f32` was the safe spelling, at 136 GFLOP/s
against the multiply's 155.  An fp16-input scale-in is already `f32`.  The
scale-outs stay `f64` `ldexp` until Phase 8.

Tests: `test_emitter.test_a_scale_in_stays_in_its_operands_storage` (fails on
the old emitter), `test_launch.test_an_aligned_sum_agrees_on_hard_cases` (zero
rows, subnormals, the largest values, one half and just below it at the
largest scale).  Bench, best GFLOP/s, against the kernels before it: every
`fused_sum` design +12-26% (volta 132 -> 155, hopper 124 -> 148, cdna3.bf16
88 -> 111, cdna3.bf8 59 -> 72); cdna2 and fp64 unchanged.

### Phase 3 -- Each operand loaded once

- **What:** loads are masked by address validity alone, and a read of an
  element is reused until a store that may alias its list.
- **Why separate:** it changes what every kernel loads; measured after
  Phase 2, since before it the gain is 8%.
- **Tests:** a test counting `tl.load(A_ptr` in the Volta kernel (one per
  iteration); tracker; bench.
- **Review:** two separate principles, not an address cache.  (1) A load's
  mask is address validity -- the row mask -- never a branch's guard, which
  only decides whose result is kept (the Triton contract); with that, the
  duplicate loads become the same load.  (2) Reusing a load is common
  subexpression elimination of a pure read, valid until a store that may alias
  the list: check it against the alias analysis, inside loops and across
  branch arms.  The recomputed `a * b` is ordinary CSE; confirm Triton merges
  it before adding anything for it.

### Phase 4 -- `max(logb(x), c)` as a bitfield

- **What:** `_emit_logb` drops the NaN and infinity arms where value classes
  prove `x` finite, and a `max(logb(x), c)` with `c >= emin_S` drops the
  subnormal arm; together, the bitfield.  `exponent0`'s shape follows.
- **Tests:** `test_emitter.py` for the spelling and the conditions (a `c` below
  `emin_S` keeps the general form; an unproven-finite `x` keeps the `where`); a
  launch test on every fp16/bf16/f32 special and subnormal; tracker; bench.
- **Review:** prefer two general rules to one pattern.  (1) `logb(x)` where
  value classes prove `x` finite needs no NaN or infinity arm.  (2) Under
  `max(., c)` with `c >= emin_S`, the subnormal arm is dead: its result is
  below `c`.  Together they give the bitfield, and each applies elsewhere.
  Check `c` against the *storage's* `emin` (E4M3 held in f16 has
  `c = -6 >= -14`), that `max(c, logb(x))` and a `max` over a comprehension
  reach it, and that the integer result is cast into its class storage.

### Phase 5 -- Skip an arm no row takes

- **What:** `_emit_branch` wraps a flattened arm whose work is large in a
  block-uniform `if` on its guard, masked by the rows.
- **Tests:** a launch test on hard test cases in which one row of a block is special
  and the rest are not; tracker; bench.
- **Review:** the rule is for any flattened `if`, not the special-value arm: an
  arm whose guard is false on every live row may be skipped, since its
  results are selected only where the guard holds and its stores are masked by
  it.  Check the guard is reduced over live rows only (a garbage row may only
  turn it on), nested arms, loops inside an arm, and a cost threshold stated
  once (a reduction per iteration is not free for a small arm).

### Phase 6 -- Gathers as `reshape` + `split`

- **What:** the emitter spells a stride-2 gather of a tile with `reshape` and
  `split`; other strides keep the extract loop.
- **Tests:** `test_expect.py` for the spelling; a launch test; tracker; bench
  bf8.
- **Review:** it is the lowering of a strided slice of a register tile, not of
  `gtr_fdpa`'s gather: any power-of-two stride is repeated `split`s.  Check a
  tile with a tail, a slice whose start is not zero, and masked lanes.

### Phase 7 -- 2-D output tiles

- **What:** `vectorize.py` tiles the grid's second axis by a `BM`; the emitter
  gives values a `[BM, ...]` leading dimension and computes one-operand work at
  its own shape; the launcher tunes `BM` with `BLOCK`.
- **Why last:** the largest change, touching tiling, emission and launch; the
  phases before it keep their gains inside it.
- **Tests:** `test_vectorize.py` for the chosen axes; a launch test of the FPy
  FP32 matmul against the interpreter at sizes not divisible by `BM`; tracker;
  bench, with the FP32 matmul among the designs.
- **Review:** the shape of a value should follow from which tiled loops it
  depends on (its reads' definitions), so one rule gives `[BM, 1, L]` for
  what depends only on `A` and `[1, BLOCK, L]` for `B` -- not per-operand
  special cases.  Check the tiling legality (`_why_writes_refuse`) with two
  tiled axes, masks on both, and the 3-D layout cost from the open item.

### Phase 8 -- Integer accumulation

- **What:** the aligned exact sum becomes an `int32` sum of the truncated
  terms, scaled once; conditions from the digit bounds (every term on one grid,
  the sum within 31 bits).
- **Tests:** tracker; the NV all-`-0` directed case agrees with MMA-Sim; bench.
- **Review:** it is an identity, `sum(q_i * 2 ** k) == sum(q_i) * 2 ** k`,
  exact under `REAL`: factoring a common scale out of an exact sum.  The
  preconditions -- one grid (the same `k` definition for every term, from
  Phase 2's fused rounds), integer terms with no `-0`, and a sum that fits the
  integer type -- come from the analyses.  Check it is stated as that rewrite
  and not matched on `fused_sum`'s shape.

### After the last phase

```
.venv/bin/python -m pytest tests/unit -q -n auto
FPY_REQUIRE_GPU=1 .venv/bin/python -m pytest tests/unit/backend/triton -q -n auto
.venv/bin/python -m mypy fpy2
.venv/bin/python -m ruff check fpy2 tests
cd examples/mmasim && ../../.venv/bin/python compile_triton.py -r 256
cd examples/mmasim && ../../.venv/bin/python compile.py -o /tmp/cpp && \
  ../../.venv/bin/python tests/test_nv.py && ../../.venv/bin/python tests/test_amd.py
.venv/bin/python examples/mmasim/bench/speed.py
```

and update the Performance table in `triton-mmasim-matmul.md`.

## Open items

### Load caching: in the emitter, or a CSE pass before it?

A pass over the normal form (common subexpressions, `a * b` computed twice)
would also help the C++ backend and is easier to test in isolation; the
emitter sees the one thing the AST does not -- that two reads differ only by a
mask it added.  **Provisional:** the emitter, keyed by address.  Reopen if a
duplicate that is not a load (the recomputed product) survives Triton's own
CSE.

### Is a power-of-two scale always provable where the designs need it?

**Settled in Phase 2:** every `fused_sum` design qualifies (the 8 NV, the 3
cdna3); `n` stays within `2 * (bias - 1)` there.

### How does a 2-D tile meet the lane tiles?

Lane tiles become 3-D.  A hand-written 3-D Volta kernel with one row per
program ran at 212 against the emitted kernel's 373, so the shape itself may
cost what the tile saves; the +46% was measured within the hand-written
kernel.  **Provisional:** Phase 7 lands first for kernels without lane tiles
(the FP32 matmul, `fp64_fma`, `cdna2`), then the MMA designs, measured against
the Phase 6 kernel.

### Wider `k` loads?

+26% with one row per program, nothing with a 2-D tile.  **Provisional:** not
planned; reopen if Phase 7 does not reach the MMA designs.

### Do these gains hold on other GPUs?

The TITAN V runs `f64` at half the `f32` rate; most GPUs run it at 1/32-1/64,
so Phases 2 and 8 would matter more there, and a different L1 would move
Phase 3.  **Provisional:** measure on the TITAN V; revisit when another GPU is
available.
