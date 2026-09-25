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
the arm's and the guard's -- so Triton does not merge them.  A later load under
a narrower mask may reuse an earlier one: the Triton contract already lets an
inactive row hold garbage until it is masked.

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

**Done.** A load under mask `M2` reuses an earlier load of the same element
under `M1` where `M2` implies `M1` (`M1`'s conjuncts are among `M2`'s): every
row `M2` keeps was loaded, and the rest may hold garbage until masked.  It
needs no argument about address validity -- the review note's first
principle was wrong: a branch's guard can be what keeps an address in bounds
(`if i < len(xs): y = xs[i]`), so a load cannot simply drop it.
`_IndentedWriter` holds the loads per open block and value-numbers pure
bindings, so two lane indices `tl.arange(0, 4)[None, :]` bound to different
names, or a row mask re-bound under a new temporary, are one address and one
conjunct; a broadcast `(a & b)[:, None]` is distributed over its conjuncts.
A reuse ends at any `tl.store` (two arguments may be one tensor), at a
reassignment of any name in the address, and at a loop body's edge.  Every
load is bound to a temporary, which changes the text of every kernel.

Volta goes from six operand loads per iteration to two.  Bench against
Phase 2: +1-20% on twelve designs (volta 155 -> 174, hopper 148 -> 173,
cdna3.bf8 72 -> 87), but **cdna3.f16 131 -> 126**: reuse holds the `A` and
`B` tiles live across the special-value arm, where a reload was an L1 hit.
Reloading the two exponent operands recovers only 123 -> 126, so it is the
tiles held across the arm.  Not answered with a live-range heuristic:
Phase 5, which takes the arm off the common path, is expected to remove the
pressure; re-measure cdna3.f16 there.

Tests: `test_emitter.TestLoadReuse` (reuse under a narrower mask -- fails on
the old emitter; none across a store, after the index is reassigned, or into
a loop body), and the exact-text tests updated for bound loads.

- **What:** loads are masked by address validity alone, and a read of an
  element is reused until a store that may alias its list.
- **Why separate:** it changes what every kernel loads; measured after
  Phase 2, since before it the gain is 8%.
- **Tests:** a test counting `tl.load(A_ptr` in the Volta kernel (one per
  iteration); tracker; bench.
- **Review:** reuse is common subexpression elimination of a read, valid
  where the reusing mask implies the loaded one and until a store that may
  alias.  Check the implication is decided on canonical conjuncts, the
  invalidation on stores, reassignments and loop bodies, and the recomputed
  `a * b` (ordinary CSE -- Triton merges it once its loads are shared).

### Phase 4 -- `max(logb(x), c)` as a bitfield

**Done**, as the review's two rules.  The emitter now runs `ValueClassInfer`
itself and hands it to `FormatInfer` (no kernel changes by that alone).
(1) `_emit_logb` drops the infinity and NaN arms where
`classes.is_finite(x)`.  (2) `_clamp_logbs` marks a `logb` operand of a `max`
one of whose other operands' least value is at or above `emin` of the
argument's *storage*; a clamped `logb` is its field minus the bias, a zero or subnormal
reading `emin - 1`.  The condition is on bounds, not literals, and holds for
either operand order.  The format bound of `x` does not carry finiteness --
`logb`'s arguments were full `IEEEFormat`s -- so value classes, not formats,
answer rule (1).

In Volta every exponent, the accumulator's too, is a bitfield and a clamp.
Bench against Phase 3: +10-51% on every design with exponents (volta
174 -> 234, turing 176 -> 262, ampere.tf32 146 -> 220, cdna3.f16 126 -> 174,
nvfp4 124 -> 143), cdna2 and fp64 unchanged.  cdna3.f16's Phase 3 regression
is more than recovered.

Tests: `test_emitter.TestLogb` (three fail on the old emitter; a clamp below
`emin` keeps the subnormal arm), `test_launch.test_logb_agrees_on_hard_cases`
over fp16 and fp32 zeros, subnormal edges, the largest values, infinities
and NaN, clamped at, above and below `emin`, and proven finite with and
without a clamp.

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

**Done**, for a `then` arm: the special-value arm always is one, and an
`else` arm is the same rule with the guard negated, left for when a design
needs it.  An arm of at least `_SKIP_SIZE` AST nodes (64) runs under
`if tl.max(live.to(tl.int32)) != 0:`, `live` its rows' mask; lane loops are
left flattened, their values being 2-D.  Triton keeps a name's type across
an `if`, so the arm's merge temporaries are given a placeholder of their
final type in front of it (`_placeholder`, inserted once the arm has decided
the shape), and every name the arm reassigns that existed before is saved in
front and put back at its end -- harmless, as the `if` does not merge it.
Leaving a block now forgets the values it assigned in the writer's value
numbering, since a skipped arm leaves placeholders, not them.

Bench against Phase 4, on the benchmark's inputs, which hold no special
value, so every block skips: +3-16% on every design with a special-value arm
(turing 260 -> 300, hopper 252 -> 285, volta 234 -> 260, cdna3.f16
174 -> 182), cdna2 and fp64 unchanged.  Where every block has a special row,
the reduction is the cost.  cdna3.f16, re-measured as Phase 3 asked: 182,
against 131 before Phase 3.

Tests: `test_emitter.TestSkippedArm` (a long arm is skipped -- fails on the
old emitter; a short one is not), and
`test_launch.test_a_skipped_arm_agrees_where_one_row_takes_it`.

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

**Done**, as two rules.  First, the Triton normal form lowers a
comprehension over `range(a, b, s)` to a count from zero,
`for k in range(len(range(a, b, s))): i = a + s * k; t[k] = ...`
(`CompToLoop`'s `index_ranges`, set only by `normalize`): the write index is
the lane, and the read `xs[i]` a lane-varying index into a tile.  Second, the
emitter lowers any such read as a gather (`_gather`), replacing Phase 1's
refusal: an index proven within the tile; `a + s * k` with `s` a power of two,
`a < s`, and no tail is a `reshape` and a `split` per bit of `a`; anything
else is a one-hot sum over a `[BLOCK, lanes, width]` compare (filled with
`-0.0`, so the sum is the element), plus the tail.

Digit bounds paired the parts of a gather by the loop's `range`, which the
new loop hides -- and `len(range(1, 16, 2))` folds to `8`.  An index
`a + s * k` with `k` over `range(n)` is now keyed by its index set
`(a, s, n)` (`_elt_key`), so `es` and `prods` gathered over the evens still
share their variables; the key types are named (`_Operand`, `_RangeKey`,
`_EltKey`).

Bench against Phase 5: cdna3.bf8 99 -> 143 (+44%; the hand edit of Phase 0
gave +29%); every other design unchanged.
Trackers unchanged: Triton 14/16, all agree; C++ 14/16.

Tests: `test_comp_to_loop.test_index_ranges_counts_the_trip` (a start, a
step, a negative step), `test_emitter.TestGather` (split, one-hot, an index
past the tile refused), `test_launch.test_a_gather_agrees_on_hard_cases`;
`test_normalize.test_a_gathered_sum_keeps_its_bound_past_a_guard` now runs
over the new loop, as does `examples/mmasim/tests/test_triton.py` on bf8.

- **What:** the emitter spells a stride-2 gather of a tile with `reshape` and
  `split`; other strides keep the extract loop.
- **Tests:** `test_expect.py` for the spelling; a launch test; tracker; bench
  bf8.
- **Review:** it is the lowering of a strided slice of a register tile, not of
  `gtr_fdpa`'s gather: any power-of-two stride is repeated `split`s.  Check a
  tile with a tail, a slice whose start is not zero, and masked lanes.

### Phase 7 -- 2-D output tiles

**Done for kernels without lanes** (`cdna2`, `fp64 (fma)`), per the open
item; for the MMA designs it was built and measured, and not kept (Phase 7b
below).  No AST change: `_grid` already proves
the loop around the tile tileable, so `_emit_grid` evaluates it as a tile of
`BLOCK_M` rows (`i = program_id(1) * BLOCK_M + arange(BLOCK_M)[:, None]`,
guarded by `i < m`) where it used to be one program per iteration.  In such a
kernel every row value is rank 2: the column index is `[1, BLOCK]`, so a
value's shape is `[BM, 1]`, `[1, BLOCK]` or `[BM, BLOCK]` by broadcasting.
The writer now tracks, per name, the tile axes it varies along
(`_IndentedWriter.along`, replacing the `rows` set); from that come the
shape a carried value or a skipped arm's placeholder is given (`_shape`), and
a load's mask under the tiles' own guards: the guards of the axes its address
varies along (`_masked_load`), so `A`'s loads stay `[BM, 1]` -- the
generalization of Phase 3's unmasked scalar load, sound for the same reason
(every program has a live row on each axis).  `emit_kernel` adds a
`BLOCK_M: tl.constexpr` beside `BLOCK`, the launcher takes `block_m`
(default 1; ignored for a kernel without one) and tunes it from
`TUNING_2D`, and the bench sweeps it (`--blocks-m`).  The tracker launches
with `BLOCK_M = 4`, so a draw of fewer rows masks the rest.

Bench against Phase 6 (best over the sweep):

| design | Phase 6 | Phase 7 | BM x BLOCK |
|---|---|---|---|
| cdna2.bf16 | 195 | 1,282 | 64 x 32 |
| cdna2.f16 | 196 | 1,229 | 64 x 16 |
| fp64 (fma) | 185 | 1,519 | 32 x 128 |

The MMA designs are unchanged (their kernels are byte-identical).  The FP32
matmul was not added to the designs: `fp64 (fma)` is the same FPy loop.

Tests: `test_launch.test_a_tile_of_rows_agrees_past_its_end` (a runtime-`k`
dot-product matmul at counts not a multiple of either tile, `BLOCK_M` 1 and
4), `test_emitter.test_a_load_is_masked_along_the_axes_its_address_varies`,
`test_launch.test_a_skipped_arm_agrees_in_a_tile_of_rows`, and
`test_both_output_dimensions_are_program_ids` now launching a lane
kernel with `block_m=4`, which the launcher first divided the grid by.

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

### Phase 7b -- 2-D tiles beside lanes

**Measured and not kept.**  Lane tiles made `[BM, BLOCK, P]` (every spelling
that assumed one row axis made rank-free, as the open item listed) agree on
every design, but only Volta gains (TITAN V, best over `BLOCK` 64 and 128,
`BM` 1, 2, 4; 4 and 8 warps):

| design | Phase 7a | `BM = 1`, rank 3 | best `BM > 1` |
|---|---|---|---|
| volta | 260 | 253 | 278 (2 x 128) |
| turing | 299 | 295 | 280 |
| ampere tf32 / bf16 | 252 / 231 | 249 / 228 | 253 / 216 |
| ada / hopper | 284 / 286 | 281 / 283 | 262 / 263 |
| mxfp8 / nvfp4 | 154 / 145 | 152 / 145 | 150 / 133 |
| cdna3.f16 / bf16 / bf8 | 183 / 147 / 143 | 182 / 147 / 135 | 160 / 144 / 130 |

A 2 x 128 tile of rows is already 1,024 lane elements per tile, and 8 warps
are slower everywhere; past `BM = 4` with `BLOCK = 128`, `ptxas` takes
minutes per configuration.  Kept lanes at one row per program: +7% on one
design does not pay for a rank-3 lane path the others would not use.
Numbers vary about 10% between sessions (mxfp8 at 7a read 171 once, 154 back
to back with this), so each row above is back to back.

Two bugs found on the way, fixed in this commit:
- A skipped arm (Phase 5) restored every name it reassigned, a list included:
  its writes, which are how a list merges, were undone.  It now keeps them,
  a tail element broadcast to the rows ahead of the `if`
  (`test_launch.test_a_skipped_arm_keeps_its_writes_to_a_list`).
- Under a tile of rows but outside the column tile (Phase 7a), a runtime
  loop's carried value was not broadcast, and `_shape` gave it one dimension:
  a row's reduction ahead of the column tile did not compile.  Carries and
  pins broadcast under any tile axis, and a kernel with a tile of rows has two
  dimensions throughout
  (`test_launch.test_a_loop_ahead_of_the_column_tile_carries_a_column`).
Every design's kernel is byte-identical to Phase 7a's.

### Phase 8 -- Integer accumulation

**Done**, as revised below: `HoistInvariant` then `HoistScale` run after the
normal form under `optimize=True` (`TritonCompiler._compile_one`).  On
`fused_sum` the scale-in `2 ** -k` and scale-out `2 ** k` leave the element
loop and the scale-out leaves the sum, so the terms are the unscaled rounds,
which digit bounds put in an `int32`: the integer accumulation comes out of
storage selection with no rule of its own.  Two emitter rules follow:

- `2 ** n` on its own (`_emit_pow2`), since the hoisted power is a binding:
  the bit pattern of a float whose normal range holds every finite `n` (an
  integer), converted to the power's storage, which holds it exactly.  Where
  value classes cannot show `n` finite, `+inf`, `-inf` and NaN are selected
  apart (`inf`, `+0`, NaN); the power is computed once per row, so the selects
  are cheap.
- Phase 2's fused round follows a scale bound to a name (`_scale_in`): it
  re-emits `n` where the binding reads it unchanged, and a binding whose every
  use is fused is not materialized.  It now also requires `n` finite, which
  Phase 2 did not check: the scale-in of a fixed-point position `n` is only
  reached with `n` finite where the source program is defined (the context
  refuses anything else), but the emitter sees the rewritten program.

At first `cdna3` and `blackwell` could not show it, so their scale-in was a
multiply by the materialized power and `HoistScale` (which needs the factor
finite too) kept the scale-out on each term.  The exponent is finite -- it is
`logb` of operands under "every product is finite" -- and five gaps in
`ValueClassInfer` hid it, each fixed as a rule:

- **A join** (`_finite_source`): an overflow's lowering joins `+-inf` with the
  product, so a finite join is the product.  Through a phi, finiteness
  passes to the one side that can be finite.
- **Nested operands, and scalars** (`_exact_leaves`): "a list filled with
  exact ops of same-index reads is finite, so the lists read are" looked one
  level deep; blackwell fills `((a * b) * alpha) * beta`.  Scalar operands are
  refined too where the list is non-empty -- no element, no product -- and not
  a name the fill loop's own phi binds, which past the loop holds what the
  last round left.
- **Per-list element facts** (`_ElementKey`, `_list_key`): a fact about every
  element of a list was kept under its alias region, and every row of a 2-D
  argument shares one.  It is kept under the definition naming the list where
  the region holds more than one: a definition names one list, and the fact
  lasts until a store into its region, which the same stamp catches.
- **List parameters in their format** (`_seed_params`): a scalar parameter's
  class came from its format and a list's elements' did not.  A region is
  seeded where every list it holds is such a parameter (E4M3 scales: no
  infinity).
- **The fixpoint budget** (`_fixpoint`): rounds were counted per phi, but the
  element map is iterated too; a loop with no phis that stored anything got
  one round and dropped every element fact to the top.  Counted per phi and
  per region.
- **A scan inside an arm** (`_stored_at`, `_holds_the_scan`): a join re-stamps
  what an arm stored into, to void the arm's facts, and that made a mask
  scanned inside an arm look stored into after its scan.  A scan survives
  real stores, which are now recorded apart.

With these, `aligned_sum` (the tests' `fused_sum` shape, which gained
`fused_sum`'s special-value arm, since its grid is only finite behind one)
guards its row of `xss` directly.  The C++ output of every design is
unchanged.

Bench, back to back against Phase 7a (TITAN V, best over `BLOCK`):

| design | Phase 7a | Phase 8 | |
|---|---|---|---|
| turing | 301 | 465 | +55% |
| ada | 284 | 433 | +52% |
| volta | 260 | 372 | +43% |
| hopper | 286 | 405 | +42% |
| ampere tf32 | 252 | 359 | +42% |
| ampere bf16 | 231 | 325 | +41% |
| cdna3.f16 | 183 | 236 | +29% |
| cdna3.bf8 | 143 | 179 | +25% |
| mxfp8 | 171 | 209 | +23% |
| cdna3.bf16 | 147 | 168 | +14% |
| nvfp4 | 160 | 176 | +10% |

cdna2 and `fp64 (fma)` unchanged (no fixed-point sum).  Trackers: 14/16, all
agree, at `-r 64` and at 5 x 9.

With the value-class rules, back to back against the table above: mxfp8
207 -> 284 (+37%), cdna3.f16 235 -> 313 (+33%), cdna3.bf16 168 -> 201 (+20%),
nvfp4 176 -> 199 (+13%); the rest unchanged.

Tests: `test_emitter.test_the_scale_out_leaves_the_sum`,
`test_emitter.test_a_power_of_two_is_its_bits`, and
`test_launch.test_a_power_of_two_agrees_on_special_exponents` (against values
computed in Python: the interpreter has no `pow` under `REAL`);
`test_a_scale_in_stays_in_its_operands_storage` now checks floor and ceil
widen to `f64` rather than an `ldexp` spelling;
`test_value_class.TestFiniteSources` (a rule per case, and a rebound scalar
the fill loop's phi holds, whose program returns `inf`) and
`TestAGuardOverARow` (the scanned row learns it, a store into another row
voids it); `examples/mmasim/tests/test_triton.py::test_all_negative_zeros_agree`,
the directed NV case the plan named.

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
- **Revised:** the identity is `HoistScale` (`fpy2.strategies.hoist_scale`),
  after `HoistInvariant` hoists the scale out of the loop: on `fused_sum`
  after `comp_to_loop`, `rescale_fixed` and `simplify`, the two leave
  `t14 * sum(ts)` with `ts` the unscaled rounds.  So Phase 8 is those two in
  the Triton pipeline under `optimize=True`, not an emitter rule.  Phase 2's
  fused round then has to follow the hoisted `t = 2 ** -_k` to its definition,
  as it matches `2 ** n * x` inline today, and the sum of integer-valued
  rounds has to find a storage (digit bounds: the sum within 24 bits for
  `f32`, or an `int32`).

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
the Phase 6 kernel.  **Settled** (Phase 7b): the tile of rows beside lanes
was built and gains only on Volta (+7%), so lanes keep one row per program.
Reopen on a GPU with more registers per thread, or if an `A`-only tile
allocated at `[BM, 1, P]` (the writer's `along`, from its reads) cuts the
register pressure.

### Wider `k` loads?

+26% with one row per program, nothing with a 2-D tile.  **Provisional:** not
planned; reopen if Phase 7 does not reach the MMA designs.

### Do these gains hold on other GPUs?

The TITAN V runs `f64` at half the `f32` rate; most GPUs run it at 1/32-1/64,
so Phases 2 and 8 would matter more there, and a different L1 would move
Phase 3.  **Provisional:** measure on the TITAN V; revisit when another GPU is
available.
