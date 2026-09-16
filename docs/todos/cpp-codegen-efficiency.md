# C++ codegen: materialization, dead emissions, and integer exponents

Five shortcomings visible in one program, a baseline that measures them, and a
phased plan. The program:

```python
@fp.fpy(ctx=fp.REAL)
def f(xs):
    if any([fp.isnan(x) for x in xs]):
        return fp.nan(), 0
    elif any([fp.isinf(x) for x in xs]):
        return fp.inf(), 0
    else:
        max_e = max([fp.logb(x) for x in xs])
        with fp.FP16:
            ys = [fp.round(x) for x in xs]
        with fp.FP32:
            acc = sum(ys)
        return acc, max_e
```

Compiled with `unfold=ROUNDINGS` over `list[FP32][32]`. It is a clean A/B: the
same syntactic shape three times, and only `any` fuses.

| source | emitted | why |
|---|---|---|
| `any([isnan(x) for x in xs])` | running `acc6`, no array | `ReduceFusion` fires |
| `max([logb(x) for x in xs])` | `std::array<float,32>` + a second loop | `ReduceFusion` covers only `any`/`all` |
| `ys = [round(x)...]; sum(ys)` | `std::array<float,32>` + `accumulate` | the fuse matches syntax, and the name blocks it |

## The five

**S1 — `ReduceFusion` covers only the boolean reductions.** `Sum` / `AMin` /
`AMax` materialize. Tracked in `backend-cpp.md` §*Narrowing inside
`std::accumulate`*, which blames `_maybe_cast`; `backend-independence.md` §Open
records that the FPy-level unfold dissolves that blocker, since an FPy-level
accumulator is an ordinary variable `StorageInfer` classes like any other.

**S2 — the fuse is syntactic, so a named intermediate defeats it.** Measured
directly: `any([...])` fuses, `bs = [...]; any(bs)` does not. Not tracked
anywhere. `CopyPropagate` cannot help — it substitutes var-to-var only.

**S3 — `_emit_empty` binds dimensions the type already carries.** It binds each
non-constant dimension *before* testing `_all_sized`, so a fixed-size
comprehension emits a dead `auto&& _tmp1 = static_cast<uint8_t>(xs.size());`.
This is the tracked "CSE, for the two names one `len(t)` gets". The bind is
load-bearing on the unsized path (`std::vector<bool>(_tmp1, ...)` reads it), so
the fix is the ordering, not the bind.

**S4 — `_emit_sum` emits a dead empty guard.** `ys.size() == 0 ? 0 : ...` where
`arg_ty.size` is statically 32.

**S5 — `logb` is integer-valued but stored as `float`.** The inference is
*exact*, and shows why:

```
MPBFixedFormat(nmin=-1, pos_maxval=127, neg_maxval=-149,
               enable_nan=True, enable_inf=True, ...)
```

An integer grid over `[-149, 127]` — `int16_t` territory — that also admits NaN
and ±inf. No integer rung admits those, so the search falls through to `float`.
`exact_logb` sets `has_neg_inf=True` *unconditionally*, because `logb(±0)` is
`-inf` and every format contains a zero.

`ValueClassInfer` already answers this. `_LOGB` maps `NaN -> NaN`,
`Zero -> Inf`, `Inf -> Inf`, `Finite -> Zero|Finite` (`logb(1.5)` is `0`), and on
a program guarding NaN, Inf and zero under `REAL` the `Logb` expression reports
`ZERO|FINITE` — NaN and Inf both excluded. That is exactly what is missing from
the format.

**The one gap is that `StorageInfer` never sees it.** `compiler.py` computes
`class_info` and hands it to the emitter; `StorageInfer.infer` takes formats
only, so the `enable_nan` / `enable_inf` flags stand and the search falls
through to `float`.

`ValueClassInfer` does lose the class under a *concrete* context — `_rounded`
replaces the exact class with `representable_classes(ctx)` wholesale rather than
intersecting, so the same program at `FP32` reports `TOP`. That is sound
(rounding can move a value between classes: overflow to `Inf`, underflow to
`Zero`, saturation back to finite) and it does not matter here, because the
exact integer format only survives under `REAL` anyway.

Integer storage is otherwise reachable — a loop counter already gets `int8_t`
and an integer accumulator `int16_t` — so the ladder works; the special-value
flags are the whole blocker.

## Baseline

`-O2`, pinned with `taskset`, min over 15 trials, each kernel's result folded
into a checksum so a speed difference is never an answer difference. The
harness was a one-off and is not in the tree; these numbers are the record. At
`49e471b9`, `n=1024`:

| kernel | ns/call | ns/elt | what it isolates |
|---|---|---|---|
| `sandbox` | 23850 | 23.3 | the whole program; the FP16 ladder dominates |
| `reduce_sum` | 985 | 0.96 | S1, `Sum` over a comprehension |
| `reduce_max` | 3887 | 3.80 | S1 + S5, `AMax` over a comprehension |
| `any_named` | 477 | 0.47 | S2, materialized |
| `any_direct` | 719 | 0.70 | control, already fused |

Per-element cost is flat from `n=256` up (swept 32 / 256 / 1024 / 4096 /
16384), so 1024 is chosen for the 4KB intermediate, not for a cache effect.

### The baseline contradicts the premise

**`any_direct` — the fused one — is 51% *slower* than `any_named`.** Measured
both ways, at `n=1024`:

| kernel | sized (`std::array`) | unsized (`std::vector`) |
|---|---|---|
| `any_named` (materialized) | **476** | **3491** |
| `any_direct` (fused) | 717 | 720 |

`ReduceFusion`'s docstring claims the fused form "measures 2-4x faster", and
that is right — for the unsized path, where it is 4.85x. On the sized path it
is a 1.50x *loss*. The fused number is the same either way because fusing
removes the list, so no representation is chosen.

### But the representation is the bug, not the fuse

The cliff is not heap-versus-stack. `reduce_sum` is 988 sized against 976
unsized and `reduce_max` 3915 against 3779 — a heap allocation costs those
nothing. It is `std::vector<bool>` specifically, and isolating it in plain C++
says so:

| representation | ns/call |
|---|---|
| `std::array<bool, 1024>` | 478 |
| `std::vector<bool>` | **3483** |
| `std::vector<uint8_t>` | **476** |
| fused loop | 724 |

Bit-packing is the entire 7.3x. A `std::vector<uint8_t>` matches the stack
array to within noise, so the allocation is free and the packing is not.

**The spelling is where the cost is** — `CppList.format()` spells a `bool`
element as `bool`, which is `std::vector<bool>` in the two unsized spellings
(`std::vector<bool>` and `std::shared_ptr<std::vector<bool>>`);
`std::array<bool, K>` is unaffected and already fast. Spelling those two
`uint8_t` would take the unsized bool reductions from 3491 to about 480.

**Decided against.** Changing the representation of `list[bool]` to buy back a
fuse's worth of time is a larger commitment than the win justifies, and it
would make one FPy type spell its element two ways depending on representation.
`ReduceFusion` stays, and its docstring now states where it pays and where it
does not. That leaves a known 1.5x loss on the sized path, accepted: the pass
runs before `Specialize`, so gating it on representation means moving it or
splitting the decision from the rewrite, and neither is worth doing for this.

What the measurement argues against — kept as optional Phases 7 and 8, with
the evidence, so they are not picked up again without it:

- **Fusing `Sum` / `AMin` / `AMax` is not worth doing.** Their sized and unsized
  numbers agree to ~1%, so the intermediate is nearly free: their element is a
  float, with no bit-packed specialization to pay for, and `reduce_sum` at
  0.96 ns/elt is already at the latency of a serial FP add chain that fusing
  does not shorten. For `AMax` fusing is actively worse — a hand-written fused
  `max([logb(x) ...])` measures 11% slower (3703ns against 3343ns), since the
  accumulator spills across each opaque `logbf` call. This retires the plan's highest-risk item — the `_eval_sum`
  seeding trap — on evidence rather than by doing it carefully.
- **Teaching the fuse to see through a name is worse than not.** It would make
  the pass fire more often, and firing is the pessimization on the sized path.

## Headroom: what any of this can be worth

The generated C++ for `sandbox` hand-rewritten, every variant checksum-identical
at `n=1024`:

| variant | ns/call | vs generated | what changed |
|---|---|---|---|
| generated | 23062 | — | — |
| reductions fused | 23472 | **+1.8%** | both intermediate arrays gone |
| one pass | 22507 | -2.4% | all four scans merged |
| hardware `fp16` | 5361 | **-77%** | the ladder replaced, `logb` untouched |
| + bit-extract `logb` | 4315 | -81% | both |

**The FP16 rounding ladder is ~77% of this program, and `logb` another ~4%.**
Fusing the reductions on the real program buys nothing — it measures slightly
negative, matching the +2.5% the pass toggle reports for `sandbox`. Merging all
four passes buys 2-6%.

So every phase in this file competes over the last quarter of the runtime, and
mostly over a few percent of it. Native lowering
([native-lowering-roadmap.md](native-lowering-roadmap.md)) is worth more than
all of them together.

One caveat against reading that as "emit `_Float16`": the hardware variant needs
`-mf16c`. Without it GCC emits `__truncsfhf2` and the same code measures 57000 ns
-- 2.3x *worse* than the ladder. The ladder is the right default; a hardware
path needs a target-feature story, not a substitution.

## Phases

Each is about one commit. Pause for review between them; run the listed tests
only, and the full suites at the end.

**Phase 0 — the benchmark.** A one-off harness plus the baseline above. Kept
as numbers rather than as code: it was worth building to answer these
questions and not worth maintaining afterwards.

**Phase 1 — stop emitting what the type proves dead** (S3, S4).
`_emit_empty` hoists the `_all_sized` test above the dimension binding, still
visiting args for their statement effects; `_emit_sum` drops the guard when
`arg_ty.size` is set. `test_bind_profile.py` moves `_emit_empty` 30 -> 28.
Tests: `test_bind_profile`, `test_emit_sum`, `test_emit_array`,
`test_emit_list`, `test_storage`. *Prototyped; lowest risk.*

**Phase 2 — resolve the fusion question.** Done, above. The answer was that
the fuse is not what needs gating and the `bool` spelling is what costs, and
then that changing the spelling is not worth it either. The commit is this
file plus `ReduceFusion`'s docstring, which now states the two cases and their
numbers. The durable statement of that now lives in `backend-cpp.md` under
"When `ReduceFusion` pays, and when it costs"; what is below is the record of
how the decision was reached. S1 and S2 move to Phases 6 and 5, optional and
argued against.

**Phase 3 — a `Logb` rule for `ValueClassInfer`** (S5a). **Not needed.** The
rule is already there and already precise under `REAL`; see S5. Verified by
compiling the guarded witness and reading the class off the `Logb` expression.
No commit.

**Phase 4 — `StorageInfer` consults value classes** (all of S5, now that
Phase 3 is closed). Pass `class_info` into `StorageInfer.infer` and clear the special-value flags a class rules out
before the domain search, joining across a storage class so one member that can
be ±inf still forces `float`. Expected: `max_e` to `int16_t` under a zero
guard, and the FP16 ladder's own `exp` to `int8_t`, dropping two
`static_cast<int>` round-trips into `std::ldexp`. `reduce_max`'s 3.8 ns/elt is
mostly the IEEE max chain's NaN and signed-zero branches, which the integer
path does not have, so this is the likeliest of the phases to show up in the
benchmark. Tests: `test_storage_infer`, `test_storage_ladder`,
`test_class_guards`, `test_unfold_round`, `test_lowered_roundtrip`,
`test_bind_profile`.

**Phase 5 (optional) — fuse through a single-use name** (S2). Match a reduction
whose argument is a `Var` whose definition is a `ListComp` with exactly one use
in the same block; leave the dead definition to `Simplify`'s DCE. Gate on no
intervening statement writing a name the comprehension reads. Tests:
`test_reduce_fusion`, plus a cpp witness. **The measurement argues against it**:
it makes the fuse fire more often, and on the sized path firing is the
pessimization (476 -> 717). Worth doing only if the sized-path loss is addressed
first, which nothing here plans to do.

**Phase 6 (optional) — cover `Sum` / `AMin` / `AMax`** (S1). The trap: `_eval_sum`
seeds with `val[0]` **unrounded** and does *n-1* rounded adds, so `acc = 0;
acc = acc + b` is wrong twice over — it rounds the seed and adds *n* times. It
needs a peeled first iteration, sound only where the length is proven nonzero.
`AMin`/`AMax` are easier: the empty list is already undefined. Would also
correct `backend-cpp.md` §*Narrowing inside `std::accumulate`*. Tests:
`test_reduce_fusion`, `test_emit_sum`, `test_emit_min_max`,
`test_lowered_roundtrip`. **The measurement argues against it too**: sized and
unsized agree to ~1% for these, and a hand-written fused `max([logb(x) ...])`
is 11% *slower*. This is the highest-risk item in the file and currently the
lowest-value one.

**Phase 7 — sign-split `ValueClass`, and order-aware `min`/`max`.** Done.
`POS_INF` and `NEG_INF` are separate atoms with `INF` kept as their composite,
so every consumer asking `cls & INF` reads unchanged; `logb`, `abs`, `Neg`,
`Add`/`Sub` and `ConstInf` became sign-aware, and `StorageInfer` now asks about
the two infinities separately. A selection is no longer the join of its
operands: `max` is `+inf` when *some* operand can be and `-inf` only when
*every* one can.

The witness is a clamp, which previously narrowed nothing:

```python
if fp.isnan(x): return 0
else:           return min(max(fp.logb(x), -126), 128)   # `int8_t`, was `float`
```

(`int8_t` rather than `int16_t` because `logb` of an FP32 tops out at 127, so
the upper clamp never binds.)

**Phase 8 — an element class for lists.** Done, and it took both halves:

- *Forward* — a list definition carries a class for its elements, joined from
  the stores that build it (`empty(...)` is bottom, a store joins in, a literal
  joins its elements, a copy inherits). Phis merge it and the loop fixpoint
  iterates on it, with *absent* reading as the top so a store on one path cannot
  look like a promise about the other. Consumed by an element read -- `ListRef`
  had no case at all and fell through to the top -- by a `for` target over a
  list, and by `AMin`/`AMax`, which returned the top outright.
- *Backward* — `_implied_elements` matches the lowered reduction loop and reads
  it as a universal: `all(...)` refines the taken arm, `any(...)` the untaken
  one. Sound because FPy has no `break`, so a `for` runs the whole iterable, and
  the match is strict: a literal seed, a step that is exactly `acc <op> b`
  naming that phi, a predicate reading only the loop target, and no store into
  the list anywhere in the body.

The motivating program now gives `int8_t max_e` unchanged. The clamp is still
load-bearing: the guards cannot exclude a *zero*, so `logb(0)` is still `-inf`,
and `max(logb(x), FP32_EMIN)` is what removes it -- which is Phase 7's ordering
rule. `if all([fp.isfinite(x) for x in xs])` works as well as the `any` form.

**Phase 9 — `min`/`max` reach the library form.** With both facts known the
open-coded predicate has nothing left to decide, so `_emit_ieee_min_max` emits
`std::min`/`std::max` -- which is what the integer path already did. `std::max`
is the predicate verbatim; `std::min` differs only on a tie, and *zero_tie_free*
is exactly the promise that a tie is between equal non-zero values. The operands
stay bound: the library form returns a *reference* to one of them.

`_emit_amin_amax` also asks now, of the element class Phase 8 gives it -- it
passed no facts at all before, so a reduction over a provably-finite list still
emitted the NaN propagation. The `signbit` tie stays, and correctly: `logb` of a
value in `[1, 2)` is a zero, and `ValueClass.ZERO` carries no sign, so
*zero_tie_free* is not provable. Splitting `±0` is the remaining half of
Phase 7's sign work and would close it.

Ordering: 1, 4, 7, 8 and 9 all landed, in that order; 3 closed with no work,
and 5 and 6 stay optional and argued against. 5 and 6 both extend `ReduceFusion` and both are argued
against by the benchmark; neither is on the path.

## Not shortcomings

- `std::array<T,N> t = std::array<T,N>{}` zero-initialises before the loop
  overwrites it. `-O2` elides it, and dropping the initialiser would be a real
  semantic loosening.
- `int8_t` loop counters compared against a `uint8_t` length: both promote to
  `int`, so there is no sign-compare warning and no bug.
- `float` for `max_e` in the program above is **correct as written**. `xs` may
  contain a zero, `logb(0)` is `-inf`, so the value really can be infinite.
  Phase 4 changes that program not at all; it pays where a guard excludes
  zero, and inside the ladder, which already does.
