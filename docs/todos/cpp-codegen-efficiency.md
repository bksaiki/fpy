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

Two things are needed, and neither alone suffices:

- `ValueClassInfer` has **no `Logb` rule**. Measured on a program guarding NaN,
  Inf and zero: the argument narrows to `ValueClass.FINITE`, and the `Logb`
  result is still `ValueClass.TOP`.
- `StorageInfer` never sees value classes. `compiler.py` computes `class_info`
  and hands it to the emitter, but `StorageInfer.infer` takes formats only.

Integer storage is otherwise reachable — a loop counter already gets `int8_t`
and an integer accumulator `int16_t` — so the ladder works; the special-value
flags are the whole blocker.

## Baseline

`tests/infra/backend/cpp_bench.py`, `-O2`, pinned, min-of-trials, checksummed.
At `49e471b9`, `n=1024`, 15 trials x 7 loops x 1953 calls:

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

**`any_direct` — the fused one — is 51% *slower* than `any_named`.** Reproduced
across a five-point sweep and two full runs; the minima are stable even where
the spread is not.

`ReduceFusion`'s docstring claims the fused form "measures 2-4x faster", and
that is not wrong — it is measured against a different representation:

```
length proven   →  std::array<bool, 1024>   stack, no allocation
length unknown  →  std::vector<bool>        heap + per-element bit twiddling
```

The 2-4x is the `std::vector<bool>` path. On the `std::array` path the
materialized form vectorizes — a fill loop with no loop-carried dependency,
then a vectorized `any_of` — while the fused form serializes on the
accumulator. Rewriting `acc || b` as `acc | b` does not recover it (measured:
no reliable difference).

**So S1 and S2 are not automatically wins.** Eliminating a materialization is a
win when it removes an allocation and a loss when it removes a vectorizable
loop. `reduce_sum` says the same from the other side: at 0.96 ns/elt it is
already at the latency of a serial FP add chain, so the array is nearly free
there and removing it cannot buy much.

This is the plan's main open question, and it has an awkward shape:
`ReduceFusion` runs *before* `Specialize`, so it cannot know whether the list
will be a `std::array` or a `std::vector`. Options, none yet chosen:

1. decide the fuse after storage is known, which means moving it or splitting it;
2. leave the decision to the emitter, against the direction `backend-independence.md` argues for;
3. fuse unconditionally and fix the fused *shape* to vectorize, which the `|` measurement suggests is not simply a matter of the operator.

## Phases

Each is about one commit. Pause for review between them; run the listed tests
only, and the full suites at the end.

**Phase 0 — the benchmark.** `tests/infra/backend/cpp_bench.py` plus the
baseline above. Done.

**Phase 1 — stop emitting what the type proves dead** (S3, S4).
`_emit_empty` hoists the `_all_sized` test above the dimension binding, still
visiting args for their statement effects; `_emit_sum` drops the guard when
`arg_ty.size` is set. `test_bind_profile.py` moves `_emit_empty` 30 -> 28.
Tests: `test_bind_profile`, `test_emit_sum`, `test_emit_array`,
`test_emit_list`, `test_storage`. *Prototyped; lowest risk.*

**Phase 2 — resolve the fusion question.** Not code first: decide, against the
benchmark, whether fusion should be gated on representation. Phases 3 and 4
below are contingent on the answer, and may reduce to "gate the existing fuse"
rather than "extend it".

**Phase 3 — fuse through a single-use name** (S2). Match a reduction whose
argument is a `Var` whose definition is a `ListComp` with exactly one use in the
same block; leave the dead definition to `Simplify`'s DCE. Gate on no
intervening statement writing a name the comprehension reads. Tests:
`test_reduce_fusion`, plus a cpp witness.

**Phase 4 — cover `Sum` / `AMin` / `AMax`** (S1). The trap: `_eval_sum` seeds
with `val[0]` **unrounded** and does *n-1* rounded adds, so `acc = 0;
acc = acc + b` is wrong twice over — it rounds the seed and adds *n* times. It
needs a peeled first iteration, sound only where the length is proven nonzero.
`AMin`/`AMax` are easier: the empty list is already undefined. Corrects
`reduce_fusion.py`'s docstring, `backend-cpp.md` §*Narrowing inside
`std::accumulate`*, and `backend-independence.md`'s "`ReduceFusion` never
fires". Tests: `test_reduce_fusion`, `test_emit_sum`, `test_emit_min_max`,
`test_lowered_roundtrip`.

**Phase 5 — a `Logb` rule for `ValueClassInfer`** (S5a). `logb(FINITE)` is
finite, `logb(ZERO)` and `logb(INF)` are infinite, `logb(NAN)` is NaN. Analysis
only, no codegen change. Tests: `test_value_class`.

**Phase 6 — `StorageInfer` consults value classes** (S5b). Pass `class_info`
into `StorageInfer.infer` and clear the special-value flags a class rules out
before the domain search, joining across a storage class so one member that can
be ±inf still forces `float`. Expected: `max_e` to `int16_t` under a zero
guard, and the FP16 ladder's own `exp` to `int8_t`, dropping two
`static_cast<int>` round-trips into `std::ldexp`. `reduce_max`'s 3.8 ns/elt is
mostly the IEEE max chain's NaN and signed-zero branches, which the integer
path does not have, so this is the likeliest of the phases to show up in the
benchmark. Tests: `test_storage_infer`, `test_storage_ladder`,
`test_class_guards`, `test_unfold_round`, `test_lowered_roundtrip`,
`test_bind_profile`.

Ordering: 5 before 6, or 6 does nothing for `logb`. 1 is independent of
everything. 3 and 4 are both downstream of 2's decision.

## Not shortcomings

- `std::array<T,N> t = std::array<T,N>{}` zero-initialises before the loop
  overwrites it. `-O2` elides it, and dropping the initialiser would be a real
  semantic loosening.
- `int8_t` loop counters compared against a `uint8_t` length: both promote to
  `int`, so there is no sign-compare warning and no bug.
- `float` for `max_e` in the program above is **correct as written**. `xs` may
  contain a zero, `logb(0)` is `-inf`, so the value really can be infinite.
  Phases 5 and 6 change that program not at all; they pay where a guard
  excludes zero, and inside the ladder, which already does.
