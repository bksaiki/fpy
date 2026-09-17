# Casts: the third question, and four defects on its seam

The cpp backend converts a value at sixteen distinct kinds of place — a return, a
slot store, a container field, an operand rebind, a loop target, a callee
argument. Thirteen functions in `emitter.py` decide what to emit at one, and they
do not agree: the same conversion is silent at one place, refused at another, and
unchecked at a third. This reads as a pile of special cases. It is not — it is
**one model with three questions**, and only two of them have vocabulary.

| # | question | asked by |
|---|---|---|
| 1 | what type does the **place** hold? | `StorageInfer`, `choose_storage` |
| 2 | what type does the **expression** have? | `format_info.by_expr`, `_storage_for_expr` |
| 3 | does every **value** fit? | `bound_fits_in_scalar` — named nowhere, called almost nowhere |

Questions 1 and 2 are the storage/format distinction `backend-cpp.md` states at
the top. Question 3 is the one that decides whether a conversion is *sound*, and
it has no place in the vocabulary: `scalar_fits_in` asks whether the types nest,
`bound_fits_in_scalar` asks whether the values do, the two disagree by design,
and the guards keep reaching for the first where they mean the second.

`round_SINT64(x: FP32)` is the standing example. Its storage is `int64_t`
(question 1), its expression type is `int64_t` (question 2), and its value is 24
significand bits, which a `float` holds exactly (question 3).
`scalar_fits_in(S64, F32)` is `False` and the conversion is exact.

Every defect below sits on that seam.

## The four defects

All four reproduce on `cpp` at `ba9f2aff`. Each was
emitted, compiled, and — where the answer is wrong rather than merely
uncompilable — run against the interpreter.

### F5 — `std::abs` on an unsigned type does not compile

`target.py:238` builds the `Abs` row over `_int_ctxs()`, which includes `UINT32`
and `UINT64`. `std::abs` has no exact overload for either, and unlike `uint8_t` /
`uint16_t` they do not promote to `int`, so the call is ambiguous.

`tests/infra/examples/ops.py` `test_unrelated_named_guard` at
`arg_types=[UINT32, UINT32]`:

```cpp
uint32_t test_unrelated_named_guard(uint32_t a, uint32_t b) {
    ...  return std::abs(a);   // error: call of overloaded 'abs(uint32_t&)' is ambiguous
}
```

`c++ -std=c++17 -fsyntax-only` exits 1. This breaks the primary criterion — the
emitted C++ must compile — and it is the only hard compile failure found across a
4,880-TU syntax sweep.

**Fix.** `Abs` over an unsigned context is the identity. Either add
`CppOpStyle.IDENTITY` and spell it as the operand, or drop the two rows and let
`Abs` refuse there. Refusing is permitted by the criterion but loses coverage for
an operation that is trivially correct, so the identity is preferred.

### F6 — the overflow assertion admits the one value it exists to reject

`_bound_test` (`emitter.py:3633`) emits its bound through `_emit_numeric_literal`,
which prints an integral bound as an integer token (`:1706`). The comparison then
happens in the **operand's** type. For a `float` operand and an `int32_t` bound,
`2147483647` converts to `2147483648.0f`, so the test admits `2**31` and the
`static_cast` that follows is undefined. `ty` is consulted for `fabs`-vs-not and
for skipping the lower bound on unsigned — never for whether the literal survives
conversion to the type the comparison runs in.

`test_round_int_nearest_away` at `FP32`:

```cpp
auto&& _tmp1 = std::round(x);
assert((-2147483648 <= _tmp1 && _tmp1 <= 2147483647) && "fpy: overflow occurred so rounding is undefined");
int32_t _tmp2 = static_cast<int32_t>(_tmp1);
```

At `x = 2147483648.0f` the assertion passes and the compiled kernel answers
`-2147483648`; the interpreter raises `OverflowError`. `_emit_integral_round`'s
comment says `ASSERT` "is a claim the edge is never reached, which an assertion
states exactly" — it does not.

**Fix.** Round the bound *inward* in the operand's format before emitting it: the
upper bound down, the lower bound up. `_SIGMA` in `storage.py` already maps a
`CppScalar` to its format, so the directed rounding is available. This is exact,
not conservative: the values it stops admitting are the ones the operand type
cannot represent anyway — near `2**31` a `float`'s neighbours are `2147483520`
and `2147483648`, with nothing between.

`_bound_test` predates this work (`3cd2daab`, #261). What is new is its reach:
the four `test_round_int_*` corpus functions all carry the shape.

### F4 — `round` under `INTEGER` is a wrong answer at the default settings

`CppCompiler` defaults `unsafe_cast_int=True`. Under it:

```python
with fp.INTEGER:
    y = fp.round(x)
```
```cpp
assert((std::isfinite(x)) && "fpy: rounding is undefined for this value");
double y = static_cast<int64_t>(x);        // x = 1e300 -> undefined
```

`_guard_float_to_integer` asserts finiteness only. `_emit_integral_round` returns
`None` (the context is native) and `_emit_wrapping_float_to_integer` returns
`None` (`MPFixedContext` is not `MPBFixedContext`), so nothing bounds the
magnitude. `round` under `INTEGER` is fully defined in FPy, so this violates the
criterion. With `unsafe_cast_int=False` it is correctly refused.

The trap is that the guard *looks* like it covers the case — its own docstring
says the conversion is undefined "for any value the destination cannot hold, not
just NaN and infinity" — and then discharges only the second half.

**Fix.** Emit the magnitude bound beside the finiteness assertion. The bound is
`int64_t`'s, which is a *storage*-imposed range rather than a format one; that is
exactly what `unsafe_cast_int=True` means the user has accepted, and stating it
in a debug-build assertion is consistent with how the other rounding checks are
handled.

### F2 — the same conversion is silent in one place and refused in another

`_maybe_cast`'s docstring: "Pass *src* to fall back on `bound_fits_in_scalar`
when the type-level test refuses." Two of its seven call sites pass it
(`:3824`, `:3825`), and the rest do not — so a conversion the emitter performs
unconditionally one function over is refused here.

```python
def as_field(a, b, c):        # accepted
    t = (a, b)
    ...
    return t

def as_operand(a, b, c):      # refused
    s = b
    if c > 0:
        with fp.FP32:
            s = a + a
    return s
```

At `arg_types=[FP32, FP64, FP64]`, `a` has storage `double` and an FP32 value
bound. As a tuple field it emits `static_cast<float>(a)`. As an operand:
*"cannot implicitly cast `double` to `float`: conversion is lossy."* Same
definition, same storage, same target.

**Fix.** Pass `src=` at `_dispatch:1915` and `_try_widen:2022`. Measured A/B over
6,633 corpus compilations per side: 5,195 → 5,200 accepted, **5 gained, 0
regressed** (`azimuth[S32]`, `fast_2mul[U8]`, `instCurrent[S32]`,
`lod_anisotropic[U8]`, `nmse3_1[S32]`), all five clean under
`-Wall -Wextra -Werror=narrowing`, and the gained programs bit-exact against the
interpreter.

`_visit_compare:2560/2561` and `_emit_min_max:2694` also omit `src=`, but a
supremum never narrows, so those two are correct as written and should say so.

## The invariant `_convert_storage` does not have

`backend-cpp.md` records, under [One question answered in four
places](backend-cpp.md#one-question-answered-in-four-places):

> `_emit_at` returns early when `want` is a scalar, so `_convert_storage` sees
> scalars only as tuple fields, where the target is the join and never narrower
> than a contributor. Measured — zero lossy narrowings through it across the
> corpus.

Both halves are false. `_rebuild_list` (`:1363`) sends a scalar pair from a
**list element**, and the conversion narrows. `tests/infra/examples/misc.py`
`example_set` at `list[SINT8]`:

```cpp
std::array<int16_t, 2> x = ...;
...
_tmp2[_tmp1] = static_cast<int8_t>(x[_tmp1]);      // int16_t -> int8_t
```

The cast is exact, but only because `x`'s `int16_t` is an artifact of
`uint8 ⊔ int8` in the *storage* lattice while the return element came from the
*format* lattice where `{0,1} ∪ SINT8 = int8`. That is two analyses agreeing by
luck, and `_convert_storage` consults neither. A second narrowing,
`int64_t → uint8_t`, reaches it through the tuple-field recursion at `:1203`,
where `scalar_fits_in(S64, U8)` is `False` — so the target is strictly narrower
than the contributor there too.

The comment at `emitter.py:1096` asserting this invariant is wrong and must go
with the fix.

## Why the deferral no longer holds

`backend-cpp.md` defers the restructuring — one predicate per place kind — on the
grounds that it "buys maintainability rather than correctness", measured by
instrumenting every `raise` and finding 8 of ~75 sites reachable.

Re-measured with `sys.monitoring` over 13,605 compilations (237 corpus functions
× 10 argument formats × 7 contexts × 5 compiler configurations, plus mixed-format
argument pairs): `emitter.py` has 85 `raise` statements, 55 in the cast/storage
path; **13 fired, 11 of them in the cast path**. Higher than recorded, and the
shape of the conclusion survives — roughly 85% is dead under this input
distribution.

What does not survive is the conclusion. The deferral rests on "no correctness
defects here", and this document lists four, two of which (F5, F6) break the
stated criterion outright and one of which (F4) is a wrong answer at the default
settings. Two further consequences of the same seam, both closed in the working
tree, make the point about recurrence:

- `_emit_at`'s `cannot_convert` branch refused `xs[0] = 1` into a `list[SINT8]`
  with *"storing a `uint8_t` into a slot of `int8_t` would narrow it"*, although
  `1` is representable — `_require_no_narrowing` asked `scalar_fits_in` where it
  meant `bound_fits_in_scalar`. Fixed by adding the value test.
- That fix landed in `_require_no_narrowing` and **not** in `_maybe_cast`, which
  is F2. The two guards diverged again, in the same session, in the same way.

## What to do, and what not to

The doc's proposal — one predicate per place kind, at ~33 call sites — is the
wrong shape. There are sixteen place kinds, nine of them currently unchecked, and
rewriting all of them is mostly churn in the most delicate part of the emitter.

The narrower change is to **give question 3 a name and use it at every guard**:
one `value_fits` predicate over (expression, target), reading `by_expr` and
answering with `bound_fits_in_scalar`, called by `_require_no_narrowing`,
`_maybe_cast`, `_emit_deduced` and `_convert_storage`. That is four call sites,
not thirty-three, and it is the seam all four defects and both recurrences sit
on.

**Phases.** Each is commit-sized and independent; the first three are the
correctness fixes and do not depend on the fourth.

| phase | change | witness |
|---|---|---|
| 1 | F5 — `Abs` is the identity on an unsigned context | `test_unrelated_named_guard` at `UINT32` / `UINT64`, syntax-checked |
| 2 | F6 — `_bound_test` rounds its bound inward in the operand's format | `test_round_int_nearest_away` at `FP32`, run at `2**31` against the interpreter |
| 3 | F4 — magnitude bound beside the finiteness assertion under `INTEGER` | `round` under `INTEGER` at `1e19` and `1e300`, run |
| 4 | F2 — `src=` at `_dispatch` and `_try_widen`; the two supremum sites documented | `as_operand` above; the five gained corpus instantiations |
| 5 | `value_fits` named and called from the four guards; the `:1096` comment removed | the existing slot-store and ternary tests, unchanged |

Phase 2 wants a differential run rather than a unit test alone: the defect is a
boundary value, and only a bit-comparison against the interpreter pins which side
of it the emitted code lands on.

## Out of scope

- The remaining nine unchecked place kinds. Several are exact by construction
  (`_call_arg`'s literal spelling, `range` bounds) and the rest are unreachable
  while `Specialize` keys a callee on its call-site types. Recorded in
  `backend-cpp.md`; not worth closing until something changes representation
  handling.
- The unspelled implicit narrowings. `_emit_at`'s scalar early return and
  `_require_bridgeable`'s scalar pass-through both emit nothing where the doc's
  rule says every conversion is an explicit `static_cast`; a `-Wconversion`
  census flags 357 of 4,880 emitted TUs. All are exact in value today. The loop
  target (`_foreach_decl`) is the interesting one — there is genuinely nowhere to
  put a cast, and `for (int8_t x : xs)` over a `std::vector<double>` is two
  storages for one datum with no cast spelled.
- `_narrow_result`'s unguarded float-to-integer conversions (45 measured). Dead
  today — the operands come from integer storage — but `ValueClassInfer` claims
  NaN is possible for a value whose storage is `uint8_t`, so the class analysis
  and the storage ladder already disagree in the direction that would make these
  undefined. Worth a note in `backend-cpp.md`, not a fix.
