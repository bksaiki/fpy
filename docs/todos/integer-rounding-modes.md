# Non-RTZ rounding into integer storage

`Round` under an integer context lowers to a `static_cast`, and C++ integer
conversion truncates toward zero. So the backend accepts `RTZ` there and refuses
every other mode.

It does not have to. Round to an integral *value* in the float type first, and
the cast that follows is a pure conversion:

```cpp
static_cast<int32_t>(std::ceil(x))    // RTP
```

`_emit_integral_value` already spells each mode this way for float storage; the
integer path is the same lowering with a cast on the end.

## Why the float step does not round twice

`trunc` / `floor` / `ceil` / `round` / `nearbyint` each return a value that is
already an integer *and* exactly representable in the same float type — above
`2 ** p` every float is an integer, and below it every integer is exact. So the
float step is exact and the cast performs no rounding of its own.

What the cast can still do is fail: a NaN, an infinity, or a magnitude the
integer type cannot hold is undefined in C++. The bound assertion covers all
three, and it has to be taken on the *rounded* value rather than the operand --
under `RTP` an operand inside the bound can round to one outside it.

## Scope

Only the modes with a direct call: `RTZ`, `RTN`, `RTP`, `RNA`, `RNE`
(`_INTEGRAL_ONE_CALL`). `RAZ` / `RTO` / `RTE` are composed from several calls;
`_emit_integral_value` spells them and the float path uses them, so widening to
them later costs one line and some tests.

`RNE` is `std::nearbyint`, which follows the *live* `fenv` mode, so it carries
the same `FE_TONEAREST` precondition the float path checks.

## The overflow rule is a separate gate

A non-RTZ integer context can never be native -- `_NATIVE_CTXS` is fixed and
entirely `RTZ` -- so it always takes the non-native path, where only
`OverflowMode.ASSERT` is accepted. That is not a rule about rounding: storage is
chosen to *contain* a format rather than equal it, so the C++ type's wrapping
happens at the type's bound and not the context's, and `SATURATE` / `WRAP` /
`OVERFLOW` are behaviour the cast does not perform. `ASSERT` alone needs none.

So the usable shape is `fp.SINT32.with_params(rm=RM.RTP, overflow=OM.ASSERT)`,
or `unfold_overflow` first. Today the `RTZ` gate fires before the overflow one
and hides it; once it is relaxed, programs that saw the rounding refusal will
start seeing the overflow refusal, so each message should name which it is.

## Plan

**1. The lowering.** Round through `_INTEGRAL_ONE_CALL[rm]` in
`_emit_cast_round`, assert the bound on the rounded value, and cast that.
Relax `_validate_ctx_storage`'s integer check from `== RTZ` to membership.
Carry the `FE_TONEAREST` precondition. Unit tests per mode, plus the case where
`RTP` rounds an in-bounds operand out of bounds.

**2. Bit-exact witnesses.** `test_round_int_up` / `_down` / `_nearest_away` /
`_nearest_even` in `tests/infra/examples/ops.py`, one per mode the cast does not
perform, compared against the interpreter by `tests.infra.backend.cpp --mode
run`. Each guards its operand: a NaN or an infinity has no integer to convert
to, and the generated pool contains both.

## Left over

A context whose *range* matches a C++ type but whose mode is not `RTZ` cannot
reach the wrapping lowering: `_emit_wrapping_float_to_integer` implements
float-to-integer `WRAP` but is gated on `is_native_ctx`, which keys on the whole
context including `rm`. So `fp.SINT32.with_params(rm=RM.RTP)` is refused for its
*overflow* rule although both halves of what it needs exist. The two workarounds
are `overflow=ASSERT` and `unfold_overflow`.

That refusal also mis-states its reason for this case -- it says the storage is
wider than the format, where for `SINT32` the two coincide exactly; what the
cast actually cannot do is wrap, C++ promising that only for integer sources.
And it advises `unfold_overflow`, which leaves such a program unchanged.
