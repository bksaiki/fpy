# `unfold_neg_zero`: state the sign of zero as program text

A third instance of the pattern behind `unfold_overflow` and `rescale_fixed`'s
`fold_specials`: take an edge rule out of the format and write it down, so a
backend that cannot express the rule can still compile the program.  See
[rounding-operator-basis.md](rounding-operator-basis.md) on whether all three
should be one operator.

Orthogonal to the native-lowering path — see "Why it is not urgent" below.

## The rule

A format with `enable_neg_zero` keeps the sign of a value that rounds to zero:
`round_C(-1e-30)` is `-0.0`, not `+0.0`. A format without it returns `+0.0`.

Stated as program text, the first is the second plus a sign restoration:

```python
# before
with C:                       # enable_neg_zero=True
    r = fp.round(x)

# after
with C':                      # C with enable_neg_zero=False
    r = fp.round(x)
with fp.REAL:
    if r == 0:
        r = fp.copysign(r, x)
```

The sign comes from the operand, which is what the format would have kept.
`fp.copysign` under `fp.REAL` is exact and already implemented in the engine.

## Why

C++ has no integer type with a signed zero, so *no* integer rung on the storage
ladder admits one:

```
MPBFixedContext(-1, 1024, enable_neg_zero=True)   -> CppScalar.F32
MPBFixedContext(-1, 1024, enable_neg_zero=False)  -> CppScalar.S16
```

One flag decides whether a rounding reaches integer storage. Anything targeting
integer arithmetic — fixed-point hardware, an integer-only DSP, a bit-exact
reference in `int16_t` — needs the sign out of the format.

## Shape

Mirrors `unfold_overflow` closely enough to share most of its structure:

- **Guard**: an unbound context (`with C:`, not `with C as c:`) whose body is
  entirely `x = fp.round(v)` over variables; `Cast` excluded; `num_randbits == 0`.
- **Candidate**: a context whose format has `enable_neg_zero` set *and* which
  can be rebuilt without it. `MPFixedContext` and `MPBFixedContext` take the
  flag directly; `SMFixedContext` has a signed zero intrinsically and would have
  to become `MPBFixedContext`; `FixedContext` (two's complement) already has
  none, so it is not a candidate.
- **Emission**: the rounding under the rebuilt context, then the `r == 0` branch
  above. Only where the source actually keeps a negative zero — probe
  `C.round(Float(c=0, s=True)).s`, as `unfold_overflow` already does.

`unfold_overflow` contains the *inverse* fixup already: its `drop_neg_zero` field
handles a source that drops the sign where its unbounded counterpart keeps it,
emitting `elif t == 0: y = 0`. This operator is that logic run the other way, so
the probe and the branch shape can be lifted from it.

## Naming

`unfold_neg_zero`, with `fold_neg_zero` for the inverse, following
`unfold_overflow` / `fold_overflow`. Same sense of "unfold": the format is a
definition and this replaces it with its body.

## Why it is not urgent

The C++ lowering no longer needs it. Integer *rounding* does not require integer
*storage* — `std::trunc` and friends are `double -> double`, so the rounding can
stay in a float type that has a signed zero. On that path
`enable_neg_zero=True` is correct and `F32`/`F64` storage is the right answer,
not a symptom.

So this is for targets that genuinely want integer arithmetic, not for unblocking
the native path. It also stands alone as a scheduling operator: a user
translating a fixed-point benchmark may want the sign rule visible in the
program regardless of any backend.

## Open questions

- Is `r == 0` the right test, or should it be `fp.signbit(r)`-free by
  construction? Under `C'` the result is `+0.0`, so `r == 0` catches exactly the
  zero case and the branch is dead for every other value.
- `SMFixedContext` cannot drop its signed zero without changing type. Decline, or
  rebuild as `MPBFixedContext` and accept that the emitted context no longer
  names the source's own class?
- Does this compose with `unfold_overflow`'s `drop_neg_zero`? Running both should
  be idempotent rather than emitting two zero branches.
