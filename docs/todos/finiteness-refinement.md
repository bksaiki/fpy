# Finiteness refinement: what is left, on `digit-bound`

The analysis work landed on `main` in #311: value-class analysis reads an
`isfinite` guard through a materialised mask as well as through a fold, and
`HoistScale` proves loop coverage from a literal trip count as well as from
`len(xs)`.  Together those closed the three things wrong with the code
`digit-bound` emitted for `fused_sum` -- the scale is hoisted, the exponent is
an integer, and the dead `isfinite` assertion is gone.

What follows is what did *not* close, and only reproduces here: every listing
below is from `digit-bound` rebased onto `main`, at `list[Real[FP16]]` of 32.

## The regression test this work still owes

- [ ] **A golden listing for `fused_sum`.**  The net on `main`
      (`tests/unit/transform/test_fused_sum_schedule.py`) stops at the
      analyses, because `fused_sum` does not compile there at all:
      storage selection refuses the return value (`no storage format contains
      MPBFloatFormat(pmax=89, emin=62, ...)`), which only digit-bound inference
      narrows.  So the two symptoms that are *facts about emitted code* -- the
      exponent's C++ type, and the absence of the `isfinite` assertion -- are
      pinned nowhere.  They belong in a test here, and it passes as of the
      rebase.

## Weirdness still in the output

None is a wrong answer; they are all code nobody would write.

- [ ] **(C6) The overflow assert and the storage disagree on width.**  The
      assert checks the `int64_t` range, then the value is narrowed to
      `int16_t` by an implicit conversion with no check and no cast:

      ```cpp
      assert((-9.2e18 <= std::trunc(_t) && std::trunc(_t) <= 9223371487098961920));
      int64_t _tmp7 = static_cast<int64_t>(_t);
      int16_t _t16  = _tmp7;
      ```

      Digit-bound inference is what proves 13 bits suffice, so the narrowing is
      sound -- but the assertion guards a bound the program does not rely on
      and says nothing about the one it does.  Worth checking whether
      `-Wconversion` fires.
- [ ] **A double cast.**  `static_cast<float>(static_cast<float>(-14))`.
- [ ] **Reciprocal scales computed twice.**  `t = 2 ** -_k` and `t14 = 2 ** _k`
      are each their own `std::ldexp`; one is the other's reciprocal.
- [ ] **The `isfinite` mask is materialised.**  All 32 elements are computed
      into a `std::array<bool, 32>` before `std::all_of`, so a non-finite first
      element still costs the whole pass.  A short-circuit loop would do.  (The
      mask in the *AST* is deliberate and is what the value-class work learned
      to read; this is only about what the backend emits for it.)

Four more went away with the finiteness work, so they are not worth hunting
again: the `std::isnan` test on a literal, the `std::signbit` tie-break in the
`max` fold, and the `std::pow(2.0, _k) * 1` fallback in both scale expressions
-- all of which existed because `e` and `_k` were floats that might have been
special, and all of which the integer exponent removed.

## Not a bug, but unstated

- [ ] **The `max(logb(x), emin)` clamp has an unproved precondition.**  It is
      `rescale_fixed` on this branch that inserts it, not the source.  It is
      semantics-preserving here -- checked against the untransformed function
      on all-subnormal, mixed and all-normal FP16 inputs, identical results --
      because a clamp to `emin` only moves the rounding position, and
      `emin - P` still sits below the finest representable digit when
      `P >= p - 1` (here `12 >= 10`).  **That inequality is nowhere asserted**,
      and it is load-bearing: the clamp is also what rules `logb(0)`'s `-inf`
      out of `e`, and so what stands between `int8_t` and `float` for it.
      Assert it where the clamp is introduced.

## Not this branch's problem

`HoistScale.refusals` reports *"no scaled list write fills the reduction"*
twice, for the program's other two reductions.  Those appear on `main` too,
they persist in every lowering of this program, and they have never been looked
at.
