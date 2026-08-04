# A real stored in an integer

The C++ backend narrows a `RealType` value to an integer type when its bound is a
set of small integers — `acc = 0.0` becomes `uint8_t`, and a list of integral
constants becomes `std::vector<uint8_t>`. That is unsound, and it is the root
cause of several wrong answers.

An integer holds neither a **signed zero** nor a **NaN**, and both are reachable:

```python
def f(y: fp.Real) -> fp.Real:
    return 0.0 * y                  # uint8_t
```
```cpp
uint8_t f(double y) {
    uint8_t _t = 0;
    uint8_t _t1 = static_cast<uint8_t>((static_cast<double>(_t) * y));
    return _t1;
}
```

| `y` | interpreter | compiled |
|---|---|---|
| `inf` | `nan` | `0` |
| `nan` | `nan` | `0` |
| `-0.0` | `-0.0` | `+0.0` |

`static_cast<uint8_t>` of a NaN is undefined behaviour, so the `0` is not even a
reliable wrong answer.

Two things conspire. `exact_binop` reports `SetFormat({0})` for `0 * x` — knowingly
unsound, see the comment there; it is kept because the alternative is a degenerate
abstract format. And the storage ladder then picks the narrowest type containing
`{0}`, which is `uint8_t`.

## The fix has to be uniform

> **A `RealType` value is stored in a float. Never in an integer.**

Half-measures do not work, and this was measured rather than guessed. Refusing an
integer storage only for a zero-containing set costs 9 corpus functions;
restricting it to *exactly* `{0}` costs 7. In both cases the failures are
*disagreements* rather than missing features — some zeros became `float` while
others stayed `uint8_t`, and then they met:

```
test_ife2:      cannot implicitly cast `float` to `uint8_t`: conversion is lossy
test_list_set1: storing a `float` into a slot of `uint8_t` would narrow it
```

Applied to every real, nothing disagrees: there are no integer-stored reals left
to meet. It also makes the inexact `{0}` bound harmless, because a float holds
whatever the multiply actually produced.

## What it costs

The narrowing is a size optimization, and dropping it is visible: 21 of the
corpus's 112 list element types are value-narrowed reals (all in `test_list*`,
`test_range*`, `test_list_comp*`), plus scalars like `uint8_t acc` inside FP64
functions. So expect test churn proportional to that, and larger emitted objects
in those cases — against emitted code that finally says `double` where FPy says
real.

*Integer*-typed FPy values keep integer storage; this is only about reals.
`range(...)` needs an integer list and must stay one — that is what the first
measured attempt broke.

## Already fixed, separately

A **negative-zero literal** no longer reports `SetFormat({0})`; it reports the
narrowest float format (`format_infer._literal_bound`), so `return -0.0` keeps its
sign and `[-0.0, -0.0]` compiles. That was free — no corpus change — because it
only touches the one literal whose exact value a `Fraction` cannot represent.
Pinned by `test_emit_scalar.py::TestNegativeZero`.
