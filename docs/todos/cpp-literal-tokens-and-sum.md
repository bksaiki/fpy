# Three more places the emitted C++ disagrees with FPy

Found by an adversarial audit against the criterion *"if the compiler succeeds,
the C++ compiles and behaves the same."* All three predate the join/widening
work and are independent of it. None is fixed.

## 1. A literal's predicted storage is not the type of the token written

The emitter picks a literal's storage from its *value* — `1.5` fits binary32, so
the prediction is `F32` — and `_maybe_cast` then inserts nothing because the
prediction already matches the target. But the **token** emitted is `1.5`, a C++
`double`. Where the callee is generic, the argument's real type wins:

```python
with fp.FP32:
    return fp.fmax(y, 1.5)          # y: FP32
```
```cpp
float f(float y) { return fpy::max(y, 1.5); }
// error: no matching function for call to 'max(float&, double)'
//   note: deduced conflicting types for parameter 'T' ('float' and 'double')
```

*Does not compile.* Triggered by any literal whose own narrowest storage is
already `F32` — `1.5`, `0.5`, `0.25` — in an FP32 context. FP64 contexts are safe
because the cast *is* inserted. This was 20 of 31 failures in one 3,200-program
sweep.

The silent twin picks a wide **overload** instead of failing to deduce:

```python
with fp.FP32:
    return fp.fma(y, z, 0.25)
```
emits `std::fma(y, z, 0.25)`, which resolves to the `double` overload — so the
product is rounded to binary64 and *then* to binary32, where FPy rounds the exact
result once. With `y = 24929·2^60` and `z = 673`, chosen so `y·z` is exactly a
binary32 midpoint: interpreter `0x1.000002p+84`, compiled `0x1p+84`.

Most other `<cmath>` ops are unaffected in practice — for them the double
rounding is provably harmless since 53 ≥ 2·24+2. `fma` is the exception because
its exact product can exceed 53 bits.

Fix: spell the literal at the target type (an `f` suffix, or a cast) rather than
trusting the predicted storage. One change fixes both.

## 2. An integral literal too large for a C++ integer literal is printed verbatim

`_emit_numeric_literal` prints any value with `denominator == 1` as decimal
digits, with no range check and no floating spelling. Storage selection catches
this at scalar level (it refuses), but a **list element or slot** never asks:

```python
zs = [1e300, y]
```
```cpp
std::vector<double> zs = std::vector<double>{1000000000000000052504...0160, y};
```

301 digits. GCC accepts it with only *"integer constant is too large for its
type"* and the value becomes 0. Interpreter `1e300`, compiled `0`. Same for
`xs[0] = 1e300`.

Fix: print a floating literal (`repr(float(v))`) when the value is outside the
integer range, or refuse.

## 3. `sum(xs)` does not match the interpreter's reduction

The emitter lowers it to `std::accumulate(begin, end, static_cast<T>(0))` — *n*
additions from a typed zero. `interpret/byte.py`'s `_eval_sum` starts from
`xs[0]` **unrounded** and does *n−1*. Two consequences:

- `sum([-0.0])` — interpreter `-0.0`, compiled `+0.0`, because `0.0 + (-0.0)` is
  `+0.0`.
- Under a narrower accumulator the seed rounds the element away: `sum(xs)` under
  FP32 with `xs = [5e-324]` gives the interpreter `4.94e-324` (the element,
  untouched) and the compiled code `0`.

Note the compiler *refuses* an explicit `f32 + f64` addition, but `sum` performs
exactly that addition without the check.

Fix: seed from the first element and reduce over the rest, or refuse when the
element format differs from the accumulator's.
