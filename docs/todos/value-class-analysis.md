# Path-sensitive value classes

A four-atom lattice — is this value a NaN, an infinity, a zero, or a finite
non-zero? — refined at every branch. Much weaker than the numeric reasoning in
[symbolic-exponent-inference.md](symbolic-exponent-inference.md), and it closes a
different and larger set of holes.

## Why

The lowered `FP16` rounding emits four guards, all from one fact nothing can
prove:

```c++
assert(std::isfinite(_t)  && "fpy: rounding is undefined for this value");
double _t7 = (std::isfinite(_tmp2) ? std::ldexp(x, static_cast<int>(_tmp2))
                                   : std::pow(2.0, _tmp2) * x);
assert(std::isfinite(_t7) && "fpy: rounding is undefined for this value");
t = (std::isfinite(exp) ? std::ldexp(_t8, static_cast<int>(exp))
                        : std::pow(2.0, exp) * _t8);
```

Two exist because the target context refuses NaN and the infinities, so the
backend must assert they do not arrive.  Two exist because `std::ldexp` takes an
`int` exponent and converting a non-finite value to one is undefined.

All four are unreachable.  The program says so, three branches earlier:

```python
if fp.isnan(x):   ...
elif fp.isinf(x): ...
elif x == 0:      ...
else:                     # every guard lives here
    e = fp.logb(x)
```

Format inference cannot see it.  `x` is `FP32`, so its bound admits a NaN and
both infinities, and `logb` faithfully passes that on.

## The domain

Per value, a subset of `{NaN, Inf, Zero, Finite}` — sixteen elements, where
`Finite` means *finite and non-zero*.  The join is union, the meet is
intersection, and the height is 4, so no widening is needed.

Forward transfer functions are the special-value tables the interpreter already
implements.  `logb` is the one that matters:

| argument | `logb` |
|---|---|
| `Zero` | `Inf` (negative, but the atom does not track sign) |
| `Inf` | `Inf` |
| `NaN` | `NaN` |
| `Finite` | `{Zero, Finite}` — `logb(1.5)` is `0` |

Verified against the interpreter.  The rest are the familiar rules: NaN
propagates through arithmetic; `Zero * Inf` is `NaN`; `Finite ± Finite` is
`{Zero, Finite}`; and `min`/`max` union their operands.

Rounding is the one to get right.  Under a **bounded** context it maps `Finite`
to `{Zero, Finite, Inf}`: a small enough value underflows to zero and a large
enough one overflows — `FP16.round(1e30)` is `+inf`, and `FP16.round(1e-30)` is
`+0`, both verified.  Only an **unbounded** context — or one whose overflow has
already been moved into program text by
[`unfold_overflow`](../source/strategies.rst) — drops the `Inf`.  Reading the
bounded case as `{Zero, Finite}` would prove exactly the guards that exist to
catch it.

## Refinement

On a branch, intersect the tested value's class with what the test implies:

| test | true arm | false arm |
|---|---|---|
| `fp.isnan(v)` | `{NaN}` | `{Inf, Zero, Finite}` |
| `fp.isinf(v)` | `{Inf}` | `{NaN, Zero, Finite}` |
| `fp.isfinite(v)` | `{Zero, Finite}` | `{NaN, Inf}` |
| `v == 0` | `{Zero}` | `{NaN, Inf, Finite}` |

**The `v == 0` row is the trap.**  A false result does *not* mean non-zero: a
NaN compares false to everything, so `NaN` stays in the false arm.  Verified —
`nan == 0` and `inf == 0` are both false.  An implementation that reads
`not (x == 0)` as "non-zero and finite" is unsound.

The chain works because refinements *intersect* down an `elif` ladder.  Reaching
the final `else`:

```
x : {NaN, Inf, Zero, Finite}
  ∩ {Inf, Zero, Finite}      not isnan
  ∩ {NaN, Zero, Finite}      not isinf
  ∩ {NaN, Inf, Finite}       not (x == 0)
  = {Finite}
```

and then `logb(x) : {Zero, Finite}`, `e - 10 : {Zero, Finite}` — neither `NaN`
nor `Inf`, which is exactly what the four guards ask.

## What it closes

- **The four guards above.** Two runtime branches and two assertions gone from
  every lowered rounding.
- **The libm mapping's specials condition** — the one unmet correctness
  side-condition in that lowering.  The interpreter raises where `std::trunc`
  returns a NaN; today only a branch nothing reads reconciles them, so the
  mapping rests on an unproven claim.  See gap 3 of
  [native-lowering-roadmap.md](native-lowering-roadmap.md).
- **`float_to_fixed`'s deliberate `enable_nan` omission.** It leaves the flag off
  because "a NaN reaches a rounding only as its operand, and the branches above
  take that case" — a comment asserting precisely what this analysis would prove.

## What it does not close

Anything about *magnitude*.  The `FP64` source still has no storage, because
that needs `|x| < 2^16` from a comparison and the relational
`2^-exp · x ∈ [2^10, 2^11)`.  Classes and magnitudes are independent, and the
numeric half is recorded separately.

The split is clean enough to state as a rule: **classes fix the specials
problems, magnitudes fix the width problems.**  Four of the five open items in
the roadmap are the former.

## Why a separate lattice, not three more flags

`AbstractFormat` already carries `has_nan`, `has_pos_inf`, `has_neg_inf` — three
of the four atoms.  What it structurally cannot say is **not zero**: its own
documentation notes that `pos_bound >= 0 >= neg_bound` holds by convention and
nothing excludes zero, so every grid contains a `+0.0`.

That bit is load-bearing.  Knowing only `x : {Zero, Finite}` leaves `logb(x)`
possibly an infinity and every guard stays.  The chain needs `x : {Finite}`.

So this wants to be its own analysis rather than a fourth flag on the format
lattice — which also keeps it independent of the format lattice's arithmetic,
where a no-zero bit would have to be threaded through `__add__`, `__mul__`, the
join, and storage selection for no benefit.

## Open questions

- **Where does it live?**  A standalone analysis whose result the emitter and
  the transforms both query, or a pre-pass that annotates the AST?  The emitter
  wants it per-expression, like `format_info.by_expr`.
- **Does it need sign?**  Splitting `Inf` into `±Inf` and `Zero` into `±0` would
  let `signbit` refine too, and would let `unfold_overflow` decide its
  sign-choice branches statically.  Sixteen elements becomes sixty-four; the
  chain above needs none of it.
- **Should `unfold_overflow` and friends consume it?**  They currently probe the
  context at transform time and emit branches for whatever they cannot rule out.
  With this analysis they could skip branches for classes the operand cannot
  hold — smaller output from the same passes.
- **Loops.**  The lattice has height 4 so a fixpoint converges immediately, but
  the refinement still has to be undone correctly at a join.
