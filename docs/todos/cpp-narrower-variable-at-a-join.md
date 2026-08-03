# One place, two element types

Where several values reach one place — a `return`, a ternary arm, a list's
elements, a tuple's field — that place admits one C++ type. `format_infer`
bounds each expression by *its own* values, which is the question it answers and
answers correctly: `[3.0]` really is bounded by `{3}`, even where it has to be a
`std::vector<double>` because the other `return` is one.

Reconciling the two is a **storage** question, so it lives in the backend, in two
halves:

- **`storage_infer.place_floors`** raises a *definition* to the join of the places
  its value reaches. Nothing then needs converting. Lists only — a scalar
  converts free at the point of use, so widening its declaration would change a
  signature to no purpose.
- **`emitter._emit_at`** builds a *constructor* at the place's type, and converts
  anything else (`_convert_storage`). Flat lists take `std::vector`'s
  iterator-range constructor, nested ones are rebuilt with a loop, tuples field
  by field.

The two must agree about which expressions contribute to a place; `_push` and
`_emit_at` are deliberately the same shape.

`place_floors` iterates, because the constraint runs both ways: a place raises a
variable, and a raised variable raises the container built from it — once `base`
is a `vector<double>`, `scratch = [base]` has to be a `vector<vector<double>>`.
Bounds only rise and the ladder is finite, so it settles.

> Do not push any of this into `format_infer`. An earlier version did, by
> overwriting each contributor's `by_expr` entry with the join. It is sound but
> strictly less precise, and it makes a backend-independent analysis answer a C++
> question: `[1.5, 2.5]` and `[3.0]` both came out bounded by `{3/2, 5/2, 3}`, so
> every consumer — `round_elim`, `const_fold`, error analysis, the FPCore backend
> — paid for a decision only the C++ backend cares about. Same principle as
> "keep `alias` free of C++" in `unboxing-gaps.md`.

Raising a *parameter* changes the ABI, and `signature()` reports the raised type.
That was already the policy for a store — `_list_set_widen` widens a def on
`xs[0] = y` — so the join arm now gives the same answer as the store arm rather
than refusing where it widens.

## What cannot be raised

A handle cannot be rebuilt. Converting allocates, a new allocation is a different
object, and a handle exists precisely so FPy's aliasing survives — so rebuilding
one would hand the caller a list its other references no longer name.
`_convert_storage` refuses. (Unboxed → boxed is fine and is done: a value has no
aliases to lose, so `std::make_shared` is free.)

Two kinds of value have no storage of their own to raise, so they still reach
that refusal.

**A name bound as a reference.** `ys = xs`, `row = xss[i]`, a loop target — the
emitter spells these `const auto&`, naming storage that already exists.
`_PlaceFloors._pinned` skips them, via the same `binds_by_reference` predicate the
emitter and `unbox` use, so the three stay in agreement.

> This was a bug first: raising them made `storage_of` report a type the
> reference did not have, and because the binding is spelled `auto` nothing
> caught it — `const auto& ys = xs;` then `return ys;` emitted an
> `fpy::list<float>` as an `fpy::list<double>` and only the C++ compiler
> objected. Pinned in `test_a_reference_bound_name_is_not_raised`.

A *parameter* is also bound by reference, but its type is spelled from
`storage_of` in the signature, so raising it does what it says.

**A callee's result**, whose representation is fixed by the callee's own body:

```python
def g(zs: list[fp.Real]) -> list[fp.Real]:      # at FP32
    with fp.FP32:
        return zs                                # returns its argument: shared

def f(xs: list[fp.Real], c: fp.Real, y: fp.Real) -> list[fp.Real]:
    with fp.FP64:
        ws = g(xs)
        return ws if c > 0 else [y]              # refused
```

```
unsupported: the list `g` returns holds `float` elements where `double` is
needed.  Changing a list's element type needs a new buffer, and this one is
shared — so the copy would not be the list its other references name.  Either
have `g` return the wider format, or do not mix formats at this point.
```

**There is a workaround, and it is the one the message names.** Widening at the
call site is enough — `g` is then specialized at the wider argument format
automatically, since a callee's formats already follow its call site. Pinned in
`test_widening_the_call_site_is_a_real_workaround`, so the advice cannot quietly
become false.

**Closing it properly is harder than it looks, and not worth it yet.** The
tempting description — "propagate the floor across the call edge" — is wrong.
A callee's return format is a *function* of its parameter formats, so a caller
cannot ask for a wider return directly; it has to find parameter formats that
*produce* one. That is inverting the callee's body, not propagating a bound, and
for many callees no answer exists: one returning `[1.5]` cannot be widened by any
argument, so the refusal survives anyway. A fixpoint over the call graph would be
the easy half.

`unboxing-gaps.md` records the related decision not to specialize on
boxed/unboxed, and the same argument applies here: nothing has measured a cost.

The reference-bound cases are harder to close and less worth it: a reference is
what makes `for a, b in zip(...)` over nested lists unbox at all, so raising one
would mean giving it storage — a copy — which is the thing the refusal exists to
avoid.

Zero corpus programs reach any of this.

## Also considered

**Stop value-narrowing list elements.** `[0.0, 1.0]` becoming `uint8_t` was the
most common accidental trigger. Measured: 21 of 112 corpus list element types are
value-narrowed, all in `test_list*` / `test_range*` / `test_list_comp*`, none in
the kernels. Raising definitions made it moot for correctness, so this is now
purely a question of whether the narrowing earns its keep.

**A language answer.** A returned list's element format is arguably part of what
the function *is*; a signature claiming `-> list[fp.Real]` at FP64 while returning
an FP32 parameter could be rejected by `TypeInfer` rather than by storage
selection. A language decision, not a codegen one.
