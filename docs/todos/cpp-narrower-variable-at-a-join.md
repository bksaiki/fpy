# One place, two element types

Where several values reach one place — a `return`, a ternary arm, a list's
elements, a tuple's field — that place admits one C++ type. `format_infer`
bounds each expression by *its own* values, which is the question it answers and
answers correctly: `[3.0]` really is bounded by `{3}`, even where it has to be a
`std::vector<double>` because the other `return` is one.

Reconciling the two is a **storage** question, so it lives in the backend, in
`emitter._emit_at`: it builds a *constructor* at the place's type, and converts
anything else (`_convert_storage`). Flat lists take `std::vector`'s
iterator-range constructor, nested ones are rebuilt with a loop, tuples field by
field.

> Do not push any of this into `format_infer`. An earlier version did, by
> overwriting each contributor's `by_expr` entry with the join. It is sound but
> strictly less precise, and it makes a backend-independent analysis answer a C++
> question: `[1.5, 2.5]` and `[3.0]` both came out bounded by `{3/2, 5/2, 3}`, so
> every consumer — `round_elim`, `const_fold`, error analysis, the FPCore backend
> — paid for a decision only the C++ backend cares about. Same principle as
> "keep `alias` free of C++" in `unboxing-gaps.md`.

## What cannot be converted

A **shared** list cannot be rebuilt. Converting allocates, a new allocation is a
different object, and sharing exists precisely so FPy's aliasing survives — so
rebuilding a shared list would hand its other references a list they no longer
name. `_convert_storage` refuses, and `_refuse_unsharing` writes the message.
(Unboxed → boxed is fine and is done: a value has no aliases to lose, so
`std::make_shared` is free.)

So a narrower list reaching a wider place compiles when it is a fresh value and
is refused when something else names it. Refusing is the point — the alternative
is silently unsharing, which is a wrong answer rather than a compile error.
`test_a_shared_narrower_list_is_refused` pins seven shapes: a local held by a
list or a tuple, a mixed-precision local, a parameter, and the three
reference-bound forms (`ys = xs`, `row = xss[i]`, a loop target).

The reference-bound ones are why this has to be a refusal and not a diagnostic
left to C++. The emitter spells them `const auto&`, so *nothing in the emitted
text states the element type* — `const auto& ys = xs;` then `return ys;` would
emit an `fpy::list<float>` where an `fpy::list<double>` is declared, and only the
C++ compiler would object.

A **callee's result** is the same refusal from the other side: its representation
is fixed by the callee's own body, so nothing at the call site can move it.

```python
def g(zs: list[fp.Real]) -> list[fp.Real]:      # at FP32
    with fp.FP32:
        return zs                                # returns its argument: shared

def f(xs: list[fp.Real], c: fp.Real, y: fp.Real) -> list[fp.Real]:
    with fp.FP64:
        return g(xs) if c > 0 else [y]           # refused
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

Zero corpus programs reach any of this.

## Also considered

**Raise the definition instead of converting at the place** (`place_floors`,
landed in `f05ea99` and removed again; the `cpp-old` branch is the last state
that had it). A pass raises each *definition* to the
join of the places its value reaches, so a narrower local is declared wide and
nothing needs converting. It iterates, because the constraint runs both ways: a
place raises a variable, and a raised variable raises the container built from it.

Measured cost of removing it: the corpus failure set is unchanged (identical 174
names), and the generated format matrix goes from 29 accepted instantiations to
27 — the two lost are `_gen_return_param_or_literal[32_64]` and
`_gen_ternary_param[32_64]`, both "return an FP32 list parameter or an FP64
literal list", i.e. the raise-a-parameter case. So it compiles the "local held in
a container" and "narrower parameter" shapes and nothing else — a pure
capability, fixing no defect. Against that:

- It has to skip exactly the values with no storage of their own to raise
  (reference-bound names, callee results), and getting that set wrong is a
  *silent* miscompile for the `const auto&` reason above. It got it wrong four
  times, each in the same way: asking a definition a question whose answer
  belongs to its whole storage class.
- Raising a *parameter* changes the ABI. Fine for an entry point, wrong for a
  function compiled code calls, which needs one signature on both sides. Both
  shapes it buys for a parameter depend on getting that distinction right.

Closing the callee-result case *properly* is harder than "propagate the floor
across the call edge": a callee's return format is a *function* of its parameter
formats, so a caller cannot ask for a wider return, only for parameter formats
that *produce* one. That is inverting the callee's body, and for many callees no
answer exists — one returning `[1.5]` cannot be widened by any argument. A
fixpoint over the call graph would be the easy half.

**Stop value-narrowing list elements.** `[0.0, 1.0]` becoming `uint8_t` is the
most common accidental trigger. Measured: 21 of 112 corpus list element types are
value-narrowed, all in `test_list*` / `test_range*` / `test_list_comp*`, none in
the kernels. This is the cheapest way to shrink the refusal set, and it is a
question about whether the narrowing earns its keep at all.

**A language answer.** A returned list's element format is arguably part of what
the function *is*; a signature claiming `-> list[fp.Real]` at FP64 while returning
an FP32 parameter could be rejected by `TypeInfer` rather than by storage
selection. A language decision, not a codegen one.
