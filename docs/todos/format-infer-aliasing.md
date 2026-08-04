# `format_infer` ignores list aliasing

**Unsound, and not a codegen bug.** The analysis reports a bound the program can
exceed; the C++ backend implements that faithfully and gives a different answer
than the interpreter.

```python
def alias_then_mutate(xs: list[fp.Real], y: fp.Real) -> fp.Real:  # xs FP32, y FP64
    with fp.FP64:
        ys = xs
        ys[0] = y
        return xs[0]
```

| | |
|---|---|
| interpreter | `0x1.999999999999ap-4` — the FP64 `0.1` |
| compiled | `0x1.99999ap-4` — the FP32 one |
| `fn_fmt.ret_fmt` | `IEEEFormat(es=8, nbits=32)` |

`ys` and `xs` are one list, so after `ys[0] = y` it holds an FP64 value and
`xs[0]` is FP64. `_list_set_widen` widens the def `ys[0] = y` produces, but
`xs`'s def keeps FP32 — `format_infer` has no aliasing model, so nothing connects
them. `ret_fmt` then claims binary32 can hold the result, and it cannot.

Every consumer inherits this, not just the C++ backend: any pass that trusts a
bound to decide a rounding is eliminable, a constant foldable, or an error bound
computable is being told something false about an aliased list.

## Why the backend cannot fix it

Tried, and backed out. Propagating the widening to the aliased parameter in
`storage_infer.place_floors` does correct the *storage* — `std::vector<double>&
xs`, so the store no longer truncates — but the return type comes from
`ret_fmt`, so the value truncates on the way out instead. The patch changed the
ABI without fixing the answer, which is the wrong trade. The fix has to be where
the bound is computed.

## Options

**Union aliased defs in `format_infer`.** Treat `ys = xs` as making the two share
one bound. Simple and sound, and it costs precision for every consumer even where
nothing is ever stored — a read-only alias would drag both defs to the join.

**Use the alias analysis.** `fpy2.analysis.alias` already answers "may these name
one list", which is exactly the missing fact. It would keep the precision, at the
cost of a dependency from `format_infer` onto another analysis — worth checking
for a cycle, since `alias` consumes type info.

**Widen only on a store.** Narrower than unioning: when an `IndexedAssign` widens
a def, also widen anything that may alias its base. Keeps read-only aliases
precise and targets exactly the unsound case.

The third looks best, and it needs the second's fact to know what "may alias"
means. Note the parallel with `docs/todos/cpp-narrower-variable-at-a-join.md`: a
store already widens a def there too (`_list_set_widen`), and this is the same
widening failing to cross an alias edge rather than a call edge.

## Where it is pinned

`tests/infra/backend/cpp.py::_generated_xfail`, the one instantiation of
`_gen_alias_then_mutate` that has a narrower list than the value stored into it.
Strict — if it starts agreeing with the interpreter, the harness fails and asks
for the entry to be removed. The shape's other three format combinations run
normally.

The bug predates the branch's C++ work (reproduces at `50fd6aa`). It survived
because no corpus program has a list at a format other than FP64, so nothing
executed the shape until the generated matrix did.
