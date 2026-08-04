# `format_infer` ignores list aliasing

The analysis reports a bound the program can exceed. It no longer produces a
wrong answer — the backend refuses these programs — but the imprecision is real
and the refusal is a rough edge, not a fix.

```python
def alias_then_mutate(xs: list[fp.Real], y: fp.Real) -> fp.Real:  # xs FP32, y FP64
    with fp.FP64:
        ys = xs
        ys[0] = y
        return xs[0]
```

`ys` and `xs` are one list, so after `ys[0] = y` it holds an FP64 value and
`xs[0]` is FP64. `_list_set_widen` widens the def that `ys[0] = y` produces, but
`xs`'s def keeps FP32 — `format_infer` has no aliasing model, so nothing connects
them, and `fn_fmt.ret_fmt` claims binary32 can hold the result.

Every consumer inherits this, not just the C++ backend: any pass that trusts a
bound to decide a rounding is eliminable, a constant foldable, or an error bound
computable is being told something false about an aliased list.

## Why the backend refuses instead of miscompiling

The store is where it becomes observable, and the C++ slot is narrower than the
value:

```
unsupported: storing a `double` into a slot of `float` would narrow it, and the
list would then not hold the value FPy says it does.  Round the value to the
list's format, or build the list at the wider one.
```

C++ *would* accept that store and narrow silently, which is the one shape where
a format disagreement is a wrong answer rather than a compile error — so
`_visit_indexed_assign` checks `scalar_fits_in` and refuses. The container's
element type cannot simply be widened from there: another name may already alias
it, and that is exactly the fact `format_infer` does not track.

## Fixing it properly

Three options, in increasing cost:

**Widen only on a store.** When an `IndexedAssign` widens a def, also widen
anything that may alias its base. Keeps read-only aliases precise and targets
exactly the unsound case. Needs the next option's fact.

**Use the alias analysis.** `fpy2.analysis.alias` already answers "may these name
one list", which is the missing input. Costs a dependency from `format_infer`
onto another analysis — check for a cycle, since `alias` consumes type info.

**Union aliased defs.** Treat `ys = xs` as making the two share one bound.
Simple and sound, and it costs precision for every consumer even where nothing is
ever stored.

The first, using the second's fact, looks right. Note the parallel with
`cpp-narrower-variable-at-a-join.md`: a store already widens a def there too, and
this is the same widening failing to cross an *alias* edge rather than a call
edge.

## Where it is pinned

`tests/infra/backend/cpp.py::_gen_alias_then_mutate`, which is now *refused*
rather than compared — it appears in the harness's "not compared" list. The bug
predates the branch's C++ work (reproduces at `50fd6aa`); it survived because no
corpus program has a list at a format other than FP64, so nothing executed the
shape until the generated matrix did.
