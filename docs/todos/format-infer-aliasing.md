# `format_infer` and list aliasing

A store mutates the list that *every* name in its alias region refers to.
`reaching_defs` refreshes only the name it was written through, so the analysis
used to report a bound the program could exceed:

```python
def alias_then_mutate(xs: list[fp.Real], y: fp.Real) -> fp.Real:  # xs FP32, y FP64
    with fp.FP64:
        ys = xs
        ys[0] = y
        return xs[0]        # holds FP64; the bound said FP32
```

`ys` was widened, `xs` was not. Every consumer inherited that — any pass that
trusts a bound to decide a rounding is eliminable, a constant foldable, or an
error bound computable was being told something false.

## What it does now

`format_infer` consumes `Alias` as a fifth member of `PreAnalyses`. An
`IndexedAssign` records its insert against the base's alias *region*, and every
bound write replays the region's record.

Replaying rather than editing bounds in place is what makes it order-independent:
a def computed after the store picks the widening up when it is first set, and one
computed before is re-widened when the store is recorded. Widening only ever
raises a bound, so a loop fixpoint still converges.

No import cycle — `Alias` consumes only define-use, reaching-defs and type info.

**Flow-sensitivity comes from an exclusion.** The two defs `reaching_defs` already
refreshes — the one the store reads and the one it creates — are skipped. The
first is the list's state *before* the store, so a read that precedes it keeps its
narrow bound. Without the exclusion this over-widens even a direct `xs[0] = x`,
marking the pre-store state with the inserted format.

Pinned by four tests in `test_format_infer.py`: the aliased store widens, a
read-only alias does not, a read before the store stays narrow, and a read after
a store *across a loop back-edge* is widened. The last is the one that would be a
wrong answer rather than a refusal.

It turned a refusal into a correct program rather than trading one for another:
`_gen_alias_then_mutate` left the harness's "not compared" list and the generated
matrix went from 27 to 28 bit-compared instantiations. The corpus emits
byte-identical C++ for all 54 compiling functions. An earlier version of this
document argued the element type could not be widened here because "another name
may already alias it" — with the alias fact every name widens together, so nothing
is left to disagree.

## What is left

**A called function's list parameter still refuses.** Widening it moves the
signature out from under the call site:

```
unsupported: this value is `std::vector<float>` where `std::vector<double>` is
needed, and C++ has no conversion between them.
```

That is the existing guard doing its job — a refusal, not a miscompile. The
workaround is the same one `cpp-narrower-variable-at-a-join.md` describes: pass
the wider list, and `Specialize` instantiates the callee at the wider format.
It does not apply here, because the argument is the *narrower* one.

**Flow-sensitivity is by exclusion, not by construction.** The upstream
alternative is to have an `IndexedAssign` refresh a def for every may-alias rather
than only the syntactic base — `DefineUse` already does exactly that for the
syntactic one, which is what `storage_infer`'s in-place mutation edge relies on.
That would make this exact by construction and would benefit every analysis, not
just this one. Worth doing if the exclusion turns out to be too blunt; nothing
measured so far says it is.

**A list aliased through a tuple still refuses, now in the backend.**
`_gen_list_into_tuple` — `t = (xs, y); zs = fp.fst(t); zs[0] = y; return xs[0]` —
gets a sound bound: `xs` widens to `fpy::list<double>` and so does the tuple's
field. But the emitter still wants a `float` element somewhere and refuses:

```
unsupported: `xs` holds `double` elements where `float` is needed.
```

The refusal predates this work (it was the mirror of this message, `float` where
`double` was needed). What changed is which half is wrong: the analysis is now
consistent, so the remaining disagreement is the backend computing a stale
element type for the tuple field.

**`Alias` runs without escape summaries.** `format_infer` calls it with none, so
it is maximally conservative about what a call may retain. That over-approximates
the aliasing, which is the safe direction here (more aliasing means more
widening), but it is imprecise: a callee that provably does not retain its
argument still forces a widening at the caller.
