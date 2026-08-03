# Unboxing: what is still boxed, and why

The C++ backend represents a list as a plain `std::vector<T>` wherever
`fpy2.analysis.alias` proves nothing can observe the difference, and keeps the
`fpy::list<T>` handle otherwise. **All 166 of the corpus's signature list levels
come out unboxed**, so the two shapes below are ones the corpus does not contain
— which is exactly why they need writing down.

`tests/unit/backend/cpp/test_unbox_profile.py` pins both the count and the corpus
size: an empty "still boxed" result only means something while the corpus is as
large as when it was measured.

## Still boxed

**A list a local name and a container both hold.** *Open.* `xs = [n, n]; return
(xs, 1.0)` keeps its handle — the name and the tuple's field are two places, even
though the name is dead after the return. Written inline as `return ([n, n], 1.0)`
it unboxes. The same conservatism applies to a list inside a list, so this is not
about tuples.

Closing it needs liveness, not just a sharing verdict, *and* the emitter must
learn to **move** into the container: `std::make_tuple(xs, 1)` copies a value
where it merely bumps a refcount for a handle. The two have to land together or
the change makes things slower.

**A projection whose slot is replaced.** *Deliberate, not a gap.* `row = xss[i]`
binds a reference, which is what lets `for a, b in zip(...)` over nested lists
unbox at all. But a C++ reference follows the *slot* while FPy keeps referring to
the list that was in it, so any `xss[i] = <list>` anywhere in the function rules
the reference out and the row keeps its handle. Function-wide by choice: nothing
else in the analysis is flow-sensitive, and making this the one exception would
set a bad precedent.

## Two things not to rediscover

**Keep an analysis free of C++.** `alias` answers whether a callee *retains* a
list — a fact about the program. Whether an argument must therefore carry a
handle is a separate question and belongs in `unbox`. Collapsing them looks
simpler and produces `fpy::borrow` calls that cannot bind to a `const`
reference.

The same principle was violated later, in `format_infer`, and cost real
precision before being reverted — see the boxed warning in
`cpp-narrower-variable-at-a-join.md`. Both times the pull was the same: the
backend needs one answer, so it is tempting to make the analysis give that
answer instead of reconciling in the backend.

**A representation decision is one change, not two.** Caller and callee must
agree or the call does not compile, so the analysis and the codegen for a
boundary land together. Same for `const`: a callee that writes its parameter
forces the caller's argument non-const.

## Not planned

Interprocedural precision beyond retention — a caller-driven representation
choice, with the callee specialized per argument representation. It needs one
body per representation vector, and nothing has measured a gain that justifies
that. If it is ever wanted, the representation must join the specialization key
rather than be patched on afterwards, since storage is decided per spec.

## Done

Interprocedural unboxing: `fpy2.analysis.escape` gives each function a
per-parameter summary of what it retains, so a call no longer forces handles.
Measured at the time: a decomposed kernel at a native boundary went to 2.1–2.4x
faster than the fully boxed build, matching what the monolithic version of the
same kernel gets — decomposing a program costs nothing at the boundary any more.
