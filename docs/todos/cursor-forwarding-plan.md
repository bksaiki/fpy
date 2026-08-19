# Plan: locations that survive rewrites

Item 3 of `docs/todos/scheduling-language.md`. Item 2 (discoverability) is
de-prioritized; the one slice of it this item needs — listing candidate sites as
cursors — is phase 6 here.

## Context

`where` is an **index into a scan**: each transform re-counts its candidate sites
over the whole program and picks the *n*th. Transforms change how many candidates
there are — `unfold_overflow` turns one rounding block into several statements —
so an index chosen against one program means something else after any rewrite.
The consequence is sharper than order-dependence: once the first operator has
rewritten at a program point, every later operator should be aimed at *that*
point, and instead each re-scans with an index that no longer refers to it.

That is why the lowering sequence in `docs/todos/native-lowering-roadmap.md`
(`monomorphize → unfold_special → unfold_overflow → float_to_fixed →
rescale_fixed → simplify`) is pinned by a test rather than exposed as an entry
point, and it is recorded as the standing blocker in
`docs/todos/rounding-operator-basis.md`. This item removes it: a reference that
survives the rewrites around it, so a schedule can aim a sequence at one site.

## The design

**A cursor is a path, not a node.** Every transform is a
`DefaultTransformVisitor` (`fpy2/ast/visitor.py`), which rebuilds every node it
visits — node identity dies at the first rewrite. A cursor is the statement's path
from the `FuncDef`: a tuple of `(field, index)` steps, `field` naming the
sub-block of the enclosing statement (`body`, `ift`, `iff`). Statement-level only.

**A cursor is owned by one program version.** It holds the `FuncDef` it resolved
against; use on another program is a `TransformReferenceError`. Holding the
reference also keeps that tree alive, so no `id()` is recycled underneath it.

**Forwarding is an edit log, not a diff.** A rewriting transform reports, per
site: the old path of the replaced statement, how many statements went out and
in, and where the *successor* sites landed. Forwarding is three rules:

- later sibling of an edit, at any level of the path → shift index by
  `inserted - removed`;
- *is* the edited statement → becomes the edit's successors;
- strictly inside the edited statement → invalidated. That subtree was rebuilt;
  only the transform could say what became of it, and it does not claim to.

**Successors are a set, not a point.** `float_to_fixed` emits a subnormal branch
and a normal branch, each with its own rounding; `unfold_special` emits one ladder
per round in the block. So `forward` returns one cursor and raises when the
successor is not unique, and `forward_all` returns the list. Real consequence for
item 7: after `float_to_fixed` a schedule is aimed at two points, not one.

**The successor scan is shared.** All five rounding transforms recognize sites
with the same predicate — `rounding_block` in `fpy2/transform/utils/`. A site's
successors are the rounding blocks in its replacement, found by one scan in
`BlockRewriter`. No subclass declares anything.

**The chain lives on `Function`.** `Function` (`fpy2/function.py`) gains a parent
link plus the log that produced it, so `f3.forward(cursor_from_f0)` walks the
chain. A pass that reports no edits is *opaque*: forwarding across it raises,
naming the pass, rather than guessing.

**No path tracking inside the visitors.** A cursor resolves against the very tree
the visitor is about to walk, so a transform matches with
`block is target_block and idx == target_idx` — identity is valid *within* one
traversal. Paths are needed only at the boundaries: one helper walks a `FuncDef`
and returns `{id(block): path}`, used to write the log and to resolve a cursor.

## Phases

Each phase lands compiling, linted (`make lint`), and tested on its own. **Stop
after each phase for review. Never commit — you commit.** Only the tests relevant
to a phase run during it; the full suite runs once, at the end.

### Phase 1 — the cursor

New `fpy2/transform/utils/cursor.py`: `Cursor` (frozen; `func: FuncDef`,
`path: tuple[tuple[str, int], ...]`), `resolve() -> Stmt`,
`parent() -> tuple[StmtBlock, int]`, `__str__` naming the statement's source
`Location` where it has one, and `block_paths(func) -> dict[int, path]`.

At the transform layer for the same reason the errors are (item 1): named for
what it is, defined where it is raised, re-exported from `fpy2/strategies`. An
unresolvable path raises `TransformReferenceError` (`fpy2/transform/utils/error.py`).

Tests (`tests/unit/transform/`): hand-built cursors over nested `if`/`for`/`with`
programs — resolution, round-trip against `block_paths`, out-of-range and
wrong-field failures.

### Phase 2 — edits and forwarding

Pure; no transform touched. In `fpy2/transform/utils/cursor.py`: `Edit` (old block path,
index, `removed`, `inserted`, `successors: tuple[tuple[int, subpath], ...]` —
offset among inserted statements plus a path within one), `EditLog`, and
`forward(path, log) -> list[path]` implementing the three rules.

Two invariants asserted on construction: edits are **disjoint** (a transform that
rewrites nested sites inside one it already rewrote records only the outermost —
`for_unroll` does this — since everything inside is invalidated anyway), and paths
are **old-tree** paths, so a sequence composes by accumulating sibling shifts
level by level.

Tests: synthetic logs only — sibling shift at depth, ancestor shift, invalidation
inside, multi-successor, empty log. The subtle piece; pin it before anything
depends on it.

### Phase 3 — the rounding transforms report, `Function` carries

Largest phase; phases 1–2 are inert until it lands.

`BlockRewriter._visit_block` (`fpy2/transform/utils/`) already knows the block
and index of every site it replaces: record an `Edit` per rewrite, scanning the
emitted statements with `rounding_block` for successors, and expose the log on the
instance. The five transforms — `unfold_special`, `unfold_neg_zero`,
`unfold_overflow`, `float_to_fixed`, `rescale_fixed` — gain `apply_with_edits`,
mirroring the existing `apply_with_status` convention (`fpy2/transform/const_fold.py`
et al.); `apply` keeps its signature and its callers.

`Function` gains `_parent` / `_edits`, `with_ast(ast, *, edits=None)`,
`forward(cursor)`, `forward_all(cursor)`, and `site` — the cursor of the single
site just rewritten, where `where` was explicit. Default is opaque, so every pass
not yet updated stays honest.

Tests (`tests/unit/strategies/`): a two-step chain (`unfold_special` then
`unfold_overflow`) aimed at the site the first rewrote, asserting the second lands
there and not at index 0; a cursor to an untouched later sibling surviving a
rewrite that grew its block; `float_to_fixed` yielding two successors — `forward`
raises, `forward_all` returns both.

### Phase 4 — `where` accepts a cursor

`check_where` (`fpy2/transform/utils/`) accepts `int | Cursor`; `BlockRewriter`
matches by identity against the resolved `(block, index)`. A cursor owned by
another program, one whose path no longer resolves, and one resolving to a
statement that is not a candidate are all `TransformReferenceError` — item 1's
"does not point at anything" covers all three. The five strategy wrappers take
`where: int | Cursor` and say so in `Parameters` and `Raises`.

Split from phase 3 deliberately: produce, then consume.

### Phase 5 — the loop and call transforms

`split_loop`, `for_unroll`, `while_unroll`, `func_inline` count sites by hand
(`self.index`) and check them inline rather than through `check_where` /
`check_site`. Unify them onto the shared helpers, then give them both
capabilities: an edit log and a cursor `where`.

Successors are transform-specific and small: the chunked loop for `split`, the
main loop for `unroll_for`, the peeled body for `unroll_while`. `inline` splices a
body where a call statement stood and has no successor of the same kind — its edit
records none, so a cursor at an inlined call invalidates. That is the answer, not
a gap.

### Phase 6 — listing sites

`sites(strategy, func) -> list[Cursor]` in `fpy2/strategies`, backed by a `sites`
classmethod per transform (free for the five via `BlockRewriter`). Without it a
cursor can only be born from an `int`, and "pin several points, then rewrite" —
item 3's stated capability — stays unreachable.

This is the half of item 2 that item 3 needs. The presentation half — rendered
listings with source locations, before/after step diffs, the Exo Appendix-A
operator table — stays deferred.

### Phase 7 — the passes that are not site-rewrites

Classify every remaining strategy and make each say which it is:

- **structure-preserving** — `monomorphize`, `elim_round`, `fuse` (verify each
  individually): statement tree unchanged, empty log, identity forwarding.
- **prepending** — `close`, `lift_context`: leading assignments; an edit at index
  0 with `removed=0`, so everything below shifts.
- **opaque** — `simplify`: dead-code elimination deletes statements it cannot
  attribute. Forwarding across it raises, naming the pass. It runs last in every
  schedule we have, so this costs nothing today.

Cheap, but each claim needs a test that a cursor survives or fails as declared: a
wrong "structure-preserving" claim is a silent mis-aim, the exact failure mode
item 1 exists to prevent.

### Phase 8 — retire the blocker

Add a test that lowers a **two-rounding** program at one site by cursor and leaves
the other untouched — the thing `where=None` cannot express — near
`tests/unit/backend/cpp/test_lowered_roundtrip.py` rather than re-pinning
bit-exactness a second time.

Then update the three places recording the blocker as standing:
`docs/todos/rounding-operator-basis.md` (*Composition has no way to carry a
location*), `docs/todos/native-lowering-roadmap.md` (*Not exposed as one entry
point, deliberately* — the stated reason is gone; the recipe itself is item 7 and
stays open), and item 3 of `docs/todos/scheduling-language.md`, including its
`value_class.py` claim (below).

## Out of scope

**Expression-level cursors, and with them the `fpy2/analysis/value_class.py`
carry-over.** Item 3 claims forwarding fixes results keyed by expression identity.
Statement-level forwarding does not: `by_expr` is keyed on `Expr` nodes and every
visitor rebuilds every expression, so carrying it needs a forwarding map through
`DefaultTransformVisitor` itself, not just at the sites a transform knows it
changed. It is not what item 7 needs, and item 6 can re-run the analysis on the
rewritten program — which is what the transforms already do. Its own roadmap item;
item 3's claim gets amended in phase 8.

**Trimming the forwarding chain.** Every `Function` in a schedule keeps its parent
alive. Schedules are short and ASTs small — a note, not a problem.

## Verification

Per phase: `make lint` (mypy + ruff) and the phase's own tests via
`python3 -m pytest tests/unit/<area> -n 8`.

Before the final review, the full set:

```
python3 -m pytest tests/unit -n 8
python3 -m tests.infra
python3 -m tests.infra.fpcore
python3 -m tests.infra.backend.cpp --mode run
make lint
```

The load-bearing end-to-end check is
`tests/unit/backend/cpp/test_lowered_roundtrip.py`: the composed sequence must
stay bit-exact against the interpreter for both `FP32` and `FP64` sources across
all fourteen targets. Phases 3–5 change how the five operators are *plumbed*, so a
break there means the edit log altered a rewrite rather than only describing it.
