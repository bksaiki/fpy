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
from the `FuncDef`: block field, statement index, block field, ... where a field
names a sub-block of the enclosing statement (`body`, `ift`, `iff`). Odd length
names a block, even a statement, so a block path is `path[:-1]`. Statement-level
only. (Exo pairs them instead — `[("body", 0), ("orelse", 2)]` — which extends to
expressions; if expression cursors ever land, that is the shape to move to.)

**A cursor is owned by one program version.** It holds the `FuncDef` it resolved
against; use on another program is a `TransformReferenceError`. Holding the
reference also keeps that tree alive, so no `id()` is recycled underneath it.

**Forwarding is an edit log, not a diff.** A rewriting transform reports, per
site: the old path of the replaced statement, how many statements it consumed,
and how many it emitted. Forwarding is three rules:

- later sibling of an edit, at any level of the path → shift index by
  `inserted - removed` (Exo's `i + n_diff * (i >= del_range.stop)`);
- *is* the edited statement → becomes the region that replaced it;
- strictly inside the edited statement → invalidated. That subtree was rebuilt;
  only the transform could say what became of it, and it does not claim to.

**A statement forwards to a region, not to a point.** Exo's `Block` cursor — a
contiguous range of statements in one block — is the honest image of a rewritten
statement: `unfold_special` replaces a block of *n* rounds with *n* ladders, and
`float_to_fixed` puts a rounding in each of two branches. So `forward` returns a
`Cursor` where the image is a single statement and a `Block` otherwise, and the
forwarding stays *purely structural* — the edit records four numbers and no
transform has to name a semantic successor.

**`where` accepts a region, and a region means *at or beneath*.** A cursor or
region names a program point, and a rewrite takes every candidate at or under it
— `where=None` scoped to a subtree rather than a new idea. The at-or-beneath part
is load-bearing, not a generalization: the statement a rewrite leaves behind is a
*wrapper* (`unfold_special` emits `with REAL: if isnan(x): … else: <the
rounding>`), so the forwarded site is one level above the next operator's site.
Selecting only the named statement makes the second step of every schedule a bad
reference; selecting beneath it makes one pinned cursor aim the whole sequence.
It also handles `float_to_fixed`'s two roundings in one call, and keeps forwarding
free of per-transform successor logic.

**The chain lives on `Function`.** `Function` (`fpy2/function.py`) gains a parent
link plus the log that produced it, so `f3.forward(cursor_from_f0)` walks the
chain — Exo's `Procedure.forward`, which collects each `_forward` back along
`_provenance_eq_Procedure` and applies them in order. A pass that reports no edits
is *opaque*: forwarding across it raises, naming the pass, rather than guessing —
Exo's default `_forward` raises too. One difference: where Exo's loop simply ends
when the cursor's procedure is not in the chain, leaving the cursor silently
misaligned, a cursor from an unrelated program is a bad reference here.

**A stale cursor is forwarded on arrival.** Exo's operators never make the user
call `forward`: `CursorArgumentProcessor.__call__` runs `p.forward(cur)` on every
cursor argument before validating it, so a cursor from any ancestor program is
rebased on entry and a schedule pins a point once. Same here, with one placement
difference — the rebase belongs in the strategy wrappers, since a transform is
handed a bare `FuncDef` and has no chain to walk, while `Function` does.

**No path tracking inside the visitors.** One helper walks a `FuncDef` and
returns `{id(block): path}`; a transform looks up the block it is visiting and
compares paths with what the cursor names — a prefix test, which is also what
makes *at or beneath* a one-liner. Nothing threads a path through the visit.

## Phases

Each phase lands compiling, linted (`make lint`), and tested on its own. **Stop
after each phase for review. Never commit — you commit.** Only the tests relevant
to a phase run during it; the full suite runs once, at the end.

### Phase 1 — the cursor

New `fpy2/transform/utils/cursor.py`: `Cursor` (frozen; `func: FuncDef`,
`path: tuple[str | int, ...]`), `resolve() -> Stmt`,
`parent() -> tuple[StmtBlock, int]`, `__str__` naming the statement's source
`Location` where it has one, and `block_paths(func) -> dict[int, path]`.

At the transform layer for the same reason the errors are (item 1): named for
what it is, defined where it is raised, re-exported from `fpy2/strategies`. An
unresolvable path raises `TransformReferenceError` (`fpy2/transform/utils/error.py`).

Tests (`tests/unit/transform/`): hand-built cursors over nested `if`/`for`/`with`
programs — resolution, round-trip against `block_paths`, out-of-range and
wrong-field failures.

### Phase 2 — edits and forwarding

Pure; no transform touched. In `fpy2/transform/utils/cursor.py`: `Block` (frozen;
`func`, `block_path`, `range`, with `__len__` / `__iter__` / `__getitem__` /
`one()`), `Edit` (old block path, index, `removed`, `inserted` — four values, no
successor logic), `EditLog`, and `forward` implementing the three rules.

Two invariants asserted on construction: edits are **disjoint** (a transform that
rewrites nested sites inside one it already rewrote records only the outermost —
`for_unroll` does this — since everything inside is invalidated anyway), and paths
are **old-tree** paths, so a sequence composes by accumulating sibling shifts
level by level.

Tests: synthetic logs only — sibling shift at depth, ancestor shift, invalidation
inside, a one-statement image forwarding to a `Cursor` and a many-statement image
to a `Block`, empty log. The subtle piece; pin it before anything depends on it.

### Phase 3 — the rounding transforms report, `Function` carries

Largest phase; phases 1–2 are inert until it lands.

`BlockRewriter._visit_block` (`fpy2/transform/utils/`) already knows the block
and index of every site it replaces, and `_rewrite` returns the statements that
replace it: record an `Edit` per rewrite and expose the log on the instance. No
subclass changes — the edit is structural. The five transforms —
`unfold_special`, `unfold_neg_zero`, `unfold_overflow`, `float_to_fixed`,
`rescale_fixed` — gain `apply_with_edits`,
mirroring the existing `apply_with_status` convention (`fpy2/transform/const_fold.py`
et al.); `apply` keeps its signature and its callers.

`fpy2/transform` already imports `fpy2.function`, so `function.py` must not import
the cursor module at runtime: annotate under `if TYPE_CHECKING` and call
`log.forward(cursor)` without naming the class, the pattern `Function` already uses
for `Interpreter`.

`Function` gains `parent` / `edits`, `with_edits(log)` beside the existing
`with_ast` (which stays, and is now the *opaque* step), and
`forward(cursor) -> Cursor | Block`. Default is opaque, so every pass not yet
updated stays honest. Forwarding a `Block` comes along here rather than in phase
2: the chain needs it as soon as any pass yields a region.

Tests (`tests/unit/strategies/`): a two-step chain (`unfold_special` then
`unfold_overflow`) aimed at the site the first rewrote, asserting the second lands
there and not at index 0; a cursor to an untouched later sibling surviving a
rewrite that grew its block; a single-statement image forwarding to a `Cursor`;
`float_to_fixed`'s two roundings both reached through one forwarded region.

### Phase 4 — `where` accepts a cursor and a region

`check_where` (`fpy2/transform/utils/`) accepts `int | Cursor | Block`, and
`BlockRewriter._selects` takes every candidate at or beneath what the cursor or
region names. The strategy wrappers call `Function.rebase` to forward a cursor
from an ancestor program before handing it down, so a schedule aims the whole
sequence with one cursor variable — declines inside it are
skipped, as under `where=None`. A cursor owned by another program, one whose path
no longer resolves, and one resolving to a statement that is not a candidate are
all `TransformReferenceError` — item 1's "does not point at anything" covers all
three, and a region whose every candidate declines is a `TransformDeclined`
carrying the reasons — an explicit aim never silently no-ops. The five strategy
wrappers spell the type out rather than sharing an alias, since phase 5's
transforms need not accept the same forms.

Split from phase 3 deliberately: produce, then consume.

### Phase 5 — the loop and call transforms

`split_loop`, `for_unroll`, `while_unroll`, `func_inline` count sites by hand
(`self.index`) and check them inline rather than through `check_where` /
`check_site`. The shared half of `BlockRewriter` — the target, the edit log, the
site check — moves up into a `SiteRewriter` base the four inherit; `BlockRewriter`
keeps only `_candidate` / `_verify` / `_rewrite`.

These four splice their replacement into the *enclosing* block through the visit
context rather than returning it, so `_visit_block` reads the edit off the
accumulator's growth (plus a flag, for a rewrite that emits exactly one
statement). `_record` drops any edit recorded beneath a statement later replaced
wholesale, which is what keeps `for_unroll`'s nested unrolling disjoint.

`inline`'s sites are `Call` *expressions*, so a statement-level cursor is coarser
than its index: a statement holding two candidate calls names both. Documented,
not worked around — expression cursors are out of scope.

### Phase 6 — listing sites

`sites(strategy, func, within=None) -> list[Cursor]` in `fpy2/strategies`, backed
by a `sites` classmethod per transform (free for the five via `BlockRewriter`).
`within` takes a `Block`, so a forwarded region can be asked what it holds.
Without this a cursor can only be born from an `int`, and "pin several points,
then rewrite" — item 3's stated capability — stays unreachable.

Same shape as Exo's `proc.find(pattern)`, which is where item 4 lands: pattern
matching that returns cursors is this function with a different predicate.

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

## Not taken from Exo

- **Gap cursors** (`before()` / `after()`, an insertion point that survives edits).
  They exist to serve an `insert` primitive, and FPy has none: no transform lets a
  schedule name where to put a statement. Revisit with the first one.
- **`InvalidCursor` as a falsy value.** Exo has both — a sentinel returned by
  `next()` past the end, and an `InvalidCursorError` raised by forwarding. A
  second failure vocabulary is what item 1 exists to remove; a bad reference
  raises, always.
- **Navigation** (`next`, `prev`, `parent`, `expand`) and **expression cursors**.
  Nothing in items 3–7 asks a schedule to walk the tree by hand.

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
