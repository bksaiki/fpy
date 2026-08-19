# Plan: locations that survive rewrites

Item 3 of `docs/todos/scheduling-language.md`. Item 2 (discoverability) is
de-prioritized; the one slice of it this item needs — listing candidate sites as
cursors — is phase 9 here.

**Done — all eleven phases.** Kept as the record of the design decisions and of
what was deliberately not taken from Exo, which item 3 of the roadmap now points
at. What each phase settled is recorded in place, including the three guesses this
plan started with that turned out wrong (`fuse` and `elim_round` are not
structure-preserving; `monomorphize` preserves expressions too).

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
visits — node identity dies at the first rewrite. So a cursor names a *path* from
the `FuncDef`.

**A path is an ADT, linked to its parent.** Phases 1–5 used a flat
`tuple[str | int, ...]` with a parity convention (odd length a block, even a
statement); phase 6 replaces it with the grammar the paths actually have:

```
BlockPath ::= FuncBody                              -- base: the function's body
            | StmtPath . field                      -- 'body' | 'ift' | 'iff'
StmtPath  ::= BlockPath [ index ]
ExprPath  ::= (StmtPath | ExprPath) . field [ index? ]
```

as parent-linked frozen dataclasses. Every ill-formed path is then
unrepresentable: a sub-block hangs off a statement, a statement sits in a block, an
expression hangs off a statement or another expression. `FuncBody` is the only
constructor without a parent, so every path is *absolute*: cursor equality is total,
and `beneath` can walk to the root without being told where to stop. A relative
path would need a second base — a hole — and every consumer would then have to ask
"relative to what?"; if item 4's pattern matching wants one, it is its own type.
The block field is a `Literal['body', 'ift', 'iff']` (precedent: `_Missing` in
`fpy2/analysis/alias.py`), so a typo'd field is a type error too; expression fields
stay `str`, there being ~20 of them.

**Parent-linked, not root-linked, and that is what buys the type safety.** A
path's type should be the type of its *leaf*, because the leaf is what the cursor
is named by — so `StmtCursor` takes a `StmtPath`, `BlockCursor` a `BlockPath`,
`ExprCursor` an `ExprPath`, and mypy checks it. Linking from the root instead would
type every path as a `FuncBody` and leave the leaf kind — the only part that
matters — invisible.

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

**A statement forwards to a region, not to a point.** Exo's block cursor — a
contiguous range of statements in one block — is the honest image of a rewritten
statement: `unfold_special` replaces a block of *n* rounds with *n* ladders, and
`float_to_fixed` puts a rounding in each of two branches. So `forward` returns a
`StmtCursor` where the image is a single statement and a `BlockCursor` otherwise,
and the forwarding stays *purely structural* — the edit records four numbers and no
transform has to name a semantic successor. (Phases 1–5 shipped these two as
`Cursor` and `Block`; phase 7 gives them their final names.)

**Three kinds, one union.** Exo's taxonomy is `ExprCursor` / `StmtCursor` /
`BlockCursor`, and FPy wants the same three — with `Cursor` as a *union alias*,
not a base class:

```python
Cursor: TypeAlias = ExprCursor | StmtCursor | BlockCursor
```

`isinstance` accepts a union alias (3.10+), including inside a tuple, so the
runtime checks in `check_where` and `Function.rebase` read the same as before, and
`match` over the three arms is exhaustively checkable. Inheritance would buy
little: `ExprCursor` carries a different payload (a statement path *plus* a path
into its expressions), and the only field all three share is `func`. The union
also collapses every signature from `int | Cursor | Block | None` to
`int | Cursor | None`, while a transform that accepts only some kinds names them
(`int | StmtCursor | None`) — the per-strategy contract, spelled out.

**Forwarding is kind-polymorphic.** The image's kind is a property of what the
rewrite did, not of the cursor handed in:

| in | out | when |
|---|---|---|
| `StmtCursor` | `StmtCursor` | shifted, or replaced one-for-one |
| `StmtCursor` | `BlockCursor` | replaced by several statements |
| `BlockCursor` | `BlockCursor` | members forwarded and re-joined |
| `BlockCursor` | `StmtCursor` | the run collapses to a single statement |
| `ExprCursor` | `ExprCursor` | its statement was left alone |
| `ExprCursor` | *raises* | its statement was rewritten |

So `forward` is typed `Cursor -> Cursor` and callers narrow — `match`, or
`BlockCursor.one()`. `StmtCursor -> ExprCursor` never happens: no pass in this
item collapses a statement into an expression, and if one appears its edit has to
say so rather than the forwarding guessing.

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
than its index: a statement holding two candidate calls names both. Documented in
phase 5, retired in phase 8.

### Phase 6 — the path ADT

New `fpy2/transform/utils/path.py` — `cursor.py` is already 386 lines and paths are
a self-contained language. It holds `FuncBody` / `SubBlock` / `StmtPath` (the
grammar above; `ExprPath` lands with `ExprCursor` in phase 8, and the grammar
already reserves its shape), plus the operations, each now structural recursion
instead of arithmetic on a heterogeneous tuple:

- `resolve_block(func, BlockPath)`, `resolve_stmt(func, StmtPath)`
- `format_path` — still `body[1].ift[0]`
- `beneath(path, block, span)` — the upward walk that replaces the slice-prefix test
- a fluent builder, so a path is written by descending and each step's type says
  what may follow it: `FuncBody().stmt(1).block('ift').stmt(0)`

What gets simpler: `_selects` and `_overlaps` become `beneath` calls, and
`block_paths` just nests constructors. What gets slightly longer: `_forward` is
inherently top-down, since shifts accumulate from the root, so it reverses the
ancestry once and folds, rebuilding nodes as it goes.

`Edit.block_path`, `Cursor.path`, and `Block.block_path` take the typed paths;
`Cursor.block_path` / `.index` survive as accessors over `path.parent` /
`path.index`. Forwarding turns into two mutually recursive functions over the
grammar (`_forward_block` / `_forward_stmt`) rather than a top-down walk with
index arithmetic, and no reversal is needed after all.

Behaviour is unchanged, so every test outside the four cursor modules passes
untouched. Inside them the path constructions change, and the ill-formed-path
cases *shrink*: a block where a statement belongs, a field where an index belongs,
and a path not rooted at the function body are no longer expressible, so only the
resolvable-but-wrong cases remain — which is the ADT paying for itself.

### Phase 7 — the cursor kinds

Mechanical; no behaviour changes. The class named `Cursor` becomes `StmtCursor`,
`Block` becomes `BlockCursor`, and `Cursor` becomes the union alias (two arms here,
three after phase 8).

The rename is not only symmetry: **`Block` collides with what "block" already
means here** — the AST has `StmtBlock`, the machinery has `BlockRewriter`, and
every docstring says "candidate rounding block" for a `with` statement, so
`where=Block(...)` reads like the one thing it is not.

Then the ten signatures of phases 3–5 collapse to `int | Cursor | None`, and their
`where` docs name the kinds. About 210 references across 33 files, nearly all
imports, annotations, and test constructions.

Landing it before phase 8 and phase 10 matters: phase 8 adds a third arm, and phase
10's classification is *per kind* — a distinction that has no name until this
lands.

Files: the definitions and the union in `fpy2/transform/utils/cursor.py`; the
`isinstance` checks in `fpy2/transform/utils/__init__.py` (`check_where`,
`_target_of`) and `fpy2/function.py` (`rebase`, `forward`); the re-export lists in
`fpy2/transform/__init__.py` and `fpy2/strategies/__init__.py`; annotations and
docstrings in the nine transform modules and their nine wrappers (e.g.
`fpy2/transform/unfold_special.py`, `fpy2/strategies/special_unfold.py`); and the
four cursor test modules under `tests/unit/`. Every other test must pass *untouched*
— that is the check that the rename changed no behaviour.

### Phase 8 — expression cursors

`ExprPath` joins the ADT of phase 6 — `parent: StmtPath | ExprPath`, a `field`, and
an optional `index`, which positions it where a field holds several. Parent-linking
gives `ExprCursor.stmt() -> StmtCursor` for free: walk up to the `StmtPath`.

Beside `sub_blocks`, a `sub_exprs(node)` giving the child expressions of a
statement or expression with the field naming each — the only place the AST's
field names appear, since `resolve_expr` reads a path by scanning it. **Every
operator stores its operands in `args`** whatever its arity (`arg` / `first` /
`second` are properties over that), so one arm covers all of them rather than one
per arity, and the field set stays small enough to declare: `ExprField` is a
`Literal` of the eighteen names, mirroring `BlockField`. Typing `sub_exprs`'
return with it also checks the two against each other — an arm returning a field
not in the alias fails to type.

**Forwarding is nearly free, and needs one honest claim.** A statement no edit
touched is *rebuilt with the same shape*, so an expression path still resolves —
no forwarding map, no visitor changes. What it needs is something no pass says
today: that expressions *outside* the recorded edits were left alone. `EditLog`
gains `exprs_preserved: bool`, which the nine transforms of phases 3–5 set and
everything else leaves `False`, so `forward(ExprCursor)` raises rather than
silently mis-aiming. An expression inside a *replaced* statement invalidates,
exactly as a statement inside one does.

One pass needs a narrower claim than that flag can make. `func_inline` splices the
callee's body *ahead of* the statement that held the call, so the statement
survives — an insertion, not a replacement — but its call became a variable. Hence
`EditLog.exprs_rewritten`: the statements a pass changed the expressions of without
replacing them. An expression cursor in one of those does not forward, while the
statements beneath it still do.

**Aiming.** `inline`'s sites *are* expressions, so an `ExprCursor` names exactly
one call: `_visit_call` compares node identity against the resolved target — the
trick `parent()` already uses, so nothing tracks expression paths during a visit.
That retires phase 5's documented coarseness, while a `StmtCursor` / `BlockCursor`
keeps the at-or-beneath reading. A statement-sited rewrite handed an `ExprCursor`
fails in `_target_of` with a message saying why: no statement sits beneath an
expression.

Tests: expression paths round-tripping over the operator groups; `inline` aimed at
one of two calls in a single statement; an expression cursor surviving a rounding
rewrite elsewhere in the program; invalidation inside a replaced statement; and
the refusal across a pass that does not preserve expressions.

### Phase 9 — listing sites

`sites(strategy, func, within=None, **kwargs) -> list[Cursor]` in
`fpy2/strategies/utils/sites.py`, dispatching through a table of the aimable strategies
to a `sites` staticmethod per transform. It returns the kind the transform is sited
on — `StmtCursor` for the rounding and loop rewrites, `ExprCursor` for `inline` —
and `within` takes any cursor, so a forwarded region can be asked what it holds.
Without this a cursor can only be born from an `int`, and "pin several points,
then rewrite" — item 3's stated capability — stays unreachable.

Two shared scanners do the work, over one `walk_stmts` that yields each statement
before the blocks it holds, which is the order a transform counts candidates in:
`stmt_sites(func, match, within)` and `expr_sites(func, match, within)`. The
predicate is syntactic and shared with the rewrite — `is_rounding_block(stmt,
casts=)` for the five, `isinstance(s, ForStmt)` for the loops — so a listing cannot
drift from what `where` accepts. A strategy absent from the table raises rather
than returning `[]`: it takes no `where` at all.

Same shape as Exo's `proc.find(pattern)`, which is where item 4 lands: pattern
matching that returns cursors is this function with a different predicate.

This is the half of item 2 that item 3 needs. The presentation half — rendered
listings with source locations, before/after step diffs, the Exo Appendix-A
operator table — stays deferred.

### Phase 10 — the passes that are not site-rewrites

Classify every remaining strategy, **per cursor kind**, and make each say which it
is. Reading them settled three of the guesses this plan started with — `elim_round`
and `fuse` are *not* structure-preserving, and `monomorphize` preserves more than
expected:

- **structure-preserving, both kinds** — `monomorphize`: it overrides only
  `_visit_argument` and `_visit_function`, so the body passes through
  `DefaultTransformVisitor` untouched. Empty log, `exprs_preserved=True`, and a
  cursor chosen before the first step of the lowering recipe survives it.
- **prepending** — `close` (a binding per captured value) and `lift_context` (one
  per lifted context): an edit at index 0 with `removed=0`, so everything below
  shifts. `close` reuses the body's statements verbatim, so expressions survive
  too; `lift_context` replaces the context expressions in place, so they do not —
  the per-kind split, in one pass each way.
- **opaque** — `simplify` (dead-code elimination deletes what it cannot
  attribute), and three that turn out to be site-rewrites without an edit log:
  `fuse` replaces one statement with a seed, a loop and a read; `elim_round`
  hoists an operation into its own `with fp.REAL:` block ahead of the statement
  that held it; `elim_iter` emits a preamble per source and rewrites comprehension
  expressions in place. Each says so in its docstring — aim before it, or re-list
  the sites after. Any of the three could report edits with phase 5's
  `_visit_block` growth pattern, if a schedule ever wants to pin through it.

Per kind is the sharp part, and the reason `exprs_preserved` exists: a single
verdict per pass would make one of the two answers a lie.

Cheap, but each claim needs a test that a cursor survives or fails as declared: a
wrong "structure-preserving" claim is a silent mis-aim, the exact failure mode
item 1 exists to prevent.

### Phase 11 — retire the blocker

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
- **Navigation** (`next`, `prev`, `parent`, `expand`), and `ExprListCursor` /
  `ArgCursor`. Nothing in items 3–7 asks a schedule to walk the tree by hand.

## Out of scope

**The `fpy2/analysis/value_class.py` carry-over.** Item 3 claims forwarding fixes
results keyed by expression identity. Expression *cursors* (phase 8) are not that:
a cursor is a path, and a path resolves in the rebuilt tree for free. Carrying an
analysis *result* needs the old-node → new-node correspondence threaded through
`DefaultTransformVisitor` itself — a much larger mechanism, in every visitor rather
than at the sites a transform knows it changed. It is not what item 7 needs, and
item 6 can re-run the analysis on the rewritten program, which is what the
transforms already do. Its own roadmap item; item 3's claim gets amended in phase
11.

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
