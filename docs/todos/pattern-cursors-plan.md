# Plan: patterns produce references

Item 4 of [scheduling-language.md](scheduling-language.md), on top of item 3
([cursor-forwarding-plan.md](cursor-forwarding-plan.md)).

## Context

`fpy2/rewrite` already does structural matching: `@fp.pattern` builds an
`ExprPattern` (one expression) or a `StmtPattern` (a run of *k* statements),
`Matcher` finds occurrences, and `Rewrite(lhs, rhs).apply(func)` rewrites one.
It is an island. Three seams show it:

- **It names locations by an occurrence index.** `Rewrite.apply(func,
  occurence=2)` counts matches in traversal order — the same index-into-a-scan
  design item 3 replaced for the strategies, with the same defect under
  composition.
- **It has its own failure vocabulary.** A rewrite that matches nothing raises
  `RewriteError`, not the `TransformDeclined` / `TransformReferenceError` pair
  item 1 settled, so `except TransformError` does not catch it.
- **It is invisible to cursors.** `Rewrite.apply` returns `func.with_ast(ast)`,
  the opaque step, so no cursor survives a user rewrite and a schedule cannot
  mix user rewrites with built-in strategies.

Unifying the layers closes all three, and gives the scheduling language a second
way to name a location: not "the *n*th candidate" but "the place that looks like
*this*".

## The design

**A match is a cursor.** The three kinds already cover what the matcher
produces, exactly:

| pattern | matches | cursor |
|---|---|---|
| `ExprPattern` | one expression, anywhere | `ExprCursor` |
| `StmtPattern` of *k* > 1 | a run of *k* statements | `BlockCursor` |
| `StmtPattern` of 1 | one statement | `StmtCursor` |

**`find` resolves to one match; `find_all` returns them all.** Exo's split, and
for the reason item 1 gives: a list-returning `find` would be indexed `[0]` at the
call site, silently taking the first of three matches — the quiet wrong-site this
whole line of work exists to prevent. A pattern is meant to be *specific*, so `find` raises on zero matches and
on several — `TransformReferenceError` either way, with the count and a pointer to
`find_all`. Both take `within`, reusing `_restrict`.

That is deliberately *not* symmetric with `sites(strategy, func, within=)`, which
returns a list and has no one-match partner: candidates are inherently plural,
while a pattern claims to identify something.

**The matcher walks with paths, not identities.** `ExprMatch` holds an `Expr` and
`StmtMatch` a `(block, idx)` pair, both by object identity, which dies at the
first rewrite. The two engines already walk the tree; they take the path with
them, reusing `walk_stmts` and the expression descent `expr_sites` uses rather
than growing a second traversal.

**A user rewrite becomes a strategy on the same terms — of type, not of trust.**
Same signature (`Function -> Function`), same `where` vocabulary, same failure
contract, same edit log. *Not* the same guarantee: a built-in rounding rewrite is
probe-verified against the format it rewrites, and `a * b + c → fma(a, b, c)` is
checked by nobody. Unification must not read as verification — see *Out of scope*.

**A statement rewrite is the first edit that removes more than one statement.**
`Edit(block, idx, k, m)` for a *k*-statement pattern replaced by *m* statements.
Everything downstream already handles it — `_forward_region` merging members that
one edit consumed together was fixed in the item-3 review — but this is what makes
that path live rather than latent, so it gets a test here.

**An expression rewrite records no statement edit at all.** It changes
expressions inside statements it does not replace, which is exactly what
`EditLog.exprs_rewritten` is for: statement cursors sail through, and an
expression cursor in a rewritten statement fails loudly.

**`where` does not take a pattern.** Exo's operators *do* accept a pattern where a
cursor is expected, so it is worth being precise about why FPy's should not. A
pattern there is well-defined only because Exo disambiguates *inside the pattern
string* — `"for i in _: _ #2"` is the third match — and requires the result to
resolve to exactly one cursor (`proc.find(pattern)` returns one; `find_all` is the
explicit opt-in for a list). FPy patterns are decorated functions, with no
occurrence syntax to carry an `#n` and no obvious place to add one. So
`where=pattern` would be either ambiguous or need a companion index —
`where=(p, 2)`, or an `occurrence=` beside `where` — which is the index item 3
removed, smuggled back in attached to a pattern.

`find_all(p, f)[2]` puts that index where it belongs: on a list, evaluated against
a named program, that a schedule can print and inspect before aiming anything. And
for `Rewrite` the question does not arise at all: the pattern **is** the rewrite,
and `where` picks among its own matches.

So `where` stays `int | Cursor | None` and `SiteRewriter` is untouched — no
generalizing `_target` to a list of regions in the path every strategy runs
through. "Apply at every match" stays a combinator (item 5), matching Exo, where
many locations means `find_all` plus a loop.

**Overlapping matches decline the whole application.** A statement pattern of *k*
statements is matched by a sliding window, so windows at *i* and *i+1* can both
match — and an edit log requires its edits to be disjoint, so they cannot both be
rewritten. Rewriting the first and skipping the second would quietly do less than
asked, which is the same defect as `find` silently taking the first of several
matches. So a `Rewrite` whose matches overlap raises :class:`TransformDeclined`
naming the pair, and the schedule narrows the pattern or aims one match with a
cursor.

## Phases

Each phase lands compiling, linted (`make lint`), and tested on its own. **Stop
after each phase for review. Never commit — you commit.** Only the tests relevant
to a phase run during it; the full suite runs once, at the end.

### Phase 1 — matches carry cursors

`_ExprMatcherEngine` and `_StmtMatcherEngine` (`fpy2/rewrite/matcher.py`) thread a
path through their walks and give `LocatedMatch` a `cursor`. `ExprMatch.expr` and
`StmtMatch.block` / `.idx` stay for now, so nothing downstream changes yet.

The sliding window is the thing to look at: `_StmtMatcherEngine` matches every
window of *k* consecutive statements, so two matches can *overlap*. That is fine
for a cursor and fatal for an edit log. This phase establishes whether the engine
can in fact produce overlapping matches today, and pins it in a test either way —
phase 4 declines them, and needs to know whether that path is reachable or
defensive.

Tests (`tests/unit/rewrite/test_matcher.py`): the cursor of every match resolves
to what the match found, for expression patterns, one-statement patterns, and
*k*-statement windows.

### Phase 2 — `find` and `find_all`

Beside `sites`, in `fpy2/strategies/utils/`:

- `find_all(pattern, func, within=None) -> list[Cursor]` — every match in
  traversal order (the order the occurrence index counted, so `find_all(p, f)[i]`
  is occurrence *i*), `[]` when none. `ExprCursor` for an expression pattern,
  `BlockCursor` / `StmtCursor` for a statement pattern.
- `find(pattern, func, within=None) -> Cursor` — the one match, raising
  `TransformReferenceError` on zero and on several.

Tests: both pattern kinds; a *k*-statement window; nested matches; `within`
narrowing to a region and to an expression; `find_all` returning `[]` where
`find` raises; and the ambiguity message naming the count, since that is the
error a user will actually hit.

### Phase 3 — one failure contract

`RewriteError` joins the hierarchy: a rewrite whose pattern matches nothing is a
`TransformReferenceError` ("names no match"), and one that matches but cannot be
applied is `TransformDeclined` with the reason. Keep the name as an alias if
anything depends on it, but `except TransformError` must catch both.

Small, and it is what lets a user rewrite sit inside a `try`/fallback beside a
built-in.

Tests: the existing `tests/unit/rewrite/test_rewrite.py` failure cases, retargeted
at the shared hierarchy.

### Phase 4 — `Rewrite` reports edits

`Rewrite.apply_with_edits(func) -> EditLog`, mirroring the nine transforms, and
`apply` delegating. A statement rewrite emits `Edit(block, idx, k, m)`; an
expression rewrite emits no edit and marks the statement in `exprs_rewritten`.
The wrapper uses `Function.with_edits`, so cursors cross a user rewrite.

Overlapping statement matches (phase 1) are declined here, whole: a `Rewrite`
whose matches overlap raises :class:`TransformDeclined` naming the pair, rather
than rewriting one and dropping the other. Detect it where the log is built —
`EditLog` already rejects overlapping edits, so the check is turning that
`ValueError` into the declined message a user can act on, before it becomes an
internal error.

Tests: a cursor surviving an expression rewrite; a statement cursor surviving a
statement rewrite elsewhere; a *k*-statement region forwarding to its replacement
(the `removed > 1` path); an expression cursor in a rewritten statement refusing
to forward; and overlapping matches declining with both locations named.

### Phase 5 — `Rewrite` takes a `where`

`Rewrite.apply(func, where=None)` replaces `occurence`, with the vocabulary every
other strategy uses: an index counts matches in traversal order, a cursor names
one (at or beneath, as everywhere else), `None` rewrites every match.

Replacing `occurence` rather than keeping it is a deliberate break: it is the
defect item 3 exists to remove, and this package is pre-1.0.

Small, now that `where` needs no new shape — this is the consume half of phase 4's
produce.

Tests: `Rewrite` aimed by index and by cursor, agreeing where both name the same
match; `find(pattern, f)[i]` aiming the same rewrite as `where=i`; a cursor naming
no match being a bad reference (unlike `find`, which returns `[]`).

### Phase 6 — surface and docs

`fpy2.rewrite` is the surface, not `fpy2.strategies`: matching and user rewrites
are their own layer, and re-exporting them into the scheduling language would blur
that. So `fpy2/rewrite/__init__.py` exposes `Rewrite`, `find`, `find_all` and the
pattern types, and what remains for this phase is deciding whether any of them
also belongs at `fpy2` top level, where `@fp.pattern` already lives.

Document them on `docs/source/strategies.rst` beside `sites` and the cursors --
the page is about aiming a rewrite, whichever package defines it. Mark item 4 done in
[scheduling-language.md](scheduling-language.md), and add the `NEXT` entry to
`HISTORY.md`.

### Phase 7 — one home for the location vocabulary (conditional)

**Trigger: what phase 3 settles.** `path.py`, `cursor.py` and `error.py` live in
`fpy2/transform/utils/`, and eighteen sites outside `transform` now reach into it —
`strategies` from eleven modules, `function.py` from three, `rewrite` from two.
That is a vocabulary shared by four packages, living inside one of them, which is
why each new consumer writes `from ..transform.utils.cursor import ...`.

Not under `rewrite`: `transform` has fifteen modules using it against `rewrite`'s
two, so that would invert the dependency. A top-level home (`fpy2/location/`) has
everything import *down* instead of sideways into a sibling's `utils`.

All three files move together or none do: `bad_path` and the cursors raise
`TransformReferenceError`, so paths cannot hoist without the error hierarchy
following. And that hierarchy is named `Transform*`, documented in
`docs/source/strategies.rst`, and deliberately named by item 1 — so the move
carries a public naming decision with it. Phase 3 is what answers it: if a *user
rewrite* raising `TransformReferenceError` reads right once rewrites are in the
hierarchy, the names stay; if it reads wrong, the neutral name is chosen with
evidence rather than guessed now.

Cheap preparation, whenever: the eleven `strategies/*.py` modules import `Cursor`
from `..transform.utils.cursor`; importing it from `..transform` — the package's
own re-export — leaves one name to redirect later instead of eleven deep paths.

## Out of scope

**Verifying a user rewrite.** The roadmap says `Rewrite.apply` should be
"probe-verified where a context is known", and this item does not do that. The
existing probe machinery (`fixed_probes`, `agrees`, `try_round` in
`fpy2/transform/utils/`) tests whether a *format's* rounding is reproduced; it has
nothing to say about whether `a * b + c` and `fma(a, b, c)` agree, which needs
interpreting both sides on sampled inputs — a different mechanism, and sampling
rather than proof (the roadmap rejects SMT for FPy on cost grounds). Worth its own
item, and worth naming: until it exists, a user rewrite has a built-in's *type*
and not a built-in's *guarantee*, and the docstrings should say so.

**Aiming another strategy at every match of a pattern.** `Rewrite` itself applies
to all of its own matches under `where=None`; what is out of scope is
`unfold_special(f, where=some_pattern)`. The honest spelling is
`for c in find_all(p, f): f = strategy(f, c)` — each step forwards the cursors that
are left and fails loudly where a rewrite invalidated one — and a combinator
wrapping that loop belongs to item 5.

**A traversal language.** Item 5's note stands: patterns give scoped matching, and
`topDown` / `bottomUp` earn nothing over `find_all` plus a loop.

## Verification

Per phase: `make lint` and the phase's own tests via
`python3 -m pytest tests/unit/<area> -n 8`.

Before the final review, the full set:

```
python3 -m pytest tests/unit -n 8
python3 -m tests.infra
python3 -m tests.infra.fpcore
python3 -m tests.infra.backend.cpp --mode run
make lint
make docs
```

`tests/unit/rewrite/` is the suite that must not regress: it pins the matcher and
applier this item re-plumbs. `tests/unit/backend/cpp/test_lowered_roundtrip.py`
stays the end-to-end check that the strategy layer still composes.
