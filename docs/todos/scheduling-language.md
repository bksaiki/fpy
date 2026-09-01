# Roadmap: growing the scheduling language

The point of this project is the FPy-specific transforms, not scheduling
infrastructure — so this roadmap records only the infrastructure worth
building, in the order it pays off. The source material is Exo 2 (Ikarashi et
al., ASPLOS '25, arXiv:2411.07211) and the systems around it: ELEVATE's
strategy combinators, the MLIR Transform dialect's handles and failure
taxonomy, and the Roly-poly user study on guided scheduling (arXiv:2107.12567).
Each item says what transfers and what it fixes here.

## 1. One failure contract

Failure used to be inconsistent: an out-of-range `where` raised `ValueError`
in `inline`, `split`, and `unroll_for`, but silently no-opped in the
`BlockRewriter`-based rounding operators, and a *declined* rewrite —
`agrees` refusing a format it cannot reproduce — was also a silent no-op, so
"it worked" and "nothing happened" printed the same.

The contract is the two-kind taxonomy every surveyed system converged on (Exo 2:
`SchedulingError` vs `InvalidCursorError`; MLIR: silenceable vs definite):

- **declined** — the transform refused, the program is unchanged, and the
  message says *why* ("declined: `inf_value` not reproducible under probes").
  Recoverable; this is what `try`/fallback strategies catch.
- **bad reference** — the `where` (later: cursor) does not point at anything.
- anything else is a bug and propagates.

**Done.** Still outside the hierarchy,
deliberately: `monomorphize`'s conflict errors (caller-supplied
contradictions, not references) and the STRICT-divisibility `ValueError`s in
`split`/`unroll_for`, which are declined-shaped but uncatchable via
`except TransformError` — revisit if a strategy ever wants to fall back on
them.

One hierarchy across both layers, no translation: the exceptions are named for
what happened (`TransformDeclined`, `TransformReferenceError`, base
`TransformError`), defined at the transform layer that raises them, and
re-exported from `fpy2.strategies`. A wrapper-level rename would force
internal callers of `Transform.apply` onto a different vocabulary than the
documented one and add a catch-and-reraise to every strategy for nothing —
Exo 2 likewise shares one `SchedulingError` between primitives and user
libraries.

An explicit `where` that matches nothing is a bad reference; `where=None`
keeps its apply-everywhere meaning, including to zero sites.

This is small, blocks nothing, and everything below assumes it. The Roly-poly
study is the evidence it matters beyond hygiene: hiding *why* a choice is
invalid measurably hurt users' understanding, and silence hides both the
invalidity and the reason.

## 2. Discoverability: candidate listing and step diffs

Roly-poly's central result: enumerating the valid next choices, with instant
feedback per step, is what made novices productive (schedules ~5× faster than
manual text scheduling). The enumeration already exists internally — each
transform scans for its candidates (`BlockRewriter._candidate`, the loop and
call scans) — it is just not exposed.

- A way to list a strategy's candidate sites with source locations, so
  `where=2` is inspectable before choosing it and explicable after.
- A way to see what a step did: a pretty-printed before/after diff. The object
  program stays printable FPy source at every step, so this is cheap.

Same bucket, for the docs: `docs/source/strategies.rst` is an API dump with no
ordering guidance, while ordering is load-bearing ("run `unfold_special`
first" lives only in docstrings). Exo 2's Appendix A format — one table row per
operator: *operator | before ⇝ after | conditions under which it declines* —
is the right shape, and item 1 makes the conditions column honest.

## 3. Locations that survive rewrites

**Done**, and it retires the blocker recorded in
[rounding-operator-basis.md](rounding-operator-basis.md): `where` counted
candidate blocks, transforms change how many there are, so composition could not
carry a location.

Exo 2's answer is cursors with *forwarding*, and that is what was built — with the
scope FPy's transforms allow, since each already knows which statements it
replaced. A cursor is a parent-linked path (`StmtCursor` / `BlockCursor` /
`ExprCursor`, united as `Cursor`); a rewriting pass reports a purely structural
edit log, which `Function.forward` replays back along the chain of programs; and
the strategies rebase a stale cursor on arrival, so a schedule pins a point once
and aims a whole sequence at it. `sites(strategy, func, within=)` lists what a
`where` may name. An un-forwardable cursor invalidates loudly (item 1's
bad-reference error) — including across the passes that rewrite at sites they do
not report, which say so rather than guessing.

The implementation is `fpy2/transform/cursor.py` and `Function.forward`.

Not delivered, and the claim this item used to make: **forwarding does not carry an
analysis result.** `fpy2/analysis/value_class.py` keys results by expression
identity, and every visitor rebuilds every expression, so carrying one needs the
old-node → new-node correspondence threaded through `DefaultTransformVisitor`
itself — a different mechanism from a path, which resolves in the rebuilt tree for
free. Item 6 can re-run the analysis on the rewritten program, which is what the
transforms already do; the carry-over is its own item if a use appears.

## 4. Patterns produce references

**Done.** `fpy2/rewrite` was an island: it named locations by an occurrence
index, raised its own `RewriteError`, and returned an opaque program no cursor
could cross. Now a match carries a cursor (`ExprCursor` for an expression
pattern, `StmtCursor` / `BlockCursor` for a statement one), `find` / `find_all`
name a location by what it looks like, `Rewrite` reports an edit log and takes
the same `where` as every other strategy, and its failures are
`TransformReferenceError` / `TransformDeclined` — so a user rule sits in a
`try`/fallback beside a built-in, and a cursor crosses it.

`where=` does *not* accept a pattern. Exo's operators do, but only because
occurrence disambiguation lives inside the pattern string (`#2`); FPy patterns
are decorated functions with nowhere to put an index, so `find_all(p, f)[i]`
keeps that index on a list instead of smuggling it back into `where`. "Apply at
every match" is a combinator (item 5), not a location.

Not delivered: **verification**. Exo 2 treats the checked user rewrite —
`rewrite_expr(p, e, e')`, verified equivalent — as a core primitive. A user
rewrite here has a built-in's *type*, failure contract and location vocabulary,
but not its guarantee: nothing checks that `l` and `r` compute the same thing.
The existing probes test whether a *format's* rounding is reproduced, which says
nothing about two expressions, so this needs interpreting both sides on sampled
inputs — its own item, and sampling rather than proof (SMT stays rejected on
cost).

## 5. Combinators, in user space

With one failure contract (1) and stable locations (3), ELEVATE's and Exo 2's
control vocabulary — `seq`, `try_else`, repeat-until-declined — is a few lines
of plain Python each, because strategies already share one type
(`Function -> Function`). Ship a handful in `fpy2.strategies` mostly as
documentation of the intended style. Do *not* build a traversal language
(`topDown`/`bottomUp`): FPy's transforms are flat and statement-level, and
traversal order earns nothing here.

## 6. Inspection for conditional schedules

Let schedules ask questions before acting: given a cursor, query the rounding
context, the inferred format, and the value class
(`fpy2/analysis/value_class.py`, `fpy2/analysis/format_infer/`). Exo 2's
claim, borne out by their `vectorize`, is that user-written automation is
impossible without inspection — "does this format state NaN?" is the guard on
whether `unfold_special` has anything to do. What it needs from item 3 is the
cursor to ask *about*, not carried analysis results — those are cheap to re-run on
the rewritten program, and forwarding does not carry them (see item 3).

## 7. The recipe as a parameterized function

Gap 2 of [native-lowering-roadmap.md](native-lowering-roadmap.md). Exo 2's
`optimize_level_1` is the model: one entry point taking the function, a
location, and a target descriptor object, built by composing the public
operators, with deviations as hooks rather than policy baked into transforms.
Item 3 has removed the stated reason the sequence was not exposed: a cursor now
aims the whole sequence at one site. Along the way, transforms without wrappers that the recipe wants
(`Specialize` first — Exo 2 generates all its tail cases from `specialize` +
simplification) get strategy wrappers, and `SplitLoopStrategy` /
`ForUnrollStrategy` get re-exported from `fpy2.strategies` so no schedule
needs a second import from the transform layer.

## Not recommended

- **SMT equivalence checking** (Exo 2's safety mechanism). Probes —
  `fixed_probes` / `agrees` — fit rounding semantics exactly and are already
  in place; Exo 2 reports 30 s – 2 min to run a schedule under SMT.
- **E-graphs / sketch-guided search.** Rewrite *search* is not the bottleneck;
  schedules here are short. The one idea worth keeping from guided equality
  saturation is sketches as postcondition asserts between phases, and item 4's
  patterns already provide that language for free.
- **Cost estimation and autotuning.** The objective here (exactness, then
  code shape) is not a scalar; Roly-poly also found cost hints double-edged —
  users follow them blindly.
- **A GUI.** Items 1 and 2 are the textual versions of the two things its
  study showed actually helped.
