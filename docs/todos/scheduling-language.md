# Roadmap: growing the scheduling language

The point of this project is the FPy-specific transforms, not scheduling
infrastructure — so this roadmap records only the infrastructure worth
building, in the order it pays off. The source material is Exo 2 (Ikarashi et
al., ASPLOS '25, arXiv:2411.07211) and the systems around it: ELEVATE's
strategy combinators, the MLIR Transform dialect's handles and failure
taxonomy, and the Roly-poly user study on guided scheduling (arXiv:2107.12567).
Each item says what transfers and what it fixes here.

## 1. One failure contract

Failure is inconsistent today: an out-of-range `where` raises `ValueError` in
`inline`, `split`, and `unroll_for`, but silently no-ops in the
`BlockRewriter`-based rounding operators (pinned as intended by
`tests/unit/transform/test_unfold_special.py`), and a *declined* rewrite —
`agrees` refusing a format it cannot reproduce — is also a silent no-op, so
"it worked" and "nothing happened" print the same.

Adopt the two-kind taxonomy every surveyed system converged on (Exo 2:
`SchedulingError` vs `InvalidCursorError`; MLIR: silenceable vs definite):

- **declined** — the transform refused, the program is unchanged, and the
  message says *why* ("declined: `inf_value` not reproducible under probes").
  Recoverable; this is what `try`/fallback strategies catch.
- **bad reference** — the `where` (later: cursor) does not point at anything.
- anything else is a bug and propagates.

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

The known blocker, recorded in
[rounding-operator-basis.md](rounding-operator-basis.md): `where` counts
candidate blocks, transforms change how many there are, so composition cannot
carry a location — which is why the lowering sequence in
[native-lowering-roadmap.md](native-lowering-roadmap.md) is pinned by a test
rather than exposed as an entry point.

Exo 2's answer is cursors with *forwarding*: a reference is a path into the
AST; every primitive edit (insert / delete / replace / wrap) carries a
function mapping old paths to new; applying a transform forwards every live
cursor, and an un-forwardable cursor invalidates loudly (the bad-reference
error from item 1). Multiple cursors coexist, so a schedule can pin several
points before rewriting any of them.

Scope here is far smaller than Exo's: transforms are statement-level and few,
and each already knows exactly which statements it replaced. Strategies accept
`where: int | Cursor`, and `Function` gains `forward(cursor)`.

Forwarding also fixes the standing hazard in `fpy2/analysis/value_class.py` —
results keyed by expression identity die on any rewrite. The forwarding map is
precisely the old-node → new-node correspondence that lets an analysis result
carry across a schedule step.

## 4. Patterns produce references

`fpy2/rewrite` (`@fp.pattern` + `Rewrite`) already does structural matching
with an occurrence index, but it is an island — unreachable from
`fpy2.strategies`. Unify the layers: matching a pattern (over a function, or
under a cursor) returns cursors; strategy `where=` accepts an int, a cursor,
or a pattern. Exo 2 treats the checked user rewrite — `rewrite_expr(p, e, e')`,
verified equivalent — as a core primitive with the same type as every built-in;
`Rewrite.apply` should become a strategy on the same terms, probe-verified
where a context is known.

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
whether `unfold_special` has anything to do. Depends on item 3: without
forwarding, an analysis result is dead after the first rewrite.

## 7. The recipe as a parameterized function

Gap 2 of [native-lowering-roadmap.md](native-lowering-roadmap.md). Exo 2's
`optimize_level_1` is the model: one entry point taking the function, a
location, and a target descriptor object, built by composing the public
operators, with deviations as hooks rather than policy baked into transforms.
Item 3 removes the stated reason the sequence is deliberately not exposed
today. Along the way, transforms without wrappers that the recipe wants
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
