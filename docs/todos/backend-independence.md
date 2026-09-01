# Roadmap: a backend-independent lowering pipeline

## Goal

Move decisions out of `emitter.py` into passes and analyses, so that each one
becomes something a consumer other than this emitter can see, test and reuse.
`round_elim`, the FPCore backend and the scheduling language all lose the work
otherwise, and the C++ backend pays for it in the most delicate code it has.

*Backend-independent* means expressible without target knowledge, not run by
every backend: §1 is a normalization FPCore would decline.

**Not a line-count goal.** That framing was tried and it misled every estimate
made under it. Across §1–§3: `emitter.py` 3815 → 3871 (**+56**),
`fpy2/backend/cpp/` −108, `fpy2/analysis/` **+621** — of which about 358 is
`storage_infer.py` relocating, so roughly 260 lines are genuinely new. The
backend shed a tenth of what the pipeline gained. What the work actually bought,
in order of how much it was worth:

1. **Whole classes of reasoning removed.** After §1 every operand the emitter
   sees is a name, a literal or a nullary constant, so a family of "where does
   the temporary go" bugs is unreachable rather than fixed. Three live
   miscompiles died with it.
2. **Decisions became testable without a C++ string.** `StorageInfer` runs
   against a synthetic three-format domain; the region facts are asserted on the
   graph directly. Both found real defects that emitted-output diffing could not
   see.
3. **Reuse by a second backend** — genuine, but contingent on a second backend
   existing. Do not let it justify work on its own.

Judge a candidate section by (1) and (2). A section that only promises (3) can
wait.

## The pass audit

Every pass `CppCompiler.specialize()` runs, classified by what happens when it is
removed — each measured by monkeypatching it to the identity over the corpus:
the 219 functions of `tests/infra/examples` plus the `core`, `eft`, `vector` and
`matrix` libraries, compiled at FP64 through `CppCompiler.compile`, of which 202
compile.

| pass | kind | corpus without it |
|---|---|---|
| `FreeVarElim` | **required** | −2 |
| `Specialize` | **required** | not ablatable — the emitter has no template |
| `Hoistable` | normalization | **195 (−7)**, all 7 `ANF` refusals |
| `ANF` | normalization | 202 (0), 66 emit differently |
| `Hoistable` *and* `ANF` | — | **202 (0)**, 74 emit differently |
| `RoundElim` | optimization | 202 (0), 53 emit differently |
| `ZipElim` | optimization | 202 (0), 6 emit differently |
| `EnumerateElim` | optimization | 202 (0), **0** emit differently |
| `ReduceFusion` | optimization | 202 (0), **0** emit differently |

**Only two passes are required, and neither is a normal form.** `FreeVarElim`
because codegen has no closure environment, `Specialize` because the emitter is
monomorphic. Everything else is a policy choice, so everything else has to earn
its place.

**The iterable optimizations are nearly inert, and not for want of programs.**
The corpus has three `enumerate` uses and thirteen `zip` uses. All three
enumerates are in *value* position — `xs = [...]; return enumerate(xs)` — which
`EnumerateElim` does not match, so `_emit_enumerate` handles all three and
removing the pass changes no output at all. `_emit_zip` still fires six times
with `ZipElim` on. `ReduceFusion` never fires. The emitter's fallbacks are not
the rare path here; they are most of the traffic.

### The pair buys nothing the corpus can see

Read the three rows together. Dropping `ANF` alone loses no program. Dropping
`Hoistable` alone loses **seven** — and every one is `ANF` refusing its input,
all the same shape:

```
cannot normalize `determinant_2x2`: a short-circuited operand may not be
evaluated, so `len(A[0]) == 2` has nowhere to put a statement.  Run `Hoistable`
first.
```

Dropping **both** loses nothing. So `Hoistable`'s entire measured contribution is
keeping `ANF` from refusing, and `ANF`'s is nothing: the pair is self-justifying,
and the emitter compiles the same 202 programs without either.

This agrees with §1's own conclusion ("its value is capability, not
normalization") but locates the capability differently. The `while (2 ** (n + 1))
> 1.0` program that motivated statement form is rescued by `Hoistable`'s loop
rotation, not by atomization — and that program is in the unit suite, not the
corpus. What the corpus says is that the *normalization* has no measurable
payoff yet. Against it:
`ANF` costs the `2 ** n * x` fusion (§6), it turned five syntactic matchers into
def-use walks, and it names redundantly. `[x + 1 for x in range(5)]`, lowered,
comes out of `ANF` as

```python
t = range(5)
t4 = len(t)
acc = fp.empty(t4)
t5 = len(t)          # the same expression, named twice
for i in range(t5):
    x = t[i]
    acc[i] = x + 1
```

and nothing downstream removes the duplicate, because this pipeline runs no copy
propagation, no CSE and no dead-code pass.

**The verdict is not "delete `ANF`" but "stop running it unconditionally."** Its
guarantee — no operand's emission ever needs a statement — is worth having, and
`Hoistable`'s slot is what every rewrite below needs. The way to get the
guarantee is to make the set of statement-needing operands *empty*, which is
[A total unfold for every surface form](#a-total-unfold-for-every-surface-form),
not to name every operand in the program. Until that lands, `ANF` is an emitter
convenience with a measured cost and belongs behind a flag; `Hoistable` stays,
because the unfolds need the slot even though today only `ANF` asks for it.

### The rule the boundary follows

Every derived form in [derived-semantics](../source/dev/derived-semantics.rst)
admits two rewrites:

- the **unfold** — the derived-semantics equation read left to right. Total,
  unconditional, target-independent, and it *deletes* the emitter's case for that
  form.
- the **fuse** — a rewrite recognizing a shape the unfold would pessimize, and
  producing something better. May decline; declining costs speed, never
  compilability.

The pipeline has this the wrong way round. `EnumerateElim`, `ZipElim` and
`ReduceFusion` are fuses and they sit behind `optimize=True`, while the *total*
fallback is the emitter: `_emit_zip`, `_emit_enumerate`, `_emit_list_comp_at`,
`_emit_sum`, `_emit_any_all`, `_emit_amin_amax`, `_emit_min_max`,
`_visit_list_slice`. That is why those methods exist, and why the invariant is
local to this backend rather than a language property — 22 files outside it still
handle `ListComp`.

Right way round: fuse first (each matches surface syntax, so it must precede
anything that rewrites that syntax), then unfold what the fuses left, then the
emitter sees core forms only.

**What legitimately stays in the emitter is what FPy has no syntax for.** That is
the general form of §6's finding. `2 ** n * x` cannot be strength-reduced at FPy
level because FPy has no `ldexp`; `for i in range(a, b)` cannot be fused at FPy
level because FPy has no counted loop that is not a list. Everything else in the
table has an FPy-level target.

| emitter group | lines | corpus calls | FPy-level target |
|---|---|---|---|
| `zip` / `enumerate` | 64 | 9 | the derived comprehension |
| reductions — `sum`, `any`/`all`, `amin`/`amax` | 120 | 13 | the derived fold |
| `min` / `max` | 82 | 27 | derived `maximum` / `minimum` |
| slice | 25 | 11 | `[xs[i] for i in range(a, b)]` |
| chained comparison | 48 | 94¹ | `t1 = a; t2 = b; (t1 < t2) and (t2 <= c)` |
| `range` | 137 | 11 | **none** for the iterated position |
| pow2 peephole | 146 | 78² | **none** — FPy has no `ldexp` (§6) |

The comprehension group is gone: §8 moved it to `CompToLoop`.

¹ all comparisons; chains are a subset.
² calls, not fires — statement form defeats the match (§6).

### Precision is the acceptance test for an unfold

An unfold that deletes an emitter case and costs an analysis its precision has
not moved work into the pipeline; it has moved it into `REAL_FORMAT`, which has
no storage and so is a refusal. So every unfold owes a precision argument before
it owes a line count. There is already one instance on the record: §1 notes that
lowering a *pure* `and`/`or` chain cost `ValueClassInfer._implied` the `And` it
read to drop a runtime guard, and the fix was to teach the analysis the lowered
ladder rather than to weaken the gate. That is the pattern, and it generalizes.

Both analyses that decide storage read surface nodes directly.
`FormatInfer` special-cases `Sum` (`n − 1` exact pairwise adds through
`AbstractFormat` rather than widening), `Min` / `Max` / `AMin` / `AMax` (the
result is exactly one operand, so the join of the operand formats, with no
widening to the active scope), and every integer-producing op including
`Range1`/`2`/`3` and `Enumerate`'s index projection. `ArraySizeInfer` has a
`ListComp` case, a `Range` case and a `ListSlice` case.

Worked through, most of those survive the unfold, and for one reason: the
lowered form recomputes the same join.

| unfold | what the analysis loses |
|---|---|
| comprehension → `fp.empty` + loop | nothing in `FormatInfer` — `Empty` is bottom and the stores join back to the element format. **`ArraySizeInfer` loses a multi-clause length**: it must read `_const_int` on `fp.empty`'s argument, which does not fold `len(xs) * len(ys)` |
| slice → comprehension | nothing — both give `ListFormat(elt)` |
| `enumerate` → comprehension over `range` | nothing — `range`'s element is `INTEGER`, which is what the index projection said |
| `min` / `max` → the derived branches | nothing *if* the phi over the arms joins the same operands; the arms are `e1` and `e2` themselves, so it should |
| `sum` → an accumulator loop | nothing **conditional on the length staying statically known** — a `for` whose iterable has a proven length is driven exactly that many times, which is the same simulation the `Sum` rule performs |

**The risk is concentrated in one place: `ArraySizeAnalysis`.** `FormatInfer`
falls back to a fixpoint with widen-mode joins — and so to `REAL_FORMAT` — for
exactly the loops whose length it cannot prove, so every row above that says
"nothing" is really saying "nothing, while the length survives the rewrite".
§8 is the worked instance, and the instructive part is that the corpus hid it.
Lowering a multi-clause comprehension allocates `fp.empty(len(xs) * len(ys))`,
which `_const_int` could not fold — so `std::array<double, 12>` became
`std::vector<double>`, changing the signature, not just the loop. The corpus
never showed it: its multi-clause comprehensions iterate literal ranges, which
`_const_int` folds through partial eval, exercising a path the analysis does not
actually have to carry.

So the acceptance test for each unfold is not the emitted C++: it is that
`ArraySizeInfer` proves the same lengths after the rewrite as before, and that no
function's `fn_fmt` moves to `REAL_FORMAT`. Neither can be asserted over the
corpus alone. The test needs a witness set whose lengths come from
*proven-length parameters*, one per size shape the unfolds construct.
The remaining precision gaps `ArraySizeInfer` has are
[array-size-integer-exactness.md](array-size-integer-exactness.md) and
[array-size-symbolic.md](array-size-symbolic.md); this makes them a prerequisite
rather than an adjacent nicety.

## Done

### 1. Statement form

`fpy2/transform/anf.py`, exposed as `fpy2.strategies.to_anf`, run unconditionally
as the last step of `CppCompiler.specialize()`, over a program
`fpy2/transform/hoistable.py` has already put in *hoistable form*.

The pass takes no target fact. It does not predict where C++ will want a
temporary; it makes the situation unreachable. A temporary goes in a statement
slot executed *exactly as often, and under exactly the same condition, as the
expression it names* — not the nearest enclosing statement, which is wrong for a
`while` condition and a ternary arm. Positions failing that test are sealed; two
get a lowering that *creates* the slot (rotation for `while`, `IfStmt` for a
ternary), and `ANF.refusals` reports the rest.

A temporary goes in a statement slot executed *exactly as often, and under
exactly the same condition, as the expression it names* — not the nearest
enclosing statement, which is wrong for a `while` condition and a ternary arm.
Positions failing that test are sealed. Creating the slot is a second, weaker
normalization — `Hoistable`, run just before `RoundElim` — which rotates a
`while`, makes an `IfStmt` of a ternary and a guarded ladder of an `and`/`or`
tail. ANF does not do this itself: it *requires* it, and raises where a sealed
position holds something `needs_slot`, so a program that cannot be normalized
says so instead of looking normalized. Splitting the two matters because a
rewrite wanting a statement slot needs only the first: over the 230-function
corpus hoistable form costs 66 statements where the atomization on top costs a
further 295. `ANF.refusals` reports what is left, entirely comprehensions, with
zero in the three positions the emitter cannot slot.

What it settled:

- **Naming materializes.** A pass that would have *deleted* an expression can
  only reach inside the name once it has one: `RoundElim` collapsing
  `fp.round(0.0)` to a literal must run first, or it leaves a `uint8_t` binding
  behind. ANF goes after everything that removes or folds, and nothing that
  removes or folds runs after it.
- **A normalization turns every downstream syntactic match into a def-use walk.**
  `DefineUseAnalysis.defining_expr` is that walk. Five matchers need it:
  `ValueClassInfer._implied`, `ArraySizeInfer._const_int` / `_len_size` /
  `_affine`, and the emitter's pow2 peephole. Each already failed on hand-written
  code such as `n = len(xs); fp.empty(n)`, so the walk is an improvement
  independent of the pass. Its result belongs to the *assignment*: read it, but
  do not re-emit it or compare it structurally against a node from elsewhere.
- **Lower on a syntactic gate.** Every lowering fires when an operand is not an
  atom — never on `needs_slot`, which predicts what an emitter wants rather than
  describing the program, and would leave `Hoistable` unable to state its own
  postcondition. Lowering a *pure* chain costs the `And` that
  `ValueClassInfer._implied` read to drop a runtime guard; the fix was to teach
  it the lowered ladder, not to weaken the gate.
- **Scalars only, by type.** A name holding a list is a second *place*, which
  decides whether a list keeps its shared handle. Chains are named at their
  outermost scalar, so no aggregate name is created. Widening this to aggregates
  is the follow-on the emitter deletions wait on.

The three miscompiles this closed are in [backend-cpp.md](backend-cpp.md).

**What it is worth today, measured by disabling it.** Over the corpus: no
program's outcome changes, 56 emit differently. Over `tests/unit/backend/cpp`:
eight tests fail, of which seven are shape assertions and **one is a capability
loss** — `while (2 ** (n + 1)) > 1.0` compiles with statement form and raises
`CppInternalError` without it, because the condition needs a statement and the
emitter has nowhere to put one. That class of program is the whole current
payoff. Against it: the `2 ** n * x` fusion, which fires with ANF off and does
not with it on.

So its value is capability, not normalization. No pass got simpler — five
matchers became def-use walks instead of syntactic matches — and the emitter did
not shrink, because its generality is driven by aggregates, which this pass
deliberately leaves nested. The normalization payoff arrives with aggregate
naming or not at all.

`ANF` should come out of the unconditional prefix and `Hoistable` should stay —
see [The pair buys nothing the corpus can see](#the-pair-buys-nothing-the-corpus-can-see).

### 2. Storage inference

`fpy2/analysis/storage_infer.py`, whose module docstring is the contract: the
judgement, the class relation, the domain, and the containment-versus-
realizability split that every consumer refusal comes from. The C++ ladder is one
instance of `StorageDomain` (`backend/cpp/storage.py`); variable materialization
— naming, declaration placement, `binds_by_reference` — is
`backend/cpp/variables.py`.

One rule is worth repeating outside the code because an interface change would
silently break it: **the join is n-ary and must never be folded.** Containment
over the domain is not a join-semilattice, so a binary `join(a, b)` folded over a
class is both less precise and less total than one search over the whole class.

Two behaviours were inherited deliberately, and both are future work:

- **The class join stays monotone.** A partly-bottom bound still contributes the
  domain's first member at its empty slots, because the join is over storages
  rather than bounds. Joining bounds first and choosing storage once is strictly
  more precise, and changes emitted output.
- **Widening stays minimal containment**, which is right for a scalar (a rung too
  narrow costs one upcast) and wrong for a list element (`std::vector<float>` and
  `std::vector<double>` are unrelated types, so it costs a new buffer and a new
  object identity). Expressing the difference needs a policy per structural
  position rather than one rule applied structurally. It has no consequence until
  `_rebuild_list` becomes reachable.

### 3. Representation inference — the backend-independent part

99 lines left `unbox.py` for `fpy2/analysis/alias.py`:

- `region_sizes(alias, array_size)` — one proven length per region, by meeting
  every contribution. A free function, because an `AliasAnalysis` is usable
  without an `ArraySizeAnalysis` and `escape` and `format_infer` both build one
  that way.
- `AliasAnalysis.written_regions` / `.slot_replaced` / `.returned_levels` /
  `.consumed_defs` — syntactic facts keyed by region, collected by a second walk
  over the finished graph, since each asks `region_of` for where a place ends up
  and that is settled only once every merge has run.

`_Scan`'s fourth fact did **not** move: *a region crosses a call boundary* is
decided by the callee's declared representation, as is the callee half of
`written`. One `_visit_call` stays.

The emitter did not shrink and was never going to — it consults an oracle either
way. What the move bought is that four questions about the region graph are now
asked and tested directly rather than through an emitted C++ string, which is how
the boxing bugs below were found.

### 8. Comprehension lowering

`CppCompiler.specialize()` runs `Hoistable` and `CompToLoop` to a fixpoint, so
no comprehension reaches the emitter and `_visit_list_comp` is a tripwire. The
pass is total: a dependent clause list, whose length is a sum rather than a
product, is built a row at a time and flattened.

Cost-free on the corpus — 202 either way — but only after five fixes, four of
which were pre-existing defects the lowering merely exposed:

- `CompToLoop` fills its assignment target, and a nested one fills the slot it
  is stored into, rather than minting an accumulator a second name then holds;
- `_emit_empty` allows an allocation with fewer dimensions than the type's
  depth, which every lowered nested comprehension needs;
- a `range` iterable stays inline, since a name holds a value and a `range` in
  a value position must be materialized — and over a real bound cannot be;
- `_const_int` folds arithmetic over lengths it knows, so
  `fp.empty(len(xs) * len(ys))` keeps a static size;
- `_join_bounds` stops collapsing a widened join of a `SetFormat` with itself.

The last is the one worth remembering. Widening exists to cut infinite ascending
chains, and the `Format ⊔ Format` case had always short-circuited an equal join
before reaching it; the `SetFormat` case had not. It fires 8 times under the
dependent lowering and 0 times otherwise, so the asymmetry was invisible until
something asked for it.

What did *not* move is the emitter: it kept `_open_list_build`, which needs no
length up front, until the lowering became total. Nothing routes there now.

## Open

### A total unfold for every surface form

Steps 1 and 2 are done — see §8. What is left, each gated on
[the precision test](#precision-is-the-acceptance-test-for-an-unfold):

**Unfold the rest**, each straight from derived-semantics, each placed after its
fuse: `zip`, `enumerate`, slice, the chained comparison, `sum` / `any` / `all` /
`amin` / `amax`, and `min` / `max`. Three pay for more than the deletion:

- the **chained comparison** is one of the two positions `Hoistable` leaves
  sealed, so unfolding it makes hoistable form total over everything but an
  `assert` message — which is never evaluated, so nothing needs a slot there.
- **`min` / `max`** lowered to the derived definition puts the NaN and
  signed-zero branches where `ValueClassInfer` can discharge them per site, the
  way `unfold_special` already prunes a branch for a class its operand cannot
  hold. `_emit_ieee_min_max` open-codes all of them unconditionally.
- **`sum`** dissolves *Narrowing inside `std::accumulate`, so `Sum` can fuse* in
  [backend-cpp.md](backend-cpp.md): an FPy-level accumulator is an ordinary
  variable `StorageInfer` gives a class like any other, so there is no implicit
  narrowing to refuse. Unfold it only where the length is proven, or
  `FormatInfer` widens the accumulator instead of simulating the adds.

**Then a cleanup.** Every lowering leaves debris this pipeline has no pass to
remove: a dead `auto&& _tmp1 = xs.size();`, a `uint8_t n = _t8;` copy, `ANF`'s
two names for one `len(t)`. `fpy2/transform/dead_code.py` and
`copy_propagate.py` exist and neither is wired in; a lowering-heavy pipeline
needs both, plus CSE for the repeated `len`. This decides whether the unfolds
are free or merely correct.

**Then demote `ANF`.** With every statement-needing form already a statement, an
operand is a C++ expression by construction and the guarantee holds without the
pass. `_emit_inline`'s tripwire stays as the check.

### `Hoistable` at the front

`Hoistable` runs *after* `Specialize`, so it normalizes each spec rather than
each source function, and every pass before it works without a statement slot.
Moving it to the front is the right shape — it is what lets `CompToLoop`,
`RoundElim` and the unfolds mint a temporary without reasoning about conditional
evaluation — but it converts "establishes the invariant" into "every later pass
preserves the invariant", which nothing states or checks.

Hoistable form is a fixpoint, so re-running it is idempotent: put it at the
front *and* keep it where it is, and the second run is a no-op wherever the
invariant survived. It costs 66 statements over the corpus (§1), so that run is
close to free, and a difference between the two is a bug report about the pass
in between.

### Recovering from an unsupported rounding

The strongest remaining item, and the only one a user would notice: today an
unsupported rounding is a refusal whose message names the operator that fixes it;
the proposal is to run it. Fully scoped in *Recovering from an unsupported
rounding instead of refusing* in [backend-cpp.md](backend-cpp.md), including the
`split_round` route for arithmetic and why the correct-double-rounding table is
what makes it exact.

It moves no line out of the backend, and it turns programs that refuse into
programs that compile. Under the goal above that is a better trade than any
section below.

### Aggregate naming in ANF

Naming an aggregate turns `return ([n, n], 1.0)` into
`xs = [n, n]; return (xs, 1.0)`. Until recently the second form kept a handle the
first did not, which would have boxed every literal-into-container in the corpus;
`AliasAnalysis.consumed_defs` and the matching `std::move` closed that (see *What
stays boxed* in [backend-cpp.md](backend-cpp.md)).

§8 widened the discount from a definition to an *object*: a list a loop fills
reaches its consumer through a phi over the allocation and the stores, all one
runtime object, and a write-through is not a second read. So a filled name moves
out too.

What is left is a name genuinely read more than once, which is a second place by
any reading — that is what aggregate naming must be measured against, and
`UnboxMode.STRICT` turns it into a refusal rather than a slower program.

## Parked, with reasons

### 4. Round and cast lowering

An FPy-level pass rewriting a position-zero fixed round into a libm-shaped op
plus an `AssertStmt` on the bound. Measured on `test_lowered_roundtrip`'s
programs, the extractable content is **two** items, not the three originally
claimed: the mode table (five modes are one libm call, three are composed) and
the two-sided bound. `_undefined_guard` returns `None` on the
`test_lowered_roundtrip` path specifically, because the value-class branches the
earlier operators state rule out NaN and infinity there — it is *not* dead in
general, and emits a guard 46 times across the unit suite.

Parked because the lowering never fires on the corpus, so this is a maintenance
win rather than a behaviour one.

**It is also not under-tested**, which was briefly offered as a reason to build a
property-based harness instead. Measured over `tests/unit/backend/cpp`: all eight
rounding modes reach `_emit_integral_value` (RNE 45, RTZ 18, RAZ 7, RTE 7, RNA 6,
RTN 6, RTO 6, RTP 6), all three `_emit_integral_round` outcomes fire (113 lowered,
49 declined, 9 refused), `_emit_cast_round` fires 12 times and `_undefined_guard`
emits a guard 46 times. Six test files cover it, not one. A harness would add
generated formats and generated input values over hand-picked boundary values —
worth having, not worth prioritising.

It also has a prerequisite that is not worth doing on its own. `FormatInfer`
bounds `nearbyint`, `trunc`, `floor`, `ceil` and `roundint` at the active
*scope's* format — `_visit_unaryop` has an exactness rule for `Abs`, `Neg`,
`Logb` and `Exp2` and no case for these — so under `REAL` the result is
`REAL_FORMAT` and has no storage. A pass inserting those ops would need the rule.
Nothing reaches it today: no transformation constructs any of those nodes, and
under a concrete context the loose bound is already the right answer, since an
FP64-precision integer reaches 2^1024 and no integer rung holds it. So the gap is
real, has no consumer, and waits on this section rather than the reverse.

### 5. Conversion insertion

**Measured, there is nothing to insert.** `_convert_storage` is reached 37 times
over the corpus and *every call has `src == want`*; `_rebuild_list` never runs.
The 114 scalar casts that do fire come from op dispatch, where C++ converts at
the point of use and no slot is needed.

The 388 lines grouped as *storage reconciliation* are not conversion insertion:
they are `_emit_at`'s **construct-at-want** path, 178 aggregate sites building a
`ListExpr` or `TupleExpr` directly at the wanted type. That is the emitter
picking a constructor's type argument, not a node a pass could hoist out. What
remains is the refusal path, which goes live with `_rebuild_list`.

### 6. Pow2 and literal peepholes

Algebraic rewrites gated on exactness proofs that are already generic; only
`std::ldexp` is C++. Statement form names the power, so `_emit_scale_by_pow2` no
longer matches `2 ** n * x` (see [backend-cpp.md](backend-cpp.md)). Restoring it
needs a language-level scale operation — FPy has no `ldexp`/`scalb`, so an
FPy-level transform has nothing to rewrite into. A language change, not a pass
move.  Demoting `ANF` restores the fusion for free, which is one more entry on
that side of the ledger.

### 7. Library-op lowering

515 lines, the largest single group, and it splits: reducing `sum` / `amin` /
`amax` / `any` / `all` to a loop is generic, with `ZipElim`, `EnumerateElim`,
`ReduceFusion` and `CompToLoop` as precedent, while `_emit_ieee_min_max`'s
NaN-propagating, signed-zero-aware predicate stays.

It also owns all 113 `_bind_operand` mints on the corpus —
`_emit_ieee_min_max` (28), `_emit_empty` (25), `_list_range` (21), `_emit_sum`
(13), `_emit_zip` (12), `_visit_list_slice` (11), `_emit_enumerate` (3) — which
build a loop or emit an operand twice, and which die when the lowering moves to
FPy level where ANF names the temporaries itself.

**Unparked by the audit.** The capability blocker is smaller than recorded: with
`Hoistable` and `CompToLoop` run to a fixpoint, wiring comprehension lowering
into `specialize()` loses no corpus program and changes four emissions, the
`std::array` regression is stale, and the dependent-clause refusal has a
derived-semantics rewrite. `_emit_ieee_min_max` moves too, since the branches it
open-codes are what `ValueClassInfer` is for. The work is step 3 of
[A total unfold for every surface form](#a-total-unfold-for-every-surface-form).

### 3's second part — `_shares_storage`

Generic in shape but calibrated to this emitter: it counts only names that get
their own storage, and that set is `binds_by_reference`. The extracted form takes
the binding rule as a parameter, which would have exactly one implementation
until a second backend exists to check it against.

Likewise `_regions` / `_at_depth` / `_fields` / `_stamp`: §2 made a storage class
a `FormatBound`, so a generic traversal is now expressible, but `_stamp`'s
decision logic is already target-neutral and what is backend-specific is only the
type it reads and the type it builds. Splitting means a higher-order traversal
with one consumer — the same objection, with more added abstraction.

## Staying in the backend

- `types.py`, `ops.py`, `target.py` — what C++ can spell, and how.
- Declaration and binding shape, list and array spelling, control-flow printing.
  This is the 1:1 emitter the roadmap aims at.
- `fenv` boundaries. Generic in principle — any target with a global
  rounding-mode register — but there is one such target, so extracting it would
  abstract over a single instance.
- The representation alphabet in `unbox.py` — handle, value, fixed array — and
  the C++ questions asked of it.
- `_emit_ieee_min_max`, and `UnboxMode` as a policy knob.
