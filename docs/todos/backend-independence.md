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

## Open

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

Still the highest-risk item here: a name holding a list is a second *place*, and
`UnboxMode.STRICT` turns that into a refusal. What closed covers a name handed
straight to a container; a name read more than once is genuinely shared, and that
is what the pass must be measured against.

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
move.

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

Blocked on capability: [backend-cpp.md](backend-cpp.md) measures regressions from
lowering comprehensions ahead of the emitter — a multi-clause comprehension loses
its `std::array`, a dependent-clause list stops compiling — so the FPy-level
lowerings have to become *more* capable first, not merely run earlier.

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
