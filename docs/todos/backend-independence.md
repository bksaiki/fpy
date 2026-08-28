# Roadmap: a backend-independent lowering pipeline

## Goal

Make `CppCompiler` a wrapper: run passes and analyses, then emit C++ 1:1 — every
emitter method spells a construct rather than deciding one. Each decision it
gives up becomes a pass or analysis some other backend could reuse.

*Backend-independent* means expressible without target knowledge, not run by
every backend: §1 is a normalization FPCore would decline.

The point is not line count. A decision made in the emitter is a decision no
other consumer can see, test, or reuse: `round_elim`, the FPCore backend and the
scheduling language all lose the same work, and the C++ backend pays for it in
the most delicate code it has.

## Where we are

The pipeline in `compiler.py` is already backend-independent — `DefineUse`,
`ContextUse`, `ArraySizeInfer`, `FormatInfer`, `ValueClassInfer`, `Alias`,
`Escape`, `StorageInfer` all live in `fpy2/analysis/`. Four modules do not:
`types.py`, `ops.py`/`target.py`, `storage.py`/`variables.py`, `unbox.py`.

The leak is the emitter. Grouping `emitter.py`'s methods by what they *do*:

| group | lines | spelling? |
|---|---|---|
| library-op lowering (`sum`/`amin`/`min`/`any`/`zip`/`range`/`size`/compare chains) | 515 | no — loop construction |
| round/cast lowering (`_emit_integral_round`, `_bound_test`, `_undefined_guard`, …) | 482 | no — soft-float lowering |
| control-flow printing | 394 | yes |
| storage reconciliation (`_emit_at`, `_convert_storage`, `_rebuild_list`, `_adapt_*`) | 388 | no — but mostly construct-at-want, not conversion (§5) |
| fenv / context boundaries | 247 | mostly |
| declarations and bindings | 221 | yes |
| peepholes (`ldexp`, pow2, literal folding) | 191 | no — algebraic rewrites |
| op dispatch (`_dispatch`, `_try_widen`, `_maybe_cast`) | 174 | table-driven already |
| list / array spelling | 172 | yes |
| comprehension lowering | 125 | no |

About 1900 lines are decisions and about 790 are printing. That gap is this
roadmap.

Which sections close it: **§4** and **§7** shrink the emitter. **§3** moves code
out of the backend without shrinking it, as **§2** did. **§1** shrank it by
nothing — every `_bind_operand` and `_emit_at` path that survives statement form
is on an *aggregate* operand, so the deletions wait on naming aggregates.

## 1. Statement form — done

`fpy2/transform/anf.py`, exposed as `fpy2.strategies.to_anf`, run unconditionally
as the last step of `CppCompiler.specialize()`, over a program
`fpy2/transform/hoistable.py` has already put in *hoistable form*.

The pass takes no target fact. It does not predict where C++ will want a
temporary; it makes the situation unreachable, because every operand the emitter
sees is already a `Var` — the shape of every normalization here, as `ZipElim`
removes the zip rather than knowing how a backend materializes one.

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
- **Lower where it pays.** A ternary lowers whenever an arm is not an atom — an
  `IfStmt` restructures for free and buys reach no other pass provides. A bool
  chain lowers only where an operand needs a place: lowering a pure one loses the
  `And` that `ValueClassInfer` reads to drop a runtime guard. A `while` rotation
  duplicates its condition, so it is gated too.
- **Scalars only, by type.** A name holding a list is a second *place*, which
  decides whether a list keeps its shared handle. Chains are named at their
  outermost scalar, so no aggregate name is created. Widening this to aggregates
  is the follow-on the emitter deletions wait on.

The three miscompiles this closed are in [backend-cpp.md](backend-cpp.md).

## 2. Storage inference — done

`fpy2/analysis/storage_infer.py`. The C++ ladder is one instance of its domain
(`CppStorageDomain`), and variable materialization — naming, declaration
placement, `binds_by_reference` — split off into `backend/cpp/variables.py`.

Storage inference is not format inference. Format inference bounds an expression
by the smallest format its value can take; storage inference picks, from a
distinguished finite set the target can spell, one format that *contains* that
bound:

```
e : real   fmt(e) = F   F <= S
------------------------------
          store(e) = S
```

Several `S` satisfy this, so the content is which one is chosen.

### The rules

`Sigma` is the backend's ordered sequence of formats; `F <= S` is the format
ordering lifted structurally through lists and tuples.

```
ceil(F)      = first sigma in Sigma with F <= sigma    -- else fallback, else refuse
ceil(bot)    = head Sigma        -- a fresh `empty` holds no value, so any rung does
ceil(none)   = none              -- non-numeric; the backend spells it as it likes
JOIN{S1..Sn} = first sigma in Sigma containing every Si          -- n-ary
```

A class is a union-find over `reaching_defs.same_object_defs`, which unions on
phi edges and `IndexedAssign` and nothing else — so a plain rebind `x = e`
starts a *new* class with its own, possibly narrower, storage. `store` is per
runtime object, not per name, and sequential redefinition never promotes. Each
class stores at `JOIN { ceil(F) | F a member's bound, F /= bot }`, or over the
bottom bounds when every member is bottom.

An expression's storage follows its *definition* where it has one — a `Var`
reads its class's storage, a `ListRef` peels the container's element — and its
own `ceil` otherwise. Definitions win because a class is a join over its
members, so a member's own bound names a format the value is not held in.

**The join is n-ary and must never be folded.** Containment over `Sigma` is not
a join-semilattice: `{s8, u16}` has two incomparable minimal upper bounds, `s32`
and `f32`, and no least one. Over the C++ ladder, folding *overshoots* in 4 of
the 3-element combinations — `JOIN{s8, u16, f32}` is `float`, folding gives
`double` — and *fails outright* in 12 where the n-ary join succeeds. The
sequence order is therefore a tie-break with downstream consequences, not a
presentation detail, and an interface offering a binary `join(a, b)` would
reintroduce the bug.

**Containment is guaranteed; realizability is not.** For a place `p` and a value
`v` flowing into it:

```
containment    fmt(v) <= store(p)      -- the value fits
realizability  store(v) ~> store(p)    -- the backend can get it there

sigma ~> sigma'      iff sigma <= sigma'
list S ~> list S'    iff S = S'                     (free)
                      or (S ~> S' and v unshared)   (rebuild)
```

The class join gives the first by construction. The second is where every
refusal a consumer raises comes from: a list may change element type only by
becoming a different object, so it needs a sharing verdict. A scalar's
realizability is a fact about formats; a list's is a fact about the heap.

### The interface

`StorageDomain` is one input and one hook:

- **`sigma`** — the storage formats as `Format`s, smallest first. Ordered, not a
  set, for the reason above.
- **`fallback(bound)`** — a storage for a bound no member contains, or `None` to
  refuse. The one thing the sequence cannot supply: C++ stores an unbounded
  integer in `int64_t` and treats overflow as the user's problem, which no
  containment test would allow.

Nothing else. Structure needs no map, since `ListFormat` and `TupleFormat` are
already bounds. Losslessness *is* containment, so no conversion relation is
needed — a target supporting *fewer* conversions than containment allows would
need one, and would have to make `join` require reachability too.

Spelling a format in the target's own types stays in the backend, as does
*representation* — handle, value, fixed array — which no format has.

### Two behaviours inherited deliberately

**The class join stays monotone.** A partly-`bot` bound still contributes
`head Sigma` at its empty slots, because the join is over storages rather than
bounds: `JOIN{ceil(bot), ceil(s8)}` is `s16` where `ceil(bot |_| s8)` is `s8`.
Joining in the format lattice first and choosing storage once is strictly more
precise, and changes emitted output.

**Widening stays minimal containment**, which is right for a scalar and wrong
for a list element. C++ converts a scalar at the point of use, so a rung too
narrow costs one upcast; `std::vector<float>` and `std::vector<double>` are
unrelated types, so the same mistake costs a new buffer and a new object
identity.

| | too wide | too narrow |
|---|---|---|
| scalar | a few bytes | a free upcast |
| list | *n* × a few bytes | O(n) copy + allocation, or a refusal |

Headroom would not pay *inside* a class — the join is already wide up front and
every construction site builds at it. It pays where a member's storage is fixed
outside the class: a parameter, a callee's return. Expressing that needs a
policy per structural position rather than one rule applied structurally. Today
it has no consequence, because `_rebuild_list` is unreachable — the boxing
verdict refuses first. Closing that gap makes the rebuild path live, and only
then does the policy cost anything measurable.

## 3. Representation inference

`unbox.py` decides, per alias region, whether a list is a shared handle, a plain
value, or a fixed-length value. It splits three ways.

**Backend-independent — done.** 99 lines left `unbox.py` (720 → 621) for
`fpy2/analysis/alias.py`:

- `region_sizes(alias, array_size)` — one proven length per region, by meeting
  every contribution. A free function, not a method: `Alias.analyze` must not
  require an `ArraySizeAnalysis`, since `escape` and `format_infer` both build an
  `AliasAnalysis` without one.
- `AliasAnalysis.written_regions` / `.slot_replaced` / `.returned_levels` —
  syntactic facts keyed by region, collected by a second walk over the finished
  graph. Second rather than folded into `_Builder`: each asks `region_of` for
  where a place ends up, which is settled only once every merge has run. A bare
  traversal is 1.4% of the builder, measured.

`_Scan`'s fourth fact did **not** move. *A region crosses a call boundary* is
decided by `_unboxed(param.ty)` on the callee's ABI, and the callee-written half
of `written` by `param.written` — both questions about a representation this
target chose. What is left in the backend is one `_visit_call`.

**Generic algorithm, backend-supplied predicate (about 70 lines).**
`_shares_storage` asks how many *places* hold a region separately, which is
language-level, but its precision is calibrated to this emitter: it does not use
`alias.is_shared`, it counts only names that get their own storage, and that set
is `binds_by_reference`, shared with the emitter so the two cannot drift.
Discounting a name the emitter then copies is a miscompilation, so the extracted
form takes the binding rule as a parameter — which would have exactly one
implementation until there is a second backend to check it against.

**Backend-specific (about 300 lines).** The output alphabet: C++ wants
handle / value / fixed array, Rust wants `Rc<RefCell<Vec<T>>>` / `Vec` /
`&mut [T]` / `[T; K]` and a borrow checker that changes what a second place
costs, and a garbage-collected target has nothing to decide. Then
`writes_through` and `may_reference_projection`, which ask whether `const`
reaches through a handle or a value and whether a reference re-reads a slot where
**E-Deref** read it once. Then the verdict-to-type mapping (`_stamp`, `_regions`,
`annotate`), `UnboxMode` and its diagnostics, and `ParamAbi` / `CalleeAbi`.

The emitter did not shrink and was never going to: it consults an oracle either
way. What the move bought is that `Alias` now answers three questions any
consumer of the region graph would ask, that one proven length per region is
available to more than the C++ boxing decision, and that all four are tested
directly rather than through an emitted C++ string.

Still available here: `_regions` / `_at_depth` / `_fields` / `_stamp` (about 90
lines) walk a storage class's structure, which since §2 is a `FormatBound` rather
than a `CppType` — so a generic version is now expressible where it was not. But
`_stamp` writes the representation, which is the backend-specific output
alphabet, so the traversal and the verdict have to be split first.

## 4. Round and cast lowering

482 lines lowering a rounding under a *fixed-point* context: a libm call for the
mode, an assertion for the operand's undefined cases, an assertion for the
format's bound. Its refusals name what it assumes upstream — digits at position
zero (`rescale_fixed`), overflow `ASSERT` (`unfold_overflow`), no random bits.

Running `unfold_special → unfold_overflow → float_to_fixed → rescale_fixed →
simplify` does not replace it: after the full sequence `_emit_integral_round`
still fires, and the `std::nearbyint` and `assert(std::fabs(...) <= 1024)` in
[native-lowering-roadmap.md](native-lowering-roadmap.md)'s output are its work.
Those passes normalize a rounding *down to* a position-zero fixed round; nothing
upstream turns that into a libm call and a bound check.

So this section is a new transform: an FPy-level pass rewriting a position-zero
fixed round into `nearbyint` / `trunc` / `floor` / `ceil` plus an `AssertStmt` on
the bound. Every piece is already an FPy op. Only then does
`_emit_integral_round` die, leaving a `static_cast` for native contexts and the
refusals as tripwires.

Two things to weigh:

- Across the 201 compiling corpus functions the non-native lowering never fires
  (only `_fold_rounded_literal`, three times). The programs exercising it are
  built by hand in `tests/unit/backend/cpp/test_lowered_roundtrip.py`, so this is
  a maintenance win rather than a behaviour one.
- The new pass owes what the emitter owes: the mode table (five modes are one
  libm call, three are composed), the operand guard derived from the context's
  flags *and* from `value_class.py`, and the two-sided bound where the format is
  asymmetric. Most of the 482 lines move rather than disappear.

## 5. Conversion insertion — blocked, and smaller than it looks

The idea: after storage inference, insert an explicit conversion node wherever an
operand's storage differs from its place's, and refuse where nothing bridges. The
emitter then prints `static_cast` and rebuild loops without deciding anything, and
`backend-cpp.md`'s four-column table retires.

**Measured, there is nothing to insert.** Over the 201 compiling corpus functions
`_convert_storage` is reached 37 times and *every call has `src == want`*;
`_rebuild_list` never runs. The 114 scalar casts that do fire come from op
dispatch, where C++ converts at the point of use and no slot is needed at all. So
the pass would today rewrite nothing.

The 388 lines grouped as *storage reconciliation* are therefore not conversion
insertion. They are `_emit_at`'s **construct-at-want** path — 178 aggregate sites,
building a `ListExpr` or `TupleExpr` directly at the wanted type. That is the
emitter picking a constructor's type argument, not a node a pass could hoist out.

What is left is the refusal path, which is correctness-bearing even at zero
firings. It goes live together with `_rebuild_list`, i.e. once the boxing gap
below closes — the same trigger the widening policy waits on.

## 6. Pow2 and literal peepholes

`_emit_scale_by_pow2`, `_emit_pow2`, `_fold_rounded_literal`,
`_mode_independent` (191 lines) are algebraic rewrites gated on exactness proofs
— `exact_exp2`, `round_is_identity` — that are already generic. Only
`std::ldexp` is C++.

Statement form names the power, so `_emit_scale_by_pow2` no longer matches
`2 ** n * x` (see [backend-cpp.md](backend-cpp.md)). Moving the strength
reduction here restores it, and needs a language-level scale operation first:
FPy has no `ldexp`/`scalb`, so an FPy-level transform has nothing to rewrite
into. That makes this a language change, not only a pass move.

## 7. Library-op lowering

515 lines, and it splits. Reducing `sum` / `amin` / `amax` / `any` / `all` to a
loop is generic, with `ZipElim`, `EnumerateElim`, `ReduceFusion` and `CompToLoop`
as precedent. `_emit_ieee_min_max`'s NaN-propagating, signed-zero-aware predicate
is target-specific and stays.

`backend-cpp.md` measures capability regressions from lowering comprehensions
ahead of the emitter — a multi-clause comprehension loses its `std::array`, a
dependent-clause list stops compiling — so the FPy-level lowerings have to become
*more* capable first, not merely get scheduled earlier.

## Order of work

**Aggregate naming in ANF is next**, and its blocker is gone. Naming an
aggregate turns `return ([n, n], 1.0)` into `xs = [n, n]; return (xs, 1.0)`, and
until recently the second form kept a handle the first did not — so the pass
would have boxed every literal-into-container in the corpus. `consumed_defs`
and the matching `std::move` closed that; see *What stays boxed* in
[backend-cpp.md](backend-cpp.md).

It remains the highest-risk item on its own terms: a name holding a list is a
second *place*, and `UnboxMode.STRICT` turns that into a refusal. What closed
covers the name handed straight to a container; a name read more than once is
genuinely shared, and that is what the pass has to be measured against.

It also owns less than it was once credited with. All 113 `_bind_operand` mints
on the corpus come from library-op lowering — `_emit_ieee_min_max` (28),
`_emit_empty` (25), `_list_range` (21), `_emit_sum` (13), `_emit_zip` (12),
`_visit_list_slice` (11), `_emit_enumerate` (3) — which build a loop or emit an
operand twice. Those are §7's, and they die when that lowering moves to FPy
level, where ANF names the loop's temporaries itself.

**§3's first part** is done; its second waits for a second backend. **§4** is
independent, but is a new transform whose lines mostly move rather than
disappear. **§6** needs a language-level scale operation. **§7** waits for the
FPy-level lowerings to become total, and owns the 113 `_bind_operand` mints.

## Staying in the backend

- `types.py`, `ops.py`, `target.py` — what C++ can spell, and how.
- Declaration and binding shape, list and array spelling, control-flow printing
  (about 790 lines). This is the 1:1 emitter the roadmap aims at.
- `fenv` boundaries (247 lines). Generic in principle — any target with a global
  rounding-mode register — but there is one such target, so extracting it would
  abstract over a single instance.
- The representation alphabet in `unbox.py` — handle, value, fixed array — and
  the C++ questions asked of it (§3).
- `_emit_ieee_min_max`, and `UnboxMode` as a policy knob.
