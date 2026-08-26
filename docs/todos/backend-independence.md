# Roadmap: a backend-independent lowering pipeline

## Goal

Make `CppCompiler` a wrapper: run passes and analyses, then emit C++ 1:1 — every
emitter method spells a construct rather than deciding one. Each decision it
gives up becomes a pass or analysis some other backend could reuse.

*Backend-independent* here means expressible without target knowledge, not run by
every backend: §1 is a normalization FPCore would decline, and that is fine.

The point is not line count. A decision made in the emitter is a decision no
other consumer can see, test, or reuse: `round-elim`, the FPCore backend and the
scheduling language all lose the same work, and the C++ backend pays for it in
the most delicate code it has.

## Where we are

The pipeline in `compiler.py` is already backend-independent — `DefineUse`,
`ContextUse`, `ArraySizeInfer`, `FormatInfer`, `ValueClassInfer`, `Alias`,
`Escape` all live in `fpy2/analysis/`. Four modules do not: `types.py`,
`ops.py`/`target.py`, `storage.py`/`storage_infer.py`, `unbox.py`.

So the leak is not the pipeline, it is the emitter. Grouping `emitter.py`'s
methods by what they *do* (line counts by method, at `be09670`, 3815 lines):

| group | lines | spelling? |
|---|---|---|
| library-op lowering (`sum`/`amin`/`min`/`any`/`zip`/`range`/`size`/compare chains) | 521 | no — loop construction |
| round/cast lowering (`_emit_integral_round`, `_bound_test`, `_undefined_guard`, …) | 482 | no — soft-float lowering |
| storage reconciliation (`_emit_at`, `_convert_storage`, `_rebuild_list`, `_adapt_*`) | 384 | no — conversion insertion |
| control-flow printing | 382 | yes |
| fenv / context boundaries | 247 | mostly |
| declarations and bindings | 221 | yes |
| peepholes (`ldexp`, pow2, literal folding) | 191 | no — algebraic rewrites |
| op dispatch (`_dispatch`, `_try_widen`, `_maybe_cast`) | 174 | table-driven already |
| list / array spelling | 172 | yes |
| comprehension lowering | 125 | no |

About 1900 lines are decisions and about 775 are printing. That gap is this
roadmap.

## 1. Statement form — **done**

`fpy2/transform/anf.py`, exposed as `fpy2.strategies.to_anf`, and run
unconditionally as the last step of `CppCompiler.specialize()`.

**Why it is backend-independent.** The pass takes no target fact and needs none.
It does not predict where C++ will want a temporary; it makes the situation
unreachable, because every operand the emitter sees is already a `Var`. That is
the shape of every normalization here — `ZipElim` removes the zip rather than
knowing how a backend materializes one. Backend-*independent* is not
backend-*neutral*, though: FPCore is an expression language where this would emit
a `let*` chain for nothing, so it is a normalization the C++ backend opts into.

**Where a temporary goes.** In a statement slot executed *exactly as often, and
under exactly the same condition, as the expression it names* — not the nearest
enclosing statement, which is wrong for a `while` condition and a ternary arm.
Positions that fail that test are sealed; two of them get a lowering that
*creates* the slot (rotation for `while`, `IfStmt` for a ternary), and
`ANF.refusals` reports what is left. Over the 230-function corpus the residue is
entirely comprehensions, and **zero** in the three positions the emitter cannot
slot.

### What it settled

- **Naming materializes.** A pass that would have *deleted* an expression can
  only reach inside the name once it has one — `RoundElim` collapsing
  `fp.round(0.0)` to a literal must run first, or it leaves a `uint8_t` binding
  behind. So ANF goes after everything that removes or folds, and nothing that
  removes or folds runs after it. Measured: earlier placement is a strict
  superset of failures.
- **A normalization turns every downstream syntactic match into a def-use walk.**
  `DefineUseAnalysis.defining_expr` is that walk; five matchers needed it
  (`ValueClassInfer._implied`, `ArraySizeInfer._const_int` / `_len_size` /
  `_affine`, the emitter's pow2 peephole). None was an ANF regression — each
  already failed on hand-written code like `n = len(xs); fp.empty(n)`, so the
  fix is an improvement independent of the pass.
- **Lower where it pays, not for uniformity.** A ternary lowers whenever an arm
  is not an atom, because an `IfStmt` restructures for free and buys reach no
  other pass provides. A bool chain lowers only where an operand needs a place:
  lowering a pure one loses the `And` that `ValueClassInfer` reads to drop a
  runtime guard. A `while` rotation duplicates its condition, so it too is gated.
- **Scalars only, by type.** A name holding a list is a second *place*, which
  decides whether a list keeps its shared handle. Chains are named at their
  outermost scalar, so no aggregate name is ever created. Widening this is the
  remaining follow-on, and §9's measurement shows the emitter deletions below
  wait on it.

The three miscompiles this closed are in [backend-cpp.md](backend-cpp.md).

## 2. Storage inference

*Absorbs the former `docs/todos/storage.md`.*

Storage inference is not format inference. Format inference maps a real-valued
expression to the smallest format bounding its value; storage inference maps it
to a member of a distinguished, finite set of formats that *contains* that bound:

```
e : real   fmt(e) = F   F <= S
------------------------------
          store(e) = S
```

Several `S` satisfy this, so the interesting content is which one is chosen, and
what constrains the choice beyond containment.

**What is already generic.** `storage_infer.py` is 312 lines of union-find,
class naming and declaration-site placement, and it touches C++ in exactly five
places, all through `CppType` and `aggregate_storage`. Parameterize it on a
lattice — `of_bound(FormatBound) -> S`, `join([S]) -> S`, `is_aggregate(S)` —
move it to `fpy2/analysis/`, and leave `_LADDER` in the backend as one instance.

**The assignment and phi rules the doc left open.** They are already implemented,
and more narrowly than the question assumed:

- **Assignment is not a promotion site.** A class is a union-find over
  `reaching_defs.same_object_defs`, which unions on phi edges and
  `IndexedAssign` only. A plain rebind `x = e` gets a *fresh* variable with its
  own, possibly narrower, storage. `store` is per runtime object, not per name,
  so sequential redefinition never promotes.
- **A phi is the only promotion site, and the rule is structural.**
  `aggregate_storage` takes the supremum element-wise through lists and tuples,
  dropping bottom bounds (a fresh `empty(...)`, which holds no value and so
  constrains nothing).
- **Promoting compound data is the real constraint, and the rule is aliasing.**
  A container's promotion is a *rebuild* — a new buffer, hence a different
  object — so it is legal only where nothing else can observe the identity.
  `_convert_storage` performs it for a value and `_refuse_unsharing` rejects it
  for a shared list.

So `F <= S` is necessary and not sufficient: among the `S` that satisfy it, the
admissible ones are constrained by *aliasing*, not only by containment. A shared
aggregate admits no promotion at all. Stating that is the generalization; the
containment rule alone would license a silent miscompile.

Still open: a *partly* bottom bound contributes its empty slots, because the
supremum is taken over storages rather than over bounds
(`aggregate_storage`'s own note). Fixing it means joining in the format lattice
first and choosing storage once.

## 3. Representation inference

`unbox.py` (722 lines) decides, per alias region, whether a list is a shared
handle, a plain value, or a fixed-length value. It does **not** split into an
analysis and a spelling; it splits three ways, and only the first part is free.

**Backend-independent, no callback (about 190 lines).** `_region_sizes` meets
`ArraySizeInfer`'s bounds over the region graph — 42 lines with no `CppType` in
them, useful to anyone wanting one proven length per region. `_Scan`'s four facts
are the same: a region is stored into, a slot is replaced, a region crosses a call
boundary, a region is returned at depth *d*. `Alias` already owns the region graph
(`referrers`, `escapes`, `transfers_ownership`, `defs_in`, `region_at`,
`region_field`), so that is where they belong.

**Generic algorithm, backend-supplied predicate (about 70 lines).**
`_shares_storage` asks how many *places* hold a region separately, which is a
language-level question — but its precision is calibrated to this emitter. It
deliberately does not use `alias.is_shared`: it counts only names that get their
own storage, and that set is `binds_by_reference`, shared with `storage_infer` so
the two cannot drift. Discounting a name the emitter then copies is a
miscompilation, so the extracted form takes the binding rule as a parameter. The
interface is the deliverable, not the code motion.

**Inherently backend-specific (about 300 lines).** The output alphabet, first of
all: C++ wants handle / value / fixed array, Rust wants
`Rc<RefCell<Vec<T>>>` / `Vec` / `&mut [T]` / `[T; K]` and has a borrow checker
that changes what a second place costs, and a garbage-collected target has
nothing to decide. Then `writes_through` and `may_reference_projection`, which
answer C++ questions — whether `const` reaches through a handle or through a
value, and whether a reference re-reads a slot where **E-Deref** read it once.
Then the verdict-to-type mapping (`_stamp`, `_regions`, `annotate`), `UnboxMode`
and its diagnostics, and `ParamAbi` / `CalleeAbi`.

So about a third moves, and the payoff is not a smaller emitter — it consults an
oracle either way. It is that `Alias` gains facts other consumers want, and that
the sharing verdict becomes testable without a C++ string.

Do the first part on its own; it is independent of everything else here. Hold the
second until there is a second backend to check the interface against —
abstracting an output alphabet over one instance is the mistake this roadmap
already declines to make for `fenv`.

## 4. Round and cast lowering

The round/cast group (482 lines) is `unfold_special` + `unfold_overflow` +
`float_to_fixed` + `rescale_fixed` re-implemented in strings — and it already
refuses everything those passes have not normalized: a context whose digits are
not at position zero, an overflow rule other than `ASSERT`, a stochastic context.
It is, in effect, the tail of a pass pipeline that is not wired up.

Wiring the recipe from
[native-lowering-roadmap.md](native-lowering-roadmap.md) (its gap 2) in front of
codegen collapses this group to: a native cast, plus printing the assertions the
rewritten program now states itself.

Independent of §1–§3, and the largest single reduction. It is also the one
subtask whose plan is already written down elsewhere; this roadmap only claims
the consequence for the emitter.

## 5. Conversion insertion

`backend-cpp.md`'s *One question answered in four places* scopes this as one
predicate per place kind. As a **pass** it is better: after storage inference,
insert an explicit conversion node wherever an operand's storage differs from its
place's, and refuse where nothing bridges. The emitter then prints
`static_cast` and rebuild loops without deciding anything, and the four-column
table in that document retires.

Needs §1: the places have to exist as program points before a pass can put a node
at one.

## 6. Pow2 and literal peepholes

`_emit_scale_by_pow2`, `_emit_pow2`, `_fold_rounded_literal`, `_mode_independent`
(191 lines) are algebraic rewrites gated on exactness proofs — `exact_exp2`,
`round_is_identity` — that are already generic. Only `std::ldexp` is C++.

A `pow2` strength-reduction transform plus one target-table entry. Small, low
risk, orderable anywhere.

## 7. Library-op lowering

521 lines, and it splits. Reducing `sum` / `amin` / `amax` / `any` / `all` to a
loop is generic, with `ZipElim`, `EnumerateElim`, `ReduceFusion` and `CompToLoop`
as precedent. `_emit_ieee_min_max`'s NaN-propagating, signed-zero-aware predicate
is genuinely target-specific and stays.

Do this last. `backend-cpp.md` measures real capability regressions from lowering
comprehensions ahead of the emitter (a multi-clause comprehension loses its
`std::array`; a dependent-clause list stops compiling), so the FPy-level
lowerings have to become *more* capable first, not merely get scheduled earlier.

## Order of work

1. **Statement form** (§1) — unblocks §5, cheapens §2 and §7.
2. **Storage inference** (§2) — smallest change, and the one with a lattice
   interface the rest can reuse.
3. **Conversion insertion** (§5).

**§3's first part** — region sizes and `_Scan`'s facts into `Alias` — is
independent and can go at any point; its second part waits for a second backend.
**§4** runs in parallel from the start; **§6** anywhere; **§7** after the
FPy-level lowerings are total.

## Staying in the backend

Named so that "make it backend-independent" does not creep into them:

- `types.py`, `ops.py`, `target.py` — what C++ can spell, and how.
- Declaration and binding shape, list and array spelling, control-flow printing
  (about 775 lines). This is the 1:1 emitter the roadmap is aiming at.
- `fenv` boundaries (247 lines). Generic in principle — any target with a global
  rounding-mode register — and there is one such target, so extracting it now
  would abstract over a single instance.
- The representation alphabet in `unbox.py` — handle, value, fixed array — and
  the C++ questions asked of it (§3).
- `_emit_ieee_min_max`, and `UnboxMode` as a policy knob.
