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

## 1. Statement form — the enabling change

The emitter calls `_bind_operand` 23 times because FPy expressions nest and C++
needs a *place*. A pass that binds every non-atomic subexpression to a fresh name
in a statement slot — ANF / three-address form — removes the need for it.

**Why this is backend-independent.** The pass is total and syntactic: no target
fact reaches it, and none has to. It does not predict where C++ will want a
temporary; it makes the situation unreachable, because every operand the emitter
sees is already a `Var`. That is the shape of every normalization here —
`ZipElim` does not know how a backend materializes a zip, it removes the zip;
`FreeVarElim` does not know how one represents a closure, it removes the free
variables. The pass shrinks the input language the backend must handle.

Backend-*independent* is not backend-*neutral*: FPCore is an expression language,
where this would emit a `let*` chain for nothing. Like `ZipElim` and `RoundElim`,
it is a normalization the C++ backend opts into.

### Where a temporary goes

Not "the nearest enclosing statement" — that is wrong for a `while` condition
(evaluated *n* times where the slot before the loop runs once) and for a ternary
arm (conditional where the slot is not). The rule is that a temporary for `e`
goes in a statement slot executed **exactly as often, and under exactly the same
condition, as `e` itself**. That partitions the expression positions:

| class | positions | placement |
|---|---|---|
| 1. the slot is the statement | `Assign.expr`, `ReturnStmt.expr`, `AssertStmt.test`, `IndexedAssign` index and value, `ForStmt.iterable`, `ContextStmt`, `EffectStmt`, `IfStmt.cond`, `If1Stmt.cond` | before that statement |
| 2. the slot is a body | anything inside an `if` / `else` / `while` / `for` body | class 1 against its own statement — recursion, no special case |
| 3. no correct slot exists | `WhileStmt.cond`, `IfExpr.ift` / `.iff`, `And` / `Or` non-first operands, `ListComp.elt` and dependent clause iterables | — |

Total on 1 and 2, refusing on 3 for v1 — the shape `CompToLoop.refusals()` and
`iter_elim`'s statement-path / expression-path split already have. The emitter has
met this too: `_is_pure_cond` *is* "does this expression need statements before
it", and `_flattenable` uses it to decline `else if` when the answer is yes. That
predicate is the one to lift.

### Lowering class 3, and why the refusal comes first

Class 3 is lowerable — the lowering *creates* the slot that was missing, so the
positions cease to exist rather than needing a residue:

| position | lowering |
|---|---|
| `WhileStmt.cond` | `c = <cond>; while c: <body>; c = <cond>` |
| `IfExpr.ift` / `.iff` | `IfStmt` plus a phi |
| `And` / `Or` tails | the same, one `IfStmt` per short circuit |
| `ListComp.elt`, dependent clause iterables | `CompToLoop` |

The `while` form is the direct reading of the loop: evaluate the condition, run
the body, evaluate it again. `c` is then an ordinary loop-carried variable the phi
machinery already handles, and the temporaries for `<cond>` sit in a slot that runs
exactly as often as FPy evaluates it — once before the loop, once per iteration.
(FPy has no `break`, so the `while True: … if not c: break` shape is unavailable
anyway; it would not be preferable if it were.) The cost is that the condition's
emitted code appears twice, so a `max(xs)` in a `while` condition emits its
reduction loop twice.

The lowerings compose: taking `IfExpr` first *gives* a comprehension in a ternary
arm the statement slot `CompToLoop` says it lacks.

**But lowering is not the fix.** Three positions in class 3 are live miscompiles
today, because `_emit_guarded_block` and `_visit_if_expr` interpolate a
statement-emitting visitor into an f-string, so whatever it emitted lands *before*
the construct. Verified by running the binaries: a `while` condition needing a
temporary hangs where the interpreter returns (the loop tests a value computed
once); a ternary arm and an `and` / `or` tail each run the untaken path's
assertion and abort where the interpreter returns. All three are in *defined* FPy
semantics — loop termination, an untaken branch — so they violate the criterion in
[backend-cpp.md](backend-cpp.md). They belong in that document's open issues;
noted here because they decide the order of this work.

Lowering makes them *unreachable*, not impossible: the emitter still contains the
shape, guarded only by the fact that an earlier pass ran. So a future change
emitting a statement from a new expression path reopens all three with no test
failing, and `optimize=False` keeps them regardless.

Hence: **refuse first, lower second.** Extending `_is_pure_cond` to gate `while`
conditions, ternary arms and boolean tails is small, testable today, and makes the
defect impossible rather than unreachable — the net that keeps the emitter honest
once ANF is upstream of it. Then the lowering makes the refusal unreachable for
real programs.

One cost to size before the refusal lands: it rejects programs that compile today,
including ones that appear to work. Measure how many corpus functions it newly
rejects — that is the visible price of trading a miscompile for an error.

### What it actually buys

The 47 temporary sites split by *what they name*, and only one half is this
pass's:

- **`_bind_operand`, 23 sites, names an FPy operand** — a C++ spelling mentions it
  twice, or it must be evaluated once (`_emit_ieee_min_max`, `_emit_sum`,
  `_ldexp_call`, `_convert_storage`, `_adapt_arg`, the round guards). ANF removes
  essentially all of them, along with most of `_emit_at`'s build-at-`want`
  machinery and `_storage_or_none`.
- **`_fresh_temp`, 24 sites, names C++ scaffolding no FPy expression corresponds
  to** — loop indices (`_emit_range` ×3, `_rebuild_list`, `_open_fill_loop` ×2,
  `_open_comp_loop` ×2), output buffers, accumulators (`_emit_amin_amax` ×2,
  `_emit_any_all`), the saved `fenv` mode, a rounded value held for its
  assertion. ANF removes **none** of them.

So **§1 retires `_bind_operand`; §4 and §7 retire `_fresh_temp`**, because those
temporaries become ordinary `Assign` statements once the lowering that invented
them moves upstream. Complementary, and neither substitutes for the other.

Two further payoffs: storage inference gets a def for every value rather than only
for declared names, which is what §2's `store(e)` rule is stated over; and this is
the statement slot `comp_to_loop.py` and `round_insert.py` say they lack, named in
both module docstrings as the reason a comprehension is unschedulable.

### Blocking questions

**Scalars only, or aggregates too?** Naming a list-valued expression creates a def
that `same_object_defs` unions with nothing — a plain rebind is not a coalescing
edge — so `t = xs` gets its own class, possibly narrower than `xs`'s, and a
narrower class for a shared list is `_refuse_unsharing`. Worse, `_shares_storage`
counts that temporary as another place holding the region unless
`binds_by_reference` discounts it, so more temporaries mean more apparent sharing,
more boxing, and under `STRICT` more *compile failures* rather than slower code.
Scalars only for v1 is the safe scope, and it still captures every
`_bind_operand` in the round guards and op dispatch.

**What is `optimize=False` for?** Running unconditionally breaks its documented
"compiles the surface AST verbatim"; running only under `optimize=True` means the
emitter keeps both paths and none of the deletions above happen. If the flag is a
debugging aid a third mode is cheap; if something relies on it as a guarantee,
this pass cannot be unconditional. The answer decides the shape of the work — and
note it has a correctness edge, since a gated pass leaves the class-3 miscompiles
above live under `optimize=False`.

### Design questions, with a leaning

- **Proper subexpressions only.** Naming a statement's top-level expression as
  well turns every `elif` ladder into nested `if`s, since `_flattenable` declines
  whenever a condition needs statements and `_PURE_COND_OPS` is exactly
  `Var` / literals / `Compare` / `Not` / `And` / `Or` / the FP predicates. Naming
  only proper subexpressions leaves `if x < y:` alone; the residual regression is
  then just an `elif` whose condition contains a call or arithmetic.
- **Literals stay atomic** for v1 — but this is the one place where *more* naming
  buys correctness rather than costing it. `_call_arg` and `_literal_cpp_type`
  exist because a literal's token type differs from its storage, and a declared
  temporary would retire that class by construction. Revisit after v1.
- **Never hoist across a `ContextStmt`.** Crossing a `with` changes the active
  context and so the rounding. `round_insert.py` met this and binds under the
  *original* scope; treating it as class 3 makes it structurally impossible for
  the pass to change a value.
- **Runs last, inside `specialize()`.** ANF destroys the shapes `ReduceFusion`,
  `ZipElim` and `EnumerateElim` match on: `any([...])` becomes `t = [...]; any(t)`
  and can never fuse. And it belongs in `specialize()` rather than `_emit`, or
  `signature()` and `compile_module()` analyze different ASTs — which
  `_analyze_all`'s docstring records as having been a real ABI bug once.
- **An access path over a `Var` is atomic.** The emitter folds an `Fst` / `Snd`
  chain into one `std::get`; naming each level breaks the fold.
  `iter_elim.is_access_path` is already the predicate.

Settled, so not to relitigate: names come from `Gensym(reserved=def_use.names())`,
the idiom in nine transforms. And validation is this pass's one luxury — the
interpreter runs both forms, so it is checkable without the backend at all.
Nothing else in this roadmap is.

Sequencing note: §2's line counts are taken against today's def set, which this
pass changes. Design §2's lattice interface before ANF lands, or expect to
re-measure.

The phased implementation plan is [anf.md](anf.md).

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
