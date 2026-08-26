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
`Escape` all live in `fpy2/analysis/`. Four modules do not: `types.py`,
`ops.py`/`target.py`, `storage.py`/`storage_infer.py`, `unbox.py`.

The leak is the emitter. Grouping `emitter.py`'s methods by what they *do*:

| group | lines | spelling? |
|---|---|---|
| library-op lowering (`sum`/`amin`/`min`/`any`/`zip`/`range`/`size`/compare chains) | 515 | no — loop construction |
| round/cast lowering (`_emit_integral_round`, `_bound_test`, `_undefined_guard`, …) | 482 | no — soft-float lowering |
| control-flow printing | 394 | yes |
| storage reconciliation (`_emit_at`, `_convert_storage`, `_rebuild_list`, `_adapt_*`) | 388 | no — conversion insertion |
| fenv / context boundaries | 247 | mostly |
| declarations and bindings | 221 | yes |
| peepholes (`ldexp`, pow2, literal folding) | 191 | no — algebraic rewrites |
| op dispatch (`_dispatch`, `_try_widen`, `_maybe_cast`) | 174 | table-driven already |
| list / array spelling | 172 | yes |
| comprehension lowering | 125 | no |

About 1900 lines are decisions and about 790 are printing. That gap is this
roadmap.

Which sections close it: **§4** and **§7** shrink the emitter. **§2** and **§3**
move code out of the backend without shrinking it. **§1** shrank it by nothing —
every `_bind_operand` and `_emit_at` path that survives statement form is on an
*aggregate* operand, so the deletions wait on naming aggregates.

## 1. Statement form — done

`fpy2/transform/anf.py`, exposed as `fpy2.strategies.to_anf`, run unconditionally
as the last step of `CppCompiler.specialize()`.

The pass takes no target fact. It does not predict where C++ will want a
temporary; it makes the situation unreachable, because every operand the emitter
sees is already a `Var` — the shape of every normalization here, as `ZipElim`
removes the zip rather than knowing how a backend materializes one.

A temporary goes in a statement slot executed *exactly as often, and under
exactly the same condition, as the expression it names* — not the nearest
enclosing statement, which is wrong for a `while` condition and a ternary arm.
Positions failing that test are sealed; two get a lowering that *creates* the
slot (rotation for `while`, `IfStmt` for a ternary), and `ANF.refusals` reports
the rest. Over the 230-function corpus the residue is entirely comprehensions,
with zero in the three positions the emitter cannot slot.

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

## 2. Storage inference

Storage inference is not format inference. Format inference maps a real-valued
expression to the smallest format bounding its value; storage inference maps it
to a member of a distinguished, finite set of formats that *contains* that bound:

```
e : real   fmt(e) = F   F <= S
------------------------------
          store(e) = S
```

Several `S` satisfy this, so the content is which one is chosen and what
constrains the choice beyond containment.

`storage_infer.py` is 312 lines of union-find, class naming and declaration-site
placement, touching C++ in five places, all through `CppType` and
`aggregate_storage`. Parameterize it on a lattice — `of_bound(FormatBound) -> S`,
`join([S]) -> S`, `is_aggregate(S)` — move it to `fpy2/analysis/`, and leave
`_LADDER` in the backend as one instance.

The assignment and phi rules:

- **Assignment is not a promotion site.** A class is a union-find over
  `reaching_defs.same_object_defs`, which unions on phi edges and
  `IndexedAssign` only. A plain rebind `x = e` gets a *fresh* variable with its
  own, possibly narrower, storage. `store` is per runtime object, not per name,
  so sequential redefinition never promotes.
- **A phi is the only promotion site, and the rule is structural.**
  `aggregate_storage` takes the supremum element-wise through lists and tuples,
  dropping bottom bounds (a fresh `empty(...)` holds no value and constrains
  nothing).
- **Promoting compound data is constrained by aliasing.** A container's promotion
  is a *rebuild* — a new buffer, hence a different object — so it is legal only
  where nothing can observe the identity. `_convert_storage` performs it for a
  value; `_refuse_unsharing` rejects it for a shared list.

So `F <= S` is necessary and not sufficient: among the `S` satisfying it, the
admissible ones are constrained by aliasing. A shared aggregate admits no
promotion at all, and the containment rule alone would license a silent
miscompile.

Open: a *partly* bottom bound contributes its empty slots, because the supremum
is taken over storages rather than over bounds. Fixing it means joining in the
format lattice first and choosing storage once.

## 3. Representation inference

`unbox.py` (722 lines) decides, per alias region, whether a list is a shared
handle, a plain value, or a fixed-length value. It splits three ways.

**Backend-independent (about 190 lines).** `_region_sizes` meets
`ArraySizeInfer`'s bounds over the region graph — 42 lines with no `CppType`,
useful to anyone wanting one proven length per region. `_Scan`'s four facts are
the same: a region is stored into, a slot is replaced, a region crosses a call
boundary, a region is returned at depth *d*. `Alias` already owns the region
graph (`referrers`, `escapes`, `transfers_ownership`, `defs_in`, `region_at`,
`region_field`), so they belong there.

**Generic algorithm, backend-supplied predicate (about 70 lines).**
`_shares_storage` asks how many *places* hold a region separately, which is
language-level, but its precision is calibrated to this emitter: it does not use
`alias.is_shared`, it counts only names that get their own storage, and that set
is `binds_by_reference`, shared with `storage_infer` so the two cannot drift.
Discounting a name the emitter then copies is a miscompilation, so the extracted
form takes the binding rule as a parameter.

**Backend-specific (about 300 lines).** The output alphabet: C++ wants
handle / value / fixed array, Rust wants `Rc<RefCell<Vec<T>>>` / `Vec` /
`&mut [T]` / `[T; K]` and a borrow checker that changes what a second place
costs, and a garbage-collected target has nothing to decide. Then
`writes_through` and `may_reference_projection`, which ask whether `const`
reaches through a handle or a value and whether a reference re-reads a slot where
**E-Deref** read it once. Then the verdict-to-type mapping (`_stamp`, `_regions`,
`annotate`), `UnboxMode` and its diagnostics, and `ParamAbi` / `CalleeAbi`.

A third moves, and the payoff is not a smaller emitter — it consults an oracle
either way. It is that `Alias` gains facts other consumers want and the sharing
verdict becomes testable without a C++ string.

The first part is independent of everything else here. The second waits for a
second backend to check the interface against.

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

## 5. Conversion insertion

`backend-cpp.md`'s *One question answered in four places* scopes this as one
predicate per place kind. As a pass it is better: after storage inference, insert
an explicit conversion node wherever an operand's storage differs from its
place's, and refuse where nothing bridges. The emitter then prints `static_cast`
and rebuild loops without deciding anything, and that document's four-column
table retires.

Needs §1: the places must exist as program points before a pass can put a node at
one.

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

1. **Storage inference** (§2) — smallest change, and the lattice interface §3 and
   §5 reuse.
2. **Conversion insertion** (§5).

**§3's first part** — region sizes and `_Scan`'s facts into `Alias` — is
independent and can go at any point; its second part waits for a second backend.
**§4** is independent, but is a new transform whose lines mostly move rather than
disappear. **§6** needs a language-level scale operation. **§7** waits for the
FPy-level lowerings to become total.

Aggregate naming in ANF is not a numbered section. It blocks §5 and every
`_bind_operand` deletion, and it is the highest-risk item: a name holding a list
is a second place, and `UnboxMode.STRICT` turns that into a refusal.

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
