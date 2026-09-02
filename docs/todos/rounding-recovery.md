# Plan: recover from an unsupported rounding instead of refusing

Scopes the ladder sketched in *Recovering from an unsupported rounding* in
[backend-cpp.md](backend-cpp.md). Two halves, in this order:

1. **arithmetic** under a context the op table has no signature for becomes a
   deliberate double rounding — compute at a native intermediate, re-round to
   the target;
2. the **residual rounding** goes down the native-lowering sequence of
   [native-lowering-roadmap.md](native-lowering-roadmap.md).

Every operator this needs exists at the `FuncDef` level and takes a
`where` cursor. This is wiring and detection, not new rewriting.

## The flag

`CppCompiler(unfold=UnfoldMode.NONE)` — graduated opt-in, alongside
`unsafe_cast_int` and `unbox`, and an enum for the same reason `unbox` is one:

| mode | what it rewrites |
|---|---|
| `NONE` | nothing; the refusal names the operator that would fix it |
| `ROUNDINGS` | a rounding the op table cannot spell → integer arithmetic |
| `DOUBLE_ROUND` | also arithmetic under such a context → native + a rounding |

The middle level is a genuinely smaller claim, which is why it is separate:
lowering a rounding rewrites one operation into another spelling of the same
function, while rewriting arithmetic rounds *twice* and rests on the
correct-double-rounding rules holding.

`NONE` by default: the refusal is a checker's answer, and turning the compiler
into a rewriter has to be asked for. With it off the refusals are exactly what
they were, so the corpus does not move.

Placed in `specialize()` after `RoundElim` and before `Simplify`.

### There is no round-to-odd level

An RTO intermediate is what `derive_intermediate` returns and what Figure 8's
rules cover for arbitrary reals — so it is accepted exactly where every native
candidate is refused: `exp` under any target, `div` and `sqrt` under a directed
or saturating one. Measured, and it does not close.

The intermediate it gives is `MPFloatContext(pmax=13, rm=RTO)`, and no native
mode is RTO — so the split leaves the operation under a context the op table
still cannot spell. The site does not go away, it changes address. Splitting
*that* one needs either RTO over RTO, which is admissible and widens by a bit
each time, or the exactness rule, whose exact result `div`, `sqrt` and `exp` do
not have. So the regress has no bottom for precisely the operations the level
would add.

It becomes reachable if an unbounded RTO operation becomes emittable — the
bit-reinterpreting soft-float direction in the native-lowering roadmap's open
questions.

## `fpy2/backend/cpp/unfold_round.py`

One new backend-local module, holding **detection and dispatch only**:

- `UnfoldKind` — `ARITH`, `FLOAT_ROUND`, `FIXED_ROUND`. The kind *is* the row of
  the ladder table below.
- `UnfoldSite` — a cursor, its kind, and the context that made it one.
- `sites(fd) -> list[UnfoldSite]` (Phase 1) — runs `DefineUse` + `ContextUse`
  over the specialized `FuncDef` and classifies each program point.
- `unfold(fd) -> FuncDef` (Phases 2-4) — walks the sites and applies each kind's
  operator sequence.

Nothing else lands there. The rewrites stay in `fpy2/transform/`, the soundness
gate stays inside `SplitRound`, and the refusal messages stay in the emitter,
which is what runs when the flag is off.

It belongs in the backend because the question is a backend question: *which
contexts does the op table dispatch on*, answered by `target.is_native_ctx`. The
operators it drives are backend-independent, and none of them changes. Not in
`target.py`, which describes the target rather than asking anything about a
program; not in the emitter, which runs after every analysis and so too late to
rewrite.

`compiler.py` calls it as one `module.map` in `specialize()`, after `RoundElim`
and before `Simplify`, under the flag.

## Phase 1 — detection *(done)*

`sites` reports

- an **arithmetic** op whose active context is neither native nor `REAL` —
  `ARITH`. `REAL` is the one non-native context the table reaches, by widening
  to an op that gives the exact result and rounds to itself;
- a **rounding** (`Round` / `Cast`) whose target context fails `is_native_ctx` —
  `FLOAT_ROUND`, or `FIXED_ROUND` where that context is fixed-point and
  `_emit_integral_round` will not take it as it stands.

Three things settled by writing it:

**The op table's keys are the dispatch predicate.** `make_op_table()` keys
`unary` / `binary` / `ternary` by node type, so `type(e) in table.binary` is
exactly "the op table is what emits this" — no list of node classes to
maintain, and `Min` / `Max` / `Len` fall out on their own.

**Detection must not run `FormatInfer`.** `RoundingScopes` answers the same
context question but also infers formats, and format inference is one of the
things the rewrite exists to make succeed — it refuses these very programs. So
`sites` runs `DefineUse` + `ContextUse` and nothing else.

**It answers the rounding question only.** A program with no sites can still be
refused, and usually is: an `MPFixedContext(-1)` over a real argument is
lowerable as it stands and still has no storage. Whether the ladder helps and
whether the program compiles are separate questions.

Verified against the `test_lowered_roundtrip` sequence, whose stages walk the
ladder table exactly: `monomorphize` and `unfold_special` leave `FLOAT_ROUND`
(an `IEEEContext`), `unfold_overflow` leaves it there (an unbounded
`MPSFloatContext` is no more native), `float_to_fixed` trades it for
`FIXED_ROUND` at `nmin = -25`, and `rescale_fixed` clears it.

## Phase 2 — arithmetic to a double rounding *(done)*

`unfold` walks the `ARITH` sites and, per site, offers `SplitRound` a native
intermediate — the *narrowest* first, since the intermediate's width becomes
the arithmetic's storage and a wider one is never less admissible.

**Round-to-nearest only.** It is the mode the per-operation rules take, and the
exactness rule takes any mode, so both rules that matter here are reached. It is
also the mode the machine is already in: an intermediate rounding some other way
would put an `fesetround` boundary around arithmetic whose whole purpose is to be
the native one. Candidates are filtered through `is_native_ctx`, so a mode the
backend stops dispatching on stops being offered.

Not `derive_intermediate`: it returns an *unbounded* context, deliberately — an
unbounded intermediate cannot overflow, so the composition agrees at the ends of
the range. But unbounded is not native, so the arithmetic would still refuse.
The compiler wants the other trade and has to name its candidate. Soundness
stays with `SplitRound`, which holds all three rules; `unfold` only proposes,
and a refusal is an ordinary outcome.

**Operand formats are the precondition, and it is met.** The per-operation
rules hold only for operands the *target* represents, so an argument carrying no
context of its own refuses every candidate — measured, and pinned. `Specialize`
pins them in the compiler's pipeline, which answers the question this plan
opened with.

What splits, over `+ - * /`, `sqrt` and `exp` across nine targets:

| target | splits | at |
|---|---|---|
| nearest — `FP16`, `RNA`, the MX formats | `+ - * /`, `sqrt` | `FP32` |
| directed — `RTZ`, `RTP`, `RTO` | `+` `-` by exactness | `FP64` |
| directed | `*` by exactness | `FP32` |
| saturating `RNE` | `+ - *` | `FP32` |

The gaps are the shape of the rules, not missing work: `exp` has no rule
anywhere, and a directed or saturating target reaches none for `/` or `sqrt`, so
those keep their refusal while an add beside them is taken. A directed target
gets `+` at `FP64` rather than `FP32` because the route is exactness and the
exact sum of two `FP16` values wants more than 24 bits.

Termination is the `ARITH` count, which each split strictly lowers: the
operation lands under a native context and the rounding it gains is to the
target, a rounding site rather than an arithmetic one.

**Not wired into the compiler yet, on purpose.** With only this row done the
refusal moves rather than lifting — from "no matching signature for `Add`" to
"rounding under `FP16` has no C++ analogue" — and nothing compiles that did not
before. The flag goes in with Phase 3.

## Phase 3 — the residual rounding *(done)*

`UnfoldSpecial → UnfoldOverflow(early_check=True) → FloatToFixed →
RescaleFixed`, in the native-lowering roadmap's own order, and **one pass, not a
fixpoint**. The plan expected two rows keyed to what is unsupported; they turn
out to be the same call, since each step selects its own candidates and declines
the rest. A fixed-point site simply gets no work from `UnfoldSpecial` or
`FloatToFixed`.

**Isolation is the one piece of real work, and it is mandatory rather than
incidental.** Every pass of the ladder takes a *rounding block* — an
underscore-bound `with` whose every statement rounds a variable — and a
specialized function has none: `Specialize` folds a block whose context is the
function's own into the annotation, which is where *all* of these end up. So
`FloatToFixed` reports neither a site nor a refusal, and the ladder silently
does nothing. `_isolate` puts the shape back, one statement at a time, matching
`rounding_block`'s own per-statement condition; a rounding of anything but a
variable keeps its refusal, and a rounding of a literal never gets there because
the emitter folds those.

Isolating *only the sites* is what makes running the ladder over the whole
program safe: a rounding the emitter already spells is not a block, so no pass
considers it. That is what replaced the per-site cursor forwarding the plan
called for.

**The flag.** Wired in `specialize()`
after `RoundElim` — which removes roundings this would otherwise lower — and
re-normalized with `_to_statement_form` after, since the lowering emits `with`
blocks and branches of its own. `Simplify` follows from the pipeline. The three
refusals it lifts now name it.

Bit-exact against the interpreter: `test_lowered_roundtrip` drives all fourteen
targets from both an `FP32` and an `FP64` source through the flag rather than by
hand, and `TestArithRoundtrip` adds the composition the double-rounding rules
exist for — `sqrt`, `/` and `+` under `FP16`, `RNA` and `E4M3`, from a source in
the target's own format, since a wider argument refuses the split.

Three crashes reproduce well before this work and would surface as crashes
rather than refusals (gap 2 of the native-lowering roadmap): `UnfoldOverflow`
writes a `nan_value` / `inf_value` substitute as a numeric literal, so
`MPFixedContext(-4, inf_value=7)` raises; an integer-typed source crashes format
inference; `fp.round_at(x, n)` crashes type inference. None is reachable from the
tested targets.

## Phase 4 — the limits *(done)*

**Iterating is not what was needed.** Swept over thirteen targets — the IEEE
formats, `bfloat16`, the MX family, a `NEG_ZERO` substitute, saturating,
stochastic, bounded and unbounded fixed-point, and a format wider than the
storage ladder — one pass clears every site that is clearable at all. Nothing
found a second round to make, so an outer loop would be untested code.

What the sweep did find is that the rewrite could make a *diagnosis* worse.
Where the ladder rewrites a site and the result still fails, it fails further
along: a rounding the emitter could name becomes a temporary storage selection
cannot place, and `cannot pick storage for _t1` replaces `needs its digits at
position zero`. So `compile_module` asks the unrewritten program for a second
opinion and reports that instead. **The flag never costs a diagnosis** — the
property worth having, and cheap, since the retry only runs on failure.

A stochastic context is no longer a site: no step of the ladder draws random
bits, and the emitter now says exactly that rather than falling through to
advice naming the flag the caller already passed.

### What the flag reaches

| target | `ROUNDINGS` | `DOUBLE_ROUND` |
|---|---|---|
| `FP16`, `bfloat16`, `IEEEContext(4, 8)`, saturating, MX, substitute specials | rounding ✓ | arithmetic ✓ |
| wider than the storage ladder | ✓ (nothing to do) | ✓ |
| stochastic | — no lowering draws random bits | — |
| fixed-point, unbounded | — no bound to state, so `UnfoldOverflow` declines | — |
| fixed-point, bounded, away from position zero | — `RescaleFixed`'s shift needs storage the context has not got | — |
| fixed-point, bounded, `WRAP` | — `UnfoldOverflow` declines the rule | — |

The float row is the one that works, and it works everywhere. **The
fixed-point row does not help a user-written fixed target**, which the plan did
not anticipate: it is reachable only from inside the flow, where
`FloatToFixed` produces a bounded context at a known position over an operand
`UnfoldSpecial` has already classified, and there it is already handled. A
hand-written one fails the storage question instead, which is a separate gap.

## Open

**What `DOUBLE_ROUND` costs.** Unmeasured. `FloatToFixed` states a value-class
branch per site, so a program with many roundings gets a lot of emitted code —
an `FP16` add is 47 lines and a two-operation polynomial 135. Nothing on the
corpus exercises it, the flag being off there.

**A user-written fixed-point target.** Both reasons it fails are outside this
plan: `UnfoldOverflow` states no rule for `WRAP` or for an unbounded format, and
`RescaleFixed`'s shift lands under a context the storage ladder has no entry
for. The second is the same gap as `MPFixedContext(-1)` refusing with no site at
all.

**A wider source for arithmetic.** The per-operation rules hold for operands the
*target* represents, so `x + y` under `FP16` needs `FP16` arguments. A program
holding `FP32` values and rounding to `FP16` per operation is the natural shape
and reaches no rule; what it would need is a rule quantified over the operand
format, not the target's.

**`exp` and friends.** No double-rounding rule covers a transcendental, so those
keep their refusal under a nearest target. The remaining route is the exactness
one, which needs a correctly-rounded implementation to compare against.
