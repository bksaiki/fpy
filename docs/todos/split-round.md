# `split_round`: one rounding into two

Gap 3 of [rounding-axes.md](rounding-axes.md), plus the plumbing (gap 2) and the
predicate `merge_round` will reuse. Expands a correctly-rounded operation into
two roundings that compose to the same answer — the forward direction of the
double-rounding axis, and the operator §5 of *When Double Rounding is Correct*
exists to justify.

## The rewrite

An assignment does not round in FPy, so the second rounding has to be an
explicit `fp.round`, in the outer block, where it picks up `rm1`:

```python
# before                          # after, via the derived RTO intermediate
with fp.FP32:                     with fp.FP32:
    t = x * y                         with MPBFloatContext(pmax=26, …, rm=RTO):
                                          _t = x * y
                                      t = fp.round(_t)
```

**Validated before planning.** Hand-writing that pair and comparing against the
unsplit program: **4000/4000 random inputs agree**, including subnormal
underflow to `+0`, overflow to `+inf`, `-0` (`0.0 * -1.0`), and the identity
`1.0 * 1.0`.

## Follow the Lean proofs, not this page's table

`rounding-axes.md`'s table has been corrected to match
[Mpfx/DoubleRounding.lean](https://github.com/bksaiki/mpfx-lean/blob/main/Mpfx/DoubleRounding.lean)
the proofs, which are the source of truth. Repeated here as the
implementation's reference:

| final `rm1` | intermediate `rm2` | premise | Lean theorem |
|---|---|---|---|
| RTZ | RTZ | `F1 ⊆ F2` | `rndRTZ_RTZ` |
| RAZ | RAZ | `F1 ⊆ F2` | `rndRAZ_RAZ` |
| RTP | RTP | `F1 ⊆ F2` | `rndRTP_RTP` |
| RTN | RTN | `F1 ⊆ F2` | `rndRTN_RTN` |
| RTO | RTO | `F1 ⊆ F2` **and** `p2 >= 2` | `rndRTO_RTO` |
| RTZ | RTO | `𝒜(p1+1, exp1-1, next_{F1}(b1)) ⊆ F2` | `rndRTO_RTZ` |
| RAZ | RTO | `𝒜(p1+1, exp1-1, next_{F1}(b1)) ⊆ F2` | `rndRTO_RAZ` |
| RNE **or RNA** | RTO | `𝒜(p1+2, exp1-2, next_{extend(F1,1)}(b1)) ⊆ F2` | `rndRTO_RN` |

Eight rules over nine mode pairs. The three corrections:

1. **No `next(b1)` bump on the same-mode rules.** `rndRTZ_RTZ` and
   `rndRTO_RTO` both take plain `hsub : F₁.toFormat ⊆ F₂.toFormat`, exactly as
   RAZ-RAZ does. The roadmap had a bump on RTZ-RTZ *and* on RTO-RTO, and
   presented the RTZ/RAZ asymmetry as a detail that "looks like a typo and is
   not"; against the proofs there is no asymmetry.
2. **RTP-RTP and RTN-RTN are sound** on plain containment. The roadmap has them
   declining pending a `value_class` sign analysis, but the sign branching is
   *internal to the proofs* (`rndRTP_RTP` "branches to RAZ for `x > 0` or RTZ
   for `x <= 0` internally"), so the pair is sound unconditionally and no
   analysis is needed.
3. **RNA is admissible as a final mode.** `rndRTO_RN` is parametric in
   `tb : TieBreak` and covers `.toEven` and `.awayZero` alike. RNA declines only
   as an *intermediate*.

Corroborated rather than corrected: `extend k` is precision `+k` and exponent
`-k` — exactly `with_prec_offset(k).with_exp_offset(-k)`; `boundAfterNext` is
`F.next(b)` *in that format's grid* — which is why the wrapper takes no
precision argument; and `⊆` checks precision, exponent and bound, which is what
`AbstractFormat.contained_in` checks. `p2 >= 2` is explicit only for RTO-RTO;
for the RTO-to-directed rules the proofs *derive* it from the bound-aware
containment.

Still declining: **RNE-RNE** (no rule; Table 2's last row, and the pairing every
`fp.FP*` context falls into by default), **RNA or RTE as an intermediate**, and
**stochastic** either side.

## What already exists

| Needed | State |
|---|---|
| `RoundingMode` covering the table | ✅ `RNE, RNA, RTP, RTN, RTZ, RAZ, RTO, RTE` |
| `rm` on the base `Context` | ❌ absent — phase 1 |
| `next(b)` in a format's grid | ⚠️ `RealFloat.next_up(p=…, n=…)` exists; no `AbstractFormat` wrapper — phase 1 |
| `p+k, exp-k` | ✅ `with_prec_offset` / `with_exp_offset` |
| The `⊆` premise | ✅ `AbstractFormat.contained_in` |
| Building the intermediate context | ✅ `derived.format()` is an `MPBFloatFormat` whose fields are `MPBFloatContext`'s constructor |

## The work

### 1. Plumbing (roadmap gap 2)

- **`Context.rounding_mode() -> RoundingMode | None`** in
  `fpy2/number/context/context.py`, beside `format()` / `round_params()`.
  Figure 8 dispatches on the *pair* of modes and nothing exposes one today.
  Abstract, implemented across the 13 in-tree classes — mechanical, since each
  already carries `.rm` (delegating ones forward to `_mpb_ctx`) and
  `RealContext` returns `None`.
- **`AbstractFormat.next_bound()`** in `format_infer/format.py`, bumping both
  bounds to the next value *in this format's own grid* —
  `bound.next_up(p=self.prec, n=self.exp)`.

  **No precision argument.** The proofs call `boundAfterNext` on a format and
  read the grid off the receiver: `F₁.boundAfterNext` for RTO-RTZ / RTO-RAZ,
  `(F₁.extend 1).boundAfterNext` for RTO-RN.  So the caller extends first and
  asks second — `f1.next_bound()` versus
  `f1.with_prec_offset(1).with_exp_offset(-1).next_bound()` — and cannot take
  `next` at a precision inconsistent with the format the bound is installed on,
  which is the mistake the RTO-RN row invites.

  A `prec=` parameter would also have been *inert*, which is what makes this
  worth stating: `next_up` normalizes to *at most* `p` bits, so raising `p`
  alone cannot refine the grid.  On FP32's bound, `next_up()` and
  `next_up(p=25)` are both `3.40282367e+38`; only `next_up(p=25, n=-150)` gives
  `3.40282357e+38`, the true next value in the extended grid.  Passing precision
  without the exponent would have silently returned F1's answer for the one rule
  that needs the finer one.

Tests: every context reports its mode or `None`; `f1.next_bound()` and
`f1.with_prec_offset(1).with_exp_offset(-1).next_bound()` differ (`…367` vs
`…357` for FP32), which is exactly the distinction RTO-RN turns on.

### 2. `double_round_ok` and the derivation

`double_round_ok(f1, rm1, f2, rm2)` beside `round_is_identity` in
`format_infer/analysis.py` — same shape of decision procedure, same place a
reader looks.

Plus **`derive_intermediate(target: Context) -> Context`**, which is how a
caller *obtains* a `ctx` rather than hand-computing one: `k = 2` for a
round-to-nearest target, `1` for a directed one, then
`with_prec_offset(k).with_exp_offset(-k)`, the `next` bump, and an
`MPBFloatContext` built from `.format()` with `rm=RTO`. Since `ctx` is required,
this is a helper and not a `None` branch in the operator — which also settles
what to derive: RTO, because that is §5.3's modular-library recipe. A caller who
wants a same-mode intermediate (plain containment, so it succeeds more often)
passes their own `ctx` and the predicate checks it.

**The `-0` trap.** `with_exp_offset` drops `-0` from the special set:

```
F1                            S={+inf,-inf,nan,-0}
F1.with_prec_offset(2).with_exp_offset(-2)   S={+inf,-inf,nan}
F1.specials_contained_in(derived)  ->  False     # spurious decline
derived.contained_in(F2)           ->  True      # F2 regains -0 via .format()
```

So the premise's direction matters, and a derived intermediate must be
round-tripped through `.format()` before being used as `F2`. The runtime split
preserves `-0` correctly, so this is an artifact of the 𝒜 arithmetic rather
than a semantic property — but `specials_contained_in` exists because a lost
`-0` was once a real bug, so it gets a test rather than a comment.

Tests, with the proofs as oracle: the eight admitted rules hold; the unsound
pairings decline, RNE-RNE by name; RTO-RTO's `p2 >= 2` and the absence of a
bump on RTZ-RTZ or RTO-RTO each get a test, since both were corrections to
this repo's written table; and a property test that
`double_round_ok` accepts `derive_intermediate(target)` for every admitted
target mode.

### 3. The transform

`SplitRound` in `fpy2/transform/split_round.py` — a `SiteRewriter` with
`_expr_sited = True`, mirroring `RoundInsert`'s hoist and reusing `operands` /
`rebuild` from `transform/utils.py`.

- **Sites** are the complement of `insert_round`'s: an arithmetic operation
  whose enclosing scope is a *concrete* context with a mode the table admits.
  `insert_round` skips an operation that already has a format; this one requires
  one.
- **Not sites:** explicit `Round` / `Cast` nodes. Splitting a rounding is
  `merge_round`'s inverse, and admitting them makes a second application grow
  the tree twice as fast for nothing.
- **`ctx` is the intermediate** — `F2` together with `rm2`, which is exactly
  what a `Context` carries. Required, mirroring
  `insert_round(func, ctx, where=None)`, where `ctx` is likewise the context the
  rewrite installs. `F1` and `rm1` are read from the program.
- **Refusals**: a REAL or symbolic scope (nothing to split — that direction is
  `insert_round`'s), stochastic either side, a mode pair off the table, a `ctx`
  failing the premise, and the positions with no statement slot. That last set
  is now only the `while` condition, for soundness, since the `_visit_block` fix
  landed with the dead-context work.
- **Termination**: one pass over `where=None` terminates; *repeated* application
  re-splits the inner operation, which is by design for a scheduling operator
  and belongs beside `insert_round`'s divergence note.

Tests: emitted shape per admitted mode; the 4000-input equivalence sweep;
refusals by reason; the `where` contract; and that `fp.round` in the outer block
really applies `rm1` — the assumption the whole rewrite rests on.

### 4. The strategy

`split_round(func, ctx, where=None)` in `fpy2/strategies/`, the same shape as
`insert_round`. Registered in
`strategies/__init__.py`, `_SITES` / `_REFUSALS` in `sites.py`, and
`docs/source/strategies.rst`, with `ACTS` and `REFUSES` rows in
`tests/unit/strategies/test_where_contract.py`.

### 5. Roadmap and gate

Mark gaps 2 and 3 done in `rounding-axes.md` (its Figure 8 table is already
corrected), strike widening from `Order of work` (semantically
meaningless in FPy: a widening cast is a verified identity, so `elim_round`
removes it again and nothing in the semantics distinguishes the two programs —
it belongs to native lowering), and note that `merge_round` now has its
predicate waiting.

Then `pytest tests/unit` single-process, `python -m tests.infra`,
`python -m tests.infra.backend.cpp`, `mypy fpy2`, `ruff check fpy2`.

## Risks

- **The table corrections.** They come from reading the Lean statements, not
  from running Lean. Three of them contradict this repo's own written table, so
  they want the author's eye before they are encoded in tests.
- **Fixed-point intermediates.** Expected admissible — §8's MPFX accumulator is
  `𝒜(inf, -32, 2^96)` and the premises are 𝒜 containment checks indifferent to
  the format family — but this overlaps `float_to_fixed` / `rescale_fixed`, so
  test one rather than assume, and decline with a clear reason if it fails.
- **`RTE`** is in FPy's `RoundingMode` but in none of the theorems; it declines.

## Out of scope

- **`merge_round`** (gap 4). It reuses `double_round_ok` unchanged, but its site
  detection is unsurveyed — `Round` over `Round`, `Cast` over arithmetic,
  `RoundAt` over a rounded operand, nested `with` blocks — and that survey wants
  a verified predicate to survey against.
- **RTP / RTN via sign analysis.** Moot: the proofs admit those pairs outright.
- **The `finitize` search** over an environment's format list — recipes, not
  operators.
