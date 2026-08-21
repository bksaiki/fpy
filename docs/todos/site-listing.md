# Site listing: what `where` counts

## The contract we want

A strategy applies to `k` sites in a program. Then

- `where=None` rewrites all `k`, and
- `where=j` for `0 <= j < k` rewrites the `j`th.

`k` is `len(sites(strategy, func))`. Six of the ten aimable strategies break
this today.

## What actually happens

`k` counts *structural* candidates. The rounding strategies decide in two
stages — `_candidate` for shape, `_verify` for whether the rewrite applies —
and `sites()` exposes only the first. It is documented (`sites()`: "Listing is
syntactic: a site that appears here may still be declined"), so it is
deliberate, but it is a weaker guarantee than the name suggests.

Audited over three rounding programs, a loop program, a call program, and a
monomorphized sum-of-squares for `insert_round`. `k` = `len(sites(...))`;
**ok** = how many of `range(k)` actually rewrote.

| strategy | program | k | ok | |
|---|---|--:|--:|---|
| `unfold_special` | two_rounds | 2 | 2 | ok |
| `unfold_special` | declining | 2 | 1 | `where=0` declines |
| `unfold_special` | cast_and_round | 2 | 0 | all decline, `None` a no-op |
| `unfold_neg_zero` | two_rounds | 2 | 0 | all decline, `None` a no-op |
| `unfold_neg_zero` | declining | 2 | 0 | all decline, `None` a no-op |
| `unfold_neg_zero` | cast_and_round | 1 | 0 | all decline, `None` a no-op |
| `unfold_overflow` | two_rounds | 2 | 2 | ok |
| `unfold_overflow` | declining | 2 | 1 | `where=0` declines |
| `float_to_fixed` | two_rounds | 2 | 2 | ok |
| `float_to_fixed` | cast_and_round | 1 | 0 | all decline, `None` a no-op |
| `rescale_fixed` | two_rounds | 2 | 0 | all decline, `None` a no-op |
| `rescale_fixed` | cast_and_round | 2 | 2 | ok |
| `split` / `unroll_for` / `unroll_while` | loops | 1 | 1 | ok |
| `inline` | calls | 2 | 2 | ok |
| `insert_round` | sum_sq | 3 | 2 | `where=0` declines |

Two distinct failures, and the second is the worse one:

1. **`where=j` declines.** `k` overcounts, so an index in `range(k)` can raise.
2. **`where=None` is a no-op while `k > 0`.** Every candidate refused.
   `sites(rescale_fixed, two_rounds)` reports two sites in a program holding no
   fixed-point context at all.

The four strategies that pass — `split`, `unroll_for`, `unroll_while`, `inline`
— pass because their candidates are structural (a `for`, a `while`, a call) and
always verify. They are not doing anything different; they just have nothing to
refuse.

Note that [scheduling-language.md](scheduling-language.md) item 2 asked for the
contract at the top of this page, citing Roly-poly on *enumerating the valid
next choices* — and then described the implementation as exposing
`BlockRewriter._candidate`, which is the structural half. The intent was the
stronger reading; the cheaper half shipped.

## The one case that cannot be made consistent

Nesting, and it has nothing to do with verification. Where one candidate
contains another:

```python
for x in xs:          # site 0
    for y in ys:      # site 1
        ...
```

`where=0` rewrites the outer and `where=1` the inner — each is one edit, so the
indexed half of the contract holds. But `where=None` performs **one** rewrite,
not two: unrolling the outer subsumes the inner, as `_record_at` says outright
("A rewrite of an enclosing statement subsumes anything recorded inside it").
Making `where=None` descend into its own output does not terminate for
`unroll_for`, since unrolling produces more loops.

So "all `k`" and "the `j`th" cannot both range over a set that contains a nested
pair. One of them has to give.

This bites less than it reads. A rounding block cannot contain another —
`rounding_block` requires the body be entirely `x = fp.round(v)` — and
`insert_round` already handles nesting the right way: `abs(x * x)` lists two
sites and `where=None` rewrites both. It is the loop strategies where the
subsumption is real.

## The fix

**Define `sites()` as exactly the set `where=None` rewrites**: verified,
outermost-first, non-overlapping. All three properties then hold by
construction.

### 1. Derive the listing from the rewrite

Today each transform keeps a `sites()` predicate *parallel to* its
`_candidate`/`_verify` pair, and the two can drift — silently misaiming `where`,
which is the worst failure this area has. Add a listing mode to `SiteRewriter`
that walks exactly as the rewrite walks but records cursors instead of
rewriting, and make `sites()` that walk. Afterwards the two cannot differ,
because they are one traversal.

This is the phase that carries the structural benefit; the rest is consequence.

### 2. Thread the strategy's parameters into `sites()`

Verification depends on them: `insert_round` cannot know which operations verify
without its target `ctx`, and `split` needs `factor`. `sites()` already forwards
`**kwargs` to the lister (`inline` uses it for `funcs`), so the plumbing exists.
What changes is that they become *required* for some strategies —
`sites(insert_round, f)` should raise rather than answer wrongly.

### 3. Keep the diagnostic

Today `where=0` on a refusing site says why: "the format is already at digit
position zero", "rounds exactly". Filtering those out of the listing makes them
invisible rather than explained, which loses item 2's own goal that a `where` be
"explicable after". Add a companion — `refusals(strategy, func, ...) ->
list[tuple[Cursor, str]]` — so a refusal stops being *counted* without becoming
undiscoverable.

### 4. Nested sites reach by cursor

With the listing outermost-only, the inner loop above is no longer `where=1`.
`within=` already scopes a listing, so `sites(unroll_for, f, within=<outer
body>)` recovers it. Document that as the way to aim inside a site.

### 5. Docs

The `where` boilerplate is copy-pasted across all ten strategies and says
"whether or not they verify", which becomes false everywhere. `sites()`'s own
docstring promises the opposite of the new behaviour.

### 6. A conformance test

One parametrized test asserting the three properties for every strategy in
`_SITES`, so a new operator cannot reintroduce the divergence. This is the test
that would have caught it.

## What it costs

- Phase 1 touches `SiteRewriter`, which eight passes share.
- Phase 2 changes the public signature of `sites()`, and makes it
  non-uniform across strategies.
- Phase 4 is a capability loss at the index level: a nested site that today has
  an index will need a cursor.
- Listing gets slower — it runs verification, and for the rounding strategies
  that means the analysis stack. `sites()` and `apply` already run it
  independently, so this is a constant factor on an existing cost, not a new
  one.

Set against that: `where` semantics are item 1 of
[scheduling-language.md](scheduling-language.md)'s "one failure contract", and
this is the concrete gap in it.

## Open questions

- **Should `where=j` still be able to name a refusing site?** If not, the
  `TransformDeclined`-from-an-index path disappears and `check_site`'s
  int branch reduces to a range check. If so, indices have to count refusals,
  and we are back where we started. The recommendation above says not.
- **What does a cursor from an earlier program do when it forwards onto a site
  that no longer verifies?** Today that is a decline. It should probably stay
  one — a cursor is not an index, so nothing is miscounted.
- **`insert_round`'s unreachable positions.** It already refuses a candidate
  that has no statement-level slot for its preamble (a loop condition, a
  comprehension, a branch). Those are structurally ineligible rather than
  semantically refused, so under this plan they should drop out of the listing
  entirely rather than appear as refusals. Worth checking the same distinction
  does not exist elsewhere.
