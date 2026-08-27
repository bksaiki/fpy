# Plan: a name that only feeds a container is not a second place

[backend-cpp.md](backend-cpp.md)'s *What stays boxed* → **A list a local name and
a container both hold**. `xs = [n, n]; return (xs, 1.0)` keeps its handle;
`return ([n, n], 1.0)` does not.

## The mechanism, measured

Both programs allocate the same list. What differs is one integer:

| | `_names` | `_slots` | `referrers` | `transfers_ownership` | verdict |
|---|---|---|---|---|---|
| `return ([n, n], 1.0)` | `{}` | 1 | 1 | **True** | `std::array<double, 2>` |
| `xs = [n, n]; return (xs, 1.0)` | `{xs}` | 1 | **2** | False | `shared_ptr<vector<double>>` |

`referrers` is `len(names) + slots`, and `transfers_ownership` requires
`not is_shared(site)`, i.e. `referrers <= 1`. So the name is counted as a place
that coexists with the container slot, when in fact the value *moves* from the
name into the container and the name is never read again.

Note where this does **not** surface: `Unbox.decide` computes
`escapes or _shares_storage(...)`, which short-circuits, so `_shares_storage` is
never even asked in the boxing case. Fixing that function would not fix this.

## Why now

**It is what aggregate naming in ANF runs into.** The two rows above are the same
program before and after naming an aggregate — `xs = [n, n]` *is* the ANF form of
the inlined literal. So extending ANF to name aggregates, which the roadmap calls
its highest-risk item and which every `_emit_at` and `_bind_operand` deletion
waits on, would turn today's unboxed row into today's boxed row across the board.
This is a prerequisite for that work, not an optimization beside it.

The §5 justification recorded earlier does **not** hold and should be dropped:
none of the corpus's 18 refusals is aliasing-related, so nothing about conversion
insertion is gated on this.

## What can and cannot check the work

- **The emitted-code harness is blind here.** 337 corpus regions, **0 boxed**,
  `_shares_storage` never `True`. A change to the boxing rule moves no corpus
  byte, so `diff -r` proves only that nothing *else* broke.
- **The unit suite is the safety net.** Across `tests/unit/backend/cpp`, the
  sharing verdict is asked 2068 times and answers `True` 95 times, over 286 boxed
  regions. That is what a mistake would show up in.
- `test_unbox_profile.py` pins `EXPECTED_LEVELS = 170` / `EXPECTED_BOXED = 0`.
  Signature levels only, so it will not move — but it is the file to update if a
  count ever does, with the reason.
- `exploration/unboxing/box_gap.py` is the motivating pair, and
  `box_census.py` the corpus census.

## The condition, and its one trap

A name is *consumed* by a construction when the value it holds moves into the
container rather than being shared with it. The sound approximation:

```
d is consumed by use u  iff  uses(d) == {u}                     -- sole reader
                             and u constructs a container
                             and u executes at most once per d
```

**The third clause is the trap.** `DefineUseAnalysis.uses` gives sole-*syntactic*-use
directly, but a single use site inside a loop is read once per iteration:

```
xs = [n, n]
for i in range(3):
    t = (xs, 1.0)        # sole use, executed three times
```

Moving out of `xs` on the first iteration empties it for the second — a silent
wrong answer, not a compile error. `LiveVars` does not help as it stands: it
returns only the live-in set at function entry and discards the per-statement
sets.

So phase 1 takes the conservative reading — the use is in the same statement
block as the definition, with no loop between them — which covers the motivating
case and every shape ANF's aggregate naming would produce, since ANF puts the
name immediately before its use. A general version needs per-point liveness and
is a follow-on.

## Phases

Each is a commit. Pause after each for review; do not commit. Only the named
tests run per phase; full suites at the end.

### 1. `Alias` learns which names are consumed

`AliasAnalysis.consumed_names: dict[Region, set[NamedId]]` — the names whose sole
use is a construction that puts them in a container, under the conservative
reading above. Computed in the existing `_RegionFacts` walk, which already has
`def_use`.

A fact, not a policy: it changes no query and no verdict, so this phase moves no
emitted byte and no boxing decision. That is the point — it lands separately so
the behaviour change in phase 2 has a reviewable diff of its own.

Tests: `tests/unit/analysis/test_alias.py` — the sole-use case, a two-use name,
a name used in a loop (must *not* be consumed), and a name whose use is not a
construction.

### 2. The verdict uses it, and the emitter moves

This is C++'s value categories, and the emitter half is exactly the lvalue →
xvalue cast. `return ([n, n], 1.0)` hands `std::make_tuple` a **prvalue**, which
move-constructs for free; `xs` is an **lvalue**, so it copy-constructs, and
`std::move(xs)` is what makes it eligible. C++'s own implicit-move rule does not
reach this case: it covers `return xs;` — a local returned *whole* — not a local
appearing as a subexpression of the returned value. C++20 widened that, but the
target is `-std=c++11`.

**Whether the move is needed depends on the representation, so check rather than
assume.** For `std::vector<T>` a move is O(1) — it steals the buffer — and
skipping it turns a refcount bump into an O(n) copy, which is the pessimization
worth fearing. For `std::array<T, K>` there is no such win: the move is
element-wise, and for a trivially-copyable `T` that *is* a copy. The motivating
program unboxes to `std::array<double, 2>`, so there the move buys nothing and
the gain is the stack object instead of a heap allocation plus refcount.

So the "must land together" rule holds for the `vector` case and not the fixed-size
one. Phase 2 should establish which representations the discount can reach before
deciding whether both halves truly have to be one commit.

- `referrers` gains a discounted reading — `referrers(region) - |consumed|` —
  exposed as a distinct query rather than by changing `referrers` itself, which
  `is_shared`, `transfers_ownership`, `is_uniquely_owned` and `_shares_storage`
  all read. Which of those should switch to the discounted count is the decision
  this phase makes, and it should be made one caller at a time with the census
  re-run after each.
- The emitter emits `std::move(xs)` at the construction, under exactly the
  condition that discounted it. Same-condition, or the two can drift into a
  use-after-move.

Tests: a compile-and-run case for the motivating program — it must produce
`std::array` *and* still give the right answer — plus the loop program, which
must keep its handle and must not contain `std::move`. Then the full
`tests/unit/backend/cpp` suite and the census.

### 3. Documentation

Fold into `backend-cpp.md`'s *What stays boxed* (the entry stops being open) and
note in the roadmap that aggregate naming's blocker is gone. Retire this
document.

## Not in this plan

- **Per-point liveness.** The general condition — a name dead after its use
  wherever that use sits — needs `LiveVars` to keep its per-statement sets. Worth
  doing when a program needs it; nothing does yet.
- **The other open entry under *What stays boxed*** (a projection whose slot is
  replaced) is deliberate, not a gap.
- **Moving into a call argument.** The same reasoning applies to `f(xs)` where
  `xs` is dead after, but the condition there is the callee's ABI rather than a
  construction, so it belongs with the call-boundary facts that stayed in the
  backend in §3.
