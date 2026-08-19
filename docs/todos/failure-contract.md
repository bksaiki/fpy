# Plan: one failure contract for the scheduling language

Item 1 of [scheduling-language.md](scheduling-language.md), broken into
commit-sized phases. Policy: pause after each phase for review; full suites
only at the end.

The contract: one hierarchy in `fpy2/transform/error.py`, raised by the
transforms and re-exported from `fpy2.strategies` —

- `TransformDeclined(TransformError)` — the transform refused at a site it
  was aimed at; program unchanged; the message says why. Recoverable.
- `TransformReferenceError(TransformError)` — an explicit `where` named no
  candidate site. `TypeError`/`ValueError` stay for argument validation.
- `where=None` keeps its apply-everywhere meaning, including to zero sites,
  and skips declined candidates silently.

## Phases

- [x] **1. Hierarchy + loud `where` in the `BlockRewriter` transforms.**
  `error.py`; `check_site` in `transform/utils.py`; range checks in
  `unfold_special`, `unfold_neg_zero`, `unfold_overflow`, `float_to_fixed`,
  `rescale_fixed` (previously a silent no-op); exports from `fpy2.transform`
  and `fpy2.strategies`; the five `test_index_past_the_last_site` tests
  flipped to expect the error.

- [x] **2. Migrate the existing out-of-range `ValueError`s.**
  `split_loop`, `for_unroll`, `while_unroll`, `func_inline` now raise
  `TransformReferenceError` (messages unchanged; no `ValueError` compat
  inheritance — tests updated instead). Out of scope, deliberately:
  `monomorphize`'s conflict errors (caller-supplied contradictions, not
  references) and the STRICT-divisibility errors in `split_loop` /
  `for_unroll` — the latter are candidates for `TransformDeclined`, revisit
  after phase 4.

- [x] **3. Declined-with-reason: protocol + `UnfoldSpecial`.**
  `BlockRewriter` split into a structural `_candidate` (counts toward
  `where`) and a `_verify` returning info or `Declined(reason)`; the default
  `_verify` passes every match through, so the unmigrated transforms behave
  as before. Explicit `where` at a declined candidate raises
  `TransformDeclined('where=k: <reason>')`; `where=None` skips it.
  `UnfoldSpecial` declines: context not statically known; `REAL`; nothing
  to state (the idempotency case). The numbering change — declined blocks
  count, `where=k` is the k-th structurally-matching block — broke no
  existing test; new tests pin it.

- [x] **4. The remaining four rounding transforms** on the split protocol,
  each with per-reason declines (`unfold_neg_zero`: not fixed-point, one
  zero, unrebuildable, probe disagreement; `unfold_overflow`: not bounded,
  unsigned, varying overflow value, refused special; `float_to_fixed`: not a
  known float format, shifted encoding, asymmetric bounds, unreachable
  policy, no `fpy2` alias; `rescale_fixed`: already at position zero, finite
  substitute, not symbolically shiftable). The shared structural test is
  `rounding_block` in `transform/utils.py`, now used by all five.
  `test_lowered_roundtrip.py` stays bit-exact.

- [x] **5. Docs + full suites.** The rounding wrappers document the
  structural `where` numbering and both errors; the loop/inline wrappers'
  `Raises` sections retyped; the errors listed in
  `docs/source/strategies.rst`. Full runs green: unit suite (2956),
  `python -m tests.infra`, `python -m tests.infra.backend.cpp`, `make lint`.

## From the branch review

Two pre-existing bugs the review surfaced were fixed on this branch:
`FloatToFixed` no longer grows `free_vars` with the `fpy2` alias unless a
rewrite emitted a context (`used_alias`, as `unfold_overflow` does), and an
annotated assign (`y: fp.Real = fp.round(x)`) is no longer a structural
match — the rewrites cannot carry the annotation, so the block is left
alone rather than rewritten without it.

Not taken, recorded as an option: hoist the duplicated `apply()` tail
(`check_site`) and `__init__` into `BlockRewriter` so a sixth subclass
cannot forget the range check, and route the loop/inline transforms'
hand-rolled checks through `check_site`.
