# Format inference: AbstractFormat-mediated join

Make `_join_bounds` smarter for the `Format ⊔ Format` case so that joining two
distinct concrete formats (e.g. `FP32 ⊔ FP64`) yields a single `Format` that
contains both, rather than immediately widening to `REAL_FORMAT`.

`AbstractFormat` becomes pure join machinery — it never appears in the
analysis lattice. The lattice stays:

```
FormatBound ::= None | SetFormat | Format | TupleFormat | ListFormat
```

Joins of distinct Formats look like:

```python
(AbstractFormat.from_format(f1) | AbstractFormat.from_format(f2)).format()
```

so the analysis always produces `Format` instances.

## Plan

### Phase 1 — Adapt `AbstractFormat` (`fpy2/analysis/format_infer/format.py`)

- Replace `from_context(ctx: SupportedContext)` with `from_format(fmt: Format)`,
  dispatching on `Format` subclasses to extract `(prec, exp, pos_bound, neg_bound)`.
- Add `format(self) -> Format`: returns a canonical `Format` whose representable
  set is a superset of `self`'s. The fully-saturated case (all four parameters
  unbounded) returns `REAL_FORMAT`.
- Replace `AbstractableContext` with `AbstractableFormat`: the union of
  `Format` subclasses supported by `from_format`. `from_format` is partial —
  it raises `ValueError` on non-abstractable inputs. Callers gate with
  `isinstance(fmt, AbstractableFormat)`.
- `from_format` covers Formats produced by the previously-supported contexts:
  `MPFixedContext`, `MPBFixedContext`, `ExpContext`, `MPFloatContext`,
  `MPSFloatContext`, `MPBFloatContext`, `EFloatContext`. Anything else is not
  abstractable.
- `format()` picks a canonical `Format` by parameter shape:
  - All unbounded → `REAL_FORMAT`
  - Finite prec, finite exp, finite bounds → `MPBFloatContext(...).format()`
  - Finite prec, finite exp, ∞ bounds → `MPSFloatContext(...).format()`
  - Finite prec, −∞ exp, ∞ bounds → `MPFloatContext(...).format()`
  - ∞ prec, finite exp, finite bounds → `MPBFixedContext(...).format()`
  - ∞ prec, finite exp, ∞ bounds → `MPFixedContext(...).format()`
  - Mixed / odd shapes → `REAL_FORMAT` (sound fall-back)

### Phase 2 — Update mpfx call sites

Five files: `fpy2/backend/mpfx/{__init__.py,compiler.py,elim_round.py,
format_infer.py,instr.py}`.

- `AbstractFormat.from_context(ctx)` → `AbstractFormat.from_format(ctx.format())`.
- Drop the `SupportedContext` re-export from the mpfx package.

### Phase 3 — Update `analysis.py`

- `_join_bounds` `Format ⊔ Format` (different) becomes:

  ```
  if both abstractable:
      return (AbstractFormat.from_format(f1) | AbstractFormat.from_format(f2)).format()
  else:
      return REAL_FORMAT
  ```

- `FormatBound` type alias unchanged.
- Update the module docstring to describe the AbstractFormat-mediated join.

### Phase 4 — Tests (`tests/unit/analysis/test_format_infer.py`)

- `test_branch_distinct_formats_joins_to_containing_format`: two `with FPxx`
  branches → result is a single `Format` ⊇ both (not `REAL_FORMAT`).
- `test_abstract_format_round_trip`: `AbstractFormat.from_format(FP32).format()`
  returns a `Format` ⊇ FP32.
- `test_join_saturates_to_real`: a join that saturates → `REAL_FORMAT`.
- `test_loop_format_join_converges`: a loop whose body produces a different
  Format than pre-loop — fixpoint reached, result is the joined containing
  Format.

## Out of scope

- Widening / loop-iteration knob (deferred — union-only joins terminate over
  the finite set of program-derived Formats).
- `AbstractFormat` appearing in the lattice (no longer needed).
- `SetFormat ⊔ AbstractFormat` rule (no longer a case).
- Other `AbstractFormat` operations (`+`, `*`) on the lattice.
- `representable_in` on `AbstractFormat`.
