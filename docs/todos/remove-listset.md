# Remove the `ListSet` AST node

## Context

`ListSet` is an `Expr` node representing a functional update of a list:
``ListSet(value, [i1, …, iN], expr)`` ≡ "a new list where
``value[i1]…[iN]`` is replaced by ``expr``".  It was added to support
compilation to FPCore, which is a Scheme-like functional language and
has no in-place mutation.

The shape today:

- **Producer:** `fpy2/transform/func_update.py::FuncUpdate` (~30 lines).
  Sole transformation that introduces ``ListSet``.  Rewrites
  ``IndexedAssign(xs, idx, v)`` (statement) → ``Assign(xs, ListSet(xs, idx, v))``
  (statement of an expression).
- **Real consumer:** `fpy2/backend/fpc.py::_visit_list_set` (~50 lines)
  lowers the node to an FPCore ``tensor`` expression.  Its
  ``_visit_indexed_assign`` currently just raises ``FPCoreCompileError``.
- **Pass-through consumers** (small ``_visit_list_set`` stubs that
  recurse into children): `array_size`, `format_infer/analysis`,
  `mpfx/format_infer`, `live_vars`, `type_infer`, `context_infer`,
  `syntax_check`, `formatter`, `byte` (interpreter), `rewrite/matcher`,
  plus visitor dispatch entry.  The cpp emitter ``_unsupported``s it
  (cpp's pipeline never runs ``FuncUpdate``).
- **Tests:** `tests/unit/analysis/test_format_infer.py` and
  `tests/unit/backend/cpp/test_emit_indexed_assign.py`.

Classic over-engineering smell: one producer, one real consumer,
several stubs.

## Plan

1. Move fpc's ``_visit_list_set`` body into ``_visit_indexed_assign``
   on the same visitor.  The FPCore ``tensor`` lowering is identical;
   the node it's attached to is the only thing that changes.  Verify
   fpc tests pass.
2. Delete the ``FuncUpdate.apply(ast)`` call from fpc's pipeline
   (``fpy2/backend/fpc.py:1097``).  fpc now sees ``IndexedAssign``
   directly and lowers it inline.  Verify again.
3. Delete every ``_visit_list_set`` stub from analyses, visitors,
   formatter, interpreter, and rewrite/matcher.  Move any non-trivial
   logic (e.g. format inference's element-widening) into the
   corresponding ``_visit_indexed_assign`` if it isn't already there.
4. Delete ``class ListSet`` from ``fpy2/ast/fpyast.py`` and its
   dispatch entry from ``fpy2/ast/visitor.py``.
5. Delete the ``FuncUpdate`` transform and its export from
   ``fpy2/transform/__init__.py``.
6. Update the two tests that mention ``ListSet`` / ``FuncUpdate`` to
   exercise the ``IndexedAssign`` path directly.

Each step should leave the unit suite and cpp infra green; any step
can be reverted independently if it surfaces a hidden dependency.

## Trade-offs

- ``ListSet`` as a pure expression localized the "functional update"
  semantics in ``by_expr`` — convenient for analyses that key on
  per-expression bounds.  After removal, the equivalent logic moves
  to the statement visitor (``_visit_indexed_assign``), where it
  joins assign-widening logic that already exists.  Probably cleaner
  given there's only one real consumer.
- Estimated net delta: ~100 lines removed (the class, the transform,
  ~8 stub visit methods); fpc's ``_visit_indexed_assign`` grows by
  ~50 lines.  Net savings 50–100 lines plus a removed conceptual
  node — modest but real.

## Why this is non-blocking

Nothing else in the codebase depends on ``ListSet`` as a publicly
observable shape (it can't be written in surface Python).  The
removal is an internal refactor; no API surface changes.
