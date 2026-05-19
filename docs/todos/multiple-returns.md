# Support multiple return statements

## Context

FPy currently requires every function to have **exactly one** return
statement, enforced by ``Reachability.analyze(func, check=True)`` at
``fpy2/decorator.py:212``.  The check exists because the FPCore
backend can't express multiple-exit control flow — FPCore is
functional and each construct produces a single expression value.

The restriction is annoying for the common case (interpretation),
which natively supports multiple returns via the Python AST the
``byte`` interpreter emits.  We want to relax the restriction at
decoration time and surface the constraint **only when the user
actually invokes the FPCore backend**.

## Already free

- ``fpy2/interpret/byte.py`` emits one ``pyast.Return`` per
  ``ReturnStmt``.  Python handles multiple returns natively.
- ``fpy2/analysis/format_infer/analysis.py`` already runs a
  running-join across all return statements in ``_visit_return``.
- ``fpy2/backend/cpp/emitter.py:_visit_return`` already emits
  ``return rhs;`` per ``ReturnStmt``.  C++ supports multiple
  returns.
- ``fpy2/backend/fpc.py:_visit_block`` already raises
  ``FPCoreCompileError`` if it encounters a non-trailing
  ``ReturnStmt`` — the fpc-side gate is already there.

## Phases

Each phase is independent and the unit suite stays green between
them.  Pause for review between phases.

### Phase 1 — drop the single-exit check

``Reachability.analyze(func, check=True)`` asserts three
properties: all-reachable, no-fallthrough, single-return.  The
first two are still wanted; only the third is the restriction.

Change ``Reachability.analyze`` to accept a finer-grained
parameter (or split the single-exit check off entirely), and have
``fpy2/decorator.py`` skip that check.  Keep the other two.

### Phase 2 — unify return types across paths

``fpy2/analysis/type_infer.py:_visit_return`` and
``fpy2/analysis/context_infer.py:_visit_return`` both assign
``self.ret_type = …`` (overwriting).  With multiple returns only
the last one wins.  Change to "first sets, subsequent unify with
the running value" — same pattern format_infer already uses.

### Phase 3 — cpp emitter return-storage selection

``fpy2/backend/cpp/emitter.py:_infer_return_storage`` walks the
body and uses the *first* ``ReturnStmt``'s storage as the
function's return type.  Change to consult the joined bound
``format_info._return_fmt`` (which is already accumulated across
all returns).  Two-line change.

### Phase 4 — fpc error wording

``fpy2/backend/fpc.py`` already rejects non-trailing returns with
``'return statements must be at the end of blocks'``.  Reword to
``'FPCore does not support multiple return statements'`` so the
user understands the connection to the backend.  Optionally
mention the workaround (use the FPy interpreter instead).

### Phase 5a — fix the syntax check's path-termination tracking

A hidden coupling surfaced when adding examples: ``SyntaxCheck``
tracks "variable defined along all paths" by intersecting envs
across branches.  After a ``ReturnStmt`` it returns ``_Env()``
(empty), which under intersection makes *every* variable
"undefined" downstream.  Under the old single-return rule this
never mattered (the return was always the last statement, so
nothing followed).  With early returns, e.g.

.. code-block:: python

   if x > 0:
       return x
   return -x

intersecting ``{x: True}`` with ``_Env()`` produces ``{x: False}``
and the trailing ``return -x`` is flagged with
``variable 'x' not defined along all paths``.

Fix: add a terminated sentinel to ``_Env`` (e.g., a
``terminated: bool`` flag).  ``_visit_return`` returns a
terminated env.  ``_Env.merge`` treats a terminated env as "this
path doesn't reach the merge point" — the result is the other
env unchanged.  When both branches terminate, the result is also
terminated; the surrounding context handles "nothing follows"
the same way it handled the single-return case.

### Phase 5 — tests

Per-path coverage of the new behavior:

- ``tests/unit/analysis/test_reachability.py`` (if it exists) —
  drop any pin on the "single return rejected" behavior; keep
  unreachable / fallthrough pins.
- ``tests/unit/interpret/`` — a program with
  ``if c: return a else: return b`` interpreted on both
  conditions, produces both values.
- ``tests/unit/analysis/test_type_infer.py`` /
  ``test_format_infer.py`` — pin the join semantics across
  multiple returns (incompatible types → unification error;
  compatible types → joined type/format).
- ``tests/unit/backend/cpp/`` — pin that a multiple-return
  program compiles cleanly through the cpp backend.
- fpc backend test — pin the new error message for a
  multiple-return input.
- ``tests/infra/examples/unit.py`` — add canonical example
  programs exercising the new control-flow shapes (early
  return, if/else returns, returns inside loops, returns inside
  ``with`` blocks).  These get auto-collected into the unit-test
  corpus and run through every infra backend, so they exercise
  the cpp pipeline end-to-end and surface the new fpc error on
  the fpc side.  Add the ones we expect to *fail* on fpc to
  ``tests/infra/backend/fpc.py::_ignore`` (or rely on the
  precise error message in a dedicated fpc test).

## Out of scope

- **Restructuring fpc to support multiple returns.** The whole
  point of the restriction in FPCore-land is that there's no
  natural way to express it; restructuring would require
  significant work (e.g., CPS conversion).  Not worth it given
  the interpreter and cpp paths are the common ones.
- **Early-exit optimizations** (returning from a loop, etc.).
  Once multiple returns are allowed, optimizing them is a
  separate concern.

## Trade-off

Errors shift from decoration time to fpc-compile time.  That's
the desired outcome — the cost is paid only by users who actually
target FPCore.  All other consumers (interpreter, cpp backend)
benefit immediately.
