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
