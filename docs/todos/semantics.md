# FPy semantics: spec vs. implementation discrepancies

Tracks known divergences between the idealized semantics documented in
`docs/source/dev/semantics.rst` and the behavior of the actual
implementation (the AST in `fpy2/ast/fpyast.py` and the interpreters in
`fpy2/interpret`). The doc deliberately describes a clean abstract
semantics; this file records where the implementation differs and whether
the difference is a bug to fix or an intentional simplification of the doc.

## Open items

### Function bodies run under the caller's context, ignoring per-function context

**E-App** evaluates the called function's body under the *caller's* active
context `C`. Real FPy functions may pin their own rounding context
(`@fpy(ctx=...)`, surfaced as `FuncDef.ctx`); `Interpreter._func_ctx` uses that
override when present and only falls back to the caller's context otherwise.

The core fragment models neither function-definition syntax nor the context
override, so E-App's caller-context behavior is a simplification rather than a
faithful account of a function with a pinned context. Extend E-App to consult
the function's declared context if/when definitions are added to the fragment.

## Resolved

### Context statement now binds the new context (matches E-Context)

`with e as x in s` previously bound `x` to the *enclosing* context rather than
the new context `C'` that governs the body. `_visit_context` in
`fpy2/interpret/byte.py` emitted `tmp = target = __ctx__`, stashing the old
context into both the restore temporary *and* the target. Fixed to stash only
the temporary (`tmp = __ctx__`) and bind the target to the new context
alongside the active context (`target = __ctx__ = <new context>`), matching the
**E-Context** rule. Verified against the full unit + infra suites.

### Context expression is now evaluated under the real context `R`

**E-Context** evaluates the context expression `e` under the real context `R`
(`<σ, R, e> ⇓ C'`), not the enclosing active context. `_visit_context` now
installs the real context before evaluating `e`: it emits `__ctx__ = __fpy_real`
(a namespace symbol bound to `REAL`) before the
`<target> = __ctx__ = <new context>` assignment, so any rounding inside the
context expression uses `REAL`, the identity. Verified against the full unit +
infra suites.
