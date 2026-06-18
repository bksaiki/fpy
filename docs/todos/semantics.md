# FPy semantics: spec vs. implementation discrepancies

Tracks known divergences between the idealized semantics documented in
`docs/source/dev/semantics.rst` and the behavior of the actual
implementation (the AST in `fpy2/ast/fpyast.py` and the interpreters in
`fpy2/interpret`). The doc deliberately describes a clean abstract
semantics; this file records where the implementation differs and whether
the difference is a bug to fix or an intentional simplification of the doc.

## Open items

### Context statement binds the *enclosing* context, not the new one

The **E-Context** rule states that `with e as x in s` binds `x` to the new
context `C'` (the one that governs the body):

```
<σ, R, e> ⇓ C'    <σ[x ↦ C'], C', s> ⇓_S σ'
─────────────────────────────────────────────
<σ, C, with e as x in s> ⇓_S σ'
```

The `BytecodeInterpreter`, however, binds `x` to the **enclosing** context
`C` (the one active *before* the `with`), not to `C'`. Reproduction:

```python
from fpy2 import fpy, IEEEContext, RM

outer = IEEEContext(8, 32, RM.RNE)
inner = IEEEContext(11, 64, RM.RNE)

@fpy(ctx=outer)
def f():
    with inner as x:
        y = x
    return y

f()  # == outer, NOT inner
```

In `fpy2/interpret/byte.py`, `_visit_context` emits `tmp = target = __ctx__`
*before* swapping in the new context, so the target receives the old
`__ctx__`. The existing test `test_context8` (`tests/infra/examples/unit.py`)
relies on this: it reuses the bound name to re-enter the enclosing context.

**Question to resolve:** is binding the enclosing context intentional, or a
bug? The surface syntax `with inner as x` strongly reads as `x == inner`,
which argues for a fix. If the enclosing-context binding is intended, the
E-Context rule and prose should be changed to bind `x` to `C` instead, and
the syntax/naming reconsidered.

### Context expression is evaluated under the active context, not `R`

**E-Context** evaluates the context expression `e` under the real context
`R` (`<σ, R, e> ⇓ C'`). The implementation evaluates `e` under the *current*
active context `C` instead — `_visit_context` lowers `e` while the old
`__ctx__` is still installed, and only swaps afterward.

This is typically immaterial: context-constructor expressions do not perform
rounding, so the active context does not affect their value. Listed for
completeness; revisit if a context expression can ever observe rounding.
