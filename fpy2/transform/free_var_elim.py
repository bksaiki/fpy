"""
Eliminate a function's free variables by materializing each captured value
as a leading assignment ``x = <value>``, closing the function.

A free variable whose captured value has an FPy literal form (numbers, bools,
tuples/lists thereof) is inlined as an assignment at the top of the body; a
free variable bound to something with no literal form — a callable
(:class:`~fpy2.Function` / :class:`~fpy2.Primitive`) resolved at the call site,
a module, etc. — is left untouched.  After the pass the function's
``free_vars`` no longer lists any inlined (now locally-bound) variable.

Backends that cannot reference a closure environment run this first so that a
data free variable becomes an ordinary local binding.
"""

from ..analysis import SyntaxCheck
from ..ast.fpyast import Assign, FuncDef, FuncMeta, StmtBlock

from .const_fold import value_to_literal


class FreeVarElim:
    """Close a function over its free variables (see module docstring)."""

    @staticmethod
    def apply(func: FuncDef) -> FuncDef:
        """Apply the transformation to a :class:`FuncDef`.  Returns a new
        ``FuncDef`` (the input is not mutated); returns it unchanged when no
        free variable has an inlinable value."""
        if not isinstance(func, FuncDef):
            raise TypeError(f"expected a 'FuncDef', got `{func}`")

        env = func.env
        prelude: list[Assign] = []
        bound = set()
        # Sort by name for a deterministic prelude order.
        for fv in sorted(func.free_vars, key=str):
            name = str(fv)
            if name not in env:
                continue
            lit = value_to_literal(env[name], None)
            if lit is None:
                # No literal form (a callable, module, ...) — leave it free.
                continue
            prelude.append(Assign(fv, None, lit, None))
            bound.add(fv)

        if not prelude:
            return func

        new_body = StmtBlock(prelude + list(func.body.stmts))
        meta = FuncMeta(
            func.free_vars - bound,
            func.meta.ctx,
            func.meta.spec,
            func.meta.props,
            func.env,
        )
        out = FuncDef(func.name, func.args, new_body, meta, loc=func.loc)
        SyntaxCheck.check(out, ignore_unknown=True)
        return out
