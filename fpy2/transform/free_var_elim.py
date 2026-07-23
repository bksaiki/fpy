"""
Close a function over its free variables by materializing each captured *data*
value as a leading assignment ``x = <value>``.

A free variable with an FPy literal form (numbers, bools, tuples/lists thereof)
is inlined; one with no literal form — a callable resolved at the call site, a
module, a rounding context — is left free for its own resolving machinery.
Backends that can't reference a closure environment run this first.
"""

from ..analysis import SyntaxCheck
from ..ast.fpyast import Assign, Expr, FuncDef, FuncMeta, StmtBlock
from ..number import Context
from .const_fold import value_to_literal


def inline_literal(val: object) -> Expr | None:
    """The AST literal to bind for a captured value, or ``None`` if it should
    stay free (no literal form, or a rounding :class:`Context` resolved by
    context analysis rather than a value binding).

    Note the deliberate divergence from :func:`const_fold.value_to_literal`,
    which *does* give a ``Context`` a literal form (``ForeignVal``): that path
    feeds context analysis, whereas this pass emits a leading value-binding
    prelude and leaves the ``Context`` free for the context machinery.  Do not
    "unify" the two — folding a ``Context`` here would break context analysis."""
    if isinstance(val, Context):
        return None
    return value_to_literal(val, None)


def unclosed_data_free_vars(func: FuncDef) -> list[str]:
    """Free-variable names in *func* that still have a value binding to emit.
    Backends assert this is empty after :meth:`FreeVarElim.apply`."""
    return [
        str(fv) for fv in func.free_vars
        if str(fv) in func.env and inline_literal(func.env[str(fv)]) is not None
    ]


class FreeVarElim:
    """Close a function over its free variables (see module docstring)."""

    @staticmethod
    def apply(func: FuncDef) -> FuncDef:
        """Return a new ``FuncDef`` closed over its data free variables (the
        input is not mutated); returns it unchanged when there are none."""
        if not isinstance(func, FuncDef):
            raise TypeError(f"expected a 'FuncDef', got `{func}`")

        env = func.env
        prelude: list[Assign] = []
        bound = set()
        # Sorted for a deterministic prelude order.
        for fv in sorted(func.free_vars, key=str):
            name = str(fv)
            if name not in env:
                continue
            lit = inline_literal(env[name])
            if lit is None:
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
