"""
Constant propagation.
"""

from ..analysis import AssignDef, DefineUse, SyntaxCheck
from ..ast import *

from .subst_var import SubstVar


def _is_value(e: Expr) -> bool:
    """
    Is this expression a constant?

    <cprop> ::= <bool>
              | <number>
              | <foreign>
    """
    return isinstance(e, BoolVal | RealVal | ForeignVal)

class ConstPropagate:
    """
    Constant propagation.

    For any occurence of the form `x = c` where `c` is a constant,
    this transform replaces all uses of `x` with `c`.
    """

    @staticmethod
    def apply(func: FuncDef, *, names: set[NamedId] | None = None) -> FuncDef:
        """
        Applies constant propagation.

        If `names` is provided, only propagate variables in this set.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\' for {func}, got {type(func)}')

        prop: dict[AssignDef, Expr] = {}
        def_use = DefineUse.analyze(func)
        for d in def_use.defs:
            if names is not None and d.name not in names:
                continue

            if isinstance(d, AssignDef) and isinstance(d.site, Assign) and _is_value(d.site.expr):
                # direct assignment: x = c
                # substitute all occurences of this definition of `x` with `c`
                if len(def_use.uses[d]) > 0:
                    # at least one use of this definition
                    prop[d] = d.site.expr

        if prop:
            # at least one variable to propagate
            func = SubstVar.apply(func, def_use, prop)
            SyntaxCheck.check(func, ignore_unknown=True)

        return func
