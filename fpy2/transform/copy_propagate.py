"""
Copy propagation.
"""

from ..analysis import AssignDef, DefineUse, SyntaxCheck
from ..ast.fpyast import *

from .subst_var import SubstVar

class CopyPropagate:
    """
    Copy propagation.

    This transform replaces any variable that is assigned another variable.
    """

    @staticmethod
    def apply(func: FuncDef, *, names: set[NamedId] | None = None) -> FuncDef:
        """Applies copy propagation to the given AST."""
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\' for {func}, got {type(func)}')

        prop: dict[AssignDef, Var] = {}
        def_use = DefineUse.analyze(func)
        for d in def_use.defs:
            if names is not None and d.name not in names:
                continue

            if isinstance(d, AssignDef) and isinstance(d.site, Assign) and isinstance(d.site.expr, Var):
                # direct assignment: x = y
                # substitute all occurences of this definition of `x` with `y`
                if len(def_use.uses[d]) > 0:
                    # optimization: only propagate if there is at least one use
                    prop[d] = d.site.expr

        if prop:
            # at least one variable to propagate
            func = SubstVar.apply(func, def_use, prop)
            SyntaxCheck.check(func, ignore_unknown=True)

        return func
