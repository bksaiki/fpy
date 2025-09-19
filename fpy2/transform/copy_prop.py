"""
Copy propagation.
"""

from ..analysis import AssignDef, DefineUse, SyntaxCheck
from ..ast import *

from .subst_var import SubstVar


class CopyPropagate:
    """
    Copy propagation.

    For any occurence of the form `x = y`where `y` is a variable,
    this transform replaces all uses of `x` with `y`.
    """

    @staticmethod
    def apply(func: FuncDef, *, names: set[NamedId] | None = None) -> FuncDef:
        """
        Applies copy propagation.

        If `names` is provided, only propagate variables in this set.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\' for {func}, got {type(func)}')

        prop: dict[AssignDef, Var] = {}
        def_use = DefineUse.analyze(func)
        for name, defs in def_use.defs.items():
            # skip any names not matching the filter
            if names is not None and name not in names:
                continue

            for d in defs:
                if isinstance(d, AssignDef) and isinstance(d.site, Assign) and isinstance(d.site.expr, Var):
                    # direct assignment: x = y
                    # substitute all occurences of this definition of `x` with `y`
                    if len(def_use.uses[d]) > 0:
                        # at least one use of this definition
                        prop[d] = d.site.expr

        if prop:
            # at least one variable to propagate
            func = SubstVar.apply(func, def_use, prop)
            SyntaxCheck.check(func, ignore_unknown=True)

        return func
