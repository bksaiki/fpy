"""
Variable substitution. 
"""

from typing import Mapping

from ..analysis import AssignDef, DefineUseAnalysis
from ..ast import *

class _SubstVar(DefaultTransformVisitor):
    """Visitor for variable substitution."""

    func: FuncDef
    def_use: DefineUseAnalysis
    subst: Mapping[AssignDef, Expr]

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis, subst: Mapping[AssignDef, Expr]):
        self.func = func
        self.def_use = def_use
        self.subst = subst

    def _visit_var(self, e: Var, ctx: None):
        d = self.def_use.find_def_from_use(e)
        if d in self.subst:
            return self.subst[d]
        else:
            return Var(e.name, e.loc)

    def apply(self):
        """Applies the replacement to the function."""
        return self._visit_function(self.func, None)


class SubstVar:
    """
    Replaces occurence of variables with expressions.

    This transformation is the basis for:
    - copy propagation
    - constant propagation
    """

    @staticmethod
    def apply(func: FuncDef, def_use: DefineUseAnalysis, subst: Mapping[AssignDef, Expr]):
        """
        Given a substitution from variable definitions to expressions, replaces
        all occurences of the variables with the corresponding expressions.
        The original definition will not be renamed.
        """

        if not isinstance(func, FuncDef):
            raise TypeError('expected FuncDef', func)
        return _SubstVar(func, def_use, subst).apply()
