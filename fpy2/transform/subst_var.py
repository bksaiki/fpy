"""
Variable substitution. 
"""

from collections.abc import Mapping

from ..analysis import AssignDef, DefineUseAnalysis
from ..ast import *


class _SubstVar(DefaultTransformVisitor):
    """Visitor for variable substitution."""

    func: FuncDef
    def_use: DefineUseAnalysis
    subst: Mapping[AssignDef, Expr]
    changed: bool
    """whether any occurrence was actually replaced"""

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis, subst: Mapping[AssignDef, Expr]):
        self.func = func
        self.def_use = def_use
        self.subst = subst
        self.changed = False

    def _visit_var(self, e: Var, ctx: None):
        d = self.def_use.find_def_from_use(e)
        if d in self.subst:
            self.changed = True
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
        func, _ = SubstVar.apply_with_status(func, def_use, subst)
        return func

    @staticmethod
    def apply_with_status(
        func: FuncDef, def_use: DefineUseAnalysis, subst: Mapping[AssignDef, Expr],
    ) -> tuple[FuncDef, bool]:
        """Same as :meth:`apply` but also returns a ``changed`` flag — ``True``
        iff an occurrence was replaced.

        A definition can have uses that are not occurrences: an
        ``xs[i] = e`` names ``xs`` as an :class:`Id`, not a :class:`Var`, so it
        is a use that no substitution rewrites.  A caller running to a fixpoint
        needs the difference, or it never converges.
        """
        if not isinstance(func, FuncDef):
            raise TypeError('expected FuncDef', func)
        inst = _SubstVar(func, def_use, subst)
        return inst.apply(), inst.changed
