"""
Copy propagation.
"""

from ..analysis import AssignDef, DefineUse, DefineUseAnalysis, SyntaxCheck
from ..ast.fpyast import *
from .subst_var import SubstVar


def _same_source_at_uses(def_use: DefineUseAnalysis, d: AssignDef) -> bool:
    """Whether every use of the copy `x = y` that *d* defines resolves `y` to
    the definition it copied, so substituting `y` reads the same value."""
    assert isinstance(d.site, Assign) and isinstance(d.site.expr, Var)
    y = d.site.expr.name
    copied = def_use.find_def_from_use(d.site.expr)
    return all(def_use.defs_at_use[u].get(y) is copied for u in def_use.uses[d])


class CopyPropagate:
    """
    Copy propagation.

    This transform replaces any variable that is assigned another variable.
    """

    @staticmethod
    def apply(func: FuncDef, *, names: set[NamedId] | None = None) -> FuncDef:
        """Applies copy propagation to the given AST."""
        func, _ = CopyPropagate.apply_with_status(func, names=names)
        return func

    @staticmethod
    def apply_with_status(
        func: FuncDef, *, names: set[NamedId] | None = None,
    ) -> tuple[FuncDef, bool]:
        """Same as :meth:`apply` but also returns a ``changed`` flag —
        ``True`` iff at least one alias was propagated."""
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\' for {func}, got {type(func)}')

        prop: dict[AssignDef, Var] = {}
        def_use = DefineUse.analyze(func)
        for d in def_use.defs:
            if names is not None and d.name not in names:
                continue

            if (
                isinstance(d, AssignDef)
                and isinstance(d.site, Assign)
                and isinstance(d.site.target, Id)
                and isinstance(d.site.expr, Var)
                and len(def_use.uses[d]) > 0
                and _same_source_at_uses(def_use, d)
            ):
                # x = y: substitute `y` for every use of this definition of `x`
                prop[d] = d.site.expr

        if not prop:
            return func, False

        func, changed = SubstVar.apply_with_status(func, def_use, prop)
        SyntaxCheck.check(func, ignore_unknown=True)
        return func, changed
