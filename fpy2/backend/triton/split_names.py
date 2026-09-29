"""
Triton backend: one name per class of definitions.

Triton carries every name a loop body assigns at one type and shape, so a
name FPy reuses for unrelated values (two comprehensions' `p`) fails to
compile under a loop.  Each class of definitions (those a phi or mutation
joins) gets its own name; the first keeps the original, so arguments are
unchanged.
"""

from ...analysis import DefineUse, DefineUseAnalysis
from ...analysis.reaching_defs import Definition, DefSite, def_classes
from ...ast import (
    Assign,
    ContextStmt,
    DefaultTransformVisitor,
    Expr,
    ForStmt,
    FuncDef,
    Id,
    IndexedAssign,
    NamedId,
    Stmt,
    TupleBinding,
    Var,
)
from ...utils import Gensym

__all__ = ['split_names']


class _SplitNames(DefaultTransformVisitor):
    """Renames each definition, and each use, to its class's name."""

    def_use: DefineUseAnalysis
    name_of: dict[Definition, NamedId]
    """Each definition's name, by its class."""
    _site: DefSite | None
    """The statement whose targets are being visited."""

    def __init__(self, func: FuncDef) -> None:
        self.def_use = DefineUse.analyze(func)
        classes = def_classes(self.def_use.defs)
        gensym = Gensym(reserved=self.def_use.name_to_defs.keys())
        kept: set[NamedId] = set()
        by_class: dict[Definition, NamedId] = {}
        for d, c in classes.items():
            if c not in by_class:
                by_class[c] = d.name if d.name not in kept else gensym.refresh(d.name)
                kept.add(d.name)
        self.name_of = {d: by_class[c] for d, c in classes.items()}
        self._site = None

    def _renamed(self, name: NamedId) -> NamedId:
        assert self._site is not None
        return self.name_of[self.def_use.find_def_from_site(name, self._site)]

    def _visit_var(self, e: Var, ctx: None) -> Expr:
        return Var(self.name_of[self.def_use.use_to_def[e]], e.loc)

    def _visit_binding(self, binding: Id | TupleBinding, ctx: None) -> Id | TupleBinding:
        if isinstance(binding, NamedId):
            return self._renamed(binding)
        return super()._visit_binding(binding, ctx)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: None) -> tuple[IndexedAssign, None]:
        s, ctx = super()._visit_indexed_assign(stmt, ctx)
        return IndexedAssign(self._renamed(stmt.var), s.indices, s.expr, s.loc), ctx

    def _visit_context(self, stmt: ContextStmt, ctx: None) -> tuple[ContextStmt, None]:
        target = self._renamed(stmt.target) if isinstance(stmt.target, NamedId) else stmt.target
        s, ctx = super()._visit_context(stmt, ctx)
        return ContextStmt(target, s.ctx, s.body, s.loc), ctx

    def _visit_statement(self, stmt: Stmt, ctx: None) -> tuple[Stmt, None]:
        # a statement's targets are visited before any statement it nests
        if isinstance(stmt, Assign | IndexedAssign | ForStmt | ContextStmt):
            self._site = stmt
        return super()._visit_statement(stmt, ctx)


def split_names(func: FuncDef) -> FuncDef:
    """*func* with each class of definitions under its own name."""
    return _SplitNames(func)._visit_function(func, None)
