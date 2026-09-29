"""
Triton backend: one name per value.

Triton holds a name at one type and shape wherever a loop or branch joins it,
and treats every name a loop body assigns as carried.  A name FPy binds to
unrelated values -- two comprehensions' `p`, say -- then fails to compile
under a loop, though no phi ever joins them.  Here each class of definitions
(those a phi or a mutation joins) gets its own name: the first class keeps
the original, so arguments are unchanged.
"""

from ...analysis import DefineUse, DefineUseAnalysis
from ...analysis.reaching_defs import Definition, DefSite, same_object_defs
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
from ...utils import Gensym, Unionfind

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
        defs = self.def_use.defs
        uf: Unionfind[Definition] = Unionfind(defs)
        for d in defs:
            for i in same_object_defs(d):
                uf.union(d, defs[i])
        gensym = Gensym(reserved=self.def_use.name_to_defs.keys())
        kept: set[NamedId] = set()
        by_class: dict[Definition, NamedId] = {}
        for d in defs:
            c = uf.find(d)
            if c not in by_class:
                by_class[c] = d.name if d.name not in kept else gensym.refresh(d.name)
                kept.add(d.name)
        self.name_of = {d: by_class[uf.find(d)] for d in defs}
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

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: None):
        s, ctx = super()._visit_indexed_assign(stmt, ctx)
        return IndexedAssign(self._renamed(stmt.var), s.indices, s.expr, s.loc), ctx

    def _visit_context(self, stmt: ContextStmt, ctx: None):
        s, ctx = super()._visit_context(stmt, ctx)
        self._site = stmt
        target = self._renamed(stmt.target) if isinstance(stmt.target, NamedId) else stmt.target
        return ContextStmt(target, s.ctx, s.body, s.loc), ctx

    def _visit_statement(self, stmt: Stmt, ctx: None) -> tuple[Stmt, None]:
        # a statement's targets are visited before any statement it nests
        if isinstance(stmt, Assign | IndexedAssign | ForStmt | ContextStmt):
            self._site = stmt
        return super()._visit_statement(stmt, ctx)


def split_names(func: FuncDef) -> FuncDef:
    """*func* with each class of definitions under its own name."""
    return _SplitNames(func)._visit_function(func, None)
