"""Definition-use analysis for FPy ASTs"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeAlias, cast

from ..ast.fpyast import *
from ..ast.visitor import DefaultVisitor
from ..utils import default_repr

from .reaching_defs import (
    ReachingDefs, ReachingDefsAnalysis,
    AssignDef, PhiDef, Definition, DefCtx,
    DefSite, PhiSite
)

UseSite: TypeAlias = Var | IndexedAssign | Call
"""AST nodes that can use variables"""

@dataclass(frozen=True)
class DefineUseAnalysis(ReachingDefsAnalysis):
    """Result of definition-use analysis."""

    uses: dict[int, set[UseSite]]
    """mapping from definition id to use sites"""

    def format(self) -> str:
        lines = ['defs:'] + super().format().splitlines()
        lines.append('uses:')
        for def_id, use_sites in self.uses.items():
            if len(use_sites) > 0:
                lines.append(f'  def {def_id}:')
                for site in use_sites:
                    lines.append(f'    - {self._format_site(site.format())}')
        return '\n'.join(lines)


class _DefineUseInstance(DefaultVisitor):
    """Per-IR instance of definition-use analysis"""
    ast: FuncDef | StmtBlock
    reaching_defs: ReachingDefsAnalysis

    uses: dict[int, set[UseSite]] = {}
    """mapping from definition id to use sites"""

    def __init__(self, ast: FuncDef | StmtBlock, reaching_defs: ReachingDefsAnalysis):
        self.ast = ast
        self.reaching_defs = reaching_defs
        self.uses = { d: set() for d, _ in enumerate(reaching_defs.defs) }

    def analyze(self):
        match self.ast:
            case FuncDef():
                self._visit_function(self.ast, None)
            case StmtBlock():
                self._visit_block(self.ast, None)
            case _:
                raise RuntimeError(f'unreachable case: {self.ast}')

        return DefineUseAnalysis(
            self.reaching_defs.defs,
            self.reaching_defs.name_to_defs,
            self.reaching_defs.in_defs,
            self.reaching_defs.out_defs,
            self.reaching_defs.reach,
            self.reaching_defs.phis,
            self.uses
        )

    def _add_use(self, name: NamedId, use: UseSite, ctx: DefCtx):
        d = ctx[name]
        self.uses[d].add(use)

    def _visit_var(self, e: Var, ctx: DefCtx):
        self._add_use(e.name, e, ctx)

    def _visit_call(self, e: Call, ctx: DefCtx):
        if e.fn is not None:
            match e.func:
                case NamedId():
                    self._add_use(e.func, e, ctx)
                case Attribute():
                    self._visit_expr(e.func, ctx)
                case _:
                    raise RuntimeError(f'unreachable: {e.func}')
        for arg in e.args:
            self._visit_expr(arg, ctx)
        for _, kwarg in e.kwargs:
            self._visit_expr(kwarg, ctx)

    def _visit_list_comp(self, e: ListComp, ctx: DefCtx):
        for iterable in e.iterables:
            self._visit_expr(iterable, ctx)
        ctx = ctx.copy()
        for target in e.targets:
            for name in target.names():
                ctx[name] = self.reaching_defs.find_def_from_site(name, e)
        self._visit_expr(e.elt, ctx)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: DefCtx):
        self._add_use(stmt.var, stmt, ctx)
        for slice in stmt.indices:
            self._visit_expr(slice, ctx)
        self._visit_expr(stmt.expr, ctx)
    
    def _visit_statement(self, stmt, ctx: DefCtx):
        ctx = self.reaching_defs.reach[stmt]
        return super()._visit_statement(stmt, ctx)


class DefineUse:
    """
    Definition-use analysis.

    This analysis computes:
    - the set of definitions available at the entry/exit of each block;
    - the set of definitions introduced at each statement;
    - set of all definitions for each variable;
    - set of all uses for each definition.
    - the definition referenced by each use.

    Each definition has a reference its immediate previous definition(s)
    which represents a (possibly cyclic) graph of definitions.
    """

    @staticmethod
    def analyze(ast: FuncDef | StmtBlock):
        if not isinstance(ast, FuncDef | StmtBlock):
            raise TypeError(f'Expected \'FuncDef\' or \'StmtBlock\', got {type(ast)} for {ast}')
        reaching_defs = ReachingDefs.analyze(ast)
        return _DefineUseInstance(ast, reaching_defs).analyze()
