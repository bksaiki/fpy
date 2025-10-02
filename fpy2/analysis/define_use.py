"""Definition-use analysis for FPy ASTs"""

from typing import TypeAlias

from ..ast.fpyast import *
from ..ast.visitor import DefaultVisitor
from ..utils import default_repr

from .reaching_defs import (
    ReachingDefs, ReachingDefsAnalysis,
    AssignDef, PhiDef, Definition, DefCtx, DefSite, PhiSite
)

__all__ = [
    'DefineUse',
    'DefineUseAnalysis',
    'AssignDef',
    'PhiDef',
    'Definition',
    'UseSite',
    'DefCtx',
    'DefSite',
    'PhiSite',
    'UseSite',
]

UseSite: TypeAlias = Var | IndexedAssign | Call
"""AST nodes that can use variables"""

@default_repr
class DefineUseAnalysis(ReachingDefsAnalysis):
    """Result of definition-use analysis."""

    uses: dict[Definition, set[UseSite]]
    """mapping from definition id to use sites"""

    use_to_def: dict[UseSite, Definition]
    """mapping from use site to definition"""

    def __init__(
        self,
        defs: list[Definition],
        name_to_defs: dict[NamedId, set[Definition]],
        in_defs: dict[StmtBlock, DefCtx],
        out_defs: dict[StmtBlock, DefCtx],
        reach: dict[Stmt, DefCtx],
        phis: dict[Stmt, set[PhiDef]],
        uses: dict[Definition, set[UseSite]]
    ):
        super().__init__(defs, name_to_defs, in_defs, out_defs, reach, phis)
        self.uses = uses
        # compute mapping from use to def
        self.use_to_def = {}
        for d, us in uses.items():
            for u in us:
                self.use_to_def[u] = d

    def names(self) -> set[NamedId]:
        """Returns the set of variable names with definitions."""
        return set(self.name_to_defs.keys())

    def format(self) -> str:
        lines: list[str] = []
        for name, ds in self.name_to_defs.items():
            lines.append(f'def `{name}`:')
            for d in ds:
                idx = self.def_to_idx[d]
                match d:
                    case AssignDef():
                        site = self._format_site(d.site.format())
                        prev = 'None' if d.prev is None else str(d.prev)
                        lines.append(f' {idx}: {d.name} [{prev}] @ {site}')
                    case PhiDef():
                        site = self._format_site(d.site.format())
                        lines.append(f' {idx}: {d.name} [phi({d.lhs}, {d.rhs})] @ {site}')
                    case _:
                        raise RuntimeError(f'unexpected definition {d}')
                lines.append(f' use `{name}`:')
                for u in self.uses[d]:
                    site = self._format_site(u.format())
                    lines.append(f'  - {site}')
        return '\n'.join(lines)

    def find_def_from_use(self, site: UseSite):
        """Finds the definition of a variable."""
        if site in self.use_to_def:
            return self.use_to_def[site]
        raise KeyError(f'no definition found for site {site}')


class _DefineUseInstance(DefaultVisitor):
    """Per-IR instance of definition-use analysis"""
    ast: FuncDef | StmtBlock
    reaching_defs: ReachingDefsAnalysis

    uses: dict[Definition, set[UseSite]] = {}
    """mapping from definition id to use sites"""

    def __init__(self, ast: FuncDef | StmtBlock, reaching_defs: ReachingDefsAnalysis):
        self.ast = ast
        self.reaching_defs = reaching_defs
        self.uses = { d: set() for d in reaching_defs.defs }

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
            self._visit_expr(e.func, ctx)
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

    def _visit_statement(self, stmt: Stmt, ctx: DefCtx):
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
