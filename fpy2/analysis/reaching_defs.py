"""
Reaching definitions analysis.
"""

from dataclasses import dataclass
from typing import Optional, TypeAlias, Union

from ..ast.fpyast import *
from ..ast.visitor import DefaultVisitor
from ..utils import Unionfind

from .defs import DefAnalysis


DefSite: TypeAlias = FuncDef | Argument | Assign | ForStmt | ContextStmt | ListComp
"""AST nodes that can define variables"""
PhiSite: TypeAlias = If1Stmt | IfStmt | WhileStmt | ForStmt
"""AST nodes that can introduce phi nodes"""
UseSite: TypeAlias = Var | IndexedAssign | Call
"""AST nodes that can use variables"""

def _format_site(s: str, width: int = 20) -> str:
    """Helper function to format site strings with 20 character limit and no newlines."""
    assert width > 3
    # Remove newlines and replace with spaces
    lines = s.splitlines()
    # Limit to 20 characters with ellipses if needed
    line = lines[0]
    if len(line) > width:
        return line[:width - 3] + '...'
    else:
        return line


@dataclass
class AssignDef:
    """A definition introduced by an assignment statement."""

    name: NamedId
    """the name of the variable being defined"""
    site: DefSite
    """the statement introducing the definition"""
    prev: Optional['Definition']

    def __hash__(self):
        return hash((self.name, self.site))

    def __eq__(self, other):
        return (
            isinstance(other, AssignDef)
            and self.name == other.name
            and self.site == other.site
        )

    def __lt__(self, other: 'AssignDef'):
        if not isinstance(other, AssignDef):
            raise TypeError(f"'<' not supported between instances '{type(self)}' and '{type(other)}'")
        return self.name < other.name

@dataclass
class PhiDef:
    """A definition introduced by merging two execution paths."""

    name: NamedId
    """the name of the variable being defined"""
    site: PhiSite
    """the statement introducing the phi node"""
    lhs: 'Definition'
    """first argument of the phi node"""
    rhs: 'Definition'
    """second argument of the phi node"""

    def __hash__(self):
        return hash((self.name, self.lhs, self.rhs))

    def __eq__(self, other):
        return (
            isinstance(other, PhiDef)
            and self.name == other.name
            and self.lhs == other.lhs
            and self.rhs == other.rhs
        )


Definition: TypeAlias = AssignDef | PhiDef
"""definition: either an assignment or a phi node"""

_DefCtx: TypeAlias = dict[NamedId, Definition]
"""visitor context: variable to current definition"""

@dataclass
class ReachingDefsAnalysis:
    """Result of reaching definitions analysis."""

    defs: dict[NamedId, set[Definition]]
    """mapping from variable names to all (re-)definitions"""
    in_defs: dict[StmtBlock, _DefCtx]
    """mapping from block to definitions available at entry"""
    out_defs: dict[StmtBlock, _DefCtx]
    """mapping from block to definitions available at exit"""
    phis: dict[Stmt, dict[NamedId, PhiDef]]
    """
    mapping from block to phi nodes at each statement:
    - for `If1` and `If`, the phi nodes are at the end of the block;
    - for `While` and `For`, the phi nodes are at the beginning of the block;
    """

    def format(self) -> str:
        lines: list[str] = []
        # print integer -> definition map
        indexed: dict[int, Definition] = {}
        inv_indexed: dict[Definition, int] = {}
        for name, ds in self.defs.items():
            lines.append(f'{name}:')
            for d in ds:
                i = len(indexed)
                indexed[i] = d
                inv_indexed[d] = i
                lines.append(f' {i}: {d.name} @ {_format_site(d.site.format())}')
        # print predecessors
        lines.append('pred:')
        for i in range(len(indexed)):
            d = indexed[i]
            match d:
                case AssignDef():
                    lines.append(f' {i} <- {d.prev and inv_indexed[d.prev]}')
                case PhiDef():
                    lhs = inv_indexed[d.lhs]
                    rhs = inv_indexed[d.rhs]
                    lines.append(f' {i} <- phi({lhs}, {rhs})')
                case _:
                    raise RuntimeError(f'unexpected definition {d}')
        return '\n'.join(lines)

class _ReachingDefs(DefaultVisitor):
    """Visitor for reaching definitions analysis."""

    ast: FuncDef | StmtBlock
    def_ids: dict[StmtBlock, set[NamedId]]

    defs: Unionfind[Definition]
    in_defs: dict[StmtBlock, _DefCtx]
    out_defs: dict[StmtBlock, _DefCtx]
    phis: dict[Stmt, dict[NamedId, PhiDef]]

    def __init__(self, ast: FuncDef | StmtBlock, def_ids: dict[StmtBlock, set[NamedId]]):
        self.ast = ast
        self.def_ids = def_ids
        self.defs = Unionfind()
        self.in_defs = {}
        self.out_defs = {}
        self.phis = {}

    def _add_def(self, d: Definition) -> Definition:
        """Add a definition to the union-find structure."""
        return self.defs.add(d)

    def _add_assign(self, name: NamedId, site: DefSite, ctx: _DefCtx) -> tuple[AssignDef, _DefCtx]:
        """Adds a new assignment definition, returning the new context."""
        d = AssignDef(name, site, ctx.get(name))
        new_ctx = ctx.copy()
        new_ctx[name] = self._add_def(d)
        return d, new_ctx

    def _add_phi(self, name: NamedId, site: PhiSite, lhs: Definition, rhs: Definition, ctx: _DefCtx) -> tuple[PhiDef, _DefCtx]:
        """Adds a new phi node definition, returning the new context."""
        d = PhiDef(name, site, lhs, rhs)
        new_ctx = ctx.copy()
        new_ctx[name] = self._add_def(d)
        return d, new_ctx

    def _unify_def(self, d1: Definition, d2: Definition) -> Definition:
        """
        Unify two definitions in the union-find structure.
        The representative is the representative of `d1`.
        """
        return self.defs.union(d1, d2)

    def _visit_list_comp(self, e: ListComp, ctx: _DefCtx):
        for target, arg in zip(e.targets, e.iterables):
            self._visit_expr(arg, ctx)
            for name in target.names():
                _, ctx = self._add_assign(name, e, ctx)
        self._visit_expr(e.elt, ctx)

    def _visit_assign(self, stmt: Assign, ctx: _DefCtx):
        # visit expression and introduce new definitions
        self._visit_expr(stmt.expr, ctx)
        for name in stmt.target.names():
            _, ctx = self._add_assign(name, stmt, ctx)
        return ctx

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: _DefCtx):
        for index in stmt.indices:
            self._visit_expr(index, ctx)
        self._visit_expr(stmt.expr, ctx)
        return ctx

    def _visit_if1(self, stmt: If1Stmt, ctx: _DefCtx):
        # visit condition
        self._visit_expr(stmt.cond, ctx)
        # visit body
        body_out = self._visit_block(stmt.body, ctx)
        # introduce phi nodes for any definitions that are re-defined in the body
        phis: dict[NamedId, PhiDef] = {}
        for name, orig in ctx.items():
            if orig != body_out[name]:
                phi, ctx = self._add_phi(name, stmt, orig, body_out[name], ctx)
                phis[name] = phi
        self.phis[stmt] = phis
        return ctx

    def _visit_if(self, stmt: IfStmt, ctx: _DefCtx):
        # visit condition
        self._visit_expr(stmt.cond, ctx)
        # visit both true and false branches
        ift_out = self._visit_block(stmt.ift, ctx)
        iff_out = self._visit_block(stmt.iff, ctx)
        # introduce phi nodes for:
        # (i) redefinitions in the branches
        # (ii) introductions in both branches
        phis: dict[NamedId, PhiDef] = {}
        for name in ift_out.keys() & iff_out.keys():
            d_ift = ift_out[name]
            d_iff = iff_out[name]
            if d_ift != d_iff:
                phi, ctx = self._add_phi(name, stmt, d_ift, d_iff, ctx)
                phis[name] = phi
        # record the phi nodes and return the updated context
        self.phis[stmt] = phis
        return ctx

    def _visit_while(self, stmt: WhileStmt, ctx: _DefCtx):
        # visit condition
        self._visit_expr(stmt.cond, ctx)
        # create (temporary) phi nodes for any mutated variable
        body_in = ctx.copy()
        mutated = ctx.keys() & self.def_ids[stmt.body]
        for intro in mutated:
            # create (temporary) phi node `x' = phi(x, x)`
            _, body_in = self._add_phi(intro, stmt, ctx[intro], ctx[intro], body_in)
        # visit body
        body_out = self._visit_block(stmt.body, body_in)
        # update phi nodes for any definitions that are re-defined in the body
        phis: dict[NamedId, PhiDef] = {}
        phi_lhs = { name: ctx[name] for name in mutated }
        for name in mutated:
            # create actual phi node `x' = phi(x, x'')` where
            # `x` is on entry and `x''` is after the loop body
            phi, ctx = self._add_phi(name, stmt, phi_lhs[name], body_out[name], ctx)
            self._unify_def(phi, body_in[name])
            phis[name] = phi
        # record the phi nodes and return the updated context
        self.phis[stmt] = phis
        return ctx

    def _visit_for(self, stmt: ForStmt, ctx: _DefCtx):
        raise NotImplementedError

    def _visit_context(self, stmt: ContextStmt, ctx: _DefCtx):
        # visit context expression and possibly introduce a binding
        self._visit_expr(stmt.ctx, ctx)
        if isinstance(stmt.target, NamedId):
            _, ctx = self._add_assign(stmt.target, stmt, ctx)
        # visit the body and continue with the resulting environment
        return self._visit_block(stmt.body, ctx)

    def _visit_assert(self, stmt: AssertStmt, ctx: _DefCtx):
        self._visit_expr(stmt.test, ctx)
        return ctx

    def _visit_effect(self, stmt: EffectStmt, ctx: _DefCtx):
        self._visit_expr(stmt.expr, ctx)
        return ctx

    def _visit_return(self, stmt: ReturnStmt, ctx: _DefCtx):
        self._visit_expr(stmt.expr, ctx)
        return ctx

    def _visit_pass(self, stmt: PassStmt, ctx: _DefCtx):
        return ctx

    def _visit_block(self, block: StmtBlock, ctx: _DefCtx) -> _DefCtx:
        self.in_defs[block] = ctx
        for stmt in block.stmts:
            ctx = self._visit_statement(stmt, ctx)
        self.out_defs[block] = ctx
        return ctx

    def _visit_function(self, func: FuncDef, ctx: _DefCtx):
        # process named arguments
        for arg in func.args:
            if isinstance(arg.name, NamedId):
                _, ctx = self._add_assign(arg.name, arg, ctx)
        # process free variables
        for v in func.free_vars:
            _, ctx = self._add_assign(v, func, ctx)
        # visit body
        self._visit_block(func.body, ctx)

    def _normalize_def(self, d: Definition) -> Definition:
        d = self.defs.find(d)
        match d:
            case AssignDef():
                if d.prev is not None:
                    d_prev = self._normalize_def(d.prev)
                    new_d = self._unify_def(d, AssignDef(d.name, d.site, d_prev))

                    if d_prev != d.prev:
                        d = self._unify_def(d, AssignDef(d.name, d.site, d_prev))


    def _normalize(self):
        # definitions may refer to non-representative definitions
        reprs: dict[Definition, Definition] = {}
        for d in self.defs:
            self._normalize_def(d, reprs)

        # the representatives are the values of `reprs`
        defs: dict[NamedId, set[Definition]] = {}
        for d in reprs.values():
            name = d.name
            if name not in defs:
                defs[name] = set()
            defs[name].add(d)

        # normalize the IN and OUT sets
        for block, ctx in self.in_defs.items():
            self.in_defs[block] = {name: reprs[d] for name, d in ctx.items()}
        for block, ctx in self.out_defs.items():
            self.out_defs[block] = {name: reprs[d] for name, d in ctx.items()}

        # normalize the phi nodes
        for stmt, phi_map in self.phis.items():
            new_phis: dict[NamedId, PhiDef] = {}
            for name, d in phi_map.items():
                d = reprs[d]
                assert isinstance(d, PhiDef), f'expected a phi node {d}'
                new_phis[name] = d
            self.phis[stmt] = new_phis

        return ReachingDefsAnalysis(defs, self.in_defs, self.out_defs, self.phis)


    def analyze(self) -> ReachingDefsAnalysis:
        match self.ast:
            case FuncDef():
                self._visit_function(self.ast, {})
            case StmtBlock():
                self._visit_block(self.ast, {})
            case _:
                raise RuntimeError(f'unexpected AST node {self.ast}')
        return self._normalize()



class ReachingDefs:
    """
    Reaching definitions analysis.

    This analysis computes:
    - the set of definitions available at the entry/exit of each block;
    - the set of definitions introduced at each statement;
    - set of all definitions for each variable;

    Each definition has a reference its immediate previous definition(s)
    which represents a chain of definitions.
    """

    @staticmethod
    def analyze(ast: FuncDef | StmtBlock):
        if not isinstance(ast, FuncDef | StmtBlock):
            raise RuntimeError(f'unexpected AST node {ast}')
        def_ids = DefAnalysis.analyze(ast)
        return _ReachingDefs(ast, def_ids).analyze()
