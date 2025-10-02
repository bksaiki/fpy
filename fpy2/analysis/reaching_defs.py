"""
Reaching definitions analysis.
"""

from dataclasses import dataclass
from typing import TypeAlias

from ..ast.fpyast import *
from ..ast.visitor import DefaultVisitor
from ..utils import Unionfind, default_repr

from .defs import DefAnalysis

__all__ = [
    'ReachingDefs',
    'ReachingDefsAnalysis',
    'AssignDef',
    'PhiDef',
    'Definition',
    'DefCtx',
    'DefSite',
    'PhiSite',
]


DefSite: TypeAlias = FuncDef | Argument | Assign | ForStmt | ContextStmt | ListComp
"""AST nodes that can define variables"""
PhiSite: TypeAlias = If1Stmt | IfStmt | WhileStmt | ForStmt
"""AST nodes that can introduce phi nodes"""


@dataclass
class AssignDef:
    """A definition introduced by an assignment statement."""

    name: NamedId
    """the name of the variable being defined"""
    site: DefSite
    """the statement introducing the definition"""
    prev: int | None
    """index of the previous definition, or `None` if this is the first definition"""
    is_free: bool
    """is this definition a free variable?"""

    def __hash__(self):
        return hash((self.name, self.site, self.prev, self.is_free))

    def __eq__(self, other):
        return (
            isinstance(other, AssignDef)
            and self.name == other.name
            and self.site == other.site
            and self.prev == other.prev
            and self.is_free == other.is_free
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
    lhs: int
    """first argument of the phi node"""
    rhs: int
    """second argument of the phi node"""
    is_intro: bool
    """relevant for `If`: is this a variable introduced by (both) branches?"""

    def __hash__(self):
        return hash((self.name, self.site, self.lhs, self.rhs, self.is_intro))

    def __eq__(self, other):
        return (
            isinstance(other, PhiDef)
            and self.name == other.name
            and self.site == other.site
            and self.lhs == other.lhs
            and self.rhs == other.rhs
            and self.is_intro == other.is_intro
        )


Definition: TypeAlias = AssignDef | PhiDef
"""definition: either an assignment or a phi node"""

DefCtx: TypeAlias = dict[NamedId, Definition]
"""mapping from variable name to definition"""

@default_repr
class ReachingDefsAnalysis:
    """Result of reaching definitions analysis."""

    defs: list[Definition]
    """list of all definitions"""
    name_to_defs: dict[NamedId, set[Definition]]
    """mapping from variable names to all (re-)definitions"""

    in_defs: dict[StmtBlock, DefCtx]
    """mapping from block to definitions available at entry"""
    out_defs: dict[StmtBlock, DefCtx]
    """mapping from block to definitions available at exit"""
    reach: dict[Stmt, DefCtx]
    """mapping from statement to definitions reaching that statement"""
    phis: dict[Stmt, set[PhiDef]]
    """
    mapping from block to phi nodes at each statement:
    - for `If1` and `If`, the phi nodes are at the end of the block;
    - for `While` and `For`, the phi nodes are at the beginning of the block;
    """

    def_to_idx: dict[Definition, int]
    """mapping from definition to its index"""
    site_to_def: dict[tuple[NamedId, DefSite], AssignDef]
    """mapping from (name, site) to definition"""
    successors: dict[Definition, set[Definition]]
    """mapping from definition to its successor definitions"""

    def __init__(
        self,
        defs: list[Definition],
        name_to_defs: dict[NamedId, set[Definition]],
        in_defs: dict[StmtBlock, DefCtx],
        out_defs: dict[StmtBlock, DefCtx],
        reach: dict[Stmt, DefCtx],
        phis: dict[Stmt, set[PhiDef]]
    ):
        self.defs = defs
        self.name_to_defs = name_to_defs
        self.in_defs = in_defs
        self.out_defs = out_defs
        self.reach = reach
        self.phis = phis
        # invert mapping from index to definition
        self.def_to_idx = { d: i for i, d in enumerate(defs) }
        # compute mapping from (name, site) to definition
        self.site_to_def = {}
        for name, ds in name_to_defs.items():
            for d in ds:
                if isinstance(d, AssignDef):
                    self.site_to_def[(name, d.site)] = d
        # compute mapping from definition to its successors
        self.successors = { d: set() for d in defs }
        for d in defs:
            match d:
                case AssignDef():
                    if d.prev is not None:
                        self.successors[self.defs[d.prev]].add(d)
                case PhiDef():
                    self.successors[self.defs[d.lhs]].add(d)
                    self.successors[self.defs[d.rhs]].add(d)
                case _:
                    raise RuntimeError(f'unexpected definition {d}')

    def _format_site(self, s: str, width: int = 20) -> str:
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

    def names(self) -> set[NamedId]:
        """Returns the set of variable names with definitions."""
        return set(self.name_to_defs.keys())

    def format(self) -> str:
        lines: list[str] = []
        for name, ds in self.name_to_defs.items():
            lines.append(f'{name}:')
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
        return '\n'.join(lines)

    def roots_of(self, d: Definition) -> set[AssignDef]:
        """
        Returns the set of root definitions (i.e., assignments with no predecessors)
        that contribute to this definition.
        """
        match d:
            case AssignDef():
                if d.prev is None:
                    return {d}
                else:
                    return self.roots_of(self.defs[d.prev])
            case PhiDef():
                return self.roots_of(self.defs[d.lhs]) | self.roots_of(self.defs[d.rhs])
            case _:
                raise RuntimeError(f'unexpected definition {d}')

    def phi_prevs(self, d: PhiDef) -> set[PhiDef]:
        """
        Returns the set of phi nodes that are immediate predecessors
        of this definition.
        """
        lhs = self.defs[d.lhs]
        rhs = self.defs[d.rhs]

        phis: set[PhiDef] = set()
        if isinstance(lhs, PhiDef):
            phis.add(lhs)
            phis |= self.phi_prevs(lhs)
        if isinstance(rhs, PhiDef):
            phis.add(rhs)
            phis |= self.phi_prevs(rhs)
        return phis

    def find_def_from_site(self, name: NamedId, site: DefSite) -> AssignDef:
        """Finds the definition of given a (name, site) pair."""
        key = (name, site)
        if key in self.site_to_def:
            return self.site_to_def[key]
        raise KeyError(f'no definition found for {name} at {site}')

    def mutated_in(self, block: StmtBlock) -> set[NamedId]:
        """Returns the set of variables mutated in the given block."""
        names: set[NamedId] = set()
        in_ctx = self.in_defs[block]
        out_ctx = self.out_defs[block]
        for name in in_ctx.keys() & out_ctx.keys():
            if in_ctx[name] != out_ctx[name]:
                names.add(name)
        return names

    def introed_in(self, block: StmtBlock) -> set[NamedId]:
        """Returns the set of variables freshly defined in the given block."""
        in_ctx = self.in_defs[block]
        out_ctx = self.out_defs[block]
        return set(out_ctx.keys() - in_ctx.keys())


_DefCtx: TypeAlias = dict[NamedId, int]
"""visitor context: mapping from variable name to definition index"""

class _ReachingDefs(DefaultVisitor):
    """Visitor for reaching definitions analysis."""

    ast: FuncDef | StmtBlock
    def_ids: dict[StmtBlock, set[NamedId]]

    indices: Unionfind[int]
    idx_to_def: dict[int, Definition]
    def_to_idx: dict[Definition, int]
    in_defs: dict[StmtBlock, _DefCtx]
    out_defs: dict[StmtBlock, _DefCtx]
    reach: dict[Stmt, _DefCtx]
    phis: dict[Stmt, dict[NamedId, int]]

    def __init__(self, ast: FuncDef | StmtBlock, def_ids: dict[StmtBlock, set[NamedId]]):
        self.ast = ast
        self.def_ids = def_ids

        self.indices = Unionfind()
        self.idx_to_def = {}
        self.def_to_idx = {}
        self.in_defs = {}
        self.out_defs = {}
        self.reach = {}
        self.phis = {}

    def _add_def(self, d: Definition) -> int:
        """Add a definition to the union-find structure."""
        assert d not in self.def_to_idx, f'definition {d} already added'
        i = len(self.idx_to_def)
        self.idx_to_def[i] = d
        self.def_to_idx[d] = i
        self.indices.add(i)
        return i

    def _add_assign(self, name: NamedId, site: DefSite, ctx: _DefCtx, is_free: bool = False) -> tuple[int, _DefCtx]:
        """Adds a new assignment definition, returning the new context."""
        prev = ctx.get(name, None)
        d = AssignDef(name, site, prev, is_free)
        idx = self._add_def(d)
        new_ctx = ctx.copy()
        new_ctx[name] = idx
        return idx, new_ctx

    def _add_phi(self, name: NamedId, site: PhiSite, lhs: int, rhs: int, ctx: _DefCtx) -> tuple[int, _DefCtx]:
        """Adds a new phi node definition, returning the new context."""
        d = PhiDef(name, site, lhs, rhs, name not in ctx)
        idx = self._add_def(d)
        new_ctx = ctx.copy()
        new_ctx[name] = idx
        return idx, new_ctx

    def _unify_def(self, i1: int, i2: int):
        """
        Unify two definitions in the union-find structure.
        The representative is the representative of `d1`.
        """
        return self.indices.union(i1, i2)

    def _visit_list_comp(self, e: ListComp, ctx: _DefCtx):
        for iterable in e.iterables:
            self._visit_expr(iterable, ctx)
        for target in e.targets:
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
        phis: _DefCtx = {}
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
        phis: _DefCtx = {}
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
        # create (temporary) phi nodes for any mutated variable
        body_in = ctx.copy()
        mutated = ctx.keys() & self.def_ids[stmt.body]
        for intro in mutated:
            # create (temporary) phi node `x' = phi(x, x)`
            _, body_in = self._add_phi(intro, stmt, ctx[intro], ctx[intro], body_in)
        # visit condition and body
        self._visit_expr(stmt.cond, body_in)
        body_out = self._visit_block(stmt.body, body_in)
        # update phi nodes for any definitions that are re-defined in the body
        phis: _DefCtx = {}
        for name in mutated:
            # create actual phi node `x' = phi(x, x'')` where
            # `x` is on entry and `x''` is after the loop body
            phi, ctx = self._add_phi(name, stmt, ctx[name], body_out[name], ctx)
            phis[name] = self._unify_def(phi, body_in[name])
        # record the phi nodes and return the updated context
        self.phis[stmt] = phis
        return ctx

    def _visit_for(self, stmt: ForStmt, ctx: _DefCtx):
        # visit iterable expression
        self._visit_expr(stmt.iterable, ctx)
        # create (temporary) phi nodes for any mutated variable
        body_in = ctx.copy()
        mutated = ctx.keys() & self.def_ids[stmt.body]
        for intro in mutated:
            # create (temporary) phi node `x' = phi(x, x)`
            _, body_in = self._add_phi(intro, stmt, ctx[intro], ctx[intro], body_in)
        # introduce new definition for the loop variable
        for name in stmt.target.names():
            _, body_in = self._add_assign(name, stmt, body_in)
        # visit body
        body_out = self._visit_block(stmt.body, body_in)
        # update phi nodes for any definitions that are re-defined in the body
        phis: _DefCtx = {}
        for name in mutated:
            # create actual phi node `x' = phi(x, x'')` where
            # `x` is on entry and `x''` is after the loop body
            phi, ctx = self._add_phi(name, stmt, ctx[name], body_out[name], ctx)
            phis[name] = self._unify_def(phi, body_in[name])
        # record the phi nodes and return the updated context
        self.phis[stmt] = phis
        return ctx

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
            self.reach[stmt] = ctx
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
            _, ctx = self._add_assign(v, func, ctx, is_free=True)
        # visit body
        self._visit_block(func.body, ctx)

    def _normalize(self):
        # create a map from indices to representative indices
        #  index -> canonical index -> representative index (0, 1, 2, ..., N-1)
        repr_count = 0
        repr_indices: dict[int, int] = {}
        for idx, c_idx in self.indices.items():
            if c_idx not in repr_indices:
                repr_indices[c_idx] = repr_count
                repr_count += 1
            repr_indices[idx] = repr_indices[c_idx]

        # using representative indices, create representative definitions
        idx_to_repr: dict[int, Definition] = {}
        repr_to_idx: dict[Definition, int] = {}
        for idx in self.indices.representatives():
            d = self.idx_to_def[idx]
            match d:
                case AssignDef():
                    prev = None if d.prev is None else repr_indices[d.prev]
                    d = AssignDef(d.name, d.site, prev, d.is_free)
                case PhiDef():
                    lhs = repr_indices[d.lhs]
                    rhs = repr_indices[d.rhs]
                    d = PhiDef(d.name, d.site, lhs, rhs, d.is_intro)
                case _:
                    raise RuntimeError(f'unexpected definition {d}')

            repr_idx = repr_indices[idx]
            idx_to_repr[repr_idx] = d
            repr_to_idx[d] = repr_idx

        # build list of all representative definitions
        defs = [idx_to_repr[i] for i, _ in enumerate(idx_to_repr)]

        # rebuild map from variable names to all (re-)definitions
        name_to_defs: dict[NamedId, set[Definition]] = {}
        for d in repr_to_idx:
            if d.name not in name_to_defs:
                name_to_defs[d.name] = set()
            name_to_defs[d.name].add(d)

        # rebuild map for IN definitions
        in_defs: dict[StmtBlock, DefCtx] = {}
        for block, ctx in self.in_defs.items():
            in_defs[block] = { 
                name: idx_to_repr[repr_indices[idx]]
                for name, idx in ctx.items()
            }

        # rebuild map for OUT definitions
        out_defs: dict[StmtBlock, DefCtx] = {}
        for block, ctx in self.out_defs.items():
            out_defs[block] = {
                name: idx_to_repr[repr_indices[idx]]
                for name, idx in ctx.items()
            }

        # rebuild map for reaching definitions
        reach: dict[Stmt, DefCtx] = {}
        for stmt, ctx in self.reach.items():
            reach[stmt] = { 
                name: idx_to_repr[repr_indices[idx]]
                for name, idx in ctx.items()
            }

        # rebuild map for phi nodes
        phis: dict[Stmt, set[PhiDef]] = {}
        for stmt, ctx in self.phis.items():
            new_phis: set[PhiDef] = set()
            for _, idx in ctx.items():
                d = idx_to_repr[repr_indices[idx]]
                assert isinstance(d, PhiDef), f'expected phi node, got {d}'
                new_phis.add(d)
            phis[stmt] = new_phis

        return ReachingDefsAnalysis(defs, name_to_defs, in_defs, out_defs, reach, phis)

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
    which represents a (possibly cyclic) graph of definitions.
    """

    @staticmethod
    def analyze(ast: FuncDef | StmtBlock):
        if not isinstance(ast, FuncDef | StmtBlock):
            raise RuntimeError(f'unexpected AST node {ast}')
        def_ids = DefAnalysis.analyze(ast)
        return _ReachingDefs(ast, def_ids).analyze()
