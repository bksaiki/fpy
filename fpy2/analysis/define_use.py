"""Definition use analysis for FPy ASTs"""

from abc import ABC, abstractmethod
from typing import TypeAlias, cast

from ..ast.fpyast import *
from ..ast.visitor import DefaultVisitor
from ..utils import default_repr

DefSite: TypeAlias = FuncDef | Argument | Stmt | ListComp
UseSite: TypeAlias = Var | IndexedAssign


@default_repr
class Definition(ABC):
    """
    Definition of a variable:
    - an assignment
    - merging definitions from two branches
    """

    name: NamedId

    def __init__(self, name: NamedId):
        self.name = name

    @abstractmethod
    def phis(self) -> set['PhiDef']:
        """Returns all phi nodes defined or redefined by this definition."""
        ...

    @abstractmethod
    def assigns(self) -> set['AssignDef']:
        """Returns all assignment sites defined or redefined by this definition."""
        ...


class AssignDef(Definition):
    """Concrete definition for an assignment."""

    site: DefSite
    """syntax location of the assignment"""

    parent: Definition | None
    """previous definition (if redefining a variable)"""

    def __init__(self, name: NamedId, site: DefSite, parent: Definition | None = None):
        super().__init__(name)
        self.site = site
        self.parent = parent

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

    def __hash__(self):
        return hash((self.name, self.site))

    def phis(self) -> set['PhiDef']:
        return set()

    def assigns(self) -> set['AssignDef']:
        return { self }


class PhiDef(Definition):
    """
    Merged definition from multiple branches (phi node in SSA form)
    """
    lhs: Definition
    rhs: Definition
    is_new: bool

    def __init__(self, name: NamedId, lhs: Definition, rhs: Definition, is_new: bool = False):
        super().__init__(name)
        self.lhs = lhs
        self.rhs = rhs
        self.is_new = is_new

    def __eq__(self, other):
        return (
            isinstance(other, PhiDef)
            and self.name == other.name
            and self.lhs == other.lhs
            and self.rhs == other.rhs
            and self.is_new == other.is_new
        )

    def __hash__(self):
        return hash((self.name, self.lhs, self.rhs, self.is_new))

    @staticmethod
    def union(lhs: Definition, rhs: Definition) -> Definition:
        """
        Create a phi definition from a set of definitions.
        If the first definition is a PhiDef, it is returned as is.
        """
        if lhs.name != rhs.name:
            raise ValueError(f'names must match: {lhs.name} != {rhs.name}')

        if lhs == rhs:
            return lhs
        else:
            return PhiDef(lhs.name, lhs, rhs)

    def phis(self) -> set['PhiDef']:
        return { self } | self.lhs.phis() | self.rhs.phis()

    def assigns(self) -> set['AssignDef']:
        return self.lhs.assigns() | self.rhs.assigns()


class DefinitionCtx(dict[NamedId, Definition]):
    """Mapping from variable to its definition (or possible definitions)."""

    def copy(self) -> 'DefinitionCtx':
        """Returns a shallow copy of the context."""
        return DefinitionCtx(self)

    def mutated_in(self, other: 'DefinitionCtx') -> list[NamedId]:
        """
        Returns the set of variables that are defined in `self`
        and mutated in `other`.
        """
        names: list[NamedId] = []
        for name in self.keys() & other.keys():
            if self[name] != other[name]:
                names.append(name)
        return names

    def fresh_in(self, other: 'DefinitionCtx') -> set[NamedId]:
        """
        Returns the set of variables that are defined in `other`
        but not in `self`.
        """
        return set(other.keys() - self.keys())


@default_repr
class DefineUseAnalysis:
    """Result of definition-use analysis"""
    defs: dict[NamedId, set[Definition]]
    uses: dict[Definition, set[UseSite]]
    stmts: dict[Stmt, tuple[DefinitionCtx, DefinitionCtx]]
    blocks: dict[StmtBlock, tuple[DefinitionCtx, DefinitionCtx]]
    phis: dict[Stmt, set[PhiDef]]

    def __init__(self):
        self.defs = {}
        self.uses = {}
        self.stmts = {}
        self.blocks = {}
        self.phis = {}

    @property
    def names(self) -> set[NamedId]:
        """Returns the set of all variable names in the analysis"""
        return set(self.defs.keys())

    def find_def_from_site(self, name: NamedId, site: DefSite) -> AssignDef:
        """Finds the definition of given a (name, site) pair."""
        defs = self.defs.get(name, set())
        for d in defs:
            if isinstance(d, AssignDef) and d.site == site:
                return d
        raise KeyError(f'no definition found for {name} at {site}')

    def find_def_from_use(self, site: UseSite):
        """Finds the definition of a variable."""
        # TODO: make more efficient: build inverse map?
        for d in self.uses:
            if site in self.uses[d]:
                return d
        raise KeyError(f'no definition found for site {site}')

class _DefineUseInstance(DefaultVisitor):
    """Per-IR instance of definition-use analysis"""
    ast: FuncDef | StmtBlock
    analysis: DefineUseAnalysis

    def __init__(self, ast: FuncDef | StmtBlock):
        self.ast = ast
        self.analysis = DefineUseAnalysis()

    def analyze(self):
        match self.ast:
            case FuncDef():
                self._visit_function(self.ast, DefinitionCtx())
            case StmtBlock():
                self._visit_block(self.ast, DefinitionCtx())
            case _:
                raise RuntimeError(f'unreachable case: {self.ast}')
        return self.analysis

    def _add_def(self, name: NamedId, d: Definition) -> Definition:
        if name not in self.analysis.defs:
            self.analysis.defs[name] = set()
        self.analysis.defs[name].add(d)
        self.analysis.uses[d] = set()
        return d

    def _add_assign(self, name: NamedId, site: DefSite, ctx: DefinitionCtx) -> AssignDef:
        pred = ctx.get(name)
        d = AssignDef(name, site, pred)
        return cast(AssignDef, self._add_def(name, d))

    def _add_use(self, name: NamedId, use: Var | IndexedAssign, ctx: DefinitionCtx):
        d = ctx[name]
        self.analysis.uses[d].add(use)

    def _visit_var(self, e: Var, ctx: DefinitionCtx):
        if e.name not in ctx:
            raise NotImplementedError(f'undefined variable {e.name}')
        self._add_use(e.name, e, ctx)

    def _visit_list_comp(self, e: ListComp, ctx: DefinitionCtx):
        for iterable in e.iterables:
            self._visit_expr(iterable, ctx)
        ctx = ctx.copy()
        for target in e.targets:
            for name in target.names():
                ctx[name] = self._add_assign(name, e, ctx)
        self._visit_expr(e.elt, ctx)

    def _visit_assign(self, stmt: Assign, ctx: DefinitionCtx):
        self._visit_expr(stmt.expr, ctx)
        for var in stmt.target.names():
            ctx[var] = self._add_assign(var, stmt, ctx)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: DefinitionCtx):
        self._add_use(stmt.var, stmt, ctx)
        for slice in stmt.indices:
            self._visit_expr(slice, ctx)
        self._visit_expr(stmt.expr, ctx)

    def _visit_if1(self, stmt: If1Stmt, ctx: DefinitionCtx):
        self._visit_expr(stmt.cond, ctx)
        body_ctx = self._visit_block(stmt.body, ctx.copy())
        # merge contexts along both paths
        # definitions cannot be introduced in the body
        self.analysis.phis[stmt] = set()
        for var in ctx:
            d_stmt = ctx[var]
            d_body = body_ctx[var]
            d = PhiDef.union(d_stmt, d_body)
            # update tables if the definition is new
            if d != d_stmt and d != d_body:
                assert isinstance(d, PhiDef)
                self.analysis.phis[stmt].add(d)
                self._add_def(var, d)
            ctx[var] = d

    def _visit_if(self, stmt: IfStmt, ctx: DefinitionCtx):
        self._visit_expr(stmt.cond, ctx)
        ift_ctx = self._visit_block(stmt.ift, ctx.copy())
        iff_ctx = self._visit_block(stmt.iff, ctx.copy())
        # merge contexts along both paths
        self.analysis.phis[stmt] = set()
        for var in ift_ctx.keys() & iff_ctx.keys():
            d_ift = ift_ctx[var]
            d_iff = iff_ctx[var]
            d = PhiDef.union(d_ift, d_iff)
            # update tables if the definition is new
            if d != d_ift and d != d_iff:
                assert isinstance(d, PhiDef)
                d.is_new = var not in ctx
                self.analysis.phis[stmt].add(d)
                self._add_def(var, d)
            ctx[var] = d

    def _visit_while(self, stmt: WhileStmt, ctx: DefinitionCtx):
        self._visit_expr(stmt.cond, ctx)
        body_ctx = self._visit_block(stmt.body, ctx.copy())
        # merge contexts along both paths
        # definitions cannot be introduced in the body
        self.analysis.phis[stmt] = set()
        for var in ctx:
            d_stmt = ctx[var]
            d_body = body_ctx[var]
            d = PhiDef.union(d_stmt, d_body)
            # update tables if the definition is new
            if d != d_stmt and d != d_body:
                assert isinstance(d, PhiDef)
                self.analysis.phis[stmt].add(d)
                self._add_def(var, d)
            ctx[var] = d

    def _visit_for(self, stmt: ForStmt, ctx: DefinitionCtx):
        self._visit_expr(stmt.iterable, ctx)
        body_ctx = ctx.copy()
        for var in stmt.target.names():
            body_ctx[var] = self._add_assign(var, stmt, body_ctx)

        body_ctx = self._visit_block(stmt.body, body_ctx)
        # merge contexts along both paths
        # definitions cannot be introduced in the body
        self.analysis.phis[stmt] = set()
        for var in ctx:
            d_stmt = ctx[var]
            d_body = body_ctx[var]
            d = PhiDef.union(d_stmt, d_body)
            # update tables if the definition is new
            if d != d_stmt and d != d_body:
                assert isinstance(d, PhiDef)
                self.analysis.phis[stmt].add(d)
                self._add_def(var, d)
            ctx[var] = d

    def _visit_statement(self, stmt: Stmt, ctx: DefinitionCtx):
        ctx_in = ctx.copy()
        super()._visit_statement(stmt, ctx)
        self.analysis.stmts[stmt] = (ctx_in, ctx.copy())

    def _visit_block(self, block: StmtBlock, ctx: DefinitionCtx):
        ctx_in = ctx.copy()
        for stmt in block.stmts:
            self._visit_statement(stmt, ctx)
        self.analysis.blocks[block] = (ctx_in, ctx.copy())
        return ctx

    def _visit_function(self, func: FuncDef, ctx: DefinitionCtx):
        for arg in func.args:
            if isinstance(arg.name, NamedId):
                ctx[arg.name] = self._add_assign(arg.name, arg, ctx)
        for v in func.free_vars:
            ctx[v] = self._add_assign(v, func, ctx)
        self._visit_block(func.body, ctx.copy())


class DefineUse:
    """
    Definition-use analyzer for the FPy IR.

    Computes definition-use chains for each variable.

    name ---> definition ---> use1, use2, ...
         ---> definition ---> use1, use2, ...
         ...
    """

    @staticmethod
    def analyze(ast: FuncDef | StmtBlock):
        if not isinstance(ast, FuncDef | StmtBlock):
            raise TypeError(f'Expected \'FuncDef\' or \'StmtBlock\', got {type(ast)} for {ast}')
        return _DefineUseInstance(ast).analyze()
