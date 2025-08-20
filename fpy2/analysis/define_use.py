"""Definition use analysis for FPy ASTs"""

from abc import ABC, abstractmethod
from typing import Iterable, TypeAlias

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
    def assigns(self) -> list['AssignDef']:
        """Returns the set of concrete assignments that for this variable."""
        ...


class AssignDef(Definition):
    """Concrete definition for an assignment."""

    site: DefSite
    """syntax location of the assignment"""

    def __init__(self, name: NamedId, site: DefSite):
        super().__init__(name)
        self.site = site

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

    def assigns(self) -> list['AssignDef']:
        return [self]


class PhiDef(Definition):
    """Merged definition from multiple branches (phi node in SSA form)"""

    children: tuple[AssignDef, ...]

    def __init__(self, name: NamedId, defs: Iterable[Definition]):
        children: list[AssignDef] = []
        for d in defs:
            children.extend(d.assigns())

        super().__init__(name)
        self.children = tuple(children)

    def __eq__(self, other):
        return (
            isinstance(other, PhiDef)
            and self.name == other.name
            and self.children == other.children
        )

    def __hash__(self):
        return hash((self.name, self.children))

    def assigns(self) -> list['AssignDef']:
        return list(self.children)

    @staticmethod
    def union(first: Definition, *others: Definition) -> Definition:
        """
        Create a phi definition from a set of definitions.
        If the first definition is a PhiDef, it is returned as is.
        """
        name = first.name
        uniq_assigns = set(first.assigns())
        for other in others:
            if name != other.name:
                raise ValueError(f'cannot union definitions with different names {other}')
            uniq_assigns.update(other.assigns())

        assigns = sorted(uniq_assigns)
        if len(assigns) == 0:
            raise ValueError('cannot create a phi definition from an empty set of definitions')
        elif len(assigns) == 1:
            return assigns[0]
        else:
            return PhiDef(name, assigns)


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
    defs: dict[NamedId, set[AssignDef]]
    uses: dict[AssignDef, set[UseSite]]
    stmts: dict[Stmt, tuple[DefinitionCtx, DefinitionCtx]]
    blocks: dict[StmtBlock, tuple[DefinitionCtx, DefinitionCtx]]

    def __init__(self):
        self.defs = {}
        self.uses = {}
        self.stmts = {}
        self.blocks = {}

    @property
    def names(self) -> set[NamedId]:
        """Returns the set of all variable names in the analysis"""
        return set(self.defs.keys())

    def find_def_from_site(self, name: NamedId, site: DefSite) -> Definition:
        """Finds the definition of given a (name, site) pair."""
        defs = self.defs.get(name, set())
        for def_ in defs:
            if def_.site == site:
                return def_
        raise KeyError(f'no definition found for {name} at {site}')

    def find_def_from_use(self, site: UseSite):
        """Finds the definition of a variable."""
        # TODO: make more efficient: build inverse map?
        for def_ in self.uses:
            if site in self.uses[def_]:
                return def_
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

    def _add_def(self, name: NamedId, site: DefSite):
        if name not in self.analysis.defs:
            self.analysis.defs[name] = set()
        definition = AssignDef(name, site)
        self.analysis.defs[name].add(definition)
        self.analysis.uses[definition] = set()
        return definition

    def _add_use(self, name: NamedId, use: Var | IndexedAssign, ctx: DefinitionCtx):
        for d in ctx[name].assigns():
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
                ctx[name] = self._add_def(name, e)
        self._visit_expr(e.elt, ctx)

    def _visit_assign(self, stmt: Assign, ctx: DefinitionCtx):
        self._visit_expr(stmt.expr, ctx)
        for var in stmt.binding.names():
            ctx[var] = self._add_def(var, stmt)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: DefinitionCtx):
        self._add_use(stmt.var, stmt, ctx)
        for slice in stmt.slices:
            self._visit_expr(slice, ctx)
        self._visit_expr(stmt.expr, ctx)

    def _visit_if1(self, stmt: If1Stmt, ctx: DefinitionCtx):
        self._visit_expr(stmt.cond, ctx)
        body_ctx = self._visit_block(stmt.body, ctx.copy())
        # merge contexts along both paths
        # definitions cannot be introduced in the body
        for var in ctx:
            ctx[var] = PhiDef.union(ctx[var], body_ctx[var])

    def _visit_if(self, stmt: IfStmt, ctx: DefinitionCtx):
        self._visit_expr(stmt.cond, ctx)
        ift_ctx = self._visit_block(stmt.ift, ctx.copy())
        iff_ctx = self._visit_block(stmt.iff, ctx.copy())
        # merge contexts along both paths
        for var in ift_ctx.keys() & iff_ctx.keys():
            ctx[var] = PhiDef.union(ift_ctx[var], iff_ctx[var])

    def _visit_while(self, stmt: WhileStmt, ctx: DefinitionCtx):
        self._visit_expr(stmt.cond, ctx)
        body_ctx = self._visit_block(stmt.body, ctx.copy())
        # merge contexts along both paths
        # definitions cannot be introduced in the body
        for var in ctx:
            ctx[var] = PhiDef.union(ctx[var], body_ctx[var])

    def _visit_for(self, stmt: ForStmt, ctx: DefinitionCtx):
        self._visit_expr(stmt.iterable, ctx)
        body_ctx = ctx.copy()
        match stmt.target:
            case NamedId():
                body_ctx[stmt.target] = self._add_def(stmt.target, stmt)
            case TupleBinding():
                for var in stmt.target.names():
                    body_ctx[var] = self._add_def(var, stmt)

        body_ctx = self._visit_block(stmt.body, body_ctx)
        # merge contexts along both paths
        # definitions cannot be introduced in the body
        for var in ctx:
            ctx[var] = PhiDef.union(ctx[var], body_ctx[var])

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
                ctx[arg.name] = self._add_def(arg.name, arg)
        for v in func.free_vars:
            ctx[v] = self._add_def(v, func)
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
