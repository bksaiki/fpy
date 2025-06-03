"""Definition use analysis for FPy ASTs"""

from dataclasses import dataclass
from typing import TypeAlias, Union

from ..ast.fpyast import *
from ..ast.visitor import DefaultVisitor
from ..utils import default_repr

Definition: TypeAlias = Argument | Stmt | CompExpr

@default_repr
class _DefineUnion:
    """Union of possible definition sites."""
    defs: set[Definition]

    def __init__(self, defs: set[Definition]):
        self.defs = set(defs)

    def __eq__(self, other):
        if not isinstance(other, _DefineUnion):
            return NotImplemented
        return self.defs == other.defs

    def __hash__(self):
        return hash(tuple(self.defs))

    @staticmethod
    def union(*defs_or_unions: Union[Definition, '_DefineUnion']):
        """Create a union of definitions from a set of definitions or unions."""
        defs: set[Definition] = set()
        for item in defs_or_unions:
            if isinstance(item, _DefineUnion):
                defs.update(item.defs)
            else:
                defs.add(item)

        if len(defs) == 0:
            raise ValueError('Cannot create a union of an empty set of definitions')
        elif len(defs) == 1:
            return defs.pop()
        else:
            return _DefineUnion(defs)

class DefinitionCtx(dict[NamedId, Definition | _DefineUnion]):
    """Mapping from variable to its definition (or possible definitions)."""

    def copy(self) -> 'DefinitionCtx':
        """Returns a shallow copy of the context."""
        return DefinitionCtx(self)

    def mutated_in(self, other: 'DefinitionCtx') -> list[NamedId]:
        """
        Returns the set of variables that are defined in `self`
        and mutated in `other`.
        """
        names: set[NamedId] = set()
        for name in self.keys() & other.keys():
            if self[name] != other[name]:
                names.add(name)
        return list(names)

    def fresh_in(self, other: 'DefinitionCtx') -> set[NamedId]:
        """
        Returns the set of variables that are defined in `other`
        but not in `self`.
        """
        return set(other.keys() - self.keys())


@dataclass
class DefineUseAnalysis:
    """Result of definition-use analysis"""
    defs: dict[NamedId, set[Definition]]
    uses: dict[Definition, set[Var | IndexedAssign]]
    stmts: dict[Stmt, tuple[DefinitionCtx, DefinitionCtx]]
    blocks: dict[StmtBlock, tuple[DefinitionCtx, DefinitionCtx]]

    @staticmethod
    def default():
        """Default analysis with empty definitions and uses"""
        return DefineUseAnalysis({}, {}, {}, {})

    @property
    def names(self) -> set[NamedId]:
        """Returns the set of all variable names in the analysis"""
        return set(self.defs.keys())


class _DefineUseInstance(DefaultVisitor):
    """Per-IR instance of definition-use analysis"""
    ast: FuncDef | StmtBlock
    analysis: DefineUseAnalysis

    def __init__(self, ast: FuncDef | StmtBlock):
        self.ast = ast
        self.analysis = DefineUseAnalysis.default()

    def analyze(self):
        match self.ast:
            case FuncDef():
                self._visit_function(self.ast, DefinitionCtx())
            case StmtBlock():
                self._visit_block(self.ast, DefinitionCtx())
            case _:
                raise RuntimeError(f'unreachable case: {self.ast}')
        return self.analysis

    def _add_def(self, name: NamedId, definition: Definition):
        if name not in self.analysis.defs:
            self.analysis.defs[name] = set()
        self.analysis.defs[name].add(definition)
        self.analysis.uses[definition] = set()

    def _add_use(self, name: NamedId, use: Var | IndexedAssign, ctx: DefinitionCtx):
        def_or_union = ctx[name]
        if isinstance(def_or_union, _DefineUnion):
            for def_ in def_or_union.defs:
                self.analysis.uses[def_].add(use)
        else:
            self.analysis.uses[def_or_union].add(use)

    def _visit_var(self, e: Var, ctx: DefinitionCtx):
        if e.name not in ctx:
            raise NotImplementedError(f'undefined variable {e.name}')
        self._add_use(e.name, e, ctx)

    def _visit_comp_expr(self, e: CompExpr, ctx: DefinitionCtx):
        for iterable in e.iterables:
            self._visit_expr(iterable, ctx)
        ctx = ctx.copy()
        for target in e.targets:
            for name in target.names():
                self._add_def(name, e)
                ctx[name] = e
        self._visit_expr(e.elt, ctx)

    def _visit_assign(self, stmt: Assign, ctx: DefinitionCtx):
        self._visit_expr(stmt.expr, ctx)
        for var in stmt.binding.names():
            self._add_def(var, stmt)
            ctx[var] = stmt

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
            ctx[var] = _DefineUnion.union(ctx[var], body_ctx[var])

    def _visit_if(self, stmt: IfStmt, ctx: DefinitionCtx):
        self._visit_expr(stmt.cond, ctx)
        ift_ctx = self._visit_block(stmt.ift, ctx.copy())
        iff_ctx = self._visit_block(stmt.iff, ctx.copy())
        # merge contexts along both paths
        for var in ift_ctx.keys() & iff_ctx.keys():
            ctx[var] = _DefineUnion.union(ift_ctx[var], iff_ctx[var])

    def _visit_while(self, stmt: WhileStmt, ctx: DefinitionCtx):
        self._visit_expr(stmt.cond, ctx)
        body_ctx = self._visit_block(stmt.body, ctx.copy())
        # merge contexts along both paths
        # definitions cannot be introduced in the body
        for var in ctx:
            ctx[var] = _DefineUnion.union(ctx[var], body_ctx[var])

    def _visit_for(self, stmt: ForStmt, ctx: DefinitionCtx):
        self._visit_expr(stmt.iterable, ctx)
        body_ctx = ctx.copy()
        match stmt.target:
            case NamedId():
                self._add_def(stmt.target, stmt)
                body_ctx[stmt.target] = stmt
            case TupleBinding():
                for var in stmt.target.names():
                    self._add_def(var, stmt)
                    body_ctx[var] = stmt

        body_ctx = self._visit_block(stmt.body, body_ctx)
        # merge contexts along both paths
        # definitions cannot be introduced in the body
        for var in ctx:
            ctx[var] = _DefineUnion.union(ctx[var], body_ctx[var])

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
                self._add_def(arg.name, arg)
                ctx[arg.name] = arg
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
