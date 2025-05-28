"""Definition use analysis for FPy ASTs"""

from dataclasses import dataclass
from typing import TypeAlias, Union

from ..ast.fpyast import *
from ..ast.visitor import DefaultAstVisitor
from ..utils import default_repr

_Definition: TypeAlias = Argument | Stmt | CompExpr

@default_repr
class _DefineUnion:
    """Union of possible definition sites."""
    defs: set[_Definition]

    def __init__(self, defs: set[_Definition]):
        self.defs = set(defs)

    def __eq__(self, other):
        if not isinstance(other, _DefineUnion):
            return NotImplemented
        return self.defs == other.defs

    def __hash__(self):
        return hash(tuple(self.defs))

    @staticmethod
    def union(*defs_or_unions: Union[_Definition, '_DefineUnion']):
        """Create a union of definitions from a set of definitions or unions."""
        defs: set[_Definition] = set()
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

@dataclass
class DefineUseAnalysis:
    """Result of definition-use analysis"""
    defs: dict[NamedId, set[_Definition]]
    uses: dict[_Definition, set[Var]]


_Ctx: TypeAlias = dict[NamedId, _Definition | _DefineUnion]

class _DefineUseInstance(DefaultAstVisitor):
    """Per-IR instance of definition-use analysis"""
    ast: FuncDef | StmtBlock
    defs: dict[NamedId, set[_Definition]]
    uses: dict[_Definition, set[Var]]

    def __init__(self, ast: FuncDef | StmtBlock):
        self.ast = ast
        self.defs = {}
        self.uses = {}

    def analyze(self):
        match self.ast:
            case FuncDef():
                self._visit_function(self.ast, {})
            case StmtBlock():
                self._visit_block(self.ast, {})
            case _:
                raise RuntimeError(f'unreachable case: {self.ast}')
        return DefineUseAnalysis(self.defs, self.uses)

    def _add_def(self, name: NamedId, definition: _Definition):
        if name not in self.defs:
            self.defs[name] = set()
        self.defs[name].add(definition)
        self.uses[definition] = set()

    def _visit_var(self, e: Var, ctx: _Ctx):
        if e.name not in ctx:
            raise NotImplementedError(f'undefined variable {e.name}')
        def_or_union = ctx[e.name]
        if isinstance(def_or_union, _DefineUnion):
            for def_ in def_or_union.defs:
                self.uses[def_].add(e)
        else:
            self.uses[def_or_union].add(e)

    def _visit_comp_expr(self, e: CompExpr, ctx: _Ctx):
        for iterable in e.iterables:
            self._visit_expr(iterable, ctx)
        ctx = ctx.copy()
        for target in e.targets:
            match target:
                case NamedId():
                    self._add_def(target, e)
                    ctx[target] = e
                case TupleBinding():
                    for name in target.names():
                        self._add_def(name, e)
                        ctx[name] = e
        self._visit_expr(e.elt, ctx)

    def _visit_simple_assign(self, stmt: SimpleAssign, ctx: _Ctx):
        self._visit_expr(stmt.expr, ctx)
        if isinstance(stmt.var, NamedId):
            self._add_def(stmt.var, stmt)
            ctx[stmt.var] = stmt

    def _visit_tuple_unpack(self, stmt: TupleUnpack, ctx: _Ctx):
        self._visit_expr(stmt.expr, ctx)
        for var in stmt.binding.names():
            self._add_def(var, stmt)
            ctx[var] = stmt

    def _visit_if1(self, stmt: If1Stmt, ctx: _Ctx):
        self._visit_expr(stmt.cond, ctx)
        body_ctx = self._visit_block(stmt.body, ctx.copy())
        # merge contexts along both paths
        # definitions cannot be introduced in the body
        for var in ctx:
            ctx[var] = _DefineUnion.union(ctx[var], body_ctx[var])

    def _visit_if(self, stmt: IfStmt, ctx: _Ctx):
        self._visit_expr(stmt.cond, ctx)
        ift_ctx = self._visit_block(stmt.ift, ctx.copy())
        iff_ctx = self._visit_block(stmt.iff, ctx.copy())
        # merge contexts along both paths
        for var in ift_ctx.keys() & iff_ctx.keys():
            ctx[var] = _DefineUnion.union(ift_ctx[var], iff_ctx[var])

    def _visit_while(self, stmt: WhileStmt, ctx: _Ctx):
        self._visit_expr(stmt.cond, ctx)
        body_ctx = self._visit_block(stmt.body, ctx.copy())
        # merge contexts along both paths
        # definitions cannot be introduced in the body
        for var in ctx:
            ctx[var] = _DefineUnion.union(ctx[var], body_ctx[var])

    def _visit_for(self, stmt: ForStmt, ctx: _Ctx):
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

    def _visit_block(self, block: StmtBlock, ctx: _Ctx):
        for stmt in block.stmts:
            self._visit_statement(stmt, ctx)
        return ctx

    def _visit_function(self, func: FuncDef, ctx: _Ctx):
        for arg in func.args:
            if isinstance(arg.name, NamedId):
                self._add_def(arg.name, arg)
                ctx[arg.name] = arg
        self._visit_block(func.body, ctx.copy())


class DefineUse:
    """
    Definition-use analyzer for the FPy IR.

    Computes definition-use chains for each variable.

    ```
    name ---> definition ---> use1, use2, ...
         ---> definition ---> use1, use2, ...
         ...
    ```
    """

    @staticmethod
    def analyze(ast: FuncDef | StmtBlock):
        if not isinstance(ast, FuncDef | StmtBlock):
            raise TypeError(f'Expected \'FuncDef\' or \'StmtBlock\', got {type(ast)} for {ast}')
        return _DefineUseInstance(ast).analyze()
