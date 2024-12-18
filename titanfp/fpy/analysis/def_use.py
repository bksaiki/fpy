"""
Definition-use analysis.

Computes the set of variables in scope for every node.
For each of the variables, tracks the set of uses throughout the program.
"""

from dataclasses import dataclass

from ..fpyast import *
from ..visitor import Analysis

class VarRecord:
    """Type to track variable definition/use."""
    name: str
    src: Ast
    uses: list[Ast]

    def __init__(self, name: str, src: Ast):
        self.name = name
        self.src = src
        self.uses = []

    def record_use(self, where: Ast):
        self.uses.append(where)

class _Ctx:
    """Context type for `DefUse` visitor methods."""
    env: dict[str, VarRecord]

    def __init__(self, env: Optional[dict[str, VarRecord]] = None):
        if env is None:
            self.env = dict()
        else:
            self.env = env

    def extend(self, name: str, src: Ast):
        copy = _Ctx()
        copy.env = dict(**self.env)
        copy.env[name] = VarRecord(name, src)
        return copy

    def record_use(self, name: str, where: Ast):
        if name not in self.env:
            raise ValueError(f'name \'{name}\' not in context')
        self.env[name].record_use(where)


class DefUse(Analysis):
    """
    Definition-use analysis.

    Computes the set of variables in scope for every node.
    For each of the variables, tracks the set of uses throughout the program.
    """

    def __init__(self, record=True):
        super().__init__('def_use', record)

    def _visit_decnum(self, e, ctx: _Ctx):
        return ctx.env

    def _visit_integer(self, e, ctx: _Ctx):
        return ctx.env

    def _visit_digits(self, e, ctx: _Ctx):
        return ctx.env

    def _visit_variable(self, e, ctx: _Ctx):
        ctx.record_use(e.name, e)
        return ctx.env

    def _visit_array(self, e, ctx: _Ctx):
        for c in e.children:
            self._visit(c, ctx)
        return ctx.env

    def _visit_unknown(self, e, ctx: _Ctx):
        for c in e.children:
            self._visit(c, ctx)
        return ctx.env

    def _visit_nary_expr(self, e, ctx: _Ctx):
        for c in e.children:
            self._visit(c, ctx)
        return ctx.env

    def _visit_compare(self, e, ctx: _Ctx):
        for c in e.children:
            self._visit(c, ctx)
        return ctx.env

    def _visit_if_expr(self, e, ctx: _Ctx):
        self._visit(e.cond, ctx)
        self._visit(e.ift, ctx)
        self._visit(e.iff, ctx)
        return ctx.env

    def _visit_assign(self, stmt, ctx: _Ctx):
        return self._visit(stmt.val, ctx)

    def _visit_tuple_assign(self, stmt, ctx: _Ctx):
        return self._visit(stmt.val, ctx)

    def _visit_return(self, stmt, ctx: _Ctx):
        return self._visit(stmt.e, ctx)

    def _visit_if_stmt(self, stmt, ctx: _Ctx):
        self._visit(stmt.cond, ctx)
        ift_env = self._visit(stmt.ift, ctx)
        iff_env = self._visit(stmt.iff, ctx)

        merged: dict[str, VarRecord] = dict()
        for name in ift_env.keys() & iff_env.keys():
            merged[name] = ift_env[name]
        return merged

    def _visit_block(self, block, ctx: _Ctx):
        for stmt in block.stmts:
            match stmt:
                case Assign():
                    self._visit(stmt, ctx)
                    for name in stmt.var.ids():
                        ctx = ctx.extend(name, stmt)
                case TupleAssign():
                    self._visit(stmt, ctx)
                    for name in stmt.binding.ids():
                        ctx = ctx.extend(name, stmt)
                case Return():
                    self._visit(stmt, ctx)
                case IfStmt():
                    ctx = _Ctx(self._visit(stmt, ctx))
                case _:
                    raise NotImplementedError('unreachable', stmt)
        return ctx.env

    def _visit_function(self, func, ctx: _Ctx):
        for arg in func.args:
            ctx = ctx.extend(arg.name, arg)
        return self._visit(func.body, ctx)

    # override typing hint
    def _visit(self, e, ctx: _Ctx) -> dict[str, VarRecord]:
        return super()._visit(e, ctx)

    def visit(self, e: Function | Block):
        if not (isinstance(e, Function) or isinstance(e, Block)):
            raise TypeError(f'visit() argument 1 must be Function or Block, not {e}')
        return self._visit(e, _Ctx())

