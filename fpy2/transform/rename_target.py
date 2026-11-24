"""Renames targets."""

from ..analysis import SyntaxCheck
from ..ast import *


class _RenameTargetInstance(DefaultTransformVisitor):
    """Renames targets."""
    rename: dict[NamedId, NamedId]

    def __init__(self, rename: dict[NamedId, NamedId]) -> None:
        self.rename = rename

    def _visit_var(self, var: Var, ctx: None) -> Var:
        """Renames variables."""
        name = self.rename.get(var.name, var.name)
        return Var(name, var.loc)

    def _visit_binding(self, binding: Id | TupleBinding, ctx: None) -> Id | TupleBinding:
        if isinstance(binding, NamedId):
            return self.rename.get(binding, binding)
        return super()._visit_binding(binding, ctx)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: None) -> tuple[Stmt, None]:
        var = self.rename.get(stmt.var, stmt.var)
        slices = [self._visit_expr(slice, ctx) for slice in stmt.indices]
        expr = self._visit_expr(stmt.expr, ctx)
        s = IndexedAssign(var, slices, expr, stmt.loc)
        return s, None

    def _visit_function(self, func: FuncDef, ctx: None) -> FuncDef:
        args: list[Argument] = []
        for arg in func.args:
            match arg.name:
                case NamedId():
                    name = self.rename.get(arg.name, arg.name)
                    args.append(Argument(name, arg.type, arg.loc))
                case UnderscoreId():
                    args.append(Argument(arg.name, arg.type, arg.loc))
                case _:
                    raise RuntimeError('unreachable', arg)

        free_vars = { self.rename.get(arg, arg) for arg in func.free_vars }
        body, _ = self._visit_block(func.body, ctx)
        meta = FuncMeta(free_vars, func.meta.ctx, func.meta.pre, func.meta.spec, func.meta.props, func.meta.env)
        return FuncDef(func.name, args, body, meta, loc=func.loc)


class RenameTarget:
    """Renames targets, that is, left-hand side of assignments, in an FPy program."""

    @staticmethod
    def apply(func: FuncDef, rename: dict[NamedId, NamedId]) -> FuncDef:
        """Renames targets in the function."""
        if not isinstance(func, FuncDef):
            raise TypeError(f"Expected \'FuncDef\' for {func}, got {type(func)}")
        ast = _RenameTargetInstance(rename)._visit_function(func, None)
        SyntaxCheck.check(ast, ignore_unknown=True)
        return ast

    @staticmethod
    def apply_block(block: StmtBlock, rename: dict[NamedId, NamedId]) -> StmtBlock:
        """Renames targets in a block."""
        if not isinstance(block, StmtBlock):
            raise TypeError(f"Expected \'StmtBlock\' for {block}, got {type(block)}")
        ast, _ = _RenameTargetInstance(rename)._visit_block(block, None)
        # TODO: how to check block
        return ast
