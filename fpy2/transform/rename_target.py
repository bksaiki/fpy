"""Renames targets."""

from ..ast import *
from ..analysis import SyntaxCheck

class _RenameTargetInstance(DefaultTransformVisitor):
    """Renames targets in a function."""
    ast: FuncDef | StmtBlock
    rename: dict[NamedId, NamedId]

    def __init__(self, ast: FuncDef | StmtBlock, rename: dict[NamedId, NamedId]):
        self.ast = ast
        self.rename = rename

    def apply(self):
        """Applies the renaming to the function."""
        match self.ast:
            case FuncDef():
                return self._visit_function(self.ast, None)
            case StmtBlock():
                ast, _ = self._visit_block(self.ast, None)
                return ast
            case _:
                raise RuntimeError('unreachable', self.ast)

    def _visit_var(self, var: Var, ctx: None):
        """Renames variables."""
        name = self.rename.get(var.name, var.name)
        return Var(name, var.loc)

    def _visit_list_comp(self, e: ListComp, ctx: None):
        targets = [self._visit_binding(target, ctx) for target in e.targets]
        iterables = [self._visit_expr(iterable, ctx) for iterable in e.iterables]
        elt = self._visit_expr(e.elt, ctx)
        return ListComp(targets, iterables, elt, e.loc)

    def _visit_binding(self, binding: Id | TupleBinding, ctx: None):
        match binding:
            case NamedId():
                return self.rename.get(binding, binding)
            case UnderscoreId():
                return binding
            case TupleBinding():
                return self._visit_tuple_binding(binding, ctx)
            case _:
                raise RuntimeError('unreachable', binding)

    def _visit_tuple_binding(self, binding: TupleBinding, ctx: None):
        return TupleBinding([self._visit_binding(elt, ctx) for elt in binding.elts], None)

    def _visit_assign(self, stmt: Assign, ctx: None):
        binding = self._visit_binding(stmt.target, ctx)
        expr = self._visit_expr(stmt.expr, ctx)
        s = Assign(binding, stmt.type, expr, stmt.loc)
        return s, None

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: None):
        var = self.rename.get(stmt.var, stmt.var)
        slices = [self._visit_expr(slice, ctx) for slice in stmt.indices]
        expr = self._visit_expr(stmt.expr, ctx)
        s = IndexedAssign(var, slices, expr, stmt.loc)
        return s, None

    def _visit_for(self, stmt: ForStmt, ctx: None):
        iterable = self._visit_expr(stmt.iterable, ctx)
        target = self._visit_binding(stmt.target, ctx)
        body, _ = self._visit_block(stmt.body, ctx)
        s = ForStmt(target, iterable, body, stmt.loc)
        return s, None

    def _visit_function(self, func: FuncDef, ctx: None):
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
        return FuncDef(func.name, args, free_vars, func.ctx, body, func.spec, func.meta, func.env, loc=func.loc)


class RenameTarget:
    """Renames targets, that is, left-hand side of assignments, in an FPy program."""

    @staticmethod
    def apply(func: FuncDef, rename: dict[NamedId, NamedId]):
        """Renames targets in the function."""
        if not isinstance(func, FuncDef):
            raise TypeError(f"Expected \'FuncDef\' for {func}, got {type(func)}")
        ast = _RenameTargetInstance(func, rename).apply()
        if not isinstance(ast, FuncDef):
            raise TypeError(f"Expected \'FuncDef\' for {ast}, got {type(ast)}")
        SyntaxCheck.check(ast, ignore_unknown=True)
        return ast

    @staticmethod
    def apply_block(block: StmtBlock, rename: dict[NamedId, NamedId]):
        """Renames targets in a block."""
        if not isinstance(block, StmtBlock):
            raise TypeError(f"Expected \'StmtBlock\' for {block}, got {type(block)}")
        ast = _RenameTargetInstance(block, rename).apply()
        if not isinstance(ast, StmtBlock):
            raise TypeError(f"Expected \'StmtBlock\' for {ast}, got {type(ast)}")
        # TODO: how to check block
        return ast
