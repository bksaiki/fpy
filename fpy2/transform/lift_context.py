"""
Transformation pass to lift context expressions to the top-level.
"""

from ..analysis import PartialEval, PartialEvalInfo
from ..ast.fpyast import *
from ..ast.visitor import DefaultTransformVisitor, DefaultVisitor
from ..utils import Gensym
from .utils import Edit, EditLog, FuncBody


class _ContextFinder(DefaultVisitor):
    """
    Visitor to find context expressions that can be lifted
    to the top-level.
    """
    func: FuncDef
    eval_info: PartialEvalInfo
    ctx_exprs: list[Expr]

    def __init__(self, func: FuncDef, eval_info: PartialEvalInfo):
        self.func = func
        self.eval_info = eval_info
        self.ctx_exprs = []

    def visit(self):
        self._visit_function(self.func, False)
        return self.ctx_exprs

    def _visit_context(self, stmt: ContextStmt, ctx: bool):
        return super()._visit_context(stmt, False)

    def _visit_expr(self, e: Expr, ctx: bool) -> Expr:
        # check if we know an expression evaluates
        # statically to a context; if so, we can lift it
        if e in self.eval_info.by_expr and not ctx:
            v = self.eval_info.by_expr[e]
            if (
                isinstance(v, Context)
                and not isinstance(e, Var)
                and not isinstance(e, ForeignVal)
            ):
                self.ctx_exprs.append(e)

        return super()._visit_expr(e, ctx)


class _ContextLifter(DefaultTransformVisitor):
    """
    Visitor to lift context expressions to the top-level.
    """
    func: FuncDef
    expr_to_name: dict[Expr, NamedId]
    name_to_expr: dict[NamedId, Expr]

    def __init__(self, func: FuncDef, eval_info: PartialEvalInfo, ctx_exprs: list[Expr]):
        self.func = func
        self.expr_to_name = {}
        self.name_to_expr = {}

        # bind expressions to fresh variable names
        gensym = Gensym(eval_info.def_use.names())
        for e in ctx_exprs:
            name = gensym.fresh('ctx')
            self.expr_to_name[e] = name
            self.name_to_expr[name] = e

    def apply(self) -> FuncDef:
        return self._visit_function(self.func, None)

    def _visit_function(self, func: FuncDef, ctx: None):
        # visit the function body to eliminate context expressions
        func = super()._visit_function(func, ctx)
        # prepend variable bindings for lifted context expressions
        stmts: list[Stmt] = []
        for name, expr in self.name_to_expr.items():
            stmts.append(Assign(name, None, expr, expr.loc))
        stmts.extend(func.body.stmts)

        # replace the function body with the new statements
        func.body = StmtBlock(stmts)
        return func

    def _visit_expr(self, e: Expr, ctx: None) -> Expr:
        # if this expression is a context expression, replace it with the corresponding variable
        if e in self.expr_to_name:
            return Var(self.expr_to_name[e], e.loc)
        return super()._visit_expr(e, ctx)


class LiftContext:
    """
    Transformation pass to lift context expressions to the top-level.
    """

    @staticmethod
    def apply(func: FuncDef, *, eval_info: PartialEvalInfo | None = None) -> FuncDef:
        return LiftContext.apply_with_edits(func, eval_info=eval_info).result

    @staticmethod
    def apply_with_edits(
        func: FuncDef, *, eval_info: PartialEvalInfo | None = None
    ) -> EditLog:
        """:meth:`apply`, with the record of what it added: a binding per lifted
        context, prepended, so every cursor below shifts by that many.

        Statement cursors survive; expression cursors do not, since the lifted
        context expressions are replaced by variables in place, at sites no edit
        records.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {func}')

        if eval_info is None:
            eval_info = PartialEval.apply(func)

        ctx_exprs = _ContextFinder(func, eval_info).visit()
        lifter = _ContextLifter(func, eval_info, ctx_exprs)
        out = lifter.apply()
        lifted = len(lifter.name_to_expr)
        edits = () if lifted == 0 else (Edit(FuncBody(), 0, 0, lifted),)
        return EditLog(func, out, edits)

