"""
Transformation pass to eliminate unnecessary rounding operations.
"""

from dataclasses import dataclass

from ...ast.fpyast import *
from ...ast.visitor import DefaultTransformVisitor
from ...analysis import ContextAnalysis, PartialEvalInfo
from ...transform import CopyPropagate
from ...utils import Gensym

from .format import AbstractFormat
from .format_infer import FormatInfer, FormatAnalysis, convert_type

@dataclass
class _BlockCtx:
    """Transformer context for a statement block."""
    stmts: list[Stmt]


class _ElimRoundVisitor(DefaultTransformVisitor):
    func: FuncDef
    format_info: FormatAnalysis
    gensym: Gensym

    def __init__(self, func: FuncDef, format_info: FormatAnalysis):
        self.func = func
        self.format_info = format_info
        self.gensym = Gensym(self.format_info.ctx_info.def_use.names())

    def apply(self):
        return self._visit_function(self.func, None)

    def _visit_unaryop(self, e: UnaryOp, ctx: _BlockCtx):
        match e:
            case Round():
                # check if rounding is necessary
                e_ty = self.format_info.by_expr[e]
                a_ty = self.format_info.by_expr[e.arg]
                if (
                    isinstance(a_ty, AbstractFormat)
                    and isinstance(e_ty, AbstractFormat)
                    and a_ty.contained_in(e_ty)
                ):
                    # rounding is unnecessary;
                    # replace with a `cast` operation to ensure context inference works
                    return Cast(
                        Attribute(Var(NamedId('fp', None), None), 'cast', None),
                        self._visit_expr(e.arg, ctx),
                        None
                    )

        # default transform
        return super()._visit_unaryop(e, ctx)

    def _visit_binaryop(self, e: BinaryOp, ctx: _BlockCtx):
        if e in self.format_info.preround:
            match e:
                case Add() | Sub() | Mul():
                    # check if rounding is necessary
                    r_ty = convert_type(self.format_info.ctx_info.by_expr[e])
                    e_ty = self.format_info.preround[e]
                    if (
                        isinstance(e_ty, AbstractFormat)
                        and isinstance(r_ty, AbstractFormat)
                        and e_ty.contained_in(r_ty)
                    ):
                        # rounding is unnecessary;
                        # insert a real rounding instead
                        fst = self._visit_expr(e.first, ctx)
                        snd = self._visit_expr(e.second, ctx)

                        # TODO: how to get the namespace of 'fp'?
                        cls = type(e)
                        tid = self.gensym.fresh()
                        ctx.stmts.append(ContextStmt(
                            UnderscoreId(),
                            Attribute(Var(NamedId('fp', None), None), 'REAL', None),
                            StmtBlock([Assign(tid, None, cls(fst, snd, None), None)]),
                            None
                        ))

                        # push statement and return `round(tid)`
                        # so that context inference can work correctly
                        return Cast(
                            Attribute(Var(NamedId('fp', None), None), 'cast', None),
                            Var(tid, None),
                            None
                        )

        # default transform
        return super()._visit_binaryop(e, ctx)


    def _visit_block(self, block: StmtBlock, ctx):
        stmts: list[Stmt] = []
        block_ctx = _BlockCtx(stmts)
        for stmt in block.stmts:
            stmt, _ = self._visit_statement(stmt, block_ctx)
            stmts.append(stmt)

        return StmtBlock(stmts), ctx

###########################################################
# Eliminator

class ElimRound:
    """
    Rounding eliminator.

    Removes unnecessary rounding operations based on format inference.
    """

    @staticmethod
    def apply(
        func: FuncDef, *,
        ctx_info: ContextAnalysis | None = None,
        eval_info: PartialEvalInfo | None = None,
        format_info: FormatAnalysis | None = None
    ):
        """
        Performs format inference.

        Args:
            func: Function definition to transform.
            ctx_info: Context analysis information.
            eval_info: Partial evaluation information.
            format_info: Format analysis information.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {func}')

        if format_info is None:
            format_info = FormatInfer.infer(func, ctx_info=ctx_info, eval_info=eval_info)

        # apply eliminator
        eliminator = _ElimRoundVisitor(func, format_info)
        func = eliminator.apply()

        # perform copy propagation to eliminate introduced variables
        return CopyPropagate.apply(func)
