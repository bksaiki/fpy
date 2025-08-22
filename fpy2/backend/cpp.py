"""
Compilation from FPy to (standard) C++.
"""

import dataclasses

from ..ast import *
from ..analysis import (
    ContextAnalysis, ContextInfer, ContextInferError, ContextType,
    DefineUse, DefineUseAnalysis,
    TypeAnalysis, TypeCheck, TypeInferError
)
from ..function import Function
from ..number import FP64, FP32
from ..transform import ContextInline
from ..types import *

from .backend import Backend, CompileError

@dataclasses.dataclass
class _CompileCtx:
    lines: list[str]
    indent_str: str
    indent_level: int

    @staticmethod
    def default(indent_str: str = ' ' * 4):
        return _CompileCtx([], indent_str, 0)

    def indent(self):
        return _CompileCtx(self.lines, self.indent_str, self.indent_level + 1)

    def add_line(self, line: str):
        self.lines.append(self.indent_str * self.indent_level + line)


class _CppBackendInstance(Visitor):
    """
    Per-function compilation instance.
    """

    func: FuncDef
    caller_ctx: Context | None
    def_use: DefineUseAnalysis
    type_info: TypeAnalysis
    ctx_info: ContextAnalysis
    targs: dict[NamedId, NamedId]

    def __init__(
        self,
        func: FuncDef,
        caller_ctx: Context | None,
        def_use: DefineUseAnalysis,
        type_info: TypeAnalysis,
        ctx_info: ContextAnalysis
    ):
        self.func = func
        self.caller_ctx = caller_ctx
        self.def_use = def_use
        self.type_info = type_info
        self.ctx_info = ctx_info
        self.targs = {}

    def compile(self):
        ctx = _CompileCtx.default()
        self._visit_function(self.func, ctx)
        return '\n'.join(ctx.lines)

    def _compile_context_var(self, cvar: NamedId) -> str:
        if cvar not in self.targs:
            if self.targs:
                self.targs[cvar] = NamedId('T', len(self.targs))
            else:
                self.targs[cvar] = NamedId('T')
        return str(self.targs[cvar])

    def _compile_context(self, ctx: ContextType) -> str:
        match ctx:
            case NamedId():
                return self._compile_context_var(ctx)
            case Context():
                if ctx.is_equiv(FP64):
                    return 'double'
                elif ctx.is_equiv(FP32):
                    return 'float'
                else:
                    raise CompileError(f'unsupported context: `{ctx}`')
            case _:
                raise CompileError(f'unsupported context: `{ctx}`')

    def _compile_type(self, ty: Type, ctx: ContextType) -> str:
        match ty:
            case BoolType():
                return 'bool'
            case RealType():
                return self._compile_context(ctx)
            case _:
                raise CompileError(f'unsupported type: `{ty.format()}`')

    def _visit_var(self, e: Var, ctx: _CompileCtx):
        return str(e.name)

    def _visit_bool(self, e: BoolVal, ctx: _CompileCtx):
        return 'true' if e.val else 'false'

    def _visit_foreign(self, e: ForeignVal, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_decnum(self, e: Decnum, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_hexnum(self, e: Hexnum, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_integer(self, e: Integer, ctx: _CompileCtx):
        # TODO: check type to pre-round
        return str(e.val)

    def _visit_rational(self, e: Rational, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_digits(self, e: Digits, ctx: _CompileCtx):
        # TODO: this is incorrect since it rounds
        return str(float(e.as_rational()))

    def _visit_nullaryop(self, e: NullaryOp, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_unaryop(self, e: UnaryOp, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_binaryop(self, e: BinaryOp, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_ternaryop(self, e: TernaryOp, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_naryop(self, e: NaryOp, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_compare(self, e: Compare, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_call(self, e: Call, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_tuple_expr(self, e: TupleExpr, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_list_expr(self, e: ListExpr, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_list_comp(self, e: ListComp, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_list_ref(self, e: ListRef, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_list_slice(self, e: ListSlice, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_list_set(self, e: ListSet, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_if_expr(self, e: IfExpr, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_context_expr(self, e: ContextExpr, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_assign(self, stmt: Assign, ctx: _CompileCtx):
        e = self._visit_expr(stmt.expr, ctx)
        match stmt.binding:
            case NamedId():
                d = self.def_use.find_def_from_site(stmt.binding, stmt)
                var_ty = self.type_info.by_def[d]
                var_ctx = self.ctx_info.by_def[d]
                ty = self._compile_type(var_ty, var_ctx)
                ctx.add_line(f'{ty} {stmt.binding} = {e};')
            case _:
                raise NotImplementedError(stmt.binding)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_if1(self, stmt: If1Stmt, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_if(self, stmt: IfStmt, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_while(self, stmt: WhileStmt, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_for(self, stmt: ForStmt, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_context(self, stmt: ContextStmt, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_assert(self, stmt: AssertStmt, ctx: _CompileCtx):
        e = self._visit_expr(stmt.test, ctx)
        ctx.add_line(f'assert({e});')

    def _visit_effect(self, stmt: EffectStmt, ctx: _CompileCtx):
        raise CompileError('FPy effects are not supported in C++ backend')

    def _visit_return(self, stmt: ReturnStmt, ctx: _CompileCtx):
        e = self._visit_expr(stmt.expr, ctx)
        ctx.add_line(f'return {e};')

    def _visit_block(self, block: StmtBlock, ctx: _CompileCtx):
        ctx = ctx.indent()
        for stmt in block.stmts:
            self._visit_statement(stmt, ctx)

    def _visit_function(self, func: FuncDef, ctx: _CompileCtx):
        # TODO: use caller context

        # compile arguments
        arg_strs: list[str] = []
        for arg, arg_ty, arg_ctx in zip(func.args, self.type_info.arg_types, self.ctx_info.arg_ctxs):
            ty = self._compile_type(arg_ty, arg_ctx)
            arg_strs.append(f'{ty} {arg.name}')

        ret_ty = self._compile_type(self.type_info.return_type, self.ctx_info.return_ctx)
        ctx.add_line(f'{ret_ty} {func.name}({", ".join(arg_strs)}) {{')

        # compile body
        self._visit_block(func.body, ctx)
        ctx.add_line('}')  # close function definition


class CppBackend(Backend):
    """
    Compiler from FPy to C++.
    """

    def compile(self, func: Function, ctx: Context | None = None) -> str:
        """
        Compiles the given FPy function to a C++ program
        represented as a string.
        """
        if not isinstance(func, Function):
            raise TypeError(f'Expected `Function`, got {type(func)} for {func}')
        if ctx is not None and not isinstance(ctx, Context):
            raise TypeError(f'Expected `Context`, got {type(ctx)} for {ctx}')

        # normalization passes
        ast = ContextInline.apply(func.ast, func.env)

        # analyses
        def_use = DefineUse.analyze(ast)

        try:
            type_info = TypeCheck.check(ast, def_use=def_use)
        except TypeInferError as e:
            raise ValueError(f'type inference failed') from e

        try:
            ctx_info = ContextInfer.infer(ast, def_use=def_use)
        except ContextInferError as e:
            raise ValueError(f'context inference failed') from e

        # compile
        inst = _CppBackendInstance(ast, ctx, def_use, type_info, ctx_info)
        return inst.compile()
