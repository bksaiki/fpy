"""
Compilation from FPy to (standard) C++.
"""

import dataclasses

from ..ast import *
from ..analysis import (
    ContextAnalysis, ContextInfer, ContextInferError, ContextType,
    DefineUse, DefineUseAnalysis, DefSite,
    TypeAnalysis, TypeCheck, TypeInferError
)
from ..function import Function
from ..number import FP64, FP32
from ..transform import ContextInline
from ..types import *
from ..utils import Gensym

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


class CppCompileError(CompileError):
    """Compiler error for C++ backend"""

    def __init__(self, func: FuncDef, msg: str, *args):
        lines: list[str] = [f'C++ backend: {msg} in function `{func.name}`']
        lines.extend(str(arg) for arg in args)
        super().__init__('\n '.join(lines))


class _CppBackendInstance(Visitor):
    """
    Per-function compilation instance.
    """

    func: FuncDef
    caller_ctx: Context | None
    def_use: DefineUseAnalysis
    type_info: TypeAnalysis
    ctx_info: ContextAnalysis
    ctx_args: dict[NamedId, Context]
    gensym: Gensym

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
        self.ctx_args = {}
        self.gensym = Gensym(self.def_use.names)

    def compile(self):
        ctx = _CompileCtx.default()
        self._visit_function(self.func, ctx)
        return '\n'.join(ctx.lines)

    def _compile_context(self, ctx: ContextType) -> str:
        match ctx:
            case NamedId():
                if ctx not in self.ctx_args:
                    raise CppCompileError(self.func, f'contexts must be monomorphic: `{ctx}`')
                return self._compile_context(self.ctx_args[ctx])
            case Context():
                if ctx.is_equiv(FP64):
                    return 'double'
                elif ctx.is_equiv(FP32):
                    return 'float'
                else:
                    raise CppCompileError(self.func, f'unsupported context: `{ctx}`')
            case _:
                raise CppCompileError(self.func, f'unsupported context: `{ctx}`')

    def _compile_type(self, ty: Type, ctx: ContextType) -> str:
        match ty:
            case BoolType():
                return 'bool'
            case RealType():
                return self._compile_context(ctx)
            case _:
                raise CppCompileError(self.func, f'unsupported type: `{ty.format()}`')

    def _fresh_var(self):
        return self.gensym.fresh('__fpy_tmp')

    def _var_is_decl(self, name: NamedId, site: DefSite):
        d = self.def_use.find_def_from_site(name, site)
        return d.parent is None

    def _var_type(self, name: NamedId, site: DefSite):
        d = self.def_use.find_def_from_site(name, site)
        ty = self.type_info.by_def[d]
        ctx = self.ctx_info.by_def[d]
        return ty, ctx

    def _expr_type(self, e: Expr):
        return self.type_info.by_expr[e], self.ctx_info.by_expr[e]

    def _visit_var(self, e: Var, ctx: _CompileCtx):
        return str(e.name)

    def _visit_bool(self, e: BoolVal, ctx: _CompileCtx):
        return 'true' if e.val else 'false'

    def _visit_foreign(self, e: ForeignVal, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_decnum(self, e: Decnum, ctx: _CompileCtx):
        # TODO: check context
        return str(e.val)

    def _visit_hexnum(self, e: Hexnum, ctx: _CompileCtx):
        # TODO: check context
        return str(e.val)

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

    def _visit_range(self, arg: Expr, ctx: _CompileCtx):
        arg_cpp = self._visit_expr(arg, ctx)
        e_ty, e_ctx = self._expr_type(arg)
        cpp_ty = self._compile_type(e_ty, e_ctx)
        t = self._fresh_var()

        # TODO: check that `arg` may be safely cast to `size_t`
        ctx.add_line(f'std::vector<{cpp_ty}> {t}(static_cast<size_t>({arg_cpp}));')
        ctx.add_line(f'std::iota({t}.begin(), {t}.end(), static_cast<{cpp_ty}>(0));')
        return t

    def _visit_unaryop(self, e: UnaryOp, ctx: _CompileCtx):
        match e:
            case Range():
                return self._visit_range(e.arg, ctx)
            case _:
                raise NotImplementedError(e)

    def _visit_binaryop(self, e: BinaryOp, ctx: _CompileCtx):
        raise NotImplementedError(e)

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

    def _visit_assign(self, stmt: Assign, ctx: _CompileCtx):
        e = self._visit_expr(stmt.expr, ctx)
        match stmt.binding:
            case NamedId():
                if self._var_is_decl(stmt.binding, stmt):
                    var_ty, var_ctx = self._var_type(stmt.binding, stmt)
                    cpp_ty = self._compile_type(var_ty, var_ctx)
                    ctx.add_line(f'{cpp_ty} {stmt.binding} = {e};')
                else:
                    ctx.add_line(f'{stmt.binding} = {e};')
            case _:
                raise NotImplementedError(stmt.binding)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_if1(self, stmt: If1Stmt, ctx: _CompileCtx):
        cond = self._visit_expr(stmt.cond, ctx)
        ctx.add_line(f'if ({cond}) {{')
        self._visit_block(stmt.body, ctx)
        ctx.add_line('}')  # close if

    def _visit_if(self, stmt: IfStmt, ctx: _CompileCtx):
        # TODO: fold if statements
        cond = self._visit_expr(stmt.cond, ctx)
        ctx.add_line(f'if ({cond}) {{')
        self._visit_block(stmt.ift, ctx)
        ctx.add_line('} else {')
        self._visit_block(stmt.iff, ctx)
        ctx.add_line('}')  # close if

    def _visit_while(self, stmt: WhileStmt, ctx: _CompileCtx):
        cond = self._visit_expr(stmt.cond, ctx)
        ctx.add_line(f'while ({cond}) {{')
        self._visit_block(stmt.body, ctx)
        ctx.add_line('}')  # close if

    def _visit_for(self, stmt: ForStmt, ctx: _CompileCtx):
        iterable = self._visit_expr(stmt.iterable, ctx)
        ctx.add_line(f'for (auto {stmt.target} : {iterable}) {{')
        self._visit_block(stmt.body, ctx)
        ctx.add_line('}')  # close if

    def _visit_context(self, stmt: ContextStmt, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_assert(self, stmt: AssertStmt, ctx: _CompileCtx):
        e = self._visit_expr(stmt.test, ctx)
        ctx.add_line(f'assert({e});')

    def _visit_effect(self, stmt: EffectStmt, ctx: _CompileCtx):
        raise CompileError('C++ backend: FPy effects are not supported')

    def _visit_return(self, stmt: ReturnStmt, ctx: _CompileCtx):
        e = self._visit_expr(stmt.expr, ctx)
        ctx.add_line(f'return {e};')

    def _visit_block(self, block: StmtBlock, ctx: _CompileCtx):
        ctx = ctx.indent()
        for stmt in block.stmts:
            self._visit_statement(stmt, ctx)

    def _visit_function(self, func: FuncDef, ctx: _CompileCtx):
        # TODO: use caller context
        if isinstance(self.ctx_info.body_ctx, NamedId):
            if self.caller_ctx is None:
                # TODO: what to report to user?
                raise CppCompileError(self.func, f'contexts must be monomorphic `{self.ctx_info.body_ctx}`')
            self.ctx_args[self.ctx_info.body_ctx] = self.caller_ctx

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


_HEADER = [
    '#include <cstddef>',
    '#include <numeric>',
    '#include <vector>',
]


class CppBackend(Backend):
    """
    Compiler from FPy to C++.
    """

    includes: bool

    def __init__(self, includes: bool = True):
        self.includes = includes

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
        body_str =  inst.compile()

        if self.includes:
            body_str = '\n'.join(_HEADER) + '\n\n' + body_str
        return body_str
