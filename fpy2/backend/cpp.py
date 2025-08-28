"""
Compilation from FPy to (standard) C++.
"""

import enum
import dataclasses

from typing import TypeAlias

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
from ..utils import Gensym, enum_repr

from .backend import Backend, CompileError
from .cpp_utils import CppType, ScalarOpTable, make_op_table

# TODO: support more C++ standards
@enum_repr
class CppStandard(enum.Enum):
    """C++ standards supported by the C++ backend"""
    CXX_11 = 0


class CppCompileError(CompileError):
    """Compiler error for C++ backend"""

    def __init__(self, func: FuncDef, msg: str, *args):
        lines: list[str] = [f'C++ backend: {msg} in function `{func.name}`']
        lines.extend(str(arg) for arg in args)
        super().__init__('\n '.join(lines))


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
    std: CppStandard
    op_table: ScalarOpTable

    caller_ctx: Context | None
    def_use: DefineUseAnalysis
    type_info: TypeAnalysis
    ctx_info: ContextAnalysis
    ctx_args: dict[NamedId, Context]
    gensym: Gensym

    def __init__(
        self,
        func: FuncDef,
        std: CppStandard,
        op_table: ScalarOpTable,
        caller_ctx: Context | None,
        def_use: DefineUseAnalysis,
        type_info: TypeAnalysis,
        ctx_info: ContextAnalysis
    ):
        self.func = func
        self.std = std
        self.op_table = op_table
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

    def _monomorphize_context(self, ctx: ContextType):
        match ctx:
            case NamedId():
                return self.ctx_args.get(ctx, ctx)
            case Context():
                return ctx
            case _:
                raise RuntimeError(f'unreachable: {ctx}')

    def _compile_context(self, ctx: ContextType):
        match ctx:
            case NamedId():
                raise CppCompileError(self.func, f'contexts must be monomorphic: `{ctx}`')
            case Context():
                if ctx.is_equiv(FP64):
                    return CppType.DOUBLE
                elif ctx.is_equiv(FP32):
                    return CppType.FLOAT
                else:
                    raise CppCompileError(self.func, f'unsupported context: `{ctx}`')
            case _:
                raise CppCompileError(self.func, f'unsupported context: `{ctx}`')

    def _compile_type(self, ty: Type, ctx: ContextType):
        match ty:
            case BoolType():
                return CppType.BOOL
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
        ctx = self._monomorphize_context(self.ctx_info.by_def[d])
        cpp_ty = self._compile_type(ty, ctx)
        return ty, ctx, cpp_ty

    def _expr_type(self, e: Expr):
        ty = self.type_info.by_expr[e]
        ctx = self._monomorphize_context(self.ctx_info.by_expr[e])
        cpp_ty = self._compile_type(ty, ctx)
        return ty, ctx, cpp_ty

    def _visit_var(self, e: Var, ctx: _CompileCtx):
        return str(e.name)

    def _visit_bool(self, e: BoolVal, ctx: _CompileCtx):
        return 'true' if e.val else 'false'

    def _visit_foreign(self, e: ForeignVal, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_number(self, e: Expr, s: str):
        _, _, cpp_ty = self._expr_type(e)
        # TODO: check context for rounding
        match cpp_ty:
            case CppType.DOUBLE:
                if '.' in s:
                    return s
                else:
                    return f'{s}.0'
            case CppType.FLOAT:
                return f'{s}f'
            case CppType.BOOL:
                raise RuntimeError(f'should not get {cpp_ty} for {e.format()}')
            case _:
                raise RuntimeError(f'unreachable: {cpp_ty}')

    def _visit_decnum(self, e: Decnum, ctx: _CompileCtx):
        return self._visit_number(e, str(e.val))

    def _visit_hexnum(self, e: Hexnum, ctx: _CompileCtx):
        return str(e.val)

    def _visit_integer(self, e: Integer, ctx: _CompileCtx):
        return self._visit_number(e, str(e.val))

    def _visit_rational(self, e: Rational, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_digits(self, e: Digits, ctx: _CompileCtx):
        # TODO: this is incorrect since it rounds
        return str(float(e.as_rational()))

    def _visit_nullaryop(self, e: NullaryOp, ctx: _CompileCtx):
        raise NotImplementedError

    def _visit_range(self, arg: Expr, ctx: _CompileCtx):
        arg_cpp = self._visit_expr(arg, ctx)
        _, _, cpp_ty = self._expr_type(arg)

        # TODO: check that `arg` may be safely cast to `size_t`
        t = self._fresh_var()
        ctx.add_line(f'std::vector<{cpp_ty.cpp_name}> {t}(static_cast<size_t>({arg_cpp}));')
        ctx.add_line(f'std::iota({t}.begin(), {t}.end(), static_cast<{cpp_ty.cpp_name}>(0));')
        return t

    def _visit_unaryop(self, e: UnaryOp, ctx: _CompileCtx):
        match e:
            case Range():
                return self._visit_range(e.arg, ctx)
            case Neg():
                # Handle negation with prefix operator
                arg = self._visit_expr(e.arg, ctx)
                return f'(-{arg})'
            case _:
                # Check unary operator table
                cls = type(e)
                if cls in self.op_table.unary:
                    ops = self.op_table.unary[cls]
                    arg = self._visit_expr(e.arg, ctx)
                    _, _, arg_ty = self._expr_type(e.arg)
                    _, _, e_ty = self._expr_type(e)
                    for op in ops:
                        if op.matches(arg_ty, e_ty):
                            return op.format(arg)
                    
                    # TODO: list options vs. actual signature
                    raise CppCompileError(self.func, f'no matching signature for `{e.format()}`')
                else:
                    raise CppCompileError(self.func, f'no matching operator for `{e.format()}`')

    def _visit_binaryop(self, e: BinaryOp, ctx: _CompileCtx):
        # compile children
        lhs = self._visit_expr(e.first, ctx)
        rhs = self._visit_expr(e.second, ctx)

        # check operator table
        cls = type(e)
        if cls in self.op_table.binary:
            ops = self.op_table.binary[cls]
            _, _, lhs_ty = self._expr_type(e.first)
            _, _, rhs_ty = self._expr_type(e.second)
            _, _, e_ty = self._expr_type(e)
            for op in ops:
                if op.matches(lhs_ty, rhs_ty, e_ty):
                    return op.format(lhs, rhs)

            # TODO: list options vs. actual signature
            raise CppCompileError(self.func, f'no matching signature for `{e.format()}`')
        else:
            raise CppCompileError(self.func, f'no matching operator for `{e.format()}`')

    def _visit_ternaryop(self, e: TernaryOp, ctx: _CompileCtx):
        # check operator table
        cls = type(e)
        if cls in self.op_table.ternary:
            ops = self.op_table.ternary[cls]
            arg1 = self._visit_expr(e.first, ctx)
            arg2 = self._visit_expr(e.second, ctx)
            arg3 = self._visit_expr(e.third, ctx)
            _, _, arg1_ty = self._expr_type(e.first)
            _, _, arg2_ty = self._expr_type(e.second)
            _, _, arg3_ty = self._expr_type(e.third)
            _, _, e_ty = self._expr_type(e)
            for op in ops:
                if op.matches(arg1_ty, arg2_ty, arg3_ty, e_ty):
                    return op.format(arg1, arg2, arg3)
            
            # TODO: list options vs. actual signature
            raise CppCompileError(self.func, f'no matching signature for `{e.format()}`')
        else:
            raise CppCompileError(self.func, f'no matching operator for `{e.format()}`')

    def _visit_naryop(self, e: NaryOp, ctx: _CompileCtx):
        # Handle n-ary operations
        match e:
            case And():
                # Logical AND: compile as (arg1 && arg2 && ...)
                if not e.args:
                    return 'true'
                args = [self._visit_expr(arg, ctx) for arg in e.args]
                return '(' + ' && '.join(args) + ')'
            case Or():
                # Logical OR: compile as (arg1 || arg2 || ...)
                if not e.args:
                    return 'false'
                args = [self._visit_expr(arg, ctx) for arg in e.args]
                return '(' + ' || '.join(args) + ')'
            case _:
                raise CppCompileError(self.func, f'unsupported n-ary operation: `{e.format()}`')

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
                    _, _, cpp_ty = self._var_type(stmt.binding, stmt)
                    ctx.add_line(f'{cpp_ty.cpp_name} {stmt.binding} = {e};')
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
            arg_strs.append(f'{ty.cpp_name} {arg.name}')

        body_ctx = self._monomorphize_context(self.ctx_info.return_ctx)
        ret_ty = self._compile_type(self.type_info.return_type, body_ctx)
        ctx.add_line(f'{ret_ty.cpp_name} {func.name}({", ".join(arg_strs)}) {{')

        # compile body
        self._visit_block(func.body, ctx)
        ctx.add_line('}')  # close function definition


_HEADER = [
    '#include <cassert>',
    '#include <cmath>',
    '#include <cstddef>',
    '#include <cstdint>',
    '#include <numeric>',
    '#include <vector>',
]


class CppBackend(Backend):
    """
    Compiler from FPy to C++.
    """

    std: CppStandard
    op_table: ScalarOpTable

    def __init__(
        self,
        std: CppStandard = CppStandard.CXX_11,
        op_table: ScalarOpTable | None = None
    ):
        if op_table is None:
            op_table = make_op_table()

        self.std = std
        self.op_table = op_table

    def headers(self) -> str:
        headers = _HEADER
        return '\n'.join(headers)

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
        inst = _CppBackendInstance(ast, self.std, self.op_table, ctx, def_use, type_info, ctx_info)
        body_str =  inst.compile()
        return body_str
