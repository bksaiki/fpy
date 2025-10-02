"""
C++ backend: compiler to C++
"""

import enum
import dataclasses

from typing import Iterable

from ...ast import *
from ...analysis import (
    ContextAnalysis, ContextInfer, ContextInferError,
    DefineUse, DefineUseAnalysis, Definition, DefSite, AssignDef, PhiDef,
    TypeInferError
)
from ...analysis.types import *
from ...analysis.context_types import *
from ...function import Function
from ...number import (
    RM,
    FP64, FP32,
    SINT8, SINT16, SINT32, SINT64,
    UINT8, UINT16, UINT32, UINT64,
    INTEGER
)
from ...primitive import Primitive
from ...transform import ConstFold
from ...utils import Gensym, enum_repr

from ..backend import Backend, CompileError

from .ops import UnaryCppOp, BinaryCppOp, TernaryCppOp, ScalarOpTable, make_op_table
from .types import CppType, CppScalar, CppList, CppTuple
from .utils import CPP_HEADERS, CPP_HELPERS


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
class _CppOptions:
    std: CppStandard
    op_table: ScalarOpTable
    unsafe_finitize_int: bool
    unsafe_cast_int: bool


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

    def dedent(self):
        assert self.indent_level > 0
        return _CompileCtx(self.lines, self.indent_str, self.indent_level - 1)

    def add_line(self, line: str):
        self.lines.append(self.indent_str * self.indent_level + line)


class _CppBackendInstance(Visitor):
    """
    Per-function compilation instance.
    """

    options: _CppOptions
    func: FuncDef

    arg_ctxs: tuple[Context | None, ...]
    caller_ctx: Context | None
    def_use: DefineUseAnalysis
    ctx_info: ContextAnalysis
    ctx_args: dict[NamedId, Context]

    decl_phis: set[PhiDef]
    decl_assigns: set[AssignDef]
    gensym: Gensym

    def __init__(
        self,
        options: _CppOptions,
        func: FuncDef,
        arg_ctxs: Iterable[Context | None],
        caller_ctx: Context | None,
        def_use: DefineUseAnalysis,
        ctx_info: ContextAnalysis
    ):
        self.options = options
        self.func = func

        self.arg_ctxs = tuple(arg_ctxs)
        self.caller_ctx = caller_ctx
        self.def_use = def_use
        self.ctx_info = ctx_info
        self.ctx_args = {}

        self.decl_phis = set()
        self.decl_assigns = set()
        self.gensym = Gensym(self.def_use.names())

    def compile(self):
        ctx = _CompileCtx.default()
        self._visit_function(self.func, ctx)
        return '\n'.join(ctx.lines)


    def _override_arg_type(self, ty: TypeContext, ctx: Context | None):
        match ty:
            case BoolTypeContext() | ContextTypeContext():
                pass
            case RealTypeContext():
                if isinstance(ty.ctx, NamedId) and isinstance(ctx, Context):
                    self.ctx_args[ty.ctx] = ctx
            case TupleTypeContext():
                # TODO: how does user specify overriding contexts for tuple arguments>
                for t in ty.elts:
                    self._override_arg_type(t, ctx)
            case ListTypeContext():
                self._override_arg_type(ty.elt, ctx)
            case _:
                raise RuntimeError(f'unreachable: {ty}')

    def _monomorphize_type(self, ty: TypeContext):
        match ty:
            case VarTypeContext():
                raise CppCompileError(self.func, f'types must be monomorphic `{ty}`')
            case BoolTypeContext():
                return ty
            case RealTypeContext():
                if isinstance(ty.ctx, NamedId):
                    if ty.ctx not in self.ctx_args:
                        raise CppCompileError(self.func, f'types must be monomorphic `{ty}`')
                    return RealTypeContext(self.ctx_args[ty.ctx])
                else:
                    return ty
            case TupleTypeContext():
                return TupleTypeContext(*(self._monomorphize_type(t) for t in ty.elts))
            case ListTypeContext():
                return ListTypeContext(self._monomorphize_type(ty.elt))
            case _:
                raise RuntimeError(f'unreachable: {ty}')

    def _compile_type(self, ty: TypeContext):
        match ty:
            case BoolTypeContext():
                return CppScalar.BOOL
            case RealTypeContext():
                if isinstance(ty.ctx, NamedId):
                    raise CppCompileError(self.func, f'types must be monomorphic: `{ty}`')
                else:
                    if ty.ctx.is_equiv(FP64):
                        return CppScalar.F64
                    elif ty.ctx.is_equiv(FP32):
                        return CppScalar.F32
                    elif ty.ctx.is_equiv(SINT8):
                        return CppScalar.S8
                    elif ty.ctx.is_equiv(SINT16):
                        return CppScalar.S16
                    elif ty.ctx.is_equiv(SINT32):
                        return CppScalar.S32
                    elif ty.ctx.is_equiv(SINT64):
                        return CppScalar.S64
                    elif ty.ctx.is_equiv(UINT8):
                        return CppScalar.U8
                    elif ty.ctx.is_equiv(UINT16):
                        return CppScalar.U16
                    elif ty.ctx.is_equiv(UINT32):
                        return CppScalar.U32
                    elif ty.ctx.is_equiv(UINT64):
                        return CppScalar.U64
                    elif ty.ctx.is_equiv(INTEGER):
                        if not self.options.unsafe_finitize_int:
                            raise CppCompileError(self.func, 'integer type not allowed (set `unsafe_allow_int=True` to override)')
                        return CppScalar.S64
                    else:
                        raise CppCompileError(self.func, f'unsupported context: `{ty.ctx}`')
            case TupleTypeContext():
                return CppTuple(self._compile_type(t) for t in ty.elts)
            case ListTypeContext():
                return CppList(self._compile_type(ty.elt))
            case _:
                raise CppCompileError(self.func, f'unsupported type: `{ty.format()}`')

    def _compile_rm(self, rm: RM):
        match rm:
            case RM.RNE:
                return 'FE_TONEAREST'
            case RM.RTZ:
                return 'FE_TOWARDZERO'
            case RM.RTP:
                return 'FE_UPWARD'
            case RM.RTN:
                return 'FE_DOWNWARD'
            case _:
                raise CppCompileError(self.func, f'unsupported rounding mode: `{rm}`')

    def _fresh_var(self):
        return str(self.gensym.fresh('__fpy_tmp'))

    def _var_is_decl(self, name: NamedId, site: DefSite):
        d = self.def_use.find_def_from_site(name, site)
        return d.prev is None and d not in self.decl_assigns

    def _def_type(self, d: Definition):
        ty = self._monomorphize_type(self.ctx_info.by_def[d])
        cpp_ty = self._compile_type(ty)
        return ty, cpp_ty

    def _var_type(self, name: NamedId, site: DefSite):
        d = self.def_use.find_def_from_site(name, site)
        return self._def_type(d)

    def _expr_type(self, e: Expr):
        ty = self._monomorphize_type(self.ctx_info.by_expr[e])
        cpp_ty = self._compile_type(ty)
        return ty, cpp_ty

    def _visit_var(self, e: Var, ctx: _CompileCtx):
        return str(e.name)

    def _visit_bool(self, e: BoolVal, ctx: _CompileCtx):
        return 'true' if e.val else 'false'

    def _visit_foreign(self, e: ForeignVal, ctx: _CompileCtx):
        raise NotImplementedError

    def _compile_number(self, s: str, ty: CppType):
        # TODO: check context for rounding
        match ty:
            case CppScalar.F64:
                if '.' in s:
                    return s
                else:
                    return f'{s}.0'
            case CppScalar.F32:
                if '.' in s:
                    return f'{s}f'
                else:
                    return f'{s}.0f'
            case CppScalar.U8 | CppScalar.U16 | CppScalar.U32:
                return f'{s}u'
            case CppScalar.U64:
                return f'{s}ull'
            case CppScalar.S8 | CppScalar.S16 | CppScalar.S32:
                return f'{s}'
            case CppScalar.S64:
                return f'{s}ll'
            case CppScalar.BOOL:
                raise RuntimeError(f'a number cannot be a boolean: {s}')
            case _:
                raise RuntimeError(f'unreachable: {ty}')

    def _visit_decnum(self, e: Decnum, ctx: _CompileCtx):
        if self.options.unsafe_cast_int and e.is_integer():
            # unsafe override: treat as unbounded integer
            v = int(e.as_rational())
            _, e_ty = self._expr_type(e)
            return self._compile_number(str(v), e_ty)
        else:
            # otherwise unsupported
            raise CppCompileError(self.func, 'unrounded literals are unsupported')

    def _visit_hexnum(self, e: Hexnum, ctx: _CompileCtx):
        if self.options.unsafe_cast_int and e.is_integer():
            # unsafe override: treat as unbounded integer
            v = int(e.as_rational())
            _, e_ty = self._expr_type(e)
            return self._compile_number(str(v), e_ty)
        else:
            # otherwise unsupported
            raise CppCompileError(self.func, 'unrounded literals are unsupported')

    def _visit_integer(self, e: Integer, ctx: _CompileCtx):
        if self.options.unsafe_cast_int and e.is_integer():
            # unsafe override: treat as unbounded integer
            v = int(e.as_rational())
            _, e_ty = self._expr_type(e)
            return self._compile_number(str(v), e_ty)
        else:
            # otherwise unsupported
            raise CppCompileError(self.func, 'unrounded literals are unsupported')

    def _visit_rational(self, e: Rational, ctx: _CompileCtx):
        if self.options.unsafe_cast_int and e.is_integer():
            # unsafe override: treat as unbounded integer
            v = int(e.as_rational())
            _, e_ty = self._expr_type(e)
            return self._compile_number(str(v), e_ty)
        else:
            # otherwise unsupported
            raise CppCompileError(self.func, 'unrounded literals are unsupported')

    def _visit_digits(self, e: Digits, ctx: _CompileCtx):
        if self.options.unsafe_cast_int and e.is_integer():
            # unsafe override: treat as unbounded integer
            v = int(e.as_rational())
            _, e_ty = self._expr_type(e)
            return self._compile_number(str(v), e_ty)
        else:
            # otherwise unsupported
            raise CppCompileError(self.func, 'unrounded literals are unsupported')

    def _visit_nullaryop(self, e: NullaryOp, ctx: _CompileCtx):
        raise NotImplementedError

    def _compile_size(self, e: str, ty: CppType):
        match ty:
            case CppScalar():
                # TODO: check that `arg` may be safely cast to `size_t`
                return f'static_cast<size_t>({e})'
            case _:
                raise RuntimeError(f'unexpected size type: {ty}')

    def _visit_real_val(self, e: RealVal, ty: CppType):
        match e:
            case Decnum():
                return self._compile_number(str(e.val), ty)
            case Hexnum():
                raise CppCompileError(self.func, 'hexadecimal literals are unsupported')
            case Integer():
                return self._compile_number(str(e.val), ty)
            case Rational():
                raise CppCompileError(self.func, 'rational values are unsupported')
            case Digits():
                raise CppCompileError(self.func, '`digits(m, e, b)` is unsupported')
            case _:
                raise RuntimeError(f'unreachable: {e}')

    def _visit_round(self, e: Round | RoundExact, ctx: _CompileCtx):
        if isinstance(e.arg, RealVal):
            # special case: rounded literal
            _, e_ty = self._expr_type(e)
            return self._visit_real_val(e.arg, e_ty)
        else:
            # general case: unary operator
            arg = self._visit_expr(e.arg, ctx)
            _, arg_ty = self._expr_type(e.arg)
            _, e_ty = self._expr_type(e)
            for op in self.options.op_table.unary[type(e)]:
                if op.matches(arg_ty, e_ty):
                    return op.format(arg)

            # TODO: list options vs. actual signature
            raise CppCompileError(self.func, f'no matching signature for `{e.format()}`')


    def _visit_round_at(self, e: RoundAt, ctx: _CompileCtx):
        raise CppCompileError(self.func, '`round_at` is unsupported')

    def _visit_len(self, e: Len, ctx: _CompileCtx):
        # len(x)
        arg_cpp = self._visit_expr(e.arg, ctx)
        _, cpp_ty = self._expr_type(e)
        return f'static_cast<{cpp_ty.format()}>({arg_cpp}.size())'

    def _visit_range(self, e: Range, ctx: _CompileCtx):
        # range(n)
        arg_cpp = self._visit_expr(e.arg, ctx)
        _, e_ty = self._expr_type(e)
        assert isinstance(e_ty, CppList) and isinstance(e_ty.elt, CppScalar)

        t = self._fresh_var()
        s = self._compile_size(arg_cpp, e_ty.elt)
        ctx.add_line(f'std::vector<{e_ty.elt.format()}> {t}({s});')
        ctx.add_line(f'std::iota({t}.begin(), {t}.end(), static_cast<{e_ty.elt.format()}>(0));')
        return t

    def _visit_size(self, arr: Expr, dim: Expr, ctx: _CompileCtx):
        # size(x, n)
        arr_cpp = self._visit_expr(arr, ctx)
        dim_cpp = self._visit_expr(dim, ctx)
        _, dim_ty = self._expr_type(dim)

        n = self._compile_size(dim_cpp, dim_ty)
        return f'size({arr_cpp}, {n})'

    def _visit_dim(self, e: Dim, ctx: _CompileCtx):
        # dim(x) => statically determined from type information
        t = self._fresh_var()
        e_str = self._visit_expr(e.arg, ctx)
        _, arg_ty = self._expr_type(e.arg)
        ctx.add_line(f'auto {t} = {e_str};') # bind it temporarily (unused)
        assert isinstance(arg_ty, CppList)
        # compute the size and cast to the expected type
        dim = arg_ty.dim()
        _, e_ty = self._expr_type(e)
        return f'static_cast<{e_ty.format()}>({dim})'

    def _visit_enumerate(self, e: Enumerate, ctx: _CompileCtx):
        # enumerate(x: T) =>
        # x = <x>;
        # std::vector<std::tuple<R, T>> t(x.size());
        # for (size_t i = 0; i < x.size(); ++i) {
        #     t[i] = std::make_tuple(static_cast<R>(i), x[i]);
        # }
        x = self._fresh_var()
        t = self._fresh_var()
        i = self._fresh_var()

        arr_cpp = self._visit_expr(e.arg, ctx)
        _, e_ty = self._expr_type(e)
        if not isinstance(e_ty, CppList) or not isinstance(e_ty.elt, CppTuple):
            raise CppCompileError(self.func, f'expected list or tuple for `{e.format()}`')
        enum_ty = e_ty.elt.elts[0]

        ctx.add_line(f'auto {x} = {arr_cpp};')
        ctx.add_line(f'{e_ty.format()} {t}({arr_cpp}.size());')
        ctx.add_line(f'for (size_t {i} = 0; {i} < {x}.size(); ++{i}) {{')
        ctx.indent().add_line(f'{t}[{i}] = std::make_tuple(static_cast<{enum_ty.format()}>({i}), {x}[{i}]);')
        ctx.add_line('}')

        return t

    def _visit_sum(self, e: Sum, ctx: _CompileCtx):
        arg_str = ', '.join(self._visit_expr(arg, ctx) for arg in e.args)
        _, cpp_ty = self._expr_type(e)
        return f'std::accumulate({arg_str}.begin(), {arg_str}.end(), static_cast<{cpp_ty.format()}>(0))'

    def _visit_zip(self, e: Zip, ctx: _CompileCtx):
        # zip(x1, ..., xn) =>
        # auto x1 = <x1>;
        # ...
        # auto xn = <xn>;
        # // assert <x1.size() == ... == xn.size()>
        # std::vector<std::tuple<decltype(x1)::value_type, ..., decltype(x1)::value_type>> t(x1.size());
        # for (size_t i = 0; i < x1.size(); ++i) {
        #     t[i] = std::make_tuple(x1[i], ..., xn[i]);
        # }

        xs: list[str] = []
        arg_tys: list[CppType] = []
        for arg in e.args:
            x = self._fresh_var()
            arg_cpp = self._visit_expr(arg, ctx)
            ctx.add_line(f'auto {x} = {arg_cpp};')
            xs.append(x)

            _, arg_ty = self._expr_type(arg)
            if not isinstance(arg_ty, CppList):
                raise CppCompileError(self.func, f'expected list for `{arg.format()}`')
            arg_tys.append(arg_ty.elt)

        t = self._fresh_var()
        i = self._fresh_var()
        tmpl = ', '.join(ty.format() for ty in arg_tys)
        ctx.add_line(f'std::vector<std::tuple<{tmpl}>> {t}({xs[0]}.size());')
        ctx.add_line(f'for (size_t {i} = 0; {i} < {xs[0]}.size(); ++{i}) {{')
        ctx.indent().add_line(f'{t}[{i}] = std::make_tuple({", ".join(f"{x}[{i}]" for x in xs)});')
        ctx.add_line('}')

        return t

    def _visit_mxn2(self, e: Min | Max, x1: str, x2: str, x1_ty: CppScalar, x2_ty: CppScalar, e_ty: CppScalar):
        # min(x1, x2) => std::fmin(<x1>, <x2>)
        for op in self.options.op_table.binary[type(e)]:
            if op.matches(x1_ty, x2_ty, e_ty):
                return op.format(x1, x2)
        raise CppCompileError(self.func, f'no matching signature for {e.format()}')

    def _visit_mxn(self, e: Min | Max, ctx: _CompileCtx):
        # min(x1, ..., xn)
        assert len(e.args) >= 2

        # initial 2 arguments
        # <result> = std::fmin(<arg0>, <arg1>)
        x1 = self._visit_expr(e.args[0], ctx)
        x2 = self._visit_expr(e.args[1], ctx)
        _, x1_ty = self._expr_type(e.args[0])
        _, x2_ty = self._expr_type(e.args[1])
        _, e_ty = self._expr_type(e)
        s = self._visit_mxn2(e, x1, x2, x1_ty, x2_ty, e_ty)

        # remaining arguments
        for arg in e.args[2:]:
            arg_cpp = self._visit_expr(arg, ctx)
            _, arg_ty = self._expr_type(arg)
            s = self._visit_mxn2(e, s, arg_cpp, e_ty, arg_ty, e_ty)

        return s

    def _visit_unaryop(self, e: UnaryOp, ctx: _CompileCtx):
        # Check unary operator table
        cls = type(e)
        if cls in self.options.op_table.unary:
            ops = self.options.op_table.unary[cls]
            arg = self._visit_expr(e.arg, ctx)
            _, arg_ty = self._expr_type(e.arg)
            _, e_ty = self._expr_type(e)
            for op in ops:
                if op.matches(arg_ty, e_ty):
                    return op.format(arg)

            # TODO: list options vs. actual signature
            raise CppCompileError(self.func, f'no matching signature for `{e.format()}`')

        # fallback
        match e:
            case Len():
                return self._visit_len(e, ctx)
            case Range():
                return self._visit_range(e, ctx)
            case Dim():
                return self._visit_dim(e, ctx)
            case Enumerate():
                return self._visit_enumerate(e, ctx)
            case Sum():
                return self._visit_sum(e, ctx)
            case _:
                raise CppCompileError(self.func, f'no matching operator for `{e.format()}`')

    def _visit_binaryop(self, e: BinaryOp, ctx: _CompileCtx):
        # compile children
        lhs = self._visit_expr(e.first, ctx)
        rhs = self._visit_expr(e.second, ctx)

        # check operator table
        cls = type(e)
        if cls in self.options.op_table.binary:
            ops = self.options.op_table.binary[cls]
            _, lhs_ty = self._expr_type(e.first)
            _, rhs_ty = self._expr_type(e.second)
            _, e_ty = self._expr_type(e)
            for op in ops:
                if op.matches(lhs_ty, rhs_ty, e_ty):
                    return op.format(lhs, rhs)

            # TODO: list options vs. actual signature
            raise CppCompileError(self.func, f'no matching signature for `{e.format()}`')

        # fallback
        match e:
            case Size():
                return self._visit_size(e.first, e.second, ctx)
            case _:
                raise CppCompileError(self.func, f'no matching operator for `{e.format()}`')

    def _visit_ternaryop(self, e: TernaryOp, ctx: _CompileCtx):
        # check operator table
        cls = type(e)
        if cls in self.options.op_table.ternary:
            ops = self.options.op_table.ternary[cls]
            arg1 = self._visit_expr(e.first, ctx)
            arg2 = self._visit_expr(e.second, ctx)
            arg3 = self._visit_expr(e.third, ctx)
            _, arg1_ty = self._expr_type(e.first)
            _, arg2_ty = self._expr_type(e.second)
            _, arg3_ty = self._expr_type(e.third)
            _, e_ty = self._expr_type(e)
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
            case Zip():
                return self._visit_zip(e, ctx)
            case Min() | Max():
                return self._visit_mxn(e, ctx)
            case And():
                # Logical AND: compile as (arg1 && arg2 && ...)
                assert len(e.args) >= 2
                args = [self._visit_expr(arg, ctx) for arg in e.args]
                return '(' + ' && '.join(args) + ')'
            case Or():
                # Logical OR: compile as (arg1 || arg2 || ...)
                assert len(e.args) >= 2
                args = [self._visit_expr(arg, ctx) for arg in e.args]
                return '(' + ' || '.join(args) + ')'
            case _:
                raise CppCompileError(self.func, f'unsupported n-ary operation: `{e.format()}`')

    def _visit_compare2(self, op: CompareOp, lhs: str, rhs: str):
        return f'({lhs} {op.symbol()} {rhs})'

    def _visit_call(self, e: Call, ctx: _CompileCtx):
        match e.fn:
            case Function():
                # function call
                # TODO: monomorphization
                args = [self._visit_expr(arg, ctx) for arg in e.args]
                return f'{e.fn.name}({", ".join(args)})'
            case Primitive():
                # primitive call
                if e.fn in self.options.op_table.prims:
                    args = [self._visit_expr(arg, ctx) for arg in e.args]
                    arg_tys: list[CppScalar] = []
                    for arg in e.args:
                        _, arg_ty = self._expr_type(arg)
                        arg_tys.append(arg_ty)
                    _, e_ty = self._expr_type(e)

                    ops = self.options.op_table.prims[e.fn]
                    for op in ops:
                        if isinstance(op, UnaryCppOp) and len(args) == 1 and op.matches(arg_tys[0], e_ty):
                            return op.format(args[0])
                        elif isinstance(op, BinaryCppOp) and len(args) == 2 and op.matches(arg_tys[0], arg_tys[1], e_ty):
                            return op.format(args[0], args[1])
                        elif isinstance(op, TernaryCppOp) and len(args) == 3 and op.matches(arg_tys[0], arg_tys[1], arg_tys[2], e_ty):
                            return op.format(args[0], args[1], args[2])
                    raise CppCompileError(self.func, f'no matching signature for primitive call `{e.format()}`')
                else:
                    raise CppCompileError(self.func, f'cannot compile unsupported primitive `{e.format()}`')
            case _:
                # unknown call
                raise CppCompileError(self.func, f'cannot compile unsupported call to `{e.format()}`')

    def _visit_compare(self, e: Compare, ctx: _CompileCtx):
        if len(e.args) == 2:
            # easy case: 2-argument comparison
            lhs = self._visit_expr(e.args[0], ctx)
            rhs = self._visit_expr(e.args[1], ctx)
            return self._visit_compare2(e.ops[0], lhs, rhs)
        else:
            # harder case:
            # - emit temporaries to bind expressions
            # - form (t1 op t2) && (t2 op t3) && ...
            args: list[str] = []
            for arg in e.args:
                t = self._fresh_var()
                cpp_arg = self._visit_expr(arg, ctx)
                ctx.add_line(f'auto {t} = {cpp_arg};')
                args.append(t)

            return ' && '.join(self._visit_compare2(e.ops[i], args[i], args[i + 1]) for i in range(len(args) - 1))

    def _visit_tuple_expr(self, e: TupleExpr, ctx: _CompileCtx):
        args = [self._visit_expr(arg, ctx) for arg in e.elts]
        return f'std::make_tuple({", ".join(args)})'

    def _visit_list_expr(self, e: ListExpr, ctx: _CompileCtx):
        _, cpp_ty = self._expr_type(e)
        args = ', '.join(self._visit_expr(arg, ctx) for arg in e.elts)
        return f'{cpp_ty.format()}({{{args}}})'

    def _visit_list_comp(self, e: ListComp, ctx: _CompileCtx):
        # [<e> for <x1> in <iterable1> ... for <xN> in <iterableN>] =>
        # auto <t1> = <iterable1>;
        # ...
        # auto <tN> = <iterableN>;
        # std::vector<T> t(t1.size() * ... * t2.size());
        # size_t i = 0;
        # for (auto <x1> : <t1>) {
        #   ...
        #      for (auto <xN> : <tN>) {
        #         t[i] = <e>;
        #         i++;
        #      }
        # }

        # emit temporaries for iterables
        ts = [self._fresh_var() for _ in e.iterables]
        for t, iterable in zip(ts, e.iterables):
            ctx.add_line(f'auto {t} = {self._visit_expr(iterable, ctx)};')

        # reserve output vector
        v = self._fresh_var()
        _, cpp_ty = self._expr_type(e)
        size = ' * '.join(f'{t}.size()' for t in ts)
        ctx.add_line(f'{cpp_ty.format()} {v}({size});')

        # initialize the counter
        i = self._fresh_var()
        ctx.add_line(f'size_t {i} = 0;')

        # fill the vector
        body_ctx = ctx
        for target, t in zip(e.targets, ts):
            match target:
                case Id():
                    body_ctx.add_line(f'for (auto {target} : {t}) {{')
                case TupleBinding():
                    t2 = self._fresh_var()
                    body_ctx.add_line(f'for (auto {t2} : {t}) {{')
                    self._visit_tuple_binding(t2, target, e, body_ctx.indent())
                case _:
                    raise RuntimeError(f'unreachable {target}')
            body_ctx = body_ctx.indent()

        # compute the element
        elt = self._visit_expr(e.elt, body_ctx)
        body_ctx.add_line(f'{v}[{i}] = {elt};')
        body_ctx.add_line(f'{i}++;')

        # closing braces
        while body_ctx.indent_level > ctx.indent_level:
            body_ctx = body_ctx.dedent()
            body_ctx.add_line('}')

        return v

    def _visit_list_ref(self, e: ListRef, ctx: _CompileCtx):
        value = self._visit_expr(e.value, ctx)
        index = self._visit_expr(e.index, ctx)

        _, index_ty = self._expr_type(e.index)
        size = self._compile_size(index, index_ty)
        return f'{value}[{size}]'

    def _visit_list_slice(self, e: ListSlice, ctx: _CompileCtx):
        # x[<start>:<stop>] =>
        # auto v = <x>
        # auto start = static_cast<size_t>(<start>) OR 0;
        # auto stop = static_cast<size_t>(<stop>) OR x.size();
        # <result> = std::vector<T>(v.begin() + start, v.end() + end);

        # temporarily bind array
        t = self._fresh_var()
        arr = self._visit_expr(e.value, ctx)
        _, e_ty = self._expr_type(e)
        ctx.add_line(f'auto {t} = {arr};')

        # compile start
        if e.start is None:
            start = '0'
        else:
            start = self._visit_expr(e.start, ctx)
            _, start_ty = self._expr_type(e.start)
            start = self._compile_size(start, start_ty)

        # compile stop
        if e.stop is None:
            stop = f'{t}.size()'
        else:
            stop = self._visit_expr(e.stop, ctx)
            _, stop_ty = self._expr_type(e.stop)
            stop = self._compile_size(stop, stop_ty)

        # result
        return f'{e_ty.format()}({t}.begin() + {start}, {t}.begin() + {stop})'

    def _visit_list_set(self, e: ListSet, ctx: _CompileCtx):
        raise CompileError('functional list updates is unsupported')

    def _visit_if_expr(self, e: IfExpr, ctx: _CompileCtx):
        cond = self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, ctx)
        iff = self._visit_expr(e.iff, ctx)
        return f'({cond} ? {ift} : {iff})'

    def _visit_attribute(self, e: Attribute, ctx: _CompileCtx):
        raise CompileError('attributes are unsupported')

    def _visit_decl(self, name: Id, e: str, site: DefSite, ctx: _CompileCtx):
        match name:
            case NamedId():
                if self._var_is_decl(name, site):
                    _, cpp_ty = self._var_type(name, site)
                    ctx.add_line(f'{cpp_ty.format()} {name} = {e};')
                else:
                    ctx.add_line(f'{name} = {e};')
            case UnderscoreId():
                ctx.add_line(f'auto _ = {e};')
            case _:
                raise RuntimeError(f'unreachable: {name}')

    def _visit_tuple_binding(self, t_id: str, binding: TupleBinding, site: DefSite, ctx: _CompileCtx):
        for i, elt in enumerate(binding.elts):
            match elt:
                case NamedId():
                    # bind the ith element to the name
                    self._visit_decl(elt, f'std::get<{i}>({t_id})', site, ctx)
                case UnderscoreId():
                    # do nothing
                    pass
                case TupleBinding():
                    # emit temporary variable for the tuple
                    t = self._fresh_var()
                    ctx.add_line(f'auto {t} = std::get<{i}>({t_id});')
                    self._visit_tuple_binding(t, elt, site, ctx)
                case _:
                    raise RuntimeError(f'unreachable: {elt}')

    def _visit_assign(self, stmt: Assign, ctx: _CompileCtx):
        e = self._visit_expr(stmt.expr, ctx)
        match stmt.target:
            case Id():
                self._visit_decl(stmt.target, e, stmt, ctx)
            case TupleBinding():
                # emit temporary variable for the tuple
                t = self._fresh_var()
                ctx.add_line(f'auto {t} = {e};')
                self._visit_tuple_binding(t, stmt.target, stmt, ctx)
            case _:
                raise NotImplementedError(stmt.binding)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: _CompileCtx):
        # compile indices
        indices: list[str] = []
        for index in stmt.indices:
            i = self._visit_expr(index, ctx)
            _, index_ty = self._expr_type(index)
            indices.append(self._compile_size(i, index_ty))

        # compile expression
        e = self._visit_expr(stmt.expr, ctx)
        index_str = ''.join(f'[{i}]' for i in indices)
        ctx.add_line(f'{stmt.var}{index_str} = {e};')

    def _visit_if1(self, stmt: If1Stmt, ctx: _CompileCtx):
        cond = self._visit_expr(stmt.cond, ctx)
        ctx.add_line(f'if ({cond}) {{')
        self._visit_block(stmt.body, ctx.indent())
        ctx.add_line('}')  # close if

    def _visit_if(self, stmt: IfStmt, ctx: _CompileCtx):
        # variables need to be declared if they are assigned
        # within the branches but not before defined in the branches
        for phi in self.def_use.phis[stmt]:
            if phi.is_intro and phi not in self.decl_phis:
                # need a declaration for the phi assignment
                _, cpp_ty = self._def_type(phi)
                ctx.add_line(f'{cpp_ty.format()} {phi.name}; // phi')
                # record this phi so that we don't revisit it
                self.decl_phis |= self.def_use.phi_prevs(phi)
                self.decl_assigns |= self.def_use.roots_of(phi)

        # TODO: fold if statements
        cond = self._visit_expr(stmt.cond, ctx)
        ctx.add_line(f'if ({cond}) {{')
        self._visit_block(stmt.ift, ctx.indent())
        ctx.add_line('} else {')
        self._visit_block(stmt.iff, ctx.indent())
        ctx.add_line('}')  # close if

    def _visit_while(self, stmt: WhileStmt, ctx: _CompileCtx):
        cond = self._visit_expr(stmt.cond, ctx)
        ctx.add_line(f'while ({cond}) {{')
        self._visit_block(stmt.body, ctx.indent())
        ctx.add_line('}')  # close if

    def _visit_for(self, stmt: ForStmt, ctx: _CompileCtx):
        iterable = self._visit_expr(stmt.iterable, ctx)
        match stmt.target:
            case Id():
                ctx.add_line(f'for (auto {stmt.target} : {iterable}) {{')
            case TupleBinding():
                t = self._fresh_var()
                ctx.add_line(f'for (auto {t} : {iterable}) {{')
                self._visit_tuple_binding(t, stmt.target, stmt, ctx.indent())
            case _:
                raise RuntimeError(f'unreachable {ctx}')

        self._visit_block(stmt.body, ctx.indent())
        ctx.add_line('}')  # close if

    def _visit_context(self, stmt: ContextStmt, ctx: _CompileCtx):
        if not isinstance(stmt.ctx, ForeignVal):
            raise CppCompileError(self.func, f'Rounding context cannot be compiled `{stmt.ctx}`')
        if isinstance(stmt.target, NamedId):
            raise CppCompileError(self.func, f'Cannot bind rounding context in C++: `{stmt.format()}`')
        rctx = stmt.ctx.val
        cpp_ctx = self._compile_type(RealTypeContext(rctx))
        if cpp_ctx.is_float():
            assert isinstance(rctx, type(FP64))
            fenv = self._fresh_var()
            ctx.add_line(f'const auto {fenv} = fegetround();')
            ctx.add_line(f'fesetround({self._compile_rm(rctx.rm)});')
            self._visit_block(stmt.body, ctx)
            ctx.add_line(f'fesetround({fenv});')
        elif cpp_ctx.is_integer():
            # do nothing
            self._visit_block(stmt.body, ctx)
        else:
            raise RuntimeError(f'unexpected context: `{cpp_ctx}`')

    def _visit_assert(self, stmt: AssertStmt, ctx: _CompileCtx):
        e = self._visit_expr(stmt.test, ctx)
        if stmt.msg is None:
            ctx.add_line(f'assert({e});')
        else:
            msg = self._visit_expr(stmt.msg, ctx)
            ctx.add_line(f'assert(({e}) && ({msg}));')

    def _visit_effect(self, stmt: EffectStmt, ctx: _CompileCtx):
        raise CppCompileError(self.func, 'FPy effects are not supported')

    def _visit_return(self, stmt: ReturnStmt, ctx: _CompileCtx):
        e = self._visit_expr(stmt.expr, ctx)
        ctx.add_line(f'return {e};')

    def _visit_pass(self, stmt: PassStmt, ctx: _CompileCtx):
        pass

    def _visit_block(self, block: StmtBlock, ctx: _CompileCtx):
        for stmt in block.stmts:
            self._visit_statement(stmt, ctx)

    def _visit_function(self, func: FuncDef, ctx: _CompileCtx):
        # apply caller context
        if isinstance(self.ctx_info.body_ctx, NamedId):
            if self.caller_ctx is not None:
                self.ctx_args[self.ctx_info.body_ctx] = self.caller_ctx

        # apply argument contexts
        for caller_arg, arg_ty in zip(self.arg_ctxs, self.ctx_info.arg_types):
            self._override_arg_type(arg_ty, caller_arg)

        # compile arguments
        arg_strs: list[str] = []
        for arg, arg_ty in zip(func.args, self.ctx_info.arg_types):
            arg_ty = self._monomorphize_type(arg_ty)
            ty = self._compile_type(arg_ty)
            arg_strs.append(f'{ty.format()} {arg.name}')

        ret_ty = self._monomorphize_type(self.ctx_info.return_type)
        ty = self._compile_type(ret_ty)
        ctx.add_line(f'{ty.format()} {func.name}({", ".join(arg_strs)}) {{')

        # compile body
        self._visit_block(func.body, ctx.indent())
        ctx.add_line('}')  # close function definition


class CppBackend(Backend):
    """
    Compiler from FPy to C++.

    Major options:
    - `std: CppStandard`: the C++ standard to target [default: `CppStandard.CXX_11`]
    - `op_table: ScalarOpTable`: target description to use [default: `None`]

    Unsafe options:
    - `unsafe_finitize_int`: unbounded integers will be finitized to "int64_t" [default: `False`]
    - `unsafe_cast_int`: unrounded integer literals will be assumed to be unbounded integers [default: `False`]
    """

    # major options
    std: CppStandard
    op_table: ScalarOpTable

    # unsafe options
    unsafe_finitize_int: bool
    unsafe_cast_int: bool

    # cache of compiled functions
    compiled: dict[str, Function]

    def __init__(
        self,
        *,
        std: CppStandard = CppStandard.CXX_11,
        op_table: ScalarOpTable | None = None,
        unsafe_finitize_int: bool = False,
        unsafe_cast_int: bool = False
    ):
        if op_table is None:
            op_table = make_op_table()

        self.std = std
        self.op_table = op_table
        self.unsafe_finitize_int = unsafe_finitize_int
        self.unsafe_cast_int = unsafe_cast_int
        self.compiled = {}

    def headers(self) -> list[str]:
        """
        Returns the C++ headers required for compiling an FPy program.
        """
        return list(CPP_HEADERS)

    def helpers(self) -> str:
        """
        Returns C++ helper functions for compiled FPy programs.
        """
        return str(CPP_HELPERS)

    def compile(
        self,
        func: Function,
        *,
        arg_ctxs: None | Iterable[Context | None] = None,
        ctx: Context | None = None
    ) -> str:
        """
        Compiles the given FPy function to a C++ program
        represented as a string.
        """
        if not isinstance(func, Function):
            raise TypeError(f'Expected `Function`, got {type(func)} for {func}')
        if ctx is not None and not isinstance(ctx, Context):
            raise TypeError(f'Expected `Context`, got {type(ctx)} for {ctx}')
        if arg_ctxs is None:
            arg_ctxs = tuple(None for _ in func.args)

        if func.name in self.compiled:
            raise ValueError(f'Function `{func.name}` has already been compiled')
        self.compiled[func.name] = func

        # normalization passes
        ast = ConstFold.apply(func.ast, enable_op=False)

        # analyses
        def_use = DefineUse.analyze(ast)

        try:
            ctx_info = ContextInfer.infer(ast, def_use=def_use, unsafe_cast_int=self.unsafe_cast_int)
        except (ContextInferError, TypeInferError) as e:
            raise ValueError(f'{func.name}: context inference failed') from e

        # compile
        options = _CppOptions(self.std, self.op_table, self.unsafe_finitize_int, self.unsafe_cast_int)
        inst = _CppBackendInstance(options, ast, arg_ctxs, ctx, def_use, ctx_info)
        body_str =  inst.compile()
        return body_str
