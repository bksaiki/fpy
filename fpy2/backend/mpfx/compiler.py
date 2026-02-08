"""
MPFX backend: compiler to MPFX number library.
"""

from typing import Collection, NoReturn

from ...ast import *
from ...analysis import (
    ContextInfer, ContextInferError,
    DefineUse, DefineUseAnalysis, Definition, DefSite, AssignDef, PhiDef,
    PartialEval, PartialEvalInfo, TypeInferError
)
from ...function import Function
from ...libraries.core import ldexp
from ...number import EFloatContext, IEEEContext, ExpContext, OV, RM, FP64, INTEGER
from ...strategies import simplify
from ...transform import Monomorphize, FuncInline, LiftContext
from ...types import BoolType, ContextType, RealType, VarType, TupleType, ListType, Type
from ...utils import Gensym
from ..backend import Backend

from .elim_round import ElimRound
from .format import AbstractFormat, SupportedContext
from .format_infer import (
    FormatAnalysis, FormatInfer, FormatInferError,
    FormatType, ListFormatType, TupleFormatType
)
from .instr import InstrGenerator
from .types import *
from .utils import MPFXCompileError, CompileCtx, CppOptions


def _emit_error(msg, func: FuncDef | None = None) -> NoReturn:
    if func:
        raise MPFXCompileError(func, msg)
    else:
        raise RuntimeError(msg)

def _compile_type(ty: FormatType, func: FuncDef | None = None) -> CppType:
    match ty:
        case BoolType():
            return CppBoolType()
        case AbstractFormat():
            if ty <= AbstractFormat.from_context(FP64):
                # fits within a double 
                return CppDoubleType()
            elif ty <= AbstractFormat.from_context(INTEGER):
                # fits with an int64_t
                return CppInt64Type()
            else:
                _emit_error(f'no container type for: {ty}', func)
        case TupleFormatType():
            # compile recursively
            return CppTupleType(_compile_type(elem_ty) for elem_ty in ty.elts)
        case ListFormatType():
            # compile recursively
            return CppListType(_compile_type(ty.elt))
        case ContextType():
            return CppContextType()
        case _:
            raise RuntimeError(f'Unhandled type: {ty}')

def _compile_rm(rm:RM, func: FuncDef | None) -> str:
    match rm:
        case RM.RNE:
            return 'mpfx::RoundingMode::RNE'
        case RM.RNA:
            return 'mpfx::RoundingMode::RNA'
        case RM.RTP:
            return 'mpfx::RoundingMode::RTP'
        case RM.RTN:
            return 'mpfx::RoundingMode::RTN'
        case RM.RTZ:
            return 'mpfx::RoundingMode::RTZ'
        case RM.RAZ:
            return 'mpfx::RoundingMode::RAZ'
        case RM.RTO:
            return 'mpfx::RoundingMode::RTO'
        case RM.RTE:
            return 'mpfx::RoundingMode::RTE'
        case _:
            _emit_error(f'Unsupported rounding mode to compile: {rm}', func)

def _compile_context(ctx: Context, func: FuncDef | None = None) -> str:
    match ctx:
        case IEEEContext():
            # ES <= 9
            if ctx.es > 9:
                _emit_error(f'IEEE 754: exponent too large to compile: {ctx.es}', func)
            # P <= 51
            if ctx.pmax > 51:
                _emit_error(f'IEEE 754: precision too large to compile: {ctx.pmax}', func)
            # overflow
            if ctx.overflow != OV.OVERFLOW:
                _emit_error(f'IEEE 754: overflow mode {ctx.overflow} cannot be compiled', func)
            # no random
            if ctx.num_randbits != 0:
                _emit_error('IEEE 754: stochastic rounding is unsupported', func)

            # compile rounding
            rm = _compile_rm(ctx.rm, func)
            return f'mpfx::IEEE754Context({ctx.es}, {ctx.nbits}, {rm})'
        case EFloatContext():
            # TODO: saturation, special values
            p = ctx.pmax
            n = ctx.nmin
            b = float(ctx.maxval())
            # compile rounding mode
            rm = _compile_rm(ctx.rm, func)
            return f'mpfx::Context({p}, {n}, {b}, {rm})'
        case ExpContext():
            # we include 0 within the representation
            p = 1
            n = ctx.minval().n
            b = float(ctx.maxval())
            # compile rounding mode
            rm = _compile_rm(ctx.rm, func)
            return f'mpfx::Context({p}, {n}, {b}, {rm})'
        case _:
            _emit_error(f'Unhandled context: {ctx}', func)


class _MPFXBackendInstance(Visitor):
    """
    Per-function compilation instance.

    Based on default C++ compiler.
    """

    func: FuncDef
    name: str
    options: CppOptions
    fmt_info: FormatAnalysis
    eval_info: PartialEvalInfo

    instr_gen: InstrGenerator
    decl_phis: set[PhiDef]
    decl_assigns: set[AssignDef]
    gensym: Gensym

    def __init__(
        self,
        func: FuncDef,
        name: str,
        options: CppOptions,
        fmt_info: FormatAnalysis,
        eval_info: PartialEvalInfo,
        allow_exact: bool = True,
    ):
        self.func = func
        self.name = name
        self.options = options
        self.fmt_info = fmt_info
        self.eval_info = eval_info

        self.instr_gen = InstrGenerator(func, allow_exact=allow_exact)
        self.decl_phis = set()
        self.decl_assigns = set()
        self.gensym = Gensym(self.def_use.names())

    @property
    def def_use(self) -> DefineUseAnalysis:
        return self.fmt_info.def_use

    def compile(self):
        # TODO: generate context name more intelligently
        ctx_name = self._fresh_var()
        ctx = CompileCtx.default(ctx_name)
        self._visit_function(self.func, ctx)
        return '\n'.join(ctx.lines)

    def _check_type(self, ty: FormatType):
        match ty:
            case BoolType() | AbstractFormat():
                pass
            case VarType():
                raise MPFXCompileError(self.func, f'Type is not monomorphic: {ty}')
            case RealType():
                raise MPFXCompileError(self.func, f'Cannot compile an unbounded real number: {ty.ctx}')
            case TupleFormatType():
                for elem_ty in ty.elts:
                    self._check_type(elem_ty)
            case ListFormatType():
                self._check_type(ty.elt)
            case ContextType():
                pass
            case _:
                raise RuntimeError(f'Unhandled type: {ty}')

    def _compile_context(self, ctx: Context) -> str:
        return _compile_context(ctx, self.func)

    def _compile_literal(self, s: str, ty: FormatType) -> str:
        """Compile a numerical literal given as a string."""
        # TODO: integer?
        if '.' in s:
            return s
        else:
            return f'{s}.0'

    def _compile_real(self, e: RealVal, ty: FormatType) -> str:
        """Compile a (rounded) numerical literal."""
        match e:
            case Decnum():
                return self._compile_literal(str(e.val), ty)
            case Hexnum():
                raise MPFXCompileError(self.func, 'hexadecimal literals are unsupported')
            case Integer():
                return self._compile_literal(str(e.val), ty)
            case Rational():
                raise MPFXCompileError(self.func, 'rational values are unsupported')
            case Digits():
                raise MPFXCompileError(self.func, '`digits(m, e, b)` is unsupported')
            case _:
                raise RuntimeError(f'unreachable: {e}')

    def _fresh_var(self):
        return str(self.gensym.fresh('__fpy_tmp'))

    def _var_is_decl(self, name: NamedId, site: DefSite):
        d = self.def_use.find_def_from_site(name, site)
        return d.prev is None and d not in self.decl_assigns

    def _def_type(self, d: Definition):
        ty = self.fmt_info.by_def[d]
        self._check_type(ty)
        return ty

    def _var_type(self, name: NamedId, site: DefSite):
        d = self.def_use.find_def_from_site(name, site)
        return self._def_type(d)

    def _expr_type(self, e: Expr):
        ty = self.fmt_info.by_expr[e]
        self._check_type(ty)
        return ty

    def _expr_ctx(self, e: Expr) -> Context:
        ty = self.fmt_info.ctx_info.by_expr[e]
        assert isinstance(ty, RealType) and isinstance(ty.ctx, Context)
        return ty.ctx

    def _compile_type(self, ty: FormatType) -> CppType:
        return _compile_type(ty, self.func)

    def _compile_number(self, s: str, ty: FormatType):
        # TODO: check context for rounding
        raise NotImplementedError

    def _compile_size(self, e: str):
        return f'static_cast<size_t>({e})'

    def _visit_var(self, e: Var, ctx: CompileCtx):
        return str(e.name)

    def _visit_bool(self, e: BoolVal, ctx: CompileCtx):
        return 'true' if e.val else 'false'

    def _visit_foreign(self, e: ForeignVal, ctx: CompileCtx):
        raise NotImplementedError

    def _visit_decnum(self, e, ctx):
        raise NotImplementedError

    def _visit_hexnum(self, e, ctx):
        raise NotImplementedError

    def _visit_integer(self, e, ctx):
        return str(e.val)

    def _visit_rational(self, e, ctx):
        raise NotImplementedError

    def _visit_digits(self, e, ctx):
        raise NotImplementedError

    def _visit_nullaryop(self, e, ctx: CompileCtx):
        match e:
            case _:
                raise MPFXCompileError(self.func, f'Unsupported nullary operation to compile: {e}')

    def _visit_unaryop(self, e, ctx: CompileCtx):
        match e:
            case Cast():
                arg_str = self._visit_expr(e.arg, ctx)
                arg_ty = self._expr_type(e.arg)
                e_ctx = self._expr_ctx(e)
                return self.instr_gen.cast(arg_str, ctx.ctx_name, arg_ty, e_ctx)
            case Neg():
                arg_str = self._visit_expr(e.arg, ctx)
                arg_ty = self._expr_type(e.arg)
                e_ctx = self._expr_ctx(e)
                return self.instr_gen.neg(arg_str, ctx.ctx_name, arg_ty, e_ctx)
            case Abs():
                arg_str = self._visit_expr(e.arg, ctx)
                arg_ty = self._expr_type(e.arg)
                e_ctx = self._expr_ctx(e)
                return self.instr_gen.abs(arg_str, ctx.ctx_name, arg_ty, e_ctx)
            case Sqrt():
                arg_str = self._visit_expr(e.arg, ctx)
                arg_ty = self._expr_type(e.arg)
                e_ctx = self._expr_ctx(e)
                return self.instr_gen.sqrt(arg_str, ctx.ctx_name, arg_ty, e_ctx)
            case Sum():
                tid = self._fresh_var()
                arg_str = self._visit_expr(e.arg, ctx)
                arg_ty = self._expr_type(e.arg)
                e_ty = self._expr_type(e)
                e_ctx = self._expr_ctx(e)
                self.instr_gen.sum(tid, arg_str, ctx, arg_ty.elt, e_ty, e_ctx)
                return tid
            case Len():
                return self._visit_len(e, ctx)
            case Not():
                arg_str = self._visit_expr(e.arg, ctx)
                return f'!({arg_str})'
            case IsNan():
                arg_str = self._visit_expr(e.arg, ctx)
                return f'std::isnan({arg_str})'
            case IsInf():
                arg_str = self._visit_expr(e.arg, ctx)
                return f'std::isinf({arg_str})'
            case Logb():
                arg_str = self._visit_expr(e.arg, ctx)
                return f'static_cast<int64_t>(std::logb({arg_str}))'
            case _:
                raise MPFXCompileError(self.func, f'Unsupported unary operation to compile: {e}')

    def _visit_binaryop(self, e, ctx: CompileCtx):
        lhs_str = self._visit_expr(e.first, ctx)
        rhs_str = self._visit_expr(e.second, ctx)
        match e:
            case Add():
                lhs_ty = self._expr_type(e.first)
                rhs_ty = self._expr_type(e.second)
                e_ctx = self._expr_ctx(e)
                return self.instr_gen.add(lhs_str, rhs_str, ctx.ctx_name, lhs_ty, rhs_ty, e_ctx)
            case Sub():
                lhs_ty = self._expr_type(e.first)
                rhs_ty = self._expr_type(e.second)
                e_ctx = self._expr_ctx(e)
                return self.instr_gen.sub(lhs_str, rhs_str, ctx.ctx_name, lhs_ty, rhs_ty, e_ctx)
            case Mul():
                lhs_ty = self._expr_type(e.first)
                rhs_ty = self._expr_type(e.second)
                e_ctx = self._expr_ctx(e)
                return self.instr_gen.mul(lhs_str, rhs_str, ctx.ctx_name, lhs_ty, rhs_ty, e_ctx)
            case Div():
                lhs_ty = self._expr_type(e.first)
                rhs_ty = self._expr_type(e.second)
                e_ctx = self._expr_ctx(e)
                return self.instr_gen.div(lhs_str, rhs_str, ctx.ctx_name, lhs_ty, rhs_ty, e_ctx)
            case _:
                raise MPFXCompileError(self.func, f'Unsupported binary operation to compile: {e}')

    def _visit_ternaryop(self, e, ctx: CompileCtx):
        fst_str = self._visit_expr(e.first, ctx)
        snd_str = self._visit_expr(e.second, ctx)
        trd_str = self._visit_expr(e.third, ctx)
        match e:
            case Fma():
                fst_ty = self._expr_type(e.first)
                snd_ty = self._expr_type(e.second)
                trd_ty = self._expr_type(e.third)
                e_ctx = self._expr_ctx(e)
                return self.instr_gen.fma(fst_str, snd_str, trd_str, ctx.ctx_name, fst_ty, snd_ty, trd_ty, e_ctx)
            case _:
                raise MPFXCompileError(self.func, f'Unsupported ternary operation to compile: {e}')

    def _visit_zip(self, e: Zip, ctx: CompileCtx):
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
        arg_tys: list[FormatType] = []
        for arg in e.args:
            x = self._fresh_var()
            arg_cpp = self._visit_expr(arg, ctx)
            ctx.add_line(f'auto {x} = {arg_cpp};')
            xs.append(x)

            arg_ty = self._expr_type(arg)
            if not isinstance(arg_ty, ListFormatType):
                raise MPFXCompileError(self.func, f'expected list for `{arg.format()}`')
            arg_tys.append(arg_ty.elt)

        t = self._fresh_var()
        i = self._fresh_var()
        tmpl = ', '.join(self._compile_type(ty).to_cpp() for ty in arg_tys)
        ctx.add_line(f'std::vector<std::tuple<{tmpl}>> {t}({xs[0]}.size());')
        ctx.add_line(f'for (size_t {i} = 0; {i} < {xs[0]}.size(); ++{i}) {{')
        ctx.indent().add_line(f'{t}[{i}] = std::make_tuple({", ".join(f"{x}[{i}]" for x in xs)});')
        ctx.add_line('}')

        return t

    def _visit_minmax(self, e: Min | Max, ctx: CompileCtx):
        op_str = 'std::min' if isinstance(e, Min) else 'std::max'
        arg_str = self._visit_expr(e.args[0], ctx)
        for next_arg in e.args[1:]:
            next_arg_str = self._visit_expr(next_arg, ctx)
            arg_str = f'{op_str}({arg_str}, {next_arg_str})'
        return arg_str

    def _visit_or_and(self, e: Or | And, ctx: CompileCtx):
        op_str = '||' if isinstance(e, Or) else '&&'
        arg_str = self._visit_expr(e.args[0], ctx)
        for next_arg in e.args[1:]:
            next_arg_str = self._visit_expr(next_arg, ctx)
            arg_str = f'({arg_str} {op_str} {next_arg_str})'
        return arg_str

    def _visit_naryop(self, e, ctx: CompileCtx):
        match e:
            case Zip():
                return self._visit_zip(e, ctx)
            case Min() | Max():
                return self._visit_minmax(e, ctx)
            case Or() | And():
                return self._visit_or_and(e, ctx)
            case Empty():
                return self._visit_empty(e, ctx)
            case _:
                raise MPFXCompileError(self.func, f'Unsupported nary operation to compile: {e}')

    def _visit_round(self, e: Round | RoundExact, ctx: CompileCtx):
        if isinstance(e.arg, RealVal):
            # special case: rounded literal
            ty = self._expr_type(e)
            return self._compile_real(e.arg, ty)
        else:
            # general case: unary operator
            # TODO: what about integer -> double
            arg = self._visit_expr(e.arg, ctx)
            ty = self._expr_type(e)
            return f'mpfx::round({arg}, {ctx.ctx_name})'

    def _visit_round_at(self, e: RoundAt, ctx: CompileCtx):
        raise MPFXCompileError(self.func, '`round_at` is unsupported')

    def _visit_len(self, e: Len, ctx: CompileCtx):
        # len(x)
        arg_cpp = self._visit_expr(e.arg, ctx)
        return self._compile_size(f'{arg_cpp}.size()')

    def _visit_empty(self, e: Empty, ctx: CompileCtx):
        # empty(x1, ..., xn) => std::vector<...std::vector<T>(xn)...>(x1)
        # Creates nested vectors with proper space reservation at each level
        e_ty = self._expr_type(e)

        # Extract base element type by unwrapping all ListFormatType layers
        ty = e_ty
        while isinstance(ty, ListFormatType):
            ty = ty.elt
        base_cpp_ty = self._compile_type(ty).to_cpp()

        # Compile dimension expressions to size strings
        dims = [self._compile_size(self._visit_expr(arg, ctx)) for arg in e.args]

        # Build nested constructor from innermost to outermost
        # Start with the innermost 1D vector
        res_str = f'std::vector<{base_cpp_ty}>({dims[-1]})'
        curr_type = f'std::vector<{base_cpp_ty}>'

        # Iterate backwards through dimensions (outer to inner)
        for i in range(len(dims) - 2, -1, -1):
            next_type = f'std::vector<{curr_type}>'
            res_str = f'{next_type}({dims[i]}, {res_str})'
            curr_type = next_type
        return res_str

    def _visit_call(self, e: Call, ctx: CompileCtx):
        # possibly can determine result of the call statically
        if e in self.eval_info.by_expr:
            val = self.eval_info.by_expr[e]
            if isinstance(val, Context):
                # special case: context construction
                return self._compile_context(val)

        if e.fn is ldexp:
            assert len(e.args) == 2
            lhs_str = self._visit_expr(e.args[0], ctx)
            rhs_str = self._visit_expr(e.args[1], ctx)
            return f'std::ldexp({lhs_str}, {rhs_str})'

        raise MPFXCompileError(self.func, f'cannot compile `{e.format()}`')


    def _visit_compare2(self, op: CompareOp, lhs: str, rhs: str) -> str:
        return f'{lhs} {op.symbol()} {rhs}'

    def _visit_compare(self, e, ctx):
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

            return ' && '.join(
                self._visit_compare2(e.ops[i], args[i], args[i + 1])
                for i in range(len(args) - 1)
            )

    def _visit_tuple_expr(self, e, ctx):
        args = [self._visit_expr(arg, ctx) for arg in e.elts]
        return f'std::make_tuple({", ".join(args)})'

    def _visit_list_expr(self, e: ListExpr, ctx: CompileCtx):
        cpp_ty = self._compile_type(self._expr_type(e))
        args = ', '.join(self._visit_expr(arg, ctx) for arg in e.elts)
        return f'{cpp_ty.to_cpp()}({{{args}}})'

    def _visit_list_comp_zip(self, e: ListComp, ctx: CompileCtx):
        # optimized list comprehension for zip:
        # [<e> for <x1>, ..., <xN> in zip(<iterable1>, ..., <iterableN>)] =>
        # auto <t1> = <iterable1>;
        # ...
        # auto <tN> = <iterableN>;
        # std::vector<T> v(t1.size());
        # for (size_t i = 0; i < t1.size(); ++i) {
        #   auto <x1> = t1[i];
        #   ...
        #   auto <xN> = tN[i];
        #   v[i] = <e>;
        # }
        assert len(e.targets) == 1 and isinstance(e.iterables[0], Zip)

        # result
        v = self._fresh_var()
        cpp_ty = self._compile_type(self._expr_type(e))

        # emit loop 
        i = self._visit_for_zip(e, e.targets[0], e.iterables[0], ctx, result=(v, cpp_ty))

        # assignment
        elt = self._visit_expr(e.elt, ctx)
        ctx.indent().add_line(f'{v}[{i}] = {elt};')

        # closing brace
        ctx.add_line('}')
        return v

    def _visit_list_comp_product(self, e: ListComp, ctx: CompileCtx):
        # general list comprehension with at least 2 iterables:
        # [<e> for <x1> in <iterable1> ... for <xN> in <iterableN>] =>
        # std::vector<T> v;
        # for (auto <x1> : <iterable1>) {
        #   auto <t2> = <iterable2>;  // evaluated in context of x1
        #   for (auto <x2> : <t2>) {
        #     ...
        #       auto <tN> = <iterableN>;  // evaluated in context of x1, x2, ...
        #       for (auto <xN> : <tN>) {
        #         v.push_back(<e>);
        #       }
        #   }
        # }

        # create output vector
        v = self._fresh_var()
        cpp_ty = self._compile_type(self._expr_type(e))
        ctx.add_line(f'{cpp_ty.to_cpp()} {v};')

        # build nested loops - each iterable is evaluated in its proper context
        body_ctx = ctx
        for target, iterable in zip(e.targets, e.iterables):
            # evaluate iterable in current loop context
            t = self._fresh_var()
            iterable_cpp = self._visit_expr(iterable, body_ctx)
            body_ctx.add_line(f'auto {t} = {iterable_cpp};')

            # open the for loop
            match target:
                case Id():
                    body_ctx.add_line(f'for (auto {target} : {t}) {{')
                case TupleBinding():
                    t2 = self._fresh_var()
                    body_ctx.add_line(f'for (auto {t2} : {t}) {{')
                    self._visit_binding(e, target, t2, body_ctx.indent())
                case _:
                    raise RuntimeError(f'unreachable {target}')
            body_ctx = body_ctx.indent()

        # compute the element and add to result
        elt = self._visit_expr(e.elt, body_ctx)
        body_ctx.add_line(f'{v}.push_back({elt});')

        # closing braces
        while body_ctx.indent_level > ctx.indent_level:
            body_ctx = body_ctx.dedent()
            body_ctx.add_line('}')

        return v

    def _visit_list_comp(self, e: ListComp, ctx: CompileCtx):
        if len(e.targets) >= 2:
            # general case: multiple for-loops with dynamic allocation
            return self._visit_list_comp_product(e, ctx)

        # single for-loop
        target = e.targets[0]
        iterable = e.iterables[0]

        if isinstance(iterable, Zip):
            # optimized case: zip with single target
            return self._visit_list_comp_zip(e, ctx)

        # [<e> for <x> in <iterable>] =>
        # auto <t> = <iterable>;
        # std::vector<T> v(t.size());
        # for (size_t i = 0; i < t.size(); ++i) {
        #   v[i] = <e>;
        # }

        # evaluate iterable
        t = self._fresh_var()
        iterable_cpp = self._visit_expr(iterable, ctx)
        ctx.add_line(f'auto {t} = {iterable_cpp};')

        # create output vector
        v = self._fresh_var()
        e_ty = self._expr_type(e)
        cpp_ty = self._compile_type(e_ty)
        ctx.add_line(f'{cpp_ty.to_cpp()} {v}({t}.size());')

        # build the for loop
        i = self._fresh_var()
        ctx.add_line(f'for (size_t {i} = 0; {i} < {t}.size(); ++{i}) {{')
        self._visit_binding(e, target, f'{t}[{i}]', ctx.indent())

        # compute the element and add to result
        elt = self._visit_expr(e.elt, ctx.indent())
        ctx.indent().add_line(f'{v}[{i}] = {elt};')
        ctx.add_line('}')  # closing brace
        return v

    def _visit_list_ref(self, e: ListRef, ctx: CompileCtx):
        value = self._visit_expr(e.value, ctx)
        index = self._visit_expr(e.index, ctx)
        size = self._compile_size(index)
        return f'{value}[{size}]'

    def _visit_list_slice(self, e: ListSlice, ctx: CompileCtx):
        # x[<start>:<stop>] =>
        # auto v = <x>
        # auto start = static_cast<size_t>(<start>) OR 0;
        # auto stop = static_cast<size_t>(<stop>) OR x.size();
        # <result> = std::vector<T>(v.begin() + start, v.end() + end);

        # temporarily bind array
        t = self._fresh_var()
        arr = self._visit_expr(e.value, ctx)
        e_ty = self._expr_type(e)
        ctx.add_line(f'auto {t} = {arr};')

        # compile start
        if e.start is None:
            start = '0'
        else:
            start = self._visit_expr(e.start, ctx)
            start = self._compile_size(start)

        # compile stop
        if e.stop is None:
            stop = f'{t}.size()'
        else:
            stop = self._visit_expr(e.stop, ctx)
            stop = self._compile_size(stop)

        # result
        ty_str = self._compile_type(e_ty).to_cpp()
        return f'{ty_str}({t}.begin() + {start}, {t}.begin() + {stop})'

    def _visit_list_set(self, e, ctx):
        raise NotImplementedError

    def _visit_if_expr(self, e, ctx):
        raise NotImplementedError

    def _visit_attribute(self, e, ctx):
        if e not in self.eval_info.by_expr:
            raise MPFXCompileError(self.func, f'cannot compile `{e.format()}`')
        val = self.eval_info.by_expr[e]
        if isinstance(val, Context):
            return self._compile_context(val)
        else:
            raise MPFXCompileError(self.func, f'cannot compile `{e.format()}`')

    def _visit_decl(self, name: Id, e: str, site: DefSite, ctx: CompileCtx):
        match name:
            case NamedId():
                if self._var_is_decl(name, site):
                    ty = self._var_type(name, site)
                    cpp_ty = self._compile_type(ty)
                    if not isinstance(cpp_ty, TupleType | ListType):
                        ctx.add_line(f'{cpp_ty.to_cpp()} {name} = {e};')
                    else:
                        ctx.add_line(f'auto {name} = {e};')
                else:
                    ctx.add_line(f'{name} = {e};')
            case UnderscoreId():
                ctx.add_line(f'auto _ = {e};')
            case _:
                raise RuntimeError(f'unreachable: {name}')

    def _visit_binding(self, site: DefSite, binding: Id | TupleBinding, e: str, ctx: CompileCtx):
        match binding:
            case Id():
                self._visit_decl(binding, e, site, ctx)
            case TupleBinding():
                # emit temporary variable for the tuple
                t = self._fresh_var()
                ctx.add_line(f'auto {t} = {e};')
                for i, elt in enumerate(binding.elts):
                    self._visit_binding(site, elt, f'std::get<{i}>({t})', ctx)
            case _:
                raise RuntimeError(f'unreachable: {binding}')

    def _visit_assign(self, stmt: Assign, ctx: CompileCtx):
        e = self._visit_expr(stmt.expr, ctx)
        self._visit_binding(stmt, stmt.target, e, ctx)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: CompileCtx):
        # compile indices
        indices: list[str] = []
        for index in stmt.indices:
            i = self._visit_expr(index, ctx)
            indices.append(self._compile_size(i))

        # compile expression
        e = self._visit_expr(stmt.expr, ctx)
        index_str = ''.join(f'[{i}]' for i in indices)
        ctx.add_line(f'{stmt.var}{index_str} = {e};')

    def _visit_if1(self, stmt: If1Stmt, ctx: CompileCtx):
        cond = self._visit_expr(stmt.cond, ctx)
        ctx.add_line(f'if ({cond}) {{')
        self._visit_block(stmt.body, ctx.indent())
        ctx.add_line('}')  # close if

    def _visit_if(self, stmt: IfStmt, ctx: CompileCtx):
        # variables need to be declared if they are assigned
        # within the branches but not before defined in the branches
        for phi in self.def_use.phis[stmt]:
            if phi.is_intro and phi not in self.decl_phis:
                # need a declaration for the phi assignment
                ty_str = self._compile_type(self._def_type(phi)).to_cpp()
                ctx.add_line(f'{ty_str} {phi.name}; // phi')
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

    def _visit_while(self, stmt: WhileStmt, ctx: CompileCtx):
        cond = self._visit_expr(stmt.cond, ctx)
        ctx.add_line(f'while ({cond}) {{')
        self._visit_block(stmt.body, ctx.indent())
        ctx.add_line('}')  # close if

    def _visit_for_range(self, target: Id, iterable: Range1 | Range2 | Range3, ctx: CompileCtx):
        if isinstance(iterable, Range1):
            # special case: for i in range(n)
            # for (size_t i = 0; i < n; ++i) {
            n = self._visit_expr(iterable.arg, ctx)
            ctx.add_line(f'for (size_t {target} = 0; {target} < {n}; ++{target}) {{')
        elif isinstance(iterable, Range2):
            # special case: for i in range(m, n)
            # for (size_t i = m; i < n; ++i) {
            m = self._visit_expr(iterable.first, ctx)
            n = self._visit_expr(iterable.second, ctx)
            ctx.add_line(f'for (size_t {target} = {m}; {target} < {n}; ++{target}) {{')
        elif isinstance(iterable, Range3):
            # special case: for i in range(m, n, s)
            # for (size_t i = m; i < n; i += s) {
            m = self._visit_expr(iterable.first, ctx)
            n = self._visit_expr(iterable.second, ctx)
            s = self._visit_expr(iterable.third, ctx)
            ctx.add_line(f'for (size_t {target} = {m}; {target} < {n}; {target} += {s}) {{')
        else:
            raise RuntimeError(f'unreachable: {iterable}')


    def _visit_for_zip(self, site: DefSite, target: Id | TupleBinding, iterable: Zip, ctx: CompileCtx, result: tuple[str, CppType] | None = None):
        # special case: for (a, b, ...) in zip(x1, x2, ...)
        # auto t1 = x1;
        # ...
        # auto tn = xn;
        # for (size_t i = 0; i < t1.size(); ++i) {
        #     auto a = t1[i];
        #     auto b = t2[i];
        #     ...

        # bind zip arguments to temporaries
        ts: list[str] = []
        for arg in iterable.args:
            t = self._fresh_var()
            arg_cpp = self._visit_expr(arg, ctx)
            ctx.add_line(f'auto {t} = {arg_cpp};')
            ts.append(t)

        # optionally allocate result vector if this is a list comprehension
        if result is not None:
            res_name, res_ty = result
            ctx.add_line(f'{res_ty.to_cpp()} {res_name}({ts[0]}.size());')

        # generate loop
        i = self._fresh_var()
        ctx.add_line(f'for (size_t {i} = 0; {i} < {ts[0]}.size(); ++{i}) {{')
        ctx = ctx.indent()
        match target:
            case Id():
                # single variable: bind to `make_tuple(t1[i], ..., tn[i])`
                match target:
                    case NamedId():
                        tid = target
                    case UnderscoreId():
                        tid = self._fresh_var()
                    case _:
                        raise RuntimeError(f'unreachable: {target}')

                e = ', '.join(f'{t}[{i}]' for t in ts)
                self._visit_decl(tid, f'make_tuple({e})', site, ctx)
            case TupleBinding():
                for elt, t in zip(target.elts, ts):
                    self._visit_binding(site, elt, f'{t}[{i}]', ctx)
            case _:
                raise RuntimeError(f'unreachable: {target}')

        # return the index variable
        return i


    def _visit_for(self, stmt: ForStmt, ctx: CompileCtx):
        if isinstance(stmt.target, Id) and isinstance(stmt.iterable, Range1 | Range2 | Range3):
            # optimized: iterating over [0, n)
            self._visit_for_range(stmt.target, stmt.iterable, ctx)
        elif isinstance(stmt.iterable, Zip):
            # optimized: iterating over zip(...)
            self._visit_for_zip(stmt, stmt.target, stmt.iterable, ctx)
        else:
            # general case: for x in iterable
            iterable = self._visit_expr(stmt.iterable, ctx)
            match stmt.target:
                case Id():
                    ctx.add_line(f'for (auto {stmt.target} : {iterable}) {{')
                case TupleBinding():
                    t = self._fresh_var()
                    ctx.add_line(f'for (auto {t} : {iterable}) {{')
                    self._visit_binding(stmt, stmt.target, t, ctx.indent())
                case _:
                    raise RuntimeError(f'unreachable {ctx}')

        self._visit_block(stmt.body, ctx.indent())
        ctx.add_line('}')  # close if

    def _visit_context(self, stmt: ContextStmt, ctx: CompileCtx):
        if isinstance(stmt.ctx, Var):
            # context variable: must be bound already
            ctx_name: str | None = str(stmt.ctx.name)
        elif isinstance(stmt.ctx, DeclContext):
            # don't bind it and hope it compiles
            ctx_name = None
        else:
            # context must be statically known
            ctx_val = self.eval_info.by_expr[stmt.ctx]

            # bind it if we can compile it
            if isinstance(ctx_val, SupportedContext) and ctx is not INTEGER:
                ctx_name = self._fresh_var()
                ctx_str = self._compile_context(ctx_val)
                ctx.add_line(f'auto {ctx_name} = {ctx_str};')
            else:
                ctx_name = None

        # compile body
        self._visit_block(stmt.body, ctx.with_ctx(ctx_name))

    def _visit_assert(self, stmt: AssertStmt, ctx: CompileCtx):
        e = self._visit_expr(stmt.test, ctx)
        if stmt.msg is None:
            ctx.add_line(f'assert({e});')
        else:
            ctx.add_line(f'assert({e}); // {stmt.msg.format()}')

    def _visit_effect(self, stmt: EffectStmt, ctx: CompileCtx):
        raise MPFXCompileError(self.func, 'FPy effects are not supported')

    def _visit_return(self, stmt: ReturnStmt, ctx: CompileCtx):
        e = self._visit_expr(stmt.expr, ctx)
        ctx.add_line(f'return {e};')

    def _visit_pass(self, stmt: PassStmt, ctx: CompileCtx):
        pass

    def _visit_block(self, block: StmtBlock, ctx: CompileCtx):
        for stmt in block.stmts:
            self._visit_statement(stmt, ctx)

    def _visit_function(self, func: FuncDef, ctx: CompileCtx):
        # compile arguments
        arg_strs: list[str] = []
        for arg, arg_ty in zip(func.args, self.fmt_info.arg_types):
            self._check_type(arg_ty)
            ty_str = self._compile_type(arg_ty).to_cpp(is_arg=True)
            arg_strs.append(f'{ty_str} {arg.name}')

        ret_ty = self.fmt_info.return_type
        self._check_type(ret_ty)
        ty_str = self._compile_type(ret_ty).to_cpp()
        ctx.add_line(f'{ty_str} {self.name}({", ".join(arg_strs)}) {{')

        # compile body
        self._visit_block(func.body, ctx.indent())
        ctx.add_line('}')


class MPFXCompiler(Backend):
    """
    MPFX backend: compiler to MPFX number library.
    """

    # unsafe options
    unsafe_cast_int: bool
    unsafe_finitize_int: bool

    # compilation options
    inline: bool
    elim_round: bool
    allow_exact: bool
    optimize: bool

    def __init__(
        self, *,
        unsafe_cast_int: bool = False,
        unsafe_finitize_int: bool = False,
        inline: bool = True,
        elim_round: bool = True,
        allow_exact: bool = True,
        optimize: bool = True
    ):
        self.unsafe_cast_int = unsafe_cast_int
        self.unsafe_finitize_int = unsafe_finitize_int
        self.inline = inline
        self.elim_round = elim_round
        self.allow_exact = allow_exact
        self.optimize = optimize

    def compile_context(self, ctx: Context):
        return _compile_context(ctx)

    def compile_type(self, ty: Type):
        match ty:
            case BoolType() | ContextType():
                return _compile_type(ty)
            case VarType():
                raise TypeError(f'Type is not monomorphic: {ty}')
            case RealType():
                if isinstance(ty.ctx, SupportedContext):
                    return _compile_type(AbstractFormat.from_context(ty.ctx))
                else:
                    raise TypeError(f'Cannot compile an unbounded real number: {ty.ctx}')
            case TupleType():
                # compile recursively
                return CppTupleType([self.compile_type(elem_ty) for elem_ty in ty.elts])
            case ListType():
                # compile recursively
                return CppListType(self.compile_type(ty.elt))
            case _:
                raise RuntimeError(f'Unhandled type: {ty}')

    def compile(
        self,
        func: Function,
        *,
        name: str | None = None,
        ctx: Context | None = None,
        arg_types: Collection[Type | None] | None = None,
    ) -> str:
        """
        Compiles the given FPy function to a C++ program
        represented as a string.
        """
        if not isinstance(func, Function):
            raise TypeError(f'Expected `Function`, got {type(func)} for {func}')
        if ctx is not None and not isinstance(ctx, Context):
            raise TypeError(f'Expected `Context` or `None`, got {type(ctx)} for {ctx}')
        if arg_types is not None and not isinstance(arg_types, Collection):
            raise TypeError(f'Expected `Collection` or `None`, got {type(arg_types)} for {arg_types}')

        # compiled name
        if name is None:
            name = func.name

        # inlining
        if self.inline:
            ast = FuncInline.apply(func.ast)
            func = func.with_ast(ast)

        if self.optimize:
            # lift contexts so they are constructed at the top level
            ast = func.ast
            ast = LiftContext.apply(ast)
            func = func.with_ast(ast)

            # apply copy propagation and dead code elimination to clean up after inlining
            func = simplify(func, enable_const_fold=False, enable_const_prop=False)

        # monomorphizing
        ast = func.ast
        if arg_types is None:
            arg_types = [None for _ in func.args]
        if ast.ctx is not None:
            ctx = None
        ast = Monomorphize.apply_by_arg(ast, ctx, arg_types)

        # partial evaluation
        def_use = DefineUse.analyze(ast)
        eval_info = PartialEval.apply(ast, def_use=def_use)

        # run type checking with static context inference
        try:
            ctx_info = ContextInfer.infer(ast, def_use=def_use, eval_info=eval_info, unsafe_cast_int=self.unsafe_cast_int)
            format_info = FormatInfer.infer(ast, ctx_info=ctx_info)
        except (ContextInferError, TypeInferError, FormatInferError) as e:
            raise ValueError(f'{func.name}: context inference failed') from e

        if self.elim_round:
            # perform rounding elimination
            ast = ElimRound.apply(ast, eval_info=eval_info)
            # print(ast.format())

            # apply copy propagation and dead code elimination to clean up after inlining
            func = simplify(func, enable_const_fold=False, enable_const_prop=False)

            # reanalyze
            eval_info = PartialEval.apply(ast)
            ctx_info = ContextInfer.infer(ast, def_use=eval_info.def_use, eval_info=eval_info, unsafe_cast_int=self.unsafe_cast_int)
            format_info = FormatInfer.infer(ast, ctx_info=ctx_info)

        # compile
        options = CppOptions(self.unsafe_finitize_int, self.unsafe_cast_int)
        inst = _MPFXBackendInstance(ast, name, options, format_info, eval_info, allow_exact=self.allow_exact)
        body_str = inst.compile()

        return body_str
