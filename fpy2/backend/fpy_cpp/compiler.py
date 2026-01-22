"""
C++/FPy backend: compiler to C++ number library.
"""

from typing import Collection

from ...ast import *
from ...analysis import (
    ContextInfer, ContextInferError,
    DefineUseAnalysis, Definition, DefSite, AssignDef, PhiDef,
    TypeInferError
)
from ...function import Function
from ...number import IEEEContext, OV, RM, FP64, INTEGER
from ...transform import Monomorphize
from ...strategies import simplify
from ...types import BoolType, RealType, VarType, Type
from ...utils import Gensym
from ..backend import Backend

from .format import AbstractFormat
from .format_infer import (
    FormatAnalysis, FormatInfer, FormatInferError,
    FormatType, ListFormatType, TupleFormatType
)
from .types import *
from .utils import CppFpyCompileError, CompileCtx, CppOptions


class _CppBackendInstance(Visitor):
    """
    Per-function compilation instance.

    Based on default C++ compiler.
    """

    func: FuncDef
    name: NamedId
    options: CppOptions
    fmt_info: FormatAnalysis

    decl_phis: set[PhiDef]
    decl_assigns: set[AssignDef]
    gensym: Gensym

    def __init__(
        self,
        func: FuncDef,
        name: NamedId,
        options: CppOptions,
        fmt_info: FormatAnalysis,
    ):
        self.func = func
        self.name = name
        self.options = options
        self.fmt_info = fmt_info

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
                raise CppFpyCompileError(self.func, f'Type is not monomorphic: {ty}')
            case RealType():
                raise CppFpyCompileError(self.func, f'Cannot compile an unbounded real number: {ty.ctx}')
            case TupleFormatType():
                for elem_ty in ty.elts:
                    self._check_type(elem_ty)
            case ListFormatType():
                self._check_type(ty.elt)
            case _:
                raise RuntimeError(f'Unhandled type: {ty}')

    def _compile_rm(self, rm: RM) -> str:
        match rm:
            case RM.RNE:
                return 'fpy::RoundingMode::RNE'
            case RM.RNA:
                return 'fpy::RoundingMode::RNA'
            case RM.RTP:
                return 'fpy::RoundingMode::RTP'
            case RM.RTN:
                return 'fpy::RoundingMode::RTN'
            case RM.RTZ:
                return 'fpy::RoundingMode::RTZ'
            case RM.RAZ:
                return 'fpy::RoundingMode::RAZ'
            case RM.RTO:
                return 'fpy::RoundingMode::RTO'
            case RM.RTE:
                return 'fpy::RoundingMode::RTE'
            case _:
                raise CppFpyCompileError(self.func, f'Unsupported rounding mode to compile: {rm}')

    def _compile_context(self, ctx: Context) -> str:
        match ctx:
            case IEEEContext():
                # ES <= 9
                if ctx.es > 9:
                    raise CppFpyCompileError(self.func, f'IEEE 754: exponent too large to compile: {ctx.es}')
                # P <= 51
                if ctx.pmax > 51:
                    raise CppFpyCompileError(self.func, f'IEEE 754: precision too large to compile: {ctx.pmax}')
                # overflow
                if ctx.overflow != OV.OVERFLOW:
                    raise CppFpyCompileError(self.func, 'IEEE 754: overflow mode RAISE cannot be compiled')
                # no random
                if ctx.num_randbits != 0:
                    raise CppFpyCompileError(self.func, 'IEEE 754: stochastic rounding cannot be compiled')

                # compile rounding
                rm = self._compile_rm(ctx.rm)
                return f'fpy::IEEE754Context({ctx.es}, {ctx.nbits}, {rm})'
            case _:
                raise RuntimeError(f'Unhandled context: {ctx}')

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
                raise CppFpyCompileError(self.func, 'hexadecimal literals are unsupported')
            case Integer():
                return self._compile_literal(str(e.val), ty)
            case Rational():
                raise CppFpyCompileError(self.func, 'rational values are unsupported')
            case Digits():
                raise CppFpyCompileError(self.func, '`digits(m, e, b)` is unsupported')
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

    def _compile_type(self, ty: FormatType) -> CppType:
        match ty:
            case BoolType():
                return CppBoolType()
            case AbstractFormat():
                if ty.contained_in(AbstractFormat.from_context(FP64)):
                    # fits within a double 
                    return CppDoubleType()
                elif ty.contained_in(AbstractFormat.from_context(INTEGER)):
                    # fits with an int64_t
                    return CppInt64Type()
                else:
                    raise CppFpyCompileError(self.func, f'no container type for: {ty}')
            case TupleFormatType():
                # compile recursively
                return CppTupleType(self._compile_type(elem_ty) for elem_ty in ty.elts)
            case ListFormatType():
                # compile recursively
                return CppListType(self._compile_type(ty.elt))
            case _:
                raise RuntimeError(f'Unhandled type: {ty}')

    def _compile_number(self, s: str, ty: FormatType):
        # TODO: check context for rounding
        raise NotImplementedError

    def _compile_size(self, e: str, ty: FormatType):
        if not isinstance(ty, RealType | AbstractFormat):
            raise RuntimeError(f'size type must be RealType, got {ty}')
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
        raise NotImplementedError

    def _visit_rational(self, e, ctx):
        raise NotImplementedError

    def _visit_digits(self, e, ctx):
        raise NotImplementedError

    def _visit_nullaryop(self, e, ctx: CompileCtx):
        match e:
            case _:
                raise CppFpyCompileError(self.func, f'Unsupported nullary operation to compile: {e}')

    def _visit_unaryop(self, e, ctx: CompileCtx):
        arg_str = self._visit_expr(e.arg, ctx)
        match e:
            case Neg():
                return f'fpy::neg({arg_str}, {ctx.ctx_name})'
            case Abs():
                return f'fpy::abs({arg_str}, {ctx.ctx_name})'
            case Sqrt():
                return f'fpy::sqrt({arg_str}, {ctx.ctx_name})'
            case Len():
                return self._visit_len(e, ctx)
            case _:
                raise CppFpyCompileError(self.func, f'Unsupported unary operation to compile: {e}')

    def _visit_binaryop(self, e, ctx: CompileCtx):
        lhs_str = self._visit_expr(e.first, ctx)
        rhs_str = self._visit_expr(e.second, ctx)
        match e:
            case Add():
                return f'fpy::add({lhs_str}, {rhs_str}, {ctx.ctx_name})'
            case Sub():
                return f'fpy::sub({lhs_str}, {rhs_str}, {ctx.ctx_name})'
            case Mul():
                return f'fpy::mul({lhs_str}, {rhs_str}, {ctx.ctx_name})'
            case Div():
                return f'fpy::div({lhs_str}, {rhs_str}, {ctx.ctx_name})'
            case _:
                raise CppFpyCompileError(self.func, f'Unsupported binary operation to compile: {e}')

    def _visit_ternaryop(self, e, ctx: CompileCtx):
        fst_str = self._visit_expr(e.first, ctx)
        snd_str = self._visit_expr(e.second, ctx)
        trd_str = self._visit_expr(e.third, ctx)
        match e:
            case Fma():
                return f'fpy::fma({fst_str}, {snd_str}, {trd_str}, {ctx.ctx_name})'
            case _:
                raise CppFpyCompileError(self.func, f'Unsupported ternary operation to compile: {e}')

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
                raise CppFpyCompileError(self.func, f'expected list for `{arg.format()}`')
            arg_tys.append(arg_ty.elt)

        t = self._fresh_var()
        i = self._fresh_var()
        tmpl = ', '.join(self._compile_type(ty).to_cpp() for ty in arg_tys)
        ctx.add_line(f'std::vector<std::tuple<{tmpl}>> {t}({xs[0]}.size());')
        ctx.add_line(f'for (size_t {i} = 0; {i} < {xs[0]}.size(); ++{i}) {{')
        ctx.indent().add_line(f'{t}[{i}] = std::make_tuple({", ".join(f"{x}[{i}]" for x in xs)});')
        ctx.add_line('}')

        return t

    def _visit_naryop(self, e, ctx: CompileCtx):
        args_str = [self._visit_expr(arg, ctx) for arg in e.args]
        match e:
            case Zip():
                return self._visit_zip(e, ctx)
            case _:
                raise CppFpyCompileError(self.func, f'Unsupported nary operation to compile: {e}')

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
            return f'fpy::round({arg}, {ctx.ctx_name})'

    def _visit_round_at(self, e: RoundAt, ctx: CompileCtx):
        raise CppFpyCompileError(self.func, '`round_at` is unsupported')

    def _visit_len(self, e: Len, ctx: CompileCtx):
        # len(x)
        arg_cpp = self._visit_expr(e.arg, ctx)
        ty = self._expr_type(e)
        return self._compile_size(f'{arg_cpp}.size()', ty)

    def _visit_call(self, e, ctx):
        raise NotImplementedError

    def _visit_compare(self, e, ctx):
        raise NotImplementedError

    def _visit_tuple_expr(self, e, ctx):
        raise NotImplementedError

    def _visit_list_expr(self, e, ctx):
        raise NotImplementedError

    def _visit_list_comp(self, e, ctx):
        raise NotImplementedError

    def _visit_list_ref(self, e: ListRef, ctx: CompileCtx):
        value = self._visit_expr(e.value, ctx)
        index = self._visit_expr(e.index, ctx)
        index_ty = self._expr_type(e.index)
        size = self._compile_size(index, index_ty)
        return f'{value}[{size}]'

    def _visit_list_slice(self, e, ctx):
        raise NotImplementedError

    def _visit_list_set(self, e, ctx):
        raise NotImplementedError

    def _visit_if_expr(self, e, ctx):
        raise NotImplementedError

    def _visit_attribute(self, e, ctx):
        raise NotImplementedError

    def _visit_decl(self, name: Id, e: str, site: DefSite, ctx: CompileCtx):
        match name:
            case NamedId():
                if self._var_is_decl(name, site):
                    ty_str = self._compile_type(self._var_type(name, site)).to_cpp()
                    ctx.add_line(f'{ty_str} {name} = {e};')
                else:
                    ctx.add_line(f'{name} = {e};')
            case UnderscoreId():
                ctx.add_line(f'auto _ = {e};')
            case _:
                raise RuntimeError(f'unreachable: {name}')

    def _visit_tuple_binding(self, t_id: str, binding: TupleBinding, site: DefSite, ctx: CompileCtx):
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

    def _visit_assign(self, stmt: Assign, ctx: CompileCtx):
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

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: CompileCtx):
        # compile indices
        indices: list[str] = []
        for index in stmt.indices:
            i = self._visit_expr(index, ctx)
            idx_ty = self._expr_type(index)
            indices.append(self._compile_size(i, idx_ty))

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

    def _visit_for(self, stmt: ForStmt, ctx: CompileCtx):
        if isinstance(stmt.target, Id) and isinstance(stmt.iterable, Range1):
            # special case: for i in range(n)
            # for (size_t i = 0; i < n; ++i) {
            n = self._visit_expr(stmt.iterable.arg, ctx)
            ctx.add_line(f'for (size_t {stmt.target} = 0; {stmt.target} < {n}; ++{stmt.target}) {{')
        else:
            # general case: for x in iterable
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

    def _visit_context(self, stmt: ContextStmt, ctx: CompileCtx):
        if not isinstance(stmt.ctx, ForeignVal):
            raise CppFpyCompileError(self.func, f'Rounding context cannot be compiled `{stmt.ctx}`')

        # name to bind context
        if isinstance(stmt.target, NamedId):
            ctx_name = str(stmt.target)
        else:
            ctx_name = self._fresh_var()

        # compile context
        ctx_str = self._compile_context(stmt.ctx.val)
        ctx.add_line(f'auto {ctx_name} = {ctx_str};')

        # compile body
        self._visit_block(stmt.body, ctx.with_ctx(ctx_name))

    def _visit_assert(self, stmt: AssertStmt, ctx: CompileCtx):
        e = self._visit_expr(stmt.test, ctx)
        if stmt.msg is None:
            ctx.add_line(f'assert({e});')
        else:
            msg = self._visit_expr(stmt.msg, ctx)
            ctx.add_line(f'assert(({e}) && ({msg}));')

    def _visit_effect(self, stmt: EffectStmt, ctx: CompileCtx):
        raise CppFpyCompileError(self.func, 'FPy effects are not supported')

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
            ty_str = self._compile_type(arg_ty).to_cpp()
            arg_strs.append(f'{ty_str} {arg.name}')

        ret_ty = self.fmt_info.return_type
        self._check_type(ret_ty)
        ty_str = self._compile_type(ret_ty).to_cpp()
        ctx.add_line(f'{ty_str} {self.name}({", ".join(arg_strs)}) {{')

        # compile body
        self._visit_block(func.body, ctx.indent())
        ctx.add_line('}')


class CppFpyBackend(Backend):
    """
    C++/FPy backend: compiler to C++ number library.

    TODO: need a name for the library.
    """

    # unsafe options
    unsafe_cast_int: bool
    unsafe_finitize_int: bool

    def __init__(self, *, unsafe_cast_int: bool = False, unsafe_finitize_int: bool = False):
        self.unsafe_cast_int = unsafe_cast_int
        self.unsafe_finitize_int = unsafe_finitize_int

    def compile(
        self,
        func: Function,
        *,
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

        # monomorphizing
        ast = func.ast
        if arg_types is None:
            arg_types = [None for _ in func.args]
        if ast.ctx is not None:
            ctx = None
        ast = Monomorphize.apply_by_arg(ast, ctx, arg_types)

        # normalization passes
        func = simplify(func.with_ast(ast))
        ast = func.ast

        # run type checking with static context inference
        try:
            ctx_info = ContextInfer.infer(ast, unsafe_cast_int=self.unsafe_cast_int)
            format_info = FormatInfer.infer(ast, ctx_info=ctx_info)
        except (ContextInferError, TypeInferError, FormatInferError) as e:
            raise ValueError(f'{func.name}: context inference failed') from e

        # compile
        options = CppOptions(self.unsafe_finitize_int, self.unsafe_cast_int)
        inst = _CppBackendInstance(ast, func.name, options,format_info)
        body_str = inst.compile()

        return body_str
