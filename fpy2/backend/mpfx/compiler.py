"""
MPFX backend: compiler to MPFX number library.
"""

from typing import Collection

from ...ast import *
from ...analysis import (
    ContextInfer, ContextInferError,
    DefineUse, DefineUseAnalysis, Definition, DefSite, AssignDef, PhiDef,
    PartialEval, PartialEvalInfo, TypeInferError
)
from ...function import Function
from ...number import IEEEContext, OV, RM, FP64, INTEGER
from ...transform import Monomorphize
from ...strategies import simplify
from ...types import BoolType, ContextType, RealType, VarType, Type
from ...utils import Gensym
from ..backend import Backend

from .format import AbstractFormat
from .format_infer import (
    FormatAnalysis, FormatInfer, FormatInferError,
    FormatType, ListFormatType, TupleFormatType
)
from .instr import (
    NegInstr, AbsInstr, SqrtInstr,
    AddInstr, SubInstr, MulInstr, DivInstr, FMAInstr
)
from .types import *
from .utils import MPFXCompileError, CompileCtx, CppOptions


class _MPFXBackendInstance(Visitor):
    """
    Per-function compilation instance.

    Based on default C++ compiler.
    """

    func: FuncDef
    name: NamedId
    options: CppOptions
    fmt_info: FormatAnalysis
    eval_info: PartialEvalInfo

    decl_phis: set[PhiDef]
    decl_assigns: set[AssignDef]
    gensym: Gensym

    def __init__(
        self,
        func: FuncDef,
        name: NamedId,
        options: CppOptions,
        fmt_info: FormatAnalysis,
        eval_info: PartialEvalInfo,
    ):
        self.func = func
        self.name = name
        self.options = options
        self.fmt_info = fmt_info
        self.eval_info = eval_info

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

    def _compile_rm(self, rm: RM) -> str:
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
                raise MPFXCompileError(self.func, f'Unsupported rounding mode to compile: {rm}')

    def _compile_context(self, ctx: Context) -> str:
        match ctx:
            case IEEEContext():
                # ES <= 9
                if ctx.es > 9:
                    raise MPFXCompileError(self.func, f'IEEE 754: exponent too large to compile: {ctx.es}')
                # P <= 51
                if ctx.pmax > 51:
                    raise MPFXCompileError(self.func, f'IEEE 754: precision too large to compile: {ctx.pmax}')
                # overflow
                if ctx.overflow != OV.OVERFLOW:
                    raise MPFXCompileError(self.func, 'IEEE 754: overflow mode RAISE cannot be compiled')
                # no random
                if ctx.num_randbits != 0:
                    raise MPFXCompileError(self.func, 'IEEE 754: stochastic rounding cannot be compiled')

                # compile rounding
                rm = self._compile_rm(ctx.rm)
                return f'mpfx::IEEE754Context({ctx.es}, {ctx.nbits}, {rm})'
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
                    raise MPFXCompileError(self.func, f'no container type for: {ty}')
            case TupleFormatType():
                # compile recursively
                return CppTupleType(self._compile_type(elem_ty) for elem_ty in ty.elts)
            case ListFormatType():
                # compile recursively
                return CppListType(self._compile_type(ty.elt))
            case ContextType():
                return CppContextType()
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
        arg_str = self._visit_expr(e.arg, ctx)
        match e:
            case Neg():
                arg_ty = self._expr_type(e.arg)
                e_ctx = self._expr_ctx(e)
                gen = NegInstr.generator(arg_ty, e_ctx)
                if gen is None:
                    raise MPFXCompileError(self.func, f'No suitable implementation found for negation: {e}')
                return gen(arg_str, ctx.ctx_name)
            case Abs():
                arg_ty = self._expr_type(e.arg)
                e_ctx = self._expr_ctx(e)
                gen = AbsInstr.generator(arg_ty, e_ctx)
                if gen is None:
                    raise MPFXCompileError(self.func, f'No suitable implementation found for absolute value: {e}')
                return gen(arg_str, ctx.ctx_name)
            case Sqrt():
                arg_ty = self._expr_type(e.arg)
                e_ctx = self._expr_ctx(e)
                gen = SqrtInstr.generator(arg_ty, e_ctx)
                if gen is None:
                    raise MPFXCompileError(self.func, f'No suitable implementation found for square root: {e}')
                return gen(arg_str, ctx.ctx_name)
            case Len():
                return self._visit_len(e, ctx)
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
                gen = AddInstr.generator(lhs_ty, rhs_ty, e_ctx)
                if gen is None:
                    raise MPFXCompileError(self.func, f'No suitable implementation found for addition: {e}')
                return gen(lhs_str, rhs_str, ctx.ctx_name)
            case Sub():
                lhs_ty = self._expr_type(e.first)
                rhs_ty = self._expr_type(e.second)
                e_ctx = self._expr_ctx(e)
                gen = SubInstr.generator(lhs_ty, rhs_ty, e_ctx)
                if gen is None:
                    raise MPFXCompileError(self.func, f'No suitable implementation found for subtraction: {e}')
                return gen(lhs_str, rhs_str, ctx.ctx_name)
            case Mul():
                lhs_ty = self._expr_type(e.first)
                rhs_ty = self._expr_type(e.second)
                e_ctx = self._expr_ctx(e)
                gen = MulInstr.generator(lhs_ty, rhs_ty, e_ctx)
                if gen is None:
                    raise MPFXCompileError(self.func, f'No suitable implementation found for multiplication: {e}')
                return gen(lhs_str, rhs_str, ctx.ctx_name)
            case Div():
                lhs_ty = self._expr_type(e.first)
                rhs_ty = self._expr_type(e.second)
                e_ctx = self._expr_ctx(e)
                gen = DivInstr.generator(lhs_ty, rhs_ty, e_ctx)
                if gen is None:
                    raise MPFXCompileError(self.func, f'No suitable implementation found for division: {e}')
                return gen(lhs_str, rhs_str, ctx.ctx_name)
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
                gen = FMAInstr.generator(fst_ty, snd_ty, trd_ty, e_ctx)
                if gen is None:
                    raise MPFXCompileError(self.func, f'No suitable implementation found for fused multiply-add: {e}')
                return gen(fst_str, snd_str, trd_str, ctx.ctx_name)
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

    def _visit_naryop(self, e, ctx: CompileCtx):
        match e:
            case Zip():
                return self._visit_zip(e, ctx)
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
        ty = self._expr_type(e)
        return self._compile_size(f'{arg_cpp}.size()', ty)

    def _visit_call(self, e, ctx):
        if e not in self.eval_info.by_expr:
            raise MPFXCompileError(self.func, f'cannot compile `{e.format()}`')
        val = self.eval_info.by_expr[e]
        if isinstance(val, Context):
            return self._compile_context(val)
        else:
            raise MPFXCompileError(self.func, f'cannot compile `{e.format()}`')

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
        # name to bind context
        if isinstance(stmt.target, NamedId):
            ctx_name = str(stmt.target)
        else:
            ctx_name = self._fresh_var()

        # compile context
        ctx_str = self._visit_expr(stmt.ctx, ctx)
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
            ty_str = self._compile_type(arg_ty).to_cpp()
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

        # partial evaluation
        def_use = DefineUse.analyze(ast)
        eval_info = PartialEval.apply(ast, def_use=def_use)

        # run type checking with static context inference
        try:
            ctx_info = ContextInfer.infer(ast, def_use=def_use, unsafe_cast_int=self.unsafe_cast_int)
            format_info = FormatInfer.infer(ast, ctx_info=ctx_info)
        except (ContextInferError, TypeInferError, FormatInferError) as e:
            raise ValueError(f'{func.name}: context inference failed') from e

        # compile
        options = CppOptions(self.unsafe_finitize_int, self.unsafe_cast_int)
        inst = _MPFXBackendInstance(ast, func.name, options, format_info, eval_info)
        body_str = inst.compile()

        return body_str
