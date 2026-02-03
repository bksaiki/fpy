"""
Abstract format inference for FPy programs.
"""

from dataclasses import dataclass
from typing import NoReturn, TypeAlias, Iterable

from ...ast import *
from ...analysis import (
    ContextAnalysis, ContextInfer, PartialEval, PartialEvalInfo,
    Definition, DefSite
)
from ...number import Context, INTEGER, REAL
from ...types import *
from ...utils import default_repr

from .format import AbstractFormat, SupportedContext

__all__ = [
    'FormatInfer',
    'FormatInferError',
    'FormatAnalysis',
    'FormatType',
    'ListFormatType',
    'TupleFormatType'
]


@default_repr
class ListFormatType:
    """List type: element type can be an abstract format"""
    elt: 'FormatType'

    def __init__(self, elt: 'FormatType'):
        self.elt = elt

    def __hash__(self):
        return hash((ListFormatType, self.elt))

    def __eq__(self, other):
        return isinstance(other, ListFormatType) and self.elt == other.elt

@default_repr
class TupleFormatType:
    """Tuple type: element types can be abstract formats"""
    elts: tuple['FormatType', ...]

    def __init__(self, elts: Iterable['FormatType']):
        self.elts = tuple(elts)

    def __hash__(self):
        return hash((TupleFormatType, self.elts))

    def __eq__(self, other):
        return isinstance(other, TupleFormatType) and self.elts == other.elts

@default_repr
class FunctionFormatType:
    """Function type: argument and return types can be abstract formats"""
    arg_types: tuple['FormatType', ...]
    ret_type: 'FormatType'

    def __init__(self, arg_types: Iterable['FormatType'], ret_type: 'FormatType'):
        self.arg_types = tuple(arg_types)
        self.ret_type = ret_type

    def __hash__(self):
        return hash((FunctionFormatType, self.arg_types, self.ret_type))

    def __eq__(self, other):
        return (
            isinstance(other, FunctionFormatType)
            and self.arg_types == other.arg_types
            and self.ret_type == other.ret_type
        )


FormatType: TypeAlias = (
    ListFormatType | TupleFormatType | FunctionFormatType
    | BoolType | ContextType | RealType
    | AbstractFormat
)
"""The normal type system extended with abstract formats."""

def _cvt_context(ctx: Context):
    if isinstance(ctx, SupportedContext):
        return AbstractFormat.from_context(ctx)
    else:
        return None

def convert_type(ty: Type) -> FormatType:
    """
    Converts a normal type to a format type by replacing real types
    with abstract formats.

    Args:
        ty: The type to convert.
    Returns:
        The converted format type.
    """
    match ty:
        case BoolType() | ContextType():
            return ty
        case RealType():
            if isinstance(ty.ctx, Context):
                fmt = _cvt_context(ty.ctx)
                return ty if fmt is None else fmt
            else:
                return ty
        case ListType():
            elt = convert_type(ty.elt)
            return ListFormatType(elt)
        case TupleType():
            elts = [convert_type(t) for t in ty.elts]
            return TupleFormatType(elts)
        case _:
            raise RuntimeError(f'Unsupported type: {ty}')


class FormatInferError(Exception):
    """
    Exception raised when format inference fails.
    """
    pass


@dataclass(frozen=True)
class FormatAnalysis:
    """
    Format analysis result.
    - `by_def`: mapping from definitions to inferred abstract formats
    - `by_expr`: mapping from expressions to inferred abstract formats
    """
    fn_type: FunctionFormatType
    by_def: dict[Definition, FormatType]
    by_expr: dict[Expr, FormatType]
    preround: dict[Expr, FormatType]
    ctx_info: ContextAnalysis

    @property
    def arg_types(self):
        return self.fn_type.arg_types

    @property
    def return_type(self):
        return self.fn_type.ret_type

    @property
    def def_use(self):
        return self.ctx_info.def_use


class _FormatInfernce(Visitor):
    """
    Visitor for format inference.
    """

    func: FuncDef
    ctx_info: ContextAnalysis
    eval_info: PartialEvalInfo

    by_def: dict[Definition, FormatType]
    by_expr: dict[Expr, FormatType]
    preround: dict[Expr, FormatType]
    ret_ty: FormatType | None

    def __init__(self, func: FuncDef, ctx_info: ContextAnalysis, eval_info: PartialEvalInfo):
        self.func = func
        self.ctx_info = ctx_info
        self.eval_info = eval_info
        self.by_def = {}
        self.by_expr = {}
        self.preround = {}
        self.ret_ty = None

    @property
    def def_use(self):
        return self.ctx_info.def_use

    def infer(self):
        fn_ty = self._visit_function(self.func, None)
        return FormatAnalysis(fn_ty, self.by_def, self.by_expr, self.preround, self.ctx_info)

    def raise_error(self, msg: str) -> NoReturn:
        raise FormatInferError(f'In function {self.func.name}: {msg}')

    def _expr_type(self, e: Expr):
        ty = self.ctx_info.by_expr[e]
        return convert_type(ty)

    def _visit_binding(self, site: DefSite, target: Id | TupleBinding, ty: FormatType):
        match target:
            case NamedId():
                # TODO: union with existing type?
                d = self.def_use.find_def_from_site(target, site)
                self.by_def[d] = ty
            case UnderscoreId():
                pass
            case TupleBinding():
                assert isinstance(ty, TupleFormatType)
                for elt, elt_ty in zip(target.elts, ty.elts, strict=True):
                    self._visit_binding(site, elt, elt_ty)
            case _:
                raise RuntimeError(f'unreachable: {target}')

    def _visit_var(self, e: Var, ctx: Context):
        d = self.def_use.find_def_from_use(e)
        return self.by_def[d]

    def _visit_bool(self, e: BoolVal, ctx: Context):
        return self._expr_type(e)

    def _visit_foreign(self, e: ForeignVal, ctx: Context):
        return self._expr_type(e)

    def _visit_decnum(self, e: Decnum, ctx: Context):
        return self._expr_type(e)

    def _visit_hexnum(self, e: Hexnum, ctx: Context):
        return self._expr_type(e)

    def _visit_integer(self, e: Integer, ctx: Context):
        return AbstractFormat.from_context(INTEGER)

    def _visit_rational(self, e: Rational, ctx: Context):
        raise NotImplementedError

    def _visit_digits(self, e: Digits, ctx: Context):
        raise NotImplementedError

    def _visit_nullaryop(self, e: NullaryOp, ctx: Context):
        raise NotImplementedError

    def _record_preround(self, e: Expr, m_ty: AbstractFormat, e_ty: AbstractFormat | RealType):
        """
        Records that an expression `e` has inferred context `m_ty`
        but has expected context `e_ty`.
        """
        if isinstance(e_ty, AbstractFormat):
            if m_ty.contained_in(e_ty):
                self.preround[e] = m_ty
            # intersect with expected type to get the most precise type needed
            return e_ty & m_ty
        elif isinstance(e_ty, RealType) and e_ty.ctx == REAL:
            # intersecting with REAL
            self.preround[e] = m_ty
            return m_ty
        else:
            return m_ty


    def _visit_unaryop(self, e: UnaryOp, ctx: Context):
        a_ty = self._visit_expr(e.arg, ctx)
        e_ty = self._expr_type(e) # get the expected type

        if isinstance(a_ty, AbstractFormat):
            # possible optimization: if A are both abstract formats,
            # then the "minimal" format can be computed directly
            match e:
                case Round() | Cast():
                    m_ty: AbstractFormat | None = a_ty
                case Neg():
                    m_ty = -a_ty
                case _:
                    m_ty = None

            if m_ty is not None:
                assert isinstance(e_ty, AbstractFormat | RealType)
                e_ty = self._record_preround(e, m_ty, e_ty)

        return e_ty

    def _visit_binaryop(self, e: BinaryOp, ctx: Context):
        a_ty = self._visit_expr(e.first, ctx)
        b_ty = self._visit_expr(e.second, ctx)
        e_ty = self._expr_type(e)

        if isinstance(a_ty, AbstractFormat) and isinstance(b_ty, AbstractFormat):
            # possible optimization: if A and B are both abstract formats,
            # then the "minimal" format can be computed directly
            match e:
                case Add():
                    m_ty: AbstractFormat | None = a_ty + b_ty
                case Sub():
                    m_ty = a_ty - b_ty
                case Mul():
                    m_ty = a_ty * b_ty
                case _:
                    m_ty = None

            if m_ty is not None:
                assert isinstance(e_ty, AbstractFormat | RealType)
                e_ty = self._record_preround(e, m_ty, e_ty)

        # return the most conservative type
        return e_ty

    def _visit_ternaryop(self, e: TernaryOp, ctx: Context):
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        self._visit_expr(e.third, ctx)
        return self._expr_type(e) # get the expected type

    def _visit_naryop(self, e: NaryOp, ctx: Context):
        for arg in e.args:
            self._visit_expr(arg, ctx)
        return self._expr_type(e) # get the expected type

    def _visit_call(self, e: Call, ctx: Context):
        for arg in e.args:
            self._visit_expr(arg, ctx)
        for _, kwarg in e.kwargs:
            self._visit_expr(kwarg, ctx)

        val = self.eval_info.by_expr[e]
        if isinstance(val, Context):
            return ContextType()
        else:
            raise NotImplementedError(e)

    def _visit_compare(self, e: Compare, ctx: Context):
        for arg in e.args:
            self._visit_expr(arg, ctx)
        return self._expr_type(e)

    def _visit_tuple_expr(self, e: TupleExpr, ctx: Context):
        elts = [self._visit_expr(elt, ctx) for elt in e.elts]
        return TupleFormatType(elts)

    def _visit_list_expr(self, e: ListExpr, ctx: Context):
        elts = [self._visit_expr(elt, ctx) for elt in e.elts]
        # TODO: unify element types?
        return ListFormatType(elts[0])

    def _visit_list_comp(self, e: ListComp, ctx: Context):
        for target, iterable in zip(e.targets, e.iterables, strict=True):
            iter_ty = self._visit_expr(iterable, ctx)
            assert isinstance(iter_ty, ListFormatType)
            self._visit_binding(e, target, iter_ty.elt)

        return ListFormatType(self._visit_expr(e.elt, ctx))

    def _visit_list_ref(self, e: ListRef, ctx: Context):
        # Γ |- e: list T   Γ |- i: real A
        # --------------------------------
        #         Γ |- e[i]: T
        ty = self._visit_expr(e.value, ctx)
        self._visit_expr(e.index, ctx)
        assert isinstance(ty, ListFormatType)
        return ty.elt

    def _visit_list_slice(self, e: ListSlice, ctx: Context):
        if e.start is not None:
            self._visit_expr(e.start, ctx)
        if e.stop is not None:
            self._visit_expr(e.stop, ctx)
        ty = self._visit_expr(e.value, ctx)
        assert isinstance(ty, ListFormatType)
        return ty

    def _visit_list_set(self, e: ListSet, ctx: Context):
        raise NotImplementedError

    def _visit_if_expr(self, e: IfExpr, ctx: Context):
        self._visit_expr(e.cond, ctx)
        ift_ty = self._visit_expr(e.ift, ctx)
        _ = self._visit_expr(e.iff, ctx)
        # TODO: unify ift_ty and iff_ty?
        return ift_ty

    def _visit_attribute(self, e: Attribute, ctx: Context):
        if e not in self.eval_info.by_expr:
            self.raise_error(f'cannot determine a compilable type for `{e.format()}`')
        val = self.eval_info.by_expr[e]
        if isinstance(val, Context):
            return ContextType()
        else:
            self.raise_error(f'cannot determine a compilable type for `{e.format()}`')

    def _visit_assign(self, stmt: Assign, ctx: Context):
        ty = self._visit_expr(stmt.expr, ctx)
        self._visit_binding(stmt, stmt.target, ty)
        return ctx

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: Context):
        for idx in stmt.indices:
            self._visit_expr(idx, ctx)
        self._visit_expr(stmt.expr, ctx)

    def _visit_if1(self, stmt: If1Stmt, ctx: Context):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)

        # add types to phi variables
        for phi in self.def_use.phis[stmt]:
            # TODO: unify?
            lhs_ty = self.by_def[self.def_use.defs[phi.lhs]]
            self.by_def[phi] = lhs_ty

        return ctx

    def _visit_if(self, stmt: IfStmt, ctx: Context):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.ift, ctx)
        self._visit_block(stmt.iff, ctx)

        # unify any merged variable
        # add types to phi variables
        for phi in self.def_use.phis[stmt]:
            # TODO: unify?
            lhs_ty = self.by_def[self.def_use.defs[phi.lhs]]
            _ = self.by_def[self.def_use.defs[phi.rhs]]
            self.by_def[phi] = lhs_ty

        return ctx

    def _visit_while(self, stmt: WhileStmt, ctx: Context):
        raise NotImplementedError

    def _visit_for(self, stmt: ForStmt, ctx: Context):
        # get the type of the iterable and bind it
        iter_ty = self._visit_expr(stmt.iterable, ctx)
        assert isinstance(iter_ty, ListFormatType)
        self._visit_binding(stmt, stmt.target, iter_ty.elt)

        # add types to phi variables
        for phi in self.def_use.phis[stmt]:
            # TODO: unify?
            lhs_ty = self.by_def[self.def_use.defs[phi.lhs]]
            self.by_def[phi] = lhs_ty

        # visit the body
        self._visit_block(stmt.body, ctx)

        # TODO: unify?
        for phi in self.def_use.phis[stmt]:
            _ = self.by_def[self.def_use.defs[phi.lhs]]
            _ = self.by_def[self.def_use.defs[phi.rhs]]

        return ctx

    def _visit_context(self, stmt: ContextStmt, ctx: Context):
        if isinstance(stmt.ctx, ForeignVal) and isinstance(stmt.ctx.val, Context):
            # check if the context is a concrete context
            body_ctx = stmt.ctx.val
        elif stmt.ctx in self.eval_info.by_expr:
            # backup is to lookup partial eval info
            val = self.eval_info.by_expr[stmt.ctx]
            if isinstance(val, Context):
                body_ctx = val
            else:
                self.raise_error(f'cannot infer context for `{stmt.ctx.format()}` at `{stmt.format()}`')
        else:
            self.raise_error(f'cannot infer context for `{stmt.ctx.format()}` at `{stmt.format()}`')

        self._visit_block(stmt.body, body_ctx)

    def _visit_assert(self, stmt: AssertStmt, ctx: Context):
        self._visit_expr(stmt.test, ctx)

    def _visit_effect(self, stmt: EffectStmt, ctx: Context):
        self._visit_expr(stmt.expr, ctx)

    def _visit_return(self, stmt: ReturnStmt, ctx: Context):
        self.ret_ty = self._visit_expr(stmt.expr, ctx)

    def _visit_pass(self, stmt: PassStmt, ctx: Context):
        pass

    def _visit_block(self, block: StmtBlock, ctx: Context):
        for stmt in block.stmts:
            ctx = self._visit_statement(stmt, ctx)

    def _visit_expr(self, expr: Expr, ctx: Context) -> FormatType:
        ty = super()._visit_expr(expr, ctx)
        # print(expr.format(), '::', ty)
        self.by_expr[expr] = ty
        return ty

    def _visit_function(self, func: FuncDef, _: None):
        # must have a concrete context from caller
        if not isinstance(func.ctx, Context):
            raise FormatInferError('no concrete context')

        # function arguments
        arg_types: list[FormatType] = []
        for arg, ty in zip(func.args, self.ctx_info.arg_types, strict=True):
            fmt_ty = convert_type(ty)
            arg_types.append(fmt_ty)
            if isinstance(arg.name, NamedId):
                d = self.def_use.find_def_from_site(arg.name, arg)
                self.by_def[d] = fmt_ty

        # function body
        self._visit_block(func.body, func.ctx)

        # function return type
        assert self.ret_ty is not None
        return FunctionFormatType(arg_types, self.ret_ty)


###########################################################
# Format inference

class FormatInfer:
    """
    Format inference for the FPy language.

    For every real typed expression, infer an abstract format
    that safely contains all possible values of the expression.
    """

    @staticmethod
    def infer(
        func: FuncDef, *,
        ctx_info: ContextAnalysis | None = None,
        eval_info: PartialEvalInfo | None = None
    ):
        """
        Performs format inference on the given function definition.

        Args:
            func: The function definition to analyze.
            ctx_info: Optional context analysis information.
        Returns:
            ???
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {func}')

        # run context inference if need be
        if ctx_info is None:
            ctx_info = ContextInfer.infer(func)

        if eval_info is None:
            eval_info = PartialEval.apply(func, def_use=ctx_info.def_use)

        # perform format inference
        format_info = _FormatInfernce(func, ctx_info, eval_info).infer()

        return format_info
