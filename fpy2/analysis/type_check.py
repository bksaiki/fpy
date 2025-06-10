"""Type checking for FPy programs."""

from typing import TypeAlias

from ..ast import *
from ..number import Context
from ..utils import Gensym, Unionfind
from .define_use import DefineUse, DefineUseAnalysis, Definition

_Type: TypeAlias = TypeAnn | NamedId
"""type: either a type annotation or type variable"""

_RCtx: TypeAlias = Context | NamedId
"""rounding context: either a rounding context or a context variable"""

_TCtx: TypeAlias = dict[NamedId, _Type]
"""typing context: mapping from variable to type"""

_TRCtx: TypeAlias = tuple[_TCtx, _RCtx]
"""typing context extended with a rounding context"""


_Real1ary = FuncTypeAnn(
    NamedId('r'),
    [RealTypeAnn(NamedId('a'), None)],
    RealTypeAnn(NamedId('r'), None),
    None
)

_Bool1ary = FuncTypeAnn(
    NamedId('r'),
    [BoolTypeAnn(None)],
    BoolTypeAnn(None),
    None
)

_Real2ary = FuncTypeAnn(
    NamedId('r'),
    [RealTypeAnn(NamedId('a'), None), RealTypeAnn(NamedId('b'), None)],
    RealTypeAnn(NamedId('r'), None),
    None
)

_Real3ary = FuncTypeAnn(
    NamedId('r'),
    [RealTypeAnn(NamedId('a'), None), RealTypeAnn(NamedId('b'), None), RealTypeAnn(NamedId('c'), None)],
    RealTypeAnn(NamedId('r'), None),
    None
)

_Predicate = FuncTypeAnn(
    NamedId('r'),
    [RealTypeAnn(NamedId('a'), None)],
    BoolTypeAnn(None),
    None
)

_Compare = FuncTypeAnn(
    NamedId('r'),
    [RealTypeAnn(NamedId('a'), None), RealTypeAnn(NamedId('b'), None)],
    BoolTypeAnn(None),
    None
)


_unary_table: dict[type[UnaryOp], FuncTypeAnn] = {
    Fabs: _Real1ary,
    Sqrt: _Real1ary,
    Neg: _Real1ary,
    Cbrt: _Real1ary,
    Ceil: _Real1ary,
    Floor: _Real1ary,
    NearbyInt: _Real1ary,
    Round: _Real1ary,
    Trunc: _Real1ary,
    Acos: _Real1ary,
    Asin: _Real1ary,
    Atan: _Real1ary,
    Cos: _Real1ary,
    Sin: _Real1ary,
    Tan: _Real1ary,
    Acosh: _Real1ary,
    Asinh: _Real1ary,
    Atanh: _Real1ary,
    Cosh: _Real1ary,
    Sinh: _Real1ary,
    Tanh: _Real1ary,
    Exp: _Real1ary,
    Exp2: _Real1ary,
    Expm1: _Real1ary,
    Log: _Real1ary,
    Log10: _Real1ary,
    Log1p: _Real1ary,
    Log2: _Real1ary,
    Erf: _Real1ary,
    Erfc: _Real1ary,
    Lgamma: _Real1ary,
    Tgamma: _Real1ary,
    IsFinite: _Predicate,
    IsInf: _Predicate,
    IsNan: _Predicate,
    IsNormal: _Predicate,
    Signbit: _Predicate,
    Not: _Bool1ary,
    Cast: _Real1ary,
}

_binary_table: dict[type[BinaryOp], FuncTypeAnn] = {
    Add: _Real2ary,
    Sub: _Real2ary,
    Mul: _Real2ary,
    Div: _Real2ary,
    Copysign: _Real2ary,
    Fdim: _Real2ary,
    Fmax: _Real2ary,
    Fmin: _Real2ary,
    Fmod: _Real2ary,
    Remainder: _Real2ary,
    Hypot: _Real2ary,
    Atan2: _Real2ary,
    Pow: _Real2ary,
}

_ternary_table: dict[type[TernaryOp], FuncTypeAnn] = {
    Fma: _Real3ary,
}


class _TypeCheckInstance(Visitor):
    """Single-use instance of type checking."""
    func: FuncDef
    def_use: DefineUseAnalysis
    types: dict[Definition, _Type]
    ret_type: Optional[_Type]
    tvars: Unionfind[_TCtx | _RCtx]
    gensym: Gensym

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        self.func = func
        self.def_use = def_use
        self.types = {}
        self.ret_type = None
        self.tvars = Unionfind()
        self.gensym = Gensym()

    def analyze(self):
        return self._visit_function(self.func, None)

    def _inst_rec(self, ty: _Type, to_tvar: dict[NamedId, NamedId]) -> _Type:
        match ty:
            case NamedId():
                if ty not in to_tvar:
                    to_tvar[ty] = self.gensym.fresh()
                return to_tvar[ty]
            case BoolTypeAnn():
                return BoolTypeAnn(None)
            case RealTypeAnn():
                match ty.ctx:
                    case None | Context():
                        return RealTypeAnn(ty.ctx, None)
                    case NamedId():
                        if ty not in to_tvar:
                            to_tvar[ty.ctx] = self.gensym.fresh()
                        return RealTypeAnn(to_tvar[ty.ctx], None)
                    case _:
                        raise RuntimeError(f'unreachable type context: {ty.ctx}')
            case FuncTypeAnn():
                if isinstance(ty.ctx, NamedId):
                    if ty.ctx not in to_tvar:
                        to_tvar[ty.ctx] = self.gensym.fresh()
                    rctx: _RCtx = to_tvar[ty.ctx]
                else:
                    rctx = ty.ctx
                args = [self._inst_rec(arg, to_tvar) for arg in ty.args]
                ret = self._inst_rec(ty.ret, to_tvar)
                return FuncTypeAnn(rctx, args, ret, None)
            case _:
                raise RuntimeError(f'unreachable type: {ty}')

    def _inst(self, ty: _Type):
        """Instantiates a function type annotation with fresh type variables."""
        to_tvar: dict[NamedId, NamedId] = {}
        return self._inst_rec(ty, to_tvar)

    def _unify(self, a: _Type, b: _Type) -> _Type:
        """Unifies two types, returning the most general unifier."""
        match a, b:
            case NamedId(), NamedId():
                # TODO: unify
                return a
            case NamedId(), _:
                # if `a` is a type variable, unify it with `b`
                # TODO: unify
                return b
            case _, NamedId():
                # if `b` is a type variable, unify it with `a`
                # TODO: unify
                return a
            case BoolTypeAnn(), BoolTypeAnn():
                # always equal
                return a
            case RealTypeAnn(), RealTypeAnn():
                # might need to unify the rounding context
                raise NotImplementedError(a, b)
            case _:
                raise ValueError(f'cannot unify types: a={a} and b={b}')

    def _visit_var(self, e: Var, trctx: _TRCtx):
        tctx, _ = trctx
        if e.name not in tctx:
            raise RuntimeError(f'variable {e.name} not found in typing context: {tctx}')
        return tctx[e.name]

    def _visit_bool(self, e: BoolVal, trctx: _TRCtx):
        return BoolTypeAnn(None)

    def _visit_foreign(self, e: ForeignVal, trctx: _TRCtx):
        raise RuntimeError('cannot type check foreign values')

    def _visit_decnum(self, e: Decnum, trctx: _TRCtx):
        _, rctx = trctx
        return RealTypeAnn(rctx, None)

    def _visit_hexnum(self, e: Hexnum, trctx: _TRCtx):
        _, rctx = trctx
        return RealTypeAnn(rctx, None)

    def _visit_integer(self, e: Integer, trctx: _TRCtx):
        _, rctx = trctx
        return RealTypeAnn(rctx, None)

    def _visit_rational(self, e: Rational, trctx: _TRCtx):
        _, rctx = trctx
        return RealTypeAnn(rctx, None)

    def _visit_digits(self, e: Digits, trctx: _TRCtx):
        _, rctx = trctx
        return RealTypeAnn(rctx, None)

    def _visit_constant(self, e: Constant, trctx: _TRCtx):
        _, rctx = trctx
        return RealTypeAnn(rctx, None)

    def _visit_unaryop(self, e: UnaryOp, trctx: _TRCtx):
        raise NotImplementedError

    def _visit_binaryop(self, e: BinaryOp, trctx: _TRCtx):
        # lookup type signature for the operator
        cls = type(e)
        if cls not in _binary_table:
            raise TypeError(f'unknown binary operator: {cls}')
        fun_ty = _binary_table[cls]

        # instantiate the function type with fresh type variables
        fun_ty = self._inst(fun_ty)
        assert isinstance(fun_ty, FuncTypeAnn)

        # compute the types of the arguments
        arg1_ty = self._visit_expr(e.first, trctx)
        arg2_ty = self._visit_expr(e.second, trctx)

        # unify the types
        self._unify(fun_ty.args[0], arg1_ty)
        self._unify(fun_ty.args[1], arg2_ty)

        # return type is valid
        # TODO: apply unionfind?
        return fun_ty.ret

    def _visit_ternaryop(self, e: TernaryOp, trctx: _TRCtx):
        raise NotImplementedError

    def _visit_naryop(self, e: NaryOp, trctx: _TRCtx):
        raise NotImplementedError

    def _visit_compare(self, e: Compare, trctx: _TRCtx):
        raise NotImplementedError

    def _visit_call(self, e: Call, trctx: _TRCtx):
        raise NotImplementedError

    def _visit_tuple_expr(self, e: TupleExpr, trctx: _TRCtx):
        raise NotImplementedError

    def _visit_comp_expr(self, e: CompExpr, trctx: _TRCtx):
        raise NotImplementedError

    def _visit_tuple_ref(self, e: TupleRef, trctx: _TRCtx):
        raise NotImplementedError

    def _visit_tuple_set(self, e: TupleSet, trctx: _TRCtx):
        raise NotImplementedError

    def _visit_if_expr(self, e: IfExpr, trctx: _TRCtx):
        raise NotImplementedError

    def _visit_context_expr(self, e: ContextExpr, trctx: _TRCtx):
        raise NotImplementedError

    def _visit_assign(self, stmt: Assign, trctx: _TRCtx):
        # evaluate the expression to get its type
        ty = self._visit_expr(stmt.expr, trctx)
        if stmt.type is not None:
            ty = self._unify(stmt.type, ty)

        # bind to variables
        tctx, rctx = trctx
        match stmt.binding:
            case NamedId():
                # update the typing context
                tctx = { **tctx, stmt.binding: ty}
            case Id():
                # do nothing to the typing context
                pass
            case TupleBinding():
                # TODO: unpack tuple
                raise NotImplementedError(stmt)
            case _:
                raise RuntimeError(f'unreachable {stmt.binding}')

        return (tctx, rctx)

    def _visit_indexed_assign(self, stmt: IndexedAssign, trctx: _TRCtx):
        raise NotImplementedError

    def _visit_if1(self, stmt: If1Stmt, trctx: _TRCtx):
        raise NotImplementedError

    def _visit_if(self, stmt: IfStmt, trctx: _TRCtx):
        raise NotImplementedError

    def _visit_while(self, stmt: WhileStmt, trctx: _TRCtx):
        raise NotImplementedError

    def _visit_for(self, stmt: ForStmt, trctx: _TRCtx):
        raise NotImplementedError

    def _visit_context(self, stmt: ContextStmt, trctx: _TRCtx):
        raise NotImplementedError

    def _visit_assert(self, stmt: AssertStmt, trctx: _TRCtx):
        raise NotImplementedError

    def _visit_effect(self, stmt: EffectStmt, trctx: _TRCtx):
        raise NotImplementedError

    def _visit_return(self, stmt: ReturnStmt, trctx: _TRCtx) -> _TRCtx:
        ty = self._visit_expr(stmt.expr, trctx)
        # check if the return type is already set
        if self.ret_type is None:
            self.ret_type = ty
        else:
            self.ret_type = self._unify(self.ret_type, ty)
        # dummy return value
        return ({}, trctx[1])

    def _visit_block(self, block: StmtBlock, trctx: _TRCtx):
        for stmt in block.stmts:
            trctx = self._visit_statement(stmt, trctx)
        return trctx

    def _visit_function(self, func: FuncDef, _: None):
        tctx: _TCtx = {}
        rctx: _RCtx = self.gensym.fresh('r')

        arg_tys: list[TypeAnn] = []
        for arg in func.args:
            match arg.type:
                case None | AnyTypeAnn():
                    arg_ty = self.gensym.fresh()
                    self.tvars.add(arg_ty)
                case _:
                    arg_ty = arg.type

            arg_tys.append(arg_ty)
            if isinstance(arg.name, NamedId):
                tctx[arg.name] = arg_ty

        self._visit_block(func.body, (tctx, rctx))

        # must be a return type
        if self.ret_type is None:
            raise RuntimeError(f'function unexpectedly has no return type: {func}')

        return FuncTypeAnn(rctx, arg_tys, self.ret_type, None)

    # override for typing hint
    def _visit_statement(self, stmt: Stmt, trctx: _TRCtx) -> _TRCtx:
        return super()._visit_statement(stmt, trctx)

    # override for typing hint
    def _visit_expr(self, expr: Expr, trctx: _TRCtx) -> _Type:
        return super()._visit_expr(expr, trctx)


class TypeCheck:
    """
    Type checker for the FPy language.

    Unlike Python, FPy is statically typed.

    When the `@fpy` decorator runs, it also type checks the function
    and raises an error if the function is not well-typed.
    """

    @staticmethod
    def check(func):
        """
        Analyzes the function for type errors.

        Produces a type signature for the function if it is well-typed.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got {func}')

        def_use = DefineUse.analyze(func)
        ty = _TypeCheckInstance(func, def_use).analyze()
        print(ty)

