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

_LETTERS = 'abcdefghijklmnopqrstuvwxyz'


class _TypeCheckInstance(Visitor):
    """Single-use instance of type checking."""
    func: FuncDef
    def_use: DefineUseAnalysis
    types: dict[Definition, _Type]
    ret_type: Optional[_Type]
    tvars: Unionfind[_Type]
    rvars: Unionfind[_RCtx]
    gensym: Gensym

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        self.func = func
        self.def_use = def_use
        self.types = {}
        self.ret_type = None
        self.tvars = Unionfind()
        self.rvars = Unionfind()
        self.gensym = Gensym()

    def analyze(self):
        return self._visit_function(self.func, None)

    def _fresh_type_var(self) -> NamedId:
        """Generates a fresh type variable."""
        tvar = self.gensym.fresh('t')
        self.tvars.add(tvar)
        return tvar

    def _fresh_context_var(self) -> NamedId:
        """Generates a fresh context variable."""
        rvar = self.gensym.fresh('r')
        self.rvars.add(rvar)
        return rvar

    def _generalized_type_var(self, counter: int) -> NamedId:
        """Generates a type variable during generalization."""
        assert counter >= 0
        quo, rem = divmod(counter, 26)
        suffix = '' if quo == 0 else str(quo)
        return NamedId(f't{_LETTERS[rem]}{suffix}')

    def _generalized_context_var(self, counter: int) -> NamedId:
        """Generates a context variable during generalization."""
        assert counter >= 0
        quo, rem = divmod(counter, 26)
        suffix = '' if quo == 0 else str(quo)
        return NamedId(f'r{_LETTERS[rem]}{suffix}')

    def _instantiate_context_var(
        self,
        ctx: _RCtx, *,
        fresh: Optional[dict[NamedId, NamedId]] = None
    ) -> _RCtx:
        """Instantiates a rounding context with fresh context variables."""
        if fresh is None:
            fresh = {}

        match ctx:
            case NamedId():
                if ctx not in fresh:
                    fresh[ctx] = self._fresh_context_var()
                return fresh[ctx]
            case Context():
                return ctx
            case _:
                raise RuntimeError(f'unreachable rounding context: {ctx}')

    def _instantiate_type_var(
        self,
        ty: _Type, *,
        fresh: Optional[dict[NamedId, NamedId]] = None
    ) -> _Type:
        """Instantiates a type with fresh type variables."""
        if fresh is None:
            fresh = {}

        match ty:
            case NamedId():
                if ty not in fresh:
                    fresh[ty] = self._fresh_type_var()
                return fresh[ty]
            case BoolTypeAnn():
                return BoolTypeAnn(None)
            case RealTypeAnn():
                ctx = None if ty.ctx is None else self._instantiate_context_var(ty.ctx, fresh=fresh)
                return RealTypeAnn(ctx, None)
            case FuncTypeAnn():
                rctx = self._instantiate_context_var(ty.ctx, fresh=fresh)
                args = [self._instantiate_type_var(arg, fresh=fresh) for arg in ty.args]
                ret = self._instantiate_type_var(ty.ret, fresh=fresh)
                return FuncTypeAnn(rctx, args, ret, None)
            case _:
                raise RuntimeError(f'unreachable type: {ty}')

    def _generalize_context(
        self,
        ctx: _RCtx, *,
        counters: Optional[list[int]] = None, # mutable_tuple[int, int]
        rename: Optional[dict[NamedId, NamedId]] = None,
    ) -> _RCtx:
        """Generalizes a rounding context, replacing context variables with canonical ones."""
        if counters is None:
            counters = [0, 0]
        if rename is None:
            rename = {}

        match ctx:
            case NamedId():
                if ctx not in rename:
                    rename[ctx] = self._generalized_context_var(counters[1])
                    counters[1] += 1
                return rename[ctx]
            case Context():
                return ctx
            case _:
                raise RuntimeError(f'unreachable rounding context: {ctx}')

    def _generalize_type(
        self,
        ty: _Type, *,
        counters: Optional[list[int]] = None, # mutable_tuple[int, int]
        rename: Optional[dict[NamedId, NamedId]] = None,
    ) -> _Type:
        """Generalizes a type, replacing type variables with canonical ones."""
        if counters is None:
            counters = [0, 0]
        if rename is None:
            rename = {}

        ty = self.tvars.find(ty)
        match ty:
            case NamedId():
                if ty not in rename:
                    rename[ty] = self._generalized_type_var(counters[0])
                    counters[0] += 1
                return rename[ty]
            case BoolTypeAnn():
                return BoolTypeAnn(None)
            case RealTypeAnn():
                ctx = None if ty.ctx is None else self._generalize_context(ty.ctx, counters=counters, rename=rename)
                return RealTypeAnn(ctx, None)
            case FuncTypeAnn():
                rctx = self._generalize_context(ty.ctx, counters=counters, rename=rename)
                args = [self._generalize_type(arg, counters=counters, rename=rename) for arg in ty.args]
                ret = self._generalize_type(ty.ret, counters=counters, rename=rename)
                return FuncTypeAnn(rctx, args, ret, None)
            case _:
                raise RuntimeError(f'unreachable type: {ty}')

    def _unify_contexts(self, a: _RCtx, b: _RCtx) -> _RCtx:
        """Unifies two rounding contexts, returning the most general unifier."""
        match a, b:
            case NamedId(), NamedId():
                return self.rvars.union(a, b)
            case NamedId(), _:
                # if `a` is a context variable, unify it with `b`
                # TODO: unify
                raise NotImplementedError
            case _, NamedId():
                # if `b` is a context variable, unify it with `a`
                # TODO: unify
                raise NotImplementedError
            case Context(), Context():
                # if both are contexts, unify them
                # TODO: context equality
                if a != b:
                    raise ValueError(f'cannot unify contexts: a={a} and b={b}')
                return a
            case _:
                raise RuntimeError(f'unreachable a={a}, b={b}')

    def _unify_types(self, a: _Type, b: _Type) -> _Type:
        """Unifies two types, returning the most general unifier."""
        match a, b:
            case NamedId(), NamedId():
                # TODO: unify
                return a
            case NamedId(), _:
                # if `a` is a type variable, unify it with `b`
                # TODO: unify
                raise NotImplementedError
            case _, NamedId():
                # if `b` is a type variable, unify it with `a`
                # TODO: unify
                raise NotImplementedError
            case BoolTypeAnn(), BoolTypeAnn():
                # always equal
                return a
            case RealTypeAnn(), RealTypeAnn():
                # might need to unify the rounding context
                match a.ctx, b.ctx:
                    case None, _:
                        # if `a` has no context, use `b`'s context
                        return b
                    case _, None:
                        # if `b` has no context, use `a`'s context
                        return a
                    case _:
                        # both have contexts, unify them
                        ctx = self._unify_contexts(a.ctx, b.ctx)
                        ty = self.tvars.add(RealTypeAnn(ctx, None))
                        self.tvars.union(ty, a)
                        self.tvars.union(ty, b)
                        return ty
            case _:
                raise ValueError(f'cannot unify types: a={a} and b={b}')

    def _resolve_context(self, ctx: _RCtx) -> _RCtx:
        """Resolves a rounding context to its representative."""
        return self.rvars.find(ctx)

    def _resolve_type(self, ty: _Type) -> _Type:
        """Resolves a type variable to its representative."""
        ty = self.tvars.find(ty)
        match ty:
            case NamedId() | BoolTypeAnn():
                return ty
            case RealTypeAnn():
                # resolve the rounding context
                ctx = None if ty.ctx is None else self._resolve_context(ty.ctx)
                return RealTypeAnn(ctx, None)
            case _:
                raise RuntimeError(f'unexpected type: {ty}')

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
        fun_ty: _Type = _binary_table[cls]

        # instantiate the function type with fresh type variables
        fun_ty = self._instantiate_type_var(fun_ty)
        assert isinstance(fun_ty, FuncTypeAnn)

        # compute the types of the arguments
        arg1_ty = self._visit_expr(e.first, trctx)
        arg2_ty = self._visit_expr(e.second, trctx)

        # unify the types
        _, rctx = trctx
        self._unify_contexts(rctx, fun_ty.ctx)
        self._unify_types(fun_ty.args[0], arg1_ty)
        self._unify_types(fun_ty.args[1], arg2_ty)

        # return type is valid
        # TODO: apply unionfind?
        return fun_ty.ret

    def _visit_ternaryop(self, e: TernaryOp, trctx: _TRCtx):
        raise NotImplementedError

    def _visit_naryop(self, e: NaryOp, trctx: _TRCtx):
        raise NotImplementedError

    def _visit_compare(self, e: Compare, trctx: _TRCtx):
        for arg in e.args:
            arg_ty = self._visit_expr(arg, trctx)
            self._unify_types(arg_ty, RealTypeAnn(None, None))
        return BoolTypeAnn(None)

    def _visit_call(self, e: Call, trctx: _TRCtx):
        raise NotImplementedError

    def _visit_tuple_expr(self, e: TupleExpr, trctx: _TRCtx):
        elt_tys = [self._visit_expr(elts, trctx) for elts in e.args]
        return TupleTypeAnn(elt_tys, None)

    def _visit_comp_expr(self, e: CompExpr, trctx: _TRCtx):
        raise NotImplementedError

    def _visit_tuple_ref(self, e: TupleRef, trctx: _TRCtx):
        raise NotImplementedError

    def _visit_tuple_set(self, e: TupleSet, trctx: _TRCtx):
        raise NotImplementedError

    def _visit_if_expr(self, e: IfExpr, trctx: _TRCtx):
        cond_ty = self._visit_expr(e.cond, trctx)
        cond_ty = self._unify_types(cond_ty, BoolTypeAnn(None))
        ift_ty = self._visit_expr(e.ift, trctx)
        iff_ty = self._visit_expr(e.iff, trctx)
        return self._unify_types(ift_ty, iff_ty)

    def _visit_context_expr(self, e: ContextExpr, trctx: _TRCtx):
        raise NotImplementedError

    def _visit_assign(self, stmt: Assign, trctx: _TRCtx):
        # evaluate the expression to get its type
        ty = self._visit_expr(stmt.expr, trctx)
        if stmt.type is not None:
            ty = self._unify_types(stmt.type, ty)

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
            self.ret_type = self._unify_types(self.ret_type, ty)
        # dummy return value
        return ({}, trctx[1])

    def _visit_block(self, block: StmtBlock, trctx: _TRCtx):
        for stmt in block.stmts:
            trctx = self._visit_statement(stmt, trctx)
        return trctx

    def _visit_function(self, func: FuncDef, _: None):
        tctx: _TCtx = {}
        rctx: _RCtx = self._fresh_context_var()

        arg_tys: list[_Type] = []
        for arg in func.args:
            match arg.type:
                case None | AnyTypeAnn():
                    arg_ty: _Type = self._fresh_type_var()
                case _:
                    arg_ty = arg.type

            arg_tys.append(arg_ty)
            if isinstance(arg.name, NamedId):
                tctx[arg.name] = arg_ty

        self._visit_block(func.body, (tctx, rctx))

        # must be a return type
        if self.ret_type is None:
            raise RuntimeError(f'function unexpectedly has no return type: {func}')

        # resolve types
        rctx = self._resolve_context(rctx)
        arg_tys = [self._resolve_type(arg_ty) for arg_ty in arg_tys]
        self.ret_type = self._resolve_type(self.ret_type)

        # generalize the return type
        fun_ty = FuncTypeAnn(rctx, arg_tys, self.ret_type, None)
        self.tvars.add(fun_ty)
        return self._generalize_type(fun_ty)

    # override for typing hint
    def _visit_statement(self, stmt: Stmt, trctx: _TRCtx) -> _TRCtx:
        return super()._visit_statement(stmt, trctx)

    def _visit_expr(self, expr: Expr, trctx: _TRCtx) -> _Type:
        ty = super()._visit_expr(expr, trctx)
        return self.tvars.add(ty)


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
        inst = _TypeCheckInstance(func, def_use)
        ty = inst.analyze()
        print(ty)

