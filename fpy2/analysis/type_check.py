"""
Type checking for FPy programs.

FPy has a simple type system:

    t ::= bool
        | real
        | t1 x t2
        | list t

"""

from ..ast import *
from ..function import Function
from ..primitive import Primitive
from ..types import Type, NullType, BoolType, RealType, VarType, FunctionType, TupleType, ListType
from ..utils import NamedId

from .define_use import DefineUse, DefineUseAnalysis, Definition, DefSite
from .live_vars import LiveVars

#####################################################################
# Type Inference

_Bool1ary = FunctionType([BoolType()], BoolType())
_Real0ary = FunctionType([], RealType())
_Real1ary = FunctionType([RealType()], RealType())
_Real2ary = FunctionType([RealType(), RealType()], RealType())
_Real3ary = FunctionType([RealType(), RealType(), RealType()], RealType())
_Predicate = FunctionType([RealType()], BoolType())

_nullary_table: dict[type[NullaryOp], FunctionType] = {
    ConstNan: _Real0ary,
    ConstInf: _Real0ary,
    ConstPi: _Real0ary,
    ConstE: _Real0ary,
    ConstLog2E: _Real0ary,
    ConstLog10E: _Real0ary,
    ConstLn2: _Real0ary,
    ConstPi_2: _Real0ary,
    ConstPi_4: _Real0ary,
    Const1_Pi: _Real0ary,
    Const2_Pi: _Real0ary,
    Const2_SqrtPi: _Real0ary,
    ConstSqrt2: _Real0ary,
    ConstSqrt1_2: _Real0ary
}

_unary_table: dict[type[UnaryOp], FunctionType] = {
    Fabs: _Real1ary,
    Sqrt: _Real1ary,
    Neg: _Real1ary,
    Cbrt: _Real1ary,
    Ceil: _Real1ary,
    Floor: _Real1ary,
    NearbyInt: _Real1ary,
    RoundInt: _Real1ary,
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
    Round: _Real1ary,
    RoundExact: _Real1ary,
}

_binary_table: dict[type[BinaryOp], FunctionType] = {
    Add: _Real2ary,
    Sub: _Real2ary,
    Mul: _Real2ary,
    Div: _Real2ary,
    Copysign: _Real2ary,
    Fdim: _Real2ary,
    Fmod: _Real2ary,
    Remainder: _Real2ary,
    Hypot: _Real2ary,
    Atan2: _Real2ary,
    Pow: _Real2ary,
    RoundAt: _Real1ary,
}

_ternary_table: dict[type[TernaryOp], FunctionType] = {
    Fma: _Real3ary,
}


class _TypeCheckInstance(Visitor):
    """Single-use instance of type checking."""

    func: FuncDef
    def_use: DefineUseAnalysis
    types: dict[Definition, Type]
    ret_type: Type | None

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        self.func = func
        self.def_use = def_use
        self.types = {}
        self.ret_type = None

    def analyze(self) -> FunctionType:
        return self._visit_function(self.func, None)

    def _unify(self, a_ty: Type, b_ty: Type):
        # TODO: implement
        return a_ty

    def _annotation_to_type(self, ty: TypeAnn | None) -> Type:
        match ty:
            case AnyTypeAnn():
                # TODO: actually implement this
                return NullType()
            case RealTypeAnn():
                return RealType()
            case _:
                raise NotImplementedError(ty)

    def _visit_var(self, e: Var, ctx: None) -> Type:
        d = self.def_use.find_def_from_use(e)
        return self.types[d]

    def _visit_bool(self, e: BoolVal, ctx: None) -> BoolType:
        return BoolType()

    def _visit_foreign(self, e: ForeignVal, ctx: None) -> Type:
        raise NotImplementedError

    def _visit_decnum(self, e: Decnum, ctx: None) -> RealType:
        return RealType()

    def _visit_hexnum(self, e: Hexnum, ctx: None) -> RealType:
        return RealType()

    def _visit_integer(self, e: Integer, ctx: None) -> RealType:
        return RealType()

    def _visit_rational(self, e: Rational, ctx: None) -> RealType:
        return RealType()

    def _visit_digits(self, e: Digits, ctx: None) -> RealType:
        return RealType()

    def _visit_nullaryop(self, e: NullaryOp, ctx: None) -> Type:
        cls = type(e)
        if cls in _nullary_table:
            fn_ty = _nullary_table[cls]
            return fn_ty.return_type
        else:
            raise ValueError(f'unknown nullary operator: {cls}')

    def _visit_unaryop(self, e: UnaryOp, ctx: None) -> Type:
        cls = type(e)
        arg_ty = self._visit_expr(e.arg, None)
        if cls in _unary_table:
            fn_ty = _unary_table[cls]
            self._unify(fn_ty.arg_types[0], arg_ty)
            return fn_ty.return_type
        else:
            match e:
                case Sum():
                    # sum operator
                    self._unify(arg_ty, ListType(RealType()))
                    return RealType()
                case Range():
                    # range operator
                    self._unify(arg_ty, RealType())
                    return ListType(RealType())
                case Dim():
                    # dimension operator
                    # TODO: unify with list[A]
                    self._unify(arg_ty, ListType(NullType()))
                    return RealType()
                case Enumerate():
                    # enumerate operator
                    # TODO: unify with list[A]
                    self._unify(arg_ty, ListType(NullType()))
                    # TODO: produce list[tuple[Real, A]]
                    return ListType(TupleType(RealType(), NullType()))
                case _:
                    raise ValueError(f'unknown unary operator: {cls}')

    def _visit_binaryop(self, e: BinaryOp, ctx: None) -> Type:
        cls = type(e)
        lhs_ty = self._visit_expr(e.first, None)
        rhs_ty = self._visit_expr(e.second, None)
        if cls in _binary_table:
            fn_ty = _binary_table[cls]
            self._unify(fn_ty.arg_types[0], lhs_ty)
            self._unify(fn_ty.arg_types[1], rhs_ty)
            return fn_ty.return_type
        else:
            match e:
                case Size():
                    # size operator
                    # TODO: unify with list[A]
                    self._unify(lhs_ty, ListType(NullType()))
                    self._unify(rhs_ty, RealType())
                    return RealType()
                case _:
                    raise ValueError(f'unknown binary operator: {cls}')

    def _visit_ternaryop(self, e: TernaryOp, ctx: None) -> Type:
        cls = type(e)
        first = self._visit_expr(e.first, None)
        second = self._visit_expr(e.second, None)
        third = self._visit_expr(e.third, None)
        if cls in _ternary_table:
            fn_ty = _ternary_table[cls]
            self._unify(fn_ty.arg_types[0], first)
            self._unify(fn_ty.arg_types[1], second)
            self._unify(fn_ty.arg_types[2], third)
            return fn_ty.return_type
        else:
            raise ValueError(f'unknown ternary operator: {cls}')

    def _visit_naryop(self, e: NaryOp, ctx: None) -> Type:
        match e:
            case Min() | Max():
                for arg in e.args:
                    ty = self._visit_expr(arg, None)
                    self._unify(ty, RealType())
                return RealType()
            case And() | Or():
                for arg in e.args:
                    ty = self._visit_expr(arg, None)
                    self._unify(ty, BoolType())
                return BoolType()
            case Zip():
                arg_tys: list[Type] = []
                for arg in e.args:
                    arg_ty = self._visit_expr(arg, None)
                    # TODO: unify with list[A]
                    self._unify(arg_ty, ListType(NullType()))
                    # TODO: extract A
                    arg_tys.append(NullType())
                return ListType(TupleType(*arg_tys))
            case _:
                raise ValueError(f'unknown n-ary operator: {type(e)}')

    def _visit_compare(self, e: Compare, ctx: None) -> BoolType:
        for arg in e.args:
            ty = self._visit_expr(arg, None)
            self._unify(ty, BoolType())
        return BoolType()

    def _visit_call(self, e: Call, ctx: None) -> Type:
        match e.fn:
            case Primitive():
                for arg, ann in zip(e.args, e.fn.arg_types):
                    ty = self._visit_expr(arg, None)
                    self._unify(ty, self._annotation_to_type(ann))
                return self._annotation_to_type(e.fn.return_type)
            case Function():
                if e.fn.sig is None or len(e.fn.sig.arg_types) != len(e.args):
                    # no function signature / signature mismatch
                    return NullType()
                else:
                    # signature matches
                    for arg, expect_ty in zip(e.args, e.fn.sig.arg_types):
                        ty = self._visit_expr(arg, None)
                        self._unify(ty, expect_ty)
                    return e.fn.sig.return_type
            case _:
                raise NotImplementedError(f'cannot type check {e.fn}')

    def _visit_tuple_expr(self, e: TupleExpr, ctx: None) -> TupleType:
        elt_tys = [self._visit_expr(arg, None) for arg in e.args]
        return TupleType(*elt_tys)

    def _visit_list_expr(self, e: ListExpr, ctx: None) -> ListType:
        arg_tys = [self._visit_expr(arg, None) for arg in e.args]
        if len(arg_tys) == 0:
            # empty list
            # TODO: return list[A]
            return ListType(NullType())
        else:
            ty = arg_tys[0]
            for arg_ty in arg_tys[1:]:
                ty = self._unify(ty, arg_ty)
            return ListType(ty)

    def _visit_binding(self, site: DefSite, binding: Id | TupleBinding, ty: Type):
        match binding:
            case NamedId():
                d = self.def_use.find_def_from_site(binding, site)
                self.types[d] = ty
            case UnderscoreId():
                pass
            case TupleBinding():
                if isinstance(ty, TupleType) and len(binding.elts) == len(ty.elt_types):
                    # type has expected shape
                    for elt_ty, elt in zip(ty.elt_types, binding.elts):
                        self._visit_binding(site, elt, elt_ty)
                else:
                    # type does not have expected shape
                    for elt in binding.elts:
                        self._visit_binding(site, elt, NullType())
            case _:
                raise RuntimeError(f'unreachable: {binding}')

    def _visit_list_comp(self, e: ListComp, ctx: None) -> ListType:
        for target, iterable in zip(e.targets, e.iterables):
            iter_ty = self._visit_expr(iterable, None)
            match iter_ty:
                case ListType():
                    # expected type: list a
                    self._visit_binding(e, target, iter_ty.elt_type)
                case _:
                    # otherwise
                    self._visit_binding(e, target, NullType())

        elt_ty = self._visit_expr(e.elt, None)
        return ListType(elt_ty)

    def _visit_list_ref(self, e: ListRef, ctx: None) -> Type:
        # type check array
        value_ty = self._visit_expr(e.value, None)
        # TODO: unify with list[A]
        self._unify(value_ty, ListType(NullType()))
        # type check index
        index_ty = self._visit_expr(e.index, None)
        self._unify(index_ty, RealType())
        # TODO: return A
        return NullType()

    def _visit_list_slice(self, e: ListSlice, ctx: None) -> ListType:
        # type check array
        value_ty = self._visit_expr(e.value, None)
        # TODO: unify with list[A]
        self._unify(value_ty, ListType(NullType()))
        # type check endpoints
        if e.start is not None:
            start_ty = self._visit_expr(e.start, None)
            self._unify(start_ty, RealType())
        if e.stop is not None:
            stop_ty = self._visit_expr(e.stop, None)
            self._unify(stop_ty, RealType())
        # same type as value_ty
        return value_ty

    def _visit_list_set(self, e: ListSet, ctx: None) -> Type:
        # type check array
        value_ty = self._visit_expr(e.value, None)
        # TODO: unify with list[A]
        self._unify(value_ty, ListType(NullType()))
        # type check index
        index_ty = self._visit_expr(e.index, None)
        self._unify(index_ty, RealType())
        # same type as value_ty
        return value_ty


    def _visit_if_expr(self, e: IfExpr, ctx: None) -> Type:
        # type check condition
        cond_ty = self._visit_expr(e.cond, None)
        self._unify(cond_ty, BoolType())

        # type check branches
        ift_ty = self._visit_expr(e.ift, None)
        iff_ty = self._visit_expr(e.iff, None)
        return self._unify(ift_ty, iff_ty)

    def _visit_context_expr(self, e: ContextExpr, ctx: None) -> Type:
        raise NotImplementedError

    def _visit_assign(self, stmt: Assign, ctx: None):
        ty = self._visit_expr(stmt.expr, None)
        self._visit_binding(stmt, stmt.binding, ty)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: None):
        raise NotImplementedError

    def _visit_if1(self, stmt: If1Stmt, ctx: None):
        # type check condition
        cond_ty = self._visit_expr(stmt.cond, None)
        self._unify(cond_ty, BoolType())
        # type check body
        self._visit_block(stmt.body, None)
        # TODO: merge variables

    def _visit_if(self, stmt: IfStmt, ctx: None):
        # type check condition
        cond_ty = self._visit_expr(stmt.cond, None)
        self._unify(cond_ty, BoolType())
        # type check branches
        self._visit_block(stmt.ift, None)
        self._visit_block(stmt.iff, None)
        # TODO: merge variables

    def _visit_while(self, stmt: WhileStmt, ctx: None):
        raise NotImplementedError

    def _visit_for(self, stmt: ForStmt, ctx: None):
        # type check iterable
        iter_ty = self._visit_expr(stmt.iterable, None)
        match iter_ty:
            case ListType():
                # expected type: list a
                self._visit_binding(stmt, stmt.target, iter_ty.elt_type)
            case _:
                # otherwise
                self._visit_binding(stmt, stmt.target, NullType())

        # type check body
        self._visit_block(stmt.body, None)

    def _visit_context(self, stmt: ContextStmt, ctx: None):
        # TODO: type check context
        # type check body
        self._visit_block(stmt.body, None)

    def _visit_assert(self, stmt: AssertStmt, ctx: None):
        self._visit_expr(stmt.test, None)

    def _visit_effect(self, stmt: EffectStmt, ctx: None):
        self._visit_expr(stmt.expr, None)

    def _visit_return(self, stmt: ReturnStmt, ctx: None):
        self.ret_type = self._visit_expr(stmt.expr, None)

    def _visit_block(self, block: StmtBlock, ctx: None):
        for stmt in block.stmts:
            self._visit_statement(stmt, None)

    def _visit_function(self, func: FuncDef, ctx: None) -> FunctionType:
        # infer types from annotations
        arg_tys: list[Type] = []
        for arg in func.args:
            arg_ty = self._annotation_to_type(arg.type)
            if isinstance(arg.name, NamedId):
                d = self.def_use.find_def_from_site(arg.name, arg)
                self.types[d] = arg_ty
            arg_tys.append(arg_ty)

        self._visit_block(func.body, None)
        if self.ret_type is None:
            raise TypeError(f'function {func.name} has no return type')
        return FunctionType(arg_tys, self.ret_type)


class TypeCheck:
    """
    Type checker for the FPy language.

    Unlike Python, FPy is statically typed.

    When the `@fpy` decorator runs, it also type checks the function
    and raises an error if the function is not well-typed.
    """

    @staticmethod
    def check(func: FuncDef):
        """
        Analyzes the function for type errors.

        Produces a type signature for the function if it is well-typed.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got {func}')

        def_use = DefineUse.analyze(func)
        inst = _TypeCheckInstance(func, def_use)
        ty = inst.analyze()
        print(func.name, ty)
        return ty
