"""
Type checking for FPy programs.

FPy has a simple type system:

    t ::= bool
        | real
        | t1 x t2
        | list t

"""

from typing import TypeAlias, Sequence

from ..ast import *
from ..function import Function
from ..primitive import Primitive
from ..utils import NamedId, default_repr
from .define_use import DefineUse, DefineUseAnalysis, Definition
from .live_vars import LiveVars

#####################################################################
# Abstract type

@default_repr
class Type:
    """Base class for all FPy types."""
    pass

_Type: TypeAlias = Type | NamedId
"""type: either a type annotation or type variable"""

_TCtx: TypeAlias = dict[NamedId, _Type]
"""typing context: mapping from variable to type"""

#####################################################################
# Types

class NullType(Type):
    """Placeholder for an ill-typed value."""

    def __eq__(self, other):
        return isinstance(other, NullType)

    def __hash__(self):
        return hash(type(self))

class BoolType(Type):
    """Type of boolean values"""

    def __eq__(self, other):
        return isinstance(other, BoolType)

    def __hash__(self):
        return hash(type(self))

class RealType(Type):
    """Real number type."""

    def __eq__(self, other):
        return isinstance(other, RealType)

    def __hash__(self):
        return hash(type(self))

class TupleType(Type):
    """Tuple type."""

    elt_types: tuple[_Type, ...]
    """type of elements"""

    def __init__(self, *elts: _Type):
        self.elt_types = elts

    def __eq__(self, other):
        return isinstance(other, TupleType) and self.elt_types == other.elt_types

    def __hash__(self):
        return hash(self.elt_types)

class ListType(Type):
    """List type."""

    elt_type: _Type
    """element type"""

    def __init__(self, elt: _Type):
        self.elt_type = elt

    def __eq__(self, other):
        return isinstance(other, ListType) and self.elt_type == other.elt_type

    def __hash__(self):
        return hash(self.elt_type)

class FunctionType(Type):
    """Function type."""

    arg_types: tuple[_Type, ...]
    """argument types"""

    return_type: _Type
    """return type"""

    def __init__(self, arg_types: Sequence[_Type], return_type: _Type):
        self.arg_types = tuple(arg_types)
        self.return_type = return_type

    def __eq__(self, other):
        return isinstance(other, FunctionType) and self.arg_types == other.arg_types and self.return_type == other.return_type

    def __hash__(self):
        return hash((self.arg_types, self.return_type))


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
    types: dict[Definition, _Type]

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        self.func = func
        self.def_use = def_use
        self.types = {}

    def analyze(self) -> FunctionType:
        return self._visit_function(self.func, None)

    def _unify(self, a_ty: _Type, b_ty: _Type):
        # TODO: implement
        pass

    def _annotation_to_type(self, ty: TypeAnn | None) -> _Type:
        match ty:
            case RealTypeAnn():
                return RealType()
            case _:
                raise NotImplementedError(ty)

    def _visit_var(self, e: Var, ctx: None) -> _Type:
        d = self.def_use.find_def_from_use(e)
        return self.types[d]

    def _visit_bool(self, e: BoolVal, ctx: None) -> BoolType:
        return BoolType()

    def _visit_foreign(self, e: ForeignVal, ctx: None) -> _Type:
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

    def _visit_nullaryop(self, e: NullaryOp, ctx: None) -> _Type:
        cls = type(e)
        if cls in _nullary_table:
            fn_ty = _nullary_table[cls]
            return fn_ty.return_type
        else:
            raise ValueError(f'unknown nullary operator: {cls}')

    def _visit_unaryop(self, e: UnaryOp, ctx: None) -> _Type:
        cls = type(e)
        arg_ty = self._visit_expr(e.arg, None)
        if cls in _unary_table:
            fn_ty = _unary_table[cls]
            self._unify(fn_ty.arg_types[0], arg_ty)
            return fn_ty.return_type
        else:
            raise ValueError(f'unknown unary operator: {cls}')

    def _visit_binaryop(self, e: BinaryOp, ctx: None) -> _Type:
        cls = type(e)
        lhs_ty = self._visit_expr(e.first, None)
        rhs_ty = self._visit_expr(e.second, None)
        if cls in _binary_table:
            fn_ty = _binary_table[cls]
            self._unify(fn_ty.arg_types[0], lhs_ty)
            self._unify(fn_ty.arg_types[1], rhs_ty)
            return fn_ty.return_type
        else:
            raise ValueError(f'unknown binary operator: {cls}')

    def _visit_ternaryop(self, e: TernaryOp, ctx: None) -> _Type:
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

    def _visit_naryop(self, e: NaryOp, ctx: None) -> _Type:
        raise NotImplementedError

    def _visit_compare(self, e: Compare, ctx: None) -> BoolType:
        for arg in e.args:
            ty = self._visit_expr(arg, None)
            self._unify(ty, BoolType())
        return BoolType()

    def _visit_call(self, e: Call, ctx: None) -> _Type:
        match e.fn:
            case Primitive():
                for arg, expect_ty in zip(e.args, e.fn.arg_types):
                    ty = self._visit_expr(arg, None)
                    self._unify(ty, expect_ty)
                return self._annotation_to_type(e.fn.return_type)
            case _:
                raise NotImplementedError(f'cannot type check {e.fn}')

    def _visit_tuple_expr(self, e: TupleExpr, ctx: None) -> TupleType:
        elt_tys = [self._visit_expr(arg, None) for arg in e.args]
        return TupleType(*elt_tys)

    def _visit_list_expr(self, e: ListExpr, ctx: None) -> ListType:
        raise NotImplementedError

    def _visit_list_comp(self, e: ListComp, ctx: None) -> ListType:
        raise NotImplementedError

    def _visit_list_ref(self, e: ListRef, ctx: None) -> _Type:
        raise NotImplementedError

    def _visit_list_slice(self, e: ListSlice, ctx: None) -> ListType:
        raise NotImplementedError

    def _visit_list_set(self, e: ListSet, ctx: None) -> _Type:
        raise NotImplementedError

    def _visit_if_expr(self, e: IfExpr, ctx: None) -> _Type:
        raise NotImplementedError

    def _visit_context_expr(self, e: ContextExpr, ctx: None) -> _Type:
        raise NotImplementedError

    def _visit_assign(self, stmt: Assign, ctx: None):
        ty = self._visit_expr(stmt.expr, None)
        if isinstance(stmt.binding, NamedId):
            d = self.def_use.find_def_from_site(stmt.binding, stmt)
            self.types[d] = ty

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: None):
        raise NotImplementedError

    def _visit_if1(self, stmt: If1Stmt, ctx: None):
        raise NotImplementedError

    def _visit_if(self, stmt: IfStmt, ctx: None):
        self._visit_expr(stmt.cond, None)
        self._visit_block(stmt.ift, None)
        self._visit_block(stmt.iff, None)
        # TODO: merge variables

    def _visit_while(self, stmt: WhileStmt, ctx: None):
        raise NotImplementedError

    def _visit_for(self, stmt: ForStmt, ctx: None):
        raise NotImplementedError

    def _visit_context(self, stmt: ContextStmt, ctx: None):
        raise NotImplementedError

    def _visit_assert(self, stmt: AssertStmt, ctx: None):
        raise NotImplementedError

    def _visit_effect(self, stmt: EffectStmt, ctx: None):
        raise NotImplementedError

    def _visit_return(self, stmt: ReturnStmt, ctx: None):
        raise NotImplementedError

    def _visit_block(self, block: StmtBlock, ctx: None):
        for stmt in block.stmts:
            self._visit_statement(stmt, None)

    def _visit_function(self, func: FuncDef, ctx: None) -> FunctionType:
        # infer types from annotations
        arg_tys: list[_Type] = []
        for arg in func.args:
            arg_ty = self._annotation_to_type(arg.type)
            if isinstance(arg.name, NamedId):
                d = self.def_use.find_def_from_site(arg.name, arg)
                self.types[d] = arg_ty
            arg_tys.append(arg_ty)

        self._visit_block(func.body, None)
        raise NotImplementedError



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
