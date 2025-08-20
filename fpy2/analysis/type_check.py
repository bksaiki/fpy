"""
Type checking for FPy programs.
"""

from dataclasses import dataclass
from typing import cast

from ..ast import *
from ..primitive import Primitive
from ..types import Type, NullType, BoolType, RealType, VarType, FunctionType, TupleType, ListType
from ..utils import Gensym, NamedId, Unionfind

from .define_use import DefineUse, DefineUseAnalysis, Definition, DefSite

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

class FPyTypeError(Exception):
    """Type error for FPy programs."""
    pass

@dataclass(frozen=True)
class TypeAnalysis:
    fn_type: FunctionType
    by_def: dict[Definition, Type]
    by_expr: dict[Expr, Type]


class _TypeCheckInstance(Visitor):
    """Single-use instance of type checking."""

    func: FuncDef
    def_use: DefineUseAnalysis
    by_def: dict[Definition, Type]
    by_expr: dict[Expr, Type]
    ret_type: Type | None
    tvars: Unionfind[Type]
    gensym: Gensym

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        self.func = func
        self.def_use = def_use
        self.by_def = {}
        self.by_expr = {}
        self.ret_type = None
        self.tvars = Unionfind()
        self.gensym = Gensym()

    def analyze(self) -> TypeAnalysis:
        ty = self._visit_function(self.func, None)
        by_defs = {
            name: self._resolve_type(ty)
            for name, ty in self.by_def.items()
        }
        by_expr = {
            e: self._resolve_type(ty)
            for e, ty in self.by_expr.items()
        }
        return TypeAnalysis(ty, by_defs, by_expr)

    def _set_type(self, site: Definition, ty: Type):
        self.by_def[site] = ty

    def _fresh_type_var(self) -> VarType:
        """Generates a fresh type variable."""
        ty = VarType(self.gensym.fresh('t'))
        self.tvars.add(ty)
        return ty

    def _resolve_type(self, ty: Type):
        match ty:
            case NullType() | BoolType() | RealType() | VarType():
                return self.tvars.get(ty, ty)
            case TupleType():
                elts = [self._resolve_type(elt) for elt in ty.elt_types]
                return self.tvars.add(TupleType(*elts))
            case ListType():
                elt_ty = self._resolve_type(ty.elt_type)
                return self.tvars.add(ListType(elt_ty))
            case _:
                raise NotImplementedError(f'cannot resolve type {ty}')

    def _unify(self, a_ty: Type, b_ty: Type):
        a_ty = self.tvars.get(a_ty, a_ty)
        b_ty = self.tvars.get(b_ty, b_ty)
        match a_ty, b_ty:
            case _, VarType():
                a_ty = self.tvars.add(a_ty)
                return self.tvars.union(a_ty, b_ty)
            case VarType(), _:
                b_ty = self.tvars.add(b_ty)
                return self.tvars.union(b_ty, a_ty)
            case RealType(), RealType():
                return a_ty
            case BoolType(), BoolType():
                return b_ty
            case ListType(), ListType():
                elt_ty = self._unify(a_ty.elt_type, b_ty.elt_type)
                elt_ty = self.tvars.add(elt_ty)
                elt_ty = self.tvars.union(elt_ty, self.tvars.add(a_ty.elt_type))
                elt_ty = self.tvars.union(elt_ty, self.tvars.add(b_ty.elt_type))
                return self.tvars.add(ListType(elt_ty))
            case TupleType(), TupleType():
                # TODO: what if the length doesn't match
                if len(a_ty.elt_types) != len(b_ty.elt_types):
                    raise FPyTypeError(f'attempting to unify `{a_ty.format()}` and `{b_ty.format()}`')
                elts = [self._unify(a_elt, b_elt) for a_elt, b_elt in zip(a_ty.elt_types, b_ty.elt_types)]
                ty = self.tvars.add(TupleType(*elts))
                ty = self.tvars.union(ty, self.tvars.add(a_ty))
                ty = self.tvars.union(ty, self.tvars.add(b_ty))
                return ty
            case NullType(), NullType():
                return a_ty
            case _:
                raise FPyTypeError(f'attempting to unify `{a_ty.format()}` and `{b_ty.format()}`')

    def _instantiate(self, ty: Type) -> Type:
        subst: dict[VarType, Type] = {}
        for fv in sorted(ty.free_vars()):
            subst[fv] = self._fresh_type_var()
        return ty.subst(subst)

    def _generalize(self, ty: Type) -> Type:
        subst: dict[VarType, Type] = {}
        for i, fv in enumerate(sorted(ty.free_vars())):
            t = self.tvars.find(fv)
            match t: 
                case VarType():
                    subst[fv] = VarType(NamedId(f't{i + 1}'))
                case _:
                    subst[fv] = t
        return ty.subst(subst)

    def _annotation_to_type(self, ty: TypeAnn | None) -> Type:
        match ty:
            case None | AnyTypeAnn():
                return self._fresh_type_var()
            case BoolTypeAnn():
                # boolean type
                return BoolType()
            case RealTypeAnn():
                return RealType()
            case TupleTypeAnn():
                # tuple type
                elt_tys = [self._annotation_to_type(elt) for elt in ty.elts]
                return TupleType(*elt_tys)
            case ListTypeAnn():
                # list type
                return ListType(self._annotation_to_type(ty.elt))
            case SizedTensorTypeAnn():
                if len(ty.dims) == 0:
                    return ListType(self._fresh_type_var())
                else:
                    arr_ty = ListType(self._annotation_to_type(ty.elt))
                    for _ in ty.dims[1:]:
                        arr_ty = ListType(arr_ty)
                    return arr_ty
            case _:
                raise NotImplementedError(ty)

    def _visit_var(self, e: Var, ctx: None) -> Type:
        d = self.def_use.find_def_from_use(e)
        return self.by_def[d]

    def _visit_bool(self, e: BoolVal, ctx: None) -> BoolType:
        return BoolType()

    def _visit_foreign(self, e: ForeignVal, ctx: None) -> Type:
        return self._fresh_type_var()

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
                case Empty():
                    # arg : real
                    self._unify(arg_ty, RealType())
                    # result is list[A]
                    ty = self._fresh_type_var()
                    return ListType(ty)
                case Dim():
                    # dimension operator
                    self._unify(arg_ty, ListType(self._fresh_type_var()))
                    return RealType()
                case Enumerate():
                    # enumerate operator
                    ty = self._fresh_type_var()
                    self._unify(arg_ty, ListType(ty))
                    return ListType(TupleType(RealType(), ty))
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
                    self._unify(lhs_ty, ListType(self._fresh_type_var()))
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
                    ty = self._fresh_type_var()
                    arg_ty = self._visit_expr(arg, None)
                    self._unify(arg_ty, ListType(ty))
                    arg_tys.append(ty)
                return ListType(TupleType(*arg_tys))
            case _:
                raise ValueError(f'unknown n-ary operator: {type(e)}')

    def _visit_compare(self, e: Compare, ctx: None) -> BoolType:
        for arg in e.args:
            ty = self._visit_expr(arg, None)
            self._unify(ty, RealType())
        return BoolType()

    def _visit_call(self, e: Call, ctx: None) -> Type:
        # get around circular imports
        from ..function import Function

        match e.fn:
            case None:
                # unbound call
                return self._fresh_type_var()
            case Primitive():
                # calling a primitive
                for arg, ann in zip(e.args, e.fn.arg_types):
                    ty = self._visit_expr(arg, None)
                    self._unify(ty, self._annotation_to_type(ann))
                return self._annotation_to_type(e.fn.return_type)
            case Function():
                # calling a function
                if e.fn.sig is None:
                    # type checking not run
                    # TODO: guard against recursion
                    fn_info = TypeCheck.check(e.fn.ast)
                    fn_ty = fn_info.fn_type
                else:
                    fn_ty = e.fn.sig

                if len(fn_ty.arg_types) != len(e.args):
                    # no function signature / signature mismatch
                    return NullType()
                else:
                    # signature matches
                    # instantiate the function type
                    fn_ty = cast(FunctionType, self._instantiate(fn_ty))
                    # merge arguments
                    for arg, expect_ty in zip(e.args, fn_ty.arg_types):
                        ty = self._visit_expr(arg, None)
                        self._unify(ty, expect_ty)
                    return fn_ty.return_type
            case _:
                raise NotImplementedError(f'cannot type check {e.fn} {e.func}')

    def _visit_tuple_expr(self, e: TupleExpr, ctx: None) -> TupleType:
        elt_tys = [self._visit_expr(arg, None) for arg in e.args]
        return TupleType(*elt_tys)

    def _visit_list_expr(self, e: ListExpr, ctx: None) -> ListType:
        arg_tys = [self._visit_expr(arg, None) for arg in e.args]
        if len(arg_tys) == 0:
            # empty list
            return ListType(self._fresh_type_var())
        else:
            elt_ty = arg_tys[0]
            for arg_ty in arg_tys[1:]:
                elt_ty = self._unify(elt_ty, arg_ty)
            ty = ListType(elt_ty)
            return ty

    def _visit_binding(self, site: DefSite, binding: Id | TupleBinding, ty: Type):
        match binding:
            case NamedId():
                d = self.def_use.find_def_from_site(binding, site)
                self._set_type(d, ty)
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
        # val : list[A]
        value_ty = self._visit_expr(e.value, None)
        ty = self._fresh_type_var()
        self._unify(value_ty, ListType(ty))
        # index : real
        index_ty = self._visit_expr(e.index, None)
        self._unify(index_ty, RealType())
        # val[index] : A
        return ty

    def _visit_list_slice(self, e: ListSlice, ctx: None):
        # type check array
        value_ty = self._visit_expr(e.value, None)
        self._unify(value_ty, ListType(self._fresh_type_var()))
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
        arr_ty = self._visit_expr(e.array, None)

        iter_ty = arr_ty
        for s in e.slices:
            ty = self._visit_expr(s, None)
            elt_ty = self._fresh_type_var()
            self._unify(arr_ty, ListType(elt_ty))
            self._unify(ty, RealType())
            iter_ty = elt_ty

        val_ty = self._visit_expr(e.value, None)
        self._unify(val_ty, iter_ty)
        return arr_ty

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
        d = self.def_use.find_def_from_use(stmt)
        arr_ty = self.by_def[d]

        for s in stmt.slices:
            # arr : list[A]
            elt_ty = self._fresh_type_var()
            self._unify(arr_ty, ListType(elt_ty))
            # s : real
            ty = self._visit_expr(s, None)
            self._unify(ty, RealType())
            # arr [idx] : A
            arr_ty = elt_ty

        # val : A
        val_ty = self._visit_expr(stmt.expr, None)
        self._unify(val_ty, arr_ty)

    def _visit_if1(self, stmt: If1Stmt, ctx: None):
        # type check condition
        cond_ty = self._visit_expr(stmt.cond, None)
        self._unify(cond_ty, BoolType())
        # type check body
        self._visit_block(stmt.body, None)

    def _select_def_repr(self, d: Definition):
        return d.assigns()[0]

    def _visit_if(self, stmt: IfStmt, ctx: None):
        # type check condition
        cond_ty = self._visit_expr(stmt.cond, None)
        self._unify(cond_ty, BoolType())
        # type check branches
        self._visit_block(stmt.ift, None)
        self._visit_block(stmt.iff, None)

        # need to merge variables introduced on both sides
        defs_in_ift, defs_out_ift = self.def_use.blocks[stmt.ift]
        defs_in_iff, defs_out_iff = self.def_use.blocks[stmt.iff]
        intros_ift = defs_in_ift.fresh_in(defs_out_ift)
        intros_iff = defs_in_iff.fresh_in(defs_out_iff)
        for intro in intros_ift & intros_iff:
            ift_def = self._select_def_repr(defs_out_ift[intro])
            iff_def = self._select_def_repr(defs_out_iff[intro])
            self._unify(self.by_def[ift_def], self.by_def[iff_def])

    def _visit_while(self, stmt: WhileStmt, ctx: None):
        cond_ty = self._visit_expr(stmt.cond, None)
        self._unify(cond_ty, BoolType())
        # type check body
        self._visit_block(stmt.body, None)

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
                self._set_type(d, arg_ty)
            arg_tys.append(arg_ty)

        # generate free variables types
        for v in func.free_vars:
            d = self.def_use.find_def_from_site(v, func)
            self._set_type(d, self._fresh_type_var())

        # type check body
        self._visit_block(func.body, None)
        if self.ret_type is None:
            raise FPyTypeError(f'function {func.name} has no return type')

        # generalize the function type
        ty = FunctionType(arg_tys, self.ret_type)
        return cast(FunctionType, self._generalize(ty))

    def _visit_expr(self, expr: Expr, ctx: None) -> Type:
        ret_ty = super()._visit_expr(expr, ctx)
        self.by_expr[expr] = ret_ty
        return ret_ty


class TypeCheck:
    """
    Type inference for the FPy language.

    FPy is not statically typed, but compilation may require statically
    determining the types throughout the program.
    The FPy type inference algorithm is a Hindley-Milner based algorithm.
    """

    #
    # <type> ::= bool
    #          | real
    #          | <var>
    #          | <type> x <type>
    #          | list <type>
    #          | <type> -> <type>
    #

    @staticmethod
    def check(func: FuncDef) -> TypeAnalysis:
        """
        Analyzes the function for type errors.

        Produces a type signature for the function if it is well-typed
        and a mapping from definition to type.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got {func}')

        def_use = DefineUse.analyze(func)
        inst = _TypeCheckInstance(func, def_use)
        return inst.analyze()

