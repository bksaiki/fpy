"""
Type checking for FPy programs.
"""

from dataclasses import dataclass
from typing import Callable, cast

from ..ast import *
from ..primitive import Primitive
from ..utils import Gensym, NamedId, Unionfind

from .types import *
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


def _ann_to_type(ty: TypeAnn | None, fresh_var: Callable[[], VarType]) -> Type:
    match ty:
        case None | AnyTypeAnn():
            return fresh_var()
        case BoolTypeAnn():
            # boolean type
            return BoolType()
        case RealTypeAnn():
            return RealType()
        case TupleTypeAnn():
            # tuple type
            elt_tys = [_ann_to_type(elt, fresh_var) for elt in ty.elts]
            return TupleType(*elt_tys)
        case ListTypeAnn():
            # list type
            return ListType(_ann_to_type(ty.elt, fresh_var))
        case SizedTensorTypeAnn():
            if len(ty.dims) == 0:
                return ListType(fresh_var())
            else:
                arr_ty = ListType(_ann_to_type(ty.elt, fresh_var))
                for _ in ty.dims[1:]:
                    arr_ty = ListType(arr_ty)
                return arr_ty
        case _:
            raise RuntimeError(f'unreachable: {ty}')



class TypeInferError(Exception):
    """Type error for FPy programs."""
    pass

@dataclass(frozen=True)
class TypeAnalysis:
    fn_type: FunctionType
    by_def: dict[Definition, Type]
    by_expr: dict[Expr, Type]
    tvars: Unionfind[Type]

    @property
    def arg_types(self):
        return self.fn_type.arg_types

    @property
    def return_type(self):
        return self.fn_type.return_type


class _TypeInferInstance(Visitor):
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

    def _set_type(self, site: Definition, ty: Type):
        self.by_def[site] = ty

    def _fresh_type_var(self) -> VarType:
        """Generates a fresh type variable."""
        ty = VarType(self.gensym.fresh('t'))
        self.tvars.add(ty)
        return ty

    def _resolve_type(self, ty: Type):
        match ty:
            case VarType():
                ty = self.tvars.get(ty, ty)
                if isinstance(ty, VarType):
                    return ty
                else:
                    return self._resolve_type(ty)
            case BoolType() | RealType() | ContextType():
                return self.tvars.get(ty, ty)
            case TupleType():
                elts = [self._resolve_type(elt) for elt in ty.elts]
                return self.tvars.add(TupleType(*elts))
            case ListType():
                elt_ty = self._resolve_type(ty.elt)
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
            case (RealType(), RealType()) | (BoolType(), BoolType()) | (ContextType(), ContextType()):
                return a_ty
            case ListType(), ListType():
                elt_ty = self._unify(a_ty.elt, b_ty.elt)
                elt_ty = self.tvars.add(elt_ty)
                elt_ty = self.tvars.union(elt_ty, self.tvars.add(a_ty.elt))
                elt_ty = self.tvars.union(elt_ty, self.tvars.add(b_ty.elt))
                return self.tvars.add(ListType(elt_ty))
            case TupleType(), TupleType():
                # TODO: what if the length doesn't match
                if len(a_ty.elts) != len(b_ty.elts):
                    raise TypeInferError(f'attempting to unify `{a_ty.format()}` and `{b_ty.format()}`')
                elts = [self._unify(a_elt, b_elt) for a_elt, b_elt in zip(a_ty.elts, b_ty.elts)]
                ty = self.tvars.add(TupleType(*elts))
                ty = self.tvars.union(ty, self.tvars.add(a_ty))
                ty = self.tvars.union(ty, self.tvars.add(b_ty))
                return ty
            case _:
                raise TypeInferError(f'attempting to unify `{a_ty.format()}` and `{b_ty.format()}`')

    def _instantiate(self, ty: Type) -> Type:
        subst: dict[VarType, Type] = {}
        for fv in sorted(ty.free_vars()):
            subst[fv] = self._fresh_type_var()
        return ty.subst(subst)

    def _generalize(self, ty: Type) -> tuple[Type, dict[VarType, Type]]:
        subst: dict[VarType, Type] = {}
        for i, fv in enumerate(sorted(ty.free_vars())):
            t = self.tvars.find(fv)
            match t: 
                case VarType():
                    subst[fv] = VarType(NamedId(f't{i + 1}'))
                case _:
                    subst[fv] = t
        ty = ty.subst(subst)
        return ty, subst

    def _ann_to_type(self, ty: TypeAnn | None) -> Type:
        return _ann_to_type(ty, self._fresh_type_var)

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
                case Len():
                    # length operator
                    self._unify(arg_ty, ListType(self._fresh_type_var()))
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
                case Sum():
                    # sum operator
                    self._unify(arg_ty, ListType(RealType()))
                    return RealType()
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

        arg_tys = [self._visit_expr(arg, None) for arg in e.args]

        match e.fn:
            case None:
                # unbound call
                ty = self._fresh_type_var()
                return ty
            case Primitive():
                # calling a primitive

                # type check the primitive and instantiate
                fn_ty = TypeInfer.infer_primitive(e.fn)
                fn_ty = cast(FunctionType, self._instantiate(fn_ty))

                # check arity
                if len(fn_ty.arg_types) != len(e.args):
                    actual_sig = f'function[{", ".join(arg.format() for arg in arg_tys)}]'
                    raise TypeInferError(f'primitive {e.fn.name}` has signature`{fn_ty.format()}`, but calling with `{actual_sig}``')
                # merge arguments
                for arg_ty, expect_ty in zip(arg_tys, fn_ty.arg_types):
                    self._unify(arg_ty, expect_ty)

                return fn_ty.return_type
            case Function():
                # calling a function

                # type check the function and instantiate
                if e.fn.sig is None:
                    # type checking not run
                    # TODO: guard against recursion
                    fn_info = TypeInfer.check(e.fn.ast)
                    fn_ty = fn_info.fn_type
                else:
                    fn_ty = e.fn.sig
                fn_ty = cast(FunctionType, self._instantiate(fn_ty))

                # check arity
                if len(fn_ty.arg_types) != len(e.args):
                    # no function signature / signature mismatch
                    actual_sig = f'function[{", ".join(arg.format() for arg in arg_tys)}]'
                    raise TypeInferError(f'function {e.fn.name}` has signature`{fn_ty.format()}`, but calling with `{actual_sig}`')
                # merge arguments
                for arg_ty, expect_ty in zip(arg_tys, fn_ty.arg_types):
                    self._unify(arg_ty, expect_ty)

                return fn_ty.return_type
            case type() if issubclass(e.fn, Context):
                # calling context constructor
                # TODO: type check constructor arguments based on Python typing hints
                return ContextType()
            case _:
                raise NotImplementedError(f'cannot type check {e.fn} {e.func}')

    def _visit_tuple_expr(self, e: TupleExpr, ctx: None) -> TupleType:
        elt_tys = [self._visit_expr(arg, None) for arg in e.elts]
        return TupleType(*elt_tys)

    def _visit_list_expr(self, e: ListExpr, ctx: None) -> ListType:
        arg_tys = [self._visit_expr(arg, None) for arg in e.elts]
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
                if not isinstance(ty, TupleType) or len(binding.elts) != len(ty.elts):
                    raise TypeInferError(f'cannot unpack `{ty.format()}` for `{binding.format()}`')
                # type has expected shape
                for elt_ty, elt in zip(ty.elts, binding.elts):
                    self._visit_binding(site, elt, elt_ty)
            case _:
                raise RuntimeError(f'unreachable: {binding}')

    def _visit_list_comp(self, e: ListComp, ctx: None) -> ListType:
        for target, iterable in zip(e.targets, e.iterables):
            iter_ty = self._visit_expr(iterable, None)
            if not isinstance(iter_ty, ListType):
                raise TypeInferError(f'iterator must be of type `list`, got `{iter_ty.format()}`')
            # expected type: list a
            self._visit_binding(e, target, iter_ty.elt)

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
        arr_ty = self._visit_expr(e.value, None)

        iter_ty = arr_ty
        for s in e.indices:
            ty = self._visit_expr(s, None)
            elt_ty = self._fresh_type_var()
            self._unify(arr_ty, ListType(elt_ty))
            self._unify(ty, RealType())
            iter_ty = elt_ty

        val_ty = self._visit_expr(e.expr, None)
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

    def _visit_attribute(self, e: Attribute, ctx: None):
        # TODO: how to type check attributes?
        # we expected the attribute value to be a module, but how do we propogate this information?
        self._visit_expr(e.value, None)
        return self._fresh_type_var()

    def _visit_assign(self, stmt: Assign, ctx: None):
        ty = self._visit_expr(stmt.expr, None)
        self._visit_binding(stmt, stmt.target, ty)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: None):
        d = self.def_use.find_def_from_use(stmt)
        arr_ty = self.by_def[d]

        for s in stmt.indices:
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

        # unify any merged variable
        for phi in self.def_use.phis[stmt]:
            lhs_ty = self.by_def[self.def_use.defs[phi.lhs]]
            rhs_ty = self.by_def[self.def_use.defs[phi.rhs]]
            ty = self._unify(lhs_ty, rhs_ty)
            self._set_type(phi, ty)

    def _visit_if(self, stmt: IfStmt, ctx: None):
        # type check condition
        cond_ty = self._visit_expr(stmt.cond, None)
        self._unify(cond_ty, BoolType())

        # type check branches
        self._visit_block(stmt.ift, None)
        self._visit_block(stmt.iff, None)

        # unify any merged variable
        for phi in self.def_use.phis[stmt]:
            lhs_ty = self.by_def[self.def_use.defs[phi.lhs]]
            rhs_ty = self.by_def[self.def_use.defs[phi.rhs]]
            ty = self._unify(lhs_ty, rhs_ty)
            self._set_type(phi, ty)

    def _visit_while(self, stmt: WhileStmt, ctx: None):
        # add types to phi variables
        for phi in self.def_use.phis[stmt]:
            lhs_ty = self.by_def[self.def_use.defs[phi.lhs]]
            self._set_type(phi, lhs_ty)

        cond_ty = self._visit_expr(stmt.cond, None)
        self._unify(cond_ty, BoolType())

        # type check body
        self._visit_block(stmt.body, None)

        # unify phi variables
        for phi in self.def_use.phis[stmt]:
            lhs_ty = self.by_def[self.def_use.defs[phi.lhs]]
            rhs_ty = self.by_def[self.def_use.defs[phi.rhs]]
            self._unify(lhs_ty, rhs_ty)

    def _visit_for(self, stmt: ForStmt, ctx: None):
        # type check iterable
        iter_ty = self._visit_expr(stmt.iterable, None)
        if not isinstance(iter_ty, ListType):
            raise TypeInferError(f'iterator must be of type `list`, got `{iter_ty.format()}`')

        # expected type: list a
        self._visit_binding(stmt, stmt.target, iter_ty.elt)

        # add types to phi variables
        for phi in self.def_use.phis[stmt]:
            lhs_ty = self.by_def[self.def_use.defs[phi.lhs]]
            self._set_type(phi, lhs_ty)

        # type check body
        self._visit_block(stmt.body, None)

        # unify phi variables
        for phi in self.def_use.phis[stmt]:
            lhs_ty = self.by_def[self.def_use.defs[phi.lhs]]
            rhs_ty = self.by_def[self.def_use.defs[phi.rhs]]
            self._unify(lhs_ty, rhs_ty)

    def _visit_context(self, stmt: ContextStmt, ctx: None):
        ty = self._visit_expr(stmt.ctx, None)
        if isinstance(stmt.target, NamedId):
            d = self.def_use.find_def_from_site(stmt.target, stmt)
            self._set_type(d, ty)
        self._visit_block(stmt.body, None)

    def _visit_assert(self, stmt: AssertStmt, ctx: None):
        ty = self._visit_expr(stmt.test, None)
        self._unify(ty, BoolType())
        if stmt.msg is not None:
            self._visit_expr(stmt.msg, None)

    def _visit_effect(self, stmt: EffectStmt, ctx: None):
        self._visit_expr(stmt.expr, None)

    def _visit_return(self, stmt: ReturnStmt, ctx: None):
        self.ret_type = self._visit_expr(stmt.expr, None)

    def _visit_pass(self, stmt: PassStmt, ctx: None):
        pass

    def _visit_block(self, block: StmtBlock, ctx: None):
        for stmt in block.stmts:
            self._visit_statement(stmt, None)

    def _visit_function(self, func: FuncDef, ctx: None) -> FunctionType:
        # infer types from annotations
        arg_tys: list[Type] = []
        for arg in func.args:
            arg_ty = self._ann_to_type(arg.type)
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
            raise TypeInferError(f'function {func.name} has no return type')

        # generalize the function type
        arg_tys = [self._resolve_type(ty) for ty in arg_tys]
        ret_ty = self._resolve_type(self.ret_type)
        return FunctionType(arg_tys, ret_ty)

    def _visit_expr(self, expr: Expr, ctx: None) -> Type:
        ret_ty = super()._visit_expr(expr, ctx)
        self.by_expr[expr] = ret_ty
        return ret_ty

    def analyze(self) -> TypeAnalysis:
        # type check the body
        ty = self._visit_function(self.func, None)

        # generalize the output type
        fn_ty, subst = self._generalize(ty)
        fn_ty = cast(FunctionType, fn_ty)

        # rename unbound type variables
        for t in self.tvars:
            if isinstance(t, VarType) and t not in subst:
                subst[t] = VarType(NamedId(f't{len(subst) + 1}'))

        # resolve definition/expr types
        by_defs = {
            name: self._resolve_type(ty).subst(subst)
            for name, ty in self.by_def.items()
        }
        by_expr = {
            e: self._resolve_type(ty).subst(subst)
            for e, ty in self.by_expr.items()
        }
        return TypeAnalysis(fn_ty, by_defs, by_expr, self.tvars)

###########################################################
# Primitives

class _TypeInferPrimitive:
    """
    Type inference for primitives.

    Converts typing annotations to types.
    """

    prim: Primitive
    gensym: Gensym

    def __init__(self, prim: Primitive):
        self.prim = prim
        self.gensym = Gensym()

    def _fresh_type_var(self) -> VarType:
        """Generates a fresh type variable."""
        return VarType(self.gensym.fresh('t'))

    def _ann_to_type(self, ty: TypeAnn | None) -> Type:
        return _ann_to_type(ty, self._fresh_type_var)

    def infer(self) -> FunctionType:
        arg_tys = [self._ann_to_type(ty) for ty in self.prim.arg_types]
        ret_ty = self._ann_to_type(self.prim.ret_type)
        return FunctionType(arg_tys, ret_ty)


###########################################################
# Type checker

class TypeInfer:
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
    def check(func: FuncDef, def_use: DefineUseAnalysis | None = None) -> TypeAnalysis:
        """
        Analyzes the function for type errors.

        Produces a type signature for the function if it is well-typed
        and a mapping from definition to type.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got {func}')

        if def_use is None:
            def_use = DefineUse.analyze(func)
        inst = _TypeInferInstance(func, def_use)
        return inst.analyze()

    @staticmethod
    def infer_primitive(prim: Primitive) -> FunctionType:
        """
        Returns the type signature of a primitive.
        """
        if not isinstance(prim, Primitive):
            raise TypeError(f'expected a \'Primitive\', got `{prim}`')

        inst = _TypeInferPrimitive(prim)
        return inst.infer()
