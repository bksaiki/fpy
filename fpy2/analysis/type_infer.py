"""
Type checking for FPy programs.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, cast

from ..ast import *
from ..number import Context, Float, RealFloat
from ..primitive import Primitive
from ..utils import Gensym, NamedId, Unionfind

from ..types import *
from .call_graph import CallGraph, CallGraphError
from .define_use import DefineUse, DefineUseAnalysis, Definition, DefSite

#####################################################################
# Type Inference

_Bool1ary = FunctionType(None, [BoolType()], BoolType())
_Real0ary = FunctionType(None, [], RealType(None))
_Real1ary = FunctionType(None, [RealType(None)], RealType(None))
_Real2ary = FunctionType(None, [RealType(None), RealType(None)], RealType(None))
_Real3ary = FunctionType(None, [RealType(None), RealType(None), RealType(None)], RealType(None))
_Predicate = FunctionType(None, [RealType(None)], BoolType())

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
    Abs: _Real1ary,
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
    Cast: _Real1ary,
    Logb: _Real1ary
}

_binary_table: dict[type[BinaryOp], FunctionType] = {
    Add: _Real2ary,
    Sub: _Real2ary,
    Mul: _Real2ary,
    Div: _Real2ary,
    Copysign: _Real2ary,
    Fdim: _Real2ary,
    Mod: _Real2ary,
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


def _merge_length(a: int | NamedId | None, b: int | NamedId | None) -> int | NamedId | None:
    """Keep the more specific of two list lengths: ``concrete > symbolic >
    None``.  Used when unifying two list types; never fails (length is
    metadata)."""
    if isinstance(a, int):
        return a
    if isinstance(b, int):
        return b
    return a if a is not None else b


def _drop_symbolic_lengths(ty: Type) -> Type:
    """Replace every *symbolic* (``NamedId``) list length with ``None``,
    keeping concrete ``int`` lengths.  Applied when instantiating a callee's
    type so its dimension variables don't leak across the call boundary."""
    match ty:
        case ListType():
            length = ty.length if isinstance(ty.length, int) else None
            return ListType(_drop_symbolic_lengths(ty.elt), length)
        case TupleType():
            return TupleType(*[_drop_symbolic_lengths(e) for e in ty.elts])
        case FunctionType():
            return FunctionType(
                ty.ctx,
                [_drop_symbolic_lengths(a) for a in ty.arg_types],
                _drop_symbolic_lengths(ty.return_type),
            )
        case _:
            return ty


def _ann_to_type(ty: TypeAnn | None, fresh_var: Callable[[], VarType]) -> Type:
    match ty:
        case AnyTypeAnn():
            return fresh_var()
        case BoolTypeAnn():
            # boolean type
            return BoolType()
        case RealTypeAnn():
            # Preserve the annotated context so that downstream analyses
            # (notably format inference) can recover the format pinned by
            # a monomorphizing pass via ``RealType.ctx``.
            return RealType(ty.ctx)
        case TupleTypeAnn():
            # tuple type
            elt_tys = [_ann_to_type(elt, fresh_var) for elt in ty.elts]
            return TupleType(*elt_tys)
        case ListTypeAnn():
            # list type — carry the optional size annotation onto the type
            return ListType(_ann_to_type(ty.elt, fresh_var), ty.length)
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
    def_use: DefineUseAnalysis

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
    sigs: dict[FuncDef, 'FunctionType']
    by_def: dict[Definition, Type]
    by_expr: dict[Expr, Type]
    ret_type: Type | None
    tvars: Unionfind[Type]
    gensym: Gensym

    def __init__(
        self,
        func: FuncDef,
        def_use: DefineUseAnalysis,
        sigs: dict[FuncDef, 'FunctionType'] | None = None,
    ):
        self.func = func
        self.def_use = def_use
        # Generalized signatures of already-checked callees, keyed by
        # ``FuncDef``.  `TypeInfer.check` pre-fills this in leaves-first
        # call-graph order, so a call's `_visit_call` is a lookup rather
        # than a recursive re-check.  Empty/partial caches fall back to
        # lazy checking (see `_visit_call`).
        self.sigs = {} if sigs is None else sigs
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

    def _value_to_type(self, val: object) -> Type | None:
        """The FPy type of a captured free-variable *value*, or ``None`` for a
        value with no FPy type (a callable, module, class, or a heterogeneous /
        empty container), in which case the caller keeps a fresh type variable.

        The capture is fixed at reparse time, so its type is authoritative:
        seeding it lets a free var used polymorphically (or only passed through)
        recover a concrete type instead of generalizing to a bare variable, and
        gives :class:`ArraySizeInfer` the arity of captured tuples/lists.  The
        numeric cases mirror format inference's capture matcher so the two
        analyses agree on what counts as a real value."""
        match val:
            case bool():
                # `bool` subclasses `int`; match before the numeric case.
                return BoolType()
            case Float() | RealFloat() | Fraction() | int() | float():
                return RealType(None)
            case Context():
                return ContextType()
            case tuple():
                elts = [self._value_to_type(x) for x in val]
                if any(e is None for e in elts):
                    return None
                return TupleType(*cast(list[Type], elts))
            case list():
                # Lists are homogeneous.  An empty list has a known length (0)
                # but no element to type — use a fresh element variable.
                if not val:
                    return ListType(self._fresh_type_var(), 0)
                elt_tys = [self._value_to_type(x) for x in val]
                if any(e is None for e in elt_tys):
                    return None
                first = cast(Type, elt_tys[0])
                if any(e != first for e in elt_tys):
                    return None
                return ListType(first, len(val))
            case _:
                return None

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
                return self.tvars.add(ListType(elt_ty, ty.length))
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
                # length is metadata: keep the more specific, never fail
                length = _merge_length(a_ty.length, b_ty.length)
                return self.tvars.add(ListType(elt_ty, length))
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
        subst: dict[NamedId, Type] = {}
        for fv in sorted(ty.free_type_vars()):
            subst[fv] = self._fresh_type_var()
        # a callee's *symbolic* list lengths name the callee's own
        # dimensions and are meaningless at the call site — drop them
        # (concrete lengths stay; they hold regardless of caller).
        return _drop_symbolic_lengths(ty.subst_type(subst))

    def _generalize(self, ty: Type) -> tuple[Type, dict[NamedId, Type]]:
        subst: dict[NamedId, Type] = {}
        for i, fv in enumerate(sorted(ty.free_type_vars())):
            t = self.tvars.find(VarType(fv))
            match t: 
                case VarType():
                    subst[fv] = VarType(NamedId(f't{i + 1}'))
                case _:
                    subst[fv] = t
        ty = ty.subst_type(subst)
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
        return RealType(None)

    def _visit_hexnum(self, e: Hexnum, ctx: None) -> RealType:
        return RealType(None)

    def _visit_integer(self, e: Integer, ctx: None) -> RealType:
        return RealType(None)

    def _visit_rational(self, e: Rational, ctx: None) -> RealType:
        return RealType(None)

    def _visit_digits(self, e: Digits, ctx: None) -> RealType:
        return RealType(None)

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
                    return RealType(None)
                case Range1():
                    # range operator
                    self._unify(arg_ty, RealType(None))
                    return ListType(RealType(None))
                case Dim():
                    # dimension operator
                    self._unify(arg_ty, ListType(self._fresh_type_var()))
                    return RealType(None)
                case Fst():
                    # `fst : 'a * 'b -> 'a`, the first projection of a pair.
                    # Unifying the operand with a fresh pair both constrains an
                    # open type variable and rejects non-pairs (a real, a list,
                    # or a tuple of arity != 2).
                    head = self._fresh_type_var()
                    tail = self._fresh_type_var()
                    self._unify(arg_ty, TupleType(head, tail))
                    return head
                case Snd():
                    # `snd : 'a * 'b -> 'b`, the second projection (see `Fst`).
                    head = self._fresh_type_var()
                    tail = self._fresh_type_var()
                    self._unify(arg_ty, TupleType(head, tail))
                    return tail
                case Enumerate():
                    # enumerate operator
                    ty = self._fresh_type_var()
                    self._unify(arg_ty, ListType(ty))
                    return ListType(TupleType(RealType(None), ty))
                case Sum() | AMin() | AMax():
                    # list-reduce operators: list[real] -> real
                    self._unify(arg_ty, ListType(RealType(None)))
                    return RealType(None)
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
                    self._unify(rhs_ty, RealType(None))
                    return RealType(None)
                case Range2():
                    # range2 operator
                    self._unify(lhs_ty, RealType(None))
                    self._unify(rhs_ty, RealType(None))
                    return ListType(RealType(None))
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
            match e:
                case Range3():
                    # range3 operator
                    self._unify(first, RealType(None))
                    self._unify(second, RealType(None))
                    self._unify(third, RealType(None))
                    return ListType(RealType(None))
                case _:
                    raise ValueError(f'unknown ternary operator: {cls}')

    def _visit_naryop(self, e: NaryOp, ctx: None) -> Type:
        match e:
            case Min() | Max():
                for arg in e.args:
                    ty = self._visit_expr(arg, None)
                    self._unify(ty, RealType(None))
                return RealType(None)
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
                    arg_tys.append(self._resolve_type(ty))
                return ListType(TupleType(*arg_tys))
            case Empty():
                #    Γ |- e1 : real  ...  Γ |- eN : real
                # ---------------------------------------
                #   Γ |- empty(e1, ..., en) : list[list[... [A]]
                for arg in e.args:
                    arg_ty = self._visit_expr(arg, None)
                    self._unify(arg_ty, RealType())

                ty = self._fresh_type_var()
                for _ in e.args:
                    ty = ListType(ty)
                return ty
            case _:
                raise ValueError(f'unknown n-ary operator: {type(e)}')

    def _visit_compare(self, e: Compare, ctx: None) -> BoolType:
        for arg in e.args:
            ty = self._visit_expr(arg, None)
            self._unify(ty, RealType(None))
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
                # calling a function: look up the callee's generalized
                # signature and instantiate it at this site.  `check`
                # pre-fills `sigs` in leaves-first call-graph order, so
                # this is normally a hit; the miss branch lazily checks
                # a callee that wasn't walked (and can't loop — `check`
                # runs a `CallGraph` acyclicity guard at its entry).
                callee = e.fn.ast
                if callee not in self.sigs:
                    self.sigs[callee] = TypeInfer.check(callee).fn_type
                fn_ty = cast(FunctionType, self._instantiate(self.sigs[callee]))

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
        # ``ty`` may be a ``VarType`` whose union-find representative is a
        # concrete shape; resolve so tuple unpacking sees the shape rather
        # than the placeholder.
        ty = self._resolve_type(ty)
        match binding:
            case NamedId():
                d = self.def_use.find_def_from_site(binding, site)
                self._set_type(d, ty)
            case UnderscoreId():
                pass
            case TupleBinding():
                # unify with a tuple of this arity (constrains an open
                # type, raises on a real mismatch)
                elts = [self._fresh_type_var() for _ in binding.elts]
                self._unify(ty, TupleType(*elts))
                for elt_ty, elt in zip(elts, binding.elts):
                    self._visit_binding(site, elt, elt_ty)
            case _:
                raise RuntimeError(f'unreachable: {binding}')

    def _visit_list_comp(self, e: ListComp, ctx: None) -> ListType:
        for target, iterable in zip(e.targets, e.iterables):
            iter_ty = self._visit_expr(iterable, None)
            # unify with a list (constrains an open type, raises on a non-list)
            item_ty = self._fresh_type_var()
            self._unify(iter_ty, ListType(item_ty))
            self._visit_binding(e, target, item_ty)

        elt_ty = self._visit_expr(e.elt, None)
        return ListType(elt_ty)

    def _visit_list_ref(self, e: ListRef, ctx: None) -> Type:
        # val : list[A]
        value_ty = self._visit_expr(e.value, None)
        ty = self._fresh_type_var()
        self._unify(value_ty, ListType(ty))
        # index : real
        index_ty = self._visit_expr(e.index, None)
        self._unify(index_ty, RealType(None))
        # val[index] : A
        return ty

    def _visit_list_slice(self, e: ListSlice, ctx: None):
        # type check array
        value_ty = self._visit_expr(e.value, None)
        self._unify(value_ty, ListType(self._fresh_type_var()))
        # type check endpoints
        if e.start is not None:
            start_ty = self._visit_expr(e.start, None)
            self._unify(start_ty, RealType(None))
        if e.stop is not None:
            stop_ty = self._visit_expr(e.stop, None)
            self._unify(stop_ty, RealType(None))
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
            self._unify(ty, RealType(None))
            # arr [idx] : A
            arr_ty = elt_ty

        # val : A
        val_ty = self._visit_expr(stmt.expr, None)
        self._unify(val_ty, arr_ty)

        # ``xs[i] = e`` produces a fresh SSA def of ``xs`` (per
        # ``reaching_defs``).  The static type of ``xs`` is unchanged by
        # element mutation, so propagate the same type.
        d_new = self.def_use.find_def_from_site(stmt.var, stmt)
        self.by_def[d_new] = self.by_def[d]

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
        # unify with a list (constrains an open type, raises on a non-list)
        iter_ty = self._visit_expr(stmt.iterable, None)
        elt_ty = self._fresh_type_var()
        self._unify(iter_ty, ListType(elt_ty))
        self._visit_binding(stmt, stmt.target, elt_ty)

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
        self._unify(ty, ContextType())
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
        # Multiple returns: the function's return type is the
        # unification of every return path's type.  The first
        # encountered return seeds ``ret_type``; subsequent ones
        # unify against the existing value, surfacing a
        # ``TypeInferError`` if the paths disagree on a structural
        # type.
        ty = self._visit_expr(stmt.expr, None)
        if self.ret_type is None:
            self.ret_type = ty
        else:
            self.ret_type = self._unify(self.ret_type, ty)

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
        # A captured value's type is authoritative; seed it when known so a
        # free var recovers a concrete type instead of a bare variable.  A
        # foreign value (callable, module, ...) keeps a fresh type variable and
        # is refined by use, as before.
        for v in func.free_vars:
            d = self.def_use.find_def_from_site(v, func)
            ty = self._value_to_type(func.env.get(str(v)))
            if ty is None:
                ty = self._fresh_type_var()
            self._set_type(d, ty)

        # type check body
        self._visit_block(func.body, None)
        if self.ret_type is None:
            raise TypeInferError(f'function {func.name} has no return type')

        # generalize the function type
        arg_tys = [self._resolve_type(ty) for ty in arg_tys]
        ret_ty = self._resolve_type(self.ret_type)
        return FunctionType(None, arg_tys, ret_ty)

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
                subst[t.name] = VarType(NamedId(f't{len(subst) + 1}'))

        # resolve definition/expr types
        by_defs = {
            name: self._resolve_type(ty).subst_type(subst)
            for name, ty in self.by_def.items()
        }
        by_expr = {
            e: self._resolve_type(ty).subst_type(subst)
            for e, ty in self.by_expr.items()
        }
        return TypeAnalysis(fn_ty, by_defs, by_expr, self.tvars, self.def_use)

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
        return FunctionType(None, arg_tys, ret_ty)


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

        # Build the call graph: this guards against recursion (FPy
        # forbids it; `CallGraph` raises on any reachable cycle) and
        # gives the leaves-first order to check callees before callers.
        try:
            cg = CallGraph.analyze(func)
        except CallGraphError as e:
            raise TypeInferError(str(e)) from e

        # Walk callees-before-callers, caching each generalized
        # signature so `_visit_call` reuses it instead of re-checking
        # the callee at every call site.  The root is last in the
        # order, so its full analysis is what we return.
        sigs: dict[FuncDef, FunctionType] = {}
        result: TypeAnalysis | None = None
        for fdef in cg.order:
            fdef_du = def_use if fdef is func and def_use is not None \
                else DefineUse.analyze(fdef)
            analysis = _TypeInferInstance(fdef, fdef_du, sigs).analyze()
            sigs[fdef] = analysis.fn_type
            if fdef is func:
                result = analysis
        assert result is not None  # func is always in cg.order
        return result

    @staticmethod
    def infer_primitive(prim: Primitive) -> FunctionType:
        """
        Returns the type signature of a primitive.
        """
        if not isinstance(prim, Primitive):
            raise TypeError(f'expected a \'Primitive\', got `{prim}`')

        inst = _TypeInferPrimitive(prim)
        return inst.infer()
