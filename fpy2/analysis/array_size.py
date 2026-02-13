"""
Array size inference.

TODO: inference algorithm likely broken for list assignment
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import TypeAlias, Union

from ..ast.fpyast import *
from ..ast.visitor import DefaultVisitor
from ..number import Float, INTEGER
from ..types import ListType, TupleType, Type
from ..utils import default_repr

from .define_use import Definition, DefSite, DefineUseAnalysis
from .partial_eval import PartialEval, PartialEvalInfo, Value
from .type_infer import TypeInfer, TypeAnalysis


@default_repr
class ArraySize:
    """array size: a set of possible sizes"""
    sizes: set[int]

    def __init__(self, sizes: set[int] | None = None):
        self.sizes = set(sizes) if sizes is not None else set()

    def __eq__(self, other):
        if not isinstance(other, ArraySize):
            return NotImplemented
        return self.sizes == other.sizes

    def __hash__(self):
        return hash(frozenset(self.sizes))

    def union(self, other: 'ArraySize') -> 'ArraySize':
        return ArraySize(self.sizes.union(other.sizes))

    def resolve(self) -> int | None:
        """resolve array size to a single concrete size if possible"""
        if len(self.sizes) == 1:
            return next(iter(self.sizes))
        else:
            return None

_Type: TypeAlias = Union['_Array', '_Tuple', None]
"""types for array size analysis: list, tuple, or scalar"""

@dataclass
class _Array:
    """symbolic array"""
    elt: _Type
    size: ArraySize

@dataclass
class _Tuple:
    """symbolic tuple"""
    elts: tuple[_Type, ...]

@dataclass
class ArraySizeAnalysis:
    """result of array size analysis"""
    by_expr: dict[Expr, _Type]
    by_def: dict[Definition, _Type]
    ret_size: _Type | None
    def_use: DefineUseAnalysis


class _ArraySizeVisitor(DefaultVisitor):
    func: FuncDef
    partial_eval: PartialEvalInfo
    type_info: TypeAnalysis

    by_expr: dict[Expr, _Type]
    by_def: dict[Definition, _Type]
    ret_size: _Type | None

    def __init__(self, func: FuncDef, partial_eval: PartialEvalInfo, type_info: TypeAnalysis):
        self.func = func
        self.partial_eval = partial_eval
        self.type_info = type_info
        self.by_expr = {}
        self.by_def = {}
        self.ret_size = None

    @property
    def def_use(self) -> DefineUseAnalysis:
        return self.partial_eval.def_use

    def infer(self):
        self._visit_function(self.func, None)
        return ArraySizeAnalysis(self.by_expr, self.by_def, self.ret_size, self.partial_eval.def_use)

    def _cvt_type(self, ty: Type) -> _Type:
        match ty:
            case ListType():
                elt = self._cvt_type(ty.elt)
                return _Array(elt, ArraySize())
            case TupleType():
                elts = tuple(self._cvt_type(e) for e in ty.elts)
                return _Tuple(elts)
            case _:
                return None
    
    def _get_eval(self, e: Expr) -> Value | None:
        if e in self.partial_eval.by_expr:
            return self.partial_eval.by_expr[e]
        else:
            return None

    def _unify(self, t1: _Type, t2: _Type) -> _Type:
        match t1, t2:
            case _Array(), _Array():
                elt = self._unify(t1.elt, t2.elt)
                size = t1.size.union(t2.size)
                return _Array(elt, size)
            case _Tuple(), _Tuple():
                elts = tuple(self._unify(e1, e2) for e1, e2 in zip(t1.elts, t2.elts, strict=True))
                return _Tuple(elts)
            case None, None:
                return None
            case _:
                raise TypeError(f'Cannot unify types: {t1} and {t2}')

    def _visit_binding(self, site: DefSite, target: Id | TupleBinding, ty: _Array | _Tuple | None):
        match target:
            case NamedId():
                d = self.def_use.find_def_from_site(target, site)
                self.by_def[d] = ty
            case TupleBinding():
                assert isinstance(ty, _Tuple), f'Expected tuple type for tuple binding, got {ty} for {target}'
                for elt, s in zip(target.elts, ty.elts, strict=True):
                    self._visit_binding(site, elt, s)
            case _:
                pass

    def _visit_var(self, e: Var, ctx: None):
        d = self.def_use.find_def_from_use(e)
        return self.by_def[d]

    def _visit_unaryop(self, e: UnaryOp, ctx: None):
        ty = self._visit_expr(e.arg, ctx)
        match e:
            case Range1():
                stop = self._get_eval(e.arg)
                if isinstance(stop, Float | Fraction):
                    size = int(INTEGER.round(stop))
                    return _Array(None, ArraySize({size}))
                else:
                    return _Array(None, ArraySize())
            case Enumerate():
                assert isinstance(ty, _Array)
                return _Array(_Tuple((None, ty.elt)), ty.size)

    def _visit_binaryop(self, e: BinaryOp, ctx: None):
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        match e:
            case Range2():
                start = self._get_eval(e.first)
                stop = self._get_eval(e.second)
                if isinstance(start, Float | Fraction) and isinstance(stop, Float | Fraction):
                    start_i = int(INTEGER.round(start))
                    stop_i = int(INTEGER.round(stop))
                    size = max(0, stop_i - start_i)
                    return _Array(None, ArraySize({size}))
                else:
                    return _Array(None, ArraySize())

    def _visit_ternaryop(self, e: TernaryOp, ctx: None):
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        self._visit_expr(e.third, ctx)
        match e:
            case Range3():
                # TODO: implement
                # for now just return a symbolic list
                return _Array(None, ArraySize())

    def _visit_naryop(self, e: NaryOp, ctx: None):
        tys = [self._visit_expr(arg, ctx) for arg in e.args]
        match e:
            case Zip():
                if len(e.args) == 0:
                    return _Array(None, ArraySize({0}))
                else:
                    assert isinstance(tys[0], _Array)
                    elt_tys: list[_Type] = [tys[0].elt]
                    size = tys[0].size
                    for ty in tys[1:]:
                        assert isinstance(ty, _Array)
                        elt_tys.append(ty.elt)
                        size = size.union(ty.size)

                    return _Array(_Tuple(tuple(elt_tys)), size)

            case Empty():
                # iterate from the inner dimension outwards to compute size
                arg_rev = list(reversed(e.args))

                # extract the type of the innermost dimension
                elt_ty = self._cvt_type(self.type_info.by_expr[e])
                for _ in arg_rev:
                    assert isinstance(elt_ty, _Array)
                    elt_ty = elt_ty.elt

                # innermost dimension
                size_v = self._get_eval(arg_rev[0])
                if isinstance(size_v, Float | Fraction):
                    size = ArraySize({int(INTEGER.round(size_v))})
                else:
                    size = ArraySize()

                # type of the innermost dimension
                ty = _Array(elt_ty, size)

                # outer dimensions
                for arg in arg_rev[1:]:
                    size_v = self._get_eval(arg)
                    if isinstance(size_v, Float | Fraction):
                        size = ArraySize({int(INTEGER.round(size_v))})
                    else:
                        size = ArraySize()

                    ty = _Array(ty, size)

                return ty

    def _visit_list_expr(self, e: ListExpr, ctx: None):
        elt_sizes = [self._visit_expr(elt, ctx) for elt in e.elts]
        ty = self.type_info.by_expr[e]
        assert isinstance(ty, ListType)
        if isinstance(ty.elt, ListType):
            assert isinstance(elt_sizes[0], _Array)
            elt_size = elt_sizes[0]
        else:
            elt_size = None
        return _Array(elt_size, ArraySize({len(e.elts)}))

    def _visit_list_comp(self, e: ListComp, ctx: None):
        # process iterables and bindings
        iter_tys: list[_Array] = []
        for target, iterable in zip(e.targets, e.iterables, strict=True):
            ty = self._visit_expr(iterable, ctx)
            assert isinstance(ty, _Array)
            self._visit_binding(e, target, ty.elt)
            iter_tys.append(ty)

        # process element expression
        elt_ty = self._visit_expr(e.elt, ctx)

        # try to compute size
        size = 1
        for ty in iter_tys:
            iter_size = ty.size.resolve()
            if iter_size is None:
                return _Array(elt_ty, ArraySize())
            size *= iter_size

        return _Array(elt_ty, ArraySize({size}))

    def _visit_list_ref(self, e: ListRef, ctx: None):
        ty = self._visit_expr(e.value, ctx)
        self._visit_expr(e.index, ctx)
        assert isinstance(ty, _Array)
        return ty.elt

    def _visit_list_slice(self, e: ListSlice, ctx: None):
        ty = self._visit_expr(e.value, ctx)
        assert isinstance(ty, _Array)

        if e.start is None:
            start = 0
        else:
            self._visit_expr(e.start, ctx)
            start_val = self._get_eval(e.start)
            if isinstance(start_val, Float | Fraction):
                start = int(INTEGER.round(start_val))
            else:
                start = None

        if e.stop is None:
            stop = None
        else:
            self._visit_expr(e.stop, ctx)
            stop_val = self._get_eval(e.stop)
            if isinstance(stop_val, Float | Fraction):
                stop = int(INTEGER.round(stop_val))
            else:
                stop = None

        size = ty.size.resolve()
        if size is not None and start is not None and stop is not None:
            # can compute size of the slice
            size = max(0, min(stop, size) - min(start, size))

        return _Array(ty.elt, ArraySize({size} if size is not None else set()))

    def _visit_tuple_expr(self, e: TupleExpr, ctx: None):
        return _Tuple(tuple(self._visit_expr(elt, ctx) for elt in e.elts))

    def _visit_if_expr(self, e: IfExpr, ctx: None):
        self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, ctx)
        iff = self._visit_expr(e.iff, ctx)
        return self._unify(ift, iff)

    def _visit_call(self, e: Call, ctx: None):
        for arg in e.args:
            self._visit_expr(arg, ctx)
        # just convert type for now
        ty = self.type_info.by_expr[e]
        return self._cvt_type(ty)

    def _visit_assign(self, stmt: Assign, ctx: None):
        ty = self._visit_expr(stmt.expr, ctx)
        self._visit_binding(stmt, stmt.target, ty)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: None):
        d = self.def_use.find_def_from_use(stmt)

        def recur(indices: tuple[Expr, ...], target_ty: _Type) -> _Type:
            if len(indices) == 0:
                ty = self._visit_expr(stmt.expr, ctx)
                return self._unify(target_ty, ty)
            else:
                self._visit_expr(indices[0], ctx)
                assert isinstance(target_ty, _Array), f'Expected array type for indexed assignment, got {target_ty} for {stmt}'
                elt_ty = recur(indices[1:], target_ty.elt)
                return _Array(elt_ty, target_ty.size)

        ty = recur(stmt.indices, self.by_def[d])
        self.by_def[d] = ty


    def _visit_if1(self, stmt: If1Stmt, ctx: None):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)

        # unify any merged variable
        for phi in self.def_use.phis[stmt]:
            lhs_ty = self.by_def[self.def_use.defs[phi.lhs]]
            rhs_ty = self.by_def[self.def_use.defs[phi.rhs]]
            self.by_def[phi] = self._unify(lhs_ty, rhs_ty)

    def _visit_if(self, stmt: IfStmt, ctx: None):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.ift, ctx)
        self._visit_block(stmt.iff, ctx)

        # unify any merged variable
        for phi in self.def_use.phis[stmt]:
            lhs_ty = self.by_def[self.def_use.defs[phi.lhs]]
            rhs_ty = self.by_def[self.def_use.defs[phi.rhs]]
            self.by_def[phi] = self._unify(lhs_ty, rhs_ty)

    def _visit_while(self, stmt: WhileStmt, ctx: None):
        # add types to phi variables
        for phi in self.def_use.phis[stmt]:
            lhs_ty = self.by_def[self.def_use.defs[phi.lhs]]
            self.by_def[phi] = lhs_ty

        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)

        for phi in self.def_use.phis[stmt]:
            lhs_ty = self.by_def[self.def_use.defs[phi.lhs]]
            rhs_ty = self.by_def[self.def_use.defs[phi.rhs]]
            self.by_def[phi] = self._unify(lhs_ty, rhs_ty)

    def _visit_for(self, stmt, ctx):
        # process iterable and binding
        iter_ty = self._visit_expr(stmt.iterable, ctx)
        assert isinstance(iter_ty, _Array)
        self._visit_binding(stmt, stmt.target, iter_ty.elt)

        # add types to phi variables
        for phi in self.def_use.phis[stmt]:
            lhs_ty = self.by_def[self.def_use.defs[phi.lhs]]
            self.by_def[phi] = lhs_ty

        # visit body
        self._visit_block(stmt.body, ctx)

        # unify any merged variable
        for phi in self.def_use.phis[stmt]:
            lhs_ty = self.by_def[self.def_use.defs[phi.lhs]]
            rhs_ty = self.by_def[self.def_use.defs[phi.rhs]]
            self.by_def[phi] = self._unify(lhs_ty, rhs_ty)

    def _visit_context(self, stmt, ctx):
        ty = self._visit_expr(stmt.ctx, ctx)
        if isinstance(stmt.target, NamedId):
            d = self.def_use.find_def_from_site(stmt.target, stmt)
            self.by_def[d] = ty
        self._visit_block(stmt.body, ctx)

    def _visit_return(self, stmt: ReturnStmt, ctx: None):
        ret_size = self._visit_expr(stmt.expr, ctx)
        if isinstance(ret_size, _Array):
            self.ret_size = ret_size

    def _visit_expr(self, expr: Expr, ctx: None) -> _Type:
        ty = super()._visit_expr(expr, ctx)
        self.by_expr[expr] = ty
        return ty

    def _visit_function(self, func: FuncDef, ctx: None):
        # process arguments
        for arg, ty in zip(func.args, self.type_info.arg_types):
            if isinstance(arg.name, NamedId):
                d = self.def_use.find_def_from_site(arg.name, arg)
                self.by_def[d] = self._cvt_type(ty)

        # process free variables
        for fv in func.free_vars:
            d = self.def_use.find_def_from_site(fv, func)
            ty = self.type_info.by_def[d]
            self.by_def[d] = self._cvt_type(ty)

        # visit body
        self._visit_block(func.body, ctx)

class ArraySizeInfer:
    """Array size inference."""

    @staticmethod
    def infer(
        func: FuncDef,
        *,
        partial_eval: PartialEvalInfo | None = None,
        type_info: TypeAnalysis | None = None
    ):
        """Infer array sizes in the given function definition.

        Args:
            func: Function definition to analyze.
            partial_eval: Optional partial evaluation information.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected `FuncDef`, got {type(func)} for {func}')

        if partial_eval is None:
            partial_eval = PartialEval.apply(func)
        if type_info is None:
            type_info = TypeInfer.check(func, def_use=partial_eval.def_use)

        visitor = _ArraySizeVisitor(func, partial_eval, type_info)
        return visitor.infer()
