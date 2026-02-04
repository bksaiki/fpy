"""
Array size inference.
"""

import math

from dataclasses import dataclass
from fractions import Fraction
from typing import TypeAlias, Union

from ..ast.fpyast import *
from ..ast.visitor import DefaultVisitor
from ..number import Float, INTEGER
from ..types import ListType
from ..utils import Gensym, Unionfind

from .define_use import Definition, DefSite, DefineUseAnalysis
from .partial_eval import PartialEval, PartialEvalInfo
from .type_infer import TypeInfer, TypeAnalysis


ArraySize: TypeAlias = int | None
"""array size: either symbolic or concrete"""

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
        self.by_def = {}
        self.by_expr = {}
        self.ret_size = None

    @property
    def def_use(self) -> DefineUseAnalysis:
        return self.partial_eval.def_use

    def infer(self):
        self._visit_function(self.func, None)
        return ArraySizeAnalysis(self.by_expr, self.by_def, self.ret_size, self.partial_eval.def_use)

    def _cvt_list_type(self, ty: ListType) -> _Array:
        if isinstance(ty.elt, ListType):
            elt = self._cvt_list_type(ty.elt)
        else:
            elt = None
        return _Array(elt, None)

    def _unify(self, t1: _Type, t2: _Type) -> _Type:
        match t1, t2:
            case _Array(), _Array():
                elt = self._unify(t1.elt, t2.elt)
                size = t1.size if t1.size == t2.size else None
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
                if isinstance(ty, _Array):
                    self.by_def[d] = ty
            case TupleBinding():
                assert isinstance(ty, _Tuple)
                for elt, s in zip(target.elts, ty.elts, strict=True):
                    self._visit_binding(site, elt, s)
            case _:
                pass

    def _visit_var(self, e: Var, ctx: None):
        d = self.def_use.find_def_from_use(e)
        if d in self.by_def:
            return self.by_def[d]

    def _visit_unaryop(self, e: UnaryOp, ctx: None):
        self._visit_expr(e.arg, ctx)
        match e:
            case Range1():
                stop = self.partial_eval.by_expr[e.arg]
                if isinstance(stop, Float | Fraction):
                    size = int(INTEGER.round(stop))
                    return _Array(None, size)

    def _visit_binaryop(self, e: BinaryOp, ctx: None):
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        match e:
            case Range2():
                start = self.partial_eval.by_expr[e.first]
                stop = self.partial_eval.by_expr[e.second]
                if isinstance(start, Float | Fraction) and isinstance(stop, Float | Fraction):
                    start_i = int(INTEGER.round(start))
                    stop_i = int(INTEGER.round(stop))
                    size = max(0, stop_i - start_i)
                    return _Array(None, size)

    def _visit_ternaryop(self, e, ctx):
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        self._visit_expr(e.third, ctx)
        match e:
            case Range3():
                # TODO: implement
                # for now just return a symbolic list
                return _Array(None, None)

    def _visit_list_expr(self, e: ListExpr, ctx: None):
        elt_sizes = [self._visit_expr(elt, ctx) for elt in e.elts]
        ty = self.type_info.by_expr[e]
        assert isinstance(ty, ListType)
        if isinstance(ty.elt, ListType):
            print(e.format(), elt_sizes[0])
            assert isinstance(elt_sizes[0], _Array)
            elt_size = elt_sizes[0]
        else:
            elt_size = None
        return _Array(elt_size, len(e.elts))

    def _visit_list_comp(self, e: ListComp, ctx: None):
        # process iterables and bindings
        iter_tys: list[_Array] = []
        for target, iterable in zip(e.targets, e.iterables, strict=True):
            ty = self._visit_expr(iterable, ctx)
            assert isinstance(ty, _Array)
            self._visit_binding(e, target, ty)
            iter_tys.append(ty)

        # process element expression
        elt_ty = self._visit_expr(e.elt, ctx)

        # try to compute size
        size: int = 1
        for ty in iter_tys:
            if isinstance(ty.size, int):
                size *= ty.size
            else:
                return _Array(elt_ty, None)

        return _Array(elt_ty, size)

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
            start_val = self.partial_eval.by_expr[e.start]
            if isinstance(start_val, Float | Fraction):
                start = int(INTEGER.round(start_val))
            else:
                start = None

        if e.stop is None:
            stop = None
        else:
            self._visit_expr(e.stop, ctx)
            stop_val = self.partial_eval.by_expr[e.stop]
            if isinstance(stop_val, Float | Fraction):
                stop = int(INTEGER.round(stop_val))
            else:
                stop = None

        if ty.size is None or start is None or stop is None:
            # list size or slice indices are unknown
            size = None
        else:
            # can compute size of the slice
            size = max(0, (stop % ty.size) - (start % ty.size))

        return _Array(ty.elt, size)

    def _visit_tuple_expr(self, e: TupleExpr, ctx: None):
        return _Tuple(tuple(self._visit_expr(elt, ctx) for elt in e.elts))

    def _visit_if_expr(self, e: IfExpr, ctx: None):
        self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, ctx)
        iff = self._visit_expr(e.iff, ctx)
        return self._unify(ift, iff)

    def _visit_assign(self, stmt: Assign, ctx: None):
        ty = self._visit_expr(stmt.expr, ctx)
        self._visit_binding(stmt, stmt.target, ty)

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

        # visit body
        self._visit_block(stmt.body, ctx)

        # unify any merged variable
        for phi in self.def_use.phis[stmt]:
            lhs_ty = self.by_def[self.def_use.defs[phi.lhs]]
            rhs_ty = self.by_def[self.def_use.defs[phi.rhs]]
            self.by_def[phi] = self._unify(lhs_ty, rhs_ty)

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
            if isinstance(arg.name, NamedId) and isinstance(ty, ListType):
                d = self.def_use.find_def_from_site(arg.name, arg)
                self.by_def[d] = self._cvt_list_type(ty)

        # process free variables
        for fv in func.free_vars:
            d = self.def_use.find_def_from_site(fv, func)
            ty = self.type_info.by_def[d]
            if isinstance(ty, ListType):
                self.by_def[d] = self._cvt_list_type(ty)

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
