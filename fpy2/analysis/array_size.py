"""
Array size inference.
"""

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


ArraySize: TypeAlias = int | NamedId
"""array size: either symbolic or concrete"""

_Type: TypeAlias = Union['_Array', '_Tuple', None]
"""types for array size analysis: list, tuple, or scalar"""

@dataclass
class _Array:
    """symbolic array"""
    elt: Union['_Array', '_Tuple', None]
    size: ArraySize

@dataclass
class _Tuple:
    """symbolic tuple"""
    elts: tuple[Union['_Array', '_Tuple', None], ...]

@dataclass
class ArraySizeAnalysis:
    """result of array size analysis"""
    by_expr: dict[Expr, _Array]
    by_def: dict[Definition, _Array]
    ret_size: _Array | None
    def_use: DefineUseAnalysis


class _ArraySizeVisitor(DefaultVisitor):
    func: FuncDef
    partial_eval: PartialEvalInfo
    type_info: TypeAnalysis

    gensym: Gensym
    size_vars: Unionfind[ArraySize]
    by_expr: dict[Expr, _Array]
    by_def: dict[Definition, _Array]
    ret_size: _Array | None

    def __init__(self, func: FuncDef, partial_eval: PartialEvalInfo, type_info: TypeAnalysis):
        self.func = func
        self.partial_eval = partial_eval
        self.type_info = type_info
        self.gensym = Gensym()
        self.size_vars = Unionfind()
        self.by_def = {}
        self.by_expr = {}
        self.ret_size = None

    @property
    def def_use(self) -> DefineUseAnalysis:
        return self.partial_eval.def_use

    def infer(self):
        self._visit_function(self.func, None)
        return ArraySizeAnalysis(self.by_expr, self.by_def, self.ret_size, self.partial_eval.def_use)

    def _fresh_var(self) -> NamedId:
        var = self.gensym.fresh('n')
        self.size_vars.add(var)
        return var

    def _cvt_list_type(self, ty: ListType) -> _Array:
        if isinstance(ty.elt, ListType):
            elt = self._cvt_list_type(ty.elt)
            return _Array(elt, self._fresh_var())
        else:
            return _Array(None, self._fresh_var())

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
                stop = self.partial_eval.by_expr[e]
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
                # TODO: implemnent
                # for now just return a symbolic list
                return _Array(None, self._fresh_var())

    def _visit_list_comp(self, e: ListComp, ctx: None):
        # process iterables and bindings
        iter_tys: list[_Type] = []
        for target, iterable in zip(e.targets, e.iterables, strict=True):
            ty = self._visit_expr(iterable, ctx)
            assert isinstance(ty, _Array)
            self._visit_binding(e, target, ty)
            iter_tys.append(ty)

        # process element expression
        elt_ty = self._visit_expr(e.elt, ctx)
        if all([ ])


        return super()._visit_list_comp(e, ctx)

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

    def _visit_tuple_expr(self, e: TupleExpr, ctx: None):
        return _Tuple(tuple(self._visit_expr(elt, ctx) for elt in e.elts))

    def _visit_assign(self, stmt: Assign, ctx: None):
        ty = self._visit_expr(stmt.expr, ctx)
        self._visit_binding(stmt, stmt.target, ty)

    def _visit_return(self, stmt: ReturnStmt, ctx: None):
        ret_size = self._visit_expr(stmt.expr, ctx)
        if isinstance(ret_size, _Array):
            self.ret_size = ret_size

    def _visit_expr(self, expr: Expr, ctx: None) -> _Array | _Tuple | None:
        ty = super()._visit_expr(expr, ctx)
        if isinstance(ty, _Array):
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
