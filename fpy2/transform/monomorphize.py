"""
Monomorphize pass.

Both type and context monomorphization.
"""

from typing import Collection, Iterable

from ..ast.fpyast import *
from ..ast.visitor import DefaultTransformVisitor
from ..analysis import TypeInfer, TypeAnalysis
from ..fpc_context import FPCoreContext
from ..types import *


class _MonomorphizeVisitor(DefaultTransformVisitor):
    """Monomorphize visitor."""

    func: FuncDef
    ctx: Context | None
    arg_types: list[Type | None]

    def __init__(self, func: FuncDef, ctx: Context | None, arg_types: Iterable[Type | None]):
        self.func = func
        self.ctx = ctx
        self.arg_types = list(arg_types)

    def _cvt_arg_type(self, ty: Type) -> TypeAnn:
        match ty:
            case VarType():
                return AnyTypeAnn(None)
            case BoolType():
                return BoolTypeAnn(None)
            case RealType():
                if isinstance(ty.ctx, NamedId):
                    return RealTypeAnn(None, None)
                else:
                    return RealTypeAnn(ty.ctx, None)
            case ContextType():
                return ContextTypeAnn(None)
            case TupleType():
                elts = [ self._cvt_arg_type(t) for t in ty.elts ]
                return TupleTypeAnn(elts, None)
            case ListType():
                elt = self._cvt_arg_type(ty.elt)
                return ListTypeAnn(elt, None)
            case _:
                raise RuntimeError(f'Unsupported argument type `{ty}`')

    def _merge_annotation(self, a: TypeAnn, b: TypeAnn | None) -> TypeAnn:
        """
        Merge two type annotations `a` and `b`.

        Returns the most specific annotation.
        """

        if b is None:
            return a

        match a, b:
            case AnyTypeAnn(), _:
                return b
            case _, AnyTypeAnn():
                return a
            case BoolTypeAnn(), BoolTypeAnn():
                return a
            case RealTypeAnn(), RealTypeAnn():
                match a.ctx, b.ctx:
                    case None, _:
                        return b
                    case _, None:
                        return a
                    case Context(), Context():
                        if not a.ctx.is_equiv(b.ctx):
                            raise RuntimeError(f'Cannot merge different contexts `{a.ctx}` and `{b.ctx}`')
                        return a
                    case _:
                        raise RuntimeError('unreachable')
            case ContextTypeAnn(), ContextTypeAnn():
                return a
            case TupleTypeAnn(), TupleTypeAnn():
                if len(a.elts) != len(b.elts):
                    raise RuntimeError(f'Cannot merge different tuple types `{a}` and `{b}`')
                elts = [ self._merge_annotation(x, y) for x, y in zip(a.elts, b.elts) ]
                return TupleTypeAnn(elts, None)
            case ListTypeAnn(), ListTypeAnn():
                elt = self._merge_annotation(a.elt, b.elt)
                return ListTypeAnn(elt, None)
            case _:
                raise RuntimeError(f'Cannot merge different types `{a}` and `{b}`')

    def _visit_argument(self, arg: Argument, ty: Type | None):
        ann = None if ty is None else self._cvt_arg_type(ty)
        ann = self._merge_annotation(arg.type, ann)
        return Argument(arg.name, ann, arg.loc)

    def _visit_function(self, func: FuncDef, ctx: None) -> FuncDef:
        match func.ctx, self.ctx:
            case None, _:
                fn_ctx: Context | FPCoreContext | None = self.ctx
            case _, None:
                fn_ctx = func.ctx
            case Context(), Context():
                if not func.ctx.is_equiv(self.ctx):
                    raise RuntimeError(f'Cannot merge different contexts `{func.ctx}` and `{self.ctx}`')
                fn_ctx = func.ctx
            case FPCoreContext(), _:
                fn_ctx = self.ctx
            case _:
                raise RuntimeError(f'unreachable: {func.ctx}, {self.ctx}')

        args = [self._visit_argument(arg, ty) for arg, ty in zip(func.args, self.arg_types)]
        body, _ = self._visit_block(func.body, None)
        meta = FuncMeta(func.free_vars, fn_ctx, func.meta.spec, func.meta.props, func.env)
        return FuncDef(func.name, args, body, meta, loc=func.loc)

    def apply(self):
        return self._visit_function(self.func, None)


class Monomorphize:
    """
    Monomorphize pass.

    This pass overrides type or context variables with more specific types.
    """

    @staticmethod
    def apply(
        func: FuncDef,
        ctx: Context | None,
        subst: dict[NamedId, Type],
        *,
        ty_info: TypeAnalysis | None = None
    ) -> FuncDef:
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {func}')

        if ty_info is None:
            ty_info = TypeInfer.check(func)

        free_vars = ty_info.fn_type.free_type_vars()
        for key in subst:
            if key not in free_vars:
                raise ValueError(f'Unbound type variable `{key}` in {func.name} : {ty_info.fn_type.format()}')

        if (
            isinstance(ctx, Context)
            and isinstance(ty_info.fn_type.ctx, Context)
            and not ctx.is_equiv(ty_info.fn_type.ctx)
        ):
            raise ValueError(f'Conflicting context info: cannot override {ty_info.fn_type.ctx} with {ctx}')

        fn_type = ty_info.fn_type.subst_type(subst)
        assert isinstance(fn_type, FunctionType)
        return _MonomorphizeVisitor(func, ctx, fn_type.arg_types).apply()

    @staticmethod
    def apply_by_arg(
        func: FuncDef,
        ctx: Context | None,
        arg_types: Collection[Type | None],
        *,
        ty_info: TypeAnalysis | None = None
    ):
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got `{func}`')
        if not isinstance(arg_types, Collection):
            raise TypeError(f'Expected \'Collection\', got `{arg_types}`')
        if len(func.args) != len(arg_types):
            raise ValueError(f'Expected {len(func.args)} types, got {len(arg_types)}')

        if ty_info is None:
            ty_info = TypeInfer.check(func)

        ty_subst: dict[NamedId, Type] = {}
        ctx_subst: dict[NamedId, Context] = {}

        def _raise_conflict(curr_ty: Type, new_ty: Type):
            raise ValueError(f'Conflicting type info: cannot override {new_ty.format()} with {curr_ty.format()}')

        def _check_merge(curr_ty: Type, new_ty: Type, a_ty: Type, b_ty: Type):
            match a_ty, b_ty:
                case VarType(), _:
                    if a_ty.name in ty_subst:
                        if ty_subst[a_ty.name] != b_ty:
                            _raise_conflict(curr_ty, new_ty)
                    else:
                        ty_subst[a_ty.name] = b_ty
                case BoolType(), BoolType():
                    pass
                case RealType(), RealType():
                    # TODO: how should we handle context merging?
                    match a_ty.ctx, b_ty.ctx:
                        case NamedId(), Context():
                            if a_ty.ctx in ctx_subst:
                                if not ctx_subst[a_ty.ctx].is_equiv(b_ty.ctx):
                                    raise ValueError(f'Conflicting context info: cannot override {new_ty} with {curr_ty}')
                            else:
                                ctx_subst[a_ty.ctx] = b_ty.ctx
                        case _:
                            pass
                case ContextType(), ContextType():
                    pass
                case TupleType(), TupleType():
                    if len(a_ty.elts) != len(b_ty.elts):
                        _raise_conflict(curr_ty, new_ty)
                    for a_elt, b_elt in zip(a_ty.elts, b_ty.elts):
                        _check_merge(curr_ty, new_ty, a_elt, b_elt)
                case ListType(), ListType():
                    _check_merge(curr_ty, new_ty, a_ty.elt, b_ty.elt)
                case _:
                    _raise_conflict(curr_ty, new_ty)

        if (
            isinstance(ctx, Context)
            and isinstance(ty_info.fn_type.ctx, Context)
            and not ctx.is_equiv(ty_info.fn_type.ctx)
        ):
            raise ValueError(f'Conflicting context info: cannot override {ty_info.fn_type.ctx} with {ctx}')

        new_arg_types: list[Type | None] = []
        for curr_ty, new_ty in zip(ty_info.arg_types, arg_types):
            if new_ty is None:
                new_arg_types.append(None)
            else:
                if new_ty is not None:
                    _check_merge(curr_ty, new_ty, curr_ty, new_ty)
                new_arg_types.append(new_ty)

        func = _MonomorphizeVisitor(func, ctx, new_arg_types).apply()
        return func
