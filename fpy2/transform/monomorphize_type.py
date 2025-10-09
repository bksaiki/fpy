"""
Monomorphize pass.

Both type and context monomorphization.
"""

from ..ast.fpyast import *
from ..ast.visitor import DefaultTransformVisitor
from ..types import *
from ..analysis.type_infer import TypeInfer, TypeAnalysis

class _MonomorphizeVisitor(DefaultTransformVisitor):
    """Monormize visitor."""

    func: FuncDef
    fn_ty: FunctionType

    def __init__(self, func: FuncDef, fn_ty: FunctionType):
        self.func = func
        self.fn_ty = fn_ty

    def _cvt_arg_type(self, ty: Type) -> TypeAnn:
        match ty:
            case VarType():
                raise RuntimeError(f'Unsubstituted type variable `{ty}`')
            case BoolType():
                return BoolTypeAnn(None)
            case RealType():
                return RealTypeAnn(None, None)
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

    def _visit_argument(self, arg: Argument, ty: Type):
        if ty.is_monomorphic():
            ann = self._cvt_arg_type(ty)
            return Argument(arg.name, ann, arg.loc)
        else:
            return Argument(arg.name, AnyTypeAnn(None), arg.loc)

    def _visit_function(self, func: FuncDef, ctx: None) -> FuncDef:
        args = [self._visit_argument(arg, ty) for arg, ty in zip(func.args, self.fn_ty.arg_types)]
        body, _ = self._visit_block(func.body, None)
        return FuncDef(func.name, args, func.free_vars, func.ctx, body, func.spec, func.meta, func.env, loc=func.loc)

    def apply(self):
        return self._visit_function(self.func, None)


class MonomorphizeType:
    """
    Monomorphize pass.

    This pass overrides type or context variables with more specific types.
    """

    @staticmethod
    def apply(
        func: FuncDef,
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

        fn_type = ty_info.fn_type.subst_type(subst)
        assert isinstance(fn_type, FunctionType)
        return _MonomorphizeVisitor(func, fn_type).apply()

    @staticmethod
    def apply_by_arg(
        func: FuncDef,
        arg_types: list[Type | None],
        *,
        ty_info: TypeAnalysis | None = None
    ):
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got `{func}`')
        if not isinstance(arg_types, list):
            raise TypeError(f'Expected \'list\', got `{arg_types}`')
        if len(func.args) != len(arg_types):
            raise ValueError(f'Expected {len(func.args)} types, got {len(arg_types)}')

        if ty_info is None:
            ty_info = TypeInfer.check(func)

        subst: dict[NamedId, Type] = {}

        def _raise_conflict(curr_ty: Type, new_ty: Type):
            raise ValueError(f'Conflicting type info: cannot override {new_ty.format()} with {curr_ty.format()}')

        def _merge(curr_ty: Type, new_ty: Type, a_ty: Type, b_ty: Type):
            match a_ty, b_ty:
                case VarType(), _:
                    if a_ty.name in subst:
                        if subst[a_ty.name] != b_ty:
                            _raise_conflict(curr_ty, new_ty)
                    else:
                        subst[a_ty.name] = b_ty
                case BoolType(), BoolType():
                    pass
                case RealType(), RealType():
                    pass
                case ContextType(), ContextType():
                    pass
                case TupleType(), TupleType():
                    if len(a_ty.elts) != len(b_ty.elts):
                        _raise_conflict(curr_ty, new_ty)
                    for a_elt, b_elt in zip(a_ty.elts, b_ty.elts):
                        _merge(curr_ty, new_ty, a_elt, b_elt)
                case ListType(), ListType():
                    _merge(curr_ty, new_ty, a_ty.elt, b_ty.elt)
                case _:
                    _raise_conflict(curr_ty, new_ty)

        for curr_ty, new_ty in zip(ty_info.arg_types, arg_types):
            if new_ty is not None:
                _merge(curr_ty, new_ty, curr_ty, new_ty)

        fn_type = ty_info.fn_type.subst_type(subst)
        assert isinstance(fn_type, FunctionType)
        return _MonomorphizeVisitor(func, fn_type).apply()
