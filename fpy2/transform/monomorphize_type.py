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
            return Argument(arg.name, AnyTypeAnn(None), arg.loc)
        else:
            ann = self._cvt_arg_type(ty)
            return Argument(arg.name, ann, arg.loc)

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

        subst_ty: dict[VarType, Type] = {}
        for k, v in subst.items():
            subst_ty[VarType(k)] = v

        free_vars = ty_info.fn_type.free_vars()
        for key in subst_ty:
            if key not in free_vars:
                raise ValueError(f'Unbound type variable `{key}` in {func.name} : {ty_info.fn_type.format()}')

        fn_type = ty_info.fn_type.subst(subst_ty)
        assert isinstance(fn_type, FunctionType)
        return _MonomorphizeVisitor(func, fn_type).apply()
