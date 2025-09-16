"""
Context types are used for context inference.

Context types extends FPy standard type system with inferred rounding contexts.

    t' ::= bool
         | real C
         | context
         | t1' x t2'
         | list t'
         | [C] t1' -> t2'
         | a

where C is an inferred context variable or a context variable.

Compared to the standard FPy type system, the differences are:
- real types are annotated with a context to indicate
the rounding context under which the number is constructed
- function types have a caller context to indicate
the context in which the function is called (this is usually a variable).
"""

from abc import ABC, abstractmethod
from typing import Iterable

from ..number import Context
from ..utils import NamedId, default_repr
from .types import *

@default_repr
class TypeContext(ABC):
    """Base class for all FPy context types."""

    @abstractmethod
    @classmethod
    def from_type(self, ty: Type, ctx: Context | NamedId | None):
        """Constructs a context type from a standard FPy type and a context."""
        ...

    @abstractmethod
    def as_type(self) -> Type:
        """Converts this context type to a standard FPy type by erasing contexts."""
        ...

    @abstractmethod
    def format(self) -> str:
        """Returns this type as a formatted string."""
        ...

    @abstractmethod
    def free_vars(self) -> set[NamedId]:
        """Returns the free context variables in the type."""
        ...

    @abstractmethod
    def subst(self, subst: dict[NamedId, Context | NamedId]) -> 'TypeContext':
        """Substitutes context variables in the type."""
        ...

    def _subst(self, ctx: Context | NamedId, subst: dict[NamedId, Context | NamedId]) -> Context | NamedId:
        if isinstance(ctx, NamedId) and ctx in subst:
            return subst[ctx]
        else:
            return ctx


class VarTypeContext(TypeContext):
    """Type variable"""

    name: NamedId
    """identifier"""

    def __init__(self, name: NamedId):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, VarTypeContext) and self.name == other.name

    def __lt__(self, other: 'VarTypeContext'):
        if not isinstance(other, VarTypeContext):
            raise TypeError(f"'<' not supported between instances '{type(self)}' and '{type(other)}'")
        return self.name < other.name

    def __hash__(self):
        return hash(self.name)

    @classmethod
    def from_type(self, ty: Type, ctx: Context | NamedId | None):
        if not isinstance(ty, VarType):
            raise TypeError(f'expected a \'VarType\', got {ty}')
        if ctx is not None:
            raise ValueError(f'cannot attach context to type variable {ty}')
        return VarTypeContext(ty.name)

    def as_type(self):
        return VarType(self.name)

    def format(self) -> str:
        return str(self.name)

    def free_vars(self) -> set[NamedId]:
        return set()

    def subst(self, subst: dict[NamedId, Context | NamedId]) -> TypeContext:
        return self


class BoolTypeContext(TypeContext):
    """Type of boolean values"""

    @classmethod
    def from_type(self, ty: Type, ctx: Context | NamedId | None):
        if not isinstance(ty, BoolType):
            raise TypeError(f'expected a \'BoolType\', got {ty}')
        if ctx is not None:
            raise ValueError(f'cannot attach context to boolean type {ty}')
        return BoolTypeContext()

    def as_type(self):
        return BoolType()

    def format(self) -> str:
        return "bool"

    def free_vars(self) -> set[NamedId]:
        return set()

    def subst(self, subst: dict[NamedId, Context | NamedId]) -> TypeContext:
        return self


class RealTypeContext(TypeContext):
    """Type of real numbers with an associated context."""

    ctx: Context | NamedId
    """Rounding context"""

    def __init__(self, ctx: Context | NamedId):
        self.ctx = ctx

    @classmethod
    def from_type(self, ty: Type, ctx: Context | NamedId | None):
        if not isinstance(ty, RealType):
            raise TypeError(f'expected a \'RealType\', got {ty}')
        if ctx is None:
            raise ValueError(f'missing context for real type {ty}')
        return RealTypeContext(ctx)

    def as_type(self):
        return RealType()

    def format(self) -> str:
        return f"real[{self.ctx}]"

    def free_vars(self) -> set[NamedId]:
        if isinstance(self.ctx, NamedId):
            return { self.ctx }
        else:
            return set()

    def subst(self, subst: dict[NamedId, Context | NamedId]) -> TypeContext:
        return RealTypeContext(self._subst(self.ctx, subst))


class ContextTypeContext(TypeContext):
    """Type of rounding contexts."""

    @classmethod
    def from_type(self, ty: Type, ctx: Context | NamedId | None):
        if not isinstance(ty, ContextType):
            raise TypeError(f'expected a \'ContextType\', got {ty}')
        if ctx is not None:
            raise ValueError(f'cannot attach context to context type {ty}')
        return ContextTypeContext()

    def as_type(self):
        return ContextType()

    def format(self) -> str:
        return "context"

    def free_vars(self) -> set[NamedId]:
        return set()

    def subst(self, subst: dict[NamedId, Context | NamedId]) -> TypeContext:
        return self


class TupleTypeContext(TypeContext):
    """Tuple type."""

    elts: tuple[TypeContext, ...]
    """type of elements"""

    def __init__(self, *elts: TypeContext):
        self.elts = elts

    @classmethod
    def from_type(self, ty: Type, ctx: Context | NamedId | None):
        if not isinstance(ty, TupleType):
            raise TypeError(f'expected a \'TupleType\', got {ty}')
        if ctx is not None:
            raise ValueError(f'cannot attach context to tuple type {ty}')
        return TupleTypeContext(*[TypeContext.from_type(elt, None) for elt in ty.elts])

    def as_type(self):
        return TupleType(*[elt.as_type() for elt in self.elts])

    def format(self) -> str:
        return f'tuple[{", ".join(elt.format() for elt in self.elts)}]'

    def free_vars(self) -> set[NamedId]:
        fvs: set[NamedId] = set()
        for elt in self.elts:
            fvs |= elt.free_vars()
        return fvs

    def subst(self, subst: dict[NamedId, Context | NamedId]) -> TypeContext:
        return TupleTypeContext(*[elt.subst(subst) for elt in self.elts])


class ListTypeContext(TypeContext):
    """List type."""

    elt: TypeContext
    """type of elements"""

    def __init__(self, elt: TypeContext):
        self.elt = elt

    @classmethod
    def from_type(self, ty: Type, ctx: Context | NamedId | None):
        if not isinstance(ty, ListType):
            raise TypeError(f'expected a \'ListType\', got {ty}')
        if ctx is not None:
            raise ValueError(f'cannot attach context to list type {ty}')
        return ListTypeContext(TypeContext.from_type(ty.elt, None))

    def as_type(self):
        return ListType(self.elt.as_type())

    def format(self) -> str:
        return f'list[{self.elt.format()}]'

    def free_vars(self) -> set[NamedId]:
        return self.elt.free_vars()

    def subst(self, subst: dict[NamedId, Context | NamedId]) -> TypeContext:
        return ListTypeContext(self.elt.subst(subst))


class FunctionTypeContext(TypeContext):
    """Function type with caller context."""

    ctx: Context | NamedId
    """caller context"""

    arg_types: tuple[TypeContext, ...]
    """argument types"""

    ret_type: TypeContext
    """return type"""

    def __init__(self, ctx: Context | NamedId, arg_types: Iterable[TypeContext], ret_type: TypeContext):
        self.ctx = ctx
        self.arg_types = tuple(arg_types)
        self.ret_type = ret_type

    @classmethod
    def from_type(self, ty: Type, ctx: Context | NamedId | None):
        if not isinstance(ty, FunctionType):
            raise TypeError(f'expected a \'FunctionType\', got {ty}')
        if ctx is None:
            raise ValueError(f'missing caller context for function type {ty}')
        return FunctionTypeContext(
            ctx,
            [TypeContext.from_type(arg_ty, None) for arg_ty in ty.arg_types],
            TypeContext.from_type(ty.return_type, None)
        )

    def as_type(self):
        return FunctionType(
            [arg_ty.as_type() for arg_ty in self.arg_types],
            self.ret_type.as_type()
        )

    def format(self) -> str:
        return f'function[{", ".join(arg.format() for arg in self.arg_types)}] -> {self.ret_type.format()} [{self.ctx}]'

    def free_vars(self) -> set[NamedId]:
        fvs: set[NamedId] = set()
        if isinstance(self.ctx, NamedId):
            fvs.add(self.ctx)
        for arg in self.arg_types:
            fvs |= arg.free_vars()
        fvs |= self.ret_type.free_vars()
        return fvs

    def subst(self, subst: dict[NamedId, Context | NamedId]) -> TypeContext:
        return FunctionTypeContext(
            self._subst(self.ctx, subst),
            [arg.subst(subst) for arg in self.arg_types],
            self.ret_type.subst(subst)
        )
