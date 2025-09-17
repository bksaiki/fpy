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
from typing import Iterable, TypeAlias

from ..number import Context
from ..utils import NamedId, default_repr
from .types import *

__all__ = [
    'TypeContext',
    'VarTypeContext',
    'BoolTypeContext',
    'RealTypeContext',
    'ContextTypeContext',
    'TupleTypeContext',
    'ListTypeContext',
    'FunctionTypeContext',
    'ContextParam'
]


ContextParam: TypeAlias = NamedId | Context

@default_repr
class TypeContext(ABC):
    """Base class for all FPy context types."""

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
    def subst(self, subst: dict[NamedId, ContextParam]) -> 'TypeContext':
        """Substitutes context variables in the type."""
        ...

    def _subst(self, ctx: ContextParam, subst: dict[NamedId, ContextParam]) -> ContextParam:
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

    def as_type(self):
        return VarType(self.name)

    def format(self) -> str:
        return str(self.name)

    def free_vars(self) -> set[NamedId]:
        return set()

    def subst(self, subst: dict[NamedId, ContextParam]) -> TypeContext:
        return self


class BoolTypeContext(TypeContext):
    """Type of boolean values"""

    def as_type(self):
        return BoolType()

    def format(self) -> str:
        return "bool"

    def free_vars(self) -> set[NamedId]:
        return set()

    def subst(self, subst: dict[NamedId, ContextParam]) -> TypeContext:
        return self


class RealTypeContext(TypeContext):
    """Type of real numbers with an associated context."""

    ctx: ContextParam
    """Rounding context"""

    def __init__(self, ctx: ContextParam):
        self.ctx = ctx

    def as_type(self):
        return RealType()

    def format(self) -> str:
        return f"real[{self.ctx}]"

    def free_vars(self) -> set[NamedId]:
        if isinstance(self.ctx, NamedId):
            return { self.ctx }
        else:
            return set()

    def subst(self, subst: dict[NamedId, ContextParam]) -> TypeContext:
        return RealTypeContext(self._subst(self.ctx, subst))


class ContextTypeContext(TypeContext):
    """Type of rounding contexts."""

    def as_type(self):
        return ContextType()

    def format(self) -> str:
        return "context"

    def free_vars(self) -> set[NamedId]:
        return set()

    def subst(self, subst: dict[NamedId, ContextParam]) -> TypeContext:
        return self


class TupleTypeContext(TypeContext):
    """Tuple type."""

    elts: tuple[TypeContext, ...]
    """type of elements"""

    def __init__(self, *elts: TypeContext):
        self.elts = elts

    def as_type(self):
        return TupleType(*[elt.as_type() for elt in self.elts])

    def format(self) -> str:
        return f'tuple[{", ".join(elt.format() for elt in self.elts)}]'

    def free_vars(self) -> set[NamedId]:
        fvs: set[NamedId] = set()
        for elt in self.elts:
            fvs |= elt.free_vars()
        return fvs

    def subst(self, subst: dict[NamedId, ContextParam]) -> TypeContext:
        return TupleTypeContext(*[elt.subst(subst) for elt in self.elts])


class ListTypeContext(TypeContext):
    """List type."""

    elt: TypeContext
    """type of elements"""

    def __init__(self, elt: TypeContext):
        self.elt = elt

    def as_type(self):
        return ListType(self.elt.as_type())

    def format(self) -> str:
        return f'list[{self.elt.format()}]'

    def free_vars(self) -> set[NamedId]:
        return self.elt.free_vars()

    def subst(self, subst: dict[NamedId, ContextParam]) -> TypeContext:
        return ListTypeContext(self.elt.subst(subst))


class FunctionTypeContext(TypeContext):
    """Function type with caller context."""

    ctx: ContextParam
    """caller context"""

    args: tuple[TypeContext, ...]
    """argument types"""

    ret: TypeContext
    """return type"""

    def __init__(self, ctx: ContextParam, args: Iterable[TypeContext], ret: TypeContext):
        self.ctx = ctx
        self.args = tuple(args)
        self.ret = ret

    def as_type(self):
        return FunctionType(
            [arg_ty.as_type() for arg_ty in self.args],
            self.ret.as_type()
        )

    def format(self) -> str:
        return f'function[{", ".join(arg.format() for arg in self.args)}] -> {self.ret.format()} [{self.ctx}]'

    def free_vars(self) -> set[NamedId]:
        fvs: set[NamedId] = set()
        if isinstance(self.ctx, NamedId):
            fvs.add(self.ctx)
        for arg in self.args:
            fvs |= arg.free_vars()
        fvs |= self.ret.free_vars()
        return fvs

    def subst(self, subst: dict[NamedId, ContextParam]) -> TypeContext:
        return FunctionTypeContext(
            self._subst(self.ctx, subst),
            [arg.subst(subst) for arg in self.args],
            self.ret.subst(subst)
        )
