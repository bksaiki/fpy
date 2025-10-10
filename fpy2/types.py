"""
FPy types:

FPy has a simple type system.

    t ::= bool
        | real
        | context
        | t1 x t2
        | list t
        | t1 -> t2
        | a

There are boolean and real number scalar types,
rounding contexts, (heterogenous) tuples, (homogenous) lists,
function types, and type variables.

The type system may be extended with statically known rounding contexts.

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

from .number import Context
from .utils import NamedId, default_repr

__all__ = [
    'Type',
    'ContextParam',
    'BoolType',
    'RealType',
    'ContextType',
    'VarType',
    'TupleType',
    'ListType',
    'FunctionType'
]


ContextParam: TypeAlias = NamedId | Context
"""context parameter: either a context variable or a concrete context"""


@default_repr
class Type(ABC):
    """Base class for all FPy types."""

    @abstractmethod
    def is_context_type(self):
        """Does this type also encode rounding context information?"""
        ...

    @abstractmethod
    def format(self) -> str:
        """Returns this type as a formatted string."""
        ...

    @abstractmethod
    def free_type_vars(self) -> set[NamedId]:
        """Returns the free type variables in the type."""
        ...

    @abstractmethod
    def free_context_vars(self) -> set[NamedId]:
        """Returns the free context variables in the type."""
        ...

    @abstractmethod
    def subst_type(self, subst: dict[NamedId, 'Type']) -> 'Type':
        """Substitutes type variables in the type."""
        ...

    @abstractmethod
    def subst_context(self, subst: dict[NamedId, ContextParam]) -> 'Type':
        """Substitutes context variables in the type."""
        ...

    def is_monomorphic(self) -> bool:
        """The type has no free variables."""
        return not self.free_type_vars()


class VarType(Type):
    """Type variable"""

    name: NamedId
    """identifier"""

    def __init__(self, name: NamedId):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, VarType) and self.name == other.name

    def __lt__(self, other: 'VarType'):
        if not isinstance(other, VarType):
            raise TypeError(f"'<' not supported between instances '{type(self)}' and '{type(other)}'")
        return self.name < other.name

    def __hash__(self):
        return hash(self.name)

    def is_context_type(self):
        return True

    def format(self) -> str:
        return str(self.name)

    def free_type_vars(self) -> set[NamedId]:
        return { self.name }

    def free_context_vars(self) -> set[NamedId]:
        return set()

    def subst_type(self, subst: dict[NamedId, Type]) -> Type:
        return subst.get(self.name, self)

    def subst_context(self, subst: dict[NamedId, ContextParam]) -> Type:
        return self


class BoolType(Type):
    """Type of boolean values"""

    def __eq__(self, other):
        return isinstance(other, BoolType)

    def __hash__(self):
        return hash(type(self))

    def is_context_type(self):
        return True

    def format(self) -> str:
        return "bool"

    def free_type_vars(self) -> set[NamedId]:
        return set()

    def free_context_vars(self) -> set[NamedId]:
        return set()

    def subst_type(self, subst: dict[NamedId, Type]) -> Type:
        return self

    def subst_context(self, subst: dict[NamedId, ContextParam]) -> Type:
        return self


class RealType(Type):
    """Real number type."""

    ctx: ContextParam | None

    def __init__(self, ctx: ContextParam | None):
        self.ctx = ctx

    def __eq__(self, other):
        return isinstance(other, RealType)

    def __hash__(self):
        return hash(type(self))

    def is_context_type(self):
        return self.ctx is not None

    def format(self) -> str:
        if self.ctx is None:
            return "real"
        else:
            return f"real[{self.ctx}]"

    def free_type_vars(self) -> set[NamedId]:
        return set()

    def free_context_vars(self) -> set[NamedId]:
        if isinstance(self.ctx, NamedId):
            return { self.ctx }
        else:
            return set()

    def subst_type(self, subst: dict[NamedId, Type]) -> Type:
        return self

    def subst_context(self, subst: dict[NamedId, ContextParam]) -> Type:
        ctx = subst.get(self.ctx, self.ctx) if isinstance(self.ctx, NamedId) else self.ctx
        return RealType(ctx)


class ContextType(Type):
    """Rounding context type."""

    def __eq__(self, other):
        return isinstance(other, ContextType)

    def __hash__(self):
        return hash(type(self))

    def is_context_type(self):
        return True

    def format(self) -> str:
        return "context"

    def free_type_vars(self) -> set[NamedId]:
        return set()

    def free_context_vars(self) -> set[NamedId]:
        return set()

    def subst_type(self, subst: dict[NamedId, Type]) -> Type:
        return self

    def subst_context(self, subst: dict[NamedId, ContextParam]) -> Type:
        return self


class TupleType(Type):
    """Tuple type."""

    elts: tuple[Type, ...]
    """type of elements"""

    def __init__(self, *elts: Type):
        self.elts = elts

    def __eq__(self, other):
        return isinstance(other, TupleType) and self.elts == other.elts

    def __hash__(self):
        return hash(self.elts)

    def is_context_type(self):
        return all(elt.is_context_type() for elt in self.elts)

    def format(self) -> str:
        return f'tuple[{", ".join(elt.format() for elt in self.elts)}]'

    def free_type_vars(self) -> set[NamedId]:
        fvs: set[NamedId] = set()
        for elt in self.elts:
            fvs |= elt.free_type_vars()
        return fvs

    def free_context_vars(self) -> set[NamedId]:
        fvs: set[NamedId] = set()
        for elt in self.elts:
            fvs |= elt.free_context_vars()
        return fvs

    def subst_type(self, subst: dict[NamedId, Type]) -> Type:
        return TupleType(*[elt.subst_type(subst) for elt in self.elts])

    def subst_context(self, subst: dict[NamedId, ContextParam]) -> Type:
        return TupleType(*[elt.subst_context(subst) for elt in self.elts])


class ListType(Type):
    """List type."""

    elt: Type
    """element type"""

    def __init__(self, elt: Type):
        self.elt = elt

    def __eq__(self, other):
        return isinstance(other, ListType) and self.elt == other.elt

    def __hash__(self):
        return hash(self.elt)

    def is_context_type(self):
        return self.elt.is_context_type()

    def format(self) -> str:
        return f'list[{self.elt.format()}]'

    def free_type_vars(self) -> set[NamedId]:
        return self.elt.free_type_vars()

    def free_context_vars(self) -> set[NamedId]:
        return self.elt.free_context_vars()

    def subst_type(self, subst: dict[NamedId, Type]) -> Type:
        return ListType(self.elt.subst_type(subst))

    def subst_context(self, subst: dict[NamedId, ContextParam]) -> Type:
        return ListType(self.elt.subst_context(subst))


class FunctionType(Type):
    """Function type."""

    ctx: ContextParam | None
    """caller context"""

    arg_types: tuple[Type, ...]
    """argument types"""

    return_type: Type
    """return type"""

    def __init__(self, ctx: ContextParam | None, arg_types: Iterable[Type], return_type: Type):
        self.ctx = ctx
        self.arg_types = tuple(arg_types)
        self.return_type = return_type

    def __eq__(self, other):
        return (
            isinstance(other, FunctionType)
            and self.ctx == other.ctx
            and self.arg_types == other.arg_types
            and self.return_type == other.return_type
        )

    def __hash__(self):
        return hash((self.ctx, self.arg_types, self.return_type))

    def is_context_type(self):
        return (
            self.ctx is not None
            and all(arg.is_context_type() for arg in self.arg_types)
            and self.return_type.is_context_type()
        )

    def format(self) -> str:
        if len(self.arg_types) == 0:
            raw_fmt = '() -> ' + self.return_type.format()
        else:
            raw_fmt = ' -> '.join([arg.format() for arg in self.arg_types] + [self.return_type.format()])

        if self.ctx is None:
            return raw_fmt
        else:
            return f'[{self.ctx}] {raw_fmt}'

    def free_type_vars(self) -> set[NamedId]:
        fvs: set[NamedId] = set()
        for arg in self.arg_types:
            fvs |= arg.free_type_vars()
        fvs |= self.return_type.free_type_vars()
        return fvs

    def free_context_vars(self) -> set[NamedId]:
        fvs: set[NamedId] = set()
        if isinstance(self.ctx, NamedId):
            fvs.add(self.ctx)
        for arg in self.arg_types:
            fvs |= arg.free_context_vars()
        fvs |= self.return_type.free_context_vars()
        return fvs

    def subst_type(self, subst: dict[NamedId, Type]) -> Type:
        arg_types = [arg.subst_type(subst) for arg in self.arg_types]
        return_type = self.return_type.subst_type(subst)
        return FunctionType(self.ctx, arg_types, return_type)

    def subst_context(self, subst: dict[NamedId, ContextParam]) -> Type:
        ctx = subst.get(self.ctx, self.ctx) if isinstance(self.ctx, NamedId) else self.ctx
        arg_types = [arg.subst_context(subst) for arg in self.arg_types]
        return_type = self.return_type.subst_context(subst)
        return FunctionType(ctx, arg_types, return_type)
