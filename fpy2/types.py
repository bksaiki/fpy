"""
FPy types:

FPy has a simple type system.

    t ::= bool
        | real
        | t1 x t2
        | list t
        | t1 -> t2
        | a

There are boolean and real number scalar types, (heterogenous) tuples,
and (homogenous) lists, function types, and type variables.
"""

from abc import ABC, abstractmethod
from typing import Sequence

from .utils import NamedId, default_repr

@default_repr
class Type(ABC):
    """Base class for all FPy types."""

    @abstractmethod
    def format(self) -> str:
        """Returns this type as a formatted string."""
        ...

    @abstractmethod
    def free_vars(self) -> set['VarType']:
        """Returns the free type variables in the type."""
        ...

    @abstractmethod
    def subst(self, subst: dict['VarType', 'Type']) -> 'Type':
        """Substitutes type variables in the type."""
        ...


class NullType(Type):
    """Placeholder for an ill-typed value."""

    def format(self) -> str:
        return "⊥"

    def free_vars(self) -> set['VarType']:
        return set()

    def subst(self, subst: dict['VarType', 'Type']) -> 'Type':
        return self

    def __eq__(self, other):
        return isinstance(other, NullType)

    def __hash__(self):
        return hash(type(self))

class BoolType(Type):
    """Type of boolean values"""

    def format(self) -> str:
        return "bool"

    def free_vars(self) -> set['VarType']:
        return set()

    def subst(self, subst: dict['VarType', 'Type']) -> 'Type':
        return self

    def __eq__(self, other):
        return isinstance(other, BoolType)

    def __hash__(self):
        return hash(type(self))

class RealType(Type):
    """Real number type."""

    def format(self) -> str:
        return "real"

    def free_vars(self) -> set['VarType']:
        return set()

    def subst(self, subst: dict['VarType', 'Type']) -> 'Type':
        return self

    def __eq__(self, other):
        return isinstance(other, RealType)

    def __hash__(self):
        return hash(type(self))

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

    def format(self):
        return self.name

    def free_vars(self) -> set['VarType']:
        return {self}

    def subst(self, subst: dict['VarType', 'Type']) -> 'Type':
        return subst.get(self, self)

class TupleType(Type):
    """Tuple type."""

    elt_types: tuple[Type, ...]
    """type of elements"""

    def __init__(self, *elts: Type):
        self.elt_types = elts

    def format(self) -> str:
        return f'tuple[{", ".join(elt.format() for elt in self.elt_types)}]'

    def free_vars(self) -> set['VarType']:
        fvs: set[VarType] = set()
        for elt in self.elt_types:
            fvs |= elt.free_vars()
        return fvs

    def subst(self, subst: dict['VarType', 'Type']) -> 'Type':
        return TupleType(*[elt.subst(subst) for elt in self.elt_types])

    def __eq__(self, other):
        return isinstance(other, TupleType) and self.elt_types == other.elt_types

    def __hash__(self):
        return hash(self.elt_types)

class ListType(Type):
    """List type."""

    elt_type: Type
    """element type"""

    def __init__(self, elt: Type):
        self.elt_type = elt

    def format(self) -> str:
        return f'list[{self.elt_type.format()}]'

    def free_vars(self) -> set['VarType']:
        return self.elt_type.free_vars()

    def subst(self, subst: dict['VarType', 'Type']) -> 'Type':
        return ListType(self.elt_type.subst(subst))

    def __eq__(self, other):
        return isinstance(other, ListType) and self.elt_type == other.elt_type

    def __hash__(self):
        return hash(self.elt_type)

class FunctionType(Type):
    """Function type."""

    arg_types: tuple[Type, ...]
    """argument types"""

    return_type: Type
    """return type"""

    def __init__(self, arg_types: Sequence[Type], return_type: Type):
        self.arg_types = tuple(arg_types)
        self.return_type = return_type

    def format(self) -> str:
        return f'function[{", ".join(arg.format() for arg in self.arg_types)}] -> {self.return_type.format()}'

    def free_vars(self) -> set['VarType']:
        fvs: set[VarType] = set()
        for arg in self.arg_types:
            fvs |= arg.free_vars()
        fvs |= self.return_type.free_vars()
        return fvs

    def subst(self, subst: dict['VarType', 'Type']) -> 'Type':
        return FunctionType(
            [arg.subst(subst) for arg in self.arg_types],
            self.return_type.subst(subst)
        )

    def __eq__(self, other):
        return isinstance(other, FunctionType) and self.arg_types == other.arg_types and self.return_type == other.return_type

    def __hash__(self):
        return hash((self.arg_types, self.return_type))
