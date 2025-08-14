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

from typing import Sequence

from .utils import NamedId, default_repr

@default_repr
class Type:
    """Base class for all FPy types."""
    pass

class NullType(Type):
    """Placeholder for an ill-typed value."""

    def __eq__(self, other):
        return isinstance(other, NullType)

    def __hash__(self):
        return hash(type(self))

class BoolType(Type):
    """Type of boolean values"""

    def __eq__(self, other):
        return isinstance(other, BoolType)

    def __hash__(self):
        return hash(type(self))

class RealType(Type):
    """Real number type."""

    def __eq__(self, other):
        return isinstance(other, RealType)

    def __hash__(self):
        return hash(type(self))

class VarType(Type):
    """Type variable"""

    name: NamedId
    """identifier"""

    def __eq__(self, other):
        return isinstance(other, VarType) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

class TupleType(Type):
    """Tuple type."""

    elt_types: tuple[Type, ...]
    """type of elements"""

    def __init__(self, *elts: Type):
        self.elt_types = elts

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

    def __eq__(self, other):
        return isinstance(other, FunctionType) and self.arg_types == other.arg_types and self.return_type == other.return_type

    def __hash__(self):
        return hash((self.arg_types, self.return_type))
