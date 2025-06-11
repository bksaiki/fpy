"""
FPy types.
"""

from typing import Sequence

from ..utils import NamedId, default_repr
from ..number import Context

@default_repr
class Type:
    """Base class for all FPy types."""
    pass

class ScalarType(Type):
    """Base class for scalar types."""
    pass

class TensorType(Type):
    """Base class for tensor types."""
    pass


class BoolType(ScalarType):
    """Boolean type."""
    def __eq__(self, other):
        return isinstance(other, BoolType)

    def __hash__(self):
        return hash(())

class RealType(ScalarType):
    """Real number type."""
    ctx: Context | NamedId

    def __init__(self, ctx: Context | NamedId):
        self.ctx = ctx

    def __eq__(self, other):
        # TODO: how to check if two contexts are equivalent?
        return isinstance(other, RealType) and self.ctx == other.ctx

    def __hash__(self):
        return hash((self.ctx,))

class TupleType(Type):
    """Tuple type."""
    elts: tuple[Type | NamedId, ...]

    def __init__(self, *elts: Type | NamedId):
        self.elts = elts

    def __eq__(self, other):
        return isinstance(other, TupleType) and self.elts == other.elts

    def __hash__(self):
        return hash(self.elts)

class SizedTensorType(Type):
    dims: list[int | NamedId]
    elt: Type | NamedId

    def __init__(self, dims: Sequence[int | NamedId], elt: Type | NamedId):
        self.dims = list(dims)
        self.elt = elt

    def __eq__(self, other):
        return (isinstance(other, SizedTensorType) and
                self.dims == other.dims and
                self.elt == other.elt)

    def __hash__(self):
        return hash((tuple(self.dims), self.elt))

class FuncType(Type):
    """Function type."""
    ctx: Context | NamedId
    args: tuple[Type | NamedId, ...]
    ret: Type | NamedId

    def __init__(self, ctx: Context | NamedId, args: Sequence[Type | NamedId], ret: Type | NamedId):
        self.ctx = ctx
        self.args = tuple(args)
        self.ret = ret

    def __eq__(self, other):
        return (isinstance(other, FuncType) and
                self.ctx == other.ctx and
                self.args == other.args and
                self.ret == other.ret)

    def __hash__(self):
        return hash((self.ctx, self.args, self.ret))
