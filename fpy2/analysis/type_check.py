"""
Type checking for FPy programs.

FPy has a simple type system:

    t ::= bool
        | real
        | t1 x t2
        | list t

"""

from typing import TypeAlias

from ..ast import *
from ..utils import NamedId, default_repr
from .define_use import DefineUse, DefineUseAnalysis, Definition
from .live_vars import LiveVars

#####################################################################
# Abstract type

@default_repr
class Type:
    """Base class for all FPy types."""
    pass

_Type: TypeAlias = Type | NamedId
"""type: either a type annotation or type variable"""

_TCtx: TypeAlias = dict[NamedId, _Type]
"""typing context: mapping from variable to type"""

#####################################################################
# Types

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

class TupleType(Type):
    """Tuple type."""

    elt_types: tuple[_Type, ...]
    """type of elements"""

    def __init__(self, *elts: _Type):
        self.elt_types = elts

    def __eq__(self, other):
        return isinstance(other, TupleType) and self.elt_types == other.elt_types

    def __hash__(self):
        return hash(self.elt_types)

class ListType(Type):
    """List type."""

    elt_type: _Type
    """element type"""

    def __init__(self, elt: _Type):
        self.elt_type = elt

    def __eq__(self, other):
        return isinstance(other, ListType) and self.elt_type == other.elt_type

    def __hash__(self):
        return hash(self.elt_type)

class FunctionType(Type):
    """Function type."""

    arg_types: tuple[_Type, ...]
    """argument types"""

    return_type: _Type
    """return type"""

    def __init__(self, arg_types: tuple[_Type, ...], return_type: _Type):
        self.arg_types = arg_types
        self.return_type = return_type

    def __eq__(self, other):
        return isinstance(other, FunctionType) and self.arg_types == other.arg_types and self.return_type == other.return_type

    def __hash__(self):
        return hash((self.arg_types, self.return_type))


#####################################################################
# Type Inference

class _TypeCheckInstance(Visitor):
    """Single-use instance of type checking."""

    func: FuncDef
    def_use: DefineUseAnalysis

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis):
        self.func = func
        self.def_use = def_use

    def analyze(self):
        return self._visit_function(self.func, None)

    def _visit_var(self, e, ctx):
        raise NotImplementedError

    def _visit_bool(self, e, ctx):
        raise NotImplementedError

    def _visit_foreign(self, e, ctx):
        raise NotImplementedError

    def _visit_decnum(self, e, ctx):
        raise NotImplementedError

    def _visit_hexnum(self, e, ctx):
        raise NotImplementedError

    def _visit_integer(self, e, ctx):
        raise NotImplementedError

    def _visit_rational(self, e, ctx):
        raise NotImplementedError

    def _visit_digits(self, e, ctx):
        raise NotImplementedError

    def _visit_nullaryop(self, e, ctx):
        raise NotImplementedError

    def _visit_unaryop(self, e, ctx):
        raise NotImplementedError

    def _visit_binaryop(self, e, ctx):
        raise NotImplementedError

    def _visit_ternaryop(self, e, ctx):
        raise NotImplementedError

    def _visit_naryop(self, e, ctx):
        raise NotImplementedError

    def _visit_compare(self, e, ctx):
        raise NotImplementedError

    def _visit_call(self, e, ctx):
        raise NotImplementedError

    def _visit_tuple_expr(self, e, ctx):
        raise NotImplementedError

    def _visit_list_expr(self, e, ctx):
        raise NotImplementedError

    def _visit_list_comp(self, e, ctx):
        raise NotImplementedError

    def _visit_list_ref(self, e, ctx):
        raise NotImplementedError

    def _visit_list_slice(self, e, ctx):
        raise NotImplementedError

    def _visit_list_set(self, e, ctx):
        raise NotImplementedError

    def _visit_if_expr(self, e, ctx):
        raise NotImplementedError

    def _visit_context_expr(self, e, ctx):
        raise NotImplementedError

    def _visit_assign(self, stmt, ctx):
        raise NotImplementedError

    def _visit_indexed_assign(self, stmt, ctx):
        raise NotImplementedError

    def _visit_if1(self, stmt, ctx):
        raise NotImplementedError

    def _visit_if(self, stmt, ctx):
        raise NotImplementedError

    def _visit_while(self, stmt, ctx):
        raise NotImplementedError

    def _visit_for(self, stmt, ctx):
        raise NotImplementedError

    def _visit_context(self, stmt, ctx):
        raise NotImplementedError

    def _visit_assert(self, stmt, ctx):
        raise NotImplementedError

    def _visit_effect(self, stmt, ctx):
        raise NotImplementedError

    def _visit_return(self, stmt, ctx):
        raise NotImplementedError

    def _visit_block(self, block, ctx):
        raise NotImplementedError

    def _visit_function(self, func, ctx):
        raise NotImplementedError



class TypeCheck:
    """
    Type checker for the FPy language.

    Unlike Python, FPy is statically typed.

    When the `@fpy` decorator runs, it also type checks the function
    and raises an error if the function is not well-typed.
    """

    @staticmethod
    def check(func: FuncDef):
        """
        Analyzes the function for type errors.

        Produces a type signature for the function if it is well-typed.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'expected a \'FuncDef\', got {func}')

        def_use = DefineUse.analyze(func)
        inst = _TypeCheckInstance(func, def_use)
        ty = inst.analyze()
        print(func.name, ty)
