"""
This module contains the AST for FPy programs.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Self, Sequence
from fractions import Fraction

from ..fpc_context import FPCoreContext
from ..number import Context
from ..utils import (
    CompareOp,
    Id, NamedId, UnderscoreId,
    Location,
    default_repr,
    decnum_to_fraction, hexnum_to_fraction, digits_to_fraction
)

__all__ = [
    # Re-exports
    'Id',
    'NamedId',
    'UnderscoreId',
    'CompareOp',
    'Location',
    'Context',

    # Base classes
    'Ast',
    'TypeAnn',
    'Expr',
    'Stmt',
    'ValueExpr',
    'NaryExpr',

    # Type annotations
    'AnyTypeAnn',
    'ScalarTypeAnn',
    'RealTypeAnn',
    'BoolTypeAnn',
    'TensorTypeAnn',
    'TupleTypeAnn',
    'SizedTensorTypeAnn',

    # Value expressions
    'Var',
    'BoolVal',
    'RealVal',
    'RationalVal',
    'Decnum',
    'Hexnum',
    'Integer',
    'Rational',
    'Digits',
    'ForeignVal',

    # N-ary operations
    'NullaryOp',
    'UnaryOp',
    'NamedUnaryOp',
    'BinaryOp',
    'NamedBinaryOp',
    'TernaryOp',
    'NamedTernaryOp',
    'NaryOp',
    'NamedNaryOp',

    # Constants
    'ConstNan',
    'ConstInf',
    'ConstPi',
    'ConstE',
    'ConstLog2E',
    'ConstLog10E',
    'ConstLn2',
    'ConstPi_2',
    'ConstPi_4',
    'Const1_Pi',
    'Const2_Pi',
    'Const2_SqrtPi',
    'ConstSqrt2',
    'ConstSqrt1_2',

    # IEEE 754 arithmetic
    'Add',
    'Sub',
    'Mul',
    'Div',
    'Fabs',
    'Sqrt',
    'Fma',

    # Sign operations
    'Neg',
    'Copysign',

    # Composite arithmetic
    'Fdim',
    'Hypot',

    # Other arithmetic
    'Max',
    'Min',
    'Fmod',
    'Remainder',
    'Cbrt',
    'Sum',

    # Rounding and truncation
    'Ceil',
    'Floor',
    'NearbyInt',
    'RoundInt',
    'Trunc',

    # Trigonometric functions
    'Acos',
    'Asin',
    'Atan',
    'Atan2',
    'Cos',
    'Sin',
    'Tan',

    # Hyperbolic functions
    'Acosh',
    'Asinh',
    'Atanh',
    'Cosh',
    'Sinh',
    'Tanh',

    # Exponential/logarithmic functions
    'Exp',
    'Exp2',
    'Expm1',
    'Log',
    'Log10',
    'Log1p',
    'Log2',
    'Pow',

    # Integral functions
    'Erf',
    'Erfc',
    'Lgamma',
    'Tgamma',

    # Classification
    'IsFinite',
    'IsInf',
    'IsNan',
    'IsNormal',
    'Signbit',

    # Logical operators
    'Not',
    'Or',
    'And',

    # Rounding operator
    'Round',
    'RoundExact',

    # Tensor operators
    'Range',
    'Dim',
    'Size',
    'Zip',
    'Enumerate',

    # Other expressions
    'Call',
    'Compare',
    'TupleExpr',
    'ListExpr',
    'TupleBinding',
    'ListComp',
    'ListSet',
    'ListRef',
    'ListSlice',
    'IfExpr',
    'ForeignAttribute',
    'ContextExpr',
    'ContextAttribute',
    'ContextUpdate',

    # Statements
    'StmtBlock',
    'Assign',
    'IndexedAssign',
    'If1Stmt',
    'IfStmt',
    'WhileStmt',
    'ForStmt',
    'ContextStmt',
    'AssertStmt',
    'EffectStmt',
    'ReturnStmt',

    # Function definition
    'Argument',
    'FuncDef',

    # Formatter
    'BaseFormatter',
    'get_default_formatter',
    'set_default_formatter'
]


@default_repr
class Ast(ABC):
    """FPy AST: abstract base class for all AST nodes."""
    __slots__ = ('_loc',)
    _loc: Optional[Location]

    def __init__(self, loc: Optional[Location]):
        self._loc = loc

    @property
    def loc(self):
        """Get the location of the AST node."""
        return self._loc

    def format(self) -> str:
        """Format the AST node as a string."""
        formatter = get_default_formatter()
        return formatter.format(self)


class TypeAnn(Ast):
    """FPy AST: typing annotation"""
    __slots__ = ()

    def __init__(self, loc: Optional[Location]):
        super().__init__(loc)

class AnyTypeAnn(TypeAnn):
    """FPy AST: any type annotation"""
    __slots__ = ()

    def __init__(self, loc: Optional[Location]):
        super().__init__(loc)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AnyTypeAnn)

    def __hash__(self) -> int:
        return hash(())

class ScalarTypeAnn(TypeAnn):
    """FPy AST: scalar type annotation"""
    __slots__ = ()

    def __init__(self, loc: Optional[Location]):
        super().__init__(loc)

class RealTypeAnn(TypeAnn):
    """FPy AST: real type annotation"""
    __slots__ = ()

    def __init__(self, loc: Optional[Location]):
        super().__init__(loc)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RealTypeAnn)

    def __hash__(self) -> int:
        return hash(())

class BoolTypeAnn(TypeAnn):
    """FPy AST: boolean type annotation"""
    __slots__ = ()

    def __init__(self, loc: Optional[Location]):
        super().__init__(loc)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BoolTypeAnn)

    def __hash__(self) -> int:
        return hash(())

class TensorTypeAnn(TypeAnn):
    """FPy AST: tensor type annotation"""
    __slots__ = ()

    def __init__(self, loc: Optional[Location]):
        super().__init__(loc)

class TupleTypeAnn(TensorTypeAnn):
    """FPy AST: native tuple type annotation"""
    __slots__ = ('elts',)
    elts: tuple[TypeAnn, ...]

    def __init__(self, elts: list[TypeAnn], loc: Optional[Location]):
        super().__init__(loc)
        self.elts = tuple(elts)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TupleTypeAnn):
            return False
        return self.elts == other.elts

    def __hash__(self) -> int:
        return hash(self.elts)

class SizedTensorTypeAnn(TensorTypeAnn):
    """FPy AST: sized, homogenous tensor type annotation"""
    __slots__ = ('dims', 'elt')
    dims: tuple[int | NamedId, ...]
    elt: TypeAnn

    def __init__(self, dims: list[int | NamedId], elt: TypeAnn, loc: Optional[Location]):
        super().__init__(loc)
        self.dims = tuple(dims)
        self.elt = elt

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SizedTensorTypeAnn):
            return False
        return self.dims == other.dims and self.elt == other.elt

    def __hash__(self) -> int:
        return hash((self.dims, self.elt))

class ForeignAttribute(Ast):
    """
    FPy AST: attribute of a foreign object, e.g., `x.y`
    Attributes may be nested, e.g., `x.y.z`.
    """
    __slots__ = ('name', 'attrs')
    name: NamedId
    attrs: tuple[NamedId, ...]

    def __init__(self, name: NamedId, attrs: Sequence[NamedId], loc: Optional[Location]):
        super().__init__(loc)
        self.name = name
        self.attrs = tuple(attrs)

    def __eq__(self, other):
        return (
            isinstance(other, ForeignAttribute)
            and self.name == other.name
            and len(self.attrs) == len(other.attrs)
            and all(a == b for a, b in zip(self.attrs, other.attrs))
        )

    def __hash__(self):
        return hash((self.name, self.attrs))

    def is_equiv(self, other) -> bool:
        return self == other

class Expr(Ast):
    """FPy AST: expression"""
    __slots__ = ()

    def __init__(self, loc: Optional[Location]):
        super().__init__(loc)

    @abstractmethod
    def is_equiv(self, other) -> bool:
        """
        Check if this expression is structuarlly equivalent to another expression.

        This is essentially a recursive equality check.
        The dunder method `__eq__` is used to check if two expressions
        represent exactly the same tree, e.g., `id(self) == id(other)`.
        """
        ...

class Stmt(Ast):
    """FPy AST: statement"""
    __slots__ = ()

    def __init__(self, loc: Optional[Location]):
        super().__init__(loc)

    @abstractmethod
    def is_equiv(self, other) -> bool:
        """
        Check if this statement is structurally equivalent to another statement.

        This is essentially a recursive equality check.
        The dunder method `__eq__` is used to check if two statements
        represent exactly the same tree, e.g., `id(self) == id(other)`.
        """
        ...

class ValueExpr(Expr):
    """FPy Ast: terminal expression"""
    __slots__ = ()

    def __init__(self, loc: Optional[Location]):
        super().__init__(loc)

class Var(ValueExpr):
    """FPy AST: variable"""
    __slots__ = ('name',)
    name: NamedId

    def __init__(self, name: NamedId, loc: Optional[Location]):
        super().__init__(loc)
        self.name = name

    def is_equiv(self, other) -> bool:
        return isinstance(other, Var) and self.name == other.name

class BoolVal(ValueExpr):
    """FPy AST: boolean value"""
    __slots__ = ('val',)
    val: bool

    def __init__(self, val: bool, loc: Optional[Location]):
        super().__init__(loc)
        self.val = val

    def is_equiv(self, other) -> bool:
        return isinstance(other, BoolVal) and self.val == other.val

class RealVal(ValueExpr):
    """FPy AST: real value"""
    __slots__ = ()

    def __init__(self, loc: Optional[Location]):
        super().__init__(loc)

class RationalVal(RealVal):
    """FPy AST: abstract rational value"""
    __slots__ = ()

    def __init__(self, loc: Optional[Location]):
        super().__init__(loc)

    @abstractmethod
    def as_rational(self) -> Fraction:
        """Returns the represented rational value as a Fraction in simplest form."""
        ...

class Decnum(RationalVal):
    """FPy AST: decimal number"""
    __slots__ = ('val',)
    val: str

    def __init__(self, val: str, loc: Optional[Location]):
        super().__init__(loc)
        self.val = val

    def is_equiv(self, other) -> bool:
        return isinstance(other, Decnum) and self.val == other.val

    def as_rational(self) -> Fraction:
        return decnum_to_fraction(self.val)

class Hexnum(RationalVal):
    """FPy AST: hexadecimal number"""
    __slots__ = ('func', 'val')
    func: NamedId | ForeignAttribute
    val: str

    def __init__(self, func: NamedId | ForeignAttribute, val: str, loc: Optional[Location]):
        super().__init__(loc)
        self.func = func
        self.val = val

    def is_equiv(self, other) -> bool:
        return isinstance(other, Hexnum) and self.val == other.val

    def as_rational(self) -> Fraction:
        return hexnum_to_fraction(self.val)

class Integer(RationalVal):
    """FPy AST: integer"""
    __slots__ = ('val',)
    val: int

    def __init__(self, val: int, loc: Optional[Location]):
        super().__init__(loc)
        self.val = val

    def is_equiv(self, other) -> bool:
        return isinstance(other, Integer) and self.val == other.val

    def as_rational(self) -> Fraction:
        return Fraction(self.val)

class Rational(RationalVal):
    """FPy AST: rational number"""
    __slots__ = ('func', 'p', 'q')
    func: NamedId | ForeignAttribute
    p: int
    q: int

    def __init__(self, func: NamedId | ForeignAttribute, p: int, q: int, loc: Optional[Location]):
        super().__init__(loc)
        self.func = func
        self.p = p
        self.q = q

    def is_equiv(self, other) -> bool:
        return isinstance(other, Rational) and self.p == other.p and self.q == other.q

    def as_rational(self) -> Fraction:
        return Fraction(self.p, self.q)

class Digits(RationalVal):
    """FPy AST: scientific notation"""
    __slots__ = ('func', 'm', 'e', 'b')
    func: NamedId | ForeignAttribute
    m: int
    e: int
    b: int

    def __init__(self, func: NamedId | ForeignAttribute, m: int, e: int, b: int, loc: Optional[Location]):
        super().__init__(loc)
        self.func = func
        self.m = m
        self.e = e
        self.b = b

    def is_equiv(self, other) -> bool:
        return (
            isinstance(other, Digits)
            and self.m == other.m
            and self.e == other.e
            and self.b == other.b
        )

    def as_rational(self) -> Fraction:
        return digits_to_fraction(self.m, self.e, self.b)

class ForeignVal(ValueExpr):
    """FPy AST: native Python value"""
    __slots__ = ('val',)
    val: Any

    def __init__(self, val: Any, loc: Optional[Location]):
        super().__init__(loc)
        self.val = val

    def is_equiv(self, other) -> bool:
        return isinstance(other, ForeignVal) and self.val == other.val

class NaryExpr(Expr):
    """FPy AST: expression with N arguments"""
    __slots__ = ()

class NullaryOp(NaryExpr):
    """FPy AST: (named) nullary operation"""
    __slots__ = ('func',)
    func: NamedId | ForeignAttribute

    def __init__(self, func: NamedId | ForeignAttribute, loc: Optional[Location]):
        super().__init__(loc)
        self.func = func

    def is_equiv(self, other) -> bool:
        return isinstance(other, NullaryOp) and type(self) is type(other)

class UnaryOp(NaryExpr):
    """FPy AST: unary operation"""
    __slots__ = ('arg',)
    arg: Expr

    def __init__(
        self,
        arg: Expr,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.arg = arg

    def is_equiv(self, other) -> bool:
        return (
            isinstance(other, UnaryOp)
            and type(self) is type(other)
            and self.arg.is_equiv(other.arg)
        )

class NamedUnaryOp(UnaryOp):
    """FPy AST: unary operation with a named function"""
    __slots__ = ('func',)
    func: NamedId | ForeignAttribute

    def __init__(
        self,
        func: NamedId | ForeignAttribute,
        arg: Expr,
        loc: Optional[Location]
    ):
        super().__init__(arg, loc)
        self.func = func

class BinaryOp(NaryExpr):
    """FPy AST: binary operation"""
    __slots__ = ('first', 'second')
    first: Expr
    second: Expr

    def __init__(
        self,
        first: Expr,
        second: Expr,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.first = first
        self.second = second

    def is_equiv(self, other) -> bool:
        return (
            isinstance(other, BinaryOp)
            and type(self) is type(other)
            and self.first.is_equiv(other.first)
            and self.second.is_equiv(other.second)
        )

class NamedBinaryOp(BinaryOp):
    """FPy AST: binary operation with a named function"""
    __slots__ = ('func',)
    func: NamedId | ForeignAttribute

    def __init__(
        self,
        func: NamedId | ForeignAttribute,
        first: Expr,
        second: Expr,
        loc: Optional[Location]
    ):
        super().__init__(first, second, loc)
        self.func = func

class TernaryOp(NaryExpr):
    """FPy AST: ternary operation"""
    __slots__ = ('first', 'second', 'third')
    first: Expr
    second: Expr
    third: Expr

    def __init__(
        self,
        first: Expr,
        second: Expr,
        third: Expr,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.first = first
        self.second = second
        self.third = third

    def is_equiv(self, other) -> bool:
        return (
            isinstance(other, TernaryOp)
            and type(self) is type(other)
            and self.first.is_equiv(other.first)
            and self.second.is_equiv(other.second)
            and self.third.is_equiv(other.third)
        )

class NamedTernaryOp(TernaryOp):
    """FPy AST: ternary operation with a named function"""
    __slots__ = ('func',)
    func: NamedId | ForeignAttribute

    def __init__(
        self,
        func: NamedId | ForeignAttribute,
        first: Expr,
        second: Expr,
        third: Expr,
        loc: Optional[Location]
    ):
        super().__init__(first, second, third, loc)
        self.func = func

class NaryOp(NaryExpr):
    """FPy AST: n-ary operation"""
    __slots__ = ('args',)
    args: tuple[Expr, ...]

    def __init__(self, args: Sequence[Expr], loc: Optional[Location]):
        super().__init__(loc)
        self.args = tuple(args)

    def is_equiv(self, other) -> bool:
        return (
            isinstance(other, NaryOp)
            and type(self) is type(other)
            and all(a.is_equiv(b) for a, b in zip(self.args, other.args))
        )

class NamedNaryOp(NaryOp):
    """FPy AST: n-ary operation with a named function"""
    __slots__ = ('func',)
    func: NamedId | ForeignAttribute

    def __init__(
        self,
        func: NamedId | ForeignAttribute,
        args: Sequence[Expr],
        loc: Optional[Location]
    ):
        super().__init__(args, loc)
        self.func = func

# Constants

class ConstNan(NullaryOp):
    """FPy node: NaN (constant)"""
    __slots__ = ()

class ConstInf(NullaryOp):
    """FPy node: infinity (constant)"""
    __slots__ = ()

class ConstPi(NullaryOp):
    """FPy node: π (constant)"""
    __slots__ = ()

class ConstE(NullaryOp):
    """FPy node: e (constant)"""
    __slots__ = ()

class ConstLog2E(NullaryOp):
    """FPy node: log₂(e) (constant)"""
    __slots__ = ()

class ConstLog10E(NullaryOp):
    """FPy node: log₁₀(e) (constant)"""
    __slots__ = ()

class ConstLn2(NullaryOp):
    """FPy node: ln(2) (constant)"""
    __slots__ = ()

class ConstPi_2(NullaryOp):
    """FPy node: π/2 (constant)"""
    __slots__ = ()

class ConstPi_4(NullaryOp):
    """FPy node: π/4 (constant)"""
    __slots__ = ()

class Const1_Pi(NullaryOp):
    """FPy node: 1/π (constant)"""
    __slots__ = ()

class Const2_Pi(NullaryOp):
    """FPy node: 2/π (constant)"""
    __slots__ = ()

class Const2_SqrtPi(NullaryOp):
    """FPy node: 2/√π (constant)"""
    __slots__ = ()

class ConstSqrt2(NullaryOp):
    """FPy node: √2 (constant)"""
    __slots__ = ()

class ConstSqrt1_2(NullaryOp):
    """FPy node: √(1/2) (constant)"""
    __slots__ = ()

# IEEE 754 required arithmetic

class Add(BinaryOp):
    """FPy node: addition"""
    __slots__ = ()

class Sub(BinaryOp):
    """FPy node: subtraction"""
    __slots__ = ()

class Mul(BinaryOp):
    """FPy node: subtraction"""
    __slots__ = ()

class Div(BinaryOp):
    """FPy node: subtraction"""
    __slots__ = ()

class Fabs(NamedUnaryOp):
    """FPy node: absolute value"""
    __slots__ = ()

class Sqrt(NamedUnaryOp):
    """FPy node: square-root"""
    __slots__ = ()

class Fma(NamedTernaryOp):
    """FPy node: fused-multiply add"""
    __slots__ = ()

# Sign operations

class Neg(UnaryOp):
    """FPy node: negation"""
    __slots__ = ()

class Copysign(NamedBinaryOp):
    """FPy node: copysign"""
    __slots__ = ()

# Composite arithmetic

class Fdim(NamedBinaryOp):
    """FPy node: `max(x - y, 0)`"""
    __slots__ = ()

class Hypot(NamedBinaryOp):
    """FPy node: `sqrt(x ** 2 + y ** 2)`"""
    __slots__ = ()

# Other arithmetic

class Max(NamedNaryOp):
    """FPy node: `max(x, y, ...)`"""
    __slots__ = ()

class Min(NamedNaryOp):
    """FPy node: `min(x, y, ...)`"""
    __slots__ = ()

class Fmod(NamedBinaryOp):
    """FPy node: modulus"""
    __slots__ = ()

class Remainder(NamedBinaryOp):
    """FPy node: remainder"""
    __slots__ = ()

class Cbrt(NamedUnaryOp):
    """FPy node: cube-root"""
    __slots__ = ()

class Sum(NamedUnaryOp):
    """FPy node: summation"""
    __slots__ = ()

# Rounding and truncation

class Ceil(NamedUnaryOp):
    """FPy node: ceiling"""
    __slots__ = ()

class Floor(NamedUnaryOp):
    """FPy node: floor"""
    __slots__ = ()

class NearbyInt(NamedUnaryOp):
    """FPy node: nearest integer"""
    __slots__ = ()

class RoundInt(NamedUnaryOp):
    """FPy node: round"""
    __slots__ = ()

class Trunc(NamedUnaryOp):
    """FPy node: truncation"""
    __slots__ = ()

# Trigonometric functions

class Acos(NamedUnaryOp):
    """FPy node: inverse cosine"""
    __slots__ = ()

class Asin(NamedUnaryOp):
    """FPy node: inverse sine"""
    __slots__ = ()

class Atan(NamedUnaryOp):
    """FPy node: inverse tangent"""
    __slots__ = ()

class Atan2(NamedBinaryOp):
    """FPy node: `atan(y / x)` with correct quadrant"""
    __slots__ = ()

class Cos(NamedUnaryOp):
    """FPy node: cosine"""
    __slots__ = ()

class Sin(NamedUnaryOp):
    """FPy node: sine"""
    __slots__ = ()

class Tan(NamedUnaryOp):
    """FPy node: tangent"""
    __slots__ = ()

# Hyperbolic functions

class Acosh(NamedUnaryOp):
    """FPy node: inverse hyperbolic cosine"""
    __slots__ = ()

class Asinh(NamedUnaryOp):
    """FPy node: inverse hyperbolic sine"""
    __slots__ = ()

class Atanh(NamedUnaryOp):
    """FPy node: inverse hyperbolic tangent"""
    __slots__ = ()

class Cosh(NamedUnaryOp):
    """FPy node: hyperbolic cosine"""
    __slots__ = ()

class Sinh(NamedUnaryOp):
    """FPy node: hyperbolic sine"""
    __slots__ = ()

class Tanh(NamedUnaryOp):
    """FPy node: hyperbolic tangent"""
    __slots__ = ()

# Exponential / logarithmic functions

class Exp(NamedUnaryOp):
    """FPy node: exponential (base e)"""
    __slots__ = ()

class Exp2(NamedUnaryOp):
    """FPy node: exponential (base 2)"""
    __slots__ = ()

class Expm1(NamedUnaryOp):
    """FPy node: `exp(x) - 1`"""
    __slots__ = ()

class Log(NamedUnaryOp):
    """FPy node: logarithm (base e)"""
    __slots__ = ()

class Log10(NamedUnaryOp):
    """FPy node: logarithm (base 10)"""
    __slots__ = ()

class Log1p(NamedUnaryOp):
    """FPy node: `log(x + 1)`"""
    __slots__ = ()

class Log2(NamedUnaryOp):
    """FPy node: logarithm (base 2)"""
    __slots__ = ()

class Pow(NamedBinaryOp):
    """FPy node: `x ** y`"""
    __slots__ = ()

# Integral functions

class Erf(NamedUnaryOp):
    """FPy node: error function"""
    __slots__ = ()

class Erfc(NamedUnaryOp):
    """FPy node: complementary error function"""
    __slots__ = ()

class Lgamma(NamedUnaryOp):
    """FPy node: logarithm of the absolute value of the gamma function"""
    __slots__ = ()

class Tgamma(NamedUnaryOp):
    """FPy node: gamma function"""
    __slots__ = ()


# Classification

class IsFinite(NamedUnaryOp):
    """FPy node: is the value finite?"""
    __slots__ = ()

class IsInf(NamedUnaryOp):
    """FPy node: is the value infinite?"""
    __slots__ = ()

class IsNan(NamedUnaryOp):
    """FPy node: is the value NaN?"""
    __slots__ = ()

class IsNormal(NamedUnaryOp):
    """FPy node: is the value normal?"""
    __slots__ = ()

class Signbit(NamedUnaryOp):
    """FPy node: is the signbit 1?"""
    __slots__ = ()

# Logical operators

class Not(UnaryOp):
    """FPy node: logical negation"""
    __slots__ = ()

class Or(NaryOp):
    """FPy node: logical disjunction"""
    __slots__ = ()

class And(NaryOp):
    """FPy node: logical conjunction"""
    __slots__ = ()

# Rounding operator

class Round(NamedUnaryOp):
    """FPy node: inter-format rounding"""

class RoundExact(NamedUnaryOp):
    """FPy node: inter-format rounding"""
    __slots__ = ()

# Tensor operators

class Range(NamedUnaryOp):
    """FPy node: range constructor"""
    __slots__ = ()

class Dim(NamedUnaryOp):
    """FPy node: dimension operator"""
    __slots__ = ()

class Size(NamedBinaryOp):
    """FPy node: size operator"""
    __slots__ = ()

class Zip(NamedNaryOp):
    """FPy node: zip operator"""
    __slots__ = ()

class Enumerate(NamedUnaryOp):
    """FPy node: enumerate operator"""
    __slots__ = ()

class Call(NaryExpr):
    """FPy AST: function call"""
    __slots__ = ('func', 'fn', 'args')
    func: NamedId | ForeignAttribute
    fn: object
    args: tuple[Expr, ...]

    def __init__(
        self,
        func: NamedId | ForeignAttribute,
        fn: object,
        args: Sequence[Expr],
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.func = func
        self.fn = fn
        self.args = tuple(args)

    def is_equiv(self, other):
        if not isinstance(other, Call):
            return False

        match self.fn, other.fn:
            case None, None:
                if self.func != other.func:
                    return False
            case _, _:
                if self.fn != other.fn:
                    return False

        return (
            len(self.args) == len(other.args)
            and all(a.is_equiv(b) for a, b in zip(self.args, other.args))
        )

class Compare(Expr):
    """FPy AST: comparison chain"""
    __slots__ = ('ops', 'args')
    ops: tuple[CompareOp, ...]
    args: tuple[Expr, ...]

    def __init__(
        self,
        ops: Sequence[CompareOp],
        args: Sequence[Expr],
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.ops = tuple(ops)
        self.args = tuple(args)

    def is_equiv(self, other) -> bool:
        return (
            isinstance(other, Compare)
            and len(self.ops) == len(other.ops)
            and all(op == other_op for op, other_op in zip(self.ops, other.ops))
            and all(arg.is_equiv(other_arg) for arg, other_arg in zip(self.args, other.args))
        )

class TupleExpr(Expr):
    """FPy AST: tuple expression"""
    __slots__ = ('args',)
    args: tuple[Expr, ...]

    def __init__(
        self,
        args: Sequence[Expr],
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.args = tuple(args)

    def is_equiv(self, other) -> bool:
        return (
            isinstance(other, TupleExpr)
            and len(self.args) == len(other.args)
            and all(arg.is_equiv(other_arg) for arg, other_arg in zip(self.args, other.args))
        )

class TupleBinding(Ast):
    """FPy AST: tuple binding"""
    __slots__ = ('elts',)
    elts: tuple[Id | Self, ...]

    def __init__(
        self,
        elts: Sequence[Id | Self],
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.elts = tuple(elts)

    def __iter__(self):
        return iter(self.elts)

    def is_equiv(self, other) -> bool:
        return (
            isinstance(other, TupleBinding)
            and len(self.elts) == len(other.elts)
            and all(self_elt.is_equiv(other_elt) for self_elt, other_elt in zip(self.elts, other.elts))
        )

    def names(self) -> set[NamedId]:
        ids: set[NamedId] = set()
        for v in self.elts:
            if isinstance(v, NamedId):
                ids.add(v)
            elif isinstance(v, UnderscoreId):
                pass
            elif isinstance(v, TupleBinding):
                ids |= v.names()
            else:
                raise NotImplementedError('unexpected tuple identifier', v)
        return ids

class ListExpr(Expr):
    """FPy AST: list expression"""
    __slots__ = ('args',)
    args: tuple[Expr, ...]

    def __init__(
        self,
        args: Sequence[Expr],
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.args = tuple(args)

    def is_equiv(self, other) -> bool:
        return (
            isinstance(other, TupleExpr)
            and len(self.args) == len(other.args)
            and all(arg.is_equiv(other_arg) for arg, other_arg in zip(self.args, other.args))
        )

class ListComp(Expr):
    """FPy AST: list comprehension expression"""
    __slots__ = ('targets', 'iterables', 'elt')
    targets: tuple[Id | TupleBinding, ...]
    iterables: tuple[Expr, ...]
    elt: Expr

    def __init__(
        self,
        targets: Sequence[Id | TupleBinding],
        iterables: Sequence[Expr],
        elt: Expr,
        loc: Optional[Location]
    ):
        assert len(targets) == len(iterables)
        super().__init__(loc)
        self.targets = tuple(targets)
        self.iterables = tuple(iterables)
        self.elt = elt

    def is_equiv(self, other) -> bool:
        return (
            isinstance(other, ListComp)
            and len(self.targets) == len(other.targets)
            and all(t.is_equiv(o_t) for t, o_t in zip(self.targets, other.targets))
            and all(i.is_equiv(o_i) for i, o_i in zip(self.iterables, other.iterables))
            and self.elt.is_equiv(other.elt)
        )

class ListSet(Expr):
    """
    FPy node: list set expression (functional)

    Generated by the `FuncUpdate` transform.
    """
    __slots__ = ('array', 'slices', 'value')
    array: Expr
    slices: tuple[Expr, ...]
    value: Expr

    def __init__(self, array: Expr, slices: Sequence[Expr], value: Expr, loc: Optional[Location]):
        super().__init__(loc)
        self.array = array
        self.slices = tuple(slices)
        self.value = value

    def is_equiv(self, other) -> bool:
        return (
            isinstance(other, ListSet)
            and self.array.is_equiv(other.array)
            and len(self.slices) == len(other.slices)
            and all(slice.is_equiv(other_slice) for slice, other_slice in zip(self.slices, other.slices))
            and self.value.is_equiv(other.value)
        )

class ListRef(Expr):
    """FPy AST: list indexing expression"""
    __slots__ = ('value', 'index')
    value: Expr
    index: Expr

    def __init__(self, value: Expr, index: Expr, loc: Optional[Location]):
        super().__init__(loc)
        self.value = value
        self.index = index

    def is_equiv(self, other) -> bool:
        return (
            isinstance(other, ListRef)
            and self.value.is_equiv(other.value)
            and self.index.is_equiv(other.index)
        )

class ListSlice(Expr):
    """FPy AST: list slicing expression"""
    __slots__ = ('value', 'start', 'stop')
    value: Expr
    start: Expr | None
    stop: Expr | None

    def __init__(
        self,
        value: Expr,
        start: Optional[Expr],
        stop: Optional[Expr],
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.value = value
        self.start = start
        self.stop = stop

    def is_equiv(self, other) -> bool:
        if not isinstance(other, ListSlice):
            return False

        if not self.value.is_equiv(other.value):
            return False

        match self.start, other.start:
            case Expr(), Expr():
                return self.start.is_equiv(other.start)
            case None, None:
                return True
            case _:
                return False


class IfExpr(Expr):
    """FPy AST: if expression"""
    __slots__ = ('cond', 'ift', 'iff')
    cond: Expr
    ift: Expr
    iff: Expr

    def __init__(
        self,
        cond: Expr,
        ift: Expr,
        iff: Expr,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.cond = cond
        self.ift = ift
        self.iff = iff

    def is_equiv(self, other) -> bool:
        return (
            isinstance(other, IfExpr)
            and self.cond.is_equiv(other.cond)
            and self.ift.is_equiv(other.ift)
            and self.iff.is_equiv(other.iff)
        )

class ContextExpr(Expr):
    """FPy AST: context constructor"""
    __slots__ = ('ctor', 'args', 'kwargs')
    ctor: Var | ForeignAttribute
    args: tuple[Expr | ForeignAttribute, ...]
    kwargs: tuple[tuple[str, Expr | ForeignAttribute], ...]

    def __init__(
        self,
        ctor: Var | ForeignAttribute,
        args: Sequence[Expr | ForeignAttribute],
        kwargs: Sequence[tuple[str, Expr | ForeignAttribute]],
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.ctor = ctor
        self.args = tuple(args)
        self.kwargs = tuple(kwargs)

    def is_equiv(self, other):
        return (
            isinstance(other, ContextExpr)
            and self.ctor.is_equiv(other.ctor)
            and len(self.args) == len(other.args)
            and all(a.is_equiv(b) for a, b in zip(self.args, other.args))
            and len(self.kwargs) == len(other.kwargs)
            and all(k1 == k2 and v1.is_equiv(v2) for (k1, v1), (k2, v2) in zip(self.kwargs, other.kwargs))
        )

class ContextAttribute(Ast):
    """FPy AST: context attribute"""
    __slots__ = ('expr', 'name')
    expr: Expr
    name: str

    def __init__(self, expr: Expr, name: str, loc: Optional[Location]):
        super().__init__(loc)
        self.expr = expr
        self.name = name

class ContextUpdate(Ast):
    """FPy AST: context update"""
    __slots__ = ('expr', 'kwargs')
    expr: Expr
    kwargs: dict[str, Expr]

    def __init__(self, expr: Expr, kwargs: dict[str, Expr], loc: Optional[Location]):
        super().__init__(loc)
        self.expr = expr
        self.kwargs = kwargs

class StmtBlock(Ast):
    """FPy AST: list of statements"""
    __slots__ = ('stmts',)

    # list makes it easier to modify in-place
    # not ideal since it's ideally immutable, but it simplifies the code
    stmts: list[Stmt]

    def __init__(self, stmts: list[Stmt]):
        if stmts == []:
            loc = None
        else:
            first_loc = stmts[0].loc
            last_loc = stmts[-1].loc
            if first_loc is None or last_loc is None:
                loc = None
            else:
                loc = Location(
                    first_loc.source,
                    first_loc.start_line,
                    first_loc.start_column,
                    last_loc.end_line,
                    last_loc.end_column
                )

        super().__init__(loc)
        self.stmts = stmts

    def is_equiv(self, other):
        return (
            isinstance(other, StmtBlock)
            and len(self.stmts) == len(other.stmts)
            and all(s1.is_equiv(s2) for s1, s2 in zip(self.stmts, other.stmts))
        )

class Assign(Stmt):
    """FPy AST: variable assignment"""
    __slots__ = ('binding', 'type', 'expr')
    binding: Id | TupleBinding
    type: Optional[TypeAnn]
    expr: Expr

    def __init__(
        self,
        binding: Id | TupleBinding,
        type: Optional[TypeAnn],
        expr: Expr,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.binding = binding
        self.type = type
        self.expr = expr

    def is_equiv(self, other):
        return (
            isinstance(other, Assign)
            and self.binding.is_equiv(other.binding)
            and self.expr.is_equiv(other.expr)
        )

class IndexedAssign(Stmt):
    """FPy AST: assignment to tuple indexing"""
    __slots__ = ('var', 'slices', 'expr')
    var: NamedId
    slices: tuple[Expr, ...]
    expr: Expr

    def __init__(
        self,
        var: NamedId,
        slices: Sequence[Expr],
        expr: Expr,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.var = var
        self.slices = tuple(slices)
        self.expr = expr

    def is_equiv(self, other) -> bool:
        return (
            isinstance(other, IndexedAssign)
            and self.var == other.var
            and len(self.slices) == len(other.slices)
            and all(s1.is_equiv(s2) for s1, s2 in zip(self.slices, other.slices))
            and self.expr.is_equiv(other.expr)
        )

class If1Stmt(Stmt):
    """FPy AST: if statement with one branch"""
    __slots__ = ('cond', 'body')
    cond: Expr
    body: StmtBlock

    def __init__(
        self,
        cond: Expr,
        body: StmtBlock,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.cond = cond
        self.body = body

    def is_equiv(self, other) -> bool:
        return (
            isinstance(other, If1Stmt)
            and self.cond.is_equiv(other.cond)
            and self.body.is_equiv(other.body)
        )

class IfStmt(Stmt):
    """FPy AST: if statement (with two branhces)"""
    __slots__ = ('cond', 'ift', 'iff')
    cond: Expr
    ift: StmtBlock
    iff: StmtBlock

    def __init__(
        self,
        cond: Expr,
        ift: StmtBlock,
        iff: StmtBlock,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.cond = cond
        self.ift = ift
        self.iff = iff

    def is_equiv(self, other) -> bool:
        return (
            isinstance(other, IfStmt)
            and self.cond.is_equiv(other.cond)
            and self.ift.is_equiv(other.ift)
            and self.iff.is_equiv(other.iff)
        )

class WhileStmt(Stmt):
    """FPy AST: while statement"""
    __slots__ = ('cond', 'body')
    cond: Expr
    body: StmtBlock

    def __init__(
        self,
        cond: Expr,
        body: StmtBlock,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.cond = cond
        self.body = body

    def is_equiv(self, other) -> bool:
        return (
            isinstance(other, WhileStmt)
            and self.cond.is_equiv(other.cond)
            and self.body.is_equiv(other.body)
        )

class ForStmt(Stmt):
    """FPy AST: for statement"""
    __slots__ = ('target', 'iterable', 'body')
    target: Id | TupleBinding
    iterable: Expr
    body: StmtBlock

    def __init__(
        self,
        target: Id | TupleBinding,
        iterable: Expr,
        body: StmtBlock,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.target = target
        self.iterable = iterable
        self.body = body

    def is_equiv(self, other) -> bool:
        return (
            isinstance(other, ForStmt)
            and self.target.is_equiv(other.target)
            and self.iterable.is_equiv(other.iterable)
            and self.body.is_equiv(other.body)
        )

class ContextStmt(Stmt):
    """FPy AST: with statement"""
    __slots__ = ('name', 'ctx', 'body')
    name: Id
    ctx: Var | ForeignVal | ForeignAttribute | ContextExpr
    body: StmtBlock

    def __init__(
        self,
        name: Id,
        ctx: Var | ForeignVal | ForeignAttribute | ContextExpr,
        body: StmtBlock,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.ctx = ctx
        self.name = name
        self.body = body

    def is_equiv(self, other) -> bool:
        return (
            isinstance(other, ContextStmt)
            and self.name == other.name
            and self.ctx.is_equiv(other.ctx)
            and self.body.is_equiv(other.body)
        )

class AssertStmt(Stmt):
    """FPy AST: assert statement"""
    __slots__ = ('test', 'msg')
    test: Expr
    msg: Optional[str]

    def __init__(
        self,
        test: Expr,
        msg: Optional[str],
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.test = test
        self.msg = msg

    def is_equiv(self, other) -> bool:
        return (
            isinstance(other, AssertStmt)
            and self.test.is_equiv(other.test)
            and self.msg == other.msg
        )

class EffectStmt(Stmt):
    """FPy AST: an expression without a result"""
    __slots__ = ('expr',)
    expr: Expr

    def __init__(
        self,
        expr: Expr,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.expr = expr

    def is_equiv(self, other) -> bool:
        return (
            isinstance(other, EffectStmt)
            and self.expr.is_equiv(other.expr)
        )

class ReturnStmt(Stmt):
    """FPy AST: return statement"""
    __slots__ = ('expr',)
    expr: Expr

    def __init__(
        self,
        expr: Expr,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.expr = expr

    def is_equiv(self, other) -> bool:
        return (
            isinstance(other, ReturnStmt)
            and self.expr.is_equiv(other.expr)
        )

class Argument(Ast):
    """FPy AST: function argument"""
    __slots__ = ('name', 'type')
    name: Id
    type: Optional[TypeAnn]

    def __init__(
        self,
        name: Id,
        type: Optional[TypeAnn],
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.name = name
        self.type = type

    def is_equiv(self, other) -> bool:
        return isinstance(other, Argument) and self.name == other.name

class FuncDef(Ast):
    """FPy AST: function definition"""
    __slots__ = ('name', 'args', 'body', 'metadata', 'free_vars', 'ctx')
    name: str
    args: tuple[Argument, ...]
    body: StmtBlock
    metadata: dict[str, Any]
    free_vars: set[NamedId]
    ctx: Optional[Context | FPCoreContext]

    def __init__(
        self,
        name: str,
        args: Sequence[Argument],
        body: StmtBlock,
        *,
        metadata: Optional[dict[str, Any]] = None,
        free_vars: Optional[set[NamedId]] = None,
        ctx: Optional[Context | FPCoreContext] = None,
        loc: Optional[Location] = None
    ):
        if metadata is None:
            metadata = {}
        else:
            metadata = dict(metadata)

        if free_vars is None:
            free_vars = set()
        else:
            free_vars = set(free_vars)

        super().__init__(loc)
        self.name = name
        self.args = tuple(args)
        self.body = body
        self.metadata = metadata
        self.free_vars = free_vars
        self.ctx = ctx

    def is_equiv(self, other) -> bool:
        return (
            isinstance(other, FuncDef)
            and self.name == other.name
            and len(self.args) == len(other.args)
            and all(a.is_equiv(b) for a, b in zip(self.args, other.args))
            and self.body.is_equiv(other.body)
        )


class BaseFormatter:
    """Abstract base class for AST formatters."""

    @abstractmethod
    def format(self, ast: Ast) -> str:
        ...

_default_formatter: Optional[BaseFormatter] = None

def get_default_formatter() -> BaseFormatter:
    """Get the default formatter for FPy AST."""
    global _default_formatter
    if _default_formatter is None:
        raise RuntimeError('no default formatter available')
    return _default_formatter

def set_default_formatter(formatter: BaseFormatter):
    """Set the default formatter for FPy AST."""
    global _default_formatter
    if not isinstance(formatter, BaseFormatter):
        raise TypeError(f'expected BaseFormatter, got {formatter}')
    _default_formatter = formatter
