"""
This module contains the intermediate representation (IR).
"""

from enum import Enum
from typing import Self

from .types import IRType

class IR:
    """FPy IR: base class for all IR nodes."""

    def __repr__(self):
        name = self.__class__.__name__
        items = ', '.join(f'{k}={repr(v)}' for k, v in self.__dict__.items())
        return f'{name}({items})'

class Expr(IR):
    """FPy AST: expression"""

class Stmt(IR):
    """FPy AST: statement"""

class Block(Stmt):
    """FPy AST: block statement"""
    stmts: list[Stmt]

    def __init__(self, stmts: list[Stmt]):
        self.stmts = stmts

class ValueExpr(Expr):
    """FPy node: abstract terminal"""

class Var(ValueExpr):
    """FPy node: variable"""
    name: str

    def __init__(self, name: str):
        self.name = name

class Decnum(ValueExpr):
    """FPy node: (decimal) number"""
    val: str

    def __init__(self, val: str):
        self.val = val

class Integer(ValueExpr):
    """FPy node: numerical constant (integer)"""
    val: int

    def __init__(self, val: int):
        self.val = val

class Digits(ValueExpr):
    """FPy node: numerical constant in scientific notation"""
    m: int
    e: int
    b: int

    def __init__(self, m: int, e: int, b: int):
        super().__init__()
        self.m = m
        self.e = e
        self.b = b

class NaryExpr(Expr):
    """FPy node: application expression"""
    name: str
    children: list[Expr]

    def __init__(self, *children: Expr):
        super().__init__()
        self.children = list(children)

class UnaryExpr(NaryExpr):
    """FPy node: abstract unary application"""

    def __init__(self, e: Expr):
        super().__init__(e)

class BinaryExpr(NaryExpr):
    """FPy node: abstract binary application"""

    def __init__(self, e1: Expr, e2: Expr):
        super().__init__(e1, e2)

class TernaryExpr(NaryExpr):
    """FPy node: abstract ternary application"""

    def __init__(self, e1: Expr, e2: Expr, e3: Expr):
        super().__init__(e1, e2, e3)

class UnknownCall(NaryExpr):
    """FPy node: abstract application"""

    def __init__(self, name: str, *children: Expr):
        super().__init__(*children)
        self.name = name

# IEEE 754 required arithmetic

class Add(BinaryExpr):
    """FPy node: addition"""
    name: str = '+'

class Sub(BinaryExpr):
    """FPy node: subtraction"""
    name: str = '-'

class Mul(BinaryExpr):
    """FPy node: subtraction"""
    name: str = '*'
    
class Div(BinaryExpr):
    """FPy node: subtraction"""
    name: str = '/'

class Fabs(UnaryExpr):
    """FPy node: absolute value"""
    name: str = 'fabs'

class Sqrt(UnaryExpr):
    """FPy node: square-root"""
    name: str = 'sqrt'

class Fma(TernaryExpr):
    """FPy node: square-root"""
    name: str = 'fma'

# Sign operations

class Neg(UnaryExpr):
    """FPy node: negation"""
    # to avoid confusion with subtraction
    # this should not be the display name
    name: str = 'neg'

class Copysign(BinaryExpr):
    """FPy node: copysign"""
    name: str = 'copysign'

# Composite arithmetic

class Fdim(BinaryExpr):
    """FPy node: `max(x - y, 0)`"""
    name: str = 'fdim'

class Fmax(BinaryExpr):
    """FPy node: `max(x, y)`"""
    name: str = 'fmax'

class Fmin(BinaryExpr):
    """FPy node: `min(x, y)`"""
    name: str = 'fmin'

class Fmod(BinaryExpr):
    name: str = 'fmod'

class Remainder(BinaryExpr):
    name: str = 'remainder'

class Hypot(BinaryExpr):
    """FPy node: `sqrt(x ** 2 + y ** 2)`"""
    name: str = 'hypot'

# Other arithmetic

class Cbrt(UnaryExpr):
    """FPy node: cube-root"""
    name: str = 'cbrt'

# Rounding and truncation

class Ceil(UnaryExpr):
    """FPy node: ceiling"""
    name: str = 'ceil'

class Floor(UnaryExpr):
    """FPy node: floor"""
    name: str = 'floor'

class Nearbyint(UnaryExpr):
    """FPy node: nearby-integer"""
    name: str = 'nearbyint'

class Round(UnaryExpr):
    """FPy node: round"""
    name: str = 'round'

class Trunc(UnaryExpr):
    """FPy node: truncation"""
    name: str = 'trunc'

# Trigonometric functions

class Acos(UnaryExpr):
    """FPy node: inverse cosine"""
    name: str = 'acos'

class Asin(UnaryExpr):
    """FPy node: inverse sine"""
    name: str = 'asin'

class Atan(UnaryExpr):
    """FPy node: inverse tangent"""
    name: str = 'atan'

class Atan2(BinaryExpr):
    """FPy node: `atan(y / x)` with correct quadrant"""
    name: str = 'atan2'

class Cos(UnaryExpr):
    """FPy node: cosine"""
    name: str = 'cos'

class Sin(UnaryExpr):
    """FPy node: sine"""
    name: str = 'sin'

class Tan(UnaryExpr):
    """FPy node: tangent"""
    name: str = 'tan'

# Hyperbolic functions

class Acosh(UnaryExpr):
    """FPy node: inverse hyperbolic cosine"""
    name: str = 'acosh'

class Asinh(UnaryExpr):
    """FPy node: inverse hyperbolic sine"""
    name: str = 'asinh'

class Atanh(UnaryExpr):
    """FPy node: inverse hyperbolic tangent"""
    name: str = 'atanh'

class Cosh(UnaryExpr):
    """FPy node: hyperbolic cosine"""
    name: str = 'cosh'

class Sinh(UnaryExpr):
    """FPy node: hyperbolic sine"""
    name: str = 'sinh'

class Tanh(UnaryExpr):
    """FPy node: hyperbolic tangent"""
    name: str = 'tanh'

# Exponential / logarithmic functions

class Exp(UnaryExpr):
    """FPy node: exponential (base e)"""
    name: str = 'exp'

class Exp2(UnaryExpr):
    """FPy node: exponential (base 2)"""
    name: str = 'exp2'

class Expm1(UnaryExpr):
    """FPy node: `exp(x) - 1`"""
    name: str = 'expm1'

class Log(UnaryExpr):
    """FPy node: logarithm (base e)"""
    name: str = 'log'

class Log10(UnaryExpr):
    """FPy node: logarithm (base 10)"""
    name: str = 'log10'

class Log1p(UnaryExpr):
    """FPy node: `log(x + 1)`"""
    name: str = 'log1p'

class Log2(UnaryExpr):
    """FPy node: logarithm (base 2)"""
    name: str = 'log2'

class Pow(BinaryExpr):
    """FPy node: `x ** y`"""
    name: str = 'pow'

# Integral functions

class Erf(UnaryExpr):
    """FPy node: error function"""
    name: str = 'erf'

class Erfc(UnaryExpr):
    """FPy node: complementary error function"""
    name: str = 'erfc'

class Lgamma(UnaryExpr):
    """FPy node: logarithm of the absolute value of the gamma function"""
    name: str = 'lgamma'

class Tgamma(UnaryExpr):
    """FPy node: gamma function"""
    name: str = 'tgamma'

# Classification

class IsFinite(UnaryExpr):
    """FPy node: is the value finite?"""
    name: str = 'isfinite'

class IsInf(UnaryExpr):
    """FPy node: is the value infinite?"""
    name: str = 'isinf'

class IsNan(UnaryExpr):
    """FPy node: is the value NaN?"""
    name: str = 'isnan'

class IsNormal(UnaryExpr):
    """FPy node: is the value normal?"""
    name: str = 'isnormal'

class Signbit(UnaryExpr):
    """FPy node: is the signbit 1?"""
    name: str = 'signbit'

# Logical operators

class Not(UnaryExpr):
    """FPy node: logical negation"""
    name: str = 'not'

class Or(NaryExpr):
    """FPy node: logical disjunction"""
    name: str = 'or'

class And(NaryExpr):
    """FPy node: logical conjunction"""
    name: str = 'and'

# Comparisons

class CompareOp(Enum):
    LT = 0
    LE = 1
    GE = 2
    GT = 3
    EQ = 4
    NE = 5

class Compare(Expr):
    """FPy node: N-argument comparison (N >= 2)"""
    ops: list[CompareOp]
    children: list[Expr]
        
    def __init__(self, ops: list[CompareOp], children: list[Expr]):
        if not isinstance(children, list) or len(children) < 2:
            raise TypeError('expected list of length >= 2', children)
        if not isinstance(ops, list) or len(ops) != len(children) - 1:
            raise TypeError(f'expected list of length >= {len(children)}', children)
        super().__init__()
        self.ops = ops
        self.children = children

class TupleExpr(Expr):
    """FPy node: tuple expression"""
    children: list[Expr]

    def __init__(self, *children: Expr):
        self.children = list(children)

class IfExpr(Expr):
    """FPy node: if expression (ternary)"""
    cond: Expr
    ift: Expr
    iff: Expr

    def __init__(self, cond: Expr, ift: Expr, iff: Expr):
        super().__init__()
        self.cond = cond
        self.ift = ift
        self.iff = iff

class VarAssign(Stmt):
    """FPy node: assignment to a single variable"""
    var: str
    ty: IRType
    expr: Expr

    def __init__(self, var: str, ty: IRType, expr: Expr):
        self.var = var
        self.ty = ty
        self.expr = expr

class TupleBinding(IR):
    """FPy AST: tuple binding"""
    elts: list[str | Self]

    def __init__(self, elts: list[str | Self]):
        self.elts = elts

class TupleAssign(Stmt):
    """FPy node: assignment to a tuple"""
    vars: TupleBinding
    tys: list[IRType]
    val: Expr

    def __init__(self, vars: TupleBinding, tys: list[IRType], val: Expr):
        assert len(vars.elts) == len(tys), "length mismatch"
        self.vars = vars
        self.tys = tys
        self.val = val

class If1Stmt(Stmt):
    """FPy AST: one-armed if statement"""
    cond: Expr
    body: Block

    def __init__(self, cond: Expr, body: Block):
        self.cond = cond
        self.body = body

class IfStmt(Stmt):
    """FPy AST: if statement"""
    cond: Expr
    ift: Block
    iff: Block

    def __init__(self, cond: Expr, ift: Block, iff: Block):
        self.cond = cond
        self.ift = ift
        self.iff = iff

class WhileStmt(Stmt):
    """FPy AST: while statement"""
    cond: Expr
    body: Block

    def __init__(self, cond: Expr, body: Block):
        self.cond = cond
        self.body = body

class ForStmt(Stmt):
    """FPy AST: for statement"""
    var: str
    ty: IRType
    iter: Expr
    body: Block

    def __init__(self, var: str, ty: IRType, iter: Expr, body: Block):
        self.var = var
        self.ty = ty
        self.iter = iter
        self.body = body

class Return(Stmt):
    """FPy AST: return statement"""
    expr: Expr

    def __init__(self, expr: Expr):
        self.expr = expr

class Argument(IR):
    """FPy AST: function argument"""
    name: str
    ty: IRType

    def __init__(self, name: str, ty: IRType):
        self.name = name
        self.ty = ty

class Function(IR):
    """FPy AST: function"""
    name: str
    args: list[Argument]
    body: Block
    ty: IRType

    def __init__(self,
        name: str,
        args: list[Argument],
        body: Block,
        ty: IRType
    ):
        self.name = name
        self.args = args
        self.body = body
        self.ty = ty
