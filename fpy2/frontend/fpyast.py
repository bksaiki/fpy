"""
This module contains the AST for FPy programs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Self, Sequence
from ..utils import CompareOp, Id, NamedId, UnderscoreId, Location

class UnaryOpKind(Enum):
    # unary operators
    NEG = 0
    NOT = 1
    # unary functions
    FABS = 2
    SQRT = 3
    CBRT = 4
    CEIL = 5
    FLOOR = 6
    NEARBYINT = 7
    ROUND = 8
    TRUNC = 9
    ACOS = 10
    ASIN = 11
    ATAN = 12
    COS = 13
    SIN = 14
    TAN = 15
    ACOSH = 16
    ASINH = 17
    ATANH = 18
    COSH = 19
    SINH = 20
    TANH = 21
    EXP = 22
    EXP2 = 23
    EXPM1 = 24
    LOG = 25
    LOG10 = 26
    LOG1P = 27
    LOG2 = 28
    ERF = 29
    ERFC = 30
    LGAMMA = 31
    TGAMMA = 32
    ISFINITE = 33
    ISINF = 34
    ISNAN = 35
    ISNORMAL = 36
    SIGNBIT = 37
    CAST = 38
    # tensor operations
    SHAPE = 39
    RANGE = 40
    DIM = 41

    def __str__(self):
        return self.name.lower()

class BinaryOpKind(Enum):
    # binary operators
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3
    # binary functions
    COPYSIGN = 4
    FDIM = 5
    FMAX = 6
    FMIN = 7
    FMOD = 8
    REMAINDER = 9
    HYPOT = 10
    ATAN2 = 11
    POW = 12
    # tensor operations
    SIZE = 13

    def __str__(self):
        return self.name.lower()

class TernaryOpKind(Enum):
    # ternary operators
    FMA = 0

class NaryOpKind(Enum):
    # boolean operations
    AND = 1
    OR = 2

class Ast(ABC):
    """FPy AST: abstract base class for all AST nodes."""
    loc: Optional[Location]
    attribs: dict[str, Any]

    def __init__(self, loc: Optional[Location]):
        self.loc = loc
        self.attribs = {}

    def __repr__(self):
        name = self.__class__.__name__
        items = ', '.join(f'{k}={repr(v)}' for k, v in self.__dict__.items())
        return f'{name}({items})'

    def format(self) -> str:
        """Format the AST node as a string."""
        formatter = get_default_formatter()
        return formatter.format(self)


class TypeAnn(Ast):
    """FPy AST: typing annotation"""

    def __init__(self, loc: Optional[Location]):
        super().__init__(loc)

class ScalarType(Enum):
    ANY = 0
    REAL = 1
    BOOL = 2

class AnyTypeAnn(TypeAnn):
    """FPy AST: any type annotation"""

    def __init__(self, loc: Optional[Location]):
        super().__init__(loc)

class ScalarTypeAnn(TypeAnn):
    """FPy AST: scalar type annotation"""
    kind: ScalarType

    def __init__(self, kind: ScalarType, loc: Optional[Location]):
        super().__init__(loc)
        self.kind = kind

class TupleTypeAnn(TypeAnn):
    """FPy AST: tuple type annotation"""
    elts: list[TypeAnn]

    def __init__(self, elts: list[TypeAnn], loc: Optional[Location]):
        super().__init__(loc)
        self.elts = elts

class Expr(Ast):
    """FPy AST: expression"""

    def __init__(self, loc: Optional[Location]):
        super().__init__(loc)

class Stmt(Ast):
    """FPy AST: statement"""

    def __init__(self, loc: Optional[Location]):
        super().__init__(loc)

class ValueExpr(Expr):
    """FPy Ast: terminal expression"""

    def __init__(self, loc: Optional[Location]):
        super().__init__(loc)

class Var(ValueExpr):
    """FPy AST: variable"""
    name: NamedId

    def __init__(self, name: NamedId, loc: Optional[Location]):
        super().__init__(loc)
        self.name = name

class Bool(ValueExpr):
    """FPy AST: boolean"""
    val: bool

    def __init__(self, val: bool, loc: Optional[Location]):
        super().__init__(loc)
        self.val = val

class String(ValueExpr):
    """FPy AST: string"""
    val: str

    def __init__(self, val: str, loc: Optional[Location]):
        super().__init__(loc)
        self.val = val

class Decnum(ValueExpr):
    """FPy AST: decimal number"""
    val: str

    def __init__(self, val: str, loc: Optional[Location]):
        super().__init__(loc)
        self.val = val   

class Hexnum(ValueExpr):
    """FPy AST: hexadecimal number"""
    val: str

    def __init__(self, val: str, loc: Optional[Location]):
        super().__init__(loc)
        self.val = val

class Integer(ValueExpr):
    """FPy AST: integer"""
    val: int

    def __init__(self, val: int, loc: Optional[Location]):
        super().__init__(loc)
        self.val = val

class Rational(ValueExpr):
    """FPy AST: rational number"""
    p: int
    q: int

    def __init__(self, p: int, q: int, loc: Optional[Location]):
        super().__init__(loc)
        self.p = p
        self.q = q

class Digits(ValueExpr):
    """FPy AST: scientific notation"""
    m: int
    e: int
    b: int

    def __init__(self, m: int, e: int, b: int, loc: Optional[Location]):
        super().__init__(loc)
        self.m = m
        self.e = e
        self.b = b

class Constant(ValueExpr):
    """FPy AST: constant expression"""
    val: str

    def __init__(self, val: str, loc: Optional[Location]):
        super().__init__(loc)
        self.val = val

class UnaryOp(Expr):
    """FPy AST: unary operation"""
    op: UnaryOpKind
    arg: Expr

    def __init__(
        self,
        op: UnaryOpKind,
        arg: Expr,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.op = op
        self.arg = arg

class BinaryOp(Expr):
    """FPy AST: binary operation"""
    op: BinaryOpKind
    left: Expr
    right: Expr

    def __init__(
        self,
        op: BinaryOpKind,
        left: Expr,
        right: Expr,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.op = op
        self.left = left
        self.right = right

class TernaryOp(Expr):
    """FPy AST: ternary operation"""
    op: TernaryOpKind
    arg0: Expr
    arg1: Expr
    arg2: Expr

    def __init__(
        self,
        op: TernaryOpKind,
        arg0: Expr,
        arg1: Expr,
        arg2: Expr,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.op = op
        self.arg0 = arg0
        self.arg1 = arg1
        self.arg2 = arg2

class NaryOp(Expr):
    """FPy AST: n-ary operation"""
    op: NaryOpKind
    args: list[Expr]

    def __init__(
        self,
        op: NaryOpKind,
        args: list[Expr],
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.op = op
        self.args = args

class Call(Expr):
    """FPy AST: function call"""
    op: str
    args: list[Expr]

    def __init__(
        self,
        op: str,
        args: list[Expr],
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.op = op
        self.args = args

class Compare(Expr):
    """FPy AST: comparison chain"""
    ops: list[CompareOp]
    args: list[Expr]

    def __init__(
        self,
        ops: list[CompareOp],
        args: list[Expr],
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.ops = ops
        self.args = args

class TupleExpr(Expr):
    """FPy AST: tuple expression"""
    args: list[Expr]

    def __init__(
        self,
        args: list[Expr],
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.args = args

class CompExpr(Expr):
    """FPy AST: comprehension expression"""
    vars: list[Id]
    iterables: list[Expr]
    elt: Expr

    def __init__(
        self,
        vars: Sequence[Id],
        iterables: Sequence[Expr],
        elt: Expr,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.vars = list(vars)
        self.iterables = list(iterables)
        self.elt = elt

class RefExpr(Expr):
    """FPy AST: tuple indexing expression"""
    value: Expr
    slices: list[Expr]

    def __init__(self, value: Expr, slices: Sequence[Expr], loc: Optional[Location]):
        super().__init__(loc)
        self.value = value
        self.slices = list(slices)

class IfExpr(Expr):
    """FPy AST: if expression"""
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

class Block(Ast):
    """FPy AST: list of statements"""
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

class VarAssign(Stmt):
    """FPy AST: variable assignment"""
    var: Id
    expr: Expr
    ann: Optional[TypeAnn]

    def __init__(
        self,
        var: Id,
        expr: Expr,
        ann: Optional[TypeAnn],
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.var = var
        self.expr = expr
        self.ann = ann

class TupleBinding(Ast):
    """FPy AST: tuple binding"""
    elts: list[Id | Self]

    def __init__(
        self,
        vars: Sequence[Id | Self],
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.elts = list(vars)

    def __iter__(self):
        return iter(self.elts)

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

class TupleAssign(Stmt):
    """FPy AST: tuple assignment"""
    binding: TupleBinding
    expr: Expr

    def __init__(
        self,
        vars: TupleBinding,
        expr: Expr,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.binding = vars
        self.expr = expr

class RefAssign(Stmt):
    """FPy AST: assignment to tuple indexing"""
    var: NamedId
    slices: list[Expr]
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
        self.slices = list(slices)
        self.expr = expr

class IfStmt(Stmt):
    """FPy AST: if statement"""
    cond: Expr
    ift: Block
    iff: Optional[Block]

    def __init__(
        self,
        cond: Expr,
        ift: Block,
        iff: Optional[Block],
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.cond = cond
        self.ift = ift
        self.iff = iff

class WhileStmt(Stmt):
    """FPy AST: while statement"""
    cond: Expr
    body: Block

    def __init__(
        self,
        cond: Expr,
        body: Block,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.cond = cond
        self.body = body

class ForStmt(Stmt):
    """FPy AST: for statement"""
    var: Id
    iterable: Expr
    body: Block

    def __init__(
        self,
        var: Id,
        iterable: Expr,
        body: Block,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.var = var
        self.iterable = iterable
        self.body = body

class ContextStmt(Stmt):
    """FPy AST: with statement"""
    name: Optional[Id]
    props: dict[str, Any]
    body: Block

    def __init__(
        self,
        name: Optional[Id],
        props: dict[str, Any],
        body: Block,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.props = props
        self.name = name
        self.body = body

class AssertStmt(Stmt):
    """FPy AST: assert statement"""
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

class Return(Stmt):
    """FPy AST: return statement"""
    expr: Expr

    def __init__(
        self,
        expr: Expr,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.expr = expr

class Argument(Ast):
    """FPy AST: function argument"""
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

class FunctionDef(Ast):
    """FPy AST: function definition"""
    name: str
    args: list[Argument]
    body: Block
    ctx: dict[str, Any]
    fvs: set[str]

    def __init__(
        self,
        name: str,
        args: Sequence[Argument],
        body: Block,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.name = name
        self.args = list(args)
        self.body = body
        self.ctx = {}
        self.fvs = set()

class BaseFormatter:
    """Abstract base class for AST formatters."""

    @abstractmethod
    def format(self, ast: Ast) -> str:
        raise NotImplementedError('virtual method')

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
