"""
This module contains the AST for FPy programs.
"""

from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Self, Sequence
from ..utils import CompareOp

@dataclass
class Location:
    """Parse location: line and column number."""
    source: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int

    def __key(self):
        return (
            self.source,
            self.start_line,
            self.start_column,
            self.end_line,
            self.end_column
        )

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if not isinstance(other, Location):
            return False
        return self.__key() == other.__key()


class Ast(ABC):
    """FPy AST: abstract base class for all AST nodes."""
    loc: Location
    attribs: dict[str, Any]

    def __init__(self, loc: Location):
        self.loc = loc
        self.attribs = {}

    def __repr__(self):
        name = self.__class__.__name__
        items = ', '.join(f'{k}={repr(v)}' for k, v in self.__dict__.items())
        return f'{name}({items})'

class TypeAnn(Ast):
    """FPy AST: typing annotation"""

    def __init__(self, loc: Location):
        super().__init__(loc)

class ScalarType(Enum):
    ANY = 0
    REAL = 1
    BOOL = 2

class AnyTypeAnn(TypeAnn):
    """FPy AST: any type annotation"""

    def __init__(self, loc: Location):
        super().__init__(loc)

class ScalarTypeAnn(TypeAnn):
    """FPy AST: scalar type annotation"""
    kind: ScalarType

    def __init__(self, kind: ScalarType, loc: Location):
        super().__init__(loc)
        self.kind = kind

class TupleTypeAnn(TypeAnn):
    """FPy AST: tuple type annotation"""
    elts: list[TypeAnn]

    def __init__(self, elts: list[TypeAnn], loc: Location):
        super().__init__(loc)
        self.elts = elts

class Expr(Ast):
    """FPy AST: expression"""

    def __init__(self, loc: Location):
        super().__init__(loc)

class Stmt(Ast):
    """FPy AST: statement"""

    def __init__(self, loc: Location):
        super().__init__(loc)

class ValueExpr(Expr):
    """FPy Ast: terminal expression"""

    def __init__(self, loc: Location):
        super().__init__(loc)

class Var(ValueExpr):
    """FPy AST: variable"""
    name: str

    def __init__(self, name: str, loc: Location):
        super().__init__(loc)
        self.name = name

class Decnum(ValueExpr):
    """FPy AST: decimal number"""
    val: str

    def __init__(self, val: str, loc: Location):
        super().__init__(loc)
        self.val = val   

class Integer(ValueExpr):
    """FPy AST: integer"""
    val: int

    def __init__(self, val: int, loc: Location):
        super().__init__(loc)
        self.val = val

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
    # unary generator
    RANGE = 38

class UnaryOp(Expr):
    """FPy AST: unary operation"""
    op: UnaryOpKind
    arg: Expr

    def __init__(
        self,
        op: UnaryOpKind,
        arg: Expr,
        loc: Location
    ):
        super().__init__(loc)
        self.op = op
        self.arg = arg

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
        loc: Location
    ):
        super().__init__(loc)
        self.op = op
        self.left = left
        self.right = right

class TernaryOpKind(Enum):
    # ternary operators
    FMA = 0
    DIGITS = 1

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
        loc: Location
    ):
        super().__init__(loc)
        self.op = op
        self.arg0 = arg0
        self.arg1 = arg1
        self.arg2 = arg2

class NaryOpKind(Enum):
    OR = 2
    AND = 1

class NaryOp(Expr):
    """FPy AST: n-ary operation"""
    op: NaryOpKind
    args: list[Expr]

    def __init__(
        self,
        op: NaryOpKind,
        args: list[Expr],
        loc: Location
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
        loc: Location
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
        loc: Location
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
        loc: Location
    ):
        super().__init__(loc)
        self.args = args

class CompExpr(Expr):
    """FPy AST: comprehension expression"""
    vars: list[str]
    iterables: list[Expr]
    elt: Expr

    def __init__(
        self,
        vars: Sequence[str],
        iterables: Sequence[Expr],
        elt: Expr,
        loc: Location
    ):
        super().__init__(loc)
        self.vars = list(vars)
        self.iterables = list(iterables)
        self.elt = elt

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
        loc: Location
    ):
        super().__init__(loc)
        self.cond = cond
        self.ift = ift
        self.iff = iff

class Block(Ast):
    """FPy AST: list of statements"""
    stmts: list[Stmt]

    def __init__(self, stmts: list[Stmt]):
        assert stmts != [], "block must contain at least one statement"
        super().__init__(Location(
            stmts[0].loc.source,
            stmts[0].loc.start_line,
            stmts[0].loc.start_column,
            stmts[-1].loc.end_line,
            stmts[-1].loc.end_column
        ))
        self.stmts = stmts

class VarAssign(Stmt):
    """FPy AST: variable assignment"""
    var: str
    expr: Expr
    ann: Optional[TypeAnn]

    def __init__(
        self,
        var: str,
        expr: Expr,
        ann: Optional[TypeAnn],
        loc: Location
    ):
        super().__init__(loc)
        self.var = var
        self.expr = expr
        self.ann = ann

class TupleBinding(Ast):
    """FPy AST: tuple binding"""
    elts: list[str | Self]

    def __init__(
        self,
        vars: list[str | Self],
        loc: Location
    ):
        super().__init__(loc)
        self.elts = vars

    def names(self) -> set[str]:
        ids: set[str] = set()
        for v in self.elts:
            if isinstance(v, TupleBinding):
                ids |= v.names()
            elif isinstance(v, str):
                ids.add(v)
            else:
                raise NotImplementedError('unexpected tuple identifier', v)
        return ids
    
    def __iter__(self):
        return iter(self.elts)

class TupleAssign(Stmt):
    """FPy AST: tuple assignment"""
    binding: TupleBinding
    expr: Expr

    def __init__(
        self,
        vars: TupleBinding,
        expr: Expr,
        loc: Location
    ):
        super().__init__(loc)
        self.binding = vars
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
        loc: Location
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
        loc: Location
    ):
        super().__init__(loc)
        self.cond = cond
        self.body = body

class ForStmt(Stmt):
    """FPy AST: for statement"""
    var: str
    iterable: Expr
    body: Block

    def __init__(
        self,
        var: str,
        iterable: Expr,
        body: Block,
        loc: Location
    ):
        super().__init__(loc)
        self.var = var
        self.iterable = iterable
        self.body = body

class ContextStmt(Stmt):
    """FPy AST: with statement"""
    name: Optional[str]
    props: dict[str, Any]
    body: Block

    def __init__(
        self,
        name: Optional[str],
        props: dict[str, Any],
        body: Block,
        loc: Location
    ):
        super().__init__(loc)
        self.props = props
        self.name = name
        self.body = body

class Return(Stmt):
    """FPy AST: return statement"""
    expr: Expr

    def __init__(
        self,
        expr: Expr,
        loc: Location
    ):
        super().__init__(loc)
        self.expr = expr

class Argument(Ast):
    """FPy AST: function argument"""
    name: str
    type: Optional[TypeAnn]

    def __init__(
        self,
        name: str,
        type: Optional[TypeAnn],
        loc: Location
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
    globals: dict[str, Any]

    def __init__(
        self,
        name: str,
        args: list[Argument],
        body: Block,
        loc: Location
    ):
        super().__init__(loc)
        self.name = name
        self.args = args
        self.body = body
        self.ctx = {}
        self.globals = {}
