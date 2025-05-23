"""
This module contains the AST for FPy programs.
"""

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any, Optional, Self, Sequence

from ..fpc_context import FPCoreContext
from ..number import Context
from ..utils import CompareOp, Id, NamedId, UnderscoreId, Location, default_repr


class UnaryOpKind(IntEnum):
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

class BinaryOpKind(IntEnum):
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

class TernaryOpKind(IntEnum):
    # ternary operators
    FMA = 0

class NaryOpKind(IntEnum):
    # boolean operations
    AND = 1
    OR = 2

@default_repr
class Ast(ABC):
    """FPy AST: abstract base class for all AST nodes."""
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

    def __init__(self, loc: Optional[Location]):
        super().__init__(loc)

class ScalarType(IntEnum):
    ANY = 0
    REAL = 1
    BOOL = 2

class AnyTypeAnn(TypeAnn):
    """FPy AST: any type annotation"""

    def __init__(self, loc: Optional[Location]):
        super().__init__(loc)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AnyTypeAnn)

    def __hash__(self) -> int:
        return hash(())

class ScalarTypeAnn(TypeAnn):
    """FPy AST: scalar type annotation"""
    kind: ScalarType

    def __init__(self, kind: ScalarType, loc: Optional[Location]):
        super().__init__(loc)
        self.kind = kind

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ScalarTypeAnn) and self.kind == other.kind

    def __hash__(self) -> int:
        return hash(self.kind)

class TupleTypeAnn(TypeAnn):
    """FPy AST: tuple type annotation"""
    elts: list[TypeAnn]

    def __init__(self, elts: list[TypeAnn], loc: Optional[Location]):
        super().__init__(loc)
        self.elts = elts

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TupleTypeAnn):
            return False
        return self.elts == other.elts

    def __hash__(self) -> int:
        return hash(tuple(self.elts))

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Var):
            return False
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

class BoolVal(ValueExpr):
    """FPy AST: boolean value"""
    val: bool

    def __init__(self, val: bool, loc: Optional[Location]):
        super().__init__(loc)
        self.val = val

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoolVal):
            return False
        return self.val == other.val

    def __hash__(self) -> int:
        return hash(self.val)

class RealVal(ValueExpr):
    """FPy AST: real value"""

    def __init__(self, loc: Optional[Location]):
        super().__init__(loc)

class ForeignVal(ValueExpr):
    """FPy AST: native Python value"""
    val: Any

    def __init__(self, val: Any, loc: Optional[Location]):
        super().__init__(loc)
        self.val = val

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ForeignVal):
            return False
        return self.val == other.val

    def __hash__(self) -> int:
        return hash(self.val)

class Decnum(RealVal):
    """FPy AST: decimal number"""
    val: str

    def __init__(self, val: str, loc: Optional[Location]):
        super().__init__(loc)
        self.val = val

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Decnum):
            return False
        return self.val == other.val

    def __hash__(self) -> int:
        return hash(self.val)

class Hexnum(RealVal):
    """FPy AST: hexadecimal number"""
    val: str

    def __init__(self, val: str, loc: Optional[Location]):
        super().__init__(loc)
        self.val = val

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Hexnum):
            return False
        return self.val == other.val

    def __hash__(self) -> int:
        return hash(self.val)

class Integer(RealVal):
    """FPy AST: integer"""
    val: int

    def __init__(self, val: int, loc: Optional[Location]):
        super().__init__(loc)
        self.val = val

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Integer):
            return False
        return self.val == other.val

    def __hash__(self) -> int:
        return hash(self.val)


class Rational(RealVal):
    """FPy AST: rational number"""
    p: int
    q: int

    def __init__(self, p: int, q: int, loc: Optional[Location]):
        super().__init__(loc)
        self.p = p
        self.q = q

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rational):
            return False
        return self.p == other.p and self.q == other.q

    def __hash__(self) -> int:
        return hash((self.p, self.q))

class Digits(RealVal):
    """FPy AST: scientific notation"""
    m: int
    e: int
    b: int

    def __init__(self, m: int, e: int, b: int, loc: Optional[Location]):
        super().__init__(loc)
        self.m = m
        self.e = e
        self.b = b

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Digits):
            return False
        return self.m == other.m and self.e == other.e and self.b == other.b

    def __hash__(self) -> int:
        return hash((self.m, self.e, self.b))

class Constant(RealVal):
    """FPy AST: constant expression"""
    val: str

    def __init__(self, val: str, loc: Optional[Location]):
        super().__init__(loc)
        self.val = val

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Constant):
            return False
        return self.val == other.val

    def __hash__(self) -> int:
        return hash(self.val)

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnaryOp):
            return False
        return self.op == other.op and self.arg == other.arg

    def __hash__(self) -> int:
        return hash((self.op, self.arg))

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BinaryOp):
            return False
        return self.op == other.op and self.left == other.left and self.right == other.right

    def __hash__(self) -> int:
        return hash((self.op, self.left, self.right))

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TernaryOp):
            return False
        return self.op == other.op and self.arg0 == other.arg0 and self.arg1 == other.arg1 and self.arg2 == other.arg2
    
    def __hash__(self) -> int:
        return hash((self.op, self.arg0, self.arg1, self.arg2))

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NaryOp):
            return False
        return self.op == other.op and self.args == other.args
    
    def __hash__(self) -> int:
        return hash((self.op, tuple(self.args)))

class Call(Expr):
    """FPy AST: function call"""
    op: str
    args: list[Expr]

    def __init__(
        self,
        op: str,
        args: Sequence[Expr],
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.op = op
        self.args = list(args)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Call):
            return False
        return self.op == other.op and self.args == other.args

    def __hash__(self) -> int:
        return hash((self.op, tuple(self.args)))

class Compare(Expr):
    """FPy AST: comparison chain"""
    ops: list[CompareOp]
    args: list[Expr]

    def __init__(
        self,
        ops: Sequence[CompareOp],
        args: Sequence[Expr],
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.ops = list(ops)
        self.args = list(args)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Compare):
            return False
        return self.ops == other.ops and self.args == other.args

    def __hash__(self) -> int:
        return hash((tuple(self.ops), tuple(self.args)))

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TupleExpr):
            return False
        return self.args == other.args

    def __hash__(self) -> int:
        return hash(tuple(self.args))

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompExpr):
            return False
        return self.vars == other.vars and self.iterables == other.iterables and self.elt == other.elt
    
    def __hash__(self) -> int:
        return hash((tuple(self.vars), tuple(self.iterables), self.elt))

class TupleRef(Expr):
    """FPy AST: tuple indexing expression"""
    value: Expr
    slices: list[Expr]

    def __init__(self, value: Expr, slices: Sequence[Expr], loc: Optional[Location]):
        super().__init__(loc)
        self.value = value
        self.slices = list(slices)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TupleRef):
            return False
        return self.value == other.value and self.slices == other.slices

    def __hash__(self) -> int:
        return hash((self.value, tuple(self.slices)))

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IfExpr):
            return False
        return self.cond == other.cond and self.ift == other.ift and self.iff == other.iff

    def __hash__(self) -> int:
        return hash((self.cond, self.ift, self.iff))

class ForeignAttribute(Ast):
    """
    FPy AST: attribute of a foreign object, e.g., `x.y`
    Attributes may be nested, e.g., `x.y.z`.
    """
    name: NamedId
    attrs: list[NamedId]

    def __init__(self, name: NamedId, attrs: Sequence[NamedId], loc: Optional[Location]):
        super().__init__(loc)
        self.name = name
        self.attrs = list(attrs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ForeignAttribute):
            return False
        return self.name == other.name and self.attrs == other.attrs

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.attrs)))


class ContextExpr(Expr):
    """FPy AST: context constructor"""
    ctor: Var | ForeignAttribute
    args: list[Expr | ForeignAttribute]
    kwargs: list[tuple[str, Expr | ForeignAttribute]]

    def __init__(
        self,
        ctor: Var | ForeignAttribute,
        args: Sequence[Expr | ForeignAttribute],
        kwargs: Sequence[tuple[str, Expr | ForeignAttribute]],
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.ctor = ctor
        self.args = list(args)
        self.kwargs = list(kwargs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContextExpr):
            return False
        return self.ctor == other.ctor and self.args == other.args and self.kwargs == other.kwargs

    def __hash__(self) -> int:
        return hash((self.ctor, tuple(self.args), tuple(self.kwargs)))


class ContextAttribute(Ast):
    """FPy AST: context attribute"""
    expr: Expr
    name: str

    def __init__(self, expr: Expr, name: str, loc: Optional[Location]):
        super().__init__(loc)
        self.expr = expr
        self.name = name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContextAttribute):
            return False
        return self.expr == other.expr and self.name == other.name

    def __hash__(self) -> int:
        return hash((self.expr, self.name))

class ContextUpdate(Ast):
    """FPy AST: context update"""
    expr: Expr
    kwargs: dict[str, Expr]

    def __init__(self, expr: Expr, kwargs: dict[str, Expr], loc: Optional[Location]):
        super().__init__(loc)
        self.expr = expr
        self.kwargs = kwargs

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContextUpdate):
            return False
        return self.expr == other.expr and self.kwargs == other.kwargs

    def __hash__(self) -> int:
        return hash((self.expr, tuple(self.kwargs.items())))


class StmtBlock(Ast):
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StmtBlock):
            return False
        return self.stmts == other.stmts

    def __hash__(self) -> int:
        return hash(tuple(self.stmts))

class SimpleAssign(Stmt):
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimpleAssign):
            return False
        return self.var == other.var and self.expr == other.expr and self.ann == other.ann
    
    def __hash__(self) -> int:
        return hash((self.var, self.expr, self.ann))

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TupleBinding):
            return False
        return self.elts == other.elts

    def __hash__(self) -> int:
        return hash(tuple(self.elts))

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

class TupleUnpack(Stmt):
    """FPy AST: unpacking / destructing a tuple"""
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TupleUnpack):
            return False
        return self.binding == other.binding and self.expr == other.expr

    def __hash__(self) -> int:
        return hash((self.binding, self.expr))

class IndexAssign(Stmt):
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IndexAssign):
            return False
        return self.var == other.var and self.slices == other.slices and self.expr == other.expr

    def __hash__(self) -> int:
        return hash((self.var, tuple(self.slices), self.expr))

class If1Stmt(Stmt):
    """FPy AST: if statement with one branch"""
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, If1Stmt):
            return False
        return self.cond == other.cond and self.body == other.body

    def __hash__(self) -> int:
        return hash((self.cond, self.body))

class IfStmt(Stmt):
    """FPy AST: if statement (with two branhces)"""
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IfStmt):
            return False
        return self.cond == other.cond and self.ift == other.ift and self.iff == other.iff

    def __hash__(self) -> int:
        return hash((self.cond, self.ift, self.iff))

class WhileStmt(Stmt):
    """FPy AST: while statement"""
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WhileStmt):
            return False
        return self.cond == other.cond and self.body == other.body

    def __hash__(self) -> int:
        return hash((self.cond, self.body))

class ForStmt(Stmt):
    """FPy AST: for statement"""
    var: Id
    iterable: Expr
    body: StmtBlock

    def __init__(
        self,
        var: Id,
        iterable: Expr,
        body: StmtBlock,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.var = var
        self.iterable = iterable
        self.body = body

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ForStmt):
            return False
        return self.var == other.var and self.iterable == other.iterable and self.body == other.body

    def __hash__(self) -> int:
        return hash((self.var, self.iterable, self.body))

class ContextStmt(Stmt):
    """FPy AST: with statement"""
    name: Id
    ctx: ContextExpr | Var | ForeignVal
    body: StmtBlock

    def __init__(
        self,
        name: Id,
        ctx: ContextExpr | Var | ForeignVal,
        body: StmtBlock,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.ctx = ctx
        self.name = name
        self.body = body

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContextStmt):
            return False
        return self.name == other.name and self.ctx == other.ctx and self.body == other.body

    def __hash__(self) -> int:
        return hash((self.name, self.ctx, self.body))

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AssertStmt):
            return False
        return self.test == other.test and self.msg == other.msg

    def __hash__(self) -> int:
        return hash((self.test, self.msg))

class EffectStmt(Stmt):
    """FPy AST: an expression without a result"""
    expr: Expr

    def __init__(
        self,
        expr: Expr,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.expr = expr

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EffectStmt):
            return False
        return self.expr == other.expr

    def __hash__(self) -> int:
        return hash(self.expr)

class ReturnStmt(Stmt):
    """FPy AST: return statement"""
    expr: Expr

    def __init__(
        self,
        expr: Expr,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.expr = expr

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReturnStmt):
            return False
        return self.expr == other.expr

    def __hash__(self) -> int:
        return hash(self.expr)

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Argument):
            return False
        return self.name == other.name and self.type == other.type

    def __hash__(self) -> int:
        return hash((self.name, self.type))

class FuncDef(Ast):
    """FPy AST: function definition"""
    name: str
    args: list[Argument]
    body: StmtBlock
    ctx: dict[str, Any]
    free_vars: set[NamedId]

    def __init__(
        self,
        name: str,
        args: Sequence[Argument],
        body: StmtBlock,
        loc: Optional[Location]
    ):
        super().__init__(loc)
        self.name = name
        self.args = list(args)
        self.body = body
        self.ctx = {}
        self.free_vars = set()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FuncDef):
            return False
        return self.name == other.name and self.args == other.args and self.body == other.body

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.args), self.body))


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
