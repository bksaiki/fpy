"""
Defines a range table, a map from variable to interval.
"""

import math

from fractions import Fraction

from ...ir import *
from ...utils import digits_to_fraction, hexnum_to_fraction, default_repr

_POS_INF = math.inf
_NEG_INF = -math.inf

class DisjointUnionError(Exception):
    """Exception raised when taking the union of disjoint intervals."""

    def __init__(self, msg: str):
        super().__init__(msg)

class DisjointIntersectionError(Exception):
    """Exception raised when taking the intersection of disjoint intervals."""

    def __init__(self, msg: str):
        super().__init__(msg)

class RangeTableParseError(Exception):
    """Parsing error for `RangeTable.from_precondition()`."""

    def __init__(self, msg: str):
        super().__init__(msg)

class Endpoint:
    """An interval endpoint."""

    val: Fraction | float
    """
    Value of the endpoint.

    Any finite endpoint is a `Fraction`.
    Any infinite endpoint is a `float`, specifically `float('inf')` or `float('-inf')`.
    """

    closed: bool
    """Is the endpoint closed?"""

    def __init__(self, val: Fraction | float, closed: bool):
        if isinstance(val, float):
            if math.isfinite(val) or closed:
                raise ValueError(f'invalid endpoint val={val}, closed={closed}')
        elif not isinstance(val, Fraction):
            raise TypeError(f'expected Fraction | float, got {type(val)}')

        self.val = val
        self.closed = closed

# TODO: merge with other interval class
class Interval:
    lo: Endpoint
    hi: Endpoint

    def __init__(
        self,
        lo: Fraction | float,
        hi: Fraction | float,
        lo_closed: bool,
        hi_closed: bool,
    ):
        if lo > hi:
            raise ValueError(f'invalid interval lo={lo}, hi={hi}')
        self.lo = Endpoint(lo, lo_closed)
        self.hi = Endpoint(hi, hi_closed)

    def __repr__(self):
        lo = '[' if self.lo.closed else '('
        hi = ']' if self.hi.closed else ')'
        return f'{lo}{self.lo.val}, {self.hi.val}{hi}'

    def __and__(self, other):
        """Intersection of two intervals."""
        if not isinstance(other, Interval):
            raise TypeError(f'expected Interval, got {type(other)}')

        lo = max(self.lo.val, other.lo.val)
        hi = min(self.hi.val, other.hi.val)
        if lo > hi:
            raise DisjointIntersectionError(f'disjoint intervals {self} and {other}')

        if self.lo.val < other.lo.val:
            lo_closed = other.lo.closed
        elif self.lo.val > other.lo.val:
            lo_closed = self.lo.closed
        else:
            lo_closed = self.lo.closed and other.lo.closed

        if self.hi.val < other.hi.val:
            hi_closed = self.hi.closed
        elif self.hi.val > other.hi.val:
            hi_closed = other.hi.closed
        else:
            hi_closed = self.hi.closed and other.hi.closed

        return Interval(lo, hi, lo_closed, hi_closed)     


    def __or__(self, other):
        """
        Union of two intervals.

        If the intervals are non-overlapping, raises a `DisjointUnionError`.
        """
        if not isinstance(other, Interval):
            raise TypeError(f'expected Interval, got {type(other)}')

        raise NotImplementedError(self, other)

class RangeTable:
    """Mapping from variable to interval."""

    table: dict[NamedId, Interval]
    """mapping from variable to interval"""
    valid: bool
    """does any variable have no valid interval?"""
    sound: bool
    """is the range table sound?"""

    def __init__(
        self,
        table: Optional[dict[NamedId, Interval]] = None,
        valid: bool = True,
        sound: bool = True,
    ):
        if table is None:
            self.table = {}
        else:
            self.table = table

        self.valid = valid
        self.sound = sound

    def __repr__(self):
        return f'{self.__class__.__name__}(table={str(self.table)}, valid={self.valid}, sound={self.sound})'

    def __getitem__(self, key: NamedId) -> Interval:
        return self.table[key]

    def __setitem__(self, key: NamedId, value: Interval):
        self.table[key] = value

    @staticmethod
    def null():
        """Creates an invalid range table."""
        return RangeTable(valid=False)

    @staticmethod
    def unsound():
        """Creates an empty unsound range table."""
        return RangeTable(sound=False)

    @staticmethod
    def default_interval():
        return Interval(_NEG_INF, _POS_INF, False, False)

    @staticmethod
    def from_condition(cond: FunctionDef):
        """Creates a range table from an expression."""
        stmts = cond.body.stmts
        if len(stmts) != 1 or not isinstance(stmts[0], Return):
            raise ValueError(f'precondition must be a single return statement {cond.format()}')
        return _parse_expr(stmts[0].expr)

    def __and__(self, other):
        if not isinstance(other, RangeTable):
            raise TypeError(f'expected RangeTable, got {type(other)}')

        if not self.valid or not other.valid:
            return RangeTable.null()

        # process `self`
        sound = self.sound and other.sound
        merged = RangeTable(sound=sound)
        for var, ival in self.table.items():
            if var in other.table:
                merged.table[var] = ival & other.table[var]
            else:
                merged.table[var] = ival

        # process `other`
        for var, ival in other.table.items():
            if var not in self.table:
                merged.table[var] = ival

        return merged

    def __or__(self, other):
        if not isinstance(other, RangeTable):
            raise TypeError(f'expected RangeTable, got {type(other)}')

        if not self.valid and not other.valid:
            return RangeTable.null()

        # process `self`
        sound = self.sound and other.sound
        merged = RangeTable(sound=sound)
        for var, ival in self.table.items():
            if var in other.table:
                merged.table[var] = ival | other.table[var]
            else:
                merged.table[var] = ival

        # process `other`
        for var, ival in other.table.items():
            if var not in self.table:
                merged.table[var] = ival

        return merged


def _parse_number(e: RealExpr) -> Fraction | float:
    """Parses a real expression into a fraction."""
    match e:
        case Decnum() | Integer():
            return Fraction(e.val)
        case Hexnum():
            return hexnum_to_fraction(e.val)
        case Digits():
            return digits_to_fraction(e.m, e.e, e.b)
        case Rational():
            return Fraction(e.p, e.q)
        case _:
            raise RangeTableParseError(f'cannot represent {e} as a fraction')


def _parse_cmp2(op: CompareOp, x: NamedId, n: Fraction | float) -> RangeTable:
    match op:
        case CompareOp.EQ:
            return RangeTable({x: Interval(n, n, True, True)})
        case CompareOp.NE:
            # TODO: unsupported
            return RangeTable.unsound()
        case CompareOp.LT:
            return RangeTable({x: Interval(_NEG_INF, n, False, False)})
        case CompareOp.LE:
            return RangeTable({x: Interval(_NEG_INF, n, False, True)})
        case CompareOp.GT:
            return RangeTable({x: Interval(n, _POS_INF, False, False)})
        case CompareOp.GE:
            return RangeTable({x: Interval(n, _POS_INF, True, False)})
        case _:
            raise RuntimeError(f'unreachable {op}')

def _parse_cmp(op: CompareOp, lhs: Expr, rhs: Expr):
    match (lhs, rhs):
        case (Var(), Var()):
            # unsupported
            return RangeTable.unsound()
        case (_, Var()):
            return _parse_cmp(op.invert(), rhs, lhs)
        case (Var(), RealExpr()):
            try:
                n = _parse_number(rhs)
                return _parse_cmp2(op, lhs.name, n)
            except RangeTableParseError:
                return RangeTable.unsound()
        case (Var(), Neg()):
            if not isinstance(rhs.children[0], RealExpr):
                # unsupported
                return RangeTable.unsound()
            try:
                n = _parse_number(rhs.children[0])
                return _parse_cmp2(op, lhs.name, -n)
            except RangeTableParseError:
                return RangeTable.unsound()
        case _:
            # unsupported
            return RangeTable.unsound()

def _parse_expr(e: Expr) -> RangeTable:
    """Parses a range expression."""
    match e:
        case Bool():
            if e.val:
                return RangeTable()
            else:
                return RangeTable.null()
        case Compare():
            table = RangeTable()
            for op, lhs, rhs in zip(e.ops, e.children, e.children[1:]):
                table &= _parse_cmp(op, lhs, rhs)
            return table
        case And():
            table = RangeTable()
            for child in e.children:
                table &= _parse_expr(child)
            return table
        case Or():
            table = RangeTable.null()
            for child in e.children:
                table |= _parse_expr(child)
            return table
        case _:
            # unsupported
            return RangeTable.unsound()
