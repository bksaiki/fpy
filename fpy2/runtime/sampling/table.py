"""
Defines a range table, a map from variable to interval.
"""

import math

from fractions import Fraction
from typing import Optional

from ...ir import *

_POS_INF = math.inf
_NEG_INF = -math.inf

class DisjointUnionError(Exception):
    """Exception raised when taking the union of disjoint intervals."""

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
        self.lo = Endpoint(lo, lo_closed)
        self.hi = Endpoint(hi, hi_closed)

    def __and__(self, other):
        """Intersection of two intervals."""
        if not isinstance(other, Interval):
            raise TypeError(f'expected Interval, got {type(other)}')

        lo = max(self.lo.val, other.lo.val)
        hi = min(self.hi.val, other.hi.val)

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
    valid: bool

    def __init__(
        self,
        table: dict[NamedId, Interval] = {},
        valid: bool = True
    ):
        self.table = table
        self.valid = valid

    @staticmethod
    def null():
        """Creates an invalid range table."""
        return RangeTable(valid=False)

    @staticmethod
    def from_precondition(pre: FunctionDef):
        """Creates a range table from an expression."""
        stmts = pre.body.stmts
        if len(stmts) != 1 or not isinstance(stmts[0], Return):
            raise ValueError(f'precondition must be a single return statement {pre.format()}')
        return _parse_expr(stmts[0].expr)

    def __and__(self, other):
        if not isinstance(other, RangeTable):
            raise TypeError(f'expected RangeTable, got {type(other)}')

        if not self.valid or not other.valid:
            return RangeTable.null()

        # process `self`
        merged = RangeTable()
        for var, ival in self.table.items():
            if var in other.table:
                merged.table[var] &= ival
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
        merged = RangeTable()
        for var, ival in self.table.items():
            if var in other.table:
                merged.table[var] |= ival
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
        case Integer():
            return Fraction(e.val)
        case _:
            raise RangeTableParseError(f'cannot represent {e} as a fraction')

def _parse_cmp(op: CompareOp, lhs: Expr, rhs: Expr):
    match (lhs, rhs):
        case (Var(), RealExpr()):
            n = _parse_number(rhs)
            match op:
                case CompareOp.EQ:
                    return RangeTable({lhs.name: Interval(n, n, True, True)})
                case CompareOp.NE:
                    raise RangeTableParseError(f'unsupported comparison {op}')
                case CompareOp.LT:
                    return RangeTable({lhs.name: Interval(_NEG_INF, n, False, False)})
                case CompareOp.LE:
                    return RangeTable({lhs.name: Interval(_NEG_INF, n, False, True)})
                case CompareOp.GT:
                    return RangeTable({lhs.name: Interval(n, _POS_INF, False, False)})
                case CompareOp.GE:
                    return RangeTable({lhs.name: Interval(n, _POS_INF, True, False)})
                case _:
                    raise RuntimeError(f'unreachable {op}')
        case (RealExpr(), Var()):
            return _parse_cmp(op.invert(), rhs, lhs)
        case _:
            raise RangeTableParseError(f'unsupported comparison {lhs} {op} {rhs}')

def _parse_expr(e: Expr) -> RangeTable:
    """Parses a range expression."""
    match e:
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
            raise RangeTableParseError(f'unsupported expression {e}')
