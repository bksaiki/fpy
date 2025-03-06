"""
Defines a range table, a map from variable to interval.
"""

from fractions import Fraction
from typing import Optional

from ...ir import *

class Endpoint:
    """An interval endpoint."""

    val: Optional[Fraction]
    """Value of the endpoint. If `None`, the endpoint is infinite."""

    closed: bool
    """Is the endpoint closed?"""

    def __init__(self, val: Optional[Fraction], closed: bool):
        if val is None and closed:
            raise ValueError(f'invalid endpoint val={val}, closed={closed}')
        self.val = val
        self.closed = closed

# TODO: merge with other interval class
class Interval:
    lo: Endpoint
    hi: Endpoint

    def __init__(
        self,
        lo: Optional[Fraction],
        hi: Optional[Fraction],
        lo_closed: bool,
        hi_closed: bool,
    ):
        self.lo = Endpoint(lo, lo_closed)
        self.hi = Endpoint(hi, hi_closed)

    def __and__(self, other):
        raise NotImplementedError


class RangeTable:
    """Mapping from variable to interval."""
    table: dict[str, Interval]
    valid: bool

    def __init__(self, valid: bool = True):
        self.table = {}
        self.valid = True

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


def _parse_expr(e: Expr) -> RangeTable:
    """Parses a range expression."""
    match e:
        case Compare():
            if len(e.children) == 3 and e.ops[0] == e.ops[1]:
                # a cmp b cmp c
                raise NotImplementedError(e)
            else:
                raise NotImplementedError(e)
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
            raise NotImplementedError(e)
