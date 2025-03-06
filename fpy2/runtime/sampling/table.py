"""
Defines a range table, a map from variable to interval.
"""

from fractions import Fraction
from typing import Optional

from ...ir import Expr

class Endpoint:
    val: Optional[Fraction]
    closed: bool

    def __init__(self, val: Optional[Fraction], closed: bool):
        if val is None and closed:
            raise ValueError(f'invalid endpoint val={val}, closed={closed}')
        self.val = val
        self.closed = closed

# TODO: merge with other interval class
class Interval:
    lo: Endpoint
    hi: Endpoint


class RangeTable:
    """Mapping from variable to interval."""
    table: dict[str, Interval]

    def __init__(self):
        self.table = {}

    @staticmethod
    def from_expr(e: Expr):
        """Creates a range table from an expression."""
        return _parse_range(e)

def _parse_range(e: Expr) -> RangeTable:
    """Parses a range expression."""
    raise NotImplementedError
