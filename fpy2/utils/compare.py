"""Comparison operators"""

from enum import Enum

class CompareOp(Enum):
    """Comparison operators as an enumeration"""
    LT = 0
    LE = 1
    GE = 2
    GT = 3
    EQ = 4
    NE = 5

    def symbol(self):
        """Get the symbol for the operator."""
        return _symbol_table[self]

    def invert(self):
        """Assuming `a op b`, returns `op` such that `b op a`."""
        match self:
            case CompareOp.LT:
                return CompareOp.GT
            case CompareOp.LE:
                return CompareOp.GE
            case CompareOp.GE:
                return CompareOp.LE
            case CompareOp.GT:
                return CompareOp.LT
            case CompareOp.EQ:
                return CompareOp.EQ
            case CompareOp.NE:
                return CompareOp.NE
            case _:
                raise RuntimeError('unreachable')


_symbol_table = {
    CompareOp.LT: '<',
    CompareOp.LE: '<=',
    CompareOp.GE: '>=',
    CompareOp.GT: '>',
    CompareOp.EQ: '==',
    CompareOp.NE: '!='
}
