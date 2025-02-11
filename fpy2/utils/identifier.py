"""
Module for identifiers.

Strings are not sufficient as identifiers, especially when (re)-generating
unique identifiers.
"""

from abc import ABC
from typing import Optional


class Id(ABC):
    """Abstract base class for identifiers."""

    def __repr__(self):
        return f'Id(\'{str(self)}\')'

class UnderscoreId(Id):
    """
    Placeholder identifier.

    When used in an assignment, the value is not bound.
    This identifier is illegal as a variable.
    """

    def __str__(self):
        return '_'

class NamedId(Id):
    """
    Named identifier.

    A named identifier consists of a base name and an optional
    count to indicate its version.
    """

    base: str
    count: Optional[int]

    def __init__(self, base: str, count: Optional[int] = None):
        if not isinstance(base, str):
            raise TypeError(f'expected a str, for {base}')
        self.base = base
        self.count = count

    def __str__(self):
        if self.count is None:
            return self.base
        else:
            return f'{self.base}{self.count}'

    def __eq__(self, other):
        if not isinstance(other, NamedId):
            return False
        return self.base == other.base and self.count == other.count

    def __hash__(self):
        return hash((self.base, self.count))
