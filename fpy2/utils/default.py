from typing import TypeVar, Union, Literal, TypeAlias
from enum import Enum

"""
Default value instead of `None`
"""

class Default(Enum):
    """
    `None`-like object which is not `None`

    Useful for functions that allow a keyword to be not provided
    but `None` is a valid "provided" value.
    """
    DEFAULT = "DEFAULT"

    def __repr__(self):
        return "DEFAULT"

DEFAULT = Default.DEFAULT


T = TypeVar('T')
DefaultOr: TypeAlias = Union[T, Literal[Default.DEFAULT]]
