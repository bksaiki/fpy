from enum import Enum
from typing import Literal, cast


class Default(Enum):
    """
    `None`-like object which is not `None`

    Useful for functions that allow a keyword to be not provided
    but `None` is a valid "provided" value.
    """
    DEFAULT = "DEFAULT"

    def __repr__(self) -> str:
        return "DEFAULT"

DEFAULT = Default.DEFAULT


type DefaultOr[T] = T | Literal[Default.DEFAULT]


def default_or[T](x: object, default: T) -> T:
    """*x*, or *default* where *x* is `DEFAULT`."""
    return default if x is DEFAULT else cast(T, x)
