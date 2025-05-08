"""
Runtime utilities for integrating with FPCore from Titanic.
"""

from typing import Any

from ..number import Context
from ..utils import default_repr

@default_repr
class FPCoreContext:
    """
    FPCore rounding context.

    FPCore defines a rounding context to be a dictionary of properties.
    Each property consists of an arbitrary string key and an arbitrary value.

    The FPCore standard defines a set of standard properties:
    - `precision`: the precision of the floating-point numbers
    - `round`: the rounding mode to use
    - `overflow`: overflow behavior for fixed-point numbers

    This class provides a way to define an FPCore-style rounding context
    and convert to and from rounding contexts in this library.
    """

    props: dict[str, Any]


    def __init__(self, props: dict[str, Any]):
        """
        Initialize the FPCore context with the given properties.

        :param props: A dictionary of properties.
        """
        self.props = dict(props)

    def with_prop(self, key: str, value: Any) -> 'FPCoreContext':
        """
        Create a new FPCore context with the given property.

        :param key: The property key.
        :param value: The property value.
        :return: A new FPCore context with the given property.
        """
        new_props = self.props.copy()
        new_props[key] = value
        return FPCoreContext(new_props)

    @staticmethod
    def from_context(ctx: Context) -> 'FPCoreContext':
        """
        Create an FPCore context from a given context.

        :param ctx: The context to convert.
        :return: An FPCore context with the properties of the given context.
        """
        if not isinstance(ctx, Context):
            raise TypeError(f'Expected \'Context\' for ctx={ctx}, got {type(ctx)}')
        raise NotImplementedError


    def to_context(self) -> Context:
        raise NotImplementedError
