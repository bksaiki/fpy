"""
Runtime utilities for integrating with FPCore from Titanic.
"""

from typing import Any

from ..number import Context, IEEEContext, RM
from ..utils import default_repr


def _cvt_round_mode(mode: str):
    match mode:
        case 'nearestEven':
            return RM.RNE
        case 'nearestAway':
            return RM.RNA
        case 'toPositive':
            return RM.RTP
        case 'toNegative':
            return RM.RTN
        case 'toZero':
            return RM.RTZ
        case 'awayZero':
            return RM.RAZ
        case _:
            raise ValueError(f'Unknown rounding mode: {mode}')

class NoSuchContextError(Exception):
    """
    Exception raised when a context is not found.
    """

    def __init__(self, ctx: 'FPCoreContext'):
        """
        Initialize the exception with the given context.

        :param ctx: The context that was not found.
        """
        self.ctx = ctx

    def __str__(self):
        return f'No such context: {self.ctx}'


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

    def __init__(self, **kwargs):
        """
        Initialize the FPCore context with the given properties.

        :param props: A dictionary of properties.
        """
        self.props = kwargs


    def __enter__(self) -> 'FPCoreContext':
        raise RuntimeError('do not call')

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise RuntimeError('do not call')

    def with_prop(self, key: str, value: Any) -> 'FPCoreContext':
        """
        Create a new FPCore context with the given property.

        :param key: The property key.
        :param value: The property value.
        :return: A new FPCore context with the given property.
        """
        new_props = self.props.copy()
        new_props[key] = value
        return FPCoreContext(**new_props)

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
        """
        Converts the FPCore context to a context in this library.
        """
        prec = self.props.get('precision', 'binary64')
        rnd = self.props.get('round', 'nearestEven')
        try:
            match prec:
                # IEEE 754 shorthands
                case 'binary128':
                    return IEEEContext(15, 128, _cvt_round_mode(rnd))
                case 'binary64':
                    return IEEEContext(11, 64, _cvt_round_mode(rnd))
                case 'binary32':
                    return IEEEContext(8, 32, _cvt_round_mode(rnd))
                case 'binary16':
                    return IEEEContext(5, 16, _cvt_round_mode(rnd))
                case _:
                    raise NoSuchContextError(self)
        except ValueError:
            raise NoSuchContextError(self) from None
