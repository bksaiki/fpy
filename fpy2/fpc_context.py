"""
Runtime utilities for integrating with FPCore from Titanic.
"""

from typing import Any, Self

from .number import (
    FP16,
    FP32,
    FP64,
    FP128,
    INTEGER,
    OV,
    REAL,
    RM,
    Context,
    FixedContext,
    IEEEContext,
    MPFixedContext,
)

__all__ = [
    'FPCoreContext',
    'NoSuchContextError',
]

###########################################################
# Rounding and overflow modes

_RM_FROM_FPC = {
    'nearestEven': RM.RNE,
    'nearestAway': RM.RNA,
    'toPositive': RM.RTP,
    'toNegative': RM.RTN,
    'toZero': RM.RTZ,
    'awayZero': RM.RAZ,
}
_RM_TO_FPC = {v: k for k, v in _RM_FROM_FPC.items()}

_OV_FROM_FPC = {
    'wrap': OV.WRAP,
    'clamp': OV.SATURATE,
    'infinity': OV.OVERFLOW,
}
_OV_TO_FPC = {v: k for k, v in _OV_FROM_FPC.items()}


def _lookup[K, V](table: dict[K, V], key: K, what: str) -> V:
    if key not in table:
        raise ValueError(f'Unknown {what}: {key}')
    return table[key]

###########################################################
# FPCore (legacy) context

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

    def __repr__(self):
        return self.__class__.__name__ + '(' + ', '.join(f'{k}={v!r}' for k, v in self.props.items()) + ')'

    def __enter__(self) -> Self:
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

        match ctx:
            case IEEEContext():
                rm = _lookup(_RM_TO_FPC, ctx.rm, 'rounding mode')
                match (ctx.es, ctx.nbits):
                    case (15, 128):
                        return FPCoreContext(precision='binary128', round=rm)
                    case (15, 79):
                        return FPCoreContext(precision='binary80', round=rm)
                    case (11, 64):
                        return FPCoreContext(precision='binary64', round=rm)
                    case (8, 32):
                        return FPCoreContext(precision='binary32', round=rm)
                    case (5, 16):
                        return FPCoreContext(precision='binary16', round=rm)
                    case _:
                        return FPCoreContext(precision=['float', ctx.es, ctx.nbits], round=rm)
            case MPFixedContext():
                rm = _lookup(_RM_TO_FPC, ctx.rm, 'rounding mode')
                if ctx.nmin == -1:
                    return FPCoreContext(precision='integer', round=rm)
                else:
                    return FPCoreContext(n=ctx.nmin, round=rm)
            case FixedContext():
                if not ctx.signed:
                    raise RuntimeError('Cannot convert unsigned FixedContext to an FPCore context')
                rm = _lookup(_RM_TO_FPC, ctx.rm, 'rounding mode')
                of = _lookup(_OV_TO_FPC, ctx.overflow, 'overflow mode')
                return FPCoreContext(precision=['fixed', ctx.nbits, ctx.scale], round=rm, overflow=of)
            case _ if ctx is REAL:
                return FPCoreContext(precision='real')
            case _:
                raise RuntimeError(f'Cannot convert to an FPCore context {type(ctx)}')


    def to_context(self) -> Context:
        """
        Converts the FPCore context to a context in this library.
        """
        prec = self.props.get('precision', 'binary64')
        rnd = self.props.get('round', 'nearestEven')
        ov = self.props.get('overflow', 'infinity')
        if 'n' in self.props and 'precision' not in self.props:
            # FPy's own property for an unbounded fixed-point context (`from_context`)
            return MPFixedContext(int(self.props['n']), _lookup(_RM_FROM_FPC, rnd, 'rounding mode'))
        try:
            rm = _lookup(_RM_FROM_FPC, rnd, 'rounding mode')
            match prec:
                # IEEE 754 long form
                case ['float', es, nbits]:
                    return IEEEContext(int(es), int(nbits), rm)
                # IEEE 754 shorthands
                case 'binary128':
                    return FP128.with_params(rm=rm)
                case 'binary80':
                    return IEEEContext(15, 79, rm)
                case 'binary64':
                    return FP64.with_params(rm=rm)
                case 'binary32':
                    return FP32.with_params(rm=rm)
                case 'binary16':
                    return FP16.with_params(rm=rm)
                # fixed-point context
                case ['fixed', nbits, scale]:
                    return FixedContext(True, int(scale), int(nbits), rm, _lookup(_OV_FROM_FPC, ov, 'overflow mode'))
                # integer context
                case 'integer':
                    return INTEGER
                # real context
                case 'real':
                    return REAL
                case _:
                    raise NoSuchContextError(self)
        except ValueError:
            raise NoSuchContextError(self) from None
