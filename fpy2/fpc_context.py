"""
Runtime utilities for integrating with FPCore from Titanic.
"""

from typing import Any

from .number import (
    Context,
    FixedContext,
    IEEEContext,
    MPFixedContext,
    RealContext,
    RM, OV,
    FP128, FP64, FP32, FP16,
    INTEGER
)

###########################################################
# Rounding modes

_round_mode = {
    'nearestEven': RM.RNE,
    'nearestAway': RM.RNA,
    'toPositive': RM.RTP,
    'toNegative': RM.RTN,
    'toZero': RM.RTZ,
    'awayZero': RM.RAZ,
}

def _round_mode_to_fpy(mode: str):
    if mode not in _round_mode:
        raise ValueError(f'Unknown rounding mode: {mode}')
    return _round_mode[mode]

def _round_mode_from_fpc(mode: RM):
    _invert_round_mode = { v: k for k, v in _round_mode.items() }
    if mode not in _invert_round_mode:
        raise ValueError(f'Unknown rounding mode: {mode}')
    return _invert_round_mode[mode]

###########################################################
# Overflow mode

_overflow_mode = {
    'wrap': OV.WRAP,
    'clamp': OV.SATURATE,
    'infinity': OV.OVERFLOW,
}

def _overflow_mode_to_fpc(mode: str):
    if mode not in _overflow_mode:
        raise ValueError(f'Unknown overflow mode: {mode}')
    return _overflow_mode[mode]

def _overflow_mode_from_fpc(mode: OV):
    _invert_overflow_mode = { v: k for k, v in _overflow_mode.items() }
    if mode not in _invert_overflow_mode:
        raise ValueError(f'Unknown overflow mode: {mode}')
    return _invert_overflow_mode[mode]

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

        match ctx:
            case IEEEContext():
                rm = _round_mode_from_fpc(ctx.rm)
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
                rm = _round_mode_from_fpc(ctx.rm)
                if ctx.nmin == -1:
                    return FPCoreContext(precision='integer', round=rm)
                else:
                    return FPCoreContext(n=ctx.nmin, round=rm)
            case FixedContext():
                if not ctx.signed:
                    raise RuntimeError('Cannot convert unsigned FixedContext to an FPCore context')
                rm = _round_mode_from_fpc(ctx.rm)
                of = _overflow_mode_from_fpc(ctx.overflow)
                return FPCoreContext(precision=['fixed', ctx.nbits, ctx.scale], round=rm, overflow=of)
            case RealContext():
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
        try:
            match prec:
                # IEEE 754 long form
                case ['float', es, nbits]:
                    return IEEEContext(int(es), int(nbits), _round_mode_to_fpy(rnd))
                # IEEE 754 shorthands
                case 'binary128':
                    return FP128.with_params(rm=_round_mode_to_fpy(rnd))
                case 'binary80':
                    return IEEEContext(15, 79, _round_mode_to_fpy(rnd))
                case 'binary64':
                    return FP64.with_params(rm=_round_mode_to_fpy(rnd))
                case 'binary32':
                    return FP32.with_params(rm=_round_mode_to_fpy(rnd))
                case 'binary16':
                    return FP16.with_params(rm=_round_mode_to_fpy(rnd))
                # fixed-point context
                case ['fixed', nbits, scale]:
                    return FixedContext(True, int(nbits), int(scale), _round_mode_to_fpy(rnd), _overflow_mode_to_fpc(ov))
                # integer context
                case 'integer':
                    return INTEGER.with_params(rm=_round_mode_to_fpy(rnd))
                # real context
                case 'real':
                    return RealContext()
                case _:
                    raise NoSuchContextError(self)
        except ValueError:
            raise NoSuchContextError(self) from None
