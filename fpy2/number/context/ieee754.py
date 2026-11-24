"""
This module defines floating-point numbers as defined
by the IEEE 754 standard.
"""

from typing import Any

from ...utils import DEFAULT, DefaultOr, default_or
from ..number import RNG, Float
from ..round import OverflowMode, RoundingMode
from .efloat import EFloatContext, EFloatFormat, EFloatNanKind


class IEEEFormat(EFloatFormat):
    """
    Number format for IEEE 754 floating-point values.

    This format is parameterized by the exponent field size `es`
    and total representation size `nbits`.
    It describes the set of representable values for `IEEEContext`.
    """

    def __init__(self, es: int, nbits: int) -> None:
        super().__init__(es, nbits, True, EFloatNanKind.IEEE_754, 0)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(es={self.es}, nbits={self.nbits})'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IEEEFormat) and self.es == other.es and self.nbits == other.nbits

    def __hash__(self) -> int:
        return hash((self.__class__, self.es, self.nbits))


class IEEEContext(EFloatContext):
    """
    Rounding context for IEEE 754 floating-point values.

    This context is parameterized by the size of
    the exponent field `es`, the size of the total
    representation `nbits`, and the rounding mode `rm`.

    This context is implemented as a subclass of `EFloatContext` which is
    a more general definition of IEEE 754-like floating-point numbers.
    """

    def __init__(
        self,
        es: int,
        nbits: int,
        rm: RoundingMode = RoundingMode.RNE,
        overflow: OverflowMode = OverflowMode.OVERFLOW,
        num_randbits: int | None = 0,
        *,
        rng: RNG | None = None
    ) -> None:
        super().__init__(es, nbits, True, EFloatNanKind.IEEE_754, 0, rm, overflow, num_randbits, rng=rng)

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(es={self.es}, nbits={self.nbits}, rm={self.rm!r}, overflow={self.overflow!r}, num_randbits={self.num_randbits!r})'

    def with_params(
        self, *,
        es: DefaultOr[int] = DEFAULT,
        nbits: DefaultOr[int] = DEFAULT,
        rm: DefaultOr[RoundingMode] = DEFAULT,
        overflow: DefaultOr[OverflowMode] = DEFAULT,
        num_randbits: DefaultOr[int | None] = DEFAULT,
        rng: DefaultOr[RNG | None] = DEFAULT,
        **kwargs: Any
    ) -> 'IEEEContext':
        if kwargs:
            raise TypeError(f'Unexpected parameters {kwargs} for IEEEContext')
        return IEEEContext(
            default_or(es, self.es),
            default_or(nbits, self.nbits),
            default_or(rm, self.rm),
            default_or(overflow, self.overflow),
            default_or(num_randbits, self.num_randbits),
            rng=default_or(rng, self.rng),
        )

    def format(self) -> IEEEFormat:
        return IEEEFormat(self.es, self.nbits)

    @classmethod
    def from_format(
        cls,
        fmt: EFloatFormat,
        *,
        rm: RoundingMode = RoundingMode.RNE,
        overflow: OverflowMode | None = None,
        num_randbits: int | None = 0,
        rng: RNG | None = None,
        nan_value: Float | None = None,
        inf_value: Float | None = None,
    ) -> 'IEEEContext':
        """Creates a context from an `IEEEFormat` and rounding parameters."""
        if not isinstance(fmt, IEEEFormat):
            raise TypeError(f'Expected \'IEEEFormat\', got {type(fmt)}')
        if nan_value is not None or inf_value is not None:
            raise ValueError('an IEEE format holds NaN and infinity: nan_value and inf_value do not apply')
        if overflow is None:
            overflow = OverflowMode.OVERFLOW
        return cls(fmt.es, fmt.nbits, rm, overflow, num_randbits, rng=rng)
