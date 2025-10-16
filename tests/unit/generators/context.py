"""
Custom generators for rounding contexts.
"""

import fpy2 as fp
from hypothesis import strategies as st

from .round import overflow_modes, rounding_modes

###########################################################
# Floating-point contexts
#
# TODO: MPBFloatContext

@st.composite
def mp_float_contexts(
    draw,
    max_p: int | None = None,
    rm: fp.RM | None = None,
    max_randbits: int | None = 0
) -> fp.MPFloatContext:
    """
    Returns a strategy for generating a `fp.MPFloatContext`.

    Args:
        max_p: Maximum precision for the context.
        rm: Rounding mode for the context. If `None`, a random rounding mode is chosen.
        max_randbits: Maximum number of random bits for the context. If `0`, rounding is
            deterministic. If `None`, no limit is set.
    """
    p = draw(st.integers(1, max_p))
    if rm is None:
        rm = draw(rounding_modes())
    num_randbits = None if max_randbits is None else draw(st.integers(0, max_randbits))
    return fp.MPFloatContext(p, rm=rm, num_randbits=num_randbits)

@st.composite
def mps_float_contexts(
    draw,
    max_p: int | None = None,
    min_emin: int | None = None,
    max_emin: int | None = None,
    rm: fp.RM | None = None,
    max_randbits: int | None = 0,
):
    """
    Returns a strategy for generating a `fp.MPSFloatContext`.

    Args:
        max_p: Maximum precision for the context.
        min_emin: Smallest minimum exponent for the context.
        max_emin: Largest minimum exponent for the context.
        rm: Rounding mode for the context. If `None`, a random rounding mode is chosen.
        max_randbits: Maximum number of random bits for the context. If `0`, rounding is
            deterministic. If `None`, no limit is set.
    """
    p = draw(st.integers(1, max_p))
    emin = draw(st.integers(min_emin, max_emin))
    if rm is None:
        rm = draw(rounding_modes())
    num_randbits = None if max_randbits is None else draw(st.integers(0, max_randbits))
    return fp.MPSFloatContext(p, emin, rm=rm, num_randbits=num_randbits)

@st.composite
def ieee_contexts(
    draw,
    max_es: int | None = None,
    max_nbits: int | None = None,
    rm: fp.RM | None = None,
    max_randbits: int | None = 0
):
    """
    Returns a strategy for generating a `fp.IEEEContext`.

    Args:
        max_es: Maximum exponent width for the context (must be >= 2).
        max_nbits: Maximum number of bits for the context (must be >= 2 + `max_es`).
        rm: Rounding mode for the context. If `None`, a random rounding mode is chosen.
        max_randbits: Maximum number of random bits for the context. If `0`, rounding is
            deterministic. If `None`, no limit is set.
    """
    es = draw(st.integers(2, max_es))
    nbits = draw(st.integers(es + 2, max_nbits))
    if rm is None:
        rm = draw(rounding_modes())
    num_randbits = None if max_randbits is None else draw(st.integers(0, max_randbits))
    return fp.IEEEContext(es, nbits, rm=rm, num_randbits=num_randbits)

@st.composite
def efloat_contexts( 
    draw,
    max_es: int | None = None,
    max_nbits: int | None = None,
    enable_inf: bool | None = None,
    nan_kind: fp.EFloatNanKind | None = None,
    min_eoffset: int | None = None,
    max_eoffset: int | None = None,
    rm: fp.RM | None = None,
    ov: fp.OV | None = None,
    max_randbits: int | None = 0
):
    """
    Returns a strategy for generating an `fp.EFloatContext`.

    Args:
        max_es: Maximum exponent width for the context (must be >= 2).
        max_nbits: Maximum number of bits for the context (must be >= 2 + `max_es`).
        enable_inf: Whether to enable infinities in the context. If `None`, a random choice is made.
        nan_kind: Kind of NaN representation for the context. If `None`, a random choice is made.
        min_eoffset: Smallest exponent offset for the context.
        max_eoffset: Largest exponent offset for the context.
        rm: Rounding mode for the context. If `None`, a random rounding mode is chosen.
        overflow: Overflow mode for the context. If `None`, a random mode is chosen.
        max_randbits: Maximum number of random bits for the context. If `0`, rounding is
            deterministic. If `None`, no limit is set.
    """
    if nan_kind is None:
        nan_kind = draw(st.sampled_from(list(fp.EFloatNanKind)))

    match nan_kind:
        case fp.EFloatNanKind.IEEE_754:
            es = draw(st.integers(1, max_es))
            nbits = draw(st.integers(es + 1, max_nbits))
            p = nbits - es
            if p == 1:
                enable_inf = False

        case fp.EFloatNanKind.MAX_VAL:
            es = draw(st.integers(0, max_es))
            if es == 0:
                nbits = draw(st.integers(es + 2, max_nbits))
                p = nbits - es
                if p == 2:
                    enable_inf = False
            elif es == 1:
                nbits = draw(st.integers(es + 1, max_nbits))
                p = nbits - es
                if p == 1:
                    enable_inf = False
            else:
                nbits = draw(st.integers(es + 1, max_nbits))

        case fp.EFloatNanKind.NEG_ZERO | fp.EFloatNanKind.NONE:
            es = draw(st.integers(0, max_es))
            if es == 0:
                nbits = draw(st.integers(es + 1, max_nbits))
                p = nbits - es
                if p == 1:
                    enable_inf = False
            else:
                nbits = draw(st.integers(es + 1, max_nbits))

        case _:
            raise ValueError(f'Unknown EFloatNanKind: {nan_kind}')

    if enable_inf is None:
        enable_inf = draw(st.booleans())
    eoffset = draw(st.integers(min_eoffset, max_eoffset))
    if rm is None:
        rm = draw(rounding_modes())
    if ov is None:
        ov = draw(overflow_modes().filter(lambda x: x != fp.OV.WRAP))
    num_randbits = None if max_randbits is None else draw(st.integers(0, max_randbits))
    return fp.EFloatContext(es, nbits, enable_inf, nan_kind, eoffset, rm, ov, num_randbits)

@st.composite
def exp_contexts(
    draw,
    max_nbits: int | None = None,
    min_eoffset: int | None = None,
    max_eoffset: int | None = None,
    rm: fp.RM | None = None,
    ov: fp.OV | None = None,
):
    """
    Returns a strategy for generating an `fp.ExpContext`.

    Args:
        max_nbits: Maximum number of bits for the context (must be >= 1).
        min_eoffset: Smallest exponent offset for the context.
        max_eoffset: Largest exponent offset for the context.
        rm: Rounding mode for the context. If `None`, a random rounding mode is chosen.
        overflow: Overflow mode for the context. If `None`, a random mode is chosen.
        max_randbits: Maximum number of random bits for the context. If `0`, rounding is
            deterministic. If `None`, no limit is set.
    """
    nbits = draw(st.integers(1, max_nbits))
    eoffset = draw(st.integers(min_eoffset, max_eoffset))
    if rm is None:
        rm = draw(rounding_modes())
    if ov is None:
        ov = draw(overflow_modes().filter(lambda x: x != fp.OV.WRAP))

    return fp.ExpContext(nbits, eoffset, rm, ov)

###########################################################
# Fixed-point contexts

@st.composite
def mp_fixed_contexts(
    draw,
    min_n: int | None = None,
    max_n: int | None = None,
    rm: fp.RM | None = None,
    max_randbits: int | None = 0,
    enable_inf: bool = False,
    enable_nan: bool = False
) -> fp.MPFixedContext:
    """
    Returns a strategy for generating a `fp.MPFixedContext`.

    Args:
        min_n: Minimum position for the most significant unrepresentable digit.
        max_n: Maximum position for the most significant unrepresentable digit.
        rm: Rounding mode for the context. If `None`, a random rounding mode is chosen.
        max_randbits: Maximum number of random bits for the context. If `0`, rounding is
            deterministic. If `None`, no limit is set.
    """
    n = draw(st.integers(min_n, max_n))
    if rm is None:
        rm = draw(rounding_modes())
    num_randbits = None if max_randbits is None else draw(st.integers(0, max_randbits))
    return fp.MPFixedContext(n, rm, num_randbits, enable_inf=enable_inf, enable_nan=enable_nan)

@st.composite
def fixed_contexts(
    draw,
    signed: bool | None = None,
    min_scale: int | None = None,
    max_scale: int | None = None,
    max_nbits: int | None = None,
    rm: fp.RM | None = None,
    ov: fp.OV | None = None,
    max_randbits: int | None = 0,
):
    """
    Returns a strategy for generating a `fp.FixedContext`.

    Args:
        signed: Whether the context is signed. If `None`, a random choice is made.
        min_scale: Minimum scale for the context.
        max_scale: Maximum scale for the context.
        max_nbits: Maximum number of bits for the context (must be >= 2 if signed and >= 1 otherwise).
        rm: Rounding mode for the context. If `None`, a random rounding mode is chosen.
        overflow: Overflow mode for the context. If `None`, a random mode is chosen.
        max_randbits: Maximum number of random bits for the context. If `0`, rounding is
            deterministic. If `None`, no limit is set.
    """
    if signed is None:
        signed = draw(st.booleans())
    scale = draw(st.integers(min_scale, max_scale))
    if signed:
        nbits = draw(st.integers(2, max_nbits))
    else:
        nbits = draw(st.integers(1, max_nbits))
    if rm is None:
        rm = draw(rounding_modes())
    if ov is None:
        ov = draw(overflow_modes())
    num_randbits = None if max_randbits is None else draw(st.integers(0, max_randbits))
    return fp.FixedContext(signed, scale, nbits, rm, ov, num_randbits=num_randbits)

@st.composite
def sm_fixed_contexts(
    draw,
    min_scale: int | None = None,
    max_scale: int | None = None,
    max_nbits: int | None = None,
    rm: fp.RM | None = None,
    ov: fp.OV | None = None,
    max_randbits: int | None = 0,
):
    """
    Returns a strategy for generating a `fp.SMFixedContext`.

    Args:
        min_scale: Minimum scale for the context.
        max_scale: Maximum scale for the context.
        max_nbits: Maximum number of bits for the context (must be >= 2).
        rm: Rounding mode for the context. If `None`, a random rounding mode is chosen.
        overflow: Overflow mode for the context. If `None`, a random mode is chosen.
        max_randbits: Maximum number of random bits for the context. If `0`, rounding is
            deterministic. If `None`, no limit is set.
    """
    scale = draw(st.integers(min_scale, max_scale))
    nbits = draw(st.integers(2, max_nbits))
    if rm is None:
        rm = draw(rounding_modes())
    if ov is None:
        ov = draw(overflow_modes())
    num_randbits = None if max_randbits is None else draw(st.integers(0, max_randbits))
    return fp.SMFixedContext(scale, nbits, rm, ov, num_randbits=num_randbits)

###########################################################
# Common contexts

_common_contexts: list[fp.Context] | None = None

@st.composite
def common_contexts(draw):
    global _common_contexts
    if _common_contexts is None:
        _common_contexts = [
            ctx for ctx in vars(fp).values()
            if isinstance(ctx, fp.Context)
        ]
    return draw(st.sampled_from(_common_contexts))

