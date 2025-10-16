"""
Custom generators for rounding contexts.
"""

import fpy2 as fp
from hypothesis import strategies as st

from .round import rounding_modes

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
    max_p: int | None = None,
    rm: fp.RM | None = None,
    max_randbits: int | None = 0
):
    """
    Returns a strategy for generating a `fp.IEEEContext`.

    Args:
        max_es: Maximum exponent width for the context.
        max_p: Maximum precision for the context.
        rm: Rounding mode for the context. If `None`, a random rounding mode is chosen.
        max_randbits: Maximum number of random bits for the context. If `0`, rounding is
            deterministic. If `None`, no limit is set.
    """
    es = draw(st.integers(2, max_es))
    p = draw(st.integers(2, max_p))
    nbits = es + p
    if rm is None:
        rm = draw(rounding_modes())
    num_randbits = None if max_randbits is None else draw(st.integers(0, max_randbits))
    return fp.IEEEContext(es, nbits, rm=rm, num_randbits=num_randbits)

@st.composite
def efloat_contexts(
    draw,
    max_es: int | None = None,
    max_p: int | None = None,
    enable_inf: bool | None = None,
    nan_kind: fp.EFloatNanKind | None = None,
    min_eoffset: int | None = None,
    max_eoffset: int | None = None,
    rm: fp.RM | None = None,
    max_randbits: int | None = 0
):
    """
    Returns a strategy for generating an `fp.EFloatContext`.
    """
    if nan_kind is None:
        nan_kind = draw(st.sampled_from(list(fp.EFloatNanKind)))

    match nan_kind:
        case fp.EFloatNanKind.IEEE_754:
            es = draw(st.integers(1, max_es))
            p = draw(st.integers(2, max_p))
            if p == 1:
                enable_inf = False

        case fp.EFloatNanKind.MAX_VAL:
            es = draw(st.integers(0, max_es))
            if es == 0:
                p = draw(st.integers(2, max_p))
                if p < 2:
                    enable_inf = False
            elif es == 1:
                p = draw(st.integers(1, max_p))
                if p == 1:
                    enable_inf = False
            else:
                p = draw(st.integers(1, max_p))

        case fp.EFloatNanKind.NONE | fp.EFloatNanKind.NONE:
            es = draw(st.integers(0, max_es))
            if es == 0:
                p = draw(st.integers(1, max_p))
                if p == 1:
                    enable_inf = False
            else:
                p = draw(st.integers(1, max_p))

        case _:
            raise ValueError(f'Unknown EFloatNanKind: {nan_kind}')

    nbits = es + p
    if enable_inf is None:
        enable_inf = draw(st.booleans())
    eoffset = draw(st.integers(min_eoffset, max_eoffset))
    if rm is None:
        rm = draw(rounding_modes())
    num_randbits = None if max_randbits is None else draw(st.integers(0, max_randbits))
    return fp.EFloatContext(es, nbits, enable_inf, nan_kind, eoffset, rm=rm, num_randbits=num_randbits)

###########################################################
# Abstract contexts

@st.composite
def encodable_contexts(
    draw,
    max_p: int | None = None,
    max_es: int | None = None,
    enable_inf: bool | None = None,
    nan_kind: fp.EFloatNanKind | None = None,
    min_eoffset: int | None = None,
    max_eoffset: int | None = None,
    rm: fp.RM | None = None,
    max_randbits: int | None = 0
):
    """
    Returns a strategy for generating an `EncodableContext`.
    """
    strategies = [efloat_contexts(max_es, max_p, enable_inf, nan_kind, min_eoffset, max_eoffset, rm, max_randbits)]
    if (
        enable_inf
        and nan_kind == fp.EFloatNanKind.IEEE_754
        and (min_eoffset is None or min_eoffset <= 0)
        and (max_eoffset is None or max_eoffset >= 0)
        and (max_es is None or max_es >= 2)
        and (max_p is None or max_p >= 2)
    ):
        strategies.append(ieee_contexts(max_es, max_p, rm, max_randbits))
    return st.one_of(strategies)

@st.composite
def sized_contexts(
    draw,
    max_p: int | None = None,
    max_es: int | None = None,
    rm: fp.RM | None = None,
    max_randbits: int | None = 0
):
    """
    Returns a strategy for generating a `SizedContext`.
    """
    return st.one_of(
        encodable_contexts(max_es=max_es, max_p=max_p, rm=rm, max_randbits=max_randbits),
    )

@st.composite
def ordinal_contexts(
    draw,
    max_p: int | None = None,
    max_es: int | None = None,
    min_emin: int | None = None,
    max_emin: int | None = None,
    rm: fp.RM | None = None,
    max_randbits: int | None = 0
):
    """
    Returns a strategy for generating an `OrdinalContext`.
    """
    return (
        mps_float_contexts(max_p=max_p, min_emin=min_emin, max_emin=max_emin, rm=rm, max_randbits=max_randbits)
        | sized_contexts(max_es=max_es, max_p=max_p, rm=rm, max_randbits=max_randbits),
    )

@st.composite
def contexts(
    draw,
    max_p: int | None = None,
    max_es: int | None = None,
    min_emin: int | None = None,
    max_emin: int | None = None,
    rm: fp.RM | None = None,
    max_randbits: int | None = 0
):
    """
    Returns a strategy for generating a `Context`.
    """
    return (
        mp_float_contexts(max_p=max_p, rm=rm, max_randbits=max_randbits)
        | ordinal_contexts(max_p=max_p, max_es=max_es, min_emin=min_emin, max_emin=max_emin, rm=rm, max_randbits=max_randbits)
    )
