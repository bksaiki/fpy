"""
Double-rounding soundness: when one rounding may be split into two.

Figure 8 of *When Double Rounding is Correct*, as proved in `Mpfx/DoubleRounding.lean
<https://github.com/bksaiki/mpfx-lean/blob/main/Mpfx/DoubleRounding.lean>`_, which
is the source of truth for these rules.  :func:`double_round_ok` decides a
candidate pair and :func:`derive_intermediate` produces one.

Sibling of :func:`fpy2.analysis.format_infer.round_is_identity`, which decides the
*single*-rounding question -- whether a rounding may be dropped rather than split.
"""

from ...number import (
    Context,
    MPBFixedContext,
    MPBFloatContext,
    MPFixedContext,
    OverflowMode,
    RoundingMode,
)
from ...number.context.mp_fixed import MPFixedFormat
from ...number.context.mpb_fixed import MPBFixedFormat
from ...number.context.mpb_float import MPBFloatFormat
from .format import AbstractableFormat, AbstractFormat

__all__ = ['derive_intermediate', 'double_round_ok']


_NEAREST = frozenset({RoundingMode.RNE, RoundingMode.RNA})
"""Round-to-nearest modes; Figure 8 admits them only over an RTO intermediate."""

_DIRECTED = frozenset({RoundingMode.RTZ, RoundingMode.RAZ})
"""Directed modes with both a same-mode rule and an RTO rule."""

_SAME_MODE = _DIRECTED | frozenset({RoundingMode.RTP, RoundingMode.RTN})
"""Modes sound against themselves on plain containment."""


def _extend(f: AbstractFormat, k: int) -> AbstractFormat:
    """The paper's ``F.extend k``: precision ``+k``, exponent ``-k``."""
    return f.with_prec_offset(k).with_exp_offset(-k)


def double_round_ok(
    f1: AbstractFormat,
    rm1: RoundingMode,
    f2: AbstractFormat,
    rm2: RoundingMode,
) -> bool:
    """
    Decide whether ``rnd_{f1,rm1} . rnd_{f2,rm2} == rnd_{f1,rm1}`` — i.e.
    rounding through the intermediate ``(f2, rm2)`` and then to the target
    ``(f1, rm1)`` gives what rounding straight to the target would.

    This is Figure 8 of *When Double Rounding is Correct*, as proved in
    `Mpfx/DoubleRounding.lean
    <https://github.com/bksaiki/mpfx-lean/blob/main/Mpfx/DoubleRounding.lean>`_.
    Eight rules over nine mode pairs:

    - ``rm1 == rm2`` in RTZ / RAZ / RTP / RTN — plain containment
      (``rndRTZ_RTZ``, ``rndRAZ_RAZ``, ``rndRTP_RTP``, ``rndRTN_RTN``).  The
      sign branching RTP and RTN need is internal to those proofs, so no
      value-range analysis is required here.
    - RTO over RTO — plain containment and ``p2 >= 2`` (``rndRTO_RTO``).
    - RTZ or RAZ over RTO — ``extend(f1, 1)`` carrying ``f1``'s next bound
      (``rndRTO_RTZ``, ``rndRTO_RAZ``).
    - RNE or RNA over RTO — ``extend(f1, 2)`` carrying the *once-extended*
      format's next bound (``rndRTO_RN``, which is parametric in the
      tie-break).  Which grid the bound comes from is the only difference
      between this premise and the one above.

    Everything else is unsound and returns ``False``, including RNE-RNE — the
    pairing every ``fp.FP*`` context falls into by default, and the one a
    well-meaning patch is most likely to try to admit.

    The theorems are about finite values, but FPy tracks special values too, and
    a special the target represents and the intermediate does not would be lost
    on the way through.  No separate check is needed: containment already makes
    ``specials_contained_in`` its first condition.

    ``p2 >= 2`` is *derived* in the proofs for the RTO-to-directed and
    RTO-to-nearest rules, so it is only checked where it is a stated
    hypothesis.

    Stochastic rounding is invisible here — this predicate sees formats and
    modes, not contexts — so a caller holding contexts must decline a
    stochastic one itself.
    """
    if rm2 is RoundingMode.RTO:
        if rm1 is RoundingMode.RTO:
            return f2.prec >= 2 and f1.contained_in(f2)
        if rm1 in _DIRECTED:
            return _extend(f1.next_bound(), 1).contained_in(f2)
        if rm1 in _NEAREST:
            return _extend(_extend(f1, 1).next_bound(), 1).contained_in(f2)
        return False

    return rm1 is rm2 and rm1 in _SAME_MODE and f1.contained_in(f2)


def derive_intermediate(target: Context) -> Context:
    """
    The tightest RTO intermediate that :func:`double_round_ok` accepts for
    *target*, as a context ready to install.

    This is how a caller obtains the ``ctx`` a split needs without computing
    `p+k` / `exp-k` by hand.  RTO is the intermediate because that is §5.3's
    modular-library recipe: compute wide under round-to-odd, then re-round to
    the target under whatever mode it wants.  A caller who wants a same-mode
    intermediate instead — plain containment, so it succeeds more often —
    passes their own.

    Raises
    ------
    ValueError
        If *target* has no rounding mode (:data:`REAL`), or its mode has no
        rule over an RTO intermediate (RTP and RTN, which are proved only
        against themselves).
    """
    rm1 = target.rounding_mode()
    if rm1 is None:
        raise ValueError(f'`{target}` does not round, so it has no intermediate')

    fmt1 = target.format()
    if not isinstance(fmt1, AbstractableFormat):
        # a format kind with no abstract form is a domain condition, the same
        # one `AbstractFormat.from_format` reports
        raise ValueError(f'`{target}` has no abstract-format form')  # noqa: TRY004

    f1 = AbstractFormat.from_format(fmt1)
    if rm1 in _NEAREST:
        derived = _extend(_extend(f1, 1).next_bound(), 1)
    elif rm1 in _DIRECTED:
        derived = _extend(f1.next_bound(), 1)
    elif rm1 is RoundingMode.RTO:
        # `extend(f1, 1)` rather than `f1` itself, so `p2 >= 2` holds even for
        # a one-bit target
        derived = _extend(f1, 1)
    else:
        raise ValueError(
            f'no double-rounding rule for a {rm1.name} target over a '
            'round-to-odd intermediate'
        )
    return _context_of(derived, RoundingMode.RTO)


def _context_of(f: AbstractFormat, rm: RoundingMode) -> Context:
    """A context representing *f*, rounding under *rm*.

    Saturating, always.  The premises are containment checks on `A`, which says
    what is representable and nothing about what happens above it -- so an
    intermediate that overflowed to infinity would send a value the target
    clamps to its maxval to `inf` instead, and the re-rounding could not pull it
    back.  Saturating is what makes the composition agree at the top of the
    range.
    """
    fmt = f.format()
    match fmt:
        case MPBFloatFormat():
            return MPBFloatContext(
                fmt.pmax, fmt.emin, fmt.pos_maxval, rm, OverflowMode.SATURATE,
                neg_maxval=fmt.neg_maxval,
                enable_nan=fmt.enable_nan, enable_inf=fmt.enable_inf,
            )
        case MPBFixedFormat():
            return MPBFixedContext(
                fmt.nmin, fmt.pos_maxval, rm, OverflowMode.SATURATE,
                neg_maxval=fmt.neg_maxval,
                enable_nan=fmt.enable_nan, enable_inf=fmt.enable_inf,
                enable_neg_zero=fmt.enable_neg_zero,
            )
        case MPFixedFormat():
            # unbounded, so there is nothing to saturate against
            return MPFixedContext(
                fmt.nmin, rm,
                enable_nan=fmt.enable_nan, enable_inf=fmt.enable_inf,
                enable_neg_zero=fmt.enable_neg_zero,
            )
        case _:
            raise ValueError(
                f'cannot build a context for {type(fmt).__name__}'
            )
