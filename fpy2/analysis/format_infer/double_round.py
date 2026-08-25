"""
Double-rounding soundness: when one rounding may be split into two.

Figure 8 of *When Double Rounding is Correct*, as proved in `Mpfx/DoubleRounding.lean
<https://github.com/bksaiki/mpfx-lean/blob/main/Mpfx/DoubleRounding.lean>`_, which
is the source of truth for these rules.  :func:`double_round_ok` decides a
candidate pair and :func:`derive_intermediate` produces one.

Sibling of :func:`fpy2.analysis.format_infer.round_is_identity`, which decides the
*single*-rounding question -- whether a rounding may be dropped rather than split.
"""

from ...number import Context, MPFixedContext, MPFloatContext, RoundingMode
from .format import AbstractFormat

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

    - ``rm1 == rm2`` in RTZ / RAZ / RTP / RTN — plain containment.  The sign
      branching RTP and RTN need is internal to those proofs, so no value-range
      analysis is required here.
    - RTO over RTO — plain containment and ``p2 >= 2``.
    - RTZ or RAZ over RTO — ``extend(f1, 1)`` carrying ``f1``'s next bound.
    - RNE or RNA over RTO — ``extend(f1, 2)`` carrying the *once-extended*
      format's next bound.  Which grid the bound comes from is the only
      difference between this premise and the one above.

    Everything else returns ``False``, RNE-RNE included.

    Special values need no separate check: containment already makes
    ``specials_contained_in`` its first condition.  ``p2 >= 2`` is *derived* in
    the proofs for the RTO rules, so it is checked only where it is stated.
    Stochastic rounding is invisible here — this sees formats and modes, not
    contexts — so a caller holding contexts declines it itself.
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
    An intermediate that :func:`double_round_ok` accepts for *target*, as a
    context ready to install.

    This is how a caller obtains the `ctx` a split needs without computing
    `p+k` / `exp-k` by hand.  Round-to-odd, because that is §5.3's
    modular-library recipe: compute wide under RTO, then re-round to the target
    under whatever mode it wants.

    **Unbounded**, which is what makes the composition agree at the ends of
    the range: it cannot overflow or underflow, so the only rounding that can is
    the target's, exactly as before the split.  A bounded intermediate has an
    overflow of its own that the premises do not constrain, and no single
    behaviour for it suits every target — see gap 4 of
    ``docs/todos/rounding-axes.md``.

    Raises
    ------
    ValueError
        If *target* has no rounding mode (:data:`REAL`), rounds stochastically,
        or its mode has no rule over an RTO intermediate (RTP and RTN, proved
        only against themselves).
    """
    rm1 = target.rounding_mode()
    if rm1 is None:
        raise ValueError(f'`{target}` does not round, so it has no intermediate')
    if target.is_stochastic():
        # not a function of its input, so no composition reproduces it
        raise ValueError(f'`{target}` rounds stochastically')
    if rm1 in _NEAREST:
        k = 2
    elif rm1 in _DIRECTED or rm1 is RoundingMode.RTO:
        k = 1
    else:
        raise ValueError(
            f'no double-rounding rule for a {rm1.name} target over a '
            'round-to-odd intermediate'
        )

    # `round_params` names the family: a precision means floating-point style, a
    # minimum digit means fixed-point.  `k` more of whichever it is covers every
    # premise, since an unbounded format contains any bound.
    max_p, min_n = target.round_params()
    if max_p is not None:
        return MPFloatContext(max_p + k, RoundingMode.RTO)
    if min_n is not None:
        return MPFixedContext(min_n - k, RoundingMode.RTO)
    raise ValueError(f'`{target}` names neither a precision nor a scale')
