"""
Double-rounding soundness: when one rounding may be split into two.

Two families of rule, both mechanised in `mpfx-lean
<https://github.com/bksaiki/mpfx-lean>`_, which is the source of truth:

- :func:`double_round_ok` -- Figure 8 of *When Double Rounding is Correct*
  (``Mpfx/DoubleRounding.lean``), which holds for *every* real and so needs
  nothing but the two formats and modes.  It refuses nearest over nearest at
  every width.
- :func:`double_round_op_ok` -- Roux 2014, *Innocuous Double Rounding of Basic
  Arithmetic Operations* (``Mpfx/DoubleRoundingAdd.lean`` and siblings), which
  holds only for the *results of one operation* but admits nearest over nearest,
  and so is what lets a hardware format serve as the intermediate.

:func:`derive_intermediate` produces a pair the first accepts.

Sibling of :func:`fpy2.analysis.format_infer.round_is_identity`, which decides the
*single*-rounding question -- whether a rounding may be dropped rather than split.
"""

from enum import Enum

from ...number import (
    Context,
    MPFixedContext,
    MPFloatContext,
    MPSFloatContext,
    RoundingMode,
)
from .format import AbstractableFormat, AbstractFormat

__all__ = [
    'DoubleRoundOp', 'derive_intermediate', 'double_round_ok',
    'double_round_op_ok',
]


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


class DoubleRoundOp(Enum):
    """An operation with a double-rounding rule of its own.

    Multiplication has one too (Roux Theorem 10), but its proof is ``rndExact``
    -- the exact product is representable in the intermediate, so the inner
    rounding is the identity.  A caller holding the operand formats checks that
    directly, and more generally than the closed form ``p2 >= 2*p1`` can, so it
    is not listed here.
    """

    ADD = 'add'
    """Addition and subtraction: ``rndAdd`` / ``rndDiff``, Roux Theorem 20."""

    DIV = 'div'
    """``rndDiv_FLX`` / ``rndDiv_FLT``, Roux Theorem 29 -- tight (Remark 30)."""

    SQRT = 'sqrt'
    """``rndSqrt_FLX`` / ``rndSqrt_FLT``, Roux Theorem 25."""


def _flx(e: float) -> bool:
    """Whether a format has no minimum quantum -- Flocq's ``FLX``."""
    return e == float('-inf')


def double_round_op_ok(
    op: DoubleRoundOp,
    f1: AbstractFormat,
    rm1: RoundingMode,
    f2: AbstractFormat,
    rm2: RoundingMode,
) -> bool:
    """
    Decide the same question as :func:`double_round_ok`, but only for the
    *results of* *op* -- which buys a much narrower intermediate.

    **The caller must have checked that every operand of the operation is
    representable in** ``f1``. Every one of these theorems assumes it, and it is
    load-bearing rather than presentational: an operand on a finer grid than the
    target lets the exact result land within half an intermediate ulp of a target
    midpoint, which is exactly what the proofs rule out. FPy's signature is
    ``op: Fx -> Fy -> F1``, so this does not come for free.

    Both roundings must be to nearest; the tie-breaks may differ.  Any special
    the target represents the intermediate must too, or the split loses it --
    the premises are about finite values, but a program is not.  The bound
    plays no part -- the theorems are stated on unbounded formats, and a bounded
    rounding is the unbounded one followed by a bound check that reads only its
    result, so the conclusion survives it.  A bounded *intermediate* is another
    matter, since its check sits between the two roundings; that stays the
    caller's to gate.

    ``ADD`` takes ``exp2 <= exp1`` over ``WithBot``, so it spans both families.
    ``DIV`` and ``SQRT`` are proved separately for ``FLX`` and ``FLT`` and have
    no mixed-family statement, so an unbounded exponent on one side only is
    refused rather than assumed.
    """
    if rm1 not in _NEAREST or rm2 not in _NEAREST:
        return False
    if not f1.specials_contained_in(f2):
        # the theorems quantify over finite values, but a program carries
        # specials through the split, and they survive it only if the
        # intermediate represents them
        return False

    p1, p2, e1, e2 = f1.prec, f2.prec, f1.exp, f2.exp
    if not isinstance(p1, int) or not isinstance(p2, int):
        return False                # the theorems take a finite precision
    if p1 == 1 and _flx(e1) and rm1 is RoundingMode.RNE:
        return False                # `IsUndefined`: the degenerate format

    match op:
        case DoubleRoundOp.ADD:
            return p2 >= 2 * p1 + 1 and e2 <= e1
        case DoubleRoundOp.DIV:
            if _flx(e1) or _flx(e2):
                return p2 >= 2 * p1 and _flx(e1) and _flx(e2)
            return p2 >= 2 * p1 and e2 <= e1 - p1 - 2
        case DoubleRoundOp.SQRT:
            if _flx(e1) or _flx(e2):
                return p2 >= 2 * p1 + 2 and _flx(e1) and _flx(e2)
            return (
                p2 >= 2 * p1 + 2 and e1 <= 0
                and (e2 <= e1 - p1 - 2 or 2 * e2 <= e1 - 4 * p1 - 2)
            )


_OP_WIDTH: dict[DoubleRoundOp, tuple[int, int, int]] = {
    # op -> (precision multiple, precision slack, quanta dropped below `exp1`)
    DoubleRoundOp.ADD: (2, 1, 0),
    DoubleRoundOp.DIV: (2, 0, 2),
    DoubleRoundOp.SQRT: (2, 2, 2),
}
"""The width each rule asks for, as read off :func:`double_round_op_ok`.  The
quanta dropped are ``p1 + this`` where nonzero -- the underflow margin `div` and
`sqrt` need because their results are irrational."""


def _derive_for_op(
    target: Context, op: DoubleRoundOp, rm1: RoundingMode,
) -> Context:
    """:func:`derive_intermediate` for an operation-specific rule."""
    if rm1 not in _NEAREST:
        raise ValueError(
            f'the {op.value} rule is proved for round-to-nearest, and '
            f'`{target}` rounds {rm1.name}'
        )
    fmt = target.format()
    if not isinstance(fmt, AbstractableFormat):
        raise TypeError(f'`{target}` has no abstract format')
    f1 = AbstractFormat.from_format(fmt)
    if f1.prec == float('inf'):
        raise ValueError(
            f'the {op.value} rule takes a finite precision, and `{target}` '
            'names none'
        )
    p1, e1 = int(f1.prec), f1.exp

    mult, slack, drop = _OP_WIDTH[op]
    p2 = mult * p1 + slack
    if not isinstance(e1, int):
        # an FLX target takes an FLX intermediate: `div` and `sqrt` are proved
        # per family, so a minimum quantum on one side only is not a theorem
        return MPFloatContext(p2, rm1)
    if op is DoubleRoundOp.SQRT and e1 > 0:
        raise ValueError(
            f'the sqrt rule needs `exp1 <= 0`, and `{target}` represents '
            'nothing below one'
        )
    # `MPSFloatContext` takes the least *normalized* exponent, which sits
    # `p2 - 1` above the least quantum
    e2 = e1 - (p1 + drop if drop else 0)
    return MPSFloatContext(p2, e2 + p2 - 1, rm1)


def derive_intermediate(
    target: Context, op: DoubleRoundOp | None = None,
) -> Context:
    """
    An intermediate that :func:`double_round_ok` accepts for *target*, as a
    context ready to install.

    This is how a caller obtains the `ctx` a split needs without computing
    `p+k` / `exp-k` by hand.  Round-to-odd, because that is §5.3's
    modular-library recipe: compute wide under RTO, then re-round to the target
    under whatever mode it wants.

    With *op*, the rule is that operation's own (:func:`double_round_op_ok`) and
    the intermediate is a *nearest* one instead -- the target's own mode, since
    the tie-breaks are independent.  That is what a hardware format can be, so
    this is the derivation to ask for when the intermediate has to be one: FP64
    satisfies every rule for an FP32 target.  ``MUL`` is not offered, because its
    rule is exactness -- any format holding the exact product works under any
    mode, and the transform checks that against the real operand formats.

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
        only against themselves).  With *op*: if *target* does not round to
        nearest, has no finite precision, or -- for ``SQRT`` -- has no
        representable value below one.
    TypeError
        If *target*'s format has no abstract form.
    """
    rm1 = target.rounding_mode()
    if rm1 is None:
        raise ValueError(f'`{target}` does not round, so it has no intermediate')
    if target.is_stochastic():
        # not a function of its input, so no composition reproduces it
        raise ValueError(f'`{target}` rounds stochastically')
    if op is not None:
        return _derive_for_op(target, op, rm1)
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
