"""
Double rounding tests for ``A(p, exp, bound)``.

For each rule, F1 = A(p1, exp1, b1) is drawn randomly, then F2 is constructed
as the *minimal* abstract format satisfying the rule's containment
precondition (using ``_extend`` and ``_bump_at`` composed per the rule), then
padded by random non-negative amounts (``pad_p``, ``pad_exp``, ``pad_steps``)
to a representative super-format. The precondition therefore holds by
construction rather than by ``assume(...)`` filtering.
"""

from typing import Callable, TypeAlias

import fpy2 as fp

from hypothesis import given, strategies as st

from fpy2.analysis.format_infer import AbstractFormat
from fpy2.number.context.mp_fixed import MPFixedFormat
from fpy2.number.context.mps_float import MPSFloatFormat

from ...generators import real_floats


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# (p, exp, b) triple representing an abstract format A(p, exp, b).
# p is `float('inf')` for the fixed-point case, otherwise a positive int.
Prec: TypeAlias = int | float
Format: TypeAlias = tuple[Prec, int, fp.RealFloat]
MinimalFn: TypeAlias = Callable[[Format], Format]


# ---------------------------------------------------------------------------
# (p, exp, b) <-> context / grid helpers
# ---------------------------------------------------------------------------

def _cvt_to_context(p: Prec, exp: int, b: fp.RealFloat) -> fp.Context:
    if p == float('inf'):
        return fp.MPBFixedContext(
            nmin=exp - 1, maxval=b,
            enable_inf=True, overflow=fp.OverflowMode.OVERFLOW,
        )
    else:
        emin = exp + p - 1
        return fp.MPBFloatContext(pmax=p, emin=emin, maxval=b)


def _round_up_to_grid(p: Prec, exp: int, b: fp.RealFloat) -> fp.RealFloat:
    """Round `b` away from zero onto the (p, exp) grid."""
    if p == float('inf'):
        return b.round(min_n=exp - 1, rm=fp.RM.RAZ)
    else:
        return b.round(p, exp - 1, rm=fp.RM.RAZ)


def _round_down_to_grid(p: Prec, exp: int, b: fp.RealFloat) -> fp.RealFloat:
    """Round `b` toward zero onto the (p, exp) grid."""
    if p == float('inf'):
        return b.round(min_n=exp - 1, rm=fp.RM.RTZ)
    else:
        return b.round(p, exp - 1, rm=fp.RM.RTZ)


def _next_pe(p: Prec, exp: int, b: fp.RealFloat) -> fp.RealFloat:
    """next_{p, exp}(b): one ordinal step above b in the A(p, exp, ·) grid.

    b must already be representable at (p, exp); otherwise the result is the
    successor of whatever b snaps to, which is not what the rules require.
    """
    if p == float('inf'):
        fmt = MPFixedFormat(exp - 1)
    else:
        emin = exp + p - 1
        fmt = MPSFloatFormat(p, emin)
    o = fmt.to_ordinal(fp.Float.from_real(b))
    return fmt.from_ordinal(o + 1).as_real()


# ---------------------------------------------------------------------------
# Abstract operations on (p, exp, b) triples
# ---------------------------------------------------------------------------

def _extend(F: Format, k: int) -> Format:
    """Move to a finer grid: (p + k, exp - k, b). Bound unchanged."""
    p, exp, b = F
    return p + k, exp - k, b


def _bump_at(F: Format, p_ref: Prec, exp_ref: int) -> Format:
    """Replace F's bound with next_{p_ref, exp_ref}(F.b). F's (p, exp) is unchanged."""
    p, exp, b = F
    new_b = _next_pe(p_ref, exp_ref, b)
    assert new_b > b, f"bump did not advance bound: {b} -> {new_b}"
    return p, exp, new_b


# ---------------------------------------------------------------------------
# Per-rule minimal F2
# ---------------------------------------------------------------------------

def _min_rtz_rtz(F1: Format) -> Format:
    p1, exp1, _ = F1
    return _bump_at(F1, p1, exp1)

def _min_raz_raz(F1: Format) -> Format:
    return F1

def _min_rto_rto(F1: Format) -> Format:
    p1, exp1, _ = F1
    return _bump_at(F1, p1, exp1)

def _min_rto_rtz(F1: Format) -> Format:
    p1, exp1, _ = F1
    return _bump_at(_extend(F1, 1), p1, exp1)

def _min_rto_raz(F1: Format) -> Format:
    p1, exp1, _ = F1
    return _bump_at(_extend(F1, 1), p1, exp1)

def _min_rto_rne(F1: Format) -> Format:
    p1, exp1, _ = F1
    return _bump_at(_extend(F1, 2), p1 + 1, exp1 - 1)


# ---------------------------------------------------------------------------
# F1 construction and padding to F2
# ---------------------------------------------------------------------------

def _make_F1(p1: Prec, exp1: int, k1: int) -> Format:
    b1_raw = fp.RealFloat(exp=exp1, c=k1)
    b1 = _round_down_to_grid(p1, exp1, b1_raw)
    return p1, exp1, b1


def _pad(F_min: Format, pad_p: int, pad_exp: int, pad_steps: int) -> Format:
    """Super-format of F_min, independently widening p, exp, and bound."""
    p_min, exp_min, b_min = F_min
    p: Prec = p_min if p_min == float('inf') else p_min + pad_p
    exp = exp_min - pad_exp
    if pad_steps == 0:
        b = b_min
    else:
        b_raw = b_min + fp.RealFloat(exp=exp, c=pad_steps)
        b = _round_up_to_grid(p, exp, b_raw)
    return p, exp, b


def _run(
    F1: Format,
    minimal_fn: MinimalFn,
    rm1: fp.RM,
    rm2: fp.RM,
    pad_p: int,
    pad_exp: int,
    pad_steps: int,
    x: fp.RealFloat,
) -> None:
    F2 = _pad(minimal_fn(F1), pad_p, pad_exp, pad_steps)

    ctx1 = _cvt_to_context(*F1).with_params(rm=rm1)
    ctx2 = _cvt_to_context(*F2).with_params(rm=rm2)

    A1 = AbstractFormat.from_format(ctx1.format())
    A2 = AbstractFormat.from_format(ctx2.format())
    assert A1 <= A2, (
        f"containment precondition violated by construction: "
        f"F1={F1}, F2={F2}, A1={A1}, A2={A2}"
    )

    y1 = ctx1.round(x)
    y2 = ctx2.round(x)
    y3 = ctx1.round(y2)
    assert y1.overflow or y1 == y3, (
        f"double-round mismatch: y1={float(y1)}, y2={float(y2)}, "
        f"y3={float(y3)}, x={float(x)}"
    )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_p1 = st.one_of(st.integers(1, 5), st.just(float('inf')))
_p1_ge2 = st.one_of(st.integers(2, 5), st.just(float('inf')))  # RTO-RTO needs p2 >= 2
_exp1 = st.integers(-8, 0)
_k1 = st.integers(1, 32)
_padN = st.integers(0, 5)
_pad_steps = st.integers(0, 8)
_x = real_floats(prec_max=16, exp_min=-16, exp_max=16)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDoubleRound:
    """Double-rounding identities ctx1[rm1] ∘ ctx2[rm2] = ctx1[rm1].

    F2 is built as a super-format of the rule-specific minimal F2, so the
    A(F1) ⊆ A(F2) precondition holds by construction.
    """

    @given(_p1, _exp1, _k1, _padN, _padN, _pad_steps, _x)
    def test_rtz_rtz(
        self, p1: Prec, exp1: int, k1: int,
        pad_p: int, pad_exp: int, pad_steps: int, x: fp.RealFloat,
    ) -> None:
        """rnd-RTZ-RTZ"""
        F1 = _make_F1(p1, exp1, k1)
        _run(F1, _min_rtz_rtz, fp.RM.RTZ, fp.RM.RTZ, pad_p, pad_exp, pad_steps, x)

    @given(_p1, _exp1, _k1, _padN, _padN, _pad_steps, _x)
    def test_raz_raz(
        self, p1: Prec, exp1: int, k1: int,
        pad_p: int, pad_exp: int, pad_steps: int, x: fp.RealFloat,
    ) -> None:
        """rnd-RAZ-RAZ"""
        F1 = _make_F1(p1, exp1, k1)
        _run(F1, _min_raz_raz, fp.RM.RAZ, fp.RM.RAZ, pad_p, pad_exp, pad_steps, x)

    @given(_p1_ge2, _exp1, _k1, _padN, _padN, _pad_steps, _x)
    def test_rto_rto(
        self, p1: Prec, exp1: int, k1: int,
        pad_p: int, pad_exp: int, pad_steps: int, x: fp.RealFloat,
    ) -> None:
        """rnd-RTO-RTO (requires p2 >= 2)"""
        F1 = _make_F1(p1, exp1, k1)
        _run(F1, _min_rto_rto, fp.RM.RTO, fp.RM.RTO, pad_p, pad_exp, pad_steps, x)

    @given(_p1, _exp1, _k1, _padN, _padN, _pad_steps, _x)
    def test_rto_rtz(
        self, p1: Prec, exp1: int, k1: int,
        pad_p: int, pad_exp: int, pad_steps: int, x: fp.RealFloat,
    ) -> None:
        """rnd-RTO-RTZ"""
        F1 = _make_F1(p1, exp1, k1)
        _run(F1, _min_rto_rtz, fp.RM.RTZ, fp.RM.RTO, pad_p, pad_exp, pad_steps, x)

    @given(_p1, _exp1, _k1, _padN, _padN, _pad_steps, _x)
    def test_rto_raz(
        self, p1: Prec, exp1: int, k1: int,
        pad_p: int, pad_exp: int, pad_steps: int, x: fp.RealFloat,
    ) -> None:
        """rnd-RTO-RAZ"""
        F1 = _make_F1(p1, exp1, k1)
        _run(F1, _min_rto_raz, fp.RM.RAZ, fp.RM.RTO, pad_p, pad_exp, pad_steps, x)

    @given(_p1, _exp1, _k1, _padN, _padN, _pad_steps, _x)
    def test_rto_rne(
        self, p1: Prec, exp1: int, k1: int,
        pad_p: int, pad_exp: int, pad_steps: int, x: fp.RealFloat,
    ) -> None:
        """rnd-RTO-RNE"""
        F1 = _make_F1(p1, exp1, k1)
        _run(F1, _min_rto_rne, fp.RM.RNE, fp.RM.RTO, pad_p, pad_exp, pad_steps, x)
