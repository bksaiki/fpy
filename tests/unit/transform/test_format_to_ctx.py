"""
Testing `Format -> Context` recovery in `fpy2.transform.specialize`.

`_format_to_ctx` rebuilds a `Context` from a `Format` by dispatching on the
format type and calling that context class's `from_format`. The properties
that matter downstream are that the recovered context describes the *same*
format, that it is the right context class, and that the canonical contexts
in `fpy2.libraries.base` recover as themselves.
"""

import fpy2 as fp
import pytest

from fpy2.number.context.efloat import EFloatContext, EFloatFormat
from fpy2.number.context.exponential import ExpContext, ExpFormat
from fpy2.number.context.fixed import FixedContext, FixedFormat
from fpy2.number.context.ieee754 import IEEEContext, IEEEFormat
from fpy2.number.context.mp_fixed import MPFixedContext, MPFixedFormat
from fpy2.number.context.mp_float import MPFloatContext, MPFloatFormat
from fpy2.number.context.mpb_fixed import MPBFixedContext, MPBFixedFormat
from fpy2.number.context.real import REAL_FORMAT
from fpy2.number.context.sm_fixed import SMFixedContext, SMFixedFormat
from fpy2.transform.specialize import _format_to_ctx

# every canonical context, by name, from `fpy2.libraries.base`
_CANONICAL = [
    'FP16', 'FP32', 'FP64', 'FP128', 'FP256', 'BF16', 'TF32', 'INTEGER',
    'SINT8', 'SINT16', 'SINT32', 'SINT64',
    'UINT8', 'UINT16', 'UINT32', 'UINT64',
    'S1E5M2', 'S1E4M3',
    'MX_E5M2', 'MX_E4M3', 'MX_E3M2', 'MX_E2M3', 'MX_E2M1',
    'MX_E8M0', 'MX_INT8',
    'FP8P1', 'FP8P2', 'FP8P3', 'FP8P4', 'FP8P5', 'FP8P6', 'FP8P7',
]


def _canonical(name: str):
    from fpy2 import libraries
    return getattr(libraries.base, name)


class TestFormatRoundTrip():
    """The recovered context must describe the format it came from."""

    @pytest.mark.parametrize('name', _CANONICAL)
    def test_canonical_format_roundtrips(self, name: str):
        fmt = _canonical(name).format()
        ctx = _format_to_ctx(fmt)
        assert ctx is not None, f'{name} did not recover'
        assert ctx.format() == fmt

    @pytest.mark.parametrize('ctx', [
        fp.FixedContext(True, 3, 12),
        fp.FixedContext(False, -2, 5),
        fp.FixedContext(True, 0, 2),
        fp.SMFixedContext(0, 9),
        fp.SMFixedContext(-3, 16),
        fp.MPFloatContext(37),
        fp.MPFixedContext(-7),
        fp.IEEEContext(6, 23),
        fp.IEEEContext(4, 8),
    ])
    def test_non_canonical_format_roundtrips(self, ctx):
        # recovery is not limited to a fixed registry of known contexts
        rec = _format_to_ctx(ctx.format())
        assert rec is not None, f'{ctx} did not recover'
        assert rec.format() == ctx.format()


class TestContextClassDispatch():
    """Each format must map to the context class that describes it."""

    @pytest.mark.parametrize('ctx,fmt_cls,ctx_cls', [
        (fp.IEEEContext(8, 32), IEEEFormat, IEEEContext),
        (fp.FixedContext(True, 0, 8), FixedFormat, FixedContext),
        (fp.SMFixedContext(0, 8), SMFixedFormat, SMFixedContext),
        (fp.MPFixedContext(-1), MPFixedFormat, MPFixedContext),
        (fp.MPFloatContext(53), MPFloatFormat, MPFloatContext),
    ])
    def test_dispatch(self, ctx, fmt_cls, ctx_cls):
        fmt = ctx.format()
        assert isinstance(fmt, fmt_cls)
        assert type(_format_to_ctx(fmt)) is ctx_cls

    def test_ieee_before_efloat(self):
        # `IEEEFormat` is an `EFloatFormat`, so case order decides this
        fmt = fp.FP64.format()
        assert isinstance(fmt, IEEEFormat) and isinstance(fmt, EFloatFormat)
        assert type(_format_to_ctx(fmt)) is IEEEContext

    def test_plain_efloat_recovers_as_efloat(self):
        from fpy2.libraries.base import MX_E4M3
        fmt = MX_E4M3.format()
        assert isinstance(fmt, EFloatFormat) and not isinstance(fmt, IEEEFormat)
        assert type(_format_to_ctx(fmt)) is EFloatContext

    def test_fixed_before_mpb_fixed(self):
        # `FixedFormat` is an `MPBFixedFormat`, so case order decides this
        fmt = fp.SINT8.format()
        assert isinstance(fmt, FixedFormat) and isinstance(fmt, MPBFixedFormat)
        assert type(_format_to_ctx(fmt)) is FixedContext

    def test_sm_fixed_before_mpb_fixed(self):
        fmt = fp.SMFixedContext(0, 8).format()
        assert isinstance(fmt, SMFixedFormat) and isinstance(fmt, MPBFixedFormat)
        assert type(_format_to_ctx(fmt)) is SMFixedContext

    def test_plain_mpb_fixed_recovers_as_mpb_fixed(self):
        ctx = fp.MPBFixedContext(-1, fp.RealFloat(exp=0, c=127),
                                 neg_maxval=fp.RealFloat(s=True, exp=0, c=128))
        fmt = ctx.format()
        assert not isinstance(fmt, FixedFormat | SMFixedFormat)
        assert type(_format_to_ctx(fmt)) is MPBFixedContext

    def test_exp_format(self):
        from fpy2.libraries.base import MX_E8M0
        fmt = MX_E8M0.format()
        assert isinstance(fmt, ExpFormat)
        assert type(_format_to_ctx(fmt)) is ExpContext


class TestCanonicalContextsRecover():

    @pytest.mark.parametrize('name', [n for n in _CANONICAL if n != 'MX_INT8'])
    def test_canonical_recovers_exactly(self, name: str):
        ctx = _canonical(name)
        assert _format_to_ctx(ctx.format()) == ctx

    def test_integer_contexts_recover_rtz(self):
        # a `Format` carries no rounding mode, so RTZ is chosen for the
        # fixed-point family; the cpp backend requires it of integer storage
        for name in ('INTEGER', 'SINT8', 'SINT16', 'SINT32', 'SINT64',
                     'UINT8', 'UINT16', 'UINT32', 'UINT64'):
            ctx = _canonical(name)
            rec = _format_to_ctx(ctx.format())
            assert rec is not None
            assert rec.rm == fp.RoundingMode.RTZ, name
            assert rec.rm == ctx.rm, name

    def test_float_contexts_recover_rne(self):
        for name in ('FP16', 'FP32', 'FP64', 'BF16', 'TF32', 'MX_E4M3'):
            ctx = _canonical(name)
            rec = _format_to_ctx(ctx.format())
            assert rec is not None
            assert rec.rm == fp.RoundingMode.RNE, name
            assert rec.rm == ctx.rm, name

    def test_mx_int8_rounding_mode_differs(self):
        # documented deviation: `MX_INT8` is built with RNE, but a
        # `FixedFormat` recovers as RTZ like the other fixed-point formats.
        # It uses float storage, so the integer-RTZ requirement is not in play.
        from fpy2.libraries.base import MX_INT8
        rec = _format_to_ctx(MX_INT8.format())
        assert rec is not None
        assert rec.format() == MX_INT8.format()
        assert MX_INT8.rm == fp.RoundingMode.RNE
        assert rec.rm == fp.RoundingMode.RTZ


class TestUnrecoverableFormats():

    def test_real_format_is_none(self):
        # the polymorphic top; the caller falls back to `RealType(None)`
        assert _format_to_ctx(REAL_FORMAT) is None

    def test_unknown_format_is_none(self):
        # a `Format` subclass no context knows how to build
        class _Bogus(fp.number.context.format.Format):
            def __eq__(self, other): return isinstance(other, _Bogus)
            def __hash__(self): return hash(_Bogus)
            def is_close_to(self, other): return False
            def representable_in(self, x): return False
            def canonical_under(self, x): return False
            def normal_under(self, x): return False
            def normalize(self, x): return x
            def round_params(self): return (None, None)

        assert _format_to_ctx(_Bogus()) is None
