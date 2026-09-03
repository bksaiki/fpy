"""
`enable_neg_zero` across every format and context that carries it.

The invariant it exists for: a context's `round` may never produce a value its
own `format()` rejects.
"""

import pytest

import fpy2 as fp
from fpy2.number import Float, RealFloat, RoundingMode

_NEG_ZERO = Float(s=True, exp=0, c=0)
_TINY = RealFloat(s=True, exp=-300, c=1)


def _contexts(enable_neg_zero: bool):
    kw = {'enable_neg_zero': enable_neg_zero}
    return [
        ('MPFloat', fp.MPFloatContext(24, **kw)),
        ('MPSFloat', fp.MPSFloatContext(24, -10, **kw)),
        ('MPBFloat', fp.MPBFloatContext(24, -10, RealFloat.from_int(1), **kw)),
        ('MPFixed', fp.MPFixedContext(-8, RoundingMode.RNE, **kw)),
        ('MPBFixed', fp.MPBFixedContext(-8, RealFloat.from_int(1), **kw)),
    ]


_IDS = [name for name, _ in _contexts(True)]


class TestTheFormatRefusesIt:
    @pytest.mark.parametrize('_name,ctx', _contexts(False), ids=_IDS)
    def test_off(self, _name, ctx):
        assert not ctx.format().representable_in(_NEG_ZERO)

    @pytest.mark.parametrize('_name,ctx', _contexts(True), ids=_IDS)
    def test_on_by_default(self, _name, ctx):
        assert ctx.format().representable_in(_NEG_ZERO)


class TestRoundingStaysInsideTheFormat:
    """The reason the context needs the flag as well as the format."""

    @pytest.mark.parametrize('_name,ctx', _contexts(False), ids=_IDS)
    def test_a_negative_value_that_lands_on_zero(self, _name, ctx):
        """`MPFloatContext` is unbounded below, so its result is the operand
        itself -- signed, and representable."""
        r = ctx.round(_TINY)
        assert ctx.format().representable_in(r)
        if r.is_zero():
            assert not r.s, r

    @pytest.mark.parametrize('_name,ctx', _contexts(False), ids=_IDS)
    def test_an_exact_negative_zero(self, _name, ctx):
        r = ctx.round(_NEG_ZERO)
        assert not r.s, r
        assert ctx.format().representable_in(r)

    @pytest.mark.parametrize('_name,ctx', _contexts(True), ids=_IDS)
    def test_the_sign_survives_by_default(self, _name, ctx):
        assert ctx.round(_NEG_ZERO).s


class TestThePlumbing:
    @pytest.mark.parametrize('_name,ctx', _contexts(False), ids=_IDS)
    def test_from_format_carries_it(self, _name, ctx):
        """Otherwise a context built from a format would round outside it."""
        assert not type(ctx).from_format(ctx.format()).enable_neg_zero

    @pytest.mark.parametrize('_name,ctx', _contexts(False), ids=_IDS)
    def test_with_params_carries_it(self, _name, ctx):
        assert not ctx.with_params().enable_neg_zero

    @pytest.mark.parametrize('_name,ctx', _contexts(False), ids=_IDS)
    def test_it_separates_formats(self, _name, ctx):
        """A format that refuses `-0` is not the one that admits it."""
        other = ctx.with_params(enable_neg_zero=True)
        assert ctx.format() != other.format()
        assert hash(ctx.format()) != hash(other.format())
        assert ctx != other

    @pytest.mark.parametrize('_name,ctx', _contexts(False), ids=_IDS)
    def test_a_bool_is_required(self, _name, ctx):
        with pytest.raises(TypeError, match='enable_neg_zero'):
            ctx.with_params(enable_neg_zero=0)   # type: ignore[arg-type]


class TestZeroRefuses:
    """`zero(s=True)` has no value to return, so it raises."""

    @pytest.mark.parametrize('_name,ctx', [
        c for c in _contexts(False) if c[0] in ('MPSFloat', 'MPBFloat')
    ], ids=['MPSFloat', 'MPBFloat'])
    def test_it_raises(self, _name, ctx):
        fmt = ctx.format()
        assert not fmt.zero(s=False).s
        with pytest.raises(ValueError, match='not representable'):
            fmt.zero(s=True)

    @pytest.mark.parametrize('_name,ctx', [
        c for c in _contexts(True) if c[0] in ('MPSFloat', 'MPBFloat')
    ], ids=['MPSFloat', 'MPBFloat'])
    def test_it_does_not_by_default(self, _name, ctx):
        assert ctx.format().zero(s=True).s
