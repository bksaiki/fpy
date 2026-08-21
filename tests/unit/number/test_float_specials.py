"""
Unit tests for `enable_nan`/`enable_inf` on the multi-precision float contexts.

`MPFloatContext`, `MPSFloatContext` and `MPBFloatContext` state NaN and infinity
as flags, defaulting to *enabled* -- the opposite of the fixed-point contexts.
With one turned off, a special operand is substituted by `nan_value`/`inf_value`
where the context names one and refused otherwise, and the flag has to survive
the `format()`/`from_format()` round trip.
"""

import fpy2 as fp
import pytest

from fpy2.number import Float, RealFloat

_MAXVAL = RealFloat(exp=5, c=0x7ff)
"""an `FP16`-shaped bound for the bounded context"""

_NAN = Float(isnan=True)
_INF = Float(isinf=True)
_NEG_INF = Float(isinf=True, s=True)


def _contexts(**kwargs) -> list[fp.Context]:
    """The three float contexts, each built with the same special-value
    keywords."""
    return [
        fp.MPFloatContext(11, **kwargs),
        fp.MPSFloatContext(11, -14, **kwargs),
        fp.MPBFloatContext(11, -14, _MAXVAL, **kwargs),
    ]


def _ids(prefix: str = '') -> list[str]:
    return [f'{prefix}{n}' for n in ('mp', 'mps', 'mpb')]


class TestDefaults:
    """Both flags default on, so every existing context is unaffected."""

    @pytest.mark.parametrize('ctx', _contexts(), ids=_ids())
    def test_the_specials_pass_through(self, ctx):
        assert ctx.round(_NAN).isnan
        assert ctx.round(_INF).isinf and not ctx.round(_INF).s
        assert ctx.round(_NEG_INF).isinf and ctx.round(_NEG_INF).s

    @pytest.mark.parametrize('ctx', _contexts(), ids=_ids())
    def test_the_format_says_so(self, ctx):
        fmt = ctx.format()
        assert fmt.enable_nan and fmt.enable_inf
        assert fmt.representable_in(_NAN)
        assert fmt.representable_in(_INF)


class TestRefusal:
    """With a rule off and no substitute named, rounding that special raises.

    `float_to_fixed` probes exactly this way and catches only `ValueError`, so
    the exception type is load-bearing.
    """

    @pytest.mark.parametrize('ctx', _contexts(enable_nan=False), ids=_ids())
    def test_nan_is_refused(self, ctx):
        with pytest.raises(ValueError):
            ctx.round(_NAN)
        # the infinity is untouched
        assert ctx.round(_INF).isinf

    @pytest.mark.parametrize('ctx', _contexts(enable_inf=False), ids=_ids())
    def test_infinity_is_refused(self, ctx):
        for x in (_INF, _NEG_INF):
            with pytest.raises(ValueError):
                ctx.round(x)
        assert ctx.round(_NAN).isnan

    @pytest.mark.parametrize('ctx', _contexts(enable_nan=False, enable_inf=False),
                             ids=_ids())
    def test_a_finite_value_still_rounds(self, ctx):
        assert ctx.round(1) == 1
        assert ctx.round(0).is_zero()


class TestSubstitution:
    """A named substitute is returned in place of the refused special."""

    @pytest.mark.parametrize('ctx', _contexts(enable_nan=False, nan_value=Float(c=3, exp=0)),
                             ids=_ids())
    def test_nan_takes_its_value_verbatim(self, ctx):
        """A NaN's sign carries no meaning, so the substitute is used as
        written for either one."""
        assert ctx.round(_NAN) == 3
        assert ctx.round(Float(isnan=True, s=True)) == 3

    @pytest.mark.parametrize('ctx', _contexts(enable_inf=False, inf_value=Float(c=3, exp=0)),
                             ids=_ids())
    def test_infinity_takes_its_value_with_the_operand_sign(self, ctx):
        """An infinity is a signed magnitude, so the substitute names the
        magnitude and the operand supplies the sign -- as `EFloatContext` has
        always done.  A float format is sign-symmetric, so both are available.
        """
        assert ctx.round(_INF) == 3
        assert ctx.round(_NEG_INF) == -3

    @pytest.mark.parametrize('ctx', _contexts(enable_inf=False, inf_value=Float(isnan=True)),
                             ids=_ids())
    def test_a_special_may_substitute_another(self, ctx):
        """NaN is still enabled, so an infinity may round to it."""
        assert ctx.round(_INF).isnan


class TestValidation:
    """A substitute the format cannot represent is rejected at construction."""

    def test_an_unrepresentable_finite_value_is_rejected(self):
        # 7 needs 3 digits; `pmax=2` cannot hold it
        with pytest.raises(ValueError):
            fp.MPFloatContext(2, enable_inf=False, inf_value=Float(c=7, exp=0))

    def test_a_substitute_needing_a_disabled_rule_is_rejected(self):
        """Rounding an infinity to NaN needs NaN, which is also off."""
        with pytest.raises(ValueError):
            fp.MPFloatContext(11, enable_nan=False, enable_inf=False,
                              inf_value=Float(isnan=True))

    def test_a_value_past_the_bound_is_rejected(self):
        with pytest.raises(ValueError):
            fp.MPBFloatContext(11, -14, _MAXVAL, enable_inf=False,
                               inf_value=Float(c=1, exp=40))

    @pytest.mark.parametrize('flag', ['enable_nan', 'enable_inf'])
    def test_a_non_bool_flag_is_rejected(self, flag):
        with pytest.raises(TypeError):
            fp.MPFloatContext(11, **{flag: 1})

    def test_a_substitute_that_is_not_a_float_is_rejected(self):
        with pytest.raises(TypeError):
            fp.MPFloatContext(11, enable_nan=False, nan_value=1.0)


_SAT = fp.OverflowMode.SATURATE


class TestOverflow:
    """Only `MPBFloatContext` has a bound, and an overflow that would round to
    infinity is substituted exactly like an infinite operand."""

    def _ctx(self, **kwargs):
        return fp.MPBFloatContext(11, -14, _MAXVAL, **kwargs)

    def test_an_enabled_infinity_still_receives_the_overflow(self):
        assert self._ctx().round(1e9).isinf

    def test_a_refused_infinity_refuses_the_overflow(self):
        with pytest.raises(ValueError):
            self._ctx(enable_inf=False).round(1e9)

    @pytest.mark.parametrize('sign', [False, True], ids=['pos', 'neg'])
    def test_a_substituted_infinity_receives_the_overflow(self, sign):
        """The operand is a finite value whose magnitude ran out of format, so
        its sign is carried through -- as the `SATURATE` arm does."""
        ctx = self._ctx(enable_inf=False, inf_value=Float(x=_MAXVAL))
        x = -1e9 if sign else 1e9
        assert ctx.round(x) == ctx.maxval(sign)
        # ... and it agrees with saturating the same operand
        assert ctx.round(x) == self._ctx(overflow=_SAT).round(x)

    def test_an_overflow_that_saturates_is_untouched(self):
        """`RTZ` rounds an overflow down to the bound, so it never wanted the
        infinity and the flag does not enter into it."""
        ctx = self._ctx(rm=fp.RM.RTZ, enable_inf=False)
        assert ctx.round(1e9) == ctx.maxval()

    def test_saturating_overflow_is_untouched(self):
        ctx = self._ctx(overflow=_SAT, enable_inf=False)
        assert ctx.round(1e9) == ctx.maxval()


class TestWithParams:
    """`with_params` carries the new fields, which is what the transforms use
    to shed a rule."""

    @pytest.mark.parametrize('ctx', _contexts(), ids=_ids())
    def test_a_rule_can_be_turned_off(self, ctx):
        off = ctx.with_params(enable_nan=False, enable_inf=False)
        assert not off.enable_nan and not off.enable_inf
        # the change reaches the format, which is what the analyses read
        assert not off.format().enable_nan
        assert not off.format().enable_inf
        with pytest.raises(ValueError):
            off.round(_NAN)

    @pytest.mark.parametrize('ctx', _contexts(), ids=_ids())
    def test_the_other_fields_survive(self, ctx):
        off = ctx.with_params(enable_nan=False)
        assert off.pmax == ctx.pmax
        assert off.rm == ctx.rm
        assert off.enable_inf == ctx.enable_inf


class TestEquality:
    """The flags take part in identity, so two contexts differing only in one
    are distinct and hash apart."""

    @pytest.mark.parametrize('ctx', _contexts(), ids=_ids())
    def test_a_disabled_rule_is_a_different_context(self, ctx):
        off = ctx.with_params(enable_nan=False)
        assert off != ctx
        assert off.format() != ctx.format()
        assert len({off, ctx}) == 2

    @pytest.mark.parametrize('ctx', _contexts(), ids=_ids())
    def test_the_same_flags_compare_equal(self, ctx):
        same = ctx.with_params(enable_nan=True)
        assert same == ctx
        assert hash(same) == hash(ctx)


class TestSubstituteIdentity:
    """A context has to equal itself, so the substitutes are compared by
    encoding rather than by `==`, which is numeric: it calls a NaN unequal to
    itself and the two zeros equal."""

    @pytest.mark.parametrize('ctx', _contexts(enable_inf=False, inf_value=Float(isnan=True)),
                             ids=_ids())
    def test_a_nan_substitute_still_equals_itself(self, ctx):
        same = type(ctx)(*_args(ctx), enable_inf=False, inf_value=Float(isnan=True))
        assert ctx == ctx
        assert ctx == same
        assert len({ctx, same}) == 1
        assert {ctx: 1}[same] == 1

    @pytest.mark.parametrize('ctx', _contexts(), ids=_ids())
    def test_the_two_zeros_are_different_substitutes(self, ctx):
        pos = ctx.with_params(enable_nan=False, nan_value=Float(c=0))
        neg = ctx.with_params(enable_nan=False, nan_value=Float(c=0, s=True))
        assert pos != neg
        assert pos.round(_NAN).s is False
        assert neg.round(_NAN).s is True


def _args(ctx) -> tuple:
    """The positional arguments `ctx` was built from."""
    if isinstance(ctx, fp.MPBFloatContext):
        return (ctx.pmax, ctx.emin, ctx.pos_maxval)
    if isinstance(ctx, fp.MPSFloatContext):
        return (ctx.pmax, ctx.emin)
    return (ctx.pmax,)
