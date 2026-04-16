import fpy2 as fp

from hypothesis import given, strategies as st

class EncodeTestCase():
    """Testing `SMFixedContext.encode()`"""

    @given(st.integers(-3, 3), st.integers(2, 8))
    def test_encode(self, scale: int, nbits: int):
        ctx = fp.SMFixedContext(scale, nbits)
        for i in range(1 - (1 << (ctx.nbits - 1)), 1 << (ctx.nbits - 1)):
            x = fp.Float(exp=scale, m=i, ctx=ctx)
            assert ctx.representable_under(x), f'x={x} is not representable under ctx={ctx}'

            expect = (1 if x.s else 0) << (ctx.nbits - 1) | abs(x.c)

            encoded = ctx.encode(x)
            assert isinstance(encoded, int), f'x={x}, encoded={encoded}'
            assert encoded == expect, f'x={x}, encoded={encoded}'


class DecodeTestCase():
    """Testing `SMFixedContext.decode()`"""

    @given(st.integers(-3, 3), st.integers(2, 8))
    def test_decode(self, scale: int, nbits: int):
        ctx = fp.SMFixedContext(scale, nbits)
        for i in range(1 << ctx.nbits):
            decoded = ctx.decode(i)

            s = bool(i >> (ctx.nbits - 1))
            exp = ctx.scale
            c = i & fp.utils.bitmask(ctx.nbits - 1)
            expect = fp.Float(s, exp, c)

            assert isinstance(decoded, fp.Float), f'i={i}, decoded={decoded}'
            assert ctx.representable_under(decoded), f'i={i}, decoded={decoded}'
            assert decoded == expect, f'i={i}, decoded={decoded}'


class EncodeRoundTripTestCase():
    """Ensure `SMFixedContext.decode()` and `SMFixedContext.encode()` roundtrips."""

    @given(st.integers(-3, 3), st.integers(2, 8))
    def test_roundtrip(self, scale: int, nbits: int):
        ctx = fp.SMFixedContext(scale, nbits)
        for i in range(1 << ctx.nbits):
            x = ctx.decode(i)
            j = ctx.encode(x)
            assert isinstance(j, int), f'i={i}, x={x}, j={j}'
            assert j == i, f'i={i}, x={x}, j={j}'
