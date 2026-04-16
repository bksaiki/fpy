import fpy2 as fp

from hypothesis import given, strategies as st

class EncodeTestCase():
    """Testing `ExpContext.encode()`"""

    @given(st.integers(-127, 127), st.integers(0, 8))
    def test_encode_e8m0(self, exp: int, shift: int):
        x = fp.Float(c=1 << shift, exp=exp - shift)
        i = fp.MX_E8M0.encode(x)
        assert isinstance(i, int), f'x={x}, i={i}'
        assert i == exp + 127

    def test_encode_nan_e8m0(self):
        x = fp.Float.nan()
        i = fp.MX_E8M0.encode(x)
        assert isinstance(i, int), f'x={x}, i={i}'
        assert i == 255

class DecodeTestCase():
    """Testing `ExpContext.decode()`"""

    @given(st.integers(0, 254))
    def test_decode_e8m0(self, i: int):
        x = fp.MX_E8M0.decode(i)
        assert isinstance(x, fp.Float), f'i={i}, x={x}'
        assert x.c == 1
        assert x.exp == i - 127, f'i={i}, x={x}'

    def test_decode_nan_e8m0(self):
        i = 255
        x = fp.MX_E8M0.decode(i)
        assert isinstance(x, fp.Float), f'i={i}, x={x}'
        assert x.isnan, f'i={i}, x={x}'

class EncodeRoundTripTestCase():
    """Ensure `ExpContext.decode()` and `ExpContext.encode()` roundtrips."""

    @given(st.sampled_from(['nan', 'finite']), st.integers(-127, 127), st.integers(0, 8))
    def test_roundtrip(self, kind: str, exp: int, shift: int):
        if kind == 'nan':
            x = fp.Float.nan()
        else:
            x = fp.Float(c=1 << shift, exp=exp - shift)

        i = fp.MX_E8M0.encode(x)
        y = fp.MX_E8M0.decode(i)
        if y.isnan:
            assert x.isnan, f'x={x}, y={y}'
        else:
            assert x == y
