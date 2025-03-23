import math
import random
import re
import unittest

from fpy2 import Float, IEEEContext, RM


_FLOAT_RE = re.compile(r'([+/-]?)0x(0|1).([0-9a-f]+)p([+/-]?[0-9]+)')

def _float_as_real(x: float):
    if math.isnan(x):
        t = math.copysign(1, x)
        return Float(s=t < 0, isnan=True)
    elif math.isinf(x):
        return Float(s=x < 0, isinf=True)
    elif x == 0:
        t = math.copysign(1, x)
        return Float(s=t < 0)
    else:
        m = re.match(_FLOAT_RE, float.hex(x))
        assert m is not None, f'x={x} ({float.hex(x)})'
        s_str, i_str, c_str, e_str = m.groups()

        s = s_str == '-'
        c = int(f'1{c_str}' if i_str == '1' else c_str, 16)
        exp = int(e_str) - 4 * len(c_str)
        return Float(s=s, exp=exp, c=c)

class RoundTestCase(unittest.TestCase):
    """Testing `IEEEContext.round()`"""

    def test_native(self, num_values: int = 10_000, mantissa_len=128):
        # rounding context for native Python floats
        fp64 = IEEEContext(11, 64, RM.RNE)
        # sample 10_000 floating-point strings
        random.seed(1)
        xs: list[str] = []
        for _ in range(num_values):
            s = random.choice(['-', '+'])
            c = ''.join(random.choices('0123456789', k=mantissa_len))
            e = random.randint(-320, 308)
            x = f'{s}1.{c}e{e}'
            xs.append(x)
        # run conversion to float
        for x in xs:
            f1 = _float_as_real(float(x))
            f2 = fp64.round(x)
            self.assertEqual(f1, f2, f'x={x}, f1={f1}, f2={f2}')
