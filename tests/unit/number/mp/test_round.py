import gmpy2 as gmp
import random
import unittest

from fpy2 import RealFloat, MPFloatContext, RM

def _mpfr_rm(rm: RM):
    match rm:
        case RM.RNE:
            return gmp.RoundToNearest
        case RM.RTP:
            return gmp.RoundUp
        case RM.RTN:
            return gmp.RoundDown
        case RM.RTZ:
            return gmp.RoundToZero
        case RM.RAZ:
            return gmp.RoundAwayZero
        case _:
            raise ValueError(f'unsupported rounding mode {rm}')


def _round_mpfr(x: RealFloat, ctx: MPFloatContext) -> RealFloat:
    s_str = '-' if x.s else '+'
    xf = gmp.mpfr(f'{s_str}{hex(x.c)}p{x.exp}', precision=x.p, base=16)
    with gmp.context(
        precision=ctx.pmax,
        emin=gmp.get_emin_min(),
        emax=gmp.get_emax_max(),
        trap_underflow=True,
        trap_overflow=True,
        trap_inexact=False,
        trap_divzero=False,
        round=_mpfr_rm(ctx.rm)
    ):
        yf = gmp.mpfr(xf)

    s = yf.is_signed()
    m_, exp_ = yf.as_mantissa_exp()
    c = int(abs(m_))
    exp = int(exp_)

    return RealFloat(s=s, exp=exp, c=c)

class RoundTestCase(unittest.TestCase):
    """Testing `IEEEContext.round()`"""

    def test_mpfr(self, num_values: int = 10_000, p_max: int = 128, e_max=1024):
        random.seed(1)
        for _ in range(num_values):
            # sample random value
            s = random.choice([False, True])
            c = random.randint(0, 2**p_max - 1)
            e = random.randint(-e_max, e_max)
            x = RealFloat(s=s, e=e, c=c)

            # sample rounding mode
            p = random.randint(2, p_max)
            rm = random.choice([RM.RNE, RM.RTP, RM.RTN, RM.RTZ, RM.RAZ])
            ctx = MPFloatContext(p, rm)

            # round value
            y = ctx.round(x)

            # round value in MPFR
            yf = _round_mpfr(x, ctx)

            # check that they are equal
            self.assertEqual(y, yf, f'x={x}, y={y}, yf={yf}')
