import fpy2 as fp
import unittest

class TestRepresentable(unittest.TestCase):
    """Test `ExpContext` representability."""

    def test_e8m0(self):
        E8M0 = fp.ExpContext(8, 0)
        NAN = fp.Float.nan()
        POS_INF = fp.Float.inf()
        NEG_INF = fp.Float.inf(s=True)
        ZERO = fp.Float.zero()
        MAX_VAL = fp.Float(c=1, exp=127)
        MIN_VAL = fp.Float(c=1, exp=-127)
        HUGE = fp.Float(c=1, exp=128)
        TINY = fp.Float(c=1, exp=-128)
        THREE_HALF = fp.Float(c=3, exp=-1)
        FOUR = fp.Float.from_int(4)
        NEG_ONE = fp.Float.from_int(-1)

        self.assertTrue(E8M0.representable_under(NAN))
        self.assertFalse(E8M0.representable_under(POS_INF))
        self.assertFalse(E8M0.representable_under(NEG_INF))
        self.assertFalse(E8M0.representable_under(ZERO))
        self.assertTrue(E8M0.representable_under(MAX_VAL))
        self.assertTrue(E8M0.representable_under(MIN_VAL))
        self.assertFalse(E8M0.representable_under(HUGE))
        self.assertFalse(E8M0.representable_under(TINY))
        self.assertFalse(E8M0.representable_under(THREE_HALF))
        self.assertTrue(E8M0.representable_under(FOUR))
        self.assertFalse(E8M0.representable_under(NEG_ONE))
