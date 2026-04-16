import fpy2 as fp

class TestRepresentable():
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

        assert E8M0.representable_under(NAN)
        assert not E8M0.representable_under(POS_INF)
        assert not E8M0.representable_under(NEG_INF)
        assert not E8M0.representable_under(ZERO)
        assert E8M0.representable_under(MAX_VAL)
        assert E8M0.representable_under(MIN_VAL)
        assert not E8M0.representable_under(HUGE)
        assert not E8M0.representable_under(TINY)
        assert not E8M0.representable_under(THREE_HALF)
        assert E8M0.representable_under(FOUR)
        assert not E8M0.representable_under(NEG_ONE)
