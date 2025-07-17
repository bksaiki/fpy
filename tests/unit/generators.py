"""
Custom generators for Hypothesis tests.
"""

import fpy2 as fp
from hypothesis import strategies as st

@st.composite
def rounding_modes(draw):
    """
    Returns a strategy for generating a rounding mode.
    """
    return draw(st.sampled_from([fp.RM.RNE, fp.RM.RNA, fp.RM.RTP, fp.RM.RTN, fp.RM.RTZ, fp.RM.RAZ, fp.RM.RTO, fp.RM.RTE]))

@st.composite
def real_floats(draw, p):
    """
    Returns a strategy for generating a RealFloat with some maximum precision `p`.
    """
    s = draw(st.booleans())
    exp = draw(st.integers())
    c = draw(st.integers(min_value=0, max_value=2**p - 1))
    return fp.RealFloat(s, exp, c)
