"""
Custom generators for rounding modes.
"""

import fpy2 as fp
from hypothesis import strategies as st

@st.composite
def rounding_modes(draw):
    """
    Returns a strategy for generating a rounding mode.
    """
    return draw(st.sampled_from([fp.RM.RNE, fp.RM.RNA, fp.RM.RTP, fp.RM.RTN, fp.RM.RTZ, fp.RM.RAZ, fp.RM.RTO, fp.RM.RTE]))
