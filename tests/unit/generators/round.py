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
    return draw(st.sampled_from(list(fp.RM)))

@st.composite
def overflow_modes(draw):
    """
    Returns a strategy for generating an overflow mode.
    """
    return draw(st.sampled_from(list(fp.OV)))
