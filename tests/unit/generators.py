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
def real_floats(draw, prec: int | None = None, exp_min: int | None = None, exp_max: int | None = None):
    """
    Returns a strategy for generating a `RealFloat` with a
    maximum precision `prec`, and exponent range between
    `exp_min` and `exp_max`.
    """
    if prec is None:
        max_value = None
    else:
        max_value = (1 << prec) - 1

    s = draw(st.booleans())
    exp = draw(st.integers(min_value=exp_min, max_value=exp_max))
    c = draw(st.integers(min_value=0, max_value=max_value))
    return fp.RealFloat(s, exp, c)

@st.composite
def floats(
    draw,
    prec: int | None = None,
    exp_min: int | None = None,
    exp_max: int | None = None,
    allow_nan: bool = True,
    allow_infinity: bool = True,
    ctx: fp.Context | None = None
):
    """
    Returns a strategy for generating a Float. If ctx is provided,
    the generated Float must be representable in that context.
    """
    if ctx is None and prec is None:
        raise ValueError("either 'ctx' or 'prec' must be provided.")

    # Choose between finite, infinite, or NaN values
    classes = ['finite']
    if allow_infinity:
        classes.append('inf')
    if allow_nan:
        classes.append('nan')

    float_type = draw(st.sampled_from(classes))
    if float_type == 'nan':
        return fp.Float.nan()
    elif float_type == 'inf':
        s = draw(st.booleans())
        return fp.Float.inf(s)
    else:  # finite
        if ctx is not None:
            # Generate representable finite float
            x = draw(real_floats(prec, exp_min, exp_max).filter(lambda x: ctx.representable_under(x)))
            return fp.Float.from_real(x, ctx, checked=False)
        else:
            # Generate arbitrary finite float
            x = draw(real_floats(prec, exp_min, exp_max))
            return fp.Float.from_real(x)
