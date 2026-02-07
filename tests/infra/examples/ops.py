"""
Example functions for each operation.
"""

import fpy2 as fp

@fp.fpy
def test_logb(x: fp.Real) -> fp.Real:
    """Example function for `logb`."""
    return fp.logb(x)

@fp.fpy
def test_declcontext(x: fp.Real) -> list[fp.Real]:
    """Example function for `declcontext`."""
    with fp.declcontext(x):
        y = fp.round(x)
    return [x, y]
