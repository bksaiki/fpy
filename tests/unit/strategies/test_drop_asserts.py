"""Unit tests for :func:`fpy2.strategies.drop_asserts`."""


import fpy2 as fp

from fpy2.strategies import drop_asserts


@fp.fpy(ctx=fp.FP64)
def _recip(x: fp.Real) -> fp.Real:
    assert x != 0, 'nonzero'
    return 1 / x


def test_it_agrees_where_the_assertions_held():
    assert repr(drop_asserts(_recip)(4.0)) == repr(_recip(4.0))
