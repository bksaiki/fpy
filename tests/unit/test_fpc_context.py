"""Unit tests for :class:`fpy2.fpc_context.FPCoreContext`."""

import pytest

import fpy2 as fp
from fpy2.fpc_context import FPCoreContext


@pytest.mark.parametrize('ctx', [
    fp.FP32,
    fp.FP64,
    fp.INTEGER,
    fp.MPFixedContext(-8),
    fp.FixedContext(True, -4, 16),
], ids=repr)
def test_a_context_round_trips(ctx):
    assert FPCoreContext.from_context(ctx).to_context() == ctx
