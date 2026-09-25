"""
`decode.greedy` on a small random Qwen3.  Needs a GPU and `transformers`;
skipped without either.

    pytest serve/tests
"""

import pytest

from fpy2.backend.triton import unavailable

_WHY = unavailable()
pytestmark = pytest.mark.skipif(_WHY is not None, reason=_WHY or '')

import decode
import swap


def test_a_run_stops_at_its_first_departure(model, tokens) -> None:
    """R0 against itself never diverges; against a reference altered at
    position 3, decoding stops there and reports 3."""
    swap.patch(model)
    ref = decode.greedy(model, tokens, 8, eos=())
    assert len(ref) == 8
    assert decode.divergence(ref, decode.greedy(model, tokens, 8, (), ref)) is None
    altered = [*ref[:3], (ref[3] + 1) % 96, *ref[4:]]
    got = decode.greedy(model, tokens, 8, (), altered)
    assert len(got) == 4 and decode.divergence(altered, got) == 3
