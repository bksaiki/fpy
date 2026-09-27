"""
`decode.greedy`, and teacher-forced divergence, on a small random Qwen3.  Needs a GPU and `transformers`;
skipped without either.

    pytest serve/tests
"""

import pytest
import torch

from fpy2.backend.triton import unavailable

_WHY = unavailable()
pytestmark = pytest.mark.skipif(_WHY is not None, reason=_WHY or '')

import decode
import metrics
import perplexity
import swap


def test_a_run_stops_at_its_first_departure(model: torch.nn.Module, tokens: torch.Tensor) -> None:
    """R0 against itself never diverges; against a reference altered at
    position 3, decoding stops there and reports 3."""
    swap.patch(model)
    ref = decode.greedy(model, tokens, 8, eos=())
    assert len(ref) == 8
    assert decode.divergence(ref, decode.greedy(model, tokens, 8, (), ref)) is None
    altered = [*ref[:3], (ref[3] + 1) % 96, *ref[4:]]
    got = decode.greedy(model, tokens, 8, (), altered)
    assert len(got) == 4 and decode.divergence(altered, got) == 3


def test_forced_divergence_indexes_the_reply(model: torch.nn.Module, tokens: torch.Tensor) -> None:
    """Forced on its own greedy reply, R0 never misses a token nor departs
    from itself, and a design here no more than free-running does; with the
    reply altered at position 3, R0's first miss is 3."""
    run = swap.patch(model)
    ref = decode.greedy(model, tokens, 8, eos=())
    seq = torch.cat([tokens, tokens.new_tensor([ref])], -1)
    design = 'amd.cdna2.bf16'
    run.mode = design
    free = decode.divergence(ref, decode.greedy(model, tokens, 8, (), ref))
    run.mode = 'fp32'
    totals = perplexity.evaluate(model, run, [seq], [design], starts=[tokens.shape[1]])
    assert totals['fp32'].first_miss == [None] and totals['fp32'].first['fp32'] == [None]
    assert totals[design].first['fp32'] == [free]
    altered = seq.clone()
    altered[0, tokens.shape[1] + 3] = (ref[3] + 1) % 96
    assert perplexity.evaluate(model, run, [altered], [], starts=[tokens.shape[1]])[
        'fp32'].first_miss == [3]


def test_divergences_count_and_bound_the_index() -> None:
    t = perplexity.Totals(first={'fp32': [None, 4, 10, None]})
    got = decode.divergences({'fp32': perplexity.Totals(), 'd': t})['fp32']['d']
    assert got['diverged'] == 0.5 and got['median'] == 7.0
    assert 4 <= got['median_lo'] <= got['median'] <= got['median_hi'] <= 10
    lo, hi = metrics.bootstrap([1.0, 2.0, 3.0], lambda xs: sum(xs) / len(xs))
    assert 1.0 <= lo <= 2.0 <= hi <= 3.0

