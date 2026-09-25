"""
`perplexity.evaluate` on a small random Qwen3.  Needs a GPU and
`transformers`; skipped without either.

    pytest serve/tests
"""

import math

import pytest

from fpy2.backend.triton import unavailable

_WHY = unavailable()
pytest.importorskip('transformers')
pytestmark = pytest.mark.skipif(_WHY is not None, reason=_WHY or '')

import perplexity
import swap


def test_the_baseline_is_its_own_reference(model, tokens) -> None:
    """R0 against itself is no distance at all, and its perplexity is the
    model's cross-entropy over the segments; another run is some distance."""
    import torch
    run = swap.patch(model)
    segs = perplexity.segments(torch.cat([tokens, tokens.flip(-1)], -1), context=16)
    totals = perplexity.evaluate(model, run, segs, ['bf16-exact'])
    r0, r1 = totals['fp32'].report(), totals['bf16-exact'].report()
    assert (r0['kl'], r0['top1'], r0['rms_dp']) == (0.0, 1.0, 0.0)
    with torch.no_grad():
        nll = torch.cat([torch.nn.functional.cross_entropy(
            model(s).logits[0, :-1], s[0, 1:], reduction='none') for s in segs])
    assert math.isclose(r0['ppl'], math.exp(nll.mean().item()), rel_tol=1e-6)
    assert r0['tokens'] == r1['tokens'] == 2 * 15
    assert r1['kl'] > 0
    assert run.mode == 'fp32'
