"""
`scoring.frozen` on a small random Qwen3.  Needs a GPU and `transformers`;
skipped without either.

    pytest serve/tests
"""

import math

import pytest
import torch

from fpy2.backend.triton import unavailable

_WHY = unavailable()
pytestmark = pytest.mark.skipif(_WHY is not None, reason=_WHY or '')

from core import quant, scoring, swap, workloads


def test_a_frozen_run_is_its_arithmetic_alone(model: torch.nn.Module, tokens: torch.Tensor) -> None:
    """Frozen to the exact run, the exact run is no distance at all and rounds
    nothing differently; R1 an ulp up or down at random is some distance,
    less than Hopper's truncating chain; free runs are reported beside."""
    run = swap.patch(model)
    swap.use(run, model, quant.SCHEMES['fp8-row'])
    segs = workloads.segments(torch.cat([tokens, tokens.flip(-1)], -1), context=16)
    rand, hopper = 'fp8-row-exact-rand', 'nv.hopper.e4m3.f32'
    got = scoring.frozen(model, run, segs, ['fp8-row-exact', rand, hopper], free={rand})
    held = got[scoring.Kind.FROZEN]
    exact = held['fp8-row-exact'].report()
    assert (exact['kl'], exact['disagree'], exact['rerounded']) == (0.0, 0.0, 0.0)
    assert 0 < held[rand].report()['kl'] < held[hopper].report()['kl']
    assert set(got[scoring.Kind.FREE]) == {rand}
    assert (run.mode, run.record, run.frozen) == ('fp32', None, None)
    swap.use(run, model, swap.BF16)


def test_random_rounding_is_the_seeds_alone(model: torch.nn.Module, tokens: torch.Tensor) -> None:
    """R1 an ulp up or down at random draws its signs from the seed and the
    segment: the same seed gives the same KL, whatever ran before; another
    seed, another."""
    run = swap.patch(model)
    segs = workloads.segments(torch.cat([tokens, tokens.flip(-1)], -1), context=16)
    rand = 'bf16-exact-rand'
    a = scoring.frozen(model, run, segs, [rand], seed=3)[scoring.Kind.FROZEN][rand].kl
    scoring.frozen(model, run, segs, ['bf16-exact', rand], seed=7)
    b = scoring.frozen(model, run, segs, [rand], seed=3)[scoring.Kind.FROZEN][rand].kl
    c = scoring.frozen(model, run, segs, [rand], seed=4)[scoring.Kind.FROZEN][rand].kl
    assert a == b != c


def test_divergence_averages_over_every_position() -> None:
    """KL and disagreement are means over every position, across blocks."""
    g = torch.Generator().manual_seed(0)
    ref = torch.log_softmax(torch.randn(10, 7, generator=g, dtype=torch.float64), -1)
    run = torch.log_softmax(torch.randn(10, 7, generator=g, dtype=torch.float64), -1)
    d = scoring.Divergence()
    d.add(list(ref.split(4)), run.split(4))
    assert math.isclose(d.kl[0], (ref.exp() * (ref - run)).sum(-1).mean().item(), rel_tol=1e-12)
    assert d.disagree[0] == (ref.argmax(-1) != run.argmax(-1)).double().mean().item()
