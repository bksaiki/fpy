"""
`layers.evaluate` on a small random Qwen3.  Needs a GPU and `transformers`;
skipped without either.

    pytest serve/tests
"""

import math

import pytest

from fpy2.backend.triton import unavailable

_WHY = unavailable()
pytestmark = pytest.mark.skipif(_WHY is not None, reason=_WHY or '')

import layers
import swap


def _pooled(stats: dict[str, layers.Stats]) -> layers.Stats:
    total = layers.Stats()
    for s in stats.values():
        total += s
    return total


def test_bf16_exact_is_correctly_rounded_and_a_design_is_not(model, tokens) -> None:
    """`bf16-exact` rounds the exact product once, so every element is
    `fl(y)`, within half an ulp; its propagated error is its input rounding.
    A design errs locally in every block."""
    run = swap.patch(model)
    stats = layers.evaluate(model, run, [tokens], ['bf16-exact', 'nv.hopper.bf16.f32'])
    assert run.mode == 'fp32'
    exact = _pooled(stats['bf16-exact']).report(layers.METRICS)
    assert exact['rounded'] == 1.0 and exact['ulp_max'] <= math.log2(1.5)
    assert 0 < exact['normwise'] <= layers.U and exact['propagated'] > 0
    blocks = layers.by_block(stats['nv.hopper.bf16.f32'])
    assert list(blocks) == ['block 0', 'block 1', 'lm_head']
    assert all(s.report(['backward'])['backward'] > 0 for s in blocks.values())


def test_only_the_selected_metrics_are_computed(model, tokens) -> None:
    run = swap.patch(model)
    stats = layers.evaluate(model, run, [tokens], ['amd.cdna2.bf16'], ['magnitude_bias'])
    s = _pooled(stats['amd.cdna2.bf16'])
    assert s.magnitude_bias != 0 and s.n > 0
    assert s.err == s.prop_ref == s.backward == s.ulp == s.rounded == s.bias == 0
