"""
`swap.patch` on a small random Qwen3.  Needs a GPU and `transformers`;
skipped without either.

    pytest serve/tests
"""

import pytest
import torch

from fpy2.backend.triton import unavailable

_WHY = unavailable()
pytestmark = pytest.mark.skipif(_WHY is not None, reason=_WHY or '')

import kernels
import swap


def _logits(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(tokens).logits


def test_each_run_computes_as_it_says(
    model: torch.nn.Module, tokens: torch.Tensor, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`fp32` is the unpatched model, bit for bit; a design routes every
    linear layer, `lm_head` included, through its kernel and lands near
    `bf16-exact`; and switching back restores `fp32`."""
    before = _logits(model, tokens)
    run = swap.patch(model)
    assert torch.equal(_logits(model, tokens), before)

    run.mode = 'bf16-exact'
    exact = _logits(model, tokens)
    calls = []
    real = kernels.matmul
    monkeypatch.setattr(kernels, 'matmul', lambda *a, **kw: calls.append(1) or real(*a, **kw))
    run.mode = 'nv.hopper.bf16.f32'
    design = _logits(model, tokens)
    assert len(calls) == sum(isinstance(m, torch.nn.Linear) for m in model.modules())
    assert not torch.equal(exact, before)
    assert torch.allclose(design, exact, rtol=0, atol=1e-3 * exact.abs().max().item())

    run.mode = 'fp32'
    assert torch.equal(_logits(model, tokens), before)


def test_a_design_through_the_run_is_kernels_linear(
    model: torch.nn.Module, tokens: torch.Tensor,
) -> None:
    """Weights prepared once and an input rounded once for the layers that
    share it change nothing: every layer's output, at `split_k` 1 and 4, and
    after a weight changes in place, is `kernels.linear`'s on the same input,
    bit for bit."""
    run = swap.patch(model)
    pairs = []

    def check(layer: torch.nn.Linear, inputs: tuple[torch.Tensor], y: torch.Tensor) -> None:
        pairs.append((y, kernels.linear(inputs[0], layer.weight, run.mode,
                                         split_k=run.split_k, combine=run.combine)))

    handles = [m.register_forward_hook(check) for m in model.modules()
               if isinstance(m, torch.nn.Linear)]
    w = model.lm_head.weight
    try:
        for split_k, combine in ((1, 'linear'), (4, 'linear'), (4, 'tree')):
            run.mode, run.split_k, run.combine = 'amd.cdna2.bf16', split_k, combine
            _logits(model, tokens)
        with torch.no_grad():
            w.neg_()
        _logits(model, tokens)
    finally:
        with torch.no_grad():
            w.neg_()
        for h in handles:
            h.remove()
    assert pairs and all(torch.equal(got, want) for got, want in pairs)
