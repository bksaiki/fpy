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
import quant
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


def test_a_scheme_picks_its_designs_and_quantizes_both_operands(
    model: torch.nn.Module, tokens: torch.Tensor,
) -> None:
    """`fp8-row`'s designs are the E4M3 ones (`:fnuz`'s CDNA3's); its exact run
    is, layer by layer, the quantized operands' FP64 product rounded once;
    each design, its scales applied after the kernel, is nearer the exact run
    than quantizing moved it from R0 (Ada's and Hopper's 13-bit accumulators
    come closest)."""
    fp8 = quant.SCHEMES['fp8-row']
    assert swap.modes(fp8) == ('fp32', 'fp8-row-exact', 'nv.ada.e4m3.f32',
                               'nv.hopper.e4m3.f32', 'nv.blackwell.e4m3.f32')
    assert swap.modes(quant.scheme('fp8-row:fnuz'))[2:] == ('amd.cdna3.fp8',)
    run = swap.patch(model)
    r0 = _logits(model, tokens)
    run.scheme, run.mode = fp8, 'fp8-row-exact'
    pairs = []

    def check(layer: torch.nn.Linear, inputs: tuple[torch.Tensor], y: torch.Tensor) -> None:
        qa = quant.quantize(inputs[0].reshape(-1, inputs[0].shape[-1]), fp8.x)
        qw = quant.quantize(layer.weight, fp8.w)
        pairs.append((y.reshape(qa.elements.shape[0], -1),
                      (qa.dequantize() @ qw.dequantize().T).float()))

    handles = [m.register_forward_hook(check) for m in model.modules()
               if isinstance(m, torch.nn.Linear)]
    try:
        exact = _logits(model, tokens)
    finally:
        for h in handles:
            h.remove()
    assert pairs and all(torch.equal(got, want) for got, want in pairs)
    quantizing = (exact - r0).abs().max()
    for run.mode in swap.modes(fp8)[2:]:
        assert (_logits(model, tokens) - exact).abs().max() < quantizing
    run.mode, run.scheme = 'fp32', swap.BF16
