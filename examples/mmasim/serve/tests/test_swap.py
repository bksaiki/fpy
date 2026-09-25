"""
`swap.patch` on a small random Qwen3.  Needs a GPU and `transformers`;
skipped without either.

    pytest serve/tests
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fpy2.backend.triton import unavailable

_WHY = unavailable()
transformers = pytest.importorskip('transformers')
pytestmark = pytest.mark.skipif(_WHY is not None, reason=_WHY or '')

import kernels
import swap


@pytest.fixture(scope='module')
def model():
    """Qwen3's shape, small: every `k` a multiple of each design's length,
    weights BF16 values held in FP32, as a checkpoint loads."""
    import torch
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(
        vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        tie_word_embeddings=True)
    m = transformers.Qwen3ForCausalLM(cfg).float().cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(p.to(torch.bfloat16).float())
    return m


@pytest.fixture(scope='module')
def tokens():
    import torch
    return torch.randint(0, 96, (1, 16), generator=torch.Generator().manual_seed(1)).cuda()


def _logits(model, tokens):
    import torch
    with torch.no_grad():
        return model(tokens).logits


def test_each_run_computes_as_it_says(model, tokens, monkeypatch) -> None:
    """`fp32` is the unpatched model, bit for bit; a design routes every
    linear layer, `lm_head` included, through its kernel and lands near
    `bf16-exact`; and switching back restores `fp32`."""
    import torch
    before = _logits(model, tokens)
    run = swap.patch(model)
    assert torch.equal(_logits(model, tokens), before)

    run.mode = 'bf16-exact'
    exact = _logits(model, tokens)
    calls = []
    real = kernels.linear
    monkeypatch.setattr(kernels, 'linear', lambda *a, **kw: calls.append(1) or real(*a, **kw))
    run.mode = 'nv.hopper.bf16.f32'
    design = _logits(model, tokens)
    assert len(calls) == sum(isinstance(m, torch.nn.Linear) for m in model.modules())
    assert not torch.equal(exact, before)
    assert torch.allclose(design, exact, rtol=0, atol=1e-3 * exact.abs().max().item())

    run.mode = 'fp32'
    assert torch.equal(_logits(model, tokens), before)
