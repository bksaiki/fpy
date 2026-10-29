"""A small random Qwen3 for the serve tests; needs a GPU and `transformers`."""

import sys
from pathlib import Path

import pytest
import torch

# `serve`, for `core`; `examples/mmasim`, for `generator`
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]


@pytest.fixture(scope='module')
def model() -> torch.nn.Module:
    """Qwen3's shape, small: every `k` a multiple of 128 and so of each
    design's length, weights BF16 values held in FP32, as `swap.load` loads
    them."""
    transformers = pytest.importorskip('transformers')
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(
        vocab_size=96, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=64,
        tie_word_embeddings=True)
    m = transformers.Qwen3ForCausalLM(cfg).float().cuda().eval()
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(p.to(torch.bfloat16).float())
    return m


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """`kernels`' registries as they were, after the test."""
    from core import kernels

    for name in ('TILES', '_BUILDS', 'REGISTERED', '_PICKLED', '_KERNELS', '_SECONDS'):
        monkeypatch.setattr(kernels, name, dict(getattr(kernels, name)))
    monkeypatch.setattr(kernels, '_LAUNCHED', set(kernels._LAUNCHED))


@pytest.fixture(scope='module')
def tokens() -> torch.Tensor:
    return torch.randint(0, 96, (1, 16), generator=torch.Generator().manual_seed(1)).cuda()
