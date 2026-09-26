"""A small random Qwen3 for the serve tests; needs a GPU and `transformers`."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope='module')
def model() -> torch.nn.Module:
    """Qwen3's shape, small: every `k` a multiple of each design's length,
    weights BF16 values held in FP32, as a checkpoint loads."""
    transformers = pytest.importorskip('transformers')
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
def tokens() -> torch.Tensor:
    return torch.randint(0, 96, (1, 16), generator=torch.Generator().manual_seed(1)).cuda()
