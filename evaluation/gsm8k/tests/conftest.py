"""Puts the evaluation's modules and the MMA serving harness on the import path;
a tiny random Qwen3.5 for the GPU tests."""

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT.parent / 'common' / 'mma'),
                str(_ROOT.parents[1] / 'examples' / 'mmasim')]
sys.path.append(str(_ROOT.parent))  # last: `config`, not shadowing this `run`

import config

MODEL = config.MODEL


@pytest.fixture(scope='session')
def tiny(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Qwen3.5-0.8B's tokenizer and layer pattern (3 linear-attention layers,
    then full attention), small and random; every `k` a multiple of 128."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(MODEL)
    t = cfg.text_config
    t.update({'hidden_size': 128, 'intermediate_size': 256, 'num_hidden_layers': 4,
              'num_attention_heads': 2, 'num_key_value_heads': 1, 'head_dim': 64,
              'linear_num_key_heads': 2, 'linear_num_value_heads': 4,
              'linear_key_head_dim': 64, 'linear_value_head_dim': 64,
              'layer_types': t.layer_types[:4]})
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp('tiny')
    AutoModelForCausalLM.from_config(cfg).save_pretrained(path)
    AutoTokenizer.from_pretrained(MODEL).save_pretrained(path)
    return path
