"""
`run.py --smoke` on a tiny random Qwen3.5: every backend writes its record,
`fpy` and `mmasim` agree, and the report is written.  Needs a GPU.

    pytest -n 0 -q evaluation/gsm8k_speed/tests/test_run.py
"""

import json
from pathlib import Path

import pytest
import run
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')


@pytest.fixture(scope='module')
def tiny(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Qwen3.5-0.8B's tokenizer and layer pattern (3 linear-attention layers,
    then full attention), small and random; every `k` a multiple of 128."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(run.swap.MODEL)
    t = cfg.text_config
    t.update({'hidden_size': 128, 'intermediate_size': 256, 'num_hidden_layers': 4,
              'num_attention_heads': 2, 'num_key_value_heads': 1, 'head_dim': 64,
              'linear_num_key_heads': 2, 'linear_num_value_heads': 4,
              'linear_key_head_dim': 64, 'linear_value_head_dim': 64,
              'layer_types': t.layer_types[:4]})
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp('tiny')
    AutoModelForCausalLM.from_config(cfg).save_pretrained(path)
    AutoTokenizer.from_pretrained(run.swap.MODEL).save_pretrained(path)
    return path


def test_smoke(tiny: Path, tmp_path: Path):
    out = tmp_path / 'results'
    assert run.main(['--smoke', '--model', str(tiny), '-o', str(out)]) == 0
    design = 'nv.hopper.bf16.f32'
    names = ['torch', f'{design}-fpy', f'{design}-mmasim']
    report = ['gsm8k-speed.tex', 'gsm8k-speed.json', 'gsm8k-summary.txt']
    assert sorted(p.name for p in out.iterdir()) == sorted([f'{n}.json' for n in names] + report)
    records = {n: json.loads((out / f'{n}.json').read_text()) for n in names}
    for r in records.values():
        assert r['problems'] == 2 and len(r['tokens']) == 2 and r['decode'] > 0
        assert all(1 <= len(t) <= 8 for t in r['tokens'])
    assert records[f'{design}-fpy']['tokens'] == records[f'{design}-mmasim']['tokens']


def test_refuses_foreign_files(tmp_path: Path):
    (tmp_path / 'notes.txt').write_text('mine')
    with pytest.raises(SystemExit):
        run.main(['--smoke', '-o', str(tmp_path)])
    assert (tmp_path / 'notes.txt').exists()
