"""
`lmeval.py` on a tiny random Qwen3.5: lm-eval's records, and their
comparison with `run.py`'s.  Needs a GPU.

    pytest -n 0 -q evaluation/gsm8k/tests/test_lmeval.py
"""

import json
from pathlib import Path

import gsm8k
import lmeval
import pytest
import run
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')


def test_lmeval(tiny: Path, tmp_path: Path):
    ours, out = tmp_path / 'ours', tmp_path / 'lmeval'
    assert run.main(['--smoke', '--model', str(tiny), '-o', str(ours), '--backends', 'torch', 'fpy']) == 0
    assert lmeval.main(['--model', str(tiny), '--limit', '2', '-o', str(out),
                        '--against', str(ours)]) == 0
    for b in ('torch', 'fpy'):
        r = json.loads((out / f'{b}.json').read_text())
        assert [s['doc_id'] for s in r['samples']] == [0, 1]
        assert all(s['answers'].keys() == gsm8k.METRICS.keys() for s in r['samples'])
    cmp = json.loads((out / 'lmeval-compare.json').read_text())
    assert cmp.keys() == {'torch', 'fpy'}
    for c in cmp.values():
        assert c['problems'] == 2 and c['same_verdict'] == dict.fromkeys(gsm8k.METRICS, 2)
