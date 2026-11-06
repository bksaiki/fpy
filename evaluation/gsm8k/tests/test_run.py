"""
`run.py --smoke` on a tiny random Qwen3.5: every backend writes its record,
`fpy` and `mmasim` agree, MMA-Sim's
sample predicts and checks, and the report is written.  Needs a GPU.

    pytest -n 0 -q evaluation/gsm8k/tests/test_run.py
"""

import json
from pathlib import Path

import pytest
import run
import torch

D = 'nv.hopper.bf16.f32'
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')


def _records(out: Path, names: list[str]) -> dict[str, dict]:
    report = ['gsm8k-speed.tex', 'gsm8k-speed.json', 'gsm8k-summary.txt']
    assert sorted(p.name for p in out.iterdir()) == sorted([f'{n}.json' for n in names] + report)
    return {n: json.loads((out / f'{n}.json').read_text()) for n in names}


def _check_run(r: dict):
    assert r['problems'] == 2 and len(r['tokens']) == 2 and r['decode'] > 0
    assert all(1 <= len(t) <= 8 for t in r['tokens'])
    (b,) = r['batches']
    assert len(b['questions']) == 2 and len(b['steps']) == len(b['running'])
    assert max(map(len, r['tokens'])) == len(b['steps']) + 1
    assert b['prefix'] + sum(b['questions']) <= b['prefill']
    assert sum(b['steps']) <= b['decode']
    assert r['commit'] and r['versions']['mmasim']


def test_smoke_full(tiny: Path, tmp_path: Path):
    out = tmp_path / 'results'
    assert run.main(['--smoke', '--mmasim', 'full', '--model', str(tiny), '-o', str(out)]) == 0
    records = _records(out, ['torch', f'{D}-fpy', f'{D}-mmasim'])
    for r in records.values():
        _check_run(r)
    assert records[f'{D}-fpy']['tokens'] == records[f'{D}-mmasim']['tokens']


def test_smoke_sampled(tiny: Path, tmp_path: Path):
    out = tmp_path / 'results'
    assert run.main(['--smoke', '--backends', 'fpy', 'mmasim', '--model', str(tiny), '-o', str(out)]) == 0
    records = _records(out, [f'{D}-fpy', f'{D}-mmasim'])
    _check_run(records[f'{D}-fpy'])
    m = records[f'{D}-mmasim']
    assert m['sampled'] and 'tokens' not in m
    s = m['sample']
    assert s['questions'] == [0, 1] and len(s['question_seconds']) == 2
    assert 4 <= len(s['steps']) <= 16
    assert m['check']['calls'] > 0 and m['check']['differing'] == 0
    steps = len(records[f'{D}-fpy']['batches'][0]['steps'])
    assert m['decode'] == pytest.approx(steps * s['step'])
    assert m['prefill'] > m['prefix'] > 0


def test_sampling_needs_fpy(tmp_path: Path):
    with pytest.raises(SystemExit):
        run.main(['--smoke', '--backends', 'mmasim', '-o', str(tmp_path / 'r')])


def test_refuses_foreign_files(tmp_path: Path):
    (tmp_path / 'notes.txt').write_text('mine')
    with pytest.raises(SystemExit):
        run.main(['--smoke', '-o', str(tmp_path)])
    assert (tmp_path / 'notes.txt').exists()
