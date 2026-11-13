"""
`run.py --smoke` on a tiny random Qwen3.5: every backend writes its record,
MMA-Sim's prediction and in-model sample agree with FPy's kernel, and the
report is written.  Needs a GPU.

    pytest -n 0 -q evaluation/gsm8k/tests/test_run.py
"""

import json
from pathlib import Path

import pytest
import run
import torch

D, E = 'nv.hopper.bf16.f32', 'nv.blackwell.e4m3.f32'
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')


def _records(out: Path, names: list[str]) -> dict[str, dict]:
    report = [f'gsm8k-{n}' for n in ('speed.tex', 'designs.tex', 'check.tex', 'speed.json', 'summary.txt')]
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


def test_smoke_layers(tiny: Path, tmp_path: Path):
    """Two designs of different schemes; MMA-Sim predicted per layer, one
    design also sampled in the model."""
    out = tmp_path / 'results'
    assert run.main([D, E, '--smoke', '--backends', 'fpy', 'mmasim', '--sampled', E, '--full',
                     '--model', str(tiny), '-o', str(out)]) == 0
    records = _records(out, [f'{d}-{b}' for d in (D, E) for b in ('fpy', 'mmasim')])
    for d in (D, E):
        f, m = records[f'{d}-fpy'], records[f'{d}-mmasim']
        _check_run(f)
        assert m['method'] == 'layers' and 'tokens' not in m and m['scheme'] == f['scheme']
        lay = m['layers']
        assert (m['prefill'], m['decode']) == (lay['prefill'], lay['decode'])
        assert lay['decode'] == pytest.approx(len(f['batches'][0]['steps']) * lay['step'])
        a, b = lay['fit']   # on the tiny model the line is noise: check it is the one used
        f = records[f'{d}-fpy']
        n = f['batches'][0]['shared']
        assert m['prefix'] == pytest.approx(a + b * n)
        assert m['prefill'] == pytest.approx(m['prefix'] + sum(a + b * (k - n) for k in f['prompt_tokens']))
        assert len(lay['passes']['questions']) == 2 and lay['passes']['step']
        assert m['check']['calls'] > 0 and m['check']['differing'] == 0
    assert records[f'{D}-fpy']['scheme'] == 'bf16' and records[f'{E}-fpy']['scheme'] == 'fp8-row'
    assert 'in_model' not in records[f'{D}-mmasim']
    s = records[f'{E}-mmasim']['in_model']
    assert 4 <= len(s['sample']['steps']) <= 16 and s['check']['differing'] == 0


def test_smoke_full(tiny: Path, tmp_path: Path):
    out = tmp_path / 'results'
    assert run.main(['--smoke', '--mmasim', 'full', D, '--mmasim-as', 'released', '--sampled',
                     '--model', str(tiny), '-o', str(out)]) == 0
    records = _records(out, ['torch', f'{D}-fpy', f'{D}-mmasim'])
    for r in records.values():
        _check_run(r)
    m = records[f'{D}-mmasim']
    assert m['method'] == 'full' and m['baseline'] == 'released'
    assert records[f'{D}-fpy']['tokens'] == m['tokens']
    assert m['layers']['decode'] > 0 and m['check']['differing'] == 0


def test_reuse(tiny: Path, tmp_path: Path):
    """A tuned run that takes FPy's record from an earlier one, as it was;
    Hopper's MMA-Sim run measured end to end, by default (`--full`)."""
    first, second = tmp_path / 'first', tmp_path / 'second'
    assert run.main([D, '--smoke', '--backends', 'fpy', '--model', str(tiny), '-o', str(first)]) == 0
    assert run.main([D, '--smoke', '--backends', 'fpy', 'mmasim', '--sampled', D, '--reuse', str(first),
                     '--model', str(tiny), '-o', str(second)]) == 0
    old, new = (json.loads((d / f'{D}-fpy.json').read_text()) for d in (first, second))
    assert new == old
    m = json.loads((second / f'{D}-mmasim.json').read_text())
    assert m['baseline'] == 'tuned' and m['method'] == 'full' and m['tokens'] == old['tokens']
    assert m['check']['differing'] == 0 and m['layers']['decode'] > 0
    assert m['in_model']['check']['differing'] == 0


def test_sampling_needs_fpy(tmp_path: Path):
    with pytest.raises(SystemExit):
        run.main(['--smoke', '--backends', 'mmasim', '-o', str(tmp_path / 'r')])


def test_refuses_foreign_files(tmp_path: Path):
    (tmp_path / 'notes.txt').write_text('mine')
    with pytest.raises(SystemExit):
        run.main(['--smoke', '-o', str(tmp_path)])
    assert (tmp_path / 'notes.txt').exists()
