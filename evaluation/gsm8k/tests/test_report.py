"""
The report on hand-written records.

    pytest -n 8 -q evaluation/gsm8k/tests/test_report.py
"""

import json
from pathlib import Path

import report

D = 'nv.hopper.bf16.f32'


def _record(backend: str, design: str | None, tokens: list[list[int]], replies: list[str],
            prefill: float, decode: float) -> dict:
    return {'backend': backend, 'design': design, 'problems': len(tokens), 'tokens': tokens,
            'replies': replies, 'targets': ['3', '5'], 'prefill': prefill, 'decode': decode}


WRONG = ['The answer is 3.', 'The answer is 6.']
RECORDS = [
    _record('mmasim', D, [[1, 2], [3, 4, 5]], WRONG, 300.0, 900.0),
    _record('torch', None, [[1, 2], [3, 9]], ['The answer is 3.', 'So 5'], 0.5, 1.5),
    _record('fpy', D, [[1, 2], [3, 4, 5]], WRONG, 1.0, 3.0),
]


def test_rows():
    rs = report.rows(RECORDS)
    assert [(r.design, r.backend) for r in rs] == [(None, 'torch'), (D, 'fpy'), (D, 'mmasim')]
    torch, fpy, mmasim = rs
    assert (torch.tokens, torch.same, torch.speedup) == (4, None, None)
    assert torch.correct == {'strict-match': 1, 'flexible-extract': 2}
    assert (fpy.tokens, fpy.same, fpy.speedup) == (5, 2, 300.0)
    assert fpy.correct == {'strict-match': 1, 'flexible-extract': 1}
    assert fpy.rate == 5 / 4 and mmasim.same is None


def test_fpy_without_mmasim():
    (fpy,) = report.rows([RECORDS[2]])
    assert fpy.same is None and fpy.speedup is None


def test_main(tmp_path: Path):
    for i, r in enumerate(RECORDS):
        (tmp_path / f'{i}.json').write_text(json.dumps(r))
    assert report.main([str(tmp_path)]) == 0
    tex = (tmp_path / 'gsm8k-speed.tex').read_text()
    assert r'\texttt{nv.hopper.bf16.f32} & FPy & 2 & 5 & 1 & 3 & 1.25 & 50.0 & 50.0 & 2/2 & 300 \\' in tex
    assert len(json.loads((tmp_path / 'gsm8k-speed.json').read_text())) == 3
    assert report.main([str(tmp_path)]) == 0   # ignores its own outputs


SAMPLED = {'backend': 'mmasim', 'design': D, 'problems': 2, 'prefill': 30.0, 'decode': 90.0,
           'sampled': True, 'check': {'calls': 10, 'outputs': 100, 'differing': 0}}


def test_sampled():
    rs = report.rows([RECORDS[2], SAMPLED])
    fpy, mmasim = rs
    assert (fpy.same, fpy.check, fpy.speedup) == (None, SAMPLED['check'], 30.0)
    assert not fpy.disagrees
    assert mmasim.predicted and mmasim.tokens is None and mmasim.rate is None
    assert '*' in report.text(rs) and 'bit-exact' in report.text(rs)
    assert r'MMA-Sim (predicted) & 2 &  & 30 & 90 &  &  &  &  &  \\' in report.latex(rs)
    bad = SAMPLED | {'check': {'calls': 10, 'outputs': 100, 'differing': 3}}
    assert report.rows([RECORDS[2], bad])[0].disagrees
