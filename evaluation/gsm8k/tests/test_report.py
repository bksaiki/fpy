"""
The report on hand-written records.

    pytest -n 8 -q evaluation/gsm8k/tests/test_report.py
"""

import json
from pathlib import Path

import pytest
import report

D, E = 'nv.hopper.bf16.f32', 'nv.blackwell.e4m3.f32'
CHECK = {'calls': 10, 'outputs': 100, 'differing': 0, 'finite': 90}


def _record(backend: str, design: str | None, tokens: list[list[int]], replies: list[str],
            prefill: float, decode: float) -> dict:
    return {'backend': backend, 'design': design, 'scheme': design and 'bf16', 'problems': len(tokens),
            'tokens': tokens, 'replies': replies, 'targets': ['3', '5'], 'prefill': prefill,
            'decode': decode}


WRONG = ['The answer is 3.', 'The answer is 6.']
FULL = _record('mmasim', D, [[1, 2], [3, 4, 5]], WRONG, 300.0, 900.0) | {
    'method': 'full', 'check': CHECK, 'layers': {'prefill': 330.0, 'decode': 870.0},
    'in_model': {'prefill': 300.0, 'decode': 1000.0}}
RECORDS = [
    FULL,
    _record('torch', None, [[1, 2], [3, 9]], ['The answer is 3.', 'So 5'], 0.5, 1.5),
    _record('fpy', D, [[1, 2], [3, 4, 5]], WRONG, 1.0, 3.0),
]
PREDICTED = [
    _record('fpy', E, [[1, 2], [3, 4, 5]], WRONG, 2.0, 2.0) | {'scheme': 'fp8-row'},
    {'backend': 'mmasim', 'design': E, 'scheme': 'fp8-row', 'problems': 2, 'method': 'layers',
     'baseline': 'tuned', 'prefill': 40.0, 'decode': 160.0, 'check': CHECK | {'differing': 3},
     'layers': {'prefill': 40.0, 'decode': 160.0}},
]


def test_rows():
    rs = report.rows(RECORDS)
    assert [(r.design, r.backend) for r in rs] == [(None, 'torch'), (D, 'fpy'), (D, 'mmasim')]
    torch, fpy, mmasim = rs
    assert (torch.tokens, torch.same, torch.speedup) == (4, None, None)
    assert torch.correct == {'strict-match': 1, 'flexible-extract': 2}
    assert (fpy.tokens, fpy.same, fpy.check, fpy.speedup) == (5, 2, CHECK, 300.0)
    assert fpy.correct == {'strict-match': 1, 'flexible-extract': 1}
    assert fpy.rate == 5 / 4 and mmasim.same is None and not fpy.disagrees


def test_fpy_without_mmasim():
    (fpy,) = report.rows([RECORDS[2]])
    assert fpy.same is None and fpy.speedup is None


def test_predicted_rows_leave_the_main_table():
    rs = report.rows(RECORDS + PREDICTED)
    fpy, mmasim = [r for r in rs if r.design == E]
    assert mmasim.predicted and mmasim.tokens is None and mmasim.rate is None
    assert fpy.speedup == 50.0 and fpy.disagrees   # 3 outputs differ
    assert {r.design for r in report.measured(rs)} == {None, D}


def test_designs_and_summary():
    ds = report.designs(RECORDS + PREDICTED)
    assert [(d.design, d.measured, d.speedup) for d in ds] == [(E, False, 50.0), (D, True, 300.0)]
    s = report.summary(ds)
    assert (s['designs'], s['min'], s['max'], s['differing']) == (2, (50.0, E), (300.0, D), 3)
    assert s['geomean'] == pytest.approx((50.0 * 300.0) ** 0.5)


def test_checks():
    cs = report.checks(RECORDS + PREDICTED)
    assert [(c.design, c.against) for c in cs] == [(D, 'full run'), (D, 'in-model sample')]
    assert cs[0].errors == pytest.approx((0.1, -1 / 30, 0.0))
    assert cs[1].errors == pytest.approx((0.1, -0.13, 1200 / 1300 - 1))


def test_main(tmp_path: Path):
    for i, r in enumerate(RECORDS + PREDICTED):
        (tmp_path / f'{i}.json').write_text(json.dumps(r))
    assert report.main([str(tmp_path)]) == 0
    tex = (tmp_path / 'gsm8k-speed.tex').read_text()
    assert r'\texttt{nv.hopper.bf16.f32} & FPy & 2 & 5 & 1 & 3 & 1.25 & 50.0 & 50.0 & 2/2 & 300 \\' in tex
    assert E not in tex
    designs = (tmp_path / 'gsm8k-designs.tex').read_text()
    assert r'\texttt{nv.blackwell.e4m3.f32} & fp8-row & 2 & 2 & 4 & 40* & 160* & 200* & 50 & 3/100 & 90.0 \\' in designs
    assert r'\texttt{nv.hopper.bf16.f32} & full run & 1,200 & 1,200 & +10.0 & -3.3 & +0.0 \\' in \
        (tmp_path / 'gsm8k-check.tex').read_text()
    nums = json.loads((tmp_path / 'gsm8k-speed.json').read_text())
    assert (len(nums['main']), len(nums['designs']), len(nums['check'])) == (3, 2, 2)
    assert 'geomean 122' in (tmp_path / 'gsm8k-summary.txt').read_text()
    assert report.main([str(tmp_path)]) == 0   # ignores its own outputs
