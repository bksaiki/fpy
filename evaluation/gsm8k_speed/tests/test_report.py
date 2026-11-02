"""
The report on hand-written records.

    pytest -n 8 -q evaluation/gsm8k_speed/tests/test_report.py
"""

import json
from pathlib import Path

import report

D = 'nv.hopper.bf16.f32'


def _record(backend: str, design: str | None, tokens: list[list[int]], answers: list[str],
            prefill: float, decode: float) -> dict:
    return {'backend': backend, 'design': design, 'problems': len(tokens), 'tokens': tokens,
            'answers': answers, 'targets': ['3', '5'], 'prefill': prefill, 'decode': decode}


RECORDS = [
    _record('mmasim', D, [[1, 2], [3, 4, 5]], ['3', '6'], 300.0, 900.0),
    _record('torch', None, [[1, 2], [3, 9]], ['3', '5'], 0.5, 1.5),
    _record('fpy', D, [[1, 2], [3, 4, 5]], ['3', '6'], 1.0, 3.0),
]


def test_rows():
    rs = report.rows(RECORDS)
    assert [(r.design, r.backend) for r in rs] == [(None, 'torch'), (D, 'fpy'), (D, 'mmasim')]
    torch, fpy, mmasim = rs
    assert (torch.tokens, torch.correct, torch.same, torch.speedup) == (4, 2, None, None)
    assert (fpy.tokens, fpy.correct, fpy.same, fpy.speedup) == (5, 1, 2, 300.0)
    assert fpy.rate == 5 / 4 and mmasim.same is None


def test_fpy_without_mmasim():
    (fpy,) = report.rows([RECORDS[2]])
    assert fpy.same is None and fpy.speedup is None


def test_main(tmp_path: Path):
    for i, r in enumerate(RECORDS):
        (tmp_path / f'{i}.json').write_text(json.dumps(r))
    assert report.main([str(tmp_path)]) == 0
    tex = (tmp_path / 'gsm8k-speed.tex').read_text()
    assert r'\texttt{nv.hopper.bf16.f32} & FPy & 2 & 5 & 1 & 3 & 1.25 & 50.0 & 2/2 & 300 \\' in tex
    assert len(json.loads((tmp_path / 'gsm8k-speed.json').read_text())) == 3
    assert report.main([str(tmp_path)]) == 0   # ignores its own outputs
