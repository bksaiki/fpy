"""
The report from hand-written results.

    pytest -n 8 evaluation/analysis_precision/tests/test_report.py
"""

import json
import math

from report import Row, main, slacks, table


def _result(top, executed=True, digits=0, lsb=0, msb=0):
    return {'top': top, 'executed': executed, 'sound': True,
            'digits': None if top else digits, 'lsb': None if top else lsb,
            'msb': None if top else msb}


def _site(name, fmt, db):
    return {'calls': [], 'loc': f'x.py:{name}', 'name': name, 'format': fmt, 'digit_bound': db}


_VOLTA = {'design': 'nv.volta.f16.f32', 'sites': [
    _site('a', _result(False), _result(False)),
    _site('ts', _result(True), _result(False, digits=1, lsb=math.inf, msb=9)),
    _site('s', _result(True), _result(True)),
]}
_FMA = {'design': 'fp32 (fma)', 'sites': [
    _site('p', _result(False, executed=False), _result(False, executed=False)),
]}


def test_rows_by_architecture():
    assert table([_FMA, _VOLTA]) == [
        Row('Volta', 1, 3, 2, 1), Row('FMA chain', 1, 1, 0, 0), Row('Total', 2, 4, 2, 1),
    ]


def test_slacks_skip_top_and_unrun_sites():
    assert slacks([_VOLTA, _FMA]) == {'digits': [0, 1], 'lsb': [0, math.inf], 'msb': [0, 9]}


def test_writes_table_and_figure(tmp_path):
    for i, rec in enumerate([_VOLTA, _FMA, {'design': 'amd.cdna1.f16', 'error': 'boom'}]):
        (tmp_path / f'{i}.json').write_text(json.dumps(rec))
    assert main([str(tmp_path)]) == 0
    tex = (tmp_path / 'precision-tops.tex').read_text()
    assert 'Volta & 1 & 3 & 2 & 1 \\\\' in tex and 'Total & 2 & 4 & 2 & 1 \\\\' in tex
    assert (tmp_path / 'precision-slack.pdf').stat().st_size > 0
