"""
The report from hand-written results.

    pytest -n 8 evaluation/analysis_precision/tests/test_report.py
"""

import json
import math

from report import Row, coverage, load, main, slacks, table


def _p(digits, lsb, msb, mag):
    return {'digits': digits, 'lsb': lsb, 'msb': msb, 'mag': mag}


def _result(top, static=None, seen=None, executed=True):
    return {'top': top, 'executed': executed, 'sound': True,
            'static': None if top else static, 'observed': seen if executed else None}


def _site(name, fmt, db):
    return {'calls': [], 'loc': f'x.py:{name}', 'name': name, 'format': fmt, 'digit_bound': db}


_FP32 = _p(24, -149, 127, 128.0)
_TS = _p(24, -math.inf, 9, 10.0)
_VOLTA = {'design': 'nv.volta.f16.f32', 'n': 1, 'sites': [
    _site('a', _result(False, _FP32, _FP32), _result(False, _FP32, _FP32)),
    _site('ts', _result(True), _result(False, _TS, _p(23, 0, 0, 0.0))),
    _site('s', _result(True), _result(True)),
]}
_FMA = {'design': 'fp32 (fma)', 'n': 1, 'sites': [
    _site('p', _result(False, _FP32, executed=False), _result(False, _FP32, executed=False)),
]}


def test_rows_by_architecture():
    assert table([_FMA, _VOLTA]) == [
        Row('Volta', 1, 3, 2, 1), Row('FMA chain', 1, 1, 0, 0), Row('Total', 2, 4, 2, 1),
    ]


def test_slacks_skip_top_and_unrun_sites():
    assert slacks([_VOLTA, _FMA]) == {'digits': [0, 1], 'lsb': [0, math.inf], 'mag': [0.0, 10.0]}


def test_coverage_is_observed_over_allowed():
    # an unbounded quantum covers none of it
    volta = (2, {'digits': (1 + 23 / 24) / 2, 'lsb': 0.5, 'mag': (1 + 2 ** -10) / 2})
    assert coverage([_VOLTA, _FMA]) == [('Volta', *volta), ('Total', *volta)]


def test_writes_tables_and_figure(tmp_path):
    for i, rec in enumerate([_VOLTA, _FMA, {'design': 'amd.cdna1.f16', 'error': 'boom'}]):
        (tmp_path / f'{i}.json').write_text(json.dumps(rec))
    assert main([str(tmp_path)]) == 0
    tex = (tmp_path / 'precision-tops.tex').read_text()
    assert 'Volta & 1 & 3 & 2 & 1 \\\\' in tex and 'Total & 2 & 4 & 2 & 1 \\\\' in tex
    assert 'Volta & 2 & 98\\% & 50\\% & 50\\%' in (tmp_path / 'precision-coverage.tex').read_text()
    assert (tmp_path / 'precision-slack.pdf').stat().st_size > 0


def test_runs_combine_their_observations(tmp_path):
    static = _p(24, -20, 9, 10.0)
    a = {'design': 'nv.volta.f16.f32', 'n': 2, 'sites': [
        _site('t', _result(False, static, _p(23, -5, 3, 3.5)), _result(False, static, _p(23, -5, 3, 3.5))),
        _site('u', _result(False, static, executed=False), _result(False, static, executed=False)),
    ]}
    b = {'design': 'nv.volta.f16.f32', 'n': 3, 'sites': [
        _site('t', _result(False, static, _p(22, -9, 4, 3.9)), _result(False, static, _p(22, -9, 4, 3.9))),
        _site('u', _result(False, static, _p(1, 0, 0, 0.0)), _result(False, static, _p(1, 0, 0, 0.0))),
    ]}
    for d, rec in (('a', a), ('b', b)):
        (tmp_path / d).mkdir()
        (tmp_path / d / 'volta.json').write_text(json.dumps(rec))
    (rec,), failed = load([tmp_path / 'a', tmp_path / 'b'])
    assert not failed and rec['n'] == 5
    t, u = sorted(rec['sites'], key=lambda s: s['name'])
    assert t['digit_bound']['observed'] == _p(23, -9, 4, 3.9)
    assert u['digit_bound']['executed'] and u['digit_bound']['observed'] == _p(1, 0, 0, 0.0)
