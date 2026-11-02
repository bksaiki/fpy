"""
The report from hand-written results.

    pytest -n 8 evaluation/analysis_precision/tests/test_report.py
"""

import json
import math

from report import VENDORS, Row, load, main, slacks, table, tightness


def _p(digits, lsb, msb, mag):
    return {'digits': digits, 'lsb': lsb, 'msb': msb, 'mag': mag}


def _result(top, static=None, seen=None, executed=True):
    return {'top': top, 'executed': executed, 'sound': True,
            'static': None if top else static, 'observed': seen if executed else None}


def _site(name, fmt, db):
    return {'calls': [], 'loc': f'x.py:{name}', 'name': name, 'format': fmt, 'digit_bound': db}


_FP32 = _p(24, -149, 127, 128.0)
_TS = _p(24, -math.inf, 9, 10.0)
_VOLTA = {'design': 'nv.volta.f16.f32', 'n': 1, 'inputs': 'mixed', 'sites': [
    _site('a', _result(False, _FP32, _FP32), _result(False, _FP32, _FP32)),
    _site('ts', _result(True), _result(False, _TS, _p(23, 0, 0, 0.0))),
    _site('s', _result(True), _result(True)),
]}
_FMA = {'design': 'fp32 (fma)', 'n': 1, 'inputs': 'mixed', 'sites': [
    _site('p', _result(False, _FP32, executed=False), _result(False, _FP32, executed=False)),
]}


def test_rows_by_architecture():
    assert table([_FMA, _VOLTA]) == [
        Row('Volta', 1, 3, 2, 1), Row('FMA chain', 1, 1, 0, 0), Row('Total', 2, 4, 2, 1),
    ]


def test_slacks_skip_top_and_unrun_sites():
    assert slacks([_VOLTA, _FMA]) == {'digits': [0, 1], 'lsb': [0, math.inf], 'mag': [0.0, 10.0]}


def test_tightness_is_observed_over_allowed():
    # an unbounded quantum covers none of it
    volta = (2, {'digits': (1 + 23 / 24) / 2, 'lsb': 0.5, 'mag': (1 + 2 ** -10) / 2})
    assert tightness([_VOLTA, _FMA]) == [('Volta', *volta), ('Total', *volta)]


def test_writes_tables_and_figure(tmp_path):
    for i, rec in enumerate([_VOLTA, _FMA, {'design': 'amd.cdna1.f16', 'error': 'boom'}]):
        (tmp_path / f'{i}.json').write_text(json.dumps(rec))
    assert main([str(tmp_path)]) == 0
    assert main([str(tmp_path)]) == 0   # again, over its own outputs
    tex = (tmp_path / 'precision-tops.tex').read_text()
    assert 'Volta & 1 & 3 & 2 & 1 \\\\' in tex and 'Total & 2 & 4 & 2 & 1 \\\\' in tex
    assert 'Volta & 2 & 98\\% & 50\\% & 50\\%' in (tmp_path / 'precision-tightness.tex').read_text()
    assert (tmp_path / 'precision-slack.pdf').stat().st_size > 0
    text = (tmp_path / 'precision-summary.txt').read_text()
    assert 'R* without digit bounds 2, with 1' in text and 'failed, left out: amd.cdna1.f16' in text


def test_vendor_rows_pool_their_architectures():
    amd = {'design': 'amd.cdna2.f16', 'n': 1, 'sites': [
        _site('z', _result(False, _FP32, _FP32), _result(False, _FP32, _p(12, -149, 127, 128.0))),
    ]}
    assert [r.label for r in table([_VOLTA, _FMA, amd], VENDORS)] == ['NVIDIA', 'AMD', 'FMA chain', 'Total']
    rows = {label: (n, means) for label, n, means in tightness([_VOLTA, amd], VENDORS)}
    assert rows['NVIDIA'] == tightness([_VOLTA])[0][1:]
    assert rows['Total'][0] == 3 and rows['AMD'][1]['digits'] == 0.5


def test_list_allocations_are_not_sites(tmp_path):
    alloc = _site('zs', _result(False), _result(False))   # bounded by the empty set
    rec = {**_VOLTA, 'sites': [*_VOLTA['sites'], alloc]}
    (tmp_path / 'volta.json').write_text(json.dumps(rec))
    (loaded,), _ = load(tmp_path)
    assert [s['name'] for s in loaded['sites']] == ['a', 'ts', 's']
