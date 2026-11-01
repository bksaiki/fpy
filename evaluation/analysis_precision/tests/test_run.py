"""
The driver on one design: a record per site, both analyses, nothing unsound.

    pytest -n 8 evaluation/analysis_precision/tests/test_run.py
"""

import json

from run import Analysis, main


def test_one_design(tmp_path):
    assert main(['-o', str(tmp_path), '-n', '8', 'nv.volta.f16.f32']) == 0
    record = json.loads((tmp_path / 'nv.volta.f16.f32.json').read_text())
    assert record['design'] == 'nv.volta.f16.f32' and record['n'] == 8
    sites = record['sites']
    assert sites and all(s['loc'] and s['name'] for s in sites)
    results = [s[a.value] for s in sites for a in Analysis]
    assert all(r is not None and r['sound'] for r in results)
    # the truncated terms and their sum: top only without digit bounds
    tops = {a: {s['name'] for s in sites if s[a.value]['top']} for a in Analysis}
    assert tops == {Analysis.FORMAT: {'ts', 's'}, Analysis.DIGIT_BOUND: set()}


def test_one_design_in_anf(tmp_path):
    assert main(['-o', str(tmp_path / 'plain'), '-n', '8', 'nv.volta.f16.f32']) == 0
    assert main(['-o', str(tmp_path / 'anf'), '-n', '8', '--anf', 'nv.volta.f16.f32']) == 0
    plain, anf = (json.loads((tmp_path / d / 'nv.volta.f16.f32.json').read_text())['sites']
                  for d in ('plain', 'anf'))
    assert len(anf) > len(plain)
    assert all(s[a.value]['sound'] for s in anf for a in Analysis)
