"""
The driver on one design: a record per site, both analyses, nothing unsound.

    pytest -n 8 evaluation/analysis_precision/tests/test_run.py
"""

import json

from run import Analysis, main


def test_one_design(tmp_path):
    assert main(['-o', str(tmp_path), '-n', '8', '-j', '1', 'nv.volta.f16.f32']) == 0
    record = json.loads((tmp_path / 'nv.volta.f16.f32.json').read_text())
    assert record['design'] == 'nv.volta.f16.f32' and record['n'] == 8
    sites = record['sites']
    assert sites and all(s['loc'] and s['name'] for s in sites)
    results = [s[a.value] for s in sites for a in Analysis]
    assert all(r is not None and r['sound'] for r in results)
    # the truncated terms and their sum: top only without digit bounds
    tops = {a: {s['name'] for s in sites if s[a.value]['top']} for a in Analysis}
    assert tops == {Analysis.FORMAT: {'ts', 's'}, Analysis.DIGIT_BOUND: set()}


def test_one_design_pushes_through_to_the_report(tmp_path):
    out = tmp_path / 'results'
    out.mkdir()
    (out / 'stale.json').write_text('{}')   # replaced, not merged
    assert main(['-o', str(out), '-n', '10', '-j', '1', 'nv.volta.f16.f32']) == 0
    assert not (out / 'stale.json').exists()
    plain = json.loads((out / 'nv.volta.f16.f32.json').read_text())['sites']
    anf = json.loads((out / 'anf' / 'nv.volta.f16.f32.json').read_text())['sites']
    assert len(anf) > len(plain)
    # every fifth input adversarial: the random ones are observed apart
    assert any(s['digit_bound']['observed_random'] for s in plain)
    text = (out / 'precision-summary.txt').read_text()
    assert 'random inputs alone' in text and 'anf:' in text
    assert (out / 'precision-tightness-vendor.tex').exists()
    paper = json.loads((out / 'precision-paper.json').read_text())
    assert paper['run'] == {**paper['run'], 'designs': 1, 'inputs_per_design': 10,
                            'random_inputs_per_design': 8, 'unsound': 0}
    assert [r['label'] for r in paper['tops']['by_vendor']] == ['NVIDIA', 'Total']
    assert {'precision', 'min_quantum', 'bound'} <= paper['tightness']['random_inputs_alone'].keys()
