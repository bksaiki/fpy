"""
The driver on one small design, under every backend, through to the report.

    pytest -n 8 evaluation/solver_backends/tests
"""

import importlib.util
import json

import pytest
from run import compile_once, main

_MISSING = [p for p in ('cvc5', 'highspy') if importlib.util.find_spec(p) is None]


def test_one_compilation_records_its_solver():
    rec = compile_once('nv.volta.f16.f32', 'z3')
    assert 'cpp_sha256' in rec, rec.get('error')
    st = rec['stats']
    assert st['queries'] > 0 and st['constraints'] > 0 and st['calls'] >= st['queries']
    assert 0 < st['solver_time'] <= rec['seconds']


@pytest.mark.skipif(bool(_MISSING), reason=f'not installed: {_MISSING}')
def test_one_design_under_every_backend(tmp_path):
    out = tmp_path / 'results'
    out.mkdir()
    (out / 'stale.json').write_text('{}')   # replaced, not merged
    assert main(['-o', str(out), '-r', '1', '-j', '4', 'fp32 (fma)']) == 0
    assert not (out / 'stale.json').exists()
    record = json.loads((out / 'fp32_fma.json').read_text())
    assert set(record['backends']) == {'z3', 'z3-opt', 'cvc5', 'highs'}
    assert all(r['outcome'] == 'ok' for r in record['backends'].values())
    hashes = {r['cpp_sha256'] for r in record['backends'].values()}
    assert len(hashes) == 1
    # the table and the figures are all the report writes
    made = {p.name for p in out.iterdir() if not p.name.endswith('.json')}
    assert made == {f'backends-scatter{v}.{e}' for v in ('', '-all', '-linear', '-all-linear')
                    for e in ('pdf', 'png')} | {'backends-totals.tex'}


def test_a_foreign_file_stops_the_run(tmp_path):
    (tmp_path / 'notes.md').write_text('mine')
    with pytest.raises(SystemExit):
        main(['-o', str(tmp_path), '-r', '1', 'fp32 (fma)'])
    assert (tmp_path / 'notes.md').exists()
