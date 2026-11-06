"""
The driver on one small design, under every backend, through to the report.

    pytest -n 8 evaluation/solver_backends/tests
"""

import importlib.util
import json

import pytest
from run import _summarize, compile_once, main

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
    assert main(['-o', str(out), '-r', '1', '-j', '4', '-c', '30', 'fp32 (fma)']) == 0
    record = json.loads((out / 'fp32_fma.json').read_text())
    assert set(record['backends']) == {'z3', 'z3-opt', 'cvc5', 'highs'}
    assert all(r['outcome'] == 'ok' for r in record['backends'].values())
    hashes = {r['cpp_sha256'] for r in record['backends'].values()}
    assert len(hashes) == 1
    # the median run's queries are kept, the others' dropped
    for r in record['backends'].values():
        assert len(r['trace']['seconds']) == r['stats']['queries']
        assert all('trace' not in run for run in r['runs'])
    assert json.loads((out / 'backends-run.json').read_text())['call_timeout'] == 30
    # the run's data at the top, the report apart in `report/`
    assert {p.name for p in out.iterdir() if not p.name.endswith('.json')} == {'report'}
    made = {p.name for p in (out / 'report').iterdir()}
    assert made == {'sensitivity-scatter.pdf', 'sensitivity-scatter.png',
                    'backends-totals.tex'}


def test_the_time_and_the_counts_come_from_one_run():
    def run(seconds, solver, constraints):
        return {'seconds': seconds, 'cpp_sha256': 'x', 'trace': {'seconds': [solver]},
                'stats': {'solver_time': solver, 'constraints': constraints}}
    s = _summarize([run(58, 43, 61144), run(51, 39, 54868), run(38, 28, 48592)])
    assert (s['seconds'], s['solver_time'], s['stats']['constraints']) == (51, 39, 54868)
    assert s['trace'] == {'seconds': [39]}
    # a run that did not compile is not a candidate
    s = _summarize([{'timeout': 600}, run(5, 1, 10), run(7, 2, 12)])
    assert s['seconds'] == 5


def test_each_run_gets_a_new_folder(tmp_path, monkeypatch):
    import run
    monkeypatch.setattr(run, 'RESULTS', tmp_path)
    (tmp_path / 'earlier').mkdir()
    (tmp_path / 'earlier' / 'a.json').write_text('{}')
    assert main(['-b', 'z3', '-r', '1', 'fp32 (fma)']) == 0
    new = [p for p in tmp_path.iterdir() if p.name != 'earlier']
    assert len(new) == 1 and (new[0] / 'fp32_fma.json').exists()
    assert (tmp_path / 'earlier' / 'a.json').exists()   # nothing deleted


def test_a_folder_with_files_is_not_written_into(tmp_path):
    (tmp_path / 'notes.md').write_text('mine')
    with pytest.raises(SystemExit):
        main(['-o', str(tmp_path), '-r', '1', 'fp32 (fma)'])
    assert [p.name for p in tmp_path.iterdir()] == ['notes.md']
