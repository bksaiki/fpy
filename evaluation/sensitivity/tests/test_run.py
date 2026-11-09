"""
The driver on one small design, under every backend, through to the report.

    pytest -n 8 evaluation/sensitivity/tests
"""

import json

import pytest
from run import _summarize, compile_once, main

def test_one_compilation_records_its_solver():
    rec = compile_once('nv.volta.f16.f32', 'z3')
    assert 'cpp_sha256' in rec, rec.get('error')
    st = rec['stats']
    assert st['queries'] > 0 and st['constraints'] > 0
    assert 0 < st['solver_time'] <= rec['seconds']


def test_one_design_under_every_backend(tmp_path):
    out = tmp_path / 'results'
    assert main(['-o', str(out), '-r', '1', '-j', '4', '-c', '30', 'fp32 (fma)']) == 0
    record = json.loads((out / 'fp32_fma.json').read_text())
    assert set(record['backends']) == {'z3', 'z3-opt', 'cvc5'}
    assert all(r['outcome'] == 'ok' for r in record['backends'].values())
    hashes = {r['cpp_sha256'] for r in record['backends'].values()}
    assert len(hashes) == 1
    # the median run's queries are kept, the others' dropped
    for r in record['backends'].values():
        assert len(r['trace']['seconds']) == r['stats']['queries']
        assert all('trace' not in run for run in r['runs'])
    settings = json.loads((out / 'sensitivity-run.json').read_text())
    assert settings['call_timeout'] == 30
    assert settings['bisects'] == {'z3': True, 'z3-opt': False, 'cvc5': True}
    # the run's data, and beside it the report
    assert {p.name for p in out.iterdir() if p.name.startswith('sensitivity-')} == {
        'sensitivity-run.json', 'sensitivity-scatter.pdf', 'sensitivity-scatter.png',
        'sensitivity-table.tex', 'sensitivity-paper.json', 'sensitivity-summary.txt'}


def test_the_time_and_the_counts_come_from_one_run():
    def run(seconds, solver, constraints):
        return {'seconds': seconds, 'cpp_sha256': 'x', 'trace': {'seconds': [solver]},
                'stats': {'solver_time': solver, 'constraints': constraints}}
    s = _summarize([run(58, 43, 61144), run(51, 39, 54868), run(38, 28, 48592)])
    assert (s['seconds'], s['solver_time'], s['stats']['constraints']) == (51, 39, 54868)
    assert s['trace'] == {'seconds': [39]}


def test_a_design_compiles_only_where_every_repeat_did():
    ok = {'seconds': 5, 'cpp_sha256': 'x', 'trace': {}, 'stats': {'solver_time': 1}}
    s = _summarize([{'timeout': 600}, ok, ok])
    assert (s['outcome'], s['compiled']) == ('timeout', 2)
    assert 'seconds' not in s and 'cpp_sha256' not in s
    # every way they failed, whichever came first
    for runs in ([{'timeout': 600}, {'error': 'e'}], [{'error': 'e'}, {'timeout': 600}]):
        s = _summarize(runs)
        assert (s['outcome'], s['compiled']) == ('error+timeout', 0)
    assert _summarize([{'error': 'e'}, {'error': 'e'}])['outcome'] == 'error'


def test_a_run_replaces_the_last(tmp_path):
    (tmp_path / 'stale.json').write_text('{}')
    (tmp_path / 'sensitivity-table.tex').write_text('')
    assert main(['-o', str(tmp_path), '-b', 'z3', '-r', '1', 'fp32 (fma)']) == 0
    assert not (tmp_path / 'stale.json').exists()
    assert (tmp_path / 'fp32_fma.json').exists()


def test_files_a_run_did_not_write_are_kept(tmp_path):
    (tmp_path / 'notes.md').write_text('mine')
    with pytest.raises(SystemExit):
        main(['-o', str(tmp_path), '-r', '1', 'fp32 (fma)'])
    assert [p.name for p in tmp_path.iterdir()] == ['notes.md']
