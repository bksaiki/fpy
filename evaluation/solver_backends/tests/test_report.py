"""
The report on hand-built records: totals only over what every backend
compiled, differing C++ named, and only the table and the figure written.

    pytest -n 8 evaluation/solver_backends/tests
"""

import json

from report import main, numbers, plotted, summary


def _ok(seconds, solver, sha, queries=10, degraded=0):
    return {
        'outcome': 'ok', 'cpp_sha256': sha, 'seconds': seconds,
        'solver_time': solver,
        'stats': {'solver_time': solver, 'constraints': 6, 'queries': queries,
                  'calls': 5, 'degraded': degraded},
    }


RECORDS = [
    {'design': 'a', 'backends': {'z3': _ok(10, 4, 'x'), 'cvc5': _ok(12, 6, 'x'),
                                 'highs': _ok(8, 2, 'y', degraded=1)}},
    {'design': 'b', 'backends': {'z3': _ok(2, 1, 'p'), 'cvc5': _ok(3, 2, 'p'),
                                 'highs': {'outcome': 'timeout', 'runs': []}}},
]


def test_totals_cover_only_what_every_backend_compiled():
    nums = numbers(RECORDS)
    assert nums['compared'] == 1
    t = nums['totals']
    assert t['z3']['solver_seconds'] == 4
    assert t['highs']['speedup_vs_ref'] == 2
    assert 'speedup_vs_ref' not in t['z3']
    assert t['highs']['degraded'] == 1
    assert nums['cpp_differs'] == {'cvc5': [], 'highs': ['a']}
    assert nums['failed']['highs'] == {'b': 'timeout'}


def test_the_summary_names_what_differs():
    text = summary(numbers(RECORDS))
    assert 'C++ differs from z3 under HiGHS: a' in text
    assert 'not compiled under HiGHS: b (timeout)' in text
    assert '6 constraints, 10 queries' in text


def test_it_names_designs_whose_repeats_did_different_work():
    assert 'counts the same across every backend' in summary(numbers(RECORDS))
    varied = _ok(9, 3, 'x')
    varied['runs'] = [_ok(9, 3, 'x'), {**_ok(9, 3, 'x'), 'stats': {
        **_ok(9, 3, 'x')['stats'], 'constraints': 7}}]
    records = [{'design': 'a', 'backends': {'z3': _ok(10, 4, 'x'), 'highs': varied}}]
    nums = numbers(records)
    assert nums['counts_vary'] == {'z3': [], 'highs': ['a']}
    assert 'counts vary across repeats under HiGHS (a call hit its limit): a' in summary(nums)


def test_it_writes_every_output(tmp_path):
    (tmp_path / 'backends-run.json').write_text(json.dumps({'designs': ['a', 'b']}))
    for rec in RECORDS:
        (tmp_path / f'{rec["design"]}.json').write_text(json.dumps(rec))
    (tmp_path / 'report').mkdir()
    (tmp_path / 'report' / 'stale.png').write_text('')   # replaced, not merged
    assert main([str(tmp_path)]) == 0
    made = {p.name for p in (tmp_path / 'report').iterdir()}
    assert made == {f'backends-scatter{v}.{e}' for v in ('', '-all', '-linear', '-all-linear')
                    for e in ('pdf', 'png')} | {'backends-totals.tex'}
    totals = (tmp_path / 'report' / 'backends-totals.tex').read_text()
    # design `a` alone: every backend compiled it
    assert 'HiGHS & 8\\,s & 2\\,s & 25\\% & 5 & 0.50 & 200.0\\,ms' in totals
    assert 'Constraints' not in totals and 'Queries' not in totals


def test_the_figure_leaves_out_designs_with_no_solver_work():
    idle = {'design': 'fma', 'backends': {b: _ok(0.1, 0.001, 'q') for b in ('z3', 'cvc5')}}
    busy = {'design': 'big', 'backends': {'z3': _ok(5, 2, 'r'), 'cvc5': _ok(5, 0.001, 'r')}}
    assert [r['design'] for r in plotted([idle, busy])] == ['big']
