"""
The report on hand-built records: totals only over what every backend
compiled, differing C++ named, and only the table and the figure written.

    pytest -n 8 evaluation/sensitivity/tests
"""

import json

from report import main, numbers, queries, summary


def _ok(seconds, solver, sha, queries=10, degraded=0):
    return {
        'outcome': 'ok', 'cpp_sha256': sha, 'seconds': seconds,
        'solver_time': solver,
        'stats': {'solver_time': solver, 'queries': queries, 'degraded': degraded},
        'trace': {'constraints': [0, 4, 6], 'seconds': [solver / 4, solver / 4, solver / 2]},
    }


RECORDS = [
    {'design': 'a', 'backends': {'z3': _ok(10, 4, 'x'), 'cvc5': _ok(12, 6, 'x'),
                                 'highs': _ok(8, 2, 'y', degraded=1)}},
    {'design': 'b', 'backends': {'z3': _ok(2, 1, 'p'), 'cvc5': _ok(3, 2, 'p'),
                                 'highs': {'outcome': 'timeout', 'runs': []}}},
]

BISECTS = {'z3': True, 'cvc5': True, 'highs': False}


def test_totals_cover_only_what_every_backend_compiled():
    nums = numbers(RECORDS, BISECTS)
    assert nums['compared'] == 1
    t = nums['totals']
    assert t['z3']['solver_seconds'] == 4
    assert t['highs']['degraded'] == 1
    assert nums['cpp_differs'] == {'cvc5': [], 'highs': ['a']}
    assert nums['failed']['highs'] == {'b': 'timeout'}


def test_a_partly_compiled_design_is_a_failure_and_says_how_much():
    partial = {'outcome': 'timeout', 'compiled': 2, 'runs': [{}, {}, {'timeout': 600}]}
    records = [{'design': 'a', 'backends': {'z3': _ok(10, 4, 'x'), 'highs': partial}}]
    nums = numbers(records, BISECTS)
    assert nums['compared'] == 0
    assert nums['failed']['highs'] == {'a': 'timeout, 2 of 3 compiled'}
    assert 'not compiled under HiGHS: a (timeout, 2 of 3 compiled)' in summary(nums)


def test_the_method_comes_from_the_run():
    t = numbers(RECORDS, BISECTS)['totals']
    assert [t[b]['method'] for b in ('z3', 'cvc5', 'highs')] == [
        'Binary search', 'Binary search', 'Optimize']


def test_the_summary_names_what_differs():
    text = summary(numbers(RECORDS, BISECTS))
    assert 'C++ differs from z3 under HiGHS: a' in text
    assert 'not compiled under HiGHS: b (timeout)' in text
    assert '10 queries, 1 degraded' in text


def test_it_writes_every_output(tmp_path):
    (tmp_path / 'sensitivity-run.json').write_text(json.dumps(
        {'designs': ['a', 'b'], 'bisects': BISECTS}))
    for rec in RECORDS:
        (tmp_path / f'{rec["design"]}.json').write_text(json.dumps(rec))
    (tmp_path / 'notes.md').write_text('mine')   # kept: only its own files
    assert main([str(tmp_path)]) == 0
    assert (tmp_path / 'notes.md').read_text() == 'mine'
    made = {p.name for p in tmp_path.iterdir() if p.name.startswith('sensitivity-')}
    assert made == {'sensitivity-run.json', 'sensitivity-scatter.pdf', 'sensitivity-scatter.png',
                    'sensitivity-totals.tex', 'sensitivity-paper.json', 'sensitivity-summary.txt'}
    paper = json.loads((tmp_path / 'sensitivity-paper.json').read_text())
    assert paper['totals']['highs']['seconds_per_query'] == 0.2
    assert paper['totals']['highs']['method'] == 'Optimize'
    assert '1 compiled by every backend' in (tmp_path / 'sensitivity-summary.txt').read_text()
    totals = (tmp_path / 'sensitivity-totals.tex').read_text()
    # design `a` alone: every backend compiled it
    assert 'HiGHS & Optimize & 8\\,s & 2\\,s & 25\\% & 200.0\\,ms' in totals
    assert 'Constraints' not in totals and 'Queries' not in totals


def test_the_figure_plots_every_query_log_axes_can_show():
    # design `a`'s and `b`'s queries under z3; HiGHS has none from `b`, a timeout
    assert queries(RECORDS, 'z3') == ([4, 6, 4, 6], [1, 2, 0.25, 0.5])
    assert queries(RECORDS, 'highs') == ([4, 6], [0.5, 1])
