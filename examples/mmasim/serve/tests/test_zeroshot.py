"""
`zeroshot`'s flips, subsets and paired Δ; `swap.pick`; `metrics`' paired
statistics.

    pytest serve/tests
"""

import math
import random

import metrics
import swap
import zeroshot


def test_pick_is_seeded_sorted_and_whole_when_asked_for_all() -> None:
    assert swap.pick(100, 10) == swap.pick(100, 10, 0) == sorted(swap.pick(100, 10))
    assert len(set(swap.pick(100, 10))) == 10 and swap.pick(100, 10, 1) != swap.pick(100, 10)
    assert swap.pick(5, None) == swap.pick(5, 5) == swap.pick(5, 9) == [0, 1, 2, 3, 4]


def test_subsets_cap_every_task_and_keep_the_old_hellaswag_one() -> None:
    """At seed 0, HellaSwag's 2,000 are those it has always run (`Random(0)`);
    a task no larger than the cap runs whole."""
    got = zeroshot.subsets({'hellaswag': 10042, 'piqa': 1838}, 2000)
    assert set(got) == {'hellaswag'}
    assert got['hellaswag'] == sorted(random.Random(0).sample(range(10042), 2000))
    assert zeroshot.subsets({'hellaswag': 10042}, None) == {}


def test_flips_count_changed_items_over_those_both_runs_have() -> None:
    """Either direction counts; an item one run lacks, or a metric one item
    lacks, is not compared."""
    ref = {'0': {'acc': 1.0, 'acc_norm': 0.0}, '1': {'acc': 0.0, 'acc_norm': 0.0},
           '2': {'acc': 1.0}}
    got = {'0': {'acc': 0.0, 'acc_norm': 0.0}, '1': {'acc': 1.0, 'acc_norm': 1.0},
           '3': {'acc': 1.0}}
    assert zeroshot.flips(ref, ref) == {'acc': (0, 3), 'acc_norm': (0, 2)}
    assert zeroshot.flips(ref, got) == {'acc': (2, 2), 'acc_norm': (1, 2)}


def test_paired_statistics_by_hand() -> None:
    assert metrics.paired([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == (0.0, 0.0)
    mean, se = metrics.paired([3.0, 1.0, 5.0], [1.0, 1.0, 1.0])   # d = 2, 0, 4
    assert mean == 2.0 and math.isclose(se, 2.0 / math.sqrt(3))
    assert math.isnan(metrics.paired([1.0])[1])
    assert math.isclose(metrics.p_value(1.959963984540054, 1.0), 0.05)
    assert metrics.p_value(0.0, 0.0) == 1.0 and metrics.p_value(1.0, 0.0) == 0.0
    assert [round(p, 12) for p in metrics.holm([0.01, 0.04, 0.03])] == [0.03, 0.06, 0.06]
    got = metrics.holm([0.01, math.nan])
    assert got[0] == 0.01 and math.isnan(got[1])


def test_against_pairs_items_per_reference() -> None:
    """Every run against R0, a design against the exact run too; Δ over the
    items both have, macro-averaged over tasks."""
    r0 = {'a': {'0': {'acc': 1.0}, '1': {'acc': 0.0}}, 'b': {'0': {'acc': 0.0}}}
    exact = {'a': {'0': {'acc': 1.0}, '1': {'acc': 1.0}}, 'b': {'0': {'acc': 0.0}}}
    design = {'a': {'0': {'acc': 0.0}, '1': {'acc': 1.0}}, 'b': {'0': {'acc': 1.0}}}
    got = zeroshot.against({'fp32': r0, 'bf16-exact': exact, 'd': design}, 'bf16-exact')
    assert set(got['fp32']) == {'bf16-exact', 'd'} and set(got['bf16-exact']) == {'d'}
    d = got['fp32']['d']
    assert d['tasks']['a']['acc'][0] == 0.0 and d['tasks']['b']['acc'][0] == 1.0
    assert d['macro']['acc'][0] == 0.5
    assert got['bf16-exact']['d']['tasks']['a']['acc'][0] == -0.5

