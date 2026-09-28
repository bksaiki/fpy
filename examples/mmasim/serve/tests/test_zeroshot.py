"""
`zeroshot`'s flips, subsets and paired Δ; `workloads.pick`; `metrics`' paired
statistics.

    pytest serve/tests
"""

import math
import random

import torch
import zeroshot
from core import stats, workloads


def test_pick_is_seeded_sorted_and_whole_when_asked_for_all() -> None:
    assert workloads.pick(100, 10) == workloads.pick(100, 10, 0) == sorted(workloads.pick(100, 10))
    assert len(set(workloads.pick(100, 10))) == 10 and workloads.pick(100, 10, 1) != workloads.pick(100, 10)
    assert workloads.pick(5, None) == workloads.pick(5, 5) == workloads.pick(5, 9) == [0, 1, 2, 3, 4]


def test_subsets_cap_every_task_by_seed() -> None:
    """At seed 0, a capped task's items are `Random(0)`'s sample;
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
    assert stats.paired([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == (0.0, 0.0)
    mean, se = stats.paired([3.0, 1.0, 5.0], [1.0, 1.0, 1.0])   # d = 2, 0, 4
    assert mean == 2.0 and math.isclose(se, 2.0 / math.sqrt(3))
    assert math.isnan(stats.paired([1.0])[1])
    assert math.isclose(stats.p_value(1.959963984540054, 1.0), 0.05)
    assert stats.p_value(0.0, 0.0) == 1.0 and stats.p_value(1.0, 0.0) == 0.0
    assert [round(p, 12) for p in stats.holm([0.01, 0.04, 0.03])] == [0.03, 0.06, 0.06]
    got = stats.holm([0.01, math.nan])
    assert got[0] == 0.01 and math.isnan(got[1])


def test_against_pairs_items_per_reference() -> None:
    """Every run against R0, a design against the exact run too; Δ over the
    items both have, macro-averaged over tasks."""
    def item(acc: float) -> dict[str, object]:
        return {'acc': acc, 'lls': [-1.0, -2.0], 'gold': 0}

    r0 = {'a': {'0': item(1.0), '1': item(0.0)}, 'b': {'0': item(0.0)}}
    exact = {'a': {'0': item(1.0), '1': item(1.0)}, 'b': {'0': item(0.0)}}
    design = {'a': {'0': item(0.0), '1': item(1.0)}, 'b': {'0': item(1.0)}}
    got = zeroshot.against({'fp32': r0, 'bf16-exact': exact, 'd': design}, 'bf16-exact')
    assert set(got['fp32']) == {'bf16-exact', 'd'} and set(got['bf16-exact']) == {'d'}
    d = got['fp32']['d']
    assert d['tasks']['a']['acc'][0] == 0.0 and d['tasks']['b']['acc'][0] == 1.0
    assert d['macro']['acc'][0] == 0.5
    assert got['bf16-exact']['d']['tasks']['a']['acc'][0] == -0.5


def test_scores_by_hand() -> None:
    """Δ of the correct choice's log-likelihood and of its margin; KL of the
    softmax over the choices; one choice has only the first."""
    ref = {'lls': [-1.0, -2.0, -3.0], 'gold': 0}
    got = {'lls': [-1.5, -1.0, -3.0], 'gold': 0}
    s = zeroshot._scores(ref, got)
    assert s['ll'] == -0.5 and s['margin'] == -0.5 - 1.0
    p = torch.tensor(ref['lls']).log_softmax(0)
    q = torch.tensor(got['lls']).log_softmax(0)
    assert math.isclose(s['kl'], float((p.exp() * (p - q)).sum()), rel_tol=1e-6)
    assert zeroshot._scores(ref, ref)['kl'] == 0
    assert zeroshot._scores({'lls': [-2.0], 'gold': 0}, {'lls': [-2.5], 'gold': 0}) == {'ll': -0.5}
    assert zeroshot._gold('winogrande', {'doc': {'answer': '2'}, 'target': 'x'}) == 1

