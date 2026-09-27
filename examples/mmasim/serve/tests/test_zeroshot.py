"""
`zeroshot.flips` and its subsets, and `swap.pick`.

    pytest serve/tests
"""

import random

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
