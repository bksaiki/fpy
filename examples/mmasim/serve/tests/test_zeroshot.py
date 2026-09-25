"""
`zeroshot.flips`.

    pytest serve/tests
"""

import zeroshot


def test_flips_count_changed_items_over_those_both_runs_have() -> None:
    """Either direction counts; an item one run lacks, or a metric one item
    lacks, is not compared."""
    ref = {'0': {'acc': 1.0, 'acc_norm': 0.0}, '1': {'acc': 0.0, 'acc_norm': 0.0},
           '2': {'acc': 1.0}}
    got = {'0': {'acc': 0.0, 'acc_norm': 0.0}, '1': {'acc': 1.0, 'acc_norm': 1.0},
           '3': {'acc': 1.0}}
    assert zeroshot.flips(ref, ref) == {'acc': (0, 3), 'acc_norm': (0, 2)}
    assert zeroshot.flips(ref, got) == {'acc': (2, 2), 'acc_norm': (1, 2)}
