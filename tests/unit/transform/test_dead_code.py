"""
Unit tests for dead code elimination.
"""

import fpy2 as fp
import unittest

@fp.fpy
def _example_simple_1():
    x = 1
    y = 2
    return x

@fp.fpy
def _example_simple_1_expect():
    x = 1
    return x


@fp.fpy
def _example_simple_2():
    z = 3 * 4
    x = 1
    return x

@fp.fpy
def _example_simple_2_expect():
    x = 1
    return x

@fp.fpy
def _example_dead_if1_1():
    if False:
        x = 1
    return 2

@fp.fpy
def _example_dead_if1_1_expect():
    return 2

@fp.fpy
def _example_dead_if1_2(t: fp.Real):
    if t < 0:
        pass
    return t

@fp.fpy
def _example_dead_if1_2_expect(t: fp.Real):
    return t

@fp.fpy
def _example_if1_1(t: fp.Real):
    if t < 0:
        x = 1
    return t

@fp.fpy
def _example_if1_1_expect(t: fp.Real):
    return t

@fp.fpy
def _example_if1_2(t: fp.Real):
    if t < 0:
        x = 1
        t = 2
    return t

@fp.fpy
def _example_if1_2_expect(t: fp.Real):
    if t < 0:
        t = 2
    return t

@fp.fpy
def _example_dead_if_1(t: fp.Real):
    if False:
        t = 1
    else:
        t = 2
    return t

@fp.fpy
def _example_dead_if_1_expect(t: fp.Real):
    t = 2
    return t

@fp.fpy
def _example_dead_if_2(t: fp.Real):
    if True:
        t = 1
    else:
        t = 2
    return t

@fp.fpy
def _example_dead_if_2_expect(t: fp.Real):
    t = 1
    return t

@fp.fpy
def _example_dead_if_3(t: fp.Real):
    if t < 0:
        pass
    else:
        t = 2
    return t

@fp.fpy
def _example_dead_if_3_expect(t: fp.Real):
    if not t < 0:
        t = 2
    return t

@fp.fpy
def _example_dead_if_4(t: fp.Real):
    if t < 0:
        t = 1
    else:
        pass
    return t

@fp.fpy
def _example_dead_if_4_expect(t: fp.Real):
    if t < 0:
        t = 1
    return t

@fp.fpy
def _example_dead_if_5(t: fp.Real):
    if t < 0:
        pass
    else:
        pass
    return t

@fp.fpy
def _example_dead_if_5_expect(t: fp.Real):
    return t

@fp.fpy
def _example_if_1(t: fp.Real):
    if t < 0:
        x = 1
    else:
        t = 2
    return t

@fp.fpy
def _example_if_1_expect(t: fp.Real):
    if not t < 0:
        t = 2
    return t

@fp.fpy
def _example_if_2(t: fp.Real):
    if t < 0:
        t = 2
    else:
        x = 1
    return t

@fp.fpy
def _example_if_2_expect(t: fp.Real):
    if t < 0:
        t = 2
    return t

@fp.fpy
def _example_if_3(t: fp.Real):
    if t < 0:
        x = 1
        t = 2
    else:
        t = 1
    return t

@fp.fpy
def _example_if_3_expect(t: fp.Real):
    if t < 0:
        t = 2
    else:
        t = 1
    return t

@fp.fpy
def _example_if_4(t: fp.Real):
    if t < 0:
        x = 1
        y = 0
    else:
        x = 2
        z = 0
    return t

@fp.fpy
def _example_if_4_expect(t: fp.Real):
    return t

@fp.fpy
def _example_dead_while_1():
    x = 0
    while False:
        x = x + 1
    return x

@fp.fpy
def _example_dead_while_1_expect():
    x = 0
    return x

@fp.fpy
def _example_dead_while_2():
    x = 0
    while x < 10:
        pass
    return x

@fp.fpy
def _example_dead_while_2_expect():
    x = 0
    return x

@fp.fpy
def _example_while_1():
    x = 0
    while x < 10:
        y = 5
        x = x + 1
    return x

@fp.fpy
def _example_while_1_expect():
    x = 0
    while x < 10:
        x = x + 1
    return x

@fp.fpy
def test_simple_3():
    x = 1
    y = 2
    x = 3
    return x

@fp.fpy
def test_simple_3_expect():
    x = 3
    return x

@fp.fpy
def test_simple_4():
    x = 1
    x = 2
    x = 3
    return x

@fp.fpy
def test_simple_4_expect():
    x = 3
    return x


_examples: list[tuple[fp.Function, fp.Function]] = [
    (_example_simple_1, _example_simple_1_expect),
    (_example_simple_2, _example_simple_2_expect),
    (_example_dead_if1_1, _example_dead_if1_1_expect),
    (_example_dead_if1_2, _example_dead_if1_2_expect),
    (_example_if1_1, _example_if1_1_expect),
    (_example_if1_2, _example_if1_2_expect),
    (_example_dead_if_1, _example_dead_if_1_expect),
    (_example_dead_if_2, _example_dead_if_2_expect),
    (_example_dead_if_3, _example_dead_if_3_expect),
    (_example_dead_if_4, _example_dead_if_4_expect),
    (_example_dead_if_5, _example_dead_if_5_expect),
    (_example_if_1, _example_if_1_expect),
    (_example_if_2, _example_if_2_expect),
    (_example_if_3, _example_if_3_expect),
    (_example_if_4, _example_if_4_expect),
    (_example_dead_while_1, _example_dead_while_1_expect),
    (_example_dead_while_2, _example_dead_while_2_expect),
    (_example_while_1, _example_while_1_expect),
    (test_simple_3, test_simple_3_expect),
    (test_simple_4, test_simple_4_expect)
]


class TestDeadCode(unittest.TestCase):

    def test_examples(self):
        for f, f_expect in _examples:
            f_opt = fp.transform.DeadCodeEliminate.apply(f.ast)
            f_opt.name = f_expect.name
            self.assertTrue(f_opt.is_equiv(f_expect.ast), f'expect:\n{f_expect.format()}\nactual:\n{f_opt.format()}')
