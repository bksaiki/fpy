"""
Observed values against static bounds, on programs whose answers are known.

    pytest -n 8 evaluation/analysis_precision/tests/test_observe.py
"""

import os
import sys

import fpy2 as fp
from fpy2.analysis import FormatInfer
from fpy2.analysis.format_infer import SetFormat
from fpy2.strategies import monomorphize
from fpy2.types import ListType, RealType

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from observe import Observed, compare, compiler_view, key, observe, statics

F16 = RealType(fp.FP16)


@fp.fpy(ctx=fp.REAL)
def _prod(x, y):
    z = x * y
    return z


@fp.fpy(ctx=fp.REAL)
def _trunc(xs, n):
    with fp.MPFixedContext(n, fp.RM.RTZ):
        ts = [fp.round(x) for x in xs]
    return sum(ts)


@fp.fpy(ctx=fp.REAL)
def _fdpa(xs):
    emax = max([fp.logb(x) for x in xs])
    s = _trunc(xs, emax - 10)
    return s


@fp.fpy(ctx=fp.REAL)
def _acc(xs):
    s = 0
    for x in xs:
        s = s + x
    return s


@fp.fpy(ctx=fp.REAL)
def _pair(x, y):
    p, q = (x * y, x + y)
    return p + q


def _results(fn, arg_types, inputs, digit_bounds):
    """Each site's result, by source key."""
    entry, analyses = compiler_view(fn, arg_types, digit_bounds)
    st = statics(entry, analyses)
    obs = observe(entry, inputs, st)
    return {key(s): compare(b, obs[s]) for s, b in st.items()}


def _by_name(results):
    """Each site's result by name and call depth."""
    return {(name, len(calls)): r for (calls, _, name), r in results.items()}


def test_exact_product_is_tight():
    # FP16 max squared has 22 digits up to 2^31; min subnormal squared is 2^-48
    mx, tiny = 65504.0, 2.0 ** -24
    for db in (False, True):
        r = _by_name(_results(_prod, [F16, F16], [(mx, mx), (tiny, tiny), (-mx, mx)], db))[('z', 0)]
        assert (r.top, r.sound, r.digits, r.lsb, r.msb) == (False, True, 0, 0, 0)


def test_digit_bounds_bound_a_data_dependent_truncation():
    xs = [([1.5, -0.375, 1024.0, 2.0 ** -20],)]
    without = _results(_fdpa, [ListType(F16, 4)], xs, False)
    with_db = _results(_fdpa, [ListType(F16, 4)], xs, True)
    assert without.keys() == with_db.keys()
    without, with_db = _by_name(without)[('ts', 1)], _by_name(with_db)[('ts', 1)]
    assert without.top and not with_db.top
    assert with_db.sound and with_db.digits is not None


def test_every_iteration_is_observed():
    entry, analyses = compiler_view(_acc, [ListType(F16, 3)], False)
    obs = observe(entry, [([1.0, 2.0, 4.0],)], statics(entry, analyses))
    (o,) = [o for (_, stmt, n), o in obs.items() if str(n) == 's' and o.runs > 1]
    assert (o.runs, o.msb, o.prec) == (3, 2, 3)   # 1, 3, 7


def test_tuple_targets_are_separate_sites():
    r = _by_name(_results(_pair, [F16, F16], [(3.0, 5.0)], False))
    assert r[('p', 0)].executed and r[('q', 0)].executed and r[('p', 0)].sound


def test_a_wrong_bound_is_unsound():
    o = Observed()
    o.add(fp.Float.from_float(1 + 2 ** -10))   # 11 digits
    assert not compare(fp.BF16.format(), o).sound
    assert compare(fp.FP16.format(), o).sound
    assert not compare(SetFormat.from_value(1), o).sound
