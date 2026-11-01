"""
The tracer sees every value at every site, keyed per call path.

    pytest -n 8 evaluation/analysis_precision/tests/test_trace.py
"""


from tracing import trace

import fpy2 as fp
from fpy2.ast.fpyast import Add, Call, ListComp, Mul


@fp.fpy(ctx=fp.REAL)
def _double(x):
    return x + x


@fp.fpy(ctx=fp.REAL)
def _quad(x):
    return _double(_double(x))


@fp.fpy(ctx=fp.REAL)
def _loop(xs):
    s = 0
    for x in xs:
        s = s + _double(x)
    return _quad(s)


@fp.fpy(ctx=fp.REAL)
def _squares(xs):
    return [x * x for x in xs]


def _values(seen, kind, depth):
    """Each value seen at a site of expression type *kind* reached through
    *depth* calls, as floats, by call path."""
    return {
        call: [float(v) for v in vs]
        for (call, e), vs in seen.items()
        if isinstance(e, kind) and len(call) == depth
    }


def test_every_iteration_and_call_path():
    _, seen = trace(_loop, [1.0, 2.0, 3.0])
    # the loop's call: every iteration; `_quad`'s two nested calls: once each
    doubles = _values(seen, Add, 1) | _values(seen, Add, 2)
    assert sorted(doubles.values()) == [[2.0, 4.0, 6.0], [24.0], [48.0]]
    # the loop body's own sum, at top level
    assert list(_values(seen, Add, 0).values()) == [[2.0, 6.0, 12.0]]
    # `_quad`'s inner and outer calls are distinct paths through it
    paths = {call for call, _ in seen if len(call) == 2}
    assert len(paths) == 2 and all(isinstance(c, Call) for p in paths for c in p)


def test_each_comprehension_element():
    _, seen = trace(_squares, [1.0, 2.0, 3.0])
    assert list(_values(seen, Mul, 0).values()) == [[1.0, 4.0, 9.0]]
    (comp,) = [vs for (_, e), vs in seen.items() if isinstance(e, ListComp)]
    assert [[float(x) for x in v] for v in comp] == [[1.0, 4.0, 9.0]]


def test_result_unchanged():
    for fn, arg in ((_loop, [0.5, -1.25, 3.0]), (_squares, [1.5, -2.0])):
        assert trace(fn, arg)[0] == fn(arg)
