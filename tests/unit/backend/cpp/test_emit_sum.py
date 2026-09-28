"""``sum(xs)`` must be the fold the interpreter performs.

``_eval_sum`` seeds with ``xs[0]`` **unrounded** and performs *n-1* additions,
each rounded to the active context; an empty list is an exact ``+0``.  Seeding a
typed zero over the whole range did *n* additions instead, so ``sum([-0.0])``
came out ``+0.0`` and a narrower seed rounded the first element away
(``sum([5e-324])`` under FP32 gave ``0``).

**Wider accumulator yes, narrower no.**  ``accumulate``'s step is
``init = init + *first``, so the addition is ``T + E`` under the usual arithmetic
conversions.  When ``E`` converts exactly the common type *is* ``T``, so the step
promotes exactly, adds once and rounds once — uni-precision at the accumulator,
as the interpreter is.  When ``E`` is wider the step computes in ``E`` and
narrows on assignment, rounding twice; widening ``T`` to hold the unrounded seed
would instead round every addition at the wrong format.
"""

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import fpy2 as fp
import pytest

from fpy2.backend.cpp import CppCompiler, CppCompileError
from fpy2.number import RealFloat
from fpy2.types import ListType, RealType

_L64 = ListType(RealType(fp.FP64))
_CXX = shutil.which('c++') or shutil.which('g++')


def _run(func, arg_types, main: str) -> str:
    """The output of *func*'s C++ with *main* appended."""
    cc = CppCompiler()
    src = (
        cc.prelude() + '\n' + cc.compile(func, arg_types=arg_types) + '\n'
        + '#include <cstdio>\n' + main
    )
    with tempfile.TemporaryDirectory() as d:
        cpp, exe = Path(d) / 's.cpp', Path(d) / 's'
        cpp.write_text(src)
        assert _CXX is not None
        subprocess.run([_CXX, '-std=c++11', '-o', str(exe), str(cpp)], check=True)
        return subprocess.run(
            [str(exe)], capture_output=True, text=True, check=True).stdout


class TestTheEmittedFold:
    def test_seeds_from_the_first_element_and_guards_the_empty_list(self):
        """``begin() + 1`` and ``xs[0]`` are both undefined on an empty vector,
        and the differential harness runs length zero.  A named operand is read
        directly: only a prvalue needs binding, so that ``begin()``/``end()``
        name the same object (see ``test_emit_bool.py``)."""

        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real]) -> fp.Real:
            return sum(xs)

        out = CppCompiler().compile(f, arg_types=[_L64])
        assert 'std::accumulate(xs.begin() + 1, xs.end(), ' in out, out
        # ...and the empty answer is a positive zero, per `_eval_sum`
        assert 'xs.size() == 0 ? static_cast<double>(0)' in out, out
        assert 'auto&&' not in out, out


class TestAccumulatorWidth:
    def test_a_wider_accumulator_is_allowed_and_seeds_with_a_cast(self):
        """``int8`` elements into FP64: the conversion is exact, so the fold is
        uni-precision at ``double``.

        The cast on the seed is load-bearing -- ``accumulate`` deduces ``T``
        from it, so without one the whole fold would run in the element type.
        """

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.SINT8:
                q = [fp.round(x) for x in xs]
            with fp.FP64:
                return sum(q)

        out = CppCompiler(optimize=False).compile(
            f, ctx=fp.FP64, arg_types=[_L64],
        )
        assert 'std::accumulate' in out, out
        # the seed's cast, not the empty-list guard's `static_cast<double>(0)`
        assert 'static_cast<double>(q[static_cast<size_t>(0)])' in out, out

    def test_a_narrower_accumulator_is_refused(self):
        """FP64 elements into an FP32 accumulator.

        The seed would round, which is the ``[5e-324]`` case, and no accumulator
        type avoids it without breaking the per-addition rounding.  A refusal is
        an acceptable answer; a wrong sum is not.
        """

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP32:
                return sum(xs)

        with pytest.raises(CppCompileError, match='cannot hold one exactly'):
            CppCompiler(optimize=False).compile(
                f, ctx=fp.FP32, arg_types=[_L64],
            )


@fp.fpy(ctx=fp.REAL)
def _scaled_sum(xs: list[fp.Real], k: fp.Real) -> fp.Real:
    ts = fp.empty(len(xs))
    for i in range(len(xs)):
        y = xs[i] * 2 ** k
        with fp.MPFixedContext(-1, fp.RM.RTZ, enable_neg_zero=False):
            ts[i] = fp.round(y)
    return sum(ts)


# 11-bit integers up to 2^54: `float` elements, an `int64_t` sum
_SCALED_ARGS = [
    ListType(RealType(fp.FP16), 5),
    RealType(fp.MPBFixedContext(-1, RealFloat.from_int(38)).format()),
]


class TestIntegralFloatElements:
    """Elements whose *values* fit the accumulator though their type does not.

    ``int64_t + float`` computes in ``float``, so each element is cast first;
    the sum below is off by ``2^24`` otherwise.
    """

    def test_each_element_is_cast_before_it_is_added(self):
        out = CppCompiler().compile(_scaled_sum, arg_types=_SCALED_ARGS)
        assert 'return _tmp2 + static_cast<int64_t>(_tmp3); })' in out, out

    @pytest.mark.skipif(_CXX is None, reason='no C++ compiler')
    def test_it_agrees_with_the_interpreter(self):
        xs, k = [65504.0, 2.0 ** -14, 0.0, 0.0, 0.0], 38
        want = int(_scaled_sum(xs, k))
        assert want == 65504 * 2 ** 38 + 2 ** 24
        got = _run(_scaled_sum, _SCALED_ARGS, (
            'int main() { std::array<float, 5> xs{' + ', '.join(map(repr, xs))
            + f'}}; printf("%lld\\n", (long long) _scaled_sum(xs, {k})); }}\n'
        ))
        assert int(got) == want


@fp.fpy(ctx=fp.REAL)
def _fsum(xs, k):
    ts = fp.empty(len(xs))
    for i in range(len(xs)):
        with fp.MPFixedContext(-1, fp.RM.RTZ, enable_neg_zero=False):
            r = fp.round(xs[i])
        t = 2 ** k * r
        ts[i] = t
    return sum(ts)


@fp.fpy(ctx=fp.REAL)
def _guarded_fsum(xs, y):
    if fp.isfinite(y):
        return _fsum(xs, max(fp.logb(y), -10))
    return 0.0


_GUARDED_ARGS = [ListType(RealType(fp.FP16), 4), RealType(fp.FP16)]


class TestAScaleOnlyTheCallerProvesFinite:
    """`2 ** k` may leave the sum only where `k` is finite, and only the
    caller's guard says so: the callee sees it through its parameter's format.
    """

    def test_the_factor_leaves_the_loop(self):
        out = CppCompiler().compile(_guarded_fsum, arg_types=_GUARDED_ARGS)
        assert '* static_cast<double>(std::accumulate(' in out, out

    @pytest.mark.skipif(_CXX is None, reason='no C++ compiler')
    def test_it_agrees_with_the_interpreter(self):
        cases = [
            ([1.5, -3.0, 7.9, 0.0], 12.0),
            ([65504.0, 1.0, -2.0, 3.0], 2.0 ** -14),
            ([1.0, 2.0, 3.0, 4.0], 0.0),
            ([1.0, 2.0, 3.0, 4.0], math.inf),
        ]
        calls = ''.join(
            'printf("%a\\n", (double) _guarded_fsum(std::array<float, 4>{'
            + ', '.join(map(repr, xs)) + f'}}, {"INFINITY" if math.isinf(y) else repr(y)}));'
            for xs, y in cases
        )
        got = _run(_guarded_fsum, _GUARDED_ARGS, f'int main() {{ {calls} }}\n')
        want = [float(_guarded_fsum(xs, y)).hex() for xs, y in cases]
        assert [float.fromhex(g).hex() for g in got.split()] == want
