"""
Guards the emitter drops because a branch already ruled the value out.

Every assertion here protects an operation against a NaN, an infinity or a zero,
and every one of them is unnecessary where the program has already tested for
those.
Value classes are what read the test; format inference cannot, since a format
says whether *some* value in it is a NaN, not whether *this* one is.

Each site is checked as a **pair of programs differing only in a branch**, so a
missing guard is attributable to the branch rather than to the operand's format
-- which is identical in both halves.  Two compile-and-run differentials back
that up: dropping a guard must not change what the program computes, nor where it
aborts.

A class is a fact about the FPy value where the guard protects a C++ operation on
its *storage*.  ``TestStorageIsNotTheContext`` pins the case that keeps those two
apart: rounding to a narrower format can *produce* an infinity, and the analysis
has to say so.
"""

import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import pytest

import fpy2 as fp
from fpy2.backend.cpp import CppCompiler
from fpy2.number import MPBFixedContext
from fpy2.types import ListType, RealType

_CXX = shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')

ASSERT = fp.OverflowMode.ASSERT
RTZ = fp.RM.RTZ

# refuses a NaN and both infinities; `enable_neg_zero` picks the storage kind,
# and no integer type has a signed zero
FLOAT_STORAGE = MPBFixedContext(
    -1, fp.RealFloat(exp=10, c=1), rm=RTZ, overflow=ASSERT,
    enable_neg_zero=True)
INT_STORAGE = MPBFixedContext(
    -1, fp.RealFloat(exp=10, c=1), rm=RTZ, overflow=ASSERT,
    enable_neg_zero=False)
# represents both specials, so its bound assert carries an exemption for them
WITH_SPECIALS = MPBFixedContext(
    -1, fp.RealFloat(exp=10, c=1), rm=RTZ, overflow=ASSERT,
    enable_nan=True, enable_inf=True)

_DRIVER = r'''
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
int main(int argc, char** argv) {
    (void) argc;
    uint64_t b = (uint64_t) std::strtoull(argv[1], nullptr, 16);
    double x; std::memcpy(&x, &b, 8);
    std::printf("%.17g\n", (double) q(x));
    return 0;
}
'''

_INPUTS = [
    0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 100.7, -100.7, 1023.0, 1024.0, 1025.0,
    2048.0, 1e300, float('inf'), float('-inf'), float('nan'),
]


def _round_pair(ctx):
    """``round(v)`` under *ctx*, once bare and once behind a finiteness test."""
    @fp.fpy(ctx=fp.REAL)
    def bare(v: fp.Real) -> fp.Real:
        with ctx:
            y = fp.round(v)
        return y

    @fp.fpy(ctx=fp.REAL)
    def guarded(v: fp.Real) -> fp.Real:
        if fp.isfinite(v):
            with ctx:
                y = fp.round(v)
        else:
            y = 0
        return y

    return bare, guarded


def _emit(func, arg_ctx=fp.FP64) -> str:
    return CppCompiler().compile(func, arg_types=[RealType(arg_ctx)])


def _bits(x: float) -> int:
    return struct.unpack('<Q', struct.pack('<d', x))[0]


def _asserts(src: str) -> str:
    """Only the assertion lines.  The programs here *test* for a NaN themselves,
    so searching the whole output would find the branch and not the guard."""
    return '\n'.join(ln for ln in src.splitlines() if 'assert(' in ln)


def _compiles(src: str) -> None:
    """*src* is C++ a compiler accepts.

    Every assertion in this file greps emitted text, which a type error passes
    unnoticed -- a narrowed element type once produced ``std::max(int16_t,
    float)`` under a passing test.  Warnings are errors here: a narrowing
    inside a braced initializer is only a warning on GCC and ill-formed in the
    standard.
    """
    if _CXX is None:
        pytest.skip('no C++ compiler')
    from fpy2.backend.cpp.utils import CPP_HEADERS
    with tempfile.TemporaryDirectory() as td:
        cpp = Path(td) / 'm.cpp'
        cpp.write_text('\n'.join(CPP_HEADERS) + '\n' + src)
        out = subprocess.run(
            [_CXX, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-fsyntax-only',
             str(cpp)],
            capture_output=True, text=True,
        )
    assert out.returncode == 0, out.stderr[-2000:]


def _build(src: str, name: str, td: str) -> Path:
    from fpy2.backend.cpp.utils import CPP_HEADERS
    cpp, exe = Path(td) / 'm.cpp', Path(td) / 'm'
    cpp.write_text('\n'.join(CPP_HEADERS) + '\n' + src
                   + _DRIVER.replace('q(x)', f'{name}(x)'))
    build = subprocess.run([_CXX, '-std=c++17', '-O0', '-o', str(exe), str(cpp)],
                           capture_output=True, text=True)
    assert build.returncode == 0, build.stderr[-2000:]
    return exe


class TestRoundingSpecialsGuard:
    """``_undefined_guard``: an operand a context has no result for."""

    @pytest.mark.parametrize('ctx', [FLOAT_STORAGE, INT_STORAGE],
                             ids=['float_storage', 'integer_storage'])
    def test_a_finiteness_test_removes_the_assert(self, ctx):
        bare, guarded = _round_pair(ctx)
        assert 'undefined for this value' in _asserts(_emit(bare))
        assert 'undefined for this value' not in _asserts(_emit(guarded))

    def test_only_the_refused_half_goes(self):
        """The guard is two tests, and each is dropped on its own: an operand
        that can still be an infinity keeps the ``isinf`` half."""
        @fp.fpy(ctx=fp.REAL)
        def q(v: fp.Real) -> fp.Real:
            if not fp.isnan(v):
                with FLOAT_STORAGE:
                    y = fp.round(v)
            else:
                y = 0
            return y

        guards = _asserts(_emit(q))
        assert 'std::isinf(v)' in guards
        assert 'std::isnan(v)' not in guards
        # the combined spelling is only for both halves at once
        assert 'std::isfinite(v)' not in guards


class TestBoundExemption:
    """The ``!isfinite(operand) ||`` disjunct in front of the bound test."""

    def test_a_context_holding_specials_exempts_them(self):
        bare, _ = _round_pair(WITH_SPECIALS)
        assert '!std::isfinite(' in _asserts(_emit(bare))

    def test_an_operand_that_cannot_be_one_needs_no_exemption(self):
        """Not a guard but a *widening*: an operand that is never a NaN or an
        infinity never takes the exemption, so the bound test stands alone."""
        _, guarded = _round_pair(WITH_SPECIALS)
        guards = _asserts(_emit(guarded))
        assert '!std::isfinite(' not in guards
        assert 'overflow occurred' in guards


class TestFloatToIntegerConversion:
    """``_guard_float_to_integer``: the conversion itself is undefined."""

    def test_a_finiteness_test_removes_the_assert(self):
        """A native integer context rounds by the bare cast, and converting a
        NaN or an infinity to an integer type is undefined in C++."""
        bare, guarded = _round_pair(fp.SINT8)
        assert 'std::isfinite' in _asserts(_emit(bare))
        out = _emit(guarded)
        assert not _asserts(out)
        assert 'static_cast<int8_t>' in out


class TestCastExactness:
    def test_the_specials_assert_goes(self):
        @fp.fpy(ctx=fp.REAL)
        def bare(v: fp.Real) -> fp.Real:
            with FLOAT_STORAGE:
                y = fp.cast(v)
            return y

        @fp.fpy(ctx=fp.REAL)
        def guarded(v: fp.Real) -> fp.Real:
            if fp.isfinite(v):
                with FLOAT_STORAGE:
                    y = fp.cast(v)
            else:
                y = 0
            return y

        assert 'a NaN or an infinity is not representable' in _emit(bare)
        guards = _asserts(_emit(guarded))
        assert 'a NaN or an infinity is not representable' not in guards
        # the claims that remain are about the value, not its kind
        assert 'only integers' in guards
        assert "outside the context's bound" in guards

    def test_the_nan_aware_equality_goes(self):
        """``NaN == NaN`` is false, so the storage roundtrip carries an extra
        disjunct -- needed only where a NaN can reach it."""
        @fp.fpy(ctx=fp.REAL)
        def bare(v: fp.Real) -> fp.Real:
            with fp.FP32:
                y = fp.cast(v)
            return y

        @fp.fpy(ctx=fp.REAL)
        def guarded(v: fp.Real) -> fp.Real:
            if not fp.isnan(v):
                with fp.FP32:
                    y = fp.cast(v)
            else:
                y = 0
            return y

        assert 'std::isnan' in _asserts(_emit(bare))
        assert 'std::isnan' not in _asserts(_emit(guarded))


class TestAClampReachesIntegerStorage:
    """A selection's *order* is what lets a clamp narrow storage.

    `logb` is integer-valued, but its format admits both infinities --
    ``logb(0)`` is ``-inf`` and ``logb(inf)`` is ``+inf`` -- so no integer rung
    contains it.  Clamping removes them, and only an order-aware `min`/`max`
    rule can see that: the join of the operand classes carries the infinity
    straight through.
    """

    def test_a_double_clamp_gives_an_integer(self):
        @fp.fpy
        def unclamped(x: fp.Real) -> fp.Real:
            if fp.isnan(x):
                return 0
            else:
                return fp.logb(x)

        @fp.fpy
        def clamped(x: fp.Real) -> fp.Real:
            if fp.isnan(x):
                return 0
            else:
                return min(max(fp.logb(x), -126), 128)

        tys = [RealType(fp.FP32)]
        assert 'float unclamped(' in CppCompiler().compile(
            unclamped, ctx=fp.REAL, arg_types=tys)
        # `logb` of an FP32 tops out at 127, so `min(., 128)` bounds the range
        # to [-126, 127] -- which is why this is `int8_t` and not `int16_t`
        assert 'int8_t clamped(' in CppCompiler().compile(
            clamped, ctx=fp.REAL, arg_types=tys)

    def test_one_clamp_is_not_enough(self):
        """Clamping below leaves ``+inf`` from ``logb(inf)``, and an integer
        rung holds neither infinity."""
        @fp.fpy
        def half(x: fp.Real) -> fp.Real:
            if fp.isnan(x):
                return 0
            else:
                return max(fp.logb(x), -126)

        out = CppCompiler().compile(half, ctx=fp.REAL, arg_types=[RealType(fp.FP32)])
        assert 'float half(' in out


class TestAResultStorageIsNotAnOperandTarget:
    """A class narrows where a result *goes*, never what is fed in.

    Which C++ signature runs is a question about the values that occur, so a
    `logb` a branch has made finite reaches the integer one -- and then nothing
    converts to a float and back to use it.
    """

    def test_a_guarded_logb_reaches_the_integer_op(self):
        @fp.fpy
        def q(x: fp.Real) -> fp.Real:
            if fp.isnan(x) or fp.isinf(x) or x == 0:
                return 0
            else:
                with fp.REAL:
                    return fp.logb(x)

        out = CppCompiler().compile(q, ctx=fp.REAL, arg_types=[RealType(fp.FP32)])
        assert 'std::ilogb(' in out
        assert 'std::logb(' not in out

    def test_an_unguarded_logb_stays_on_the_float_op(self):
        """``logb(0)`` is ``-inf``, and converting one to an ``int`` is
        undefined -- ``std::ilogb`` would be a wrong answer, not a wider one."""
        @fp.fpy
        def q(x: fp.Real) -> fp.Real:
            with fp.REAL:
                return fp.logb(x)

        out = CppCompiler().compile(q, ctx=fp.REAL, arg_types=[RealType(fp.FP32)])
        assert 'std::logb(' in out
        assert 'std::ilogb(' not in out

    def test_an_operand_is_not_narrowed_by_the_result(self):
        """``max`` is finite where ``logb`` is not, so its operands stay
        ``float``."""
        @fp.fpy
        def q(x: fp.Real) -> fp.Real:
            if fp.isnan(x):
                return 0
            else:
                return min(max(fp.logb(x), -126), 128)

        out = CppCompiler().compile(q, ctx=fp.REAL, arg_types=[RealType(fp.FP32)])
        assert 'int8_t q(' in out
        assert 'std::logb(' in out          # the operand keeps its float op
        assert 'std::ilogb(' not in out


class TestATupleReturnNarrowsPerField:
    """A function's return type is its ABI, so a field that collapses to one
    class for the whole tuple stays wide in every caller too."""

    def test_the_narrow_field_stays_narrow(self):
        @fp.fpy
        def q(x: fp.Real) -> tuple[fp.Real, fp.Real]:
            if fp.isnan(x) or fp.isinf(x) or x == 0:
                return x, 0
            else:
                with fp.REAL:
                    return x, fp.logb(x)

        out = CppCompiler().compile(
            q, ctx=fp.REAL, arg_types=[RealType(fp.FP32)])
        assert 'std::tuple<float, int16_t> q(' in out

    def test_a_field_that_can_be_an_infinity_does_not(self):
        """The same shape with the guard removed: `logb(0)` is ``-inf``."""
        @fp.fpy
        def q(x: fp.Real) -> tuple[fp.Real, fp.Real]:
            with fp.REAL:
                return x, fp.logb(x)

        out = CppCompiler().compile(
            q, ctx=fp.REAL, arg_types=[RealType(fp.FP32)])
        assert 'std::tuple<float, float> q(' in out


class TestAListStoresAtItsElements:
    """The same narrowing, one level in: a list stores at what its elements can
    be rather than at their format.

    Over a list the guard has to be a guard over the *whole* list, since nothing
    else rules a NaN out of every element -- so this is what the reduction
    refinement buys.  `float` here would be four bytes per element to hold a
    value in ``[-126, 127]``, and a `float` reduction to fold them.
    """

    @staticmethod
    def _guarded():
        @fp.fpy(ctx=fp.REAL)
        def q(xs) -> fp.Real:
            if all([fp.isfinite(x) and x != 0 for x in xs]):
                ys = [max(fp.logb(x), -126) for x in xs]
                return max(ys)
            else:
                return 0
        return q

    @staticmethod
    def _emit(q):
        out = CppCompiler().compile(
            q, arg_types=[ListType(RealType(fp.FP32), 8)])
        _compiles(out)
        return out

    def test_the_buffer_holds_the_element_type(self):
        assert 'std::array<int8_t, 8>' in self._emit(self._guarded())

    def test_the_store_spells_its_conversion(self):
        """The *value* fits where the expression's storage does not: ``max``
        computes at ``float`` because ``logb`` does."""
        assert 'static_cast<int8_t>(std::max(' in self._emit(self._guarded())

    def test_the_reduction_folds_on_the_integer_path(self):
        out = self._emit(self._guarded())
        assert re.search(r'= std::max\(\w+, ys\[', out), out
        assert 'signbit' not in out
        assert 'quiet_NaN' not in out

    def test_without_the_guard_it_stays_a_float(self):
        """``logb(0)`` is ``-inf`` and ``logb(inf)`` is ``+inf``, so an element
        can be one and no integer rung holds it."""
        @fp.fpy(ctx=fp.REAL)
        def q(xs) -> fp.Real:
            ys = [max(fp.logb(x), -126) for x in xs]
            return max(ys)

        out = self._emit(q)
        assert 'std::array<float, 8>' in out
        assert 'std::array<int8_t' not in out


class TestMinMax:
    """``_emit_ieee_min_max``: IEEE ``minimum`` propagates a NaN."""

    def test_the_propagation_goes_when_neither_operand_can_be_one(self):
        @fp.fpy
        def bare(a: fp.Real, b: fp.Real) -> fp.Real:
            with fp.FP64:
                return fp.fmin(a, b)

        @fp.fpy(ctx=fp.REAL)
        def guarded(a: fp.Real, b: fp.Real) -> fp.Real:
            if not fp.isnan(a) and not fp.isnan(b):
                with fp.FP64:
                    y = fp.fmin(a, b)
            else:
                y = 0
            return y

        tys = [RealType(fp.FP64)] * 2
        assert 'quiet_NaN' in CppCompiler().compile(bare, arg_types=tys)
        out = CppCompiler().compile(guarded, arg_types=tys)
        assert 'quiet_NaN' not in out
        # the signed-zero half of the predicate is not what a NaN test rules out
        assert 'std::signbit' in out

    def test_an_earlier_step_of_a_fold_can_go_alone(self):
        """Three operands are two pairwise steps.  Only the last operand can be
        a NaN, so the first step needs no propagation and the second does."""
        @fp.fpy(ctx=fp.REAL)
        def q(a: fp.Real, b: fp.Real, c: fp.Real) -> fp.Real:
            if not fp.isnan(a) and not fp.isnan(b):
                with fp.FP64:
                    y = min(a, b, c)
            else:
                y = 0
            return y

        out = CppCompiler().compile(q, arg_types=[RealType(fp.FP64)] * 3)
        assert out.count('quiet_NaN') == 1
        assert out.count('std::signbit') == 2

    def test_the_signbit_term_goes_when_an_operand_cannot_be_zero(self):
        """Only ``a = -0`` against ``b = +0`` needs the term, so ruling out a
        zero on *either* side is enough."""
        @fp.fpy
        def bare(a: fp.Real, b: fp.Real) -> fp.Real:
            with fp.FP64:
                return fp.fmin(a, b)

        @fp.fpy(ctx=fp.REAL)
        def guarded(a: fp.Real, b: fp.Real) -> fp.Real:
            if a != 0:
                with fp.FP64:
                    y = fp.fmin(a, b)
            else:
                y = 0
            return y

        tys = [RealType(fp.FP64)] * 2
        assert 'std::signbit' in CppCompiler().compile(bare, arg_types=tys)
        out = CppCompiler().compile(guarded, arg_types=tys)
        assert 'std::signbit' not in out
        # a zero test says nothing about a NaN, so that half stays
        assert 'quiet_NaN' in out

    def test_a_non_zero_literal_is_enough_on_either_side(self):
        @fp.fpy
        def second(a: fp.Real) -> fp.Real:
            with fp.FP64:
                return fp.fmin(a, 1)

        @fp.fpy
        def first(a: fp.Real) -> fp.Real:
            with fp.FP64:
                return fp.fmax(1, a)

        for q in (second, first):
            assert 'std::signbit' not in CppCompiler().compile(
                q, arg_types=[RealType(fp.FP64)])

    def test_a_later_step_of_a_fold_keeps_the_term_alone(self):
        """The middle operand cannot be zero, so the first step needs no term;
        the accumulator it produces can be, so the second step does."""
        @fp.fpy
        def q(a: fp.Real, c: fp.Real) -> fp.Real:
            with fp.FP64:
                return min(a, 1, c)

        out = CppCompiler().compile(q, arg_types=[RealType(fp.FP64)] * 2)
        assert out.count('std::signbit') == 1


_BIN_DRIVER = r'''
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
int main(int argc, char** argv) {
    (void) argc;
    uint64_t ba = (uint64_t) std::strtoull(argv[1], nullptr, 16);
    uint64_t bb = (uint64_t) std::strtoull(argv[2], nullptr, 16);
    double a, b; std::memcpy(&a, &ba, 8); std::memcpy(&b, &bb, 8);
    double r = q(a, b);
    uint64_t out; std::memcpy(&out, &r, 8);
    std::printf("%016lx\n", (unsigned long) out);
    return 0;
}
'''

_PAIRS = [
    (-0.0, 0.0), (0.0, -0.0), (-0.0, -0.0), (0.0, 0.0), (1.0, 2.0), (2.0, 1.0),
    (-1.0, 1.0), (float('inf'), 1.0), (-float('inf'), 1.0), (5.0, 5.0),
    (float('nan'), 1.0),      # takes the else arm, so the guard itself is live
]


class TestMinWithoutPropagationStillAgrees:
    """The predicate alone has to be the whole operation.

    Bit-exact, because what the NaN select is *not* protecting is the signed-zero
    half of the predicate -- and ``min(-0.0, +0.0)`` differing from ``+0.0`` is
    invisible to a value comparison.
    """

    def test_value_for_value(self):
        if _CXX is None:
            pytest.skip('no C++ compiler')

        @fp.fpy(ctx=fp.REAL)
        def q(a: fp.Real, b: fp.Real) -> fp.Real:
            if not fp.isnan(a) and not fp.isnan(b):
                with fp.FP64:
                    y = fp.fmin(a, b)
            else:
                y = 0
            return y

        src = CppCompiler().compile(q, arg_types=[RealType(fp.FP64)] * 2)
        assert 'quiet_NaN' not in src
        with tempfile.TemporaryDirectory() as td:
            from fpy2.backend.cpp.utils import CPP_HEADERS
            cpp, exe = Path(td) / 'm.cpp', Path(td) / 'm'
            cpp.write_text('\n'.join(CPP_HEADERS) + '\n' + src + _BIN_DRIVER)
            build = subprocess.run(
                [_CXX, '-std=c++17', '-O0', '-o', str(exe), str(cpp)],
                capture_output=True, text=True)
            assert build.returncode == 0, build.stderr[-2000:]

            bad = []
            for a, b in _PAIRS:
                r = subprocess.run(
                    [str(exe), f'{_bits(a):016x}', f'{_bits(b):016x}'],
                    capture_output=True, text=True)
                got, want = int(r.stdout.strip(), 16), _bits(float(q(a, b)))
                if got != want:
                    g = struct.unpack('<d', struct.pack('<Q', got))[0]
                    bad.append(f'({a!r},{b!r}) cpp {g!r} vs py {float(q(a, b))!r}')
        assert not bad, '; '.join(bad)


class TestStorageIsNotTheContext:
    """A class describes the FPy value; the guard protects C++ on its storage.

    The two come apart exactly where a rounding *produces* a special, so the
    analysis has to round through the target context rather than pass the operand
    class along.  It does, which is why the guard below survives.
    """

    def test_narrowing_to_a_bounded_format_can_make_an_infinity(self):
        assert float(fp.FP32.round(1e300)) == float('inf')

    def test_so_a_guard_after_a_narrowing_round_stays(self):
        """``v`` is finite by the branch, but ``y`` need not be: `FP32` overflows
        at ``1e300``.  Passing the operand's class through would have dropped the
        second guard and left ``std::trunc`` on an infinity."""
        @fp.fpy(ctx=fp.REAL)
        def q(v: fp.Real) -> fp.Real:
            if fp.isfinite(v):
                with fp.FP32:
                    y = fp.round(v)
                with FLOAT_STORAGE:
                    z = fp.round(y)
            else:
                z = 0
            return z

        assert 'undefined for this value' in _asserts(_emit(q))


class TestAgreesWithTheInterpreter:
    """Removing a guard must change neither the value nor where it aborts."""

    @pytest.mark.parametrize('ctx', [
        pytest.param(FLOAT_STORAGE, id='float_storage'),
        pytest.param(INT_STORAGE, id='integer_storage'),
        pytest.param(WITH_SPECIALS, id='with_specials'),
        pytest.param(fp.SINT8, id='sint8'),
    ])
    def test_the_guarded_program_value_for_value(self, ctx):
        if _CXX is None:
            pytest.skip('no C++ compiler')
        _, guarded = _round_pair(ctx)
        src = _emit(guarded)
        with tempfile.TemporaryDirectory() as td:
            exe = _build(src, guarded.name, td)
            bad = []
            for x in _INPUTS:
                bits = struct.unpack('<Q', struct.pack('<d', x))[0]
                r = subprocess.run([str(exe), f'{bits:016x}'],
                                   capture_output=True, text=True)
                try:
                    want, py_ok = float(guarded(x)), True
                except Exception:
                    want, py_ok = None, False
                if (r.returncode == 0) != py_ok:
                    bad.append(f'{x:g}: cpp '
                               f'{"accepts" if r.returncode == 0 else "aborts"}, '
                               f'py {"accepts" if py_ok else "raises"}')
                elif py_ok and float(r.stdout) != want:
                    bad.append(f'{x:g}: cpp {r.stdout.strip()} vs py {want:g}')
        assert not bad, '; '.join(bad[:6])
