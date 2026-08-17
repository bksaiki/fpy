"""
Guards the emitter drops because a branch already ruled the value out.

Every assertion here protects an operation against a NaN or an infinity, and
every one of them is unnecessary where the program has already tested for those.
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

import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import pytest

import fpy2 as fp
from fpy2.backend.cpp import CppCompiler
from fpy2.number import MPBFixedContext
from fpy2.types import RealType

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
