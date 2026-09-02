"""
``Min``/``Max`` are IEEE 754-2019 ``minimum``/``maximum``, emitted inline.

``std::fmin``/``fmax`` are a different function on two counts: they *ignore* a
NaN where these propagate it, and they leave the choice between ``-0.0`` and
``+0.0`` unspecified -- libstdc++ compiles the variable-operand path to
``(a < b) ? a : b``, so ``fmin(-0.0, +0.0)`` gives ``+0.0`` where ``minimum``
gives ``-0.0``.

Emitted inline rather than as an ``fpy::`` template, so the emitted program
depends on ``std::`` alone.
"""

import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import pytest

import fpy2 as fp
from fpy2.backend.cpp import CppCompiler
from fpy2.backend.cpp.utils import CPP_HEADERS, CPP_HELPERS
from fpy2.types import ListType, RealType

_CXX = shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')

_NAN, _INF = float('nan'), float('inf')

_PAIRS = [
    (_NAN, 1.0), (1.0, _NAN), (_NAN, _NAN), (_NAN, -1.0), (0.0, _NAN),
    (-0.0, 0.0), (0.0, -0.0), (-0.0, -0.0), (0.0, 0.0),
    (1.0, 2.0), (2.0, 1.0), (-1.0, 1.0), (1.0, -1.0),
    (_INF, 1.0), (-_INF, 1.0), (_INF, -_INF), (_NAN, _INF), (_INF, _INF),
]

_LISTS = [
    [1.0], [_NAN], [-0.0], [1.0, 2.0], [2.0, 1.0], [-0.0, 0.0], [0.0, -0.0],
    [1.0, _NAN, 2.0], [_NAN, 1.0], [3.0, -1.0, 2.0], [_INF, -_INF],
    [-0.0, 0.0, -0.0], [5.0, 5.0],
]

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

_RED_DRIVER = r'''
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
int main(int argc, char** argv) {
    std::vector<double> xs;
    for (int k = 1; k < argc; ++k) {
        uint64_t b = (uint64_t) std::strtoull(argv[k], nullptr, 16);
        double v; std::memcpy(&v, &b, 8); xs.push_back(v);
    }
    double r = q(xs);
    uint64_t out; std::memcpy(&out, &r, 8);
    std::printf("%016lx\n", (unsigned long) out);
    return 0;
}
'''


def _bits(x: float) -> int:
    return struct.unpack('<Q', struct.pack('<d', x))[0]


def _binary(is_min: bool):
    if is_min:
        @fp.fpy
        def q(a: fp.Real, b: fp.Real) -> fp.Real:
            with fp.FP64:
                return fp.fmin(a, b)
    else:
        @fp.fpy
        def q(a: fp.Real, b: fp.Real) -> fp.Real:
            with fp.FP64:
                return fp.fmax(a, b)
    return q


def _reduction(is_min: bool):
    if is_min:
        @fp.fpy
        def q(xs) -> fp.Real:
            with fp.FP64:
                return min(xs)
    else:
        @fp.fpy
        def q(xs) -> fp.Real:
            with fp.FP64:
                return max(xs)
    return q


def _binary_nonzero(is_min: bool):
    """As :func:`_binary`, with a literal operand no zero can equal.  ``b`` is
    unused, so one driver serves both."""
    if is_min:
        @fp.fpy
        def q(a: fp.Real, b: fp.Real) -> fp.Real:
            with fp.FP64:
                return fp.fmin(a, 1)
    else:
        @fp.fpy
        def q(a: fp.Real, b: fp.Real) -> fp.Real:
            with fp.FP64:
                return fp.fmax(a, 1)
    return q


def _diff_pairs(q) -> None:
    """Run the compiled *q* on :data:`_PAIRS` and diff against the interpreter."""
    if _CXX is None:
        pytest.skip('no C++ compiler')
    src = CppCompiler().compile(q, arg_types=[RealType(fp.FP64)] * 2)
    with tempfile.TemporaryDirectory() as td:
        exe = _build(src, _BIN_DRIVER, td)
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


def _build(src: str, driver: str, td: str) -> Path:
    cpp, exe = Path(td) / 'm.cpp', Path(td) / 'm'
    cpp.write_text('\n'.join(CPP_HEADERS) + '\n' + src + driver)
    build = subprocess.run([_CXX, '-std=c++17', '-O0', '-o', str(exe), str(cpp)],
                           capture_output=True, text=True)
    assert build.returncode == 0, build.stderr[-2000:]
    return exe


class TestAgreesWithTheInterpreter:
    """Bit-exact, so the sign of a zero is not lost in the comparison."""

    @pytest.mark.parametrize('is_min', [True, False], ids=['min', 'max'])
    def test_binary(self, is_min):
        _diff_pairs(_binary(is_min))

    @pytest.mark.parametrize('is_min', [True, False], ids=['min', 'max'])
    def test_binary_without_the_signbit_term(self, is_min):
        """A non-zero second operand drops the term, and ``a`` still ranges over
        both zeros -- so the values it was protecting are exercised without it."""
        q = _binary_nonzero(is_min)
        assert 'std::signbit' not in CppCompiler().compile(
            q, arg_types=[RealType(fp.FP64)] * 2)
        _diff_pairs(q)

    @pytest.mark.parametrize('is_min', [True, False], ids=['min', 'max'])
    def test_reduction(self, is_min):
        """The fold in ``_emit_amin_amax`` uses the same emitter, inside a loop
        where the accumulator is read twice per step."""
        if _CXX is None:
            pytest.skip('no C++ compiler')
        q = _reduction(is_min)
        src = CppCompiler().compile(q, arg_types=[ListType(RealType(fp.FP64))])
        with tempfile.TemporaryDirectory() as td:
            exe = _build(src, _RED_DRIVER, td)
            bad = []
            for xs in _LISTS:
                r = subprocess.run(
                    [str(exe)] + [f'{_bits(v):016x}' for v in xs],
                    capture_output=True, text=True)
                got, want = int(r.stdout.strip(), 16), _bits(float(q(xs)))
                if got != want:
                    g = struct.unpack('<d', struct.pack('<Q', got))[0]
                    bad.append(f'{xs!r} cpp {g!r} vs py {float(q(xs))!r}')
        assert not bad, '; '.join(bad)


class TestTheInterpreterReference:
    """The premise the differentials above compare against -- interpreter only."""

    def test_a_nan_propagates(self):
        """``std::fmin(NaN, 1.0)`` is ``1.0``; ``minimum`` is a NaN."""
        assert _binary(True)(_NAN, 1.0).isnan
        assert _binary(True)(1.0, _NAN).isnan
        assert _binary(False)(_NAN, 1.0).isnan

    def test_a_negative_zero_wins_a_min(self):
        """Either operand order, where libstdc++'s ``(a < b) ? a : b`` gives
        ``+0.0`` for one of them."""
        for a, b in ((-0.0, 0.0), (0.0, -0.0)):
            assert _binary(True)(a, b).s, (a, b)      # min is -0.0
            assert not _binary(False)(a, b).s, (a, b)  # max is +0.0


class TestNoSupportLibrary:
    def test_nothing_emits_fpy(self):
        src = CppCompiler().compile(
            _binary(True), arg_types=[RealType(fp.FP64)] * 2)
        assert 'fpy::' not in src

    def test_the_helper_block_is_empty(self):
        assert CPP_HELPERS == ''
        assert CppCompiler().helpers() == ''

    def test_a_min_program_compiles_without_helpers(self):
        if _CXX is None:
            pytest.skip('no C++ compiler')
        src = CppCompiler().compile(
            _binary(False), arg_types=[RealType(fp.FP64)] * 2)
        with tempfile.TemporaryDirectory() as td:
            cpp = Path(td) / 'm.cpp'
            cpp.write_text('\n'.join(CPP_HEADERS) + '\n' + src)
            r = subprocess.run([_CXX, '-std=c++17', '-fsyntax-only', str(cpp)],
                               capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]


class TestTheNaryFold:
    def test_three_operands_nest_two_steps(self):
        """Each step's result becomes the next step's first operand, which the
        predicate names twice -- so the intermediate has to be bound, and the
        emitter binds it itself."""
        @fp.fpy
        def q(a: fp.Real, b: fp.Real, c: fp.Real) -> fp.Real:
            with fp.FP64:
                return fp.fmin(a, fp.fmin(b, c))

        out = CppCompiler().compile(q, arg_types=[RealType(fp.FP64)] * 3)
        assert out.count('std::signbit(') == 2
        # the inner select is emitted once and read twice, not duplicated
        assert out.count('std::isnan(b)') == 1


class TestIntegerPathUnchanged:
    def test_an_integer_reduction_keeps_the_library_form(self):
        """The fold in ``_emit_amin_amax`` chooses per storage kind too."""
        @fp.fpy
        def q(xs) -> fp.Real:
            with fp.SINT32:
                return min(xs)

        out = CppCompiler().compile(q, arg_types=[ListType(RealType(fp.SINT32))])
        assert 'std::min(' in out
        assert 'signbit' not in out

    def test_integers_keep_the_library_form(self):
        """No NaN and no signed zero, so ``std::min``/``max`` is already exact
        and there is nothing to inline."""
        @fp.fpy
        def q(a: fp.Real, b: fp.Real) -> fp.Real:
            with fp.SINT32:
                return fp.fmin(a, b)

        out = CppCompiler().compile(q, arg_types=[RealType(fp.SINT32)] * 2)
        assert 'std::min(' in out
        assert 'isnan' not in out and 'signbit' not in out
