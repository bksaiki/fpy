"""
`fp.cast` asserts exactness *in the context*; storage only contains a context.

``_emit_exact_cast`` tests the cast by round-tripping through storage
(``assert(x == static_cast<T>(x))``), which answers the context's question only
where the two formats agree.  Under a context bounded at 1024 whose representable
values are the integers, both ``cast(2048.0)`` (past the bound) and ``cast(0.5)``
(not an integer) raise in the interpreter and both satisfy that round-trip.

The interesting check here is the differential: compile the cast, run it once per
input, and compare "did it abort?" against "did the interpreter raise?".  Source
text alone cannot show that the assertions admit the right set.
"""

import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import pytest

import fpy2 as fp
from fpy2.backend.cpp import CppCompiler
from fpy2.backend.cpp.compiler import CppCompileError
from fpy2.backend.cpp.utils import CPP_HEADERS
from fpy2.types import RealType

_CXX = shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')

# takes the input as a hex bit pattern in argv so an abort is per-value
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

_BOUNDED_1024 = fp.MPBFixedContext(-1, fp.RealFloat(exp=10, c=1), rm=fp.RM.RTZ)
_ASYMMETRIC = fp.MPBFixedContext(
    -1, fp.RealFloat(exp=0, c=127),
    neg_maxval=fp.RealFloat(s=True, exp=0, c=128), rm=fp.RM.RTZ)

# every edge the assertions decide on: both zeros, the specials, non-integers,
# and off-by-one at each bound
_INPUTS = [
    0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.5, 0.25,
    127.0, 128.0, -128.0, -129.0,
    1023.0, 1024.0, 1025.0, 2048.0, 65535.0, 65536.0,
    1e300, float('inf'), float('-inf'), float('nan'),
]


def _cast_fn(ctx):
    @fp.fpy(ctx=fp.REAL)
    def q(v: fp.Real) -> fp.Real:
        with ctx:
            y = fp.cast(v)
        return y
    return q


def _emit(ctx, arg_ctx=fp.FP64) -> str:
    return CppCompiler().compile(_cast_fn(ctx), arg_types=[RealType(arg_ctx)])


def _interpreter_accepts(ctx, x: float) -> bool:
    try:
        _cast_fn(ctx)(x)
        return True
    except Exception:
        return False


class TestAgreesWithTheInterpreter:
    """The assertions must admit exactly the representable values."""

    @pytest.mark.parametrize('ctx', [
        pytest.param(_BOUNDED_1024, id='bounded_1024'),
        pytest.param(_ASYMMETRIC, id='asymmetric'),
        pytest.param(fp.SINT8, id='sint8'),
        pytest.param(fp.UINT16, id='uint16'),
    ])
    def test_it_aborts_exactly_where_the_interpreter_raises(self, ctx):
        if _CXX is None:
            pytest.skip('no C++ compiler')
        src = _emit(ctx)
        with tempfile.TemporaryDirectory() as td:
            cpp, exe = Path(td) / 'm.cpp', Path(td) / 'm'
            cpp.write_text('\n'.join(CPP_HEADERS) + '\n' + src + _DRIVER)
            build = subprocess.run(
                [_CXX, '-std=c++17', '-O0', '-o', str(exe), str(cpp)],
                capture_output=True, text=True)
            assert build.returncode == 0, build.stderr[-2000:]

            disagree = []
            for x in _INPUTS:
                bits = struct.unpack('<Q', struct.pack('<d', x))[0]
                r = subprocess.run([str(exe), f'{bits:016x}'],
                                   capture_output=True, text=True)
                if (r.returncode == 0) != _interpreter_accepts(ctx, x):
                    disagree.append((x, r.returncode == 0))

        assert not disagree, '; '.join(
            f'{x!r}: compiled {"accepts" if ok else "aborts"}, '
            f'interpreter {"accepts" if not ok else "raises"}'
            for x, ok in disagree
        )


class TestValuesExactInStorageOnly:
    @pytest.mark.parametrize('x, why', [
        (2048.0, 'past the bound'),
        (0.5, 'not an integer'),
    ])
    def test_a_value_exact_in_storage_but_not_in_the_context(self, x, why):
        assert not _interpreter_accepts(_BOUNDED_1024, x), why
        # exact in `float`, so the storage round-trip alone said nothing
        assert float(struct.unpack('<f', struct.pack('<f', x))[0]) == x

        out = _emit(_BOUNDED_1024)
        assert 'std::trunc' in out          # only integers are representable
        assert '1024' in out                # and only up to the bound


class TestWhatIsEmitted:
    def test_asymmetric_bounds_get_two_comparisons(self):
        """``fabs`` states one magnitude; these two bounds are independent."""
        out = _emit(_ASYMMETRIC)
        assert '-128 <= v && v <= 127' in out
        assert 'fabs' not in out

    def test_an_integer_operand_needs_only_the_bound(self):
        """An integer is already representable wherever the integers are, and is
        never a NaN or an infinity -- so only its magnitude is in question."""
        out = _emit(fp.SINT8, arg_ctx=fp.SINT32)
        assert 'outside the context\'s bound' in out
        assert 'std::trunc' not in out
        assert 'isfinite' not in out and 'isnan(v)' not in out

    def test_a_float_operand_is_checked_three_ways(self):
        out = _emit(fp.SINT8)
        # specials, integral, bound, and the storage round-trip
        assert out.count('assert(') == 4
        assert 'isfinite' in out
        assert 'std::trunc' in out
        assert '-128 <= v && v <= 127' in out

    def test_a_native_float_context_is_unchanged(self):
        """`FP32`'s format *equals* its storage's, so the round-trip already was
        the containment test and nothing is added."""
        out = _emit(fp.FP32)
        assert 'std::trunc' not in out
        assert 'cast is not exact' not in out
        assert out.count('assert(') == 1


class TestUncheckableIsRefused:
    """At a non-zero position the representable values are multiples of
    ``2 ** (nmin + 1)``; scaling the operand to test that would round it first,
    so there is no test to emit."""

    # integer storage, so `_validate_context_rm` (which checks the position only
    # for float storage) does not catch it first
    _INT_STORAGE = fp.MPBFixedContext(
        3, fp.RealFloat(exp=8, c=1), rm=fp.RM.RTZ, enable_neg_zero=False)

    @pytest.mark.parametrize('arg_ctx', [fp.FP64, fp.SINT32],
                             ids=['float_operand', 'integer_operand'])
    def test_a_non_zero_position_is_refused_not_assumed(self, arg_ctx):
        """Both operand types, because the integer one skipped this check: it
        takes a shortcut past the representability test, and would have claimed
        ``cast(5)`` exact where only multiples of 16 are representable."""
        with pytest.raises(CppCompileError, match='cannot be checked'):
            _emit(self._INT_STORAGE, arg_ctx=arg_ctx)

    def test_float_storage_is_refused_earlier(self):
        """`_validate_context_rm` gets there first for float storage, naming the
        same fix."""
        ctx = fp.MPBFixedContext(-8, fp.RealFloat(exp=4, c=1), rm=fp.RM.RTZ)
        with pytest.raises(CppCompileError, match='digits at position zero'):
            _emit(ctx)
