"""
`Round` into a fixed-point context narrower than its storage.

Storage *contains* a format rather than equalling it, so the cast that rounds
also keeps values the format would have bounded.  A context bounded at 100 in an
``int8_t`` wraps at 128, not at 101 -- so this emitted a bare
``static_cast<int8_t>(v)`` and returned 120 where the interpreter raised
``OverflowError``, and for an operand past the type's range the conversion was
outright undefined.

The exception is a context the op table dispatches on: `SINT8`'s format *is*
``int8_t``'s and its ``WRAP`` *is* what the cast does, so that one needs no help.
Whole contexts have to be compared rather than formats -- the same -128..127
values under ``ASSERT`` still need the assertion.
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

A = fp.OverflowMode.ASSERT

# bound 100 is narrower than either storage type; `enable_neg_zero` picks which
_INT_STORAGE = fp.MPBFixedContext(
    -1, fp.RealFloat(exp=0, c=100), rm=fp.RM.RTZ, overflow=A,
    enable_neg_zero=False)
_FLOAT_STORAGE = fp.MPBFixedContext(
    -1, fp.RealFloat(exp=0, c=100), rm=fp.RM.RTZ, overflow=A,
    enable_neg_zero=True)
# format-equal to `int8_t`, but `ASSERT` where `SINT8` says `WRAP`
_ASYMMETRIC = fp.MPBFixedContext(
    -1, fp.RealFloat(exp=0, c=127), neg_maxval=fp.RealFloat(s=True, exp=0, c=128),
    rm=fp.RM.RTZ, overflow=A, enable_neg_zero=False)

_INPUTS = [
    0.0, -0.0, 1.0, -1.0, 50.0, 99.5, 100.0, 100.7, -100.7, 101.0,
    127.0, 128.0, 200.0, 255.0, 256.0, -128.0, -129.0, 300.0,
    65535.0, 65536.0, 1e300, float('inf'), float('-inf'), float('nan'),
]


def _round_fn(ctx):
    @fp.fpy(ctx=fp.REAL)
    def q(v: fp.Real) -> fp.Real:
        with ctx:
            y = fp.round(v)
        return y
    return q


def _emit(ctx, arg_ctx=fp.FP64) -> str:
    return CppCompiler().compile(_round_fn(ctx), arg_types=[RealType(arg_ctx)])


class TestAgreesWithTheInterpreter:
    @pytest.mark.parametrize('ctx', [
        pytest.param(_INT_STORAGE, id='int_storage'),
        pytest.param(_FLOAT_STORAGE, id='float_storage'),
        pytest.param(_ASYMMETRIC, id='asymmetric_assert'),
    ])
    def test_value_for_value(self, ctx):
        """Same value where both succeed, and an abort exactly where the
        interpreter raises."""
        if _CXX is None:
            pytest.skip('no C++ compiler')
        q = _round_fn(ctx)
        src = _emit(ctx)
        with tempfile.TemporaryDirectory() as td:
            cpp, exe = Path(td) / 'm.cpp', Path(td) / 'm'
            cpp.write_text('\n'.join(CPP_HEADERS) + '\n' + src + _DRIVER)
            build = subprocess.run(
                [_CXX, '-std=c++17', '-O0', '-o', str(exe), str(cpp)],
                capture_output=True, text=True)
            assert build.returncode == 0, build.stderr[-2000:]

            bad = []
            for x in _INPUTS:
                bits = struct.unpack('<Q', struct.pack('<d', x))[0]
                r = subprocess.run([str(exe), f'{bits:016x}'],
                                   capture_output=True, text=True)
                try:
                    want, py_ok = float(q(x)), True
                except Exception:
                    want, py_ok = None, False
                if (r.returncode == 0) != py_ok:
                    bad.append(f'{x:g}: cpp '
                               f'{"accepts" if r.returncode == 0 else "aborts"}, '
                               f'py {"accepts" if py_ok else "raises"}')
                elif py_ok and float(r.stdout) != want:
                    bad.append(f'{x:g}: cpp {r.stdout.strip()} vs py {want:g}')
        assert not bad, '; '.join(bad[:6])


class TestTheOriginalHole:
    def test_the_bound_is_asserted_in_integer_storage(self):
        """This emitted only ``int8_t y = static_cast<int8_t>(v);``."""
        out = _emit(_INT_STORAGE)
        assert 'overflow occurred' in out
        # 120 is representable in `int8_t` but not in this context
        with pytest.raises(Exception):
            _round_fn(_INT_STORAGE)(120.0)

    def test_the_bound_is_asserted_on_the_rounded_value(self):
        """``100.7`` rounds to ``100`` under ``RTZ`` and is *in* bounds, so the
        test cannot be applied to the operand."""
        assert float(_round_fn(_INT_STORAGE)(100.7)) == 100.0
        out = _emit(_INT_STORAGE)
        assert 'std::trunc(' in out

    def test_specials_are_guarded_before_an_integer_conversion(self):
        """A NaN or infinity converted to an integer type is undefined -- on
        x86-64 it gives ``INT_MIN`` -- where the interpreter raises."""
        out = _emit(_INT_STORAGE)
        assert 'std::isfinite' in out


class TestEdgeRulesAreRefused:
    @pytest.mark.parametrize('overflow', [
        fp.OverflowMode.SATURATE, fp.OverflowMode.WRAP, fp.OverflowMode.OVERFLOW,
    ], ids=['saturate', 'wrap', 'overflow'])
    @pytest.mark.parametrize('neg_zero', [True, False], ids=['float', 'integer'])
    def test_a_rule_this_lowering_does_not_implement(self, overflow, neg_zero):
        """Emitting the rounding and dropping the rule is a miscompile: at 120,
        `SATURATE` says 100 and `WRAP` says -81, and the old output said 120."""
        ctx = fp.MPBFixedContext(
            -1, fp.RealFloat(exp=0, c=100), rm=fp.RM.RTZ, overflow=overflow,
            enable_neg_zero=neg_zero)
        with pytest.raises(CppCompileError, match='has no C'):
            _emit(ctx)


class TestNativeContextsAreUntouched:
    @pytest.mark.parametrize('ctx, ty', [
        pytest.param(fp.SINT8, 'int8_t', id='sint8'),
        pytest.param(fp.UINT16, 'uint16_t', id='uint16'),
    ])
    def test_a_native_integer_context_keeps_its_bare_cast(self, ctx, ty):
        """Its whole context matches what the cast does, wrapping included --
        `SINT8` maps 128 to -128 and -129 to 127 in both languages."""
        out = _emit(ctx)
        assert f'static_cast<{ty}>' in out
        assert 'assert(' not in out

    def test_wrapping_matches_for_a_native_context(self):
        q = _round_fn(fp.SINT8)
        assert float(q(128.0)) == -128.0
        assert float(q(-129.0)) == 127.0
        assert float(q(200.0)) == -56.0


class TestIntegerOperand:
    def test_no_pointless_float_tests_for_an_integer_operand(self):
        """An ``int32_t`` is always finite and always integral, and ``fabs``
        would promote it to ``double`` -- lossy past 2**53.  Only its magnitude
        is in question, tested with an exact two-sided comparison."""
        out = _emit(_INT_STORAGE, arg_ctx=fp.SINT32)
        assert 'isfinite' not in out
        assert 'std::trunc' not in out
        assert 'fabs' not in out
        assert '-100 <= v && v <= 100' in out
