"""
End-to-end: a float rounding lowered by the scheduling operators, compiled to
C++, must agree with the FPy reference bit for bit.

This is the property the whole lowering path exists for, and the one that
catches what the source-text assertions in ``test_emit_integral_round`` cannot:
that the emitted program *computes* the same thing.  Two real defects would
have been caught here --

- ``_validate_context_rm`` asserting ``EFloatContext`` for any float storage,
  which crashed on a fixed-point context that landed in a float type;
- ``Round`` under such a context emitting ``static_cast<float>``, a float
  narrowing rather than a rounding to an integral value.

The pipeline is ``monomorphize -> unfold_overflow -> float_to_fixed ->
rescale_fixed``; an ``FP32`` source is used because an ``FP64`` one still
exceeds every storage type (see ``docs/todos/symbolic-exponent-inference.md``).
"""

import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import pytest

import fpy2 as fp
import fpy2.strategies as st
from fpy2.backend.cpp.compiler import CppCompiler
from fpy2.module import Module
from fpy2.number import EFloatContext, EFloatNanKind, RealFloat
from fpy2.types import RealType

_CXX = shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
_OPTS = ['-std=c++17', '-O2', '-w']

_DRIVER = r'''
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
int main() {
    char buf[32];
    while (std::scanf("%31s", buf) == 1) {
        uint32_t b = (uint32_t) std::strtoul(buf, nullptr, 16);
        float x; std::memcpy(&x, &b, 4);
        float r = q(x);
        uint32_t out; std::memcpy(&out, &r, 4);
        /* NaN payloads are unspecified in both languages; canonicalize */
        if (r != r) out = 0x7fc00000u;
        std::printf("%08x\n", out);
    }
    return 0;
}
'''


def _bits(v: fp.Float) -> int:
    """`v` as FP32 bits, NaN canonicalized -- payload *and* sign, both
    unspecified in either language.  A restored sign is still tested: the
    ``neg_zero`` target turns a NaN into a signed finite value, which is
    compared as-is.
    """
    if v.isnan:
        return 0x7fc00000
    if v.isinf:
        return 0xff800000 if v.s else 0x7f800000
    return struct.unpack('<I', struct.pack('<f', float(v)))[0]


def _inputs() -> list[fp.Float]:
    """FP32 values covering every class the lowering branches on."""
    B = fp.FP32.maxval().as_real()
    out = [
        fp.Float(c=0), fp.Float(c=0, s=True), fp.Float(isnan=True),
        fp.Float(isinf=True), fp.Float(isinf=True, s=True),
    ]
    grid = [
        (5, 2047),          # FP16 maxval
        (5, 2048),          # just past it
        (4, 4095),          # the RNE tie above maxval
        (B.exp, B.c),       # FP32 maxval
        (fp.FP32.expmin, 1),
        (fp.FP16.expmin, 1), (fp.FP16.expmin - 1, 1),
        (fp.FP16.emin, 1), (fp.FP16.emin - 1, 3),
        (0, 1), (3, 3), (-4, 11), (-30, 1), (10, 1365),
    ]
    for exp, c in grid:
        for s in (False, True):
            out.append(fp.FP32.round(RealFloat(s=s, exp=exp, c=c)))
    return out


def _run(target) -> None:
    """Lower ``round`` into *target* from an FP32 source, compile, and diff."""
    if _CXX is None:
        pytest.skip('no C++ compiler')

    @fp.fpy(ctx=fp.REAL)
    def q(x: fp.Real) -> fp.Real:
        with target:
            y = fp.round(x)
        return y

    ref = st.monomorphize(q, args=[RealType(fp.FP32)])
    low = st.rescale_fixed(st.float_to_fixed(
        st.unfold_overflow(ref, early_check=True)))

    cc = CppCompiler()
    mod = Module()
    mod.add(low)
    src = '\n'.join([*cc.headers(), cc.helpers(), cc.compile_module(mod), _DRIVER])

    xs = _inputs()
    stdin = '\n'.join(f'{_bits(x):08x}' for x in xs)
    want = [f'{_bits(fp.FP32.round(ref(x))):08x}' for x in xs]

    with tempfile.TemporaryDirectory() as td:
        cpp, exe = Path(td) / 'm.cpp', Path(td) / 'm'
        cpp.write_text(src)
        build = subprocess.run([_CXX, *_OPTS, '-o', str(exe), str(cpp)],
                               capture_output=True, text=True)
        assert build.returncode == 0, (
            f'lowered program does not compile:\n{build.stderr[-3000:]}'
        )
        run = subprocess.run([str(exe)], input=stdin,
                             capture_output=True, text=True)
    assert run.returncode == 0, (
        f'lowered program aborted (an assertion fired): rc={run.returncode}'
    )
    got = run.stdout.split()
    assert len(got) == len(want)
    bad = [(x, a, b) for x, a, b in zip(xs, got, want) if a != b]
    assert not bad, (
        f'{len(bad)} of {len(xs)} inputs differ; first few: '
        + '; '.join(f'{x} -> got {a} want {b}' for x, a, b in bad[:5])
    )


class TestLoweredRoundtrip:
    """Compiled output against the interpreter, bit for bit."""

    @pytest.mark.parametrize('target', [
        fp.FP16,
        fp.IEEEContext(5, 16, fp.RoundingMode.RTZ),
        fp.IEEEContext(5, 16, fp.RoundingMode.RTP),
        fp.IEEEContext(5, 16, fp.RoundingMode.RTN),
        fp.IEEEContext(5, 16, fp.RoundingMode.RNA),
        fp.IEEEContext(4, 8),
        fp.IEEEContext(5, 16, fp.RoundingMode.RNE, fp.OverflowMode.SATURATE),
        fp.MX_E5M2,
        fp.MX_E4M3,
        fp.MX_E2M1,
        EFloatContext(4, 8, False, EFloatNanKind.NEG_ZERO, 0),
    ], ids=['fp16_rne', 'fp16_rtz', 'fp16_rtp', 'fp16_rtn', 'fp16_rna',
            'ieee_4_8', 'ieee_saturating', 'e5m2', 'e4m3', 'e2m1', 'neg_zero'])
    def test_matches_the_interpreter(self, target):
        _run(target)

    def test_needs_no_support_library(self):
        """The goal is C++ that depends on nothing of ours.  The lowered
        program uses only ``std::`` and ``std::numeric_limits``, so stripping
        the ``fpy::`` helpers must leave it compiling."""
        if _CXX is None:
            pytest.skip('no C++ compiler')

        @fp.fpy(ctx=fp.REAL)
        def q(x: fp.Real) -> fp.Real:
            with fp.FP16:
                y = fp.round(x)
            return y

        ref = st.monomorphize(q, args=[RealType(fp.FP32)])
        low = st.rescale_fixed(st.float_to_fixed(
            st.unfold_overflow(ref, early_check=True)))
        cc = CppCompiler()
        mod = Module()
        mod.add(low)
        body = cc.compile_module(mod)
        assert 'fpy::' not in body

        src = '\n'.join([*cc.headers(), body])   # no helpers
        with tempfile.TemporaryDirectory() as td:
            cpp = Path(td) / 'm.cpp'
            cpp.write_text(src)
            r = subprocess.run([_CXX, *_OPTS, '-fsyntax-only', str(cpp)],
                               capture_output=True, text=True)
        assert r.returncode == 0, (
            f'needs the support library after all:\n{r.stderr[-2000:]}'
        )
