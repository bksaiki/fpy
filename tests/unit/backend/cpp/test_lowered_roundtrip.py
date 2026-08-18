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
    char buf[64];
    while (std::scanf("%63s", buf) == 1) {
        uint64_t b = (uint64_t) std::strtoull(buf, nullptr, 16);
        SRCTY x; std::memcpy(&x, &b, sizeof(x));
        /* the result is a value of the *target* format, exact in either
           storage, so widening it makes one comparison serve both sources */
        double r = (double) q(x);
        uint64_t out; std::memcpy(&out, &r, 8);
        /* NaN payloads are unspecified in both languages; canonicalize */
        if (r != r) out = 0x7ff8000000000000ull;
        std::printf("%016llx\n", (unsigned long long) out);
    }
    return 0;
}
'''

_CTYPE = {32: 'float', 64: 'double'}


def _in_bits(v: fp.Float, width: int) -> int:
    """`v` as source-format bits, zero-extended to 64 so one hex column feeds
    either driver."""
    if width == 32:
        if v.isnan:
            return 0x7fc00000
        if v.isinf:
            return 0xff800000 if v.s else 0x7f800000
        return struct.unpack('<I', struct.pack('<f', float(v)))[0]
    return _bits(v)


def _bits(v: fp.Float) -> int:
    """`v` as FP64 bits, NaN canonicalized -- payload *and* sign, both
    unspecified in either language.  A restored sign is still tested: the
    ``neg_zero`` target turns a NaN into a signed finite value, which is
    compared as-is.
    """
    if v.isnan:
        return 0x7ff8000000000000
    if v.isinf:
        return 0xfff0000000000000 if v.s else 0x7ff0000000000000
    return struct.unpack('<Q', struct.pack('<d', float(v)))[0]


def _inputs(src) -> list[fp.Float]:
    """Values of the source format covering every class the lowering branches
    on."""
    B = src.maxval().as_real()
    out = [
        fp.Float(c=0), fp.Float(c=0, s=True), fp.Float(isnan=True),
        fp.Float(isinf=True), fp.Float(isinf=True, s=True),
    ]
    points = [
        (5, 2047),          # FP16 maxval
        (5, 2048),          # just past it
        (4, 4095),          # the RNE tie above maxval
        (B.exp, B.c),       # the source's own maxval
        (src.expmin, 1),
        (fp.FP16.expmin, 1), (fp.FP16.expmin - 1, 1),
        (fp.FP16.emin, 1), (fp.FP16.emin - 1, 3),
        (0, 1), (3, 3), (-4, 11), (-30, 1), (10, 1365),
    ]
    for exp, c in points:
        for s in (False, True):
            out.append(src.round(RealFloat(s=s, exp=exp, c=c)))
    return out


def _run(target, src=fp.FP32) -> None:
    """Lower ``round`` into *target* from a *src* source, compile, and diff."""
    if _CXX is None:
        pytest.skip('no C++ compiler')

    @fp.fpy(ctx=fp.REAL)
    def q(x: fp.Real) -> fp.Real:
        with target:
            y = fp.round(x)
        return y

    ref = st.monomorphize(q, args=[RealType(src)])
    low = st.rescale_fixed(st.float_to_fixed(
        st.unfold_overflow(ref, early_check=True)))

    cc = CppCompiler()
    mod = Module()
    mod.add(low)
    width = src.nbits
    driver = _DRIVER.replace('SRCTY', _CTYPE[width])
    text = '\n'.join([*cc.headers(), cc.helpers(), cc.compile_module(mod), driver])

    xs = _inputs(src)
    stdin = '\n'.join(f'{_in_bits(x, width):016x}' for x in xs)
    want = [f'{_bits(ref(x)):016x}' for x in xs]

    with tempfile.TemporaryDirectory() as td:
        cpp, exe = Path(td) / 'm.cpp', Path(td) / 'm'
        cpp.write_text(text)
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


_TARGETS = [
    fp.FP16,
    fp.IEEEContext(5, 16, fp.RoundingMode.RTZ),
    fp.IEEEContext(5, 16, fp.RoundingMode.RTP),
    fp.IEEEContext(5, 16, fp.RoundingMode.RTN),
    fp.IEEEContext(5, 16, fp.RoundingMode.RNA),
    fp.IEEEContext(5, 16, fp.RoundingMode.RAZ),
    fp.IEEEContext(5, 16, fp.RoundingMode.RTO),
    fp.IEEEContext(5, 16, fp.RoundingMode.RTE),
    fp.IEEEContext(4, 8),
    fp.IEEEContext(5, 16, fp.RoundingMode.RNE, fp.OverflowMode.SATURATE),
    fp.MX_E5M2,
    fp.MX_E4M3,
    fp.MX_E2M1,
    EFloatContext(4, 8, False, EFloatNanKind.NEG_ZERO, 0),
]
_TARGET_IDS = ['fp16_rne', 'fp16_rtz', 'fp16_rtp', 'fp16_rtn', 'fp16_rna',
               'fp16_raz', 'fp16_rto', 'fp16_rte',
               'ieee_4_8', 'ieee_saturating', 'e5m2', 'e4m3', 'e2m1', 'neg_zero']


class TestLoweredRoundtrip:
    """Compiled output against the interpreter, bit for bit."""

    @pytest.mark.parametrize('target', _TARGETS, ids=_TARGET_IDS)
    def test_matches_the_interpreter(self, target):
        _run(target)

    @pytest.mark.parametrize('target', _TARGETS, ids=_TARGET_IDS)
    def test_matches_the_interpreter_from_fp64(self, target):
        """The source format that matters, and the one this path could not
        reach until branch refinement read the guards.

        Storage selection used to fail here: the scale-in was inferred at
        ``2 ** 2108`` against a true ``[2 ** 10, 2 ** 11)``, and its finest digit
        at ``2 ** -1090``.  Compiling is only half of it -- this is the half that
        says the answer is right.
        """
        _run(target, src=fp.FP64)

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
