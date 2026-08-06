"""Caller-observable effects: does the caller see what FPy says it should?

The differential harness compares *return values*.  FPy's list semantics are
about the caller's own storage: `xs[i] = e` in a callee is visible to whoever
passed `xs`.  Nothing checks that, which is why a rebound unboxed parameter
silently swallowed a write.

The oracle here is the **boxed compilation of the same function**.  Boxing is
the reference semantics -- an `fpy::list<T>` shares, always -- so for any
program, `unbox=True` and `unbox=False` must produce the same return value
*and* leave the caller's vectors in the same state.  Restricted to flat
`list[Real]` parameters because those are the ones a native caller can share
under both representations (`fpy::borrow(v)` vs `v` itself); a nested boxed
list can only be `copy_in`'d, so there is nothing to compare.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import fpy2 as fp
import pytest

from fpy2.backend.cpp.compiler import CppCompiler
from fpy2.backend.cpp.types import CppList
from tests.infra.backend.cpp import CPP_INTEROP
from fpy2.module import Module
from fpy2.types import BoolType, ListType, RealType

R = RealType(fp.FP64)
_CXX = shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
_OPTS = ['-std=c++11', '-O0']

pytestmark = pytest.mark.skipif(_CXX is None, reason='no C++ compiler')


def _adapt(param, name: str) -> str:
    """How a native caller hands *name* to a parameter of type *param*."""
    if isinstance(param, CppList):
        return name if not param.boxed else f'fpy::borrow({name})'
    return name


def _run(func, arg_types, args, *, unbox: bool) -> tuple[str, list[list[float]]]:
    """``(printed result, post-call contents of each list argument)``.

    Builds native storage the *caller* owns, hands it over however the
    signature says to, and reads the caller's own vectors back afterwards.
    """
    cc = CppCompiler(unbox=unbox)
    m = Module()
    m.add(func, ctx=fp.FP64, arg_types=list(arg_types))
    params, _ret = cc.signature(
        func, ctx=fp.FP64, arg_types=list(arg_types), module=m,
    )

    decls, actuals, prints = [], [], []
    for i, (p, ty, val) in enumerate(zip(params, arg_types, args)):
        if isinstance(ty, ListType):
            init = ', '.join(f'{v!r}' for v in val)
            decls.append(f'std::vector<double> a{i}; '
                         f'{"".join(f"a{i}.push_back({v!r}); " for v in val)}'
                         f'(void){init.count(",")};')
            actuals.append(_adapt(p, f'a{i}'))
            prints.append(
                f'std::printf("ARG{i}");'
                f'for (size_t k = 0; k < a{i}.size(); ++k) '
                f'std::printf(" %.17g", a{i}[k]);'
                f'std::printf("\\n");'
            )
        else:
            decls.append(f'double a{i} = {val!r};')
            actuals.append(f'a{i}')

    src = '\n'.join([
        *cc.headers(), '#include <cstdio>', cc.helpers(), CPP_INTEROP,
        cc.compile_module(m),
        'int main() {',
        *decls,
        f'double r = {func.name}({", ".join(actuals)});',
        'std::printf("RET %.17g\\n", r);',
        *prints,
        'return 0; }',
    ])

    with tempfile.TemporaryDirectory() as td:
        cpp = Path(td) / 'k.cpp'
        cpp.write_text(src)
        exe = Path(td) / 'k.exe'
        b = subprocess.run([_CXX, *_OPTS, '-o', str(exe), str(cpp)],
                           capture_output=True, text=True)
        assert b.returncode == 0, (
            f'unbox={unbox} did not compile:\n{b.stderr[-2500:]}\n--- src ---\n{src}'
        )
        r = subprocess.run([str(exe)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr

    ret = ''
    out: list[list[float]] = []
    for line in r.stdout.splitlines():
        if line.startswith('RET '):
            ret = line[4:]
        elif line.startswith('ARG'):
            out.append([float(t) for t in line.split()[1:]])
    return ret, out


# --------------------------------------------------------------------------
# Programs.  Each is a shape where the *only* difference between right and
# wrong is what the caller's vector holds afterwards.

@fp.fpy
def c_scale_in_place(xs: list[fp.Real], k: fp.Real) -> fp.Real:
    """Plain mutation through a parameter."""
    with fp.FP64:
        acc = 0.0
        for i in range(len(xs)):
            xs[i] = xs[i] * k
            acc = acc + xs[i]
        return acc


@fp.fpy
def c_write_then_rebind(xs: list[fp.Real], c: fp.Real) -> fp.Real:
    """Bug #1's shape: mutate the caller's, *then* rebind locally.

    An unboxed list cannot express this with one variable -- the emitter
    passes a rebound parameter by value.  The return value is right either
    way; only the caller's vector tells them apart.
    """
    with fp.FP64:
        xs[0] = 99.0
        if c > 0:
            xs = [7.0]
        return xs[0]


@fp.fpy
def c_alias_then_write(xs: list[fp.Real]) -> fp.Real:
    """`ys = xs` is a reference binding; the write must still reach the caller."""
    with fp.FP64:
        ys = xs
        ys[0] = 42.0
        return xs[0]


@fp.fpy
def c_rebind_only(xs: list[fp.Real], c: fp.Real) -> fp.Real:
    """Rebinding alone must *not* reach the caller -- the opposite error."""
    with fp.FP64:
        if c > 0:
            xs = [7.0, 8.0]
        return xs[0]


@fp.fpy
def c_write_after_rebind(xs: list[fp.Real], c: fp.Real) -> fp.Real:
    """Rebind, then write.  Whether the caller sees it depends on the branch,
    which is precisely the semantics an unboxed copy cannot reproduce."""
    with fp.FP64:
        if c > 0:
            xs = [7.0, 8.0]
        xs[1] = 5.0
        return xs[1]


@fp.fpy
def c_two_params_written(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
    with fp.FP64:
        for i in range(len(xs)):
            xs[i] = xs[i] + ys[i]
            ys[i] = xs[i] * 2.0
        return xs[0] + ys[0]


@fp.fpy
def c_slice_is_a_copy(xs: list[fp.Real]) -> fp.Real:
    """A slice is a fresh list, so writing it must *not* reach the caller."""
    with fp.FP64:
        ys = xs[0:2]
        ys[0] = 123.0
        return ys[0] + xs[0]


_L = ListType(R)
CASES = [
    ('scale_in_place', c_scale_in_place, [_L, R], [[1.0, 2.0, 3.0], 3.0]),
    ('write_then_rebind_taken', c_write_then_rebind, [_L, R], [[1.0, 2.0], 1.0]),
    ('write_then_rebind_untaken', c_write_then_rebind, [_L, R], [[1.0, 2.0], -1.0]),
    ('alias_then_write', c_alias_then_write, [_L], [[1.0, 2.0]]),
    ('rebind_only_taken', c_rebind_only, [_L, R], [[1.0, 2.0], 1.0]),
    ('rebind_only_untaken', c_rebind_only, [_L, R], [[1.0, 2.0], -1.0]),
    ('write_after_rebind_taken', c_write_after_rebind, [_L, R], [[1.0, 2.0], 1.0]),
    ('write_after_rebind_untaken', c_write_after_rebind, [_L, R], [[1.0, 2.0], -1.0]),
    ('two_params_written', c_two_params_written, [_L, _L], [[1.0, 2.0], [3.0, 4.0]]),
    ('slice_is_a_copy', c_slice_is_a_copy, [_L], [[1.0, 2.0, 3.0]]),
]


@pytest.mark.parametrize(
    'name,func,arg_types,args', CASES, ids=[c[0] for c in CASES],
)
def test_unboxing_preserves_caller_visible_effects(name, func, arg_types, args):
    """Unboxing is an optimization, so it must be unobservable -- including
    through the caller's own storage, which is the half the differential
    harness cannot see.

    Regression class: any change that lets an unboxed list become a *copy* the
    caller does not share -- a new by-value declaration rule, a lost
    `_is_rebound` check, a discount that stops mirroring the emitter.  Loud:
    the boxed run is the oracle and it disagrees numerically.
    """
    boxed = _run(func, arg_types, args, unbox=False)
    unboxed = _run(func, arg_types, args, unbox=True)
    assert boxed == unboxed, (
        f'{name}: boxed gave {boxed}, unboxed gave {unboxed}'
    )
