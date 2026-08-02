"""Representation across a compiled-to-compiled call boundary.

`signature()` disagreeing with `compile_module()` was bug #3, and what pins it
is one two-function module checked for one parameter of one callee.  The rule
it is testing ("a function compiled code calls keeps its handles") applies to
*every* function in *every* module, on both sides of the boundary and to the
return type as well as the parameters -- and `signature`'s new `_find_spec`
now has to pick the right spec out of a list, which is a new way to be wrong.

Nothing in the unit suite ever hands a multi-function module to a C++
compiler, so a caller and callee that disagree about a representation is a
hard error with no test between it and a user.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import fpy2 as fp
import pytest

from fpy2.backend.cpp.compiler import CppCompileError, CppCompiler
from fpy2.backend.cpp.types import CppList
from fpy2.module import Module
from fpy2.types import ListType, RealType

R = RealType(fp.FP64)
L = ListType(R)
N = ListType(L)
_CXX = shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
_OPTS = ['-std=c++11', '-O0', '-Wall', '-Wextra']


def _typecheck(cc: CppCompiler, m: Module) -> str:
    src = '\n'.join([*cc.headers(), cc.helpers(), cc.compile_module(m)])
    if _CXX is None:
        pytest.skip('no C++ compiler')
    with tempfile.TemporaryDirectory() as td:
        cpp = Path(td) / 'm.cpp'
        cpp.write_text(src)
        r = subprocess.run([_CXX, *_OPTS, '-fsyntax-only', str(cpp)],
                           capture_output=True, text=True)
    assert r.returncode == 0, (
        f'module does not typecheck:\n{r.stderr[-3000:]}\n--- emitted ---\n{src}'
    )
    return src


# --------------------------------------------------------------------------
# A call graph with every shape that matters: a leaf that mutates, a leaf that
# returns a fresh list, a middle function that is both caller and callee, and
# an entry that is only a caller.

@fp.fpy
def g_mutates(zs: list[fp.Real]) -> fp.Real:
    with fp.FP64:
        zs[0] = 1.0
        return zs[0]


@fp.fpy
def g_fresh(n: fp.Real) -> list[fp.Real]:
    """A callee handing a *fresh* list back: its result is the caller's
    storage, so the return type is a boundary too."""
    with fp.FP64:
        return [n, n + 1.0]


@fp.fpy
def g_middle(ws: list[fp.Real]) -> fp.Real:
    """Both a caller and a callee."""
    with fp.FP64:
        a = g_mutates(ws)
        bs = g_fresh(a)
        return bs[0] + ws[0]


@fp.fpy
def g_entry(vs: list[fp.Real]) -> fp.Real:
    with fp.FP64:
        return g_middle(vs) + g_mutates(vs)


@fp.fpy
def g_nested_callee(mss: list[list[fp.Real]]) -> fp.Real:
    with fp.FP64:
        mss[0][0] = 2.0
        return mss[0][0]


@fp.fpy
def g_nested_caller(mss: list[list[fp.Real]]) -> fp.Real:
    with fp.FP64:
        return g_nested_callee(mss)


def _module():
    m = Module()
    m.add(g_entry, ctx=fp.FP64, arg_types=[L])
    m.add(g_middle, ctx=fp.FP64, arg_types=[L])
    m.add(g_mutates, ctx=fp.FP64, arg_types=[L])
    m.add(g_fresh, ctx=fp.FP64, arg_types=[R])
    m.add(g_nested_caller, ctx=fp.FP64, arg_types=[N])
    m.add(g_nested_callee, ctx=fp.FP64, arg_types=[N])
    return m


ENTRIES = [g_entry, g_middle, g_mutates, g_fresh, g_nested_caller, g_nested_callee]
ARGS = {
    'g_entry': [L], 'g_middle': [L], 'g_mutates': [L], 'g_fresh': [R],
    'g_nested_caller': [N], 'g_nested_callee': [N],
}


def test_the_whole_module_typechecks():
    """A caller and callee that disagree about a representation is a C++ type
    error, and nothing in the unit suite runs a C++ compiler.

    Regression class: the `is_called` rule, the `call`-site boundary rule, or
    `annotate_return` stops covering one side of a boundary.
    """
    _typecheck(CppCompiler(), _module())


@pytest.mark.parametrize('func', ENTRIES, ids=[f.name for f in ENTRIES])
def test_signature_matches_the_module_for_every_function(func):
    """Bug #3 generalized: `signature(f, module=m)` is what an embedding
    program builds arguments from, so it has to be right for *every* `f` in
    the module -- not just the last one, which is all `specs[-1]` gave you.

    Regression class: `_find_spec` picks the wrong spec (name mangling, a
    duplicate name, a changed emission order) and a native caller builds a
    `std::vector` for a parameter that is really an `fpy::list`.
    """
    m = _module()
    cc = CppCompiler()
    params, ret = cc.signature(func, ctx=fp.FP64, arg_types=ARGS[func.name], module=m)
    src = cc.compile_module(m)

    # locate the emitted definition and compare it token for token
    sig_line = next(
        (ln for ln in src.splitlines() if f' {func.name}(' in ln), None,
    )
    assert sig_line is not None, f'{func.name} not emitted:\n{src}'
    assert sig_line.startswith(ret.format() + ' '), (
        f'{func.name}: signature() says the result is `{ret.format()}` but '
        f'the module emitted `{sig_line}`'
    )
    for p in params:
        assert p.format() in sig_line, (
            f'{func.name}: signature() says a parameter is `{p.format()}` but '
            f'the module emitted `{sig_line}`'
        )


def test_a_called_functions_result_is_what_its_caller_expects():
    """`g_fresh` returns a list nothing else refers to, so it hands back a
    value — in a module too, since the callers now read that from its
    signature rather than assuming a handle.

    Regression class: the two ends disagree about the return type.  Whether
    they settle on a value or a handle matters less than that they settle on
    the same one, so this asserts agreement and typechecks the result.
    """
    cc = CppCompiler()
    _p, alone = cc.signature(g_fresh, ctx=fp.FP64, arg_types=[R])
    assert isinstance(alone, CppList) and not alone.boxed, alone.format()

    m = _module()
    _p, in_module = cc.signature(g_fresh, ctx=fp.FP64, arg_types=[R], module=m)
    out = _typecheck(cc, m)
    assert f'{in_module.format()} g_fresh(' in out, out


def test_signature_without_a_module_still_describes_a_lone_function():
    """The other direction: passing no module must not start requiring one."""
    cc = CppCompiler()
    params, _ret = cc.signature(g_nested_callee, ctx=fp.FP64, arg_types=[N])
    assert params[0].format() == 'std::vector<std::vector<double>>'


def test_asking_for_a_function_that_is_not_in_the_module_is_an_error():
    """`_find_spec` falls back to `specs[0]` when there is exactly one spec.
    With more than one it must refuse rather than silently answer about some
    other function -- a wrong signature is worse than no signature.
    """
    m = _module()
    cc = CppCompiler()

    @fp.fpy
    def stranger(qs: list[fp.Real]) -> fp.Real:
        with fp.FP64:
            return qs[0]

    with pytest.raises(CppCompileError):
        cc.signature(stranger, ctx=fp.FP64, arg_types=[L], module=m)


def test_a_two_function_module_is_not_answered_by_emission_order():
    """With `specs[-1]` the answer depended on which function happened to be
    emitted last.  What it *is* matters less than that it does not move."""
    cc = CppCompiler()
    seen = set()
    for order in ([g_mutates, g_middle], [g_middle, g_mutates]):
        m = Module()
        for f in order:
            m.add(f, ctx=fp.FP64, arg_types=[L])
        params, _ = cc.signature(
            g_mutates, ctx=fp.FP64, arg_types=[L], module=m,
        )
        seen.add(params[0].format())
    assert len(seen) == 1, f'answer depends on module order: {seen}' 
