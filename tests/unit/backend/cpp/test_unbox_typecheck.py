"""Mixed per-level representations, and the return type.

`unbox` decides each list *level* independently, so
`std::vector<std::shared_ptr<std::vector<double>>>` is legal and does
occur.  Every unit test in
`test_unbox.py` stops at the emitted *string* -- nothing hands the result to a
C++ compiler, and the differential harness only ever sees whole-corpus
programs whose levels happen to agree.  A mixed nesting that does not typecheck
is a hard error nobody would see until a user hit it.

Also here: the return type.  `annotate_return` is a third place a
representation is chosen, and a `return` whose type disagrees with the
function's is a compile error, not a wrong answer.
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import fpy2 as fp
import pytest

from fpy2.backend.cpp.compiler import CppCompiler
from fpy2.backend.cpp.types import CppList
from fpy2.backend.cpp.unbox import UnboxMode
from fpy2.module import Module
from fpy2.types import ListType, RealType

R = RealType(fp.FP64)
L = ListType(R)
N = ListType(L)
_CXX = shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
_OPTS = ['-std=c++11', '-O0', '-Wall', '-Wextra', '-Werror=return-type']

pytestmark = pytest.mark.skipif(_CXX is None, reason='no C++ compiler')


def _typecheck(module: Module, *, unbox: UnboxMode = UnboxMode.ALLOW) -> str:
    """Compile *module* to a translation unit and put it through the C++
    compiler.  Returns the source on success; fails the test on a diagnostic."""
    cc = CppCompiler(unbox=unbox)
    src = '\n'.join([*cc.headers(), cc.helpers(), cc.compile_module(module)])
    with tempfile.TemporaryDirectory() as td:
        cpp = Path(td) / 'u.cpp'
        cpp.write_text(src)
        r = subprocess.run(
            [_CXX, *_OPTS, '-fsyntax-only', str(cpp)],
            capture_output=True, text=True,
        )
    assert r.returncode == 0, (
        f'emitted C++ does not typecheck (unbox={unbox}):'
        f'\n{r.stderr[-3000:]}\n--- emitted ---\n{src[-3000:]}'
    )
    return src


def _levels(ty) -> list[bool]:
    out = []
    while isinstance(ty, CppList):
        out.append(ty.boxed)
        ty = ty.elt
    return out


# --------------------------------------------------------------------------

@fp.fpy
def m_hand_out_a_row(xss: list[list[fp.Real]]) -> list[fp.Real]:
    """Outer unboxed (nothing else names it), inner boxed (a row is handed to
    the caller).  The mixed type: a vector of boxed rows."""
    with fp.FP64:
        xss[0][0] = 99
        return xss[1]


@fp.fpy
def m_fresh_outer_shared_inner(ys: list[fp.Real]) -> fp.Real:
    """A fresh outer list holding the caller's row twice."""
    with fp.FP64:
        zss = [ys, ys]
        zss[0][0] = 5.0
        acc = 0.0
        for row in zss:
            acc = acc + row[0]
        return acc


@fp.fpy
def m_three_deep(xsss: list[list[list[fp.Real]]]) -> list[list[fp.Real]]:
    """Three levels, and the middle one is handed out."""
    with fp.FP64:
        xsss[0][0][0] = 1.0
        return xsss[0]


@fp.fpy
def m_return_fresh_nested(x: fp.Real) -> list[list[fp.Real]]:
    """A fresh nested result: the return transfers ownership, so both levels
    may be values -- and the return type has to say so."""
    with fp.FP64:
        return [[x, x], [x, x]]


@fp.fpy
def m_return_a_parameter(xs: list[fp.Real]) -> list[fp.Real]:
    """Returning a parameter leaves the caller with two handles: sharing, so
    the return type must keep its handle and match the parameter."""
    with fp.FP64:
        xs[0] = 1.0
        return xs


@fp.fpy
def m_two_returns_disagree(xs: list[fp.Real], c: fp.Real) -> list[fp.Real]:
    """Two `return`s, one fresh and one the parameter.  They are one C++
    return type, so the conservative one has to win at both."""
    with fp.FP64:
        if c > 0:
            return [c, c]
        else:
            return xs


@fp.fpy
def m_list_of_tuple_of_list(x: fp.Real) -> fp.Real:
    """A list inside a tuple inside a list: `regions_in_a_tuple` has to reach
    it, or the tuple's field and the list's own type disagree."""
    with fp.FP64:
        ys = [x, x]
        ts = [(ys, x)]
        zs = fp.fst(ts[0])
        zs[0] = 7.0
        return ys[0]


CASES = [
    ('hand_out_a_row', m_hand_out_a_row, [N]),
    ('fresh_outer_shared_inner', m_fresh_outer_shared_inner, [L]),
    ('three_deep', m_three_deep, [ListType(N)]),
    ('return_fresh_nested', m_return_fresh_nested, [R]),
    ('return_a_parameter', m_return_a_parameter, [L]),
    ('two_returns_disagree', m_two_returns_disagree, [L, R]),
    ('list_of_tuple_of_list', m_list_of_tuple_of_list, [R]),
]


@pytest.mark.parametrize('name,func,arg_types', CASES, ids=[c[0] for c in CASES])
def test_representation_stressing_programs_typecheck(name, func, arg_types):
    """Every per-level and per-return representation choice has to produce a
    program C++ accepts.

    Regression class: a level, a return, or a container field gets stamped
    with a representation that disagrees with the storage around it.  Loud,
    but only if something actually runs a C++ compiler -- and no unit test
    does.
    """
    m = Module()
    m.add(func, ctx=fp.FP64, arg_types=list(arg_types))
    _typecheck(m, unbox=UnboxMode.ALLOW)


def test_mixed_nesting_is_actually_produced():
    """The guard on the test above: if nothing ever comes out mixed, the
    typecheck is pinning a case that does not exist."""
    cc = CppCompiler(unbox=UnboxMode.ALLOW)
    params, ret = cc.signature(m_hand_out_a_row, ctx=fp.FP64, arg_types=[N])
    assert _levels(params[0]) == [False, True], (
        f'expected a mixed nesting, got {params[0].format()}'
    )
    assert _levels(ret) == [True]


def test_return_type_matches_the_parameter_it_hands_back():
    """`return xs` on a list parameter: two names for one C++ type, decided in
    two places (`_read` for the parameter, `annotate_return` for the result).

    Regression class: the two disagree and the emitted `return` needs a
    conversion that does not exist.
    """
    cc = CppCompiler(unbox=UnboxMode.ALLOW)
    params, ret = cc.signature(m_return_a_parameter, ctx=fp.FP64, arg_types=[L])
    assert params[0].format() == ret.format(), (
        f'parameter is `{params[0].format()}` but the result is `{ret.format()}`'
    )


def test_a_fresh_nested_result_is_fully_unboxed():
    """The positive direction for the return type: a transfer of ownership
    costs nothing, so both levels may be values.  Without this the typecheck
    above would pass on an all-boxed answer."""
    cc = CppCompiler()
    _params, ret = cc.signature(m_return_fresh_nested, ctx=fp.FP64, arg_types=[R])
    assert _levels(ret) == [False, False], ret.format()


@fp.fpy
def m_tuple_with_list_via_a_local(y: fp.Real):
    """A tuple holding a list, bound to a local before being returned."""
    with fp.FP64:
        t = [y, y], 1.0
        return t


def test_a_declaration_agrees_with_what_is_handed_through_it():
    """One representation per place, including a list inside a tuple.

    Regression: `unbox` had *two* traversals stamping representations onto a
    type, and only one descended into tuples.  The declaration came from the one
    that did not, the return type from the one that did, so this emitted

        std::tuple<std::vector<double>, uint8_t> f() {
            std::tuple<std::shared_ptr<std::vector<double>>, uint8_t> t = ...;
            return t;
        }

    which no C++ compiler accepts.  Returned *inline* both paths agreed, which is
    why every other tuple case compiled and this one did not.
    """
    m = Module()
    m.add(m_tuple_with_list_via_a_local, ctx=fp.FP64, arg_types=[R])
    _typecheck(m)

    # The point is agreement, not which answer: the two spellings of the tuple
    # type in the emitted function must be the same one.  Read the function
    # alone -- the runtime helpers legitimately use `make_shared` for
    # allocation of its own.
    body = CppCompiler().compile_module(m)
    tuples = set(re.findall(r'std::tuple<[^>]*>', body))
    assert len(tuples) == 1, f'declaration and return type disagree: {tuples}'
    # ...and here the answer should be unboxed: nothing else holds the list, so
    # a handle would be a pointless allocation.
    assert 'std::shared_ptr' not in body, body
    assert 'make_shared' not in body, body
