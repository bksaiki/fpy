"""Mixed per-level representations, and the return type.

`unbox` decides each list *level* independently, so
`std::vector<fpy::list<double>>` is legal and does occur.  Every unit test in
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

from fpy2.backend.cpp.compiler import CppCompileError, CppCompiler
from fpy2.backend.cpp.types import CppList
from fpy2.module import Module
from fpy2.types import ListType, RealType

R = RealType(fp.FP64)
L = ListType(R)
N = ListType(L)
_CXX = shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
_OPTS = ['-std=c++11', '-O0', '-Wall', '-Wextra', '-Werror=return-type']

pytestmark = pytest.mark.skipif(_CXX is None, reason='no C++ compiler')


def _typecheck(module: Module, *, unbox=True, optimize=True) -> str:
    """Compile *module* to a translation unit and put it through the C++
    compiler.  Returns the source on success; fails the test on a diagnostic."""
    cc = CppCompiler(unbox=unbox, optimize=optimize)
    src = '\n'.join([*cc.headers(), cc.helpers(), cc.compile_module(module)])
    with tempfile.TemporaryDirectory() as td:
        cpp = Path(td) / 'u.cpp'
        cpp.write_text(src)
        r = subprocess.run(
            [_CXX, *_OPTS, '-fsyntax-only', str(cpp)],
            capture_output=True, text=True,
        )
    assert r.returncode == 0, (
        f'emitted C++ does not typecheck (unbox={unbox}, optimize={optimize}):'
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
    the caller).  The mixed type `std::vector<fpy::list<double>>`."""
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


@pytest.mark.parametrize('optimize', [True, False])
@pytest.mark.parametrize('name,func,arg_types', CASES, ids=[c[0] for c in CASES])
def test_representation_stressing_programs_typecheck(name, func, arg_types, optimize):
    """Every per-level and per-return representation choice has to produce a
    program C++ accepts.

    Regression class: a level, a return, or a container field gets stamped
    with a representation that disagrees with the storage around it.  Loud,
    but only if something actually runs a C++ compiler -- and no unit test
    does.
    """
    m = Module()
    m.add(func, ctx=fp.FP64, arg_types=list(arg_types))
    _typecheck(m, unbox=True, optimize=optimize)


def test_mixed_nesting_is_actually_produced():
    """The guard on the test above: if nothing ever comes out mixed, the
    typecheck is pinning a case that does not exist."""
    cc = CppCompiler()
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
    cc = CppCompiler()
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


# --------------------------------------------------------------------------
# Found while writing the tests above, and none of it is an unbox bug -- all of
# it reproduces with `unbox=False`.  It is the first thing that fell out of
# running a C++ compiler over emitted code, which nothing else does.
#
# `format_infer` picks a bound per expression and joins where several values
# reach one place.  The join was never pushed back *down*, so each contributor
# kept its own narrower bound and the backend gave one place two storages.
# Scalars survive that on implicit conversion; `std::vector` has no converting
# constructor across element types, so these are hard errors.

@fp.fpy
def j_two_returned_literals(c: fp.Real) -> list[fp.Real]:
    """Two returns: `{1.5, 2.5}` narrows differently from `{3}`."""
    with fp.FP64:
        if c > 0:
            return [1.5, 2.5]
        else:
            return [3.0]


@fp.fpy
def j_nested_literal() -> fp.Real:
    """No returns involved -- a nested literal's own rows disagree."""
    with fp.FP64:
        xss = [[1.5], [3.0, 4.0]]
        return xss[0][0]


@fp.fpy
def j_ternary_over_lists(c: fp.Real) -> fp.Real:
    """One C++ ternary, so one type across both arms."""
    with fp.FP64:
        xs = [1.5, 2.5] if c > 0 else [3.0]
        return xs[0]


@fp.fpy
def j_list_inside_a_returned_tuple(c: fp.Real):
    """`std::tuple` does convert across element types -- but only when its
    elements do, and two `std::vector`s do not."""
    with fp.FP64:
        if c > 0:
            return ([1.5, 2.5], 1.5)
        else:
            return ([3.0], 3.0)


@fp.fpy
def j_comprehension_against_a_literal(c: fp.Real) -> list[fp.Real]:
    """A comprehension builds its vector element by element, so its body is a
    contributor too."""
    with fp.FP64:
        if c > 0:
            return [1.5 for _ in range(3)]
        else:
            return [3.0]


JOIN_CASES = [
    (j_two_returned_literals, [R]),
    (j_nested_literal, []),
    (j_ternary_over_lists, [R]),
    (j_list_inside_a_returned_tuple, [R]),
    (j_comprehension_against_a_literal, [R]),
]


@pytest.mark.parametrize('unbox', [True, False])
@pytest.mark.parametrize(
    'func,arg_types', JOIN_CASES, ids=[f.name for f, _ in JOIN_CASES],
)
def test_a_joined_place_has_one_element_type(func, arg_types, unbox):
    m = Module()
    m.add(func, ctx=fp.FP64, arg_types=list(arg_types))
    _typecheck(m, unbox=unbox)
    # One element type throughout -- which one it is, is storage selection's
    # business, and `unbox=False` spells the same list `fpy::list`.  Read off
    # the function alone; the runtime helpers are templates and would
    # contribute a `T`.
    body = CppCompiler(unbox=unbox).compile_module(m)
    elts = re.findall(r'(?:std::vector|fpy::(?:list|make_list))<(\w+)>', body)
    assert len(set(elts)) == 1, body


# --------------------------------------------------------------------------
# The other half: a *variable* reaching a joined place.  `_push_format` cannot
# re-decide one -- its storage was fixed by its own definition -- so the
# backend converts at the boundary instead.

@fp.fpy
def v_narrower_variable(c: fp.Real, y: fp.Real) -> list[fp.Real]:
    """`xs` narrows to `uint8_t` on its own; the other return is `double`."""
    with fp.FP64:
        xs = [1.0, 2.0]
        if c > 0:
            return xs
        else:
            return [y]


@fp.fpy
def v_narrower_variable_nested(c: fp.Real, y: fp.Real) -> list[list[fp.Real]]:
    """Nested, where the range constructor does not reach -- the rows need
    converting too."""
    with fp.FP64:
        xss = [[1.0, 2.0]]
        if c > 0:
            return xss
        else:
            return [[y]]


CONVERT_CASES = [v_narrower_variable, v_narrower_variable_nested]


@pytest.mark.parametrize(
    'func', CONVERT_CASES, ids=[f.name for f in CONVERT_CASES],
)
def test_a_narrower_variable_is_converted_at_the_boundary(func):
    m = Module()
    m.add(func, ctx=fp.FP64, arg_types=[R, R])
    _typecheck(m)


# A boxed list is the one thing conversion must not rebuild: a new allocation
# is a different object, and the handle is there precisely so FPy's aliasing
# survives.  Both cases below are refused; only the first is refused for a
# reason that will not go away.

@fp.fpy
def r_genuinely_shared(c: fp.Real, y: fp.Real) -> list[fp.Real]:
    """`xs` is a name *and* a container slot, so rebuilding it would hand the
    caller something `zss` no longer aliases."""
    with fp.FP64:
        xs = [1.0, 2.0]
        zss = [xs]
        zss[0][0] = 1.0
        if c > 0:
            return xs
        else:
            return [y]


@fp.fpy
def r_boxed_only_by_conservatism(c: fp.Real, y: fp.Real):
    """Nothing here outlives the return -- `xs` keeps its handle only because
    a name plus a container field reads as two places.  That is the
    `named_in_tuple` entry in `docs/todos/unboxing-gaps.md`; closing it turns
    this case into a conversion."""
    with fp.FP64:
        xs = [1.0, 2.0]
        if c > 0:
            return (xs, 1.0)
        else:
            return ([y], y)


REFUSE_CASES = [r_genuinely_shared, r_boxed_only_by_conservatism]


@pytest.mark.parametrize(
    'func', REFUSE_CASES, ids=[f.name for f in REFUSE_CASES],
)
def test_a_shared_list_is_refused_rather_than_unshared(func):
    m = Module()
    m.add(func, ctx=fp.FP64, arg_types=[R, R])
    with pytest.raises(CppCompileError, match='rebuilding a shared list'):
        CppCompiler().compile_module(m)
