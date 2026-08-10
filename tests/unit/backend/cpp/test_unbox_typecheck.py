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
from fpy2.backend.cpp.unbox import UnboxMode
from fpy2.module import Module
from fpy2.types import ListType, RealType

R = RealType(fp.FP64)
L = ListType(R)
N = ListType(L)
_CXX = shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
_OPTS = ['-std=c++11', '-O0', '-Wall', '-Wextra', '-Werror=return-type']

pytestmark = pytest.mark.skipif(_CXX is None, reason='no C++ compiler')


def _typecheck(module: Module, *, unbox=UnboxMode.ALLOW, optimize=True) -> str:
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
    _typecheck(m, unbox=UnboxMode.ALLOW, optimize=optimize)


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


# --------------------------------------------------------------------------
# Found while writing the tests above, and none of it is an unbox bug -- all of
# it reproduces with `UnboxMode.NEVER`.  It is the first thing that fell out of
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


@pytest.mark.parametrize('unbox', [UnboxMode.ALLOW, UnboxMode.NEVER])
@pytest.mark.parametrize(
    'func,arg_types', JOIN_CASES, ids=[f.name for f, _ in JOIN_CASES],
)
def test_a_joined_place_has_one_element_type(func, arg_types, unbox):
    m = Module()
    m.add(func, ctx=fp.FP64, arg_types=list(arg_types))
    _typecheck(m, unbox=unbox)
    # One element type throughout -- which one it is, is storage selection's
    # business, and `UnboxMode.NEVER` spells the same list `fpy::list`.  Read off
    # the function alone; the runtime helpers are templates and would
    # contribute a `T`.
    body = CppCompiler(unbox=unbox).compile_module(m)
    elts = re.findall(r'(?:std::vector|fpy::(?:list|make_list))<(\w+)>', body)
    assert len(set(elts)) == 1, body


# --------------------------------------------------------------------------
# The other half: a *variable* reaching a joined place.  Emitting the
# contributors at the place's type cannot reach one -- a variable's storage was
# fixed by its own definition -- so the backend converts at the boundary.

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


# --------------------------------------------------------------------------
# And the boundary of that: a narrower list something *else* also names.
#
# The conversion above rebuilds the list, which is invisible only because
# nothing else holds that buffer.  Once something does, the rebuilt copy is a
# different list from the one the other references see, and there is no sound
# lowering -- so the compiler refuses.  These are all legal FPy programs; the
# limitation is the C++ backend's.  `docs/todos/backend-cpp.md`
# records what it would take to compile them.

@fp.fpy
def s_local_in_a_list(c: fp.Real, y: fp.Real) -> list[fp.Real]:
    """`xs` is a name *and* a slot of `zss`, so it keeps its handle -- and a
    handle cannot be rebuilt without `zss` still naming the old buffer."""
    with fp.FP64:
        xs = [1.0, 2.0]
        zss = [xs]
        zss[0][0] = 1.0
        if c > 0:
            return xs
        else:
            return [y]


@fp.fpy
def s_local_in_a_tuple(c: fp.Real, y: fp.Real):
    """The same, one level in: the tuple's field fixes the list's type."""
    with fp.FP64:
        xs = [1.0, 2.0]
        if c > 0:
            return (xs, 1.0)
        else:
            return ([y], y)


@fp.fpy
def s_mixed_precision_local(c: fp.Real, y: fp.Real) -> list[fp.Real]:
    """The format the program asked for, not a narrowing accident: `lo`'s
    elements are FP32-rounded *values*, and `zss` holds the same buffer."""
    with fp.FP32:
        lo = [fp.round(y), fp.round(y)]
    with fp.FP64:
        zss = [lo]
        if c > 0:
            return lo
        else:
            return [y]


@fp.fpy
def s_parameter(xs: list[fp.Real], c: fp.Real, y: fp.Real) -> list[fp.Real]:
    """A parameter: the caller holds the same list, and the signature already
    committed to its element type, so neither side can move."""
    with fp.FP64:
        if c > 0:
            return xs
        else:
            return [y]


@fp.fpy
def s_alias(xs: list[fp.Real], c: fp.Real, y: fp.Real) -> list[fp.Real]:
    """`ys = xs` binds `const auto&`, so `ys` has no buffer of its own."""
    with fp.FP64:
        ys = xs
        if c > 0:
            return ys
        else:
            return [y]


@fp.fpy
def s_projection(xss: list[list[fp.Real]], c: fp.Real, y: fp.Real) -> list[fp.Real]:
    """`row = xss[0]` binds `const auto&` to a slot."""
    with fp.FP64:
        row = xss[0]
        if c > 0:
            return row
        else:
            return [y]


@fp.fpy
def s_loop_target(xss: list[list[fp.Real]], c: fp.Real, y: fp.Real) -> list[fp.Real]:
    """A loop target binds `const auto&` to each element."""
    with fp.FP64:
        out = [y]
        for row in xss:
            if c > 0:
                out = row
        return out


L32 = ListType(RealType(fp.FP32))
N32 = ListType(L32)

SHARED_CASES = [
    (s_local_in_a_list, [R, R]),
    (s_local_in_a_tuple, [R, R]),
    (s_mixed_precision_local, [R, R]),
    (s_parameter, [L32, R, R]),
    (s_alias, [L32, R, R]),
    (s_projection, [N32, R, R]),
    (s_loop_target, [N32, R, R]),
]


@pytest.mark.parametrize(
    'func,arg_types', SHARED_CASES, ids=[f.name for f, _ in SHARED_CASES],
)
def test_a_shared_narrower_list_is_refused(func, arg_types):
    """Refusing is the whole point: the alternative is silently unsharing.

    The last three matter most, because there the mismatch would be invisible.
    A reference binding is spelled `const auto&`, so nothing in the emitted text
    states its element type -- `const auto& ys = xs;` followed by `return ys;`
    would hand back an `fpy::list<float>` as an `fpy::list<double>` and only the
    C++ compiler would object.
    """
    m = Module()
    m.add(func, ctx=fp.FP64, arg_types=list(arg_types))
    with pytest.raises(CppCompileError, match='is shared'):
        CppCompiler(unbox=UnboxMode.ALLOW).compile_module(m)


def test_a_callee_result_is_refused_rather_than_unshared():
    """The refusal names the callee, since that is the only place to fix it.

    `g`'s return type is fixed by `g`'s own body, so nothing on the caller's
    side can change it, and rebuilding the result would copy a list out of its
    aliases.  Returned straight out of the call there is no local to blame, so
    the message has to reach for the callee's name instead.
    """
    @fp.fpy
    def g(zs: list[fp.Real]) -> list[fp.Real]:
        with fp.FP32:
            return zs

    @fp.fpy
    def f(xs: list[fp.Real], c: fp.Real, y: fp.Real) -> list[fp.Real]:
        with fp.FP64:
            if c > 0:
                return g(xs)
            else:
                return [y]

    m = Module()
    m.add(g, ctx=fp.FP32, arg_types=[L32])
    m.add(f, ctx=fp.FP64, arg_types=[L32, R, R])
    with pytest.raises(CppCompileError, match='is shared') as exc:
        CppCompiler(unbox=UnboxMode.ALLOW).compile_module(m)
    msg = str(exc.value)
    assert '`g`' in msg, msg
    assert 'element type' in msg, msg


def test_a_callees_parameter_at_a_join_is_refused():
    """The refusal fires inside a callee too, reached through the call.

    A function compiled code calls keeps one signature on both sides -- the same
    rule `unbox` states for representations -- so the callee's narrower list
    parameter is as immovable as an entry point's.  The fix is on the caller's
    side: pass the wider list and specialization carries the format down, which
    `test_widening_the_call_site_is_a_real_workaround` pins.
    """
    @fp.fpy
    def g(zs: list[fp.Real], c: fp.Real, y: fp.Real) -> list[fp.Real]:
        with fp.FP64:
            return zs if c > 0 else [y]

    @fp.fpy
    def f(xs: list[fp.Real], c: fp.Real, y: fp.Real) -> fp.Real:
        with fp.FP64:
            w = g(xs, c, y)
            return w[0]

    m = Module()
    m.add(f, ctx=fp.FP64, arg_types=[L32, R, R])
    with pytest.raises(CppCompileError, match='is shared'):
        CppCompiler(unbox=UnboxMode.ALLOW).compile_module(m)


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
            std::tuple<fpy::list<double>, uint8_t> t = ...;   // disagrees
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
    # `fpy::make_list`.
    body = CppCompiler().compile_module(m)
    tuples = set(re.findall(r'std::tuple<[^>]*>', body))
    assert len(tuples) == 1, f'declaration and return type disagree: {tuples}'
    # ...and here the answer should be unboxed: nothing else holds the list, so
    # a handle would be a pointless allocation.
    assert 'fpy::list' not in body, body
    assert 'make_shared' not in body, body


def test_widening_the_call_site_is_a_real_workaround():
    """The escape hatch the refusal above recommends, pinned.

    A callee's formats already follow its call site, so passing the wider list
    specializes `g` at the wider format and nothing needs converting.  Since
    that is the only advice the error can give, it must not quietly stop being
    true -- if specialization ever stopped tracking argument formats, the
    message would become a lie and this is what would notice.
    """
    @fp.fpy
    def g(zs: list[fp.Real]) -> list[fp.Real]:
        with fp.FP32:
            return zs

    @fp.fpy
    def f(xs: list[fp.Real], c: fp.Real, y: fp.Real) -> list[fp.Real]:
        with fp.FP64:
            ws = g(xs)
            return ws if c > 0 else [y]

    m = Module()
    m.add(f, ctx=fp.FP64, arg_types=[L, R, R])      # a *FP64* list, not FP32
    _typecheck(m)
    # `g` came along specialized at the caller's format: had it stayed FP32 the
    # body would say `float` somewhere, and the refusal would have fired.
    body = CppCompiler(unbox=UnboxMode.ALLOW).compile_module(m)
    assert 'float' not in body, body
    assert 'double' in body, body
