"""Generated coverage for the shapes the corpus does not have.

Every bug found while building the join/widening machinery lived in one of four
shapes, and the corpus barely contains them.  Of its 217 functions: 6 have more
than one ``return``, 6 have a ternary, 7 have a nested list literal, and **none**
has a list at a format other than FP64.  So the differential harness stayed green
through a nested-literal miscompile, a return-join type disagreement, a
``signature()`` mismatch, and a reference-bound name reporting a storage type its
reference did not have.  Each was found by hand-writing a shape, which is the
thing to automate.

Three axes:

- **Shape** -- hand-enumerated below, because a failure has to be readable.
  Adding one is a single function plus an entry in ``SHAPES``.
- **Format** -- generated.  This is where the corpus has nothing, and it is also
  the cheap axis: a program's formats come entirely from ``arg_types``, so
  one source function yields four programs.
- **Length** -- generated; see ``LENGTHS``.  A concrete length turns a
  parameter into ``std::array`` and collides with in-body literals' own
  sizes at joins.

The assertion is deliberately weak on purpose: a program may legitimately be
*refused* (``CppEmitError``) -- a shared list cannot change element type, and
saying so is correct.  What may never happen is emitting C++ that does not
typecheck.  Since "everything refused" would satisfy that vacuously,
:func:`test_enough_of_the_matrix_compiles` pins the floor.
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

import fpy2 as fp

from fpy2.backend.cpp.compiler import CppCompileError, CppCompiler
from fpy2.backend.cpp.unbox import UnboxMode
from fpy2.module import Module
from fpy2.types import ListType, RealType

_CXX = shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
_OPTS = ['-std=c++11', '-O0', '-Wall', '-Wextra', '-Werror=return-type']

pytestmark = pytest.mark.skipif(_CXX is None, reason='no C++ compiler')


# --------------------------------------------------------------------------
# Shapes.  Each takes one of the four signatures in `SIGS`, so `arg_types` can
# be generated for it.

@fp.fpy
def s_two_literal_returns(c: fp.Real, y: fp.Real) -> list[fp.Real]:
    with fp.FP64:
        if c > 0:
            return [1.5, 2.5]
        else:
            return [y]


@fp.fpy
def s_return_param_or_literal(
    xs: list[fp.Real], c: fp.Real, y: fp.Real,
) -> list[fp.Real]:
    with fp.FP64:
        if c > 0:
            return xs
        else:
            return [y]


@fp.fpy
def s_ternary_literals(c: fp.Real, y: fp.Real) -> fp.Real:
    with fp.FP64:
        zs = [1.5, 2.5] if c > 0 else [y]
        return zs[0]


@fp.fpy
def s_ternary_param(xs: list[fp.Real], c: fp.Real, y: fp.Real) -> fp.Real:
    with fp.FP64:
        zs = xs if c > 0 else [y]
        return zs[0]


@fp.fpy
def s_nested_literal(c: fp.Real, y: fp.Real) -> fp.Real:
    with fp.FP64:
        zss = [[1.5], [y, 3.0]]
        return zss[0][0]


@fp.fpy
def s_three_deep_literal(c: fp.Real, y: fp.Real) -> fp.Real:
    with fp.FP64:
        zsss = [[[1.5]], [[y, 3.0]]]
        return zsss[0][0][0]


@fp.fpy
def s_nested_returns(c: fp.Real, y: fp.Real) -> list[list[fp.Real]]:
    with fp.FP64:
        if c > 0:
            return [[1.5], [2.5]]
        else:
            return [[y]]


@fp.fpy
def s_list_in_a_tuple(c: fp.Real, y: fp.Real):
    with fp.FP64:
        if c > 0:
            return ([1.5, 2.5], 1.5)
        else:
            return ([y], y)


@fp.fpy
def s_two_lists_in_a_tuple(xs: list[fp.Real], c: fp.Real, y: fp.Real):
    with fp.FP64:
        return (xs, [y])


@fp.fpy
def s_tuple_with_list_via_a_local(c: fp.Real, y: fp.Real):
    """Bound to a local first, which is the whole point.

    `unbox` used to stamp representations in two places and only one descended
    into a tuple, so the declaration and the return type disagreed about the
    list field.  Returned inline both went through the same path and agreed,
    which is why every other tuple shape here compiled and this one did not.
    """
    with fp.FP64:
        t = [y, y], 1.0
        return t


@fp.fpy
def s_alias_then_return(
    xs: list[fp.Real], c: fp.Real, y: fp.Real,
) -> list[fp.Real]:
    with fp.FP64:
        ys = xs
        if c > 0:
            return ys
        else:
            return [y]


@fp.fpy
def s_projection_then_return(
    xss: list[list[fp.Real]], c: fp.Real, y: fp.Real,
) -> list[fp.Real]:
    with fp.FP64:
        row = xss[0]
        if c > 0:
            return row
        else:
            return [y]


@fp.fpy
def s_loop_target(
    xss: list[list[fp.Real]], c: fp.Real, y: fp.Real,
) -> list[fp.Real]:
    with fp.FP64:
        out = [y]
        for row in xss:
            if c > 0:
                out = row
        return out


@fp.fpy
def s_comprehension_or_literal(c: fp.Real, y: fp.Real) -> list[fp.Real]:
    with fp.FP64:
        if c > 0:
            return [1.5 for _ in range(3)]
        else:
            return [y]


@fp.fpy
def s_slice_or_literal(
    xs: list[fp.Real], c: fp.Real, y: fp.Real,
) -> list[fp.Real]:
    with fp.FP64:
        if c > 0:
            return xs[0:1]
        else:
            return [y]


@fp.fpy
def s_name_and_container(c: fp.Real, y: fp.Real) -> list[fp.Real]:
    with fp.FP64:
        zs = [1.5, 2.5]
        zss = [zs]
        zss[0][0] = 1.0
        if c > 0:
            return zs
        else:
            return [y]


@fp.fpy
def s_indexed_write_then_return(
    xs: list[fp.Real], c: fp.Real, y: fp.Real,
) -> list[fp.Real]:
    with fp.FP64:
        xs[0] = y
        if c > 0:
            return xs
        else:
            return [y]


@fp.fpy
def s_nested_param_returned(
    xss: list[list[fp.Real]], c: fp.Real, y: fp.Real,
) -> list[list[fp.Real]]:
    with fp.FP64:
        if c > 0:
            return xss
        else:
            return [[y]]


@fp.fpy
def s_write_a_row(
    xss: list[list[fp.Real]], c: fp.Real, y: fp.Real,
) -> fp.Real:
    with fp.FP64:
        xss[0][0] = y
        return xss[0][0]


# `SCALARS`: (c, y).  The list parameter's element format and `y`'s format are
# what make two contributors disagree; `c` is only a condition.
SCALARS, FLAT, NESTED = 'scalars', 'flat', 'nested'

SHAPES = [
    (s_two_literal_returns, SCALARS),
    (s_return_param_or_literal, FLAT),
    (s_ternary_literals, SCALARS),
    (s_ternary_param, FLAT),
    (s_nested_literal, SCALARS),
    (s_three_deep_literal, SCALARS),
    (s_nested_returns, SCALARS),
    (s_list_in_a_tuple, SCALARS),
    (s_two_lists_in_a_tuple, FLAT),
    (s_tuple_with_list_via_a_local, SCALARS),
    (s_alias_then_return, FLAT),
    (s_projection_then_return, NESTED),
    (s_loop_target, NESTED),
    (s_comprehension_or_literal, SCALARS),
    (s_slice_or_literal, FLAT),
    (s_name_and_container, SCALARS),
    (s_indexed_write_then_return, FLAT),
    (s_nested_param_returned, NESTED),
    (s_write_a_row, NESTED),
]

FORMATS = [fp.FP32, fp.FP64]

# `1` agrees with the `[y]` arm every shape has (so sized results appear), `2`
# disagrees (so the join must demote).  Only list-taking signatures vary over
# this; a scalar shape would repeat byte-identically.
LENGTHS = [None, 1, 2]


def _arg_types(sig: str, elt_fmt, y_fmt, length=None):
    """*sig*'s parameters, at the given formats and list length."""
    scalars = [RealType(fp.FP64), RealType(y_fmt)]      # c, y
    match sig:
        case 'scalars':
            return scalars
        case 'flat':
            return [ListType(RealType(elt_fmt), length), *scalars]
        case 'nested':
            return [
                ListType(ListType(RealType(elt_fmt), length), length),
                *scalars,
            ]
    raise AssertionError(sig)


def _matrix():
    """Every (shape, element format, scalar format, length) combination.

    No outer-context axis: every shape pins its context with ``with fp.FP64:``,
    so varying it produced byte-identical programs and only doubled the count.
    """
    for func, sig in SHAPES:
        lengths = LENGTHS if sig != SCALARS else [None]
        for elt_fmt in FORMATS:
            for y_fmt in FORMATS:
                for length in lengths:
                    label = (
                        f'{func.name}__{elt_fmt.nbits}_{y_fmt.nbits}'
                        f'_L{length}'
                    )
                    yield (
                        label, func,
                        _arg_types(sig, elt_fmt, y_fmt, length),
                    )


@pytest.fixture(scope='module')
def emitted():
    """``(namespaced sources, refusals)`` over the whole matrix.

    A refusal is a legitimate answer, so it is collected rather than raised.
    Each program goes in its own ``namespace`` -- two specializations of one
    shape are both named after it, and this is also what lets a few hundred
    programs share a single compiler invocation.
    """
    sources: list[str] = []
    refused: list[str] = []
    for i, (label, func, arg_types) in enumerate(_matrix()):
        m = Module()
        try:
            m.add(func, ctx=fp.FP64, arg_types=list(arg_types))
            # ALLOW: the matrix deliberately includes sharing shapes, and
            # strict refusals would hollow out `test_enough_of_the_matrix_compiles`.
            body = CppCompiler(unbox=UnboxMode.ALLOW).compile_module(m)
        except CppCompileError as e:
            refused.append(f'{label}: {" ".join(str(e).split())[:100]}')
            continue
        sources.append(f'namespace p{i:04d} {{\n{body}\n}}  // {label}')
    return sources, refused


def test_every_emitted_program_typechecks(emitted):
    """The property nothing else checks.

    A refusal is fine; C++ that does not compile is not.  One translation unit
    for the whole matrix, so this costs a single compiler invocation.
    """
    sources, _refused = emitted
    cc = CppCompiler()
    tu = '\n\n'.join([*cc.headers(), cc.helpers(), *sources])
    with tempfile.TemporaryDirectory() as td:
        cpp = Path(td) / 'generated.cpp'
        cpp.write_text(tu)
        r = subprocess.run(
            [_CXX, *_OPTS, '-fsyntax-only', str(cpp)],
            capture_output=True, text=True,
        )
    assert r.returncode == 0, (
        f'{len(sources)} generated programs, and the emitted C++ does not '
        f'compile.  The failing namespace names the shape and its formats.\n\n'
        + r.stderr[:4000]
    )


def test_enough_of_the_matrix_compiles(emitted):
    """A guard on the guard.

    Everything above passes vacuously if every program is refused, and a change
    that turns compiles into refusals is a regression even though no C++ breaks.
    The bound is loose -- it is here to catch a collapse, not to pin a number.
    """
    sources, refused = emitted
    total = len(sources) + len(refused)
    assert total > 60, f'the matrix shrank to {total} programs'
    assert len(sources) >= total * 0.6, (
        f'only {len(sources)}/{total} generated programs compile; the rest are '
        f'refused, so the typecheck above is checking less than it looks.\n  '
        + '\n  '.join(refused[:20])
    )

R = RealType(fp.FP64)
L = ListType(R)
L32 = ListType(RealType(fp.FP32))
N32 = ListType(L32)


def _typecheck(module: Module, *, unbox=UnboxMode.ALLOW) -> str:
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


# --------------------------------------------------------------------------
# The hand-written shapes the matrix above generalizes.  None of it is an unbox
# bug -- all of it reproduces with `UnboxMode.NEVER`.
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
def j_nested_literal(a: fp.Real) -> fp.Real:
    """No returns involved -- a nested literal's own rows disagree.

    A row comes from a parameter, or `Simplify` folds the whole program to its
    result and there is no place left to have a type."""
    with fp.FP64:
        xss = [[a], [3.0, 4.0]]
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
    (j_nested_literal, [R]),
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
    # business, and `UnboxMode.NEVER` wraps the same list in a `shared_ptr`.  Read off
    # the function alone; the runtime helpers are templates and would
    # contribute a `T`.
    body = CppCompiler(unbox=unbox).compile_module(m)
    # both spellings of an unboxed list: `std::vector<T>` and
    # `std::array<T, K>`; the boxed handle contains the vector form
    elts = re.findall(r'std::(?:vector|array)<(\w+)[,>]', body)
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
def sh_local_in_a_list(c: fp.Real, y: fp.Real) -> list[fp.Real]:
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
def sh_local_in_a_tuple(c: fp.Real, y: fp.Real):
    """The same, one level in: the tuple's field fixes the list's type."""
    with fp.FP64:
        xs = [1.0, 2.0]
        if c > 0:
            return (xs, 1.0)
        else:
            return ([y], y)


@fp.fpy
def sh_mixed_precision_local(c: fp.Real, y: fp.Real) -> list[fp.Real]:
    """The format the program asked for, not a narrowing accident: `lo`'s
    elements are FP32-rounded *values*, and `zss` holds the same buffer.

    `zss` is read, or dead-code elimination removes it and with it the sharing
    the refusal is about."""
    with fp.FP32:
        lo = [fp.round(y), fp.round(y)]
    with fp.FP64:
        zss = [lo]
        if c > 0:
            return zss[0]
        else:
            return [y]


@fp.fpy
def sh_parameter(xs: list[fp.Real], c: fp.Real, y: fp.Real) -> list[fp.Real]:
    """A parameter: the caller holds the same list, and the signature already
    committed to its element type, so neither side can move."""
    with fp.FP64:
        if c > 0:
            return xs
        else:
            return [y]


@fp.fpy
def sh_alias(xs: list[fp.Real], c: fp.Real, y: fp.Real) -> list[fp.Real]:
    """`ys = xs` binds `const auto&`, so `ys` has no buffer of its own."""
    with fp.FP64:
        ys = xs
        if c > 0:
            return ys
        else:
            return [y]


@fp.fpy
def sh_projection(xss: list[list[fp.Real]], c: fp.Real, y: fp.Real) -> list[fp.Real]:
    """`row = xss[0]` binds `const auto&` to a slot."""
    with fp.FP64:
        row = xss[0]
        if c > 0:
            return row
        else:
            return [y]


@fp.fpy
def sh_loop_target(xss: list[list[fp.Real]], c: fp.Real, y: fp.Real) -> list[fp.Real]:
    """A loop target binds `const auto&` to each element."""
    with fp.FP64:
        out = [y]
        for row in xss:
            if c > 0:
                out = row
        return out


SHARED_CASES = [
    (sh_local_in_a_list, [R, R]),
    (sh_local_in_a_tuple, [R, R]),
    (sh_mixed_precision_local, [R, R]),
    (sh_parameter, [L32, R, R]),
    (sh_alias, [L32, R, R]),
    (sh_projection, [N32, R, R]),
    (sh_loop_target, [N32, R, R]),
]


@pytest.mark.parametrize(
    'func,arg_types', SHARED_CASES, ids=[f.name for f, _ in SHARED_CASES],
)
def test_a_shared_narrower_list_is_refused(func, arg_types):
    """Refusing is the whole point: the alternative is silently unsharing.

    The last three matter most, because there the mismatch would be invisible.
    A reference binding is spelled `const auto&`, so nothing in the emitted text
    states its element type -- `const auto& ys = xs;` followed by `return ys;`
    would hand back a boxed `float` list as a boxed `double` list and only the
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
