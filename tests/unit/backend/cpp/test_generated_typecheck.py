"""Generated coverage for the shapes the corpus does not have.

Every bug found while building the join/widening machinery lived in one of four
shapes, and the corpus barely contains them.  Of its 217 functions: 6 have more
than one ``return``, 6 have a ternary, 7 have a nested list literal, and **none**
has a list at a format other than FP64.  So the differential harness stayed green
through a nested-literal miscompile, a return-join type disagreement, a
``signature()`` mismatch, and a reference-bound name reporting a storage type its
reference did not have.  Each was found by hand-writing a shape, which is the
thing to automate.

Two axes:

- **Shape** -- hand-enumerated below, because a failure has to be readable.
  Adding one is a single function plus an entry in ``SHAPES``.
- **Format** -- generated.  This is where the corpus has nothing, and it is also
  the cheap axis: a program's formats come entirely from ``arg_types``, so
  one source function yields four programs.

The assertion is deliberately weak on purpose: a program may legitimately be
*refused* (``CppEmitError``) -- a shared list cannot change element type, and
saying so is correct.  What may never happen is emitting C++ that does not
typecheck.  Since "everything refused" would satisfy that vacuously,
:func:`test_enough_of_the_matrix_compiles` pins the floor.
"""

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


def _arg_types(sig: str, elt_fmt, y_fmt):
    """*sig*'s parameters, at the given formats."""
    scalars = [RealType(fp.FP64), RealType(y_fmt)]      # c, y
    match sig:
        case 'scalars':
            return scalars
        case 'flat':
            return [ListType(RealType(elt_fmt)), *scalars]
        case 'nested':
            return [ListType(ListType(RealType(elt_fmt))), *scalars]
    raise AssertionError(sig)


def _matrix():
    """Every (shape, element format, scalar format) combination.

    No outer-context axis: every shape pins its context with ``with fp.FP64:``,
    so varying it produced byte-identical programs and only doubled the count.
    """
    for func, sig in SHAPES:
        for elt_fmt in FORMATS:
            for y_fmt in FORMATS:
                label = f'{func.name}__{elt_fmt.nbits}_{y_fmt.nbits}'
                yield label, func, _arg_types(sig, elt_fmt, y_fmt)


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
