"""The ``std::array`` surface.

`TestSizedType` / `TestConstruction` / `TestConversionLattice` drive the
emitter's fixed-size arms with hand-built types; `TestSizeSpecialization`
pins size-keyed specs; `TestEndToEnd` runs the size decision through the
whole pipeline, including the demotions (joins, runtime lengths,
conditionally-proven sizes) and the two off switches (``arrays=False``,
``UnboxMode.NEVER``).
"""

import re

import pytest

import fpy2 as fp

from fpy2.backend.cpp.compiler import CppCompiler
from fpy2.backend.cpp.emitter import (
    CppEmitError,
    CppEmitter,
    CppInternalError,
)
from fpy2.backend.cpp.types import CppList, CppScalar
from fpy2.backend.cpp.unbox import UnboxMode
from fpy2.module import Module
from fpy2.types import BoolType, ListType, RealType

F64 = CppScalar.F64
ARR3 = CppList(F64, boxed=False, size=3)
ARR0 = CppList(F64, boxed=False, size=0)
VEC = CppList(F64, boxed=False)
BOXED = CppList(F64)


@fp.fpy
def _trivial(x: fp.Real) -> fp.Real:
    with fp.FP64:
        return x + 1.0


def _emitter() -> CppEmitter:
    """An emitter over a trivial function, to drive helpers by hand."""
    cc = CppCompiler(unbox=UnboxMode.ALLOW)
    m = Module()
    m.add(_trivial, ctx=fp.FP64, arg_types=[RealType(fp.FP64)])
    a = cc.analyze(cc.specialize(m)[-1])
    return CppEmitter(
        ast=a.ast,
        storage=a.storage,
        variables=a.variables,
        def_use=a.def_use,
        format_info=a.format_info,
        class_info=a.class_info,
        ctx_use=a.ctx_use,
        call_names={},
        unsafe_cast_int=True,
        unbox=a.unbox,
        callee_params={},
    )


def _any_expr(em: CppEmitter):
    """An arbitrary expression of *em*'s AST -- the helpers taking an ``at``
    only read it for an error location."""
    return next(iter(em.format_info.by_expr))


class TestSizedType:
    """`CppList.size` is identity, not metadata."""

    def test_format(self):
        assert ARR3.format() == 'std::array<double, 3>'
        assert VEC.format() == 'std::vector<double>'
        assert BOXED.format() == 'std::shared_ptr<std::vector<double>>'

    def test_size_participates_in_equality(self):
        assert ARR3 != VEC
        assert ARR3 != CppList(F64, boxed=False, size=4)
        assert ARR3 == CppList(F64, boxed=False, size=3)
        assert hash(ARR3) != hash(VEC)

    def test_a_boxed_list_cannot_carry_a_size(self):
        with pytest.raises(AssertionError):
            CppList(F64, boxed=True, size=3)

    def test_nested_format(self):
        m = CppList(ARR3, boxed=False, size=2)
        assert m.format() == 'std::array<std::array<double, 3>, 2>'


class TestConstruction:
    """The constructor forms `std::array` lacks, respelled."""

    def test_init_takes_double_braces(self):
        em = _emitter()
        s = em._list_new_init(ARR3, ['a', 'b', 'c'])
        assert s == 'std::array<double, 3>{{a, b, c}}'

    def test_init_element_count_is_checked(self):
        em = _emitter()
        with pytest.raises(CppInternalError):
            em._list_new_init(ARR3, ['a', 'b'])

    def test_empty_init_is_value_init(self):
        em = _emitter()
        assert em._list_new_init(ARR0, []) == 'std::array<double, 0>{}'
        assert em._list_empty(ARR3) == 'std::array<double, 3>{}'

    def test_sized_ignores_the_runtime_count(self):
        em = _emitter()
        assert em._list_new_sized(ARR3, 'n') == 'std::array<double, 3>{}'

    def test_filled_repeats_the_fill(self):
        em = _emitter()
        s = em._list_new_filled(ARR3, 'n', 'double{}')
        assert s == 'std::array<double, 3>{{double{}, double{}, double{}}}'

    def test_range_copies_into_a_declared_temp(self):
        em = _emitter()
        out = em._list_new_range(ARR3, 'first', 'last')
        body = em.writer.render()
        assert f'std::array<double, 3> {out}{{}};' in body
        assert f'std::copy(first, last, {out}.begin());' in body

    def test_push_back_on_an_array_is_a_backend_bug(self):
        em = _emitter()
        with pytest.raises(CppInternalError):
            em._list_push(ARR3, 'xs', 'x')

    def test_builder_stores_by_index(self):
        em = _emitter()
        out, append = em._open_list_build(ARR3)
        stmt = append('x')
        body = em.writer.render()
        assert f'std::array<double, 3> {out}{{}};' in body
        assert '= 0;' in body  # the running index
        assert stmt.endswith('] = x') and '++' in stmt

    def test_builder_still_pushes_for_vectors(self):
        em = _emitter()
        out, append = em._open_list_build(VEC)
        assert append('x') == f'{out}.push_back(x)'


class TestConversionLattice:
    """array→vector converts; vector→array and K1→K2 refuse."""

    def test_array_to_vector_is_a_copy(self):
        em = _emitter()
        at = _any_expr(em)
        s = em._convert_storage('xs', ARR3, VEC, at=at)
        assert s == 'std::vector<double>(xs.begin(), xs.end())'

    def test_vector_to_array_refuses(self):
        em = _emitter()
        at = _any_expr(em)
        with pytest.raises(CppEmitError):
            em._convert_storage('xs', VEC, ARR3, at=at)

    def test_size_mismatch_refuses(self):
        em = _emitter()
        at = _any_expr(em)
        with pytest.raises(CppEmitError):
            em._convert_storage(
                'xs', ARR3, CppList(F64, boxed=False, size=4), at=at,
            )

    def test_array_to_boxed_drops_the_size_before_the_handle(self):
        """`make_shared<vector>(array)` does not compile: the value must be
        rebuilt as a vector first, then boxed."""
        em = _emitter()
        at = _any_expr(em)
        s = em._convert_storage('xs', ARR3, BOXED, at=at)
        assert s == (
            'std::make_shared<std::vector<double>>'
            '(std::vector<double>(xs.begin(), xs.end()))'
        )

    def test_same_size_element_widening_rebuilds_by_index(self):
        src = CppList(CppScalar.U8, boxed=False, size=3)
        em = _emitter()
        at = _any_expr(em)
        out = em._convert_storage('xs', src, ARR3, at=at)
        body = em.writer.render()
        assert f'std::array<double, 3> {out}{{}};' in body
        assert f'{out}[' in body and 'static_cast<double>' in body
        assert 'push_back' not in body and 'reserve' not in body


@fp.fpy
def _dot(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
    with fp.FP64:
        acc = 0.0
        for i in range(len(xs)):
            acc = acc + xs[i] * ys[i]
        return acc


@fp.fpy
def _calls_dot_at_two_lengths(x: fp.Real) -> fp.Real:
    with fp.FP64:
        a = _dot([x, x], [x, x])
        b = _dot([x, x, x], [x, x, x])
        return a + b


class TestSizeSpecialization:
    """Sizes join the specialization key: a callee compiles once per
    distinct argument-length vector, its parameters carrying the length."""

    def test_a_callee_parameter_becomes_an_array(self):
        @fp.fpy
        def caller(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return _dot([x, x, x], [x, x, x])

        m = Module()
        m.add(caller, ctx=fp.FP64, arg_types=[RealType(fp.FP64)])
        out = CppCompiler().compile_module(m)
        assert 'const std::array<double, 3>& xs' in out
        assert 'std::vector' not in out

    def test_two_lengths_make_two_specs(self):
        m = Module()
        m.add(
            _calls_dot_at_two_lengths, ctx=fp.FP64,
            arg_types=[RealType(fp.FP64)],
        )
        out = CppCompiler().compile_module(m)
        assert 'const std::array<double, 2>& xs' in out
        assert 'const std::array<double, 3>& xs' in out
        assert out.count('double _dot__') == 2

    def test_size_free_specs_are_byte_identical(self):
        """`size_key` must not move a size-free program: same mangled
        names, same code."""
        @fp.fpy
        def caller(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return _dot(xs, xs)

        L = ListType(RealType(fp.FP64))
        m = Module()
        m.add(caller, ctx=fp.FP64, arg_types=[L])
        on = CppCompiler(arrays=True).compile_module(m)
        off = CppCompiler(arrays=False).compile_module(m)
        assert on == off
        assert 'std::array' not in on

    def test_entries_differing_only_in_length_get_distinct_specs(self):
        """The public key carries lengths too: two `Module.add` entries at
        different lengths must not collapse to the first one's signature."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return xs[0]

        R = RealType(fp.FP64)
        m = Module()
        m.add(f, name='f2', ctx=fp.FP64, arg_types=[ListType(R, 2)])
        m.add(f, name='f5', ctx=fp.FP64, arg_types=[ListType(R, 5)])
        out = CppCompiler().compile_module(m)
        assert 'double f2(const std::array<double, 2>& xs)' in out
        assert 'double f5(const std::array<double, 5>& xs)' in out


class TestEndToEnd:
    """The size decision, through the whole pipeline."""

    def test_literal_and_comprehension(self):
        @fp.fpy
        def f() -> fp.Real:
            with fp.FP64:
                xs = [1.5, 2.5, 3.5]
                ys = [x * 2 for x in xs]
                return ys[0]

        out = CppCompiler().compile(f, ctx=fp.FP64, arg_types=[])
        # the element scalar is the ladder's business (it is value-precise
        # enough to fit {3, 5, 7} in uint8_t); what this pins is the shape
        assert out.count('std::array<') >= 2
        assert 'std::vector' not in out
        assert 'push_back' not in out

    def test_empty_of_constant_dims_is_one_value_init(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                m = fp.empty(2, 3)
                m[0][0] = x
                return m[0][0]

        out = CppCompiler().compile(
            f, ctx=fp.FP64, arg_types=[RealType(fp.FP64)],
        )
        assert 'std::array<std::array<double, 3>, 2>' in out
        assert '{};' in out
        assert 'std::vector' not in out

    def test_empty_may_give_fewer_dims_than_the_type_has(self):
        """``fp.empty(n)`` for a ``list[list[T]]`` allocates the outer layer
        only; each cell is ``UNINIT`` until a store replaces it, which is what
        every lowered comprehension over a nested list does."""
        @fp.fpy
        def f(A: list[list[fp.Real]]) -> list[list[fp.Real]]:
            with fp.FP64:
                out = fp.empty(len(A))
                for i in range(len(A)):
                    out[i] = A[i][:]
                return out

        s = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[ListType(ListType(RealType(fp.FP64)))],
        )
        assert re.search(
            r'std::vector<std::vector<double>>\('
            r'static_cast<uint64_t>\(\w+\), std::vector<double>\{\}\)', s,
        )

    def test_slice_of_whole_keeps_the_size(self):
        @fp.fpy
        def f() -> fp.Real:
            with fp.FP64:
                xs = [1.5, 2.5, 3.5]
                ys = xs[:]
                ys[0] = 9.0
                return xs[0]

        out = CppCompiler().compile(f, ctx=fp.FP64, arg_types=[])
        assert out.count('std::array<') >= 2
        assert 'std::vector' not in out
        assert 'std::copy(' in out

    def test_sized_entry_parameter_from_arg_types(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return xs[0]

        out = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[ListType(RealType(fp.FP64), 4)],
        )
        assert 'const std::array<double, 4>& xs' in out

    def test_unsized_parameter_stays_a_vector(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return xs[0]

        out = CppCompiler().compile(
            f, ctx=fp.FP64, arg_types=[ListType(RealType(fp.FP64))],
        )
        assert 'const std::vector<double>& xs' in out

    def test_a_join_of_two_sizes_demotes_to_vector(self):
        @fp.fpy
        def f(c: bool, x: fp.Real) -> fp.Real:
            with fp.FP64:
                if c:
                    xs = [x, x]
                else:
                    xs = [x, x, x]
                return xs[0]

        out = CppCompiler().compile(
            f, ctx=fp.FP64, arg_types=[BoolType(), RealType(fp.FP64)],
        )
        assert 'std::vector<double> xs' in out
        assert 'std::array' not in out

    def test_disagreeing_returns_demote_the_return_type(self):
        @fp.fpy
        def f(c: bool, x: fp.Real) -> list[fp.Real]:
            with fp.FP64:
                if c:
                    return [x, x]
                return [x, x, x]

        out = CppCompiler().compile(
            f, ctx=fp.FP64, arg_types=[BoolType(), RealType(fp.FP64)],
        )
        assert out.startswith('std::vector<double> f(')

    def test_agreeing_returns_keep_the_size(self):
        @fp.fpy
        def f(c: bool, x: fp.Real) -> list[fp.Real]:
            with fp.FP64:
                if c:
                    return [x, x]
                return [x, x + 1.0]

        out = CppCompiler().compile(
            f, ctx=fp.FP64, arg_types=[BoolType(), RealType(fp.FP64)],
        )
        assert out.startswith('std::array<double, 2> f(')

    def test_empty_sized_from_len_is_an_array(self):
        """``fp.empty(len(xs))`` over a proven-length list is a `std::array`.

        The dimension folds through `ArraySizeInfer`'s `_const_int`; before that
        it resolved only partial-eval constants, so this shape lost its length
        and fell back to `std::vector` even though the length was proven.
        """

        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            acc = fp.empty(len(xs))
            for i in range(len(xs)):
                acc[i] = xs[i] * 2.0
            return acc

        out = CppCompiler(unsafe_cast_int=True).compile(
            f, ctx=fp.FP64, arg_types=[ListType(RealType(fp.FP64), 3)],
        )
        assert 'std::array<double, 3> acc' in out
        assert 'std::vector' not in out

    def test_runtime_length_stays_a_vector(self):
        @fp.fpy
        def f(n: fp.Real) -> fp.Real:
            with fp.INTEGER:
                k = fp.round(n)
            xs = fp.empty(k)
            with fp.FP64:
                xs[0] = 1.5
                return xs[0]

        out = CppCompiler(unsafe_cast_int=True).compile(
            f, ctx=fp.FP64, arg_types=[RealType(fp.FP64)],
        )
        assert 'std::vector<' in out
        assert 'std::array' not in out

    def test_a_conditional_zip_does_not_pin_a_parameter(self):
        """Regression: `zip(xs, [«3 elements»])` under a branch used to pin
        `len(xs) == 3` globally, compiling the parameter to
        `std::array<double, 3>` -- a stack overflow (ASan-confirmed) for a
        caller whose list is longer and whose execution never takes the
        branch.  The pin holds only where the zip runs.
        """
        @fp.fpy
        def f(xs: list[fp.Real], c: fp.Real) -> fp.Real:
            with fp.FP64:
                acc = 0.0
                if c > 0:
                    for a, b in zip(xs, [1.0, 2.0, 3.0]):
                        acc = acc + a * b
                return acc + xs[0]

        cc = CppCompiler(unbox=UnboxMode.ALLOW)
        params, _ = cc.signature(
            f, ctx=fp.FP64,
            arg_types=[ListType(RealType(fp.FP64)), RealType(fp.FP64)],
        )
        assert params[0].format() == 'std::vector<double>'

    def test_an_unconditional_zip_still_pins(self):
        """The counterweight: reached on every execution, the strict-zip
        equality is a global fact (a mismatched call is undefined anyway),
        so the parameter does become an array.  Expression position, since
        ``ZipElim`` rewrites a for-header zip away before the analysis."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                t = zip(xs, [1.0, 2.0, 3.0])
                acc = 0.0
                for i in range(len(t)):
                    a, b = t[i]
                    acc = acc + a * b
                return acc

        cc = CppCompiler(unbox=UnboxMode.ALLOW)
        params, _ = cc.signature(
            f, ctx=fp.FP64, arg_types=[ListType(RealType(fp.FP64))],
        )
        assert params[0].format() == 'std::array<double, 3>'

    def test_mixed_sized_and_runtime_dims(self):
        """`empty(2, k)`: the outer layer is an array of two *runtime-sized*
        vectors -- the repeated-fill path, with the dimension bound to a
        name so it is evaluated once."""
        @fp.fpy
        def f(n: fp.Real) -> fp.Real:
            with fp.INTEGER:
                k = fp.round(n)
            m = fp.empty(2, k)
            with fp.FP64:
                m[0][0] = 1.5
                return m[0][0]

        out = CppCompiler(unsafe_cast_int=True).compile(
            f, ctx=fp.FP64, arg_types=[RealType(fp.FP64)],
        )
        assert 'std::array<std::vector<' in out

    def test_strict_mode_emits_arrays(self):
        """STRICT gates handles, not sizes: a fully-unboxable program keeps
        its arrays under the default mode."""
        @fp.fpy
        def f() -> fp.Real:
            with fp.FP64:
                xs = [1.5, 2.5, 3.5]
                return xs[0]

        out = CppCompiler(unbox=UnboxMode.STRICT).compile(
            f, ctx=fp.FP64, arg_types=[],
        )
        assert 'std::array<float, 3>' in out

    def test_assert_len_pins_a_parameter(self):
        """The route the `arrays` docstring advertises: a trusted top-level
        `assert len(xs) == K` becomes a type-level commitment."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                assert len(xs) == 4
                return xs[0]

        out = CppCompiler().compile(
            f, ctx=fp.FP64, arg_types=[ListType(RealType(fp.FP64))],
        )
        assert 'const std::array<double, 4>& xs' in out

    def test_flag_off_drops_a_sized_annotation_silently(self):
        """`arrays=False` compiles a length-annotated parameter as a plain
        vector -- the length is advisory metadata again, not an error."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return xs[0]

        out = CppCompiler(arrays=False).compile(
            f, ctx=fp.FP64, arg_types=[ListType(RealType(fp.FP64), 4)],
        )
        assert 'const std::vector<double>& xs' in out

    def test_flag_off_is_a_clean_bypass(self):
        @fp.fpy
        def f() -> fp.Real:
            with fp.FP64:
                xs = [1.5, 2.5, 3.5]
                return xs[0]

        on = CppCompiler().compile(f, ctx=fp.FP64, arg_types=[])
        off = CppCompiler(arrays=False).compile(f, ctx=fp.FP64, arg_types=[])
        assert 'std::array<float, 3>' in on
        assert 'std::array' not in off
        assert 'std::vector<float>' in off

    def test_never_mode_has_no_arrays(self):
        @fp.fpy
        def f() -> fp.Real:
            with fp.FP64:
                xs = [1.0, 2.0, 3.0]
                return xs[0]

        out = CppCompiler(unbox=UnboxMode.NEVER).compile(
            f, ctx=fp.FP64, arg_types=[],
        )
        assert 'std::array' not in out
