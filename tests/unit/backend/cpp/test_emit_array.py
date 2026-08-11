"""The ``std::array`` surface, ahead of anything that produces it.

Phase 2 of the sized-list feature: `CppList` carries a ``size`` and the
emitter can construct, rebuild, and convert fixed-size lists -- but no
analysis stamps a size yet, so every case here drives the machinery with
hand-built types.  End-to-end coverage arrives with the size decision.
"""

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
from fpy2.types import RealType

F64 = CppScalar.F64
ARR3 = CppList(F64, boxed=False, size=3)
ARR0 = CppList(F64, boxed=False, size=0)
VEC = CppList(F64, boxed=False)
BOXED = CppList(F64)


@fp.fpy
def _trivial(x: fp.Real) -> fp.Real:
    with fp.FP64:
        return x + 1.0


def _emitter(mode=UnboxMode.ALLOW):
    """``(emitter, at)`` over a trivial function, to drive helpers by hand.

    *at* is an arbitrary expression of the analyzed AST -- the helpers only
    read it for an error location.
    """
    cc = CppCompiler(unbox=mode)
    m = Module()
    m.add(_trivial, ctx=fp.FP64, arg_types=[RealType(fp.FP64)])
    a = cc.analyze(cc.specialize(m)[-1])
    em = CppEmitter(
        ast=a.ast,
        storage=a.storage,
        def_use=a.def_use,
        format_info=a.format_info,
        ctx_use=a.ctx_use,
        call_names={},
        unsafe_cast_int=True,
        unbox=a.unbox,
        callee_params={},
    )
    return em, next(iter(a.format_info.by_expr))


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
        em, _ = _emitter()
        s = em._list_new_init(ARR3, ['a', 'b', 'c'])
        assert s == 'std::array<double, 3>{{a, b, c}}'

    def test_init_element_count_is_checked(self):
        em, _ = _emitter()
        with pytest.raises(CppInternalError):
            em._list_new_init(ARR3, ['a', 'b'])

    def test_empty_init_is_value_init(self):
        em, _ = _emitter()
        assert em._list_new_init(ARR0, []) == 'std::array<double, 0>{}'
        assert em._list_empty(ARR3) == 'std::array<double, 3>{}'

    def test_sized_ignores_the_runtime_count(self):
        em, _ = _emitter()
        assert em._list_new_sized(ARR3, 'n') == 'std::array<double, 3>{}'

    def test_filled_repeats_the_fill(self):
        em, _ = _emitter()
        s = em._list_new_filled(ARR3, 'n', 'double{}')
        assert s == 'std::array<double, 3>{{double{}, double{}, double{}}}'

    def test_range_copies_into_a_declared_temp(self):
        em, _ = _emitter()
        out = em._list_new_range(ARR3, 'first', 'last')
        body = em.writer.render()
        assert f'std::array<double, 3> {out}{{}};' in body
        assert f'std::copy(first, last, {out}.begin());' in body

    def test_push_back_on_an_array_is_a_backend_bug(self):
        em, _ = _emitter()
        with pytest.raises(CppInternalError):
            em._list_push(ARR3, 'xs', 'x')

    def test_builder_stores_by_index(self):
        em, _ = _emitter()
        out, append = em._open_list_build(ARR3)
        stmt = append('x')
        body = em.writer.render()
        assert f'std::array<double, 3> {out}{{}};' in body
        assert '= 0;' in body  # the running index
        assert stmt.endswith('] = x') and '++' in stmt

    def test_builder_still_pushes_for_vectors(self):
        em, _ = _emitter()
        out, append = em._open_list_build(VEC)
        assert append('x') == f'{out}.push_back(x)'


class TestConversionLattice:
    """array→vector converts; vector→array and K1→K2 refuse."""

    def test_array_to_vector_is_a_copy(self):
        em, at = _emitter()
        s = em._convert_storage('xs', ARR3, VEC, at=at)
        assert s == 'std::vector<double>(xs.begin(), xs.end())'

    def test_vector_to_array_refuses(self):
        em, at = _emitter()
        with pytest.raises(CppEmitError):
            em._convert_storage('xs', VEC, ARR3, at=at)

    def test_size_mismatch_refuses(self):
        em, at = _emitter()
        with pytest.raises(CppEmitError):
            em._convert_storage(
                'xs', ARR3, CppList(F64, boxed=False, size=4), at=at,
            )

    def test_array_to_boxed_drops_the_size_before_the_handle(self):
        """`make_shared<vector>(array)` does not compile: the value must be
        rebuilt as a vector first, then boxed."""
        em, at = _emitter()
        s = em._convert_storage('xs', ARR3, BOXED, at=at)
        assert s == (
            'std::make_shared<std::vector<double>>'
            '(std::vector<double>(xs.begin(), xs.end()))'
        )

    def test_same_size_element_widening_rebuilds_by_index(self):
        src = CppList(CppScalar.U8, boxed=False, size=3)
        em, at = _emitter()
        out = em._convert_storage('xs', src, ARR3, at=at)
        body = em.writer.render()
        assert f'std::array<double, 3> {out}{{}};' in body
        assert f'{out}[' in body and 'static_cast<double>' in body
        assert 'push_back' not in body and 'reserve' not in body


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
        from fpy2.types import ListType

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
        from fpy2.types import ListType

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return xs[0]

        out = CppCompiler().compile(
            f, ctx=fp.FP64, arg_types=[ListType(RealType(fp.FP64))],
        )
        assert 'const std::vector<double>& xs' in out

    def test_a_join_of_two_sizes_demotes_to_vector(self):
        from fpy2.types import BoolType

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
        from fpy2.types import BoolType

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
        from fpy2.types import BoolType

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
