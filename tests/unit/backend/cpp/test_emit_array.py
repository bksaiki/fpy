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


class TestInert:
    """Until the analysis stamps sizes, no program produces an array."""

    def test_a_list_program_is_array_free(self):
        @fp.fpy
        def f() -> fp.Real:
            with fp.FP64:
                xs = [1.0, 2.0, 3.0]
                ys = [x * 2 for x in xs]
                return ys[0]

        out = CppCompiler().compile(f, ctx=fp.FP64, arg_types=[])
        assert 'std::array' not in out
