"""Storage inference, driven without a backend.

The analysis is generic over a *domain* -- an ordered sequence of formats the
target can spell -- so these tests supply one of their own rather than the cpp
ladder.  If any of them needed `fpy2.backend`, the move to `fpy2/analysis/`
would not have been real.
"""

import pytest

import fpy2 as fp
from fpy2.analysis import (
    ArraySizeInfer,
    ContextUse,
    DefineUse,
    StorageInfer,
    StorageSelectionError,
    of_bound,
)
from fpy2.analysis.format_infer import FormatInfer, ListFormat, SetFormat
from fpy2.analysis.storage_infer import join

_I8 = fp.SINT8.format()
_I64 = fp.SINT64.format()
_F64 = fp.FP64.format()


class _Tiny:
    """Three formats, deliberately not the cpp ladder."""

    @property
    def sigma(self):
        return [_I8, _I64, _F64]

    def fallback(self, bound):
        return None


class _NoIntegers:
    """One format.  A domain may be as poor as it likes."""

    @property
    def sigma(self):
        return [_F64]

    def fallback(self, bound):
        return None


def _analyze(func, domain):
    ast = func.ast
    du = DefineUse.analyze(ast)
    cu = ContextUse.analyze(ast, def_use=du)
    sz = ArraySizeInfer.analyze(ast)
    fmt = FormatInfer.analyze(ast, def_use=du, ctx_use=cu, array_size=sz)
    return fmt, StorageInfer.infer(
        fmt.type_info.def_use, fmt.by_def, fmt.by_expr, domain,
    )


def _classes(storage, name):
    """The distinct classes holding definitions called *name*."""
    return {
        id(storage.def_class[d]) for d in storage.def_class
        if str(d.name) == name
    }


class TestTheDomainIsTheBackends:
    def test_of_bound_takes_the_first_containing_member(self):
        assert of_bound(_Tiny(), SetFormat({1})) is _I8
        assert of_bound(_Tiny(), _F64) is _F64

    def test_a_poorer_domain_gives_a_wider_answer(self):
        """Nothing about the answer is intrinsic -- it is the domain's."""
        assert of_bound(_NoIntegers(), SetFormat({1})) is _F64

    def test_a_domain_may_contain_nothing_that_fits(self):
        with pytest.raises(StorageSelectionError):
            of_bound(_NoIntegers(), _I64)

    def test_structure_recurses(self):
        chosen = of_bound(_Tiny(), ListFormat(SetFormat({1})))
        assert isinstance(chosen, ListFormat)
        assert chosen.elt is _I8

    def test_a_non_numeric_bound_has_no_numeric_storage(self):
        assert of_bound(_Tiny(), None) is None


class TestTheJoin:
    def test_it_is_the_first_member_containing_every_input(self):
        assert join(_Tiny(), [_I8, _F64]) is _F64
        assert join(_Tiny(), [_I8, _I8]) is _I8

    def test_it_recurses_through_lists(self):
        merged = join(_Tiny(), [ListFormat(_I8), ListFormat(_F64)])
        assert isinstance(merged, ListFormat)
        assert merged.elt is _F64

    def test_it_refuses_where_no_member_contains_all(self):
        with pytest.raises(StorageSelectionError):
            join(_NoIntegers(), [_I64, _F64])


class TestClasses:
    """`store` is per runtime object, not per name."""

    def test_a_rebind_starts_a_new_class(self):
        @fp.fpy
        def f() -> fp.Real:
            with fp.FP64:
                y = 1.0
                y = 2.0 * 3.0
                return y

        _fmt, storage = _analyze(f, _Tiny())
        assert len(_classes(storage, 'y')) == 2

    def test_a_phi_joins_one_class(self):
        @fp.fpy
        def f(c: bool) -> fp.Real:
            with fp.FP64:
                if c:
                    y = 1.0
                else:
                    y = 2.0 * 3.0
                return y

        _fmt, storage = _analyze(f, _Tiny())
        assert len(_classes(storage, 'y')) == 1

    def test_an_in_place_update_stays_one_class(self):
        @fp.fpy
        def f() -> list[fp.Real]:
            with fp.FP64:
                ys = [1.0, 2.0]
                ys[0] = 3.0
                return ys

        _fmt, storage = _analyze(f, _Tiny())
        assert len(_classes(storage, 'ys')) == 1

    def test_every_member_fits_its_class(self):
        """Containment, which the join guarantees by construction."""

        @fp.fpy
        def f(c: bool) -> fp.Real:
            with fp.FP64:
                if c:
                    y = 1.0
                else:
                    y = 2.0 * 3.0
                return y

        fmt, storage = _analyze(f, _Tiny())
        for d, cls in storage.def_class.items():
            bound = fmt.by_def.get(d)
            if bound is None or storage.class_storage[cls] is None:
                continue
            assert of_bound(_Tiny(), bound) is not None


class TestIsRebound:
    def test_an_in_place_update_is_not_a_rebind(self):
        @fp.fpy
        def f() -> list[fp.Real]:
            with fp.FP64:
                ys = [1.0, 2.0]
                ys[0] = 3.0
                return ys

        _fmt, storage = _analyze(f, _Tiny())
        d = next(d for d in storage.def_class if str(d.name) == 'ys')
        assert not storage.is_rebound(d)
