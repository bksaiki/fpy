"""
Tests for cpp storage-type selection (Phase 1 of the backend-cpp plan).
"""

import fpy2 as fp
import pytest

from fractions import Fraction
from fpy2.analysis.format_infer import ListFormat, SetFormat, TupleFormat
from fpy2.backend.cpp.storage import (
    StorageSelectionError,
    aggregate_storage,
    choose_storage,
    choose_storage_scalar,
)
from fpy2.backend.cpp.types import CppList, CppScalar, CppTuple
from fpy2.number.context.real import REAL_FORMAT


class TestStorageScalar:
    """``choose_storage_scalar`` covers scalar bounds."""

    def test_none_bound_is_bool(self):
        """Non-numeric (e.g., from a comparison) → BOOL."""
        assert choose_storage_scalar(None) == CppScalar.BOOL

    def test_fp32_format_picks_f32(self):
        assert choose_storage_scalar(fp.FP32.format()) == CppScalar.F32

    def test_fp64_format_picks_f64(self):
        assert choose_storage_scalar(fp.FP64.format()) == CppScalar.F64

    def test_sint8_format_picks_s8(self):
        assert choose_storage_scalar(fp.SINT8.format()) == CppScalar.S8

    def test_sint64_format_picks_s64(self):
        assert choose_storage_scalar(fp.SINT64.format()) == CppScalar.S64

    def test_uint16_format_picks_u16(self):
        assert choose_storage_scalar(fp.UINT16.format()) == CppScalar.U16

    def test_setformat_zero_fits_smallest(self):
        """A SetFormat({0}) is dyadic and trivially fits the smallest int."""
        s = SetFormat(frozenset((Fraction(0),)))
        assert choose_storage_scalar(s) == CppScalar.U8

    def test_setformat_negative_picks_signed(self):
        """SetFormat({-1, 1}) needs at least an int8."""
        s = SetFormat(frozenset((Fraction(-1), Fraction(1))))
        assert choose_storage_scalar(s) == CppScalar.S8

    def test_real_format_rejected(self):
        """REAL_FORMAT can't be stored in any finite C++ type."""
        with pytest.raises(StorageSelectionError, match='unconstrained real'):
            choose_storage_scalar(REAL_FORMAT)

    def test_unbounded_integer_falls_back_to_s64(self):
        """``MPFixedFormat`` representing unbounded integers — e.g.,
        the result of ``range(...)`` — falls back to ``S64``.

        Strictly speaking S64 doesn't *contain* an unbounded integer,
        but the alternative is rejecting every loop counter that the
        format-inference doesn't tighten with a numeric bound.  Overflow
        is the user's responsibility; multi-precision storage is
        out-of-scope per ``docs/todos/backend-cpp.md``.
        """
        from fpy2.number.context.mp_fixed import MPFixedFormat
        unbounded_int = MPFixedFormat(nmin=-1)
        assert choose_storage_scalar(unbounded_int) == CppScalar.S64

    def test_unbounded_integer_fallback_still_checks_special_values(self):
        """The ``S64`` fallback ignores the *magnitude* bound, not the rest.

        Regression: it ran after the ladder search and re-checked nothing, so a
        bound the ladder had just rejected could still land in ``int64_t``.
        ``int64_t`` holds no NaN, no infinity and no signed zero, and an
        ``MPFixedFormat`` can carry any of the three.
        """
        from fpy2.number.context.mp_fixed import MPFixedFormat
        for kwargs in ({'enable_nan': True}, {'enable_inf': True}):
            fmt = MPFixedFormat(nmin=-1, **kwargs)
            with pytest.raises(StorageSelectionError):
                choose_storage_scalar(fmt)
        # ...while a plain unbounded integer still takes the fallback.
        assert choose_storage_scalar(MPFixedFormat(nmin=-1)) == CppScalar.S64

    def test_an_unbounded_integers_signed_zero_is_not_seen_here(self):
        """``enable_neg_zero`` is the one flag this guard cannot act on.

        Not because the guard is wrong, but because
        ``AbstractFormat.from_format`` declines to believe an ``MPFixedFormat``
        about its signed zero -- so the flag is already gone by the time storage
        selection runs.  See the ``TODO`` there and
        ``docs/todos/reals-in-integer-storage.md``: this is what lets ``-x`` on an
        ``INTEGER`` parameter return ``0`` where the interpreter returns ``-0.0``.

        Pinned so that lifting the carve-out shows up here as a change in
        behaviour rather than passing unnoticed.
        """
        from fpy2.number.context.mp_fixed import MPFixedFormat
        fmt = MPFixedFormat(nmin=-1, enable_neg_zero=True)
        assert fmt.representable_in(fp.RealFloat(s=True, exp=0, c=0))
        assert choose_storage_scalar(fmt) == CppScalar.S64

    def test_a_signed_zero_bound_does_not_narrow_to_an_integer(self):
        """A bound carrying a ``-0.0`` selects a float, not an integer.

        The ladder is built by abstracting each C++ type's own ``Format``, so
        the integer rungs report ``has_neg_zero=False`` and containment rejects
        them.  Without that, the narrowest type containing "a zero" is
        ``uint8_t``, which holds the integer 0 and neither sign — the mechanism
        behind every signed-zero wrong answer in
        ``docs/todos/reals-in-integer-storage.md``.
        """
        from fpy2.analysis.format_infer.format import AbstractFormat
        pz = fp.RealFloat(s=False, exp=0, c=0)
        nz = fp.RealFloat(s=True, exp=0, c=0)
        neg = AbstractFormat(1, 0, pz, neg_bound=nz, has_neg_zero=True)
        assert choose_storage_scalar(neg.format()) == CppScalar.F32

    def test_a_range_counter_keeps_an_integer_storage(self):
        """The shape ``_range_counter_scalar`` builds, reaching the ladder via
        ``.format()``.

        A counter is an integer and never a ``-0.0``.  It only stays an integer
        because ``enable_neg_zero`` lets the materialized format say so; without
        it every loop counter becomes ``float``.
        """
        from fpy2.analysis.format_infer.format import AbstractFormat
        counter = AbstractFormat(float('inf'), 0, fp.RealFloat.from_int(10))
        assert choose_storage_scalar(counter.format()) == CppScalar.S8

    def test_integer_arithmetic_keeps_an_integer_storage(self):
        """``int8 + int8`` is ``int16_t``, not ``float``.

        The sum has a finite precision, so it materializes as a *float*-shaped
        format on the way to the ladder — and a float format admits a negative
        zero unless ``enable_neg_zero`` says otherwise.
        """
        from fpy2.analysis.format_infer.format import AbstractFormat
        a = AbstractFormat.from_format(fp.SINT8.format())
        assert choose_storage_scalar((a + a).format()) == CppScalar.S16

    def test_a_positive_zero_bound_still_narrows(self):
        """The counterweight: ``+0.0`` is exactly the integer 0, so the
        value-narrowing this backend relies on is untouched.  A blanket "no
        zero narrows" rule would cost the ~20% of corpus list element types
        that are value-narrowed."""
        assert choose_storage(SetFormat(frozenset((Fraction(0),)))) == CppScalar.U8


class TestStorageStructural:
    """``choose_storage`` recurses through TupleFormat / ListFormat."""

    def test_list_of_fp32(self):
        bound = ListFormat(fp.FP32.format())
        assert choose_storage(bound) == CppList(CppScalar.F32)

    def test_tuple_mixed(self):
        bound = TupleFormat((fp.FP32.format(), fp.SINT8.format()))
        assert choose_storage(bound) == CppTuple((CppScalar.F32, CppScalar.S8))

    def test_nested_list(self):
        bound = ListFormat(ListFormat(fp.FP64.format()))
        assert choose_storage(bound) == CppList(CppList(CppScalar.F64))


class TestStorageAggregate:
    """``aggregate_storage`` widens across multiple SSA defs."""

    def test_single_def(self):
        assert (
            aggregate_storage([fp.FP32.format()]) == CppScalar.F32
        )

    def test_widen_fp32_and_fp64(self):
        result = aggregate_storage([fp.FP32.format(), fp.FP64.format()])
        assert result == CppScalar.F64

    def test_widen_int_and_float(self):
        # Mixing FP32 and S32: F32's 24-bit mantissa is strictly less
        # than S32's 32-bit precision, so the ladder picks F64 (53-bit
        # mantissa, covers both).
        result = aggregate_storage([fp.FP32.format(), fp.SINT32.format()])
        assert result == CppScalar.F64

    def test_widen_unrepresentable_pair_rejects(self):
        """``[F32, S64]`` has no covering type — F64 doesn't have enough
        mantissa bits to hold a full S64.  Storage selection rejects
        rather than silently picking a lossy widening."""
        with pytest.raises(StorageSelectionError, match='no storage type'):
            aggregate_storage([fp.FP32.format(), fp.SINT64.format()])

    def test_widen_setformat_with_float(self):
        s = SetFormat(frozenset((Fraction(0),)))
        result = aggregate_storage([s, fp.FP32.format()])
        assert result == CppScalar.F32

    def test_aggregate_lists(self):
        result = aggregate_storage([
            ListFormat(fp.FP32.format()),
            ListFormat(fp.FP64.format()),
        ])
        assert result == CppList(CppScalar.F64)

    def test_aggregate_real_format_rejected(self):
        with pytest.raises(StorageSelectionError):
            aggregate_storage([REAL_FORMAT])
