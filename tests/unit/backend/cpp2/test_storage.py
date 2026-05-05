"""
Tests for cpp2 storage-type selection (Phase 1 of the backend-cpp plan).
"""

import fpy2 as fp
import pytest

from fractions import Fraction
from fpy2.analysis.format_infer import ListFormat, SetFormat, TupleFormat
from fpy2.backend.cpp2.storage import (
    StorageSelectionError,
    aggregate_storage,
    choose_storage,
    choose_storage_scalar,
)
from fpy2.backend.cpp2.types import CppList, CppScalar, CppTuple
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
