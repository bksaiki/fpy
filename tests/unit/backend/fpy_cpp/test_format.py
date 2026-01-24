"""
Test for the abstract number format `A(p, exp, bound)`.
"""

import fpy2 as fp
import unittest

from fpy2.backend.fpy_cpp.format import AbstractFormat

class TestAbstractFormat(unittest.TestCase):

    def test_construct(self):
        """Testing construction of AbstractFormat."""
        # MPFloatContext(24)
        fmt = AbstractFormat.from_context(fp.MPFloatContext(24))
        self.assertEqual(fmt.prec, 24)
        self.assertEqual(fmt.exp, float('-inf'))
        self.assertEqual(fmt.bound, float('inf'))
        # MPSFloatContext(24, -10)
        fmt = AbstractFormat.from_context(fp.MPSFloatContext(24, -10))
        self.assertEqual(fmt.prec, 24)
        self.assertEqual(fmt.exp, -33)
        self.assertEqual(fmt.bound, float('inf'))
        # MPBFloatContext(24, -10, 1.0)
        fmt = AbstractFormat.from_context(fp.MPBFloatContext(24, -10, fp.RealFloat.from_int(1)))
        self.assertEqual(fmt.prec, 24)
        self.assertEqual(fmt.exp, -33)
        self.assertEqual(fmt.bound, fp.RealFloat.from_int(1))
        # FP64
        fmt = AbstractFormat.from_context(fp.FP64)
        self.assertEqual(fmt.prec, 53)
        self.assertEqual(fmt.exp, -1074)
        self.assertEqual(fmt.bound, fp.RealFloat(exp=971, c=(1 << 53) - 1))
        # FP32
        fmt = AbstractFormat.from_context(fp.FP32)
        self.assertEqual(fmt.prec, 24)
        self.assertEqual(fmt.exp, -149)
        self.assertEqual(fmt.bound, fp.RealFloat(exp=104, c=(1 << 24) - 1))
        # MPFixedContext(-8)
        fmt = AbstractFormat.from_context(fp.MPFixedContext(-8))
        self.assertEqual(fmt.prec, float('inf'))
        self.assertEqual(fmt.exp, -7)
        self.assertEqual(fmt.bound, float('inf'))
        # MPBFixedContext(-8, 1.0)
        fmt = AbstractFormat.from_context(fp.MPBFixedContext(-8, fp.RealFloat.from_int(1)))
        self.assertEqual(fmt.prec, float('inf'))
        self.assertEqual(fmt.exp, -7)
        self.assertEqual(fmt.bound, fp.RealFloat.from_int(1))
        # INT8
        fmt = AbstractFormat.from_context(fp.SINT8)
        self.assertEqual(fmt.prec, float('inf'))
        self.assertEqual(fmt.exp, 0)
        self.assertEqual(fmt.bound, fp.RealFloat.from_int(128))

    def test_contains(self):
        """Testing containment check."""
        # FP32 \subseteq FP64
        CTX1 = AbstractFormat.from_context(fp.FP32)
        CTX2 = AbstractFormat.from_context(fp.FP64)
        self.assertTrue(CTX1.contained_in(CTX2), "Expected FP32 to be contained in FP64.")

        # MX_E5M2 \subseteq FP32
        CTX1 = AbstractFormat.from_context(fp.MX_E5M2)
        CTX2 = AbstractFormat.from_context(fp.FP32)
        self.assertTrue(CTX1.contained_in(CTX2), "Expected MX_E5M2 to be contained in FP32.")

        # FP64 ⊄ FP32
        CTX1 = AbstractFormat.from_context(fp.FP64)
        CTX2 = AbstractFormat.from_context(fp.FP32)
        self.assertFalse(CTX1.contained_in(CTX2), "Expected FP64 to not be contained in FP32.")

        # MX_E4M3 ⊄ MX_E5M2
        CTX1 = AbstractFormat.from_context(fp.MX_E4M3)
        CTX2 = AbstractFormat.from_context(fp.MX_E5M2)
        self.assertFalse(CTX1.contained_in(CTX2), "Expected MX_E4M3 to not be contained in MX_E5M2.")

        # MX_E4M3 \subseteq fixed<-9, 32>
        CTX1 = AbstractFormat.from_context(fp.MX_E4M3)
        CTX2 = AbstractFormat.from_context(fp.FixedContext(True, -9, 32))
        self.assertTrue(CTX1.contained_in(CTX2), "Expected MX_E4M3 to be contained in fixed<-9, 32>.")

        # MX_E5M2 ⊄ fixed<-9, 32>
        CTX1 = AbstractFormat.from_context(fp.MX_E5M2)
        CTX2 = AbstractFormat.from_context(fp.FixedContext(True, -9, 32))
        self.assertFalse(CTX1.contained_in(CTX2), "Expected MX_E5M2 to not be contained in fixed<-9, 32>.")

        # INT8 \subseteq FP32
        CTX1 = AbstractFormat.from_context(fp.SINT8)
        CTX2 = AbstractFormat.from_context(fp.FP32)
        self.assertTrue(CTX1.contained_in(CTX2), "Expected INT8 to be contained in FP32.")

        # INT4 \subseteq A(3, 0, 4)
        CTX1 = AbstractFormat.from_context(fp.FixedContext(True, 0, 4))
        CTX2 = AbstractFormat(3, 0, fp.RealFloat.from_int(8))
        self.assertTrue(CTX1.contained_in(CTX2), "Expected INT4 to be contained in A(3, 0, 4).")

        # INT4 \subseteq A(4, 0, 12)
        CTX1 = AbstractFormat.from_context(fp.FixedContext(True, 0, 4))
        CTX2 = AbstractFormat(4, 0, fp.RealFloat.from_int(12))
        self.assertTrue(CTX1.contained_in(CTX2), "Expected INT4 to be contained in A(4, 0, 12).")
