"""
Test for the abstract number format `A(p, exp, bound)`.
"""

import fpy2 as fp
import itertools
import unittest

from fpy2.backend.mpfx.format import AbstractFormat

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

    def test_effective_prec(self):
        """Testing effective precision calculation."""
        precs: list[int | float] = [2, 4, 8, float('inf')]
        exps: list[int | float] = [-10, -5, 0, 5, float('-inf')]
        bounds: list[fp.RealFloat | float] = [fp.RealFloat.from_int(64), fp.RealFloat.from_int(1024), float('inf')]

        for p, e, b in itertools.product(precs, exps, bounds):
            if p == float('inf') and e == float('-inf'):
                continue  # skip invalid format
            fmt = AbstractFormat(p, e, b)
            self.assertLessEqual(fmt.effective_prec(), p)

    def test_add(self, logging: bool = True):
        precs: list[int | float] = [2, 4, 8, float('inf')]
        exps: list[int | float] = [-10, -5, 0, 5, float('-inf')]
        bounds: list[fp.RealFloat | float] = [fp.RealFloat.from_int(64), fp.RealFloat.from_int(1024), float('inf')]

        # iterator over all combinations
        for p1, e1, b1 in itertools.product(precs, exps, bounds):
            if p1 == float('inf') and e1 == float('-inf'):
                continue  # skip invalid format

            fmt1 = AbstractFormat(p1, e1, b1)
            for p2, e2, b2 in itertools.product(precs, exps, bounds):
                if p2 == float('inf') and e2 == float('-inf'):
                    continue  # skip invalid format

                fmt2 = AbstractFormat(p2, e2, b2)
                fmt = fmt1 + fmt2

                if logging:
                    fmt1_str = f"A({fmt1.prec}, {fmt1.exp}, {float(fmt1.pos_bound)})"
                    fmt2_str = f"A({fmt2.prec}, {fmt2.exp}, {float(fmt2.pos_bound)})"
                    fmt_str = f"A({fmt.prec}, {fmt.exp}, {float(fmt.pos_bound)})"
                    print(f"{fmt1_str} + {fmt2_str} = {fmt_str}")

                self.assertEqual(fmt.exp, min(e1, e2))
                self.assertEqual(fmt.pos_bound, b1 + b2)


    def test_mul(self):
        precs: list[int | float] = [2, 4, 8, float('inf')]
        exps: list[int | float] = [-10, -5, 0, 5, float('-inf')]
        bounds: list[fp.RealFloat | float] = [fp.RealFloat.from_int(64), fp.RealFloat.from_int(1024), float('inf')]

        # iterator over all combinations
        params1 = itertools.product(precs, exps, bounds)
        params2 = itertools.product(precs, exps, bounds)

        for p1, e1, b1 in params1:
            if p1 == float('inf') and e1 == float('-inf'):
                continue  # skip invalid format

            fmt1 = AbstractFormat(p1, e1, b1)
            for p2, e2, b2 in params2:
                if p2 == float('inf') and e2 == float('-inf'):
                    continue  # skip invalid format

                fmt2 = AbstractFormat(p2, e2, b2)
                fmt = fmt1 * fmt2

                self.assertEqual(fmt.effective_prec(), fmt1.effective_prec() + fmt2.effective_prec())
                self.assertEqual(fmt.exp, e1 + e2)
                self.assertEqual(fmt.pos_bound, b1 * b2)
