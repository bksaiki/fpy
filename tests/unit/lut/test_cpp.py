import fpy2 as fp
import unittest

# Small floating-point contexts
FP8 = fp.IEEEContext(4, 8, fp.RM.RNE)
FP4 = fp.IEEEContext(2, 4, fp.RM.RNE)


# Define a simple square function using fpy2.ops
def square(x, ctx):
    """Compute x^2"""
    return fp.ops.mul(x, x, ctx=ctx)


class TestCppLUTCompilation(unittest.TestCase):
    """Tests LUT compilation to C++."""

    def test_compile_square_function(self):
        """Test compiling a LUT for the square function."""
        lut = fp.LUTGenerator.generate(square, FP8, ctx=FP8)
        cpp_code = fp.lut.backend.cpp.CppLUT.compile(lut, func_name="fp8_square", method="array")
        self.assertIn("uint8_t fp8_square(uint8_t arg0)", cpp_code)

    def test_compile_switch_method(self):
        """Test compiling a LUT using the switch method."""
        lut = fp.LUTGenerator.generate(square, FP4, ctx=FP4)
        cpp_code = fp.lut.backend.cpp.CppLUT.compile(lut, func_name="fp4_square", method="switch")
        self.assertIn("uint8_t fp4_square(uint8_t arg0)", cpp_code)
        self.assertIn("switch (arg0) {", cpp_code)
        self.assertIn("case 0: return", cpp_code)

    def test_compile_two_argument_function(self):
        """Test compiling a LUT for a two-argument function."""
        lut = fp.LUTGenerator.generate(fp.add, FP4, FP8, ctx=FP4)
        cpp_code = fp.lut.backend.cpp.CppLUT.compile(lut, func_name="fp4_add", method="array")
        self.assertIn("uint8_t fp4_add(uint8_t arg0, uint8_t arg1)", cpp_code)
