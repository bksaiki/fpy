from typing import TypeAlias

# Numbers
from .number import Float
from .real import RealFloat

# Contexts
from .context import Context, OrdinalContext, SizedContext, EncodableContext
from .ext import ExtContext, ExtNanKind
from .ieee754 import IEEEContext
from .mp import MPContext
from .mpb import MPBContext
from .mps import MPSContext

# Rounding
from .round import RoundingMode, RoundingDirection

# Miscellaneous
from .native import default_float_convert, default_str_convert

###########################################################
# Type aliases

RM: TypeAlias = RoundingMode
"""alias for `RoundingMode`"""

###########################################################
# Format aliases

FP256 = IEEEContext(19, 256, RM.RNE)
"""
Alias for the IEEE 754 octuple precision (256-bit) floating point format
with round nearest, ties-to-even rounding mode.
"""

FP128 = IEEEContext(15, 128, RM.RNE)
"""
Alias for the IEEE 754 quadruple precision (128-bit) floating point format
with round nearest, ties-to-even rounding mode.
"""

FP64 = IEEEContext(11, 64, RM.RNE)
"""
Alias for the IEEE 754 double precision (64-bit) floating point format
with round nearest, ties-to-even rounding mode.

This context is Python's native float type.
"""

FP32 = IEEEContext(8, 32, RM.RNE)
"""
Alias for the IEEE 754 single precision (32-bit) floating point format
with round nearest, ties-to-even rounding mode.
"""

FP16 = IEEEContext(5, 16, RM.RNE)
"""
Alias for the IEEE 754 half precision (16-bit) floating point format
with round nearest, ties-to-even rounding mode.
"""

TF32 = IEEEContext(8, 19, RM.RNE)
"""
Alias for Nvidia's TensorFloat-32 (TF32) floating point format
with round nearest, ties-to-even rounding mode.
"""

BF16 = IEEEContext(5, 16, RM.RNE)
"""
Alias for Google's Brain Floating Point (BF16) floating point format
with round nearest, ties-to-even rounding mode.
"""

S1E5M2 = ExtContext(5, 8, False, ExtNanKind.NEG_ZERO, -1, RM.RNE)
"""
Alias for Graphcore's FP8 format with 5 bits of exponent
with round nearest, ties-to-even rounding mode.

See Graphcore's FP8 proposal for more information: https://arxiv.org/pdf/2206.02915.
"""

S1E4M3 = ExtContext(4, 8, False, ExtNanKind.NEG_ZERO, -1, RM.RNE)
"""
Alias for Graphcore's FP8 format with 4 bits of exponent
with round nearest, ties-to-even rounding mode.

See Graphcore's FP8 proposal for more information: https://arxiv.org/pdf/2206.02915.
"""

MX_E5M2 = IEEEContext(5, 8, RM.RNE)
"""
Alias for the FP8 format with 5 bits of exponent in
the Open Compute Project (OCP) Microscaling Formats (MX) specification
with round nearest, ties-to-even rounding mode.

See the OCP MX specification for more information:
https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
"""

MX_E4M3 = ExtContext(4, 8, False, ExtNanKind.MAX_VAL, 0, RM.RNE)
"""
Alias for the FP8 format with 4 bits of exponent in
the Open Compute Project (OCP) Microscaling Formats (MX) specification
with round nearest, ties-to-even rounding mode.

See the OCP MX specification for more information:
https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
"""

MX_E3M2 = ExtContext(3, 6, False, ExtNanKind.NONE, 0, RM.RNE)
"""
Alias for the FP6 format with 3 bits of exponent in
the Open Compute Project (OCP) Microscaling Formats (MX) specification
with round nearest, ties-to-even rounding mode.

See the OCP MX specification for more information:
https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
"""

MX_E2M3 = ExtContext(2, 6, False, ExtNanKind.NONE, 0, RM.RNE)
"""
Alias for the FP6 format with 2 bits of exponent in
the Open Compute Project (OCP) Microscaling Formats (MX) specification
with round nearest, ties-to-even rounding mode.

See the OCP MX specification for more information:
https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
"""

MX_E2M1 = ExtContext(2, 4, False, ExtNanKind.NONE, 0, RM.RNE)
"""
Alias for the FP4 format with 2 bits of exponent in
the Open Compute Project (OCP) Microscaling Formats (MX) specification
with round nearest, ties-to-even rounding mode.

See the OCP MX specification for more information:
https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
"""

# TODO: MX_INT8

FP8P1 = ExtContext(7, 8, True, ExtNanKind.NEG_ZERO, 0, RM.RNE)
"""
Alias for the FP8 format with 7 bits of exponent found in
a draft proposal by the IEEE P3109 working group
with round nearest, ties-to-even rounding mode.
Format subject to change.

See the IEEE P3109 working group for more information:
https://github.com/P3109/Public/blob/main/IEEE%20WG%20P3109%20Interim%20Report.pdf
"""

FP8P2 = ExtContext(6, 8, True, ExtNanKind.NEG_ZERO, -1, RM.RNE)
"""
Alias for the FP8 format with 6 bits of exponent found in
a draft proposal by the IEEE P3109 working group
with round nearest, ties-to-even rounding mode.
Format subject to change.

See the IEEE P3109 working group for more information:
https://github.com/P3109/Public/blob/main/IEEE%20WG%20P3109%20Interim%20Report.pdf
"""

FP8P3 = ExtContext(5, 8, True, ExtNanKind.NEG_ZERO, -1, RM.RNE)
"""
Alias for the FP8 format with 5 bits of exponent found in
a draft proposal by the IEEE P3109 working group
with round nearest, ties-to-even rounding mode.
Format subject to change.

See the IEEE P3109 working group for more information:
https://github.com/P3109/Public/blob/main/IEEE%20WG%20P3109%20Interim%20Report.pdf
"""

FP8P4 = ExtContext(4, 8, True, ExtNanKind.NEG_ZERO, -1, RM.RNE)
"""
Alias for the FP8 format with 4 bits of exponent found in
a draft proposal by the IEEE P3109 working group
with round nearest, ties-to-even rounding mode.
Format subject to change.

See the IEEE P3109 working group for more information:
https://github.com/P3109/Public/blob/main/IEEE%20WG%20P3109%20Interim%20Report.pdf
"""

FP8P5 = ExtContext(3, 8, True, ExtNanKind.NEG_ZERO, -1, RM.RNE)
"""
Alias for the FP8 format with 3 bits of exponent found in
a draft proposal by the IEEE P3109 working group
with round nearest, ties-to-even rounding mode.
Format subject to change.

See the IEEE P3109 working group for more information:
https://github.com/P3109/Public/blob/main/IEEE%20WG%20P3109%20Interim%20Report.pdf
"""

FP8P6 = ExtContext(2, 8, True, ExtNanKind.NEG_ZERO, -1, RM.RNE)
"""
Alias for the FP8 format with 2 bits of exponent found in
a draft proposal by the IEEE P3109 working group
with round nearest, ties-to-even rounding mode.
Format subject to change.

See the IEEE P3109 working group for more information:
https://github.com/P3109/Public/blob/main/IEEE%20WG%20P3109%20Interim%20Report.pdf
"""

FP8P7 = ExtContext(1, 8, True, ExtNanKind.NEG_ZERO, -1, RM.RNE)
"""
Alias for the FP8 format with 1 bit of exponent found in
a draft proposal by the IEEE P3109 working group
with round nearest, ties-to-even rounding mode.
Format subject to change.

See the IEEE P3109 working group for more information:
https://github.com/P3109/Public/blob/main/IEEE%20WG%20P3109%20Interim%20Report.pdf
"""
