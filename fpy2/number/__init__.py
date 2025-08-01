from typing import TypeAlias

# Numbers
from .number import RealFloat, Float, Real

# Contexts
from .context import Context, OrdinalContext, SizedContext, EncodableContext
from .ext_float import ExtFloatContext, ExtFloatNanKind
from .fixed import FixedContext
from .ieee754 import IEEEContext
from .mp_fixed import MPFixedContext
from .mp_float import MPFloatContext
from .mpb_fixed import MPBFixedContext
from .mpb_float import MPBFloatContext
from .mps_float import MPSFloatContext
from .real import RealContext
from .sm_fixed import SMFixedContext

# Rounding
from .round import RoundingMode, RoundingDirection, OverflowMode

# Miscellaneous
from .native import default_float_convert, default_str_convert

###########################################################
# Type aliases

RM: TypeAlias = RoundingMode
"""alias for `RoundingMode`"""

OV: TypeAlias = OverflowMode
"""alias for `OverflowMode`"""

###########################################################
# Format aliases

REAL = RealContext()
"""
Alias for exact computation.
Operations are never rounded under this context.
"""

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

BF16 = IEEEContext(8, 16, RM.RNE)
"""
Alias for Google's Brain Floating Point (BF16) floating point format
with round nearest, ties-to-even rounding mode.
"""

S1E5M2 = ExtFloatContext(5, 8, False, ExtFloatNanKind.NEG_ZERO, -1, RM.RNE)
"""
Alias for Graphcore's FP8 format with 5 bits of exponent
with round nearest, ties-to-even rounding mode.

See Graphcore's FP8 proposal for more information: https://arxiv.org/pdf/2206.02915.
"""

S1E4M3 = ExtFloatContext(4, 8, False, ExtFloatNanKind.NEG_ZERO, -1, RM.RNE)
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

MX_E4M3 = ExtFloatContext(4, 8, False, ExtFloatNanKind.MAX_VAL, 0, RM.RNE)
"""
Alias for the FP8 format with 4 bits of exponent in
the Open Compute Project (OCP) Microscaling Formats (MX) specification
with round nearest, ties-to-even rounding mode.

See the OCP MX specification for more information:
https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
"""

MX_E3M2 = ExtFloatContext(3, 6, False, ExtFloatNanKind.NONE, 0, RM.RNE)
"""
Alias for the FP6 format with 3 bits of exponent in
the Open Compute Project (OCP) Microscaling Formats (MX) specification
with round nearest, ties-to-even rounding mode.

See the OCP MX specification for more information:
https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
"""

MX_E2M3 = ExtFloatContext(2, 6, False, ExtFloatNanKind.NONE, 0, RM.RNE)
"""
Alias for the FP6 format with 2 bits of exponent in
the Open Compute Project (OCP) Microscaling Formats (MX) specification
with round nearest, ties-to-even rounding mode.

See the OCP MX specification for more information:
https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
"""

MX_E2M1 = ExtFloatContext(2, 4, False, ExtFloatNanKind.NONE, 0, RM.RNE)
"""
Alias for the FP4 format with 2 bits of exponent in
the Open Compute Project (OCP) Microscaling Formats (MX) specification
with round nearest, ties-to-even rounding mode.

See the OCP MX specification for more information:
https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
"""

# TODO: MX_INT8

FP8P1 = ExtFloatContext(7, 8, True, ExtFloatNanKind.NEG_ZERO, 0, RM.RNE)
"""
Alias for the FP8 format with 7 bits of exponent found in
a draft proposal by the IEEE P3109 working group
with round nearest, ties-to-even rounding mode.
Format subject to change.

See the IEEE P3109 working group for more information:
https://github.com/P3109/Public/blob/main/IEEE%20WG%20P3109%20Interim%20Report.pdf
"""

FP8P2 = ExtFloatContext(6, 8, True, ExtFloatNanKind.NEG_ZERO, -1, RM.RNE)
"""
Alias for the FP8 format with 6 bits of exponent found in
a draft proposal by the IEEE P3109 working group
with round nearest, ties-to-even rounding mode.
Format subject to change.

See the IEEE P3109 working group for more information:
https://github.com/P3109/Public/blob/main/IEEE%20WG%20P3109%20Interim%20Report.pdf
"""

FP8P3 = ExtFloatContext(5, 8, True, ExtFloatNanKind.NEG_ZERO, -1, RM.RNE)
"""
Alias for the FP8 format with 5 bits of exponent found in
a draft proposal by the IEEE P3109 working group
with round nearest, ties-to-even rounding mode.
Format subject to change.

See the IEEE P3109 working group for more information:
https://github.com/P3109/Public/blob/main/IEEE%20WG%20P3109%20Interim%20Report.pdf
"""

FP8P4 = ExtFloatContext(4, 8, True, ExtFloatNanKind.NEG_ZERO, -1, RM.RNE)
"""
Alias for the FP8 format with 4 bits of exponent found in
a draft proposal by the IEEE P3109 working group
with round nearest, ties-to-even rounding mode.
Format subject to change.

See the IEEE P3109 working group for more information:
https://github.com/P3109/Public/blob/main/IEEE%20WG%20P3109%20Interim%20Report.pdf
"""

FP8P5 = ExtFloatContext(3, 8, True, ExtFloatNanKind.NEG_ZERO, -1, RM.RNE)
"""
Alias for the FP8 format with 3 bits of exponent found in
a draft proposal by the IEEE P3109 working group
with round nearest, ties-to-even rounding mode.
Format subject to change.

See the IEEE P3109 working group for more information:
https://github.com/P3109/Public/blob/main/IEEE%20WG%20P3109%20Interim%20Report.pdf
"""

FP8P6 = ExtFloatContext(2, 8, True, ExtFloatNanKind.NEG_ZERO, -1, RM.RNE)
"""
Alias for the FP8 format with 2 bits of exponent found in
a draft proposal by the IEEE P3109 working group
with round nearest, ties-to-even rounding mode.
Format subject to change.

See the IEEE P3109 working group for more information:
https://github.com/P3109/Public/blob/main/IEEE%20WG%20P3109%20Interim%20Report.pdf
"""

FP8P7 = ExtFloatContext(1, 8, True, ExtFloatNanKind.NEG_ZERO, -1, RM.RNE)
"""
Alias for the FP8 format with 1 bit of exponent found in
a draft proposal by the IEEE P3109 working group
with round nearest, ties-to-even rounding mode.
Format subject to change.

See the IEEE P3109 working group for more information:
https://github.com/P3109/Public/blob/main/IEEE%20WG%20P3109%20Interim%20Report.pdf
"""

INTEGER = MPFixedContext(-1, RoundingMode.RTZ)
"""
Alias for an arbitrary-precision integer context with
round towards zero rounding mode.

Numbers rounded under this context behave like Python's native `int` type.
"""

SINT8 = FixedContext(True, 0, 8, RM.RTZ, OV.WRAP)
"""
Alias for a signed 8-bit integer context with
round towards zero rounding mode and wrapping overflow behavior.

Rounding infinity or NaN under this context produces an OverflowError.
"""

SINT16 = FixedContext(True, 0, 16, RM.RTZ, OV.WRAP)
"""
Alias for a signed 16-bit integer context with
round towards zero rounding mode and wrapping overflow behavior.

Rounding infinity or NaN under this context produces an OverflowError.
"""

SINT32 = FixedContext(True, 0, 32, RM.RTZ, OV.WRAP)
"""
Alias for a signed 32-bit integer context with
round towards zero rounding mode and wrapping overflow behavior.

Rounding infinity or NaN under this context produces an OverflowError.
"""

SINT64 = FixedContext(True, 0, 64, RM.RTZ, OV.WRAP)
"""
Alias for a signed 64-bit integer context with
round towards zero rounding mode and wrapping overflow behavior.

Rounding infinity or NaN under this context produces an OverflowError.
"""

UINT8 = FixedContext(False, 0, 8, RM.RTZ, OV.WRAP)
"""
Alias for an unsigned 8-bit integer context with
round towards zero rounding mode and wrapping overflow behavior.

Rounding infinity or NaN under this context produces an OverflowError.
"""

UINT16 = FixedContext(False, 0, 16, RM.RTZ, OV.WRAP)
"""
Alias for an unsigned 16-bit integer context with
round towards zero rounding mode and wrapping overflow behavior.

Rounding infinity or NaN under this context produces an OverflowError.
"""

UINT32 = FixedContext(False, 0, 32, RM.RTZ, OV.WRAP)
"""
Alias for an unsigned 32-bit integer context with
round towards zero rounding mode and wrapping overflow behavior.

Rounding infinity or NaN under this context produces an OverflowError.
"""

UINT64 = FixedContext(False, 0, 64, RM.RTZ, OV.WRAP)
"""
Alias for an unsigned 64-bit integer context with
round towards zero rounding mode and wrapping overflow behavior.

Rounding infinity or NaN under this context produces an OverflowError.
"""
