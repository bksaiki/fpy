"""
Constants for Python's native `float` type.
"""

FP64_NBITS = 64
"""size of a `float` in bits"""

FP64_ES = 11
"""number of exponent bits in a `float`"""

FP64_P = FP64_NBITS - FP64_ES
"""precision of a `float`"""

FP64_M = FP64_P - 1
"""number of mantissa bits in a `float`"""

FP64_EMAX = (1 << (FP64_ES - 1)) - 1
"""maximum (normalized) exponent value of a `float`"""

FP64_EMIN = 1 - FP64_EMAX
"""minimum (normalized) exponent value of a `float`"""

FP64_EXPMAX = FP64_EMAX - FP64_P + 1
"""maximum (unnormalized) exponent value of a `float`"""

FP64_EXPMIN = FP64_EMIN - FP64_P + 1
"""minimum (unnormalized) exponent value of a `float`"""

FP64_BIAS = FP64_EMAX
"""exponent bias of a `float`"""
