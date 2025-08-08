Number Systems
==================

FPy supports a variety of number systems including
multi-precision floating-point and fixed-point numbers.

Philosophy
------------------

This library embraces design principles found in the FPCore standard.
Most importantly, numerical programs need only be specified by
(i) real-number mathematical operations and (ii) rounding.
Number formats, e.g., `double` or `float`, are not first-class, as in most programming languages.
Rather, formats are just *side effects* of rounding and should be de-emphasized or eliminated entirely.
Furthermore, all values within a numerical program should be viewed as (extended) real numbers.

Rounding Contexts
------------------

FPy provides a hierarchy of abstract classes describing rounding contexts.

Abstract Contexts
^^^^^^^^^^^^^^^^^^

.. autoclass:: fpy2.Context
   :members:
   :show-inheritance:

.. autoclass:: fpy2.OrdinalContext
   :members:
   :show-inheritance:

.. autoclass:: fpy2.SizedContext
   :members:
   :show-inheritance:

.. autoclass:: fpy2.EncodableContext
   :members:
   :show-inheritance:

FPy provides a number of concrete rounding contexts.
Each context implements a particular flavor of rounding.

Floating-Point Contexts
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: fpy2.MPFloatContext
   :members:
   :show-inheritance:

.. autoclass:: fpy2.MPSFloatContext
   :members:
   :show-inheritance:

.. autoclass:: fpy2.MPBFloatContext
   :members:
   :show-inheritance:

.. autoclass:: fpy2.IEEEContext
   :members:
   :show-inheritance:

.. autoclass:: fpy2.EFloatContext
   :members:
   :show-inheritance:

.. autoclass:: fpy2.ExpContext
   :members:
   :show-inheritance:

Fixed-Point Contexts
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: fpy2.MPFixedContext
   :members:
   :show-inheritance:

.. autoclass:: fpy2.MPBFixedContext
   :members:
   :show-inheritance:

.. autoclass:: fpy2.FixedContext
   :members:
   :show-inheritance:

.. autoclass:: fpy2.SMFixedContext
   :members:
   :show-inheritance:

Real Context
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: fpy2.RealContext
   :members:
   :show-inheritance:

Common Rounding Contexts
--------------------------
.. manually documented: keep in sync with `fpy2/number/__init__.py`

FPy provides a number of aliases for common rounding contexts.

Floating-Point Contexts
^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:data:: fpy2.FP256
   :value: IEEEContext(19, 256, RM.RNE)

   Alias for the IEEE 754 octuple precision (256-bit) floating point format
   with round nearest, ties-to-even rounding mode.

.. py:data:: fpy2.FP128
   :value: IEEEContext(15, 128, RM.RNE)

   Alias for the IEEE 754 quadruple precision (128-bit) floating point format
   with round nearest, ties-to-even rounding mode.

.. py:data:: fpy2.FP64
   :value: IEEEContext(11, 64, RM.RNE)

   Alias for the IEEE 754 double precision (64-bit) floating point format
   with round nearest, ties-to-even rounding mode.

.. py:data:: fpy2.FP32
   :value: fpy2.IEEEContext(8, 32, RM.RNE)

   Alias for the IEEE 754 single precision (32-bit) floating point format
   with round nearest, ties-to-even rounding mode.

.. py:data:: fpy2.FP16
   :value: IEEEContext(5, 16, RM.RNE)

   Alias for the IEEE 754 half precision (16-bit) floating point format
   with round nearest, ties-to-even rounding mode.

.. py:data:: fpy2.TF32
   :value: IEEEContext(8, 19, RM.RNE)

   Alias for Nvidia's TensorFloat-32 (TF32) floating point format
   with round nearest, ties-to-even rounding mode.

.. py:data:: fpy2.BF16
   :value: IEEEContext(5, 16, RM.RNE)

   Alias for the BFloat16 (BF16) floating point format
   with round nearest, ties-to-even rounding mode.

.. py:data:: fpy2.S1E5M2
   :value: EFloatContext(5, 8, False, ExtNanKind.NEG_ZERO, -1, RM.RNE)

   Alias for Graphcore's FP8 format with 5 bits of exponent
   with round nearest, ties-to-even rounding mode.

   See Graphcore's `FP8 proposal <https://arxiv.org/pdf/2206.02915>`_ for more information.

.. py:data:: fpy2.S1E4M3
   :value: EFloatContext(4, 8, False, ExtNanKind.NEG_ZERO, -1, RM.RNE)

   Alias for Graphcore's FP8 format with 4 bits of exponent
   with round nearest, ties-to-even rounding mode.

   See Graphcore's `FP8 proposal <https://arxiv.org/pdf/2206.02915>`_ for more information..

.. py:data:: fpy2.MX_E5M2
   :value: IEEEContext(5, 8, RM.RNE)

   Alias for the FP8 format with 5 bits of exponent in
   the Open Compute Project (OCP) Microscaling Formats (MX) specification
   with round nearest, ties-to-even rounding mode.

   See the
   `OCP MX specification <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`_
   for more information.

.. py:data:: fpy2.MX_E4M3
   :value: EFloatContext(4, 8, False, ExtNanKind.MAX_VAL, 0, RM.RNE)

   Alias for the FP8 format with 4 bits of exponent in
   the Open Compute Project (OCP) Microscaling Formats (MX) specification
   with round nearest, ties-to-even rounding mode.

   See the
   `OCP MX specification <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`_
   for more information.

.. py:data:: fpy2.MX_E3M2
   :value: EFloatContext(3, 6, False, ExtNanKind.NONE, 0, RM.RNE)

   Alias for the FP6 format with 3 bits of exponent in
   the Open Compute Project (OCP) Microscaling Formats (MX) specification
   with round nearest, ties-to-even rounding mode.

   See the
   `OCP MX specification <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`_
   for more information.

.. py:data:: fpy2.MX_E2M3
   :value: EFloatContext(2, 6, False, ExtNanKind.NONE, 0, RM.RNE)

   Alias for the FP6 format with 2 bits of exponent in
   the Open Compute Project (OCP) Microscaling Formats (MX) specification
   with round nearest, ties-to-even rounding mode.

   See the
   `OCP MX specification <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`_
   for more information.

.. py:data:: fpy2.MX_E2M1
   :value: EFloatContext(2, 4, False, ExtNanKind.NONE, 0, RM.RNE)

   Alias for the FP4 format with 2 bits of exponent in
   the Open Compute Project (OCP) Microscaling Formats (MX) specification
   with round nearest, ties-to-even rounding mode.

   See the
   `OCP MX specification <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`_
   for more information.

.. py:data:: fpy2.MX_E8M0
   :value: ExpContext(8, 0)

   Alias for the MX scaling format in the Open Compute Project (OCP)
   Microscaling Formats (MX) specification with round nearest, ties-to-even
   rounding mode.

   This is just the exponent field of a single-precision floating-point value (`FP32`)
   with `encode(x) == 0xFF` representing NaN and `encode(x) == 0x00` representing
   the exponent one below the minimum normal exponent value.

.. py:data:: fpy2.MX_INT8
   :value: FixedContext(True, -6, 8, RM.RNE)

   Alias for the MX 8-bit integer format in the Open Compute Project (OCP)
   Microscaling Formats (MX) specification with round nearest, ties-to-even
   rounding mode.

   This implementation uses the standard asymmetric encoding inherited
   from fixed-point formats, with `+MAX_VAL = +1 63/64` and `-MAX_VAL = -2`.

.. py:data:: fpy2.FP8P1
   :value: EFloatContext(7, 8, True, ExtNanKind.NEG_ZERO, 0, RM.RNE)

   Alias for the FP8 format with 7 bits of exponent found in a draft proposal
   by the IEEE P3109 working group with round nearest, ties-to-even rounding mode.
   Format subject to change.

   See the IEEE P3109 working group 
   `draft <https://github.com/P3109/Public/blob/main/IEEE%20WG%20P3109%20Interim%20Report.pdf>`_
   for more information.

.. py:data:: fpy2.FP8P2
   :value: EFloatContext(6, 8, True, ExtNanKind.NEG_ZERO, -1, RM.RNE)

   Alias for the FP8 format with 6 bits of exponent found in a draft proposal
   by the IEEE P3109 working group with round nearest, ties-to-even rounding mode.
   Format subject to change.

   See the IEEE P3109 working group 
   `draft <https://github.com/P3109/Public/blob/main/IEEE%20WG%20P3109%20Interim%20Report.pdf>`_
   for more information.

.. py:data:: fpy2.FP8P3
   :value: EFloatContext(5, 8, True, ExtNanKind.NEG_ZERO, -1, RM.RNE)

   Alias for the FP8 format with 5 bits of exponent found in a draft proposal
   by the IEEE P3109 working group with round nearest, ties-to-even rounding mode.
   Format subject to change.

   See the IEEE P3109 working group 
   `draft <https://github.com/P3109/Public/blob/main/IEEE%20WG%20P3109%20Interim%20Report.pdf>`_
   for more information.

.. py:data:: fpy2.FP8P4
   :value: EFloatContext(4, 8, True, ExtNanKind.NEG_ZERO, -1, RM.RNE)

   Alias for the FP8 format with 4 bits of exponent found in a draft proposal
   by the IEEE P3109 working group with round nearest, ties-to-even rounding mode.
   Format subject to change.

   See the IEEE P3109 working group 
   `draft <https://github.com/P3109/Public/blob/main/IEEE%20WG%20P3109%20Interim%20Report.pdf>`_
   for more information.

.. py:data:: fpy2.FP8P5
   :value: EFloatContext(3, 8, True, ExtNanKind.NEG_ZERO, -1, RM.RNE)

   Alias for the FP8 format with 3 bits of exponent found in a draft proposal
   by the IEEE P3109 working group with round nearest, ties-to-even rounding mode.
   Format subject to change.

   See the IEEE P3109 working group 
   `draft <https://github.com/P3109/Public/blob/main/IEEE%20WG%20P3109%20Interim%20Report.pdf>`_
   for more information.

.. py:data:: fpy2.FP8P6
   :value: EFloatContext(2, 8, True, ExtNanKind.NEG_ZERO, -1, RM.RNE)

   Alias for the FP8 format with 2 bits of exponent found in a draft proposal
   by the IEEE P3109 working group with round nearest, ties-to-even rounding mode.
   Format subject to change.

   See the IEEE P3109 working group 
   `draft <https://github.com/P3109/Public/blob/main/IEEE%20WG%20P3109%20Interim%20Report.pdf>`_
   for more information.

.. py:data:: fpy2.FP8P7
   :value: EFloatContext(1, 8, True, ExtNanKind.NEG_ZERO, -1, RM.RNE)

   Alias for the FP8 format with 1 bit of exponent found in a draft proposal
   by the IEEE P3109 working group with round nearest, ties-to-even rounding mode.
   Format subject to change.

   See the IEEE P3109 working group 
   `draft <https://github.com/P3109/Public/blob/main/IEEE%20WG%20P3109%20Interim%20Report.pdf>`_
   for more information.

Fixed-Point Contexts
^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:data:: fpy2.Integer
   :value: MPFixedContext(-1, RM.RTZ)

   Alias for an arbitrary-precision integer context with
   round towards zero rounding mode.

   Numbers rounded under this context behave like Python's native `int` type.

.. py:data:: fpy2.SINT8
   :value: FixedContext(True, 0, 8, RM.RTZ, OF.WRAP)

   Alias for a signed 8-bit integer context with
   round towards zero rounding mode and wrapping overflow behavior.

   Rounding infinity or NaN under this context produces an OverflowError.

.. py:data:: fpy2.SINT16
   :value: FixedContext(True, 0, 16, RM.RTZ, OF.WRAP)

   Alias for a signed 16-bit integer context with
   round towards zero rounding mode and wrapping overflow behavior.

   Rounding infinity or NaN under this context produces an OverflowError.

.. py:data:: fpy2.SINT32
   :value: FixedContext(True, 0, 32, RM.RTZ, OF.WRAP)

   Alias for a signed 32-bit integer context with
   round towards zero rounding mode and wrapping overflow behavior.

   Rounding infinity or NaN under this context produces an OverflowError.

.. py:data:: fpy2.SINT64
   :value: FixedContext(True, 0, 64, RM.RTZ, OF.WRAP)

   Alias for a signed 64-bit integer context with
   round towards zero rounding mode and wrapping overflow behavior.

   Rounding infinity or NaN under this context produces an OverflowError.

.. py:data:: fpy2.UINT8
   :value: FixedContext(False, 0, 8, RM.RTZ, OF.WRAP)

   Alias for an unsigned 8-bit integer context with
   round towards zero rounding mode and wrapping overflow behavior.

   Rounding infinity or NaN under this context produces an OverflowError.

.. py:data:: fpy2.UINT16
   :value: FixedContext(False, 0, 16, RM.RTZ, OF.WRAP)

   Alias for an unsigned 16-bit integer context with
   round towards zero rounding mode and wrapping overflow behavior.

   Rounding infinity or NaN under this context produces an OverflowError.

.. py:data:: fpy2.UINT32
   :value: FixedContext(False, 0, 32, RM.RTZ, OF.WRAP)

   Alias for an unsigned 32-bit integer context with
   round towards zero rounding mode and wrapping overflow behavior.

   Rounding infinity or NaN under this context produces an OverflowError.

.. py:data:: fpy2.UINT64
   :value: FixedContext(False, 0, 64, RM.RTZ, OF.WRAP)

   Alias for an unsigned 64-bit integer context with
   round towards zero rounding mode and wrapping overflow behavior.

   Rounding infinity or NaN under this context produces an OverflowError.


Number Types
------------------

FPy only provides two number types.

.. autoclass:: fpy2.RealFloat
   :members:
   :show-inheritance:

.. autoclass:: fpy2.Float
   :members:
   :show-inheritance:

Rounding Utilities
------------------

Additional types for specifying rounding operations.

.. autoclass:: fpy2.RoundingMode
   :members:
   :show-inheritance:

.. autoclass:: fpy2.RoundingDirection
   :members:
   :show-inheritance:

.. autoclass:: fpy2.OverflowMode
   :members:
   :show-inheritance:

.. autoclass:: fpy2.EFloatNanKind
   :members:
   :show-inheritance:
