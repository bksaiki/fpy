Examples
==================

Exact Dot Product
------------------

The following program computes the dot product of two vectors
but with only a single rounding operation::

   import fpy2 as fp

   @fp.fpy
   def dot_prod(a: list[fp.Real], b: list[fp.Real]) -> fp.Real:
      assert len(a) == len(b)
      sum = 0
      with fp.REAL:
         for ai, bi in zip(a, b):
            sum += ai * bi
      return fp.round(sum)

We briefly note some important FPy features:

* the :py:deco:`fpy2.fpy` decorator declares the function is an FPy program; the FPy language supports only a subset of Python features.
* the user may assume the function takes two tuples of (ideal) real numbers and produces a real number result.
* each numerical computation occurs under a *rounding context* which specifies how the ideal exact result should be rounded.

Note that :py:class:`fpy2.REAL` specifies the operation is performed exactly,
that is, without any rounding.
