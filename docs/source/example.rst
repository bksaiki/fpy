Examples
==================

Mixed-Precision Dot Product
---------------------------

Low-precision hardware rarely picks one format and sticks with it: a dot
product may multiply exactly, accumulate blocks of 32 elements in a narrow
format, and keep the running total in ``float32``.
In FPy that datapath *is* the program.
Every operation is correctly rounded by the context it appears in, and
contexts are ordinary values that can be passed around::

   import fpy2 as fp

   @fp.fpy
   def dot(xs: list[fp.Real], ys: list[fp.Real], K: int, block: fp.Context) -> fp.Real:
      """A blocked dot product: exact products, blocks of `K` summed in `block`."""
      assert len(xs) == len(ys) and len(xs) % K == 0
      acc = 0
      for start in range(0, len(xs), K):
         with block:
            inner_acc = 0
            for x, y in zip(xs[start:start + K], ys[start:start + K]):
               with fp.REAL:
                  p = x * y      # every product is exact ...
               inner_acc += p    # ... but block sums round to `block`
         with fp.FP32:
            acc += inner_acc     # one fp32 addition per block
      return acc

   @fp.fpy(ctx=fp.REAL)
   def dot_ref(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
      """The same dot product, with no rounding anywhere."""
      return sum([x * y for x, y in zip(xs, ys)])

The block size and the block format are just arguments, so a Python loop
can sweep over datapaths and measure each one against the exact answer::

   xs = ys = [0.1] * 4096
   exact = dot_ref(xs, ys).as_rational()   # the true value, as a Fraction

   print(f'{"block":>6} {"format":>9} {"result":>11} {"rel. error":>11}')
   for K, name, ctx in [(4096, 'float16', fp.FP16), (32, 'float16', fp.FP16),
                        (4096, 'bfloat16', fp.BF16), (32, 'bfloat16', fp.BF16),
                        (32, 'float32', fp.FP32)]:
      r = dot(xs, ys, K, ctx)
      err = abs(r.as_rational() - exact) / exact
      print(f'{K:>6} {name:>9} {float(r):>11.6f} {float(err):>11.3%}')

which prints

.. code-block:: text

    block    format      result  rel. error
     4096   float16   32.000000     21.875%
       32   float16   40.968750      0.021%
     4096  bfloat16    4.000000     90.234%
       32  bfloat16   40.250000      1.733%
       32   float32   40.959972      0.000%

Every row runs the same kernel.
:py:data:`fpy2.REAL` is the context that never rounds, so the products inside
``dot`` stay exact whatever the accumulator does, and ``dot_ref`` computes the
true real-number answer to measure against.
The table makes the case for blocking: a flat ``float16`` accumulator stalls out
once the running total is large enough that adding another ``0.1 * 0.1``
changes nothing, while summing 32 elements at a time keeps the error three
orders of magnitude smaller.

Interoperating with Python
--------------------------

Everything outside the decorated function is ordinary Python.
An FPy program is a :py:class:`fpy2.Function`, callable like any other Python
function, and it takes Python values as arguments.
Rounding contexts are ordinary values too: the sweep above keeps them in a list
and hands one to ``dot`` as a plain argument, while the ``ctx`` keyword argument
chooses the context that the body starts in.

Results come back as :py:class:`fpy2.Float` values, which support the usual
Python numeric protocol, plus :py:meth:`fpy2.Float.as_rational` for the exact
value of the number::

   r = dot(xs, ys, 32, fp.FP16)
   float(r)                                # 40.96875
   r > 40                                  # True
   r.as_rational()                         # Fraction(1311, 32)

Under :py:data:`fpy2.REAL` there is no format at all, and a value that is not a
binary float — anything that came out of a division like ``1 / 3`` — comes back
as a plain ``fractions.Fraction`` instead.
Either way the value is an exact rational, so the error of a rounded run can be
computed exactly in Python::

   err = abs(r.as_rational() - exact) / exact
   float(err)                              # 0.00021362304687488895

Both conversions are strict.
``float()`` refuses a lossy conversion, raising ``ValueError`` if the value is
not exactly representable as a Python ``float``, as with a result under, say,
``fp.MPFloatContext(100)``; and :py:meth:`fpy2.Float.as_rational` raises
``ValueError`` on an infinity or NaN, which have no rational value.

Contexts may also be used directly from Python, without an FPy program,
to round a single value::

   fp.FP16.round(0.1)                      # Float('0.099976')
   fp.FP16.round(0.1).as_rational()        # Fraction(819, 8192)

In the other direction, an arbitrary Python function can be made callable
*from* FPy code with the :py:deco:`fpy2.fpy_primitive` decorator, as long as
all of its arguments and its return value are annotated::

   @fp.fpy_primitive
   def clamp(x: fp.Real, lo: fp.Real, hi: fp.Real) -> fp.Real:
      return min(max(x, lo), hi)

   @fp.fpy
   def dot_clamped(xs: list[fp.Real], ys: list[fp.Real], K: int, block: fp.Context) -> fp.Real:
      return clamp(dot(xs, ys, K, block), 0, 40)

   dot_clamped(xs, ys, 32, fp.FP16)        # Float('40.0')

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

Note that :py:data:`fpy2.REAL` specifies the operation is performed exactly,
that is, without any rounding.

Muller's Recurrence
-------------------

Muller's recurrence is the sequence

.. math::

   x_0 = 4, \quad x_1 = \frac{17}{4}, \quad
   x_{n+1} = 108 - \frac{815 - 1500 / x_{n-1}}{x_n}

whose exact limit is 5.
The value 5 is a *repelling* fixed point of the recurrence and 100 is an
attracting one, so any rounding error at all, however small, eventually carries
the computed sequence to 100 instead::

   import fpy2 as fp

   @fp.fpy
   def muller(n: int) -> fp.Real:
      """Muller's recurrence: the exact sequence converges to 5."""
      a, b = 4, 4.25
      for _ in range(n):
         a, b = b, 108 - (815 - 1500 / a) / b
      return b

   formats = {
      'float16': fp.FP16,
      'bfloat16': fp.BF16,
      'float32': fp.FP32,
      'float64': fp.FP64,
      'exact': fp.REAL,
   }

   print('  n ' + ''.join(f'{name:>10}' for name in formats))
   for n in range(0, 26, 5):
      xs = [float(muller(n, ctx=ctx)) for ctx in formats.values()]
      print(f'{n:>3} ' + ''.join(f'{x:>10.4f}' for x in xs))

which prints

.. code-block:: text

     n    float16  bfloat16   float32   float64     exact
     0     4.2500    4.2500    4.2500    4.2500    4.2500
     5    98.5625   97.5000    4.9831    4.9108    4.9108
    10   100.0000  100.0000   99.9626    4.9914    4.9928
    15   100.0000  100.0000  100.0000  102.0400    4.9994
    20   100.0000  100.0000  100.0000  100.0000    5.0000
    25   100.0000  100.0000  100.0000  100.0000    5.0000

Each column runs the same FPy program under a different rounding context.
The first four are IEEE 754 formats, and more precision only buys more
iterations before the sequence collapses.
The last column uses :py:data:`fpy2.REAL`, which computes the real sequence
exactly, and is the only one that shows what the recurrence actually does.
