Builtins
========

The :doc:`expansion <expansion>` page covers FPy's surface syntax, independent
of how an implementation spells it; this page covers the names FPy's library
provides and what they mean. *Primitives* are core operators. Every other
builtin function could be written as an FPy function and is given as an ``@fp.fpy``
program: a call expands to that body, under the rounding context where it
expands; its parameters are the call's arguments.

.. note::

   A call hoists to a fresh variable first; in an ``@fp.fpy`` program,
   ``return e`` is the assignment to that variable.

Primitives
----------

Rounded arithmetic
~~~~~~~~~~~~~~~~~~

The full language supports many rounded operations;
these operations are elements of :math:`\mathit{Arith}`.

.. list-table::
   :widths: 26 74
   :header-rows: 1

   * - Kind
     - Operators
   * - Arithmetic
     - ``e1 + e2``, ``e1 - e2``, ``e1 * e2``, ``e1 / e2``, ``-e``, ``fp.fabs(e)``
   * - Fused
     - ``fp.fma(e1, e2, e3)``
   * - Algebraic
     - ``fp.sqrt(e)``, ``fp.cbrt(e)``, ``e1 ** e2``
   * - Trigonometric
     - ``fp.sin(e)``, ``fp.cos(e)``, ``fp.tan(e)``, ``fp.asin(e)``,
       ``fp.acos(e)``, ``fp.atan(e)``, ``fp.atan2(e1, e2)``
   * - Hyperbolic
     - ``fp.sinh(e)``, ``fp.cosh(e)``, ``fp.tanh(e)``, ``fp.asinh(e)``,
       ``fp.acosh(e)``, ``fp.atanh(e)``
   * - Exponential
     - ``fp.exp(e)``, ``fp.exp2(e)``, ``fp.expm1(e)``, ``fp.log(e)``,
       ``fp.log2(e)``, ``fp.log10(e)``, ``fp.log1p(e)``
   * - Special
     - ``fp.erf(e)``, ``fp.erfc(e)``, ``fp.lgamma(e)``, ``fp.tgamma(e)``
   * - Remainder
     - ``e1 % e2``, ``fp.fmod(e1, e2)``, ``fp.remainder(e1, e2)``
   * - Integer-valued
     - ``fp.ceil(e)``, ``fp.floor(e)``, ``fp.trunc(e)``, ``fp.roundint(e)``,
       ``fp.nearbyint(e)``
   * - Rounding
     - ``fp.round(e)``, ``fp.round_at(e, n)``
   * - Sign and exponent
     - ``fp.copysign(e1, e2)``, ``fp.logb(e)``
   * - Constant
     - ``fp.const_pi()``

``fp.fma(e1, e2, e3)`` computes ``e1 * e2 + e3`` with a *single* rounding,
:math:`C(\exact{e_1 \cdot e_2 + e_3})`. The three remainders differ in sign
convention: the divisor's, the dividend's, and nearest-zero. The integer-valued
operators differ in which integer they choose. ``fp.round(e)`` is idempotent,
and ``fp.round_at(e, n)`` rounds at digit position ``n`` first.

Exact operations
~~~~~~~~~~~~~~~~

The full language supports many exact operations;
these operations are elements of :math:`\mathit{Exact}`.

.. list-table::
   :widths: 26 74
   :header-rows: 1

   * - Kind
     - Operators
   * - Comparison
     - ``e1 < e2``, ``e1 <= e2``, ``e1 > e2``, ``e1 >= e2``, ``e1 == e2``,
       ``e1 != e2``
   * - Classification
     - ``fp.isfinite(e)``, ``fp.isinf(e)``, ``fp.isnan(e)``,
       ``fp.isnormal(e)``, ``fp.signbit(e)``
   * - Logical
     - ``not e``
   * - Size
     - ``len(xs)``, ``fp.size(xs, k)``, ``fp.dim(xs)``
   * - Special values
     - ``fp.nan()``, ``fp.inf()``

A chained comparison is the conjunction of adjacent pairwise tests, and all six
chain. The four ordering tests take numbers, while ``==`` and ``!=`` compare
lists element-wise and tuples field-wise, and reject operands of unequal type.

Literals
--------

These are numeric literals spelled as calls: their arguments are literals, not
expressions. Each denotes its number exactly (**X-Num**).

.. list-table::
   :widths: 42 58
   :header-rows: 1

   * - FPy form
     - Denotes
   * - ``fp.hexfloat(h)``
     - the value of the hexadecimal float string ``h``
   * - ``fp.rational(p, q)``
     - :math:`p/q`
   * - ``fp.digits(m, e, b)``
     - :math:`m \cdot b^{e}`

Contexts
--------

``fp.REAL`` is the context literal :math:`\R`. Every other context is a
:math:`\mathsf{ctx}\ e`. The full language provides them as *values*—``fp.FP64``
and the other named contexts—and as *context constructors*, functions such as
``fp.IEEEContext`` that build one from its parameters. Each is
:math:`\mathsf{ctx}` applied to a record describing the context.

Accessors and casts
-------------------

``fp.fst(pair)`` and ``fp.snd(pair)`` take the halves of a pair. Both require a
tuple of exactly two elements::

    @fp.fpy
    def fst(pair: tuple[Any, Any]) -> Any:
        a, b = pair
        return a

    @fp.fpy
    def snd(pair: tuple[Any, Any]) -> Any:
        a, b = pair
        return b

``fp.cast(e)`` rounds, then requires the result to be exact::

    @fp.fpy
    def cast(e: fp.Real) -> fp.Real:
        t = fp.round(e)
        assert t == e
        return t

Lists
-----

``range(start, stop, step)`` counts its iterations before filling rather than
dividing to get the length: ``step`` may be negative, and a rounded division
must not fix a list's length::

    @fp.fpy
    def range(start: int, stop: int, step: int) -> list[fp.Real]:
        with fp.REAL:
            n = 0
            i = start
            while (i < stop and step > 0) or (i > stop and step < 0):
                n = n + 1
                i = i + step
        acc = fp.empty(n)
        with fp.REAL:
            i = start
            j = 0
            while j < n:
                acc[j] = i
                i = i + step
                j = j + 1
        return acc

``range(start, stop)`` defaults the step::

    range(start, stop, 1)

``range(stop)`` defaults the start as well::

    range(0, stop)

``slice(xs, start, stop)`` takes exactly ``stop - start`` elements, and bounds
are not clamped::

    @fp.fpy
    def slice(xs: list[Any], start: int, stop: int) -> list[Any]:
        return [xs[i] for i in range(start, stop)]

``zip(xs1, ..., xsk)`` takes ``len(xs1)`` elements and asserts that every other
iterable is that long; unequal lengths are undefined, so the assertion is a
claim rather than a check with a defined failure. ``enumerate(xs)`` pairs each
element with its index::

    @fp.fpy
    def zip2(xs: list[Any], ys: list[Any]) -> list[tuple[Any, Any]]:
        assert len(ys) == len(xs)
        return [(xs[i], ys[i]) for i in range(len(xs))]

    @fp.fpy
    def enumerate(xs: list[Any]) -> list[tuple[fp.Real, Any]]:
        return [(i, xs[i]) for i in range(len(xs))]

.. note::

   Rebuilding allocates a fresh cell per element, so a slice copies the cells
   rather than sharing them: a write to ``ys[k]`` of ``ys = xs[i:j]`` does not
   reach ``xs``. Those cells hold the same rows, though, so ``ys[k][l] = e``
   does.

Selection and composites
------------------------

``max(e1, e2)`` and ``min(e1, e2)`` propagate NaN and break ``±0`` ties by sign,
independent of argument order::

    @fp.fpy
    def maximum(e1: fp.Real, e2: fp.Real) -> fp.Real:
        if fp.isnan(e1) or fp.isnan(e2):
            return e1 if fp.isnan(e1) else e2   # any NaN operand propagates
        return e1 if e1 > e2 or (e1 == e2 and not fp.signbit(e1)) else e2  # tie: +0

    @fp.fpy
    def minimum(e1: fp.Real, e2: fp.Real) -> fp.Real:
        if fp.isnan(e1) or fp.isnan(e2):
            return e1 if fp.isnan(e1) else e2
        return e1 if e1 < e2 or (e1 == e2 and fp.signbit(e1)) else e2  # tie: -0

Their variadic form folds left-to-right, so ``max(e1, e2, e3)`` is
``max(max(e1, e2), e3)``; the single-list ``max(xs)`` folds over ``xs`` and has
no empty case.

``fp.fdim(e1, e2)`` and ``fp.hypot(e1, e2)`` are *composite*: each computes its
defining expression exactly and rounds **once**; rounding each step would give a
different result::

    @fp.fpy
    def fdim(e1: fp.Real, e2: fp.Real) -> fp.Real:
        with fp.REAL:
            t = max(e1 - e2, 0)
        return fp.round(t)

    @fp.fpy
    def hypot(e1: fp.Real, e2: fp.Real) -> fp.Real:
        with fp.REAL:
            t = e1 * e1 + e2 * e2
        return fp.sqrt(t)

Reductions
----------

``sum(xs)`` folds with ``+``, rounding each step; the empty sum is exact ``0``::

    @fp.fpy
    def sum(xs: list[fp.Real]) -> fp.Real:
        if len(xs) == 0:
            return 0
        acc = xs[0]
        for x in xs[1:]:
            acc = acc + x
        return acc

``any(bs)`` and ``all(bs)`` fold with the logical operators, so nothing rounds.
Each seeds with its operator's identity, which is also its empty case, so unlike
``min`` and ``max`` both are total on the empty list::

    @fp.fpy
    def any_(bs: list[bool]) -> bool:
        acc = False
        for b in bs:
            acc = acc or b
        return acc

    @fp.fpy
    def all_(bs: list[bool]) -> bool:
        acc = True
        for b in bs:
            acc = acc and b
        return acc

The element type is exactly ``bool``: FPy has no truthiness, so
``any([1.0, 0.0])`` is a type error rather than a zero test.

Constants
---------

Every constant expands to an expression that rounds exactly once.
``fp.const_pi()`` is the primitive. The simple cases round in their outermost
operation.

.. list-table::
   :widths: 42 58
   :header-rows: 1

   * - FPy form
     - Equivalent FPy form
   * - ``fp.const_e()``
     - ``fp.exp(1)``
   * - ``fp.const_ln2()``
     - ``fp.log(2)``
   * - ``fp.const_sqrt2()``
     - ``fp.sqrt(2)``
   * - ``fp.const_sqrt1_2()``
     - ``fp.sqrt(0.5)``

A *composed* constant also rounds exactly once, with every other operation
exact under ``fp.REAL``. Scaling by a power of two is exact, so
``fp.const_pi_2()`` and ``fp.const_pi_4()`` round first and scale after::

    @fp.fpy
    def const_pi_2() -> fp.Real:
        t = fp.const_pi()
        with fp.REAL:
          return t / 2

    @fp.fpy
    def const_pi_4() -> fp.Real:
        t = fp.const_pi()
        with fp.REAL:
          return t / 4

``fp.const_1_pi()``, ``fp.const_2_pi()``, ``fp.const_2_sqrt_pi()``,
``fp.const_log2e()``, and ``fp.const_log10e()`` compute their operand exactly
and round in the root operation::

    @fp.fpy
    def const_1_pi() -> fp.Real:
        with fp.REAL:
            t = fp.const_pi()
        return 1 / t

    @fp.fpy
    def const_2_pi() -> fp.Real:
        with fp.REAL:
            t = fp.const_pi()
        return 2 / t

    @fp.fpy
    def const_2_sqrt_pi() -> fp.Real:
        with fp.REAL:
            t = fp.sqrt(fp.const_pi())
        return 2 / t

    @fp.fpy
    def const_log2e() -> fp.Real:
        with fp.REAL:
            t = fp.exp(1)
        return fp.log2(t)

    @fp.fpy
    def const_log10e() -> fp.Real:
        with fp.REAL:
            t = fp.exp(1)
        return fp.log10(t)

.. note::

   Unlike ``fp.const_pi_2()`` and ``fp.const_pi_4()``, these five have no
   evaluation as written: they need an exact transcendental intermediate, which
   ``fp.REAL`` cannot represent. They exist as specifications for compatibility
   with
   `FPCore <https://fptalks.org/spec/index.html>`_.
