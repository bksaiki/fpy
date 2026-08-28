Derived Semantics
=================

The :doc:`core semantics <semantics>` covers a minimal fragment of FPy; this
page gives the rest as *rewrites*. Some forms rewrite to core syntax, others to
further FPy forms, and elaboration is the fixpoint: rewrite until only core
syntax is left.

Every rewrite is a *macro*: elaboration substitutes operands in place. A form
shown as an ``@fp.fpy`` program expands to that body, under the rounding context
where it expands.

Translating to core semantics
-----------------------------

Pure expressions
~~~~~~~~~~~~~~~~

A pure expression translates directly to a core form.
In the surface syntax, ``n`` is any integer or decimal literal, and
:math:`\text{to\_rational}(s)` converts a hexadecimal float string to a
rational number.

.. list-table::
   :widths: 42 58
   :header-rows: 1

   * - FPy form
     - Core form
   * - ``False`` / ``True``
     - :math:`\mathsf{false}` / :math:`\mathsf{true}`
   * - ``n``
     - :math:`n`
   * - ``fp.hexfloat(s)``
     - :math:`\exact{\text{to\_rational}(s)}`
   * - ``fp.rational(p, q)``
     - :math:`\exact{p/q}`
   * - ``fp.digits(m, e, b)``
     - :math:`\exact{m \cdot b^{e}}`
   * - ``fp.REAL``
     - :math:`\R`
   * - ``fp.IEEEContext(8, 32)``
     - :math:`\mathsf{ctx}\ \{ \ldots \}`
   * - ``x``
     - :math:`x`
   * - ``(e1, ..., em)``
     - :math:`(\, e_1, \ldots, e_m \,)`
   * - ``op(e1, ..., ek)``
     - :math:`\mathit{op}(e_1, \ldots, e_k)`
   * - ``xs[i]``
     - :math:`\mathsf{!}\,(xs[i])`

.. note::

  Literals are **exact**; they do not round. 
  For example, ``0.1`` is exactly :math:`1/10`.

Rounded arithmetic
^^^^^^^^^^^^^^^^^^

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
     - ``fp.fma(a, b, c)``
   * - Algebraic
     - ``fp.sqrt(e)``, ``fp.cbrt(e)``, ``e1 ** e2``
   * - Trigonometric
     - ``fp.sin(e)``, ``fp.cos(e)``, ``fp.tan(e)``, ``fp.asin(e)``,
       ``fp.acos(e)``, ``fp.atan(e)``, ``fp.atan2(y, x)``
   * - Hyperbolic
     - ``fp.sinh(e)``, ``fp.cosh(e)``, ``fp.tanh(e)``, ``fp.asinh(e)``,
       ``fp.acosh(e)``, ``fp.atanh(e)``
   * - Exponential
     - ``fp.exp(e)``, ``fp.exp2(e)``, ``fp.expm1(e)``, ``fp.log(e)``,
       ``fp.log2(e)``, ``fp.log10(e)``, ``fp.log1p(e)``
   * - Special
     - ``fp.erf(e)``, ``fp.erfc(e)``, ``fp.lgamma(e)``, ``fp.tgamma(e)``
   * - Remainder
     - ``e1 % e2``, ``fp.fmod(x, y)``, ``fp.remainder(x, y)``
   * - Integer-valued
     - ``fp.ceil(e)``, ``fp.floor(e)``, ``fp.trunc(e)``, ``fp.roundint(e)``,
       ``fp.nearbyint(e)``
   * - Rounding
     - ``fp.round(e)``, ``fp.round_at(e, n)``
   * - Sign and exponent
     - ``fp.copysign(x, y)``, ``fp.logb(e)``
   * - Constant
     - ``fp.const_pi()``

``fp.fma(a, b, c)`` computes ``a * b + c`` with a *single* rounding,
:math:`C(\exact{a \cdot b + c})`. The three remainders share a shape and differ
in the exact value they take: the sign of the divisor, the sign of the dividend,
and nearest-zero. The integer-valued operators differ in which integer they
choose. ``fp.round(e)`` is idempotent, and ``fp.round_at(e, n)`` rounds at digit
position ``n`` first.

Exact operations
^^^^^^^^^^^^^^^^

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
lists and tuples element-wise and reject operands of unequal type.

Effectful expressions
~~~~~~~~~~~~~~~~~~~~~

The full FPy language has *effectful* expressions, but the core language does
not: there, calls and allocations are statements. The translation inserts those
statements, binding each result to a fresh temporary. Below, a variable written
:math:`t`, :math:`t_1`, :math:`t_2`, and so on is fresh. This is also what makes
macro substitution safe: an effectful operand is already a variable by the time
a macro duplicates it.

.. list-table::
   :widths: 42 58
   :header-rows: 1

   * - FPy form
     - Equivalent FPy form
   * - ``... f(e) ...``
     - ``t = f(e) ; ... t ...``
   * - ``... [e1, ..., em] ...``
     - ``t = [e1, ..., em] ; ... t ...``

``fp.empty(d1, ..., dn)`` allocates too. The core's list constructor is
fixed-width, so nothing there allocates a run-time number of cells; one surface
form has to be primitive, and this is it. The comprehensions and ``range`` calls
below
build on it. Its cells start unspecified, so a program that reads one before
writing it is undefined.

.. note::

   Hoisting is a post-order traversal.
   For example, ``z = xs[0] + f(xs)`` becomes
   ``t1 = xs[0] ; t2 = f(xs) ; z = t1 + t2``.

Once in statement position, each translates to core syntax.

.. list-table::
   :widths: 42 58
   :header-rows: 1

   * - FPy form
     - Core form
   * - ``z = f(e)``
     - :math:`z = f\ e`
   * - ``z = [e1, ..., em]``
     - :math:`t_1 = \mathsf{ref}\ e_1 \,\mathsf{;}\, \cdots \,\mathsf{;}\,
       t_m = \mathsf{ref}\ e_m \,\mathsf{;}\, z = [\, t_1, \ldots, t_m \,]`

.. note::

   A list is a list of *references*: construction allocates one cell per
   element, so ``z`` binds to a list of locations. **E-Update** replaces a
   cell's contents and no rule changes a list's length, so FPy has no
   ``append``.

A call is **E-App** generalized to many arguments, so the
function map :math:`\Phi` takes a name to a parameter *list* and a body. The
body runs under the callee's declared context if it has one, else the caller's
:math:`C`.

Patterns
~~~~~~~~

An assignment's target is a *pattern*. FPy allows a wildcard where the core does
not, so it takes a fresh variable that nothing reads.

.. list-table::
   :widths: 42 58
   :header-rows: 1

   * - FPy pattern
     - Core pattern
   * - ``x``
     - :math:`x`
   * - ``_``
     - :math:`t`
   * - ``p1, ..., pm``
     - :math:`(\, p_1, \ldots, p_m \,)`

Tuple patterns nest, so ``a, (b, c) = e`` binds all three.

Statements
~~~~~~~~~~

These follow the core's statement grammar. Only the indexed assignment inserts
a statement of its own, binding the cell before writing through it.

.. list-table::
   :widths: 42 58
   :header-rows: 1

   * - FPy form
     - Core form
   * - ``p = e``
     - :math:`p = e`
   * - ``xs[i] = e``
     - :math:`t = xs[i] \,\mathsf{;}\, t := e`
   * - ``s1 ; s2``
     - :math:`s_1 \,\mathsf{;}\, s_2`
   * - ``if c: s``
     - :math:`\mathsf{if}\ c\ \mathsf{then}\ s\ \mathsf{else}\ \mathsf{skip}`
   * - ``if c: s1 else: s2``
     - :math:`\mathsf{if}\ c\ \mathsf{then}\ s_1\ \mathsf{else}\ s_2`
   * - ``while c: s``
     - :math:`\mathsf{while}\ c\ \mathsf{do}\ s`
   * - ``return e``
     - :math:`\mathsf{ret}\ e`
   * - ``with e as x: s``
     - :math:`\mathsf{with}\ e\ \mathsf{as}\ x\ \mathsf{in}\ s`
   * - ``assert e``
     - :math:`\mathsf{assert}\ e`
   * - ``pass``
     - :math:`\mathsf{skip}`

A ``while`` re-tests each iteration, so anything hoisted from its condition
repeats at the end of the body. **E-Context** evaluates a ``with``'s context
expression under :math:`\R`, so anything hoisted out of it runs there too, not
before the ``with``. An ``assert``'s optional message is used only on failure.

Derived forms
-------------

Each form below rewrites to another term in the full FPy language.
Running these rewrites to fixpoint translates a program into the core language.

Conditional expressions
~~~~~~~~~~~~~~~~~~~~~~~

The core has no conditional expression, only the statement, so one nested in a
larger expression hoists to assignment position and the statement form then
applies. Unlike a call, it carries no effect.

.. list-table::
   :widths: 42 58
   :header-rows: 1

   * - FPy form
     - Equivalent FPy form
   * - ``... (a if c else b) ...``
     - ``t = a if c else b ; ... t ...``
   * - ``z = a if c else b``
     - ``if c: z = a else: z = b``
   * - ``a and b``
     - ``b if a else False``
   * - ``a or b``
     - ``True if a else b``
   * - ``a < b <= c``
     - ``t1 = a ; t2 = b ; (t1 < t2) and (t2 <= c)``

``and`` and ``or`` short-circuit through the conditional. A chain binds every
operand but the last, so a middle one is evaluated once rather than once per
test, and the last only if the tests before it pass.

Accessors and casts
~~~~~~~~~~~~~~~~~~~

``fp.fst(e)`` and ``fp.snd(e)`` take the halves of a pair. Both require a tuple
of exactly two elements; a longer one is an error, not a shorter tuple::

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

Loops and comprehensions
~~~~~~~~~~~~~~~~~~~~~~~~

``for x in xs: s`` is an index loop over a ``while``::

    with fp.REAL:
        t = 0
    while t < len(xs):
        x = xs[t]
        s
        with fp.REAL:
            t = t + 1

``range(start, stop, step)`` counts its iterations before filling rather than
dividing to get the length: ``step`` may be negative, and a rounded division
must not fix a list's length::

    @fp.fpy
    def range3(start: int, stop: int, step: int) -> list[fp.Real]:
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

    @fp.fpy
    def range2(start: int, stop: int) -> list[fp.Real]:
        return range(start, stop, 1)

    @fp.fpy
    def range1(stop: int) -> list[fp.Real]:
        return range(0, stop)

A comprehension allocates the result, then fills it. A target may be a tuple
pattern, and *k* generators nest, giving a result whose length is the product of
their sizes. The free ``g`` is a schema variable standing for the element
expression; the single-generator case::

    @fp.fpy
    def comp(ps: list[tuple[Any, Any]]) -> list[Any]:
        acc = fp.empty(len(ps))
        with fp.REAL:
            j = 0
        for x, y in ps:
            acc[j] = g(x, y)
            with fp.REAL:
                j = j + 1
        return acc

``xs[start:stop]``, ``zip(xs, ys)``, and ``enumerate(xs)`` are comprehensions
over a ``range``. A slice takes exactly ``stop - start`` elements; an omitted
bound defaults to ``0`` or ``len(xs)``, and bounds are not clamped::

    @fp.fpy
    def slice(xs: list[Any], start: int, stop: int) -> list[Any]:
        return [xs[i] for i in range(start, stop)]

``zip(xs, ys, ...)`` takes any number of lists; the two-list case is shown, and
unequal lengths are undefined::

    @fp.fpy
    def zip2(xs: list[Any], ys: list[Any]) -> list[tuple[Any, Any]]:
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
~~~~~~~~~~~~~~~~~~~~~~~~

``max(x, y)`` and ``min(x, y)`` propagate NaN and break ``±0`` ties by sign,
independent of argument order::

    @fp.fpy
    def maximum(x: fp.Real, y: fp.Real) -> fp.Real:
        if fp.isnan(x) or fp.isnan(y):
            return x if fp.isnan(x) else y   # any NaN operand propagates
        return x if x > y or (x == y and not fp.signbit(x)) else y  # tie: +0

    @fp.fpy
    def minimum(x: fp.Real, y: fp.Real) -> fp.Real:
        if fp.isnan(x) or fp.isnan(y):
            return x if fp.isnan(x) else y
        return x if x < y or (x == y and fp.signbit(x)) else y      # tie: -0

Their variadic and single-list forms fold this binary operation left-to-right. A
*composite* operator computes its defining expression exactly and rounds
**once**; rounding each step would differ::

    @fp.fpy
    def fdim(x: fp.Real, y: fp.Real) -> fp.Real:
        with fp.REAL:
            t = max(x - y, 0)
        return fp.round(t)

    @fp.fpy
    def hypot(x: fp.Real, y: fp.Real) -> fp.Real:
        with fp.REAL:
            t = x * x + y * y
        return fp.sqrt(t)

Reductions
~~~~~~~~~~

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
~~~~~~~~~

Every constant expands to an expression whose final operation rounds, built on
the primitive ``fp.const_pi()``.

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
   * - ``fp.const_pi_2()``
     - ``fp.const_pi() / 2``
   * - ``fp.const_pi_4()``
     - ``fp.const_pi() / 4``

The last two round twice, since their operand is itself a rounded result.

A *composed* constant keeps its inner value exact under ``fp.REAL`` and rounds
only at the end::

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

.. note::

  These constants are for specification purposes and cannot be
  evaluated without runtime support. They are intended for FPCore compatability.
