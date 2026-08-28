Derived Semantics
=================

The :doc:`core semantics <semantics>` page covers semantics for
a minimal fragment of FPy; this page presents the semantics for
the full language by elaborating the surface syntax to the core syntax.

Elaboration is given as *rewrite rules* of two kinds: a syntactic form rewrites
either directly to core syntax, or to another FPy form. Rewriting to fixpoint
leaves only core syntax.

Every rewrite is a *macro*: elaboration substitutes operands in place, binding
each to a fresh variable so that it is evaluated once however often the body
mentions it. A syntactic form shown as an ``@fp.fpy`` program expands to that
body, under the rounding context where it expands, its parameters bound to the
operands.

Translating to core semantics
-----------------------------

Each syntactic form below rewrites directly to core syntax.

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

Every other context is a :math:`\mathsf{ctx}\ \{ \ldots \}`. The full language
provides them as *values*—``fp.FP64`` and the other named contexts—and as
*context constructors*, functions such as ``fp.IEEEContext`` that build one from
its parameters.

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
:math:`C(\exact{e_1 \cdot e_2 + e_3})`. The three remainders share a shape and
differ in the exact value they take: the sign of the divisor, the sign of the
dividend, and nearest-zero. The integer-valued operators differ in which integer
they choose. ``fp.round(e)`` is idempotent, and ``fp.round_at(e, n)`` rounds at
digit position ``n`` first.

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
:math:`t`, :math:`t_1`, :math:`t_2`, and so on is fresh.

.. list-table::
   :widths: 42 58
   :header-rows: 1

   * - FPy form
     - Equivalent FPy form
   * - ``... f(e) ...``
     - ``t = f(e) ; ... t ...``
   * - ``... [e1, ..., em] ...``
     - ``t = [e1, ..., em] ; ... t ...``

.. note::

   Hoisting is a post-order traversal.
   For example, ``z = xs[0] + f(xs)`` becomes
   ``t1 = xs[0] ; t2 = f(xs) ; z = t1 + t2``.

``fp.empty(d1, ..., dn)`` allocates too, creating a nested ``d1 x ... x dn``
list. Its cells start unspecified, so a program that reads one before writing
it is undefined.

.. admonition:: Open issue

   ``fp.empty`` is the one syntactic form with no rewrite: the core's list
   constructor is fixed-width, so nothing there allocates a run-time number of
   cells. Its semantics is that of a list constructor whose width is a run-time
   value: ``z = fp.empty(n)`` allocates :math:`n` fresh cells and binds ``z`` to
   the list of their locations, nesting for higher dimensions.

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
   * - ``e``
     - :math:`t = e`
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

A bare expression statement discards its value, so it binds a fresh variable
that nothing reads; it is worth writing only for the effects inside ``e``.
A ``while`` re-tests each iteration, so anything hoisted from its condition
repeats at the end of the body. **E-Context** evaluates a ``with``'s context
expression under :math:`\R`, so anything hoisted out of it runs there too, not
before the ``with``. An ``assert``'s optional message is used only on failure.

Derived forms
-------------

Each syntactic form below rewrites to another term in the full FPy language.

.. note::

   A rewrite whose right side is a statement block is written in assignment
   position; in expression position the form hoists to a fresh variable first,
   as a call does. In an ``@fp.fpy`` program, ``return e`` is the assignment to
   that target.

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

``fp.fst(pair)`` and ``fp.snd(pair)`` take the halves of a pair. Both require a
tuple of exactly two elements; a longer one is an error, not a shorter tuple::

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

``for p in e: s`` is an index loop over a ``while``::

    t1 = e
    t2 = 0
    while t2 < len(t1):
        p = t1[t2]
        s
        with fp.REAL:
            t2 = t2 + 1

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

``z = [e2 for p in e1]`` allocates the result, then fills it. A target may be a
tuple pattern::

    t1 = e1
    z = fp.empty(len(t1))
    t2 = 0
    for p in t1:
        z[t2] = e2
        with fp.REAL:
            t2 = t2 + 1

``z = [e3 for p1 in e1 for p2 in e2]`` nests, and ``e2`` may mention ``p1``, so
the result's length is a sum of the inner lengths rather than a product. Build the
rows with the rewrite above, then flatten; *k* generators nest the same way::

    t1 = [[e3 for p2 in e2] for p1 in e1]
    t2 = 0
    for t3 in t1:
        with fp.REAL:
            t2 = t2 + len(t3)
    z = fp.empty(t2)
    t4 = 0
    for t3 in t1:
        for t5 in t3:
            z[t4] = t5
            with fp.REAL:
                t4 = t4 + 1

``xs[start:stop]`` takes exactly ``stop - start`` elements; an omitted bound
defaults to ``0`` or ``len(xs)``, and bounds are not clamped::

    @fp.fpy
    def slice(xs: list[Any], start: int, stop: int) -> list[Any]:
        return [xs[i] for i in range(start, stop)]

``zip(xs, ys, ...)`` takes any number of lists; the two-list case is shown, and
unequal lengths are undefined. ``enumerate(xs)`` pairs each element with its
index::

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

Their variadic and single-list forms fold this binary operation left-to-right.

``fp.fdim(e1, e2)`` and ``fp.hypot(e1, e2)`` are *composite*: each computes its
defining expression exactly and rounds **once**, where rounding each step would
differ::

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

   These constants have no evaluation as written: they need an
   exact transcendental intermediate, which ``fp.REAL`` cannot represent.
   They exist as specifications for compatibility with
   `FPCore <https://fptalks.org/spec/index.html>`_.
