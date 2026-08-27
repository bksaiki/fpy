Derived Semantics
=================

The :doc:`core semantics <semantics>` covers a minimal fragment of FPy. This
page gives the rest, as *rewrites* that end in core syntax. Elaboration is their
fixpoint: rewrite until nothing but core syntax is left.

Two kinds of rewrite do the work. A *core* rewrite maps an FPy form onto core
syntax; a *surface* rewrite replaces it with other FPy forms, which are then
rewritten in turn. FPy is written ``like this`` and core syntax
:math:`\mathsf{like}\ \mathsf{this}`.

The sections build up in layers: pure expressions, then the expressions that
carry effects, then statements, then the forms that rewrite to other FPy, and
last those standing for whole programs.

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

``fp.fma`` computes ``a * b + c`` with a *single* rounding,
:math:`C(\exact{a \cdot b + c})`. The three remainders share a shape and differ
in the exact value they take: the sign of the divisor, the sign of the dividend,
and nearest-zero. The integer-valued operators differ in which integer they
choose. ``fp.round`` is idempotent, and ``fp.round_at`` rounds at digit position
``n`` first.

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

A chained comparison is the conjunction of adjacent pairwise tests, and all six
chain. The four ordering tests take numbers, while ``==`` and ``!=`` compare
lists and tuples element-wise and reject operands of unequal type.

Effectful expressions
~~~~~~~~~~~~~~~~~~~~~

The full FPy language has *effectful* expressions; the core does not, since it
makes calls and allocation statements. Such an expression hoists into a
preceding statement, leaving a fresh temporary behind; a variable written
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

``fp.empty(d1, ..., dn)`` allocates too, and is the one form with no core
equivalent: the core's list constructor is fixed-width, while these sizes are
run-time values. Its cells start unspecified, so a program that reads one before
writing it is undefined.

.. note::

   Hoisting runs left to right, so a read stays ahead of a call that may
   overwrite it: ``z = xs[0] + f(xs)`` becomes
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
   * - ``x1, ..., xm = e``
     - :math:`(\, x_1, \ldots, x_m \,) = e`
   * - ``xs[i] = e``
     - :math:`t = xs[i] \,\mathsf{;}\, t := e`
   * - ``_ = e``
     - :math:`t = e`, unused
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
repeats at the end of the body. A ``with``'s context expression translates like
any other, and a constructor with literal arguments such as
``fp.IEEEContext(8, 32)`` is a :math:`\mathsf{ctx}\ \{ \ldots \}` constant; what
differs is the context it runs under. **E-Context** evaluates it under
:math:`\R`, so a constructor whose arguments are computed at run time is
evaluated exactly too, and anything hoisted out of it runs under :math:`\R` as
well, not before the ``with``. An ``assert``'s optional message is used only on
failure. A call is **E-App** generalized to many arguments and foreign
callables, so the function map :math:`\Phi` takes a name to a parameter *list*
and a body; the body runs under the callee's declared context if it has one,
else the caller's :math:`C`.

Surface rewrites
----------------

These forms become other FPy, which is then rewritten in turn. They come after
hoisting because a call under a short-circuit form must be inside a branch
before anything moves.

.. list-table::
   :widths: 42 58
   :header-rows: 1

   * - FPy form
     - Equivalent FPy form
   * - ``a and b``
     - ``b if a else False``
   * - ``a or b``
     - ``True if a else b``
   * - ``a < b <= c``
     - ``(a < b) and (b <= c)``
   * - ``z = a if c else b``
     - ``if c: z = a else: z = b``
   * - ``z = fp.fst(e)``
     - ``(z, t) = e``
   * - ``z = fp.snd(e)``
     - ``(t, z) = e``

Since ``and`` short-circuits, each operand of a chained comparison is evaluated
at most once. ``fp.fst`` and ``fp.snd`` require a tuple of exactly two elements;
a longer one is an error, not a shorter tuple.

Derived programs
----------------

The remaining forms stand for whole FPy programs, which elaborate by everything
above. ``for x in xs: s`` is an index loop over a ``while``::

    @fp.fpy
    def for_loop(xs: list[fp.Real]) -> fp.Real:
        acc = 0
        i = 0
        while i < len(xs):
            x = xs[i]
            acc = acc + x  # loop body s
            i = i + 1
        return acc

Comprehensions and slices
~~~~~~~~~~~~~~~~~~~~~~~~~

``[g(x, y) for x, y in ps]`` is a list-building loop; a target may be a tuple
binding, and several generators nest as in Python, *k* of them giving *k* nested
loops whose result is the product of their sizes. The single-generator case::

    @fp.fpy
    def comp(ps: list[tuple[Any, Any]]) -> list[Any]:
        acc = fp.empty(len(ps))
        j = 0
        for x, y in ps:
            acc[j] = g(x, y)
            j = j + 1
        return acc

``xs[start:stop]`` extracts exactly ``stop - start`` elements. An omitted bound
defaults to ``0`` or ``len(xs)``, as in ``xs[1:]``; bounds are not clamped::

    @fp.fpy
    def slice(xs: list[Any], start: int, stop: int) -> list[Any]:
        return [xs[i] for i in range(start, stop)]

Reading each element and rebuilding allocates a fresh cell per element, so a
slice copies the cells rather than sharing them: for ``ys = xs[i:j]``, a write to
``ys[k]`` does not reach ``xs``. Those cells hold the same rows, though, so
``ys[k][l] = e`` does.

``zip`` takes any number of lists; the two-list case is shown, and unequal
lengths are undefined::

    @fp.fpy
    def zip2(xs: list[Any], ys: list[Any]) -> list[tuple[Any, Any]]:
        return [(xs[i], ys[i]) for i in range(len(xs))]

``enumerate(xs)`` pairs each element with its integer index::

    @fp.fpy
    def enumerate(xs: list[Any]) -> list[tuple[fp.Real, Any]]:
        return [(i, xs[i]) for i in range(len(xs))]

The one- and two-argument ``range`` are the three-argument form with defaults::

    @fp.fpy
    def range1(stop: int) -> list[fp.Real]:
        return range(0, stop)

    @fp.fpy
    def range2(start: int, stop: int) -> list[fp.Real]:
        return range(start, stop, 1)

The three-argument form counts the iterations before filling, rather than
dividing to get the length: ``step`` may be negative, and a rounded division
must not fix a list's length::

    @fp.fpy
    def range3(start: int, stop: int, step: int) -> list[fp.Real]:
        n = 0
        i = start
        while (i < stop and step > 0) or (i > stop and step < 0):
            n = n + 1
            i = i + step
        acc = fp.empty(n)
        i = start
        j = 0
        while j < n:
            acc[j] = i
            i = i + step
            j = j + 1
        return acc

``fp.cast(e)`` rounds ``e`` under the rounding context but is stuck unless the
result is exact::

    @fp.fpy
    def cast(e: fp.Real) -> fp.Real:
        t = fp.round(e)
        assert t == e
        return t

Composite and selection
~~~~~~~~~~~~~~~~~~~~~~~

A *composite* operator computes its defining expression exactly and rounds
**once**—a naive expression that rounded each step would differ::

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

Selection returns one operand exactly. ``max`` and ``min`` propagate NaN and
break ``±0`` ties by sign, independent of argument order::

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

The variadic ``max`` / ``min`` and the single-list reduce forms fold this binary
operation left-to-right.

Reductions
~~~~~~~~~~

``sum(xs)`` is a left fold with ``+``, rounding each step; the empty sum is
exact ``0``::

    @fp.fpy
    def sum(xs: list[fp.Real]) -> fp.Real:
        if len(xs) == 0:
            return 0
        acc = xs[0]
        for x in xs[1:]:
            acc = acc + x
        return acc

The *boolean* reductions fold a ``list[bool]`` with the logical operators, so nothing
rounds and the rounding context is irrelevant. Each seeds with its operator's
identity, which is also the empty case's value; unlike ``min`` / ``max``, both
are total on the empty list::

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

Only ``ConstPi`` (π) is primitive: a transcendental with no finite expression,
a single correctly-rounded value,
:math:`\langle \sigma, \mu, C, \pi \rangle \Downarrow C(\exact{\pi})`.

Every other constant is an FPy function whose **final operation rounds** (the
one in the ``return``). Where the operand is exact, one rounded operation is the
whole function:

* ``ConstE`` (e) — ``fp.exp(1)``
* ``ConstLn2`` (ln 2) — ``fp.log(2)``
* ``ConstSqrt2`` (√2) — ``fp.sqrt(2)``

Three more round twice, since their operand is itself a rounded result:

* ``ConstSqrt1_2`` (1/√2) — ``fp.sqrt(1 / 2)``
* ``ConstPi_2`` (π/2) — ``fp.const_pi() / 2``
* ``ConstPi_4`` (π/4) — ``fp.const_pi() / 4``

A *truly composed* constant keeps its inner value exact under ``with fp.REAL:``
and rounds only at the ``return``. The inner operation runs under ``fp.REAL``,
which cannot represent its result, so the program below *specifies* the constant
rather than computing it; the engine evaluates these **FPCore-compatibility**
constants directly::

    @fp.fpy
    def const_2_sqrt_pi() -> fp.Real:
        with fp.REAL:
            s = fp.sqrt(fp.const_pi())
        return 2 / s

* ``ConstLog2E`` (log₂ e) — inner ``e = fp.exp(1)``, ``return fp.log2(e)``
* ``ConstLog10E`` (log₁₀ e) — inner ``e = fp.exp(1)``, ``return fp.log10(e)``
* ``Const1_Pi`` (1/π) — inner ``p = fp.const_pi()``, ``return 1 / p``
* ``Const2_Pi`` (2/π) — inner ``p = fp.const_pi()``, ``return 2 / p``
* ``Const2_SqrtPi`` (2/√π) — shown above

``ConstNan`` / ``ConstInf`` — the IEEE 754 special values NaN and
:math:`+\infty`.

Foreign values
--------------

.. TODO: explain foreign values; nothing below lowers to core syntax yet.

.. note::

   These forms are out of scope. A foreign value cannot be embedded in the core
   syntax, so nothing here lowers.

A *foreign value* is a native Python value, opaque to FPy. No operation applies
to one: a program may only pass it along, hand it to a context constructor, or
read an attribute of it. ``e.name`` reads that attribute and classifies the
result—a native number becomes a numerical value, anything opaque stays
foreign—and rounds nothing.
