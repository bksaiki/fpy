Expansion
=========

The :doc:`core semantics <semantics>` page covers a minimal fragment of FPy;
this page covers the rest of its surface syntax by expanding it into the core.
The :doc:`builtins <builtins>` page covers the functions FPy provides.

Elaboration is given as *rewrite rules* of two kinds: a syntactic form rewrites
either directly to core syntax, or to another FPy form. Rewriting to fixpoint
leaves only core syntax.

Every rewrite is a *macro*: elaboration substitutes operands in place. A rewrite
that repeats an operand binds it to a fresh variable first, so it is evaluated
once.

Translating to core semantics
-----------------------------

Each syntactic form below has a counterpart in the core. The effectful ones
reach it by hoisting to statement position first.

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
     - :math:`\{\, 1 = e_1, \ldots, m = e_m \,\}`
   * - ``op(e1, ..., ek)``
     - :math:`\mathit{op}(e_1, \ldots, e_k)`
   * - ``xs[i]``
     - :math:`\mathsf{!}\,(xs[i])`

A tuple is a record whose labels are its positions.

.. note::

   Literals are **exact**; they do not round.
   For example, ``0.1`` is exactly :math:`1/10`.

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

An assignment's target is a *pattern*. The core has none: its assignment binds
a single variable. A wildcard takes a fresh variable that nothing reads, and a
tuple pattern binds the tuple, then assigns each field to its sub-pattern.

.. list-table::
   :widths: 42 58
   :header-rows: 1

   * - FPy form
     - Core form
   * - ``x = e``
     - :math:`x = e`
   * - ``_ = e``
     - :math:`t = e`
   * - ``p1, ..., pm = e``
     - :math:`t = e \,\mathsf{;}\, p_1 = t.1 \,\mathsf{;}\, \cdots
       \,\mathsf{;}\, p_m = t.m`

Tuple patterns nest, so ``a, (b, c) = e`` binds all three. A tuple whose length
differs from its pattern's is undefined.

Statements
~~~~~~~~~~

These follow the core's statement grammar; assignment is covered under
patterns. Only the indexed assignment inserts a statement of its own, binding
the cell before writing through it.

.. list-table::
   :widths: 42 58
   :header-rows: 1

   * - FPy form
     - Core form
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
   * - ``assert e`` / ``assert e, msg``
     - :math:`\mathsf{assert}\ e`
   * - ``pass``
     - :math:`\mathsf{skip}`

A bare expression statement discards its value, so it binds a fresh variable
that nothing reads; it is worth writing only for the effects inside ``e``.
**E-Context** evaluates a ``with``'s context expression under :math:`\R`, so
anything hoisted out of it runs there too, not before the ``with``. A failing
``assert`` is stuck, so its optional message is dropped.

.. note::

   The rewrite for ``while`` statements assumes ``c`` is already a core expression.
   A condition that hoists is re-tested each iteration, so its statements run before
   the loop and again at the end of the body. Writing ``H`` for those statements
   and ``c'`` for what remains of ``c``::

       H
       while c':
           s
           H

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

The core has no conditional expression, only the statement, so a conditional in
expression position hoists even though it has no effect.

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
   * - ``... (a < b <= c) ...``
     - ``t = a < b <= c ; ... t ...``
   * - ``z = a < b <= c``
     - ``t1 = a ; t2 = b ; z = (t1 < t2) and (t2 <= c)``

``and`` and ``or`` short-circuit through the conditional. A chain binds every
operand but the last, so a middle one is evaluated once rather than once per
test, and the last only if the tests before it pass.

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

``xs[start:stop]`` is ``slice(xs, start, stop)`` (see :doc:`builtins`).
``xs[start:]`` is ``xs[start:len(xs)]``, and ``xs[:stop]`` is ``xs[0:stop]``.
