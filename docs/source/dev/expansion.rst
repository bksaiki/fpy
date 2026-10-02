Expansion
=========

The :doc:`core semantics <semantics>` page covers a minimal fragment of FPy;
this page covers the rest of its surface syntax by expanding it into the core.
The :doc:`builtins <builtins>` page covers the names FPy's library provides.

Expansion has three parts. Expressions and statements translate to the core
compositionally. The core separates effects from expressions: calls,
allocations, and heap reads are statements or explicit dereferences, so an FPy
expression may need statements that run before it. *Derived forms* rewrite to
other FPy forms; every rewrite is a *macro* that substitutes operands in place,
and a rewrite that repeats an operand binds it to a fresh variable first, so it
is evaluated once. Throughout, a variable written :math:`t`, :math:`t_1`,
:math:`t_2`, ``t1``, ``t2``, and so on is fresh.

FPy syntax is set in monospace (:math:`\fpy{while e: s}`) and core syntax in
math (:math:`\mathsf{while}\ e'\ \mathsf{do}\ s'`). A primed metavariable
names the core counterpart of its FPy namesake. In FPy code, ``ℝ`` is the real
rounding context, however an implementation spells it.

Expressions
-----------

An expression expands to a core statement and a core expression:

.. math::

   \fpy{e} \leadsto (s', e')

read ":math:`\fpy{e}` expands to :math:`s'`, after which :math:`e'` computes its
value". Running :math:`s'` and then evaluating :math:`e'` is equivalent to
evaluating :math:`\fpy{e}`.

Constants and variables expand to no statements. A numeric literal
:math:`\fpy{n}` expands to the number :math:`n` it denotes, exactly: ``0.1`` is
:math:`1/10`.

.. math::

   \frac{}{\fpy{False} \leadsto (\mathsf{skip}, \mathsf{false})}
   \qquad
   \frac{}{\fpy{True} \leadsto (\mathsf{skip}, \mathsf{true})}
   \tag{X-Bool}

.. math::

   \frac{}{\fpy{n} \leadsto (\mathsf{skip}, n)}
   \tag{X-Num}

.. math::

   \frac{}{\fpy{ℝ} \leadsto (\mathsf{skip}, \R)}
   \tag{X-Real}

.. math::

   \frac{}{\fpy{x} \leadsto (\mathsf{skip}, x)}
   \tag{X-Var}

A compound expression expands its operands left to right and concatenates their
statements. Operators and tuples add no statements; a tuple is a record whose
labels are its positions.

.. math::

   \frac{\fpy{ei} \leadsto (s_i', e_i') \quad (1 \le i \le k)}
        {\fpy{op(e1, ..., ek)} \leadsto
         (s_1' \,\mathsf{;}\, \cdots \,\mathsf{;}\, s_k',\
          \mathit{op}(e_1', \ldots, e_k'))}
   \tag{X-Op}

.. math::

   \frac{\fpy{ei} \leadsto (s_i', e_i') \quad (1 \le i \le m)}
        {\fpy{(e1, ..., em)} \leadsto
         (s_1' \,\mathsf{;}\, \cdots \,\mathsf{;}\, s_m',\
          \{\, 1 = e_1', \ldots, m = e_m' \,\})}
   \tag{X-Tuple}

Indexing reads the heap, so it binds what it reads: a later operand's
statements may write the same cell.

.. math::

   \frac{\fpy{e1} \leadsto (s_1', e_1') \quad \fpy{e2} \leadsto (s_2', e_2')}
        {\fpy{e1[e2]} \leadsto
         (s_1' \,\mathsf{;}\, s_2' \,\mathsf{;}\, t = \mathsf{!}\,(e_1'[e_2']),\ t)}
   \tag{X-Index}

A list allocates one cell per element.

.. math::

   \frac{\fpy{ei} \leadsto (s_i', e_i') \quad (1 \le i \le m)}
        {\fpy{[e1, ..., em]} \leadsto
         (s_1' \,\mathsf{;}\, \cdots \,\mathsf{;}\, s_m' \,\mathsf{;}\,
          t_1 = \mathsf{ref}\ e_1' \,\mathsf{;}\, \cdots \,\mathsf{;}\,
          t_m = \mathsf{ref}\ e_m' \,\mathsf{;}\,
          t = [\, t_1, \ldots, t_m \,],\ t)}
   \tag{X-List}

.. note::

   A list is a list of *references*: construction allocates one cell per
   element, so a list's value is a list of locations. **E-Update** replaces a
   cell's contents and no rule changes a list's length, so FPy has no
   ``append``.

A call passes its arguments as a tuple, and the callee binds each parameter to
its field.

.. math::

   \frac{\fpy{ei} \leadsto (s_i', e_i') \quad (1 \le i \le k)}
        {\fpy{f(e1, ..., ek)} \leadsto
         (s_1' \,\mathsf{;}\, \cdots \,\mathsf{;}\, s_k' \,\mathsf{;}\,
          t = f\ \{\, 1 = e_1', \ldots, k = e_k' \,\},\ t)}
   \tag{X-Call}

A conditional expression runs only the branch it takes, so each branch's
statements stay inside it.

.. math::

   \frac{\fpy{e1} \leadsto (s_1', e_1') \quad \fpy{e2} \leadsto (s_2', e_2')
         \quad \fpy{e3} \leadsto (s_3', e_3')}
        {\fpy{e2 if e1 else e3} \leadsto
         (s_1' \,\mathsf{;}\, \mathsf{if}\ e_1'\
          \mathsf{then}\ (s_2' \,\mathsf{;}\, t = e_2')\
          \mathsf{else}\ (s_3' \,\mathsf{;}\, t = e_3'),\ t)}
   \tag{X-Cond}

``fp.empty(d1, ..., dn)`` allocates too, creating a nested ``d1 x ... x dn``
list. Its cells start unspecified, so a program that reads one before writing
it is undefined.

.. admonition:: Open issue

   ``fp.empty`` is the one syntactic form with no expansion: the core's list
   constructor is fixed-width, so nothing there allocates a run-time number of
   cells. Its semantics is that of a list constructor whose width is a run-time
   value: ``z = fp.empty(n)`` allocates :math:`n` fresh cells and binds ``z`` to
   the list of their locations, nesting for higher dimensions.

Statements
----------

A statement expands to a core statement:

.. math::

   \fpy{s} \leadsto s'

Each statement places its expressions' statements before it. A bare expression
statement keeps only their effects.

.. math::

   \frac{\fpy{e} \leadsto (s_0', e') \quad \fpy{p} \triangleright e' \leadsto s_1'}
        {\fpy{p = e} \leadsto s_0' \,\mathsf{;}\, s_1'}
   \tag{X-Assign}

.. math::

   \frac{\fpy{e} \leadsto (s_0', e')}
        {\fpy{e} \leadsto s_0'}
   \tag{X-Expr}

.. math::

   \frac{\fpy{e} \leadsto (s_0', e')}
        {\fpy{return e} \leadsto s_0' \,\mathsf{;}\, \mathsf{ret}\ e'}
   \tag{X-Ret}

.. math::

   \frac{\fpy{e} \leadsto (s_0', e')}
        {\fpy{assert e} \leadsto s_0' \,\mathsf{;}\, \mathsf{assert}\ e'}
   \tag{X-Assert}

.. math::

   \frac{}{\fpy{pass} \leadsto \mathsf{skip}}
   \tag{X-Pass}

An indexed assignment binds the cell before evaluating the value it writes.

.. math::

   \frac{\fpy{e1} \leadsto (s_1', e_1') \quad \fpy{e2} \leadsto (s_2', e_2')
         \quad \fpy{e3} \leadsto (s_3', e_3')}
        {\fpy{e1[e2] = e3} \leadsto
         s_1' \,\mathsf{;}\, s_2' \,\mathsf{;}\, t = e_1'[e_2'] \,\mathsf{;}\,
         s_3' \,\mathsf{;}\, t := e_3'}
   \tag{X-Index-Assign}

Compound statements expand their parts.

.. math::

   \frac{\fpy{s1} \leadsto s_1' \quad \fpy{s2} \leadsto s_2'}
        {\fpy{s1 ; s2} \leadsto s_1' \,\mathsf{;}\, s_2'}
   \tag{X-Seq}

.. math::

   \frac{\fpy{e} \leadsto (s_0', e') \quad \fpy{s1} \leadsto s_1'
         \quad \fpy{s2} \leadsto s_2'}
        {\fpy{if e: s1 else: s2} \leadsto
         s_0' \,\mathsf{;}\, \mathsf{if}\ e'\ \mathsf{then}\ s_1'\ \mathsf{else}\ s_2'}
   \tag{X-If}

A loop re-tests its condition each iteration, so the condition's statements run
before the loop and again at the end of the body.

.. math::

   \frac{\fpy{e} \leadsto (s_0', e') \quad \fpy{s} \leadsto s'}
        {\fpy{while e: s} \leadsto
         s_0' \,\mathsf{;}\, \mathsf{while}\ e'\ \mathsf{do}\ (s' \,\mathsf{;}\, s_0')}
   \tag{X-While}

**E-Context** evaluates a ``with``'s context expression under :math:`\R`, so its
statements run under :math:`\R` too; the store is not scoped, so their bindings
remain visible to :math:`e'`.

.. math::

   \frac{\fpy{e} \leadsto (s_0', e') \quad \fpy{s} \leadsto s'}
        {\fpy{with e as x: s} \leadsto
         \mathsf{with}\ \R\ \mathsf{as}\ t\ \mathsf{in}\ s_0' \,\mathsf{;}\,
         \mathsf{with}\ e'\ \mathsf{as}\ x\ \mathsf{in}\ s'}
   \tag{X-With}

Patterns
~~~~~~~~

An assignment's target is a *pattern*. The core has none: its assignment binds
a single variable. Binding a pattern to a core expression expands to core
statements:

.. math::

   \fpy{p} \triangleright e' \leadsto s'

A wildcard drops the value, and a tuple pattern binds the tuple, then binds each
field to its sub-pattern.

.. math::

   \frac{}{\fpy{x} \triangleright e' \leadsto x = e'}
   \tag{X-Pat-Var}

.. math::

   \frac{}{\fpy{\_} \triangleright e' \leadsto \mathsf{skip}}
   \tag{X-Pat-Wild}

.. math::

   \frac{\fpy{pi} \triangleright t.i \leadsto s_i' \quad (1 \le i \le m)}
        {\fpy{(p1, ..., pm)} \triangleright e' \leadsto
         t = e' \,\mathsf{;}\, s_1' \,\mathsf{;}\, \cdots \,\mathsf{;}\, s_m'}
   \tag{X-Pat-Tuple}

Tuple patterns nest, so ``a, (b, c) = e`` binds all three. A tuple whose length
differs from its pattern's is undefined.

Derived forms
-------------

Each syntactic form below rewrites to another term in the full FPy language.

.. note::

   A rewrite whose right side is a statement block is written in assignment
   position ``z = e``. In expression position, the form expands through that
   block:

   .. math::

      \frac{\fpy{t = e} \leadsto s'}{\fpy{e} \leadsto (s', t)}

Statements
~~~~~~~~~~

A one-armed conditional has an empty ``else``. A failing ``assert`` is stuck,
so its optional message is dropped.

.. list-table::
   :widths: 42 58
   :header-rows: 1

   * - FPy form
     - Equivalent FPy form
   * - ``if c: s``
     - ``if c: s else: pass``
   * - ``assert e, msg``
     - ``assert e``

Conditional expressions
~~~~~~~~~~~~~~~~~~~~~~~

Logical operators and comparison chains rewrite to conditional expressions
(**X-Cond**).

.. list-table::
   :widths: 42 58
   :header-rows: 1

   * - FPy form
     - Equivalent FPy form
   * - ``a and b``
     - ``b if a else False``
   * - ``a or b``
     - ``True if a else b``
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
        with ℝ:
            t2 = t2 + 1

``z = [e2 for p in e1]`` allocates the result, then fills it. A target may be a
tuple pattern::

    t1 = e1
    z = fp.empty(len(t1))
    t2 = 0
    for p in t1:
        z[t2] = e2
        with ℝ:
            t2 = t2 + 1

``z = [e3 for p1 in e1 for p2 in e2]`` nests, and ``e2`` may mention ``p1``, so
the result's length is a sum of the inner lengths rather than a product. Build the
rows with the rewrite above, then flatten; *k* generators nest the same way::

    t1 = [[e3 for p2 in e2] for p1 in e1]
    t2 = 0
    for t3 in t1:
        with ℝ:
            t2 = t2 + len(t3)
    z = fp.empty(t2)
    t4 = 0
    for t3 in t1:
        for t5 in t3:
            z[t4] = t5
            with ℝ:
                t4 = t4 + 1

Slices
~~~~~~

Slice notation calls the ``slice`` builtin (see :doc:`builtins`).

.. list-table::
   :widths: 42 58
   :header-rows: 1

   * - FPy form
     - Equivalent FPy form
   * - ``xs[start:stop]``
     - ``slice(xs, start, stop)``
   * - ``xs[start:]``
     - ``xs[start:len(xs)]``
   * - ``xs[:stop]``
     - ``xs[0:stop]``
