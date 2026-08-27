Derived Semantics
=================

The :doc:`core semantics <semantics>` covers only a minimal fragment of
FPy—constants, arithmetic, function calls, and the basic statements. This page
explains every other *evaluable* node in :mod:`fpy2.ast.fpyast`, each either
**(i)** evaluating like a core rule (referenced by tag, e.g. **E-Op**),
**(ii)** desugaring to a small FPy program, or **(iii)** elaborating to core
syntax that has no FPy spelling of its own, as lists do (see *Lists*).

The core leaves its two operator sets open. Most entries below name a member of
one of them: :math:`\mathit{Op}`, whose members round their exact result
(**E-Op**), or :math:`\mathit{Pred}`, whose members yield a boolean exactly
(**E-Pred**).

A node shown with an ``@fp.fpy`` program stands for that program, which is
ordinary FPy and elaborates in turn by the entries here: an expression node
elaborates to a call to it (**E-App**), a statement node to its body. A free name
in such a program, like ``g`` or ``Any``, is a schema variable.

Core expressions are pure, so any node whose elaboration below is a statement is
*lifted* out of expression position, leaving a fresh temporary in its place
(**E-App**, **E-Ref**). Lifting preserves order: a subexpression that reads the
heap is bound before a later lifted call, so ``z = xs[0] + f(xs)`` binds
``xs[0]`` before calling ``f``, which may write ``xs``.

Literals and values
-------------------

* ``Decnum``, ``Hexnum``, ``Integer``, ``Rational``, ``Digits`` — numeric
  literals; each evaluates to the *exact* real it denotes, **E-Val**. Nothing
  rounds until arithmetic uses the value, so ``0.1`` is exactly :math:`1/10`.
* ``BoolVal`` — ``True`` / ``False``, **E-Val**.
* ``Var`` — a variable reference, **E-Var**.
* ``ForeignVal`` — a native Python value; evaluates to an opaque *foreign*
  value, **E-Val** — a seventh kind of value beyond the core's six. No
  operation applies to one: a program may only pass it along, hand it to a
  context constructor, or read an attribute of it.

Classification and inspection
-----------------------------

* ``IsFinite``, ``IsInf``, ``IsNan``, ``IsNormal``, ``Signbit`` — members of
  :math:`\mathit{Pred}`, each testing its operand.
* ``Logb`` — a member of :math:`\mathit{Op}`, the normalized (integer) exponent.

Arithmetic
----------

These are members of :math:`\mathit{Op}`, differing only in the operation:
``Sub`` (``-``), ``Mul`` (``*``), ``Div`` (``/``), ``Neg``, ``Abs``, ``Sqrt``,
``Cbrt``, ``Pow`` (``**``), ``Copysign``, ``Atan2``, ``Mod`` (``%``), ``Fmod``,
``Remainder``, and the elementary functions ``Sin``, ``Cos``, ``Tan``,
``Asin``, ``Acos``, ``Atan``, ``Sinh``, ``Cosh``, ``Tanh``, ``Asinh``,
``Acosh``, ``Atanh``, ``Exp``, ``Exp2``, ``Expm1``, ``Log``, ``Log10``,
``Log1p``, ``Log2``, ``Erf``, ``Erfc``, ``Lgamma``, ``Tgamma``.

* ``Fma`` — ``a*b + c`` with a *single* rounding, :math:`C(\exact{a \cdot b + c})`.
* ``Mod`` / ``Fmod`` / ``Remainder`` — same shape, differing in the exact value:
  the sign of the divisor, the sign of the dividend, and nearest-zero.
* ``Ceil``, ``Floor``, ``Trunc``, ``RoundInt``, ``NearbyInt`` — round the exact
  integer-valued result, differing in which integer is chosen.

Rounding operators
------------------

* ``Round`` — ``fp.round(e)`` rounds ``e`` to the rounding context, :math:`C(v)`
  (**E-Op** with the identity operation); idempotent.
* ``RoundAt`` — ``fp.round_at(e, n)`` rounds ``e`` at digit position ``n``, then
  under :math:`C`.
* ``Cast`` — ``fp.cast(e)`` rounds ``e`` but is stuck unless the result is
  exact (a guarded **E-Assert**).

Tuples
------

A tuple holds its fields' values directly and is immutable—no node writes to one.
Tuples cannot be indexed.

* ``TupleExpr`` — **E-Tuple**; ``TupleBinding`` — the tuple pattern of
  **M-Tuple**.
* ``Fst`` / ``Snd`` — pair accessors: ``fp.fst(t)`` :math:`\equiv` ``a`` and
  ``fp.snd(t)`` :math:`\equiv` ``b``, where ``a, b = t`` (**M-Tuple**). Both
  require a tuple of exactly two elements; a longer one is an error, not a
  shorter tuple.

Lists
-----

Every FPy list is a list of *references* in the core semantics. Allocation is a
statement, so ``z = [e_1, ..., e_m]`` lifts to one allocation per element and a
list of the temporaries:

.. math::

   t_1 = \texttt{ref}\ e_1 \,\texttt{;}\, \ldots \,\texttt{;}\,
   t_m = \texttt{ref}\ e_m \,\texttt{;}\, z = [\, t_1, \ldots, t_m \,]

binding :math:`z` to a list of locations
:math:`[\, \ell_1, \ldots, \ell_m \,]`—one cell per element. Three
consequences:

* **Elements are mutable, the length is not.**  **E-Update** replaces a cell's
  contents, and no rule changes a list's length once built. So FPy
  has no ``append``.
* **Binding shares cells.**  **E-Assign** copies nothing, so ``ys = xs``—and a
  tuple field holding ``xs``—sees writes to its elements.
* **Ownership stops at the cell.**  Construction gives every element a fresh
  cell, but that cell holds the element's *value*: ``[xs]`` does not copy
  ``xs``'s cells, and ``xss[i]`` dereferences to a row shared with the original.

The nodes that build and read them:

* ``ListExpr`` — an **E-Ref** per element, then **E-List** over the temporaries:
  ``z = [x, y]`` :math:`\equiv`
  :math:`t_1 = \texttt{ref}\ x \,\texttt{;}\, t_2 = \texttt{ref}\ y
  \,\texttt{;}\, z = [\, t_1, t_2 \,]`.
* ``ListRef`` — ``xs[i]`` in value position, **E-Index** to the cell then
  **E-Deref** through it: ``z = xs[i]`` :math:`\equiv`
  :math:`z = \texttt{!}\, xs[i]`.

* ``Empty`` — ``fp.empty(d1, …, dn)`` allocates an uninitialized ``n``-d list; it
  writes the heap, so it lifts like a call. No equivalent form exists:
  the core's list constructor is fixed-width and the sizes are run-time values.
* ``Len`` / ``Size`` / ``Dim`` — ``len(xs)``, ``fp.size(xs, k)``, ``fp.dim(xs)``:
  exact integer counts, no rounding.
* ``Range1`` / ``Range2`` / ``Range3`` — ``range(…)`` materialized to a list of
  integers, as in Python.

Miscellaneous
-------------

* ``IfExpr`` — ``r = a if c else b`` :math:`\equiv`
  :math:`\texttt{if}\ c\ \texttt{then}\ r = a\ \texttt{else}\ r = b`, the
  expression form of the conditional. Only the selected branch runs, so a call
  under it cannot lift out; lifting applies inside each branch.
* ``Attribute`` — ``e.name`` reads an attribute of a foreign value and
  classifies the result: a native number becomes a numerical value; anything
  opaque stays foreign. No rounding.
* ``Call`` — **E-App**, generalized to many arguments and foreign callables, so
  the function map :math:`\Phi` takes a name to a parameter *list* and a body.
  Being a statement, a call outside assignment position lifts:
  ``z = f(x) + 1`` :math:`\equiv`
  :math:`t = f\ x \,\texttt{;}\, z = t + 1`. The body runs under the callee's
  declared context if it has one, else the caller's :math:`C`.

Logical operators
-----------------

* ``Not`` — a member of :math:`\mathit{Pred}`, boolean negation. Its operand is
  a boolean, which **E-Pred** allows: it takes values, not just numbers.
* ``And`` / ``Or`` — ``a and b`` :math:`\equiv` ``b if a else False``, and
  ``a or b`` :math:`\equiv` ``True if a else b``.

Comparisons
-----------

* ``Compare`` — a chained comparison is the conjunction of adjacent pairwise
  tests, all six of ``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=`` chaining:
  ``a < b <= c`` :math:`\equiv` ``(a < b) and (b <= c)``. Since ``and``
  short-circuits, each operand is evaluated at most once. All six are members of
  :math:`\mathit{Pred}`; the four ordering tests take numbers, while ``==`` and
  ``!=`` compare lists and tuples element-wise and reject operands of unequal
  type.

Statements
----------

* ``StmtBlock`` — a statement sequence, **E-Seq-Normal** / **E-Seq-Return**;
  empty is **E-Skip**.
* ``Assign`` — **E-Assign** (pattern via **M-Var** / **M-Tuple**).
* ``IndexedAssign`` — **E-Index** to the cell, then **E-Update** through it:
  ``xs[i] = e`` :math:`\equiv`
  :math:`y = xs[i] \,\texttt{;}\, y := e`. The temporary is undereferenced,
  unlike ``ListRef``, since ``y`` has to hold the cell.
* ``If1Stmt`` — ``if c: s`` :math:`\equiv`
  :math:`\texttt{if}\ c\ \texttt{then}\ s\ \texttt{else}\ \texttt{skip}`.
* ``IfStmt`` — **E-If-True** / **E-If-False**.
* ``WhileStmt`` — **E-While-True** / **E-While-False**. The test runs each
  iteration, so anything lifted from the condition repeats at the end of the body.
* ``ForStmt`` — ``for x in xs: s`` is an index loop over a ``WhileStmt``::

    @fp.fpy
    def for_loop(xs: list[fp.Real]) -> fp.Real:
        acc = 0
        i = 0
        while i < len(xs):
            x = xs[i]
            acc = acc + x  # loop body s
            i = i + 1
        return acc

* ``ContextStmt`` — **E-Context**. The context expression elaborates like any
  other expression, and a constructor with literal arguments such as
  ``fp.IEEEContext(8, 32)`` is a :math:`\texttt{ctx}\ \{ \ldots \}` constant.
  The only difference is the context it runs under: **E-Context** evaluates it
  under :math:`\R`, so a constructor whose arguments are computed at run time is
  evaluated exactly too—and anything lifted out of it runs under :math:`\R` as
  well, not before the ``with``.
* ``AssertStmt`` — **E-Assert** (the optional message is used only on failure).
* ``EffectStmt`` — evaluate an expression and discard the result (``_ = e``).
* ``ReturnStmt`` — **E-Ret**; ``PassStmt`` — **E-Skip**.

List comprehensions
-------------------

* ``ListComp`` — a list-building loop; a target may be a tuple binding
  (**M-Tuple**), and several generators nest as in Python: *k* of them give *k*
  nested loops, and the result's length is the product of their sizes. The
  single-generator case, for an element expression ``g``,
  ``[g(x, y) for x, y in ps]``::

    @fp.fpy
    def comp(ps: list[tuple[Any, Any]]) -> list[Any]:
        acc = fp.empty(len(ps))
        j = 0
        for x, y in ps:
            acc[j] = g(x, y)
            j = j + 1
        return acc

* ``ListSlice`` — ``xs[start:stop]`` extracts exactly ``stop - start`` elements.
  An omitted bound defaults to ``0`` or ``len(xs)``, as in ``xs[1:]``; bounds are
  not clamped::

    @fp.fpy
    def slice(xs: list[Any], start: int, stop: int) -> list[Any]:
        return [xs[i] for i in range(start, stop)]

  Reading each element and rebuilding allocates a fresh cell per element, so a
  slice copies the cells rather than sharing them: for ``ys = xs[i:j]``, a write to
  ``ys[k]`` does not reach ``xs``. Those cells hold the same rows, though, so
  ``ys[k][l] = e`` does.

* ``Zip`` — corresponding elements as tuples. ``zip`` takes any number of lists;
  the two-list case is shown, and unequal lengths are undefined::

    @fp.fpy
    def zip2(xs: list[Any], ys: list[Any]) -> list[tuple[Any, Any]]:
        return [(xs[i], ys[i]) for i in range(len(xs))]

* ``Enumerate`` — pairs each element with its integer index::

    @fp.fpy
    def enumerate(xs: list[Any]) -> list[tuple[fp.Real, Any]]:
        return [(i, xs[i]) for i in range(len(xs))]

Composite and selection
-----------------------

**Selection** returns one operand exactly (no rounding). ``Max`` / ``Min``
propagate NaN and break ``±0`` ties by sign, independent of argument order::

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

The variadic ``max`` / ``min`` and the single-list reduce forms ``AMax`` /
``AMin`` fold this binary operation left-to-right.

**Composite** operators compute their defining expression exactly and round
**once** (a naive expression that rounded each step would differ):

* ``Fdim`` — ``fp.fdim(x, y)``::

    @fp.fpy
    def fdim(x: fp.Real, y: fp.Real) -> fp.Real:
        with fp.REAL:
            t = max(x - y, 0)
        return fp.round(t)

* ``Hypot`` — ``fp.hypot(x, y)``::

    @fp.fpy
    def hypot(x: fp.Real, y: fp.Real) -> fp.Real:
        with fp.REAL:
            t = x * x + y * y
        return fp.sqrt(t)

Reductions
----------

* ``Sum`` — ``sum(xs)`` is a left fold with ``+`` (rounding each step; the empty
  sum is exact ``0``)::

    @fp.fpy
    def sum(xs: list[fp.Real]) -> fp.Real:
        if len(xs) == 0:
            return 0
        acc = xs[0]
        for x in xs[1:]:
            acc = acc + x
        return acc

The *boolean* reductions fold a ``list[bool]`` with the logical operators, so
nothing rounds and the context is irrelevant. Each seeds with its operator's
identity, which is also the empty case's value; unlike ``min``/``max``, both are
total on the empty list.

* ``AnyOf`` — ``any(bs)`` is a left fold with ``or``; ``any([])`` is ``False``::

    @fp.fpy
    def any_(bs: list[bool]) -> bool:
        acc = False
        for b in bs:
            acc = acc or b
        return acc

* ``AllOf`` — ``all(bs)`` is a left fold with ``and``; ``all([])`` is ``True``::

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
