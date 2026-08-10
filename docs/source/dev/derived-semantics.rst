Derived Semantics
=================

The :doc:`core semantics <semantics>` covers only a minimal fragment of
FPy—constants, arithmetic, function calls, and the basic statements. This page
explains every other *evaluable* node in :mod:`fpy2.ast.fpyast`, each either
**(i)** evaluating like a core rule (referenced by tag, e.g. **E-Add**),
**(ii)** desugaring to a small FPy program, or **(iii)** elaborating to core
syntax that has no FPy spelling of its own, as lists do (see *Lists*).

The two rules leaned on most are **E-Add**
(:math:`\langle \sigma, \mu, C, e_1 + e_2 \rangle \Downarrow C(\exact{n_1 + n_2}) \,;\, \mu_2`—round
the exact result under the active context :math:`C`) and **E-Lt**
(:math:`\langle \sigma, \mu, C, e_1 < e_2 \rangle \Downarrow (n_1 < n_2) \,;\, \mu_2`—an exact
boolean, no rounding). The store :math:`\mu` is elided in the entries below:
every node threads it, list construction allocates a reference per element, and
``IndexedAssign`` is the only node that writes through one. Several operations move
values without inspecting them, so they are *polymorphic*: ``Any`` in the
programs below is any element type, not just ``fp.Real``. Type annotations,
abstract base classes, and re-exports carry no runtime behaviour and are omitted.

Literals and values
-------------------

* ``Decnum``, ``Hexnum``, ``Integer``, ``Rational``, ``Digits`` — numeric
  literals; each evaluates to the *exact* real it denotes, like **E-Num**. No
  rounding occurs until the value is used in arithmetic, so ``0.1`` is exactly
  :math:`1/10`.
* ``BoolVal`` — ``True`` / ``False``, like **E-True** / **E-False**.
* ``Var`` — a variable reference, **E-Var**.
* ``ForeignVal`` — a native Python value; evaluates to itself like **E-Num**,
  no rounding.

Constants
---------

Only ``ConstPi`` (π) is primitive: a transcendental with no finite expression,
a single correctly-rounded value—the nullary **E-Add**,
:math:`\langle \sigma, C, \pi \rangle \Downarrow C(\exact{\pi})`.

Every other constant is an FPy function whose **final operation rounds** (the
one in the ``return``). A single operation on exact rationals is the whole
function—an ordinary computable program::

    @fp.fpy
    def const_sqrt2() -> fp.Real:
        return fp.sqrt(2)

* ``ConstE`` (e) — ``fp.exp(1)``
* ``ConstLn2`` (ln 2) — ``fp.log(2)``
* ``ConstSqrt2`` (√2) — ``fp.sqrt(2)``
* ``ConstSqrt1_2`` (1/√2) — ``fp.sqrt(1 / 2)``
* ``ConstPi_2`` (π/2) — ``fp.const_pi() / 2``
* ``ConstPi_4`` (π/4) — ``fp.const_pi() / 4``

A *truly composed* constant keeps its inner value exact under ``with fp.REAL:``
and rounds only at the ``return``. That inner value is irrational, so these are
*uncomputable* **FPCore-compatibility** shims (the engine approximates them and
may be an ULP off)::

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

Arithmetic
----------

These evaluate their operands and round the exact result under :math:`C`, like
**E-Add** (:math:`C(\exact{\ldots})`), differing only in the function computed:
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

Reductions
----------

* ``Sum`` — ``sum(xs)`` is a left fold with ``+`` (rounding each step; the empty
  sum is exact ``0``). If ``xs`` has `n` elements, then the fold performs `n - 1` additions,
  each rounded under the active context :math:`C`.::

    @fp.fpy
    def sum(xs: list[fp.Real]) -> fp.Real:
        if len(xs) == 0:
            return 0
        acc = xs[0]
        for x in xs[1:]:
            acc = acc + x
        return acc

The *boolean* reductions fold a ``list[bool]`` with the logical operators, so
no rounding occurs and the active context is irrelevant. Each is seeded with
its operator's identity, which is also the value of the empty case—unlike
``min``/``max``, both are total on the empty list.

* ``AnyOf`` — ``any(bs)`` is a left fold with ``or``, where
  ``any([])`` is ``False``::

    @fp.fpy
    def any_(bs: list[bool]) -> bool:
        acc = False
        for b in bs:
            acc = acc or b
        return acc

* ``AllOf`` — ``all(bs)`` is a left fold with ``and``, where
  ``all([])`` is ``True``::

    @fp.fpy
    def all_(bs: list[bool]) -> bool:
        acc = True
        for b in bs:
            acc = acc and b
        return acc

``AnyOf``/``AllOf`` are to ``Or``/``And`` what ``AMax``/``AMin`` are to
``Max``/``Min``: the list-fold form of a scalar operator, and a distinct node so
typing stays syntax-directed (``list[bool] -> bool`` rather than
``bool -> ... -> bool``). The element type is exactly ``bool``: FPy has no
truthiness, so ``any([1.0, 0.0])`` is a type error rather than a zero test.

The fold is eager, and this **matches** Python for the syntax FPy supports.
Python's ``any``/``all`` short-circuit their *iterable*, but a list
comprehension is fully built before ``any`` is called, so
``any([pred(x) for x in xs])`` evaluates ``pred`` on every element in CPython
too; only a generator expression—which FPy does not have—makes the
short-circuit skip work. Evaluating every element is therefore faithful, not a
simplification.

Classification and inspection
-----------------------------

* ``IsFinite``, ``IsInf``, ``IsNan``, ``IsNormal``, ``Signbit`` — test the
  operand and yield a boolean, like **E-Lt** (no rounding).
* ``Logb`` — the (integer) normalized exponent, rounded under :math:`C` like
  **E-Add**.

Logical operators
-----------------

* ``Not`` — boolean negation, like **E-Lt**.
* ``And`` / ``Or`` — short-circuiting; each is a conditional (**E-If** as a
  value)::

    @fp.fpy
    def and_(a: bool, b: bool) -> bool:
        return b if a else False

    @fp.fpy
    def or_(a: bool, b: bool) -> bool:
        return True if a else b

Comparisons
-----------

* ``Compare`` — a chained comparison is the conjunction of adjacent pairwise
  tests (each like **E-Lt**), every operand evaluated once. All six operators
  (``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``) yield exact booleans. E.g.
  ``a < b <= c``::

    @fp.fpy
    def chain(a: fp.Real, b: fp.Real, c: fp.Real) -> bool:
        return (a < b) and (b <= c)

Rounding operators
------------------

* ``Round`` — ``fp.round(e)`` rounds ``e`` to the active context, :math:`C(v)`
  (**E-Add** with no arithmetic); idempotent.
* ``RoundAt`` — ``fp.round_at(e, n)`` rounds ``e`` at digit position ``n``, then
  under :math:`C`.
* ``Cast`` — ``fp.cast(e)`` rounds ``e`` but is stuck unless the result is
  exact (a guarded **E-Assert**).

Tuples
------

A tuple holds its fields' values directly and is immutable—no node writes to one.
It copies nothing: a field holds exactly what its expression evaluated to, so a
tuple groups without owning. Tuples cannot be indexed, and are decomposed only
by a tuple pattern.

* ``TupleExpr`` — **E-Tuple**; ``TupleBinding`` — the tuple pattern of
  **M-Tuple**.
* ``Fst`` / ``Snd`` — pair accessors. Both require a tuple of exactly two
  elements; a longer one is an error, not a shorter tuple::

    @fp.fpy
    def fst(t: tuple[Any, Any]) -> Any:
        a, b = t
        return a

    @fp.fpy
    def snd(t: tuple[Any, Any]) -> Any:
        a, b = t
        return b

Lists
-----

Every FPy list is a list of *references* in the core semantics.
The list expression ``[e_1, ..., e_m]`` elaborates to

.. math::

   [\, \texttt{ref}\ e_1, \ldots, \texttt{ref}\ e_m \,]

whose value is a list of locations :math:`[\, \ell_1, \ldots, \ell_m \,]`—one
cell per element. Three consequences:

* **Elements are mutable, the length is not.**  **E-Update** replaces a cell's
  contents; only **E-List** builds a list value, and it fixes :math:`m`. So FPy
  has no ``append``.
* **Binding shares cells.**  **E-Assign** copies nothing, so ``ys = xs``—and a
  tuple field holding ``xs``—sees writes to its elements.
* **Ownership stops at the cell.**  Construction gives every element a fresh
  cell, but that cell holds the element's *value*: ``[xs]`` does not copy
  ``xs``'s cells, and ``xss[i]`` dereferences to a row shared with the original.

The nodes that build and read them:

* ``ListExpr`` — **E-List** over an **E-Ref** per element; ``ListRef``
  (``xs[i]``) — **E-Index** then **E-Deref**, since it only ever appears in value
  position. An assignment target is ``IndexedAssign``, not a ``ListRef``.
* ``ListSlice`` — ``xs[start:stop]`` extracts exactly ``stop - start``
  elements::

    @fp.fpy
    def slice(xs: list[Any], start: int, stop: int) -> list[Any]:
        return [xs[i] for i in range(start, stop)]

  Reading each element and rebuilding allocates a fresh cell per element, so a
  slice copies the spine rather than viewing it: for ``ys = xs[i:j]``, a write to
  ``ys[k]`` does not reach ``xs``. Those cells hold the same rows, though, so
  ``ys[k][l] = e`` does.

* ``ListComp`` — a list-building loop; a target may be a tuple binding
  (**M-Tuple**), and several generators nest as in Python. For an element
  expression ``g``, ``[g(x, y) for x, y in zip(xs, ys)]``::

    @fp.fpy
    def comp(xs: list[Any], ys: list[Any]) -> list[Any]:
        pairs = zip(xs, ys)
        acc = fp.empty(len(pairs))
        j = 0
        for x, y in pairs:
            acc[j] = g(x, y)
            j = j + 1
        return acc

* ``Zip`` — corresponding elements as tuples::

    @fp.fpy
    def zip(xs: list[Any], ys: list[Any]) -> list[tuple[Any, Any]]:
        assert len(xs) == len(ys)
        return [(xs[i], ys[i]) for i in range(len(xs))]

  Its elements are tuples, so each cell holds a tuple whose fields hold what
  ``xs[i]`` and ``ys[i]`` dereference to—a copy for a scalar, the shared row for
  a list. ``Enumerate`` has the same shape.

* ``Enumerate`` — ``(i, xs[i])`` pairs with integer ``i``::

    @fp.fpy
    def enumerate(xs: list[Any]) -> list[tuple[fp.Real, Any]]:
        return [(i, xs[i]) for i in range(len(xs))]

* ``Empty`` — ``fp.empty(d1, …, dn)`` allocates an uninitialized ``n``-d list.
* ``Len`` / ``Size`` / ``Dim`` — ``len(xs)``, ``fp.size(xs, k)``, ``fp.dim(xs)``:
  exact integer counts, no rounding.
* ``Range1`` / ``Range2`` / ``Range3`` — ``range(…)`` materialized to a list of
  integers, as in Python.

Miscellaneous
-------------

* ``IfExpr`` — ``a if c else b``, the expression form of the conditional (only
  the selected branch runs)::

    @fp.fpy
    def if_expr(c: bool, a: Any, b: Any) -> Any:
        if c:
            r = a
        else:
            r = b
        return r

* ``Attribute`` — ``e.name`` reads an attribute of a foreign value (no
  rounding).
* ``Call`` — **E-App**, generalized to many arguments and foreign callables; the
  body runs under the callee's declared context if any, else the caller's
  :math:`C`. An argument passes its structure of cells, and **E-App** does not
  capture the store, so a callee's write to an element is visible to its caller.

Statements
----------

* ``StmtBlock`` — a statement sequence, **E-Seq-Normal** / **E-Seq-Return**;
  empty is **E-Skip**.
* ``Assign`` — **E-Assign** (pattern via **M-Var** / **M-Tuple**).
* ``IndexedAssign`` — **E-Index** to the cell, then **E-Update** through it.
  Because the core's update takes a variable on the left, a temporary carries the
  projection: ``x[i] = e`` :math:`\equiv` ``y = x[i] ; y := e``.
* ``If1Stmt`` — ``if c: body`` is **E-If-True** / **E-If-False** with an
  **E-Skip** else-branch.
* ``IfStmt`` — **E-If-True** / **E-If-False**.
* ``WhileStmt`` — ``while c: s`` :math:`\equiv`
  ``if c then (s ; while c: s) else skip``.
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

* ``ContextStmt`` — **E-Context**.
* ``AssertStmt`` — **E-Assert** (the optional message is used only on failure).
* ``EffectStmt`` — evaluate an expression and discard the result (``_ = e``).
* ``ReturnStmt`` — **E-Ret**; ``PassStmt`` — **E-Skip**.
