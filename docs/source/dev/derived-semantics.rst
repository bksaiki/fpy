Derived Semantics
=================

The :doc:`core semantics <semantics>` covers only a minimal fragment of
FPy—constants, arithmetic, function calls, and the basic statements.  This page
explains every other *evaluable* node in :mod:`fpy2.ast.fpyast`, each either
**(i)** evaluating like a core rule (referenced by tag, e.g. **E-Add**),
**(ii)** desugaring to a small FPy program, or **(iii)** elaborating to core
syntax that has no FPy spelling of its own, as lists do (see *Lists*).

The two rules leaned on most are **E-Add**
(:math:`\langle \sigma, \mu, C, e_1 + e_2 \rangle \Downarrow C(\exact{n_1 + n_2}) \,;\, \mu_2`—round
the exact result under the active context :math:`C`) and **E-Lt**
(:math:`\langle \sigma, \mu, C, e_1 < e_2 \rangle \Downarrow (n_1 < n_2) \,;\, \mu_2`—an exact
boolean, no rounding).  The store :math:`\mu` is elided in the entries below:
every node threads it, list construction allocates a cell per element, and
``IndexedAssign`` is the only node that overwrites one.  Type annotations,
abstract base classes, and re-exports carry no runtime behaviour and are omitted.

Literals and values
-------------------

* ``Decnum``, ``Hexnum``, ``Integer``, ``Rational``, ``Digits`` — numeric
  literals; each evaluates to the *exact* real it denotes, like **E-Num**.  No
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
one in the ``return``).  A single operation on exact rationals is the whole
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
and rounds only at the ``return``.  That inner value is irrational, so these are
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

**Selection** returns one operand exactly (no rounding).  ``Max`` / ``Min``
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
  sum is exact ``0``).  The accumulator is *seeded with the first element*, so
  a list of ``n`` elements performs ``n - 1`` additions::

    @fp.fpy
    def sum(xs: list[fp.Real]) -> fp.Real:
        if len(xs) == 0:
            return 0
        acc = xs[0]
        for x in xs[1:]:
            acc = acc + x
        return acc

  Seeding with ``0`` instead would *not* be equivalent: ``0 + xs[0]`` is itself
  a rounded operation, so it would round the first element before the fold
  begins.  Under ``fp.FP16``, summing the exact values
  ``[1 + 2**-11, 2**-11]`` gives ``1 + 2**-10`` as written above, but ``1.0``
  with a zero seed — ``2**-11`` is half of FP16's spacing above ``1.0``, so
  rounding the first element on its own is a tie that resolves down.

The *boolean* reductions fold a ``list[bool]`` with the logical operators, so
no rounding occurs and the active context is irrelevant.  Each is seeded with
its operator's identity, which is also the value of the empty case — unlike
``min``/``max``, both are total on the empty list.

* ``AnyOf`` — ``any(bs)`` is a left fold with ``or``, seeded ``False``
  (so ``any([])`` is ``False``)::

    @fp.fpy
    def any_(bs: list[bool]) -> bool:
        acc = False
        for b in bs:
            acc = acc or b
        return acc

* ``AllOf`` — ``all(bs)`` is a left fold with ``and``, seeded ``True``
  (so ``all([])`` is ``True``)::

    @fp.fpy
    def all_(bs: list[bool]) -> bool:
        acc = True
        for b in bs:
            acc = acc and b
        return acc

``AnyOf``/``AllOf`` are to ``Or``/``And`` what ``AMax``/``AMin`` are to
``Max``/``Min``: the list-fold form of a scalar operator, and a distinct node so
typing stays syntax-directed (``list[bool] -> bool`` rather than
``bool -> ... -> bool``).  The element type is exactly ``bool``: FPy has no
truthiness, so ``any([1.0, 0.0])`` is a type error rather than a zero test.

The fold is eager, and this **matches** Python for the syntax FPy supports.
Python's ``any``/``all`` short-circuit their *iterable*, but a list
comprehension is fully built before ``any`` is called, so
``any([pred(x) for x in xs])`` evaluates ``pred`` on every element in CPython
too; only a generator expression — which FPy does not have — makes the
short-circuit skip work.  Evaluating every element is therefore faithful, not a
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
  tests (each like **E-Lt**), every operand evaluated once.  All six operators
  (``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``) yield exact booleans.  E.g.
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

Lists
-----

Every FPy list is a core list of *references*.  The list expression
``[e_1, ..., e_m]`` elaborates to

.. math::

   [\, \texttt{ref}\ e_1, \ldots, \texttt{ref}\ e_m \,]

whose value is a list of locations :math:`[\, \ell_1, \ldots, \ell_m \,]`—one
cell per element.  Everything below follows from that choice and the core rules
that act on it.

**Elements are mutable; the length is not.**  **E-Update** replaces the contents
of a cell, and **E-List** is the only rule that builds a list value, fixing
:math:`m` where the list is constructed.  No core rule lengthens or shortens an
existing one, so FPy has no ``append`` and a list's length is fixed for the life
of the value.

**A projection is a cell, and its two positions differ.**  As a value, ``xs[i]``
is **E-Index** followed by **E-Deref**—:math:`\texttt{!}\, (xs[i])`—reading the
element out of its cell.  As the target of an assignment it is the cell itself,
with no dereference (see ``IndexedAssign``).  Nesting works the same way at each
level: a ``list[list[Any]]`` is cells holding list structures, so ``xss[i]``
dereferences to a row whose cells are the original's.

**Binding shares cells.**  **E-Assign** copies nothing, so ``ys = xs`` gives
``xs``'s cells a second name and a write to an element is visible through both.
Rebinding ``xs`` afterwards is not, since that changes only the environment.
Argument passing and ``return`` are the same: **E-App** does not capture the
store, so a callee's writes to an element are visible to its caller.

**Construction allocates, one level deep.**  Every element of a new list gets a
fresh cell, so ``[a, b]`` shares no cell with whatever ``a`` and ``b`` are bound
to—no copying rule is needed for that.  The fresh cell holds the element's
*value*, though, so where that value is itself a list the new cell holds the
same structure: ``[xs]`` does not copy ``xs``'s cells.  A list owns its own
cells and nothing deeper.

**Tuples hold values, not cells.**  A tuple groups without owning.  The fields
of ``(a, b)`` hold whatever ``a`` and ``b`` evaluate to, which for a list is its
structure of cells, so a list reached through a tuple is shared rather than
copied.

Compound data
-------------

These move values without inspecting them, so they are *polymorphic*: ``Any``
below is any element type, not just ``fp.Real``.

* ``TupleExpr`` — **E-Tuple**; ``TupleBinding`` — the tuple pattern of
  **M-Tuple**.
* ``ListExpr`` — **E-List** over an **E-Ref** per element; ``ListRef``
  (``xs[i]``) — **E-Index** in target position, **E-Index** then **E-Deref** as a
  value.  Both are the elaboration described under *Lists*.
* ``Fst`` / ``Snd`` — tuple accessors (``snd`` of a longer tuple is the rest)::

    @fp.fpy
    def fst(t: tuple[Any, Any]) -> Any:
        a, b = t
        return a

    @fp.fpy
    def snd(t: tuple[Any, Any]) -> Any:
        a, b = t
        return b

* ``IfExpr`` — ``a if c else b``, the expression form of the conditional (only
  the selected branch runs)::

    @fp.fpy
    def if_expr(c: bool, a: Any, b: Any) -> Any:
        if c:
            r = a
        else:
            r = b
        return r

* ``ListSlice`` — ``xs[start:stop]`` extracts exactly ``stop - start``
  elements::

    @fp.fpy
    def slice(xs: list[Any], start: int, stop: int) -> list[Any]:
        return [xs[i] for i in range(start, stop)]

  Reading each element and rebuilding allocates a fresh cell per element, so a
  slice copies the spine rather than viewing it: for ``ys = xs[i:j]``, a write to
  ``ys[k]`` does not reach ``xs``.  Those cells hold the same rows, though, so
  ``ys[k][l] = e`` does.

* ``ListComp`` — a list-building loop; a target may be a tuple binding
  (**M-Tuple**), and several generators nest as in Python.  For an element
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
  a list.  ``Enumerate`` has the same shape.

* ``Enumerate`` — ``(i, xs[i])`` pairs with integer ``i``::

    @fp.fpy
    def enumerate(xs: list[Any]) -> list[tuple[fp.Real, Any]]:
        return [(i, xs[i]) for i in range(len(xs))]

* ``Empty`` — ``fp.empty(d1, …, dn)`` allocates an uninitialized ``n``-d list.
* ``Len`` / ``Size`` / ``Dim`` — ``len(xs)``, ``fp.size(xs, k)``, ``fp.dim(xs)``:
  exact integer counts, no rounding.
* ``Range1`` / ``Range2`` / ``Range3`` — ``range(…)`` materialized to a list of
  integers, as in Python.
* ``Attribute`` — ``e.name`` reads an attribute of a foreign value (no
  rounding).
* ``Call`` — **E-App**, generalized to many arguments and foreign callables; the
  body runs under the callee's declared context if any, else the caller's
  :math:`C`.  An argument passes its structure of cells, and **E-App** does not
  capture the store, so a callee's write to an element is visible to its caller.

Statements
----------

* ``StmtBlock`` — a statement sequence, **E-Seq**; empty is **E-Skip**.
* ``Assign`` — **E-Assign** (pattern via **M-Var** / **M-Tuple**).
* ``IndexedAssign`` — ``x[i] = e`` writes through the cell at index ``i``.  The
  core's update takes a variable on the left, so it elaborates to a projection
  bound to a temporary and an **E-Update** on that: :math:`y = x[i]`,
  :math:`y := e`, where the projection is in target position and so carries no
  **E-Deref**.  Every other name for that cell observes the write.  It cannot
  change the list's length, and it cannot make the slot hold a *different*
  cell—only the contents of the cell already there.
* ``If1Stmt`` — ``if c: body`` is **E-If** with an **E-Skip** else-branch.
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
