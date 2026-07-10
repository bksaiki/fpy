Derived Semantics
=================

The :doc:`core semantics <semantics>` covers only a minimal fragment of
FPy—constants, arithmetic, function calls, and the basic statements.  This
page accounts for *every remaining node* in :mod:`fpy2.ast.fpyast`.  Each
one is explained in one of two ways:

* **(i) same as a core rule** — the node evaluates exactly like one of the
  rules in the core semantics (referenced by its tag, e.g. **E-Add**), or
* **(ii) a desugaring** — the node behaves like a small FPy program written
  in the core fragment, given in the core's notation (``:=``, ``;``,
  ``if … then … else …``, ``ret``, ``skip``, ``with … as … in …``).

The two rules the rest of this page leans on most are **E-Add**
(:math:`\langle \sigma, C, e_1 + e_2 \rangle \Downarrow C(\exact{n_1 + n_2})`—round
the exact result under the active context :math:`C`) and **E-Lt**
(:math:`\langle \sigma, C, e_1 < e_2 \rangle \Downarrow (n_1 < n_2)`—an exact
boolean, no rounding).  Nodes that carry no runtime behaviour (type
annotations, abstract base classes, re-exports) are listed at the end.

Literals and values
-------------------

* ``Decnum``, ``Hexnum``, ``Integer``, ``Rational``, ``Digits`` — numeric
  literals (surface forms ``0.1``, ``fp.hexfloat('0x1.8p+0')``, ``3``,
  ``fp.rational(1, 3)``, ``fp.digits(m, e, b)``).  Each evaluates to the
  *exact* real number it denotes, exactly like **E-Num**; no rounding
  occurs until the value is used in arithmetic, so ``0.1`` denotes exactly
  :math:`1/10`.  (``Decnum``/``Hexnum`` also preserve a negative-zero
  literal.)
* ``BoolVal`` — ``True`` / ``False``, like **E-True** / **E-False**.
* ``Var`` — a variable reference, **E-Var**.
* ``ForeignVal`` — a native Python value spliced into the program.  It
  evaluates to itself like **E-Num** but holds an opaque host value (used by
  ``Attribute`` and ``Call``); no rounding.

Constants
---------

* ``ConstPi``, ``ConstE``, ``ConstLog2E``, ``ConstLog10E``, ``ConstLn2``,
  ``ConstPi_2``, ``ConstPi_4``, ``Const1_Pi``, ``Const2_Pi``,
  ``Const2_SqrtPi``, ``ConstSqrt2``, ``ConstSqrt1_2`` — each is an irrational
  real constant *rounded under the active context*: the nullary analogue of
  **E-Add**, :math:`\langle \sigma, C, \pi \rangle \Downarrow C(\exact{\pi})`.
  Under :math:`\R` the exact value is returned.
* ``ConstNan``, ``ConstInf`` — the special IEEE 754 values NaN and
  :math:`+\infty` produced by the active context; like the constants above
  except the value is a special float rather than a real.

Arithmetic
----------

The following evaluate their operands to reals and round the exact
mathematical result under the active context :math:`C`—identical in shape to
**E-Add**, :math:`C(\exact{\ldots})`—differing only in the exact function
computed:

  ``Sub`` (``-``), ``Mul`` (``*``), ``Div`` (``/``), ``Neg`` (unary ``-``),
  ``Abs``, ``Sqrt``, ``Cbrt``, ``Pow`` (``**``), ``Copysign``, ``Atan2``,
  ``Mod`` (``%``), ``Fmod``, ``Remainder``, and every elementary function:
  ``Sin``, ``Cos``, ``Tan``, ``Asin``, ``Acos``, ``Atan``, ``Sinh``,
  ``Cosh``, ``Tanh``, ``Asinh``, ``Acosh``, ``Atanh``, ``Exp``, ``Exp2``,
  ``Expm1``, ``Log``, ``Log10``, ``Log1p``, ``Log2``, ``Erf``, ``Erfc``,
  ``Lgamma``, ``Tgamma``.

* ``Fma`` — fused multiply-add, ``a*b + c`` with a *single* rounding of the
  exact result: :math:`C(\exact{a \cdot b + c})`.
* ``Mod`` / ``Fmod`` / ``Remainder`` differ only in the exact value chosen
  before rounding: ``Mod`` follows Python (sign of the divisor), ``Fmod``
  follows C (sign of the dividend), ``Remainder`` is the IEEE 754 value
  nearest zero.

**Composite** operators compute their defining expression *exactly* and round
the result **once**.  The desugaring therefore evaluates the body under
:math:`\R` (so the intermediates do not round) and rounds only the final
result — a naive expression that rounded each intermediate would give a
different answer.

* ``Fdim`` — ``fp.fdim(x, y)`` has the semantics of::

    @fp.fpy
    def fdim(x: fp.Real, y: fp.Real) -> fp.Real:
        with fp.REAL:
            t = max(x - y, 0)
        return fp.round(t)

* ``Hypot`` — ``fp.hypot(x, y)`` has the semantics of::

    @fp.fpy
    def hypot(x: fp.Real, y: fp.Real) -> fp.Real:
        with fp.REAL:
            t = x * x + y * y
        return fp.sqrt(t)

**Selection** returns one operand exactly (no rounding, like **E-Lt** but
yielding a real).  ``Max`` / ``Min`` *propagate NaN* — any NaN operand makes
the result NaN — and break ``±0`` ties by sign (``max(-0, +0) = +0``,
``min(-0, +0) = -0``), independent of argument order.  The binary case has the
semantics of::

    @fp.fpy
    def maximum(x: fp.Real, y: fp.Real) -> fp.Real:
        if fp.isnan(x) or fp.isnan(y):
            return x if fp.isnan(x) else y   # any NaN operand propagates
        return x if x > y or (x == y and not fp.signbit(x)) else y  # tie: +0

    @fp.fpy
    def minimum(x: fp.Real, y: fp.Real) -> fp.Real:
        if fp.isnan(x) or fp.isnan(y):
            return x if fp.isnan(x) else y   # any NaN operand propagates
        return x if x < y or (x == y and fp.signbit(x)) else y      # tie: -0

The variadic ``max(x, y, …)`` / ``min(x, y, …)`` and the list-reduce forms
``AMax`` / ``AMin`` fold this binary operation left-to-right.

**Round-to-integer** (round the exact integer-valued result under :math:`C`,
like **E-Add**; they differ only in which integer is chosen):

* ``Ceil``, ``Floor``, ``Trunc``, ``RoundInt``, ``NearbyInt``.

Reductions
----------

* ``Sum`` — ``sum(xs)`` is a left fold with ``+``, so every addition rounds
  under :math:`C` (the empty sum is exact ``0``); it has the semantics of::

    @fp.fpy
    def sum(xs: list[fp.Real]) -> fp.Real:
        acc = 0
        for x in xs:
            acc = acc + x
        return acc

* ``AMax`` / ``AMin`` — the reduce form ``max(xs)`` / ``min(xs)`` over a
  single list; same select-without-rounding semantics as ``Max`` / ``Min``.

Classification and inspection
-----------------------------

* ``IsFinite``, ``IsInf``, ``IsNan``, ``IsNormal``, ``Signbit`` — test the
  operand and yield a boolean; exact, no rounding—like **E-Lt**.
* ``Logb`` — extracts the (integer) normalized exponent; an exact
  integer-valued result rounded under :math:`C`, like **E-Add**.

Logical operators
-----------------

* ``Not`` — boolean negation; like **E-Lt** (boolean, no rounding).
* ``And`` / ``Or`` — short-circuiting conjunction / disjunction; the right
  operand is evaluated only when needed, so each has the semantics of a
  conditional (**E-If** lifted to a value via ``IfExpr``)::

    @fp.fpy
    def and_(a: bool, b: bool) -> bool:
        return b if a else False

    @fp.fpy
    def or_(a: bool, b: bool) -> bool:
        return True if a else b

Comparisons
-----------

* ``Compare`` — a Python-style chained comparison is the conjunction of
  adjacent pairwise tests, each an **E-Lt**-style comparison, with every
  operand evaluated once.  All six operators (``<``, ``<=``, ``>``, ``>=``,
  ``==``, ``!=``) are exact booleans like **E-Lt**.  For example, ``a < b <= c``
  has the semantics of::

    @fp.fpy
    def chain(a: fp.Real, b: fp.Real, c: fp.Real) -> bool:
        return (a < b) and (b <= c)

Rounding operators
------------------

* ``Round`` — ``fp.round(e)`` rounds the value of ``e`` to the active
  context: literally :math:`C(v)`, i.e. **E-Add** with no arithmetic.
  Idempotent under the same context.
* ``RoundAt`` — ``fp.round_at(e, n)`` rounds ``e`` at the absolute digit
  position ``n`` (to a fixed exponent) and then under :math:`C`; ``Round``
  with an extra position argument.
* ``Cast`` — ``fp.cast(e)`` rounds ``e`` to the active context and *asserts
  the result is exact*: identical to ``Round`` when ``e`` is representable,
  otherwise stuck (like a failed **E-Assert**).

Compound data
-------------

Most operations here are *polymorphic* in the element type—they move values
around without inspecting them—so ``Any`` in the functions below stands for an
arbitrary element type, not just ``fp.Real``.

* ``TupleExpr`` — **E-Tuple**; ``ListExpr`` — **E-List**; ``ListRef`` —
  ``xs[i]``, **E-Ref**.
* ``TupleBinding`` — the tuple pattern :math:`p_1, \ldots, p_n` of
  **M-Tuple**; it *is* the core tuple pattern.
* ``Fst`` / ``Snd`` — tuple accessors that project a position via a tuple
  pattern (**M-Tuple**): ``fst(t)`` is the head; ``snd(t)`` is the second
  element of a pair, or the tuple of the remaining elements otherwise.  For a
  pair they have the semantics of::

    @fp.fpy
    def fst(t: tuple[Any, Any]) -> Any:
        a, b = t
        return a

    @fp.fpy
    def snd(t: tuple[Any, Any]) -> Any:
        a, b = t
        return b

* ``IfExpr`` — ``a if c else b`` is the expression form of the conditional; it
  evaluates ``c`` then only the selected branch, so it has the semantics of::

    @fp.fpy
    def if_expr(c: bool, a: Any, b: Any) -> Any:
        if c:
            r = a
        else:
            r = b
        return r

* ``ListSlice`` — ``xs[start:stop]`` extracts *exactly* ``stop - start``
  elements (defaults ``start = 0``, ``stop = len(xs)``; out-of-range bounds are
  stuck, not clamped).  It has the semantics of::

    @fp.fpy
    def slice(xs: list[Any], start: int, stop: int) -> list[Any]:
        return [xs[i] for i in range(start, stop)]

* ``ListComp`` — a comprehension is a list-building loop.  A target may be a
  tuple binding, unpacked via **M-Tuple** (several generators nest, as in
  Python — a cartesian product — rather than zipping).  For an element
  expression ``g``, ``[g(x, y) for x, y in zip(xs, ys)]`` has the semantics
  of::

    @fp.fpy
    def comp(xs: list[Any], ys: list[Any]) -> list[Any]:
        pairs = zip(xs, ys)
        acc = fp.empty(len(pairs))
        j = 0
        for x, y in pairs:
            acc[j] = g(x, y)
            j = j + 1
        return acc

* ``Zip`` — ``zip(xs, ys)`` is the list of tuples of corresponding elements::

    @fp.fpy
    def zip(xs: list[Any], ys: list[Any]) -> list[tuple[Any, Any]]:
        return [(xs[i], ys[i]) for i in range(len(xs))]

* ``Enumerate`` — ``enumerate(xs)`` is the list of ``(i, xs[i])`` pairs with
  integer ``i``::

    @fp.fpy
    def enumerate(xs: list[Any]) -> list[tuple[fp.Real, Any]]:
        return [(i, xs[i]) for i in range(len(xs))]
* ``Empty`` — ``fp.empty(d1, …, dn)`` allocates an ``n``-dimensional list of
  the given integer sizes (elements uninitialized); a library allocator, no
  rounding.
* ``Len`` (``len(xs)``), ``Size`` (``fp.size(xs, k)``), ``Dim``
  (``fp.dim(xs)``) — structural queries returning exact integers (element
  count, size along dimension ``k``, and number of dimensions); no rounding.
* ``Range1`` / ``Range2`` / ``Range3`` — ``range(stop)`` /
  ``range(start, stop)`` / ``range(start, stop, step)`` materialize the list
  of integers, as Python's ``range`` would.
* ``Attribute`` — ``e.name`` reads a native attribute of a foreign value; an
  opaque host operation, no rounding (companion to ``ForeignVal``).
* ``Call`` — general function application: **E-App** generalized to multiple
  positional / keyword arguments and to foreign (native) callables.  A
  pure-FPy call binds each argument and runs the body; the body runs under
  the callee's declared context if it has one (see ``FuncMeta`` below), else
  under the caller's :math:`C`.

Statements
----------

* ``StmtBlock`` — a sequence of statements, iterated **E-Seq**
  (``s1 ; s2 ; … ; sn``); an empty block is **E-Skip**.
* ``Assign`` — **E-Assign** (with **M-Var** / **M-Tuple** for the pattern).
* ``IndexedAssign`` — ``x[i] = e`` updates one element; it desugars to
  rebinding ``x`` to a copy with that position replaced,
  ``x := set(x, i, e)`` (**E-Assign**); the update is functional.
* ``If1Stmt`` — one-armed ``if c: body`` :math:`\equiv`
  ``if c then body else skip`` (**E-If** with an **E-Skip** else-branch).
* ``IfStmt`` — **E-If-True** / **E-If-False**.
* ``WhileStmt`` — ``while c: s`` :math:`\equiv`
  ``if c then (s ; while c: s) else skip``: a recursive unfolding into
  **E-If** and **E-Seq**.
* ``ForStmt`` — ``for x in xs: s`` is an index loop over a ``WhileStmt``
  (tuple targets destructure via **M-Tuple**; multiple iterables zip).  For a
  body ``s`` that accumulates, it has the semantics of::

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
* ``AssertStmt`` — **E-Assert**; the optional message is used only on
  failure, which is stuck.
* ``EffectStmt`` — evaluate an expression for its effect and discard the
  result, ``_ := e``; completes normally like **E-Assign** to an unused
  pattern.
* ``ReturnStmt`` — **E-Ret**; ``PassStmt`` — **E-Skip**.

Definitions and non-evaluable nodes
-----------------------------------

* ``FuncDef`` / ``Argument`` / ``FuncMeta`` — a function definition is the
  closure the initial environment binds for **E-App**; an ``Argument`` is a
  parameter (bound by **M-Var** on call).  ``FuncMeta`` carries metadata that
  is not itself evaluated—the foreign environment, spec, and properties—plus
  a *declared context override*: when present, the body runs under that
  context regardless of the caller's :math:`C` (otherwise **E-App**'s caller
  context applies).
* **Type annotations** (``AnyTypeAnn``, ``RealTypeAnn``, ``BoolTypeAnn``,
  ``ContextTypeAnn``, ``TupleTypeAnn``, ``ListTypeAnn``) have no runtime
  semantics—typing is out of scope (see :doc:`semantics`).
* **Abstract bases** (``Ast``, ``Expr``, ``Stmt``, ``TypeAnn``,
  ``ValueExpr``, ``RealVal``, ``RationalVal``, ``NaryExpr``, and the operator
  families ``NullaryOp``, ``UnaryOp`` / ``NamedUnaryOp``, ``BinaryOp`` /
  ``NamedBinaryOp``, ``TernaryOp`` / ``NamedTernaryOp``, and ``NaryOp`` /
  ``NamedNaryOp``) are structural only; their concrete subclasses carry the
  semantics above.
* **Re-exports** (``Id``, ``NamedId``, ``SourceId``, ``UnderscoreId``,
  ``CompareOp``, ``Location``, ``Context``, ``FuncSymbol``) and the
  **formatter** helpers (``BaseFormatter``, ``get_default_formatter``,
  ``set_default_formatter``) are not evaluable AST nodes.
