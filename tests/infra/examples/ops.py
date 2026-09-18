"""
Example functions for each operation.
"""

import fpy2 as fp

_ASSERT = fp.OverflowMode.ASSERT

_INT_RTP = fp.SINT32.with_params(rm=fp.RM.RTP, overflow=_ASSERT)
_INT_RTN = fp.SINT32.with_params(rm=fp.RM.RTN, overflow=_ASSERT)
_INT_RNA = fp.SINT32.with_params(rm=fp.RM.RNA, overflow=_ASSERT)
_INT_RNE = fp.SINT32.with_params(rm=fp.RM.RNE, overflow=_ASSERT)

# Each is a mode the cast does not perform: C++ integer conversion truncates, so
# the value is made integral in the float type first.  Each guards its operand, a
# NaN and an infinity having no integer to convert to.  The four differ only on
# which value they land on, which is what a bit-exact run checks.

@fp.fpy
def test_round_int_up(x: fp.Real) -> fp.Real:
    """`round` toward positive infinity into integer storage."""
    if fp.isnan(x) or fp.isinf(x):
        return 0
    else:
        with _INT_RTP:
            return fp.round(x)

@fp.fpy
def test_round_int_down(x: fp.Real) -> fp.Real:
    """`round` toward negative infinity into integer storage."""
    if fp.isnan(x) or fp.isinf(x):
        return 0
    else:
        with _INT_RTN:
            return fp.round(x)

@fp.fpy
def test_round_int_nearest_away(x: fp.Real) -> fp.Real:
    """`round` to nearest, ties away from zero, into integer storage."""
    if fp.isnan(x) or fp.isinf(x):
        return 0
    else:
        with _INT_RNA:
            return fp.round(x)

@fp.fpy
def test_round_int_nearest_even(x: fp.Real) -> fp.Real:
    """`round` to nearest, ties to even, into integer storage.

    ``std::nearbyint`` follows the live ``fenv`` mode, so this also pins that
    the kernel is entered under ``FE_TONEAREST``.
    """
    if fp.isnan(x) or fp.isinf(x):
        return 0
    else:
        with _INT_RNE:
            return fp.round(x)

_INT_RTP_WRAP = fp.SINT32.with_params(rm=fp.RM.RTP)

@fp.fpy
def test_round_int_up_wrapping(x: fp.Real) -> fp.Real:
    """`round` toward positive infinity into integer storage that wraps.

    ``int32_t`` holds exactly what the format does, so the type's own wrapping
    is the context's -- reduced from the *rounded* value, which under this mode
    can sit a step outside the range its operand was inside.
    """
    if fp.isnan(x) or fp.isinf(x):
        return 0
    else:
        with _INT_RTP_WRAP:
            return fp.round(x)

@fp.fpy
def test_logb(x: fp.Real) -> fp.Real:
    """Example function for `logb`."""
    return fp.logb(x)

@fp.fpy
def test_logb_guarded(x: fp.Real) -> fp.Real:
    """`logb` of a finite non-zero value, which is a small integer.

    The guards rule out the NaN and the ``+inf``, ``x == 0`` rules out the
    ``-inf`` that ``logb(0)`` gives, and the ``REAL`` block keeps the exact
    integer format -- under a concrete context the class is the one the context
    can represent, which is every class.
    """
    if fp.isnan(x) or fp.isinf(x) or x == 0:
        return 0
    else:
        with fp.REAL:
            return fp.logb(x)

@fp.fpy
def test_logb_guarded_list(xs: list[fp.Real]) -> fp.Real:
    """The same, reached through a guard over a whole list.

    ``all`` covers the list, FPy having no ``break``, so what it tests holds of
    every element -- the only way a fact reaches the elements of a parameter.
    """
    if all([fp.isfinite(x) and x != 0 for x in xs]):
        with fp.REAL:
            return max([fp.logb(x) for x in xs])
    else:
        return 0

@fp.fpy
def test_logb_guarded_list_stale(xs: list[fp.Real]) -> fp.Real:
    """The same guard, and a store that makes it say nothing.

    The fact is about the contents at the loop's *exit*, and the read here is
    of a list the loop never saw.
    """
    ok = all([fp.isfinite(x) and x != 0 for x in xs])
    xs[0] = fp.nan()
    if ok:
        with fp.REAL:
            return fp.logb(xs[0])
    else:
        return 0

@fp.fpy
def test_logb_guarded_pair(x: fp.Real) -> tuple[fp.Real, fp.Real]:
    """A tuple return narrows field by field.

    The exponent is a small integer where the value beside it is not, and one
    class for the whole tuple is the top class -- so the field would widen back
    in the function's own ABI.
    """
    if fp.isnan(x) or fp.isinf(x) or x == 0:
        return x, 0
    else:
        with fp.REAL:
            return x, fp.logb(x)

@fp.fpy
def test_specials_add(a: fp.Real, b: fp.Real) -> fp.Real:
    """Operands that exclude a NaN but span both infinities.

    ``inf + -inf`` is a NaN, so the sum keeps one although neither operand can
    be one -- the shape that tells the addition table apart from a join.
    """
    if fp.isnan(a) or fp.isnan(b):
        return 0
    else:
        with fp.REAL:
            return a + b

@fp.fpy
def test_specials_sub(a: fp.Real, b: fp.Real) -> fp.Real:
    """The same for subtraction, which negates its right operand before the
    table applies -- ``inf - inf`` is a NaN where ``inf + inf`` is not."""
    if fp.isnan(a) or fp.isnan(b):
        return 0
    else:
        with fp.REAL:
            return a - b

@fp.fpy
def test_specials_mul(a: fp.Real, b: fp.Real) -> fp.Real:
    """``0 * inf`` is a NaN, from operands neither of which can be one."""
    if fp.isnan(a) or fp.isnan(b):
        return 0
    else:
        with fp.REAL:
            return a * b

@fp.fpy
def test_cancelling_add(a: fp.Real, b: fp.Real) -> fp.Real:
    """Two finite non-zeros summing to zero, so the class keeps a zero its
    operands rule out."""
    if fp.isnan(a) or fp.isinf(a) or a == 0:
        return 0
    elif fp.isnan(b) or fp.isinf(b) or b == 0:
        return 0
    else:
        with fp.REAL:
            return a + b

@fp.fpy
def test_unrelated_named_guard(a: fp.Real, b: fp.Real) -> fp.Real:
    """``if p: t = True`` has a lowered ``and``'s shape and says nothing.

    Reaching the second guard with ``t`` true means ``p`` held or the first
    test did, and ``p`` tests something else -- so ``a`` may still be a NaN.
    The guard has to be a *name* for the match to get this far.
    """
    t = not fp.isnan(a)
    p = b > 0
    if p:
        t = True
    if t:
        with fp.REAL:
            return fp.fabs(a)
    else:
        return 0

@fp.fpy
def test_not_a_fold(xs: list[fp.Real]) -> fp.Real:
    """An ``and`` that does not carry the accumulator is not a fold.

    ``ok`` is the *last* element's predicate, so it says nothing about the
    others and the reduction over them keeps every class.
    """
    ok = True
    for x in xs:
        p = fp.isfinite(x)
        q = x != 0
        ok = p and q
    if ok:
        with fp.REAL:
            return max(xs)
    else:
        return 0

@fp.fpy
def _fill_first(ys: list[fp.Real]) -> fp.Real:
    """A callee that stores through the list it is handed."""
    ys[0] = fp.nan()
    return 0

@fp.fpy
def test_elements_after_a_call(xs: list[fp.Real]) -> fp.Real:
    """What the caller stored does not survive the call.

    The callee may store through the same list, so every element fact has to be
    dropped at the call -- the one shape that exercises that.
    """
    xs[0] = 1.0
    y = _fill_first(xs)
    return xs[0] + y

@fp.fpy
def test_finite_product(a: fp.Real, b: fp.Real) -> fp.Real:
    """A product of two finite non-zeros, under whatever context the caller
    brings.

    Exactly it is finite; *rounded* it can be an infinity, so the class holds
    only where the context is known.  The one shape that tells an exact claim
    apart from a rounded one at run time.
    """
    if fp.isnan(a) or fp.isinf(a) or a == 0:
        return 0
    elif fp.isnan(b) or fp.isinf(b) or b == 0:
        return 0
    else:
        return a * b

@fp.fpy
def test_pow(x: fp.Real, y: fp.Real) -> fp.Real:
    """Example function for `pow`."""
    return x ** y

@fp.fpy
def test_empty1(n: fp.Real):
    """Example function for `empty`."""
    arr = fp.empty(n)
    for i in range(n):
        arr[i] = i
    return arr

@fp.fpy
def test_empty2(m: fp.Real, n: fp.Real):
    """Example function for `empty`."""
    arr = fp.empty(m, n)
    for i in range(m):
        for j in range(n):
            arr[i][j] = i * n + j
    return arr

@fp.fpy
def test_empty3(k: fp.Real, m: fp.Real, n: fp.Real):
    """Example function for `empty`."""
    arr = fp.empty(k, m, n)
    for i in range(k):
        for j in range(m):
            for l in range(n):
                arr[i][j][l] = (i * m + j) * n + l
    return arr
