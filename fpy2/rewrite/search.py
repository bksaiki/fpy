"""
Naming a location by what it looks like: a pattern match as a cursor.
"""

from ..function import Function
from ..transform import Cursor, TransformReferenceError, contains
from .matcher import Matcher
from .pattern import Pattern


def find_all(
    pattern: Pattern,
    func: Function,
    within: Cursor | None = None,
) -> list[Cursor]:
    """
    Every place in `func` that matches `pattern`, in visit order.

    The kind of cursor is the kind of the pattern: an
    :class:`fpy2.strategies.ExprCursor` for an expression pattern, a
    :class:`fpy2.strategies.StmtCursor` or
    :class:`fpy2.strategies.BlockCursor` for a statement pattern of one or
    several statements.

    A statement pattern of *k* statements is matched by a sliding window, so
    two matches can share statements.  They are all listed; a rewrite cannot
    apply to more than one of an overlapping pair.

    Parameters
    ----------
    pattern : Pattern
        The pattern to match, from :func:`fpy2.pattern`.
    func : Function
        The function to search.
    within : Cursor | None
        Keep only the matches at or beneath this cursor or region; one from an
        earlier program is forwarded first. If `None`, search the whole program.

    Returns
    -------
    list[Cursor]
        The matches, in visit order, outermost-first; `[]` where the pattern
        matches nothing.

    Raises
    ------
    TransformReferenceError
        If `within` names part of another program, or names an expression while
        the pattern matches statements.
    """
    if not isinstance(pattern, Pattern):
        raise TypeError(f'Expected a \'Pattern\', got {pattern}')
    if not isinstance(func, Function):
        raise TypeError(f'Expected a \'Function\', got {func}')

    matches = Matcher(pattern).match(func)
    if within is None:
        return [m.cursor for m in matches]

    here = func.rebase(within)
    return [m.cursor for m in matches if contains(here, m.cursor)]


def find(
    pattern: Pattern,
    func: Function,
    within: Cursor | None = None,
) -> Cursor:
    """
    The one place in `func` that matches `pattern`.

    A pattern is meant to identify something, so matching nothing and matching
    several places are both bad references — use :func:`find_all` where a
    pattern is expected to match more than once.

    Parameters
    ----------
    pattern : Pattern
        The pattern to match, from :func:`fpy2.pattern`.
    func : Function
        The function to search.
    within : Cursor | None
        Keep only the matches at or beneath this cursor or region; one from an
        earlier program is forwarded first. If `None`, search the whole program.

    Returns
    -------
    Cursor
        The match, of the kind the pattern matches.

    Raises
    ------
    TransformReferenceError
        If the pattern matches nothing, or matches more than one place.

    Examples
    --------
    ::

        @fp.pattern
        def fma_l(a, b, c):
            a * b + c

        @fp.fpy
        def f(x, y, z):
            return x * y + z

    ``find(fma_l, f)`` names the sum, which ``inline`` or a
    :class:`fpy2.strategies.Rewrite` can then be aimed at.
    """
    found = find_all(pattern, func, within)
    if not found:
        raise TransformReferenceError(
            f'pattern `{pattern.name}` matches nothing in `{func.name}`'
        )
    if len(found) > 1:
        where = ', '.join(str(c) for c in found[:3])
        more = ', ...' if len(found) > 3 else ''
        raise TransformReferenceError(
            f'pattern `{pattern.name}` matches {len(found)} places in '
            f'`{func.name}` ({where}{more}); use `find_all` or refine the pattern'
        )
    return found[0]
