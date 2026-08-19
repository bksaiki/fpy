"""
Scheduling language: fixed-point rescaling
"""

from ..function import Function
from ..transform import RescaleFixed
from ..transform.utils.cursor import Block, Cursor


def rescale_fixed(func: Function, where: int | Cursor | Block | None = None) -> Function:
    """
    Rescale fixed-point rounding in `func` to digit position zero.

    A fixed-point context with scale ``n`` represents the format
    ``A(inf, n, maxval)``.  Scaling by ``2**k`` shifts that format to
    ``A(inf, n + k, maxval * 2**k)``, and rounding commutes with the shift
    since a power of two is exact.  Each rounding under a fixed-point
    context therefore becomes: scale the operand up under ``fp.REAL``,
    round under the same format at position zero, then scale the result
    back down under ``fp.REAL``.

    Applies to every fixed-point context — :class:`fpy2.FixedContext` and
    :class:`fpy2.SMFixedContext`, which name their position ``scale``, and
    :class:`fpy2.MPFixedContext` and :class:`fpy2.MPBFixedContext`, which
    name it ``nmin``, one position below the scale.

    Only blocks whose body is entirely ``x = fp.round(v)`` / ``x = fp.cast(v)``
    (or a returned round) are rewritten, since arithmetic does not commute
    with the shift.  A format that substitutes a *finite* value for NaN or an
    infinity is declined — the substitute would have to shift along with the
    format.  Run :func:`fpy2.strategies.unfold_special` first, which takes
    those rules out of the context.

    Run :func:`fpy2.strategies.simplify` afterwards to fold the scale
    constants into the surrounding expressions.

    Parameters
    ----------
    func : Function
        The function to transform.
    where : int | Cursor | Block | None
        Which block to rescale: an index counting candidate blocks (the
        structurally-matching rounding blocks, whether or not they verify)
        in visit order, outermost-first; or a
        :class:`fpy2.strategies.Cursor` / :class:`fpy2.strategies.Block`
        naming a program point, which takes every candidate *at or
        beneath* it -- so the statement an earlier rewrite left behind
        names the rounding now nested inside it. A cursor or region from
        an earlier program is forwarded to this one first. If `None`,
        rescale every candidate that verifies and skip the rest.

    Returns
    -------
    Function
        The transformed function.

    Raises
    ------
    TransformDeclined
        If an explicit `where` names a candidate this rewrite refuses, or a
        region whose every candidate it refuses; the message says why.
    TransformReferenceError
        If an explicit `where` names no candidate block, or a cursor of a
        program this one was not derived from.

    Examples
    --------
    A quantizer that rounds into a fixed-point format::

        @fp.fpy(ctx=fp.REAL)
        def quantize(a):
            with fp.FixedContext(True, -16, 32):
                aq = fp.round(a)
            return aq

    ``rescale_fixed(quantize)`` moves the format to position zero, where its
    values are integers, and scales around the round::

        @fp.fpy(
            ctx=fp.REAL,
        )
        def quantize(a):
            with fp.FixedContext(True, 0, 32):
                with fp.REAL:
                    _t = (65536 * a)
                _t3 = fp.round(_t)
                with fp.REAL:
                    aq = (fp.rational(1, 65536) * _t3)
            return aq
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")

    return func.with_edits(RescaleFixed.apply_with_edits(
        func.ast, where=func.rebase(where)
    ))
