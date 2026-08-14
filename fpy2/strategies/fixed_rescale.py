"""
Scheduling language: fixed-point rescaling
"""

from ..function import Function
from ..transform import RescaleFixed


def rescale_fixed(
    func: Function, where: int | None = None, *, fold_specials: bool = False
) -> Function:
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
    with the shift.

    Run :func:`fpy2.strategies.simplify` afterwards to fold the scale
    constants into the surrounding expressions.

    Parameters
    ----------
    func : Function
        The function to transform.
    where : int | None
        The index of the block to rescale, counting candidate blocks (those
        this rewrite could rescale) in visit order, outermost-first. If
        `None`, rescale every candidate.
    fold_specials : bool
        If `True`, precede each rounding with branches assigning what the
        format makes of NaN and the infinities, for whichever of them it
        defines. Fixed-point formats commonly leave them undefined, in which
        case nothing is folded and the rounding rejects them as before. This
        is what lets a format that substitutes a finite value for them be
        rescaled at all, since such a substitute would otherwise have to
        shift along with the format. Defaults to `False`.

    Returns
    -------
    Function
        The transformed function.

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

    ast = RescaleFixed.apply(func.ast, where=where, fold_specials=fold_specials)
    return func.with_ast(ast)
