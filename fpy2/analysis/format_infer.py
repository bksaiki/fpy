"""
Format analysis for FPy programs.

This module implements a static analysis that bounds the number format
(set of representable values) for each real-valued expression and variable
in an FPy program.

The analysis uses a flat format lattice:

    specific_format < REAL_FORMAT (top)

where ``REAL_FORMAT`` represents unrestricted real values (the top element),
and any concrete ``Format`` instance represents values restricted to that
format's representable set.

**Lattice order**: ``f1 <= f2`` iff ``f1 == f2`` or ``f2 == REAL_FORMAT``.

**Join**: ``join(f1, f2) = f1 if f1 == f2 else REAL_FORMAT``.

For each expression, the format is derived from the rounding context under
which it is evaluated (obtained from ``ContextAnalysis``):

- If the context is a concrete ``Context`` object, the format is
  ``ctx.format()``.
- If the context is an unresolved context variable (``NamedId``), the format
  is conservatively ``REAL_FORMAT``.

For phi nodes (introduced at branch-merge and loop-back edges), the format
is the join of the two incoming formats:

- At a conditional (``if``/``if1``): ``join(then_format, else_format)``.
- At a loop back-edge (``while``/``for``): ``join(pre_loop_format,
  body_format)``.

Since ``REAL_FORMAT`` is the top element, the fixpoint for loops converges
in at most two iterations.
"""

from dataclasses import dataclass

from ..ast.fpyast import *
from ..number import Context
from ..number.context.format import Format
from ..number.context.real import REAL_FORMAT
from ..types import Type, RealType

from .context_infer import ContextInfer, ContextAnalysis
from .define_use import DefineUse, DefineUseAnalysis
from .reaching_defs import AssignDef, PhiDef, Definition

__all__ = [
    'FormatInfer',
    'FormatAnalysis',
]


#####################################################################
# Lattice operations

def _join_formats(f1: Format, f2: Format) -> Format:
    """
    Computes the join (least upper bound) of two formats.

    In the flat format lattice, two different formats join to
    ``REAL_FORMAT`` (the top element).
    """
    if f1 == f2:
        return f1
    return REAL_FORMAT


def _format_of_ctx_param(ctx) -> Format:
    """
    Extracts the number format from a context parameter.

    If the context is a concrete ``Context`` object, returns its
    ``format()``.  If the context is an unresolved variable (``NamedId``)
    or ``None``, returns ``REAL_FORMAT`` (conservative top element).
    """
    if isinstance(ctx, Context):
        return ctx.format()
    return REAL_FORMAT


def _format_of_type(ty: Type | None) -> Format:
    """Extracts the number format from an inferred type."""
    match ty:
        case RealType(ctx=ctx):
            return _format_of_ctx_param(ctx)
        case _:
            # Non-real types (bool, context, tuple, list) do not carry
            # a number format; use the top element as a conservative default.
            return REAL_FORMAT


#####################################################################
# Analysis result

@dataclass
class FormatAnalysis:
    """
    Result of format analysis for an FPy function.

    Maps each real-valued definition site and expression to its inferred
    number format.  ``REAL_FORMAT`` represents the top of the lattice
    (unbounded real values, e.g. an argument with an unknown format).
    """

    by_def: dict[Definition, Format]
    """
    Format inferred for each variable definition site.

    Keys are ``AssignDef`` or ``PhiDef`` objects from the
    definition-use analysis.  For phi nodes the format is the join
    of the formats from each incoming control-flow edge.
    """

    by_expr: dict[Expr, Format]
    """
    Format inferred for each expression node.

    An expression's format is the format of the rounding context under
    which it is evaluated (from ``ContextAnalysis.by_expr``).
    """

    ctx_info: ContextAnalysis
    """Underlying context analysis used to derive format information."""


#####################################################################
# Internal analysis instance

class _FormatInferInstance:
    """
    Internal helper that derives ``FormatAnalysis`` from a completed
    ``ContextAnalysis``.

    The derivation is straightforward: every real-valued type carries a
    context parameter (concrete ``Context`` or unresolved ``NamedId``),
    and the format is extracted from that parameter.

    Phi nodes are handled specially: instead of reading the unified context
    from ``ctx_info.by_def``, the format is computed by joining the formats
    of the two predecessor definitions, making the lattice join operation
    explicit.  This also means the format analysis remains meaningful even
    when two predecessors have the same format but different rounding modes
    (which ``ContextInfer`` treats as compatible via ``is_equiv``).
    """

    func: FuncDef
    ctx_info: ContextAnalysis

    def __init__(self, func: FuncDef, ctx_info: ContextAnalysis):
        self.func = func
        self.ctx_info = ctx_info

    @property
    def def_use(self) -> DefineUseAnalysis:
        return self.ctx_info.def_use

    def _format_of_def(self, d: Definition) -> Format:
        """Read the format for a definition from ``ctx_info``."""
        ty = self.ctx_info.by_def.get(d)
        return _format_of_type(ty)

    def _phi_format(self, phi: PhiDef) -> Format:
        """
        Compute the format of a phi node by joining its two predecessors.

        The join captures the semantics of the format lattice at
        control-flow merge points:

        - **Conditional branches** (``if``/``if1``): both the then-branch
          definition and the else-branch definition flow into the phi. The
          format of the merge point is ``join(then_fmt, else_fmt)``.

        - **Loop back-edges** (``while``/``for``): the lhs predecessor is
          the pre-loop definition and the rhs predecessor is the definition
          produced by the loop body. The format is ``join(pre_fmt, body_fmt)``.

        Since ``REAL_FORMAT`` is the top element, if either predecessor is
        ``REAL_FORMAT`` the phi is also ``REAL_FORMAT``.  Convergence for
        loops is therefore guaranteed in at most two iterations.
        """
        lhs_def = self.def_use.defs[phi.lhs]
        rhs_def = self.def_use.defs[phi.rhs]
        lhs_fmt = self._format_of_def(lhs_def)
        rhs_fmt = self._format_of_def(rhs_def)
        return _join_formats(lhs_fmt, rhs_fmt)

    def analyze(self) -> FormatAnalysis:
        by_def: dict[Definition, Format] = {}
        by_expr: dict[Expr, Format] = {}

        # Derive the format for every definition site.
        for d in self.ctx_info.by_def:
            match d:
                case PhiDef():
                    # Phi nodes: use the explicit join of the two predecessors
                    # to make the lattice semantics transparent.
                    by_def[d] = self._phi_format(d)
                case _:
                    # Non-phi definitions: read directly from the resolved type.
                    by_def[d] = self._format_of_def(d)

        # Derive the format for every expression node.
        for e, ty in self.ctx_info.by_expr.items():
            by_expr[e] = _format_of_type(ty)

        return FormatAnalysis(by_def, by_expr, self.ctx_info)


#####################################################################
# Public API

class FormatInfer:
    """
    Format inference for FPy functions.

    This analysis bounds the number format for each real-valued expression
    and variable definition in an FPy program.

    **Format lattice**::

        specific_format  <  REAL_FORMAT   (top)

    The top element ``REAL_FORMAT`` represents unrestricted real values.
    Any concrete ``Format`` object represents values that are guaranteed
    to be representable under that format after rounding.

    **Join rule**::

        join(f, f)       = f
        join(f1, f2)     = REAL_FORMAT   (when f1 != f2)

    This is used at branch-merge points and loop back-edges (phi nodes).

    **Usage**::

        from fpy2.analysis import FormatInfer

        info = FormatInfer.analyze(func)
        for d, fmt in info.by_def.items():
            print(d, '->', fmt)
    """

    @staticmethod
    def analyze(
        func: FuncDef,
        *,
        ctx_info: ContextAnalysis | None = None,
        **kwargs
    ) -> FormatAnalysis:
        """
        Performs format analysis on a compiled FPy function.

        The analysis requires a completed ``ContextAnalysis`` (either
        supplied via *ctx_info* or computed internally).

        Args:
            func:     The ``FuncDef`` AST node to analyze.
            ctx_info: A pre-computed ``ContextAnalysis`` result.  When
                      ``None`` (the default), context inference is run
                      automatically.
            **kwargs: Additional keyword arguments forwarded verbatim to
                      ``ContextInfer.infer`` when *ctx_info* is ``None``.

        Returns:
            A ``FormatAnalysis`` instance whose ``by_def`` and ``by_expr``
            maps contain the inferred formats for every definition site and
            expression node in *func*.

        Raises:
            TypeError:           If *func* is not a ``FuncDef``.
            ContextInferError:   If context inference fails (only when
                                 *ctx_info* is ``None``).
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f"expected a 'FuncDef', got {type(func)}")

        if ctx_info is None:
            ctx_info = ContextInfer.infer(func, **kwargs)

        return _FormatInferInstance(func, ctx_info).analyze()
