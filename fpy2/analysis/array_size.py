"""
Array size inference.

Tracks the static size of list-valued expressions and propagates
sizes through control-flow merges.  Mirrors the structure of
:mod:`fpy2.analysis.format_infer`: the lattice value attached to each
expression / definition is a structural type that mirrors the basic-type
shape, and joins are performed at phi nodes.

Indexed assignment (``xs[i] = e``) is handled via the SSA-fresh-def
treatment that ``reaching_defs`` provides — semantically a functional
update ``xs = update(xs, [i], e)``.  ``_visit_indexed_assign`` reads
the pre-mutation bound from the use def, widens the element bound at
depth ``len(indices)``, and writes the result to the fresh def.
"""

from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from typing import TypeAlias

from ..ast.fpyast import *
from ..ast.visitor import DefaultVisitor
from ..function import Function
from ..number import Float, INTEGER, REAL
from ..types import ListType, TupleType, Type
from ..utils import Gensym, NamedId, Unionfind
from .context_use import ContextUse, ContextUseAnalysis
from .define_use import Definition, DefSite, DefineUseAnalysis
from .partial_eval import PartialEval, PartialEvalInfo, Value
from .type_infer import TypeInfer, TypeAnalysis

__all__ = [
    'ArraySize',
    'ArraySizeAnalysis',
    'ArraySizeBound',
    'ArraySizeInfer',
    'ListSize',
    'SizeUnionfind',
    'TupleSize',
    'concrete_size',
    'is_size_eq',
]


#####################################################################
# Array-size lattice

ArraySize: TypeAlias = 'int | NamedId | None'
"""
Static size of a list-valued expression:

- a concrete ``int`` when the size is a compile-time constant;
- a :class:`NamedId` *size variable* when the length is unknown but
  tracked, so equalities like ``len(ys) == len(xs)`` survive.  Two size
  variables in the same :class:`SizeUnionfind` class denote the same
  runtime length; a class whose representative is an ``int`` is pinned to
  that constant.
- ``None`` — top; no information.

``None`` is top; equal sizes (same ``int``, or co-representative size
variables) join to themselves; everything else joins to ``None``.  It is
*not* a set of possible sizes.  Size variables are minted per analysis
run and are **not** meaningful across functions.
"""

SizeUnionfind: TypeAlias = 'Unionfind[NamedId | int]'
"""Equivalence classes of size variables (and the concrete ``int``s they
pin to) for one analysis run.  By convention a concrete ``int`` is always
the representative of its class, so :func:`concrete_size` is just a
``find`` that lands on an ``int``.  Use :func:`concrete_size` /
:func:`is_size_eq` rather than poking the union-find directly."""


def _repr_size(size: ArraySize, uf: SizeUnionfind) -> int | NamedId | None:
    """The representative of *size*'s class: an ``int`` if the class is
    pinned to a constant, else the size variable itself (``None`` stays
    ``None``)."""
    if isinstance(size, NamedId):
        return uf.get(size, size)
    return size


def concrete_size(size: ArraySize, uf: SizeUnionfind) -> int | None:
    """The concrete ``int`` *size* denotes — directly, or via a pinned
    size-variable class — or ``None`` if not a compile-time constant."""
    rep = _repr_size(size, uf)
    return rep if isinstance(rep, int) else None


def _size_eq(a: ArraySize, b: ArraySize, uf: SizeUnionfind) -> bool:
    """Are two sizes provably equal: same representative (concrete int or
    co-representative size variable)?"""
    ra = _repr_size(a, uf)
    return ra is not None and ra == _repr_size(b, uf)


def is_size_eq(b1: ArraySizeBound, b2: ArraySizeBound, uf: SizeUnionfind) -> bool:
    """Structurally compare two bounds, treating equal-representative
    sizes at each level as equal."""
    match b1, b2:
        case ListSize(), ListSize():
            return _size_eq(b1.size, b2.size, uf) and is_size_eq(b1.elt, b2.elt, uf)
        case TupleSize(), TupleSize():
            return (len(b1.elts) == len(b2.elts)
                    and all(is_size_eq(a, b, uf) for a, b in zip(b1.elts, b2.elts)))
        case None, None:
            return True
        case _:
            return False


@dataclass(frozen=True)
class ListSize:
    """Array-size info for a list-valued expression."""
    elt: 'ArraySizeBound'
    size: ArraySize


@dataclass(frozen=True)
class TupleSize:
    """Array-size info for a tuple-valued expression."""
    elts: tuple['ArraySizeBound', ...]


ArraySizeBound: TypeAlias = 'None | ListSize | TupleSize'
"""
Inferred array-size info for an expression or variable definition.

- ``None`` — scalar / non-list / non-tuple value (no array-size info).
- :class:`ListSize` — list-valued expression with an element bound and
  an :class:`ArraySize`.
- :class:`TupleSize` — heterogeneous tuple, per-element bounds preserved.
"""


#####################################################################
# Analysis result

@dataclass
class ArraySizeAnalysis:
    """
    Result of array-size analysis for an FPy function.

    Maps each variable definition site and expression to its inferred
    :data:`ArraySizeBound`.  ``ret_size`` is the inferred bound of the
    function's return value (if list-shaped).
    """
    by_expr: dict[Expr, ArraySizeBound]
    by_def: dict[Definition, ArraySizeBound]
    ret_size: ArraySizeBound
    def_use: DefineUseAnalysis
    size_uf: SizeUnionfind
    """Equivalence classes for the size variables appearing in the bounds
    above.  Use :func:`concrete_size` / :func:`is_size_eq` (which take
    this) rather than comparing sizes directly."""


#####################################################################
# Internal analysis visitor

class _ArraySizeInferInstance(DefaultVisitor):
    """Single-use visitor that performs array-size inference."""

    func: FuncDef
    partial_eval: PartialEvalInfo
    type_info: TypeAnalysis

    by_expr: dict[Expr, ArraySizeBound]
    by_def: dict[Definition, ArraySizeBound]
    ret_size: ArraySizeBound
    uf: SizeUnionfind
    gensym: Gensym
    _uf_changes: int
    _callee_ret: dict[FuncDef, ArraySizeBound]
    _ctx_use_cache: ContextUseAnalysis | None

    def __init__(self, func: FuncDef, partial_eval: PartialEvalInfo, type_info: TypeAnalysis):
        self.func = func
        self.partial_eval = partial_eval
        self.type_info = type_info
        self.by_expr = {}
        self.by_def = {}
        self.ret_size = None
        self.uf = Unionfind()
        self.gensym = Gensym()
        # Bumped whenever a union actually collapses two classes — lets the
        # loop fixpoint detect union-find progress even when no stored bound
        # changes (e.g. a ``zip`` inside the body merges two size vars).
        self._uf_changes = 0
        self._callee_ret = {}
        self._ctx_use_cache = None

    def _fresh_size(self) -> NamedId:
        """Mint a fresh size variable (only ever for arguments / free
        vars, which are bound once before any fixpoint — so no AST-keyed
        memoization is needed)."""
        s = self.gensym.fresh('n')
        self.uf.add(s)
        return s

    def _pin_size(self, sym: NamedId, n: int) -> None:
        """Constrain *sym*'s class to the concrete length *n*, making the
        ``int`` the class representative."""
        self.uf.add(n)
        if self.uf.find(sym) != self.uf.find(n):
            self.uf.union(n, sym)   # ``int`` first => it leads
            self._uf_changes += 1

    def _merge_sizes(self, a: NamedId, b: NamedId) -> None:
        """Prove two size variables equal (e.g. via strict ``zip``),
        keeping a concrete ``int`` as the representative if either class
        already has one."""
        ra, rb = self.uf.get(a, a), self.uf.get(b, b)
        if ra == rb:
            return
        if isinstance(rb, int):
            ra, rb = rb, ra          # keep the concrete as leader
        self.uf.union(ra, rb)
        self._uf_changes += 1

    @property
    def def_use(self) -> DefineUseAnalysis:
        return self.partial_eval.def_use

    @property
    def _ctx_use(self) -> ContextUseAnalysis:
        # Computed lazily — only slices with symbolic-offset bounds need
        # the active rounding context, so functions without them never
        # pay for it.  Reuse our partial-eval info so it isn't recomputed.
        if self._ctx_use_cache is None:
            self._ctx_use_cache = ContextUse.analyze(
                self.func, partial_eval=self.partial_eval
            )
        return self._ctx_use_cache

    def analyze(self) -> ArraySizeAnalysis:
        self._visit_function(self.func, None)
        return ArraySizeAnalysis(
            self.by_expr, self.by_def, self.ret_size,
            self.partial_eval.def_use, self.uf,
        )

    def _cvt_type(self, ty: Type) -> ArraySizeBound:
        match ty:
            case ListType():
                elt = self._cvt_type(ty.elt)
                return ListSize(elt, None)
            case TupleType():
                elts = tuple(self._cvt_type(e) for e in ty.elts)
                return TupleSize(elts)
            case _:
                return None

    def _arg_bound(self, ty: Type) -> ArraySizeBound:
        """Like :meth:`_cvt_type`, but a list parameter's *outer* length
        gets a fresh size variable (its length is fixed per call, just
        unknown), so equalities such as ``ys = xs`` are tracked.  Inner
        dimensions stay ``None`` (one size variable per argument)."""
        match ty:
            case ListType():
                return ListSize(self._cvt_type(ty.elt), self._fresh_size())
            case TupleType():
                return TupleSize(tuple(self._arg_bound(e) for e in ty.elts))
            case _:
                return None

    def _get_eval(self, e: Expr) -> Value | None:
        if e in self.partial_eval.by_expr:
            return self.partial_eval.by_expr[e]
        else:
            return None

    def _join_size(self, a: ArraySize, b: ArraySize) -> ArraySize:
        """Lattice join: equal sizes (same int, or co-representative size
        variables) survive as their representative — concrete when the
        class is pinned; everything else goes to ``None``."""
        ra = _repr_size(a, self.uf)
        return ra if ra is not None and ra == _repr_size(b, self.uf) else None

    def _unify(self, t1: ArraySizeBound, t2: ArraySizeBound) -> ArraySizeBound:
        match t1, t2:
            case ListSize(), ListSize():
                elt = self._unify(t1.elt, t2.elt)
                size = self._join_size(t1.size, t2.size)
                return ListSize(elt, size)
            case TupleSize(), TupleSize():
                elts = tuple(self._unify(e1, e2) for e1, e2 in zip(t1.elts, t2.elts, strict=True))
                return TupleSize(elts)
            case None, None:
                return None
            case _:
                raise TypeError(f'Cannot unify types: {t1} and {t2}')

    def _visit_binding(self, site: DefSite, target: Id | TupleBinding, ty: ArraySizeBound):
        match target:
            case NamedId():
                d = self.def_use.find_def_from_site(target, site)
                self.by_def[d] = ty
            case TupleBinding():
                assert isinstance(ty, TupleSize), f'Expected tuple type for tuple binding, got {ty} for {target}'
                for elt, s in zip(target.elts, ty.elts, strict=True):
                    self._visit_binding(site, elt, s)
            case _:
                pass

    def _visit_var(self, e: Var, ctx: None):
        d = self.def_use.find_def_from_use(e)
        return self.by_def[d]

    def _visit_unaryop(self, e: UnaryOp, ctx: None):
        ty = self._visit_expr(e.arg, ctx)
        match e:
            case Range1():
                # ``range(stop)`` -> ``max(0, stop)`` when ``stop`` is a
                # known integer (clamped: ``stop <= 0`` is empty, like
                # Python).  ``_const_int`` also resolves ``len(xs)`` of a
                # known-size list, so ``range(len(xs))`` is covered.
                n = self._const_int(e.arg)
                if n is not None:
                    return ListSize(None, max(0, n))
                return ListSize(None, None)
            case Enumerate():
                assert isinstance(ty, ListSize)
                return ListSize(TupleSize((None, ty.elt)), ty.size)
            case _:
                return None

    def _visit_binaryop(self, e: BinaryOp, ctx: None):
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        match e:
            case Range2():
                # ``range(start, stop)`` -> ``max(0, stop - start)``.  The
                # difference is constant when both bounds are concrete or
                # share a symbolic base that cancels (e.g.
                # ``range(i, i + 16)`` under REAL -> 16).
                diff = self._affine_diff(
                    self._affine(e.first), self._affine(e.second)
                )
                if diff is not None:
                    return ListSize(None, max(0, diff))
                return ListSize(None, None)
            case _:
                return None

    def _visit_ternaryop(self, e: TernaryOp, ctx: None):
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        self._visit_expr(e.third, ctx)
        match e:
            case Range3():
                # The count depends only on the span ``stop - start`` and
                # the step: ``len(range(a, b, s)) == len(range(0, b-a, s))``.
                # So a constant span (concrete bounds, or a cancelling
                # symbolic base) plus a concrete non-zero step is enough —
                # e.g. ``range(i, i + 16, 2)`` under REAL -> 8.
                span = self._affine_diff(
                    self._affine(e.first), self._affine(e.second)
                )
                step = self._const_int(e.third)
                if span is not None and step is not None and step != 0:
                    return ListSize(None, len(range(0, span, step)))
                return ListSize(None, None)
            case _:
                return None

    def _visit_naryop(self, e: NaryOp, ctx: None):
        tys = [self._visit_expr(arg, ctx) for arg in e.args]
        match e:
            case Zip():
                if len(e.args) == 0:
                    return ListSize(None, 0)

                # FPy's ``zip`` is *strict*: it raises unless every input
                # has the same length (no truncation to the shortest), so
                # the result length *equals* every input length.  We use
                # that runtime-enforced equality:
                #   * any input concrete -> all inputs equal it (or the zip
                #     raises); result is that int.  Conflicting concretes
                #     => always-raising => unknown.
                #   * else all symbolic -> merge their classes (sound: the
                #     zip enforces equality) and keep the representative.
                #   * any input truly unknown (``None``) -> unknown.
                elt_tys: list[ArraySizeBound] = []
                sizes: list[ArraySize] = []
                for ty in tys:
                    assert isinstance(ty, ListSize)
                    elt_tys.append(ty.elt)
                    sizes.append(ty.size)

                concretes = {concrete_size(s, self.uf) for s in sizes}
                concretes.discard(None)
                symbols = [s for s in sizes if isinstance(s, NamedId)]

                if any(s is None for s in sizes):
                    size: ArraySize = None
                elif concretes:
                    # all inputs must equal the concrete length(s)
                    size = next(iter(concretes)) if len(concretes) == 1 else None
                    if size is not None:
                        for s in symbols:
                            self._pin_size(s, size)
                else:
                    # all symbolic: strict zip proves them equal
                    rep = symbols[0]
                    for s in symbols[1:]:
                        self._merge_sizes(rep, s)
                    size = self.uf.find(rep)

                return ListSize(TupleSize(tuple(elt_tys)), size)

            case Empty():
                # iterate from the inner dimension outwards to compute size
                arg_rev = list(reversed(e.args))

                # extract the type of the innermost dimension
                elt_ty = self._cvt_type(self.type_info.by_expr[e])
                for _ in arg_rev:
                    assert isinstance(elt_ty, ListSize)
                    elt_ty = elt_ty.elt

                # innermost dimension
                size_v = self._get_eval(arg_rev[0])
                size = (
                    int(INTEGER.round(size_v))
                    if isinstance(size_v, Float | Fraction)
                    else None
                )

                # type of the innermost dimension
                ty = ListSize(elt_ty, size)

                # outer dimensions
                for arg in arg_rev[1:]:
                    size_v = self._get_eval(arg)
                    size = (
                        int(INTEGER.round(size_v))
                        if isinstance(size_v, Float | Fraction)
                        else None
                    )

                    ty = ListSize(ty, size)

                return ty

            case _:
                return None

    def _visit_list_expr(self, e: ListExpr, ctx: None):
        elt_sizes = [self._visit_expr(elt, ctx) for elt in e.elts]
        if elt_sizes:
            # Unify across all elements so heterogeneous inner sizes
            # widen correctly (e.g., ``[[1.0], [1.0, 2.0]]`` collapses
            # the inner size to ``None`` rather than retaining the
            # first element's size).
            elt_size = reduce(self._unify, elt_sizes)
        else:
            # Empty list literal: derive the structural top of the
            # element type from the type checker's verdict.  This
            # avoids an IndexError on ``[]`` annotated as
            # ``list[list[...]]`` and gives the right shape for any
            # nested type (lists, tuples).
            list_ty = self.type_info.by_expr[e]
            assert isinstance(list_ty, ListType)
            elt_size = self._cvt_type(list_ty.elt)
        return ListSize(elt_size, len(e.elts))

    def _visit_list_comp(self, e: ListComp, ctx: None):
        # process iterables and bindings
        iter_tys: list[ListSize] = []
        for target, iterable in zip(e.targets, e.iterables, strict=True):
            ty = self._visit_expr(iterable, ctx)
            assert isinstance(ty, ListSize)
            self._visit_binding(e, target, ty.elt)
            iter_tys.append(ty)

        # process element expression
        elt_ty = self._visit_expr(e.elt, ctx)

        # A single-iterable comprehension has exactly the iterable's
        # length — propagate its size verbatim, so a *symbolic* size flows
        # through (``[f(x) for x in xs]`` stays length-linked to ``xs``).
        if len(iter_tys) == 1:
            return ListSize(elt_ty, iter_tys[0].size)

        # Multiple iterables form a cartesian product; the size is the
        # product of the lengths, known only when every one is a concrete
        # ``int`` (symbolic sizes don't multiply).
        size = 1
        for ty in iter_tys:
            if not isinstance(ty.size, int):
                return ListSize(elt_ty, None)
            size *= ty.size

        return ListSize(elt_ty, size)

    def _visit_list_ref(self, e: ListRef, ctx: None):
        ty = self._visit_expr(e.value, ctx)
        self._visit_expr(e.index, ctx)
        assert isinstance(ty, ListSize)
        return ty.elt

    def _visit_list_slice(self, e: ListSlice, ctx: None):
        ty = self._visit_expr(e.value, ctx)
        assert isinstance(ty, ListSize)
        if e.start is not None:
            self._visit_expr(e.start, ctx)
        if e.stop is not None:
            self._visit_expr(e.stop, ctx)

        # ``xs[:]`` spans the whole list — propagate its (possibly
        # symbolic) size verbatim.
        if e.start is None and e.stop is None:
            return ListSize(ty.elt, ty.size)

        # The slice size is ``stop - start``, which we can pin whenever
        # that difference is a compile-time constant.  Decomposing each
        # bound into ``base + offset`` (see ``_affine``) covers two cases
        # with one rule:
        #   * both bounds concrete (base ``None``): ``x[1:3]`` -> 2;
        #   * both bounds share a symbolic base that cancels:
        #     ``x[i : i + 16]`` -> 16.
        # Omitted bounds: ``start`` defaults to 0; ``stop`` defaults to
        # the list's own size — usable as an offset only if it's concrete.
        start = (None, 0) if e.start is None else self._affine(e.start)
        if e.stop is None:
            stop = (None, ty.size) if isinstance(ty.size, int) else None
        else:
            stop = self._affine(e.stop)

        slice_size: int | None = None
        if stop is not None:
            diff = self._affine_diff(start, stop)
            # FPy slicing is *strict*: ``xs[a:b]`` is valid only when
            # ``0 <= a <= b <= len(xs)`` — out-of-range / inverted bounds
            # raise rather than clamp.  So a negative difference is an
            # always-raising slice: report unknown, never a bogus size.
            if diff is not None and diff >= 0:
                slice_size = diff

        return ListSize(ty.elt, slice_size)

    def _affine(self, e: Expr) -> tuple[Expr | None, int]:
        """Decompose *e* into ``(base, offset)`` with ``e == base +
        offset``, where *offset* is a compile-time integer and *base* is
        the residual expression (``None`` when *e* is a pure constant).

        Only descends through ``+`` / ``-`` whose result is computed
        under the exact (``REAL``) context: under a rounding context the
        addition could perturb the value, so ``(i + 16) - i`` would not
        be guaranteed to equal ``16``.  Falls back to ``(e, 0)`` when no
        constant offset can be peeled off — sound, just less precise.
        """
        c = self._const_int(e)
        if c is not None:
            return (None, c)

        match e:
            case Add() if self._is_exact(e):
                c = self._const_int(e.second)
                if c is not None:
                    base, off = self._affine(e.first)
                    return (base, off + c)
                c = self._const_int(e.first)
                if c is not None:
                    base, off = self._affine(e.second)
                    return (base, off + c)
            case Sub() if self._is_exact(e):
                c = self._const_int(e.second)
                if c is not None:
                    base, off = self._affine(e.first)
                    return (base, off - c)

        return (e, 0)

    def _affine_diff(
        self,
        lo: tuple[Expr | None, int],
        hi: tuple[Expr | None, int],
    ) -> int | None:
        """``hi - lo`` as a compile-time constant, or ``None``.

        Constant exactly when the two affine forms share a base — both
        pure constants, or the same symbolic base (so it cancels).
        """
        (lbase, loff), (hbase, hoff) = lo, hi
        same_base = (
            (lbase is None and hbase is None)
            or (lbase is not None and hbase is not None
                and lbase.is_equiv(hbase))
        )
        return hoff - loff if same_base else None

    def _const_int(self, e: Expr) -> int | None:
        """The integer value *e* statically evaluates to, or ``None``.

        Covers partial-eval constants and ``len(xs)`` of a list whose
        size is statically known.
        """
        val = self._get_eval(e)
        if isinstance(val, Float | Fraction):
            return int(INTEGER.round(val))
        if isinstance(e, Len):
            inner = self.by_expr.get(e.arg)
            if isinstance(inner, ListSize) and isinstance(inner.size, int):
                return inner.size
        return None

    def _is_exact(self, e: Expr) -> bool:
        """True iff *e* is evaluated under the exact (``REAL``) rounding
        context, so its arithmetic introduces no rounding."""
        scope = self._ctx_use.use_to_scope.get(e)
        return scope is not None and scope.ctx == REAL

    def _visit_tuple_expr(self, e: TupleExpr, ctx: None):
        return TupleSize(tuple(self._visit_expr(elt, ctx) for elt in e.elts))

    def _visit_if_expr(self, e: IfExpr, ctx: None):
        self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, ctx)
        iff = self._visit_expr(e.iff, ctx)
        return self._unify(ift, iff)

    def _visit_call(self, e: Call, ctx: None):
        for arg in e.args:
            self._visit_expr(arg, ctx)
        # Start from the call's type shapeeton (correct shape, all-``None``
        # sizes), then overlay any statically-known sizes the callee's own
        # return-size analysis proved.  A concrete size in a callee's
        # ``ret_size`` is independent of the call's arguments (arg-derived
        # sizes resolve to ``None``), so it's a constant property of the
        # callee and sound to propagate to every call site.
        shapeeton = self._cvt_type(self.type_info.by_expr[e])
        return self._refine_sizes(shapeeton, self._callee_ret_size(e))

    def _callee_ret_size(self, e: Call) -> ArraySizeBound:
        """The callee's inferred return-size bound, or ``None`` when the
        call target isn't an analyzable FPy function (primitive, context
        constructor, or unbound call)."""
        if not isinstance(e.fn, Function):
            return None
        callee = e.fn.ast
        if callee not in self._callee_ret:
            # The call graph is acyclic (enforced by ``TypeInfer.check``,
            # which has already run), so this recursion terminates;
            # memoize so each callee is analyzed once per analysis run.
            self._callee_ret[callee] = ArraySizeInfer.analyze(callee).ret_size
        return self._callee_ret[callee]

    def _refine_sizes(self, shape: ArraySizeBound, src: ArraySizeBound) -> ArraySizeBound:
        """Overlay concrete sizes from *src* onto the structural *shape*.

        *shape* (from the call's type) has the correct shape; *src* (the
        callee's ``ret_size``) may carry concrete sizes but — for a
        generic callee monomorphized at this site — a possibly different
        element shape.  Where shapes agree, prefer *src*'s concrete size;
        on any mismatch, keep *shape* unchanged.

        Only *concrete* ``int`` sizes are adopted: a :class:`SymbolicSize`
        in *src* belongs to the callee's own analysis run and is
        meaningless in ours, so it must not leak across the call.
        """
        match shape, src:
            case ListSize(), ListSize():
                elt = self._refine_sizes(shape.elt, src.elt)
                size = src.size if isinstance(src.size, int) else shape.size
                return ListSize(elt, size)
            case TupleSize(), TupleSize() if len(shape.elts) == len(src.elts):
                elts = tuple(
                    self._refine_sizes(a, b)
                    for a, b in zip(shape.elts, src.elts)
                )
                return TupleSize(elts)
            case _:
                return shape

    def _visit_assign(self, stmt: Assign, ctx: None):
        ty = self._visit_expr(stmt.expr, ctx)
        self._visit_binding(stmt, stmt.target, ty)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: None):
        # ``xs[i1]…[iN] = expr`` is treated as a functional update:
        # ``xs = update(xs, [i1, …, iN], expr)``.  The use of ``xs`` reads
        # the pre-mutation bound; the fresh def at this site (introduced
        # by ``reaching_defs``) receives the widened bound — the inserted
        # value's bound joined into the element bound at depth
        # ``len(indices)``.
        d_use = self.def_use.find_def_from_use(stmt)
        target_ty = self.by_def[d_use]

        def recur(indices: tuple[Expr, ...], target_ty: ArraySizeBound) -> ArraySizeBound:
            if len(indices) == 0:
                ty = self._visit_expr(stmt.expr, ctx)
                return self._unify(target_ty, ty)
            else:
                self._visit_expr(indices[0], ctx)
                assert isinstance(target_ty, ListSize), f'Expected list type for indexed assignment, got {target_ty} for {stmt}'
                elt_ty = recur(indices[1:], target_ty.elt)
                return ListSize(elt_ty, target_ty.size)

        new_ty = recur(stmt.indices, target_ty)
        d_def = self.def_use.find_def_from_site(stmt.var, stmt)
        self.by_def[d_def] = new_ty


    def _visit_if1(self, stmt: If1Stmt, ctx: None):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)

        # unify any merged variable
        for phi in self.def_use.phis[stmt]:
            lhs_ty = self.by_def[self.def_use.defs[phi.lhs]]
            rhs_ty = self.by_def[self.def_use.defs[phi.rhs]]
            self.by_def[phi] = self._unify(lhs_ty, rhs_ty)

    def _visit_if(self, stmt: IfStmt, ctx: None):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.ift, ctx)
        self._visit_block(stmt.iff, ctx)

        # unify any merged variable
        for phi in self.def_use.phis[stmt]:
            lhs_ty = self.by_def[self.def_use.defs[phi.lhs]]
            rhs_ty = self.by_def[self.def_use.defs[phi.rhs]]
            self.by_def[phi] = self._unify(lhs_ty, rhs_ty)

    def _iterate_to_fixpoint(self, phis, run_body):
        """
        Drive a loop's phi-bound fixpoint to convergence.

        Initialises each phi from its pre-loop (lhs) definition, then
        repeatedly runs *run_body* and unifies the post-body (rhs) into
        each phi until no phi changes *and* the union-find is stable.  The
        height is finite (sizes are ``int``/symbol/``None``; the structural
        lattice is shape-preserving; UF merges are monotone), so this
        terminates.

        The UF-stability check matters because a merge inside the body
        (e.g. a ``zip``) can change which sizes are *equivalent* without
        changing any stored phi value; without it the loop could stop one
        iteration before that equivalence propagates.
        """
        for phi in phis:
            self.by_def[phi] = self.by_def[self.def_use.defs[phi.lhs]]
        while True:
            prev = {phi: self.by_def[phi] for phi in phis}
            prev_changes = self._uf_changes
            run_body()
            for phi in phis:
                lhs = self.by_def[self.def_use.defs[phi.lhs]]
                rhs = self.by_def[self.def_use.defs[phi.rhs]]
                self.by_def[phi] = self._unify(lhs, rhs)
            if (self._uf_changes == prev_changes
                    and all(self.by_def[phi] == prev[phi] for phi in phis)):
                break

    def _visit_while(self, stmt: WhileStmt, ctx: None):
        def body():
            self._visit_expr(stmt.cond, ctx)
            self._visit_block(stmt.body, ctx)

        self._iterate_to_fixpoint(self.def_use.phis[stmt], body)

    def _visit_for(self, stmt, ctx):
        # process iterable and binding (once, before the fixpoint)
        iter_ty = self._visit_expr(stmt.iterable, ctx)
        assert isinstance(iter_ty, ListSize)
        self._visit_binding(stmt, stmt.target, iter_ty.elt)

        self._iterate_to_fixpoint(
            self.def_use.phis[stmt], lambda: self._visit_block(stmt.body, ctx)
        )

    def _visit_context(self, stmt, ctx):
        ty = self._visit_expr(stmt.ctx, ctx)
        if isinstance(stmt.target, NamedId):
            d = self.def_use.find_def_from_site(stmt.target, stmt)
            self.by_def[d] = ty
        self._visit_block(stmt.body, ctx)

    def _visit_return(self, stmt: ReturnStmt, ctx: None):
        ret_size = self._visit_expr(stmt.expr, ctx)
        if not isinstance(ret_size, ListSize):
            return
        # Multiple returns: unify the list-size bound across paths.
        # First-seen seeds; subsequent unify against the running
        # value (taking the meet on the size — concrete iff all paths
        # agree, else ``None``).
        if self.ret_size is None:
            self.ret_size = ret_size
        else:
            self.ret_size = self._unify(self.ret_size, ret_size)

    def _visit_expr(self, expr: Expr, ctx: None) -> ArraySizeBound:
        ty = super()._visit_expr(expr, ctx)
        self.by_expr[expr] = ty
        return ty

    def _visit_function(self, func: FuncDef, ctx: None):
        # process arguments — list parameters get a fresh size variable
        for arg, ty in zip(func.args, self.type_info.arg_types):
            if isinstance(arg.name, NamedId):
                d = self.def_use.find_def_from_site(arg.name, arg)
                self.by_def[d] = self._arg_bound(ty)

        # process free variables — same treatment as arguments
        for fv in func.free_vars:
            d = self.def_use.find_def_from_site(fv, func)
            ty = self.type_info.by_def[d]
            self.by_def[d] = self._arg_bound(ty)

        # visit body
        self._visit_block(func.body, ctx)


#####################################################################
# Public API

class ArraySizeInfer:
    """Array size inference."""

    @staticmethod
    def analyze(
        func: FuncDef,
        *,
        partial_eval: PartialEvalInfo | None = None,
        type_info: TypeAnalysis | None = None,
    ) -> ArraySizeAnalysis:
        """Analyze a function definition to infer array sizes.

        Args:
            func: Function definition to analyze.
            partial_eval: Optional pre-computed partial-evaluation info.
            type_info: Optional pre-computed type analysis.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected `FuncDef`, got {type(func)} for {func}')

        if partial_eval is None:
            partial_eval = PartialEval.apply(func)
        if type_info is None:
            type_info = TypeInfer.check(func, def_use=partial_eval.def_use)

        return _ArraySizeInferInstance(func, partial_eval, type_info).analyze()
