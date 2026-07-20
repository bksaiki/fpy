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

from contextlib import contextmanager
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
from .context_use import ContextUse, ContextUseAnalysis, ContextUseSite
from .define_use import Definition, DefSite, DefineUseAnalysis
from .partial_eval import PartialEval, PartialEvalInfo, Value
from .type_infer import TypeInfer, TypeAnalysis

__all__ = [
    'ArraySize',
    'ArraySizeAnalysis',
    'ArraySizeBound',
    'ArraySizeInfer',
    'ListSize',
    'TupleSize',
    'concrete_size',
    'is_size_eq',
]


#####################################################################
# Array-size lattice

ArraySize: TypeAlias = 'int | NamedId | None'
"""
Static size of a list-valued expression — a flat lattice:

- a concrete ``int`` — a known, compile-time-constant length;
- a :class:`NamedId` *size variable* — an unknown-but-definite length,
  tracked so equalities (``len(ys) == len(xs)``, strict ``zip``) survive:
  in the resolved result, two equal variables denote equal lengths;
- ``None`` — top: no tracked information.  Arises from a join of
  disagreeing sizes *or* from an unknown we don't bother tracking; either
  way it is never equal to anything, even another ``None``.

Any two distinct atoms (``int``s / variable classes) join to ``None``.
During analysis the variables live in a union-find; :class:`ArraySizeInfer`
returns a result **resolved** to representatives (see :meth:`_resolve`), so
afterwards a size is known iff it is an ``int`` and provable equality is
just ``==`` (via :func:`is_size_eq`).  Variables are minted per run and are
not meaningful across functions.
"""

SizeUnionfind: TypeAlias = 'Unionfind[NamedId | int]'
"""Internal: equivalence classes of size variables (and the concrete
``int``s they are pinned to), used *during* analysis.  A concrete ``int``
is always the class representative, so resolving a variable is a ``find``
that lands on an ``int`` when known.  It is resolved away before the
result is returned (see :meth:`_ArraySizeInferInstance._resolve`), so
consumers never see it."""


def _repr_size(size: ArraySize, uf: SizeUnionfind) -> int | NamedId | None:
    """The representative of *size*'s class: an ``int`` if the class is
    pinned to a constant, else the size variable itself (``None`` stays
    ``None``)."""
    if isinstance(size, NamedId):
        return uf.get(size, size)
    return size


def concrete_size(size: ArraySize) -> int | None:
    """The concrete ``int`` *size* denotes, or ``None`` if unknown.
    Assumes *size* comes from a resolved :class:`ArraySizeAnalysis`."""
    return size if isinstance(size, int) else None


def list_depth(bound: 'ArraySizeBound') -> int:
    """Number of nested list dimensions in *bound* (== ``dim``)."""
    depth = 0
    while isinstance(bound, ListSize):
        depth += 1
        bound = bound.elt
    return depth


def _size_eq(a: ArraySize, b: ArraySize) -> bool:
    """Are two (resolved) sizes provably equal — identical and not an
    untracked unknown (``None`` is never equal, even to itself)?"""
    return a is not None and a == b


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

def is_size_eq(b1: ArraySizeBound, b2: ArraySizeBound) -> bool:
    """Structurally compare two (resolved) bounds, treating equal sizes
    at each level as equal."""
    match b1, b2:
        case ListSize(), ListSize():
            return _size_eq(b1.size, b2.size) and is_size_eq(b1.elt, b2.elt)
        case TupleSize(), TupleSize():
            return (len(b1.elts) == len(b2.elts)
                    and all(is_size_eq(a, b) for a, b in zip(b1.elts, b2.elts)))
        case None, None:
            return True
        case _:
            return False


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
    _cond_depth: int
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
        # Counts unions; lets the loop fixpoint detect progress when a
        # merge changes equivalences without changing any stored bound.
        self._uf_changes = 0
        # Nesting depth in conditionally-executed regions (if / loop
        # bodies); only depth-0 asserts hold on every execution.
        self._cond_depth = 0
        self._callee_ret = {}
        self._ctx_use_cache = None

    @contextmanager
    def _branch(self):
        """Mark the enclosed block as conditionally executed."""
        self._cond_depth += 1
        try:
            yield
        finally:
            self._cond_depth -= 1

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
        # Lazy: only symbolic-offset slices need it.  Reuses partial_eval.
        if self._ctx_use_cache is None:
            self._ctx_use_cache = ContextUse.analyze(
                self.func, partial_eval=self.partial_eval
            )
        return self._ctx_use_cache

    def analyze(self) -> ArraySizeAnalysis:
        self._visit_function(self.func, None)
        # Resolve every size to its representative so the result is
        # self-contained (cf. type inference's ``_resolve_type``).
        by_expr = {e: self._resolve(b) for e, b in self.by_expr.items()}
        by_def = {d: self._resolve(b) for d, b in self.by_def.items()}
        ret_size = self._resolve(self.ret_size)
        return ArraySizeAnalysis(by_expr, by_def, ret_size, self.partial_eval.def_use)

    def _resolve(self, bound: ArraySizeBound) -> ArraySizeBound:
        """Replace every size variable in *bound* with its union-find
        representative (an ``int`` if pinned, else the canonical var)."""
        match bound:
            case ListSize():
                return ListSize(self._resolve(bound.elt), _repr_size(bound.size, self.uf))
            case TupleSize():
                return TupleSize(tuple(self._resolve(e) for e in bound.elts))
            case _:
                return None

    def _annotated_size(self, length: int | NamedId | None) -> ArraySize:
        """The array-size seeded by a :class:`ListType`'s optional length:
        a concrete ``int``; a symbolic ``NamedId`` (registered in the
        union-find, so equal dim names share a class); or ``None`` when the
        annotation gives no size."""
        if isinstance(length, NamedId):
            self.uf.add(length)
        return length

    def _cvt_type(self, ty: Type) -> ArraySizeBound:
        match ty:
            case ListType():
                elt = self._cvt_type(ty.elt)
                return ListSize(elt, self._annotated_size(ty.length))
            case TupleType():
                elts = tuple(self._cvt_type(e) for e in ty.elts)
                return TupleSize(elts)
            case _:
                return None

    def _arg_bound(self, ty: Type) -> ArraySizeBound:
        """Like :meth:`_cvt_type`, but a list parameter's *outer* length,
        when the annotation gives none, gets a fresh size variable (its
        length is fixed per call, just unknown), so equalities such as
        ``ys = xs`` are tracked."""
        match ty:
            case ListType():
                size = self._annotated_size(ty.length)
                if size is None:
                    size = self._fresh_size()
                return ListSize(self._cvt_type(ty.elt), size)
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
                # ``range(stop)`` -> ``max(0, stop)`` for a known ``stop``
                # (``_const_int`` also resolves ``len(xs)``).
                n = self._const_int(e.arg)
                if n is not None:
                    return ListSize(None, max(0, n))
                return ListSize(None, None)
            case Enumerate():
                assert isinstance(ty, ListSize)
                return ListSize(TupleSize((None, ty.elt)), ty.size)
            case Fst():
                # tuple head — the first element's bound (if tracked).
                if isinstance(ty, TupleSize) and len(ty.elts) >= 1:
                    return ty.elts[0]
                return None
            case Snd():
                # tuple tail — the second element's bound for a pair, else
                # the bounds of the remaining elements.
                if isinstance(ty, TupleSize) and len(ty.elts) >= 2:
                    rest = ty.elts[1:]
                    return rest[0] if len(rest) == 1 else TupleSize(rest)
                return None
            case _:
                return None

    def _visit_binaryop(self, e: BinaryOp, ctx: None):
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        match e:
            case Range2():
                # ``range(start, stop)`` -> ``max(0, stop - start)`` when
                # the difference is constant (see ``_affine_diff``).
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
                # ``len(range(a, b, s)) == len(range(0, b - a, s))``, so a
                # constant span plus a concrete non-zero step suffices.
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

                # ``zip`` is *strict*: it raises unless every input has the
                # same length, so the result length equals them all.  A
                # concrete input fixes the size (and pins the rest); else
                # the unknowns are merged; any ``None`` input -> unknown.
                elt_tys: list[ArraySizeBound] = []
                sizes: list[ArraySize] = []
                for ty in tys:
                    assert isinstance(ty, ListSize)
                    elt_tys.append(ty.elt)
                    sizes.append(ty.size)

                concretes = {r for s in sizes
                             if isinstance((r := _repr_size(s, self.uf)), int)}
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
                # Nest dimension sizes from the inside out; each dim is its
                # concrete ``int`` if known, else ``None``.
                arg_rev = list(reversed(e.args))
                elt_ty = self._cvt_type(self.type_info.by_expr[e])
                for _ in arg_rev:
                    assert isinstance(elt_ty, ListSize)
                    elt_ty = elt_ty.elt

                acc: ArraySizeBound = elt_ty
                for arg in arg_rev:
                    size_v = self._get_eval(arg)
                    size = (
                        int(INTEGER.round(size_v))
                        if isinstance(size_v, Float | Fraction)
                        else None
                    )
                    acc = ListSize(acc, size)
                return acc

            case _:
                return None

    def _visit_list_expr(self, e: ListExpr, ctx: None):
        elt_sizes = [self._visit_expr(elt, ctx) for elt in e.elts]
        if elt_sizes:
            # Unify so conflicting inner sizes widen (``[[1.0], [1.0, 2.0]]``
            # collapses the inner size to ``None``).
            elt_size = reduce(self._unify, elt_sizes)
        else:
            # Empty literal: shape the element bound from its declared type
            # (avoids indexing ``elt_sizes[0]`` and handles nested ``[]``).
            list_ty = self.type_info.by_expr[e]
            assert isinstance(list_ty, ListType)
            elt_size = self._cvt_type(list_ty.elt)
        return ListSize(elt_size, len(e.elts))

    def _visit_list_comp(self, e: ListComp, ctx: None):
        iter_tys: list[ListSize] = []
        for target, iterable in zip(e.targets, e.iterables, strict=True):
            ty = self._visit_expr(iterable, ctx)
            assert isinstance(ty, ListSize)
            self._visit_binding(e, target, ty.elt)
            iter_tys.append(ty)
        elt_ty = self._visit_expr(e.elt, ctx)

        # One iterable: same length (propagates a symbolic size, so
        # ``[f(x) for x in xs]`` stays length-linked to ``xs``).
        if len(iter_tys) == 1:
            return ListSize(elt_ty, iter_tys[0].size)

        # Several iterables: cartesian product, known only if all concrete.
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

        # Size is ``stop - start`` when that difference is a compile-time
        # constant (``x[1:3]`` -> 2, or ``x[i:i+16]`` where the base
        # cancels; see ``_affine``).  An omitted ``stop`` is the list's own
        # size, usable only when concrete.
        start: tuple[Expr | None, int]
        start = (None, 0) if e.start is None else self._affine(e.start)
        stop: tuple[Expr | None, int] | None
        if e.stop is None:
            stop = (None, ty.size) if isinstance(ty.size, int) else None
        else:
            stop = self._affine(e.stop)

        slice_size: int | None = None
        if stop is not None:
            diff = self._affine_diff(start, stop)
            # Slicing is strict: an inverted/out-of-range slice raises, so a
            # negative difference stays unknown rather than a bogus size.
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

        Covers partial-eval constants and the integer-valued list queries
        ``len(xs)`` / ``size(xs, d)`` / ``dim(xs)`` when the queried size
        is statically known (``dim`` always is — it's the nesting depth).
        """
        val = self._get_eval(e)
        if isinstance(val, Float | Fraction):
            return int(INTEGER.round(val))
        match e:
            case Len():
                bound = self.by_expr.get(e.arg)
                if isinstance(bound, ListSize) and isinstance(bound.size, int):
                    return bound.size
            case Dim():
                bound = self.by_expr.get(e.arg)
                if isinstance(bound, ListSize):
                    return self._list_depth(bound)
            case Size():
                d = self._const_int(e.second)
                if d is None:
                    return None
                bound = self.by_expr.get(e.first)
                for _ in range(d):
                    if not isinstance(bound, ListSize):
                        return None
                    bound = bound.elt
                if isinstance(bound, ListSize) and isinstance(bound.size, int):
                    return bound.size
        return None

    @staticmethod
    def _list_depth(bound: ArraySizeBound) -> int:
        """Number of nested list dimensions in *bound* (== ``dim``)."""
        return list_depth(bound)

    def _is_exact(self, e: ContextUseSite) -> bool:
        """True iff *e* is evaluated under the exact (``REAL``) rounding
        context, so its arithmetic introduces no rounding.  Only called on
        arithmetic nodes (always context-use sites), so the lookup is safe
        even though ``e`` is typed wider than the map's key."""
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
        # Take the call's type shape, then overlay any concrete size the
        # callee's own return-size analysis proved — a concrete callee
        # ``ret_size`` is arg-independent, so it holds at every call site.
        shape = self._cvt_type(self.type_info.by_expr[e])
        return self._refine_sizes(shape, self._callee_ret_size(e))

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

        Only *concrete* ``int`` sizes are adopted: a *src* size variable
        belongs to the callee's own run and must not leak across the call.
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
        # ``xs[i1]…[iN] = expr`` is a functional update: read the
        # pre-mutation bound off the use, then write the fresh def with the
        # inserted value joined into the element bound at depth N.
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
        with self._branch():
            self._visit_block(stmt.body, ctx)

        # unify any merged variable
        for phi in self.def_use.phis[stmt]:
            lhs_ty = self.by_def[self.def_use.defs[phi.lhs]]
            rhs_ty = self.by_def[self.def_use.defs[phi.rhs]]
            self.by_def[phi] = self._unify(lhs_ty, rhs_ty)

    def _visit_if(self, stmt: IfStmt, ctx: None):
        self._visit_expr(stmt.cond, ctx)
        with self._branch():
            self._visit_block(stmt.ift, ctx)
            self._visit_block(stmt.iff, ctx)

        # unify any merged variable
        for phi in self.def_use.phis[stmt]:
            lhs_ty = self.by_def[self.def_use.defs[phi.lhs]]
            rhs_ty = self.by_def[self.def_use.defs[phi.rhs]]
            self.by_def[phi] = self._unify(lhs_ty, rhs_ty)

    def _iterate_to_fixpoint(self, phis, run_body):
        """
        Drive a loop's phi-bound fixpoint to convergence: seed each phi
        from its pre-loop value, then run the body and unify the post-body
        value into each phi until both the phis and the union-find are
        stable.  The UF check is needed because a body merge (e.g. a
        ``zip``) can change equivalences without moving any stored phi.
        Terminates: finite-height lattice and monotone UF merges.
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

        with self._branch():
            self._iterate_to_fixpoint(self.def_use.phis[stmt], body)

    def _visit_for(self, stmt, ctx):
        # iterable + target bound once, before the fixpoint
        iter_ty = self._visit_expr(stmt.iterable, ctx)
        assert isinstance(iter_ty, ListSize)
        self._visit_binding(stmt, stmt.target, iter_ty.elt)

        with self._branch():
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
        # Across multiple returns, unify: concrete iff all paths agree.
        if self.ret_size is None:
            self.ret_size = ret_size
        else:
            self.ret_size = self._unify(self.ret_size, ret_size)

    def _visit_assert(self, stmt: AssertStmt, ctx: None):
        self._visit_expr(stmt.test, ctx)
        # Only an *unconditional* assert holds on every execution, so only
        # then may it constrain sizes globally (cf. strict ``zip``).
        if self._cond_depth == 0:
            self._seed_from_assert(stmt.test)

    def _seed_from_assert(self, test: Expr):
        """Learn size equalities from ``len(a) == len(b)`` / ``len(a) == N``
        (and chained ``==``) — pins or merges the size variables."""
        match test:
            case And():
                for arg in test.args:
                    self._seed_from_assert(arg)
            case Compare():
                for op, lhs, rhs in zip(test.ops, test.args, test.args[1:]):
                    if op == CompareOp.EQ:
                        self._relate_sizes(self._len_size(lhs), self._len_size(rhs))
            case _:
                pass

    def _len_size(self, e: Expr) -> ArraySize:
        """The size ``e`` constrains: the list's size for ``len(xs)``, the
        constant for an integer expression, else ``None`` (not relatable)."""
        if isinstance(e, Len):
            bound = self.by_expr.get(e.arg)
            return bound.size if isinstance(bound, ListSize) else None
        return self._const_int(e)

    def _relate_sizes(self, a: ArraySize, b: ArraySize):
        """Record that two sizes are equal: merge two variables, or pin a
        variable to a concrete length."""
        match a, b:
            case NamedId(), NamedId():
                self._merge_sizes(a, b)
            case NamedId(), int():
                self._pin_size(a, b)
            case int(), NamedId():
                self._pin_size(b, a)
            case _:
                pass  # both concrete (already constrained) or unknown

    def _visit_expr(self, expr: Expr, ctx: None) -> ArraySizeBound:
        ty = super()._visit_expr(expr, ctx)
        self.by_expr[expr] = ty
        return ty

    def _visit_function(self, func: FuncDef, ctx: None):
        # arguments and free variables: list params get a fresh size var
        for arg, ty in zip(func.args, self.type_info.arg_types):
            if isinstance(arg.name, NamedId):
                d = self.def_use.find_def_from_site(arg.name, arg)
                self.by_def[d] = self._arg_bound(ty)
        for fv in func.free_vars:
            d = self.def_use.find_def_from_site(fv, func)
            ty = self.type_info.by_def[d]
            self.by_def[d] = self._arg_bound(ty)

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
