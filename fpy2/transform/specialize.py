"""
Module-level specialization.

Expands a :class:`~fpy2.Module` into a new ``Module`` where every function is
fully monomorphized at a specific ``(FuncDef, calling-ctx, argument-formats)``
spec.  Each unique spec becomes one entry; cross-function calls are rewired
to the appropriate spec.

The spec key is ``(FuncDef, calling-ctx, fingerprint of per-argument
FormatBounds)`` — the natural domain produced by :class:`FormatInfer`.
Public entries convert their user-supplied ``arg_types`` to
:class:`FormatBound`\\s via :func:`_type_to_fmt`; callees take their
``arg_fmts`` directly from FormatInfer's per-call-site analysis.  Trivial
bounds (``None`` for non-numeric args, ``REAL_FORMAT`` for the polymorphic
top) fingerprint to the empty string, so polymorphic specs pass through
unchanged.

Callee monomorphization converts ``arg_fmts`` back to ``Type``\\s via
:func:`_bound_to_type` only to feed :class:`Monomorphize` — the key itself
lives in pure :class:`FormatBound` space.  Backends (notably cpp's storage
selection) rely on the resulting per-arg ctx annotations to pick concrete
representations.
"""

import hashlib
from typing import NamedTuple

from ..analysis.array_size import (
    ArraySizeBound,
    ListSize,
    TupleSize,
    concrete_size,
)
from ..analysis.format_infer import (
    FormatBound,
    FormatInfer,
    ListFormat,
    SetFormat,
    TupleFormat,
)
from ..ast import Call, FuncDef
from ..ast.visitor import DefaultTransformVisitor
from ..function import Function
from ..module import Module
from ..number import Context, RoundingMode
from ..number.context.efloat import EFloatContext, EFloatFormat
from ..number.context.exponential import ExpContext, ExpFormat
from ..number.context.fixed import FixedContext, FixedFormat
from ..number.context.format import Format
from ..number.context.ieee754 import IEEEContext, IEEEFormat
from ..number.context.mp_fixed import MPFixedContext, MPFixedFormat
from ..number.context.mp_float import MPFloatContext, MPFloatFormat
from ..number.context.mpb_fixed import MPBFixedContext, MPBFixedFormat
from ..number.context.mpb_float import MPBFloatContext, MPBFloatFormat
from ..number.context.mps_float import MPSFloatContext, MPSFloatFormat
from ..number.context.real import REAL_FORMAT, RealFormat
from ..number.context.sm_fixed import SMFixedContext, SMFixedFormat
from ..types import BoolType, ListType, RealType, TupleType, Type
from .monomorphize import Monomorphize

# ----------------------------------------------------------------------
# Format -> Context recovery + FormatBound -> Type conversion (used only
# to feed `Monomorphize` at callees — the spec key does *not* go through
# this conversion).

def _format_to_ctx(fmt: Format) -> Context | None:
    """Best-effort recovery of a :class:`Context` from a :class:`Format`.

    Each format is paired with the context class that describes it, and the
    context is rebuilt via that class's ``from_format``.  Returns ``None``
    when no context can describe the format — the caller falls back to
    ``RealType(None)``.

    The cases are ordered most-derived first, since ``IEEEFormat`` is an
    ``EFloatFormat`` and ``FixedFormat``/``SMFixedFormat`` are both
    ``MPBFixedFormat``\\s.
    """
    # A `Format` describes a set of values, not how to round into it, so the
    # rounding mode has to be chosen here.  RNE is `from_format`'s default and
    # matches every canonical float context; the fixed-point family instead
    # uses RTZ, which is what every canonical integer context (`SINT*`,
    # `UINT*`, `INTEGER`) is built with and what the cpp backend requires of
    # integer storage, since C++ integer arithmetic truncates.
    if isinstance(fmt, MPFixedFormat | MPBFixedFormat):
        rm = RoundingMode.RTZ
    else:
        rm = RoundingMode.RNE

    try:
        match fmt:
            # `IEEEFormat` before `EFloatFormat`
            case IEEEFormat():
                return IEEEContext.from_format(fmt, rm=rm)
            case EFloatFormat():
                return EFloatContext.from_format(fmt, rm=rm)
            # `FixedFormat` and `SMFixedFormat` before `MPBFixedFormat`
            case FixedFormat():
                return FixedContext.from_format(fmt, rm=rm)
            case SMFixedFormat():
                return SMFixedContext.from_format(fmt, rm=rm)
            case MPBFixedFormat():
                return MPBFixedContext.from_format(fmt, rm=rm)
            case MPFixedFormat():
                return MPFixedContext.from_format(fmt, rm=rm)
            case ExpFormat():
                return ExpContext.from_format(fmt, rm=rm)
            case MPBFloatFormat():
                return MPBFloatContext.from_format(fmt, rm=rm)
            case MPSFloatFormat():
                return MPSFloatContext.from_format(fmt, rm=rm)
            case MPFloatFormat():
                return MPFloatContext.from_format(fmt, rm=rm)
            case RealFormat():
                # the polymorphic top: callers treat it as "no context"
                return None
            case _:
                return None
    except (NotImplementedError, TypeError, ValueError):
        # some `from_format`s reject formats their context cannot express
        # (e.g. NaN/Inf disabled); recovery is best-effort
        return None


def _bound_to_type(
    bound: FormatBound, size: ArraySizeBound = None,
) -> Type | None:
    """Convert a :class:`FormatBound` to a :class:`Type` for use as a
    ``Monomorphize`` argument override.

    Scalar ``Format`` bounds attempt ``Format → Context`` recovery via
    :func:`_format_to_ctx` and become ``RealType(<recovered ctx>)`` on
    success (fallback ``RealType(None)`` otherwise).  ``SetFormat`` and
    ``None`` collapse to ``RealType(None)`` / ``None``.

    *size* rides along structurally: a ``ListSize`` whose length is a
    concrete ``int`` puts that length on the ``ListType``, which is how a
    caller's proven argument length reaches the callee's annotations (and
    from there its own array-size analysis).  A shape mismatch or ``None``
    contributes nothing.
    """
    if bound is None:
        return None
    if isinstance(bound, TupleFormat):
        sizes: tuple[ArraySizeBound, ...] = (
            size.elts
            if isinstance(size, TupleSize) and len(size.elts) == len(bound.elts)
            else (None,) * len(bound.elts)
        )
        elt_types: list[Type] = []
        for e, se in zip(bound.elts, sizes):
            t = _bound_to_type(e, se)
            if t is None:
                return None
            elt_types.append(t)
        return TupleType(*elt_types)
    if isinstance(bound, ListFormat):
        elt_size = size.elt if isinstance(size, ListSize) else None
        elt_type = _bound_to_type(bound.elt, elt_size)
        if elt_type is None:
            return None
        length = concrete_size(size.size) if isinstance(size, ListSize) else None
        return ListType(elt_type, length)
    if isinstance(bound, SetFormat):
        return RealType(None)
    assert isinstance(bound, Format), f'unexpected FormatBound: {type(bound)}'
    return RealType(_format_to_ctx(bound))


def _arg_fmts_to_arg_types(
    arg_fmts: tuple[FormatBound, ...] | None,
    arg_sizes: 'tuple[ArraySizeBound, ...] | None' = None,
) -> tuple[Type | None, ...] | None:
    """Per-argument ``FormatBound → Type`` for ``Monomorphize``."""
    if arg_fmts is None:
        return None
    if arg_sizes is None or len(arg_sizes) != len(arg_fmts):
        arg_sizes = (None,) * len(arg_fmts)
    return tuple(_bound_to_type(b, s) for b, s in zip(arg_fmts, arg_sizes))


class _SpecKey(NamedTuple):
    """A specialization is identified by the original ``FuncDef``, the
    calling (outer) context, a stable fingerprint of the per-argument
    :class:`FormatBound`\\s, and -- when the caller asked for size keying --
    one of the per-argument concrete lengths."""
    fdef: FuncDef
    ctx: Context | None
    arg_fmts_fp: str   # '' when no arg formats are pinned
    arg_sizes_fp: str  # '' when no argument carries a concrete length


# ----------------------------------------------------------------------
# Type -> FormatBound conversion (for public keying).


def _type_to_fmt(t: Type | None) -> FormatBound:
    """Convert a :class:`Type` to a :class:`FormatBound` for spec keying.

    ``RealType(ctx)`` → ``ctx.format()`` (a ``Format``).  Aggregates
    recurse (``TupleType`` → ``TupleFormat``, ``ListType`` →
    ``ListFormat``).  Non-numeric and ctx-less types yield ``None`` —
    no specialization info to key on."""
    if t is None or isinstance(t, BoolType):
        return None
    if isinstance(t, RealType):
        if t.ctx is None or not isinstance(t.ctx, Context):
            return None
        return t.ctx.format()
    if isinstance(t, TupleType):
        return TupleFormat(tuple(_type_to_fmt(e) for e in t.elts))
    if isinstance(t, ListType):
        return ListFormat(_type_to_fmt(t.elt))
    return None


# ----------------------------------------------------------------------
# Spec-key fingerprints.


def _ctx_fingerprint(ctx: Context) -> str:
    """A short, stable, identifier-safe fingerprint for a context.  Used
    in mangled private-spec names *and* in the spec key.  Matches cpp's
    ``_ctx_fingerprint`` in shape (SHA-1 of ``str(ctx)`` truncated to 8
    hex chars) so the two layers can eventually share a mangling scheme."""
    return hashlib.sha1(str(ctx).encode()).hexdigest()[:8]


def _is_trivial_fmt(f: FormatBound) -> bool:
    """A :class:`FormatBound` that conveys no specialization information:
    ``None`` (non-numeric) or ``REAL_FORMAT`` (the polymorphic scalar
    top)."""
    return f is None or f is REAL_FORMAT or f == REAL_FORMAT


def _arg_fmts_fingerprint(
    arg_fmts: tuple[FormatBound, ...] | None,
) -> str:
    """A short fingerprint of a per-argument :class:`FormatBound` tuple.
    Returns ``''`` when *arg_fmts* is ``None`` or every entry is trivial
    — so polymorphic specs pass through unchanged.  Otherwise distinct
    bound reprs dedupe to one spec via SHA-1 (matching cpp's mangling
    shape)."""
    if arg_fmts is None or all(_is_trivial_fmt(f) for f in arg_fmts):
        return ''
    parts = [repr(f) if f is not None else 'X' for f in arg_fmts]
    raw = '|'.join(parts)
    return hashlib.sha1(raw.encode()).hexdigest()[:8]


def _sanitize_size(b: ArraySizeBound) -> ArraySizeBound:
    """*b* with every non-concrete size dropped.

    Size *variables* (``NamedId``) are per-analysis gensyms: letting one
    into a fingerprint would make spec keys and mangled names differ from
    run to run.  Only ``int`` lengths survive; the shape is kept so nested
    and tuple-carried lengths line up positionally.
    """
    match b:
        case ListSize():
            return ListSize(_sanitize_size(b.elt), concrete_size(b.size))
        case TupleSize():
            return TupleSize(tuple(_sanitize_size(e) for e in b.elts))
        case None:
            return None


def _type_to_size(t: Type | None) -> ArraySizeBound:
    """The concrete lengths a user-supplied ``arg_type`` carries, as an
    :class:`ArraySizeBound` -- the size analog of :func:`_type_to_fmt`, for
    public keying.  Two entries differing only in length must not collapse
    to one spec: the length reaches the annotations and, with the cpp
    backend's arrays, the ABI."""
    if isinstance(t, TupleType):
        return TupleSize(tuple(_type_to_size(e) for e in t.elts))
    if isinstance(t, ListType):
        length = t.length if isinstance(t.length, int) else None
        return ListSize(_type_to_size(t.elt), length)
    return None


def _has_concrete_size(b: ArraySizeBound) -> bool:
    """Whether *b* carries an ``int`` length at any level."""
    match b:
        case ListSize():
            return isinstance(b.size, int) or _has_concrete_size(b.elt)
        case TupleSize():
            return any(_has_concrete_size(e) for e in b.elts)
        case None:
            return False


def _arg_sizes_fingerprint(
    arg_sizes: 'tuple[ArraySizeBound, ...] | None',
) -> str:
    """A short fingerprint of *sanitized* per-argument sizes.  Returns
    ``''`` when nothing carries a concrete length -- so a size-free program
    produces byte-identical keys and mangled names to a size-blind run."""
    if arg_sizes is None or not any(_has_concrete_size(s) for s in arg_sizes):
        return ''
    parts = [repr(s) if s is not None else 'X' for s in arg_sizes]
    return hashlib.sha1('|'.join(parts).encode()).hexdigest()[:8]


def _mangle_private(
    name: str, ctx: Context | None, arg_fmts_fp: str, arg_sizes_fp: str = '',
) -> str:
    """Build a stable name for a private spec.  Includes the ctx
    fingerprint (when present) and the arg-format and arg-size
    fingerprints (when non-empty), so two specs of the same function with
    different ``(ctx, arg_fmts, arg_sizes)`` produce distinguishable
    names."""
    parts = [name]
    if ctx is not None:
        parts.append(_ctx_fingerprint(ctx))
    if arg_fmts_fp:
        parts.append(arg_fmts_fp)
    if arg_sizes_fp:
        parts.append(arg_sizes_fp)
    return '__'.join(parts)


# ----------------------------------------------------------------------
# Per-call-site rebinder (same shape as v1).


class _RebindCallSites(DefaultTransformVisitor):
    """Rebuild a function body, swapping each ``Call.fn`` per a
    *per-call-site* map (``Call → Function``).  Within one specialized
    caller, the same callee can be invoked at different specs from
    different sites, so the rebind is keyed on the ``Call`` node itself."""

    def __init__(self, mapping: dict[Call, Function]):
        self._mapping = mapping

    def _visit_call(self, e: Call, ctx):
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        kwargs = [(k, self._visit_expr(v, ctx)) for k, v in e.kwargs]
        fn = self._mapping.get(e, e.fn)
        return Call(e.func, fn, args, kwargs, e.loc)

    def apply(self, func: FuncDef) -> FuncDef:
        return self._visit_function(func, None)


# ----------------------------------------------------------------------
# The pass.


class Specialize:
    """Module → Module pass that expands public entries into a flat set
    of fully-monomorphized specializations.

    Each ``(FuncDef, calling-ctx, arg-types-fingerprint)`` triple becomes
    one entry; cross-function calls are rewired to the appropriate spec.
    Public entries' user-given names are preserved; transitively-reached
    private specs get a stable mangled name combining the original name
    with the ctx and arg-types fingerprints.

    The output is assembled by registering only the public specs with
    :meth:`Module.add`; private specs surface through ``add``'s eager
    call-graph discovery (they're reachable from the publics' rewired
    ``Call.fn`` references).

    Cyclic input call graphs surface at :meth:`Module.add` time on the
    input module, before this pass runs.  Any cycle introduced by
    specialization itself would surface at the output ``add`` call.
    """

    @staticmethod
    def apply(module: Module, *, size_key: bool = False) -> Module:
        """Specialize *module*.

        *size_key* additionally keys each spec on its arguments' concrete
        lengths, so one function called with 3- and 5-element lists compiles
        twice, each spec's annotations carrying its length -- what lets the
        cpp backend's arrays cross call edges.  One spec per distinct
        length vector is the same template-instantiation economics as the
        ctx and format axes; lengths originate in program text (literals,
        ``empty(K)``, ``range(K)``, annotations), so the worklist stays
        finite.  ``False`` keeps keys and mangled names byte-identical to a
        size-blind run.
        """
        if not isinstance(module, Module):
            raise TypeError(f'expected a `Module`, got {type(module)} for {module}')

        # --- 1. Enumerate specs.  Start with each public entry; walk
        #        callees via `FormatInfer.by_call` (which gives the
        #        calling ctx *and* per-argument formats per call site).
        monos: dict[_SpecKey, FuncDef] = {}
        call_targets: dict[_SpecKey, dict[Call, _SpecKey]] = {}
        callees_of: dict[_SpecKey, list[_SpecKey]] = {}
        orig_func: dict[_SpecKey, Function] = {}
        # The arg_types tuple used to monomorphize each spec.  For public
        # roots this is the user-supplied ``entry.arg_types``; for callees
        # it is derived from ``sub_fa.fn_fmt.arg_fmts`` via
        # ``_arg_fmts_to_arg_types`` so the body's arg annotations get
        # per-arg ctx pinning (needed by cpp's storage selection).
        arg_types_for: dict[_SpecKey, tuple[Type | None, ...] | None] = {}

        public_keys: list[tuple[str, _SpecKey]] = []   # (entry_name, key) per public

        worklist: list[_SpecKey] = []
        for entry in module:
            atypes = entry.arg_types
            # Derive arg_fmts from the user-supplied arg_types so the key
            # lives in FormatBound space (matching what callees produce).
            pub_arg_fmts = (
                tuple(_type_to_fmt(t) for t in atypes)
                if atypes is not None else None
            )
            # ...and the lengths those arg_types carry, so two entries
            # differing only in length get distinct specs.
            pub_arg_sizes = (
                tuple(_type_to_size(t) for t in atypes)
                if size_key and atypes is not None else None
            )
            key = _SpecKey(
                fdef=entry.func.ast,
                ctx=entry.ctx,
                arg_fmts_fp=_arg_fmts_fingerprint(pub_arg_fmts),
                arg_sizes_fp=_arg_sizes_fingerprint(pub_arg_sizes),
            )
            public_keys.append((entry.name, key))
            if key not in orig_func:
                orig_func[key] = entry.func
                arg_types_for[key] = atypes
                worklist.append(key)

        seen: set[_SpecKey] = set(worklist)
        while worklist:
            key = worklist.pop(0)
            atypes = arg_types_for.get(key)
            mono = Monomorphize.apply(key.fdef, key.ctx, atypes)
            monos[key] = mono

            # FormatInfer gives, for each Function-targeted Call in
            # ``mono``, the sub-analysis whose ``fn_fmt`` describes the
            # callee at that call site — calling ctx + per-argument
            # format bounds.  Both feed the callee's spec identity.
            fa = FormatInfer.analyze(mono)
            site_map: dict[Call, _SpecKey] = {}
            local_callees: list[_SpecKey] = []
            local_seen: set[_SpecKey] = set()
            for call, sub_fa in fa.by_call.items():
                callee_fn = call.fn
                assert isinstance(callee_fn, Function)  # FormatInfer only records these
                callee_ctx_raw = sub_fa.fn_fmt.ctx
                # Only concrete ``Context``s count; symbolic / None collapse.
                callee_ctx = callee_ctx_raw if isinstance(callee_ctx_raw, Context) else None
                callee_arg_fmts = sub_fa.fn_fmt.arg_fmts
                # The caller-side proven length of each argument expression,
                # sanitized to concrete ints (a symbolic size is a per-run
                # gensym and must never reach a fingerprint).
                callee_arg_sizes = (
                    tuple(
                        _sanitize_size(fa.array_size.by_expr.get(a))
                        for a in call.args
                    )
                    if size_key else None
                )
                callee_key = _SpecKey(
                    fdef=callee_fn.ast,
                    ctx=callee_ctx,
                    arg_fmts_fp=_arg_fmts_fingerprint(callee_arg_fmts),
                    arg_sizes_fp=_arg_sizes_fingerprint(callee_arg_sizes),
                )

                site_map[call] = callee_key
                if callee_key not in local_seen:
                    local_seen.add(callee_key)
                    local_callees.append(callee_key)
                if callee_key not in seen:
                    seen.add(callee_key)
                    orig_func[callee_key] = callee_fn
                    # ``Monomorphize`` takes Types; convert the arg_fmts
                    # here (and only here — the key already lives in
                    # FormatBound space) so the body's arg annotations
                    # get per-arg ctx pinning that backends need.
                    arg_types_for[callee_key] = _arg_fmts_to_arg_types(
                        callee_arg_fmts, callee_arg_sizes,
                    )
                    worklist.append(callee_key)

            call_targets[key] = site_map
            callees_of[key] = local_callees

        # --- 2. Topological sort (leaves-first).
        order: list[_SpecKey] = []
        visited: set[_SpecKey] = set()

        def _post_order(k: _SpecKey):
            if k in visited:
                return
            visited.add(k)
            for cee in callees_of[k]:
                _post_order(cee)
            order.append(k)

        for k in monos:
            _post_order(k)

        # --- 3. Decide names.  Publics: first registering entry's name
        #        (no mangling).  Privates: original name + ctx + arg_types
        #        fingerprints.
        spec_to_public_name: dict[_SpecKey, str] = {}
        for entry_name, k in public_keys:
            spec_to_public_name.setdefault(k, entry_name)

        names: dict[_SpecKey, str] = {}
        for k in monos:
            if k in spec_to_public_name:
                names[k] = spec_to_public_name[k]
            else:
                names[k] = _mangle_private(
                    orig_func[k].name, k.ctx, k.arg_fmts_fp, k.arg_sizes_fp,
                )

        # --- 4. Build new ``Function``s in leaves-first order, rewiring
        #        each spec's body to point at the already-built callee
        #        specs (per-call-site).
        new_funcs: dict[_SpecKey, Function] = {}
        for k in order:
            site_to_func = {
                call: new_funcs[callee_k]
                for call, callee_k in call_targets[k].items()
            }
            rewired = _RebindCallSites(site_to_func).apply(monos[k])
            rewired.name = names[k]
            new_funcs[k] = orig_func[k].with_ast(rewired)

        # --- 5. Assemble the output module.  Each public entry is re-added
        #        with its original name; private specs are picked up
        #        automatically by ``add``'s eager call-graph discovery,
        #        which walks the rewired ``Call.fn`` references.
        out = Module(module.name)
        for entry_name, k in public_keys:
            out.add(new_funcs[k], name=entry_name)
        return out
