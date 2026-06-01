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
from ..number import Context
from ..number.context.format import Format
from ..number.context.real import REAL_FORMAT
from ..types import BoolType, ListType, RealType, TupleType, Type
from .monomorphize import Monomorphize


# ----------------------------------------------------------------------
# Format -> Context recovery + FormatBound -> Type conversion (used only
# to feed `Monomorphize` at callees — the spec key does *not* go through
# this conversion).


_FORMAT_TO_CTX: dict[Format, Context] | None = None


def _format_to_ctx(fmt: Format) -> Context | None:
    """Best-effort recovery of a canonical :class:`Context` from a
    :class:`Format`.  Returns ``None`` for formats outside the canonical
    registry — the caller falls back to ``RealType(None)``."""
    global _FORMAT_TO_CTX
    if _FORMAT_TO_CTX is None:
        from ..libraries.base import (
            FP16, FP32, FP64, FP128, FP256, BF16, TF32, INTEGER,
            SINT8, SINT16, SINT32, SINT64,
            UINT8, UINT16, UINT32, UINT64,
            S1E5M2, S1E4M3,
            MX_E5M2, MX_E4M3, MX_E3M2, MX_E2M3, MX_E2M1,
            MX_E8M0, MX_INT8,
            FP8P1, FP8P2, FP8P3, FP8P4, FP8P5, FP8P6, FP8P7,
        )
        canonical = (
            FP16, FP32, FP64, FP128, FP256, BF16, TF32, INTEGER,
            SINT8, SINT16, SINT32, SINT64,
            UINT8, UINT16, UINT32, UINT64,
            S1E5M2, S1E4M3,
            MX_E5M2, MX_E4M3, MX_E3M2, MX_E2M3, MX_E2M1,
            MX_E8M0, MX_INT8,
            FP8P1, FP8P2, FP8P3, FP8P4, FP8P5, FP8P6, FP8P7,
        )
        _FORMAT_TO_CTX = {}
        for ctx in canonical:
            _FORMAT_TO_CTX.setdefault(ctx.format(), ctx)
    return _FORMAT_TO_CTX.get(fmt)


def _bound_to_type(bound: FormatBound) -> Type | None:
    """Convert a :class:`FormatBound` to a :class:`Type` for use as a
    ``Monomorphize`` argument override.

    Scalar ``Format`` bounds attempt ``Format → Context`` recovery via
    :func:`_format_to_ctx` and become ``RealType(<recovered ctx>)`` on
    success (fallback ``RealType(None)`` otherwise).  ``SetFormat`` and
    ``None`` collapse to ``RealType(None)`` / ``None``.
    """
    if bound is None:
        return None
    if isinstance(bound, TupleFormat):
        elt_types: list[Type] = []
        for e in bound.elts:
            t = _bound_to_type(e)
            if t is None:
                return None
            elt_types.append(t)
        return TupleType(*elt_types)
    if isinstance(bound, ListFormat):
        elt_type = _bound_to_type(bound.elt)
        if elt_type is None:
            return None
        return ListType(elt_type)
    if isinstance(bound, SetFormat):
        return RealType(None)
    assert isinstance(bound, Format), f'unexpected FormatBound: {type(bound)}'
    return RealType(_format_to_ctx(bound))


def _arg_fmts_to_arg_types(
    arg_fmts: tuple[FormatBound, ...] | None,
) -> tuple[Type | None, ...] | None:
    """Per-argument ``FormatBound → Type`` for ``Monomorphize``."""
    if arg_fmts is None:
        return None
    return tuple(_bound_to_type(b) for b in arg_fmts)


class _SpecKey(NamedTuple):
    """A specialization is identified by the original ``FuncDef``, the
    calling (outer) context, and a stable fingerprint of the per-argument
    :class:`FormatBound`\\s."""
    fdef: FuncDef
    ctx: Context | None
    arg_fmts_fp: str   # '' when no arg formats are pinned


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


def _mangle_private(
    name: str, ctx: Context | None, arg_fmts_fp: str,
) -> str:
    """Build a stable name for a private spec.  Includes the ctx
    fingerprint (when present) and the arg-format fingerprint (when
    non-empty), so two specs of the same function with different
    ``(ctx, arg_fmts)`` produce distinguishable names."""
    parts = [name]
    if ctx is not None:
        parts.append(_ctx_fingerprint(ctx))
    if arg_fmts_fp:
        parts.append(arg_fmts_fp)
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
    def apply(module: Module) -> Module:
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
            key = _SpecKey(
                fdef=entry.func.ast,
                ctx=entry.ctx,
                arg_fmts_fp=_arg_fmts_fingerprint(pub_arg_fmts),
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
                callee_key = _SpecKey(
                    fdef=callee_fn.ast,
                    ctx=callee_ctx,
                    arg_fmts_fp=_arg_fmts_fingerprint(callee_arg_fmts),
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
                        callee_arg_fmts,
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
                    orig_func[k].name, k.ctx, k.arg_fmts_fp,
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
