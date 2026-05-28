"""
Module-level specialization.

Expands a :class:`~fpy2.Module` into a new ``Module`` where every function is
fully monomorphized at a specific ``(FuncDef, calling-ctx, argument-formats)``
spec.  Each unique spec becomes one entry; cross-function calls are rewired
to the appropriate spec.

**v2 — arg-types in the key.** The spec key extends v1's
``(FuncDef, ctx)`` with a fingerprint of the per-argument ``Type``\\s the
spec is monomorphized at.  This closes the public-level dedup gap of v1:
two registrations of the same function at the same outer ctx but with
different user-provided ``arg_types`` now produce distinct specs (v1
silently collapsed them onto the first registration).

Trivial arg types (``None`` or ``RealType(None)`` — i.e., no shape or ctx
info pinned) fingerprint to the empty string, so polymorphic callees pass
through unchanged.  Scalar **per-argument context** at *callee* call sites
is intentionally not part of the key: the outer calling context already
flows through the callee's body — the callee at FP32 vs. FP64 is a
distinct spec via the ``ctx`` field — so finer per-arg ctx differentiation
adds no semantic content.  (Storage-level per-arg-format differentiation
is a cpp-emission concern handled by cpp's own pipeline, not by
``Specialize``.)
"""

import hashlib

from typing import NamedTuple

from ..analysis.format_infer import (
    FormatBound,
    FormatInfer,
    ListFormat,
    TupleFormat,
)
from ..ast import Call, FuncDef
from ..ast.visitor import DefaultTransformVisitor
from ..function import Function
from ..module import Module
from ..number import Context
from ..types import ListType, RealType, TupleType, Type
from .monomorphize import Monomorphize


class _SpecKey(NamedTuple):
    """A specialization is identified by the original ``FuncDef``, the
    calling (outer) context, and a stable fingerprint of the per-argument
    types it is monomorphized at."""
    fdef: FuncDef
    ctx: Context | None
    arg_types_fp: str   # '' when no arg_types are pinned


# ----------------------------------------------------------------------
# FormatBound -> Type conversion (for Monomorphize input).


def _bound_to_type(bound: FormatBound) -> Type | None:
    """Best-effort conversion of a ``FormatBound`` to a :class:`Type` for
    use as a ``Monomorphize`` argument override.

    Returns ``None`` when the bound carries no useful structural
    information (Monomorphize then leaves the parameter's annotated type
    alone).  Scalar ``Format`` / ``SetFormat`` bounds become
    ``RealType(None)`` — they pin "this argument is a scalar real" without
    yet recovering the underlying ``Context`` (deferred).
    """
    if bound is None:
        return None
    if isinstance(bound, TupleFormat):
        # If any element couldn't be converted, bail on the whole arg.
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
    # Scalar bound (concrete ``Format`` or ``SetFormat``).
    return RealType(None)


def _arg_fmts_to_arg_types(
    arg_fmts: tuple[FormatBound, ...] | None,
) -> tuple[Type | None, ...] | None:
    """Convert a per-argument ``FormatBound`` tuple to a per-argument
    ``Type | None`` tuple consumable by :class:`Monomorphize`."""
    if arg_fmts is None:
        return None
    return tuple(_bound_to_type(b) for b in arg_fmts)


# ----------------------------------------------------------------------
# Spec-key fingerprints.


def _ctx_fingerprint(ctx: Context) -> str:
    """A short, stable, identifier-safe fingerprint for a context.  Used
    in mangled private-spec names *and* in the spec key.  Matches cpp's
    ``_ctx_fingerprint`` in shape (SHA-1 of ``str(ctx)`` truncated to 8
    hex chars) so the two layers can eventually share a mangling scheme."""
    return hashlib.sha1(str(ctx).encode()).hexdigest()[:8]


def _is_trivial_type(t: Type | None) -> bool:
    """A type that conveys no specialization information: ``None`` (no
    override) or ``RealType(None)`` (just "scalar real")."""
    return t is None or (isinstance(t, RealType) and t.ctx is None)


def _arg_types_fingerprint(
    arg_types: tuple[Type | None, ...] | None,
) -> str:
    """A short fingerprint of a per-argument types tuple.  Returns an
    empty string when *arg_types* is ``None`` or every entry is "trivial"
    (no shape or context info pinned) — so polymorphic specs pass through
    unchanged.  Otherwise, distinct ``Type.format()`` strings dedupe to one
    spec via BLAKE2."""
    if arg_types is None or all(_is_trivial_type(t) for t in arg_types):
        return ''
    parts = [t.format() if t is not None else 'X' for t in arg_types]
    raw = '|'.join(parts)
    return hashlib.sha1(raw.encode()).hexdigest()[:8]


def _mangle_private(
    name: str, ctx: Context | None, arg_types_fp: str,
) -> str:
    """Build a stable name for a private spec.  Includes the ctx
    fingerprint (when present) and the arg-types fingerprint (when
    non-empty), so two specs of the same function with different
    ``(ctx, arg_types)`` produce distinguishable names."""
    parts = [name]
    if ctx is not None:
        parts.append(_ctx_fingerprint(ctx))
    if arg_types_fp:
        parts.append(arg_types_fp)
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

    Raises :class:`CallGraphError` if the input has a cyclic call graph.
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
        # ``_bound_to_type``.
        arg_types_for: dict[_SpecKey, tuple[Type | None, ...] | None] = {}

        public_keys: list[tuple[str, _SpecKey]] = []   # (entry_name, key) per public

        worklist: list[_SpecKey] = []
        for entry in module:
            atypes = entry.arg_types
            key = _SpecKey(
                fdef=entry.func.ast,
                ctx=entry.ctx,
                arg_types_fp=_arg_types_fingerprint(atypes),
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
            # callee at that call site — both its calling ctx and the
            # per-argument format bounds.
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
                callee_arg_types = _arg_fmts_to_arg_types(sub_fa.fn_fmt.arg_fmts)
                callee_key = _SpecKey(
                    fdef=callee_fn.ast,
                    ctx=callee_ctx,
                    arg_types_fp=_arg_types_fingerprint(callee_arg_types),
                )

                site_map[call] = callee_key
                if callee_key not in local_seen:
                    local_seen.add(callee_key)
                    local_callees.append(callee_key)
                if callee_key not in seen:
                    seen.add(callee_key)
                    orig_func.setdefault(callee_key, callee_fn)
                    arg_types_for.setdefault(callee_key, callee_arg_types)
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
                    orig_func[k].name, k.ctx, k.arg_types_fp,
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
        #        with its original name; private specs surface through the
        #        module's lazy private derivation via rewired call refs.
        out = Module(module.name)
        for entry_name, k in public_keys:
            out.add(new_funcs[k], name=entry_name)
        return out
