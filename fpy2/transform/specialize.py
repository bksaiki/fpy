"""
Module-level specialization.

Expands a :class:`~fpy2.Module` into a new ``Module`` where every function is
fully monomorphized at a specific ``(FuncDef, calling-ctx, argument-formats)``
spec.  Each unique spec becomes one entry; cross-function calls are rewired
to the appropriate spec.

**v2 — arg formats in the key.** The spec key extends v1's
``(FuncDef, ctx)`` with a fingerprint of the per-argument
:class:`FormatBound`\\s — the natural domain produced by
:class:`FormatInfer`.

- **Public** entries convert their user-supplied ``arg_types`` to
  ``FormatBound``\\s via :func:`_type_to_fmt` (``RealType(ctx) →
  ctx.format()``, aggregates recurse).
- **Callee** entries use their ``sub_fa.fn_fmt.arg_fmts`` directly.

The fingerprint hashes the tuple representation; trivial bounds
(``None`` for non-numeric args, ``REAL_FORMAT`` for the polymorphic
top) fingerprint to the empty string, so polymorphic specs pass
through unchanged.

Per-spec monomorphization is ctx-only (``Monomorphize.apply(fdef,
ctx, None)``): callees do not receive per-argument type overrides
beyond the outer ctx, so two callee specs that differ only in arg
format share an identical FuncDef body.  They are still distinct
specs (distinct keys, names, and ``Function`` objects) — the
differentiation lives in the key, not the body.  This keeps the
output module's bodies free of per-arg ctx annotations baked in by
``Specialize``, leaving any such pinning to the consuming backend if
it wants it.
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
from ..number.context.real import REAL_FORMAT
from ..types import BoolType, ListType, RealType, TupleType, Type
from .monomorphize import Monomorphize


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
        # it is ``None`` (callees are monomorphized at the outer ctx only;
        # arg formats live in the spec key, not in the body).
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
                    # Monomorphize the callee with the outer ctx only —
                    # arg formats live in the spec key, not in the body.
                    arg_types_for[callee_key] = None
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
        #        with its original name; private specs surface through the
        #        module's lazy private derivation via rewired call refs.
        out = Module(module.name)
        for entry_name, k in public_keys:
            out.add(new_funcs[k], name=entry_name)
        return out
