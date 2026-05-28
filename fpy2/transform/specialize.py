"""
Module-level specialization.

Expands a :class:`~fpy2.Module` into a new ``Module`` where every function is
fully monomorphized at a specific calling context.  Each
``(FuncDef, calling-ctx)`` pair becomes one entry; cross-function calls are
rewired to the appropriate spec.

First-cut design: **context-only** key (no per-argument format keying — to be
added later as a strategy option).  Public entries with no ``ctx`` spec pass
through (their bodies and their transitively-reached callees stay
polymorphic); the same ``(FuncDef, None)`` spec is reused across all such
publics.
"""

import hashlib

from typing import NamedTuple

from ..analysis.format_infer import FormatInfer
from ..ast import Call, FuncDef
from ..ast.visitor import DefaultTransformVisitor
from ..function import Function
from ..module import Module
from ..number import Context
from .monomorphize import Monomorphize


class _SpecKey(NamedTuple):
    """A specialization is identified by the original ``FuncDef`` and the
    calling (outer) context it is monomorphized at."""
    fdef: FuncDef
    ctx: Context | None


class _RebindCallSites(DefaultTransformVisitor):
    """Rebuild a function body, swapping each ``Call.fn`` per a
    *per-call-site* map (``Call → Function``).  Used by :class:`Specialize`:
    within one specialized caller, the same callee can be invoked at
    different specs from different sites, so the rebind is keyed on the
    ``Call`` node itself rather than on the original callee ``Function``."""

    def __init__(self, mapping: dict[Call, Function]):
        self._mapping = mapping

    def _visit_call(self, e: Call, ctx):
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        kwargs = [(k, self._visit_expr(v, ctx)) for k, v in e.kwargs]
        fn = self._mapping.get(e, e.fn)
        return Call(e.func, fn, args, kwargs, e.loc)

    def apply(self, func: FuncDef) -> FuncDef:
        return self._visit_function(func, None)


def _ctx_fingerprint(ctx: Context) -> str:
    """A short, stable, identifier-safe fingerprint for a context, used in
    mangled private-spec names.  An 8-hex-character BLAKE2 digest of
    ``repr(ctx)`` — long enough to be collision-free in practice, short
    enough to keep names readable."""
    return hashlib.blake2b(repr(ctx).encode('utf-8'), digest_size=4).hexdigest()


def _mangle_private(name: str, ctx: Context | None) -> str:
    return name if ctx is None else f'{name}__{_ctx_fingerprint(ctx)}'


class Specialize:
    """Module → Module pass that expands public entries into a flat set of
    fully-monomorphized specializations.

    Each ``(FuncDef, calling-context)`` pair becomes one function in the
    output module; cross-function calls in the output reference the
    appropriate spec.  Public entries' user-given names are preserved
    (they become the FuncDef name of their specialization), while
    transitively-reached private specs get a stable mangled name
    (``original_name__<ctx-fingerprint>``).

    Raises ``CallGraphError`` if the input has a cyclic call graph (FPy
    forbids recursion).
    """

    @staticmethod
    def apply(module: Module) -> Module:
        if not isinstance(module, Module):
            raise TypeError(f'expected a `Module`, got {type(module)} for {module}')

        # --- 1. Enumerate specs.  Start with each public entry; walk callees
        #        by querying `FormatInfer.by_call` for the calling ctx.
        monos: dict[_SpecKey, FuncDef] = {}
        call_targets: dict[_SpecKey, dict[Call, _SpecKey]] = {}
        callees_of: dict[_SpecKey, list[_SpecKey]] = {}
        orig_func: dict[_SpecKey, Function] = {}
        # Public roots may carry an `arg_types` override; callees never do.
        arg_types_for: dict[_SpecKey, tuple | None] = {}

        public_keys: list[tuple[str, _SpecKey]] = []  # (entry_name, spec) per public entry

        worklist: list[_SpecKey] = []
        for entry in module:
            key = _SpecKey(entry.func.ast, entry.ctx)
            public_keys.append((entry.name, key))
            if key not in orig_func:
                orig_func[key] = entry.func
                arg_types_for[key] = entry.arg_types
                worklist.append(key)

        seen: set[_SpecKey] = set(worklist)
        while worklist:
            key = worklist.pop(0)
            atypes = arg_types_for.get(key)
            mono = Monomorphize.apply(key.fdef, key.ctx, atypes)
            monos[key] = mono

            # FormatInfer gives, for each top-level Call in `mono`, the
            # sub-analysis whose `fn_fmt.ctx` is the callee's calling context.
            fa = FormatInfer.analyze(mono)
            site_map: dict[Call, _SpecKey] = {}
            local_callees: list[_SpecKey] = []
            local_seen: set[_SpecKey] = set()
            for call, sub_fa in fa.by_call.items():
                callee_fn = call.fn
                assert isinstance(callee_fn, Function)  # FormatInfer only records these
                callee_ctx_raw = sub_fa.fn_fmt.ctx
                # Only concrete `Context`s count; symbolic (NamedId) or None
                # collapse to the polymorphic spec.
                callee_ctx = callee_ctx_raw if isinstance(callee_ctx_raw, Context) else None
                callee_key = _SpecKey(callee_fn.ast, callee_ctx)

                site_map[call] = callee_key
                if callee_key not in local_seen:
                    local_seen.add(callee_key)
                    local_callees.append(callee_key)
                if callee_key not in seen:
                    seen.add(callee_key)
                    orig_func.setdefault(callee_key, callee_fn)
                    arg_types_for.setdefault(callee_key, None)
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

        # --- 3. Decide names.  Publics: the first registering entry's name
        #        (no mangling).  Privates: original name + ctx fingerprint.
        spec_to_public_name: dict[_SpecKey, str] = {}
        for entry_name, k in public_keys:
            spec_to_public_name.setdefault(k, entry_name)

        names: dict[_SpecKey, str] = {}
        for k in monos:
            if k in spec_to_public_name:
                names[k] = spec_to_public_name[k]
            else:
                names[k] = _mangle_private(orig_func[k].name, k.ctx)

        # --- 4. Build new `Function`s in leaves-first order, rewiring each
        #        spec body's `Call.fn` to the already-built callee specs.
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
        #        with its original user-given name; private specs are reached
        #        through the rewired call references and surface via the
        #        module's lazy private derivation.
        out = Module(module.name)
        for entry_name, k in public_keys:
            out.add(new_funcs[k], name=entry_name)
        return out
