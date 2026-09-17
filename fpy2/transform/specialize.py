"""
Module-level specialization.

Expands a :class:`~fpy2.Module` into a new ``Module`` where every function is
fully monomorphized at a specific ``(FuncDef, calling-ctx, argument-types)``
spec.  Each unique spec becomes one entry; cross-function calls are rewired
to the appropriate spec.

A spec is keyed on two axes.  The refined argument *types* say what each
argument may be -- a public entry supplies them directly, a callee gets them
from FormatInfer's per-call-site ``arg_fmts`` and the caller-proven lengths.
The pinned argument *values* say which value it is, for the contexts and
rounding modes partial evaluation resolved at the call site; those are
substituted into the body rather than annotated, since a value is not a type.
Both axes are exactly what ``Monomorphize`` is given, so one fingerprint means
one body.  An argument that pins nothing contributes nothing, leaving
polymorphic specs unchanged.
"""

import hashlib
from typing import NamedTuple

from ..analysis.array_size import (
    ArraySizeBound,
    ListSize,
    TupleSize,
    concrete_size,
)
from ..analysis.define_use import AssignDef, DefineUse
from ..analysis.format_infer import (
    FormatBound,
    FormatInfer,
    ListFormat,
    SetFormat,
    TupleFormat,
    VarFormat,
    to_abstract,
)
from ..analysis.partial_eval import PartialEvalInfo
from ..ast import Call, Expr, ForeignVal, FuncDef
from ..ast.visitor import DefaultTransformVisitor
from ..function import Function
from ..interpret.value import Foreign
from ..module import Module
from ..number import Context, RoundingMode
from ..number.context.format import Format
from ..number.context.real import REAL_FORMAT
from ..types import ListType, RealType, TupleType, Type
from ..utils import NamedId
from .monomorphize import Monomorphize
from .subst_var import SubstVar

# ----------------------------------------------------------------------
# FormatBound -> Type conversion, used only to feed `Monomorphize` at
# callees — the spec key does *not* go through it.

def _bound_to_type(
    bound: FormatBound, size: ArraySizeBound = None,
) -> Type | None:
    """Convert a :class:`FormatBound` to a :class:`Type` for use as a
    ``Monomorphize`` argument override.

    A scalar ``Format`` becomes ``RealType(fmt)`` directly.  A ``SetFormat``
    names no ``Format``, so it is widened to the tightest one containing its
    values -- a superset, which is what the callee's storage has to hold
    anyway.  ``None`` and :class:`VarFormat` name no type at all.

    *size* rides along structurally: a ``ListSize`` with a concrete ``int``
    length puts that length on the ``ListType``, which is how a caller's proven
    argument length reaches the callee's annotations, and from there the
    callee's own array-size analysis.  A shape mismatch or ``None`` contributes
    nothing.
    """
    if bound is None or isinstance(bound, VarFormat):
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
        af = to_abstract(bound)
        return RealType(None if af is None else af.format())
    assert isinstance(bound, Format), f'unexpected FormatBound: {type(bound)}'
    return RealType(bound)


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
    """A specialization is identified by the original ``FuncDef``, the calling
    (outer) context, and a stable fingerprint of the refined argument types --
    which are exactly what :class:`Monomorphize` is given."""
    fdef: FuncDef
    ctx: Context | None
    arg_types_fp: str  # '' when the argument types constrain nothing
    arg_vals_fp: str = ''  # '' when no argument is pinned to a value


def _ctx_fingerprint(ctx: Context) -> str:
    """A short, stable, identifier-safe fingerprint for a context.  Used
    in mangled private-spec names *and* in the spec key.  Matches cpp's
    ``_ctx_fingerprint`` in shape (SHA-1 of ``str(ctx)`` truncated to 8
    hex chars) so the two layers can eventually share a mangling scheme."""
    return hashlib.sha1(str(ctx).encode()).hexdigest()[:8]


def _is_trivial_fmt(f: 'FormatBound') -> bool:
    """A :class:`FormatBound` that conveys no specialization information:
    ``None`` (non-numeric) or ``REAL_FORMAT`` (the polymorphic scalar
    top)."""
    return f is None or f is REAL_FORMAT or f == REAL_FORMAT


def _type_pin(t: Type | None, size_key: bool) -> str | None:
    """What *t* pins, as a canonical string, or ``None`` when it pins nothing.

    The format of a real, the shape of an aggregate, and the length of a list
    all distinguish one spec from another.  *size_key* off drops lengths,
    keeping keys byte-identical to a size-blind run.
    """
    match t:
        case RealType():
            return None if _is_trivial_fmt(t.fmt) else f'r{t.fmt!r}'
        case ListType():
            # the shape pins even when the leaves do not: a spec whose argument
            # is known to be a list differs from one where nothing is known
            elt = _type_pin(t.elt, size_key)
            # a symbolic length is a per-run gensym and must never be keyed on
            n = concrete_size(t.length) if size_key else None
            return f'l[{elt or "_"};{"_" if n is None else n}]'
        case TupleType():
            return 't[' + ';'.join(
                _type_pin(e, size_key) or '_' for e in t.elts
            ) + ']'
        case _:
            return None


def _arg_vals_fingerprint(vals: tuple[object, ...] | None) -> str:
    """A short fingerprint of the argument values a call pins.

    A *separate* axis from :func:`_arg_types_fingerprint`: a format or a length
    constrains what a value may be, while a pin says which value it is.  That
    is partial evaluation, not typing, so it keys on its own and is applied by
    substitution rather than by an annotation.
    """
    if vals is None or all(v is None for v in vals):
        return ''
    raw = '|'.join('X' if v is None else f'{type(v).__name__}:{v}' for v in vals)
    return hashlib.sha1(raw.encode()).hexdigest()[:8]


def _arg_types_fingerprint(
    atypes: 'tuple[Type | None, ...] | None', size_key: bool,
) -> str:
    """A short fingerprint of the refined argument types.

    ``''`` when nothing is pinned, so a polymorphic spec passes through
    unchanged.  These types are exactly what :class:`Monomorphize` is given,
    so two specs with the same fingerprint have identical bodies.
    """
    if atypes is None:
        return ''
    pins = [_type_pin(t, size_key) for t in atypes]
    if all(p is None for p in pins):
        return ''
    raw = '|'.join(p if p is not None else 'X' for p in pins)
    return hashlib.sha1(raw.encode()).hexdigest()[:8]


def _sanitize_size(b: ArraySizeBound) -> ArraySizeBound:
    """*b* with every non-concrete size dropped, and ``None`` when nothing
    concrete survives at any level.

    Size *variables* (``NamedId``) are per-analysis gensyms: letting one into a
    fingerprint would make spec keys and mangled names differ from run to run.
    So only ``int`` lengths survive, and the structure around them is kept only
    where one does, so that nested and tuple-carried lengths line up
    positionally.
    """
    match b:
        case ListSize():
            elt, k = _sanitize_size(b.elt), concrete_size(b.size)
            if elt is None and k is None:
                return None
            return ListSize(elt, k)
        case TupleSize():
            elts = tuple(_sanitize_size(e) for e in b.elts)
            if all(e is None for e in elts):
                return None
            return TupleSize(elts)
        case None:
            return None


def _mangle_private(
    name: str, ctx: Context | None, arg_types_fp: str, arg_vals_fp: str = '',
) -> str:
    """Build a stable name for a private spec, from the ctx, argument-type and
    pinned-value fingerprints, so two specs of the same function are
    distinguishable."""
    parts = [name]
    if ctx is not None:
        parts.append(_ctx_fingerprint(ctx))
    if arg_types_fp:
        parts.append(arg_types_fp)
    if arg_vals_fp:
        # `v`-tagged so a type and a value fingerprint cannot collide
        parts.append(f'v{arg_vals_fp}')
    return '__'.join(parts)


_PINNABLE = (Context, RoundingMode)
"""Value kinds a call site may pin into a callee.

Both reach a rounding: a :class:`Context` directly, a :class:`RoundingMode`
through a context constructor.  Numbers are left out -- their formats already
travel as types, and pinning every constant would multiply specs for no gain.
"""


def _pinned_value(v: object) -> object | None:
    """*v* if a call site may pin it, else ``None``.

    Partial evaluation wraps a value FPy cannot compute on in a ``Foreign``,
    which is how a rounding mode arrives.
    """
    if isinstance(v, Foreign):
        v = v.val
    return v if isinstance(v, _PINNABLE) else None


def _pinnable_args(
    callee: FuncDef, args: tuple[Expr, ...], pe: PartialEvalInfo,
) -> tuple[object, ...]:
    """The value each of *args* pins in *callee*, or ``None`` where it pins
    nothing.

    A discarded parameter takes no substitution, so keying on its value would
    only split one spec into identical copies.  An unused *named* parameter
    still splits, which costs a duplicate body rather than a wrong one.
    """
    out: list[object] = []
    for i, a in enumerate(args):
        param = callee.args[i].name if i < len(callee.args) else None
        val = _pinned_value(pe.by_expr.get(a))
        out.append(val if isinstance(param, NamedId) else None)
    return tuple(out)


def _pin_arg_values(
    func: FuncDef, vals: tuple[object, ...] | None,
) -> FuncDef:
    """*func* with each pinned parameter replaced by its value.

    The pin is a partial-evaluation fact, not a type, so it is applied by
    substituting the value into the body -- which is what makes a ``with`` over
    a context *parameter* resolvable.  The parameter is left in place and
    becomes dead; dropping it would have to rewrite every call site too.
    """
    if vals is None or all(v is None for v in vals) or not func.args:
        return func

    def_use = DefineUse.analyze(func)
    subst: dict[AssignDef, Expr] = {}
    for arg, val in zip(func.args, vals):
        if val is None or not isinstance(arg.name, NamedId):
            continue
        # both kinds are opaque to FPy and spell as a foreign literal
        subst[def_use.find_def_from_site(arg.name, arg)] = ForeignVal(val, None)
    if not subst:
        return func
    return SubstVar.apply(func, def_use, subst)


# ----------------------------------------------------------------------
# Per-call-site rebinder.


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
        lengths, so a function called with 3- and 5-element lists compiles
        twice, each spec's annotations carrying its length -- which is what lets
        the cpp backend's arrays cross call edges.  Lengths originate in program
        text (literals, ``empty(K)``, ``range(K)``, annotations), so the
        worklist stays finite; the cost is the same template-instantiation
        economics as the ctx and format axes.  ``False`` keeps keys and mangled
        names byte-identical to a size-blind run.
        """
        if not isinstance(module, Module):
            raise TypeError(f'expected a `Module`, got {type(module)} for {module}')

        # --- 1. Enumerate specs: each public entry, then its callees.
        monos: dict[_SpecKey, FuncDef] = {}
        call_targets: dict[_SpecKey, dict[Call, _SpecKey]] = {}
        callees_of: dict[_SpecKey, list[_SpecKey]] = {}
        orig_func: dict[_SpecKey, Function] = {}
        # What each spec is monomorphized at: a public root's own
        # `arg_types`, or a callee's derived from the call site.
        arg_types_for: dict[_SpecKey, tuple[Type | None, ...] | None] = {}
        # Per-spec argument values the caller pinned, applied by substitution.
        arg_vals_for: dict[_SpecKey, tuple[object, ...] | None] = {}

        public_keys: list[tuple[str, _SpecKey]] = []   # (entry_name, key) per public

        worklist: list[_SpecKey] = []
        for entry in module:
            atypes = entry.arg_types
            key = _SpecKey(
                fdef=entry.func.ast,
                ctx=entry.ctx,
                arg_types_fp=_arg_types_fingerprint(atypes, size_key),
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
            mono = _pin_arg_values(mono, arg_vals_for.get(key))
            monos[key] = mono

            fa = FormatInfer.analyze(mono)
            pe = fa.partial_eval
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
                # concrete ints only: a symbolic size is a per-run gensym
                callee_arg_sizes = (
                    tuple(
                        _sanitize_size(fa.array_size.by_expr.get(a))
                        for a in call.args
                    )
                    if size_key else None
                )
                # both the spec's identity and what `Monomorphize` is given
                callee_atypes = _arg_fmts_to_arg_types(
                    callee_arg_fmts, callee_arg_sizes,
                )
                # a `Context` makes a callee's `with` resolvable; a rounding
                # mode reaches one through a context constructor
                callee_arg_vals = _pinnable_args(callee_fn.ast, call.args, pe)
                callee_key = _SpecKey(
                    fdef=callee_fn.ast,
                    ctx=callee_ctx,
                    arg_types_fp=_arg_types_fingerprint(callee_atypes, size_key),
                    arg_vals_fp=_arg_vals_fingerprint(callee_arg_vals),
                )

                site_map[call] = callee_key
                if callee_key not in local_seen:
                    local_seen.add(callee_key)
                    local_callees.append(callee_key)
                if callee_key not in seen:
                    seen.add(callee_key)
                    orig_func[callee_key] = callee_fn
                    arg_types_for[callee_key] = callee_atypes
                    arg_vals_for[callee_key] = callee_arg_vals
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

        # --- 3. Name each spec: a public keeps its entry name, a private
        #        is mangled from the fingerprints.
        spec_to_public_name: dict[_SpecKey, str] = {}
        for entry_name, k in public_keys:
            spec_to_public_name.setdefault(k, entry_name)

        names: dict[_SpecKey, str] = {}
        for k in monos:
            if k in spec_to_public_name:
                names[k] = spec_to_public_name[k]
            else:
                names[k] = _mangle_private(
                    orig_func[k].name, k.ctx, k.arg_types_fp, k.arg_vals_fp,
                )

        # --- 4. Build leaves-first, rewiring each body per call site.
        new_funcs: dict[_SpecKey, Function] = {}
        for k in order:
            site_to_func = {
                call: new_funcs[callee_k]
                for call, callee_k in call_targets[k].items()
            }
            rewired = _RebindCallSites(site_to_func).apply(monos[k])
            rewired.name = names[k]
            new_funcs[k] = orig_func[k].with_ast(rewired)

        # --- 5. Re-add the publics; `add` discovers the privates through
        #        the rewired `Call.fn` references.
        out = Module(module.name)
        for entry_name, k in public_keys:
            out.add(new_funcs[k], name=entry_name)
        return out
