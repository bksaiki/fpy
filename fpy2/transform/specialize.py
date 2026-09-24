"""
Module-level specialization.

Expands a :class:`~fpy2.Module` into a new ``Module`` where every function is
fully monomorphized at one ``(FuncDef, _Instance)`` spec.  Each unique spec
becomes one entry; cross-function calls are rewired to the appropriate spec.

:class:`_Instance` names the axes a spec is keyed on.  The refined argument
*types* say what each argument may be -- a public entry supplies them
directly, a callee gets them from FormatInfer's per-call-site ``arg_fmts``
and the caller-proven lengths.  The pinned argument *values* say which value
it is, for the contexts and rounding modes partial evaluation resolved at the
call site; those are substituted into the body rather than annotated, since a
value is not a type.  The derived *bounds* say what the caller's digit-bound
analysis proved inside the callee.  Types and values are exactly what
``Monomorphize`` is given, so one key means one body.  An argument that pins
nothing contributes nothing, leaving polymorphic specs unchanged.
"""

import hashlib
from collections import Counter
from dataclasses import dataclass, replace
from typing import NamedTuple, TypeAlias

from ..analysis.alias import Alias
from ..analysis.array_size import (
    ArraySizeBound,
    ListSize,
    TupleSize,
    concrete_size,
)
from ..analysis.define_use import AssignDef, DefineUse
from ..analysis.digit_bound import DigitBoundParams
from ..analysis.escape import Escape, EscapeSummary
from ..analysis.format_infer import (
    FormatAnalysis,
    FormatBound,
    FormatInfer,
    ListFormat,
    SetFormat,
    TupleFormat,
    VarFormat,
    to_abstract,
)
from ..analysis.partial_eval import PartialEvalInfo
from ..analysis.type_infer import TypeInferError
from ..analysis.value_class import ValueClassInfer
from ..ast import Call, Expr, ForeignVal, FuncDef
from ..ast.visitor import DefaultTransformVisitor, DefaultVisitor
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


def _escape_summaries(
    fd: FuncDef, memo: dict[FuncDef, EscapeSummary],
) -> dict[FuncDef, EscapeSummary]:
    """*memo*, holding a summary for every function *fd* calls, transitively.

    From each callee's own body, leaves first.  One that does not type on its
    own is left out, which reads as retaining everything.
    """
    callees: list[FuncDef] = []

    class _Calls(DefaultVisitor):
        def _visit_call(self, e: Call, ctx):
            if isinstance(e.fn, Function):
                callees.append(e.fn.ast)
            super()._visit_call(e, ctx)

    _Calls()._visit_function(fd, None)
    for callee in callees:
        if callee not in memo:
            _escape_summaries(callee, memo)
            try:
                memo[callee] = Escape.analyze(callee, memo)
            except TypeInferError:
                pass
    return memo


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
    arg_sizes: tuple[ArraySizeBound, ...] | None = None,
) -> tuple[Type | None, ...] | None:
    """Per-argument ``FormatBound → Type`` for ``Monomorphize``."""
    if arg_fmts is None:
        return None
    if arg_sizes is None or len(arg_sizes) != len(arg_fmts):
        arg_sizes = (None,) * len(arg_fmts)
    return tuple(_bound_to_type(b, s) for b, s in zip(arg_fmts, arg_sizes))


@dataclass(frozen=True)
class _ListPin:
    """What a list argument pins: its element's pin and, where the key is
    size-sensitive, its length."""
    elt: '_Pin | None'
    length: int | None


@dataclass(frozen=True)
class _TuplePin:
    """What a tuple argument pins: one pin per field."""
    elts: 'tuple[_Pin | None, ...]'


_Pin: TypeAlias = FormatBound | _ListPin | _TuplePin
"""What one argument's type pins, or ``None`` where it pins nothing."""


@dataclass(frozen=True)
class _DefBound:
    """A bound the caller derived for one named definition in the callee."""
    name: NamedId
    fmt: FormatBound


@dataclass(frozen=True)
class _ExprBound:
    """A bound the caller derived for *count* of the callee's expressions.

    Unkeyed; see :func:`_bounds_key`.
    """
    fmt: FormatBound
    count: int


_Bound: TypeAlias = _DefBound | _ExprBound

_PinnedValue: TypeAlias = Context | RoundingMode | None
"""A value a call site may pin into a callee -- see :data:`_PINNABLE`."""


class _Instance(NamedTuple):
    """How a function is instantiated -- everything about a spec except
    which function it is.

    Every field has its own equality, which is what lets
    :meth:`Specialize.apply` compare a round's output against the one before
    it without going through a name.  :func:`_mangle_private` is the only
    place any of it becomes text.
    """
    ctx: Context | None

    arg_types: tuple[_Pin | None, ...] = ()
    """What :class:`Monomorphize` is given, shaped by :func:`_type_pin`;
    empty when the argument types constrain nothing."""

    arg_vals: tuple[_PinnedValue, ...] = ()
    """The values a call site pins into the callee; empty when none are."""

    bounds: frozenset[_Bound] = frozenset()
    """What the caller's analysis derived *inside* the callee; empty for a
    public entry, which has no caller."""


class _SpecKey(NamedTuple):
    """Which function, instantiated how."""
    fdef: FuncDef
    inst: _Instance


_Shape: TypeAlias = 'frozenset[tuple[_Instance, int]]'
"""How many specs of each instantiation one expansion produced.

What :meth:`Specialize.apply` compares to decide it has reached a fixpoint.
The names would nearly do, except that a *public* keeps its entry name, so
its instantiation would be invisible to the comparison however much it
sharpened.
"""


def _is_trivial_fmt(f: FormatBound) -> bool:
    """A :class:`FormatBound` that conveys no specialization information:
    ``None`` (non-numeric) or ``REAL_FORMAT`` (the polymorphic scalar
    top)."""
    return f is None or f is REAL_FORMAT or f == REAL_FORMAT


def _type_pin(t: Type | None, size_key: bool) -> _Pin | None:
    """What *t* pins, or ``None`` when it pins nothing.

    The format of a real, the shape of an aggregate, and the length of a list
    all distinguish one spec from another.  *size_key* off drops lengths,
    keeping keys identical to a size-blind run.
    """
    match t:
        case RealType():
            return None if _is_trivial_fmt(t.fmt) else t.fmt
        case ListType():
            # the shape pins even when the leaves do not: a spec whose argument
            # is known to be a list differs from one where nothing is known
            # ... and a symbolic length is a per-run gensym, never keyed on
            return _ListPin(
                _type_pin(t.elt, size_key),
                concrete_size(t.length) if size_key else None,
            )
        case TupleType():
            return _TuplePin(tuple(_type_pin(e, size_key) for e in t.elts))
        case _:
            return None


def _is_trivial_bound(f: FormatBound) -> bool:
    """*f* constrains nothing, at any depth -- so a callee whose bounds are all
    trivial keys and names exactly as it did before it had any."""
    match f:
        case ListFormat():
            return _is_trivial_bound(f.elt)
        case TupleFormat():
            return all(_is_trivial_bound(e) for e in f.elts)
        case VarFormat():
            return True
        case _:
            return _is_trivial_fmt(f)


def _bounds_key(sub: FormatAnalysis) -> frozenset[_Bound]:
    """What a caller's analysis derives *inside* a callee, as a set.

    A relation between arguments cannot be keyed directly (see
    :class:`DigitBoundParams`), but what it yields can: two callers that
    bound a callee differently derive different bounds here and so take
    separate specs.  Sharing one would let whichever caller was analyzed
    first decide the other's storage.

    Over the *expressions* as well as the definitions: a callee that assigns
    nothing -- ``with ctx: return round(x)`` -- has only its parameters in
    ``by_def``, so two callers would key the same.  The two maps together are
    also what the backend reads.

    A set, because both maps enumerate in AST-node order, an artefact of the
    walk.  ``by_def`` carries its name; ``by_expr`` contributes bounds
    without keys, so an occurrence *count* is what tells two callers apart.
    """
    bounds = (*sub.by_def.values(), *sub.by_expr.values())
    if all(_is_trivial_bound(f) for f in bounds):
        return frozenset()
    named: set[_Bound] = {_DefBound(d.name, fmt) for d, fmt in sub.by_def.items()}
    counted = Counter(sub.by_expr.values())
    return frozenset(named | {_ExprBound(f, n) for f, n in counted.items()})


def _arg_vals_key(
    vals: tuple[_PinnedValue, ...] | None,
) -> tuple[_PinnedValue, ...]:
    """The argument values a call pins.

    A *separate* axis from :func:`_arg_types_key`: a format or a length
    constrains what a value may be, while a pin says which value it is.  That
    is partial evaluation, not typing, so it keys on its own and is applied by
    substitution rather than by an annotation.
    """
    if vals is None or all(v is None for v in vals):
        return ()
    return tuple(vals)


def _arg_types_key(
    atypes: tuple[Type | None, ...] | None, size_key: bool,
) -> tuple[_Pin | None, ...]:
    """The refined argument types, as :func:`_type_pin` shapes them.

    Empty when nothing is pinned, so a polymorphic spec passes through
    unchanged.  These types are exactly what :class:`Monomorphize` is given,
    so two specs with the same key have identical bodies.
    """
    if atypes is None:
        return ()
    pins = tuple(_type_pin(t, size_key) for t in atypes)
    return () if all(p is None for p in pins) else pins


def _sanitize_size(b: ArraySizeBound) -> ArraySizeBound:
    """*b* with every non-concrete size dropped, and ``None`` when nothing
    concrete survives at any level.

    Size *variables* (``NamedId``) are per-analysis gensyms: letting one into a
    key would make spec keys and mangled names differ from run to run.
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
        case _:
            return None


def _digest(x: object) -> str:
    """A short, reproducible digest of *x*'s structure.

    A `frozenset` is sorted first: it enumerates in hash order, which does
    not survive across processes, and this reaches generated code.
    """
    raw = repr(sorted(map(repr, x))) if isinstance(x, frozenset) else repr(x)
    return hashlib.sha1(raw.encode()).hexdigest()[:8]


def _mangle_private(base: str, inst: _Instance) -> str:
    """A name for a private spec, so two specs of one function are
    distinguishable in the emitted code.

    The key decides identity; this only has to label it, uniquely and
    reproducibly.

    *base* is the unmangled name, which the caller tracks: specialization
    re-reads its own output, and recovering the base by stripping the suffix
    would take `helper__deadbeef` apart too and collide it with a `helper`
    beside it.
    """
    parts = [base]
    if inst.ctx is not None:
        parts.append(_digest(inst.ctx))
    if inst.arg_types:
        parts.append(_digest(inst.arg_types))
    # tagged so two digests of different kinds cannot collide
    if inst.arg_vals:
        parts.append('v' + _digest(inst.arg_vals))
    if inst.bounds:
        parts.append('b' + _digest(inst.bounds))
    return '__'.join(parts)


_PINNABLE = (Context, RoundingMode)
"""Value kinds a call site may pin into a callee.

Both reach a rounding: a :class:`Context` directly, a :class:`RoundingMode`
through a context constructor.  Numbers are left out -- their formats already
travel as types, and pinning every constant would multiply specs for no gain.
"""


def _pinned_value(v: object) -> _PinnedValue:
    """*v* if a call site may pin it, else ``None``.

    Partial evaluation wraps a value FPy cannot compute on in a ``Foreign``,
    which is how a rounding mode arrives.
    """
    if isinstance(v, Foreign):
        v = v.val
    return v if isinstance(v, _PINNABLE) else None


def _pinnable_args(
    callee: FuncDef, args: tuple[Expr, ...], pe: PartialEvalInfo,
) -> tuple[_PinnedValue, ...]:
    """The value each of *args* pins in *callee*, or ``None`` where it pins
    nothing.

    A discarded parameter takes no substitution, so keying on its value would
    only split one spec into identical copies.  An unused *named* parameter
    still splits, which costs a duplicate body rather than a wrong one.
    """
    out: list[_PinnedValue] = []
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
# Dead-parameter elimination.


def _dead_args(func: FuncDef) -> tuple[int, ...]:
    """The positions of parameters *func*'s body has no use for.

    A parameter reaching only a phi still carries a value into the merge, so a
    phi operand counts as a use.
    """
    du = DefineUse.analyze(func)
    merged = {
        du.defs[i]
        for phis in du.phis.values() for phi in phis for i in (phi.lhs, phi.rhs)
    }
    out: list[int] = []
    for i, arg in enumerate(func.args):
        if not isinstance(arg.name, NamedId):
            continue
        d = du.find_def_from_site(arg.name, arg)
        if not du.uses[d] and d not in merged:
            out.append(i)
    return tuple(out)


class _DropCallArgs(DefaultTransformVisitor):
    """Drop, at every call site, the arguments whose parameters went away."""

    def __init__(self, dropped: dict[FuncDef, tuple[int, ...]]):
        self._dropped = dropped

    def _visit_call(self, e: Call, ctx):
        args = [self._visit_expr(a, ctx) for a in e.args]
        kwargs = [(k, self._visit_expr(v, ctx)) for k, v in e.kwargs]
        gone = set(
            self._dropped.get(e.fn.ast, ()) if isinstance(e.fn, Function) else ()
        )
        return Call(
            e.func, e.fn,
            [a for i, a in enumerate(args) if i not in gone],
            kwargs, e.loc,
        )

    def apply(self, func: FuncDef) -> FuncDef:
        return self._visit_function(func, None)


def _drop_dead_args(
    module: Module,
    bound_params: dict[str, DigitBoundParams] | None = None,
) -> Module:
    """Drop parameters no private spec's body uses any more.

    Pinning substitutes a value into every use of a parameter, leaving the
    parameter dead -- and a dead parameter has nothing left to infer a type
    from, which the C++ backend refuses.  A public entry keeps its signature:
    its callers are outside the module.

    Once the specs have settled, not per round: a spec's identity is partly
    the values its caller pinned, which a dropped argument no longer spells.

    The argument goes with the parameter, so an expression that only ever fed
    a dead one is no longer evaluated.  That is a change in what runs -- an
    out-of-range index there stops raising -- and is sound only because FPy
    leaves such a read undefined.

    *bound_params* is reindexed alongside: its terms are bound to parameters by
    position, so dropping a parameter without dropping its term hands every
    later parameter the one before it.
    """
    public = {f.ast.name for f in module.call_graph().publics}
    dropped: dict[FuncDef, tuple[int, ...]] = {}

    def step(_m: Module, func: FuncDef) -> FuncDef:
        # callees first, so what each one shed is already known
        func = _DropCallArgs(dropped).apply(func)
        gone = () if func.name in public else _dead_args(func)
        if not gone:
            return func
        gone_set = set(gone)
        kept = [a for i, a in enumerate(func.args) if i not in gone_set]
        out = FuncDef(func.name, kept, func.body, func.meta, loc=func.loc)
        dropped[out] = gone
        params = bound_params.get(func.name) if bound_params is not None else None
        if params is not None:
            bound_params[func.name] = DigitBoundParams(  # type: ignore[index]
                params.store,
                tuple(a for i, a in enumerate(params.args)
                      if i not in gone_set),
                params.assume,
            )
        return out

    return module.map(step)


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

    Each ``(FuncDef, _Instance)`` pair becomes one entry; cross-function
    calls are rewired to the appropriate spec.  Public entries' user-given
    names are preserved; transitively-reached private specs get a stable
    mangled name combining the original name with a digest of the
    instantiation.

    The output is assembled by registering only the public specs with
    :meth:`Module.add`; private specs surface through ``add``'s eager
    call-graph discovery (they're reachable from the publics' rewired
    ``Call.fn`` references).

    Cyclic input call graphs surface at :meth:`Module.add` time on the
    input module, before this pass runs.  Any cycle introduced by
    specialization itself would surface at the output ``add`` call.
    """

    _MAX_ROUNDS = 8
    """How many expansions to allow before treating the lack of a fixpoint as
    a defect.  Each round can only sharpen what the round before it knew, so
    reaching this means something is not monotone -- which is a bug to find,
    not a budget to raise."""

    @staticmethod
    def apply(
        module: Module,
        *,
        size_key: bool = False,
        bound_params: dict[str, DigitBoundParams] | None = None,
    ) -> Module:
        """Specialize *module*, to a fixpoint.

        One expansion is not enough where a callee's argument comes back from
        the call: a loop-carried accumulator only takes its format once the
        callee's return is known, and the callee is specialized from that
        argument.  Expanding again feeds each spec the sharper annotations the
        round before it derived, and the specs stop changing once nothing
        more is learned.  A round is compared to the last by the
        instantiations it produced; see :class:`_Shape`.

        *bound_params* is filled as :meth:`_expand` describes.
        """
        if not isinstance(module, Module):
            raise TypeError(f'expected a `Module`, got {type(module)} for {module}')

        previous: _Shape | None = None
        # every name this pass has coined, and the name it was coined from
        bases: dict[str, str] = {}
        for _ in range(Specialize._MAX_ROUNDS):
            if bound_params is not None:
                bound_params.clear()   # only the final round's bound_params describe the output
            out, shape = Specialize._expand(
                module, size_key=size_key, bases=bases, bound_params=bound_params,
            )
            if shape == previous:
                return _drop_dead_args(out, bound_params)
            previous = shape
            # re-registered as they were, since a public entry's name, context
            # and argument types are the caller's and not the spec's
            module = Module(out.name)
            for entry in out:
                module.add(
                    entry.func, name=entry.name, ctx=entry.ctx,
                    arg_types=entry.arg_types,
                )
        raise RuntimeError(
            f'specialization did not settle in {Specialize._MAX_ROUNDS} '
            'rounds: either a round is not monotone, or a program needs a '
            'deeper chain than this allows'
        )

    @staticmethod
    def _expand(
        module: Module,
        *,
        size_key: bool = False,
        bases: dict[str, str] | None = None,
        bound_params: dict[str, DigitBoundParams] | None = None,
    ) -> tuple[Module, _Shape]:
        """One expansion of *module*, and the shape of what it produced.

        *size_key* additionally keys each spec on its arguments' concrete
        lengths, so a function called with 3- and 5-element lists compiles
        twice, each spec's annotations carrying its length -- which is what lets
        the cpp backend's arrays cross call edges.  Lengths originate in program
        text (literals, ``empty(K)``, ``range(K)``, annotations), so the
        worklist stays finite; the cost is the same template-instantiation
        economics as the ctx and format axes.  ``False`` keeps keys and mangled
        names identical to a size-blind run.

        *bound_params* is filled, by spec name, with the constraint store
        and argument terms each callee's caller bound it to; replaying them
        recovers what a spec analyzed on its own would lose.
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
        arg_vals_for: dict[_SpecKey, tuple[_PinnedValue, ...] | None] = {}
        # Per-spec digit-bound params, from the caller that first reached it.
        params_for: dict[_SpecKey, DigitBoundParams] = {}
        escapes: dict[FuncDef, EscapeSummary] = {}

        public_keys: list[tuple[str, _SpecKey]] = []   # (entry_name, key) per public

        worklist: list[_SpecKey] = []
        for entry in module:
            atypes = entry.arg_types
            key = _SpecKey(entry.func.ast, _Instance(
                ctx=entry.ctx,
                arg_types=_arg_types_key(atypes, size_key),
            ))
            public_keys.append((entry.name, key))
            if key not in orig_func:
                orig_func[key] = entry.func
                arg_types_for[key] = atypes
                worklist.append(key)

        seen: set[_SpecKey] = set(worklist)
        while worklist:
            key = worklist.pop(0)
            atypes = arg_types_for.get(key)
            mono = Monomorphize.apply(key.fdef, key.inst.ctx, atypes)
            mono = _pin_arg_values(mono, arg_vals_for.get(key))
            monos[key] = mono

            # `use_digit_bounds`: the spec key carries the relational bounds, so two
            # callers whose context differs must not share a spec -- and the
            # params this captures are what carry that context into the emit.
            # With the callees' escape summaries, so a list handed to one keeps
            # the element facts a callee's params assume.
            classes = ValueClassInfer.analyze(
                mono, alias=Alias.analyze(mono, summaries=_escape_summaries(mono, escapes)))
            fa = FormatInfer.analyze(mono, use_digit_bounds=True, value_classes=classes)
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
                callee_key = _SpecKey(callee_fn.ast, _Instance(
                    ctx=callee_ctx,
                    arg_types=_arg_types_key(callee_atypes, size_key),
                    arg_vals=_arg_vals_key(callee_arg_vals),
                    bounds=_bounds_key(sub_fa),
                ))

                site_map[call] = callee_key
                if callee_key not in local_seen:
                    local_seen.add(callee_key)
                    local_callees.append(callee_key)
                db = fa.digit_bound
                if callee_key not in seen:
                    seen.add(callee_key)
                    orig_func[callee_key] = callee_fn
                    arg_types_for[callee_key] = callee_atypes
                    arg_vals_for[callee_key] = callee_arg_vals
                    if db is not None and call in db.by_call:
                        params_for[callee_key] = DigitBoundParams(
                            db.store, db.by_call[call].args, db.assume_at(call),
                        )
                    worklist.append(callee_key)
                elif (params := params_for.get(callee_key)) is not None and params.assume:
                    # the spec runs wherever any of its calls does, so it
                    # assumes only what all of them prove -- and a literal
                    # names something only in the store that minted it
                    also = (db.assume_at(call) if db is not None and db.store is params.store
                            else frozenset())
                    params_for[callee_key] = replace(params, assume=params.assume & also)

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
        #        is mangled from its instantiation.
        spec_to_public_name: dict[_SpecKey, str] = {}
        for entry_name, k in public_keys:
            spec_to_public_name.setdefault(k, entry_name)

        names: dict[_SpecKey, str] = {}
        for k in monos:
            if k in spec_to_public_name:
                names[k] = spec_to_public_name[k]
            else:
                name = orig_func[k].name
                base = name if bases is None else bases.get(name, name)
                names[k] = _mangle_private(base, k.inst)
                if bases is not None:
                    bases[names[k]] = base

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
        if bound_params is not None:
            for k, params in params_for.items():
                bound_params[names[k]] = params

        out = Module(module.name)
        for entry_name, k in public_keys:
            out.add(new_funcs[k], name=entry_name)
        shape = frozenset(Counter(k.inst for k in monos).items())
        return out, shape
