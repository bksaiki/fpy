"""
A module: a collection of functions registered for compilation.

Registering a function (:meth:`Module.add`) records it as a *public* entry,
optionally with a monomorphization spec (a rounding context and/or per-argument
types).  The functions reachable from the public entries through their call
chains are *private* (internal).

Private functions are **not stored** — they are derived on demand from the
call graph (:class:`~fpy2.analysis.CallGraph`).  Because functions reference
their callees by live ``Call.fn`` object references, the private set is always
derivable from the public entries, and stays consistent as passes rewrite the
program.
"""

from dataclasses import dataclass
from typing import Callable, Collection, Iterator

from .analysis import CallGraph
from .ast import Call, FuncDef
from .ast.visitor import DefaultTransformVisitor
from .function import Function
from .number import Context
from .types import Type


class _RebindCalls(DefaultTransformVisitor):
    """Rewrites a function body's ``Call.fn`` references according to an
    ``old Function -> new Function`` map.  Used by :meth:`Module.map` to point
    callers at their transformed callees; non-``Function`` call targets
    (primitives, builtins, ...) are left unchanged."""

    def __init__(self, mapping: dict[Function, Function]):
        self._mapping = mapping

    def _visit_call(self, e: Call, ctx):
        args = [self._visit_expr(arg, ctx) for arg in e.args]
        kwargs = [(k, self._visit_expr(v, ctx)) for k, v in e.kwargs]
        # ``Call.fn`` is ``object`` (may be a primitive, builtin, None, ...);
        # only rebind user functions we have a replacement for.
        fn = e.fn
        if isinstance(fn, Function) and fn in self._mapping:
            fn = self._mapping[fn]
        return Call(e.func, fn, args, kwargs, e.loc)

    def apply(self, func: FuncDef) -> FuncDef:
        return self._visit_function(func, None)


@dataclass
class _Derived:
    """Memoized derivation over the module's public entries."""
    private: list[Function]
    """internal functions, leaves-first"""
    order: list[Function]
    """all functions (public + private) in global leaves-first order"""
    fdef_to_func: dict[FuncDef, Function]
    """every reachable ``FuncDef`` to its ``Function`` wrapper"""


@dataclass
class ModuleEntry:
    """A public function registered with a :class:`Module`.

    *ctx* and *arg_types* are an optional monomorphization spec, consumed by
    the compiler when the module is compiled (they do not eagerly transform
    the function)."""

    func: Function
    name: str
    ctx: Context | None = None
    arg_types: tuple[Type | None, ...] | None = None


class Module:
    """A collection of functions for compilation.

    The stored state is the ordered list of *public* entries.  Public-vs-private
    classification and the all-functions view are derived from the call graph on
    demand (memoized; invalidated whenever a function is added)."""

    name: str | None
    _entries: list[ModuleEntry]
    _by_name: dict[str, ModuleEntry]
    _derived: _Derived | None  # memoized; invalidated on `add`

    def __init__(self, name: str | None = None):
        self.name = name
        self._entries = []
        self._by_name = {}
        self._derived = None

    # ------------------------------------------------------------------
    # Registration

    def add(
        self,
        func: Function,
        *,
        name: str | None = None,
        ctx: Context | None = None,
        arg_types: Collection[Type | None] | None = None,
    ) -> None:
        """Register *func* as a public function of the module.

        Args:
            func: the function to register.
            name: export name; defaults to ``func.name``.  Must be unique.
            ctx: optional rounding context to monomorphize against.
            arg_types: optional per-argument types to monomorphize against;
                if given, its length must match the function's arity.
        """
        if not isinstance(func, Function):
            raise TypeError(f'expected a \'Function\', got {func!r}')
        if name is None:
            name = func.name
        if name in self._by_name:
            raise ValueError(f'module already has a function named {name!r}')
        if ctx is not None and not isinstance(ctx, Context):
            raise TypeError(f'expected a \'Context\' for ctx, got {ctx!r}')
        if arg_types is not None:
            if not isinstance(arg_types, Collection):
                raise TypeError(f'expected a collection for arg_types, got {arg_types!r}')
            arg_types = tuple(arg_types)
            if len(arg_types) != len(func.args):
                raise ValueError(
                    f'arg_types length mismatch for {name!r}: expected '
                    f'{len(func.args)}, got {len(arg_types)}'
                )

        entry = ModuleEntry(func=func, name=name, ctx=ctx, arg_types=arg_types)
        self._entries.append(entry)
        self._by_name[name] = entry
        self._derived = None  # invalidate the derived views

    # ------------------------------------------------------------------
    # Lookup / iteration over the public entries (stored)

    def __iter__(self) -> Iterator[ModuleEntry]:
        """Iterate the public entries in registration order."""
        return iter(self._entries)

    def __len__(self) -> int:
        """The number of public entries."""
        return len(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def get(self, name: str) -> ModuleEntry:
        """The public entry registered under *name*."""
        return self._by_name[name]

    # ------------------------------------------------------------------
    # Derived views (public stored, private/all computed lazily)

    def public(self) -> list[Function]:
        """The registered functions, deduplicated by identity, in
        registration order."""
        seen: set[int] = set()
        out: list[Function] = []
        for entry in self._entries:
            if id(entry.func) not in seen:
                seen.add(id(entry.func))
                out.append(entry.func)
        return out

    def private(self) -> list[Function]:
        """The internal functions, reached only through the call chains of the
        public entries (derived from the call graph), in leaves-first order."""
        return self._derive().private

    def functions(self) -> list[Function]:
        """All functions in the module: public entries followed by the derived
        private functions."""
        return self.public() + self._derive().private

    def _derive(self) -> _Derived:
        if self._derived is None:
            self._derived = self._compute_derived()
        return self._derived

    def _compute_derived(self) -> _Derived:
        public = self.public()
        public_fdefs = {f.ast for f in public}

        # Map every reachable ``FuncDef`` to its ``Function`` wrapper.  The
        # public roots are the registered functions; each callee's wrapper is
        # recovered from the ``Call.fn`` at its call sites (``CallGraph`` is
        # ``FuncDef``-keyed, but the module needs the wrappers).
        fdef_to_func: dict[FuncDef, Function] = {}
        order: list[Function] = []
        seen: set[FuncDef] = set()

        for f in public:
            fdef_to_func[f.ast] = f
            cg = CallGraph.analyze(f.ast)
            for calls in cg.call_sites.values():
                for call in calls:
                    callee = call.fn
                    # CallGraph only records call sites with `Function` targets.
                    assert isinstance(callee, Function)
                    fdef_to_func[callee.ast] = callee
            # leaves-first within each root; merging across roots with a shared
            # `seen` keeps the union globally leaves-first (callee before caller)
            # and analyzes a shared callee once.
            for fdef in cg.order:
                if fdef in seen:
                    continue
                seen.add(fdef)
                order.append(fdef_to_func[fdef])

        private = [f for f in order if f.ast not in public_fdefs]
        return _Derived(private=private, order=order, fdef_to_func=fdef_to_func)

    # ------------------------------------------------------------------
    # Per-function passes

    def map(self, transform: Callable[['Module', FuncDef], FuncDef]) -> 'Module':
        """Apply *transform* to every function in the module, returning a
        **new** module.

        *transform* takes ``(module, func)`` — this module (so a pass can
        consult the surrounding program, e.g. its call graph or other
        functions) and the ``FuncDef`` to rewrite — and returns the rewritten
        ``FuncDef``.

        Functions are transformed in leaves-first order, and each caller's
        ``Call.fn`` references are rebound to the already-transformed callees
        before the caller is transformed — so cross-function calls keep
        resolving correctly (and a transform like ``FuncInline`` inlines the
        transformed callee bodies).  The new module re-registers the
        transformed public entries with their original names and
        monomorphization specs; the private functions re-derive from them.
        """
        derived = self._derive()
        old_to_new: dict[Function, Function] = {}
        for old in derived.order:  # leaves-first: callees before callers
            rebound = _RebindCalls(old_to_new).apply(old.ast)
            new_ast = transform(self, rebound)
            old_to_new[old] = old.with_ast(new_ast)

        result = Module(self.name)
        for entry in self._entries:
            result.add(
                old_to_new[entry.func],
                name=entry.name,
                ctx=entry.ctx,
                arg_types=entry.arg_types,
            )
        return result
