# TODO: a `Module` abstraction for compilation

## Goal

Add a backend-agnostic **`Module`** — a collection of functions registered
for compilation, each with an optional monomorphization spec. A module can
be handed to a compiler, transformed per-function, and (eventually) used for
structured cross-function specialization.

Registering a function marks it **public**; every function in its call chain
is **private** (internal). So the module covers the whole reachable program,
not just the entry points.

**Eager vs lazy (recommended: lazy/derived).** Functions reference their
callees by live `Call.fn` object references, so the private set is *derivable*
from the public entries at any time via `CallGraph`. The stored state is just
the public registrations (+ specs); public-vs-private is a computed projection
over the call graph, recomputed on demand (and memoized with invalidation on
`add`). This stays fresh when a pass rewrites the graph, and matches how the
cpp compiler already discovers callees. The alternative — eagerly snapshotting
the call chain at `add` time — is simpler to query but can go stale. See the
open decisions.

This generalizes the cpp backend's existing `CppTranslationUnit` (a
cpp-specific, ad-hoc "collection of functions to compile together") into a
reusable container that fpc and future backends can also consume. It also
gives fpc the multi-function story it currently lacks.

## Grounding (what exists today)

- `Function` wraps a `FuncDef` (`.ast`); `Function.with_ast(new_ast)` rebuilds
  it while preserving env/runtime.
- `Monomorphize.apply(func, ctx, args)` / `apply_by_arg(...)` specialize a
  `FuncDef` against a context + per-argument types.
- cpp: `CppCompiler.compile(func, *, ctx, arg_types) -> str`; and
  `CppCompiler.unit() -> CppTranslationUnit` with `unit.add(func, *, ctx,
  arg_types)` + `unit.render()`. The unit shares a specialization cache across
  added functions (callees auto-included, deduped, ODR-safe).
- fpc: `FPCoreCompiler.compile(func, ctx) -> FPCore` — single function, no unit
  concept, no `arg_types`.
- `CallGraph.analyze(funcdef)` gives the reachable functions + leaves-first
  order (used for the whole-module call graph / acyclicity check).

## Data model

**Stored state** is an ordered collection of **public entries** — each a
registered function plus an optional monomorphization spec:

```python
@dataclass
class ModuleEntry:
    func: Function                                       # compilers take Function; .ast is the FuncDef
    name: str                                            # export name; defaults to func.name
    ctx: Context | None = None                           # optional monomorphization context
    arg_types: tuple[Type | None, ...] | None = None     # optional per-arg monomorphization
```

Public entries are keyed by `name`, so the same `FuncDef` can be registered
under several specs/names.

**Private functions are derived, not stored** (lazy model): the set of
internal functions is computed on demand by walking the call graph from the
public entries. A function reachable only as a callee is private; if a
function is both registered and a callee, public wins; a private callee
reached from several publics is one function (deduped by `FuncDef` identity).

Note `CallGraph` nodes are `FuncDef`s, but the module needs `Function`
wrappers (compilers consume `Function`, and `map` rebuilds via
`Function.with_ast`). The wrappers are available at the call sites
(`Call.fn` is the callee `Function`), so the derivation builds a
`dict[FuncDef, Function]` alongside `CallGraph` — the root's `Function` is
the registered one; each callee's comes from `Call.fn`. (A small helper, or
an optional `CallGraph` mode that also returns the `Function` per node, can
provide this; keep `CallGraph` itself `FuncDef`-keyed.)

## API surface

```python
class Module:
    def __init__(self, name: str | None = None): ...

    # registration — "optional monomorphization" = the optional ctx/arg_types spec
    def add(self, func: Function, *, name: str | None = None,
            ctx: Context | None = None,
            arg_types: Collection[Type | None] | None = None) -> None: ...

    # introspection — public is stored; private/all are derived (lazy)
    def __iter__(self) -> Iterator[ModuleEntry]   # public entries (stored order)
    def __len__(self) -> int                      # number of public entries
    def __contains__(self, name: str) -> bool
    def get(self, name: str) -> ModuleEntry
    def public(self) -> list[Function]            # registered functions
    def private(self) -> list[Function]           # internal callees (derived)
    def functions(self) -> list[Function]         # public + private (derived)

    # whole-program structure (uses CallGraph)
    def call_graph(self) -> CallGraphAnalysis

    # per-function passes — "apply passes on each function"
    def map(self, transform: Callable[['Module', FuncDef], FuncDef]) -> 'Module': ...
```

- `private`/`functions`/`call_graph` are derived on demand from the public
  entries (memoized, invalidated on `add`).
- `map` applies a `(Module, FuncDef) -> FuncDef` transform across the program
  and returns a **new** `Module`, rebuilding each `Function` via
  `Function.with_ast`. The module argument is the one `.map` was called on, so a
  pass can consult the surrounding program (call graph, other functions).
  Composes with existing transforms via a thin wrapper:
  `module.map(lambda m, fd: FuncInline.apply(fd))`.
  **Caveat (the hard part):** callers reference callees by `Call.fn` (a live
  `Function`). Transforming a private callee in isolation leaves callers
  pointing at the *old* `Function`. So `map` must rewrite the call graph
  consistently — transform in leaves-first `CallGraph` order, and rebind each
  caller's `Call.fn` to the transformed callee. This call-graph-aware rewrite
  is the main reason a `Module` is more useful than a loose list of functions;
  it is also the subtlest piece to get right (flagged in the open decisions).
- `call_graph` ties in the existing analysis; the module is the natural owner
  of the whole-program graph (dedup/ordering, cross-entry recursion checks).

## Compiler integration

Add `compile_module` to each backend; the return type is backend-specific.

- `CppCompiler.compile_module(module) -> str` — thin adapter over the unit:
  ```python
  unit = self.unit()
  for e in module:
      unit.add(e.func, ctx=e.ctx, arg_types=e.arg_types)
  return unit.render()
  ```
  Reuses the shared specialization cache / ODR-safety the unit already
  provides, now driven by a portable `Module`.
- `FPCoreCompiler.compile_module(module) -> dict[str, FPCore]` — compile each
  entry (`compile(e.func, e.ctx)`), keyed by `e.name`. (fpc + `arg_types`: see
  open decisions.)

`Backend` gets a loosely-typed `compile_module` declaration.

## Monomorphization: stored spec, not eager (recommended)

The entry stores the `(ctx, arg_types)` **spec**; the compiler applies it
(cpp's `_run_pipeline` already calls `Monomorphize`, so eager monomorphization
in the module would double-apply). Callers wanting concrete functions up front
use the pass API instead:

```python
module.map(lambda m, fd: Monomorphize.apply(fd, ctx, arg_types))   # or a dedicated module.monomorphized()
```

Recommendation: **lazy spec consumed by the compiler** — maps 1:1 onto today's
working cpp path, avoids re-monomorphization.

## Where it lives

`fpy2/module.py` (top-level, alongside `function.py`), exported as `fp.Module`
(and `ModuleEntry`).

## Phases

1. **[DONE] Core container + registration + public/private derivation**:
   `Module`, `ModuleEntry`, `add`, iteration/lookup, validation (duplicate-name,
   `Function` type, `ctx` type, `arg_types` arity). Derive `public()` /
   `private()` / `functions()` via `CallGraph` + the `FuncDef -> Function` map
   (memoized, invalidated on `add`). Exported from `fpy2/__init__.py`.

   *Notes learned in phase 1 (relevant later):*
   - `_compute_derived` already builds **and memoizes** the
     `FuncDef -> Function` map (third element of `_derive()`); phase 2's
     rewiring should reuse it rather than rebuild.
   - **Acyclicity is already enforced** — `CallGraph.analyze` runs inside the
     derivation, so `private()`/`functions()` (and any compile/`map`) raise
     `CallGraphError` on a cyclic module today. Phase 4 narrows to just the
     public `call_graph()` accessor (a combined whole-module graph).
   - `public()` dedupes by `Function` identity (same func under two names ->
     one public function, two entries).

2. **[DONE] `map` (per-function passes)** via `Function.with_ast`. *Refined design
   (from phase 1):* because **private is derived**, `map` only needs to return
   a new `Module` whose **public entries** are the transformed registered
   functions (same names/specs) — the transformed privates re-derive
   automatically, *provided* the transformed public bodies' `Call.fn` references
   point at the transformed callees. So `map` must:
   1. compute a **module-wide leaves-first order** over all functions (merge the
      per-root `CallGraph.order`s with a shared visited set — a combined
      post-order DFS — since a callee shared across roots is transformed once);
   2. for each function in that order, **rebind its `Call.fn`** references to
      the already-transformed callees (using an `old Function -> new Function`
      map), *then* apply the user transform, then wrap via `with_ast` and record
      the old->new mapping;
   3. build the new `Module`, re-adding the transformed **public** functions
      with their original names/specs.

   This makes a transform like `FuncInline` inline the *transformed* callees
   (bottom-up), and a structure-agnostic transform like `ConstFold` is
   unaffected by the rebinding. A small `Call.fn`-rebinding helper is needed
   (a `DefaultTransformVisitor` overriding `_visit_call` to swap `fn` per the
   map) — verify none exists before writing it.

   Test that a transform applied through the module matches applying it directly
   *and* that cross-function calls still resolve to the transformed callees.

   *Notes learned in phase 2 (relevant later):*
   - Implemented as a `_RebindCalls(DefaultTransformVisitor)` helper +
     `_compute_derived` now also produces the module-wide leaves-first `order`
     (all functions), reused by `map`.
   - **`Call.is_equiv` compares `fn` by identity**, so "transform through module
     == transform applied directly" can only be asserted *structurally* for a
     **leaf** function (no calls). For functions that call others, the rebound
     `Call.fn` points at fresh `Function` objects, so those cases must be tested
     by **semantics** (execution equivalence) + checking callees are rewired —
     not `is_equiv`. (cpp's phase 3 compares source strings, so it is unaffected.)
   - **`map` then compile composes for free**: a mapped module's public entries
     reference the transformed callees (via the rewired `Call.fn`), so a backend
     that auto-includes callees (cpp) will pull in the *transformed* privates.
     Phase 3 `compile_module` therefore just iterates the public `_entries`
     (with their specs); it does not need to register privates.
3. **[DONE] Compiler integration**: `CppCompiler.compile_module(module) -> str`
   (reuses `unit()`; `compile` left as-is), `FPCoreCompiler.compile_module(module)
   -> dict[str, FPCore]`, and a base `Backend.compile_module` stub
   (`NotImplementedError`; backend-specific return type). Both validate the
   argument is a `Module`. `Module` is referenced via a `TYPE_CHECKING` import +
   a local runtime import (avoids any backend<->module import-order issue).

   *Notes learned in phase 3:*
   - cpp `compile_module` output is byte-identical to a manual `unit.add` loop.
   - **`map` + compile compose**: a mapped module compiles fine (its public
     entries reference the transformed callees, which cpp auto-includes).
   - Decision 3 resolved: **fpc ignores `arg_types`** (not expressible in
     FPCore) while honoring `ctx`, so one module stays portable across backends.
4. **[DONE] `call_graph()` accessor — `ModuleCallGraph`**: rather than
   shoehorning the multi-root union into a single-rooted `CallGraphAnalysis`,
   added a dedicated `ModuleCallGraph` dataclass — keyed on `Function` (the
   wrappers the module actually holds), with public-vs-private baked in. It
   carries the merged `callees` / `callers` / global leaves-first `order`,
   plus `is_public`, `callees_of`, `callers_of`, iteration, `__contains__`,
   `format()` (a forest — one indented tree per public entry, shared callees
   marked `(*)` across both trees), and `dot()` (publics styled `style=bold`).
   `Module.call_graph()` returns it; backed by the same memoized derivation
   that drives `public()`/`private()`. Acyclicity is enforced for free — per-root
   `CallGraph.analyze` already raises during the derivation. Exported as
   `fp.ModuleCallGraph`.

   *Notes from phase 4:*
   - Per-root `CallGraph.callees` lists for the same `FuncDef` are identical
     across roots (the function's body doesn't change), so merging is just
     "first writer wins" per `FuncDef`.
   - The base `Backend.compile_module` stub from phase 3 was reverted —
     `compile_module` is implemented per backend with no abstract declaration
     (matches how `compile` is also per-backend with no base method).
5. **[DONE — v1] Structured specialization**: `Module → Module` pass that
   expands public entries' `(ctx, arg_types)` specs into a flat set of
   fully-monomorphized functions. Each `(FuncDef, calling-ctx)` pair becomes
   one entry; cross-function calls in the output are rewired to the
   appropriate spec. Lives in `fpy2/transform/specialize.py` as a `Specialize`
   class with `apply(module) -> Module`; `Module.specialized()` is a thin
   shorthand. Public entries' user-given names are preserved; private specs
   get a `name__<8hex>` fingerprint of the ctx.

   *v1 design choices:*
   - **ctx-only key** `(FuncDef, ctx)`. Per-arg-format keying (the cpp
     `_discover_specializations` shape) is deferred; can be added as a
     `Specialize.apply(module, key=...)` strategy in v2.
   - **Per-call contexts via `FormatInfer.by_call`** — `sub_fa.fn_fmt.ctx`
     gives the callee's calling context at each call site. Heavier than a
     custom context walker, but a known-correct path the cpp pipeline
     already uses.
   - **`Monomorphize.apply(fdef, ctx, arg_types)`** per spec for the actual
     body specialization.
   - **`_RebindCallSites` visitor** — a per-`Call`-identity variant of
     `_RebindCalls`, since within one specialized caller the same callee
     can be invoked at different specs from different sites (rare but
     correct to support).
   - **Mangling**: `hashlib.blake2b(repr(ctx), digest_size=4).hexdigest()`.

   *Shape note (greedy-feel output, no model change):* the **pre-specialization**
   `Module` keeps the lazy/derived model — auto-consistent under `map`-style
   rewrites. The **post-specialization output** is a regular `Module` where
   every spec is `add`-ed explicitly — the "greedy/flat" feel comes from how
   the output is built, not from changing the underlying design.

   *Future / deferred for v2+:*
   - Format-aware specialization (`(FuncDef, ctx, param_formats)`), matching
     cpp's current key — accept a `key=` callable.
   - Have `CppTranslationUnit` detect a pre-specialized input and skip its
     own walk (currently it just re-discovers each unique FuncDef → one spec,
     which is a wasted pass but correct).
   - Per-backend name-mangling hook for callers who want a particular scheme.

## Open decisions

1. **[RESOLVED] Public/private tracking — lazy/derived.** Store only public
   entries; compute the private set from the call graph on demand (memoized,
   invalidated on `add`). Stays fresh when a pass rewrites the graph.
2. **[RESOLVED] Monomorphization timing — lazy spec.** The entry stores the
   `(ctx, arg_types)` spec; the compiler applies it at `compile_module` time
   (no eager monomorphization in the module).
3. **[RESOLVED] fpc + `arg_types`** — **ignore** it (FPCore can't express
   per-arg monomorphization); `ctx` is still honored. Keeps a module portable
   across backends rather than rejecting at compile time.
4. **[RESOLVED] `map` granularity** — a plain `Callable[[Module, FuncDef],
   FuncDef]` (the module + the function). Passing the module lets a pass consult
   the surrounding program; existing single-arg transforms compose via a thin
   `lambda m, fd: T.apply(fd)`.
5. **[RESOLVED] `map` call-graph rewiring** — transform leaves-first; before
   transforming each function, rebind its `Call.fn` to already-transformed
   callees via an `old Function -> new Function` map (the `_RebindCalls`
   visitor). Implemented in phase 2.

(Resolved: **callee inclusion** — the module owns it. Public = registered;
private = derived call-chain functions. Earlier draft left this to the
compiler; superseded by the public/private model.)

## Testing

- Phase 1: registration, duplicate names, defaulting `name` to `func.name`,
  type validation, iteration order; `public()`/`private()` classification —
  a registered function with a private callee, a callee shared across two
  publics (deduped, one private), a function that is both registered and a
  callee (public wins).
- Phase 2: `module.map(lambda m, fd: T.apply(fd))` equals applying `T.apply` to each function
  directly (structural / `is_equiv`), and a public caller's `Call.fn` resolves
  to the *transformed* private callee afterward.
- Phase 3: cpp `compile_module` output matches a hand-built `unit.add` loop;
  fpc `compile_module` returns the expected `dict[str, FPCore]`.
- Phase 4: `call_graph()` over a module whose entries share a callee; recursion
  across entries raises.
