# Scheduling operators: expose existing transforms

## Context

`fpy2/strategies` is the user-facing scheduling language (a la Halide);
`fpy2/transform` holds the rewrites. Today only 6 of the 24 transforms
are reachable through a scheduling operator:

| Strategy | Transform(s) |
|---|---|
| `split` | `SplitLoop` |
| `unroll_for` / `unroll_while` | `ForUnroll` / `WhileUnroll` |
| `simplify` | `ConstFold` + `CopyPropagate` + `DeadCodeEliminate` (fixpoint) |

Several remaining transforms are user-meaningful scheduling decisions,
not just backend lowering. This plan exposes them, ordered by precedent
in Halide/TVM/Exo first, then by impact for the FPy-novel
(precision-scheduling) ones.

Every operator follows the established wrapper pattern
(`fpy2/strategies/loop_split.py` is the model):

- signature `Function -> Function`, `TypeError` on non-`Function`,
  body is argument validation + `Transform.apply(func.ast, ...)` +
  `func.with_ast(...)`;
- loop/site targeting uses the existing `where: int | None` convention
  (`None` = everywhere);
- exported from `fpy2/strategies/__init__.py`, one `autofunction`
  entry added to `docs/source/strategies.rst`;
- a unit test module under `tests/unit/strategies/`.

Each phase below is one commit.

## Guiding examples

Two schedules that should run end-to-end once every phase lands; they
become an integration test (`tests/unit/strategies/test_schedules.py`)
in the final phase. Until then each phase checks its own operator in
isolation.

**Loop schedule** — a dot product with a helper call, a derived
iterable, and a baked constant:

```python
SCALE = 2.0

@fp.fpy
def mul_add(acc: fp.Real, x: fp.Real, y: fp.Real) -> fp.Real:
    return acc + SCALE * (x * y)

@fp.fpy
def dot(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
    acc = 0.0
    for x, y in zip(xs, ys):
        acc = mul_add(acc, x, y)
    return acc

sched = inline(dot)              # phase 1: expose the loop body
sched = close(sched)             # phase 7: bake SCALE (now free in dot)
sched = elim_iter(sched)         # phase 4: indexed loop, no tuple list
sched = unroll_for(sched, times=1)  # existing
sched = simplify(sched)          # existing
```

**Precision schedule** — pin formats, then delete the roundings the
pinned context made identities:

```python
@fp.fpy
def prod3(x: fp.Real, y: fp.Real, z: fp.Real) -> fp.Real:
    return (x * y) * z

@fp.fpy
def any_small(xs: list[fp.Real]) -> bool:
    return any([abs(x) < 1e-6 for x in xs])

sched = monomorphize(prod3, ctx=fp.FP64,
                     args=[RealType(fp.FP32)] * 3)  # phase 2
sched = elim_round(sched)        # phase 3: FP32*FP32 exact in FP64
sched = lift_context(sched)      # phase 6
sched = simplify(sched)          # existing

sched2 = fuse(any_small)         # phase 5: no intermediate list[bool]
```

Both schedules must preserve interpreter behavior on concrete inputs;
the integration test asserts that plus the key structural facts (no
remaining user-function calls, no `zip` node, no `Round` under FP64,
no comprehension under `any`).

## Phase 1: `inline` (`FuncInline`)

Halide `compute_inline` / TVM `compute_inline` / Exo `inline`. The
biggest enabler: no loop operator can see a loop nest hiding behind a
call.

```python
def inline(func: Function, where: int | None = None, *,
           funcs: Iterable[Function] | None = None,
           recursive: bool = True) -> Function
```

`funcs` restricts which callees are candidates (`None` = every FPy
call); `where` selects a single candidate site by index in visit
order, matching the `where` convention of `split` / `unroll_*`
(out-of-range raises `ValueError`, like `WhileUnroll`). The two
compose: `inline(f, 1, funcs=[g])` is "the second call to `g`".
With `recursive=True` a selected site's callee is fully flattened.
Site targeting required adding `where` to `FuncInline` itself
(candidate-site counter in `_visit_call`; the bottom-up fast path
applies only when `funcs` and `where` are both `None`).

## Phase 2: `monomorphize` (`Monomorphize`)

The type/argument half is Halide `Func::specialize()`; pinning a
rounding context is the FPy-novel half and the scheduling decision the
whole precision pipeline hangs on — it is what makes `elim_round`
(phase 3) and `simplify`'s context folding fire.

```python
def monomorphize(func: Function, ctx: Context | None = None,
                 args: Collection[Type | None] | None = None) -> Function
```

Thin wrapper over `Monomorphize.apply(func, ctx, args)`; the analysis
argument (`ty_info`) stays internal.

## Phase 3: `elim_round` (`RoundElim`)

The payoff step of a precision schedule: after `monomorphize`, delete
the roundings the pinned context made provably identity. Already
trusted in the C++ backend pipeline. No Halide/TVM analogue (they do
not model rounding).

```python
def elim_round(func: Function) -> Function
```

`RoundElim.apply` takes no options, so the wrapper is trivial. Known
follow-ups for the transform itself are tracked in
[round-elim.md](round-elim.md); this phase only exposes it.

## Phase 4: `elim_iter` (`EnumerateElim` + `ZipElim`)

Analogue of inlining a derived view so the intermediate tensor is never
materialized. Schedule-relevant because `split` / `unroll_for`
materialize the iterable into a temp — run on a `zip(...)` loop that
builds exactly the tuple list `ZipElim` avoids — and the window is
one-way: once a target rewrite (e.g. `ForUnpack`, or split/unroll)
runs, the guards never fire again.

One bundle, not two operators: the passes have hard ordering
constraints (`EnumerateElim` must itself handle `enumerate(zip(...))`,
since after its rewrite the `zip` sits on an assignment RHS out of
`ZipElim`'s reach). Mirrors `simplify`'s flag style:

```python
def elim_iter(func: Function, *, enable_enumerate: bool = True,
              enable_zip: bool = True) -> Function
```

Internal order: `EnumerateElim`, then `ZipElim`. `ForUnpack` is
deliberately excluded — it is backend normalization that buys the user
nothing and destroys the patterns this operator targets.

Docstring caveat: the rewrites take the loop bound from the first
source, so on mismatched-length `zip` — undefined behavior in FPy —
the transformed program may observably differ from the interpreter.

## Phase 5: `fuse` (`ReduceFusion`)

Producer–consumer fusion of an `any` / `all` comprehension into a
single loop (what Halide `compute_at` / TVM fusing an injective stage
into a reduction does, narrowed to these two reducers).

```python
def fuse(func: Function) -> Function
```

## Phase 6: `lift_context` (`LiftContext`)

FPy-novel, enabler-shaped: hoists `with`-context expressions to the
top level, normalizing programs so `elim_round` / `simplify` apply
more often. Least useful standalone, hence last of the exposures.

```python
def lift_context(func: Function) -> Function
```

`eval_info` stays internal.

## Phase 7: `close` (`FreeVarElim`)

Constant/parameter baking from staging systems (weakest precedent).
Materializes captured data values as leading assignments so a schedule
is self-contained and later passes (const-fold especially) can see the
constants.

```python
def close(func: Function) -> Function
```

Raises (via the transform) on free variables with no literal form;
say so in the docstring.

## Validation

Per phase: the new unit test module plus the existing strategy tests.
After the final phase, the full pre-merge suites: `python -m
tests.infra` and `python -m tests.infra.backend.cpp`.

## Out of scope / follow-ups

- **`reorder` and `tile`** — the most-precedented Halide operators of
  all, but no transform exists yet; tile is sugar over two `split`s +
  reorder once reorder lands. New-transform work, not wrappers.
- **Adjacent-loop fusion** — distinct from `ReduceFusion`; no
  transform exists.
- **Targeted context rewrite** — change the context at a specific
  site, the inverse of `monomorphize`'s whole-function pinning.
- **`specialize_module`** — `Specialize` is `Module -> Module`, so it
  does not fit the `Function -> Function` strategy signature; expose
  as a separate entry point if scripting is wanted.
- **Keep internal**: `ForBundling` / `WhileBundling` / `IfBundling`,
  `SimplifyIf` (could surface as `if_convert` if vectorization ever
  lands), `SubstVar` / `RenameTarget`, and the individual
  `ConstFold` / `CopyPropagate` / `DeadCodeEliminate` (reachable via
  `simplify` flags).
