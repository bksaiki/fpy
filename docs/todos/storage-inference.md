# Audit: storage inference

What `fpy2/backend/cpp/storage_infer.py` and `storage.py` actually do today, what
the rules are at each kind of definition, and what an interface would have to
carry for the analysis to live in `fpy2/analysis/`. Not a plan — see
[backend-independence.md](backend-independence.md) §2 for where it sits.

## What the module is

Three things are bundled under one name.

1. **Storage assignment** — partition the SSA defs into classes, give each class
   one storage type. Backend-independent apart from the type domain.
2. **Variable materialization** — give each class an identifier (`def_to_name`),
   and decide where it is declared (`declare_at_assign`, `hoists_before`).
   Generic for an imperative target, meaningless for a functional one.
3. **Emitter helpers** — `is_rebound`, `binds_by_reference`, `is_single_def`.
   `binds_by_reference` is explicitly shared between the emitter and `unbox.py`
   so the two cannot drift; it answers a C++ question (does this name bind a
   reference to storage that already exists).

Only (1) is the analysis. (2) and (3) travel with it because they are keyed by
the same classes.

## The rules

### (i) Expressions — not covered

`StorageInfer` assigns storage to **definitions only**. An expression's storage
is computed independently by the emitter: `_storage_for_expr(e)` is
`choose_storage(format_info.by_expr[e])`.

The two disagree, and `backend-cpp.md` documents it as a hazard: *"an
expression's storage can disagree with its declaration's … anything deciding
whether a cast is needed has to read the one the operand is actually emitted
as"*. `_storage_or_none` exists to paper over exactly this — a `Var` reads as its
*declaration*, a `ListRef` peels its container's declared element type, and
everything else falls back to `by_expr`.

A generic analysis has to decide whether it owns this. Two coherent answers:
answer for every expression (and reconcile with the class storage), or answer for
definitions only and state that expression storage is `of_bound(by_expr[e])` with
the reconciliation left to the consumer. Today's split is the second, but
undeclared, which is why the hazard needed documenting.

Two more places pick storage outside the analysis: the **function return type**
(`compiler.py` calls this out as *"the only one `StorageInfer` does not cover"*)
and every **callee ABI** slot.

### (ii) New definitions

A definition's storage is its *class's* storage. A class is a connected component
of `reaching_defs.same_object_defs`, which unions on exactly two edges: a
`PhiDef`'s two operands, and an `IndexedAssign`'s def with its `prev`.

A definition connected to nothing is a singleton class, and its storage is
`of_bound` of its own `FormatBound`. So a fresh definition is as narrow as its
own bound allows.

### (iii) Re-definitions

**A plain rebind is not a widening — it is a new class.** `y = 1.0; y = x * x`
gives two classes with independent storage:

```c++
uint8_t y = 1;          // bound {1}
double y_1 = (x * x);
```

`store` is per runtime object, not per name. Sequential redefinition never
promotes, and the second definition never sees the first's bound. The cost is
cosmetic — a dead narrow variable, above — and the benefit is that an unrelated
reuse of a name cannot widen anything.

**An `IndexedAssign` *is* the same class.** `xs[i] = e` writes through the
existing cells, so its def joins `prev`'s class and the element bound of `e`
participates in the class join.

### (iv) Phi definitions

The only join site. `aggregate_storage(bounds)` takes the supremum over every
member's bound:

- bottom bounds (a fresh `empty(...)`, which holds no value) are dropped when any
  other member constrains, since keeping one would widen for nothing;
- storage is chosen per bound, then joined **over storages**, structurally —
  scalars by ladder supremum, lists element-wise, tuples field-wise.

Joining over storages rather than over bounds is where the known imprecision
lives: a *partly* bottom bound still contributes its empty slots, because by the
time the join happens the bound has already become a type.

The scalar join is not simply "the wider of the two": `S32 ⊔ U32` is `S64`,
because the ladder search asks for the first entry containing *both*, and no
32-bit entry contains both.

## The rules, formally

### Domains

```
F  ::= sigma | list F | tuple F1..Fn | bot | none | real     -- FormatBound
S  ::= sigma | list S | tuple S1..Sn | bool                  -- storage
sigma in Sigma                        -- the backend's distinguished set
```

`Sigma` is a finite **sequence** of scalar formats — the order is load-bearing,
see the audit below. `F <= S` is the format ordering lifted structurally. `bot` is a bound holding no value (a fresh `empty`); `none` is a
non-numeric bound (a boolean); `real` is the unconstrained real.

### Choice and join

```
ceil(F)   =  first sigma in Sigma with F <= sigma        -- none => error
ceil(list F)      = list ceil(F)
ceil(tuple F..)   = tuple ceil(F)..
ceil(none)        = bool
ceil(bot)         = head Sigma                            -- vacuous; any rung holds it

JOIN{S1..Sn}  =  first sigma in Sigma with Si <= sigma for all i    -- n-ary
JOIN{list S1..list Sn}    = list JOIN{S1..Sn}
JOIN{tuple S1..}..        = tuple JOIN{..}..
JOIN{bool..} = bool ;  bool with sigma undefined          -- a typing error upstream
```

**`JOIN` is n-ary and cannot be folded** — see the audit. It is also not "the
wider of the two": `JOIN{s32, u32} = s64`, because no 32-bit member contains
both.

### Classes

`~` is the least equivalence relation on definitions containing:

```
        d = phi(d1, d2)                    d @ IndexedAssign, prev(d) = d'
       ------------------ S-Phi            ---------------------------- S-Update
        d ~ d1    d ~ d2                              d ~ d'
```

and nothing else. In particular:

```
        d @ Assign(y, e),  d' an earlier def of y,  d not a phi operand
       ---------------------------------------------------------------- S-Rebind
                              not (d ~ d')
```

A rebind starts a new class. `store` is per runtime object, not per name.

### Assignment

```
     members([d]) = d1..dk       |- di : Fi       G = { Fi | Fi /= bot }
    ------------------------------------------------------------------- S-Class
              store([d])  =  JOIN { ceil(F) | F in (G or {bot}) }

                        store([d]) = S
                       ---------------- S-Def
                         store(d) = S
```

Bottom bounds drop when any member constrains, since `ceil(bot)` is the smallest
rung and keeping it would widen for nothing (`u8 |_| s8 = s16`).

### Expressions

```
         fmt(e) = F                        e = Var(x),  x reads d
        --------------- S-Expr             ----------------------- S-Var
        store(e) = ceil(F)                    store(e) = store(d)
```

**These two overlap and disagree.** `S-Var` is what the value is actually held
in; `S-Expr` is what its own bound would justify, and `ceil(fmt(e)) <= store(d)`
but not conversely. The emitter resolves the overlap in `_storage_or_none` by
preferring `S-Var` and following reference bindings; nothing states the
precedence. Making it a rule is the smallest fix.

### What the assignment does and does not guarantee

For a place `p` (a definition, a return, a parameter slot) and a value `v`
flowing into it, two obligations:

```
   containment      fmt(v) <= store(p)          -- the value fits
   realizability    store(v) ~> store(p)        -- the backend can get it there
```

`S-Class` guarantees **containment** for members of a class, by construction of
the join. It says nothing about **realizability**, which is where every refusal
comes from. `~>` is where the aliasing side-condition lives:

```
   sigma ~> sigma'                   iff sigma <= sigma'
   tuple S.. ~> tuple S'..           iff Si ~> S'i for each i
   list S ~> list S'                 iff S = S'                        (free)
                                      or (S ~> S' and v unshared)      (rebuild)
```

The last line is the whole aggregate story: a list may change element type only
by becoming a different object, so it needs a sharing verdict. A scalar's
realizability is a fact about formats; a list's is a fact about the heap.

### Audit of these rules

**`Sigma` must be a sequence, not a set.** `ceil` cannot be `min` over the
containment order, because that order has no least element above an arbitrary
bound: `u8` and `s8` are *incomparable* and both are minimal upper bounds of the
value `{1}`. The implementation resolves it by scanning `_LADDER` in a fixed
order and taking the first — so the answer depends on the enumeration, and the
enumeration is part of the backend's contribution. The ladder *is* a linear
extension of containment (checked), so "first containing" is always a **minimal**
upper bound; it is not the **least**, because none need exist.

**The join is n-ary and folding it is wrong in both directions.** Over the ten
ladder entries, taking all 3-element combinations:

- 4 orderings where folding **overshoots**: `JOIN{s8, u16, f32}` is `float`, but
  `(s8 |_| u16) |_| f32` is `double` — because `s8 |_| u16` picks `s32`, and no
  32-bit integer fits in `float`.
- 12 orderings where folding **fails** and the n-ary join succeeds:
  `JOIN{s8, u32, f32}` is `double`, but `(s8 |_| u32) |_| f32` finds nothing,
  since `s64` does not fit `float` or `double`.

So `Sigma` under containment is **not a join-semilattice**: `{s8, u16}` has two
incomparable minimal upper bounds, `s32` and `f32`, and no least one. The
tie-break is therefore not neutral — choosing `s32` is exactly what makes the
later join fail. `aggregate_storage` avoids this by joining a whole class at once;
an interface exposing a binary `join(a, b)` would reintroduce it.

**Termination rests on a premise the rules do not state.** Class construction is
union-find over a finite set of definitions; `ceil` and `JOIN` are finite scans of
`Sigma`. The only unbounded recursion is structural — `ceil(list F)`,
`JOIN{list ..}`, `~>` on lists — and it terminates because `FormatBound` is
finite-depth, which holds because FPy has no recursive list types (a cycle would
need `xs[0] = xs`, which fails to unify). That should be a premise, not a
coincidence.

**`store(e)` is ambiguous as written.** `S-Expr` and `S-Var` have overlapping
conclusions, so `store` on expressions is a *relation*, not a function. The
emitter picks `S-Var` and follows reference bindings; the rules must either say so
or use two judgment forms.

**The rule is stronger than the code.** `S-Class` requires a bound for every
member. `aggregate_storage` skips members absent from `def_to_bound`
(`if d in def_to_bound`), so a member without one contributes no constraint and
the class may not contain its values. Whether that is reachable is unclear; the
rule is the right statement and the code should assert it.

**The partly-`bot` imprecision, concretely.** `ceil` before `JOIN` costs two
rungs: `JOIN{ceil(bot), ceil(s8)}` is `JOIN{u8, s8}` = `s16`, where
`ceil(bot |_| s8)` = `ceil(s8)` = `s8`. The empty slot forces an *unsigned* rung
that then cannot merge with a signed one.

### Where the implementation is underdetermined

- **`ceil` then join, or join then `ceil`.** `S-Class` as written is
  `JOIN{ceil(Fi)}`, matching the code. `ceil(JOIN{Fi})` — joining in the format
  lattice first — is at least as precise and strictly more so for a partly-`bot`
  bound, as above.
- **`ceil` is minimal by fiat.** Nothing forces the first containing member; it is
  a policy suiting a target where upcasting is free. See *Widening a list is a
  different decision*.
- **The tie-break among minimal upper bounds is a real choice**, not a detail of
  iteration order, and it has downstream cost.
- **A side rule escapes the lattice.** An `MPFixedContext` with `expmin >= 0` and
  no specials falls back to the widest integer member, ignoring its magnitude
  bound entirely. It is not derivable from `Sigma` and belongs in the interface.

## Real-valued versus list-valued

For a scalar the join is free. C++ converts at the assignment, so a member whose
own bound was narrower simply stores into a wider variable:

```c++
double y{};
if (c) { y = 1; } else { y = (x * x); }   // `1`'s bound is {1}; no cost
```

For a list the join is element-wise, and **what it costs depends on where the
value comes from** — which is a fact about aliasing, not about types:

| the value | cost of widening | mechanism |
|---|---|---|
| built at the site (`[1.5, 2.5]`, a comprehension, `empty`) | free | `_emit_at` constructs at the wanted type |
| held in a variable with its own storage | O(n) copy, new object | `_rebuild_list` |
| **shared** (aliased, or a parameter, or a callee's return) | **impossible** | `_refuse_unsharing` |

Measured: `zs = [1.5, 2.5]; if c: zs = [y]` compiles — both arms *construct*, so
both build `vector<double>` directly. `return xs` where `xs: list[FP32]` and the
other arm returns `[y: FP64]` refuses, because `xs`'s storage is fixed by the
signature and rebuilding it would hand back a different object.

So for aggregates, storage assignment and aliasing are not separable. `F <= S` is
necessary and not sufficient: among the `S` that contain the bound, the
admissible ones are those every member can *reach* — and a shared aggregate can
reach only its own.

Nesting recurses structurally, and each level carries the same question
independently: `list[list[real]]` widening its innermost element is a rebuild at
every level, O(total elements), and the sharing verdict is per level (`unbox.py`
keys it per alias region, per depth).

Tuples are element-wise like lists but always by value, so a tuple field's
widening is a `make_tuple` of converted fields — no identity to lose.

## Performance considerations

- **Widening is monotone and class-wide.** One wide member widens every member.
  A loop-carried list that is wide on one path is wide on all of them, for the
  whole life of the variable.
- **Narrowing never happens.** The analysis only joins. There is no pass that
  observes "every value in this class is actually small" and shrinks it — the
  join of the members' bounds is the answer, and a bound is already the smallest
  format containing the values.
- **The narrow-rebind artifact.** Because a rebind is a fresh class, a program
  that assigns a placeholder and then the real value declares two variables, the
  first often dead. Cheap for scalars, an allocation for aggregates.
- **The cost model lives in the consumer, not the analysis.** The analysis picks
  the smallest containing type; whether that choice is cheap depends on the
  construct/rebuild/refuse table above, which it does not consult. A backend that
  wanted to avoid a rebuild would have to *change the class's storage*, which is
  the analysis's output, not its input.
- **Integer narrowing of real values is deliberate and load-bearing.**
  `acc = 0.0` becomes `uint8_t` when the bound says every value is a small
  integer. `backend-cpp.md` measures 21 of 112 corpus list element types this
  way. It is a size decision, gated on the bound genuinely excluding NaN and
  signed zero.

## Widening a list is a different decision from widening a scalar

The two are usually treated as one operation applied structurally. They are not,
and the difference is not only the boxing refusal.

**A scalar's storage is a local decision; a list's is a commitment.** C++ converts
a scalar at the point of use, so choosing `float` and needing `double` later costs
one upcast at one site. `std::vector<float>` and `std::vector<double>` are
*unrelated types*, so the same mistake costs a new buffer — and with it a new
object identity, which is why a shared list cannot pay at all. The element format
propagates to every place the list flows; the scalar format does not.

So the cost of being wrong is asymmetric, and asymmetric in a different direction
for each kind:

| | too wide | too narrow |
|---|---|---|
| scalar | a few bytes | a free upcast |
| list | *n* × a few bytes | O(n) copy + allocation, or a refusal |

Minimal containment — pick the smallest type holding the bound — is the right
policy for the left column and the wrong one for the right.

**Where headroom would actually pay.** Not inside a class: the class join is
already "wide up front", and every construction site builds at it. Measured —
`zs = [1.5, 2.5]; if c: zs = [y]` emits `std::vector<double>` at *both* arms with
no conversion, because the class settled on `double` before either was built.

The rebuild arises only where a member's storage is fixed *outside* the class: a
parameter, a callee's return, a reference binding. No amount of headroom in the
class helps there, because the class is not what chose. Headroom pays at exactly
the positions that cannot be revised later — **signatures and returns**. Widening
a list parameter is what `backend-cpp.md`'s call-site workaround achieves
indirectly, by having `Specialize` instantiate the callee at the wider argument
format.

**The rebuild path may not be reachable at all.** Across the 201 compiling corpus
functions `_convert_storage` fires 37 times and `_rebuild_list` **never** does.
That measurement is rigged: the harness instantiates every real as `FP64`
(`_inst_type`), so no list parameter has a narrow element and mixed formats are
unreachable by construction.

Built by hand, the picture does not improve, but for a different reason:

```python
with fp.FP32:  a = [1.5, 2.5]
with fp.FP64:  b = [y, y]
return [a, b]              # refused: `a` holds float where double is needed
```

`a` is a fresh local list used once, so rebuilding it is **safe** — the refusal is
not a semantic one. It is refused because `a` is *boxed*, and it is boxed because
a name and a container both hold it, which is the open imprecision
`backend-cpp.md` records under *What stays boxed* (closing it needs liveness).
Inline the same literal and it compiles, because a value built at the site is
built at the wide type directly:

```c++
std::vector<double>{static_cast<double>(1.5), static_cast<double>(2.5)}
```

So today, widening a list element is **free when the value is constructed at the
site, and refused when it is named** — and `_rebuild_list` did not fire in any of
three hand-built cases or anywhere in the corpus. The middle path the design
provides for is one I could not reach.

**The boxing verdict is the only thing holding it shut.** Forcing every region
unboxed — simulating the liveness fix — makes the refusing program above compile,
and the rebuild fires:

```c++
std::array<double, 2> _tmp2{};
for (size_t _tmp1 = 0; _tmp1 < a.size(); ++_tmp1) {
    _tmp2[_tmp1] = static_cast<double>(a[_tmp1]);
}
```

That couples two work items that look unrelated. The liveness gap is filed as a
*precision* improvement in `unbox`, but closing it converts this class of program
from a compile error into a silent O(n) copy — a refusal becoming a cost rather
than becoming a success. Whichever lands first decides whether the widening
policy is ever observable: today its cost is zero or a diagnostic, never a loop.

**The cost is placed by dataflow, not by the programmer.** The one-off O(n) is
the smaller half. A conversion sits wherever a value crosses into a place with
different storage, and nothing bounds how often that site executes — a merge
inside a loop converts once per iteration, for a decision made statically about a
type. Choosing the width at the *definition* happens once; converting at the
*use* happens as often as control flow says. That asymmetry, rather than the
constant factor, is the argument for deciding early and generously.

**Second-order effects of headroom.** A wider element makes every copy dearer,
which feeds back into `unbox`: the value-versus-handle decision is partly about
copy cost, and a `std::array<double, K>` is twice the stack object that
`std::array<float, K>` is. It also collides with the open question in
`backend-cpp.md` on whether integer narrowing of real values earns its keep — 21
of 112 corpus list element types are value-narrowed reals. That discussion is
currently all-or-nothing; "narrow scalars, widen list elements" is a third
position nobody has tried.

**What the interface would need.** A backend cannot express any of this against
an analysis that always picks minimal containment. The minimum is a policy per
structural position — minimal at a scalar, generous at a list element — rather
than one rule applied structurally. "Generous" needs its own definition against a
finite set: the next rung up, or the target's natural width, are the two obvious
readings, and the second is the one a backend can state and an analysis cannot
guess.

**A reference binding already breaks the class abstraction.** `ys = xs` is a plain
rebind, so it is *not* a coalescing edge and `ys` gets its own class with its own
storage — while the emitter binds `const auto& ys = xs`, giving it `xs`'s type in
fact. `_storage_or_none` exists to follow that alias, and its comment records the
bug it was written for. Whatever policy widening takes, the partition should
either coalesce reference bindings or the analysis should be told which bindings
are references; today the emitter patches over a divergence the analysis created.

## The domain is `FormatBound`, not a type vocabulary

`CppScalar` predates the `Format` abstraction, and is now mostly a *label* for a
format the backend can spell. The identification is exact: `S64` is
`SINT64.format()` — a `FixedFormat`, hashable, with working equality, lifting to
precisely the `AbstractFormat` the ladder holds for it. Every operation defined
on `CppScalar` is defined *through* that lift:

| today | is |
|---|---|
| `scalar_fits_in(a, b)` | `_LADDER_LOOKUP[a] <= _LADDER_LOOKUP[b]` |
| `scalar_sup([...])` | the first ladder entry containing all of them |
| `choose_storage_scalar(b)` | the first ladder entry containing `b` |
| `exact_integer_bits(ty)` | `int(af.prec)` for a float rung |
| `is_float()` | finite `prec` and specials; an integer rung is `A(inf, 0, …)` |

So the analysis can work directly on `FormatBound`. `ListFormat` and
`TupleFormat` already exist there, `AbstractFormat` already has the ordering, and
the two operations storage inference needs are format operations:

- `of_bound(b)` — the smallest member of the backend's set containing `b`
- `join(s, t)` — the smallest member containing both

Its **input** domain is all of `FormatBound` (whatever format inference
produced); its **output** domain is the backend's distinguished set, applied
structurally. That is exactly the `F <= S` rule, with no separate type
vocabulary in between.

What that removes from `storage.py`: the ladder-lookup table, `scalar_fits_in`,
`scalar_sup`, `exact_integer_bits`, `bound_fits_in_scalar`, and the
`CppScalar`↔format plumbing. What is left of that module is the *set* itself and
a spelling function.

### What does not collapse

**The representation axis.** `CppList(elt, boxed, size)` fuses two orthogonal
things: the element format, and whether the list is a shared handle, a value, or
a fixed-length value. Only the first is a format. `unbox.py` decides the second
and today patches it in by *mutating the analysis's output*
(`storage.class_storage.update(unbox.storage)`), which the split would remove:
the analysis answers a `FormatBound`, `unbox` answers a representation, and the
backend's own type is a view of the pair.

`CppList.__eq__` deliberately includes `boxed` and `size`, so that
`std::array<double, 3>` cannot pass for `std::array<double, 4>`. That constraint
belongs to the representation, not the format, and survives in whatever the
backend keeps.

**Booleans.** There is no `BoolFormat`; `FormatBound` is `None` for a boolean,
and `CppScalar.BOOL` is the backend's stand-in. In the format domain the answer
is simply `None`, which the backend spells `bool` — a special case that stops
being one, except that `join` still needs a rule for `None` against a numeric
format. Today `scalar_sup` raises there; a type error upstream would be better.

**The spelling.** `format()` on each `CppType` is target syntax and stays in the
backend, as a function from (storage format, representation) to a string.

## What an interface would have to carry

### What the backend must supply

**An ordered sequence of storage formats**, given as `Format`s
(`SINT64.format()`, `FP64.format()`, …). The analysis derives `of_bound` and an
**n-ary** `join` from it and the existing `AbstractFormat` ordering.

The order is not presentation: containment over the set is not a join-semilattice
(`{s8, u16}` has two incomparable minimal upper bounds), so the sequence *is* the
tie-break, and a different one changes which programs compile. `join` must take a
collection — a binary operator folded over a class is both less precise and less
total than joining the class at once.

**Nothing for structure.** `ListFormat` and `TupleFormat` are already
`FormatBound`s, and the analysis joins their components independently. Today's
`CppList` / `CppTuple` exist only because the domain was a type vocabulary.

**A non-numeric answer.** `FormatBound` is `None` for a boolean, and `BOOL` is
not on the ladder — `scalar_sup` refuses to mix it with numeric storage rather
than joining. Whatever replaces it needs the same "not a number format, joins
only with itself" behaviour.

**A fallback policy.** `choose_storage_scalar` has one: an `MPFixedContext` with
`expmin >= 0` and no specials falls back to `S64`, *deliberately ignoring the
magnitude bound* — an unbounded integer has none, and overflow is the user's
problem. That is a backend policy, not derivable from the set.

### Lossless conversions: probably redundant, worth confirming

The obvious extra input is a set of supported lossless conversions. It looks
unnecessary, because **containment in the format lattice already is
losslessness**: `scalar_fits_in(a, b)` is `_LADDER_LOOKUP[a] <= _LADDER_LOOKUP[b]`,
and that already answers `int64_t → double` as *false* (precision 64 does not fit
in 53) while `int32_t → double` is true. A backend that supplies its formats
gets the conversion relation for free.

Two directions could break that, and a design should say which it assumes:

- **The backend supports fewer conversions than containment allows** — a target
  with no integer-to-float instruction, say. Then a conversion set is a genuine
  extra input, and `join` must additionally require reachability.
- **The backend supports more** — a conversion that is lossless for a reason the
  format lattice does not model. Then the set is a refinement, and containment is
  the conservative default.

Note that today `_maybe_cast` and `bound_fits_in_scalar` already diverge from
type-level containment at the *use* site: a value whose *bound* fits may be cast
even where its *type* does not. That is a third relation, and it lives in the
emitter.

### What should not be in the interface

- **Naming.** `def_to_name` and the `x_1` suffixing are variable materialization.
- **Declaration placement.** `declare_at_assign` / `hoists_before` assume a
  target with declarations and block scope.
- **`binds_by_reference`.** A C++ question, and one deliberately shared with
  `unbox.py`; extracting it would need the binding rule as a backend callback.

## Open questions

**The constraint that settles most of them: this replaces `storage_infer.py`, so
it must keep its capability.** That turns nine of the design questions from
"decide" into "preserve", and de-scopes the rest. Behavioural equivalence is also
cheap to check — the emitted C++ should be identical across the corpus before and
after.

### Settled by the replacement constraint

| | question | answer |
|---|---|---|
| 1 | expressions, or only definitions | **definitions only**; the consumer derives expression storage as today |
| 2 | owns the return type and ABI slots | **no** — both stay where they are |
| 3 | picks, or reports constraints | **picks** |
| 4 | which minimal upper bound | **first in the backend's sequence** |
| 5 | minimal or generous per position | **minimal everywhere** |
| 6 | join over bounds or over storages | **over storages**, keeping the partly-`bot` imprecision |
| 7 | aliasing oracle, or report-and-refuse | **reports**; the consumer refuses |
| 9 | a lossless-conversion relation | **no** — containment is it |
| 10 | the non-numeric case | `ceil(none) = bool`, joining it with a numeric member is an error |

Two consequences worth naming.

**Question 7's answer is what makes the move possible.** The analysis does not
need an aliasing oracle — it guarantees `containment` and leaves `realizability`
to the consumer — so it can live in `fpy2/analysis/` as a standalone pass. Had it
needed the oracle, it could not.

**Question 5's answer de-scopes the widening discussion entirely.** Generous
storage for list elements is a *later* change with its own measurement, not part
of the replacement. See *Widening a list is a different decision*.

### Phase 0, before the replacement

8. **Reference bindings.** The invariant is that a reference binding and a shared
   storage class are the same claim:

   > if the emitter binds `d` by reference to `src`, then
   > `storage_of(d) == storage_of(src)`

   Today they may disagree and `_storage_or_none` compensates at every read; the
   comment there records the miscompile that came from missing one. Two designs:
   coalesce on the alias relation (truthful, but a class may then contain a def
   whose storage is fixed elsewhere, making the join unsatisfiable), or **bind a
   reference only where the storages agree** and copy otherwise. The second is
   chosen: no analysis change, capability preserved, and the alias chase in
   `_storage_or_none` becomes redundant rather than load-bearing. Assert the
   invariant either way.

   Doing this first also makes question 1 cheap — see below.

### Settled by measurement

12. **Every class member has a bound.** 1195 of 1195 definitions across the 201
    compiling corpus functions are in `def_to_bound`; the `if d in def_to_bound`
    guard never excludes one. The new code should **assert** rather than skip: a
    member without a bound contributes no constraint, so silently skipping it can
    mis-store a class.
13. **`_rebuild_list` is unreachable today**, and reachable once the boxing
    verdict is fixed — forcing every region unboxed makes the refusing program
    compile with an element-wise copy loop. So there is no list-widening
    behaviour to preserve, which shortens the equivalence obligation.

### Deferred

11. **The boxing/liveness fix stays parked** until a widening policy exists.
    Question 5 keeps the policy as-is for the replacement, so nothing governs the
    silent O(n) copies that fix would enable. It would also move the baseline the
    replacement is measured against.

### The one open decision

1. **Definitions only, or extend to expressions?** The replacement constraint
   answers *definitions only*. But `_storage_or_none` (34 lines) already computes
   `store(e)` — a `Var` reads its def's storage, a `ListRef` peels the
   container's element, everything else is `ceil(by_expr[e])` — so extending is
   mostly relocation, and it turns the `S-Expr` / `S-Var` ambiguity into a
   defined function. Its `Var` case depends on the reference-source chase, which
   phase 0 removes. So the order matters: after 8, extending is cheap; before it,
   the chase moves into the analysis, which is the wrong direction.


## Plan of action

Each numbered phase is a commit. Pause after each for review; the executor does
not commit. Only the named tests run per phase — full suites at the end.

The settled answers this plan assumes are in *Open questions*; the numbers below
in brackets refer to them.

### 0. The equivalence harness, and the reference-binding invariant

**0a. Build the harness first**, because every later phase is checked with it: a
script that emits C++ for every compiling corpus function (`all_unit_tests` +
`all_example_tests` + the four libraries, instantiated through `_inst_type`) and
writes it to a directory. A phase is equivalent iff the directory is
byte-identical. Keep it out of `tests/` — it is a development tool, not a test.

Record the baseline before touching anything.

**0b. Re-measure the invariant.** Whether
`storage_of(d) == storage_of(_reference_source(d))` already holds everywhere.
An earlier attempt reported zero reference bindings across 199 functions, which
is not believable; confirm the harness intercepts before trusting the result. If
the invariant already holds, 0c is a pure assertion and costs nothing.

**0c. Enforce it [8].** `binds_by_reference` gains a storage-equality condition,
so a name binds a reference only where its class storage equals the source's;
otherwise it copies and converts. Assert the invariant at the binding site — it
is what would have caught the boxed-`uint8_t` miscompile `_storage_or_none`'s
comment records.

Then check whether `_storage_or_none`'s alias chase is now redundant. Leave it
if unsure; phase 3 removes it properly.

```
pytest tests/unit/backend/cpp -q -n 8
```

### 1. The storage domain becomes `FormatBound`

In place, still in the backend — no files move, so the diff is the domain change
alone.

- `Sigma` becomes an ordered **sequence** of `Format`s (`SINT64.format()`,
  `FP64.format()`, …) rather than a `CppScalar` enum [4].
- `ceil(F)` is the first member containing `F`; `JOIN` is **n-ary** over a
  collection, never a folded binary — folding is both less precise and less
  total (`JOIN{s8, u16, f32}` is `float`; folded it is `double`, and
  `JOIN{s8, u32, f32}` folded fails outright).
- `scalar_fits_in`, `scalar_sup`, `_LADDER_LOOKUP`, `exact_integer_bits` and
  `bound_fits_in_scalar` collapse into format operations or vanish.
- `choose_storage`'s unbounded-integer fallback stays, marked as the policy hook
  it is.
- Assert that every class member has a bound rather than skipping [12].
- `CppList` / `CppTuple` do **not** vanish: they carry the representation axis
  (`boxed`, `size`), which is not a format. What changes is that the analysis no
  longer speaks in them; the backend builds one from (storage format,
  representation).

The `class_storage` mutation `unbox.py` performs
(`storage.class_storage.update(...)`) is the seam here. It stays for now, but the
types on either side change.

Risk: this is where emitted output could shift. Byte-identical is the bar.

```
pytest tests/unit/backend/cpp -q -n 8    +    the 0a harness diff
```

### 2. Split assignment from materialization

`storage_infer.py` keeps the partition and `class_storage`. `def_to_name`,
`hoists_before`, `declare_at_assign` and `_is_external` move to a sibling module
in the backend — they are variable materialization, not storage. `is_rebound`,
`binds_by_reference` and `is_single_def` go with them.

Nothing changes semantically; this is the cut line for phase 5.

```
pytest tests/unit/backend/cpp -q -n 8    +    harness diff
```

### 3. Extend `store` to expressions [1]

`_storage_or_none`'s 34 lines become the analysis's `store(e)`: a `Var` reads its
def's storage, a `ListRef` peels the container's element type, everything else is
`ceil(by_expr[e])`. Phase 0 removed the reference chase, so the `Var` case is
direct.

This makes `store` total over expressions and turns the `S-Expr` / `S-Var`
overlap into a defined function. State the precedence in the docstring.

Note this widens the change beyond a like-for-like replacement, so the harness no
longer covers all of it — the new expression answers were previously computed by
the emitter. Expect the diff to be empty anyway; investigate if not.

```
pytest tests/unit/backend/cpp -q -n 8    +    harness diff
```

### 4. The backend interface

A protocol the backend implements:

- the ordered sequence of storage formats;
- the non-numeric answer (`bool`) and the rule that joining it with a numeric
  member is an error [10];
- the fallback policy for an unbounded integer format.

No conversion relation — containment is losslessness [9]. No structure map —
`ListFormat` and `TupleFormat` are already `FormatBound`s. `join` takes a
collection.

Parameterize the assignment on it, with the C++ ladder as the one instance.

```
pytest tests/unit/backend/cpp -q -n 8    +    harness diff
```

### 5. Relocate

`fpy2/analysis/storage_infer.py`, exported from `fpy2.analysis`. Pure motion;
the C++ ladder and the spelling function stay in `fpy2/backend/cpp/storage.py`.

Add tests that exercise the analysis without a backend — the point of the move.
State the termination premise (`FormatBound` is finite-depth because FPy has no
recursive list types) in the module docstring.

```
pytest tests/unit -q -n 8
python -m tests.infra
python -m tests.infra.backend.cpp
```

### 6. Documentation

Fold the settled rules into the new module's docstring. Retire this document,
keeping in [backend-independence.md](backend-independence.md) §2 only what a
reader of the roadmap needs: the rules, the interface, and the two things
deliberately inherited — the class-wide monotone join, and the widening policy
staying minimal.

## What this plan does not do

- **Widening policy** stays minimal everywhere [5]. Generous storage for list
  elements is a later change with its own measurement.
- **The boxing/liveness fix** stays parked [11]. It converts refusals into silent
  O(n) copies, and nothing governs them yet.
- **Return type and callee ABI slots** stay where they are [2].
- **Reference bindings are not coalesced into the class.** Phase 0 makes the
  divergence impossible rather than modelling it.
