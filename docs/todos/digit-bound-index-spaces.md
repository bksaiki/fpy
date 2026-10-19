# Digit-bound inference: index spaces by identity

Design note, for review before any code.  It replaces the loop-depth proxy
digit-bound inference uses to decide what a variable varies over; see
`digit-bound-inference.md`, "What is left".

## Context

A list contributes one *element summary* to the store: variables that stand
for an arbitrary element, universally quantified over its index.  Three
places need to know whether a variable varies with some index:

| consumer | question | today |
|---|---|---|
| `_instance` (a part of a list: `xs[0]`, `xs[k + 1]`, a gather) | which variables does the part rename? | every variable in `_elt_vars` |
| `_elt_lsb` | is a list's grid uniform, or does it move with the list's index? | `_term_level(lsb) > _elt_depth` |
| `_grid_ge` | what a grid bound inherits from the terms bounding it | the deepest `_term_level` |

`_elt_vars` is one global set, every variable minted inside any loop or
comprehension, or a summary.  `_term_level` is a loop *depth*: `_var_level`
(the depth a variable is minted at) and `_anchor_level` (the depth of what
bounds it; a list summary sits one deeper than its list).

Depth stands in for identity, and two different index spaces at one depth
are conflated:

```python
for j in range(2):                        # depth 1: j
    es = [fp.logb(x) for x in xs]         # es, xs summaries: depth 1, xs's index
    t = xs[0]                             # a part of xs
    e = max(es)                           # depth 1: varies with j, not xs's index
    with fp.MPFixedContext(e - 12, fp.RM.RTZ):
        out[j] = fp.round(t)
```

Instantiating `{0}` of `xs` renames everything in `_elt_vars`, `e`
included, so the envelope `es_i <= e` reaches the part as `es@0 <= e@0` and
says nothing about `e`.  Renaming by depth instead ("at least as deep as the
list's summary") fails the other way: `e` and `xs`'s summary are both at
depth 1.  The precision is 288 where 12 is right.  The mmasim generator's tree
before CDNA3's separate join is this shape.

## What an index space is

Covering is by length: `_covers` lets any `range(n)` loop read every list of
length `n` through the list's own summary variables, and two lists are paired
exactly when one loop body reads both.  So the identity is not a loop but the
class covering builds:

- every list definition has an **element space**;
- every binder (a `for`, a comprehension clause) has an **index space**;
- a binder that covers a list, or iterates it directly, *unifies* its index
  space with the list's element space (union-find);
- a list a binder builds -- a comprehension, an `fp.empty` filled by a covering
  loop -- has its element space unified with that binder's.

`es = [logb(x) for x in xs]` thus puts `es`, `xs` and the comprehension in one
class, which is the pairing the summaries already rely on.  `j`'s loop covers
nothing here, so it is a class of its own.

## What a variable varies over

Each variable carries a set of classes, `spaces(v)`:

- **minted** inside binders `B1 .. Bn`: their classes;
- a list's **summary**: its element space, plus wherever the list itself
  sits;
- a **grid** bounded below by terms (`_grid_ge`, the `_anchor_level` role):
  the union of theirs, except that an lsb's own bound does not add its
  list's space -- the exception `_moves_at` already makes.  Other
  constraints, a `max`'s among them, propagate nothing, as today;
- a callee's, called from inside binders: the callee's plus the call site's
  -- `_raise_anchor(i, _elt_depth)` today.

An aggregate over a list -- `max`, `min`, `sum` -- is minted *outside* the
binder that built the list, so it does not vary over the list's space: `e`
above is `{j}`.  No rule removes a space; it is never added.

## The consumers

- `_instance` for a part of a list in class `S` renames exactly the variables
  with `S` in their spaces.  `e` keeps its name, and the envelope reaches the
  part as `es@0 <= e`.
- `_elt_lsb` keeps a summary's grid when the list's own class is not among the
  grid bound's spaces: uniform over the list's index.
- `_grid_ge` unions spaces instead of taking a maximum depth.

`_elt_vars` becomes "has a non-empty set", and the depth fields go.

## What does not change

- Covering stays by length, and what it claims stays the same; this only
  gives the claim a name.
- The store, its constraints, and every operator rule.
- Parts keep their keys, so `es` and `prods` over one range still share a
  renaming.

## Tests

- The outer-loop case above: precision 12.
- The existing pairing and part tests (`TestAPartOfAListKeepsItsPairing`,
  `TestAnElementReadAnyWay`, the gathers), unchanged.
- `_elt_lsb` both ways: a grid that moves with the list's index dropped, one
  bound only by an outer loop kept.
- The corpus: `compile.py` at 60/62, and CDNA3's tree in the sweep compiling.

## Open questions

### Is unification by length sound for lists that are never read together?

Two unrelated lists of one length land in one class as soon as one loop
covers both.  That is today's claim made explicit, and it is sound because a
covering read *is* the same index; but a class grows monotonically, so a
variable can be renamed for a part of a list it never met.  Renaming more is
the safe direction, so the provisional call is to accept it, and revisit if a
design loses a bound to it.

### Should a part get a class of its own?

A part's variables are copies, so they could be given a fresh class, a subset
of their list's.  Nothing reads it yet; the provisional call is no, until
loop unrolling (the other gap in "What is left") needs per-iteration
identity, which is the same question.
